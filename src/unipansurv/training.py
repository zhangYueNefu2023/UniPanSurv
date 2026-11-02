from __future__ import annotations

import argparse
import contextlib
import itertools
import json
import math
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Sequence

import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from .config import UniPanSurvConfig
from .data import SurvivalBatch, TensorSurvivalDataset, collate_survival_records
from .losses import cancer_balance_weights, survival_objective
from .metrics import harrell_c_index
from .model import UniPanSurv


@dataclass
class DataSettings:
    train: str
    validation: str | None = None
    test: str | None = None
    modalities: list[int] | None = None
    ppi_edges: str | None = None


@dataclass
class OptimisationSettings:
    learning_rate: float = 2e-4
    weight_decay: float = 1e-4
    max_epochs: int = 200
    patience: int = 20
    max_grad_norm: float = 1.0
    cancer_balance_gamma: float = 0.5
    per_device_batch_size: int = 2
    gradient_accumulation_steps: int = 1
    expected_global_batch_size: int | None = 8
    stage: str = "selection"
    refit_epochs: int = 43


@dataclass
class RuntimeSettings:
    seed: int = 2026
    num_workers: int = 0
    amp: str = "bf16"
    deterministic: bool = True
    output_dir: str = "runs/unipansurv"
    resume: str | None = None


@dataclass
class RunConfig:
    model: UniPanSurvConfig
    data: DataSettings
    optimisation: OptimisationSettings = field(default_factory=OptimisationSettings)
    runtime: RuntimeSettings = field(default_factory=RuntimeSettings)

    @classmethod
    def from_json(cls, path: str | Path) -> "RunConfig":
        config_path = Path(path).resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
        base = config_path.parent
        data_values = dict(raw["data"])
        for key in ("train", "validation", "test", "ppi_edges"):
            value = data_values.get(key)
            if value is not None and not Path(value).is_absolute():
                data_values[key] = str((base / value).resolve())
        runtime_values = dict(raw.get("runtime", {}))
        for key in ("output_dir", "resume"):
            value = runtime_values.get(key)
            if value is not None and not Path(value).is_absolute():
                runtime_values[key] = str((base / value).resolve())
        return cls(
            model=UniPanSurvConfig.from_dict(raw["model"]),
            data=DataSettings(**data_values),
            optimisation=OptimisationSettings(**raw.get("optimisation", {})),
            runtime=RuntimeSettings(**runtime_values),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.to_dict(),
            "data": asdict(self.data),
            "optimisation": asdict(self.optimisation),
            "runtime": asdict(self.runtime),
        }


@dataclass
class DistributedContext:
    rank: int
    local_rank: int
    world_size: int
    device: torch.device
    distributed: bool

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def initialise_distributed() -> DistributedContext:
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if distributed:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            device = torch.device("cuda", local_rank)
            backend = "nccl"
        else:
            device = torch.device("cpu")
            backend = "gloo"
        dist.init_process_group(backend=backend, init_method="env://")
    else:
        rank, local_rank, world_size = 0, 0, 1
        device = torch.device("cuda", 0) if torch.cuda.is_available() else torch.device("cpu")
    return DistributedContext(rank, local_rank, world_size, device, distributed)


def seed_everything(seed: int, deterministic: bool) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False


def load_ppi_edges(path: str | Path) -> Tensor:
    rows: list[tuple[int, int]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            fields = stripped.replace(",", "\t").split("\t")
            if len(fields) != 2:
                raise ValueError(f"{path}:{line_number}: expected two integer columns")
            try:
                rows.append((int(fields[0]), int(fields[1])))
            except ValueError:
                if line_number == 1:
                    continue
                raise
    if not rows:
        raise ValueError(f"{path}: no PPI edges were read")
    return torch.tensor(rows, dtype=torch.long)


def _loader(
    dataset: TensorSurvivalDataset,
    settings: RunConfig,
    context: DistributedContext,
    *,
    training: bool,
) -> tuple[DataLoader[SurvivalBatch], DistributedSampler | None]:
    sampler = None
    if context.distributed:
        sampler = DistributedSampler(
            dataset,
            num_replicas=context.world_size,
            rank=context.rank,
            shuffle=training,
            seed=settings.runtime.seed,
            drop_last=False,
        )
    generator = torch.Generator().manual_seed(settings.runtime.seed + context.rank)
    loader = DataLoader(
        dataset,
        batch_size=settings.optimisation.per_device_batch_size,
        shuffle=training and sampler is None,
        sampler=sampler,
        num_workers=settings.runtime.num_workers,
        pin_memory=context.device.type == "cuda",
        collate_fn=collate_survival_records,
        generator=generator,
        persistent_workers=settings.runtime.num_workers > 0,
    )
    return loader, sampler


def _autocast(context: DistributedContext, amp: str) -> contextlib.AbstractContextManager:
    if context.device.type != "cuda" or amp == "none":
        return contextlib.nullcontext()
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(amp)
    if dtype is None:
        raise ValueError("runtime.amp must be one of: none, bf16, fp16")
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 was requested but this GPU does not support it")
    return torch.autocast(device_type="cuda", dtype=dtype)


def _model_loss(
    model: nn.Module,
    batch: SurvivalBatch,
    config: RunConfig,
    training_counts: Tensor,
) -> tuple[Tensor, Tensor]:
    output = model(batch.patients, mode="source")
    weight = cancer_balance_weights(
        batch.cancer_id,
        training_counts,
        config.optimisation.cancer_balance_gamma,
    )
    loss = survival_objective(
        head=config.model.survival_head,
        objective_logits=output.objective_logits,
        hazard_logits=output.hazard_logits,
        interval_index=batch.interval_index,
        event=batch.event,
        observed_time=batch.observed_time,
        sample_weight=weight,
        deephit_ranking_weight=config.model.deephit_ranking_weight,
        deephit_ranking_temperature=config.model.deephit_ranking_temperature,
    )
    return loss, output.risk_score


def train_epoch(
    model: nn.Module,
    loader: DataLoader[SurvivalBatch],
    optimizer: AdamW,
    scaler: Any,
    config: RunConfig,
    context: DistributedContext,
    training_counts: Tensor,
) -> float:
    model.train()
    optimizer.zero_grad(set_to_none=True)
    total_loss = torch.zeros(2, device=context.device)
    accumulation = config.optimisation.gradient_accumulation_steps
    iterator = iter(loader)
    while True:
        cpu_group = list(itertools.islice(iterator, accumulation))
        if not cpu_group:
            break
        # Normalise once over the effective global batch, not separately in
        # each microbatch. With batch=1, local normalisation cancels N_c^-0.5.
        local_weight_sums = [
            training_counts[cpu_batch.cancer_id.to(context.device)].float()
            .pow(-config.optimisation.cancer_balance_gamma).sum()
            for cpu_batch in cpu_group
        ]
        global_weight_sum = torch.stack(local_weight_sums).sum()
        if context.distributed:
            dist.all_reduce(global_weight_sum, op=dist.ReduceOp.SUM)
        for index, cpu_batch in enumerate(cpu_group):
            batch = cpu_batch.to(context.device)
            synchronize = index + 1 == len(cpu_group)
            sync_context = (
                contextlib.nullcontext()
                if synchronize or not isinstance(model, DistributedDataParallel)
                else model.no_sync()
            )
            with sync_context:
                with _autocast(context, config.runtime.amp):
                    loss, _ = _model_loss(model, batch, config, training_counts)
                    # DDP averages gradients over workers; undo that averaging
                    # after applying the common global-batch denominator.
                    factor = (context.world_size * local_weight_sums[index]
                              / global_weight_sum.clamp_min(1e-8))
                    scaled_loss = loss * factor
                scaler.scale(scaled_loss).backward()
            total_loss[0] += loss.detach() * local_weight_sums[index]
            total_loss[1] += local_weight_sums[index]
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimisation.max_grad_norm)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)
    if context.distributed:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    return float((total_loss[0] / total_loss[1].clamp_min(1e-8)).item())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[SurvivalBatch],
    config: RunConfig,
    context: DistributedContext,
    training_counts: Tensor,
) -> dict[str, float]:
    model.eval()
    # Gather predictions, remove DistributedSampler padding, and compute the
    # weighted validation objective once. Averaging microbatch-normalised
    # losses would make early stopping depend on batch size/cancer ordering.
    local_rows: list[tuple[Any, ...]] = []
    for cpu_batch in loader:
        batch = cpu_batch.to(context.device)
        with _autocast(context, config.runtime.amp):
            output = model(batch.patients, mode="source")
        for index, patient_id in enumerate(batch.patient_id):
            local_rows.append((
                patient_id,
                float(batch.observed_time[index]),
                float(batch.event[index]),
                float(output.risk_score[index]),
                int(batch.cancer_id[index]),
                output.objective_logits[index].float().cpu().tolist(),
                output.hazard_logits[index].float().cpu().tolist(),
                int(batch.interval_index[index]),
            ))
    if context.distributed:
        gathered: list[Any] = [None for _ in range(context.world_size)]
        dist.all_gather_object(gathered, local_rows)
        rows = [row for partition in gathered for row in partition]
    else:
        rows = local_rows
    unique = {row[0]: row for row in rows}
    ordered = list(unique.values())
    times = torch.tensor([row[1] for row in ordered])
    events = torch.tensor([row[2] for row in ordered])
    risks = torch.tensor([row[3] for row in ordered])
    cancers = torch.tensor([row[4] for row in ordered])
    weights = cancer_balance_weights(
        cancers, training_counts.cpu(), config.optimisation.cancer_balance_gamma
    )
    loss = survival_objective(
        head=config.model.survival_head,
        objective_logits=torch.tensor([row[5] for row in ordered]),
        hazard_logits=torch.tensor([row[6] for row in ordered]),
        interval_index=torch.tensor([row[7] for row in ordered]),
        event=events, observed_time=times, sample_weight=weights,
        deephit_ranking_weight=config.model.deephit_ranking_weight,
        deephit_ranking_temperature=config.model.deephit_ranking_temperature,
    )
    cancer_scores = []
    for cancer in torch.unique(cancers):
        selected = cancers == cancer
        score = harrell_c_index(times[selected], events[selected], risks[selected])
        if math.isfinite(score):
            cancer_scores.append(score)
    return {
        "loss": float(loss),
        "c_index": (sum(cancer_scores) / len(cancer_scores)
                    if cancer_scores else float("nan")),
        "pooled_c_index": harrell_c_index(times, events, risks),
        "evaluated_cancers": len(cancer_scores),
        "patients": len(ordered),
    }


def _unwrap(model: nn.Module) -> UniPanSurv:
    return model.module if isinstance(model, DistributedDataParallel) else model  # type: ignore[return-value]


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    epoch: int,
    best_loss: float | None,
    config: RunConfig,
) -> None:
    torch.save(
        {
            "model": _unwrap(model).state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_validation_loss": best_loss,
            "config": config.to_dict(),
        },
        path,
    )


def _validate_settings(config: RunConfig, context: DistributedContext) -> None:
    opt = config.optimisation
    if opt.stage not in {"selection", "refit"}:
        raise ValueError("optimisation.stage must be 'selection' or 'refit'")
    if opt.stage == "selection" and config.data.validation is None:
        raise ValueError("selection requires a separate validation dataset")
    if opt.stage == "selection" and (opt.max_epochs <= 0 or opt.patience <= 0):
        raise ValueError("selection epoch limit and patience must be positive")
    if opt.stage == "refit":
        if config.data.validation is not None:
            raise ValueError("refit uses development training data without validation stopping")
        if config.runtime.resume is not None:
            raise ValueError("refit must start from fresh model parameters")
        if opt.refit_epochs <= 0:
            raise ValueError("refit_epochs must be positive")
    if opt.per_device_batch_size <= 0 or opt.gradient_accumulation_steps <= 0:
        raise ValueError("batch size and accumulation steps must be positive")
    actual = opt.per_device_batch_size * opt.gradient_accumulation_steps * context.world_size
    if opt.expected_global_batch_size is not None and actual != opt.expected_global_batch_size:
        raise ValueError(
            "effective global batch mismatch: "
            f"{opt.per_device_batch_size} x {context.world_size} x "
            f"{opt.gradient_accumulation_steps} = {actual}, expected "
            f"{opt.expected_global_batch_size}"
        )
    if config.model.relation_mode == "ppi" and config.data.ppi_edges is None:
        raise ValueError("PPI ablation requires data.ppi_edges")


def run_training(config: RunConfig) -> dict[str, Any]:
    context = initialise_distributed()
    try:
        _validate_settings(config, context)
        seed_everything(config.runtime.seed + context.rank, config.runtime.deterministic)
        output_dir = Path(config.runtime.output_dir)
        if context.is_main:
            output_dir.mkdir(parents=True, exist_ok=True)
            with (output_dir / "resolved_config.json").open("w", encoding="utf-8") as handle:
                json.dump(config.to_dict(), handle, indent=2, sort_keys=True)
        if context.distributed:
            dist.barrier()

        train_data = TensorSurvivalDataset(
            config.data.train, allowed_modalities=config.data.modalities
        )
        validation_data = (
            TensorSurvivalDataset(config.data.validation, allowed_modalities=config.data.modalities)
            if config.data.validation is not None else None
        )
        test_data = (
            TensorSurvivalDataset(config.data.test, allowed_modalities=config.data.modalities)
            if config.data.test is not None
            else None
        )
        train_loader, train_sampler = _loader(train_data, config, context, training=True)
        validation_loader = (
            _loader(validation_data, config, context, training=False)[0]
            if validation_data is not None else None
        )
        test_loader = (
            _loader(test_data, config, context, training=False)[0]
            if test_data is not None
            else None
        )
        training_counts = train_data.cancer_counts
        if training_counts.numel() != config.model.num_source_cancers:
            raise ValueError(
                "training dataset cancer IDs do not match model.num_source_cancers"
            )
        training_counts = training_counts.to(context.device)

        model = UniPanSurv(config.model).to(context.device)
        if config.model.relation_mode == "ppi":
            assert config.data.ppi_edges is not None  # checked by _validate_settings
            model.set_ppi_edges(load_ppi_edges(config.data.ppi_edges).to(context.device))
        # Validate one record before wrapping in DDP so input errors are immediate.
        first = train_data[0].patient.to(context.device)
        first.validate(
            num_features=config.model.num_features,
            num_omics=config.model.num_omics,
            num_clinical_features=config.model.num_clinical_features,
            num_source_cancers=config.model.num_source_cancers,
        )
        optimizer = AdamW(
            model.parameters(),
            lr=config.optimisation.learning_rate,
            weight_decay=config.optimisation.weight_decay,
        )
        start_epoch, best_loss = 0, math.inf
        if config.runtime.resume is not None:
            checkpoint = torch.load(
                config.runtime.resume, map_location=context.device, weights_only=False
            )
            model.load_state_dict(checkpoint["model"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            start_epoch = int(checkpoint["epoch"]) + 1
            best_loss = float(checkpoint["best_validation_loss"])
        if context.distributed:
            model = DistributedDataParallel(
                model,
                device_ids=[context.local_rank] if context.device.type == "cuda" else None,
                output_device=context.local_rank if context.device.type == "cuda" else None,
                find_unused_parameters=True,
            )
        scaler_enabled = context.device.type == "cuda" and config.runtime.amp == "fp16"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            scaler = torch.amp.GradScaler("cuda", enabled=scaler_enabled)
        else:  # pragma: no cover - compatibility with the oldest supported torch.
            scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        if context.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(context.device)
        history_path = output_dir / "history.jsonl"
        epochs_without_improvement = 0
        started = time.perf_counter()
        refit = config.optimisation.stage == "refit"
        epoch_limit = (config.optimisation.refit_epochs if refit
                       else config.optimisation.max_epochs)
        for epoch in range(start_epoch, epoch_limit):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)
            train_loss = train_epoch(
                model,
                train_loader,
                optimizer,
                scaler,
                config,
                context,
                training_counts,
            )
            validation = (
                evaluate(model, validation_loader, config, context, training_counts)
                if validation_loader is not None else None
            )
            improved = validation is not None and validation["loss"] < best_loss - 1e-8
            if improved:
                best_loss = validation["loss"]
                epochs_without_improvement = 0
                if context.is_main:
                    _save_checkpoint(
                        output_dir / "best.pt",
                        model,
                        optimizer,
                        epoch,
                        best_loss,
                        config,
                    )
            elif not refit:
                epochs_without_improvement += 1
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "stage": config.optimisation.stage,
                "validation_loss": None if validation is None else validation["loss"],
                "validation_c_index": None if validation is None else validation["c_index"],
                "best_validation_loss": None if refit else best_loss,
            }
            if context.is_main:
                with history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)
            stop = not refit and epochs_without_improvement >= config.optimisation.patience
            stop_tensor = torch.tensor(int(stop), device=context.device)
            if context.distributed:
                dist.broadcast(stop_tensor, src=0)
            if bool(stop_tensor.item()):
                break

        fit_wall_seconds = time.perf_counter() - started
        checkpoint_path = output_dir / ("final.pt" if refit else "best.pt")
        if refit and context.is_main:
            _save_checkpoint(checkpoint_path, model, optimizer, epoch, None, config)
        if context.distributed:
            dist.barrier()
        checkpoint = torch.load(
            checkpoint_path, map_location=context.device, weights_only=False
        )
        _unwrap(model).load_state_dict(checkpoint["model"])
        validation = (
            evaluate(model, validation_loader, config, context, training_counts)
            if validation_loader is not None else None
        )
        test = (
            evaluate(model, test_loader, config, context, training_counts)
            if test_loader is not None
            else None
        )
        elapsed = time.perf_counter() - started
        peak_memory = (
            torch.cuda.max_memory_allocated(context.device) if context.device.type == "cuda" else 0
        )
        local_profile = {"rank": context.rank, "peak_memory_bytes": peak_memory}
        if context.distributed:
            profiles: list[dict[str, int] | None] = [None] * context.world_size
            dist.all_gather_object(profiles, local_profile)
        else:
            profiles = [local_profile]
        summary = {
            "stage": config.optimisation.stage,
            "best_epoch": None if refit else int(checkpoint["epoch"]),
            "final_epoch": int(checkpoint["epoch"]),
            "epochs_completed": epoch + 1,
            "checkpoint": checkpoint_path.name,
            "validation": validation,
            "test": test,
            "wall_seconds": elapsed,
            "fit_wall_seconds": fit_wall_seconds,
            "world_size": context.world_size,
            "effective_global_batch_size": (
                config.optimisation.per_device_batch_size
                * context.world_size
                * config.optimisation.gradient_accumulation_steps
            ),
            "gpu_profiles": [profile for profile in profiles if profile is not None],
        }
        if context.is_main:
            with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
                json.dump(summary, handle, indent=2, sort_keys=True)
        return summary
    finally:
        if context.distributed and dist.is_initialized():
            dist.destroy_process_group()


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train UniPanSurv with optional DDP")
    parser.add_argument("--config", required=True, help="JSON training configuration")
    args = parser.parse_args(argv)
    run_training(RunConfig.from_json(args.config))


if __name__ == "__main__":
    main()
