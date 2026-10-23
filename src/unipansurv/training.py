from __future__ import annotations

import argparse
import contextlib
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
    validation: str
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
    for step, cpu_batch in enumerate(loader):
        batch = cpu_batch.to(context.device)
        group_start = (step // accumulation) * accumulation
        group_size = min(accumulation, len(loader) - group_start)
        synchronize = (step + 1) % accumulation == 0 or step + 1 == len(loader)
        sync_context = (
            contextlib.nullcontext()
            if synchronize or not isinstance(model, DistributedDataParallel)
            else model.no_sync()
        )
        with sync_context:
            with _autocast(context, config.runtime.amp):
                loss, _ = _model_loss(model, batch, config, training_counts)
                scaled_loss = loss / group_size
            scaler.scale(scaled_loss).backward()
        if synchronize:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.optimisation.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        total_loss[0] += loss.detach() * len(batch.patients)
        total_loss[1] += len(batch.patients)
    if context.distributed:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
    return float((total_loss[0] / total_loss[1].clamp_min(1)).item())


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader[SurvivalBatch],
    config: RunConfig,
    context: DistributedContext,
    training_counts: Tensor,
) -> dict[str, float]:
    model.eval()
    total_loss = torch.zeros(2, device=context.device)
    local_rows: list[tuple[str, float, float, float]] = []
    for cpu_batch in loader:
        batch = cpu_batch.to(context.device)
        with _autocast(context, config.runtime.amp):
            loss, risk = _model_loss(model, batch, config, training_counts)
        total_loss[0] += loss * len(batch.patients)
        total_loss[1] += len(batch.patients)
        local_rows.extend(
            zip(
                batch.patient_id,
                batch.observed_time.cpu().tolist(),
                batch.event.cpu().tolist(),
                risk.float().cpu().tolist(),
            )
        )
    if context.distributed:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        gathered: list[list[tuple[str, float, float, float]] | None] = [
            None for _ in range(context.world_size)
        ]
        dist.all_gather_object(gathered, local_rows)
        rows = [row for partition in gathered if partition is not None for row in partition]
    else:
        rows = local_rows
    # DistributedSampler can pad the validation set.  Deduplicate by patient ID.
    unique = {row[0]: row for row in rows}
    ordered = list(unique.values())
    times = torch.tensor([row[1] for row in ordered])
    events = torch.tensor([row[2] for row in ordered])
    risks = torch.tensor([row[3] for row in ordered])
    return {
        "loss": float((total_loss[0] / total_loss[1].clamp_min(1)).item()),
        "c_index": harrell_c_index(times, events, risks),
    }


def _unwrap(model: nn.Module) -> UniPanSurv:
    return model.module if isinstance(model, DistributedDataParallel) else model  # type: ignore[return-value]


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: AdamW,
    epoch: int,
    best_loss: float,
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
        validation_data = TensorSurvivalDataset(
            config.data.validation, allowed_modalities=config.data.modalities
        )
        test_data = (
            TensorSurvivalDataset(config.data.test, allowed_modalities=config.data.modalities)
            if config.data.test is not None
            else None
        )
        train_loader, train_sampler = _loader(train_data, config, context, training=True)
        validation_loader, _ = _loader(validation_data, config, context, training=False)
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
        for epoch in range(start_epoch, config.optimisation.max_epochs):
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
            validation = evaluate(
                model, validation_loader, config, context, training_counts
            )
            improved = validation["loss"] < best_loss - 1e-8
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
            else:
                epochs_without_improvement += 1
            row = {
                "epoch": epoch,
                "train_loss": train_loss,
                "validation_loss": validation["loss"],
                "validation_c_index": validation["c_index"],
                "best_validation_loss": best_loss,
            }
            if context.is_main:
                with history_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, sort_keys=True) + "\n")
                print(json.dumps(row, sort_keys=True), flush=True)
            stop = epochs_without_improvement >= config.optimisation.patience
            stop_tensor = torch.tensor(int(stop), device=context.device)
            if context.distributed:
                dist.broadcast(stop_tensor, src=0)
            if bool(stop_tensor.item()):
                break

        if context.distributed:
            dist.barrier()
        checkpoint = torch.load(
            output_dir / "best.pt", map_location=context.device, weights_only=False
        )
        _unwrap(model).load_state_dict(checkpoint["model"])
        validation = evaluate(model, validation_loader, config, context, training_counts)
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
            "best_epoch": int(checkpoint["epoch"]),
            "validation": validation,
            "test": test,
            "wall_seconds": elapsed,
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
