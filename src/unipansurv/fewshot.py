from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path
from typing import Sequence

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from .config import UniPanSurvConfig
from .data import TensorSurvivalDataset, collate_survival_records
from .losses import survival_objective
from .model import UniPanSurv


def adapt_target(
    *,
    checkpoint_path: str | Path,
    data_path: str | Path,
    output_path: str | Path,
    steps: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    reference_penalty: float = 1e-3,
    batch_size: int = 8,
    seed: int = 2026,
    device: str | None = None,
) -> None:
    """Fit only the target context, rank-4 LoRA, 64-8-64 adapter and calibration."""

    if steps <= 0 or batch_size <= 0 or reference_penalty < 0:
        raise ValueError("invalid few-shot optimisation settings")
    random.seed(seed)
    torch.manual_seed(seed)
    selected_device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    checkpoint = torch.load(checkpoint_path, map_location=selected_device, weights_only=False)
    config = UniPanSurvConfig.from_dict(checkpoint["config"]["model"])
    model = UniPanSurv(config).to(selected_device)
    ppi_key = "relation_encoder.ppi_edge_codes"
    if ppi_key in checkpoint["model"] and checkpoint["model"][ppi_key].numel():
        codes = checkpoint["model"][ppi_key]
        edges = torch.stack(
            (
                torch.div(codes, config.num_features, rounding_mode="floor"),
                codes.remainder(config.num_features),
            ),
            dim=1,
        )
        model.set_ppi_edges(edges)
    model.load_state_dict(checkpoint["model"])
    model.freeze_backbone()
    adapter = model.new_target_adapter().to(selected_device)
    adapter.train()
    reference = {
        name: parameter.detach().clone()
        for name, parameter in adapter.named_parameters()
    }
    dataset = TensorSurvivalDataset(data_path)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_survival_records,
        generator=torch.Generator().manual_seed(seed),
    )
    optimizer = AdamW(adapter.parameters(), lr=learning_rate, weight_decay=weight_decay)
    iterator = itertools.cycle(loader)
    history: list[dict[str, float | int]] = []
    for step in range(steps):
        batch = next(iterator).to(selected_device)
        optimizer.zero_grad(set_to_none=True)
        output = model(batch.patients, mode="few_shot", target_adapter=adapter)
        data_loss = survival_objective(
            head=config.survival_head,
            objective_logits=output.objective_logits,
            hazard_logits=output.hazard_logits,
            interval_index=batch.interval_index,
            event=batch.event,
            observed_time=batch.observed_time,
            deephit_ranking_weight=config.deephit_ranking_weight,
            deephit_ranking_temperature=config.deephit_ranking_temperature,
        )
        penalty = data_loss.new_zeros(())
        for name, parameter in adapter.named_parameters():
            penalty = penalty + (parameter - reference[name]).square().sum()
        loss = data_loss + reference_penalty * penalty
        loss.backward()
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
        optimizer.step()
        history.append(
            {
                "step": step + 1,
                "loss": float(loss.detach()),
                "survival_loss": float(data_loss.detach()),
                "reference_penalty": float(penalty.detach()),
            }
        )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "adapter": adapter.state_dict(),
            "source_checkpoint": str(Path(checkpoint_path).resolve()),
            "model_config": config.to_dict(),
            "steps": steps,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "reference_penalty": reference_penalty,
            "seed": seed,
        },
        output_path,
    )
    with output_path.with_suffix(".history.json").open("w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Few-shot target-cancer adaptation")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--reference-penalty", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device")
    args = parser.parse_args(argv)
    adapt_target(
        checkpoint_path=args.checkpoint,
        data_path=args.data,
        output_path=args.output,
        steps=args.steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        reference_penalty=args.reference_penalty,
        batch_size=args.batch_size,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
