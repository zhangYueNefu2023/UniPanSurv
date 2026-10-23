from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import torch

from .data import write_tensor_dataset


def _records(
    count: int,
    *,
    split: str,
    generator: torch.Generator,
    num_features: int,
    num_cancers: int,
    num_intervals: int,
    tokens_per_patient: int,
) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for index in range(count):
        cancer_id = index % num_cancers
        feature_ids = torch.randperm(num_features, generator=generator)[:tokens_per_patient]
        omics_ids = torch.arange(tokens_per_patient).remainder(3)
        order = torch.randperm(tokens_per_patient, generator=generator)
        feature_ids, omics_ids = feature_ids[order], omics_ids[order]
        values = torch.randn(tokens_per_patient, generator=generator)
        clinical = torch.randn(3, generator=generator)
        linear_risk = values[: min(4, tokens_per_patient)].mean() + 0.2 * clinical[0]
        event_time = torch.exp(-linear_risk) * (4.0 + 8.0 * torch.rand((), generator=generator))
        censor_time = 2.0 + 12.0 * torch.rand((), generator=generator)
        event = float(event_time <= censor_time)
        observed_time = float(torch.minimum(event_time, censor_time))
        interval_index = min(int(observed_time // 1.0), num_intervals - 1)
        records.append(
            {
                "feature_ids": feature_ids,
                "omics_ids": omics_ids,
                "values": values,
                "clinical_values": clinical,
                "clinical_mask": torch.ones(3, dtype=torch.bool),
                "cancer_id": cancer_id,
                "observed_time": observed_time,
                "interval_index": interval_index,
                "event": event,
                "patient_id": f"{split}-{index:05d}",
            }
        )
    return records


def generate_synthetic_dataset(
    output_dir: str | Path,
    *,
    seed: int = 17,
    num_features: int = 60,
    num_cancers: int = 3,
    num_intervals: int = 15,
    tokens_per_patient: int = 12,
    train_size: int = 48,
    validation_size: int = 18,
    test_size: int = 18,
) -> None:
    output_dir = Path(output_dir)
    generator = torch.Generator().manual_seed(seed)
    metadata = {
        "num_features": num_features,
        "num_cancers": num_cancers,
        "num_intervals": num_intervals,
        "omics_names": ["RNA", "miRNA", "protein"],
        "clinical_names": ["age", "sex", "stage"],
        "synthetic": True,
    }
    for split, count in (
        ("train", train_size),
        ("validation", validation_size),
        ("test", test_size),
    ):
        write_tensor_dataset(
            output_dir / f"{split}.pt",
            _records(
                count,
                split=split,
                generator=generator,
                num_features=num_features,
                num_cancers=num_cancers,
                num_intervals=num_intervals,
                tokens_per_patient=tokens_per_patient,
            ),
            metadata,
        )
    # A tiny directed protein edge list for testing the PPI control.
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "ppi_edges.tsv").open("w", encoding="utf-8") as handle:
        handle.write("source_feature_id\ttarget_feature_id\n")
        for source in range(0, min(num_features - 1, 20), 2):
            handle.write(f"{source}\t{source + 1}\n")
            handle.write(f"{source + 1}\t{source}\n")


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Generate a small runnable example dataset")
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args(argv)
    generate_synthetic_dataset(args.output_dir, seed=args.seed)


if __name__ == "__main__":
    main()
