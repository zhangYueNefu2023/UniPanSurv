from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .types import PatientInput


SCHEMA_VERSION = 1


@dataclass
class SurvivalRecord:
    patient: PatientInput
    observed_time: float
    interval_index: int
    event: float
    patient_id: str


@dataclass
class SurvivalBatch:
    patients: list[PatientInput]
    observed_time: Tensor
    interval_index: Tensor
    event: Tensor
    cancer_id: Tensor
    patient_id: list[str]

    def to(self, device: torch.device | str) -> "SurvivalBatch":
        return SurvivalBatch(
            patients=[patient.to(device) for patient in self.patients],
            observed_time=self.observed_time.to(device),
            interval_index=self.interval_index.to(device),
            event=self.event.to(device),
            cancer_id=self.cancer_id.to(device),
            patient_id=self.patient_id,
        )


class TensorSurvivalDataset(Dataset[SurvivalRecord]):
    """Validated model-ready survival dataset stored as a safe tensor payload."""

    def __init__(
        self,
        path: str | Path,
        *,
        allowed_modalities: Sequence[int] | None = None,
    ) -> None:
        self.path = Path(path)
        try:
            payload = torch.load(self.path, map_location="cpu", weights_only=True)
        except TypeError:  # PyTorch 2.2 compatibility.
            payload = torch.load(self.path, map_location="cpu")
        if not isinstance(payload, dict):
            raise ValueError(f"{self.path}: expected a dictionary payload")
        if payload.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"{self.path}: unsupported schema_version; expected {SCHEMA_VERSION}"
            )
        self.metadata = dict(payload.get("metadata", {}))
        raw_records = payload.get("records")
        if not isinstance(raw_records, list) or not raw_records:
            raise ValueError(f"{self.path}: records must be a non-empty list")
        self.allowed_modalities = (
            None
            if allowed_modalities is None
            else torch.tensor(sorted(set(allowed_modalities)), dtype=torch.long)
        )
        self.records = [self._decode(record, index) for index, record in enumerate(raw_records)]

    def _decode(self, record: Any, index: int) -> SurvivalRecord:
        if not isinstance(record, dict):
            raise ValueError(f"{self.path}: record {index} is not a dictionary")
        required = {
            "feature_ids",
            "omics_ids",
            "values",
            "clinical_values",
            "clinical_mask",
            "cancer_id",
            "observed_time",
            "interval_index",
            "event",
            "patient_id",
        }
        missing = required.difference(record)
        if missing:
            raise ValueError(f"{self.path}: record {index} missing {sorted(missing)}")
        feature_ids = torch.as_tensor(record["feature_ids"], dtype=torch.long)
        omics_ids = torch.as_tensor(record["omics_ids"], dtype=torch.long)
        values = torch.as_tensor(record["values"], dtype=torch.float32)
        molecular_mask = record.get("molecular_mask")
        if molecular_mask is not None:
            molecular_mask = torch.as_tensor(molecular_mask, dtype=torch.bool)
        if self.allowed_modalities is not None:
            keep = torch.zeros_like(omics_ids, dtype=torch.bool)
            for modality in self.allowed_modalities:
                keep |= omics_ids == modality
            if molecular_mask is not None:
                keep &= molecular_mask
            feature_ids, omics_ids, values = (
                feature_ids[keep],
                omics_ids[keep],
                values[keep],
            )
            molecular_mask = None
        if feature_ids.numel() == 0:
            raise ValueError(
                f"{self.path}: modality selection removed every token in record {index}"
            )
        patient = PatientInput(
            feature_ids=feature_ids,
            omics_ids=omics_ids,
            values=values,
            clinical_values=torch.as_tensor(record["clinical_values"], dtype=torch.float32),
            clinical_mask=torch.as_tensor(record["clinical_mask"], dtype=torch.bool),
            cancer_id=int(record["cancer_id"]),
            molecular_mask=molecular_mask,
        )
        observed_time = float(record["observed_time"])
        interval_index = int(record["interval_index"])
        event = float(record["event"])
        if observed_time < 0 or event not in {0.0, 1.0}:
            raise ValueError(f"{self.path}: invalid survival outcome in record {index}")
        return SurvivalRecord(
            patient=patient,
            observed_time=observed_time,
            interval_index=interval_index,
            event=event,
            patient_id=str(record["patient_id"]),
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> SurvivalRecord:
        return self.records[index]

    @property
    def cancer_counts(self) -> Tensor:
        cancer_ids = torch.tensor(
            [record.patient.cancer_id for record in self.records], dtype=torch.long
        )
        return torch.bincount(cancer_ids, minlength=int(cancer_ids.max().item()) + 1)


def collate_survival_records(records: Iterable[SurvivalRecord]) -> SurvivalBatch:
    records = list(records)
    if not records:
        raise ValueError("cannot collate an empty batch")
    return SurvivalBatch(
        patients=[record.patient for record in records],
        observed_time=torch.tensor(
            [record.observed_time for record in records], dtype=torch.float32
        ),
        interval_index=torch.tensor(
            [record.interval_index for record in records], dtype=torch.long
        ),
        event=torch.tensor([record.event for record in records], dtype=torch.float32),
        cancer_id=torch.tensor(
            [record.patient.cancer_id for record in records], dtype=torch.long
        ),
        patient_id=[record.patient_id for record in records],
    )


def write_tensor_dataset(
    path: str | Path,
    records: list[dict[str, Any]],
    metadata: dict[str, Any],
) -> None:
    """Write the documented model-ready format used by the CLI."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "schema_version": SCHEMA_VERSION,
            "metadata": metadata,
            "records": records,
        },
        path,
    )
