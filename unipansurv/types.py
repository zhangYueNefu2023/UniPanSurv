from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor


@dataclass
class PatientInput:
    """One patient's model-ready measurements.

    Molecular values must already be transformed and standardised with
    training-partition statistics. Missing molecular measurements should be
    removed or marked ``False`` in ``molecular_mask``.
    """

    feature_ids: Tensor
    omics_ids: Tensor
    values: Tensor
    clinical_values: Tensor
    clinical_mask: Tensor
    cancer_id: Optional[int] = None
    molecular_mask: Optional[Tensor] = None

    def validate(
        self,
        *,
        num_features: int,
        num_omics: int,
        num_clinical_features: int,
        num_source_cancers: int,
    ) -> None:
        n = self.feature_ids.numel()
        if self.feature_ids.ndim != 1 or self.omics_ids.ndim != 1 or self.values.ndim != 1:
            raise ValueError("feature_ids, omics_ids and values must be one-dimensional")
        if self.omics_ids.numel() != n or self.values.numel() != n:
            raise ValueError("molecular arrays must have the same length")
        if self.molecular_mask is not None:
            if self.molecular_mask.ndim != 1 or self.molecular_mask.numel() != n:
                raise ValueError("molecular_mask must match the molecular arrays")
        if self.clinical_values.shape != (num_clinical_features,):
            raise ValueError("clinical_values has the wrong shape")
        if self.clinical_mask.shape != (num_clinical_features,):
            raise ValueError("clinical_mask has the wrong shape")
        if n == 0:
            raise ValueError("each patient must contain at least one molecular token")
        if torch.any(self.feature_ids < 0) or torch.any(self.feature_ids >= num_features):
            raise ValueError("feature_ids contains an out-of-range identifier")
        if torch.any(self.omics_ids < 0) or torch.any(self.omics_ids >= num_omics):
            raise ValueError("omics_ids contains an out-of-range identifier")
        if self.cancer_id is not None and not 0 <= self.cancer_id < num_source_cancers:
            raise ValueError("cancer_id must identify a fitted source cancer")

    def to(self, device: torch.device | str) -> "PatientInput":
        return PatientInput(
            feature_ids=self.feature_ids.to(device),
            omics_ids=self.omics_ids.to(device),
            values=self.values.to(device),
            clinical_values=self.clinical_values.to(device),
            clinical_mask=self.clinical_mask.to(device),
            cancer_id=self.cancer_id,
            molecular_mask=None if self.molecular_mask is None else self.molecular_mask.to(device),
        )


@dataclass
class UniPanSurvOutput:
    """Predicted hazards, survival probabilities and compact model summaries."""

    hazard_logits: Tensor
    hazards: Tensor
    survival: Tensor
    risk: Tensor
    patient_representation: Tensor
    relation_summaries: Tensor
    relation_mask: Tensor
    omics_summaries: Tensor
    omics_mask: Tensor

