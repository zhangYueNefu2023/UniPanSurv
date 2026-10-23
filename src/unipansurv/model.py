from __future__ import annotations

from typing import Literal, Optional, Sequence

import torch
from torch import Tensor, nn

from .adaptation import TargetAdapter
from .config import UniPanSurvConfig
from .layers import (
    ClinicalEncoder,
    GatedFusion,
    GatedSetPool,
    MolecularTokenizer,
    RelationEncoder,
    build_survival_head,
)
from .types import PatientInput, UniPanSurvOutput


PredictionMode = Literal["source", "zero_shot", "few_shot"]


class UniPanSurv(nn.Module):
    """Shared pan-cancer survival model with zero- and few-shot operation."""

    def __init__(self, config: UniPanSurvConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_dim

        self.tokenizer = MolecularTokenizer(config)
        self.cancer_context = nn.Embedding(config.num_cancer_contexts, d)
        self.relation_encoder = RelationEncoder(config)
        self.omics_pool = GatedSetPool(d, num_queries=1)
        self.relation_group_pool = GatedSetPool(d, num_queries=1)
        self.omics_group_pool = GatedSetPool(d, num_queries=1)
        self.clinical_encoder = ClinicalEncoder(
            config.num_clinical_features, d, config.dropout
        )
        self.fusion = GatedFusion(d, config.dropout)
        self.survival_head = build_survival_head(config)
        self.risk_calibration = nn.Embedding(
            config.num_cancer_contexts, config.num_time_intervals
        )
        nn.init.zeros_(self.risk_calibration.weight)

    def new_target_adapter(self) -> TargetAdapter:
        """Initialise a new-cancer adapter from the learned Unknown state."""

        unknown = self.config.unknown_cancer_index
        return TargetAdapter(
            self.config,
            self.cancer_context.weight[unknown],
            self.risk_calibration.weight[unknown],
        ).to(self.cancer_context.weight.device)

    def set_ppi_edges(self, edges: Tensor) -> None:
        """Set assay-qualified directed feature pairs for the PPI ablation."""

        self.relation_encoder.set_ppi_edges(edges)

    def freeze_backbone(self) -> None:
        """Freeze all source-trained parameters before few-shot adaptation."""

        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def unfreeze_backbone(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad_(True)

    def _observed_tokens(self, patient: PatientInput) -> tuple[Tensor, Tensor, Tensor]:
        n = patient.feature_ids.numel()
        if patient.molecular_mask is None:
            observed = torch.ones(n, dtype=torch.bool, device=patient.feature_ids.device)
        else:
            observed = patient.molecular_mask.to(
                dtype=torch.bool, device=patient.feature_ids.device
            )

        if self.training and self.config.modality_dropout_probability > 0:
            present = torch.unique(patient.omics_ids[observed])
            if present.numel() > 1:
                drop = torch.rand(present.numel(), device=present.device) < self.config.modality_dropout_probability
                if torch.all(drop):
                    drop[torch.randint(present.numel(), (1,), device=present.device)] = False
                for omics_id, should_drop in zip(present, drop):
                    if bool(should_drop):
                        observed = observed & (patient.omics_ids != omics_id)

        if not torch.any(observed):
            raise ValueError("no observed molecular tokens remain for this patient")
        return (
            patient.feature_ids[observed].long(),
            patient.omics_ids[observed].long(),
            patient.values[observed].to(dtype=self.cancer_context.weight.dtype),
        )

    def _source_context_index(self, patient: PatientInput) -> int:
        if patient.cancer_id is None:
            raise ValueError("source mode requires patient.cancer_id")
        use_unknown = (
            self.training
            and self.config.unknown_context_probability > 0
            and torch.rand((), device=self.cancer_context.weight.device)
            < self.config.unknown_context_probability
        )
        return self.config.unknown_cancer_index if bool(use_unknown) else patient.cancer_id

    def _context_and_calibration(
        self,
        patient: PatientInput,
        mode: PredictionMode,
        target_adapter: Optional[TargetAdapter],
    ) -> tuple[Tensor, Tensor]:
        if mode == "source":
            index = self._source_context_index(patient)
            return self.cancer_context.weight[index], self.risk_calibration.weight[index]
        if mode == "zero_shot":
            index = self.config.unknown_cancer_index
            return self.cancer_context.weight[index], self.risk_calibration.weight[index]
        if mode == "few_shot":
            if target_adapter is None:
                raise ValueError("few_shot mode requires target_adapter")
            return target_adapter.context, target_adapter.risk_calibration
        raise ValueError(f"unsupported prediction mode: {mode}")

    def _encode_one(
        self,
        patient: PatientInput,
        mode: PredictionMode,
        target_adapter: Optional[TargetAdapter],
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        cfg = self.config
        patient.validate(
            num_features=cfg.num_features,
            num_omics=cfg.num_omics,
            num_clinical_features=cfg.num_clinical_features,
            num_source_cancers=cfg.num_source_cancers,
            validate_cancer_id=mode == "source",
        )
        feature_ids, omics_ids, values = self._observed_tokens(patient)
        tokens = self.tokenizer(feature_ids, omics_ids, values)
        context, calibration = self._context_and_calibration(patient, mode, target_adapter)

        lora = None if target_adapter is None or mode != "few_shot" else target_adapter.relation_lora
        if cfg.relation_mode == "none":
            relation_summaries = tokens.new_zeros(
                cfg.num_omics * cfg.num_omics * cfg.num_relation_queries,
                cfg.hidden_dim,
            )
            relation_mask = torch.zeros(
                relation_summaries.size(0), dtype=torch.bool, device=tokens.device
            )
            pooling_tokens = tokens
        else:
            relation_summaries, relation_mask, pooling_tokens = self.relation_encoder(
                tokens, feature_ids, omics_ids, context, lora=lora
            )

        omics_summaries = tokens.new_zeros(cfg.num_omics, cfg.hidden_dim)
        omics_mask = torch.zeros(cfg.num_omics, dtype=torch.bool, device=tokens.device)
        for omics_id in range(cfg.num_omics):
            selected = pooling_tokens[omics_ids == omics_id]
            pooled, valid = self.omics_pool(selected)
            omics_summaries[omics_id] = pooled[0]
            omics_mask[omics_id] = valid[0]

        relation_group, relation_group_valid = self.relation_group_pool(
            relation_summaries, relation_mask
        )
        omics_group, omics_group_valid = self.omics_group_pool(
            omics_summaries, omics_mask
        )
        clinical_group, clinical_valid = self.clinical_encoder(
            patient.clinical_values.to(tokens.dtype), patient.clinical_mask
        )
        groups = torch.stack((relation_group[0], omics_group[0], clinical_group), dim=0)
        group_mask = torch.stack(
            (relation_group_valid[0], omics_group_valid[0], clinical_valid), dim=0
        )
        representation = self.fusion(groups, group_mask)
        if mode == "few_shot" and target_adapter is not None:
            representation = target_adapter.patient_adapter(representation)
        (
            objective_logits,
            hazard_logits,
            hazards,
            survival,
            risk,
            risk_score,
        ) = self.survival_head(representation, calibration)
        return (
            objective_logits,
            hazard_logits,
            hazards,
            survival,
            risk,
            risk_score,
            representation,
            relation_summaries,
            relation_mask,
            omics_summaries,
            omics_mask,
        )

    def forward(
        self,
        patients: Sequence[PatientInput],
        *,
        mode: PredictionMode = "source",
        target_adapter: Optional[TargetAdapter] = None,
    ) -> UniPanSurvOutput:
        if len(patients) == 0:
            raise ValueError("patients cannot be empty")
        encoded = [self._encode_one(patient, mode, target_adapter) for patient in patients]
        stacked = [torch.stack(items, dim=0) for items in zip(*encoded)]
        return UniPanSurvOutput(
            objective_logits=stacked[0],
            hazard_logits=stacked[1],
            hazards=stacked[2],
            survival=stacked[3],
            risk=stacked[4],
            risk_score=stacked[5].squeeze(-1),
            patient_representation=stacked[6],
            relation_summaries=stacked[7],
            relation_mask=stacked[8],
            omics_summaries=stacked[9],
            omics_mask=stacked[10],
        )

