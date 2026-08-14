from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .config import UniPanSurvConfig


class ValueEncoder(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, values: Tensor) -> Tensor:
        return self.network(values.unsqueeze(-1))


class MolecularTokenizer(nn.Module):
    """Build 64-D tokens from identity, assay type and patient value."""

    def __init__(self, config: UniPanSurvConfig) -> None:
        super().__init__()
        d = config.hidden_dim
        self.feature_embedding = nn.Embedding(config.num_features, d)
        self.omics_embedding = nn.Embedding(config.num_omics, d)
        self.value_encoder = ValueEncoder(d, config.dropout)
        self.normalisation = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, feature_ids: Tensor, omics_ids: Tensor, values: Tensor) -> Tensor:
        token = (
            self.feature_embedding(feature_ids)
            + self.omics_embedding(omics_ids)
            + self.value_encoder(values)
        )
        return self.dropout(self.normalisation(token))


class GatedSetPool(nn.Module):
    """Permutation-invariant sigmoid-gated mean pooling."""

    def __init__(self, hidden_dim: int, num_queries: int = 1) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_queries = num_queries
        self.queries = nn.Parameter(torch.empty(num_queries, hidden_dim))
        self.bias = nn.Parameter(torch.zeros(num_queries))
        nn.init.normal_(self.queries, std=hidden_dim**-0.5)

    def forward(self, values: Tensor, mask: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        if values.ndim != 2 or values.size(-1) != self.hidden_dim:
            raise ValueError("values must have shape [items, hidden_dim]")
        if mask is None:
            mask = torch.ones(values.size(0), dtype=torch.bool, device=values.device)
        else:
            mask = mask.to(dtype=torch.bool, device=values.device)
        selected = values[mask]
        if selected.numel() == 0:
            pooled = values.new_zeros(self.num_queries, self.hidden_dim)
            valid = torch.zeros(self.num_queries, dtype=torch.bool, device=values.device)
            return pooled, valid
        logits = selected @ self.queries.T / math.sqrt(self.hidden_dim) + self.bias
        weights = torch.sigmoid(logits)
        numerator = weights.T @ selected
        denominator = weights.sum(dim=0).unsqueeze(-1).clamp_min(1e-8)
        pooled = numerator / denominator
        valid = torch.ones(self.num_queries, dtype=torch.bool, device=values.device)
        return pooled, valid


class LoRALinearDelta(nn.Module):
    """A zero-initialised low-rank additive update for a linear projection."""

    def __init__(self, in_features: int, out_features: int, rank: int) -> None:
        super().__init__()
        self.rank = rank
        self.down = nn.Linear(in_features, rank, bias=False)
        self.up = nn.Linear(rank, out_features, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)
        self.scale = 1.0 / rank

    def forward(self, x: Tensor) -> Tensor:
        return self.up(self.down(x)) * self.scale


class RelationLoRA(nn.Module):
    """Target-specific rank-r updates for relation query/key projections."""

    def __init__(self, hidden_dim: int, rank: int) -> None:
        super().__init__()
        self.query = LoRALinearDelta(hidden_dim, hidden_dim, rank)
        self.key = LoRALinearDelta(hidden_dim, hidden_dim, rank)


class RelationEncoder(nn.Module):
    """Score and pool every observed directed within-patient molecular pair.

    Pairs are enumerated by directed omics block and processed in chunks. The
    molecular tokens are read-only: this layer returns relation summaries and
    never writes aggregated messages back into the tokens.
    """

    def __init__(self, config: UniPanSurvConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_dim
        h = config.num_relation_heads

        self.query_projection = nn.Linear(d, d, bias=False)
        self.key_projection = nn.Linear(d, d, bias=False)
        self.context_to_query = nn.Linear(d, d, bias=False)
        self.context_to_key = nn.Linear(d, d, bias=False)
        initial_scale = 1.0 / math.sqrt(config.relation_head_dim)
        self.log_scale = nn.Parameter(torch.full((h,), math.log(initial_scale)))
        self.omics_pair_bias = nn.Parameter(torch.zeros(h, config.num_omics, config.num_omics))
        self.relation_type_embedding = nn.Embedding(config.num_omics * config.num_omics, d)

        edge_input_dim = 3 * d + h
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_input_dim, d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, d),
            nn.LayerNorm(d),
        )

        self.readout_queries = nn.Parameter(
            torch.empty(config.num_omics, config.num_omics, config.num_relation_queries, d)
        )
        self.readout_bias = nn.Parameter(
            torch.zeros(config.num_omics, config.num_omics, config.num_relation_queries)
        )
        nn.init.normal_(self.readout_queries, std=d**-0.5)

    def _project(
        self,
        tokens: Tensor,
        context: Tensor,
        lora: Optional[RelationLoRA],
    ) -> tuple[Tensor, Tensor]:
        q = self.query_projection(tokens) + self.context_to_query(context).unsqueeze(0)
        k = self.key_projection(tokens) + self.context_to_key(context).unsqueeze(0)
        if lora is not None:
            q = q + lora.query(tokens)
            k = k + lora.key(tokens)
        shape = (-1, self.config.num_relation_heads, self.config.relation_head_dim)
        return q.reshape(shape), k.reshape(shape)

    def _edge_chunk(
        self,
        source_tokens: Tensor,
        target_tokens: Tensor,
        source_q: Tensor,
        target_k: Tensor,
        source_omics: int,
        target_omics: int,
    ) -> Tensor:
        scores = torch.einsum("shd,thd->sth", source_q, target_k)
        scores = scores * self.log_scale.exp().view(1, 1, -1)
        scores = scores + self.omics_pair_bias[:, source_omics, target_omics].view(1, 1, -1)

        ns, nt = scores.shape[:2]
        source = source_tokens[:, None, :].expand(ns, nt, -1)
        target = target_tokens[None, :, :].expand(ns, nt, -1)
        relation_id = source_omics * self.config.num_omics + target_omics
        relation_type = self.relation_type_embedding.weight[relation_id]
        relation_type = relation_type.view(1, 1, -1).expand(ns, nt, -1)
        edge_input = torch.cat((source, target, scores, relation_type), dim=-1)
        return self.edge_encoder(edge_input)

    def forward(
        self,
        tokens: Tensor,
        omics_ids: Tensor,
        context: Tensor,
        lora: Optional[RelationLoRA] = None,
    ) -> tuple[Tensor, Tensor]:
        cfg = self.config
        q, k = self._project(tokens, context, lora)
        block_outputs: list[Tensor] = []
        block_masks: list[Tensor] = []
        chunk = cfg.pair_chunk_size

        for source_omics in range(cfg.num_omics):
            source_index = torch.nonzero(omics_ids == source_omics, as_tuple=False).flatten()
            for target_omics in range(cfg.num_omics):
                target_index = torch.nonzero(omics_ids == target_omics, as_tuple=False).flatten()
                numerator = tokens.new_zeros(cfg.num_relation_queries, cfg.hidden_dim)
                denominator = tokens.new_zeros(cfg.num_relation_queries)
                pair_count = 0
                query = self.readout_queries[source_omics, target_omics]
                query_bias = self.readout_bias[source_omics, target_omics]

                for s0 in range(0, source_index.numel(), chunk):
                    src_idx = source_index[s0 : s0 + chunk]
                    for t0 in range(0, target_index.numel(), chunk):
                        tgt_idx = target_index[t0 : t0 + chunk]
                        if src_idx.numel() == 0 or tgt_idx.numel() == 0:
                            continue
                        edge = self._edge_chunk(
                            tokens[src_idx],
                            tokens[tgt_idx],
                            q[src_idx],
                            k[tgt_idx],
                            source_omics,
                            target_omics,
                        )
                        valid = src_idx[:, None] != tgt_idx[None, :]
                        if not torch.any(valid):
                            continue
                        edge = edge[valid]
                        gates = torch.sigmoid(
                            edge @ query.T / math.sqrt(cfg.hidden_dim) + query_bias
                        )
                        numerator = numerator + gates.T @ edge
                        denominator = denominator + gates.sum(dim=0)
                        pair_count += int(valid.sum().item())

                if pair_count == 0:
                    summaries = tokens.new_zeros(cfg.num_relation_queries, cfg.hidden_dim)
                    valid_queries = torch.zeros(
                        cfg.num_relation_queries, dtype=torch.bool, device=tokens.device
                    )
                else:
                    summaries = numerator / denominator.unsqueeze(-1).clamp_min(cfg.eps)
                    valid_queries = torch.ones(
                        cfg.num_relation_queries, dtype=torch.bool, device=tokens.device
                    )
                block_outputs.append(summaries)
                block_masks.append(valid_queries)

        return torch.cat(block_outputs, dim=0), torch.cat(block_masks, dim=0)


class ClinicalEncoder(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2 * num_features, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, values: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        mask = mask.to(dtype=torch.bool, device=values.device)
        neutral_values = torch.where(mask, values, torch.zeros_like(values))
        encoded = self.network(torch.cat((neutral_values, mask.to(values.dtype)), dim=-1))
        return encoded, mask.any()


class GatedFusion(nn.Module):
    """Fuse relation, omics and clinical evidence with masked softmax gates."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.transforms = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(3)
            ]
        )
        self.group_embedding = nn.Parameter(torch.empty(3, hidden_dim))
        nn.init.normal_(self.group_embedding, std=hidden_dim**-0.5)
        self.gates = nn.ModuleList([nn.Linear(hidden_dim, 1) for _ in range(3)])
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, groups: Tensor, group_mask: Tensor) -> Tensor:
        transformed = torch.stack(
            [layer(groups[index]) for index, layer in enumerate(self.transforms)], dim=0
        )
        transformed = transformed + self.group_embedding
        logits = torch.cat(
            [gate(transformed[index]) for index, gate in enumerate(self.gates)], dim=0
        )
        logits = logits.masked_fill(~group_mask, torch.finfo(logits.dtype).min)
        weights = torch.softmax(logits, dim=0)
        return self.output_norm((weights.unsqueeze(-1) * transformed).sum(dim=0))


class PatientAdapter(nn.Module):
    """Zero-initialised residual adapter used only for few-shot adaptation."""

    def __init__(self, hidden_dim: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(hidden_dim, rank)
        self.up = nn.Linear(rank, hidden_dim)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, patient: Tensor) -> Tensor:
        return patient + self.up(F.gelu(self.down(patient)))


class DiscreteTimeSurvivalHead(nn.Module):
    """Map one patient representation to interval hazards and survival."""

    def __init__(self, config: UniPanSurvConfig) -> None:
        super().__init__()
        d = config.hidden_dim
        self.num_intervals = config.num_time_intervals
        self.time_embedding = nn.Embedding(config.num_time_intervals, config.time_embedding_dim)
        self.patient_projection = nn.Linear(d, d)
        self.time_projection = nn.Linear(config.time_embedding_dim, d)
        self.output = nn.Linear(d, 1)
        self.shared_baseline = nn.Parameter(torch.zeros(config.num_time_intervals))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, patient: Tensor, calibration: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        time = self.time_embedding.weight
        hidden = torch.tanh(
            self.patient_projection(patient).unsqueeze(0) + self.time_projection(time)
        )
        logits = self.output(self.dropout(hidden)).squeeze(-1)
        logits = logits + self.shared_baseline + calibration
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1.0 - hazards, dim=-1)
        risk = 1.0 - survival
        return logits, hazards, survival, risk

