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
    default model keeps molecular tokens read-only.  The explicit ``writeback``
    ablation additionally returns tokens updated with aggregated edge messages.
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
        self.score_mlp = nn.Sequential(
            nn.Linear(3 * d, d),
            nn.GELU(),
            nn.Linear(d, h),
        )
        self.writeback_projection = nn.Linear(d, d)
        self.writeback_norm = nn.LayerNorm(d)
        self.register_buffer(
            "ppi_edge_codes", torch.empty(0, dtype=torch.long), persistent=True
        )
        nn.init.normal_(self.readout_queries, std=d**-0.5)

    def set_ppi_edges(self, edges: Tensor) -> None:
        """Install allowed directed protein-feature pairs for the PPI control.

        ``edges`` must have shape ``[num_edges, 2]`` and contain assay-qualified
        feature identifiers from the same dictionary used by the dataset.  The
        caller is responsible for supplying both directions when an undirected
        PPI resource is intended.
        """

        if edges.ndim != 2 or edges.size(1) != 2:
            raise ValueError("PPI edges must have shape [num_edges, 2]")
        edges = edges.to(device=self.ppi_edge_codes.device, dtype=torch.long)
        if edges.numel() and (
            torch.any(edges < 0) or torch.any(edges >= self.config.num_features)
        ):
            raise ValueError("PPI edge contains an out-of-range feature identifier")
        codes = edges[:, 0] * self.config.num_features + edges[:, 1]
        self.ppi_edge_codes = torch.unique(codes, sorted=True)

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

    def _pair_scores(
        self,
        source_tokens: Tensor,
        target_tokens: Tensor,
        source_q: Tensor,
        target_k: Tensor,
        context: Tensor,
        source_feature_ids: Tensor,
        target_feature_ids: Tensor,
        source_omics: int,
        target_omics: int,
    ) -> Tensor:
        mode = self.config.relation_mode
        if mode == "random_scores":
            head = torch.arange(
                self.config.num_relation_heads,
                device=source_tokens.device,
                dtype=source_tokens.dtype,
            ).unsqueeze(0)
            phase = (
                source_feature_ids.to(source_tokens.dtype).unsqueeze(1) * 12.9898
                + target_feature_ids.to(source_tokens.dtype).unsqueeze(1) * 78.233
                + head * 37.719
            )
            scores = torch.remainder(torch.sin(phase) * 43758.545, 1.0) * 2.0 - 1.0
        elif mode == "mlp_control":
            repeated_context = context.unsqueeze(0).expand(source_tokens.size(0), -1)
            scores = self.score_mlp(
                torch.cat((source_tokens, target_tokens, repeated_context), dim=-1)
            )
        else:
            scores = torch.einsum("ehd,ehd->eh", source_q, target_k)
            scores = scores * self.log_scale.exp().view(1, -1)
        return scores + self.omics_pair_bias[:, source_omics, target_omics].view(1, -1)

    def _edge_chunk(
        self,
        source_tokens: Tensor,
        target_tokens: Tensor,
        scores: Tensor,
        source_omics: int,
        target_omics: int,
    ) -> Tensor:
        edge_count = scores.size(0)
        relation_id = source_omics * self.config.num_omics + target_omics
        relation_type = self.relation_type_embedding.weight[relation_id]
        relation_type = relation_type.view(1, -1).expand(edge_count, -1)
        edge_input = torch.cat(
            (source_tokens, target_tokens, scores, relation_type), dim=-1
        )
        return self.edge_encoder(edge_input)

    def _pair_chunks(
        self, source_index: Tensor, target_index: Tensor
    ) -> list[tuple[Tensor, Tensor]]:
        """Return flat directed-pair index chunks without a dense N by N grid."""

        ns, nt = source_index.numel(), target_index.numel()
        chunks: list[tuple[Tensor, Tensor]] = []
        for start in range(0, ns * nt, self.config.pair_chunk_size):
            flat = torch.arange(
                start,
                min(start + self.config.pair_chunk_size, ns * nt),
                device=source_index.device,
            )
            src = source_index[torch.div(flat, nt, rounding_mode="floor")]
            tgt = target_index[flat.remainder(nt)]
            valid = src != tgt
            if torch.any(valid):
                chunks.append((src[valid], tgt[valid]))
        return chunks

    def _ppi_filter(
        self,
        source_indices: Tensor,
        target_indices: Tensor,
        feature_ids: Tensor,
        source_omics: int,
        target_omics: int,
    ) -> tuple[Tensor, Tensor]:
        # Assay index 2 is protein in the released RNA/miRNA/protein schema.
        if source_omics != 2 or target_omics != 2:
            return source_indices[:0], target_indices[:0]
        if self.ppi_edge_codes.numel() == 0:
            raise RuntimeError(
                "relation_mode='ppi' requires set_ppi_edges() or --ppi-edges"
            )
        code = (
            feature_ids[source_indices] * self.config.num_features
            + feature_ids[target_indices]
        )
        position = torch.searchsorted(self.ppi_edge_codes, code)
        position = position.clamp_max(max(self.ppi_edge_codes.numel() - 1, 0))
        keep = self.ppi_edge_codes[position] == code
        return source_indices[keep], target_indices[keep]

    def _select_topk(
        self,
        chunks: list[tuple[Tensor, Tensor]],
        tokens: Tensor,
        q: Tensor,
        k: Tensor,
        context: Tensor,
        feature_ids: Tensor,
        source_omics: int,
        target_omics: int,
    ) -> list[tuple[Tensor, Tensor]]:
        best_score = tokens.new_empty(0)
        best_source = feature_ids.new_empty(0)
        best_target = feature_ids.new_empty(0)
        limit = self.config.topk_pairs_per_block
        with torch.no_grad():
            for source_indices, target_indices in chunks:
                scores = self._pair_scores(
                    tokens[source_indices],
                    tokens[target_indices],
                    q[source_indices],
                    k[target_indices],
                    context,
                    feature_ids[source_indices],
                    feature_ids[target_indices],
                    source_omics,
                    target_omics,
                )
                importance = scores.abs().amax(dim=-1)
                best_score = torch.cat((best_score, importance))
                best_source = torch.cat((best_source, source_indices))
                best_target = torch.cat((best_target, target_indices))
                if best_score.numel() > limit:
                    keep = torch.topk(best_score, k=limit, sorted=False).indices
                    best_score = best_score[keep]
                    best_source = best_source[keep]
                    best_target = best_target[keep]
        if best_score.numel() == 0:
            return []
        keep = torch.topk(
            best_score, k=min(limit, best_score.numel()), sorted=False
        ).indices
        return [(best_source[keep], best_target[keep])]

    def forward(
        self,
        tokens: Tensor,
        feature_ids: Tensor,
        omics_ids: Tensor,
        context: Tensor,
        lora: Optional[RelationLoRA] = None,
    ) -> tuple[Tensor, Tensor, Tensor]:
        cfg = self.config
        q, k = self._project(tokens, context, lora)
        block_outputs: list[Tensor] = []
        block_masks: list[Tensor] = []
        message_sum = tokens.new_zeros(tokens.shape)
        message_count = tokens.new_zeros(tokens.size(0), 1)

        for source_omics in range(cfg.num_omics):
            source_index = torch.nonzero(omics_ids == source_omics, as_tuple=False).flatten()
            for target_omics in range(cfg.num_omics):
                target_index = torch.nonzero(omics_ids == target_omics, as_tuple=False).flatten()
                numerator = tokens.new_zeros(cfg.num_relation_queries, cfg.hidden_dim)
                denominator = tokens.new_zeros(cfg.num_relation_queries)
                pair_count = 0
                query = self.readout_queries[source_omics, target_omics]
                query_bias = self.readout_bias[source_omics, target_omics]

                chunks = self._pair_chunks(source_index, target_index)
                if cfg.relation_mode == "ppi":
                    chunks = [
                        self._ppi_filter(
                            src,
                            tgt,
                            feature_ids,
                            source_omics,
                            target_omics,
                        )
                        for src, tgt in chunks
                    ]
                if cfg.relation_mode == "topk":
                    chunks = self._select_topk(
                        chunks,
                        tokens,
                        q,
                        k,
                        context,
                        feature_ids,
                        source_omics,
                        target_omics,
                    )

                for src_idx, tgt_idx in chunks:
                    if src_idx.numel() == 0:
                        continue
                    scores = self._pair_scores(
                        tokens[src_idx],
                        tokens[tgt_idx],
                        q[src_idx],
                        k[tgt_idx],
                        context,
                        feature_ids[src_idx],
                        feature_ids[tgt_idx],
                        source_omics,
                        target_omics,
                    )
                    edge = self._edge_chunk(
                        tokens[src_idx],
                        tokens[tgt_idx],
                        scores,
                        source_omics,
                        target_omics,
                    )
                    gates = torch.sigmoid(
                        edge @ query.T / math.sqrt(cfg.hidden_dim) + query_bias
                    )
                    numerator = numerator + gates.T @ edge
                    denominator = denominator + gates.sum(dim=0)
                    pair_count += src_idx.numel()
                    if cfg.relation_mode == "writeback":
                        message_sum = message_sum.index_add(0, tgt_idx, edge)
                        message_count = message_count.index_add(
                            0,
                            tgt_idx,
                            torch.ones(
                                tgt_idx.numel(), 1, device=tokens.device, dtype=tokens.dtype
                            ),
                        )

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

        updated_tokens = tokens
        if cfg.relation_mode == "writeback":
            message = message_sum / message_count.clamp_min(1.0)
            updated_tokens = self.writeback_norm(
                tokens + self.writeback_projection(message)
            )
        return (
            torch.cat(block_outputs, dim=0),
            torch.cat(block_masks, dim=0),
            updated_tokens,
        )


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

    def forward(
        self, patient: Tensor, calibration: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        time = self.time_embedding.weight
        hidden = torch.tanh(
            self.patient_projection(patient).unsqueeze(0) + self.time_projection(time)
        )
        logits = self.output(self.dropout(hidden)).squeeze(-1)
        logits = logits + self.shared_baseline + calibration
        hazards = torch.sigmoid(logits)
        survival = torch.cumprod(1.0 - hazards, dim=-1)
        risk = 1.0 - survival
        return logits, logits, hazards, survival, risk, risk[-1:]


class CategoricalSurvivalHead(nn.Module):
    """MTLR/DeepHit-style categorical event-time head.

    The final category represents survival beyond the last modelled interval.
    The two variants share the output parameterisation and differ in their
    training objectives (see :mod:`unipansurv.losses`).
    """

    def __init__(self, config: UniPanSurvConfig) -> None:
        super().__init__()
        d = config.hidden_dim
        self.kind = config.survival_head
        output_dim = (
            config.num_time_intervals
            if config.survival_head == "mtlr"
            else config.num_time_intervals + 1
        )
        self.output = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d, output_dim),
        )
        self.calibration = nn.Linear(
            config.num_time_intervals, output_dim, bias=False
        )

    def forward(
        self, patient: Tensor, calibration: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        raw = self.output(patient) + self.calibration(calibration)
        if self.kind == "mtlr":
            event_logits = torch.flip(
                torch.cumsum(torch.flip(raw, dims=(0,)), dim=0), dims=(0,)
            )
            logits = torch.cat((event_logits, raw.new_zeros(1)))
        else:
            logits = raw
        probability = torch.softmax(logits, dim=-1)
        event_probability = probability[:-1]
        survival = torch.flip(
            torch.cumsum(torch.flip(probability[1:], dims=(0,)), dim=0), dims=(0,)
        )
        survival_before = torch.cat((probability.new_ones(1), survival[:-1]))
        hazards = event_probability / survival_before.clamp_min(1e-8)
        hazard_logits = torch.logit(hazards.clamp(1e-7, 1.0 - 1e-7))
        risk = 1.0 - survival
        return logits, hazard_logits, hazards, survival, risk, risk[-1:]


class CoxSurvivalHead(nn.Module):
    """Scalar log-relative-risk head for the Cox ablation."""

    def __init__(self, config: UniPanSurvConfig) -> None:
        super().__init__()
        self.output = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, 1),
        )

    def forward(
        self, patient: Tensor, calibration: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        del calibration
        score = self.output(patient)
        empty = score.new_empty(0)
        return score, empty, empty, empty, empty, score


def build_survival_head(config: UniPanSurvConfig) -> nn.Module:
    if config.survival_head == "discrete":
        return DiscreteTimeSurvivalHead(config)
    if config.survival_head in {"mtlr", "deephit"}:
        return CategoricalSurvivalHead(config)
    if config.survival_head == "cox":
        return CoxSurvivalHead(config)
    raise ValueError(f"unsupported survival head: {config.survival_head}")

