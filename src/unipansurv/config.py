from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping


RelationMode = Literal[
    "full",
    "none",
    "random_scores",
    "topk",
    "ppi",
    "mlp_control",
    "writeback",
]
SurvivalHead = Literal["discrete", "mtlr", "deephit", "cox"]


@dataclass(frozen=True)
class UniPanSurvConfig:
    """Architecture settings for UniPanSurv.

    ``num_features`` counts assay-qualified molecular features. For example,
    TP53 RNA and TP53 protein must have different feature identifiers.
    ``num_source_cancers`` excludes the additional Unknown context.
    """

    num_features: int
    num_source_cancers: int
    num_clinical_features: int = 3
    num_omics: int = 3
    hidden_dim: int = 64
    num_relation_heads: int = 4
    relation_head_dim: int = 16
    num_relation_queries: int = 8
    num_time_intervals: int = 15
    time_embedding_dim: int = 16
    # Maximum number of directed edges materialised at once.  This is a count
    # of edges, not the side length of a dense square block.
    pair_chunk_size: int = 262_144
    dropout: float = 0.20
    unknown_context_probability: float = 0.25
    modality_dropout_probability: float = 0.15
    adapter_rank: int = 4
    adapter_bottleneck: int = 8
    relation_mode: RelationMode = "full"
    survival_head: SurvivalHead = "discrete"
    topk_pairs_per_block: int = 4_096
    deephit_ranking_weight: float = 0.20
    deephit_ranking_temperature: float = 0.10
    eps: float = 1e-8

    def __post_init__(self) -> None:
        if self.num_features <= 0:
            raise ValueError("num_features must be positive")
        if self.num_source_cancers <= 0:
            raise ValueError("num_source_cancers must be positive")
        if self.num_omics <= 0:
            raise ValueError("num_omics must be positive")
        if self.hidden_dim != self.num_relation_heads * self.relation_head_dim:
            raise ValueError(
                "hidden_dim must equal num_relation_heads * relation_head_dim"
            )
        if self.num_relation_queries <= 0 or self.num_time_intervals <= 0:
            raise ValueError("query and interval counts must be positive")
        if self.pair_chunk_size <= 0:
            raise ValueError("pair_chunk_size must be positive")
        if self.adapter_rank <= 0 or self.adapter_bottleneck <= 0:
            raise ValueError("adapter dimensions must be positive")
        if self.topk_pairs_per_block <= 0:
            raise ValueError("topk_pairs_per_block must be positive")
        if self.relation_mode not in {
            "full",
            "none",
            "random_scores",
            "topk",
            "ppi",
            "mlp_control",
            "writeback",
        }:
            raise ValueError(f"unsupported relation_mode: {self.relation_mode}")
        if self.survival_head not in {"discrete", "mtlr", "deephit", "cox"}:
            raise ValueError(f"unsupported survival_head: {self.survival_head}")
        if self.deephit_ranking_weight < 0 or self.deephit_ranking_temperature <= 0:
            raise ValueError("invalid DeepHit ranking settings")
        for name, value in (
            ("dropout", self.dropout),
            ("unknown_context_probability", self.unknown_context_probability),
            ("modality_dropout_probability", self.modality_dropout_probability),
        ):
            if not 0.0 <= value < 1.0:
                raise ValueError(f"{name} must lie in [0, 1)")

    @property
    def unknown_cancer_index(self) -> int:
        """Index reserved for the learned Unknown cancer context."""

        return self.num_source_cancers

    @property
    def num_cancer_contexts(self) -> int:
        return self.num_source_cancers + 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "UniPanSurvConfig":
        return cls(**dict(values))

