from __future__ import annotations

import pytest
import torch

from unipansurv import PatientInput, UniPanSurv, UniPanSurvConfig, survival_objective


def patient() -> PatientInput:
    return PatientInput(
        feature_ids=torch.tensor([0, 1, 2, 3, 4, 5]),
        omics_ids=torch.tensor([0, 0, 1, 1, 2, 2]),
        values=torch.tensor([0.2, -0.1, 0.7, -0.4, 0.5, 0.9]),
        clinical_values=torch.tensor([0.1, 1.0, -0.3]),
        clinical_mask=torch.tensor([True, True, True]),
        cancer_id=0,
    )


@pytest.mark.parametrize(
    "relation_mode",
    ["full", "none", "random_scores", "topk", "ppi", "mlp_control", "writeback"],
)
def test_relation_ablation_forward_and_gradient(relation_mode: str) -> None:
    config = UniPanSurvConfig(
        num_features=10,
        num_source_cancers=2,
        pair_chunk_size=3,
        topk_pairs_per_block=2,
        dropout=0.0,
        unknown_context_probability=0.0,
        modality_dropout_probability=0.0,
        relation_mode=relation_mode,  # type: ignore[arg-type]
    )
    model = UniPanSurv(config)
    if relation_mode == "ppi":
        model.set_ppi_edges(torch.tensor([[4, 5], [5, 4]]))
    output = model([patient()])
    loss = survival_objective(
        head="discrete",
        objective_logits=output.objective_logits,
        hazard_logits=output.hazard_logits,
        interval_index=torch.tensor([4]),
        event=torch.tensor([1.0]),
        observed_time=torch.tensor([4.2]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert output.relation_summaries.shape == (1, 72, 64)
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize("head", ["discrete", "mtlr", "deephit", "cox"])
def test_survival_head_objectives(head: str) -> None:
    config = UniPanSurvConfig(
        num_features=10,
        num_source_cancers=2,
        pair_chunk_size=4,
        dropout=0.0,
        unknown_context_probability=0.0,
        modality_dropout_probability=0.0,
        survival_head=head,  # type: ignore[arg-type]
    )
    model = UniPanSurv(config)
    output = model([patient(), patient()])
    loss = survival_objective(
        head=config.survival_head,
        objective_logits=output.objective_logits,
        hazard_logits=output.hazard_logits,
        interval_index=torch.tensor([3, 7]),
        event=torch.tensor([1.0, 0.0]),
        observed_time=torch.tensor([3.5, 7.2]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert output.risk_score.shape == (2,)
    if head == "cox":
        assert output.survival.shape == (2, 0)
    else:
        assert output.survival.shape == (2, 15)
        assert torch.all(output.survival[:, 1:] <= output.survival[:, :-1] + 1e-7)


def test_random_score_control_is_order_invariant() -> None:
    config = UniPanSurvConfig(
        num_features=10,
        num_source_cancers=2,
        pair_chunk_size=3,
        dropout=0.0,
        unknown_context_probability=0.0,
        modality_dropout_probability=0.0,
        relation_mode="random_scores",
    )
    model = UniPanSurv(config).eval()
    original = patient()
    order = torch.tensor([4, 1, 5, 0, 3, 2])
    reordered = PatientInput(
        feature_ids=original.feature_ids[order],
        omics_ids=original.omics_ids[order],
        values=original.values[order],
        clinical_values=original.clinical_values,
        clinical_mask=original.clinical_mask,
        cancer_id=original.cancer_id,
    )
    with torch.no_grad():
        first = model([original]).hazard_logits
        second = model([reordered]).hazard_logits
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


def test_pair_chunk_size_is_an_edge_count() -> None:
    config = UniPanSurvConfig(
        num_features=10,
        num_source_cancers=2,
        pair_chunk_size=3,
    )
    encoder = UniPanSurv(config).relation_encoder
    indices = torch.arange(4)
    chunks = encoder._pair_chunks(indices, indices)
    assert chunks
    assert max(source.numel() for source, _ in chunks) <= 3
    assert sum(source.numel() for source, _ in chunks) == 12
    assert all(torch.all(source != target) for source, target in chunks)


def test_target_adapter_uses_separate_lora_rank_and_patient_bottleneck() -> None:
    config = UniPanSurvConfig(
        num_features=10,
        num_source_cancers=2,
        adapter_rank=4,
        adapter_bottleneck=8,
    )
    adapter = UniPanSurv(config).new_target_adapter()
    assert adapter.relation_lora.query.rank == 4
    assert adapter.relation_lora.key.rank == 4
    assert adapter.patient_adapter.down.in_features == 64
    assert adapter.patient_adapter.down.out_features == 8
    assert adapter.patient_adapter.up.in_features == 8
    assert adapter.patient_adapter.up.out_features == 64
