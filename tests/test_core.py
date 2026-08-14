from __future__ import annotations

import torch

from unipansurv import (
    PatientInput,
    UniPanSurv,
    UniPanSurvConfig,
    discrete_time_survival_nll,
)


def config() -> UniPanSurvConfig:
    return UniPanSurvConfig(
        num_features=50,
        num_source_cancers=2,
        pair_chunk_size=2,
        dropout=0.0,
        unknown_context_probability=0.0,
        modality_dropout_probability=0.0,
    )


def patient(cancer_id: int = 0) -> PatientInput:
    return PatientInput(
        feature_ids=torch.tensor([1, 7, 13, 22, 31]),
        omics_ids=torch.tensor([0, 0, 1, 2, 2]),
        values=torch.tensor([0.1, -0.2, 0.3, 1.0, -0.5]),
        clinical_values=torch.tensor([0.4, 1.0, -0.8]),
        clinical_mask=torch.tensor([True, True, True]),
        cancer_id=cancer_id,
    )


def test_forward_shapes_and_monotone_survival() -> None:
    torch.manual_seed(1)
    model = UniPanSurv(config()).eval()
    output = model([patient(), patient(1)], mode="source")
    assert output.hazard_logits.shape == (2, 15)
    assert output.patient_representation.shape == (2, 64)
    assert output.relation_summaries.shape == (2, 9 * 8, 64)
    assert output.omics_summaries.shape == (2, 3, 64)
    assert torch.all(output.survival[:, 1:] <= output.survival[:, :-1] + 1e-7)
    assert torch.allclose(output.risk, 1.0 - output.survival)


def test_molecular_order_does_not_change_prediction() -> None:
    torch.manual_seed(2)
    model = UniPanSurv(config()).eval()
    original = patient()
    order = torch.tensor([3, 0, 4, 2, 1])
    reordered = PatientInput(
        feature_ids=original.feature_ids[order],
        omics_ids=original.omics_ids[order],
        values=original.values[order],
        clinical_values=original.clinical_values,
        clinical_mask=original.clinical_mask,
        cancer_id=original.cancer_id,
    )
    with torch.no_grad():
        first = model([original], mode="source").hazard_logits
        second = model([reordered], mode="source").hazard_logits
    assert torch.allclose(first, second, atol=1e-6, rtol=1e-6)


def test_zero_shot_ignores_source_cancer_identifier() -> None:
    torch.manual_seed(3)
    model = UniPanSurv(config()).eval()
    first_patient = patient(0)
    second_patient = patient(1)
    with torch.no_grad():
        first = model([first_patient], mode="zero_shot").hazard_logits
        second = model([second_patient], mode="zero_shot").hazard_logits
    assert torch.allclose(first, second, atol=1e-7, rtol=1e-7)


def test_single_token_patient_has_no_relation_summary() -> None:
    model = UniPanSurv(config()).eval()
    one = PatientInput(
        feature_ids=torch.tensor([1]),
        omics_ids=torch.tensor([0]),
        values=torch.tensor([0.2]),
        clinical_values=torch.tensor([0.0, 0.0, 0.0]),
        clinical_mask=torch.tensor([False, False, False]),
        cancer_id=0,
    )
    with torch.no_grad():
        output = model([one], mode="source")
    assert not torch.any(output.relation_mask)
    assert output.omics_mask.tolist() == [[True, False, False]]


def test_survival_loss_and_target_adapter_gradients() -> None:
    torch.manual_seed(4)
    model = UniPanSurv(config())
    model.freeze_backbone()
    adapter = model.new_target_adapter()
    output = model([patient()], mode="few_shot", target_adapter=adapter)
    loss = discrete_time_survival_nll(
        output.hazard_logits,
        interval_index=torch.tensor([4]),
        event=torch.tensor([1.0]),
    )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert any(parameter.grad is not None for parameter in adapter.parameters())

