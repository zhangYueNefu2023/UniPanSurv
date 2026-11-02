from __future__ import annotations

import copy
import json

import pytest
import torch
from torch.utils.data import DataLoader

from unipansurv import PatientInput, UniPanSurv, UniPanSurvConfig
from unipansurv.data import SurvivalRecord, collate_survival_records, write_tensor_dataset
from unipansurv.layers import RelationEncoder
from unipansurv.losses import discrete_time_survival_nll, categorical_survival_nll
from unipansurv.training import (
    DataSettings, DistributedContext, OptimisationSettings, RunConfig,
    RuntimeSettings, _validate_settings, evaluate, run_training, train_epoch,
)


def model_config() -> UniPanSurvConfig:
    return UniPanSurvConfig(
        num_features=10, num_source_cancers=2, dropout=0.0,
        unknown_context_probability=0.0, modality_dropout_probability=0.0,
        pair_chunk_size=2,
    )


def records() -> list[SurvivalRecord]:
    return [SurvivalRecord(
        patient=PatientInput(
            feature_ids=torch.tensor([0, 1, 2]), omics_ids=torch.tensor([0, 1, 2]),
            values=torch.tensor([0.2 + i, -0.3, 0.5]),
            clinical_values=torch.zeros(3), clinical_mask=torch.zeros(3, dtype=torch.bool),
            cancer_id=i % 2,
        ), observed_time=float(i + 1), interval_index=i + 1,
        event=1.0, patient_id=f"P{i}",
    ) for i in range(4)]


def context() -> DistributedContext:
    return DistributedContext(0, 0, 1, torch.device("cpu"), False)


def disabled_scaler():
    # Keep these CPU regression checks compatible with the declared torch>=2.2.
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cpu", enabled=False)
    return torch.cuda.amp.GradScaler(enabled=False)


@pytest.mark.parametrize("index,event,expected", [
    (0, 0.0, 1.0), (1, 0.0, 0.9), (2, 0.0, 0.9 * 0.8),
    (0, 1.0, 0.1), (1, 1.0, 0.9 * 0.2), (2, 1.0, 0.9 * 0.8 * 0.3),
])
def test_likelihood_matches_completed_bin_formula(index, event, expected) -> None:
    logits = torch.logit(torch.tensor([[0.1, 0.2, 0.3]]))
    loss = discrete_time_survival_nll(logits, torch.tensor([index]), torch.tensor([event]))
    assert float(loss) == pytest.approx(float(-torch.log(torch.tensor(expected))), abs=1e-6)


@pytest.mark.parametrize("index,probability", [(0, 1.0), (1, 0.75), (2, 0.5)])
def test_categorical_censoring_starts_at_current_category(index, probability) -> None:
    loss = categorical_survival_nll(torch.zeros(1, 4), torch.tensor([index]), torch.zeros(1))
    assert float(loss) == pytest.approx(float(-torch.log(torch.tensor(probability))), abs=1e-6)


def test_chunked_relation_readout_matches_literal_paper_formula() -> None:
    torch.manual_seed(11)
    # A visible epsilon detects clamp_min versus the paper's additive epsilon.
    cfg = UniPanSurvConfig.from_dict({**model_config().to_dict(), "eps": 0.03})
    encoder = RelationEncoder(cfg).eval()
    tokens = torch.randn(4, 64)
    ids = torch.arange(4)
    types = torch.tensor([0, 0, 1, 2])
    cancer_context = torch.randn(64)
    actual, mask, unchanged = encoder(tokens, ids, types, cancer_context)
    q = (tokens @ encoder.query_projection.weight.T
         + cancer_context @ encoder.context_to_query.weight.T).reshape(4, 4, 16)
    k = (tokens @ encoder.key_projection.weight.T
         + cancer_context @ encoder.context_to_key.weight.T).reshape(4, 4, 16)
    expected = torch.zeros_like(actual)
    for source_type in range(3):
        for target_type in range(3):
            edges = []
            for i in range(4):
                for j in range(4):
                    if i == j or types[i] != source_type or types[j] != target_type:
                        continue
                    score = ((q[i] * k[j]).sum(-1) * encoder.log_scale.exp()
                             + encoder.omics_pair_bias[:, source_type, target_type])
                    edge = encoder.edge_encoder(torch.cat((
                        tokens[i], tokens[j], score,
                        encoder.relation_type_embedding.weight[source_type * 3 + target_type],
                    )))
                    edges.append(edge)
            if not edges:
                continue
            edge_values = torch.stack(edges)
            gate = torch.sigmoid(
                edge_values @ encoder.readout_queries[source_type, target_type].T
                + encoder.readout_bias[source_type, target_type]
            )
            start = (source_type * 3 + target_type) * 8
            expected[start:start + 8] = (
                gate.T @ edge_values / (gate.sum(0).unsqueeze(1) + cfg.eps)
            )
    assert torch.allclose(actual, expected, atol=1e-6, rtol=1e-5)
    assert torch.equal(unchanged, tokens)
    assert mask.sum() == 7 * 8


def test_global_weighted_update_is_independent_of_microbatch_partition() -> None:
    torch.manual_seed(12)
    first = UniPanSurv(model_config())
    second = copy.deepcopy(first)
    counts = torch.tensor([1, 9])
    dataset = records()
    settings = OptimisationSettings(max_grad_norm=1e6)
    cfg = RunConfig(model_config(), DataSettings("unused", "unused"), settings,
                    RuntimeSettings(amp="none"))
    losses = []
    for model, size, accumulation in ((first, 4, 1), (second, 1, 4)):
        cfg.optimisation.gradient_accumulation_steps = accumulation
        loader = DataLoader(dataset, batch_size=size, collate_fn=collate_survival_records)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        losses.append(train_epoch(model, loader, optimizer,
                      disabled_scaler(), cfg, context(), counts))
    assert losses[0] == pytest.approx(losses[1], abs=1e-6)
    for name, parameter in first.named_parameters():
        assert torch.allclose(parameter, dict(second.named_parameters())[name], atol=1e-6), name


def test_validation_loss_is_independent_of_batch_partition() -> None:
    model = UniPanSurv(model_config()).eval()
    cfg = RunConfig(model_config(), DataSettings("unused", "unused"),
                    runtime=RuntimeSettings(amp="none"))
    reports = [evaluate(model, DataLoader(records(), batch_size=size,
        collate_fn=collate_survival_records), cfg, context(), torch.tensor([1, 9]))
        for size in (1, 4)]
    assert reports[0]["loss"] == pytest.approx(reports[1]["loss"], abs=1e-6)
    assert reports[0]["c_index"] == reports[1]["c_index"]
    assert reports[0]["evaluated_cancers"] == 2


def test_c_index_reports_equal_cancer_mean_separately_from_pooled() -> None:
    from types import SimpleNamespace
    class FixedPredictions(torch.nn.Module):
        def forward(self, patients, mode="source"):
            risk_by_value = {0: 0.9, 1: 0.8, 2: 0.1, 3: 0.9}
            risks = [risk_by_value[round(float(p.values[0]) - 0.2)] for p in patients]
            logits = torch.zeros(len(patients), 15)
            return SimpleNamespace(risk_score=torch.tensor(risks),
                objective_logits=logits, hazard_logits=logits)
    cfg = RunConfig(model_config(), DataSettings("unused", "unused"),
                    runtime=RuntimeSettings(amp="none"))
    report = evaluate(FixedPredictions(), DataLoader(records(), batch_size=2,
        collate_fn=collate_survival_records), cfg, context(), torch.tensor([2, 2]))
    assert report["c_index"] == pytest.approx(0.5)
    assert report["pooled_c_index"] == pytest.approx(3.5 / 6)


def test_refit_really_runs_43_epochs_without_validation_selection(tmp_path, monkeypatch) -> None:
    import unipansurv.training as training
    monkeypatch.setattr(training, "initialise_distributed", context)
    path = tmp_path / "development.pt"
    items = []
    for record in records()[:2]:
        p = record.patient
        items.append(dict(feature_ids=p.feature_ids, omics_ids=p.omics_ids,
            values=p.values, clinical_values=p.clinical_values,
            clinical_mask=p.clinical_mask, cancer_id=p.cancer_id,
            observed_time=record.observed_time, interval_index=record.interval_index,
            event=record.event, patient_id=record.patient_id))
    write_tensor_dataset(path, items, {"synthetic": True})
    cfg = RunConfig(model_config(), DataSettings(str(path)),
        OptimisationSettings(stage="refit", max_epochs=1, patience=0,
            per_device_batch_size=2, expected_global_batch_size=2),
        RuntimeSettings(amp="none", output_dir=str(tmp_path / "refit")))
    summary = run_training(cfg)
    assert summary["epochs_completed"] == 43
    assert summary["best_epoch"] is None and summary["validation"] is None
    assert summary["checkpoint"] == "final.pt"
    assert (tmp_path / "refit/final.pt").is_file()
    assert not (tmp_path / "refit/best.pt").exists()
    history = [json.loads(line) for line in (tmp_path / "refit/history.jsonl").read_text().splitlines()]
    assert len(history) == 43 and all(row["validation_loss"] is None for row in history)


def test_refit_rejects_validation_stopping_and_selection_checkpoint_reuse() -> None:
    cfg = RunConfig(model_config(), DataSettings("development", "validation"),
                    OptimisationSettings(stage="refit", expected_global_batch_size=2))
    with pytest.raises(ValueError, match="without validation"):
        _validate_settings(cfg, context())
    cfg.data.validation = None
    cfg.runtime.resume = "selection.pt"
    with pytest.raises(ValueError, match="fresh model"):
        _validate_settings(cfg, context())
