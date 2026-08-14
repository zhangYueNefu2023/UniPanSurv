from __future__ import annotations

import torch

from unipansurv import (
    PatientInput,
    UniPanSurv,
    UniPanSurvConfig,
    discrete_time_survival_nll,
)


def make_patient(cancer_id: int) -> PatientInput:
    return PatientInput(
        feature_ids=torch.tensor([2, 11, 17, 31, 45], dtype=torch.long),
        omics_ids=torch.tensor([0, 0, 1, 2, 2], dtype=torch.long),
        values=torch.tensor([0.2, -0.3, 0.7, 1.1, -0.1]),
        clinical_values=torch.tensor([0.4, 1.0, -0.5]),
        clinical_mask=torch.tensor([True, True, True]),
        cancer_id=cancer_id,
    )


def main() -> None:
    torch.manual_seed(7)
    config = UniPanSurvConfig(
        num_features=100,
        num_source_cancers=3,
        pair_chunk_size=2,
    )
    model = UniPanSurv(config)
    patients = [make_patient(0), make_patient(1)]

    model.train()
    output = model(patients, mode="source")
    interval_index = torch.tensor([3, 7])
    event = torch.tensor([1.0, 0.0])
    loss = discrete_time_survival_nll(output.hazard_logits, interval_index, event)
    loss.backward()
    print(f"source-training loss: {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        zero_shot = model(patients, mode="zero_shot")
    print("zero-shot survival shape:", tuple(zero_shot.survival.shape))

    model.freeze_backbone()
    target_adapter = model.new_target_adapter()
    optimizer = torch.optim.AdamW(target_adapter.parameters(), lr=1e-3)
    optimizer.zero_grad()
    adapted = model(patients, mode="few_shot", target_adapter=target_adapter)
    adaptation_loss = discrete_time_survival_nll(
        adapted.hazard_logits, interval_index, event
    )
    adaptation_loss.backward()
    optimizer.step()
    print(f"few-shot adaptation loss: {adaptation_loss.item():.4f}")


if __name__ == "__main__":
    main()

