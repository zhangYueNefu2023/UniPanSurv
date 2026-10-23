from __future__ import annotations

import torch

from unipansurv.data import TensorSurvivalDataset, collate_survival_records, write_tensor_dataset


def test_dataset_roundtrip_and_modality_mask(tmp_path) -> None:
    path = tmp_path / "cohort.pt"
    record = {
        "feature_ids": torch.tensor([1, 2, 3]),
        "omics_ids": torch.tensor([0, 1, 2]),
        "values": torch.tensor([0.1, 0.2, 0.3]),
        "clinical_values": torch.tensor([1.0, 0.0, -1.0]),
        "clinical_mask": torch.tensor([True, True, False]),
        "cancer_id": 0,
        "observed_time": 4.5,
        "interval_index": 4,
        "event": 1.0,
        "patient_id": "P001",
    }
    write_tensor_dataset(path, [record], {"synthetic": True})
    dataset = TensorSurvivalDataset(path, allowed_modalities=[0, 2])
    assert dataset[0].patient.omics_ids.tolist() == [0, 2]
    batch = collate_survival_records([dataset[0]])
    assert batch.patient_id == ["P001"]
    assert batch.interval_index.tolist() == [4]
    assert dataset.cancer_counts.tolist() == [1]
