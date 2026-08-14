# UniPanSurv core model

This repository contains the core PyTorch implementation of UniPanSurv. It
includes the model, censoring-aware survival loss, zero-shot inference and the
lightweight few-shot target adapter.

## Model components

For each patient, UniPanSurv:

1. converts every observed assay-qualified molecular measurement into a 64-D
   token using the molecular identifier, omics type and standardised value;
2. scores all directed non-self molecular pairs observed within that patient;
3. processes pairs in blocks and pools them into eight relation summaries for
   each of the nine directed omics blocks;
4. keeps the original molecular tokens read-only and separately constructs
   relation, omics and clinical summaries;
5. fuses the three evidence streams into one 64-D patient representation; and
6. predicts interval hazards and a survival curve over 15 time intervals.

One backbone is shared across source cancers. Zero-shot prediction uses the
learned Unknown context. Few-shot adaptation freezes the source backbone and
updates only a target context, rank-4 relation LoRA, patient adapter and
interval-specific risk calibration.

## Repository layout

```text
UniPanSurv-core/
├── pyproject.toml
├── README.md
├── examples/
│   └── minimal_usage.py
├── src/unipansurv/
│   ├── __init__.py
│   ├── adaptation.py
│   ├── config.py
│   ├── layers.py
│   ├── losses.py
│   ├── model.py
│   └── types.py
└── tests/
    └── test_core.py
```

## Installation

```bash
git clone <YOUR-GITHUB-URL>
cd UniPanSurv-core
python -m pip install -e ".[dev]"
pytest
```

## Required input format

The repository intentionally starts after cohort construction and
training-only preprocessing. Each `PatientInput` contains:

| Field | Shape | Meaning |
|---|---:|---|
| `feature_ids` | `[N]` | Assay-qualified feature identifiers |
| `omics_ids` | `[N]` | `0=RNA`, `1=miRNA`, `2=protein` |
| `values` | `[N]` | Transformed and training-standardised values |
| `molecular_mask` | `[N]` | Optional observation mask |
| `clinical_values` | `[C]` | Training-standardised baseline variables |
| `clinical_mask` | `[C]` | Observed clinical variables |
| `cancer_id` | scalar | Source-cancer identifier in source mode |

The same biological name measured by different assays must receive different
feature identifiers. Missing molecular measurements must be removed or masked;
they are never converted into artificial molecular relations. Feature maps,
means, standard deviations and time-interval boundaries must be estimated from
the training partition and then reused unchanged for validation and test data.

## Minimal use

```python
import torch
from unipansurv import PatientInput, UniPanSurv, UniPanSurvConfig

config = UniPanSurvConfig(num_features=63_028, num_source_cancers=18)
model = UniPanSurv(config)

patient = PatientInput(
    feature_ids=torch.tensor([10, 25, 900, 1200]),
    omics_ids=torch.tensor([0, 0, 1, 2]),
    values=torch.tensor([0.2, -0.4, 1.1, 0.5]),
    clinical_values=torch.tensor([0.1, 1.0, -0.3]),
    clinical_mask=torch.tensor([True, True, True]),
    cancer_id=0,
)

model.eval()
with torch.no_grad():
    output = model([patient], mode="source")
    print(output.survival.shape)  # [1, 15]
```

`PatientInput` objects and the model must be placed on the same device.

## Source-cancer training

```python
from unipansurv import cancer_balance_weights, discrete_time_survival_nll

model.train()
output = model(patients, mode="source")
weights = cancer_balance_weights(
    cancer_ids,
    training_cancer_counts,
    gamma=0.5,
)
loss = discrete_time_survival_nll(
    output.hazard_logits,
    interval_index,
    event,
    sample_weight=weights,
)
loss.backward()
```

During source training, the model jointly replaces the cancer context and risk
calibration with their Unknown versions for a random fraction of patients. It
also applies whole-modality dropout while retaining at least one observed
modality.

## Zero-shot prediction

```python
model.eval()
with torch.no_grad():
    output = model(target_patients, mode="zero_shot")
```

Zero-shot mode always uses the Unknown context and calibration. Any `cancer_id`
stored in the input is ignored. No target survival outcome is consumed by this
forward pass.

## Few-shot adaptation

```python
from unipansurv import discrete_time_survival_nll

model.freeze_backbone()
adapter = model.new_target_adapter()
adapter.train()
optimizer = torch.optim.AdamW(adapter.parameters(), lr=1e-3)

for patients, interval_index, event in adaptation_loader:
    optimizer.zero_grad()
    output = model(patients, mode="few_shot", target_adapter=adapter)
    loss = discrete_time_survival_nll(
        output.hazard_logits,
        interval_index,
        event,
    )
    loss.backward()
    optimizer.step()
```

The independent target test set must never be used to fit the adapter, choose
hyperparameters or determine stopping time.

## Notes on complete relation learning

The implementation does not use positional encoding, top-k filtering,
PPI/pathway priors or patient-to-patient edges. Pair scores are processed in
chunks controlled by `pair_chunk_size`, but every valid directed pair is still
included. Chunking reduces the largest temporary tensor; it does not change the
quadratic number of pair evaluations.

## Reproducibility boundary

This repository defines the architecture and loss but does not include trained
weights or claim any performance result. A reproducible study should separately
release cohort identifiers, partition files, training-only preprocessing
parameters, time-bin boundaries, random seeds and locked evaluation code where
data-use agreements permit.

