# UniPanSurv

UniPanSurv is a PyTorch implementation of a shared pan-cancer survival model for
incomplete RNA, miRNA and protein profiles. The package provides the core model,
a censoring-aware discrete-time survival objective, source-cancer training
primitives, zero-shot inference through a learned Unknown cancer context, and a
restricted target-cancer adapter for few-shot use.

## Core architecture

For each patient, the implemented model:

1. converts each observed assay-qualified molecular measurement into a
   64-dimensional token using its feature identifier, omics type and standardized
   value;
2. evaluates directed non-self molecular pairs within the patient;
3. processes pairs in bounded chunks and summarizes each of the nine directed
   RNA/miRNA/protein blocks with eight learned relation queries;
4. retains the original molecular tokens as a separate omics evidence stream;
5. combines relation, omics and clinical summaries with masked gated fusion; and
6. predicts hazards and survival probabilities over 15 discrete time intervals.

The source model contains cancer-specific contexts and an additional learned
Unknown context. Zero-shot inference uses the Unknown context. Few-shot
adaptation freezes the source model and updates only the target-specific context,
rank-4 relation LoRA parameters, a residual patient adapter and interval-specific
risk calibration.

## Repository layout

```text
UniPanSurv/
├── configs/
│   ├── a100_4gpu.json
│   ├── rtx4090_2gpu.json
│   └── ablations.json
├── examples/
│   └── minimal_usage.py
├── src/
│   └── unipansurv/
│       ├── __init__.py
│       ├── adaptation.py
│       ├── config.py
│       ├── layers.py
│       ├── losses.py
│       ├── model.py
│       └── types.py
├── tests/
│   └── test_core.py
├── pyproject.toml
├── requirements.txt
└── README.md
```

## Requirements and installation

- Python 3.9 or newer
- PyTorch 2.2 or newer

```bash
git clone https://github.com/zhangYueNefu2023/UniPanSurv.git
cd UniPanSurv
python -m pip install -e ".[dev]"
pytest
```

Run the included minimal example with:

```bash
python examples/minimal_usage.py
```

The tests cover output shapes, monotonic survival probabilities, permutation
invariance to molecular-token order, use of the Unknown context in zero-shot
mode, single-token inputs, and gradient flow through the target adapter.

## Model-ready input

The public model API starts after cohort construction, feature selection and
training-fold preprocessing. Each `PatientInput` contains:

| Field | Shape | Description |
|---|---:|---|
| `feature_ids` | `[N]` | Assay-qualified feature identifiers |
| `omics_ids` | `[N]` | Omics identifiers: `0=RNA`, `1=miRNA`, `2=protein` |
| `values` | `[N]` | Transformed and training-standardized molecular values |
| `molecular_mask` | `[N]` | Optional mask identifying observed molecular values |
| `clinical_values` | `[C]` | Training-standardized baseline clinical variables |
| `clinical_mask` | `[C]` | Mask identifying observed clinical variables |
| `cancer_id` | scalar | Source-cancer identifier used in source mode |

The same biological entity measured by different assays must receive different
feature identifiers. Missing molecular values must be removed or masked rather
than converted into artificial molecular relations. Feature selection, feature
identifier mapping, transformations, standardization parameters and time-bin
boundaries must be estimated using the training partition and then applied
unchanged to validation and test data.

## Minimal model use

```python
import torch

from unipansurv import PatientInput, UniPanSurv, UniPanSurvConfig

config = UniPanSurvConfig(
    num_features=2048,
    num_source_cancers=18,
    pair_chunk_size=262_144,
)
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
    print(output.survival.shape)  # torch.Size([1, 15])
```

`PatientInput` tensors and the model must be located on the same device.
`num_features` is the size of the processed assay-qualified feature vocabulary;
valid feature identifiers therefore range from `0` to `num_features - 1`.

## Source-cancer objective

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

`training_cancer_counts` must be calculated from the complete training
partition, not from an individual mini-batch. During source-mode training, the
model can replace source-cancer context and calibration with their Unknown
counterparts and can remove whole observed modalities according to the
probabilities stored in the configuration.

## Zero-shot inference

```python
model.eval()
with torch.no_grad():
    output = model(target_patients, mode="zero_shot")
```

Zero-shot mode always uses the Unknown cancer context and Unknown calibration.
Any `cancer_id` stored in a target input is ignored by the zero-shot forward
pass. No target survival outcome is consumed during prediction.

## Few-shot adaptation

```python
import torch

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

The adaptation subset and independent target test subset must remain separate.
Target-test outcomes must not be used to fit the adapter, select
hyperparameters, or determine the stopping point.

## Experiment configuration snapshots

The `configs/` directory records the experiment settings associated with the
revision:

- `a100_4gpu.json` describes a four-GPU NVIDIA A100 job with 80 GB memory per
  GPU, two patients per GPU and an expected global batch size of eight.
- `rtx4090_2gpu.json` describes a two-GPU NVIDIA RTX 4090 job with one patient
  per GPU, four gradient-accumulation steps and an expected effective global
  batch size of eight.
- `ablations.json` records the prespecified relation, modality, modality-dropout
  and survival-head comparison matrix.

The distributed cluster comprised four NVIDIA A100 80 GB GPUs and six NVIDIA
RTX 4090 24 GB GPUs. A100 jobs used four GPUs. The supplied RTX 4090 profile
describes a two-GPU job scheduled within the six-GPU RTX 4090 pool; A100 and RTX
4090 devices were not mixed within a single distributed job.

## Configuration scope

The public core forward path currently implements complete relation learning
and the discrete-time survival head. The additional values accepted by
`UniPanSurvConfig`—including `relation_mode`, alternative survival-head names,
top-k settings and DeepHit ranking settings—and the entries in
`configs/ablations.json` are experiment specifications. 

## Complete-relation computation

The default model does not use positional encodings, top-k filtering,
PPI/pathway restrictions, or patient-to-patient graph edges. Every valid
directed non-self molecular pair is included. `pair_chunk_size` is the maximum
number of directed edges materialized at one time; chunking reduces temporary
memory use but does not change the set or total number of evaluated pairs.

## Code availability

The public repository is available at:

https://github.com/zhangYueNefu2023/UniPanSurv

No software license file is currently included. In the absence of an explicit
license, use and redistribution remain subject to applicable copyright law and
permission from the copyright holders.
