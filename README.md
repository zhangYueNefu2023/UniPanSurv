# UniPanSurv

PyTorch reference implementation of UniPanSurv: a shared pan-cancer survival
model for incomplete RNA, miRNA and protein profiles.  This repository contains
the model definition, source training, zero-shot inference primitives, restricted
few-shot adaptation, prespecified ablations and multi-GPU launch configurations.

## Implemented architecture

- assay-qualified molecular feature, omics-type and continuous-value embeddings;
- 64-dimensional tokens and four 16-dimensional relation heads;
- every retained, observed directed non-self pair, streamed in fixed-size blocks;
- nine directed omics blocks and eight learned relation readout queries per block;
- read-only molecular tokens in the default model;
- separate relation, assay and clinical summaries with masked gated fusion;
- 15-interval discrete-time survival prediction;
- learned source-cancer contexts plus an Unknown context for zero-shot use;
- rank-4 relation LoRA, a 64-8-64 patient adapter and interval calibration for
  target-cancer adaptation.

The supplied full-data configurations use the manuscript settings: a training-fold
cap of 2,048 retained features, 262,144 directed pairs per streaming block,
dropout 0.20, context dropout 0.25, modality dropout 0.15, AdamW learning rate
`2e-4`, weight decay `1e-4`, maximum gradient norm 1.0, maximum 200 epochs and
patience 20.

## Repository layout

```text
src/unipansurv/
  model.py          UniPanSurv composition and source/zero/few-shot modes
  layers.py         tokenizer, relation module, fusion and survival heads
  adaptation.py     target-specific trainable modules
  fewshot.py        100-update target adaptation CLI
  losses.py         discrete, MTLR, DeepHit-style and Cox objectives
  data.py           validated model-ready dataset format
  training.py       single-process and DDP training/evaluation/checkpointing
  ablation.py       reproducible ablation-matrix launcher
  synthetic.py      small non-scientific dataset for an end-to-end smoke test
configs/
  a100_4gpu.json    four A100 80-GB workers, two patients/GPU
  rtx4090_2gpu.json two RTX 4090 workers, one patient/GPU, four accumulation steps
  ablations.json    relation, modality, dropout and survival-head ablations
scripts/
  launch_a100.sh
  launch_4090.sh
  launch_4090_ablation_cluster.sh
tests/
```

## Installation and verified smoke test

Python 3.9 or newer and PyTorch 2.2 or newer are required.

```bash
git clone https://github.com/zhangYueNefu2023/UniPanSurv.git
cd UniPanSurv
python -m pip install -e '.[dev]'
unipansurv-synthetic --output-dir data/synthetic
unipansurv-train --config configs/smoke.json
pytest
```

## Model-ready data

Training consumes one `.pt` file per split.  Each file is a dictionary with
`schema_version`, `metadata` and a non-empty `records` list.  Every record has:

```python
{
    "feature_ids": LongTensor[num_observed_tokens],
    "omics_ids": LongTensor[num_observed_tokens],   # 0=RNA, 1=miRNA, 2=protein
    "values": FloatTensor[num_observed_tokens],      # training-fold standardised
    "clinical_values": FloatTensor[3],
    "clinical_mask": BoolTensor[3],
    "cancer_id": int,                               # contiguous source ID
    "observed_time": float,
    "interval_index": int,                          # 0..14
    "event": 0.0 or 1.0,
    "patient_id": str
}
```

Feature selection, transformations, imputation, standardisation, interval-grid
construction and feature-ID mapping must be fitted on the training partition.
The same mapping and statistics must then be applied unchanged to validation and
test patients.  The model intentionally does not perform these cohort-dependent
steps internally, because doing so would make leakage easy.

For `relation_mode="ppi"`, `data.ppi_edges` must point to a two-column TSV of
directed, assay-qualified integer feature IDs.  Both directions must be listed
when an undirected interaction resource is intended.  The implementation keeps
only protein-to-protein pairs (`omics_id=2`) found in that file.

## Distributed training

Primary four-A100 job:

```bash
./scripts/launch_a100.sh configs/a100_4gpu.json
```

This starts four DDP workers with two patients per GPU, giving global batch size
eight.  The program checks that arithmetic before training.

One two-RTX-4090 auxiliary job:

```bash
./scripts/launch_4090.sh configs/rtx4090_2gpu.json 0,1
```

This starts two workers with one patient per GPU and four gradient-accumulation
steps, also giving effective global batch size eight.  A100 and RTX 4090 devices
are never mixed within one job.

Every run writes:

- `resolved_config.json`: exact paths and hyperparameters;
- `history.js