#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_CONFIG="${1:-${REPO_DIR}/configs/rtx4090_2gpu.json}"
MATRIX="${2:-${REPO_DIR}/configs/ablations.json}"

for SHARD in 0 1 2; do
  FIRST_GPU=$((SHARD * 2))
  SECOND_GPU=$((FIRST_GPU + 1))
  CUDA_VISIBLE_DEVICES="${FIRST_GPU},${SECOND_GPU}" \
    python -m unipansurv.ablation \
      --base-config "${BASE_CONFIG}" \
      --matrix "${MATRIX}" \
      --nproc-per-node 2 \
      --shard-index "${SHARD}" \
      --num-shards 3 &
done
wait
