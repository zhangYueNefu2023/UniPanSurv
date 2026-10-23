#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${REPO_DIR}/configs/rtx4090_2gpu.json}"
GPU_PAIR="${2:-0,1}"

CUDA_VISIBLE_DEVICES="${GPU_PAIR}" torchrun --standalone --nproc_per_node=2 \
  -m unipansurv.training --config "${CONFIG_PATH}"
