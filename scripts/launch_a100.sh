#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${REPO_DIR}/configs/a100_4gpu.json}"

torchrun --standalone --nproc_per_node=4 -m unipansurv.training --config "${CONFIG_PATH}"
