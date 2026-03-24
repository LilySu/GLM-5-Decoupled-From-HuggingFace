#!/usr/bin/env bash
# Quick-run the GLM-5 kernel container locally or on a cloud GPU.
#
# Usage:
#   ./docker-run.sh                           # Interactive shell
#   ./docker-run.sh python3 -m benchmark.run_all --quick   # Run benchmarks
#   ./docker-run.sh nvidia-smi               # Check GPU
#
# The current directory is mounted at /workspace/glm5 so edits are live.

set -euo pipefail

IMAGE="${GLM5_IMAGE:-glm5-kernels:latest}"

# Check if image exists
if ! docker image inspect "$IMAGE" &>/dev/null; then
    echo "Image '$IMAGE' not found. Building..."
    ./docker-build.sh
fi

exec docker run \
    --gpus all \
    --rm \
    -it \
    --shm-size=16g \
    -v "$(pwd)":/workspace/glm5 \
    -v "${HOME}/.deep_gemm_cache:/workspace/.deep_gemm_cache" \
    -w /workspace/glm5 \
    -e NVIDIA_VISIBLE_DEVICES=all \
    -e DG_JIT_CACHE_DIR=/workspace/.deep_gemm_cache \
    "$IMAGE" \
    "${@:-/bin/bash}"
