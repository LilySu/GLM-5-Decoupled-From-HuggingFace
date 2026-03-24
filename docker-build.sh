#!/usr/bin/env bash
# Build and optionally push the GLM-5 kernel Docker image.
#
# Usage:
#   ./docker-build.sh                     # Build locally
#   ./docker-build.sh --push              # Build and push to Docker Hub
#   ./docker-build.sh --push --registry ghcr.io/your-user  # Push to GHCR
#   ./docker-build.sh --runpod            # Build with RunPod base image instead
#
# The image includes: PyTorch 2.6, CUDA 12.8, FlashMLA, DeepGEMM, FlashInfer, Triton
# Target: NVIDIA H100/H200 (SM90)

set -euo pipefail

IMAGE_NAME="glm5-kernels"
IMAGE_TAG="latest"
REGISTRY=""
PUSH=false
USE_RUNPOD_BASE=false
DOCKERFILE="Dockerfile"

while [[ $# -gt 0 ]]; do
    case $1 in
        --push)     PUSH=true; shift ;;
        --registry) REGISTRY="$2"; shift 2 ;;
        --tag)      IMAGE_TAG="$2"; shift 2 ;;
        --runpod)   USE_RUNPOD_BASE=true; DOCKERFILE="Dockerfile.runpod"; shift ;;
        *)          echo "Unknown option: $1"; exit 1 ;;
    esac
done

FULL_IMAGE="${IMAGE_NAME}:${IMAGE_TAG}"
if [ -n "$REGISTRY" ]; then
    FULL_IMAGE="${REGISTRY}/${FULL_IMAGE}"
fi

echo "========================================="
echo "Building: ${FULL_IMAGE}"
echo "Dockerfile: ${DOCKERFILE}"
echo "========================================="

DOCKER_BUILDKIT=1 docker build \
    --file "${DOCKERFILE}" \
    --tag "${FULL_IMAGE}" \
    --progress=plain \
    .

echo ""
echo "Build complete: ${FULL_IMAGE}"
echo ""

# Quick local smoke test (no GPU needed for import checks)
echo "Running import smoke test..."
docker run --rm "${FULL_IMAGE}" python3 -c "
import torch, triton, safetensors, transformers
print('Core deps OK')
try:
    import flash_mla
    print('FlashMLA OK')
except Exception as e:
    print(f'FlashMLA: {e}')
try:
    import deep_gemm
    print('DeepGEMM OK')
except Exception as e:
    print(f'DeepGEMM: {e}')
try:
    import flashinfer
    print('FlashInfer OK')
except Exception as e:
    print(f'FlashInfer: {e}')
"

if [ "$PUSH" = true ]; then
    echo ""
    echo "Pushing ${FULL_IMAGE}..."
    docker push "${FULL_IMAGE}"
    echo "Pushed: ${FULL_IMAGE}"
    echo ""
    echo "To use on RunPod:"
    echo "  1. Create new pod → Select Template → Custom Docker Image"
    echo "  2. Image: ${FULL_IMAGE}"
    echo "  3. GPU: H100 SXM (80GB)"
    echo "  4. Volume mount: /workspace"
    echo ""
    echo "To use on Nebius:"
    echo "  1. Create VM with H100 GPU"
    echo "  2. docker pull ${FULL_IMAGE}"
    echo "  3. docker run --gpus all -it -v /data:/workspace/glm5 ${FULL_IMAGE}"
fi
