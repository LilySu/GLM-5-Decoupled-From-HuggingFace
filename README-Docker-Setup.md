# GLM-5 Docker Setup: GPU Kernel Environment

Pre-built container with FlashMLA, DeepGEMM, FlashInfer, and Triton — ready for H100 benchmarks on RunPod, Nebius, or local GPU.

## What's Inside

| Component | Version | Install Method |
|-----------|---------|---------------|
| CUDA | 12.8.1 | Base image |
| PyTorch | 2.6.0 (cu128) | `uv pip install` |
| Triton | >=3.1 | `uv pip install` |
| FlashMLA | latest (SM90) | Built from source |
| DeepGEMM | latest (FP8 GEMM) | Built from source |
| FlashInfer | latest (cu128) | Prebuilt wheels |
| transformers | >=4.45 | `uv pip install` |
| safetensors, tokenizers, numpy, pandas | latest | `uv pip install` |

GPU requirement: **SM90 (Hopper)** — H100, H200, H800.

## Quick Start

### Build locally

```bash
./docker-build.sh
```

### Run interactively

```bash
./docker-run.sh
```

### Run a benchmark directly

```bash
./docker-run.sh python3 -m benchmark.run_all --quick
```

### Check GPU inside container

```bash
./docker-run.sh nvidia-smi
```

## Dockerfiles

There are two variants:

| File | Base Image | Use Case |
|------|-----------|----------|
| `Dockerfile` | `nvidia/cuda:12.8.1-devel-ubuntu24.04` | Anywhere (local, Nebius, generic cloud) |
| `Dockerfile.runpod` | `runpod/pytorch:2.6.0-py3.12-cuda12.8.1-devel-ubuntu24.04` | RunPod (includes their Jupyter, SSH, monitoring) |

Both produce the same kernel environment. The RunPod variant just inherits their infrastructure tooling.

## Deploy on RunPod

### Option A: Push image to Docker Hub, use as custom template

```bash
# Build the RunPod variant and push
./docker-build.sh --runpod
docker tag glm5-kernels:latest your-user/glm5-kernels:runpod
docker push your-user/glm5-kernels:runpod
```

Then on RunPod:
1. **My Pods** > **Deploy** > **Custom Docker Image**
2. Image: `your-user/glm5-kernels:runpod`
3. GPU: **H100 SXM 80GB** (1x for kernel dev, 4x+ for multi-GPU tests)
4. Volume: `/workspace` (persists between restarts)
5. Once running, SSH in or use Jupyter, then:

```bash
cd /workspace
git clone <your-glm5-repo>
cd glm5
python3 -m benchmark.run_all --quick
```

### Option B: Start from RunPod's stock PyTorch template, install inside

```bash
# On a RunPod H100 pod with PyTorch 2.6 template:
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv pip install --system triton safetensors tokenizers "transformers>=4.45" numpy pandas scipy
uv pip install --system flashinfer-python flashinfer-cubin
uv pip install --system flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu128

cd /workspace
git clone --recurse-submodules --depth=1 https://github.com/deepseek-ai/FlashMLA.git
cd FlashMLA && FLASH_MLA_DISABLE_SM100=1 pip install -v --no-build-isolation . && cd ..

git clone --depth=1 https://github.com/deepseek-ai/DeepGEMM.git
cd DeepGEMM && git submodule update --init --recursive --depth=1 && pip install -v --no-build-isolation . && cd ..

export DG_JIT_USE_NVRTC=1
export DG_JIT_CACHE_DIR=/workspace/.deep_gemm_cache
mkdir -p $DG_JIT_CACHE_DIR
```

## Deploy on Nebius

```bash
# On a Nebius H100 instance with CUDA 12.8:
docker pull your-user/glm5-kernels:latest
docker run --gpus all -it \
    --shm-size=16g \
    -v /data/glm5:/workspace/glm5 \
    -v /data/.deep_gemm_cache:/workspace/.deep_gemm_cache \
    your-user/glm5-kernels:latest
```

Or build directly on the instance:

```bash
git clone <your-glm5-repo> && cd glm5
docker build -t glm5-kernels:latest -f Dockerfile .
docker run --gpus all -it -v $(pwd):/workspace/glm5 glm5-kernels:latest
```

## Build Options

```bash
# Standard build (NVIDIA CUDA base)
./docker-build.sh

# RunPod-optimized build
./docker-build.sh --runpod

# Build and push to Docker Hub
./docker-build.sh --push --registry your-dockerhub-user

# Build and push to GitHub Container Registry
./docker-build.sh --push --registry ghcr.io/your-user

# Custom tag
./docker-build.sh --tag v0.1.0
```

## DeepGEMM JIT Cache

DeepGEMM compiles CUDA kernels at runtime via JIT. First invocation at a new matrix size takes 10-30 seconds. Compiled kernels are cached in `DG_JIT_CACHE_DIR`.

The `docker-run.sh` script mounts `~/.deep_gemm_cache` from the host so the cache persists across container restarts. On RunPod, the `/workspace` volume persists automatically.

To pre-warm the JIT cache for all benchmark configurations:

```bash
python3 -c "
import deep_gemm, torch
# Dummy calls at expected seq_lens to trigger JIT compilation
for T in [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]:
    q = torch.randn(1, 32, 128, dtype=torch.float8_e4m3fn, device='cuda')
    k = torch.randn(T, 128, dtype=torch.float8_e4m3fn, device='cuda')
    ks = torch.zeros(1, dtype=torch.int32, device='cuda')
    ke = torch.full((1,), T, dtype=torch.int32, device='cuda')
    w = torch.randn(1, 32, device='cuda')
    k_scales = torch.ones(T, device='cuda')
    try:
        deep_gemm.fp8_mqa_logits(q, (k, k_scales), w, ks, ke)
        print(f'  T={T}: compiled OK')
    except Exception as e:
        print(f'  T={T}: {e}')
"
```

## Verifying the Installation

Inside the container:

```bash
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'SM: {torch.cuda.get_device_capability(0)}')

import triton; print(f'Triton {triton.__version__}')
import flash_mla; print('FlashMLA OK')
import deep_gemm; print('DeepGEMM OK')
import flashinfer; print('FlashInfer OK')
"
```

Expected output on H100:
```
PyTorch 2.6.0
CUDA 12.8
GPU: NVIDIA H100 80GB HBM3
SM: (9, 0)
Triton 3.1.0
FlashMLA OK
DeepGEMM OK
FlashInfer OK
```

## Troubleshooting

**FlashMLA build fails with "unsupported SM"**
You're not on SM90. FlashMLA dense decode requires H100/H200. Check `torch.cuda.get_device_capability()`.

**DeepGEMM JIT compilation hangs**
Set `DG_JIT_USE_NVRTC=1` (already set in the Dockerfile). If still slow, check that CUDA toolkit matches PyTorch's CUDA version.

**FlashInfer import error about CUDA version**
Ensure the JIT cache index URL matches your CUDA version. For CUDA 12.8: `--index-url https://flashinfer.ai/whl/cu128`.

**OOM during benchmarks**
Reduce batch size or context length. At B=32, T=65536 with full KV cache, memory usage is ~5GB for KV alone. Add `--shm-size=16g` to docker run if using multi-GPU.

**"No module named flash_mla" but build succeeded**
FlashMLA installs into the system Python. If using a venv, ensure you installed into the venv or use `--system` with uv.
