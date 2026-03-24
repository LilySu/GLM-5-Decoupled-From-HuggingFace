# GLM-5 Decoupled: Full GPU Kernel Environment
# Includes FlashMLA, DeepGEMM, FlashInfer, Triton, and all Python deps.
#
# Tested on: RunPod (H100 SXM5), Nebius (H100 SXM5)
# GPU requirement: SM90 (Hopper) — H100/H200/H800
# CUDA requirement: 12.8+
#
# Build:
#   docker build -t glm5-kernels:latest -f Dockerfile .
#
# Run (RunPod / Nebius / local):
#   docker run --gpus all -it -v $(pwd):/workspace/glm5 glm5-kernels:latest
#
# RunPod template:
#   Use "Custom Docker Image" → glm5-kernels:latest
#   Or push to Docker Hub/GHCR first:
#     docker tag glm5-kernels:latest your-user/glm5-kernels:latest
#     docker push your-user/glm5-kernels:latest

# ─────────────────────────────────────────────────────────
# Stage 1: Base with CUDA + PyTorch + uv
# ─────────────────────────────────────────────────────────
FROM nvidia/cuda:12.8.1-devel-ubuntu24.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="9.0"

# System packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl wget ca-certificates \
    build-essential cmake ninja-build \
    python3.12 python3.12-dev python3.12-venv python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.12 1

# Install uv (fast Python package manager)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Create project venv with uv
WORKDIR /workspace
RUN uv venv /workspace/.venv --python 3.12
ENV VIRTUAL_ENV=/workspace/.venv
ENV PATH="/workspace/.venv/bin:$PATH"

# ─────────────────────────────────────────────────────────
# Stage 2: Python dependencies (cached layer)
# ─────────────────────────────────────────────────────────
FROM base AS deps

# PyTorch with CUDA 12.8 (install first — largest dep, changes least)
RUN uv pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128

# Triton (required by unsloth kernels and DeepGEMM JIT)
RUN uv pip install triton>=3.1.0

# Python deps (fast with uv)
RUN uv pip install \
    safetensors \
    tokenizers \
    transformers>=4.45.0 \
    numpy \
    pandas \
    scipy

# ─────────────────────────────────────────────────────────
# Stage 3: FlashInfer (pip-installable, no build needed)
# ─────────────────────────────────────────────────────────
FROM deps AS flashinfer

RUN uv pip install flashinfer-python flashinfer-cubin \
    && uv pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu128

# ─────────────────────────────────────────────────────────
# Stage 4: FlashMLA (build from source — SM90 kernels)
# ─────────────────────────────────────────────────────────
FROM flashinfer AS flashmla

RUN cd /workspace && \
    git clone --recurse-submodules --depth=1 https://github.com/deepseek-ai/FlashMLA.git && \
    cd FlashMLA && \
    git submodule update --init --recursive --depth=1 && \
    FLASH_MLA_DISABLE_SM100=1 \
    TORCH_CUDA_ARCH_LIST="9.0" \
    MAX_JOBS=4 \
    pip install -v --no-build-isolation . && \
    cd /workspace && rm -rf FlashMLA/.git

# ─────────────────────────────────────────────────────────
# Stage 5: DeepGEMM (build from source — FP8 GEMM kernels)
# ─────────────────────────────────────────────────────────
FROM flashmla AS deepgemm

RUN cd /workspace && \
    git clone --depth=1 https://github.com/deepseek-ai/DeepGEMM.git && \
    cd DeepGEMM && \
    git submodule update --init --recursive --depth=1 && \
    pip install -v --no-build-isolation . && \
    cd /workspace && rm -rf DeepGEMM/.git

# DeepGEMM JIT settings — use NVRTC for faster JIT, cache in persistent dir
ENV DG_JIT_USE_NVRTC=1
ENV DG_JIT_CACHE_DIR=/workspace/.deep_gemm_cache

# Pre-create JIT cache dir
RUN mkdir -p /workspace/.deep_gemm_cache

# ─────────────────────────────────────────────────────────
# Stage 6: Final image — verify everything works
# ─────────────────────────────────────────────────────────
FROM deepgemm AS final

# Smoke test all installations
RUN python3 -c "\
import torch; \
print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}'); \
print(f'CUDA available: {torch.cuda.is_available()}'); \
" && \
    python3 -c "import triton; print(f'Triton {triton.__version__}')" && \
    python3 -c "import flash_mla; print('FlashMLA OK')" && \
    python3 -c "import deep_gemm; print(f'DeepGEMM OK')" && \
    python3 -c "import flashinfer; print(f'FlashInfer OK')" && \
    python3 -c "import safetensors; print('safetensors OK')" && \
    python3 -c "import transformers; print(f'transformers {transformers.__version__}')"

# Default working directory for mounting the project
WORKDIR /workspace/glm5

# RunPod expects WORKDIR=/workspace and exposes port 8888 for Jupyter
EXPOSE 8888

# Default: interactive shell. Override with your command.
CMD ["/bin/bash"]
