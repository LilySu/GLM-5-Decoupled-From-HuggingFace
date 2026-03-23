# Running GLM-5 Kernel Benchmarks on Nebius AI Cloud

Step-by-step guide to provisioning H100 GPUs on Nebius, installing dependencies,
and running the full benchmark suite for `glm5-kernels-flashmla-deepgemm`.

---

## Prerequisites

- A Nebius AI Cloud account (sign up at [nebius.com](https://nebius.com))
- A payment method configured (H100: $2.95/GPU/hr on-demand, $1.25/GPU/hr preemptible)
- A local machine with SSH client

## Cost Estimate

| Configuration | Use Case | Cost/hr | Recommended Runtime | Total |
|---|---|---|---|---|
| 1× H100 SXM | CPU tests + single-GPU kernel tests | $2.95 | 1-2 hours | ~$6 |
| 8× H100 SXM | Full benchmarks + multi-GPU tests | $23.60 | 2-4 hours | ~$70 |
| 1× H100 preemptible | Quick kernel correctness tests | $1.25 | 1 hour | ~$1.25 |

---

## Step 1: Install the Nebius CLI

```bash
# On your local machine (Linux/macOS)
curl -sSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash

# Restart shell or source profile
source ~/.bashrc  # or ~/.zshrc

# Verify installation
nebius --version
```

## Step 2: Authenticate and Configure

```bash
# Create a profile (opens browser for OAuth login)
nebius profile create

# Set your project ID (find it in the Nebius web console under Settings > Project)
nebius config set parent-id <YOUR_PROJECT_ID>
```

## Step 3: Generate SSH Keys

```bash
# If you don't already have one
ssh-keygen -t ed25519 -f ~/.ssh/nebius_h100 -C "glm5-benchmarks"
```

## Step 4: Create the GPU Instance

### Option A: Single H100 GPU (kernel correctness + component benchmarks)

Create `h100-single.json`:

```json
{
  "metadata": {
    "user-data": "#cloud-config\nusers:\n  - name: ubuntu\n    sudo: ALL=(ALL) NOPASSWD:ALL\n    shell: /bin/bash\n    ssh_authorized_keys:\n      - <PASTE_YOUR_PUBLIC_KEY_HERE>"
  },
  "resources": {
    "platform": "gpu-h100-sxm",
    "preset": "1gpu-16vcpu-200gb"
  },
  "boot_disk": {
    "attach_mode": "READ_WRITE",
    "initialize_params": {
      "image_id": "<UBUNTU_22_04_CUDA_12_IMAGE_ID>",
      "size_gibibytes": 200,
      "type": "NETWORK_SSD"
    }
  },
  "network_interfaces": [
    {
      "subnet_id": "<YOUR_SUBNET_ID>",
      "ip_address": {},
      "public_ip_address": {}
    }
  ]
}
```

```bash
# Find the Ubuntu + CUDA image ID
IMAGE_ID=$(nebius compute image list --parent-id <PROJECT_ID> \
  --format json | jq -r '.items[] | select(.name | contains("ubuntu-22.04-cuda-12")) | .metadata.id' | head -1)

# Find subnet ID
SUBNET_ID=$(nebius vpc subnet list --format json | jq -r '.items[0].metadata.id')

# Create the instance (replace placeholders in the JSON first)
nebius compute instance create --from-file h100-single.json
```

### Option B: 8× H100 GPU Node (full benchmarks + multi-GPU)

Create `h100-8gpu.json`:

```json
{
  "metadata": {
    "user-data": "#cloud-config\nusers:\n  - name: ubuntu\n    sudo: ALL=(ALL) NOPASSWD:ALL\n    shell: /bin/bash\n    ssh_authorized_keys:\n      - <PASTE_YOUR_PUBLIC_KEY_HERE>"
  },
  "resources": {
    "platform": "gpu-h100-sxm",
    "preset": "8gpu-128vcpu-1600gb"
  },
  "boot_disk": {
    "attach_mode": "READ_WRITE",
    "initialize_params": {
      "image_id": "<UBUNTU_22_04_CUDA_12_IMAGE_ID>",
      "size_gibibytes": 500,
      "type": "NETWORK_SSD"
    }
  },
  "network_interfaces": [
    {
      "subnet_id": "<YOUR_SUBNET_ID>",
      "ip_address": {},
      "public_ip_address": {}
    }
  ]
}
```

```bash
nebius compute instance create --from-file h100-8gpu.json
```

### Option C: Preemptible (cheapest, may be interrupted)

Add to your JSON inside `resources`:
```json
"resources": {
    "platform": "gpu-h100-sxm",
    "preset": "1gpu-16vcpu-200gb",
    "preemptible": true
}
```

## Step 5: Connect via SSH

```bash
# Get the public IP
PUBLIC_IP=$(nebius compute instance list --format json | \
  jq -r '.items[0].status.network_interfaces[0].public_ip_address.address')

ssh -i ~/.ssh/nebius_h100 ubuntu@$PUBLIC_IP
```

## Step 6: Environment Setup on the GPU Instance

```bash
# Verify GPU
nvidia-smi

# You should see: NVIDIA H100 80GB HBM3, CUDA 12.x, Driver 5xx+

# Install system dependencies
sudo apt-get update && sudo apt-get install -y git python3-pip python3-venv

# Create virtual environment
python3 -m venv ~/glm5-env
source ~/glm5-env/bin/activate

# Install PyTorch (CUDA 12.x)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install Triton (for Unsloth kernels)
pip install triton
```

## Step 7: Build FlashMLA and DeepGEMM from Source

These require CUDA 12.8+ and SM90 (H100). They JIT-compile CUDA kernels.

```bash
# FlashMLA
git clone --recurse-submodules https://github.com/deepseek-ai/FlashMLA.git
cd FlashMLA && pip install -v . && cd ..

# DeepGEMM
git clone https://github.com/deepseek-ai/DeepGEMM.git
cd DeepGEMM && pip install -v . && cd ..

# Verify installations
python3 -c "import flash_mla; print('FlashMLA OK')"
python3 -c "import deep_gemm; print(f'DeepGEMM {deep_gemm.__version__} OK')"
```

**Note:** First import of DeepGEMM triggers JIT compilation (~30-60s). Set these env vars to speed up:

```bash
export DG_JIT_USE_NVRTC=1              # 10× faster JIT compilation
export DG_JIT_CACHE_DIR=~/.deep_gemm   # Persist cache across sessions
```

## Step 8: Clone and Install the GLM-5 Project

```bash
cd ~
git clone <YOUR_GLM5_REPO_URL> glm5
cd glm5
pip install -e .  # if setup.py exists, otherwise just use PYTHONPATH
export PYTHONPATH=~/glm5:$PYTHONPATH
```

## Step 9: Run the Benchmarks

### Quick Validation (5 minutes)

```bash
# CPU tests — verify everything works
python3 -m glm5-kernels-flashmla-deepgemm.tests.run_all

# Expected: 29/29 passed
```

### H100 Kernel Correctness (10 minutes)

```bash
# Includes FlashMLA and DeepGEMM kernel tests
python3 -m glm5-kernels-flashmla-deepgemm.tests.run_all --h100

# Expected: 29 CPU + 36 H100 = 65 tests
```

### Component Benchmarks — 3-Way Comparison (15 minutes)

```bash
# Small dimensions (fast, good for sanity checking)
python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench_3way

# Full GLM-5 dimensions (requires more memory, more representative)
python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench_3way --full-dims

# Single component deep dive
python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench_3way --component mla --full-dims
```

### Per-Kernel Profiling (30 minutes)

```bash
# Print the ncu/nsys command lines
python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench --mode commands

# Quick timing benchmark
python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench --mode bench --full-dims

# nsys timeline capture (generates .nsys-rep file)
nsys profile -o ~/glm5_nsys \
  python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench --mode nsys --full-dims

# ncu kernel metrics (SLOW — profiles every kernel launch)
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum \
  -o ~/glm5_ncu \
  python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench --mode ncu --component flashmla
```

### FP8 Pareto Frontier (20 minutes)

```bash
cd ~/glm5
python3 -m benchmark.fp8_pareto.bench_fp8 --output ~/fp8_pareto_results.json
```

### Multi-GPU Tests (8× H100 only, 10 minutes)

```bash
# NCCL bandwidth + TP equivalence + expert partitioning
torchrun --nproc_per_node=8 \
  -m glm5-kernels-flashmla-deepgemm.tests.h100_test_multi_gpu
```

### Full Benchmark Suite (1-2 hours)

```bash
cd ~/glm5
python3 -m benchmark.run_all --output ~/benchmark_results.json
```

## Step 10: Download Results

```bash
# From your local machine
scp -i ~/.ssh/nebius_h100 ubuntu@$PUBLIC_IP:~/benchmark_results.json .
scp -i ~/.ssh/nebius_h100 ubuntu@$PUBLIC_IP:~/fp8_pareto_results.json .
scp -i ~/.ssh/nebius_h100 ubuntu@$PUBLIC_IP:~/glm5_nsys.nsys-rep .
scp -i ~/.ssh/nebius_h100 ubuntu@$PUBLIC_IP:~/glm5_ncu.ncu-rep .
```

## Step 11: Terminate the Instance

**Do not forget this — H100s cost $2.95/hr per GPU ($23.60/hr for 8-GPU).**

```bash
# List instances
nebius compute instance list

# Delete by ID
nebius compute instance delete --id <INSTANCE_ID>

# Verify deletion
nebius compute instance list
```

---

## Nebius H100 Instance Specs

| Spec | 1-GPU Preset | 8-GPU Preset |
|------|-------------|-------------|
| Platform | `gpu-h100-sxm` | `gpu-h100-sxm` |
| Preset | `1gpu-16vcpu-200gb` | `8gpu-128vcpu-1600gb` |
| GPUs | 1× H100 SXM 80GB | 8× H100 SXM 80GB |
| vCPUs | 16 | 128 |
| RAM | 200 GB | 1600 GB |
| GPU Memory | 80 GB HBM3 | 640 GB HBM3 (total) |
| GPU Bandwidth | 3.35 TB/s | 3.35 TB/s per GPU |
| Interconnect | N/A | NVLink 900 GB/s + 400Gb InfiniBand |
| On-demand Price | $2.95/hr | $23.60/hr |
| Preemptible | $1.25/hr | $10.00/hr |
| Region | `eu-north1` (Finland) | `eu-north1` |

## Memory Requirements for GLM-5 Benchmarks

| Benchmark | Minimum GPUs | GPU Memory Needed | Recommended Preset |
|-----------|-------------|-------------------|-------------------|
| CPU tests | 0 (no GPU) | 0 | Any |
| Kernel correctness (H100) | 1 | ~20 GB | `1gpu-16vcpu-200gb` |
| Component benchmarks (small dims) | 1 | ~10 GB | `1gpu-16vcpu-200gb` |
| Component benchmarks (full dims) | 1 | ~40 GB | `1gpu-16vcpu-200gb` |
| Single MoE layer (full dims) | 1 | ~25 GB | `1gpu-16vcpu-200gb` |
| 4-layer model (full dims) | 1 | ~60 GB | `1gpu-16vcpu-200gb` |
| Multi-GPU NCCL tests | 2+ | ~10 GB/GPU | `8gpu-128vcpu-1600gb` |
| Full 78-layer model (W4A8) | 8 | ~700 GB total | `8gpu-128vcpu-1600gb` |
| Full 78-layer model (BF16) | 16+ | ~1.4 TB total | 2× `8gpu-128vcpu-1600gb` |

## Troubleshooting

### FlashMLA build fails

```bash
# Ensure CUDA 12.8+ is available
nvcc --version  # Should show 12.8 or higher

# If CUDA is too old, install a newer version
# Nebius Ubuntu+CUDA images typically include 12.4 — you may need to upgrade:
wget https://developer.download.nvidia.com/compute/cuda/12.8.0/local_installers/cuda_12.8.0_570.86.10_linux.run
sudo sh cuda_12.8.0_570.86.10_linux.run --toolkit --silent
export CUDA_HOME=/usr/local/cuda-12.8
export PATH=$CUDA_HOME/bin:$PATH
```

### DeepGEMM JIT takes too long

```bash
# Use NVRTC for 10× faster compilation
export DG_JIT_USE_NVRTC=1

# Pre-warm all kernel shapes (optional, takes ~2 minutes)
python3 -c "
import deep_gemm
import torch
# Trigger JIT for common shapes
for h in [32, 64]:
    q = torch.randn(128, h, 128, device='cuda', dtype=torch.float8_e4m3fn)
    kv = torch.randn(4096, 128, device='cuda', dtype=torch.float8_e4m3fn)
    s = torch.ones(4096, device='cuda', dtype=torch.float32) / 448
    w = torch.randn(128, h, device='cuda', dtype=torch.float32)
    ks = torch.zeros(128, dtype=torch.int32, device='cuda')
    ke = torch.full((128,), 4096, dtype=torch.int32, device='cuda')
    deep_gemm.fp8_mqa_logits(q, (kv, s), w, ks, ke)
print('JIT warmup complete')
"
```

### Out of memory on single GPU

```bash
# Use small dimensions for testing
python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench_3way
# NOT --full-dims (which uses real GLM-5 744B dimensions)
```

### nsys/ncu not installed

```bash
# These come with the CUDA toolkit
sudo apt-get install -y nsight-systems nsight-compute
# Or download from NVIDIA developer
```

---

## Sources

- [Nebius GPU Pricing](https://nebius.com/prices)
- [Nebius Compute VM Types & Presets](https://docs.nebius.com/compute/virtual-machines/types)
- [Nebius Compute Quickstart](https://docs.nebius.com/compute/quickstart)
- [Nebius CLI Reference: compute instance create](https://docs.nebius.com/cli/reference/compute/instance/create)
- [Nebius Serving LLMs with vLLM Guide](https://nebius.com/blog/posts/serving-llms-with-vllm-practical-guide)
