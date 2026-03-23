# Running GLM-5 Kernel Benchmarks on RunPod

## Quick Start

1. Deploy a pod with **H100 SXM** or **H200 SXM** using template `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`
2. Open the **Web Terminal**
3. Follow the steps below

## Cost

| GPU | On-Demand | Spot |
|-----|-----------|------|
| H100 SXM 80GB | ~$2.49/hr | ~$1.69/hr |
| H200 SXM 141GB | ~$3.49/hr | ~$2.39/hr |

## Pod Configuration

| Setting | Value |
|---------|-------|
| GPU | H100 SXM 80GB or H200 SXM 141GB |
| GPU Count | 1 |
| Template | `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404` |
| Container Disk | 30 GB |
| Volume Disk | 100 GB |
| SSH Terminal Access | Checked |

## Step 1: Verify GPU

```bash
nvidia-smi
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
python3 -c "import torch; print(torch.version.cuda)"
```

Expected: H100/H200, PyTorch 2.8.0, CUDA 12.8.

## Step 2: Clone the Project

```bash
cd /workspace
git clone https://github.com/YOUR_USERNAME/glm5.git
cd glm5
```

If the repo is private, use a GitHub personal access token:

```bash
git clone https://YOUR_TOKEN@github.com/YOUR_USERNAME/glm5.git
```

Or rsync from your local machine (requires SSH key in RunPod Settings):

```bash
# From your LOCAL terminal, not the pod:
rsync -avz -e "ssh -p PORT -i ~/.ssh/id_ed25519" /home/lily/wsl_git/glm5/ root@HOST:/workspace/glm5/
```

## Step 3: Install Dependencies

```bash
cd /workspace
pip install triton
```

### Build FlashMLA

FlashMLA requires CUDA 12.8+ and the CUTLASS submodule. Use `--depth=1` to avoid disk exhaustion on the submodule checkout.

```bash
git clone --recurse-submodules --depth=1 https://github.com/deepseek-ai/FlashMLA.git
cd FlashMLA
git submodule update --init --recursive --depth=1
FLASH_MLA_DISABLE_SM100=1 pip install -v --no-build-isolation /workspace/FlashMLA
cd ..
```

**Notes:**
- `--no-build-isolation` is required because the build needs torch in the environment
- `FLASH_MLA_DISABLE_SM100=1` skips Blackwell compilation (needs CUDA 12.9, we only need SM90 for H100/H200)
- `--depth=1` prevents the CUTLASS submodule from filling disk with 100K+ HTML doc files

### Build DeepGEMM

```bash
git clone https://github.com/deepseek-ai/DeepGEMM.git
cd DeepGEMM
git submodule update --init --recursive --depth=1
pip install -v --no-build-isolation /workspace/DeepGEMM
cd ..
```

### Configure DeepGEMM JIT

```bash
export DG_JIT_USE_NVRTC=1
export DG_JIT_CACHE_DIR=/workspace/.deep_gemm
```

Add to `~/.bashrc` so it persists across terminal sessions:

```bash
echo 'export DG_JIT_USE_NVRTC=1' >> ~/.bashrc
echo 'export DG_JIT_CACHE_DIR=/workspace/.deep_gemm' >> ~/.bashrc
```

## Step 4: Verify Installations

**Important:** Run verification from `/workspace`, NOT from inside `/workspace/FlashMLA` or `/workspace/DeepGEMM`. Being inside those directories causes circular import errors because Python finds the local source folder instead of the installed package.

```bash
cd /workspace
```

```bash
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

```bash
python3 -c "import flash_mla; print('FlashMLA OK')"
```

```bash
python3 -c "import deep_gemm; print('DeepGEMM ' + deep_gemm.__version__ + ' OK')"
```

All three should print OK. First DeepGEMM import takes ~30-60s (JIT compilation).

**Build timing notes (observed 2026-03-23 on RunPod H100 SXM):**
- FlashMLA build: ~5-10 minutes (compiles SM90 CUDA kernels)
- DeepGEMM build: ~1-2 minutes (mostly Python, kernels are JIT-compiled at runtime)
- FlashMLA build runs in parallel with DeepGEMM if you open two terminals

## Step 5: Run Benchmarks

```bash
cd /workspace/glm5
export PYTHONPATH=/workspace/glm5:$PYTHONPATH
```

### CPU Tests (quick sanity check, ~10s)

```bash
python3 -m glm5-kernels-flashmla-deepgemm.tests.run_all
```

### H100 Kernel Correctness (~10min)

```bash
python3 -m glm5-kernels-flashmla-deepgemm.tests.run_all --h100
```

### 3-Way Component Benchmark

```bash
python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench_3way
```

### Full GLM-5 Dimensions

```bash
python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench_3way --full-dims
```

### Per-Kernel Profiling

```bash
python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench --mode bench --full-dims
```

## Step 6: Download Results

From your **local** terminal:

```bash
scp -P PORT -i ~/.ssh/id_ed25519 root@HOST:/workspace/glm5/bench_results.json .
```

## Step 7: Stop the Pod

Go to RunPod dashboard → click **Stop** (not Terminate).
- **Stop** preserves your volume disk — restart later without reinstalling
- **Terminate** deletes everything

## Troubleshooting

### Problem: "No module named torch" during FlashMLA or DeepGEMM build

**Date encountered:** 2026-03-23

**Symptom:** `pip install -v .` fails with `ModuleNotFoundError: No module named 'torch'` even though `python3 -c "import torch"` works.

**Cause:** pip's build isolation creates a clean virtualenv for compilation that doesn't include torch. The RunPod template installs torch system-wide, but the isolated build subprocess can't see it.

**Fix:** Always use `--no-build-isolation`:

```bash
pip install -v --no-build-isolation /workspace/FlashMLA
```

---

### Problem: FlashMLA build fails with "sm100 requires NVCC 12.9"

**Date encountered:** 2026-03-23

**Symptom:** `AssertionError: sm100 compilation for Flash MLA requires NVCC 12.9 or higher`

**Cause:** FlashMLA tries to compile Blackwell (SM100) kernels by default. CUDA 12.8 only supports up to SM90 (H100/H200). SM100 needs CUDA 12.9+.

**Fix:** Disable SM100 compilation (you only need SM90 for H100/H200):

```bash
FLASH_MLA_DISABLE_SM100=1 pip install -v --no-build-isolation /workspace/FlashMLA
```

---

### Problem: CUTLASS submodule checkout fails, fills disk, or aborts

**Date encountered:** 2026-03-23

**Symptom:** `fatal: Unable to checkout '147f567...' in submodule path 'csrc/cutlass'` or build fails with `fatal error: cutlass/bfloat16.h: No such file or directory`

**Cause:** CUTLASS repo is very large (~100K HTML doc files). Shallow clone (`--depth=1`) can fail or abort mid-checkout. If the checkout is incomplete, header files are missing and compilation fails.

**Fix:** Remove and re-initialize the submodule:

```bash
cd /workspace/FlashMLA   # or /workspace/DeepGEMM
rm -rf csrc/cutlass
git submodule deinit -f csrc/cutlass
git submodule update --init --recursive --depth=1
```

Verify the fix:

```bash
ls csrc/cutlass/include/cutlass/bfloat16.h
```

If this still fails, try without `--depth=1` (slower but reliable):

```bash
rm -rf csrc/cutlass
git submodule update --init --recursive
```

Same fix applies to DeepGEMM (`third-party/cutlass/`):

```bash
cd /workspace/DeepGEMM
rm -rf third-party/cutlass
git submodule deinit -f third-party/cutlass
git submodule update --init --recursive --depth=1
ls third-party/cutlass/include/cute
```

---

### Problem: DeepGEMM first import hangs for 30-60 seconds

**Date encountered:** 2026-03-23

**Symptom:** `import deep_gemm` takes 30-60 seconds the first time.

**Cause:** DeepGEMM JIT-compiles CUDA kernels on first import using nvcc (slow).

**Fix:** Use NVRTC for 10x faster compilation:

```bash
export DG_JIT_USE_NVRTC=1
export DG_JIT_CACHE_DIR=/workspace/.deep_gemm
```

---

### Problem: SSH "Permission denied (publickey)" for rsync/scp

**Date encountered:** 2026-03-23

**Symptom:** `Permission denied (publickey)` when trying to rsync or scp files to the pod.

**Cause:** RunPod injects SSH keys at pod boot time. If no key is configured, or the key doesn't match, SSH auth fails.

**Fix:**

1. Copy your public key: `cat ~/.ssh/id_ed25519.pub`
2. Go to RunPod → **Settings** → **SSH Public Keys** → paste → **Save**
3. **Stop** and **Start** the pod (key is injected at boot)
4. Use `-i` flag explicitly:

```bash
rsync -avz -e "ssh -p PORT -i ~/.ssh/id_ed25519" /home/lily/wsl_git/glm5/ root@HOST:/workspace/glm5/
```

**Workaround:** Use the **Web Terminal** in RunPod's dashboard instead of SSH — no key needed.

---

### Problem: Git clone with SSH fails ("Permission denied (publickey)")

**Date encountered:** 2026-03-23

**Symptom:** `git clone git@github.com:user/repo.git` fails with `Permission denied (publickey)`.

**Cause:** The pod doesn't have SSH keys configured for GitHub.

**Fix:** Use HTTPS instead:

```bash
git clone https://github.com/user/repo.git
```

For private repos, use a personal access token:

```bash
git clone https://TOKEN@github.com/user/repo.git
```

---

### Problem: Line breaks in web terminal split commands

**Date encountered:** 2026-03-23

**Symptom:** Commands pasted from documentation get split across lines, causing errors like `command not found` or `flag needs an argument`.

**Cause:** The RunPod web terminal wraps long lines and the shell interprets the wrapped text as separate commands.

**Fix:** Either:
- Type commands manually (don't paste multi-line)
- Use `=` to attach flag values: `--flag=value` instead of `--flag value`
- Paste commands one at a time, ensuring each is a single line

---

### Problem: "cannot import name '_C' from partially initialized module 'deep_gemm' (circular import)"

**Date encountered:** 2026-03-23

**Symptom:** `import deep_gemm` fails with `ImportError: cannot import name '_C' from partially initialized module 'deep_gemm' (most likely due to a circular import)`

**Cause:** You're running Python from inside `/workspace/DeepGEMM/`. Python finds the local `deep_gemm/` source directory before the installed package, causing a circular import. Same issue applies to FlashMLA if you run from inside `/workspace/FlashMLA/`.

**Fix:** Always `cd /workspace` (or any other directory) before importing:

```bash
cd /workspace
python3 -c "import deep_gemm; print('OK')"
```
