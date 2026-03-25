#!/bin/bash
# Fix DeepGEMM: remove stale editable install so site-packages version is used
cd /workspace/DeepGEMM
rm -rf deep_gemm.egg-info
rm -rf /root/.deep_gemm
rm -rf /workspace/.deep_gemm_cache/cache
find /usr/local/lib/python3.12/dist-packages -name "deep*gemm*" -type f -exec rm -f {} \; 2>/dev/null
find /usr/local/lib/python3.12/dist-packages -name "deep*gemm*" -type d -exec rm -rf {} \; 2>/dev/null
find /usr/local/lib/python3.12/dist-packages -name "__editable__*deep*" -exec rm -f {} \; 2>/dev/null
pip install --no-build-isolation .
export DG_JIT_CACHE_DIR=/workspace/.deep_gemm_cache
export DG_JIT_USE_NVRTC=1
cd /workspace
python3 -c "import deep_gemm; print('Installed from:', deep_gemm.__path__)"
python3 /workspace/GLM-5-Decoupled-From-HuggingFace/test_deepgemm_jit.py
