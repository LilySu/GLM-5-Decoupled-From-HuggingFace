#!/bin/bash
# Fix DeepGEMM JIT compilation: CUTLASS headers not found
cd /workspace/DeepGEMM
rm -rf deep_gemm.egg-info
rm -rf /root/.deep_gemm
rm -rf /workspace/.deep_gemm_cache/cache
pip install --no-build-isolation --force-reinstall .
cd /workspace
python3 -c "import deep_gemm; print('Installed from:', deep_gemm.__path__)"
export CPLUS_INCLUDE_PATH=/workspace/DeepGEMM/third-party/cutlass/include:/workspace/DeepGEMM/third-party/fmt/include:$CPLUS_INCLUDE_PATH
export DG_JIT_CACHE_DIR=/workspace/.deep_gemm_cache
export DG_JIT_USE_NVRTC=1
python3 /workspace/GLM-5-Decoupled-From-HuggingFace/test_deepgemm_jit.py
