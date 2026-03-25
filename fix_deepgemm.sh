#!/bin/bash
# Fix DeepGEMM: remove /workspace/DeepGEMM from PYTHONPATH so site-packages version is used
export PYTHONPATH=$(echo $PYTHONPATH | tr ':' '\n' | grep -v '/workspace/DeepGEMM' | tr '\n' ':' | sed 's/:$//')
export DG_JIT_CACHE_DIR=/workspace/.deep_gemm_cache
export DG_JIT_USE_NVRTC=1
rm -rf /workspace/.deep_gemm_cache/cache /root/.deep_gemm 2>/dev/null

# Patch CUDA 12.8 incompatibility: cute::numeric_limits<float>::infinity() is not constexpr
LOGITS_FILE=/usr/local/lib/python3.12/dist-packages/deep_gemm/include/deep_gemm/impls/smxx_clean_logits.cuh
if [ -f "$LOGITS_FILE" ]; then
    sed -i 's/constexpr float neg_inf = -cute::numeric_limits<float>::infinity()/constexpr float neg_inf = -__builtin_inff()/' "$LOGITS_FILE"
    echo "Patched smxx_clean_logits.cuh for CUDA 12.8 compatibility."
fi

cd /workspace
python3 -c "import deep_gemm; print('Installed from:', deep_gemm.__path__)"
python3 /workspace/GLM-5-Decoupled-From-HuggingFace/test_deepgemm_jit.py
