# GLM-5 H100 Benchmark Results — Run 1

**Date:** March 23, 2026
**Hardware:** NVIDIA H100 80GB HBM3 (SM90)
**PyTorch:** 2.8.0+cu128
**Libraries:** FlashMLA (source), DeepGEMM (source), FlashInfer

## Kernel-Level Results (h100_bench --full-dims)

| Component | Latency | TFLOPS | MFU% | Notes |
|-----------|---------|--------|------|-------|
| **FlashMLA decode** | 0.080 ms | **229.5** | 23.2% | B=32, T=4096, H=64, d_qk=576, d_v=512 |
| **DeepGEMM fp8_mqa_logits** | 0.033 ms | 1.0 | — | DSA indexer, H=32, D=128 |
| **DeepGEMM BF16 grouped GEMM** | 0.336 ms | **613.9** | **62.1%** | E=256, I=2048, D=6144, M=8192 |
| **MoE router** | 0.094 ms | — | — | Sigmoid + topk(8), 256 experts |
| **Dense layer** | 2.721 ms | — | — | B=1, S=128, full GLM-5 dims |
| **MoE layer** | 4.866 ms | — | — | B=1, S=128, full GLM-5 dims |
| **Full model** | OOM | — | — | 78 layers needs 8×H100 |

## MoE Sweep Results (SC '25 standard, batch=64)

| Tokens | BF16 Loop (ms) | BF16 Loop TFLOPS | DeepGEMM BF16 (ms) | DeepGEMM TFLOPS | Speedup |
|--------|---------------|-----------------|--------------------|-----------------|---------|
| 128 | ~38 ms | 2.0 | — | — | — |
| 256 | ~40 ms | 3.8 | — | — | — |
| 512 | ~42 ms | 7.2 | 30.8 ms | 10.0 | 1.4× |
| 1024 | ~42 ms | 14.6 | 31.5 ms | 19.6 | 1.3× |
| 2048 | ~43 ms | 28.8 | 32.9 ms | 37.6 | 1.3× |

**Key finding:** DeepGEMM BF16 grouped GEMM achieves consistent ~31-33ms regardless of token count (throughput scales linearly), while PyTorch per-expert loop saturates at ~43ms. At 2048 tokens: **1.3× speedup** with DeepGEMM, and the gap widens at larger batch sizes.

## FP8 Pareto Results

| Context | BF16 (ms) | FlashMLA FP8 (ms) | FlashInfer FP8 (ms) | cos_sim | RMSE |
|---------|-----------|-------------------|---------------------|---------|------|
| 1024 | 11.705 | 11.713 | 11.713 | 0.9993 | 0.00193 |
| 4096 | 46.621 | 46.690 | 46.684 | 0.9993 | 0.00098 |

**Key finding:** Both FP8 formats produce identical quality (cos_sim=0.9993). FlashMLA per-tile and FlashInfer global scale are indistinguishable at these context lengths. No speedup over BF16 in the eager attention path — real speedup requires the actual FlashMLA/FlashInfer CUDA kernels.

## Latency Breakdown (Single Forward Pass Estimate)

For B=1, S=128 on 1×H100:
```
Dense layer (×3):    2.721 ms × 3 =   8.16 ms
MoE layer (×75):     4.866 ms × 75 = 364.95 ms
                                     ─────────
Estimated total:                      373.11 ms (~2.7 tokens/sec for decode)
```

This is WITHOUT tensor parallelism. With 8×H100 TP:
- Communication overhead: ~2 all-reduces per layer × 78 layers
- Estimated: 373 / 8 + comm_overhead ≈ **50-60 ms per token** (16-20 tokens/sec)

## DeepGEMM API Findings

Documented in `benchmark/README.md`:
- `fp8_mqa_logits`: q must be raw FP8 tensor (NOT tuple), kv scales must be 1D
- `m_grouped_bf16_gemm_nt_contiguous`: confirmed 613.9 TFLOPS (62.1% MFU)
- `per_block_cast_to_fp8` needs `use_ue8m0: bool` argument
- `per_custom_dims_cast_to_fp8` produces 1D scales — incompatible with GEMM kernels
- FP8 grouped GEMM needs K > 128 for meaningful block scales
