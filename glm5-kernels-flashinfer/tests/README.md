# GLM-5 FlashInfer Kernel Test Suite

## Quick Start

```bash
# CPU tests (no GPU required):
python3 -m glm5-kernels-flashinfer.tests.run_all

# All tests including H100 kernel tests:
python3 -m glm5-kernels-flashinfer.tests.run_all --h100

# 3-way benchmark:
python3 -m glm5-kernels-flashinfer.tests.h100_bench_3way --full-dims

# Profiling commands:
python3 -m glm5-kernels-flashinfer.tests.h100_bench --mode commands
```

---

## Test Parity with FlashMLA+DeepGEMM

This test suite mirrors `glm5-kernels-flashmla-deepgemm/tests/` with FlashInfer-specific adaptations.

| Category | FlashMLA Tests | FlashInfer Tests | Differences |
|----------|---------------|-----------------|-------------|
| CPU equivalence | 29 (11 files) | 28 (2 files) | FlashInfer uses consolidated test_components.py; tests FlashInfer FP8 format instead of FlashMLA 656-byte format |
| H100 kernel correctness | 7 (flashmla + deepgemm) | 7 (flashinfer + deepgemm) | FlashInfer tests FA3 dense, trtllm-gen sparse, CUDA graph instead of FlashMLA kernels |
| H100 CUDA categories | 29 (10 files) | 29 (10 files) | TMA test calls FlashInfer FA3 instead of FlashMLA; all others identical |
| Benchmarks | 2 files | 2 files | bench calls FlashInfer API; 3-way imports from flashinfer package |

---

## Component Coverage (3-Way Benchmark)

| Component | Raw PyTorch | Triton | FlashInfer Kernels |
|-----------|-------------|--------|-------------------|
| **RMSNorm** | manual | `fast_rms_layernorm()` (Triton) | same Triton |
| **SwiGLU** | `F.silu(e) * g` | `swiglu_fg_kernel()` (Triton) | same Triton |
| **Cross-Entropy** | `F.cross_entropy()` | `fast_cross_entropy_loss()` (Triton) | same Triton |
| **RoPE** | PyTorch | PyTorch | PyTorch |
| **MoE Router** | `route_tokens_to_experts()` | (same) | `sigmoid_topk_route()` |
| **DSA Indexer** | PyTorch eager | (same) | DeepGEMM `fp8_mqa_logits` if avail |
| **DSA Sparse Attn** | matmul+mask | (same) | FlashInfer `trtllm_batch_decode(sparse_mla_top_k)` if avail |
| **MLA Attention** | eager | (same) | FlashInfer `BatchMLAPagedAttentionWrapper(fa3)` if avail |
| **MoE Forward** | expert loop | (same) | DeepGEMM `m_grouped_fp8_gemm` if avail |
| **Full Model** | all layers | (same) | all kernel paths active |

---

## Tolerance Reference Card

| Comparison Type | Tolerance | Rationale |
|----------------|-----------|-----------|
| BF16 vs BF16 | `atol=1e-2, rtol=1e-2` | 7-bit mantissa |
| FP8 E4M3 roundtrip | `atol=5e-2, rtol=7e-2` | 3-bit mantissa = ~6.25% worst-case |
| FP8 chained (78x) | `cos_sim > 0.90` | Errors partially cancel |
| TopK indices | `Jaccard > 0.95` | Set comparison, not ordered |
| Greedy decode | **Identical** | Deterministic = bit-exact |
| FlashInfer kernel vs PyTorch | `atol=5e-2, rtol=5e-2` | Different accumulation order |
| DeepGEMM FP8 vs BF16 | `atol=1.0, rtol=0.15` | Double quantization |
| Cross-impl logits | `atol=1e-3, rtol=1e-2` | Same weights, float ordering |

---

## File Inventory

```
tests/
  conftest.py                          # Shared helpers
  run_all.py                           # Test runner (--h100)
  README.md                            # This file
  ---- CPU Tests (28) ----
  test_equivalence.py                  # 6  vs glm5-triton reference
  test_components.py                   # 22 cu_seqlens, cache, mask, dispatch, FP8, decode, routing, grads, compat, edge
  ---- H100 Kernel Correctness (7) ----
  h100_test_flashinfer_kernels.py      # 3  FA3 dense, trtllm-gen sparse, CUDA graph
  h100_test_deepgemm_kernels.py        # 4  fp8_mqa_logits, GLM-5 dims, grouped GEMM
  ---- H100 CUDA Categories (29) ----
  h100_test_cuda_graph.py              # 3  Cat 1
  h100_test_tma.py                     # 2  Cat 2 (FlashInfer FA3 bandwidth)
  h100_test_memory.py                  # 3  Cat 3
  h100_test_fp8_edge_cases.py          # 4  Cat 4
  h100_test_multi_gpu.py               # 3  Cat 5 (torchrun)
  h100_test_launch_overhead.py         # 3  Cat 6
  h100_test_determinism.py             # 3  Cat 7
  h100_test_sparse_patterns.py         # 4  Cat 8
  h100_test_precision_chain.py         # 2  Cat 9
  h100_test_thermal.py                 # 2  Cat 10
  ---- Benchmarks ----
  h100_bench.py                        # Profiling harness (ncu/nsys)
  h100_bench_3way.py                   # 3-way comparison
```

**Total: 64 tests** (28 CPU + 7 kernel + 29 CUDA categories)
