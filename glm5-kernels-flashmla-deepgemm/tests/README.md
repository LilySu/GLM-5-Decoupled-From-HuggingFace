# GLM-5 Kernel Test Suite

## Quick Start

```bash
# CPU tests (no GPU required, ~10s):
python3 -m glm5-kernels-flashmla-deepgemm.tests.run_all

# All tests including H100 kernel tests:
python3 -m glm5-kernels-flashmla-deepgemm.tests.run_all --h100

# Multi-GPU tests (requires torchrun):
torchrun --nproc_per_node=2 -m glm5-kernels-flashmla-deepgemm.tests.h100_test_multi_gpu

# 3-way benchmark (raw PyTorch vs Triton vs Kernels):
python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench_3way --full-dims

# Generate ncu/nsys profiling commands:
python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench --mode commands
```

---

## Test Categories

### CPU Tests (29 tests, no GPU required)

| # | File | Tests | What It Validates |
|---|------|-------|-------------------|
| 1-6 | `test_equivalence.py` | 6 | Bit-exact match vs `glm5-triton` reference: router, FP8, indexer, norm, attention, full model |
| 7 | `test_deepgemm_cu_seqlens.py` | 3 | cu_seq_len_k_start/end construction for prefill, decode, chunked |
| 8 | `test_kv_cache.py` | 3 | Multi-step concatenation, reset, PagedKVCache allocation exhaustion |
| 9 | `test_dsa_mask.py` | 2 | Sparse mask pattern (selected=0, rest=-inf) and causal intersection |
| 10 | `test_moe_expert_dispatch.py` | 2 | All-to-one expert, sparse routing with empty experts |
| 11 | `test_fp8_layout.py` | 2 | 656-byte layout: nope FP8, scales power-of-2, rope BF16 lossless |
| 12 | `test_autoregressive_decode.py` | 1 | Prefill + 3 decode steps, per-step logit equivalence |
| 13 | `test_group_routing.py` | 2 | Group elimination with n_group>1, n_group=1 == flat topk |
| 14 | `test_gradient_flow.py` | 1 | Backward pass works, gradients reach all trainable params |
| 15 | `test_state_dict_compat.py` | 3 | Key names, cross-loading, parameter shapes match glm5-triton |
| 16 | `test_edge_cases.py` | 4 | B=1/S=1, topk>seq_len clamping, shared expert, empty expert |

### H100 Kernel Correctness (7 tests, require SM90 + flash-mla + deep-gemm)

| File | Tests | What It Validates |
|------|-------|-------------------|
| `h100_test_flashmla_kernels.py` | 3 | FlashMLA dense decode, sparse prefill, FP8 KV cache |
| `h100_test_deepgemm_kernels.py` | 4 | DeepGEMM fp8_mqa_logits, GLM-5 dims (H=32), grouped GEMM contiguous + masked |

### H100 CUDA-Specific Categories (29 tests)

| Cat | File | Tests | What It Validates |
|-----|------|-------|-------------------|
| 1 | `h100_test_cuda_graph.py` | 3 | Graph capture of decode step, sparse index update inside graph, graph replay speedup |
| 2 | `h100_test_tma.py` | 2 | FlashMLA bandwidth >1000 GB/s (TMA proxy), DeepGEMM TFLOPS >50 (WGMMA proxy) |
| 3 | `h100_test_memory.py` | 3 | Single MoE layer peak <60GB, KV cache linear scaling, no leak over 50 decode steps |
| 4 | `h100_test_fp8_edge_cases.py` | 4 | Outlier overflow, zero-block handling, subnormal preservation, per-tile scale correctness |
| 5 | `h100_test_multi_gpu.py` | 3 | NCCL all-reduce bandwidth, TP numerical equivalence, expert partitioning (torchrun only) |
| 6 | `h100_test_launch_overhead.py` | 3 | Empty kernel overhead, per-layer overhead <30%, graph vs eager speedup (subprocess-isolated) |
| 7 | `h100_test_determinism.py` | 3 | topk bit-identical x10, full decode bit-identical x3, DSA indexer bit-identical x5 |
| 8 | `h100_test_sparse_patterns.py` | 4 | Causality (valid-scored only), recency bias, non-degeneracy, Jaccard stability |
| 9 | `h100_test_precision_chain.py` | 2 | 78x FP8 roundtrip cos_sim >0.90, full pipeline with FP8 noise injection cos_sim >0.85 |
| 10 | `h100_test_thermal.py` | 2 | 30s sustained GEMM last/first TFLOPS >85%, GPU clock >80% of max |

### Benchmarks (run separately)

| File | Purpose |
|------|---------|
| `h100_bench.py` | Per-kernel profiling harness (ncu/nsys/timing), multi-GPU NCCL |
| `h100_bench_3way.py` | 3-way comparison: raw PyTorch vs Triton vs kernel-accelerated |

### 3-Way Benchmark Component Coverage

`h100_bench_3way.py` benchmarks **10 individual components** across all three implementations.
The table below shows which actual code path each column runs for each component:

| Component | Raw PyTorch | Triton (glm5-triton) | Kernels (glm5-kernels) |
|-----------|-------------|----------------------|------------------------|
| **RMSNorm** | `RMSNorm.forward()` (manual) | `fast_rms_layernorm()` (Unsloth Triton) | `fast_rms_layernorm()` (same Triton) |
| **SwiGLU** | `F.silu(e) * g` (PyTorch) | `swiglu_fg_kernel()` (Unsloth Triton) | `swiglu_fg_kernel()` (same Triton) |
| **Cross-Entropy** | `F.cross_entropy()` (PyTorch) | `fast_cross_entropy_loss()` (Unsloth Triton, chunked) | `fast_cross_entropy_loss()` (same Triton) |
| **RoPE (64-dim)** | `rotate_half + cat` (PyTorch) | `apply_rotary_pos_emb()` (PyTorch) | `apply_rotary_pos_emb()` (PyTorch) |
| **MoE Router** | `route_tokens_to_experts()` (PyTorch) | (same as raw) | `sigmoid_topk_route()` (PyTorch, standalone) |
| **DSA Indexer** | `DSAIndexer.forward()` (PyTorch) | (same as raw) | `DSAIndexer.forward()` (DeepGEMM fp8_mqa_logits if avail) |
| **DSA Sparse Attn** | `matmul + mask + softmax` (PyTorch) | (same as raw) | FlashMLA `flash_mla_sparse_fwd()` if avail, else same |
| **MLA Attention** | `MLAttention.forward()` (eager) | (same as raw) | FlashMLA `flash_mla_with_kvcache()` if avail, else eager |
| **MoE Forward** | `MoE.forward()` (expert loop) | (same as raw) | DeepGEMM `m_grouped_fp8_gemm` if avail, else loop |
| **Full Model** | All layers end-to-end | (same as raw) | All kernel paths active |

**Key observations:**
- Triton column differs from Raw for: RMSNorm, SwiGLU, Cross-Entropy (these have actual Triton kernels)
- Triton column = Raw for: RoPE, Router, Indexer, Sparse Attn, MLA, MoE, Full Model (no Triton kernel for these)
- Kernels column differs from Raw only when flash-mla/deep-gemm are installed (otherwise falls back to eager)

---

## Tolerance Reference Card

Use this table when deciding `atol`/`rtol` for `assert_close` or when interpreting test results.

| Comparison Type | Tolerance | Rationale |
|----------------|-----------|-----------|
| **BF16 vs BF16** (same computation) | `atol=1e-2, rtol=1e-2` | BF16 has 7-bit mantissa (~0.8% precision). Accumulated error across layers. |
| **FP8 E4M3 roundtrip** (single) | `atol=5e-2, rtol=7e-2` | 3-bit mantissa = ~6.25% worst-case relative error. |
| **FP8 chained roundtrips** (78x) | `cos_sim > 0.90` | Errors partially cancel but accumulate. Cosine similarity is more meaningful than element-wise. |
| **TopK index comparison** | `Jaccard > 0.95` | topk with `sorted=False` may return different orderings. Compare as sets, not sequences. |
| **Greedy decode tokens** | **Identical** | Deterministic model + deterministic topk = bit-exact token sequence. Any difference is a bug. |
| **FlashMLA kernel vs PyTorch** | `atol=5e-2, rtol=5e-2` | FlashMLA uses different accumulation order (online softmax). |
| **DeepGEMM FP8 GEMM vs BF16** | `atol=1.0, rtol=0.15` | FP8 quantizes both inputs. Error compounds: ~7% per input x 2 inputs. |
| **Cross-implementation logits** | `atol=1e-3, rtol=1e-2` | Same weights, same PyTorch computation path. Differences only from float ordering. |
| **NCCL TP vs single-GPU** | `atol=1e-2` | All-gather reorders tensor chunks. Floating-point addition is non-associative. |
| **Cosine similarity (cumulative)** | `> 0.90` | For measuring drift across many precision boundaries. 0.90 = strong correlation preserved. |
| **TFLOPS / bandwidth thresholds** | Varies by kernel | See individual test docstrings. Generally: >60% of peak for compute, >70% for memory. |

### When to use which metric

```
Element-wise exact?     -> torch.equal(a, b)
Element-wise close?     -> torch.allclose(a, b, atol=X, rtol=Y)
Set equality?           -> Jaccard similarity or set comparison
Distribution preserved? -> cosine_similarity(a.flatten(), b.flatten())
Ordering preserved?     -> Spearman rank correlation
```

---

## NCU Metrics Collected

When running `h100_bench.py --mode ncu`, these metrics are collected:

| Metric | What It Tells You |
|--------|-------------------|
| `sm__throughput.avg.pct_of_peak_sustained_elapsed` | SM utilization (target: >80% compute-bound) |
| `dram__throughput.avg.pct_of_peak_sustained_elapsed` | HBM bandwidth utilization (target: >70% memory-bound) |
| `sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed` | Tensor core utilization (target: >60% for GEMM) |
| `smsp__warps_issue_stalled_wait.pct` | Warps stalled waiting (high = memory-bound or barrier-bound) |
| `smsp__warps_issue_stalled_mio_throttle.pct` | Warps stalled on MIO/TMA (high = TMA descriptor bottleneck) |
| `smsp__warps_issue_stalled_math_pipe_throttle.pct` | Warps stalled on math pipe (high = compute-bound, good for GEMM) |
| `dram__bytes_read.sum` | Total HBM bytes read |
| `gpu__time_duration.sum` | Kernel execution time (ns) |
| `sm__warps_active.avg.pct_of_peak_sustained_elapsed` | Active warp occupancy (target: >50%) |

Quick profile (fewer metrics, faster):
```bash
ncu --metrics sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,gpu__time_duration.sum \
    -o glm5_quick python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench --mode ncu
```

---

## File Inventory

```
tests/
  conftest.py                          # Shared helpers, configs, skip decorators
  run_all.py                           # Test runner (--h100 for GPU tests)
  README.md                            # This file
  ---- CPU Tests (29) ----
  test_equivalence.py                  # 6  vs glm5-triton reference
  test_deepgemm_cu_seqlens.py          # 3  cu_seqlens construction
  test_kv_cache.py                     # 3  cache correctness
  test_dsa_mask.py                     # 2  sparse mask pattern
  test_moe_expert_dispatch.py          # 2  expert routing
  test_fp8_layout.py                   # 2  656-byte layout
  test_autoregressive_decode.py        # 1  multi-step decode
  test_group_routing.py                # 2  hierarchical routing
  test_gradient_flow.py                # 1  backward pass
  test_state_dict_compat.py            # 3  weight compatibility
  test_edge_cases.py                   # 4  boundary conditions
  ---- H100 Kernel Correctness (7) ----
  h100_test_flashmla_kernels.py        # 3  FlashMLA CUDA kernels
  h100_test_deepgemm_kernels.py        # 4  DeepGEMM CUDA kernels
  ---- H100 CUDA Categories (29) ----
  h100_test_cuda_graph.py              # 3  Cat 1: graph capture/replay
  h100_test_tma.py                     # 2  Cat 2: TMA verification
  h100_test_memory.py                  # 3  Cat 3: memory peak/leak
  h100_test_fp8_edge_cases.py          # 4  Cat 4: FP8 edge cases
  h100_test_multi_gpu.py               # 3  Cat 5: NCCL + TP (torchrun)
  h100_test_launch_overhead.py         # 3  Cat 6: launch overhead
  h100_test_determinism.py             # 3  Cat 7: deterministic execution
  h100_test_sparse_patterns.py         # 4  Cat 8: sparse attention
  h100_test_precision_chain.py         # 2  Cat 9: precision drift
  h100_test_thermal.py                 # 2  Cat 10: thermal throttling
  ---- Benchmarks ----
  h100_bench.py                        # Profiling harness (ncu/nsys)
  h100_bench_3way.py                   # 3-way implementation comparison
```

**Total: 65 tests** (29 CPU + 7 kernel correctness + 29 CUDA categories)
