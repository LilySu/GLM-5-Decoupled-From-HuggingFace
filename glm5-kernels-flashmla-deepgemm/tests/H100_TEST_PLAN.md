# H100 Test Plan for glm5-kernels-flashmla-deepgemm

## Current State

29 tests pass on CPU with forced eager/PyTorch paths. They validate logic correctness but **do NOT test any CUDA kernels**. See `tests/README.md` for the full inventory and CUDA-specific test recommendations (10 additional categories covering CUDA graphs, TMA verification, memory profiling, FP8 edge cases, multi-GPU NCCL, determinism, sparse attention patterns, precision boundaries, launch overhead, and thermal throttling).

This plan covers the kernel-specific tests to run on actual H100s.

---

## Test Architecture

### Three-Tier Testing Strategy

```
Tier 1: Component Kernel Tests (unit)
  Each kernel tested in ISOLATION against its PyTorch reference.
  One test file per kernel. Fast (<5s each). Run on every code change.

Tier 2: Integration Tests (subsystem)
  Test kernel COMBINATIONS that run together in a forward pass.
  Focus on precision boundaries (BF16→FP8→BF16) and data flow.
  Medium speed (~30s each). Run before merging.

Tier 3: End-to-End Tests (system)
  Full model forward/backward with real or synthetic weights.
  Validates the complete pipeline produces correct output.
  Slow (~2-5min). Run nightly or before deployment.
```

---

## Tier 1: Component Kernel Tests

### Test File: `tests/test_flashmla_kernels.py`

```
Requires: SM90, flash_mla installed

test_flashmla_dense_decode_vs_eager()
  Purpose:  Validate FlashMLA dense decode matches PyTorch eager attention
  Setup:    batch=4, num_heads=64, seq_len_kv=1024, head_dim_ckv=512, head_dim_kpe=64
  Method:   Run both FlashMLA and eager attention on same inputs
  Assert:   torch.allclose(kernel_out, eager_out, atol=1e-2, rtol=1e-2)
  Why 1e-2: FlashMLA uses BF16 internally, eager uses FP32 accumulation.
            1e-2 is the expected BF16 numerical tolerance.

test_flashmla_sparse_decode_vs_eager()
  Purpose:  Validate FlashMLA sparse decode with DSA indices
  Setup:    batch=2, num_heads=64, seq_len_kv=4096, topk=2048
            Random indices [B, 1, 2048] selecting which KV positions to attend
  Method:   FlashMLA sparse path vs masked eager attention (set non-selected to -inf)
  Assert:   torch.allclose(atol=1e-2, rtol=1e-2)

test_flashmla_sparse_prefill_vs_eager()
  Purpose:  Validate FlashMLA sparse prefill (used during initial prompt processing)
  Setup:    seq_len_q=128, seq_len_kv=128, topk=64 (smaller for prefill test)
  Method:   Compare flash_mla_sparse_fwd() vs masked eager
  Assert:   torch.allclose(atol=1e-2, rtol=1e-2)

test_flashmla_fp8_kv_cache_roundtrip()
  Purpose:  Validate FP8 KV cache quantization preserves attention quality
  Setup:    Generate BF16 KV → quantize to 656-byte FlashMLA format → decode
  Method:   Compare decode with FP8 cache vs BF16 cache
  Assert:   cosine_similarity > 0.995 (FP8 introduces small errors)

test_flashmla_paged_cache()
  Purpose:  Validate paged KV cache allocation and retrieval
  Setup:    num_pages=256, page_block_size=64, multiple sequences
  Method:   Allocate pages, populate, decode, verify output matches non-paged
  Assert:   Exact match with non-paged path (paging is a memory layout change, not compute)
```

### Test File: `tests/test_deepgemm_kernels.py`

```
Requires: SM90, deep_gemm installed

test_deepgemm_fp8_mqa_logits_vs_pytorch()
  Purpose:  Validate DeepGEMM DSA indexer scoring matches PyTorch reference
  Setup:    seq_len=1, num_heads=32, head_dim=128, seq_len_kv=4096
            cu_k_start=[0], cu_k_end=[4096] (decode mode, full context)
  Method:   Compare deep_gemm.fp8_mqa_logits(q_fp8, kv_fp8, weights, ...)
            vs torch.einsum("shd,td->sht", q, k) * scale + relu + weights
  Assert:   torch.allclose(atol=5e-2, rtol=5e-2)
  Why 5e-2: FP8 input quantization + fused computation = higher tolerance needed.
            The key metric is TOP-K AGREEMENT, not exact logit values.

test_deepgemm_fp8_mqa_logits_topk_agreement()
  Purpose:  Validate that the TOP-2048 selections match between kernel and reference
  Setup:    Same as above
  Method:   Run both paths, apply torch.topk(k=2048) to each output
  Assert:   jaccard_similarity(kernel_topk_indices, ref_topk_indices) > 0.95
  Why 0.95: FP8 rounding can flip borderline tokens near the top-2048 threshold.
            95% agreement means <100 out of 2048 tokens differ, which is acceptable.

test_deepgemm_fp8_mqa_logits_num_heads_32()
  Purpose:  Verify JIT compiles successfully for num_heads=32 (GLM-5 specific)
  Setup:    Explicitly use num_heads=32 (not 64 like DeepSeek)
  Method:   Call the kernel and verify it returns correct shape
  Assert:   output.shape == [seq_len, seq_len_kv], no JIT error

test_deepgemm_fp8_mqa_logits_causal_cu_seqlens()
  Purpose:  Validate causal cu_seqlens during prefill
  Setup:    seq_len=8, cu_k_start=[0]*8, cu_k_end=[1,2,...,8]
  Method:   Verify each query only sees its causal past
  Assert:   For position i, logits[i, j>i] should be ~0 (not attended)

test_deepgemm_grouped_gemm_vs_loop()
  Purpose:  Validate FP8 grouped GEMM matches per-expert loop
  Setup:    256 experts, 8 selected per token, hidden=6144, intermediate=2048
            Small batch: 32 tokens × 8 experts = 256 expert calls
  Method:   Compare m_grouped_fp8_gemm_nt_contiguous vs per-expert F.linear loop
  Assert:   torch.allclose(atol=5e-2, rtol=5e-2)

test_deepgemm_jit_cache_warmup()
  Purpose:  Verify JIT compilation caches correctly
  Method:   Call kernel once (cold start), time it.
            Call again (cached), verify <1ms.
  Assert:   Second call latency < 1ms (vs 10-60s for first)
```

### Test File: `tests/test_fp8_precision.py`

```
test_fp8_deepgemm_quantize_dequantize()
  Purpose:  Validate roundtrip FP8 error is within E4M3 spec
  Setup:    Random BF16 tensor [1024, 6144]
  Method:   quantize → dequantize → measure error
  Assert:   max_relative_error < 0.125 (1/8 = E4M3 worst case for mantissa=3)
            mean_relative_error < 0.03

test_fp8_flashmla_format_correctness()
  Purpose:  Validate 656-byte FlashMLA KV format is correct
  Setup:    BF16 KV [num_tokens, 576] (512 nope + 64 rope)
  Method:   quantize_kv_flashmla() → validate byte layout
  Assert:   Bytes 0-511: FP8 nope values
            Bytes 512-527: 4 float32 scales
            Bytes 528-655: BF16 rope values (unquantized)

test_fp8_accumulated_error_through_pipeline()
  Purpose:  Measure error accumulation through the BF16→FP8→BF16 pipeline
  Setup:    Simulate: BF16 input → FP8 indexer → BF16 → FP8 MoE → BF16 output
  Method:   Chain 4 quantize/dequantize operations, measure total drift
  Assert:   total_relative_error < 0.1 (10% after 4 roundtrips)
            This is the real-world pipeline: embeddings→indexer→attention→MoE→output
```

---

## Tier 2: Integration Tests

### Test File: `tests/test_attention_pipeline.py`

```
test_mla_with_dsa_indexer_integration()
  Purpose:  Test DSA indexer → sparse attention end-to-end
  Setup:    Full MLA + DSA with real dimension sizes
            batch=1, seq_len=512, all 64 heads
  Flow:     hidden → DSA indexer (→ top-2048 indices) → sparse MLA attention → output
  Method:   Run with kernels vs run with PyTorch fallback, compare outputs
  Assert:   cosine_similarity(kernel_out, eager_out) > 0.99

test_mla_prefill_then_decode()
  Purpose:  Test the prefill→decode transition
  Setup:    Prefill 32 tokens, then decode 10 tokens one at a time
  Method:   Verify KV cache grows correctly, decode output is consistent
  Assert:   Each decode step produces valid logits, cache.seq_len grows by 1

test_moe_router_plus_grouped_gemm()
  Purpose:  Test router → expert selection → grouped GEMM end-to-end
  Setup:    batch=8 tokens, 256 experts, top-8
  Flow:     router(hidden) → topk indices/weights → grouped_gemm(hidden, experts, indices)
  Method:   Compare kernel MoE vs eager per-expert loop
  Assert:   torch.allclose(atol=5e-2) — FP8 GEMM tolerance

test_dense_layer_then_moe_layer()
  Purpose:  Test the layer type transition (layers 0-2 dense, 3+ MoE)
  Setup:    2-layer model: layer 0 = dense FFN, layer 1 = MoE
  Method:   Full forward pass through both layer types
  Assert:   Output shape correct, values finite, loss computable
```

### Test File: `tests/test_weight_absorption.py`

```
test_absorb_kv_b_proj_into_q()
  Purpose:  Validate weight absorption transforms Q correctly
  Setup:    Load raw kv_b_proj weights [kv_lora_rank, num_heads * (qk_nope + v_head_dim)]
  Method:
    absorbed_q_proj = original_q_b_proj @ kv_b_proj_k_nope.T
    Verify: q_absorbed @ kv_compressed == q_original @ kv_decompressed
  Assert:   Output equivalence within BF16 tolerance

test_absorb_kv_b_proj_into_o()
  Purpose:  Validate weight absorption for output projection
  Method:   absorbed_o_proj = kv_b_proj_v.T @ original_o_proj
            Verify: absorbed_o_proj @ attention_output == original_o_proj @ v_decompressed
  Assert:   Output equivalence

test_absorbed_vs_nonabsorbed_full_attention()
  Purpose:  End-to-end: absorbed path (FlashMLA) vs non-absorbed path (eager)
  Setup:    Full MLA layer with same input, one with absorbed weights + FlashMLA, one eager
  Assert:   cosine_similarity > 0.99
```

### Test File: `tests/test_precision_boundaries.py`

```
test_bf16_to_fp8_indexer_boundary()
  Purpose:  Measure precision loss at the BF16→FP8 boundary before DSA indexer
  Setup:    BF16 hidden states [batch, seq, 6144]
  Method:   Quantize to FP8, run indexer, compare topk selections vs BF16 path
  Assert:   topk_agreement > 0.95

test_fp8_to_bf16_after_moe()
  Purpose:  Measure precision loss at FP8→BF16 boundary after MoE GEMM
  Method:   Run MoE with FP8 GEMM, dequantize, compare vs BF16 MoE
  Assert:   max_element_error < 0.5, cosine_similarity > 0.99

test_full_pipeline_precision_chain()
  Purpose:  Measure TOTAL accumulated error through one full decoder layer
  Flow:     BF16 hidden → FP8 indexer → BF16 scores → FlashMLA(BF16 KV) →
            BF16 attn_out → FP8 MoE GEMM → BF16 layer_out
  Compare:  All-BF16 eager path vs mixed-precision kernel path
  Assert:   cosine_similarity > 0.98 per layer
            After 78 layers: cosine_similarity > 0.90 (accumulated)
```

---

## Tier 3: End-to-End Tests

### Test File: `tests/test_e2e_inference.py`

```
test_e2e_generate_greedy()
  Purpose:  Full inference: tokenize → prefill → decode loop → detokenize
  Setup:    Load model (tiny config for CI, full config for nightly)
            Prompt: "The capital of France is"
  Method:   Generate 20 tokens greedily
  Assert:   Output is coherent text (not garbage)
            Each token generated in <X ms (latency check)

test_e2e_kernel_vs_eager_same_output()
  Purpose:  Verify kernel path produces IDENTICAL greedy tokens as eager path
  Setup:    Same prompt, same temperature=0 (deterministic)
  Method:   Generate 50 tokens with kernels, 50 with eager
  Assert:   Token sequences are IDENTICAL (greedy decode is deterministic)
  Why:      If tokens diverge, there's a numerical issue in the kernel path

test_e2e_long_context_dsa()
  Purpose:  Validate DSA works at long context lengths
  Setup:    Input: 8K tokens (prompt) + generate 100 tokens
  Method:   DSA indexer selects top-2048 from 8K+ KV cache each decode step
  Assert:   No crashes, output coherent, DSA indices are reasonable
            (not all clustered at one position)

test_e2e_batch_inference()
  Purpose:  Validate batched inference works correctly
  Setup:    4 different prompts, varying lengths, batch processed
  Method:   Generate 20 tokens per prompt
  Assert:   Each prompt's output is independent (no cross-contamination)
            Padded positions don't affect results
```

### Test File: `tests/test_e2e_training.py`

```
test_training_forward_backward()
  Purpose:  Verify gradients flow through the full model
  Setup:    Tiny config (128D, 2 layers), batch of training data
  Method:   Forward → compute loss → backward
  Assert:   >75% of parameters have non-zero gradients
            Loss is finite

test_training_convergence_tiny()
  Purpose:  Verify model can overfit a tiny dataset
  Setup:    Tiny config, 20 training steps on 4 samples
  Method:   Train and measure loss at step 0 vs step 20
  Assert:   Loss decreases by >80%

test_gradient_checkpointing_equivalence()
  Purpose:  Verify gradient checkpointing doesn't change gradients
  Setup:    Same input, same model, one with checkpointing, one without
  Assert:   Gradients match within BF16 tolerance
```

---

## Performance Benchmarks

### Test File: `tests/benchmark_kernels.py`

Not pytest tests — standalone benchmarking script. Output is a table.

```python
# Run: python tests/benchmark_kernels.py

Components to benchmark:
  1. FlashMLA dense decode:      measure TFLOPS at seq_len_kv = [512, 1K, 4K, 16K, 64K]
  2. FlashMLA sparse decode:     measure TFLOPS at seq_len_kv = [4K, 16K, 64K], topk=2048
  3. DeepGEMM fp8_mqa_logits:    measure TFLOPS at seq_len_kv = [512, 1K, 4K, 16K]
  4. DeepGEMM grouped GEMM:      measure TFLOPS at batch = [8, 32, 128, 512]
  5. Triton RMSNorm:              measure GB/s at hidden_size=6144
  6. Triton SwiGLU:               measure GB/s at hidden_size=6144
  7. Full decoder layer:          measure ms/layer at batch=1, seq=1 (decode)
  8. Full decoder layer:          measure ms/layer at batch=1, seq=128 (prefill)

Each benchmark:
  - 10 warmup iterations (JIT compilation, cache warming)
  - 100 timed iterations
  - Report: mean, std, min, max, TFLOPS or GB/s
  - Compare: kernel vs PyTorch eager vs Triton (where applicable)

Output format:
  | Component          | Kernel    | Eager     | Speedup | TFLOPS |
  |--------------------|-----------|-----------|---------|--------|
  | FlashMLA decode    | 0.12ms    | 2.8ms     | 23.3x   | 410    |
  | DeepGEMM indexer   | 0.05ms    | 0.9ms     | 18.0x   | -      |
  | DeepGEMM MoE       | 0.08ms    | 1.2ms     | 15.0x   | 1550   |
  | Triton RMSNorm     | 0.003ms   | 0.008ms   | 2.7x    | -      |
```

---

## Test Configuration

### conftest.py Extensions

```python
# Add to existing conftest.py:

@pytest.fixture
def h100_device():
    """Returns a CUDA device, skipping if not SM90+."""
    if not torch.cuda.is_available():
        pytest.skip("No CUDA")
    cap = torch.cuda.get_device_capability()
    if cap[0] < 9:
        pytest.skip(f"Requires SM90+ (H100), got SM{cap[0]}{cap[1]}")
    return torch.device("cuda")

@pytest.fixture
def tiny_model_on_gpu(h100_device):
    """Create a tiny GLM-5 model on H100 with kernel acceleration."""
    cfg = make_cfg()
    cfg["use_flash_mla"] = True
    cfg["use_deepgemm"] = True
    model = GlmMoeDsaForCausalLM(cfg).to(h100_device).bfloat16()
    return model, cfg

@pytest.fixture
def reference_model_on_gpu(h100_device):
    """Create a tiny reference model (PyTorch eager) on GPU."""
    cfg = make_cfg()
    cfg["use_flash_mla"] = False
    cfg["use_deepgemm"] = False
    model = GlmMoeDsaForCausalLM(cfg).to(h100_device).bfloat16()
    return model, cfg

PRECISION_TOLERANCES = {
    "bf16_vs_bf16": {"atol": 1e-3, "rtol": 1e-3},      # Same precision
    "bf16_vs_fp8":  {"atol": 5e-2, "rtol": 5e-2},      # FP8 quantization
    "kernel_vs_eager": {"atol": 1e-2, "rtol": 1e-2},    # BF16 kernel vs FP32 eager
    "topk_agreement": 0.95,                                # 95% index overlap
    "cosine_similarity": 0.99,                             # Minimum cos sim
}
```

### pytest Markers

```ini
# pytest.ini or pyproject.toml
[tool.pytest.ini_options]
markers = [
    "cpu: runs on CPU (no GPU needed)",
    "sm90: requires SM90 (H100/H800)",
    "slow: takes >30s",
    "nightly: run only in nightly CI",
    "benchmark: performance benchmark (not correctness)",
]
```

### CI Pipeline

```
On every push:       pytest -m "cpu" tests/              # 14 existing tests (~5s)
On PR merge:         pytest -m "cpu or sm90" tests/      # All unit + kernel tests (~2min)
Nightly:             pytest tests/                         # Everything including e2e (~10min)
                     python tests/benchmark_kernels.py     # Performance regression check
```

---

## Key Testing Principles

### 1. Dual-Path Architecture
Every kernel component has a PyTorch fallback. Tests always run BOTH paths and compare.
```python
kernel_out = component(x, use_kernel=True)
eager_out  = component(x, use_kernel=False)
assert_close(kernel_out, eager_out, **TOLERANCES["kernel_vs_eager"])
```

### 2. TopK Agreement > Exact Numerics
For DSA indexer, the SELECTION matters more than exact logit values. FP8 rounding can flip borderline tokens, but if 95%+ of top-2048 tokens agree, the attention output will be nearly identical.

### 3. Cosine Similarity for High-Dimensional Outputs
For attention and MoE outputs (6144-dim vectors), cosine similarity is more meaningful than element-wise error. A cos_sim > 0.99 means the direction is preserved even if magnitudes shift slightly.

### 4. Cumulative Precision Tracking
The 78-layer pipeline accumulates FP8 rounding errors. Test both:
- Per-layer: cos_sim > 0.99
- End-to-end: cos_sim > 0.90 (allows for 78 layers of accumulated drift)

### 5. Deterministic Baseline
Greedy decode (temperature=0) with the same seed should produce IDENTICAL token sequences from kernel and eager paths. Any divergence indicates a numerical bug, not just rounding.

---

## File Summary

```
tests/
├── conftest.py                     # Fixtures, tolerances, markers (EXTEND)
├── test_equivalence.py             # Existing 6 CPU tests (KEEP)
├── test_dsa_mask.py                # Existing 2 CPU tests (KEEP)
├── test_kv_cache.py                # Existing 3 CPU tests (KEEP)
├── test_deepgemm_cu_seqlens.py     # Existing 3 CPU tests (KEEP)
├── test_flashmla_kernels.py        # NEW: 5 FlashMLA kernel tests (SM90)
├── test_deepgemm_kernels.py        # NEW: 6 DeepGEMM kernel tests (SM90)
├── test_fp8_precision.py           # NEW: 3 FP8 precision tests (SM90)
├── test_attention_pipeline.py      # NEW: 4 integration tests (SM90)
├── test_weight_absorption.py       # NEW: 3 absorption tests (SM90)
├── test_precision_boundaries.py    # NEW: 3 precision chain tests (SM90)
├── test_e2e_inference.py           # NEW: 4 end-to-end tests (SM90)
├── test_e2e_training.py            # NEW: 3 training tests (SM90)
├── benchmark_kernels.py            # NEW: Performance benchmark script
└── H100_TEST_PLAN.md               # This document
```

**Total: 14 existing + 31 new = 45 tests + benchmark suite**
