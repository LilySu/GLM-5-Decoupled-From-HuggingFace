# GLM-5 FlashInfer Kernel Path — Research & Plan

## Status: IMPLEMENTED (28 CPU tests passing)

This directory contains the FlashInfer-based kernel path for GLM-5, implemented as an
alternative to the FlashMLA+DeepGEMM path (see `glm5-kernels-flashmla-deepgemm/`).

FlashInfer is preferred when:
- CUDA graph support is needed (native plan/run pattern)
- Serving via vLLM/SGLang (FlashInfer is their primary MLA backend)
- SM80 (A100) fallback is needed (FlashInfer supports SM80, FlashMLA requires SM90)
- Multi-backend flexibility (FA2/FA3/trtllm-gen selectable per workload)

---

## FlashInfer MLA Module Analysis

### Repository
- **URL:** github.com/flashinfer-ai/flashinfer
- **License:** Apache 2.0
- **MLA files analyzed:**
  - `flashinfer/mla.py` (~900 lines) — full Python API
  - `csrc/batch_decode_mla_*.cu` — decode kernels
  - `csrc/batch_mla_sm90_*.cu` — Hopper-optimized kernels
  - `csrc/cutlass_mla.cu` — CUTLASS MLA
  - `csrc/xqa/mla_sm120.cu` — Blackwell XQA MLA
  - `include/flashinfer/attention/mla.cuh`, `mla_hopper.cuh`, `mla_params.cuh`
  - `include/flashinfer/attention/cutlass_mla.cuh`
  - `include/flashinfer/attention/decode_mla_cute_sm80.cuh`
  - `include/flashinfer/attention/blackwell/` — SM100 kernels

### Key API: BatchMLAPagedAttentionWrapper

```python
wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(
    float_workspace_buffer=torch.empty(128*1024*1024, dtype=torch.int8, device="cuda"),
    backend="auto"  # "fa2", "fa3", "cutlass", "trtllm-gen"
)

# Plan phase (pre-computes scheduling metadata)
wrapper.plan(
    qo_indptr,        # [batch+1] int32
    kv_indptr,        # [batch+1] int32
    kv_indices,       # [kv_indptr[-1]] int32 — page indices
    kv_len_arr,       # [batch] int32
    num_heads=64,     # GLM-5: 64 heads
    head_dim_ckv=512, # compressed KV dim
    head_dim_kpe=64,  # RoPE dim
    page_size=1,      # or 64, etc.
    causal=True,
    sm_scale=1.0 / ((512 + 64) ** 0.5),
    q_data_type=torch.bfloat16,
    kv_data_type=torch.bfloat16,
)

# Run phase
out = wrapper.run(
    q_nope,    # [batch, num_heads, 512] BF16
    q_pe,      # [batch, num_heads, 64] BF16
    ckv_cache, # [num_pages, page_size, 512] BF16
    kpe_cache, # [num_pages, page_size, 64] BF16
)
# out shape: [batch, num_heads, 512]
```

### Key API: trtllm_batch_decode_with_kv_cache_mla

```python
out = flashinfer.mla.trtllm_batch_decode_with_kv_cache_mla(
    query,              # [batch, q_len, num_heads, 576] — concat(q_nope, q_pe)
    kv_cache,           # [num_pages, page_size, 576] — concat(ckv, kpe)
    workspace_buffer,
    qk_nope_head_dim=128,  # or 64 for smaller models
    kv_lora_rank=512,
    qk_rope_head_dim=64,
    block_tables=page_table,
    seq_lens=seq_lens,
    max_seq_len=max_seq,
    sparse_mla_top_k=2048,  # DSA sparse attention support!
    bmm1_scale=sm_scale,
    bmm2_scale=1.0,
    backend="trtllm-gen",  # or "xqa" for SM120
)
```

### Explicit GLM-5 Support

```python
# From flashinfer/mla.py:
supported_mla_layer_dimensions = [
    MLALayerDimensions(head_dimensions=deepseek_mla_dimensions, num_heads=128),  # DSR1
    MLALayerDimensions(head_dimensions=deepseek_mla_dimensions, num_heads=64),   # GLM-5
    MLALayerDimensions(head_dimensions=smaller_mla_dimensions, num_heads=32),    # Smaller
]
```

GLM-5 is explicitly listed with `num_heads=64` and `deepseek_mla_dimensions` (kv_lora_rank=512, qk_rope_head_dim=64).

---

## Why FlashInfer Was NOT Selected

### Reason 1: CUTLASS Backend Hardcodes 128 Heads

```python
# From flashinfer/mla.py _check_cutlass_shape():
if H != 128:
    raise ValueError(f"Expected 128 heads for q_nope_pe, got {H}")
if D_q != D_ckv or D_q != 576:
    raise ValueError(f"Expected head dim 576...")
```

GLM-5 has 64 heads. The CUTLASS backend would fail with a ValueError. Only the TRT-LLM gen and FA2/FA3 backends work with GLM-5.

### Reason 2: More Complex API (plan/run pattern)

FlashInfer requires a two-phase plan/run pattern with workspace buffers:
1. Allocate 128MB workspace buffer
2. Call `wrapper.plan(...)` with paging metadata
3. Call `wrapper.run(...)` with actual tensors

FlashMLA is simpler: single function call `flash_mla_with_kvcache(q, k_cache, ...)`.

### Reason 3: FlashMLA Has Higher Reported TFLOPS

FlashMLA reports 660 TFLOPS dense decode, 640 TFLOPS sparse prefill on H800. FlashInfer wraps FlashMLA as one of its backends but adds framework overhead.

### Reason 4: Separate q_nope/q_pe Tensors

FlashInfer takes `q_nope` and `q_pe` as separate tensors. FlashMLA takes a single concatenated Q. The concatenated format is simpler when working with absorbed weights.

---

## When FlashInfer Would Be Better

1. **CUDA Graph support** — FlashInfer's plan/run pattern is designed for CUDA graph capture. FlashMLA would need custom graph integration.

2. **Serving framework integration** — vLLM and SGLang use FlashInfer as a primary backend. For production serving, FlashInfer is the natural choice.

3. **Multi-backend flexibility** — FlashInfer can dispatch to FA2, FA3, CUTLASS, TRT-LLM gen, or XQA backends based on hardware and problem size. FlashMLA is SM90/SM100 only.

4. **Sparse MLA via trtllm-gen** — The `sparse_mla_top_k` parameter provides integrated DSA sparse attention decode without needing a separate sparse kernel.

5. **SM80 (A100) fallback** — FlashInfer's `decode_mla_cute_sm80.cuh` provides A100 support. FlashMLA requires SM90+.

---

## Sources Consulted

| # | URL | What was found |
|---|-----|----------------|
| 1 | `flashinfer/mla.py` lines 1-200 | MLAHeadDimensions, supported dimensions, CUTLASS shape check |
| 2 | `flashinfer/mla.py` lines 200-600 | BatchMLAPagedAttentionWrapper (plan, run methods) |
| 3 | `flashinfer/mla.py` lines 600-900 | trtllm_batch_decode, xqa_batch_decode, sparse_mla_top_k |
| 4 | GitHub API tree (MLA files) | 35 MLA-related files across csrc/, include/, tests/ |
| 5 | `benchmarks/bench_deepseek_mla.py` | Benchmark configurations |
| 6 | `tests/attention/test_deepseek_mla.py` | Test dimensions and usage patterns |
