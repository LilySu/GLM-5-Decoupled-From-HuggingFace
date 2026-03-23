# FlashInfer Uncertainties & Open Questions for GLM-5

## Previously Resolved

### CUTLASS Backend Incompatibility
- **Issue:** `_check_cutlass_shape()` hardcodes `H=128, D=576`. GLM-5 has H=64.
- **Resolution:** CUTLASS backend cannot be used. TRT-LLM gen and FA2/FA3 backends work.
- **Source:** `flashinfer/mla.py` lines 42-55

### GLM-5 Explicit Support
- **Issue:** Does FlashInfer know about GLM-5 dimensions?
- **Resolution:** Yes. `supported_mla_layer_dimensions` includes `(deepseek_mla_dimensions, num_heads=64)` with comment `# GLM-5 dimensions`.
- **Source:** `flashinfer/mla.py` lines 100-110

### Output Dimension (#6)
- **Issue:** Does FlashInfer handle the W_O projection, or is it external?
- **Resolution:** External. `wrapper.run()` returns `[batch, num_heads, head_dim_ckv]` = `[batch, 64, 512]`. The absorbed output projection `W_O` mapping `[64, 512] → [6144]` is a separate linear layer, identical to FlashMLA.
- **Source:** `flashinfer/mla.py` line 484: `out = torch.empty_like(q_nope)` — output shape matches q_nope shape `[B, H, 512]`.

---

## Newly Resolved (from source code analysis, 2026-03-22)

### 1. TRT-LLM gen Backend Performance vs FlashMLA — PARTIALLY RESOLVED

**Source:** `flashinfer/mla.py` lines 584-693, `trtllm_batch_decode_with_kv_cache_mla()`

**Findings:**

1. **Backend auto-selection on H100 (SM90):** The auto backend selection is:
   ```python
   backend = "trtllm-gen" if get_compute_capability(query.device)[0] == 10 else "xqa"
   ```
   SM90 has compute capability 9.0, so `[0] = 9 ≠ 10` → auto selects **"xqa"**. But XQA requires SM120 and FP8-only (lines 595-601). This means **on H100, you MUST explicitly set `backend="trtllm-gen"`** for the decode path. Auto-detection effectively doesn't work for H100 with this function.

2. **Sparse MLA support:** `sparse_mla_top_k` parameter is fully supported by the trtllm-gen backend (line 685). When `sparse_mla_top_k > 0`, `block_tables` is reinterpreted as `(B, Q_len, sparse_mla_top_k)` indices (line 104-108).

3. **Performance comparison:** No published TFLOPS numbers found for FlashInfer's TRT-LLM gen path with sparse MLA on H800/H100. FlashMLA reports 410 TFLOPS sparse decode on H800.

4. **NEW CRITICAL FINDING — `qk_nope_head_dim=128` hardcheck:**
   ```python
   # _check_trtllm_gen_mla_shape(), line 86-87:
   if qk_nope_head_dim != 128:
       raise ValueError(f"Expected qk_nope_head_dim == 128, got {qk_nope_head_dim}")
   ```
   GLM-5 has `qk_nope_head_dim=192`. This validation WILL FAIL. However, this is a **validation-only check** — the actual kernel infers dimensions from tensor shapes (D_q=576 and D_ckv=576 are what matters). The `qk_nope_head_dim` value is NOT passed to the underlying CUDA kernel (line 667 `run_func()` call doesn't include it).

   **Fix:** Either:
   - (a) Monkey-patch `_check_trtllm_gen_mla_shape` to accept `qk_nope_head_dim=192`
   - (b) Pass `qk_nope_head_dim=128` to satisfy validation (misleading but harmless — it's unused by the kernel)
   - (c) Call the kernel directly bypassing the validation wrapper

   **Recommended:** Option (a) — clearest intent. Add to `glm5-kernels-flashinfer/patches.py`:
   ```python
   import flashinfer.mla as _mla
   _orig_check = _mla._check_trtllm_gen_mla_shape
   def _glm5_check_trtllm_gen_mla_shape(query, kv_cache, qk_nope_head_dim, kv_lora_rank, qk_rope_head_dim, sparse_mla_top_k, page_table, page_size):
       # GLM-5 has qk_nope_head_dim=192, not 128. The kernel doesn't use this value.
       return _orig_check(query, kv_cache, 128, kv_lora_rank, qk_rope_head_dim, sparse_mla_top_k, page_table, page_size)
   _mla._check_trtllm_gen_mla_shape = _glm5_check_trtllm_gen_mla_shape
   ```

**Decision:** Use `trtllm_batch_decode_with_kv_cache_mla(backend="trtllm-gen")` for sparse MLA decode on H100, with the `qk_nope_head_dim` validation patched. Benchmark against FlashMLA sparse decode to compare.

**Risk level:** Medium. Kernel should work (D_q=576 is identical), but the patched validation means we're using the kernel outside its tested configuration.

---

### 2. FA3 Backend Status — RESOLVED, works with absorbed MLA

**Source:** `flashinfer/mla.py` lines 274-277, 329-338

**Findings:**

1. **FA3 is a first-class backend** for `BatchMLAPagedAttentionWrapper`:
   ```python
   if backend == "auto":
       self._backend = determine_mla_backend(self.device)
   else:
       self._backend = backend  # accepts "fa2", "fa3", "cutlass"
   ```

2. **FA3 supports absorbed MLA natively.** The `plan()` function passes `head_dim_ckv` (512) and `head_dim_kpe` (64) to the JIT module:
   ```python
   self._cached_module = get_batch_mla_module(
       self._backend,  # "fa3"
       q_data_type, kv_data_type, q_data_type, qo_indptr.dtype,
       head_dim_ckv,  # 512
       head_dim_kpe,  # 64
       use_profiler,
   )
   ```
   The JIT generates a FA3 kernel specialized for these dimensions. No dimension hardcoding.

3. **FA3 does NOT support sparse MLA** through this wrapper. The `BatchMLAPagedAttentionWrapper.run()` does not have a `sparse_mla_top_k` parameter. For sparse MLA, use `trtllm_batch_decode_with_kv_cache_mla()`.

4. **FA3 on H100:** FA3 uses Hopper warp specialization (producer-consumer pattern). It should outperform FA2 on H100 due to better SM utilization and TMA support.

**Decision:**
- **Dense MLA (prefill + non-sparse decode):** Use `BatchMLAPagedAttentionWrapper(backend="fa3")` — best performance on H100 via warp specialization
- **Sparse MLA (DSA decode):** Use `trtllm_batch_decode_with_kv_cache_mla(backend="trtllm-gen")` — only path that supports `sparse_mla_top_k`

**Recommended two-kernel architecture:**
```python
# Dense prefill — FA3 backend
self.dense_wrapper = BatchMLAPagedAttentionWrapper(workspace, backend="fa3")

# Sparse decode — TRT-LLM gen backend (with patched validation)
output = trtllm_batch_decode_with_kv_cache_mla(
    query, kv_cache, workspace,
    qk_nope_head_dim=192,  # patched to bypass 128 check
    kv_lora_rank=512, qk_rope_head_dim=64,
    sparse_mla_top_k=2048,  # DSA top-k
    block_tables=dsa_indices,  # [B, 1, 2048] from DSA indexer
    backend="trtllm-gen",
)
```

**Risk level:** Low. FA3 is well-tested for MLA via the FA2 codepath with Hopper extensions.

---

### 3. CUDA Graph Compatibility with Sparse MLA — RESOLVED, compatible with fixed batch size

**Source:** `flashinfer/mla.py` lines 103-108, 573-574

**Findings:**

1. **Sparse MLA index shape is fixed:** When `sparse_mla_top_k > 0`, the block_tables shape is `(B, Q_len, sparse_mla_top_k)` = `(B, 1, 2048)` for decode. This shape is **constant** across decode steps — only the VALUES change, not the shape.

2. **CUDA graphs capture kernel launch parameters (shapes, strides, pointers), not tensor values.** Since the sparse index tensor has a fixed shape, the same CUDA graph can be replayed with different index values each step.

3. **The trtllm-gen docstring confirms graph support:**
   ```python
   # line 573-574:
   # The two scale factors should be static constant for cuda graph capture.
   ```
   This implies the function is designed for CUDA graph compatibility.

4. **BatchMLAPagedAttentionWrapper also supports CUDA graphs** via `use_cuda_graph=True`, but only for DENSE MLA (no sparse_mla_top_k).

5. **Constraint:** Batch size must be fixed for CUDA graph replay. The `kv_indices_buf` is pre-allocated to a maximum size.

**Decision:** CUDA graphs ARE compatible with sparse MLA on the trtllm-gen backend. Requirements:
- Fixed batch size (standard CUDA graph constraint)
- Fixed `sparse_mla_top_k` (2048, always constant)
- `bmm1_scale` and `bmm2_scale` must be static floats (not dynamic tensors) for graph capture
- Pre-allocate all buffers (workspace, block_tables, output)

**Risk level:** Low. The design explicitly accounts for graph capture.

---

### 4. FP8 KV Cache in FlashInfer — RESOLVED, different format from FlashMLA

**Source:** `flashinfer/mla.py` lines 79-84, 543-546, 588-591

**Findings:**

1. **FlashInfer FP8 KV cache format:**
   ```
   kv_cache: [num_pages, page_size, head_dim_ckv + head_dim_kpe]
           = [num_pages, 64, 512 + 64]
           = [num_pages, 64, 576]   (dtype: float8_e4m3fn)
   ```
   ckv and kpe are stored contiguously per page slot. Scale factors are passed as **external** parameters (`bmm1_scale`, `bmm2_scale`), either as static floats or dynamic `torch.Tensor`.

2. **FlashMLA FP8 KV cache format (for comparison):**
   ```
   Per token: 656 bytes total
     512 bytes: quantized NoPE (512 × float8_e4m3)
      16 bytes: 4 × float32 scale factors (inline, 1 per 128 values)
     128 bytes: unquantized RoPE (64 × bfloat16, NOT quantized)
   ```

3. **Key differences:**

   | Aspect | FlashInfer | FlashMLA |
   |--------|-----------|----------|
   | Layout | Contiguous `[ckv \| kpe]` all FP8 | Interleaved NoPE(FP8) + scales(FP32) + RoPE(BF16) |
   | Scale factors | External tensors | Inline (embedded in cache) |
   | RoPE data | FP8 quantized | BF16 unquantized |
   | Bytes/token | 576 | 656 |
   | Per-group scaling | Via external tensor | 4 groups of 128 values each |

4. **FlashInfer is more memory-efficient** (576 vs 656 bytes/token) but quantizes the RoPE dimensions, which may slightly reduce attention precision for the positional component.

**Decision:** Use FlashInfer's simpler FP8 format. The external scale factors are easier to manage and the 12% memory savings (576 vs 656 bytes/token) compound over long sequences. If precision issues arise from quantized RoPE, fall back to BF16 KV cache.

**Risk level:** Low for ckv (512 dims). Low-medium for kpe (64 dims) — quantizing 64-dim RoPE keys to FP8 may introduce small positional errors. Monitor attention quality.

---

### 5. Page Size Constraints — RESOLVED

**Source:** `flashinfer/mla.py` lines 55-63, 634-636, and wrapper code

**Findings by backend:**

| Backend | Page Size Constraint | GLM-5 Compatible |
|---------|---------------------|-------------------|
| CUTLASS | `block_num % (128 / block_size) == 0`, H=128 required | **NO** (GLM-5 has H=64) |
| TRT-LLM gen | `block_size ∈ {32, 64}` | **YES** with block_size=64 |
| FA2 | Flexible (passed as parameter to `plan()`) | **YES** |
| FA3 | Flexible (passed as parameter to `plan()`) | **YES** |
| XQA | SM120 + FP8 only | **NO** (H100 is SM90) |

**Decision:** Use `page_size=64` for consistency across backends:
- FA3 dense MLA: `plan(..., page_size=64)`
- TRT-LLM gen sparse MLA: block_size=64 (validated at line 635)
- FlashMLA (if used as fallback): PAGE_BLOCK_SIZE=64 (hardcoded)

**Risk level:** None. page_size=64 is universally supported.

---

## NEW FINDING: `qk_nope_head_dim=128` Hardcheck in TRT-LLM gen

**This was NOT in the original uncertainties list but is CRITICAL.**

The `_check_trtllm_gen_mla_shape()` function at line 86-87 hardcodes:
```python
if qk_nope_head_dim != 128:
    raise ValueError(f"Expected qk_nope_head_dim == 128, got {qk_nope_head_dim}")
```

GLM-5 has `qk_nope_head_dim=192`. This validation was written for DeepSeek-V3 (`qk_nope_head_dim=128`) and not updated for GLM-5.

**Why it still works:** The kernel itself uses D_q=576 (the absorbed dimension), not `qk_nope_head_dim`. The raw nope dimension is only used during the Q absorption step (which happens BEFORE the kernel call). The validation is overly restrictive.

**Evidence the check is cosmetic:** Line 95-96 shows the developers already questioned whether the head count check is DeepSeek-only:
```python
# if H != 128:
#     raise ValueError(f"Expected 128 heads for query, got {H}")
# todo(Yingyi): should we check num_heads == 128? Is this deepseek only?
```
The `num_heads` check was commented out but the `qk_nope_head_dim` check was not.

**Recommended fix:** File an issue on flashinfer GitHub and use the monkey-patch described in #1 above until it's fixed upstream.

---

## Summary: Recommended FlashInfer Architecture for GLM-5 on H100

```
Dense Prefill:   BatchMLAPagedAttentionWrapper(backend="fa3")
                 → FA3 warp-specialized MLA kernel
                 → page_size=64, head_dim_ckv=512, head_dim_kpe=64

Sparse Decode:   trtllm_batch_decode_with_kv_cache_mla(backend="trtllm-gen")
                 → TRT-LLM gen fused sparse MLA kernel
                 → sparse_mla_top_k=2048, page_size=64
                 → REQUIRES qk_nope_head_dim validation patch

DSA Indexer:     DeepGEMM fp8_mqa_logits (num_heads=32)
                 + torch.topk (deterministic)

MoE GEMM:       DeepGEMM m_grouped_fp8_gemm_nt_contiguous

KV Cache:        FP8 format [num_pages, 64, 576] (FlashInfer native)
                 OR BF16 [num_pages, 64, 576] (safer for RoPE precision)

CUDA Graphs:     Compatible for decode with fixed batch size
```
