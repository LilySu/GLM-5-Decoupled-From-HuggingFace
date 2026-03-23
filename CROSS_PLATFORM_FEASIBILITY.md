# GLM-5 Cross-Platform Inference Feasibility Analysis

## Executive Summary

GLM-5 has three architecturally unique components that determine portability: **MLA** (Multi-Latent Attention with KV compression), **DSA** (Dynamic Sparse Attention with lightning indexer), and **256-expert MoE** (sigmoid routing). This analysis evaluates feasibility across 5 platforms for running GLM-5 inference.

---

## Component Feasibility Matrix

### MLA (Multi-Latent Attention)

The core challenge: MLA compresses KV from 6144D to 576D (512 nope + 64 rope) using learned projections. Efficient inference requires **weight absorption** (folding `kv_b_proj` into Q and O) and kernels that handle the absorbed 576D KV format.

| Platform | MLA Kernel | Source | Status | Notes |
|----------|-----------|--------|--------|-------|
| **NVIDIA H100 (CUDA)** | FlashMLA, FlashInfer, TRT-LLM | DeepSeek, FlashInfer, NVIDIA | **Production-ready** | d_v=512 hardcoded (absorbed mode). 660 TFLOPS dense decode on H800. GLM-5 explicitly supported in FlashInfer (`num_heads=64` listed). |
| **AMD MI300X (ROCm)** | AITER `mla_decode_fwd`, `mla_prefill_fwd` | AMD (ROCm/AITER) | **Actively developed** | Native MLA with ASM + CK backends. Supports nhead=16-128 via metadata reshaping. FP8 support with block-size mapping. SGLang PR #21166 "GLM-5 performance optimization" on AMD is open (Mar 2026). Performance gap noted: SGLang issue #21071 "GLM5 FP8: AMD MI355 slower than H200." |
| **AMD MI300X (Triton-ROCm)** | ROCm FlashAttention (Triton backend) | AMD fork of FlashAttention | **Partial** | Supports MQA/GQA but NO explicit MLA support. Would need custom Triton kernel or fall back to eager attention. The Triton backend supports head_dim up to 256 and FP8 via FA3 interface. |
| **JAX/TPU** | Custom Pallas kernel or MaxText | Google | **Research-grade** | MaxText supports DeepSeek V3.1 (671B) which uses the same MLA architecture. No explicit GLM-5 support. Would need: (1) port weight absorption logic to JAX, (2) write Pallas kernel or use JAX's `dot_product_attention` with absorbed format, (3) handle 576D KV layout on TPU. Feasible but ~2-4 weeks of work. |
| **Modular MAX/Mojo** | DeepseekV3ForCausalLM arch | Modular | **Likely compatible** | MAX lists `DeepseekV3ForCausalLM` as supported. GLM-5 shares the same MLA architecture (same kv_lora_rank=512, same absorption trick). Would need: (1) new model class or config adapter, (2) verify internal MLA kernel handles `num_heads=64` (DeepSeek uses 128). No source code access to verify kernel constraints. |

### DSA (Dynamic Sparse Attention — Lightning Indexer)

The core challenge: DSA uses a 32-head lightweight scoring layer to select top-2048 KV positions per query. Requires: (1) fused MQA scoring kernel, (2) **deterministic** top-k, (3) sparse attention over selected indices.

| Platform | DSA Indexer Kernel | Sparse Attention | Status | Notes |
|----------|-------------------|------------------|--------|-------|
| **NVIDIA H100** | DeepGEMM `fp8_mqa_logits` | FlashMLA sparse | **Production-ready** | Exact match for GLM-5's scoring formula. `torch.topk` for deterministic selection. FlashMLA sparse decode: 410 TFLOPS on H800. |
| **AMD MI300X (ROCm)** | None found (custom needed) | AITER MLA supports paged attention, no sparse | **Major gap** | No equivalent to DeepGEMM's `fp8_mqa_logits` on ROCm. The indexer scoring (`q·k^T → ReLU → weighted_sum → topk`) would need: (1) custom HIP kernel or Triton-ROCm kernel, (2) PyTorch eager fallback (~10-50x slower). Sparse attention: AITER's MLA doesn't document sparse index support. SGLang's GLM-5 PR (#21166) may address this — status unknown. |
| **AMD MI300X (Triton-ROCm)** | Custom Triton kernel needed | Custom masked attention | **Feasible, ~1-2 weeks** | The indexer scoring is 3 fused ops (einsum + relu + weighted_sum). Writable as a Triton kernel. Sparse attention via masked FlashAttention. ROCm's Triton backend supports arbitrary attention masks. |
| **JAX/TPU** | `jax.lax.dot_general` + custom | JAX masked attention | **Feasible, ~2-3 weeks** | The indexer is mathematically simple. Use `jax.numpy.einsum` for scoring, `jax.lax.top_k` for selection (deterministic on TPU). Sparse attention via `jax.nn.dot_product_attention` with custom mask. No dedicated sparse kernel → quadratic memory (works for seq ≤ 32K, problematic at 200K). |
| **Modular MAX** | Not available | Not available | **Not feasible without custom Mojo kernel** | MAX's DeepSeek support may not include DSA (it's a GLM-5/DeepSeek-V3.2 addition). No sparse attention kernel documented. Would need: write DSA indexer + sparse attention as custom Mojo GPU kernels. High effort (~4-8 weeks). |

### MoE (256 Experts, Sigmoid Routing, Top-8)

The core challenge: 256 experts with SwiGLU FFN, sigmoid (not softmax) gating, `n_group=1` flat routing, scaling factor 2.5.

| Platform | Routing Kernel | Grouped GEMM | Status | Notes |
|----------|---------------|-------------|--------|-------|
| **NVIDIA H100** | PyTorch (5 lines, n_group=1) | DeepGEMM FP8 | **Production-ready** | 1550 TFLOPS on H800. SGLang's `moe_fused_gate` also works but unnecessary for n_group=1. |
| **AMD MI300X (ROCm)** | PyTorch or AITER `moe_op.py` | AITER `moe_sorting.py` + CK/Triton GEMM | **Functional, perf unclear** | AITER has MoE ops including sorting and dispatch. vLLM/SGLang on ROCm use these. SGLang PR #21097 "[AMD] Add MoE weights and scales padding" is in draft. FP8 MoE on MI300X is under active development. |
| **AMD MI300X (Triton-ROCm)** | PyTorch | Triton grouped GEMM (from Unsloth) | **Works today** | The existing `glm5-triton/unsloth_moe/` Triton kernels run on ROCm with `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`. Performance ~60-70% of CUDA optimized path. |
| **JAX/TPU** | `jax.lax.top_k` | `jax.numpy.einsum` or Pallas | **Feasible, standard pattern** | MoE is well-supported in JAX via MaxText (supports DeepSeek V3.1 MoE). Expert parallelism via `jax.experimental.shard_map`. TPU's all-to-all is efficient for MoE dispatch. Main concern: 256 experts × 8 selected = many small GEMMs. TPU megacore helps. |
| **Modular MAX** | Likely via DeepSeek support | `Qwen3MoeForCausalLM` exists | **Likely works** | MAX supports MoE via Qwen3 and DeepSeek architectures. 256 experts with sigmoid routing may need config validation. |

### Supporting Components (RMSNorm, SwiGLU, RoPE, CE Loss)

| Platform | RMSNorm | SwiGLU | RoPE (64D) | CE Loss | FP8 |
|----------|---------|--------|-----------|---------|-----|
| **NVIDIA H100** | Unsloth Triton ✅ | Unsloth Triton ✅ | PyTorch ✅ | Unsloth Triton ✅ | E4M3 native ✅ |
| **AMD MI300X** | AITER/CK ✅ | AITER ✅ | PyTorch ✅ | PyTorch ✅ | E4M3 via AITER ✅ (MI300X supports FP8) |
| **Triton-ROCm** | Unsloth Triton ✅ | Unsloth Triton ✅ | PyTorch ✅ | Unsloth Triton ✅ | Limited (Triton FP8 on ROCm is newer) |
| **JAX/TPU** | `jax.numpy` ✅ | `jax.nn.silu` ✅ | `jax` ✅ | `optax` ✅ | TPU v5e FP8 ✅, older TPU: BF16 only |
| **Modular MAX** | Built-in ✅ | Built-in ✅ | Built-in ✅ | Built-in ✅ | Unknown |

---

## Overall Feasibility Summary

| Platform | Hardware | MLA | DSA Indexer | DSA Sparse Attn | MoE 256 | Overall | Effort | Inference Speed (est.) |
|----------|----------|-----|-------------|-----------------|---------|---------|--------|----------------------|
| **NVIDIA H100 (CUDA)** | H100/H800 80GB × 8 | ✅ Production | ✅ Production | ✅ Production | ✅ Production | **✅ Ready** | Done | 1.0× (baseline) |
| **AMD MI300X (ROCm/AITER)** | MI300X 192GB × 4-8 | ⚠️ Active dev | ❌ Custom needed | ❌ Custom needed | ⚠️ Active dev | **⚠️ 2-3 months** | High | ~0.6-0.8× (est.) |
| **AMD MI300X (Triton-ROCm)** | MI300X 192GB × 4-8 | ❌ No MLA kernel | ❌ Custom needed | ⚠️ Masked FA | ✅ Unsloth works | **⚠️ 3-4 months** | High | ~0.4-0.6× (est.) |
| **JAX/TPU** | TPU v5p × 8+ | ⚠️ Custom Pallas | ⚠️ Custom JAX | ❌ No sparse (quadratic) | ✅ MaxText pattern | **⚠️ 2-3 months** | High | ~0.5-0.7× (no sparse) |
| **Modular MAX/Mojo** | H100 or MI300X | ⚠️ DeepSeek adapter | ❌ Custom Mojo | ❌ Custom Mojo | ⚠️ Likely works | **❌ 4-8 months** | Very high | Unknown |

### Legend
- ✅ Production-ready: kernel exists, tested, documented
- ⚠️ Feasible with work: kernel exists partially or pattern is known, needs adaptation
- ❌ Major gap: no kernel exists, custom implementation required

---

## Detailed Platform Analysis

### AMD MI300X via ROCm — Most Promising Alternative

**Advantages:**
- 192GB HBM3 per GPU (vs H100's 80GB) — fits more of the 744B model per device
- AITER has native MLA kernels (decode + prefill) with ASM backends
- SGLang is actively adding GLM-5 support (PR #21166, Mar 2026)
- FP8 E4M3 supported on MI300X hardware
- vLLM has ROCm backend with MLA support for DeepSeek models

**Critical gaps:**
1. **DSA Lightning Indexer**: No equivalent to DeepGEMM's `fp8_mqa_logits`. The fused `q·k^T → ReLU → weighted_sum` scoring requires a custom kernel. Options:
   - Write a Triton-ROCm kernel (~1 week for scoring, ~1 week for integration)
   - Use PyTorch eager (functional but ~10-50x slower for this op)
   - Wait for SGLang PR #21166 (may include this)

2. **Sparse attention with arbitrary top-2048 indices**: AITER's MLA supports paged attention but doesn't document token-level sparse selection (only block-level). Need to verify if `mla_decode_fwd` can accept sparse page tables like FlashMLA's `indices` parameter.

3. **Performance parity**: SGLang issue #21071 reports "GLM5 FP8: AMD MI355 slower than H200" — suggesting current MI300X performance lags H100 for this specific workload.

**Estimated effort:** 2-3 months for a functional inference path. Key work:
- Port weight absorption logic (1 week)
- Write DSA indexer Triton-ROCm kernel (2 weeks)
- Integrate AITER MLA with sparse indices (2 weeks)
- Test and tune MoE on MI300X (2 weeks)
- End-to-end integration and benchmarking (2 weeks)

### JAX/TPU — Viable for Research, Not Serving

**Advantages:**
- MaxText already supports DeepSeek V3.1 (same MLA architecture)
- TPU v5p has excellent all-to-all performance for MoE expert dispatch
- JAX's `jit` + `shard_map` handles tensor parallelism cleanly
- Deterministic execution by default (no CUDA non-determinism issues)

**Critical gaps:**
1. **No sparse attention kernel on TPU**: JAX's `dot_product_attention` doesn't support arbitrary sparse indices. Masking the full N×N matrix works but is O(N²) — at 200K context, this is 40 billion elements per attention head. TPU can't handle this.
   - Mitigation: Limit context to ≤32K on TPU, use DSA only on GPU
   - Alternative: Write a custom Pallas kernel for sparse attention (very hard, ~4-8 weeks)

2. **No FP8 on older TPUs**: TPU v4 and earlier don't support FP8. TPU v5e and v5p do. BF16 inference works but misses the 2× throughput from FP8.

3. **Weight absorption in JAX**: Need to implement the absorption logic in JAX (PyTorch tensor ops → JAX array ops). Straightforward but tedious (~1 week).

**Estimated effort:** 2-3 months for functional inference at ≤32K context. Sparse attention (DSA at 200K) is the hard part — may require 2+ additional months for a Pallas kernel.

### Modular MAX/Mojo — Longest Path

**Advantages:**
- Supports DeepSeek V3 architecture (same MLA)
- Hardware-agnostic GPU kernels (could target both NVIDIA and AMD)
- Growing model library (500+ models)

**Critical gaps:**
1. **No GLM-5 model class**: Would need to register `GlmMoeDsaForCausalLM` as a new architecture. If MAX's `DeepseekV3ForCausalLM` handles MLA and MoE, a config adapter might suffice. But DSA is unique to GLM-5/DeepSeek-V3.2 and likely not in the DeepSeek V3 implementation.

2. **DSA requires custom Mojo kernels**: The lightning indexer and sparse attention are not standard ops. Writing them in Mojo's GPU kernel language is possible but the ecosystem is young — documentation for custom attention kernels is sparse.

3. **256-expert MoE**: MAX supports Qwen3 MoE but GLM-5's sigmoid routing (vs softmax) and n_group=1 flat selection need verification.

**Estimated effort:** 4-8 months. The Mojo kernel authoring for DSA is the bottleneck.

---

## Recommendation

**For production inference today:** NVIDIA H100 with FlashMLA + DeepGEMM (the current `glm5-kernels-flashmla-deepgemm/` path).

**For AMD MI300X inference (next priority):**
1. Start with SGLang on ROCm — it's the path of least resistance (PR #21166 is already in progress)
2. If SGLang doesn't cover DSA, write a Triton-ROCm kernel for the indexer scoring
3. Use AITER's MLA kernels for attention
4. Timeline: 2-3 months to production-quality inference

**For JAX/TPU (research):**
1. Port the model to MaxText-style JAX
2. Accept the 32K context limitation (no sparse attention on TPU)
3. Use for experimentation, not serving
4. Timeline: 2-3 months for basic inference

**For Modular MAX:** Wait. The ecosystem needs to mature before custom attention kernels are practical.

---

## Sources

| Source | URL | What it confirms |
|--------|-----|-----------------|
| AITER MLA | `github.com/ROCm/aiter` | Native MLA decode/prefill on MI300X with ASM backend |
| SGLang GLM-5 AMD PR | `github.com/sgl-project/sglang/pull/21166` | Active GLM-5 optimization on AMD (Mar 2026) |
| SGLang MI355 perf issue | `github.com/sgl-project/sglang/issues/21071` | Current AMD performance lags NVIDIA for GLM-5 FP8 |
| ROCm FlashAttention | `github.com/ROCm/flash-attention` | CK + Triton backends, MQA/GQA, FP8 via FA3, MI200x-MI355x |
| MaxText DeepSeek V3.1 | `github.com/google/maxtext` | JAX supports DeepSeek MLA + MoE on TPU |
| Modular MAX models | `docs.modular.com/max/models` | DeepSeekV3ForCausalLM supported, no GLM-5 |
| GLM-5 paper §2.2 | `arxiv.org/abs/2602.15763` | DSA requires deterministic topk (§3.2) |
| FlashMLA API | `github.com/deepseek-ai/FlashMLA` | d_v=512 hardcoded, 656-byte FP8 format |
| DeepGEMM API | `github.com/deepseek-ai/DeepGEMM` | fp8_mqa_logits, runtime num_heads, JIT compilation |
