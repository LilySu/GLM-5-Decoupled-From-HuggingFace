# GLM-5 Precision Tracking: Numerical Format at Every Pipeline Stage

## Purpose

This document maps the **exact numerical precision** (data type, scaling method, accumulation strategy) at every stage of the GLM-5 inference pipeline, across all 4 implementations. This is critical for:

1. **Debugging quality regressions** — knowing where precision drops identifies where errors accumulate
2. **FP8 Pareto analysis** — understanding which stages introduce quantization error
3. **Academic evaluation** — FA3 established that FP8 papers must report quality alongside speed
4. **Hardware planning** — different precisions have different TFLOPS ceilings on H100

---

## Precision Format Reference

| Format | Bits | Exponent | Mantissa | Range | Precision | H100 Peak TFLOPS |
|--------|------|----------|----------|-------|-----------|-----------------|
| FP32 | 32 | 8 | 23 | ±3.4e38 | ~7 digits | 67 |
| TF32 | 19 | 8 | 10 | ±3.4e38 | ~3 digits | 495 |
| BF16 | 16 | 8 | 7 | ±3.4e38 | ~2 digits | 989 |
| FP16 | 16 | 5 | 10 | ±65504 | ~3 digits | 989 |
| FP8 E4M3 | 8 | 4 | 3 | ±448 | ~1.5 digits | 1979 |
| FP8 E5M2 | 8 | 5 | 2 | ±57344 | ~1 digit | 1979 |
| INT8 | 8 | — | — | [-128, 127] | exact integer | 1979 |
| MXFP4 | 4+8 | 2 | 1 | varies | ~0.5 digits | 3958 (Blackwell) |

**Key insight from DeepSeek-V3 (arXiv:2412.19437):** FP8 E4M3 with fine-grained quantization (1×128 tiles for activations, 128×128 blocks for weights) loses <0.25% accuracy vs BF16, provided FP32 accumulation is used for all GEMMs.

---

## Stage-by-Stage Precision Tables

### Table 1: GLM-5 Raw PyTorch (`glm5-raw-decoupled-from-hf/`)

All computation in BF16 with FP32 accumulation for matmuls. No quantization.

| Stage | Operation | Input Precision | Weight Precision | Compute Precision | Accumulation | Output Precision | Notes |
|-------|-----------|----------------|-----------------|-------------------|-------------|-----------------|-------|
| **Embedding** | `nn.Embedding` | INT64 (token IDs) | BF16 | — | — | BF16 | Lookup, no compute |
| **RMSNorm (pre-attn)** | `x * rsqrt(mean(x²) + ε) * w` | BF16 | BF16 | FP32 | FP32 | BF16 | Variance in FP32 for stability |
| **MLA Q compress** | `q_a_proj` (6144→2048) | BF16 | BF16 | BF16 | FP32 (torch default) | BF16 | Linear matmul |
| **MLA Q layernorm** | `q_a_layernorm` | BF16 | BF16 | FP32 | FP32 | BF16 | — |
| **MLA Q expand** | `q_b_proj` (2048→16384) | BF16 | BF16 | BF16 | FP32 | BF16 | [64 heads × 256 dim] |
| **MLA KV compress** | `kv_a_proj_with_mqa` (6144→576) | BF16 | BF16 | BF16 | FP32 | BF16 | [512 compressed + 64 RoPE key] |
| **MLA KV layernorm** | `k_a_layernorm` | BF16 | BF16 | FP32 | FP32 | BF16 | — |
| **MLA KV expand** | `kv_b_proj` (512→28672) | BF16 | BF16 | BF16 | FP32 | BF16 | [64 × (192 nope + 256 value)] |
| **RoPE** | `cos/sin rotation` | BF16 | — | BF16 | — | BF16 | 64-dim only, element-wise |
| **KV Cache Store** | Concatenate K, V | BF16 | — | — | — | **BF16** | Full precision cached |
| **DSA Indexer Score** | `einsum(q, k) * scale` | BF16 | BF16 | BF16 | FP32 | BF16 | 32 heads × 128 dim |
| **DSA ReLU** | `F.relu(scores)` | BF16 | — | BF16 | — | BF16 | Element-wise |
| **DSA Weighted Sum** | `einsum(scores, weights)` | BF16 | BF16 (weights) | BF16 | FP32 | BF16 | — |
| **DSA TopK** | `torch.topk(logits, 2048)` | BF16 | — | — | — | BF16 + INT64 | **Deterministic** (paper §3.2) |
| **Attention QK^T** | `matmul(Q, K^T) / √256` | BF16 | — | BF16 | FP32 | BF16 | 64 heads × seq × seq |
| **Attention Softmax** | `softmax(scores)` | BF16 | — | FP32 | FP32 | BF16 | **FP32 softmax** for stability |
| **Attention PV** | `matmul(P, V)` | BF16 | — | BF16 | FP32 | BF16 | — |
| **Attention O proj** | `o_proj` (16384→6144) | BF16 | BF16 | BF16 | FP32 | BF16 | — |
| **RMSNorm (pre-MoE)** | Same as pre-attn | BF16 | BF16 | FP32 | FP32 | BF16 | — |
| **MoE Router** | `F.linear(x.float(), w.float())` | **FP32** | **FP32** | FP32 | FP32 | **FP32** | **Router runs in FP32** (stability) |
| **MoE Sigmoid** | `scores.sigmoid()` | FP32 | — | FP32 | — | FP32 | — |
| **MoE TopK** | `torch.topk(scores, 8)` | FP32 | — | — | — | FP32 + INT64 | n_group=1, flat selection |
| **MoE Expert FFN** | `gate_up + silu + down` | BF16 | BF16 | BF16 | FP32 | BF16 | Per-expert loop |
| **Shared Expert FFN** | Same as above | BF16 | BF16 | BF16 | FP32 | BF16 | Always-on expert |
| **LM Head** | `lm_head` (6144→154880) | BF16 | BF16 | BF16 | FP32 | BF16 | Logits |
| **Loss** | `F.cross_entropy` | BF16 (logits) | — | FP32 | FP32 | FP32 | **FP32 loss** |

**Precision transitions:** 0 (all BF16 except router FP32 and loss FP32)
**Quantization boundaries:** 0

---

### Table 2: GLM-5 Triton (`glm5-triton/`)

Same as Raw PyTorch EXCEPT where Triton kernels replace PyTorch ops. The Triton kernels may use different internal precision.

| Stage | Change vs Raw | Precision Impact |
|-------|--------------|-----------------|
| **RMSNorm** | Unsloth Triton `fast_rms_layernorm` | **FP32 variance computation** preserved. Internal reduction in FP32. Identical output to PyTorch. |
| **SwiGLU** | Unsloth Triton `swiglu_fg_kernel` | BF16 in/out. Fused silu(gate)*up. No precision change vs sequential PyTorch. |
| **Cross-Entropy** | Unsloth Triton `fast_cross_entropy_loss` | Chunked for vocab=154880. FP32 log-sum-exp. **Numerically identical** to PyTorch F.cross_entropy. |
| **MoE GEMM** | Unsloth Triton grouped GEMM | BF16 input, BF16 weights, **FP32 accumulation** in Triton. Same precision as per-expert loop. |
| All other stages | Unchanged | — |

**Precision transitions:** 0 additional
**Quantization boundaries:** 0

---

### Table 3: GLM-5 FlashMLA + DeepGEMM (`glm5-kernels-flashmla-deepgemm/`)

**THIS IS WHERE PRECISION CHANGES HAPPEN.** Four components switch to FP8.

| Stage | Change vs Triton | Input | Weights | Compute | Accum | Output | Quantization Details |
|-------|-----------------|-------|---------|---------|-------|--------|---------------------|
| **Embedding → RMSNorm** | No change | BF16 | BF16 | FP32 | FP32 | BF16 | — |
| **MLA Q/KV projections** | No change | BF16 | BF16 | BF16 | FP32 | BF16 | — |
| **RoPE** | No change | BF16 | — | BF16 | — | BF16 | — |
| ⚡ **KV Cache Store** | **FlashMLA FP8 format** | BF16 | — | — | — | **FP8 E4M3 + BF16** | **656 bytes/token**: 512B FP8 nope (4 tiles × 128, per-tile power-of-2 scale) + 16B FP32 scales + 128B BF16 rope (unquantized) |
| ⚡ **DSA Indexer Score** | **DeepGEMM `fp8_mqa_logits`** | **FP8 E4M3** (Q) | **FP8 E4M3** (K) | FP8 | **FP32** | BF16 | Per-token 1×128 quantization for Q. Per-token scaling for K. Fused: `ReLU(q·k^T) * weights`. |
| **DSA TopK** | No change | BF16 | — | — | — | BF16 + INT64 | Still `torch.topk` (deterministic) |
| ⚡ **MLA Attention** | **FlashMLA kernel** | BF16 (Q absorbed) | **FP8 E4M3** (KV cache) | BF16 | **FP32** (online softmax) | BF16 | FlashMLA reads FP8 KV, dequantizes on-the-fly. Softmax in FP32. d_v=512 (absorbed). |
| **Attention O proj** | No change | BF16 | BF16 | BF16 | FP32 | BF16 | — |
| **MoE Router** | No change | FP32 | FP32 | FP32 | FP32 | FP32 | — |
| ⚡ **MoE Expert GEMM** | **DeepGEMM FP8 grouped** | **FP8 E4M3** | **FP8 E4M3** | FP8 | **FP32** | BF16 | `m_grouped_fp8_gemm_nt_contiguous`. Activations: 1×128 tile quant. Weights: 128×128 block quant. |
| **LM Head** | No change | BF16 | BF16 | BF16 | FP32 | BF16 | — |

**Precision transitions per layer:** 4 (BF16→FP8 at KV store, indexer, attention, MoE; FP8→BF16 after each)
**Quantization boundaries per layer:** 4 (quantize before, dequantize after each FP8 stage)
**Total boundaries per forward pass:** 4 × 78 layers = **312 quantization boundary crossings**

**Quantization formats used:**

| Component | Format | Block Size | Scale Type | Bytes/Token/Layer |
|-----------|--------|-----------|-----------|------------------|
| KV Cache (nope) | FP8 E4M3 | 128-dim tiles | Power-of-2 (FP32) | 512 + 16 scales |
| KV Cache (rope) | **BF16** (unquantized) | — | — | 128 |
| DSA Q input | FP8 E4M3 | 1×128 per-token | FP32 per-block | — (transient) |
| DSA K input | FP8 E4M3 | 1×128 per-token | FP32 per-block | — (transient) |
| MoE activations | FP8 E4M3 | 1×128 per-token | FP32 per-block | — (transient) |
| MoE weights | FP8 E4M3 | 128×128 blocks | FP32 per-block | Persistent |

---

### Table 4: GLM-5 FlashInfer (`glm5-kernels-flashinfer/`)

Similar to FlashMLA+DeepGEMM but with DIFFERENT KV cache format.

| Stage | Change vs FlashMLA | Precision Impact |
|-------|-------------------|-----------------|
| ⚡ **KV Cache Store** | **FlashInfer FP8 format** | **576 bytes/token**: all FP8 E4M3 contiguous [512 ckv + 64 kpe]. External scale tensors (not inline). **RoPE dims ARE quantized to FP8** (unlike FlashMLA which keeps BF16). |
| ⚡ **MLA Attention** | **FlashInfer FA3 backend** | BF16 Q, FP8 KV cache, BF16 output. FA3 uses Hopper warp specialization. bmm1_scale and bmm2_scale passed as external floats. |
| ⚡ **MLA Sparse Decode** | **TRT-LLM gen backend** | Same FP8 KV. `sparse_mla_top_k=2048`. Requires patched `qk_nope_head_dim` validation. |
| All other stages | Same as FlashMLA+DeepGEMM | — |

**Key precision difference:** FlashInfer quantizes RoPE dimensions to FP8 (saves 80 bytes/token/layer but loses positional precision). FlashMLA keeps RoPE in BF16 (more precise, 80 bytes larger).

---

## Precision Comparison Summary

| Property | Raw PyTorch | Triton | FlashMLA+DeepGEMM | FlashInfer |
|----------|------------|--------|-------------------|-----------|
| **Compute dtype** | BF16 | BF16 | BF16 + **FP8** | BF16 + **FP8** |
| **KV cache dtype** | BF16 | BF16 | **FP8 (nope) + BF16 (rope)** | **FP8 (all)** |
| **KV bytes/token/layer** | 1,152 | 1,152 | **656** | **576** |
| **MoE GEMM dtype** | BF16 | BF16 (Triton) | **FP8** (DeepGEMM) | **FP8** (DeepGEMM) |
| **DSA indexer dtype** | BF16 | BF16 | **FP8** (DeepGEMM) | **FP8** (DeepGEMM) |
| **Router dtype** | FP32 | FP32 | FP32 | FP32 |
| **Softmax dtype** | FP32 | FP32 | FP32 | FP32 |
| **Accumulation dtype** | FP32 | FP32 | FP32 | FP32 |
| **RoPE precision in cache** | BF16 | BF16 | **BF16** | **FP8** ⚠️ |
| **Quant boundaries/layer** | 0 | 0 | **4** | **4** |
| **Total quant crossings** | 0 | 0 | **312** | **312** |
| **Expected cos_sim (78 layers)** | 1.000 | 1.000 | >0.95 | >0.93 (RoPE FP8 hurts) |

---

## Critical Precision Observations

### 1. Router MUST Stay FP32
The GLM-5 codebase (`model.py` line 371-373) explicitly casts router inputs and weights to FP32:
```python
def forward(self, x):
    x = x.view(-1, self.hidden_size)
    return F.linear(x.float(), self.weight.float())
```
This matches DeepSeek-V3's finding that MoE gating modules need BF16/FP32 precision for stable expert selection. Quantizing the router to FP8 causes load imbalance.

### 2. Softmax MUST Stay FP32
All implementations compute softmax in FP32 (or with FP32 accumulation for the online softmax in FlashMLA/FlashInfer). BF16 softmax causes catastrophic precision loss when attention logits have large magnitude differences.

### 3. FlashMLA's BF16 RoPE vs FlashInfer's FP8 RoPE
FlashMLA stores the 64-dim RoPE key portion in BF16 (128 bytes), while FlashInfer quantizes it to FP8 (64 bytes). This 80-byte difference per token per layer compounds:
- At 200K context, 78 layers: FlashMLA uses 1.17 GB more for RoPE
- But FlashMLA preserves positional encoding precision → better long-context quality
- **Recommendation:** Measure attention output cosine similarity at 64K+ contexts to quantify the impact

### 4. The 312 Quantization Boundary Problem
Each BF16→FP8→BF16 roundtrip introduces ~3-5% relative error per element. With 4 boundaries per layer and 78 layers:
- Worst case (if errors compound): 78 × 4 × 3% = 936% cumulative (obviously wrong — errors partially cancel)
- Realistic (errors partially cancel): cosine similarity drops from 1.0 to ~0.95 over 78 layers
- **Must benchmark:** Chain 78 quantize-dequantize roundtrips on a fixed tensor and measure actual drift

### 5. DeepSeek-V3's Precision Lesson
From the DeepSeek-V3 technical report: "the relative accuracy loss compared to BF16 remains below 0.25%" — but ONLY because they use:
- Fine-grained 1×128 tile quantization (not per-tensor)
- FP32 accumulation for ALL GEMMs
- High-precision components: embedding, output head, MoE gating, normalization, attention operators
GLM-5's kernel implementations follow the same pattern.

---

## Recommended Precision Logging for Benchmarks

Every benchmark run should log:

```python
precision_log = {
    "stage": "mla_attention",
    "input_dtype": "bf16",
    "kv_cache_dtype": "fp8_e4m3",
    "kv_scale_type": "per_tile_power_of_2",  # FlashMLA
    "compute_dtype": "bf16",
    "accumulation_dtype": "fp32",
    "output_dtype": "bf16",
    # Quality metrics
    "output_cos_sim_vs_bf16_ref": 0.9987,
    "output_rmse_vs_bf16_ref": 0.0023,
    "max_abs_error": 0.15,
    # Quantization details
    "quant_block_size": 128,
    "quant_format": "e4m3",
    "num_quant_boundaries": 2,  # one quantize, one dequantize
}
```

Sources:
- [DeepSeek-V3 FP8 Training](https://arxiv.org/abs/2412.19437)
- [FlashAttention-3 FP8 Evaluation](https://arxiv.org/abs/2407.08608)
- [Unified FP8 for MoE RL](https://lmsys.org/blog/2025-11-25-fp8-rl/)
- [InfiR2: FP8 Training Recipe](https://arxiv.org/abs/2509.22536)
- [Microscaling Formats (OCP MX Spec)](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)
- [MicroMix: Mixed-Precision MX Quantization](https://arxiv.org/abs/2508.02343)
- [MXFP4 on AMD GPUs](https://rocm.blogs.amd.com/software-tools-optimization/mxfp4-mxfp6-quantization/README.html)
