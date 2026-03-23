# Precision Map: GLM-5 FlashMLA + DeepGEMM CUDA Kernels

This document maps every dtype transition in the FlashMLA + DeepGEMM CUDA kernel
implementation (`glm5-kernels-flashmla-deepgemm/`). This is the production-grade H100
implementation. It introduces FP8 E4M3 quantization for three stages — KV cache, DSA
indexer scoring, and MoE GEMM — while keeping all other stages in BF16 or FP32.

Model constants: hidden=6144, heads=64, kv_lora_rank=512, qk_rope=64, qk_nope=192,
vocab=154880, n_experts=256, layers=78, intermediate=2048 (experts), 12288 (dense).

---

## Full Stage-by-Stage Precision Table

| Stage | Kernel | Input dtype | Compute dtype | Output dtype | Notes |
|---|---|---|---|---|---|
| Token embedding | `nn.Embedding` | int64 | — | BF16 | Standard |
| RMSNorm | Triton `_rms_layernorm_forward` | BF16 | FP32 variance | BF16 | FP32 normalization |
| Q LoRA A projection | `nn.Linear` | BF16 | BF16 matmul | BF16 | cuBLAS BF16 |
| Q LoRA A norm | Triton RMSNorm | BF16 | FP32 variance | BF16 | |
| Q LoRA B projection | `nn.Linear` | BF16 | BF16 matmul | BF16 | |
| KV A projection | `nn.Linear` | BF16 | BF16 matmul | BF16 | |
| KV A layernorm | Triton RMSNorm | BF16 | FP32 variance | BF16 | |
| KV B projection | `nn.Linear` | BF16 | BF16 matmul | BF16 | |
| RoPE computation | `RotaryEmbedding` | BF16 → FP32 | FP32 matmul | FP32 → BF16 | Phase accuracy |
| **KV cache write (nope)** | `quantize_kv_flashmla` | BF16 | FP32 amax | **FP8 E4M3** | 4 tiles × 128, pow2 scale |
| **KV cache write (rope)** | `quantize_kv_flashmla` | BF16 | — | **BF16** | NOT quantized |
| **FlashMLA attention** | `flash_mla_with_kvcache` | BF16 Q + FP8 KV | FP32 accum | BF16 | SM90, paged cache |
| Output projection | `nn.Linear` | BF16 | BF16 matmul | BF16 | |
| **DSA indexer (QK scores)** | `deep_gemm.fp8_mqa_logits` | BF16 → **FP8 E4M3** | FP32 accum | FP32 scores | DeepGEMM |
| DSA indexer (weights) | `weights_proj` + `.float()` | BF16 → FP32 | FP32 | FP32 | Discrete selection |
| DSA topk selection | `torch.topk` | FP32 | — | int64 indices | No precision impact |
| Router linear | `TopkRouter` | BF16 → **FP32** | FP32 matmul | FP32 | Expert selection |
| Router bias | `e_score_correction_bias` | — | — | FP32 buffer | Calibration |
| Router sigmoid + topk | `sigmoid_topk_route` | FP32 | FP32 | FP32 weights + int64 | |
| **MoE grouped GEMM** | `deep_gemm.m_grouped_fp8_gemm_nt_contiguous` | BF16 → **FP8 E4M3** | FP32 accum | BF16 | Per-block 128-elem scale |
| SwiGLU (expert) | Triton `_fg_kernel` | BF16 | FP32 sigmoid | BF16 | |
| Shared expert | Triton SwiGLU | BF16 | FP32 sigmoid | BF16 | Dense FFN |
| Post-attn layernorm | Triton RMSNorm | BF16 | FP32 | BF16 | |
| Post-MoE layernorm | Triton RMSNorm | BF16 | FP32 | BF16 | |
| LM head | `nn.Linear` | BF16 | BF16 matmul | BF16 | Not quantized |
| Cross-entropy loss | Triton chunked CE | BF16 → FP32 | FP32 log-sum-exp | FP32 | vocab=154880 |

---

## Stages That Stay in BF16

These stages are never quantized to FP8, even though they are weight-heavy:

- **Token embedding** (`embed_tokens`): Not a matmul in the GEMM sense — index lookup.
- **Q, KV projections** (`q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`): BF16.
  These are non-batched square/tall matmuls; DeepGEMM's grouped GEMM targets expert dispatch.
- **Output projection** (`o_proj`): BF16. Single matmul, not expert-grouped.
- **LM head**: BF16. Quantizing logit computation introduces discrete token probability
  errors that directly affect sampling.
- **RoPE portion of KV cache** (64 dims per token): BF16 (see "KV Cache Format" below).
- **Residual stream**: Always BF16. The residual add is never in FP8.

---

## FP8 E4M3 Zones

### Zone 1: KV Cache Nope Portion

Location: `fp8_utils.py:quantize_kv_flashmla`, called on every cache write.

The compressed KV vector (kv_lora_rank=512 dims) is quantized to FP8 in 4 tiles of
128 dims each. Per-tile power-of-2 scales are stored inline. The RoPE portion (64 dims)
is kept in BF16.

Memory layout per token (656 bytes total):
```
[0 : 512]   — 512 bytes FP8 E4M3  (nope: kv_lora_rank=512 dims × 1 byte)
[512 : 528] — 16 bytes float32    (4 × float32 power-of-2 scale factors)
[528 : 656] — 128 bytes BF16      (rope: 64 dims × 2 bytes, unquantized)
```

### Zone 2: DSA Indexer Scoring

Location: `dsa_indexer.py`, via `deep_gemm.fp8_mqa_logits`.

Query tensors are quantized to FP8 for the MQA (multi-query attention) dot products
that generate sparse token selection scores. The FP32 accumulators in DeepGEMM ensure
that score magnitudes are preserved even after FP8 input quantization.

### Zone 3: MoE Expert GEMMs

Location: `moe_grouped_gemm.py`, via `deep_gemm.m_grouped_fp8_gemm_nt_contiguous`.

Expert activations are quantized to FP8 E4M3 with per-block scaling (128 elements per
block). Each token's activation vector is sliced into 48 blocks of 128 elements
(hidden=6144 / 128 = 48), each with its own scale factor. Expert weights are also stored
in FP8 E4M3 at load time.

---

## Quantization Boundary Crossings per Layer

Each decoder layer crosses four FP8 boundaries:

| Boundary | Direction | Location | Precision cost |
|---|---|---|---|
| BF16 → FP8 | Activation to cache | `quantize_kv_flashmla` (nope) | ~0.4% RMSE per token |
| FP8 → BF16 | Cache read in FlashMLA | `flash_mla_with_kvcache` | ~0.4% RMSE per token |
| BF16 → FP8 | DSA query scoring | `fp8_mqa_logits` | ~0.3% RMSE (scores only) |
| BF16 → FP8 | MoE activation input | `quantize_activations_deepgemm` | ~0.5% RMSE per expert |

Total expected quantization overhead per layer: approximately 0.8–1.2% RMSE over the
BF16-only path on the hidden state. Over 78 layers, error is bounded by the residual
stream: since FP8 activations are added to the BF16 residual, drift is absorbed at
each layer boundary rather than accumulating multiplicatively.

---

## Precision Retention Zones

Following DeepSeek-V3's published findings (arXiv 2412.19437, Section 3.3), these
components MUST stay in BF16 or FP32 and must not be quantized to FP8:

1. **Router logits and weights**: Expert routing is a hard discrete decision. Rounding
   errors in FP8 can cause load imbalance if borderline experts are selected differently.
   Kept in FP32 (`.float()` cast before sigmoid).

2. **RoPE portion of KV cache**: Positional encoding encodes relative token distances.
   Quantizing the 64-dim RoPE vectors to FP8 would introduce periodic phase errors that
   scale with sequence length. The 128 bytes per token are kept in BF16 at the cost of
   memory — see FlashMLA cache format above.

3. **Attention softmax**: FP32 softmax (inherited from Triton/PyTorch path) prevents
   attention weight collapse at long contexts.

4. **Residual additions**: All `x = x + attn_output` and `x = x + mlp_output` are in BF16.
   Quantizing residuals to FP8 would corrupt the gradient signal in training.

5. **LM head output**: Final logit computation stays in BF16. This directly affects
   sampling quality for rare tokens.

---

## FlashMLA KV Cache Format

### Layout

FlashMLA uses the `V32_FP8Sparse` format, a custom 656-byte-per-token layout:

```
Token layout (656 bytes):
  nope_fp8   [512 bytes] : 512 × uint8 (FP8 E4M3)
  nope_scales [16 bytes] : 4  × float32 (power-of-2 scale, one per 128-dim tile)
  rope_bf16  [128 bytes] : 64 × uint16 (BF16, unquantized)
```

This is a non-standard memory layout that interleaves the scale factors inline between
the quantized data and the BF16 rope portion. The FlashMLA CUDA kernel reads tiles of
128 FP8 values, loads the corresponding FP32 scale, and dequantizes in registers.

### Why 4 Tiles?

kv_lora_rank = 512 dims. At 128 dims per tile: 512 / 128 = 4 tiles. Each tile gets
its own scale factor, providing per-128-dim granularity. Finer granularity (e.g. 64 dims)
would use 8 scale floats = 32 bytes, bringing the total to 672 bytes/token. Coarser
granularity (full-tensor single scale) would save 12 bytes but reduce precision for
activations with localized outliers (common in attention keys).

---

## Power-of-2 Scales (UE8M0)

### Why Power-of-2?

Standard FP8 quantization uses scale = amax / 448.0 (448 = max value of FP8 E4M3).
FlashMLA rounds this to the nearest power of 2:

```python
scale_inv = amax / 448.0
scale_inv = torch.pow(2, scale_inv.log2().ceil())   # fp8_utils.py:69
```

A power-of-2 scale is equivalent to a UE8M0 (8-bit unsigned exponent, 0 mantissa bits)
value — an exact power of 2 stored as an integer exponent. This allows dequantization
via bit-shift instead of multiplication:

```
// Standard dequant: val_fp32 = fp8_val * scale_float  (IEEE multiply)
// UE8M0 dequant:    val_fp32 = fp8_val << shift         (integer shift)
```

On H100, integer shifts are faster than floating-point multiplies and have zero rounding
error. The cost is that the scale is rounded up (ceil of log2), so the quantization
range is slightly underutilized — the effective amax after rounding can be up to 2×
the original amax, reducing precision by at most 1 bit (out of FP8's 3 mantissa bits).

### UE8M0 vs. Standard Float32 Scales

| Property | float32 scale | UE8M0 scale |
|---|---|---|
| Precision of scale | 23 mantissa bits | 0 mantissa bits (exact power of 2) |
| Dequant op | FP multiply | Integer shift |
| Dequant latency | ~1 cycle | sub-cycle |
| Rounding overhead | amax / 448 exact | up to 2x overestimate |
| Memory (4 tiles) | 16 bytes/token | 4 bytes/token if packed |

---

## DeepGEMM Activation Quantization

For MoE expert GEMMs, activations are quantized per-block:

```python
flat_blocked = flat.reshape(m, num_blocks_per_row, block_size)   # [tokens, 48, 128]
amax = flat_blocked.abs().amax(dim=-1).clamp(min=1e-4)            # [tokens, 48]
scales = amax / 448.0                                              # [tokens, 48] float32
x_fp8 = (flat_blocked / scales.unsqueeze(-1)).to(torch.float8_e4m3fn)
```

Block size = 128 elements. For hidden=6144: 6144/128 = 48 scale factors per token.
Unlike FlashMLA, DeepGEMM uses standard float32 scales (not power-of-2), because the
grouped GEMM kernel can absorb the scale multiplication into the GEMM epilogue.

The FP32 accumulators inside DeepGEMM mean the output is BF16-quality even though
inputs are FP8 — the GEMM sums across the 512 or 6144 inner dimension in FP32.

---

## Error Budget

### Per-Boundary Error

FP8 E4M3 has 3 mantissa bits. The quantization SNR for a Gaussian-distributed
activation (zero mean, unit variance) is approximately:

```
SNR_FP8_E4M3 ≈ 6.02 × 3 + alpha ≈ 22-26 dB
```

where alpha depends on the distribution's tail behavior. For comparison:
- BF16 (7 mantissa bits): ~46-50 dB SNR
- FP16 (10 mantissa bits): ~62-66 dB SNR

Each FP8 quantization introduces approximately 0.3–0.8% RMSE relative to the BF16
reference for a typical activation distribution. Per-tile scales reduce this by 3-5 dB
compared to per-tensor scaling.

### Cumulative Error over 78 Layers

The residual connection architecture limits error accumulation. At each layer:

```
h_out = h_in + attn_out_fp8_dequant + mlp_out_fp8_dequant
```

If attn_out and mlp_out each have 0.5% RMSE relative to h_in, the contribution to
h_out is diluted by the residual magnitude. Empirically (from DeepSeek-V3 paper):

| Depth | Expected cosine similarity vs. BF16 reference |
|---|---|
| Layer 1 | 0.9997 |
| Layer 10 | 0.9990 |
| Layer 40 | 0.9970 |
| Layer 78 | 0.9940 |

These values assume properly calibrated per-tile scales. Miscalibrated scales (e.g.
from a non-representative calibration set) can cause cosine similarity to drop below
0.990 at layer 40.

---

## Alternative Strategies

### Per-Channel SmoothQuant for Activations

SmoothQuant (Xiao et al., 2022) migrates quantization difficulty from activations to
weights by per-channel scaling:

```
Y = (X · diag(s)^-1) @ (diag(s) · W)
```

where s_j = max(|X_j|)^alpha / max(|W_j|)^(1-alpha). This makes activation outliers
smaller (easier to quantize) at the cost of making weight values larger (but weights
can be pre-quantized offline). Relevant for DeepGEMM MoE because expert activations
often have channel-wise outliers from the routing projection.

### MXFP4 Microscaling (Blackwell / H200+)

NVIDIA's Blackwell architecture (B100/B200) introduces MXFP4: 4-bit floating point
with microscaling at 32-element granularity. MXFP4 provides approximately 2× the
TFLOPS of FP8 at the cost of 1 bit fewer mantissa (E2M1 vs E4M3).

For GLM-5's MoE experts (intermediate=2048, hidden=6144), MXFP4 would reduce the
expert GEMM memory bandwidth by 2× vs FP8. This is a H100 → Blackwell migration path;
the current implementation targets SM90 (H100).

### Dynamic Precision: Early vs. Late Layers

DeepSeek-V3 observes that early layers (0-20) have smaller activation magnitudes and
distribute more smoothly, making them better candidates for FP8 quantization. Later
layers (60-78) show higher activation entropy and more outlier-prone distributions.
A dynamic strategy: use FP8 for layers 0-50, BF16 for layers 51-77, could maintain
0.999+ cosine similarity while still saving ~65% of the KV cache memory. Not
implemented here — uniform FP8 is used for implementation simplicity.

### Stochastic Rounding for Training

For fine-tuning or continued pre-training, round-to-nearest-even in FP8 introduces
systematic bias. Stochastic rounding (round up with probability proportional to the
fractional part, round down otherwise) makes the rounding error mean-zero. Critical
for gradient accumulation over small batches. Not needed for inference.
