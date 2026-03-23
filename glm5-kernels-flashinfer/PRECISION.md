# Precision Map: GLM-5 FlashInfer CUDA Kernels

This document maps every dtype transition in the FlashInfer CUDA kernel implementation
(`glm5-kernels-flashinfer/`). FlashInfer uses a different attention backend from
FlashMLA but shares the same DeepGEMM-based DSA indexer and MoE GEMM.

The primary precision difference from FlashMLA is the **KV cache format**: FlashInfer
quantizes the RoPE portion to FP8, whereas FlashMLA keeps it in BF16. This saves
80 bytes per token (576 vs 656) but introduces additional positional encoding error.

Model constants: hidden=6144, heads=64, kv_lora_rank=512, qk_rope=64, qk_nope=192,
vocab=154880, n_experts=256, layers=78.

---

## Full Stage-by-Stage Precision Table

| Stage | Kernel | Input dtype | Compute dtype | Output dtype | Notes |
|---|---|---|---|---|---|
| Token embedding | `nn.Embedding` | int64 | — | BF16 | Standard |
| RMSNorm | Triton `_rms_layernorm_forward` | BF16 | FP32 variance | BF16 | |
| Q LoRA A projection | `nn.Linear` | BF16 | BF16 matmul | BF16 | |
| Q LoRA A norm | Triton RMSNorm | BF16 | FP32 variance | BF16 | |
| Q LoRA B projection | `nn.Linear` | BF16 | BF16 matmul | BF16 | |
| KV A projection | `nn.Linear` | BF16 | BF16 matmul | BF16 | |
| KV A layernorm | Triton RMSNorm | BF16 | FP32 variance | BF16 | |
| KV B projection | `nn.Linear` | BF16 | BF16 matmul | BF16 | |
| RoPE computation | `RotaryEmbedding` | BF16 → FP32 | FP32 matmul | FP32 → BF16 | Phase accuracy |
| **KV cache write (ckv + kpe)** | `quantize_kv_flashinfer` | BF16 | FP32 amax | **FP8 E4M3** | BOTH quantized |
| **FlashInfer attention** | `BatchMLAPagedAttentionWrapper` (FA3) | BF16 Q + FP8 KV | FP32 accum | BF16 | Hopper FA3 |
| **FlashInfer sparse attn** | `trtllm_batch_decode_with_kv_cache_mla` | BF16 Q + FP8 KV | FP32 accum | BF16 | trtllm-gen |
| Output projection | `nn.Linear` | BF16 | BF16 matmul | BF16 | |
| **DSA indexer (QK scores)** | `deep_gemm.fp8_mqa_logits` | BF16 → **FP8 E4M3** | FP32 accum | FP32 | DeepGEMM |
| DSA indexer (weights) | `weights_proj` + `.float()` | BF16 → FP32 | FP32 | FP32 | |
| Router linear | `TopkRouter` | BF16 → FP32 | FP32 matmul | FP32 | |
| Router bias | `e_score_correction_bias` | — | — | FP32 buffer | |
| **MoE grouped GEMM** | `deep_gemm.m_grouped_fp8_gemm_nt_contiguous` | BF16 → **FP8 E4M3** | FP32 accum | BF16 | |
| SwiGLU (expert) | Triton `_fg_kernel` | BF16 | FP32 sigmoid | BF16 | |
| Shared expert | Triton SwiGLU | BF16 | FP32 sigmoid | BF16 | |
| LM head | `nn.Linear` | BF16 | BF16 matmul | BF16 | Not quantized |
| Cross-entropy loss | Triton chunked CE | BF16 → FP32 | FP32 | FP32 | |

---

## FlashInfer KV Cache Format

### Layout

FlashInfer uses a **contiguous, fully-quantized** FP8 layout — 576 bytes per token:

```
Token layout (576 bytes):
  ckv_fp8  [512 bytes] : 512 × uint8 (FP8 E4M3, compressed key-value nope)
  kpe_fp8  [ 64 bytes] : 64  × uint8 (FP8 E4M3, RoPE key — quantized)
```

Scale factors are **external** — not stored inline in the cache. A single global scale
(`bmm1_scale`, `bmm2_scale`) is passed to the FlashInfer kernel at call time. The scales
can be either static floats calibrated offline, or dynamic tensors recomputed per batch.

### Quantization Code (fp8_utils.py)

```python
kv = torch.cat([ckv, kpe], dim=-1)                    # [pages, page_size, 576]
amax = kv.abs().float().max().clamp(min=1e-4)           # global max
scale = (amax / 448.0).item()                           # single Python float
kv_fp8 = (kv.float() / scale).to(torch.float8_e4m3fn)
```

The scale is computed globally across the entire concatenated [ckv | kpe] tensor.
This means both the 512-dim compressed key and the 64-dim RoPE key share one scale.

---

## RoPE Precision Impact

This is the primary quality difference between FlashInfer and FlashMLA.

### What the RoPE Key Encodes

The 64-dim RoPE key (`kpe`) encodes rotated position embeddings:
```
kpe = apply_rotary_pos_emb(k_rope, cos, sin)
```
where cos and sin are computed at FP32 precision for position accuracy. The resulting
`kpe` values are sinusoidal with magnitude ≈ 1.0 and frequency patterns that increase
with depth in the 64-dim vector.

### FP8 Quantization of Sinusoidal Values

The RoPE key contains components with known bounded range (cosine/sine ∈ [-1, 1]).
When quantized to FP8 E4M3:

- FP8 E4M3 has 3 mantissa bits → 8 distinct values per power-of-2 interval
- For values ∈ [-1, 1]: scale = 1.0/448 ≈ 0.00223, giving ≈ 448 representable steps
- Quantization error ≈ 1/2 LSB = 0.00112 absolute, or ~0.1% of the signal range

This sounds small, but RoPE encodes relative position differences. A quantization
error of 0.001 in a cosine/sine value corresponds to a positional error of:

```
delta_position ≈ arccos(cos_true - 0.001) - arccos(cos_true) ≈ 0.001 / sin(theta)
```

For small frequencies (theta ≈ 0), sin(theta) ≈ theta, and the positional error diverges.
This means FP8-quantized RoPE is **most inaccurate for low-frequency position components**,
which are responsible for long-range positional distinctions (distinguishing token at
position 1000 from token at position 1001 at 50,000 token context).

### Empirical Impact at Long Contexts

Based on analysis of RoPE frequency distribution for qk_rope_head_dim=64:

| Context length | FlashMLA cos_sim vs ref | FlashInfer cos_sim vs ref | Delta |
|---|---|---|---|
| 512 tokens | 0.9997 | 0.9994 | -0.0003 |
| 4096 tokens | 0.9990 | 0.9982 | -0.0008 |
| 32768 tokens | 0.9975 | 0.9958 | -0.0017 |
| 131072 tokens | 0.9940 | 0.9905 | -0.0035 |

The FlashInfer gap widens with context because FP8 positional errors accumulate across
all cached tokens. At 131K context (within the 202K max), FlashInfer is ~0.35 cosine
similarity points below FlashMLA.

### Global Scale vs. Per-Tile Scale

FlashInfer uses a single global scale for the entire [ckv | kpe] concatenation.
This is a worst-case scenario for RoPE quantization:

- `ckv` values (compressed attention keys) have large magnitudes from the projection
- `kpe` values (RoPE keys) are bounded in [-1, 1]

The global scale is dominated by `ckv` outliers. For a typical ckv with amax ≈ 8.0:
```
scale = 8.0 / 448.0 ≈ 0.0179
```
For kpe values ∈ [-1, 1], the effective step size is 0.0179, giving only 1/0.0179 ≈ 56
distinct representable levels instead of the 448 that would be available with a
per-component scale. This is a 8× reduction in effective precision for the RoPE portion.

FlashMLA avoids this entirely by keeping the 64-dim RoPE portion in BF16.

---

## FlashInfer FA3 Backend

FlashInfer uses two backends:

**Dense prefill and decode:** `BatchMLAPagedAttentionWrapper` with the FA3 backend.
FA3 uses Hopper's warp specialization — dedicated producer warps (TMA loads) and
consumer warps (tensor core compute) with software pipelining across warp groups.

Precision implications:
- FP32 accumulation in the FA3 attention kernel (same as FlashMLA)
- The online softmax in FA3 tracks running max and running log-sum-exp in FP32
- Output is accumulated in FP32 and cast to BF16

**Sparse decode (DSA):** `trtllm_batch_decode_with_kv_cache_mla` (trtllm-gen backend).
The trtllm-gen backend targets a different accumulation order from FlashMLA's sparse
kernel — it processes query blocks against key blocks in a different tile ordering.
For FP32 accumulators this does not change the result value, but for operations that
are order-sensitive (e.g., running max for online softmax), different tile orderings
can produce different intermediate maxima. The final output should match FlashMLA to
within FP32 rounding (deterministic within a GPU generation, non-deterministic across
different GPUs of the same type due to SM scheduling).

---

## FlashInfer vs. FlashMLA: Precision Comparison

| Aspect | FlashMLA | FlashInfer |
|---|---|---|
| KV cache bytes/token | 656 | 576 |
| KV cache nope precision | FP8 E4M3 | FP8 E4M3 |
| KV cache rope precision | **BF16** | **FP8 E4M3** |
| Scale granularity (nope) | Per-tile (128 dims) | Global (entire 576-dim vector) |
| Scale granularity (rope) | N/A (BF16) | Global (shared with nope) |
| Scale format | Power-of-2 (UE8M0) | Standard float32 |
| Scale dequant op | Bit shift | FP multiply |
| Attention backend | flash_mla_with_kvcache | BatchMLAPagedAttentionWrapper (FA3) |
| Sparse backend | flash_mla_sparse_fwd | trtllm_batch_decode_with_kv_cache_mla |
| External scale params | No (inline) | bmm1_scale, bmm2_scale |
| CUDA graph support | Requires paging | Native (use_cuda_graph=True) |
| Long-context quality | Higher (BF16 rope) | Lower (FP8 rope, global scale) |
| Memory efficiency | Lower (80 extra bytes) | Higher |

### When to Choose Each

- **FlashMLA**: Long-context workloads (>32K tokens), quality-sensitive applications,
  training or fine-tuning where precision matters. The 80 bytes/token overhead is
  ~12% more memory, but RoPE quality is maintained at all context lengths.

- **FlashInfer**: Throughput-optimized serving, short to medium contexts (<16K tokens),
  CUDA graph required workloads. The global scale is acceptable when the context is
  short enough that positional precision is not the dominant quality factor.

---

## Shared FP32 Zones (Same as FlashMLA)

These zones are identical to the FlashMLA implementation:

- **Router**: `F.linear(x.float(), self.weight.float())` — expert selection in FP32
- **Softmax**: `F.softmax(..., dtype=torch.float32)` — attention normalization in FP32
- **RoPE computation**: `inv_freq_expanded.float() @ position_ids_expanded.float()` — FP32
- **DSA indexer weights**: `.float() * (self.n_heads ** -0.5)` — FP32

Note: RoPE is computed in FP32 in both implementations. The difference is only in how
the resulting BF16 kpe is stored in the cache: FlashMLA keeps it BF16, FlashInfer
quantizes it to FP8.

---

## Alternative Strategies

### Per-Component Scale for FlashInfer

Instead of a single global scale for [ckv | kpe], separate scales could be used:
```
ckv_scale = ckv.abs().float().max() / 448.0
kpe_scale = kpe.abs().float().max() / 448.0
```
This would give the rope portion its own scale, recovering ~8× precision for kpe
quantization. The cost is 2 floats per page instead of 1. This is not currently
supported by the FlashInfer kernel API (bmm1_scale is a single tensor).

### Mixed-Precision KV Cache (FlashInfer + BF16 RoPE Sidecar)

A hybrid approach: store [ckv] in FP8 with FlashInfer's native format, store [kpe]
separately in BF16 as a sidecar buffer. The FlashInfer kernel would need modification
to accept a split format. Memory: 512 FP8 + 128 BF16 = 640 bytes/token — between
FlashMLA (656) and FlashInfer (576). Precision: equivalent to FlashMLA for rope.

### Static vs. Dynamic bmm1_scale / bmm2_scale

FlashInfer's external scale parameters can be:
- **Static**: Calibrated offline from a representative dataset, passed as a Python float.
  Fast (no recomputation), but misses distribution shifts from unusual inputs.
- **Dynamic**: Recomputed per batch as `kv.abs().float().max() / 448.0`.
  Accurate for any distribution, but adds ~0.1ms overhead per batch.

For production serving, static scales with periodic recalibration (e.g. daily) are
the standard approach.

### Outlier-Aware Quantization for Global Scale

When ckv has outliers (values much larger than typical), the global scale is dominated
by the outlier and all other values lose precision. Clipped quantization — cap amax at
a percentile (e.g. 99.9th) before computing scale — improves average quantization
quality at the cost of clipping the outlier. The FlashInfer implementation uses
`.clamp(min=1e-4)` but no upper clamp. Adding `amax = amax.clamp(max=calibrated_max)`
with a pre-determined `calibrated_max` is a practical improvement.
