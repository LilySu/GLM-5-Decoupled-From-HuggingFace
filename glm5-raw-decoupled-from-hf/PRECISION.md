# Precision Map: GLM-5 Pure PyTorch Reference

This document maps every dtype transition in the pure PyTorch reference implementation
(`glm5-raw-decoupled-from-hf/`). This implementation is the **precision baseline**. All
other implementations must produce outputs within tolerance of this reference.

Model constants: hidden=6144, heads=64, kv_lora_rank=512, qk_rope=64, qk_nope=192,
vocab=154880, n_experts=256, layers=78.

---

## Stage-by-Stage Precision Table

| Stage | Module | Input dtype | Compute dtype | Output dtype | Source |
|---|---|---|---|---|---|
| Token embedding lookup | `nn.Embedding` | int64 indices | — | BF16 | model.py embed_tokens |
| RMSNorm (input layernorm) | `RMSNorm.forward` | BF16 | FP32 (variance) | BF16 | model.py:20-24 |
| Q LoRA A projection | `nn.Linear` | BF16 | BF16 matmul | BF16 | model.py:269 |
| Q LoRA A layernorm | `RMSNorm.forward` | BF16 | FP32 (variance) | BF16 | model.py:268 |
| Q LoRA B projection | `nn.Linear` | BF16 | BF16 matmul | BF16 | model.py:269 |
| KV A projection | `nn.Linear` | BF16 | BF16 matmul | BF16 | model.py:276 |
| KV A layernorm | `RMSNorm.forward` | BF16 | FP32 (variance) | BF16 | model.py:278 |
| KV B projection | `nn.Linear` | BF16 | BF16 matmul | BF16 | model.py:280 |
| RoPE inv_freq matmul | `RotaryEmbedding.forward` | BF16 → **FP32** | FP32 matmul | FP32 → BF16 | model.py:124-132 |
| apply_rotary_pos_emb | `apply_rotary_pos_emb` | BF16 | BF16 | BF16 | model.py:41-48 |
| QK dot product (attn score) | `torch.matmul` | BF16 | BF16 matmul | BF16 | model.py:72 |
| Attention softmax | `F.softmax` | BF16 → **FP32** | FP32 softmax | FP32 → BF16 | model.py:76 |
| AV dot product (attn output) | `torch.matmul` | BF16 | BF16 matmul | BF16 | model.py:78 |
| Output projection | `nn.Linear` | BF16 | BF16 matmul | BF16 | model.py:337 |
| DSA indexer scoring (weights) | `DSAIndexer.forward` | BF16 → **FP32** | FP32 | FP32 | model.py:197 |
| DSA indexer scoring (QK) | `torch.einsum` | BF16 → **FP32** | FP32 einsum | FP32 | model.py:200 |
| Router linear | `TopkRouter.forward` | BF16 → **FP32** | FP32 matmul | FP32 | model.py:373 |
| Router bias (e_score_correction) | `register_buffer` | — | — | FP32 buffer | model.py:369 |
| Router sigmoid + topk | `MoE.route_tokens_to_experts` | FP32 | FP32 | FP32 | model.py:431 |
| Expert gate+up projection | `F.linear` | BF16 | BF16 matmul | BF16 | model.py:402 |
| SwiGLU (SiLU gate) | `F.silu` | BF16 | BF16 | BF16 | model.py:403 |
| Expert down projection | `F.linear` | BF16 | BF16 matmul | BF16 | model.py:404 |
| Shared expert (FeedForward) | `FeedForward.forward` | BF16 | BF16 matmul | BF16 | model.py:354-355 |
| Post-attention layernorm | `RMSNorm.forward` | BF16 | FP32 (variance) | BF16 | model.py:20-24 |
| Post-MoE layernorm | `RMSNorm.forward` | BF16 | FP32 (variance) | BF16 | model.py:20-24 |
| LM head | `nn.Linear` | BF16 | BF16 matmul | BF16 | model.py lm_head |
| Cross-entropy loss | `F.cross_entropy` | BF16 → **FP32** | FP32 | FP32 scalar | PyTorch default |

---

## FP32 Zones: Where and Why

### 1. Router (model.py:373)

```python
return F.linear(x.float(), self.weight.float())
```

The router computes expert selection scores. Expert selection is a discrete decision —
routing the wrong expert wastes the entire token's compute for that layer. FP32 is used
so that sigmoid probabilities are numerically stable and group-wise topk comparisons are
consistent. The `e_score_correction_bias` (model.py:369) is registered as `dtype=torch.float32`
for the same reason: it adjusts per-expert calibration scores where small differences matter.

### 2. Softmax (model.py:76)

```python
F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
```

BF16 has only 7 mantissa bits. For long sequences, attention scores can span a wide
dynamic range. Log-sum-exp in BF16 loses the small probability tails, causing attention
to collapse to near-uniform or near-argmax behavior. FP32 softmax is 8x more precise
for the exponential, then the result is cast back to BF16 for the AV matmul.

### 3. RoPE calculation (model.py:124-132)

```python
inv_freq_expanded.float() @ position_ids_expanded.float()
```

Positional encoding accuracy degrades quadratically with sequence length in lower
precision. At max_position_embeddings=202752, BF16 cannot represent the full range of
frequency × position products without significant rounding error. FP32 computes the
correct phase angles; the result is cast back to BF16 before `cos`/`sin` are returned.

### 4. DSA indexer scoring (model.py:197, 200)

```python
weights = self.weights_proj(hidden_states).float() * (self.n_heads ** -0.5)
scores = torch.einsum("bshd,btd->bsht", q.float(), k_cached.float()) * self.softmax_scale
```

The DSA indexer selects the top-2048 tokens for sparse attention. This is also a
discrete decision — selecting the wrong token set changes which context the layer
can attend to. FP32 prevents marginal score differences from being rounded away
and causing incorrect topk selections.

### 5. Cross-entropy loss

`F.cross_entropy` converts logits to FP32 internally before the log-sum-exp. This
is PyTorch's default behavior for numerical stability. The loss gradient flows back
in BF16 through the logits.

---

## What Stays BF16

All weight matrices are stored and applied in BF16:
- `embed_tokens` embedding table
- All `nn.Linear` weights: `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa`, `kv_b_proj`, `o_proj`
- All MoE expert weights: `gate_up_proj`, `down_proj`
- Shared expert weights: `gate_proj`, `up_proj`, `down_proj`
- DSA indexer weights: `wq_b`, `wk`, `weights_proj`
- `lm_head` weight

All residual stream tensors are BF16. The hidden state is never up-cast for the
residual add — only specific operations use FP32 internally.

---

## Precision Baseline Statement

This is the precision BASELINE. All other implementations must produce outputs within
tolerance of this reference.

Acceptable tolerances (empirical, one decoder layer, hidden=6144, B=1, S=32):

| Metric | Triton | FlashMLA | FlashInfer |
|---|---|---|---|
| Cosine similarity vs. reference | > 0.9999 | > 0.999 | > 0.998 |
| RMSE vs. reference | < 1e-3 | < 5e-3 | < 8e-3 |
| Max abs error | < 0.05 | < 0.2 | < 0.3 |

These tolerances widen for longer contexts due to accumulated RoPE and attention
rounding, and for deeper layers due to residual drift.

---

## Alternative Precision Strategies

### Kahan Summation

Kahan compensated summation reduces floating-point accumulation error from O(n·eps) to
O(eps). Useful for the residual stream across 78 layers. Not implemented here — PyTorch's
`torch.sum` uses pairwise summation internally on CUDA, which gives O(log n · eps).
Kahan would require a custom kernel.

### Mixed-Precision Matmul (TF32)

NVIDIA A100/H100 hardware supports TF32 (10-bit mantissa, same range as FP32) for
FP32 matmuls, enabled by `torch.backends.cuda.matmul.allow_tf32 = True`. This gives
~8x throughput over FP32 at the cost of 3 mantissa bits. The reference implementation
does not set this flag — all FP32 matmuls use full 23-bit mantissa.

### Stochastic Rounding

Round-to-nearest-even (the default) introduces systematic bias in BF16 matmuls that
can accumulate over many layers. Stochastic rounding adds uniform noise in [-0.5 ulp, +0.5 ulp]
before rounding, making the rounding error mean-zero. Required for BF16 training
convergence at small batch sizes. Not needed for inference.

### FP8 Accumulation

FP8 E4M3 (max representable value 448) is used by FlashMLA and DeepGEMM paths.
The reference stays entirely in BF16/FP32 to provide a clean numerical baseline.
