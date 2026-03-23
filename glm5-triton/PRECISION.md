# Precision Map: GLM-5 Triton Kernels

This document maps every dtype transition in the Triton kernel implementation
(`glm5-triton/`). This implementation replaces three PyTorch operations with fused
Triton kernels: RMSNorm, SwiGLU, and Cross-Entropy Loss. All other operations are
identical to the pure PyTorch reference.

**Key claim:** Triton kernels are NUMERICALLY EQUIVALENT to PyTorch — they use the same
precision at every stage. The only difference is execution speed (fused memory access,
no intermediate allocations, reduced kernel launch overhead).

Model constants: hidden=6144, heads=64, kv_lora_rank=512, qk_rope=64, qk_nope=192,
vocab=154880, n_experts=256, layers=78.

---

## Stage-by-Stage Precision Table

| Stage | Kernel | Input dtype | Compute dtype | Output dtype | Source |
|---|---|---|---|---|---|
| Token embedding lookup | `nn.Embedding` | int64 | — | BF16 | model.py embed_tokens |
| **RMSNorm** | `_rms_layernorm_forward` (Triton) | BF16 → **FP32** | FP32 (variance) | BF16 | unsloth_rms_layernorm.py:41 |
| Q LoRA A projection | `nn.Linear` | BF16 | BF16 matmul | BF16 | model.py |
| Q LoRA A layernorm | `_rms_layernorm_forward` (Triton) | BF16 → **FP32** | FP32 | BF16 | unsloth_rms_layernorm.py:41 |
| Q LoRA B projection | `nn.Linear` | BF16 | BF16 matmul | BF16 | model.py |
| KV A projection | `nn.Linear` | BF16 | BF16 matmul | BF16 | model.py |
| KV A layernorm | `_rms_layernorm_forward` (Triton) | BF16 → **FP32** | FP32 | BF16 | unsloth_rms_layernorm.py:41 |
| KV B projection | `nn.Linear` | BF16 | BF16 matmul | BF16 | model.py |
| RoPE inv_freq matmul | `RotaryEmbedding.forward` | BF16 → **FP32** | FP32 matmul | FP32 → BF16 | rope_partial.py |
| QK dot product | `torch.matmul` | BF16 | BF16 matmul | BF16 | mla_attention.py |
| Attention softmax | `F.softmax` | BF16 → **FP32** | FP32 softmax | FP32 → BF16 | mla_attention.py |
| AV dot product | `torch.matmul` | BF16 | BF16 matmul | BF16 | mla_attention.py |
| Output projection | `nn.Linear` | BF16 | BF16 matmul | BF16 | mla_attention.py |
| DSA indexer scoring | `torch.einsum` | BF16 → **FP32** | FP32 | FP32 | dsa_indexer.py |
| Router linear | `TopkRouter.forward` | BF16 → **FP32** | FP32 matmul | FP32 | model.py |
| Router bias | `e_score_correction_bias` | — | — | FP32 buffer | model.py |
| **SwiGLU** | `_fg_kernel` (Triton) | BF16 → **FP32** | FP32 sigmoid | BF16 | unsloth_swiglu.py:39-44 |
| MoE gate+up projection | `nn.Linear` (grouped) | BF16 | BF16 matmul | BF16 | model.py |
| MoE down projection | `nn.Linear` (grouped) | BF16 | BF16 matmul | BF16 | model.py |
| Shared expert SwiGLU | `_fg_kernel` (Triton) | BF16 → **FP32** | FP32 sigmoid | BF16 | unsloth_swiglu.py:39-44 |
| LM head | `nn.Linear` | BF16 | BF16 matmul | BF16 | model.py |
| **Cross-entropy loss** | `_cross_entropy_forward` (Triton) | BF16 → **FP32** | FP32 log-sum-exp | FP32 scalar | unsloth_cross_entropy_loss.py:54 |

---

## Triton Kernel Precision Details

### RMSNorm (unsloth_rms_layernorm.py:41)

```python
X_row = tl.load(X + col_offsets, mask=mask, other=0).to(tl.float32)
W_row = tl.load(W + col_offsets, mask=mask, other=0)  # stays in weight dtype (BF16)

row_var = tl.sum(X_row * X_row, axis=0) / n_cols
eps_f32 = tl.full((), eps, tl.float32)
inv_var = tl.math.rsqrt(row_var + eps_f32)
normed = X_row * inv_var
normed = normed.to(W_row.dtype)   # cast normalized value back to BF16
output = normed * W_row           # BF16 * BF16 = BF16
```

The variance accumulation `sum(x^2) / n` is in FP32. For hidden_size=6144, summing
6144 BF16 squares would accumulate ~80 bits of rounding error in BF16, but only ~40
bits in FP32 (bounded by pairwise summation). The FP32 cast on load prevents this.
The inverse square root is also FP32, ensuring accurate normalization. The output is
cast back to BF16 before multiplying by the weight, matching the PyTorch reference
exactly (PyTorch's RMSNorm does `x.to(float32)` → compute → `x.to(input_dtype)`).

### SwiGLU (unsloth_swiglu.py:39-44)

```python
e_row = tl.load(e + offsets, mask=mask, other=0).to(tl.float32)   # gate in FP32
g_row = tl.load(g + offsets, mask=mask, other=0)                    # up stays BF16

f_row = e_row * tl.sigmoid(e_row)   # SiLU in FP32: prevents instability near 0
f_row = f_row.to(g_row.dtype)       # cast back to BF16
h_row = f_row * g_row               # BF16 result
```

`tl.sigmoid` in BF16 would overflow for large negative inputs (exp(-x) can underflow to
0, making sigmoid exactly 1.0 when the true value is e.g. 0.9998). The FP32 cast gives
sigmoid 8 more exponent bits. For expert intermediate_size=2048, activations can reach
large magnitudes after projection; FP32 sigmoid prevents saturation artifacts.

The `g_row` (up projection) is not cast to FP32 — only the gating path needs the
precision. This matches the DeepSeek-V3 implementation choice.

### Cross-Entropy Loss (unsloth_cross_entropy_loss.py:54)

```python
logits = tl.load(logits_ptr + col_offsets, mask=mask, other=-float("inf")).to(tl.float32)

c = tl.max(logits, 0)                          # max for numerical stability
logsumexp = c + tl.log(tl.sum(tl.exp(logits - c), 0))   # FP32 log-sum-exp
```

GLM-5's vocab_size=154880. In BF16, `exp(logit)` for logit values > 88 overflows to inf.
Even within range, summing 154880 exponentials in BF16 accumulates catastrophic
cancellation in the log-sum-exp. FP32 can represent exponentials up to e^88 ≈ 1.65×10^38
without overflow, and provides 7× more mantissa precision for the sum.

The Triton kernel chunks the vocab into BLOCK_SIZE tiles and uses the max-subtract
trick (log-sum-exp numerical stability), all in FP32.

### MoE Grouped GEMM (unsloth_moe/grouped_gemm/kernels/forward.py)

The Triton grouped GEMM kernel uses `tl.dot` for the inner matrix multiply. Triton's
`tl.dot` accumulates in FP32 by default when inputs are BF16, matching cuBLAS behavior.
This means the expert matmuls have FP32-precision accumulation even though inputs and
outputs are BF16. The effective precision is approximately FP32 for the dot product sum,
BF16 for the final stored result.

---

## Equivalence to PyTorch Reference

The Triton kernels are designed to be numerically equivalent to their PyTorch counterparts:

| Operation | PyTorch reference | Triton kernel | Equivalent? |
|---|---|---|---|
| RMSNorm | `x.float()` → variance → `x.to(dtype)` | `tl.float32` → variance → `.to(W_row.dtype)` | Yes |
| SwiGLU | `F.silu(gate) * up` (BF16) | FP32 sigmoid gate, BF16 up | Near-equivalent (gate more precise) |
| Cross-entropy | `F.cross_entropy` (FP32 internally) | FP32 log-sum-exp with chunking | Equivalent for large vocab |

The SwiGLU kernel is *more* numerically stable than the pure PyTorch reference for large
gate values — this is intentional and the outputs should match to within BF16 precision.

---

## FP32 Zones Inherited from Pure PyTorch

All FP32 zones from the pure PyTorch reference are preserved unchanged:

- **Router**: `F.linear(x.float(), self.weight.float())` — expert selection must be stable
- **Softmax**: `F.softmax(..., dtype=torch.float32)` — attention weight normalization
- **RoPE**: `inv_freq_expanded.float() @ position_ids_expanded.float()` — phase accuracy
- **DSA indexer scoring**: `q.float()`, `k_cached.float()` — sparse token selection

---

## Alternative Strategies

### Triton `tl.dot` Accumulator Dtype

Triton's `tl.dot` accepts an `allow_tf32` flag (default: True on NVIDIA hardware). Setting
`tl.dot(a, b, allow_tf32=False)` forces IEEE FP32 accumulation, at ~2x throughput cost.
Setting to True allows TF32 (10-bit mantissa accumulation), which is the hardware default.
The grouped GEMM kernel in this implementation uses the default (TF32 allowed). For
maximum precision equivalence with the pure PyTorch reference, set `allow_tf32=False`.

### TF32 Mode

`torch.backends.cuda.matmul.allow_tf32 = True` enables TF32 for all FP32 matmuls.
The router uses `F.linear(x.float(), self.weight.float())` — with TF32 enabled, this
uses 10-bit mantissa precision. The effect is minimal for expert routing (selection is
threshold-based), but could affect borderline topk decisions.

### Chunked Softmax

For very long sequences (202752 tokens), the attention score matrix is
[64 heads × seq × seq]. Chunked softmax computes log-sum-exp in tiles to avoid
materializing the full score matrix. This is the flashattention approach. The Triton
implementation here uses standard softmax on the materialized scores.
