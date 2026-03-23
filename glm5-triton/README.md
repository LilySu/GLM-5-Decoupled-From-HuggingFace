# glm5-triton

Self-contained GLM-5 model implementation with Triton-accelerated kernels.
No external dependencies beyond `torch` and `triton`.

## Architecture

GLM-5 is a 744B parameter (40B active) Mixture-of-Experts language model
from Zhipu AI (arXiv 2602.15763v2). Key innovations:

- **MLA (Multi-Latent Attention)**: Compresses KV into a 512-dim latent space,
  applies RoPE to only a 64-dim decoupled stream, with asymmetric head dims
  (QK=256, V=256). More memory-efficient than GQA for long contexts.

- **DSA (DeepSeek Sparse Attention)**: A learned indexer selects the top-2048
  most relevant tokens per query position. Attention is computed only over this
  sparse subset, reducing cost by ~1.5-2x on long sequences.

- **256-Expert MoE**: Sigmoid-based routing with group selection picks 8 experts
  per token (plus 1 always-active shared expert). Scaled by factor 2.5.

- **MTP (Multi-Token Prediction)**: Parameter-shared draft layers for speculative
  decoding (stub only — not in released weights).

## Component Map

Files prefixed with `unsloth_` are Triton kernels from
[unsloth](https://github.com/unslothai/unsloth). Files without the prefix
are pure-PyTorch implementations.

```
+----------------------------------+------------------------+-------------------+
| Component                        | File                   | Status            |
+----------------------------------+------------------------+-------------------+
| RMSNorm fwd+bwd                  | unsloth_rms_layernorm  | Triton kernel     |
| SwiGLU fwd+bwd                   | unsloth_swiglu         | Triton kernel     |
| Cross-Entropy Loss (chunked)     | unsloth_cross_entropy  | Triton kernel     |
| LoRA MLP/QKV/W                   | unsloth_fast_lora      | Triton kernel     |
| MoE Grouped GEMM                 | unsloth_moe/           | Triton kernel     |
| Utilities                        | unsloth_utils          | Triton support    |
+----------------------------------+------------------------+-------------------+
| Config                           | config                 | PyTorch (dict)    |
| KV Cache                         | cache                  | PyTorch           |
| Decoupled partial-dim RoPE       | rope_partial           | PyTorch           |
| DSA Lightning Indexer            | dsa_indexer            | PyTorch           |
| DSA Sparse Attention             | dsa_sparse_attention   | PyTorch           |
| MLA (Multi-Latent Attention)     | mla_attention          | PyTorch           |
| FeedForward / MoE / Router       | model                  | PyTorch           |
| DecoderLayer / Base / CausalLM   | model                  | PyTorch           |
| MTP (Multi-Token Prediction)     | mtp                    | Stub              |
+----------------------------------+------------------------+-------------------+
```

### Where Triton kernels accelerate the model

The Triton kernels are drop-in replacements for specific operations in the
PyTorch forward pass:

| PyTorch op | Triton replacement | Speedup target |
|---|---|---|
| `RMSNorm.forward(x)` | `fast_rms_layernorm(norm, x)` | 313 norm ops/fwd |
| `F.silu(gate) * up` | `swiglu_fg_kernel(gate, up)` | Dense MLP activation |
| `F.cross_entropy(logits, labels)` | `fast_cross_entropy_loss(logits, labels)` | 154K vocab chunked |
| Per-expert loop in MoeExperts | `grouped_gemm()` from `unsloth_moe/` | 75 MoE layers |

### Model config (Table 10 from paper)

```
+-------------------------+----------+
| Parameter               | GLM-5    |
+-------------------------+----------+
| Total Parameters        | 744B     |
| Activated Parameters    | 40B      |
| Dense Layers            | 3        |
| MoE Layers              | 75       |
| MTP Layers              | 1        |
| Hidden Dim              | 6144     |
| Dense Intermediate Dim  | 12288    |
| MoE Intermediate Dim    | 2048     |
| QK Head Dim (nope)      | 192      |
| QK Rope Head Dim        | 64       |
| V Head Dim              | 256      |
| Q LoRA Dim              | 2048     |
| KV LoRA Dim             | 512      |
| Attention Heads         | 64       |
| Indexer Attn Heads      | 32       |
| Indexer Head Dim        | 128      |
| Experts (total)         | 256      |
| Routed Experts (per tok)| 8        |
| Shared Experts          | 1        |
| Vocabulary Size         | 154880   |
+-------------------------+----------+
```

## Validation

Run from the `glm5/` root directory:

```bash
# Inspect sample data structure (no GPU needed)
python3 data/sample_data.py

# Run 8-test validation suite (needs GPU, <1GB VRAM, ~15 seconds)
python3 glm5-triton/validate.py
```

Tests:

```
Test 1: Forward pass              Logits shape, finite loss
Test 2: Backward pass             Gradients reach 75%+ of parameters
Test 3: Training convergence      Loss drops >80% in 20 steps
Test 4: Label masking             -100 tokens properly ignored
Test 5: KV cache decode           10 autoregressive steps
Test 6: Multi-turn padded batch   Variable-length conversations, padded
Test 7: Long sequence (256+ tok)  Attention + DSA indexer on longer input
Test 8: Gradient checkpointing    Same loss/gradients with checkpointing on
```

## Usage

```python
import sys
sys.path.insert(0, "/path/to/glm5")

from importlib import import_module
glm5 = import_module("glm5-triton")

# Use default config (or modify for a smaller test)
cfg = glm5.GLM_MOE_DSA_CONFIG.copy()
model = glm5.GlmMoeDsaForCausalLM(cfg).cuda().bfloat16()

# Forward pass
import torch
input_ids = torch.randint(0, cfg["vocab_size"], (1, 128), device="cuda")
loss, logits, kv_cache = model(input_ids=input_ids, labels=input_ids)
```

### Quick validation from Python

```python
# Use the shared test data for a structured forward pass
from data.sample_data import get_sample_batch, get_tiny_config

cfg = get_tiny_config()  # tiny model, ~395K params
model = glm5.GlmMoeDsaForCausalLM(cfg).cuda().bfloat16()

input_ids, labels = get_sample_batch(batch_size=2, device="cuda")
loss, logits, _ = model(input_ids=input_ids, labels=labels)
print(f"loss={loss.item():.4f}, logits={logits.shape}")
```

## Forward pass flow

```
Input tokens [B, S]
    |
    v
Token Embedding [B, S, 6144]                  <- nn.Embedding
    |
    v
Compute RoPE (cos, sin) for positions         <- rope_partial.RotaryEmbedding
    |
    v
For each of 78 layers:
    |
    +-- input_layernorm (RMSNorm)              <- mla_attention.RMSNorm
    +-- MLA Attention:                         <- mla_attention.MLAttention
    |     Q: hidden -> q_a_proj -> RMSNorm -> q_b_proj -> split nope/rope -> RoPE
    |     KV: hidden -> kv_a_proj -> split -> RMSNorm -> kv_b_proj -> split K/V
    |     DSA Indexer: select top-2048 tokens  <- dsa_indexer.DSAIndexer
    |     Attention: QK^T * mask -> softmax -> V
    |     Output: o_proj
    +-- residual connection
    |
    +-- post_attention_layernorm (RMSNorm)
    +-- MLP:
    |     Layers 0-2:  FeedForward (SwiGLU)    <- model.FeedForward
    |     Layers 3-77: MoE (256 experts)       <- model.MoE
    +-- residual connection
    |
    v
Final RMSNorm [B, S, 6144]
    |
    v
LM Head -> logits [B, S, 154880]              <- nn.Linear
    |
    v
Cross-Entropy Loss (if labels provided)
```

## Architecture visualization

See [viz/glm5-architecture.html](../viz/glm5-architecture.html) for a 3-column
comparison of DeepSeek V3.2 vs GLM-5 Raw vs GLM-5 Triton with color-coded
component mapping.

## Dependencies

- `torch` (PyTorch)
- `triton` (required — the unsloth kernel files import triton at module level)
