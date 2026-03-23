# GLM-5 CUDA Kernel Compatibility Research

## Context

GLM-5 is a 744B (40B active) MoE model with three novel architectural features: **MLA** (Multi-Latent Attention with compressed KV), **DSA** (DeepSeek Sparse Attention with lightning indexer), and **256-expert MoE** with sigmoid routing. The current implementations (`glm5-raw-decoupled-from-hf/` and `glm5-triton/`) use pure PyTorch for MLA, DSA, and RoPE, with Triton kernels only for RMSNorm, SwiGLU, CE Loss, LoRA, and MoE grouped GEMM (from Unsloth).

This research identifies which CUDA kernels from 7 external repos can accelerate GLM-5 inference (primary) and training (noted), while staying true to the paper's architecture.

---

## GLM-5 Key Dimensions (Table 10)

```
hidden_size:         6144        num_heads:          64
q_lora_rank:         2048        kv_lora_rank:       512
qk_nope_head_dim:    192         qk_rope_head_dim:   64
qk_head_dim:         256         v_head_dim:         256
n_routed_experts:    256         num_experts_per_tok: 8
moe_intermediate:    2048        n_shared_experts:    1
index_n_heads:       32          index_head_dim:      128
index_topk:          2048        rope_theta:          10000
num_layers:          78 (3 dense + 75 MoE)
```

### Critical Compatibility Note: MLA Weight Absorption

GLM-5 and DeepSeek-V3 share the same `kv_lora_rank=512` and `qk_rope_head_dim=64`. With **weight absorption** (absorbing `kv_b_proj` into Q), the compressed KV cache is `512+64=576` dims for both models. This means **FlashMLA kernels designed for DeepSeek-V3 work with GLM-5** when using absorbed weights — the different head dims (GLM-5: QK=256,V=256 vs DS-V3: QK=576,V=512) only affect the external matmuls, not the attention kernel itself.

---

## Kernel Availability Table (Inference-First)

### Legend
- **Yes** = Kernel exists and is directly compatible
- **Yes\*** = Kernel exists but needs adaptation (dimension changes, format conversion)
- **Absorbed** = Compatible when using MLA weight absorption technique
- **Partial** = Covers some but not all of the operation
- **No** = Not available

| GLM-5 Component | FlashMLA | DeepGEMM | FlashInfer | TensorRT-LLM | xLLM | vLLM | SGLang |
|---|---|---|---|---|---|---|---|
| **MLA Prefill** | Absorbed | — | Yes (mla module) | Yes (flashMLA/) | Partial | Absorbed | Absorbed (multi-backend) |
| **MLA Decode** | Absorbed | — | Yes (mla module) | Yes (mlaKernels.cu, XQA) | Partial | Absorbed | Absorbed (multi-backend) |
| **DSA Indexer (Lightning)** | — | Yes (fp8_mqa_logits) | No | No (infra only) | No | Yes (DeepGEMM) | Yes (fused Triton) |
| **DSA Sparse Prefill** | Absorbed | — | No (block-sparse only) | Partial (custom mask) | No | Yes (FLASHMLA_SPARSE) | Yes (NSA backends) |
| **DSA Sparse Decode** | Absorbed | — | No (block-sparse only) | Partial (custom mask) | No | Yes (FLASHMLA_SPARSE) | Yes (NSA backends) |
| **MoE Grouped GEMM** | — | Yes (FP8, BF16) | Yes (cutlass_fused_moe) | Yes (CUTLASS moe_gemm) | Partial | Yes (DeepGEMM FP8) | Yes (multi-backend) |
| **MoE Sigmoid Router** | — | — | Partial (top-k routing) | Yes (DeepSeek routing) | Yes (sigmoid kernel) | Yes | Yes (moe_fused_gate) |
| **RMSNorm** | — | — | Yes | Yes (FP8 variant) | Yes | Yes | Yes |
| **Partial RoPE (64/256)** | Yes (built-in) | — | Partial (unclear) | Yes | Yes | Yes | Yes (fused w/ KV write) |
| **SwiGLU Fused** | — | — | Partial (silu_and_mul) | Yes (GEMM+SwiGLU FP8) | No | Yes (in MoE) | Yes (silu_and_mul) |
| **Cross-Entropy Loss** | — | — | No | Yes (train_ops/) | No | No | No |
| **FP8 Quantization** | Yes (KV cache) | Yes (GEMM) | Yes (gemm, quant) | Yes (comprehensive) | Yes (W8A8) | Yes (W8A8) | Yes (W8A8, per-token) |
| **Paged KV Cache** | Yes (block=64) | — | Yes | Yes | Yes | Yes (PagedAttn v2) | Yes |
| **Top-k Sampling** | — | — | Yes | Yes | Yes | Yes (topk.cu) | Yes |
| **MTP / Spec. Decode** | — | — | No | Yes (mtpKernels.cu) | No | Yes (MTP backend) | Yes (EAGLE, MTP) |
| **Causal Mask Gen** | Built-in | — | Built-in | Built-in | Built-in | Built-in | Built-in |

---

## Detailed Component Analysis

### 1. MLA Attention (most critical)

**Best options:** FlashMLA (via vLLM/SGLang), FlashInfer MLA module

| Repo | Kernel | How it works with GLM-5 | GPU | Dtype | Train? |
|------|--------|------------------------|-----|-------|--------|
| **FlashMLA** | `sparse_prefill_fwd`, `sparse_decode_fwd`, `dense_decode_fwd` | Absorbed MLA: KV cache = 576D (512 nope + 64 rope). Q absorbed via kv_b_proj. Identical format to DS-V3. | SM90, SM100 | BF16, FP8 KV | No |
| **FlashInfer** | `BatchMLAPagedAttentionWrapper` | Native MLA module with paged cache. Weight absorption built in. | SM75+ | BF16, FP8 | No |
| **TensorRT-LLM** | `mlaKernels.cu`, `flash_fwd_mla_*.cu`, XQA | Full MLA pipeline: RoPE + KV assignment + paged cache. Absorbed format. | SM80+ | FP16, BF16, FP8 | Partial |
| **vLLM** | FlashMLABackend, FlashAttnMLABackend | Dispatches to FlashMLA or FlashAttn. FP8 KV cache. 40% throughput improvement reported. | SM90+ | BF16, FP8 | No |
| **SGLang** | FA3, FlashInfer, FlashMLA, CUTLASS, TRT-LLM backends | 5 selectable backends. 7x accel reported. Data-parallel attention reduces KV. | SM90+ | BF16, FP8 | No |

**Key file:** `glm5-triton/mla_attention.py` (currently PyTorch, STATUS: "No Triton kernel yet")

### 2. DSA Lightning Indexer

**Best options:** DeepGEMM (fp8_mqa_logits), SGLang (fused Triton)

| Repo | Kernel | Details | GPU | Dtype |
|------|--------|---------|-----|-------|
| **DeepGEMM** | `fp8_mqa_logits`, `fp8_paged_mqa_logits` | Fused: `out[i,j] = ReLU(q[i] @ kv[j]) * weights[i]` summed across heads. Exactly matches GLM-5's indexer scoring formula. | SM90, SM100 | FP8 |
| **SGLang** | Fused Triton NSA Indexer | Optimizes K/S buffer access for lightning indexer. Dedicated key & scale cache. | SM90+ | FP8 |
| **vLLM** | via DeepGEMM integration | Uses DeepGEMM's MQA logits kernel. | SM90+ | FP8 |

**Critical note from paper (Section 3.2):** Deterministic TopK is mandatory — non-deterministic CUDA topk caused "drastic performance degradation during RL." `torch.topk` is deterministic.

**Key file:** `glm5-triton/dsa_indexer.py` (currently PyTorch, STATUS: "No Triton kernel yet")

### 3. DSA Sparse Attention

**Best options:** FlashMLA sparse kernels (via vLLM/SGLang)

| Repo | Kernel | Details | GPU |
|------|--------|---------|-----|
| **FlashMLA** | `sparse_prefill_fwd`, `sparse_decode_fwd` | Token-level sparse attention. Variable topk per query via `topk_length` tensor. 640 TFlops prefill (H800). | SM90, SM100 |
| **vLLM** | FLASHMLA_SPARSE backend | Integrates FlashMLA sparse + DeepGEMM indexer. Separate prefill/decode pipelines. | SM90+ |
| **SGLang** | NSA backends (FlashMLA Sparse, TRT-LLM NSA, FA3 Sparse) | Multiple backend options. TRT-LLM NSA: 3-5x on Blackwell. | SM90+ |
| **FlashInfer** | BlockSparseAttentionWrapper | **Block-sparse only, NOT token-level.** Cannot represent DSA's arbitrary top-2048 selection. | SM75+ |
| **TensorRT-LLM** | `preparecustomMask.cu` | Custom mask infrastructure but no dedicated DSA kernel. | SM80+ |

**Key file:** `glm5-triton/dsa_sparse_attention.py` (currently PyTorch, STATUS: "No Triton kernel yet")

### 4. MoE Grouped GEMM (256 experts, top-8, SwiGLU)

**Best options:** DeepGEMM (FP8), SGLang/vLLM (multi-backend)

| Repo | Kernel | Details | GPU | Dtype | Train? |
|------|--------|---------|-----|-------|--------|
| **DeepGEMM** | `m_grouped_fp8_gemm_*_contiguous`, `m_grouped_fp8_gemm_*_masked` | Contiguous: training/prefill. Masked: decode with CUDA graphs. K-grouped for backward. | SM90, SM100 | FP8, BF16 | Yes |
| **FlashInfer** | `cutlass_fused_moe`, `trtllm_fp8_*_moe` | CUTLASS and TRT-LLM integrated MoE. Multiple quantization modes. | SM75+ | BF16, FP8, FP4 | No |
| **TensorRT-LLM** | CUTLASS moe_gemm/, `RoutingDeepSeek.cu` | DeepSeek-specific routing + grouped GEMM. Block-scale MoE. | SM80+ | FP8 | No |
| **SGLang** | DeepGEMM, FlashInfer, CUTLASS, Triton, AITER | 5+ backends. `moe_fused_gate` for hierarchical selection (256 experts, 32/group). | SM90+ | FP8, BF16 | No |
| **vLLM** | `deep_gemm_moe_fp8()` | DeepGEMM integration. `VLLM_USE_DEEP_GEMM` env var. | SM90+ | FP8 | No |
| **Existing** | `glm5-triton/unsloth_moe/grouped_gemm/` | Triton grouped GEMM with TMA, autotuning, permutation support. | SM80+ | BF16 | Yes |

**Key file:** `glm5-triton/unsloth_moe/grouped_gemm/kernels/forward.py` (already Triton-accelerated)

### 5. MoE Sigmoid Router + Group Selection

**Best options:** SGLang (moe_fused_gate), xLLM (sigmoid kernel)

| Repo | Kernel | Details |
|------|--------|---------|
| **SGLang** | `moe_fused_gate` | Fuses sigmoid + bias + group-level top-k + token-level top-k. Supports 256 experts, 32/group. |
| **xLLM** | `moe_fused_topk.cu`, `moe_topk_sigmoid_kernels.cuh` | Dedicated sigmoid-based routing kernel. |
| **TensorRT-LLM** | `RoutingDeepSeek.cu` | DeepSeek routing: histogram, cluster, offsets kernels. |
| **Existing** | `glm5-triton/model.py:MoE.route_tokens_to_experts()` | PyTorch: sigmoid → bias → group topk → expert topk → normalize → scale. |

### 6. Supporting Kernels

| Component | Best Source | Kernel | Notes |
|-----------|-----------|--------|-------|
| **RMSNorm** | Already have (Unsloth Triton) | `glm5-triton/unsloth_rms_layernorm.py` | Also: FlashInfer `fused_add_rmsnorm`, TRT-LLM FP8 variant |
| **SwiGLU** | Already have (Unsloth Triton) | `glm5-triton/unsloth_swiglu.py` | Also: TRT-LLM fused GEMM+SwiGLU (FP8) |
| **CE Loss** | Already have (Unsloth Triton) | `glm5-triton/unsloth_cross_entropy_loss.py` | TRT-LLM also has training CE kernels |
| **Partial RoPE** | PyTorch (sufficient) | `glm5-triton/rope_partial.py` | 64-dim is bandwidth-bound; PyTorch fast enough. SGLang fuses RoPE+KV write for decode. |
| **Paged KV Cache** | vLLM / FlashInfer | PagedAttention v2, page_size=64 | GLM-5's KVCache is simple Python; paged version needed for serving |
| **Top-k Sampling** | FlashInfer / vLLM | `top_k`, `top_k_sampling_from_logits` | For generation |
| **FP8 Quant** | TensorRT-LLM / DeepGEMM | `invokePerTokenQuantization`, per-token group scaling | For inference optimization |
| **MTP** | vLLM / SGLang / TRT-LLM | `mtpKernels.cu`, MTP backend | GLM-5 uses 3 shared MTP layers, 4-token speculation |

---

## Recommended Kernel Stack for GLM-5

### Inference Serving (Priority)

| Component | Recommended Source | Why |
|-----------|-------------------|-----|
| MLA + DSA Attention | **vLLM** or **SGLang** (using FlashMLA + DeepGEMM backends) | Most complete: absorbed MLA + sparse attention + paged KV + FP8, all integrated |
| DSA Lightning Indexer | **DeepGEMM** `fp8_mqa_logits` | Exact match for GLM-5's scoring formula: fused ReLU + weighted MQA logits |
| MoE Grouped GEMM | **DeepGEMM** (FP8) or existing Unsloth Triton | DeepGEMM for FP8 inference; Unsloth Triton for BF16/training |
| MoE Routing | **SGLang** `moe_fused_gate` | Fused sigmoid + group selection for 256 experts |
| RMSNorm | **Existing** (Unsloth Triton) | Already accelerated |
| SwiGLU | **Existing** (Unsloth Triton) | Already accelerated |
| Paged KV Cache | **vLLM** PagedAttention or **FlashMLA** paged | Essential for serving; FlashMLA uses block_size=64 |
| FP8 Quantization | **DeepGEMM** + **TensorRT-LLM** | Per-token FP8 for activations, block-scale for weights |
| MTP Speculation | **vLLM** MTP backend or **TRT-LLM** `mtpKernels.cu` | 4-token draft with 3 shared layers |
| Top-k Sampling | **FlashInfer** | Deterministic, GPU-optimized |

### Training (Noted)

| Component | Source | Why |
|-----------|--------|-----|
| MoE Grouped GEMM fwd+bwd | **DeepGEMM** (m-grouped + k-grouped) | Full forward and backward grouped GEMM |
| RMSNorm fwd+bwd | **Existing** Unsloth Triton | Already has backward pass |
| SwiGLU fwd+bwd | **Existing** Unsloth Triton | Already has backward pass |
| CE Loss fwd+bwd | **Existing** Unsloth Triton | Chunked for 154K vocab |

---

## Gaps: Components Needing Custom Kernels

1. **DSA Indexer deterministic TopK** — DeepGEMM's `fp8_mqa_logits` handles scoring but may use non-deterministic topk. Paper requires deterministic topk (Section 3.2). May need to combine DeepGEMM scoring + `torch.topk`.

2. **Non-absorbed MLA** — If NOT using weight absorption (e.g., for training where you need gradients through `kv_b_proj`), FlashMLA cannot be used directly. Would need custom attention kernel for QK=256, V=256 heads, or fall back to eager attention.

3. **DSA + MLA fusion for training backward** — No repo provides fused backward through sparse-attention + MLA compression. Training requires PyTorch autograd through the eager attention path.

---

## Hardware Requirements

| Kernel Source | Minimum GPU | Optimal GPU |
|---------------|-------------|-------------|
| FlashMLA | SM90 (H100/H800) | SM100 (B200) |
| DeepGEMM | SM90 (H100/H800) | SM100 (B200) |
| FlashInfer | SM75 (T4) | SM90+ |
| TensorRT-LLM | SM80 (A100) | SM90+ |
| vLLM FlashMLA | SM90 (H100) | SM100 |
| SGLang NSA | SM90 (H100) | SM100 |
| Unsloth Triton (existing) | SM80 (A100) | SM90+ |

**Minimum for full kernel coverage: SM90 (H100/H800)**

---

## Source Repos

| Repo | URL | Key Kernels for GLM-5 |
|------|-----|----------------------|
| FlashMLA | github.com/deepseek-ai/FlashMLA | Absorbed MLA attention (prefill+decode), sparse attention, paged FP8 KV cache |
| DeepGEMM | github.com/deepseek-ai/DeepGEMM | FP8 grouped GEMM (MoE), MQA logits (DSA indexer), dense GEMM |
| FlashInfer | github.com/flashinfer-ai/flashinfer | MLA module, MoE, RMSNorm, RoPE, sampling, paged KV |
| TensorRT-LLM | github.com/NVIDIA/TensorRT-LLM | MLA, MoE routing, SwiGLU+GEMM fusion, MTP, FP8, speculative decoding |
| xLLM | github.com/jd-opensource/xllm | MoE sigmoid routing, fused QKNorm+RoPE, day-0 GLM-5 support |
| vLLM | github.com/vllm-project/vllm | FlashMLA backends, FLASHMLA_SPARSE, DeepGEMM MoE, PagedAttention, MTP |
| SGLang | github.com/sgl-project/sglang | 5 MLA backends, NSA (3-5x Blackwell), moe_fused_gate, EAGLE/MTP |
