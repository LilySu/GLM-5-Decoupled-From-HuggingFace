# glm5-kernels-flashmla-deepgemm: GLM-5 model with H100 CUDA kernel acceleration.
#
# This package replaces pure PyTorch components with H100-optimized CUDA kernels:
#
# ┌──────────────────────────────────┬────────────────────────┬──────────────────────────┐
# │ Component                        │ File                   │ Kernel Source             │
# ├──────────────────────────────────┼────────────────────────┼──────────────────────────┤
# │ MLA Attention (prefill+decode)   │ mla_attention          │ FlashMLA (SM90 CUDA)     │
# │ DSA Lightning Indexer            │ dsa_indexer            │ DeepGEMM fp8_mqa_logits  │
# │ DSA Sparse Attention             │ dsa_sparse_attention   │ FlashMLA sparse kernels  │
# │ MoE Grouped GEMM (FP8)          │ moe_grouped_gemm       │ DeepGEMM FP8 grouped     │
# │ MoE Sigmoid Routing              │ moe_router             │ PyTorch (n_group=1)      │
# │ FP8 Quantization Utilities      │ fp8_utils              │ PyTorch utilities         │
# │ Paged KV Cache                  │ cache                  │ FlashMLA-compatible       │
# ├──────────────────────────────────┼────────────────────────┼──────────────────────────┤
# │ RMSNorm fwd+bwd                  │ unsloth_rms_layernorm  │ Triton kernel (kept)     │
# │ SwiGLU fwd+bwd                   │ unsloth_swiglu         │ Triton kernel (kept)     │
# │ Cross-Entropy Loss (chunked)     │ unsloth_cross_entropy  │ Triton kernel (kept)     │
# │ LoRA MLP/QKV/W                   │ unsloth_fast_lora      │ Triton kernel (kept)     │
# │ MoE Grouped GEMM (BF16 fallback)│ unsloth_moe/           │ Triton kernel (kept)     │
# │ Utilities                        │ unsloth_utils          │ Triton support (kept)    │
# ├──────────────────────────────────┼────────────────────────┼──────────────────────────┤
# │ Config                           │ config                 │ PyTorch (dict)           │
# │ Decoupled partial-dim RoPE       │ rope_partial           │ PyTorch (kept)           │
# │ Full model assembly              │ model                  │ PyTorch + kernel wrappers│
# └──────────────────────────────────┴────────────────────────┴──────────────────────────┘
#
# Dependencies:
#   pip install flash-mla   (build from source, requires CUDA 12.8+ and SM90)
#   pip install deep-gemm   (build from source, requires CUDA 12.8+ and SM90)
#
# Hardware requirement: SM90 (H100/H800) minimum

# --- Config ---
from .config import GLM_MOE_DSA_CONFIG, load_config_from_hf

# --- Kernel-accelerated components ---
from .mla_attention import MLAttention
from .dsa_indexer import DSAIndexer
from .dsa_sparse_attention import sparse_attention_forward
from .moe_router import sigmoid_topk_route
from .moe_grouped_gemm import moe_grouped_gemm_forward
from .fp8_utils import quantize_kv_flashmla, quantize_activations_deepgemm
from .cache import KVCache

# --- Kept from glm5-triton (Triton kernels) ---
from .unsloth_rms_layernorm import fast_rms_layernorm, Fast_RMS_Layernorm
from .unsloth_swiglu import swiglu_fg_kernel, swiglu_DWf_DW_dfg_kernel
from .unsloth_cross_entropy_loss import fast_cross_entropy_loss, Fast_CrossEntropyLoss
from .unsloth_fast_lora import LoRA_MLP, LoRA_QKV, LoRA_W
from .unsloth_utils import calculate_settings, torch_gpu_device, DEVICE_TYPE, MAX_FUSED_SIZE

# --- Kept from glm5-triton (PyTorch) ---
from .rope_partial import RotaryEmbedding, apply_rotary_pos_emb

# --- Full model ---
from .model import (
    make_causal_mask, FeedForward, TopkRouter, MoeExperts, MoE,
    DecoderLayer, GlmMoeDsaModel, GlmMoeDsaForCausalLM,
)
