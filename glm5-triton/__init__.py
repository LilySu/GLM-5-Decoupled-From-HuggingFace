# glm5-triton: Self-contained GLM-5 model with Triton-accelerated kernels.
#
# This package contains everything needed to instantiate and run the GLM-5
# model with zero external dependencies beyond torch and triton.
#
# Files prefixed with "unsloth_" contain Triton kernels extracted from
# unsloth (https://github.com/unslothai/unsloth) — optimized GPU ops.
#
# Files WITHOUT the prefix are pure-PyTorch implementations ported from
# the standalone GLM-5 reference (glm5-raw-decoupled-from-hf/model.py).
#
# ┌──────────────────────────────────┬────────────────────────┬───────────────────┐
# │ Component                        │ File                   │ Status            │
# ├──────────────────────────────────┼────────────────────────┼───────────────────┤
# │ RMSNorm fwd+bwd                  │ unsloth_rms_layernorm  │ Triton kernel     │
# │ SwiGLU fwd+bwd                   │ unsloth_swiglu         │ Triton kernel     │
# │ Cross-Entropy Loss (chunked)     │ unsloth_cross_entropy  │ Triton kernel     │
# │ LoRA MLP/QKV/W                   │ unsloth_fast_lora      │ Triton kernel     │
# │ MoE Grouped GEMM                 │ unsloth_moe/           │ Triton kernel     │
# │ Utilities                        │ unsloth_utils          │ Triton support    │
# ├──────────────────────────────────┼────────────────────────┼───────────────────┤
# │ Config                           │ config                 │ PyTorch (dict)    │
# │ KV Cache                         │ cache                  │ PyTorch           │
# │ Decoupled partial-dim RoPE       │ rope_partial           │ PyTorch           │
# │ DSA Lightning Indexer            │ dsa_indexer            │ PyTorch           │
# │ DSA Sparse Attention             │ dsa_sparse_attention   │ PyTorch           │
# │ MLA (Multi-Latent Attention)     │ mla_attention          │ PyTorch           │
# │ FeedForward / MoE / Router       │ model                  │ PyTorch           │
# │ DecoderLayer / Base / CausalLM   │ model                  │ PyTorch           │
# │ MTP (Multi-Token Prediction)     │ mtp                    │ Stub              │
# └──────────────────────────────────┴────────────────────────┴───────────────────┘

# --- Triton kernels (from unsloth) ---
from .unsloth_rms_layernorm import (
    fast_rms_layernorm,
    Fast_RMS_Layernorm,
)
from .unsloth_swiglu import (
    swiglu_fg_kernel,
    swiglu_DWf_DW_dfg_kernel,
)
from .unsloth_cross_entropy_loss import (
    fast_cross_entropy_loss,
    Fast_CrossEntropyLoss,
)
from .unsloth_fast_lora import (
    LoRA_MLP,
    LoRA_QKV,
    LoRA_W,
    apply_lora_mlp_swiglu,
    apply_lora_qkv,
    apply_lora_o,
    get_lora_parameters,
    get_lora_parameters_bias,
)
from .unsloth_utils import (
    calculate_settings,
    torch_gpu_device,
    DEVICE_TYPE,
    MAX_FUSED_SIZE,
    is_cdna,
    is_rdna,
    matmul_lora,
    fast_linear_forward,
)

# --- Config ---
from .config import GLM_MOE_DSA_CONFIG, load_config_from_hf

# --- KV Cache ---
from .cache import KVCache

# --- PyTorch reference implementations ---
from .rope_partial import (
    RotaryEmbedding,
    apply_rotary_pos_emb,
    apply_rope_to_query,
    apply_rope_to_compressed_kv_key,
    rotate_half,
)
from .dsa_indexer import DSAIndexer
from .dsa_sparse_attention import build_dsa_mask, eager_attention_forward
from .mla_attention import MLAttention

# --- Full model scaffolding ---
from .model import (
    make_causal_mask,
    FeedForward,
    TopkRouter,
    MoeExperts,
    MoE,
    DecoderLayer,
    GlmMoeDsaModel,
    GlmMoeDsaForCausalLM,
)
