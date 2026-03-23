# glm5-kernels-flashinfer: GLM-5 model with FlashInfer + DeepGEMM kernel acceleration.
#
# Key differences from glm5-kernels-flashmla-deepgemm:
#   MLA Attention:  FlashInfer BatchMLAPagedAttentionWrapper (FA3) + trtllm-gen sparse
#   FP8 KV Cache:   FlashInfer native [num_pages, page_size, 576] contiguous FP8
#   Sparse Decode:  trtllm_batch_decode_with_kv_cache_mla(sparse_mla_top_k=2048)
#   Patches:        Monkey-patch for qk_nope_head_dim=192 validation bypass
#   CUDA Graphs:    Native support via use_cuda_graph=True in BatchMLAPagedAttentionWrapper
#
# Same as FlashMLA path:
#   DSA Indexer:    DeepGEMM fp8_mqa_logits
#   MoE GEMM:      DeepGEMM FP8 grouped GEMM
#   MoE Router:    Pure PyTorch (n_group=1)
#   RMSNorm/SwiGLU: Unsloth Triton kernels
#   RoPE:          Pure PyTorch
#
# Dependencies:
#   pip install flashinfer  (CUDA 12.0+)
#   pip install deep-gemm   (build from source, CUDA 12.8+, SM90)

from .config import GLM_MOE_DSA_CONFIG, load_config_from_hf
from .mla_attention import MLAttention
from .dsa_indexer import DSAIndexer
from .dsa_sparse_attention import build_dsa_mask, eager_attention_forward
from .moe_router import sigmoid_topk_route
from .moe_grouped_gemm import moe_grouped_gemm_forward
from .fp8_utils import quantize_kv_flashinfer, quantize_activations_deepgemm
from .cache import KVCache
from .rope_partial import RotaryEmbedding, apply_rotary_pos_emb
from .model import (
    make_causal_mask, FeedForward, TopkRouter, MoeExperts, MoE,
    DecoderLayer, GlmMoeDsaModel, GlmMoeDsaForCausalLM,
)
