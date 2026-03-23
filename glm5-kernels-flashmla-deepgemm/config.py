# GLM-5 model configuration.
# Default values match the GLM-5 744B (40B active) architecture from Table 10
# of the paper (arXiv 2602.15763v2).
#
# This is a plain dict — no HuggingFace PretrainedConfig dependency.

import json
import os

GLM_MOE_DSA_CONFIG = {
    # --- Vocabulary & embeddings ---
    "vocab_size": 154880,        # GLM-5 token vocabulary size
    "hidden_size": 6144,         # Transformer hidden dimension
    "tie_word_embeddings": False, # Whether lm_head shares embed_tokens weights

    # --- Layers ---
    "num_hidden_layers": 78,     # Total decoder layers (3 dense + 75 MoE)
    "intermediate_size": 12288,  # Dense MLP intermediate size (for layers 0-2)

    # --- Attention ---
    "num_attention_heads": 64,   # Number of query heads
    "num_key_value_heads": 64,   # MLA uses same count (no GQA — MLA compresses differently)
    "attention_bias": False,     # No bias on Q/K/V/O projections
    "attention_dropout": 0.0,    # Attention dropout rate

    # --- MLA (Multi-head Latent Attention) ---
    "q_lora_rank": 2048,         # Query compression bottleneck (6144 -> 2048 -> 64*256)
    "kv_lora_rank": 512,         # KV compression bottleneck (6144 -> 512 -> 64*(192+256))
    "qk_rope_head_dim": 64,     # RoPE-applied portion of each head
    "qk_nope_head_dim": 192,    # Non-RoPE portion of each head (Table 10: "QK Head Dim")
    "qk_head_dim": 256,         # Total = qk_nope_head_dim + qk_rope_head_dim
    "v_head_dim": 256,           # Value head dimension

    # --- MoE (Mixture of Experts) ---
    "n_routed_experts": 256,     # Total expert count
    "n_shared_experts": 1,       # Always-active shared expert count
    "num_experts_per_tok": 8,    # Top-k experts selected per token
    "moe_intermediate_size": 2048, # Per-expert intermediate dimension
    "routed_scaling_factor": 2.5,  # Scale factor applied after expert weighted sum
    "n_group": 1,                # Number of expert groups for group-based selection
    "topk_group": 1,             # Number of groups selected before within-group topk
    "norm_topk_prob": True,      # Normalize routing weights after selection

    # --- DSA (Dynamic Sparse Attention) indexer ---
    "index_topk": 2048,         # Max tokens the indexer selects per query position
    "index_head_dim": 128,      # Dimension per indexer scoring head
    "index_n_heads": 32,        # Number of lightweight indexer scoring heads

    # --- Activation & normalization ---
    "hidden_act": "silu",       # SwiGLU uses SiLU (sigmoid linear unit)
    "rms_norm_eps": 1e-5,       # RMSNorm epsilon

    # --- Positional encoding ---
    "max_position_embeddings": 202752,  # Maximum context length
    "rope_theta": 10000.0,              # RoPE base frequency

    # --- Weight initialization ---
    "initializer_range": 0.02,  # Std dev for weight initialization

    # --- Special tokens ---
    "pad_token_id": None,
    "bos_token_id": 0,
    "eos_token_id": 1,

    # --- Cache ---
    "use_cache": True,

    # --- MLP layer types: first 3 dense, remaining 75 sparse (MoE) ---
    "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
}


def load_config_from_hf(checkpoint_dir: str) -> dict:
    """Read config.json from a HuggingFace checkpoint directory.

    Maps HF config keys to our standalone config dict format.
    Falls back to GLM_MOE_DSA_CONFIG defaults for any missing keys.
    """
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path) as f:
        hf_config = json.load(f)

    config = dict(GLM_MOE_DSA_CONFIG)

    # Keys that match directly between HF and standalone configs
    direct_keys = [
        "vocab_size", "hidden_size", "num_hidden_layers", "intermediate_size",
        "num_attention_heads", "num_key_value_heads", "attention_bias", "attention_dropout",
        "q_lora_rank", "kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim",
        "n_routed_experts", "n_shared_experts", "num_experts_per_tok",
        "moe_intermediate_size", "routed_scaling_factor", "n_group", "topk_group",
        "norm_topk_prob", "hidden_act", "rms_norm_eps", "max_position_embeddings",
        "initializer_range", "pad_token_id", "bos_token_id", "eos_token_id",
        "tie_word_embeddings", "use_cache",
        "index_topk", "index_head_dim", "index_n_heads",
    ]
    for key in direct_keys:
        if key in hf_config:
            config[key] = hf_config[key]

    # Computed: total QK head dim = nope + rope
    config["qk_head_dim"] = config["qk_nope_head_dim"] + config["qk_rope_head_dim"]

    # Extract rope_theta from nested rope_parameters if present
    rope_params = hf_config.get("rope_parameters")
    if isinstance(rope_params, dict) and "rope_theta" in rope_params:
        config["rope_theta"] = rope_params["rope_theta"]
    elif "rope_theta" in hf_config:
        config["rope_theta"] = hf_config["rope_theta"]

    # MLP layer types: first N are dense, rest are sparse (MoE)
    if "mlp_layer_types" in hf_config and hf_config["mlp_layer_types"] is not None:
        config["mlp_layer_types"] = hf_config["mlp_layer_types"]
    else:
        n = config["num_hidden_layers"]
        config["mlp_layer_types"] = ["dense"] * min(3, n) + ["sparse"] * (n - 3)

    return config
