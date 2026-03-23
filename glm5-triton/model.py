# GLM-5 full model — assembles all components into a runnable model.
#
# This file contains the architectural scaffolding that connects the individual
# components (MLA attention, DSA indexer, RoPE, SwiGLU, MoE, RMSNorm) into
# a complete GLM-5 causal language model.
#
# Components imported from other files in this package:
#   mla_attention.py  -> MLAttention (attention + DSA)
#   rope_partial.py   -> RotaryEmbedding
#   cache.py          -> KVCache
#   config.py         -> GLM_MOE_DSA_CONFIG
#
# Paper ref: GLM-5 (arXiv 2602.15763v2)
# Architecture: 744B params, 40B active, 78 layers (3 dense + 75 MoE),
#               MLA attention, DSA sparse attention, 256 experts

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mla_attention import MLAttention, RMSNorm
from .rope_partial import RotaryEmbedding
from .cache import KVCache


# ---------------------------------------------------------------------------
# Causal mask
# ---------------------------------------------------------------------------

def make_causal_mask(seq_len, past_len, dtype, device):
    """Create a [1, 1, seq_len, total_len] causal attention mask.

    Each query position i can attend to key positions 0 .. past_len+i.
    Masked positions are set to dtype.min (approx -inf) so they become
    zero after softmax.

    Args:
        seq_len:  number of new query positions
        past_len: number of previously cached key positions
        dtype:    mask dtype (should match query dtype)
        device:   target device

    Returns:
        mask: [1, 1, seq_len, total_len] float tensor
    """
    total_len = past_len + seq_len
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(total_len, device=device).unsqueeze(0)
    # Position i can see positions 0..past_len+i
    causal = cols <= (rows + past_len)
    mask = torch.where(causal, 0.0, torch.finfo(dtype).min)
    return mask.to(dtype).unsqueeze(0).unsqueeze(0)


# ---------------------------------------------------------------------------
# FeedForward — SwiGLU MLP used in dense layers and as shared MoE expert
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    """SwiGLU feed-forward network: down_proj(SiLU(gate_proj(x)) * up_proj(x)).

    Used in two places:
      1. Dense MLP layers (layers 0-2): intermediate_size = 12288
      2. Shared expert in MoE layers:   intermediate_size = 2048 * n_shared_experts

    Args:
        cfg:               model config dict
        intermediate_size: override for the intermediate dimension (default: cfg["intermediate_size"])
    """

    def __init__(self, cfg, intermediate_size=None):
        super().__init__()
        self.hidden_size = cfg["hidden_size"]
        self.intermediate_size = cfg["intermediate_size"] if intermediate_size is None else intermediate_size
        # gate_proj and up_proj both project from hidden to intermediate
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # down_proj projects back from intermediate to hidden
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # SwiGLU: SiLU(gate) * up, then project down
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# TopkRouter — sigmoid-based MoE router with group selection
# ---------------------------------------------------------------------------

class TopkRouter(nn.Module):
    """MoE routing layer that scores each token against all experts.

    Uses a learned linear projection to produce per-expert logits.
    The e_score_correction_bias is an additive bias used during expert
    selection (but not during weight computation) to balance load.

    Args:
        cfg: model config dict
    """

    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg["hidden_size"]
        self.n_routed_experts = cfg["n_routed_experts"]

        # Router weight: projects hidden states to per-expert scores
        self.weight = nn.Parameter(torch.empty(self.n_routed_experts, self.hidden_size))
        # Additive bias for expert selection (not for weight computation)
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(self.n_routed_experts, dtype=torch.float32),
        )

    def forward(self, x):
        # Flatten to 2D and compute in float32 for numerical stability
        x = x.view(-1, self.hidden_size)
        return F.linear(x.float(), self.weight.float())


# ---------------------------------------------------------------------------
# MoeExperts — collection of expert weights stored as stacked 3D tensors
# ---------------------------------------------------------------------------

class MoeExperts(nn.Module):
    """All routed expert weights stored as contiguous 3D parameter tensors.

    Each expert is a SwiGLU MLP: SiLU(x @ gate) * (x @ up), then @ down.
    Weights are stored as:
      gate_up_proj: [num_experts, 2 * intermediate_dim, hidden_dim]
      down_proj:    [num_experts, hidden_dim, intermediate_dim]

    The forward method loops over active experts (those that received at
    least one token). This is the reference implementation — the Triton
    grouped GEMM kernels in unsloth_moe/ can replace this loop.

    Args:
        cfg: model config dict
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg["n_routed_experts"]       # 256
        self.hidden_dim = cfg["hidden_size"]              # 6144
        self.intermediate_dim = cfg["moe_intermediate_size"]  # 2048

        # Stacked expert weights — ready for grouped GEMM
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        """Route tokens through selected experts and accumulate weighted outputs.

        Args:
            hidden_states: [num_tokens, hidden_dim] — flattened input
            top_k_index:   [num_tokens, top_k] — expert indices per token
            top_k_weights: [num_tokens, top_k] — routing weights per token

        Returns:
            output: [num_tokens, hidden_dim] — weighted sum of expert outputs
        """
        final_hidden_states = torch.zeros_like(hidden_states)

        # Build a mask: which tokens go to which experts
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)  # [num_experts, top_k, num_tokens]
            # Find which experts actually received tokens
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        # Process only experts that received at least one token
        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            # Find which tokens were routed to this expert, and their routing positions
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])

            # Gather the tokens for this expert
            current_state = hidden_states[token_idx]

            # SwiGLU: split gate_up into gate and up, apply SiLU activation
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = F.silu(gate) * up

            # Down projection
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])

            # Apply routing weights
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]

            # Scatter-add back to the output
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


# ---------------------------------------------------------------------------
# MoE — full Mixture of Experts layer (router + experts + shared expert)
# ---------------------------------------------------------------------------

class MoE(nn.Module):
    """Mixture of Experts layer with sigmoid routing and group-based selection.

    Architecture:
      1. Router computes per-expert logits via linear projection
      2. Sigmoid activation (not softmax) on logits
      3. Group-based selection: pick top groups, then top-k experts within
      4. Route tokens through selected experts (weighted sum)
      5. Scale by routed_scaling_factor (2.5)
      6. Add shared expert output (always-active SwiGLU MLP)

    Args:
        cfg: model config dict
    """

    def __init__(self, cfg):
        super().__init__()
        self.experts = MoeExperts(cfg)
        self.gate = TopkRouter(cfg)
        # Shared expert is an always-active SwiGLU MLP
        self.shared_experts = FeedForward(
            cfg, intermediate_size=cfg["moe_intermediate_size"] * cfg["n_shared_experts"],
        )
        self.n_routed_experts = cfg["n_routed_experts"]   # 256
        self.n_group = cfg["n_group"]                      # 1
        self.topk_group = cfg["topk_group"]                # 1
        self.norm_topk_prob = cfg["norm_topk_prob"]        # True
        self.routed_scaling_factor = cfg["routed_scaling_factor"]  # 2.5
        self.top_k = cfg["num_experts_per_tok"]            # 8

    def route_tokens_to_experts(self, router_logits):
        """Select top-k experts per token using sigmoid + group-based routing.

        Args:
            router_logits: [num_tokens, n_routed_experts] — raw router output

        Returns:
            topk_indices: [num_tokens, top_k] — selected expert indices
            topk_weights: [num_tokens, top_k] — normalized & scaled routing weights
        """
        # Sigmoid activation (not softmax — key GLM-5 design choice)
        router_logits = router_logits.sigmoid()

        # Add correction bias for expert selection (load balancing)
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias

        # Group-based selection: score each group by its top-2 experts
        group_scores = (
            router_logits_for_choice
            .view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        # Select top groups
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)

        # Expand group mask to per-expert mask
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )

        # Zero out experts in non-selected groups, then pick top-k
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]

        # Gather actual routing weights (from sigmoid logits, not bias-adjusted)
        topk_weights = router_logits.gather(1, topk_indices)

        # Normalize routing weights so they sum to 1 per token
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)

        # Scale by routed_scaling_factor (2.5 for GLM-5)
        topk_weights = topk_weights * self.routed_scaling_factor

        return topk_indices, topk_weights

    def forward(self, hidden_states):
        """Run MoE: route through experts + add shared expert.

        Args:
            hidden_states: [B, S, hidden_size]

        Returns:
            output: [B, S, hidden_size]
        """
        residuals = hidden_states
        orig_shape = hidden_states.shape

        # Router scores all experts for each token
        router_logits = self.gate(hidden_states)

        # Select top-k experts and compute weights
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)

        # Flatten to [num_tokens, hidden_size] for expert dispatch
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)

        # Add shared expert output (always processes all tokens)
        hidden_states = hidden_states + self.shared_experts(residuals)

        return hidden_states


# ---------------------------------------------------------------------------
# DecoderLayer — one transformer block (pre-norm attention + pre-norm MLP)
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    """Single GLM-5 decoder layer.

    Structure:
      residual + MLA(input_layernorm(x)) -> residual + MLP(post_attn_layernorm(x))

    The MLP is either a dense FeedForward (layers 0-2) or a full MoE (layers 3-77),
    determined by cfg["mlp_layer_types"][layer_idx].

    Args:
        cfg:       model config dict
        layer_idx: index of this layer (0-77)
    """

    def __init__(self, cfg, layer_idx):
        super().__init__()
        # MLA attention with DSA indexer
        self.self_attn = MLAttention(cfg, layer_idx)

        # Dense MLP for first 3 layers, MoE for the remaining 75
        if cfg["mlp_layer_types"][layer_idx] == "sparse":
            self.mlp = MoE(cfg)
        else:
            self.mlp = FeedForward(cfg)

        # Pre-norm layers
        self.input_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])

        self.gradient_checkpointing = False

    def _forward(self, hidden_states, attention_mask, position_embeddings, past_key_values=None, **kwargs):
        # Pre-norm -> attention -> residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states, position_embeddings, attention_mask=attention_mask,
            past_key_values=past_key_values, **kwargs,
        )
        hidden_states = residual + hidden_states

        # Pre-norm -> MLP/MoE -> residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, past_key_values=None, **kwargs):
        # Use gradient checkpointing during training if enabled
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, hidden_states, attention_mask, position_embeddings,
                past_key_values, use_reentrant=False, **kwargs,
            )
        return self._forward(hidden_states, attention_mask, position_embeddings, past_key_values, **kwargs)


# ---------------------------------------------------------------------------
# GlmMoeDsaModel — base model (embedding + 78 layers + final norm)
# ---------------------------------------------------------------------------

class GlmMoeDsaModel(nn.Module):
    """GLM-5 base model: token embedding -> 78 decoder layers -> final RMSNorm.

    This is the core transformer stack without the language model head.

    Args:
        cfg: model config dict
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.padding_idx = cfg["pad_token_id"]
        self.vocab_size = cfg["vocab_size"]

        # Token embedding: vocab_size -> hidden_size
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"], self.padding_idx)

        # 78 decoder layers: first 3 dense, remaining 75 MoE
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg, layer_idx) for layer_idx in range(cfg["num_hidden_layers"])]
        )

        # Final layer norm
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])

        # Rotary embedding (computes cos/sin for the 64-dim rope portion)
        self.rotary_emb = RotaryEmbedding(cfg)

        self._init_weights()

    def _init_weights(self):
        """Initialize all weights with normal distribution."""
        std = self.cfg["initializer_range"]
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, TopkRouter):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                nn.init.zeros_(module.e_score_correction_bias)
            elif isinstance(module, MoeExperts):
                nn.init.normal_(module.gate_up_proj, mean=0.0, std=std)
                nn.init.normal_(module.down_proj, mean=0.0, std=std)

    def set_gradient_checkpointing(self, enable=True):
        """Enable or disable gradient checkpointing on all layers."""
        for layer in self.layers:
            layer.gradient_checkpointing = enable

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        """Forward pass through the base model.

        Args:
            input_ids:       [B, S] token indices
            attention_mask:  unused (causal mask is generated internally)
            position_ids:    [B, S] position indices (auto-generated if None)
            past_key_values: KVCache instance for autoregressive decoding
            inputs_embeds:   [B, S, hidden_size] pre-computed embeddings (alternative to input_ids)
            use_cache:       whether to allocate/use KV cache

        Returns:
            hidden_states:   [B, S, hidden_size]
            past_key_values: KVCache instance (or None)
        """
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # Embed tokens
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # Allocate KV cache if needed
        if use_cache and past_key_values is None:
            past_key_values = KVCache(self.cfg["num_hidden_layers"])

        # Auto-generate position IDs based on past cache length
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # Build causal attention mask
        causal_mask = make_causal_mask(
            seq_len=inputs_embeds.shape[1],
            past_len=past_key_values.get_seq_length() if past_key_values is not None else 0,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        # Compute rotary embeddings once for all layers
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # Run through all 78 decoder layers
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values, **kwargs,
            )

        # Final layer norm
        hidden_states = self.norm(hidden_states)

        return hidden_states, past_key_values


# ---------------------------------------------------------------------------
# GlmMoeDsaForCausalLM — causal LM wrapper (base model + lm_head + loss)
# ---------------------------------------------------------------------------

class GlmMoeDsaForCausalLM(nn.Module):
    """GLM-5 causal language model: base model + linear LM head.

    Produces logits over the vocabulary for next-token prediction.
    Optionally computes cross-entropy loss when labels are provided.

    Args:
        cfg: model config dict
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = GlmMoeDsaModel(cfg)
        self.vocab_size = cfg["vocab_size"]

        # LM head: hidden_size -> vocab_size
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

        # Optionally share weights between embedding and LM head
        if cfg.get("tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, **kwargs):
        """Forward pass: embeddings -> transformer -> logits (-> loss).

        Args:
            input_ids:       [B, S] token indices
            labels:          [B, S] target token indices for loss (-100 = ignore)
            (other args forwarded to GlmMoeDsaModel)

        Returns:
            loss:            scalar cross-entropy loss (None if labels not provided)
            logits:          [B, S, vocab_size]
            past_key_values: KVCache instance
        """
        # Run the base model
        hidden_states, past_key_values = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs,
        )

        # Project to vocabulary logits
        logits = self.lm_head(hidden_states)

        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), labels.view(-1),
            )

        return loss, logits, past_key_values
