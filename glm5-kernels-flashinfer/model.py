# GLM-5 full model — FlashInfer kernel-accelerated version.
#
# Difference from FlashMLA path:
#   MLA Attention:   FlashInfer BatchMLAPagedAttentionWrapper (FA3) + trtllm-gen sparse
#   DSA Indexer:     DeepGEMM fp8_mqa_logits (same)
#   MoE GEMM:       DeepGEMM FP8 grouped GEMM (same)
#   MoE Router:     Pure PyTorch sigmoid routing (same)
#   RMSNorm/SwiGLU:  Unsloth Triton (same)

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mla_attention import MLAttention, RMSNorm
from .rope_partial import RotaryEmbedding
from .cache import KVCache
from .moe_router import TopkRouter, sigmoid_topk_route
from .moe_grouped_gemm import moe_grouped_gemm_forward


def make_causal_mask(seq_len, past_len, dtype, device):
    total_len = past_len + seq_len
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(total_len, device=device).unsqueeze(0)
    causal = cols <= (rows + past_len)
    mask = torch.where(causal, 0.0, torch.finfo(dtype).min)
    return mask.to(dtype).unsqueeze(0).unsqueeze(0)


class FeedForward(nn.Module):
    def __init__(self, cfg, intermediate_size=None):
        super().__init__()
        self.hidden_size = cfg["hidden_size"]
        self.intermediate_size = cfg["intermediate_size"] if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MoeExperts(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg["n_routed_experts"]
        self.hidden_dim = cfg["hidden_size"]
        self.intermediate_dim = cfg["moe_intermediate_size"]
        self.gate_up_proj = nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        self.down_proj = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return moe_grouped_gemm_forward(
            hidden_states, self.gate_up_proj, self.down_proj,
            top_k_index, top_k_weights, self.num_experts,
        )


class MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.experts = MoeExperts(cfg)
        self.gate = TopkRouter(cfg)
        self.shared_experts = FeedForward(
            cfg, intermediate_size=cfg["moe_intermediate_size"] * cfg["n_shared_experts"],
        )
        self.n_routed_experts = cfg["n_routed_experts"]
        self.n_group = cfg["n_group"]
        self.topk_group = cfg["topk_group"]
        self.norm_topk_prob = cfg["norm_topk_prob"]
        self.routed_scaling_factor = cfg["routed_scaling_factor"]
        self.top_k = cfg["num_experts_per_tok"]

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = sigmoid_topk_route(
            router_logits, self.gate.e_score_correction_bias,
            top_k=self.top_k, n_group=self.n_group, topk_group=self.topk_group,
            norm_topk_prob=self.norm_topk_prob, routed_scaling_factor=self.routed_scaling_factor,
        )
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


class DecoderLayer(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.self_attn = MLAttention(cfg, layer_idx)
        if cfg["mlp_layer_types"][layer_idx] == "sparse":
            self.mlp = MoE(cfg)
        else:
            self.mlp = FeedForward(cfg)
        self.input_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.gradient_checkpointing = False

    def _forward(self, hidden_states, attention_mask, position_embeddings, past_key_values=None, **kwargs):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states, position_embeddings, attention_mask=attention_mask,
            past_key_values=past_key_values, **kwargs,
        )
        hidden_states = residual + hidden_states
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, past_key_values=None, **kwargs):
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, hidden_states, attention_mask, position_embeddings,
                past_key_values, use_reentrant=False, **kwargs,
            )
        return self._forward(hidden_states, attention_mask, position_embeddings, past_key_values, **kwargs)


class GlmMoeDsaModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.padding_idx = cfg["pad_token_id"]
        self.vocab_size = cfg["vocab_size"]
        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"], self.padding_idx)
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg, i) for i in range(cfg["num_hidden_layers"])]
        )
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.rotary_emb = RotaryEmbedding(cfg)
        self._init_weights()

    def _init_weights(self):
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
        for layer in self.layers:
            layer.gradient_checkpointing = enable

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        if use_cache and past_key_values is None:
            past_key_values = KVCache(self.cfg["num_hidden_layers"])
        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)
        causal_mask = make_causal_mask(
            inputs_embeds.shape[1],
            past_key_values.get_seq_length() if past_key_values is not None else 0,
            inputs_embeds.dtype, inputs_embeds.device,
        )
        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)
        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values, **kwargs,
            )
        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values


class GlmMoeDsaForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = GlmMoeDsaModel(cfg)
        self.vocab_size = cfg["vocab_size"]
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)
        if cfg.get("tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, **kwargs):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs,
        )
        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), labels.view(-1))
        return loss, logits, past_key_values
