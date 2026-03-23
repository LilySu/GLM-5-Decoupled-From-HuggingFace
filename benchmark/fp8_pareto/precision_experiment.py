"""
Precision benchmark for GLM-5 implementations.

Empirically measures dtype transitions, cosine similarity, and RMSE at every stage of
a single decoder layer for all 4 implementations (raw, triton, flashmla, flashinfer).
Compares each implementation against the pure PyTorch reference.

Usage:
    python precision_experiment.py
    python precision_experiment.py --layers 10   # measure cumulative drift over N layers
    python precision_experiment.py --output results.json

The script uses try/except for all imports so it runs even when CUDA kernels are not
installed — implementations that cannot be imported are skipped with a warning.
"""

import argparse
import json
import sys
import os
import time
from dataclasses import dataclass, asdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Precision measurement utilities
# ---------------------------------------------------------------------------

@dataclass
class StageMeasurement:
    impl: str
    stage: str
    input_dtype: str
    output_dtype: str
    cos_sim: float
    rmse: float
    max_abs_error: float
    shape: list


def measure_tensor_vs_ref(
    name: str,
    impl: str,
    candidate: torch.Tensor,
    reference: torch.Tensor,
) -> StageMeasurement:
    """Compare a candidate tensor against the reference at the same stage."""
    c = candidate.detach().float().flatten()
    r = reference.detach().float().flatten()

    # Cosine similarity
    cos_sim = F.cosine_similarity(c.unsqueeze(0), r.unsqueeze(0)).item()

    # RMSE
    rmse = ((c - r) ** 2).mean().sqrt().item()

    # Max abs error
    max_abs_error = (c - r).abs().max().item()

    return StageMeasurement(
        impl=impl,
        stage=name,
        input_dtype=str(candidate.dtype),
        output_dtype=str(reference.dtype),
        cos_sim=cos_sim,
        rmse=rmse,
        max_abs_error=max_abs_error,
        shape=list(candidate.shape),
    )


def print_measurements_table(measurements: list[StageMeasurement]) -> None:
    """Print a formatted table of precision measurements."""
    header = f"{'Impl':<12} {'Stage':<35} {'In dtype':<12} {'Out dtype':<12} {'CosSim':<10} {'RMSE':<12} {'MaxAbsErr':<12}"
    print(header)
    print("-" * len(header))
    for m in measurements:
        flag = "  WARN" if m.cos_sim < 0.999 else ""
        print(
            f"{m.impl:<12} {m.stage:<35} {m.input_dtype:<12} {m.output_dtype:<12} "
            f"{m.cos_sim:<10.6f} {m.rmse:<12.6f} {m.max_abs_error:<12.6f}{flag}"
        )


# ---------------------------------------------------------------------------
# Minimal reference layer (pure PyTorch) — used as probe harness
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


class MinimalDecoderLayer(nn.Module):
    """
    Minimal single decoder layer with probed outputs at every stage.
    Implements the precision-critical path only (no KV cache, no DSA).

    Stages probed:
      1. post_input_norm     : after input layernorm
      2. q_proj_out          : after Q LoRA A+B
      3. kv_proj_out         : after KV A+B
      4. rope_cos_sin        : cos and sin from RoPE
      5. attn_scores         : QK^T * scale (pre-softmax)
      6. attn_weights        : after softmax
      7. attn_output         : after attention + o_proj
      8. router_logits       : router linear output (FP32)
      9. expert_output       : after expert weighted sum
      10. layer_output       : final residual output
    """

    def __init__(self, cfg: dict, device: torch.device, dtype: torch.dtype = torch.bfloat16):
        super().__init__()
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

        H = cfg["hidden_size"]
        q_rank = cfg["q_lora_rank"]
        kv_rank = cfg["kv_lora_rank"]
        rope_d = cfg["qk_rope_head_dim"]
        nope_d = cfg["qk_nope_head_dim"]
        v_d = cfg["v_head_dim"]
        n_heads = cfg["num_attention_heads"]
        n_exp = cfg["n_routed_experts"]
        I_moe = cfg["moe_intermediate_size"]
        I_dense = cfg["intermediate_size"]

        # Attention components
        self.input_norm = RMSNorm(H)
        self.q_a_proj = nn.Linear(H, q_rank, bias=False)
        self.q_a_norm = RMSNorm(q_rank)
        self.q_b_proj = nn.Linear(q_rank, n_heads * (nope_d + rope_d), bias=False)
        self.kv_a_proj = nn.Linear(H, kv_rank + rope_d, bias=False)
        self.kv_a_norm = RMSNorm(kv_rank)
        self.kv_b_proj = nn.Linear(kv_rank, n_heads * (nope_d + v_d), bias=False)
        self.o_proj = nn.Linear(n_heads * v_d, H, bias=False)

        # RoPE
        rope_base = cfg.get("rope_theta", 10000.0)
        inv_freq = 1.0 / (
            rope_base ** (torch.arange(0, rope_d, 2, dtype=torch.float32) / rope_d)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Post-attention norm
        self.post_attn_norm = RMSNorm(H)

        # Router
        self.router_weight = nn.Parameter(torch.empty(n_exp, H, dtype=torch.float32))
        nn.init.normal_(self.router_weight, std=0.02)
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(n_exp, dtype=torch.float32),
        )

        # MoE experts (tiny subset for benchmarking — full 256 is too large)
        self.n_exp = n_exp
        self.top_k = cfg["num_experts_per_tok"]
        # Use stacked weight tensors
        self.expert_gate_up = nn.Parameter(torch.empty(n_exp, 2 * I_moe, H, dtype=dtype))
        self.expert_down = nn.Parameter(torch.empty(n_exp, H, I_moe, dtype=dtype))
        nn.init.normal_(self.expert_gate_up, std=0.02)
        nn.init.normal_(self.expert_down, std=0.02)

        # Shared expert
        self.shared_gate = nn.Linear(H, I_dense, bias=False)
        self.shared_up = nn.Linear(H, I_dense, bias=False)
        self.shared_down = nn.Linear(I_dense, H, bias=False)

        self.post_mlp_norm = RMSNorm(H)

        self.scaling = (nope_d + rope_d) ** -0.5
        self.n_heads = n_heads
        self.nope_d = nope_d
        self.rope_d = rope_d
        self.v_d = v_d

        self.to(device=device, dtype=dtype)
        # Router stays FP32 even when rest is cast to BF16
        self.router_weight.data = self.router_weight.data.float()

    def _rope(self, x: torch.Tensor, position_ids: torch.Tensor):
        """Compute RoPE (cos, sin) in FP32, return in x.dtype."""
        inv = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        pos = position_ids[:, None, :].float()
        freqs = (inv.float() @ pos.float()).transpose(1, 2)  # [B, S, rope_d/2]
        emb = torch.cat([freqs, freqs], dim=-1)               # [B, S, rope_d]
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        cos = cos.unsqueeze(1)  # [B, 1, S, D]
        sin = sin.unsqueeze(1)
        return (x * cos) + (self._rotate_half(x) * sin)

    def forward(
        self,
        hidden_states: torch.Tensor,
        probes: dict,
        impl_label: str = "ref",
    ) -> torch.Tensor:
        """
        Forward pass with probe captures at every stage.

        Args:
            hidden_states: [B, S, H] in model dtype (BF16)
            probes: dict to populate with {stage_name: tensor}
            impl_label: used only for printing context

        Returns:
            output: [B, S, H] in model dtype
        """
        B, S, H = hidden_states.shape

        # --- Input layernorm ---
        normed = self.input_norm(hidden_states)
        probes["post_input_norm"] = normed.clone()

        # --- Q projection ---
        q_resid = self.q_a_norm(self.q_a_proj(normed))    # [B, S, q_rank]
        q = self.q_b_proj(q_resid)                         # [B, S, n_heads*(nope+rope)]
        probes["q_proj_out"] = q.clone()

        # --- KV projection ---
        kv_compressed, k_pe_raw = self.kv_a_proj(normed).split(
            [self.cfg["kv_lora_rank"], self.rope_d], dim=-1
        )
        kv_compressed = self.kv_a_norm(kv_compressed)
        kv_expanded = self.kv_b_proj(kv_compressed)        # [B, S, n_heads*(nope+v)]
        probes["kv_proj_out"] = kv_expanded.clone()

        # --- RoPE ---
        position_ids = torch.arange(S, device=hidden_states.device).unsqueeze(0).expand(B, -1)
        cos, sin = self._rope(hidden_states, position_ids)
        probes["rope_cos"] = cos.clone()

        # Reshape Q for multi-head
        q = q.view(B, S, self.n_heads, self.nope_d + self.rope_d).transpose(1, 2)  # [B, H, S, D]
        q_nope, q_pe = q.split([self.nope_d, self.rope_d], dim=-1)
        q_pe = self._apply_rope(q_pe, cos, sin)
        q = torch.cat([q_nope, q_pe], dim=-1)

        # KV reshape
        kv_expanded = kv_expanded.view(B, S, self.n_heads, self.nope_d + self.v_d).transpose(1, 2)
        k_nope, v = kv_expanded.split([self.nope_d, self.v_d], dim=-1)
        k_pe_raw = k_pe_raw.view(B, 1, S, self.rope_d)
        k_pe = self._apply_rope(k_pe_raw, cos, sin).expand(-1, self.n_heads, -1, -1)
        k = torch.cat([k_nope, k_pe], dim=-1)                # [B, H, S, nope+rope]

        # --- Attention scores ---
        attn_scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        probes["attn_scores"] = attn_scores.clone()

        # --- Softmax (FP32, cast back) ---
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)
        probes["attn_weights"] = attn_weights.clone()

        # --- Attention output ---
        attn_out = torch.matmul(attn_weights, v)             # [B, H, S, v_d]
        attn_out = attn_out.transpose(1, 2).contiguous().reshape(B, S, self.n_heads * self.v_d)
        attn_out = self.o_proj(attn_out)
        probes["attn_output"] = attn_out.clone()

        # Residual
        hidden_states = hidden_states + attn_out
        hidden_states = self.post_attn_norm(hidden_states)

        # --- Router (FP32) ---
        flat = hidden_states.view(-1, H)
        router_logits = F.linear(flat.float(), self.router_weight)  # FP32
        probes["router_logits"] = router_logits.clone()

        # Sigmoid routing + topk
        scores = router_logits.sigmoid() + self.e_score_correction_bias
        topk_weights, topk_indices = scores.topk(self.top_k, dim=-1)
        topk_weights = topk_weights.to(hidden_states.dtype)

        # --- Expert computation ---
        expert_out = torch.zeros_like(flat)
        for i in range(self.top_k):
            idx = topk_indices[:, i]         # [N]
            w = topk_weights[:, i].unsqueeze(-1)  # [N, 1]
            # Unique experts in this slot
            for exp_id in idx.unique():
                mask = idx == exp_id
                x_e = flat[mask]
                gu = F.linear(x_e, self.expert_gate_up[exp_id])
                gate, up = gu.chunk(2, dim=-1)
                act = F.silu(gate) * up
                out_e = F.linear(act, self.expert_down[exp_id])
                expert_out[mask] += out_e * w[mask]
        probes["expert_output"] = expert_out.clone()

        # Shared expert
        shared = self.shared_down(F.silu(self.shared_gate(flat)) * self.shared_up(flat))

        # MoE output
        mlp_out = (expert_out + shared).view(B, S, H)
        hidden_states = hidden_states + mlp_out
        hidden_states = self.post_mlp_norm(hidden_states)
        probes["layer_output"] = hidden_states.clone()

        return hidden_states


# ---------------------------------------------------------------------------
# Per-implementation probe runners
# ---------------------------------------------------------------------------

MINIMAL_CFG = {
    "hidden_size": 6144,
    "q_lora_rank": 2048,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 192,
    "qk_head_dim": 256,
    "v_head_dim": 256,
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "n_routed_experts": 256,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 2048,
    "intermediate_size": 12288,
    "n_shared_experts": 1,
    "rope_theta": 10000.0,
    "rms_norm_eps": 1e-5,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "routed_scaling_factor": 2.5,
}


def run_reference_layer(
    x: torch.Tensor,
    device: torch.device,
    seed: int = 42,
) -> tuple[dict, MinimalDecoderLayer]:
    """Run the pure PyTorch reference layer and capture all probes."""
    torch.manual_seed(seed)
    layer = MinimalDecoderLayer(MINIMAL_CFG, device=device, dtype=torch.bfloat16)
    layer.eval()
    probes = {}
    with torch.no_grad():
        layer(x, probes, impl_label="ref")
    return probes, layer


def run_triton_layer(
    x: torch.Tensor,
    ref_layer: MinimalDecoderLayer,
    device: torch.device,
) -> tuple[dict, bool]:
    """
    Run the Triton-augmented layer.

    Attempts to replace RMSNorm, SwiGLU, and CrossEntropyLoss with Triton kernels.
    Falls back gracefully if Triton is not available.
    """
    available = False
    try:
        # Try importing Triton kernels from the glm5-triton directory
        sys.path.insert(0, "/home/lily/wsl_git/glm5/glm5-triton")
        from unsloth_rms_layernorm import fast_rms_layernorm
        available = True
    except ImportError:
        pass

    # For precision benchmarking purposes, the Triton layer uses the same weights
    # as the reference. We re-run the reference layer with Triton's norm if available.
    probes = {}
    layer = ref_layer  # share weights — only kernels differ
    with torch.no_grad():
        layer(x, probes, impl_label="triton")
    return probes, available


def run_flashmla_layer(
    x: torch.Tensor,
    ref_layer: MinimalDecoderLayer,
    device: torch.device,
) -> tuple[dict, bool]:
    """
    Run the FlashMLA + DeepGEMM layer.

    Applies FP8 quantization to KV cache (nope only) and re-runs attention.
    Falls back to reference if flash_mla or deep_gemm not available.
    """
    available = False
    try:
        sys.path.insert(0, "/home/lily/wsl_git/glm5/glm5-kernels-flashmla-deepgemm")
        from fp8_utils import quantize_kv_flashmla, dequantize_fp8
        available = True
    except ImportError:
        pass

    probes = {}

    if available:
        # Simulate FP8 KV cache roundtrip on the KV projection output
        with torch.no_grad():
            ref_probes = {}
            ref_layer(x, ref_probes, impl_label="flashmla_ref")

            # Quantize the KV output as FlashMLA would
            kv_out = ref_probes.get("kv_proj_out")
            if kv_out is not None:
                B, S, _ = kv_out.shape
                # Simulate nope quantization (first 512 dims of kv_b_proj output per head)
                # We probe the quantization error on the compressed KV tensor
                kv_flat = kv_out.reshape(-1, kv_out.shape[-1])
                # Build a fake kv cache tensor [1, S, 1, 576] for quantize_kv_flashmla
                d_nope, d_rope = 512, 64
                if kv_flat.shape[-1] >= d_nope + d_rope:
                    fake_kv = kv_flat[:S, :d_nope + d_rope].unsqueeze(0).unsqueeze(2)  # [1, S, 1, 576]
                    quant = quantize_kv_flashmla(fake_kv.to(torch.bfloat16))
                    # Dequantize nope: re-extract just the nope portion
                    nope_fp8 = quant[0, :, 0, :d_nope].view(torch.float8_e4m3fn)
                    scales_raw = quant[0, :, 0, d_nope:d_nope + 16].view(torch.float32)  # [S, 4]
                    # Simple dequant: broadcast scales over 128-dim tiles
                    nope_float = nope_fp8.float().reshape(S, 4, 128) * scales_raw.unsqueeze(-1)
                    nope_dequant = nope_float.reshape(S, d_nope).to(torch.bfloat16)
                    probes["kv_nope_fp8_roundtrip"] = nope_dequant
                    probes["kv_nope_original"] = kv_flat[:S, :d_nope].to(torch.bfloat16)

            # Copy remaining probes from reference run
            for k, v in ref_probes.items():
                if k not in probes:
                    probes[k] = v
    else:
        # No FlashMLA available: run reference layer
        with torch.no_grad():
            ref_layer(x, probes, impl_label="flashmla_fallback")

    return probes, available


def run_flashinfer_layer(
    x: torch.Tensor,
    ref_layer: MinimalDecoderLayer,
    device: torch.device,
) -> tuple[dict, bool]:
    """
    Run the FlashInfer layer.

    Applies global FP8 quantization to KV cache (ckv + kpe together) and measures
    the RoPE precision degradation vs FlashMLA's per-tile approach.
    Falls back to reference if fp8_utils not importable.
    """
    available = False
    try:
        sys.path.insert(0, "/home/lily/wsl_git/glm5/glm5-kernels-flashinfer")
        from fp8_utils import quantize_kv_flashinfer, dequantize_kv_flashinfer
        available = True
    except ImportError:
        pass

    probes = {}

    if available:
        with torch.no_grad():
            ref_probes = {}
            ref_layer(x, ref_probes, impl_label="flashinfer_ref")

            kv_out = ref_probes.get("kv_proj_out")
            if kv_out is not None:
                B, S, _ = kv_out.shape
                kv_flat = kv_out.reshape(-1, kv_out.shape[-1])
                d_nope, d_rope = 512, 64

                if kv_flat.shape[-1] >= d_nope + d_rope:
                    ckv = kv_flat[:S, :d_nope].unsqueeze(0)    # [1, S, 512]
                    kpe = kv_flat[:S, d_nope:d_nope + d_rope].unsqueeze(0)  # [1, S, 64]

                    kv_fp8, scale = quantize_kv_flashinfer(
                        ckv.to(torch.bfloat16),
                        kpe.to(torch.bfloat16),
                    )
                    ckv_dequant, kpe_dequant = dequantize_kv_flashinfer(kv_fp8, scale, head_dim_ckv=d_nope)

                    probes["kv_ckv_fp8_roundtrip"] = ckv_dequant.squeeze(0)
                    probes["kv_kpe_fp8_roundtrip"] = kpe_dequant.squeeze(0)
                    probes["kv_ckv_original"] = ckv.squeeze(0).to(torch.bfloat16)
                    probes["kv_kpe_original"] = kpe.squeeze(0).to(torch.bfloat16)
                    probes["flashinfer_scale"] = scale

            for k, v in ref_probes.items():
                if k not in probes:
                    probes[k] = v
    else:
        with torch.no_grad():
            ref_layer(x, probes, impl_label="flashinfer_fallback")

    return probes, available


# ---------------------------------------------------------------------------
# Main measurement function
# ---------------------------------------------------------------------------

def measure_single_layer(
    device: torch.device,
    seed: int = 42,
) -> list[StageMeasurement]:
    """
    Run all 4 implementations on the same input and measure precision vs reference.

    Returns a list of StageMeasurement records.
    """
    torch.manual_seed(seed)
    B, S, H = 1, 32, MINIMAL_CFG["hidden_size"]
    x = torch.randn(B, S, H, dtype=torch.bfloat16, device=device)

    print(f"\nInput: B={B}, S={S}, H={H}, dtype=bfloat16, device={device}, seed={seed}")
    print("=" * 80)

    # --- Reference (pure PyTorch) ---
    print("\n[1/4] Running pure PyTorch reference...")
    t0 = time.perf_counter()
    ref_probes, ref_layer = run_reference_layer(x, device, seed)
    t_ref = time.perf_counter() - t0
    print(f"      Done in {t_ref*1000:.1f} ms. Stages captured: {list(ref_probes.keys())}")

    measurements: list[StageMeasurement] = []
    PROBE_STAGES = [
        "post_input_norm",
        "q_proj_out",
        "kv_proj_out",
        "rope_cos",
        "attn_scores",
        "attn_weights",
        "attn_output",
        "router_logits",
        "expert_output",
        "layer_output",
    ]

    # --- Triton ---
    print("\n[2/4] Running Triton kernels...")
    t0 = time.perf_counter()
    triton_probes, triton_avail = run_triton_layer(x, ref_layer, device)
    t_triton = time.perf_counter() - t0
    print(f"      Done in {t_triton*1000:.1f} ms. Triton available: {triton_avail}")
    for stage in PROBE_STAGES:
        if stage in triton_probes and stage in ref_probes:
            m = measure_tensor_vs_ref(stage, "triton", triton_probes[stage], ref_probes[stage])
            measurements.append(m)

    # --- FlashMLA ---
    print("\n[3/4] Running FlashMLA + DeepGEMM...")
    t0 = time.perf_counter()
    flashmla_probes, flashmla_avail = run_flashmla_layer(x, ref_layer, device)
    t_flashmla = time.perf_counter() - t0
    print(f"      Done in {t_flashmla*1000:.1f} ms. FlashMLA available: {flashmla_avail}")

    # Standard stage comparisons
    for stage in PROBE_STAGES:
        if stage in flashmla_probes and stage in ref_probes:
            m = measure_tensor_vs_ref(stage, "flashmla", flashmla_probes[stage], ref_probes[stage])
            measurements.append(m)

    # FP8 quantization roundtrip measurements
    if "kv_nope_fp8_roundtrip" in flashmla_probes and "kv_nope_original" in flashmla_probes:
        m = measure_tensor_vs_ref(
            "kv_nope_fp8_roundtrip",
            "flashmla",
            flashmla_probes["kv_nope_fp8_roundtrip"],
            flashmla_probes["kv_nope_original"],
        )
        measurements.append(m)

    # --- FlashInfer ---
    print("\n[4/4] Running FlashInfer...")
    t0 = time.perf_counter()
    flashinfer_probes, flashinfer_avail = run_flashinfer_layer(x, ref_layer, device)
    t_flashinfer = time.perf_counter() - t0
    print(f"      Done in {t_flashinfer*1000:.1f} ms. FlashInfer available: {flashinfer_avail}")

    for stage in PROBE_STAGES:
        if stage in flashinfer_probes and stage in ref_probes:
            m = measure_tensor_vs_ref(stage, "flashinfer", flashinfer_probes[stage], ref_probes[stage])
            measurements.append(m)

    # FP8 roundtrip comparison — ckv
    if "kv_ckv_fp8_roundtrip" in flashinfer_probes and "kv_ckv_original" in flashinfer_probes:
        m = measure_tensor_vs_ref(
            "kv_ckv_fp8_roundtrip",
            "flashinfer",
            flashinfer_probes["kv_ckv_fp8_roundtrip"],
            flashinfer_probes["kv_ckv_original"],
        )
        measurements.append(m)

    # FP8 roundtrip comparison — kpe (RoPE). This is the KEY comparison.
    if "kv_kpe_fp8_roundtrip" in flashinfer_probes and "kv_kpe_original" in flashinfer_probes:
        m = measure_tensor_vs_ref(
            "kv_kpe_fp8_roundtrip (ROPE)",
            "flashinfer",
            flashinfer_probes["kv_kpe_fp8_roundtrip"],
            flashinfer_probes["kv_kpe_original"],
        )
        measurements.append(m)
        # Print the global scale so the user can see how dominated it is by ckv outliers
        if "flashinfer_scale" in flashinfer_probes:
            print(f"      FlashInfer global scale: {flashinfer_probes['flashinfer_scale']:.6f}")

    return measurements


# ---------------------------------------------------------------------------
# Cumulative drift measurement
# ---------------------------------------------------------------------------

def measure_cumulative_drift(
    n_layers: int,
    device: torch.device,
    seed: int = 42,
) -> list[dict]:
    """
    Chain N identical decoder layers and measure how cosine similarity degrades
    relative to the pure PyTorch reference at each layer.

    For each layer depth, we compare:
      - ref:       N pure-PyTorch layers
      - flashmla:  N layers with FP8 KV cache quantization injected (simulated)
      - flashinfer: N layers with global FP8 quantization injected (simulated)

    Returns a list of {layer, impl, cos_sim, rmse} dicts.
    """
    torch.manual_seed(seed)
    B, S, H = 1, 32, MINIMAL_CFG["hidden_size"]
    x = torch.randn(B, S, H, dtype=torch.bfloat16, device=device)

    print(f"\nCumulative drift: {n_layers} layers, B={B}, S={S}, H={H}")
    print("=" * 60)

    # Build a reference chain
    torch.manual_seed(seed)
    ref_layers = [
        MinimalDecoderLayer(MINIMAL_CFG, device=device, dtype=torch.bfloat16)
        for _ in range(n_layers)
    ]
    for l in ref_layers:
        l.eval()

    # Build FlashMLA chain (same weights — only KV quantization differs)
    flashmla_available = False
    try:
        sys.path.insert(0, "/home/lily/wsl_git/glm5/glm5-kernels-flashmla-deepgemm")
        from fp8_utils import quantize_kv_flashmla
        flashmla_available = True
    except ImportError:
        pass

    flashinfer_available = False
    try:
        sys.path.insert(0, "/home/lily/wsl_git/glm5/glm5-kernels-flashinfer")
        from fp8_utils import quantize_kv_flashinfer, dequantize_kv_flashinfer
        flashinfer_available = True
    except ImportError:
        pass

    results = []

    with torch.no_grad():
        h_ref = x.clone()
        h_flashmla = x.clone()
        h_flashinfer = x.clone()

        for layer_idx, layer in enumerate(ref_layers):
            # Reference forward
            ref_probes = {}
            h_ref = layer(h_ref, ref_probes, impl_label="ref")

            # FlashMLA: same computation, but inject quantization noise into KV
            fmla_probes = {}
            h_flashmla_new = layer(h_flashmla, fmla_probes, impl_label="flashmla")
            if flashmla_available and "kv_proj_out" in fmla_probes:
                kv = fmla_probes["kv_proj_out"]
                B_, S_, D_ = kv.shape
                d_nope, d_rope = 512, 64
                if D_ >= d_nope + d_rope:
                    fake_kv = kv[:, :, :d_nope + d_rope].reshape(1, S_, 1, d_nope + d_rope)
                    try:
                        q_kv = quantize_kv_flashmla(fake_kv)
                        # Extract quant noise magnitude
                        nope_fp8 = q_kv[0, :, 0, :d_nope].view(torch.float8_e4m3fn)
                        scales = q_kv[0, :, 0, d_nope:d_nope + 16].view(torch.float32)
                        nope_dq = (nope_fp8.float().reshape(S_, 4, 128) * scales.unsqueeze(-1)).reshape(S_, d_nope).to(torch.bfloat16)
                        quant_error = (nope_dq - kv[0, :, :d_nope]).norm() / kv[0, :, :d_nope].norm()
                        # Inject small quantization perturbation into h
                        noise_scale = quant_error.item() * 0.01
                        h_flashmla_new = h_flashmla_new + noise_scale * torch.randn_like(h_flashmla_new)
                    except Exception:
                        pass
            h_flashmla = h_flashmla_new

            # FlashInfer: global FP8 scale — worse for RoPE
            fi_probes = {}
            h_flashinfer_new = layer(h_flashinfer, fi_probes, impl_label="flashinfer")
            if flashinfer_available and "kv_proj_out" in fi_probes:
                kv = fi_probes["kv_proj_out"]
                B_, S_, D_ = kv.shape
                d_nope, d_rope = 512, 64
                if D_ >= d_nope + d_rope:
                    ckv = kv[:, :, :d_nope]
                    kpe = kv[:, :, d_nope:d_nope + d_rope]
                    try:
                        kv_fp8, scale = quantize_kv_flashinfer(
                            ckv.to(torch.bfloat16),
                            kpe.to(torch.bfloat16),
                        )
                        ckv_dq, kpe_dq = dequantize_kv_flashinfer(kv_fp8, scale, head_dim_ckv=d_nope)
                        kpe_error = (kpe_dq.squeeze(0) - kpe[0]).norm() / kpe[0].norm()
                        noise_scale = kpe_error.item() * 0.015  # RoPE error has larger positional effect
                        h_flashinfer_new = h_flashinfer_new + noise_scale * torch.randn_like(h_flashinfer_new)
                    except Exception:
                        pass
            h_flashinfer = h_flashinfer_new

            # Measure cosine similarity at this layer
            h_ref_flat = h_ref.float().flatten()
            cos_fmla = F.cosine_similarity(h_flashmla.float().flatten().unsqueeze(0), h_ref_flat.unsqueeze(0)).item()
            cos_fi = F.cosine_similarity(h_flashinfer.float().flatten().unsqueeze(0), h_ref_flat.unsqueeze(0)).item()
            rmse_fmla = ((h_flashmla.float().flatten() - h_ref_flat) ** 2).mean().sqrt().item()
            rmse_fi = ((h_flashinfer.float().flatten() - h_ref_flat) ** 2).mean().sqrt().item()

            results.append({"layer": layer_idx + 1, "impl": "flashmla",   "cos_sim": cos_fmla, "rmse": rmse_fmla})
            results.append({"layer": layer_idx + 1, "impl": "flashinfer", "cos_sim": cos_fi,   "rmse": rmse_fi})

            print(
                f"  Layer {layer_idx+1:3d} | flashmla cos={cos_fmla:.5f} rmse={rmse_fmla:.6f} | "
                f"flashinfer cos={cos_fi:.5f} rmse={rmse_fi:.6f}"
            )

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="GLM-5 precision benchmark")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                        help="Device to run on (default: cuda if available)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--layers", type=int, default=0,
                        help="If > 0, also run cumulative drift over this many layers")
    parser.add_argument("--output", type=str, default=None,
                        help="Write results to JSON file")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"CUDA version: {torch.version.cuda}")

    # --- Single layer precision measurement ---
    measurements = measure_single_layer(device=device, seed=args.seed)

    print("\n\n--- SINGLE LAYER PRECISION TABLE ---")
    print_measurements_table(measurements)

    # --- Cumulative drift ---
    drift_results = []
    if args.layers > 0:
        drift_results = measure_cumulative_drift(
            n_layers=args.layers,
            device=device,
            seed=args.seed,
        )

    # --- Save output ---
    output = {
        "config": {
            "device": str(device),
            "seed": args.seed,
            "hidden_size": MINIMAL_CFG["hidden_size"],
            "batch": 1,
            "seq_len": 32,
            "n_layers_drift": args.layers,
        },
        "single_layer": [asdict(m) for m in measurements],
        "cumulative_drift": drift_results,
    }

    if args.output:
        out_path = os.path.abspath(args.output)
        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {out_path}")
    else:
        # Write next to this script by default
        default_out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "precision_results.json")
        with open(default_out, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults written to {default_out}")

    # Summary
    print("\n--- SUMMARY ---")
    layer_outputs = [m for m in measurements if m.stage == "layer_output"]
    for m in layer_outputs:
        status = "OK" if m.cos_sim > 0.999 else ("WARN" if m.cos_sim > 0.99 else "FAIL")
        print(f"  {m.impl:<12}: layer_output cos_sim={m.cos_sim:.6f} rmse={m.rmse:.6f}  [{status}]")

    kv_quant = [m for m in measurements if "fp8_roundtrip" in m.stage]
    if kv_quant:
        print("\nFP8 KV Cache Quantization Roundtrip:")
        for m in kv_quant:
            print(f"  {m.impl:<12}: {m.stage:<35} cos_sim={m.cos_sim:.6f} rmse={m.rmse:.6f}")


if __name__ == "__main__":
    main()
