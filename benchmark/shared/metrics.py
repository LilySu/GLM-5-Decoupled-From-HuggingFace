"""Compute metrics aligned with FlashAttention-3, MoE-Inference-Bench, and roofline methodology.

Metrics:
- MFU (Model FLOPs Utilization): achieved TFLOPS / peak TFLOPS × 100
  Reference: FA2 50-73% on A100, FA3 75% on H100
- HBM SOL% (Speed of Light): achieved BW / peak BW × 100
  Reference: H100 peak 3.35 TB/s
- Operational Intensity: FLOPs / bytes (roofline position)
- Roofline bound classification: compute-bound vs memory-bound

References:
- Williams et al., "Roofline Model" (2009)
- FlashAttention-3: 740 TFLOPS FP16, 1.2 PFLOPS FP8 on H100
- DeepGEMM: 1550 TFLOPS FP8 grouped GEMM on H800
"""

from .config import H100_SPECS


# ── FLOPs Computation ────────────────────────────────────────────────────

def compute_attention_flops(B: int, H: int, S_q: int, S_kv: int,
                            d_qk: int = 576, d_v: int = 512) -> int:
    """FLOPs for MLA attention: QK^T matmul + softmax + PV matmul.

    For absorbed MLA:
    - QK^T: 2 * B * H * S_q * S_kv * d_qk  (matmul)
    - PV:   2 * B * H * S_q * S_kv * d_v    (matmul)
    - Softmax: ~5 * B * H * S_q * S_kv      (exp, sum, div — negligible vs matmul)

    Total ≈ 2 * B * H * S_q * S_kv * (d_qk + d_v)
    """
    return 2 * B * H * S_q * S_kv * (d_qk + d_v)


def compute_attention_bytes(B: int, H: int, S_q: int, S_kv: int,
                            d_qk: int = 576, d_v: int = 512,
                            dtype_bytes: int = 2) -> int:
    """Bytes accessed for MLA attention.

    Read: Q [B, H, S_q, d_qk], KV cache [B, 1, S_kv, d_qk+d_v] (absorbed: 1 KV head)
    Write: O [B, H, S_q, d_v]
    """
    q_bytes = B * H * S_q * d_qk * dtype_bytes
    # Absorbed MLA: single KV head, cache is [B, S_kv, d_qk + d_v] (no per-head replication)
    kv_bytes = B * S_kv * (d_qk + d_v) * dtype_bytes
    o_bytes = B * H * S_q * d_v * dtype_bytes
    return q_bytes + kv_bytes + o_bytes


def compute_moe_flops(N_tokens: int, K_active: int, D_hidden: int,
                      D_intermediate: int) -> int:
    """FLOPs for MoE forward: gate_up + silu + down, per selected expert.

    Per expert per token:
    - gate_proj: 2 * D_hidden * D_intermediate  (matmul)
    - up_proj:   2 * D_hidden * D_intermediate  (matmul)
    - silu:      ~D_intermediate                 (negligible)
    - down_proj: 2 * D_intermediate * D_hidden   (matmul)
    Total per token: 6 * D_hidden * D_intermediate

    K_active experts × N_tokens:
    """
    flops_per_token_per_expert = 6 * D_hidden * D_intermediate
    return N_tokens * K_active * flops_per_token_per_expert


def compute_moe_bytes(N_tokens: int, K_active: int, D_hidden: int,
                      D_intermediate: int, N_experts: int,
                      dtype_bytes: int = 2) -> int:
    """Bytes accessed for MoE forward.

    Read: hidden_states [N, D], expert weights [E, 2*I, D] + [E, D, I]
    Write: output [N, D]
    Weight read is the dominant cost — each active expert's full weight matrix.
    """
    input_bytes = N_tokens * D_hidden * dtype_bytes
    # Each active expert: gate_up [2*I, D] + down [D, I]
    weight_bytes_per_expert = (2 * D_intermediate * D_hidden + D_hidden * D_intermediate) * dtype_bytes
    weight_bytes = K_active * weight_bytes_per_expert  # only active experts loaded
    output_bytes = N_tokens * D_hidden * dtype_bytes
    return input_bytes + weight_bytes + output_bytes


def compute_dsa_indexer_flops(S_q: int, S_kv: int, H_idx: int = 32,
                              D_idx: int = 128) -> int:
    """FLOPs for DSA lightning indexer scoring.

    score[s,t] = ReLU(sum_h(w_h * q_h · k_h)) for each (query, key) pair.
    Per (s,t): 2 * H_idx * D_idx (dot products) + H_idx (weighted sum) + 1 (ReLU)
    Total: S_q * S_kv * (2 * H_idx * D_idx + H_idx)
    """
    flops_per_pair = 2 * H_idx * D_idx + H_idx
    return S_q * S_kv * flops_per_pair


# ── Derived Metrics ──────────────────────────────────────────────────────

def compute_mfu(flops: int, latency_s: float, precision: str = "bf16") -> float:
    """Model FLOPs Utilization: achieved / peak × 100.

    FA3 reference: 75% MFU on H100 FP16 (740 / 989 TFLOPS).
    """
    if latency_s <= 0:
        return 0.0
    achieved_tflops = flops / latency_s / 1e12
    peak_key = f"peak_tflops_{precision}"
    peak = H100_SPECS.get(peak_key, H100_SPECS["peak_tflops_bf16"])
    return (achieved_tflops / peak) * 100


def compute_hbm_sol(bytes_accessed: int, latency_s: float) -> float:
    """HBM Speed-of-Light: achieved bandwidth / peak × 100.

    H100 SXM5 peak: 3.35 TB/s.
    """
    if latency_s <= 0:
        return 0.0
    achieved_gb_s = bytes_accessed / latency_s / 1e9
    return (achieved_gb_s / H100_SPECS["hbm_bandwidth_gb_s"]) * 100


def compute_tflops(flops: int, latency_s: float) -> float:
    """Raw TFLOPS achieved."""
    if latency_s <= 0:
        return 0.0
    return flops / latency_s / 1e12


def compute_bandwidth_gb_s(bytes_accessed: int, latency_s: float) -> float:
    """Raw bandwidth achieved in GB/s."""
    if latency_s <= 0:
        return 0.0
    return bytes_accessed / latency_s / 1e9


def compute_operational_intensity(flops: int, bytes_accessed: int) -> float:
    """Operational intensity: FLOPs per byte.

    The roofline ridge point for H100:
    ridge = peak_tflops / peak_bw = 989e12 / 3.35e12 ≈ 295 FLOPs/byte (BF16)
    Below ridge = memory-bound. Above ridge = compute-bound.
    """
    if bytes_accessed <= 0:
        return 0.0
    return flops / bytes_accessed


def classify_roofline_bound(operational_intensity: float,
                            precision: str = "bf16") -> str:
    """Classify as memory-bound or compute-bound on the roofline.

    Ridge point = peak compute / peak bandwidth.
    H100 BF16: 989 TFLOPS / 3.35 TB/s ≈ 295 FLOPs/byte
    H100 FP8:  1979 TFLOPS / 3.35 TB/s ≈ 590 FLOPs/byte
    """
    peak_key = f"peak_tflops_{precision}"
    peak_tflops = H100_SPECS.get(peak_key, H100_SPECS["peak_tflops_bf16"])
    peak_bw = H100_SPECS["hbm_bandwidth_gb_s"]
    ridge = (peak_tflops * 1e12) / (peak_bw * 1e9)  # FLOPs/byte

    if operational_intensity < ridge:
        return "memory-bound"
    else:
        return "compute-bound"


def compute_roofline_achievable(operational_intensity: float,
                                precision: str = "bf16") -> float:
    """Theoretical maximum TFLOPS at given operational intensity.

    achievable = min(peak_compute, peak_bw × OI)
    """
    peak_key = f"peak_tflops_{precision}"
    peak_tflops = H100_SPECS.get(peak_key, H100_SPECS["peak_tflops_bf16"])
    peak_bw = H100_SPECS["hbm_bandwidth_gb_s"]
    bw_limited = peak_bw * operational_intensity / 1e3  # GB/s × FLOPs/byte / 1e3 = TFLOPS
    return min(peak_tflops, bw_limited)
