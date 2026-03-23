"""GLM-5 model dimensions and H100 hardware constants.

All values sourced from:
- GLM-5 paper (arXiv:2602.15763) Table 10
- NVIDIA H100 SXM5 datasheet
- FlashAttention-3 (75% MFU reference)
"""

from dataclasses import dataclass, field
from typing import Optional, List


# ── GLM-5 Architecture Constants ────────────────────────────────────────

GLM5_CONFIG = {
    # Attention (MLA)
    "num_heads": 64,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 192,
    "qk_head_dim": 256,  # 192 + 64
    "v_head_dim": 256,
    "d_qk_absorbed": 576,  # 512 + 64 (absorbed format)
    "d_v_absorbed": 512,   # kv_lora_rank (absorbed format)
    "q_lora_rank": 2048,
    # MoE
    "hidden_size": 6144,
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 2048,
    "routed_scaling_factor": 2.5,
    "n_group": 1,         # GLM-5 uses FLAT routing (no hierarchical groups)
    "topk_group": 1,
    # DSA (Dynamic Sparse Attention)
    "index_n_heads": 32,
    "index_head_dim": 128,
    "index_topk": 2048,
    # Layers
    "num_layers": 78,     # 3 dense + 75 MoE
    "num_dense_layers": 3,
    "num_moe_layers": 75,
    # Other
    "vocab_size": 154880,
    "intermediate_size": 12288,  # dense FFN
    "max_position_embeddings": 202752,
    "page_size": 64,
}


# ── H100 SXM5 Hardware Constants ────────────────────────────────────────

H100_SPECS = {
    # Compute peaks
    "peak_tflops_fp32": 67.0,      # FP32 CUDA cores
    "peak_tflops_tf32": 495.0,     # TF32 tensor cores (w/ sparsity: 989)
    "peak_tflops_bf16": 989.0,     # BF16 tensor cores (w/ sparsity: 1979)
    "peak_tflops_fp16": 989.0,     # FP16 tensor cores
    "peak_tflops_fp8": 1979.0,     # FP8 tensor cores (w/ sparsity: 3958)
    "peak_tflops_int8": 1979.0,    # INT8 tensor cores
    # Memory
    "hbm_capacity_gb": 80.0,       # HBM3
    "hbm_bandwidth_gb_s": 3350.0,  # 3.35 TB/s
    "l2_cache_mb": 50.0,
    "sram_per_sm_kb": 256.0,       # 256 KB per SM (228 KB usable)
    "num_sms": 132,
    # Reference points
    "fa3_mfu_pct": 75.0,           # FlashAttention-3 FP16 MFU on H100
    "fa3_tflops_fp16": 740.0,      # FA3 achieved TFLOPS
    "fa3_pflops_fp8": 1.2,         # FA3 FP8 peak
    "flashmla_tflops_decode": 660.0,  # FlashMLA dense decode
    "flashmla_tflops_sparse": 410.0,  # FlashMLA sparse decode
    "deepgemm_tflops_fp8": 1550.0,    # DeepGEMM grouped GEMM
    # Interconnect
    "nvlink_bw_gb_s": 900.0,       # NVLink bidirectional
    "pcie5_bw_gb_s": 128.0,        # PCIe Gen5 x16
}


# ── MoE-Inference-Bench Sweep Ranges (SC '25 Standard) ──────────────────

MOE_BENCH_BATCHES = [1, 16, 32, 64]
MOE_BENCH_TOKENS = [128, 256, 512, 1024, 2048]
MOE_BENCH_EXPERTS = [8, 64, 128, 256]  # GLM-5 is 256
MOE_BENCH_ACTIVE = [2, 4, 8]            # GLM-5 is top-8
MOE_BENCH_FFN_DIMS = [1024, 2048, 4096]  # GLM-5 is 2048


# ── MLPerf v5.1 SLA Thresholds ──────────────────────────────────────────

MLPERF_TTFT_P99_MS = 2000.0    # p99 TTFT < 2 seconds
MLPERF_TPOT_P99_MS = 80.0      # p99 TPOT < 80 ms


@dataclass
class BenchConfig:
    """Configuration for a single benchmark run."""
    batch_size: int = 32
    seq_len: int = 128        # prompt length (prefill) or query length
    context_len: int = 4096   # KV cache length (decode)
    precision: str = "bf16"   # "bf16" | "fp8"
    mode: str = "decode"      # "prefill" | "decode"
    impl: str = "flashmla"    # "flashmla" | "flashinfer"
    component: str = "mla"    # "mla" | "dsa_indexer" | "moe_gemm" | "fp8_quant" | "full_layer" | "full_model"
    warmup: int = 10
    iters: int = 100
    # MoE-specific
    num_tokens: int = 1024    # tokens routed to MoE
    num_experts: int = 256
    active_experts: int = 8
    ffn_dim: int = 2048


@dataclass
class BenchResult:
    """Result from a benchmark run, with full statistical analysis.

    Fields aligned with FlashAttention-3 and MoE-Inference-Bench reporting standards.
    """
    name: str
    impl: str
    config: dict

    # Raw latency data (all iterations)
    latency_ms: List[float] = field(default_factory=list)

    # Latency statistics
    median_ms: float = 0.0
    mean_ms: float = 0.0
    std_ms: float = 0.0
    p5_ms: float = 0.0
    p50_ms: float = 0.0
    p95_ms: float = 0.0
    p99_ms: float = 0.0
    ci_95_low: float = 0.0
    ci_95_high: float = 0.0

    # Performance metrics (FlashAttention standard)
    tflops: float = 0.0
    mfu_pct: float = 0.0         # % of H100 peak (FA3 reference: 75%)
    bandwidth_gb_s: float = 0.0
    hbm_sol_pct: float = 0.0     # % of H100 HBM bandwidth

    # Roofline
    operational_intensity: float = 0.0  # FLOPs / byte
    roofline_bound: str = ""            # "compute" | "memory"

    # Memory
    peak_memory_gb: float = 0.0
    kv_cache_memory_gb: float = 0.0

    # Quality (for FP8 Pareto)
    cosine_similarity: float = 1.0     # vs BF16 reference
    rmse: float = 0.0                  # vs BF16 reference

    # Serving metrics (OSDI standard)
    ttft_ms: float = 0.0       # Time to First Token
    tpot_ms: float = 0.0       # Time Per Output Token
    meets_sla: bool = True     # p99 within MLPerf thresholds

    # Status
    is_oom: bool = False
    error: str = ""
