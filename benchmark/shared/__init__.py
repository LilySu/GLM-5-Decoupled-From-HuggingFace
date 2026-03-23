"""Shared benchmark utilities: timing, metrics, config, reporting."""
from .timer import cuda_timer_extended
from .metrics import compute_mfu, compute_hbm_sol, compute_attention_flops, compute_attention_bytes, compute_moe_flops
from .config import GLM5_CONFIG, H100_SPECS, BenchConfig, BenchResult
from .report import save_results, capture_environment
