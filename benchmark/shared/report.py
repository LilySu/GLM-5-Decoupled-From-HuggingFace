"""Output reporting: JSON results + environment snapshot.

Every benchmark run produces:
1. Raw JSON with all measurements (for post-hoc analysis / reproducibility)
2. Environment snapshot (GPU, CUDA, library versions, temperature, clocks)
3. Human-readable summary to stdout

Aligned with NeurIPS reproducibility checklist requirements.
"""

import json
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List

import torch

from .config import BenchResult


def capture_environment() -> Dict[str, Any]:
    """Capture full environment for reproducibility.

    Reports exact hardware, software, and environmental state
    following NeurIPS 2024+ reproducibility checklist.
    """
    env = {
        "timestamp": datetime.now().isoformat(),
        "hostname": os.uname().nodename,
        # GPU
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
        "gpu_count": torch.cuda.device_count(),
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else [],
        # Software versions
        "cuda_version": torch.version.cuda or "N/A",
        "pytorch_version": torch.__version__,
        "python_version": os.popen("python3 --version").read().strip(),
    }

    # Library versions (conditional — may not be installed)
    for lib_name in ["flash_mla", "flashinfer", "deep_gemm", "triton"]:
        try:
            mod = __import__(lib_name)
            env[f"{lib_name}_version"] = getattr(mod, "__version__", "installed (no version)")
        except ImportError:
            env[f"{lib_name}_version"] = "not installed"

    # GPU state via nvidia-smi
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.current.sm,power.draw,power.limit",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if smi.returncode == 0:
            parts = smi.stdout.strip().split(", ")
            if len(parts) >= 4:
                env["gpu_temp_c"] = float(parts[0])
                env["gpu_clock_mhz"] = float(parts[1])
                env["gpu_power_w"] = float(parts[2])
                env["gpu_power_limit_w"] = float(parts[3])
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # CUDA memory
    if torch.cuda.is_available():
        env["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_mem / 1e9
        env["gpu_memory_allocated_gb"] = torch.cuda.memory_allocated(0) / 1e9

    return env


def save_results(
    results: List[BenchResult],
    output_dir: str,
    experiment_name: str,
    env: Dict[str, Any] = None,
) -> str:
    """Save results as JSON with full reproducibility info.

    Args:
        results: List of BenchResult from benchmark runs.
        output_dir: Directory to write JSON files.
        experiment_name: Name for this experiment (e.g., "moe_sweep", "mfu_ceiling").
        env: Environment dict from capture_environment(). Captured fresh if None.

    Returns:
        Path to the saved JSON file.
    """
    if env is None:
        env = capture_environment()

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{experiment_name}_{timestamp}.json"
    filepath = os.path.join(output_dir, filename)

    # Serialize results
    serialized = []
    for r in results:
        d = {
            "name": r.name,
            "impl": r.impl,
            "config": r.config,
            "latency_ms_raw": r.latency_ms,  # ALL raw values for post-hoc analysis
            "median_ms": r.median_ms,
            "mean_ms": r.mean_ms,
            "std_ms": r.std_ms,
            "p5_ms": r.p5_ms,
            "p50_ms": r.p50_ms,
            "p95_ms": r.p95_ms,
            "p99_ms": r.p99_ms,
            "ci_95": [r.ci_95_low, r.ci_95_high],
            "tflops": r.tflops,
            "mfu_pct": r.mfu_pct,
            "bandwidth_gb_s": r.bandwidth_gb_s,
            "hbm_sol_pct": r.hbm_sol_pct,
            "operational_intensity": r.operational_intensity,
            "roofline_bound": r.roofline_bound,
            "peak_memory_gb": r.peak_memory_gb,
            "kv_cache_memory_gb": r.kv_cache_memory_gb,
            "cosine_similarity": r.cosine_similarity,
            "rmse": r.rmse,
            "ttft_ms": r.ttft_ms,
            "tpot_ms": r.tpot_ms,
            "meets_sla": r.meets_sla,
            "is_oom": r.is_oom,
            "error": r.error,
        }
        serialized.append(d)

    output = {
        "experiment": experiment_name,
        "environment": env,
        "results": serialized,
        "metadata": {
            "num_results": len(serialized),
            "num_oom": sum(1 for r in results if r.is_oom),
            "generated_by": "glm5/benchmark",
            "methodology": "MoE-Inference-Bench (SC'25) + FA3 + Sarathi-Serve",
        },
    }

    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"Results saved to: {filepath}")
    return filepath


def print_summary_table(results: List[BenchResult], title: str = ""):
    """Print a human-readable summary table to stdout."""
    if title:
        print(f"\n{'='*80}")
        print(f"  {title}")
        print(f"{'='*80}")

    # Header
    print(f"{'Name':<30} {'Impl':<12} {'Median(ms)':<12} {'p99(ms)':<10} "
          f"{'TFLOPS':<10} {'MFU%':<8} {'BW(GB/s)':<10} {'SOL%':<8} {'Bound':<10}")
    print("-" * 120)

    for r in results:
        if r.is_oom:
            print(f"{r.name:<30} {r.impl:<12} {'OOM':<12}")
            continue
        print(f"{r.name:<30} {r.impl:<12} {r.median_ms:<12.3f} {r.p99_ms:<10.3f} "
              f"{r.tflops:<10.1f} {r.mfu_pct:<8.1f} {r.bandwidth_gb_s:<10.1f} "
              f"{r.hbm_sol_pct:<8.1f} {r.roofline_bound:<10}")
