"""Extended CUDA timer with academic-grade statistical analysis.

Follows FlashAttention-3 and MoE-Inference-Bench methodology:
- 10 warmup iterations (JIT compilation, thermal stabilization)
- 100 measured iterations (sufficient for p99 estimation)
- Bootstrap 95% CI on the median (1000 resamples)
- Full latency distribution preserved for post-hoc analysis

References:
- FlashAttention-3 (Tri Dao, 2024): 75% MFU on H100, benchmarked with CUDA events
- MoE-Inference-Bench (SC '25): systematic sweeps on 4×H100
"""

import random
import statistics
from typing import Callable, Dict, List, Tuple

import torch


def cuda_timer_extended(
    fn: Callable,
    warmup: int = 10,
    iters: int = 100,
    sync: bool = True,
    bootstrap_samples: int = 1000,
) -> Tuple[List[float], Dict]:
    """Time a CUDA function with full statistical analysis.

    Args:
        fn: Zero-argument callable to benchmark.
        warmup: Number of warmup iterations (default 10, not 5 — accounts for
                DeepGEMM JIT compilation and FlashInfer module build).
        iters: Number of measured iterations (default 100, not 20 — required
               for meaningful p99 and bootstrap CI).
        sync: Whether to synchronize after each iteration.
        bootstrap_samples: Number of bootstrap resamples for CI computation.

    Returns:
        Tuple of (raw_times_ms, stats_dict) where stats_dict contains:
        - median, mean, std
        - p5, p50, p95, p99 (tail latency is critical for serving papers)
        - ci_95_low, ci_95_high (bootstrap 95% CI on the median)
        - min, max
        - cv (coefficient of variation = std/mean)
    """
    if not torch.cuda.is_available():
        return [], {"error": "no CUDA device"}

    # ── Warmup ───────────────────────────────────────────────────────────
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()

    # ── Measured iterations ──────────────────────────────────────────────
    times: List[float] = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        if sync:
            torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # milliseconds

    # ── Statistics ───────────────────────────────────────────────────────
    times_sorted = sorted(times)
    n = len(times_sorted)
    mean_val = statistics.mean(times)
    std_val = statistics.stdev(times) if n > 1 else 0.0
    median_val = statistics.median(times)

    def percentile(data, pct):
        k = (len(data) - 1) * (pct / 100)
        f = int(k)
        c = f + 1
        if c >= len(data):
            return data[-1]
        return data[f] + (k - f) * (data[c] - data[f])

    p5 = percentile(times_sorted, 5)
    p50 = percentile(times_sorted, 50)
    p95 = percentile(times_sorted, 95)
    p99 = percentile(times_sorted, 99)

    # ── Bootstrap 95% CI on the median ──────────────────────────────────
    # Non-parametric CI — no normality assumption (Mann-Whitney compatible)
    rng = random.Random(42)  # deterministic bootstrap
    medians = []
    for _ in range(bootstrap_samples):
        resample = [rng.choice(times) for _ in range(n)]
        medians.append(statistics.median(resample))
    medians.sort()
    ci_low = percentile(medians, 2.5)
    ci_high = percentile(medians, 97.5)

    stats = {
        "median": median_val,
        "mean": mean_val,
        "std": std_val,
        "p5": p5,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "min": times_sorted[0],
        "max": times_sorted[-1],
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "cv": std_val / mean_val if mean_val > 0 else 0.0,
        "num_iters": n,
        "warmup": warmup,
    }

    return times, stats


def check_outliers(times: List[float]) -> Dict:
    """Check for timing anomalies that invalidate results.

    Flags:
    - p99/p50 > 2.0: Inconsistent kernel performance (thermal throttling?)
    - CV > 0.15: High variance (background interference?)
    - First 10 measurements > 2× last 10: Thermal warmup not complete
    """
    if not times:
        return {"valid": False, "reason": "no measurements"}

    times_sorted = sorted(times)
    n = len(times_sorted)

    def pct(data, p):
        k = (len(data) - 1) * (p / 100)
        f = int(k)
        return data[min(f, len(data) - 1)]

    p50 = pct(times_sorted, 50)
    p99 = pct(times_sorted, 99)
    mean_val = statistics.mean(times)
    std_val = statistics.stdev(times) if n > 1 else 0.0
    cv = std_val / mean_val if mean_val > 0 else 0.0

    flags = []
    if p50 > 0 and p99 / p50 > 2.0:
        flags.append(f"p99/p50 = {p99/p50:.2f} > 2.0 (inconsistent, check thermal)")
    if cv > 0.15:
        flags.append(f"CV = {cv:.3f} > 0.15 (high variance, check background load)")
    if n >= 20:
        first_10 = statistics.mean(times[:10])
        last_10 = statistics.mean(times[-10:])
        if first_10 > 2 * last_10:
            flags.append(f"first_10/last_10 = {first_10/last_10:.2f} (warmup incomplete)")

    return {
        "valid": len(flags) == 0,
        "flags": flags,
        "p99_p50_ratio": p99 / p50 if p50 > 0 else 0,
        "cv": cv,
    }
