#!/usr/bin/env python3
"""Run all GLM-5 benchmark experiments.

Usage:
    # Full suite (~3 hours on 1×H100):
    python -m benchmark.run_all --output-dir results/

    # Quick mode (~30 min, reduced sweeps):
    python -m benchmark.run_all --quick --output-dir results/

    # Specific experiment only:
    python -m benchmark.run_all --experiment moe_sweep --output-dir results/

Experiments:
    1. moe_sweep:      MoE-Inference-Bench (SC '25) style sweeps
    2. triple_micro:   Kernel microbenchmarks (per-component)
    3. triple_comp:    Component integration (full decoder layer)
    4. triple_e2e:     End-to-end serving scenarios (TTFT + TPOT)
    5. mfu_ceiling:    MFU analysis vs FA3's 75% ceiling
    6. fp8_pareto:     FP8 speed-quality Pareto frontier

Methodology aligned with: MLSys 2024, OSDI 2024, NeurIPS 2024, SC '25, MLPerf v5.1
"""

import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def run_experiment(name: str, output_dir: str, quick: bool = False):
    """Run a single experiment by name."""
    print(f"\n{'#'*80}")
    print(f"# Experiment: {name}")
    print(f"# Output: {output_dir}")
    print(f"# Quick mode: {quick}")
    print(f"{'#'*80}\n")

    start = time.time()

    if name == "moe_sweep":
        from benchmark.moe_sweep.bench_moe import run_moe_sweep
        run_moe_sweep(output_dir=os.path.join(output_dir, "moe_sweep"), quick=quick)

    elif name == "triple_micro":
        from benchmark.triple_report.bench_micro import run_micro_benchmark
        run_micro_benchmark(output_dir=os.path.join(output_dir, "triple_report"))

    elif name == "triple_comp":
        from benchmark.triple_report.bench_component import run_component_benchmark
        run_component_benchmark(output_dir=os.path.join(output_dir, "triple_report"))

    elif name == "triple_e2e":
        from benchmark.triple_report.bench_e2e import run_e2e_benchmark
        run_e2e_benchmark(output_dir=os.path.join(output_dir, "triple_report"))

    elif name == "mfu_ceiling":
        from benchmark.mfu_ceiling.bench_mfu import run_mfu_benchmark
        run_mfu_benchmark(output_dir=os.path.join(output_dir, "mfu_ceiling"), quick=quick)

    elif name == "fp8_pareto":
        from benchmark.fp8_pareto.bench_fp8 import run_fp8_benchmark
        run_fp8_benchmark(output_dir=os.path.join(output_dir, "fp8_pareto"))

    else:
        print(f"Unknown experiment: {name}")
        return

    elapsed = time.time() - start
    print(f"\n  Experiment '{name}' completed in {elapsed:.1f}s ({elapsed/60:.1f} min)\n")


ALL_EXPERIMENTS = [
    "moe_sweep",
    "triple_micro",
    "triple_comp",
    "triple_e2e",
    "mfu_ceiling",
    "fp8_pareto",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run GLM-5 benchmark suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--experiment",
        choices=ALL_EXPERIMENTS + ["all"],
        default="all",
        help="Which experiment to run (default: all)",
    )
    parser.add_argument("--output-dir", default="results/", help="Output directory for JSON results")
    parser.add_argument("--quick", action="store_true", help="Quick mode: reduced sweep ranges (~30 min)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    total_start = time.time()

    if args.experiment == "all":
        for exp in ALL_EXPERIMENTS:
            try:
                run_experiment(exp, args.output_dir, args.quick)
            except Exception as e:
                print(f"  FAILED: {exp} — {e}")
                import traceback
                traceback.print_exc()
    else:
        run_experiment(args.experiment, args.output_dir, args.quick)

    total_elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"  All experiments completed in {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print(f"  Results in: {args.output_dir}")
    print(f"{'='*80}")
