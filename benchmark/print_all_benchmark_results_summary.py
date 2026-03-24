"""Print a summary of all benchmark JSON results found in the results directory.

Usage (on RunPod):
    cd /workspace/GLM-5-Decoupled-From-HuggingFace
    python3 -m benchmark.print_all_benchmark_results_summary
"""

import glob
import json
import os
import sys


def main():
    # Find results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    results_dir = os.path.join(project_root, "results")

    if not os.path.isdir(results_dir):
        print(f"No results/ directory found at {results_dir}")
        sys.exit(1)

    json_files = sorted(glob.glob(os.path.join(results_dir, "**/*.json"), recursive=True))
    if not json_files:
        json_files = sorted(glob.glob(os.path.join(results_dir, "*.json")))

    if not json_files:
        print(f"No JSON files found in {results_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} result files in {results_dir}\n")

    for path in json_files:
        name = os.path.basename(path)
        rel = os.path.relpath(path, project_root)
        print(f"{'=' * 70}")
        print(f"FILE: {rel}")
        print(f"{'=' * 70}")

        try:
            with open(path) as f:
                d = json.load(f)
        except Exception as e:
            print(f"  ERROR reading: {e}\n")
            continue

        # Print environment info if present
        if isinstance(d, dict) and "environment" in d:
            env = d["environment"]
            print(f"  GPU: {env.get('gpu_name', '?')}")
            print(f"  CUDA: {env.get('cuda_version', '?')}")
            print(f"  PyTorch: {env.get('pytorch_version', '?')}")
            print()

        # Print results
        results = None
        if isinstance(d, dict) and "results" in d:
            results = d["results"]
        elif isinstance(d, list):
            results = d

        if results and isinstance(results, list):
            for r in results[:30]:
                if isinstance(r, dict):
                    name_val = r.get("name", r.get("component", "?"))
                    ms = r.get("median_ms", r.get("p50_ms", r.get("latency_ms", 0)))
                    tflops = r.get("tflops", 0)
                    mfu = r.get("mfu_pct", 0)
                    bw = r.get("bandwidth_gb_s", 0)
                    cos = r.get("cosine_similarity", r.get("cos_sim", 0))
                    rmse = r.get("rmse", 0)
                    bound = r.get("roofline_bound", "")
                    sol = r.get("hbm_sol_pct", 0)
                    impl = r.get("impl", r.get("precision", ""))

                    line = f"  {name_val}"
                    if impl:
                        line += f" [{impl}]"
                    line += f": {ms:.3f} ms"
                    if tflops:
                        line += f", {tflops:.1f} TFLOPS"
                    if mfu:
                        line += f", MFU={mfu:.1f}%"
                    if bw:
                        line += f", BW={bw:.0f} GB/s"
                    if sol:
                        line += f", SOL={sol:.1f}%"
                    if cos and cos != 1.0:
                        line += f", cos={cos:.4f}"
                    if rmse:
                        line += f", RMSE={rmse:.4f}"
                    if bound:
                        line += f" ({bound})"

                    # Extra fields
                    skip_keys = {"name", "component", "impl", "precision", "median_ms", "p50_ms",
                                 "latency_ms", "mean_ms", "std_ms", "p5_ms", "p95_ms", "p99_ms",
                                 "ci_95_low", "ci_95_high", "min_ms", "max_ms",
                                 "tflops", "mfu_pct", "bandwidth_gb_s", "hbm_sol_pct",
                                 "operational_intensity", "roofline_bound",
                                 "cosine_similarity", "cos_sim", "rmse",
                                 "peak_memory_gb", "kv_cache_memory_gb",
                                 "ttft_ms", "tpot_ms", "meets_sla",
                                 "is_oom", "error", "is_pareto",
                                 "latency_ms_raw", "config", "num_iters"}
                    extras = {k: v for k, v in r.items()
                              if k not in skip_keys and v and v != 0 and v != "" and v != False}
                    if extras:
                        extra_str = ", ".join(f"{k}={v}" for k, v in list(extras.items())[:5])
                        line += f"  | {extra_str}"

                    print(line)
                else:
                    print(f"  {r}")
        elif isinstance(d, dict):
            # Print top-level keys
            for k, v in list(d.items())[:15]:
                if k in ("results", "environment"):
                    continue
                val_str = str(v)
                if len(val_str) > 100:
                    val_str = val_str[:100] + "..."
                print(f"  {k}: {val_str}")

        print()

    print(f"{'=' * 70}")
    print(f"TOTAL: {len(json_files)} result files printed")


if __name__ == "__main__":
    main()
