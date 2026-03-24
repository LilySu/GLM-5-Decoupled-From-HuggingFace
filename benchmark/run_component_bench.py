"""Run component benchmark in a clean subprocess to avoid CUDA state issues."""
import subprocess
import sys

print("Step 1: Verify GPU is clean...")
r = subprocess.run([sys.executable, "/workspace/GLM-5-Decoupled-From-HuggingFace/benchmark/debug_single_layer.py"])

if r.returncode != 0:
    print("GPU is in bad state. Restart the RunPod pod and try again.")
    sys.exit(1)

print("\nStep 2: Running component benchmark...")
r2 = subprocess.run([
    sys.executable, "-m", "benchmark.triple_report.bench_component",
    "--output-dir", "results/triple/",
])

if r2.returncode != 0:
    print(f"\nComponent benchmark failed with code {r2.returncode}")
else:
    print("\nComponent benchmark completed successfully.")
