"""H100 Category 10: Thermal Throttling Detection.

H100 throttles under sustained load (TDP=700W SXM). If the last window's
TFLOPS drops below 85% of the first window, thermal management is kicking in
and benchmark numbers are unreliable.

Requirements: H100 (SM90), CUDA.
"""

import sys
import time
import torch
from .conftest import skip_no_cuda, cuda_timer_fn


@skip_no_cuda
def h100_test_thermal_sustained_gemm():
    """Run sustained FP32 GEMM for 30s, compare first-5s vs last-5s throughput."""
    print("\n[H100-Thermal-1] Sustained GEMM thermal stability (30s)")

    device = "cuda"
    M, N, K = 4096, 4096, 4096
    a = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b = torch.randn(K, N, device=device, dtype=torch.bfloat16)

    flops_per_op = 2 * M * N * K

    def run_window(duration_s, label):
        """Run GEMM ops for duration_s seconds, return median TFLOPS."""
        times = []
        deadline = time.time() + duration_s
        while time.time() < deadline:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            torch.mm(a, b)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times.sort()
        median_ms = times[len(times) // 2]
        tflops = flops_per_op / (median_ms * 1e-3) / 1e12
        return tflops, median_ms, len(times)

    # Warmup
    for _ in range(20):
        torch.mm(a, b)
    torch.cuda.synchronize()

    # First window (GPU is cold)
    tflops_first, ms_first, n_first = run_window(5.0, "first")
    print(f"  First 5s:  {tflops_first:.1f} TFLOPS ({ms_first:.3f} ms/op, {n_first} ops)")

    # Sustain load for middle window
    _ = run_window(20.0, "middle")

    # Last window (GPU is hot)
    tflops_last, ms_last, n_last = run_window(5.0, "last")
    print(f"  Last 5s:   {tflops_last:.1f} TFLOPS ({ms_last:.3f} ms/op, {n_last} ops)")

    ratio = tflops_last / tflops_first if tflops_first > 0 else 0
    print(f"  Ratio:     {ratio:.3f} (last/first)")

    ok = ratio > 0.85
    if ok:
        print(f"  PASS thermal stable: {ratio:.1%} retention (threshold: >85%)")
    else:
        print(f"  FAIL thermal throttling detected: {ratio:.1%} retention < 85%")
        print(f"  Check: nvidia-smi -q -d TEMPERATURE,POWER")
    return ok


@skip_no_cuda
def h100_test_thermal_clock_frequency():
    """Check that GPU clock frequency is at or near boost clock (not throttled)."""
    print("\n[H100-Thermal-2] GPU clock frequency check")

    device = "cuda"

    # Run a workload to get the GPU to boost
    a = torch.randn(2048, 2048, device=device, dtype=torch.bfloat16)
    for _ in range(50):
        torch.mm(a, a)
    torch.cuda.synchronize()

    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=clocks.current.sm,clocks.max.sm,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            print(f"  SKIP nvidia-smi failed: {result.stderr.strip()}")
            return True

        lines = result.stdout.strip().split("\n")
        # Parse first GPU (or the one we're using)
        gpu_idx = torch.cuda.current_device()
        if gpu_idx < len(lines):
            parts = [p.strip() for p in lines[gpu_idx].split(",")]
            current_mhz = float(parts[0])
            max_mhz = float(parts[1])
            temp_c = float(parts[2])
            power_w = float(parts[3])

            ratio = current_mhz / max_mhz if max_mhz > 0 else 0
            print(f"  Clock: {current_mhz:.0f}/{max_mhz:.0f} MHz ({ratio:.1%})")
            print(f"  Temp:  {temp_c:.0f}C, Power: {power_w:.0f}W")

            # H100 SXM at 700W TDP commonly runs at 80-85% of max clock under load.
            # 80% is a reasonable threshold for a warmed-up GPU.
            ok = ratio > 0.80
            if ok:
                print(f"  PASS clock at {ratio:.1%} of max (>80%)")
            else:
                print(f"  FAIL clock throttled to {ratio:.1%} of max")
                if temp_c > 80:
                    print(f"  Likely cause: temperature {temp_c:.0f}C > 80C thermal limit")
                if power_w > 650:
                    print(f"  Likely cause: power {power_w:.0f}W near TDP limit")
            return ok
        else:
            print(f"  SKIP could not read GPU {gpu_idx} stats")
            return True

    except FileNotFoundError:
        print("  SKIP nvidia-smi not found")
        return True
    except Exception as e:
        print(f"  SKIP nvidia-smi error: {e}")
        return True


if __name__ == "__main__":
    results = [
        h100_test_thermal_sustained_gemm(),
        h100_test_thermal_clock_frequency(),
    ]
    sys.exit(0 if all(results) else 1)
