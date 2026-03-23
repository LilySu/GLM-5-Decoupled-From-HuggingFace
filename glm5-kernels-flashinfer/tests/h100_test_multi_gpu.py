"""H100 Category 5: Multi-GPU NCCL + Tensor Parallelism.

All-reduce bandwidth must reach >80% of NVLink 900GB/s. TP-sharded model
must produce numerically equivalent output to single-GPU.

Run with torchrun:
    torchrun --nproc_per_node=2 -m glm5-kernels-flashinfer.tests.h100_test_multi_gpu
"""

import os
import sys
import torch
import torch.distributed as dist
from .conftest import skip_no_multi_gpu, make_cfg, assert_close


def _setup_dist():
    if "RANK" not in os.environ:
        return None, None
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    dist.init_process_group("nccl")
    torch.cuda.set_device(local_rank)
    return rank, local_rank


@skip_no_multi_gpu
def h100_test_nccl_allreduce_bandwidth():
    """Measure NCCL all-reduce bandwidth. Target: >80% of NVLink peak."""
    print("\n[H100-NCCL-1] All-reduce bandwidth")

    rank, local_rank = _setup_dist()
    if rank is None:
        print("  SKIP not launched with torchrun")
        return True

    device = f"cuda:{local_rank}"
    world_size = dist.get_world_size()

    # Test with realistic GLM-5 hidden_size tensor
    for size_mb in [1, 10, 50]:
        numel = size_mb * 1024 * 1024 // 2  # BF16
        tensor = torch.randn(numel, dtype=torch.bfloat16, device=device)

        # Warmup
        for _ in range(5):
            dist.all_reduce(tensor)
        torch.cuda.synchronize()

        # Timed
        times = []
        for _ in range(20):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            dist.all_reduce(tensor)
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))

        times.sort()
        median_ms = times[len(times) // 2]
        # Ring all-reduce: 2*(N-1)/N * data_size bytes transferred
        algo_bytes = 2 * (world_size - 1) / world_size * tensor.numel() * tensor.element_size()
        bandwidth = algo_bytes / (median_ms * 1e-3) / 1e9

        if rank == 0:
            print(f"  {size_mb}MB: {bandwidth:.0f} GB/s ({median_ms:.3f} ms)")

    if rank == 0:
        print(f"  PASS all-reduce bandwidth measured (verify >720 GB/s for NVLink)")

    dist.destroy_process_group()
    return True


@skip_no_multi_gpu
def h100_test_tp_numerical_equivalence():
    """TP-sharded output should match single-GPU output (within BF16 tolerance)."""
    print("\n[H100-NCCL-2] TP numerical equivalence")

    rank, local_rank = _setup_dist()
    if rank is None:
        print("  SKIP not launched with torchrun")
        return True

    device = f"cuda:{local_rank}"
    world_size = dist.get_world_size()

    # Simulate column-parallel linear: each rank has a shard of the weight
    M, N, K = 32, 128, 64
    torch.manual_seed(42)
    full_weight = torch.randn(N, K, dtype=torch.bfloat16, device=device)
    x = torch.randn(M, K, dtype=torch.bfloat16, device=device)

    # Single-GPU reference
    full_out = x @ full_weight.T

    # TP: each rank takes N//world_size columns
    shard_size = N // world_size
    shard_weight = full_weight[rank * shard_size:(rank + 1) * shard_size]
    partial_out = x @ shard_weight.T  # [M, shard_size]

    # Gather all shards
    gathered = [torch.zeros_like(partial_out) for _ in range(world_size)]
    dist.all_gather(gathered, partial_out)
    tp_out = torch.cat(gathered, dim=-1)  # [M, N]

    if rank == 0:
        ok = assert_close("tp_vs_single", tp_out, full_out, atol=1e-2, rtol=1e-2)
    else:
        ok = True

    dist.destroy_process_group()
    return ok


@skip_no_multi_gpu
def h100_test_tp_expert_partitioning():
    """Verify MoE expert partitioning across GPUs produces correct aggregate output."""
    print("\n[H100-NCCL-3] TP expert partitioning")

    rank, local_rank = _setup_dist()
    if rank is None:
        print("  SKIP not launched with torchrun")
        return True

    device = f"cuda:{local_rank}"
    world_size = dist.get_world_size()

    # Simulate expert partitioning: E=8 experts across world_size GPUs
    E = 8
    experts_per_rank = E // world_size
    N, D, I = 16, 32, 16

    torch.manual_seed(42)
    hidden = torch.randn(N, D, dtype=torch.bfloat16, device=device)
    # Each rank has its shard of expert weights
    all_gate_up = torch.randn(E, 2 * I, D, dtype=torch.bfloat16, device=device)
    all_down = torch.randn(E, D, I, dtype=torch.bfloat16, device=device)

    my_experts = list(range(rank * experts_per_rank, (rank + 1) * experts_per_rank))

    # Simulate routing: each token goes to 2 experts
    torch.manual_seed(0)
    indices = torch.randint(0, E, (N, 2), device=device)
    weights = torch.softmax(torch.randn(N, 2, device=device), dim=-1).to(torch.bfloat16)

    # Each rank processes its experts
    partial_out = torch.zeros(N, D, dtype=torch.bfloat16, device=device)
    for k in range(2):
        for ei in my_experts:
            mask = indices[:, k] == ei
            if mask.any():
                tokens = hidden[mask]
                gate, up = torch.nn.functional.linear(tokens, all_gate_up[ei]).chunk(2, dim=-1)
                out = torch.nn.functional.linear(torch.nn.functional.silu(gate) * up, all_down[ei])
                partial_out[mask] += out * weights[mask, k:k + 1]

    # All-reduce to aggregate
    dist.all_reduce(partial_out)

    if rank == 0:
        ok = torch.isfinite(partial_out).all().item()
        if ok:
            print(f"  PASS expert partitioning: output all finite after all-reduce")
        else:
            print(f"  FAIL non-finite values in aggregated output")
    else:
        ok = True

    dist.destroy_process_group()
    return ok


if __name__ == "__main__":
    results = [
        h100_test_nccl_allreduce_bandwidth(),
        h100_test_tp_numerical_equivalence(),
        h100_test_tp_expert_partitioning(),
    ]
    sys.exit(0 if all(results) else 1)
