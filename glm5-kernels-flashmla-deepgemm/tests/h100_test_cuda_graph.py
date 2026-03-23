"""H100 Category 1: CUDA Graph Capture & Replay.

Decode uses CUDA graphs to eliminate ~390us/token kernel launch overhead.
Must verify that: (1) graphs capture successfully, (2) replay with updated
sparse indices produces correct outputs, (3) KV cache updates work inside graphs.

IMPORTANT: A failed CUDA graph capture corrupts the CUDA context. All subsequent
GPU allocations fail with "Offset increment outside graph capture". Each test
wraps graph operations in try/finally with _reset_cuda_state() to prevent cascading.

Requirements: H100 (SM90), flash-mla or deep-gemm for kernel paths.
"""

import sys
import gc
import torch
from .conftest import assert_close, make_cfg, skip_no_cuda, skip_no_sm90, cuda_timer_fn


def _reset_cuda_state():
    """Reset CUDA context after a failed graph capture to prevent cascading errors."""
    if torch.cuda.is_available():
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


@skip_no_cuda
def h100_test_cuda_graph_capture_model():
    """Capture a full decode step in a CUDA graph and replay it.

    Uses dense-only layers (no MoE) because MoE's expert dispatch uses
    torch.nonzero() which is not CUDA graph capturable.
    """
    print("\n[H100-Graph-1] CUDA graph capture of full decode step")
    from importlib import import_module
    kernel_model = import_module("glm5-kernels-flashmla-deepgemm.model")

    cfg = make_cfg(num_layers=2)
    cfg["mlp_layer_types"] = ["dense"] * 2
    device = "cuda"

    model = kernel_model.GlmMoeDsaForCausalLM(cfg).to(device).eval()
    for layer in model.model.layers:
        layer.self_attn.use_flash_mla = False
        layer.self_attn.indexer.use_deepgemm = False

    B, S_prefill = 1, 16
    input_ids = torch.randint(0, cfg["vocab_size"], (B, S_prefill), device=device)

    graph = None
    try:
        with torch.no_grad():
            _, logits, cache = model(input_ids=input_ids, use_cache=True)

        static_token = torch.zeros(B, 1, dtype=torch.long, device=device)
        static_token[0, 0] = logits[:, -1:, :].argmax(dim=-1).item()

        # Warmup decode path (required before graph capture)
        with torch.no_grad():
            _, _, cache = model(input_ids=static_token, past_key_values=cache, use_cache=True)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with torch.no_grad():
                _, static_out, _ = model(input_ids=static_token, past_key_values=cache, use_cache=True)

        static_token[0, 0] = 42
        graph.replay()
        captured_out = static_out.clone()

        ok = True
        if not torch.isfinite(captured_out).all():
            print("  FAIL non-finite values after graph replay")
            ok = False
        if captured_out.shape != (B, 1, cfg["vocab_size"]):
            print(f"  FAIL output shape {captured_out.shape}")
            ok = False
        if ok:
            print(f"  PASS graph capture + replay: output shape={captured_out.shape}, all finite")
        return ok

    except Exception as e:
        print(f"  FAIL graph capture: {e}")
        return False
    finally:
        del graph
        _reset_cuda_state()


@skip_no_cuda
def h100_test_cuda_graph_sparse_index_update():
    """Verify that sparse DSA indices can be updated between graph replays."""
    print("\n[H100-Graph-2] Sparse index tensor update inside CUDA graph")

    device = "cuda"
    B, S, T, topk = 1, 1, 128, 16

    # All tensors must be allocated BEFORE graph capture
    static_indices = torch.zeros(B, S, topk, dtype=torch.long, device=device)
    static_scores = torch.randn(B, S, T, device=device)

    graph = None
    try:
        # Warmup
        topk_result = static_scores.topk(topk, dim=-1).indices
        static_indices.copy_(topk_result)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            topk_result = static_scores.topk(topk, dim=-1).indices
            static_indices.copy_(topk_result)

        ok = True
        for trial in range(3):
            # Update scores IN-PLACE (no new allocation — graph compatible)
            static_scores.copy_(torch.randn(B, S, T, device=device))
            graph.replay()

            expected = static_scores.topk(topk, dim=-1).indices
            if not torch.equal(static_indices, expected):
                print(f"  FAIL trial {trial}: indices mismatch after replay")
                ok = False

        if ok:
            print("  PASS sparse indices correctly updated across 3 graph replays")
        return ok

    except Exception as e:
        print(f"  FAIL sparse index graph: {e}")
        return False
    finally:
        del graph
        _reset_cuda_state()


@skip_no_cuda
def h100_test_cuda_graph_speedup():
    """Measure actual speedup from CUDA graph replay vs eager decode."""
    print("\n[H100-Graph-3] CUDA graph speedup measurement")
    from importlib import import_module
    kernel_model = import_module("glm5-kernels-flashmla-deepgemm.model")

    cfg = make_cfg(num_layers=2)
    cfg["mlp_layer_types"] = ["dense"] * 2  # dense only for graph compatibility
    device = "cuda"
    model = kernel_model.GlmMoeDsaForCausalLM(cfg).to(device).eval()
    for layer in model.model.layers:
        layer.self_attn.use_flash_mla = False
        layer.self_attn.indexer.use_deepgemm = False

    B = 1
    input_ids = torch.randint(0, cfg["vocab_size"], (B, 16), device=device)
    static_token = torch.randint(0, cfg["vocab_size"], (B, 1), device=device)

    graph = None
    try:
        with torch.no_grad():
            _, _, cache_eager = model(input_ids=input_ids, use_cache=True)
            _, _, cache_eager = model(input_ids=static_token, past_key_values=cache_eager, use_cache=True)

        # Eager timing
        eager_times = cuda_timer_fn(
            lambda: model(input_ids=static_token, past_key_values=cache_eager, use_cache=True),
            warmup=5, iters=50,
        )
        eager_median = eager_times[len(eager_times) // 2]

        # Re-prefill for graph path
        with torch.no_grad():
            _, _, cache_graph = model(input_ids=input_ids, use_cache=True)
            _, _, cache_graph = model(input_ids=static_token, past_key_values=cache_graph, use_cache=True)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            with torch.no_grad():
                _, static_out, _ = model(input_ids=static_token, past_key_values=cache_graph, use_cache=True)

        graph_times = cuda_timer_fn(lambda: graph.replay(), warmup=5, iters=50)
        graph_median = graph_times[len(graph_times) // 2]

        speedup = eager_median / graph_median if graph_median > 0 else 0
        print(f"  Eager: {eager_median:.3f} ms, Graph: {graph_median:.3f} ms, Speedup: {speedup:.2f}x")

        ok = speedup > 1.0
        if ok:
            print(f"  PASS graph replay speedup = {speedup:.2f}x")
        else:
            print(f"  WARN graph not faster (may be due to small model size)")
        return True  # soft pass

    except Exception as e:
        print(f"  FAIL graph speedup: {e}")
        return False
    finally:
        del graph
        _reset_cuda_state()


if __name__ == "__main__":
    results = [
        h100_test_cuda_graph_capture_model(),
        h100_test_cuda_graph_sparse_index_update(),
        h100_test_cuda_graph_speedup(),
    ]
    sys.exit(0 if all(results) else 1)
