"""H100 Category 7: Deterministic Execution.

Paper Section 3.2: non-deterministic CUDA topk caused "drastic performance
degradation during RL after only a few steps." Must guarantee bit-identical
decode outputs across multiple runs with the same seed.

Requirements: CUDA GPU (any), no special libraries needed.
"""

import sys
import torch
from .conftest import make_cfg, skip_no_cuda


@skip_no_cuda
def h100_test_deterministic_topk():
    """torch.topk must produce identical results across 10 calls."""
    print("\n[H100-Det-1] Deterministic topk")

    device = "cuda"
    torch.manual_seed(42)
    scores = torch.randn(32, 256, device=device)

    results = []
    for _ in range(10):
        idx = scores.topk(8, dim=-1).indices
        results.append(idx.clone())

    ok = all(torch.equal(results[0], r) for r in results[1:])
    if ok:
        print("  PASS topk is bit-identical across 10 calls")
    else:
        mismatches = sum(not torch.equal(results[0], r) for r in results[1:])
        print(f"  FAIL topk produced {mismatches}/9 different results")
    return ok


@skip_no_cuda
def h100_test_deterministic_full_decode():
    """Full decode: 10 tokens generated from same seed must be bit-identical across 3 runs."""
    print("\n[H100-Det-2] Deterministic full decode (3 runs x 10 tokens)")
    from importlib import import_module
    kernel_model = import_module("glm5-kernels-flashmla-deepgemm.model")

    cfg = make_cfg(num_layers=2)
    device = "cuda"

    def run_decode():
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        model = kernel_model.GlmMoeDsaForCausalLM(cfg).to(device).eval()
        for layer in model.model.layers:
            layer.self_attn.use_flash_mla = False
            layer.self_attn.indexer.use_deepgemm = False

        input_ids = torch.tensor([[1, 2, 3, 4]], device=device)
        tokens = []
        with torch.no_grad():
            _, logits, cache = model(input_ids=input_ids, use_cache=True)
            for _ in range(10):
                next_token = logits[:, -1:, :].argmax(dim=-1)
                tokens.append(next_token.item())
                _, logits, cache = model(input_ids=next_token, past_key_values=cache, use_cache=True)
        return tokens

    runs = [run_decode() for _ in range(3)]

    ok = runs[0] == runs[1] == runs[2]
    if ok:
        print(f"  PASS 3 runs produced identical tokens: {runs[0]}")
    else:
        print(f"  FAIL runs differ:")
        for i, r in enumerate(runs):
            print(f"    run {i}: {r}")
    return ok


@skip_no_cuda
def h100_test_deterministic_dsa_indexer():
    """DSA indexer must select identical top-k positions across runs."""
    print("\n[H100-Det-3] Deterministic DSA indexer")
    from importlib import import_module
    dsa = import_module("glm5-kernels-flashmla-deepgemm.dsa_indexer")
    rope_mod = import_module("glm5-kernels-flashmla-deepgemm.rope_partial")

    cfg = make_cfg()
    device = "cuda"

    results = []
    for _ in range(5):
        torch.manual_seed(99)
        torch.cuda.manual_seed(99)
        indexer = dsa.DSAIndexer(cfg, 0).to(device).eval()
        indexer.use_deepgemm = False

        hidden = torch.randn(1, 16, cfg["hidden_size"], device=device)
        q_resid = torch.randn(1, 16, cfg["q_lora_rank"], device=device)
        rope = rope_mod.RotaryEmbedding(cfg).to(device)
        cos, sin = rope(hidden, torch.arange(16, device=device).unsqueeze(0))

        with torch.no_grad():
            idx = indexer(hidden, q_resid, (cos, sin), use_cache=False)
        results.append(idx.cpu())

    ok = all(torch.equal(results[0], r) for r in results[1:])
    if ok:
        print("  PASS DSA indexer is bit-identical across 5 runs")
    else:
        print("  FAIL DSA indexer produced different selections across runs")
    return ok


if __name__ == "__main__":
    results = [
        h100_test_deterministic_topk(),
        h100_test_deterministic_full_decode(),
        h100_test_deterministic_dsa_indexer(),
    ]
    sys.exit(0 if all(results) else 1)
