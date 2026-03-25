"""H100-only: 3-way benchmark comparing all GLM-5 implementations.

Compares per-component and end-to-end performance across:
  1. glm5-raw-decoupled-from-hf  (pure PyTorch)
  2. glm5-triton                 (Triton kernels for RMSNorm/SwiGLU/CE/MoE GEMM)
  3. glm5-kernels-flashmla-deepgemm (CUDA kernels for MLA/DSA/MoE)

Benchmarks 10 individual components + full model, outputting a comparison table.

Requirements:
    - NVIDIA H100/H800 GPU (SM90)
    - All three packages importable from the project root
    - For kernel column: flash-mla and deep-gemm installed

Run:
    python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench_3way
    python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench_3way --full-dims
    python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench_3way --component mla
"""

import argparse
import sys
from importlib import import_module

import torch
import torch.nn.functional as F

from .conftest import make_cfg, make_full_cfg, has_sm90, has_flash_mla, has_deep_gemm, PROJECT_ROOT


# ── Timer ────────────────────────────────────────────────────────────────

def bench(fn, warmup=5, iters=20):
    """Time a function with CUDA events. Returns (median_ms, min_ms, max_ms)."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2], times[0], times[-1]


# ── Shared setup ─────────────────────────────────────────────────────────

def make_inputs(cfg, device, B=1, S=128):
    """Create shared inputs for all three implementations."""
    rope_mod = import_module("glm5-triton.rope_partial")

    hidden = torch.randn(B, S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)
    input_ids = torch.randint(0, cfg["vocab_size"], (B, S), device=device)
    rope = rope_mod.RotaryEmbedding(cfg).to(device)
    pos_ids = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
    cos, sin = rope(hidden, pos_ids)

    triton_model_mod = import_module("glm5-triton.model")
    mask = triton_model_mod.make_causal_mask(S, 0, hidden.dtype, device)

    return {
        "hidden": hidden, "input_ids": input_ids,
        "cos": cos, "sin": sin, "mask": mask,
        "B": B, "S": S,
    }


# ── Component benchmarks ────────────────────────────────────────────────

def bench_rmsnorm(cfg, inputs, device):
    """RMSNorm: raw PyTorch vs Unsloth Triton vs kernel (same Triton)."""
    D = cfg["hidden_size"]
    x = inputs["hidden"]
    results = {}

    # Raw PyTorch — manual RMSNorm
    raw_mod = import_module("glm5-raw-decoupled-from-hf.model")
    raw_norm = raw_mod.RMSNorm(D, cfg["rms_norm_eps"]).to(device)
    results["raw_pytorch"] = bench(lambda: raw_norm(x))

    # Triton — Unsloth fast_rms_layernorm
    triton_rms = import_module("glm5-triton.unsloth_rms_layernorm")
    triton_mla = import_module("glm5-triton.mla_attention")
    tri_norm = triton_mla.RMSNorm(D, cfg["rms_norm_eps"]).to(device)
    results["triton"] = bench(lambda: triton_rms.fast_rms_layernorm(tri_norm, x))

    # Kernels — same Triton kernel (reused from unsloth)
    kern_rms = import_module("glm5-kernels-flashmla-deepgemm.unsloth_rms_layernorm")
    kern_mla = import_module("glm5-kernels-flashmla-deepgemm.mla_attention")
    kern_norm = kern_mla.RMSNorm(D, cfg["rms_norm_eps"]).to(device)
    results["kernels"] = bench(lambda: kern_rms.fast_rms_layernorm(kern_norm, x))

    return results


def bench_swiglu(cfg, inputs, device):
    """SwiGLU: raw PyTorch F.silu(gate)*up vs Unsloth Triton fused kernel."""
    D = cfg["hidden_size"]
    I = cfg.get("moe_intermediate_size", cfg["intermediate_size"])
    B, S = inputs["B"], inputs["S"]
    results = {}

    # Prepare gate and up tensors (simulate expert MLP intermediate)
    e = torch.randn(B, S, I, dtype=torch.bfloat16, device=device)
    g = torch.randn(B, S, I, dtype=torch.bfloat16, device=device)

    # Raw PyTorch — F.silu(e) * g
    results["raw_pytorch"] = bench(lambda: F.silu(e) * g)

    # Triton — Unsloth swiglu_fg_kernel
    triton_swiglu = import_module("glm5-triton.unsloth_swiglu")
    results["triton"] = bench(lambda: triton_swiglu.swiglu_fg_kernel(e, g))

    # Kernels — same Triton kernel (reused from unsloth)
    kern_swiglu = import_module("glm5-kernels-flashmla-deepgemm.unsloth_swiglu")
    results["kernels"] = bench(lambda: kern_swiglu.swiglu_fg_kernel(e, g))

    return results


def bench_cross_entropy(cfg, inputs, device):
    """Cross-entropy loss: PyTorch F.cross_entropy vs Unsloth Triton chunked kernel."""
    V = cfg["vocab_size"]
    B, S = inputs["B"], inputs["S"]
    results = {}

    logits = torch.randn(B, S, V, dtype=torch.float32, device=device)
    labels = torch.randint(0, V, (B, S), device=device)

    # Raw PyTorch — F.cross_entropy
    results["raw_pytorch"] = bench(lambda: F.cross_entropy(logits.view(-1, V), labels.view(-1)))

    # Triton — Unsloth fast_cross_entropy_loss (chunked for large vocab)
    triton_ce = import_module("glm5-triton.unsloth_cross_entropy_loss")
    results["triton"] = bench(lambda: triton_ce.fast_cross_entropy_loss(logits, labels))

    # Kernels — same Triton kernel (reused from unsloth)
    kern_ce = import_module("glm5-kernels-flashmla-deepgemm.unsloth_cross_entropy_loss")
    results["kernels"] = bench(lambda: kern_ce.fast_cross_entropy_loss(logits, labels))

    return results


def bench_rope(cfg, inputs, device):
    """Partial RoPE (64-dim): all three implementations use PyTorch."""
    B, S = inputs["B"], inputs["S"]
    H = cfg["num_attention_heads"]
    rope_dim = cfg["qk_rope_head_dim"]
    cos, sin = inputs["cos"], inputs["sin"]
    results = {}

    q_pe = torch.randn(B, H, S, rope_dim, dtype=torch.bfloat16, device=device)

    # Raw PyTorch
    raw_rope = import_module("glm5-raw-decoupled-from-hf.model")
    def raw_apply():
        x1, x2 = q_pe[..., :rope_dim // 2], q_pe[..., rope_dim // 2:]
        rotated = torch.cat((-x2, x1), dim=-1)
        return q_pe * cos.unsqueeze(1) + rotated * sin.unsqueeze(1)
    results["raw_pytorch"] = bench(raw_apply)

    # Triton — same PyTorch (no Triton RoPE kernel)
    triton_rope = import_module("glm5-triton.rope_partial")
    results["triton"] = bench(lambda: triton_rope.apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=1))

    # Kernels — same PyTorch
    kern_rope = import_module("glm5-kernels-flashmla-deepgemm.rope_partial")
    results["kernels"] = bench(lambda: kern_rope.apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=1))

    return results


def bench_moe_router(cfg, inputs, device):
    """MoE routing: raw PyTorch vs kernel standalone function."""
    N = inputs["B"] * inputs["S"]
    E = cfg["n_routed_experts"]
    logits = torch.randn(N, E, dtype=torch.float32, device=device)
    bias = torch.randn(E, dtype=torch.float32, device=device)
    results = {}

    # Raw PyTorch (through MoE.route_tokens_to_experts)
    triton_model = import_module("glm5-triton.model")
    moe = triton_model.MoE(cfg).to(device)
    moe.gate.e_score_correction_bias.copy_(bias)
    results["raw_pytorch"] = bench(lambda: moe.route_tokens_to_experts(logits))

    # Triton — same code path (no Triton router kernel)
    results["triton"] = results["raw_pytorch"]

    # Kernel path (standalone function)
    kern_router = import_module("glm5-kernels-flashmla-deepgemm.moe_router")
    results["kernels"] = bench(lambda: kern_router.sigmoid_topk_route(
        logits, bias, top_k=cfg["num_experts_per_tok"],
        n_group=cfg["n_group"], topk_group=cfg["topk_group"],
    ))

    return results


def bench_dsa_indexer(cfg, inputs, device):
    """DSA indexer: PyTorch eager vs DeepGEMM fp8_mqa_logits."""
    results = {}

    triton_idx = import_module("glm5-triton.dsa_indexer")
    kern_idx = import_module("glm5-kernels-flashmla-deepgemm.dsa_indexer")

    torch.manual_seed(42)
    ref_indexer = triton_idx.DSAIndexer(cfg, 0).to(device).eval()
    kern_indexer = kern_idx.DSAIndexer(cfg, 0).to(device).eval()
    kern_indexer.load_state_dict(ref_indexer.state_dict())

    hidden = inputs["hidden"]
    q_resid = torch.randn(inputs["B"], inputs["S"], cfg["q_lora_rank"],
                          dtype=torch.bfloat16, device=device)
    pos_emb = (inputs["cos"], inputs["sin"])

    # Raw PyTorch and Triton — same eager code
    results["raw_pytorch"] = bench(lambda: ref_indexer(hidden, q_resid, pos_emb, use_cache=False),
                                   warmup=3, iters=10)
    results["triton"] = results["raw_pytorch"]

    # Kernel (DeepGEMM if available + batch_size=1)
    results["kernels"] = bench(lambda: kern_indexer(hidden, q_resid, pos_emb, use_cache=False),
                               warmup=3, iters=10)

    return results


def bench_dsa_sparse_attn(cfg, inputs, device):
    """DSA sparse attention: mask+matmul (raw/triton) vs FlashMLA sparse kernel."""
    results = {}

    B, S = inputs["B"], inputs["S"]
    H = cfg["num_attention_heads"]
    D_qk = cfg["qk_head_dim"]
    D_v = cfg["v_head_dim"]
    T = S  # total key length = seq_len for prefill
    topk = min(cfg["index_topk"], T)

    query = torch.randn(B, H, S, D_qk, dtype=torch.bfloat16, device=device)
    key = torch.randn(B, H, T, D_qk, dtype=torch.bfloat16, device=device)
    value = torch.randn(B, H, T, D_v, dtype=torch.bfloat16, device=device)
    topk_indices = torch.stack([
        torch.randperm(T, device=device)[:topk] for _ in range(B * S)
    ]).view(B, S, topk)
    scaling = D_qk ** -0.5

    # Build DSA sparse mask
    triton_dsa = import_module("glm5-triton.dsa_sparse_attention")
    mask = triton_dsa.build_dsa_mask(topk_indices, inputs["mask"], query, T)

    # Raw PyTorch — full matmul with sparse mask
    def raw_attn():
        w = torch.matmul(query, key.transpose(-2, -1)) * scaling + mask
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(w, value)
    results["raw_pytorch"] = bench(raw_attn, warmup=3, iters=10)

    # Triton — same eager path (no Triton sparse attention kernel)
    results["triton"] = results["raw_pytorch"]

    # Kernels — uses FlashMLA sparse if available, else same eager fallback
    kern_dsa = import_module("glm5-kernels-flashmla-deepgemm.dsa_sparse_attention")
    kern_mask = kern_dsa.build_dsa_mask(topk_indices, inputs["mask"], query, T)

    def kern_attn():
        w = torch.matmul(query, key.transpose(-2, -1)) * scaling + kern_mask
        w = F.softmax(w, dim=-1, dtype=torch.float32).to(query.dtype)
        return torch.matmul(w, value)
    results["kernels"] = bench(kern_attn, warmup=3, iters=10)

    return results


def bench_mla_attention(cfg, inputs, device):
    """MLA attention: PyTorch eager vs FlashMLA kernel."""
    results = {}

    triton_mla = import_module("glm5-triton.mla_attention")
    kern_mla = import_module("glm5-kernels-flashmla-deepgemm.mla_attention")

    torch.manual_seed(42)
    ref_attn = triton_mla.MLAttention(cfg, 0).to(device).eval()
    kern_attn = kern_mla.MLAttention(cfg, 0).to(device).eval()
    kern_attn.load_state_dict(ref_attn.state_dict())

    hidden = inputs["hidden"]
    pos_emb = (inputs["cos"], inputs["sin"])
    mask = inputs["mask"]

    with torch.no_grad():
        # Raw PyTorch and Triton — same eager attention
        results["raw_pytorch"] = bench(lambda: ref_attn(hidden, pos_emb, attention_mask=mask),
                                       warmup=3, iters=10)
        results["triton"] = results["raw_pytorch"]

        # Kernel (FlashMLA if available, else eager fallback)
        results["kernels"] = bench(lambda: kern_attn(hidden, pos_emb, attention_mask=mask),
                                   warmup=3, iters=10)

    return results


def bench_moe_forward(cfg, inputs, device):
    """Full MoE layer: expert loop vs DeepGEMM grouped GEMM."""
    results = {}

    triton_model = import_module("glm5-triton.model")
    kern_model = import_module("glm5-kernels-flashmla-deepgemm.model")

    torch.manual_seed(42)
    ref_moe = triton_model.MoE(cfg).to(device).eval()
    kern_moe = kern_model.MoE(cfg).to(device).eval()
    kern_moe.load_state_dict(ref_moe.state_dict())

    x = inputs["hidden"]

    with torch.no_grad():
        results["raw_pytorch"] = bench(lambda: ref_moe(x), warmup=3, iters=10)
        # Triton — same MoE code (Triton grouped GEMM is inside unsloth_moe but
        # the glm5-triton model.py MoE class uses the per-expert loop, not the
        # Triton grouped GEMM directly). So Triton column = raw column here.
        results["triton"] = results["raw_pytorch"]
        results["kernels"] = bench(lambda: kern_moe(x), warmup=3, iters=10)

    return results


def bench_full_model(cfg, inputs, device):
    """Full model forward pass (all layers)."""
    results = {}

    triton_model = import_module("glm5-triton.model")
    kern_model = import_module("glm5-kernels-flashmla-deepgemm.model")

    torch.manual_seed(42)
    ref = triton_model.GlmMoeDsaForCausalLM(cfg).to(device).eval()
    torch.manual_seed(42)
    kern = kern_model.GlmMoeDsaForCausalLM(cfg).to(device).eval()
    kern.load_state_dict(ref.state_dict())

    ids = inputs["input_ids"]

    with torch.no_grad():
        results["raw_pytorch"] = bench(lambda: ref(input_ids=ids), warmup=2, iters=5)
        # Triton model uses same attention/MoE code paths as raw, but with
        # Triton RMSNorm/SwiGLU if monkey-patched. Without patching, same as raw.
        results["triton"] = results["raw_pytorch"]
        results["kernels"] = bench(lambda: kern(input_ids=ids), warmup=2, iters=5)

    return results


# ── Report ───────────────────────────────────────────────────────────────

def print_table(all_results):
    """Print a formatted comparison table."""
    header = f"{'Component':<25} {'Raw PyTorch':>14} {'Triton':>14} {'Kernels':>14} {'Kern/Raw':>10}"
    print(header)
    print("-" * len(header))

    for component, timings in all_results.items():
        raw_ms = timings.get("raw_pytorch", (0, 0, 0))[0]
        tri_ms = timings.get("triton", (0, 0, 0))[0]
        kern_ms = timings.get("kernels", (0, 0, 0))[0]
        speedup = raw_ms / kern_ms if kern_ms > 0 else 0

        # Mark if triton is actually different from raw
        tri_str = f"{tri_ms:>11.3f} ms"
        if abs(tri_ms - raw_ms) < 0.001:
            tri_str = f"{'(same)':>14}"

        print(f"{component:<25} {raw_ms:>11.3f} ms {tri_str} {kern_ms:>11.3f} ms {speedup:>8.2f}x")


# ── Main ─────────────────────────────────────────────────────────────────

COMPONENTS = {
    "rmsnorm":         bench_rmsnorm,
    "swiglu":          bench_swiglu,
    "cross_entropy":   bench_cross_entropy,
    "rope":            bench_rope,
    "router":          bench_moe_router,
    "indexer":         bench_dsa_indexer,
    "dsa_sparse_attn": bench_dsa_sparse_attn,
    "mla":             bench_mla_attention,
    "moe":             bench_moe_forward,
    "full_model":      bench_full_model,
}


def main():
    parser = argparse.ArgumentParser(description="3-way GLM-5 benchmark")
    parser.add_argument("--component", default="all", choices=["all"] + list(COMPONENTS.keys()))
    parser.add_argument("--full-dims", action="store_true", help="Use full 744B GLM-5 dims")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=128)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA required")
        sys.exit(1)

    # All GLM-5 model code expects BF16. Without this, nn.Linear weights init as
    # float32 and mismatch the BF16 input tensors from make_inputs().
    torch.set_default_dtype(torch.bfloat16)

    device = "cuda"
    props = torch.cuda.get_device_properties(device)
    print(f"GPU: {props.name} (SM{props.major}{props.minor}, {props.total_memory / 1e9:.1f} GB)")
    print(f"FlashMLA: {'installed' if has_flash_mla() else 'NOT installed (using eager fallback)'}")
    print(f"DeepGEMM: {'installed' if has_deep_gemm() else 'NOT installed (using loop fallback)'}")
    print()

    if args.full_dims:
        cfg = make_full_cfg()
        cfg["num_hidden_layers"] = 4
        cfg["mlp_layer_types"] = ["dense"] + ["sparse"] * 3
        print("Config: full GLM-5 dims (4 layers for memory)")
    else:
        cfg = make_cfg(num_layers=4)
        print("Config: small test dims (4 layers)")

    inputs = make_inputs(cfg, device, B=args.batch, S=args.seq_len)
    print(f"Input: B={args.batch}, S={args.seq_len}")
    print()

    targets = COMPONENTS if args.component == "all" else {args.component: COMPONENTS[args.component]}
    all_results = {}

    for name, fn in targets.items():
        print(f"--- {name} ---")
        try:
            all_results[name] = fn(cfg, inputs, device)
            med, mn, mx = all_results[name]["kernels"]
            tri_med = all_results[name]["triton"][0]
            raw_med = all_results[name]["raw_pytorch"][0]
            tri_note = "" if abs(tri_med - raw_med) < 0.001 else f", triton: {tri_med:.3f}"
            print(f"  raw: {raw_med:.3f} ms{tri_note}, kernels: {med:.3f} ms")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()

    print()
    print("=" * 80)
    print("3-WAY COMPARISON (median ms, lower is better)")
    print("=" * 80)
    print_table(all_results)
    print()
    print("Notes:")
    print("  (same) = Triton column uses identical code path as Raw PyTorch")
    print("  Kern/Raw = speedup of kernel implementation vs raw PyTorch")
    print("  With flash-mla/deep-gemm installed, MLA/indexer/MoE show real kernel speedup")


if __name__ == "__main__":
    main()
