"""H100 Category 9: Precision Boundary Stress.

312 FP8<->BF16 crossings per forward pass (4 per layer x 78 layers).
Chain multiple roundtrips to measure accumulated drift. The concern is that
quantization error compounds through the model, degrading output quality.

Requirements: CUDA GPU (any).
"""

import sys
import torch
from .conftest import assert_close, cosine_sim, skip_no_cuda


@skip_no_cuda
def h100_test_precision_chain_roundtrips():
    """Chain 78 FP8 roundtrips (simulating one per layer) and measure accumulated error."""
    print("\n[H100-Prec-1] Chained FP8 roundtrips (78 iterations)")
    from importlib import import_module
    fp8 = import_module("glm5-kernels-flashinfer.fp8_utils")

    device = "cuda"
    torch.manual_seed(42)
    x_orig = torch.randn(4, 512, device=device, dtype=torch.float32)
    x = x_orig.clone()

    # Simulate 78 layers, each with quantize->dequantize
    num_layers = 78
    for i in range(num_layers):
        x_fp8, scales = fp8.quantize_activations_deepgemm(x, block_size=128)
        x = fp8.dequantize_fp8(x_fp8, scales, block_size=128).float()

    # Measure accumulated error
    abs_err = (x - x_orig).abs()
    max_abs = abs_err.max().item()
    mean_abs = abs_err.mean().item()
    cos = cosine_sim(x, x_orig)

    # Relative error on non-tiny values
    mask = x_orig.abs() > 0.1
    rel_err = (abs_err[mask] / x_orig[mask].abs()).max().item() if mask.any() else 0

    print(f"  After {num_layers} roundtrips:")
    print(f"    max_abs_err:  {max_abs:.4f}")
    print(f"    mean_abs_err: {mean_abs:.6f}")
    print(f"    max_rel_err:  {rel_err:.4f}")
    print(f"    cosine_sim:   {cos:.6f}")

    # Tolerance: cosine similarity should stay >0.90 after 78 roundtrips.
    # FP8 E4M3 has ~6% per-roundtrip error, but errors partially cancel
    # (not strictly additive). Empirically, cos_sim stays >0.95.
    ok = cos > 0.90
    if ok:
        print(f"  PASS cosine_sim {cos:.4f} > 0.90 threshold")
    else:
        print(f"  FAIL cosine_sim {cos:.4f} < 0.90 — accumulated drift too high")
    return ok


@skip_no_cuda
def h100_test_precision_full_pipeline():
    """Run BF16→FP8→BF16 through the actual model pipeline and measure output drift."""
    print("\n[H100-Prec-2] Full pipeline precision (model with vs without FP8 roundtrips)")
    from importlib import import_module
    kernel_model = import_module("glm5-kernels-flashinfer.model")

    cfg = make_cfg_for_precision()
    device = "cuda"

    torch.manual_seed(42)
    model = kernel_model.GlmMoeDsaForCausalLM(cfg).to(device).eval()
    for layer in model.model.layers:
        layer.self_attn.use_flashinfer = False
        layer.self_attn.indexer.use_deepgemm = False

    B, S = 1, 16
    input_ids = torch.randint(0, cfg["vocab_size"], (B, S), device=device)

    # Reference: standard forward
    with torch.no_grad():
        _, logits_ref, _ = model(input_ids=input_ids)

    # Perturbed: inject FP8 roundtrip noise at each layer boundary
    fp8 = import_module("glm5-kernels-flashinfer.fp8_utils")

    def inject_fp8_hook(module, input, output):
        if isinstance(output, tuple):
            return output
        x = output
        x_fp8, scales = fp8.quantize_activations_deepgemm(x.float(), block_size=128)
        return fp8.dequantize_fp8(x_fp8, scales, block_size=128).to(x.dtype)

    hooks = []
    for layer in model.model.layers:
        h = layer.input_layernorm.register_forward_hook(inject_fp8_hook)
        hooks.append(h)

    with torch.no_grad():
        _, logits_fp8, _ = model(input_ids=input_ids)

    for h in hooks:
        h.remove()

    # Compare
    cos = cosine_sim(logits_fp8, logits_ref)
    max_diff = (logits_fp8.float() - logits_ref.float()).abs().max().item()

    print(f"  Logit cosine_sim: {cos:.6f}")
    print(f"  Logit max_diff:   {max_diff:.4f}")

    # With FP8 noise injected at every layer, logits should still be close
    ok = cos > 0.85
    if ok:
        print(f"  PASS pipeline cosine_sim {cos:.4f} > 0.85 despite FP8 noise injection")
    else:
        print(f"  FAIL pipeline cosine_sim {cos:.4f} < 0.85 — FP8 drift too severe")
    return ok


def make_cfg_for_precision():
    """4-layer config for precision testing."""
    from .conftest import make_cfg
    return make_cfg(num_layers=4)


if __name__ == "__main__":
    results = [
        h100_test_precision_chain_roundtrips(),
        h100_test_precision_full_pipeline(),
    ]
    sys.exit(0 if all(results) else 1)
