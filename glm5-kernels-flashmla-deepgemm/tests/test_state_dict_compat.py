"""Test 15: State dict key compatibility between glm5-triton and glm5-kernels.

Ensures that weights can be loaded from one implementation to the other,
which is critical for using pre-trained checkpoints.
"""

import sys
import torch
from .conftest import make_cfg


def test_state_dict_keys_match():
    """Both implementations should produce identical state_dict keys."""
    print("\n[Test 15a] State dict key compatibility")
    from importlib import import_module
    triton_model = import_module("glm5-triton.model")
    kernel_model = import_module("glm5-kernels-flashmla-deepgemm.model")

    cfg = make_cfg(num_layers=2)
    torch.manual_seed(42)
    ref = triton_model.GlmMoeDsaForCausalLM(cfg)
    torch.manual_seed(42)
    kern = kernel_model.GlmMoeDsaForCausalLM(cfg)

    ref_keys = set(ref.state_dict().keys())
    kern_keys = set(kern.state_dict().keys())

    ok = True
    only_ref = ref_keys - kern_keys
    only_kern = kern_keys - ref_keys

    if only_ref:
        print(f"  FAIL {len(only_ref)} keys only in glm5-triton:")
        for k in sorted(only_ref)[:10]:
            print(f"    {k}")
        ok = False

    if only_kern:
        print(f"  FAIL {len(only_kern)} keys only in glm5-kernels:")
        for k in sorted(only_kern)[:10]:
            print(f"    {k}")
        ok = False

    if ok:
        print(f"  PASS {len(ref_keys)} keys match exactly")
    return ok


def test_state_dict_cross_load():
    """Load glm5-triton state_dict into glm5-kernels and vice versa."""
    print("\n[Test 15b] Cross-loading state dicts")
    from importlib import import_module
    triton_model = import_module("glm5-triton.model")
    kernel_model = import_module("glm5-kernels-flashmla-deepgemm.model")

    cfg = make_cfg(num_layers=2)
    torch.manual_seed(42)
    ref = triton_model.GlmMoeDsaForCausalLM(cfg)
    kern = kernel_model.GlmMoeDsaForCausalLM(cfg)

    ok = True
    try:
        kern.load_state_dict(ref.state_dict())
        print("  PASS triton → kernels load")
    except Exception as e:
        print(f"  FAIL triton → kernels: {e}")
        ok = False

    try:
        ref.load_state_dict(kern.state_dict())
        print("  PASS kernels → triton load")
    except Exception as e:
        print(f"  FAIL kernels → triton: {e}")
        ok = False

    return ok


def test_state_dict_shapes_match():
    """All parameter shapes should match between implementations."""
    print("\n[Test 15c] Parameter shape compatibility")
    from importlib import import_module
    triton_model = import_module("glm5-triton.model")
    kernel_model = import_module("glm5-kernels-flashmla-deepgemm.model")

    cfg = make_cfg(num_layers=2)
    torch.manual_seed(42)
    ref = triton_model.GlmMoeDsaForCausalLM(cfg)
    torch.manual_seed(42)
    kern = kernel_model.GlmMoeDsaForCausalLM(cfg)

    ref_sd = ref.state_dict()
    kern_sd = kern.state_dict()

    ok = True
    for key in ref_sd:
        if key in kern_sd:
            if ref_sd[key].shape != kern_sd[key].shape:
                print(f"  FAIL {key}: {ref_sd[key].shape} vs {kern_sd[key].shape}")
                ok = False

    if ok:
        print(f"  PASS all {len(ref_sd)} parameter shapes match")
    return ok


if __name__ == "__main__":
    results = [
        test_state_dict_keys_match(),
        test_state_dict_cross_load(),
        test_state_dict_shapes_match(),
    ]
    sys.exit(0 if all(results) else 1)
