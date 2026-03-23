"""Test 12: Multi-step autoregressive decode.

Validates that prefill + sequential decode steps produce consistent outputs
between glm5-triton and glm5-kernels, using the KV cache correctly.
"""

import sys
import torch
from .conftest import assert_close, make_cfg


def test_autoregressive_3step():
    """Prefill 4 tokens, then decode 3 more tokens step by step."""
    print("\n[Test 12] Autoregressive decode (prefill=4, decode=3)")
    from importlib import import_module
    triton_model = import_module("glm5-triton.model")
    kernel_model = import_module("glm5-kernels-flashmla-deepgemm.model")

    cfg = make_cfg(num_layers=2)
    torch.manual_seed(42)
    ref = triton_model.GlmMoeDsaForCausalLM(cfg)
    torch.manual_seed(42)
    kern = kernel_model.GlmMoeDsaForCausalLM(cfg)
    kern.load_state_dict(ref.state_dict())

    for layer in kern.model.layers:
        layer.self_attn.use_flash_mla = False
        layer.self_attn.indexer.use_deepgemm = False

    ref.eval()
    kern.eval()

    B = 1
    prefill_ids = torch.randint(0, cfg["vocab_size"], (B, 4))

    ok = True
    with torch.no_grad():
        # Prefill
        _, ref_logits, ref_cache = ref(input_ids=prefill_ids, use_cache=True)
        _, kern_logits, kern_cache = kern(input_ids=prefill_ids, use_cache=True)
        ok = assert_close("prefill_logits", ref_logits, kern_logits, atol=1e-4) and ok

        # 3 decode steps
        for step in range(3):
            next_token = ref_logits[:, -1:, :].argmax(dim=-1)
            _, ref_logits, ref_cache = ref(
                input_ids=next_token, past_key_values=ref_cache, use_cache=True,
            )
            _, kern_logits, kern_cache = kern(
                input_ids=next_token, past_key_values=kern_cache, use_cache=True,
            )
            ok = assert_close(f"decode_step_{step}_logits", ref_logits, kern_logits, atol=1e-3) and ok

    return ok


if __name__ == "__main__":
    sys.exit(0 if test_autoregressive_3step() else 1)
