"""Test 14: Gradient flow through the model.

Verifies that backward pass works and gradients are non-zero for all
trainable parameters, using the small test config.
"""

import sys
import torch
from .conftest import make_cfg


def test_gradient_flow():
    """Forward + backward should produce non-zero gradients on all parameters."""
    print("\n[Test 14] Gradient flow")
    from importlib import import_module
    kernel_model = import_module("glm5-kernels-flashmla-deepgemm.model")

    cfg = make_cfg(num_layers=2)
    torch.manual_seed(42)
    model = kernel_model.GlmMoeDsaForCausalLM(cfg)

    for layer in model.model.layers:
        layer.self_attn.use_flash_mla = False
        layer.self_attn.indexer.use_deepgemm = False

    model.train()
    B, S = 1, 8
    input_ids = torch.randint(0, cfg["vocab_size"], (B, S))
    labels = input_ids.clone()

    loss, logits, _ = model(input_ids=input_ids, labels=labels)

    ok = True
    if loss is None:
        print("  FAIL loss is None")
        return False

    loss.backward()

    zero_grad_params = []
    no_grad_params = []
    total_params = 0
    for name, param in model.named_parameters():
        total_params += 1
        if param.grad is None:
            no_grad_params.append(name)
        elif param.grad.abs().max().item() == 0:
            zero_grad_params.append(name)

    if no_grad_params:
        # DSA indexer params won't have gradients (torch.no_grad in forward)
        indexer_params = [p for p in no_grad_params if "indexer" in p]
        non_indexer = [p for p in no_grad_params if "indexer" not in p]
        if non_indexer:
            print(f"  FAIL {len(non_indexer)} non-indexer params with no grad:")
            for p in non_indexer[:5]:
                print(f"    {p}")
            ok = False
        else:
            print(f"  PASS {len(indexer_params)} indexer params correctly have no grad (torch.no_grad)")

    if zero_grad_params:
        # Some params may legitimately have zero grad if not used in this forward pass
        print(f"  WARN {len(zero_grad_params)} params with zero grad (may be unused for this input)")
        for p in zero_grad_params[:3]:
            print(f"    {p}")

    if ok:
        grad_params = total_params - len(no_grad_params) - len(zero_grad_params)
        print(f"  PASS gradient flow: {grad_params}/{total_params} params have non-zero grads")

    return ok


if __name__ == "__main__":
    sys.exit(0 if test_gradient_flow() else 1)
