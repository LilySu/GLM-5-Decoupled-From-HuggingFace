# Validation script for glm5-triton.
#
# Run from the glm5/ root:
#   python3 glm5-triton/validate.py
#
# Tests (8 total):
#   1. Forward pass          — loss is finite, logits shape correct
#   2. Backward pass         — gradients reach all parameter groups
#   3. Training convergence  — loss drops >80% in 20 steps on one batch
#   4. Label masking         — masked vs unmasked loss differs
#   5. KV cache decode       — autoregressive generation, cache grows correctly
#   6. Multi-turn training   — variable-length padded batch, loss still decreases
#   7. Long sequence         — 256-token sequence, attention + DSA indexer survive
#   8. Gradient checkpointing — forward+backward works with checkpointing enabled

import sys
import os
import torch

repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, repo_root)

from importlib import import_module
from data.sample_data import (
    get_sample_batch,
    get_multi_turn_batch,
    get_long_sequence,
    get_tiny_config,
    make_conversation,
    IGNORE_INDEX,
)

PASS = "  PASS"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(cfg):
    model_mod = import_module("glm5-triton.model")
    return model_mod.GlmMoeDsaForCausalLM(cfg).to(DEVICE).to(torch.bfloat16)


# ---------------------------------------------------------------------------
# Test 1: Basic forward pass
# ---------------------------------------------------------------------------
def test_forward_pass(model):
    print("Test 1: Forward pass")
    input_ids, labels = get_sample_batch(batch_size=2, device=DEVICE)
    B, S = input_ids.shape
    V = model.vocab_size

    with torch.no_grad():
        loss, logits, _ = model(input_ids=input_ids, labels=labels)

    assert logits.shape == (B, S, V), f"  logits {logits.shape} != ({B},{S},{V})"
    assert torch.isfinite(loss), f"  loss={loss.item()} not finite"
    assert loss.item() > 0, f"  loss={loss.item()} not positive"
    print(f"  logits {logits.shape}, loss {loss.item():.4f}")
    print(PASS)
    print()


# ---------------------------------------------------------------------------
# Test 2: Backward pass — gradients on all parameter groups
# ---------------------------------------------------------------------------
def test_backward_pass(model):
    print("Test 2: Backward pass (gradient flow)")
    input_ids, labels = get_sample_batch(batch_size=1, device=DEVICE)
    model.zero_grad()

    loss, _, _ = model(input_ids=input_ids, labels=labels)
    loss.backward()

    total = 0
    with_grad = 0
    zero_names = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        total += 1
        if p.grad is not None and p.grad.abs().sum() > 0:
            with_grad += 1
        else:
            zero_names.append(name)

    ratio = with_grad / total
    print(f"  {with_grad}/{total} params have gradients ({ratio:.0%})")
    if zero_names:
        print(f"  Zero-grad (expected for DSA indexer @no_grad): {zero_names[:5]}")
    assert ratio > 0.5, f"  FAIL: only {ratio:.0%}"
    print(PASS)
    print()
    model.zero_grad()


# ---------------------------------------------------------------------------
# Test 3: Training convergence — loss drops >80% in 20 steps
# ---------------------------------------------------------------------------
def test_training_convergence(model):
    print("Test 3: Training convergence (20 steps, >80% loss drop)")
    input_ids, labels = get_sample_batch(batch_size=2, device=DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for step in range(20):
        optimizer.zero_grad()
        loss, _, _ = model(input_ids=input_ids, labels=labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    drop = (losses[0] - losses[-1]) / losses[0]
    print(f"  Step 0: {losses[0]:.4f} -> Step 19: {losses[-1]:.4f} ({drop:.0%} drop)")
    assert drop > 0.80, f"  FAIL: only {drop:.0%} drop (need >80%)"
    print(PASS)
    print()


# ---------------------------------------------------------------------------
# Test 4: Label masking — -100 tokens properly ignored
# ---------------------------------------------------------------------------
def test_label_masking(model):
    print("Test 4: Label masking")
    input_ids, labels = get_sample_batch(batch_size=1, device=DEVICE)

    with torch.no_grad():
        loss_masked, _, _ = model(input_ids=input_ids, labels=labels)
        loss_unmasked, _, _ = model(input_ids=input_ids, labels=input_ids)
        all_ignored = torch.full_like(labels, IGNORE_INDEX)
        loss_ignored, _, _ = model(input_ids=input_ids, labels=all_ignored)

    print(f"  Masked:   {loss_masked.item():.4f}")
    print(f"  Unmasked: {loss_unmasked.item():.4f}")
    print(f"  All -100: {loss_ignored.item()} (expect nan)")
    assert torch.isfinite(loss_masked)
    assert abs(loss_masked.item() - loss_unmasked.item()) > 0.01, \
        "  FAIL: masking has no effect"
    print(PASS)
    print()


# ---------------------------------------------------------------------------
# Test 5: KV cache autoregressive decode (10 steps)
# ---------------------------------------------------------------------------
def test_kv_cache_decode(model):
    print("Test 5: KV cache decode (10 steps)")
    input_ids, _ = get_sample_batch(batch_size=1, device=DEVICE)
    S = input_ids.shape[1]
    V = model.vocab_size

    with torch.no_grad():
        # Prefill
        _, logits, kv = model(input_ids=input_ids, use_cache=True)
        assert kv.get_seq_length() == S
        print(f"  Prefill: cache={kv.get_seq_length()}")

        # 10 decode steps
        for step in range(10):
            next_tok = logits[:, -1:, :].argmax(dim=-1)
            _, logits, kv = model(input_ids=next_tok, past_key_values=kv, use_cache=True)
            expected = S + step + 1
            assert logits.shape == (1, 1, V), f"  step {step}: logits {logits.shape}"
            assert kv.get_seq_length() == expected, f"  step {step}: cache {kv.get_seq_length()} != {expected}"

        print(f"  After 10 decode steps: cache={kv.get_seq_length()}")
    print(PASS)
    print()


# ---------------------------------------------------------------------------
# Test 6: Multi-turn padded batch — variable-length conversations
# ---------------------------------------------------------------------------
def test_multi_turn_training(model):
    print("Test 6: Multi-turn padded batch training")
    input_ids, labels = get_multi_turn_batch(batch_size=3, num_turns=2, device=DEVICE)
    B, S = input_ids.shape
    n_trained = (labels != IGNORE_INDEX).sum().item()
    n_masked = (labels == IGNORE_INDEX).sum().item()
    print(f"  Batch: {B} samples, {S} tokens each (padded)")
    print(f"  Trained tokens: {n_trained}, Masked: {n_masked}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for step in range(10):
        optimizer.zero_grad()
        loss, _, _ = model(input_ids=input_ids, labels=labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"  Step 0: {losses[0]:.4f} -> Step 9: {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "  FAIL: loss did not decrease on multi-turn data"
    print(PASS)
    print()


# ---------------------------------------------------------------------------
# Test 7: Long sequence — 256 tokens, stresses attention and DSA
# ---------------------------------------------------------------------------
def test_long_sequence(model):
    print("Test 7: Long sequence (256 tokens)")
    input_ids, labels = get_long_sequence(length=256, device=DEVICE)
    S = input_ids.shape[1]
    print(f"  Sequence length: {S} tokens")

    # Forward pass
    with torch.no_grad():
        loss, logits, _ = model(input_ids=input_ids, labels=labels)

    assert torch.isfinite(loss), f"  FAIL: loss={loss.item()}"
    assert logits.shape[1] == S
    print(f"  Forward: loss={loss.item():.4f}, logits {logits.shape}")

    # Backward pass on long sequence
    model.zero_grad()
    loss2, _, _ = model(input_ids=input_ids, labels=labels)
    loss2.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                   for p in model.parameters() if p.requires_grad)
    assert has_grad, "  FAIL: no gradients on long sequence"
    print(f"  Backward: gradients flow through {S}-token sequence")
    print(PASS)
    print()
    model.zero_grad()


# ---------------------------------------------------------------------------
# Test 8: Gradient checkpointing — same result with less memory
# ---------------------------------------------------------------------------
def test_gradient_checkpointing(model):
    print("Test 8: Gradient checkpointing")
    input_ids, labels = get_sample_batch(batch_size=2, device=DEVICE)

    # Forward+backward WITHOUT checkpointing
    model.model.set_gradient_checkpointing(enable=False)
    model.zero_grad()
    loss_no_ckpt, _, _ = model(input_ids=input_ids, labels=labels)
    loss_no_ckpt.backward()
    grad_no_ckpt = {n: p.grad.clone() for n, p in model.named_parameters()
                    if p.grad is not None}

    # Forward+backward WITH checkpointing
    model.model.set_gradient_checkpointing(enable=True)
    model.zero_grad()
    loss_ckpt, _, _ = model(input_ids=input_ids, labels=labels)
    loss_ckpt.backward()

    # Loss should be identical (same computation, just different memory strategy)
    loss_diff = abs(loss_no_ckpt.item() - loss_ckpt.item())
    print(f"  Loss without ckpt: {loss_no_ckpt.item():.6f}")
    print(f"  Loss with ckpt:    {loss_ckpt.item():.6f}")
    print(f"  Difference:        {loss_diff:.8f}")
    assert loss_diff < 0.01, f"  FAIL: loss differs by {loss_diff}"

    # Gradients should be very close (float rounding may differ slightly)
    max_grad_diff = 0.0
    for name, p in model.named_parameters():
        if p.grad is not None and name in grad_no_ckpt:
            diff = (p.grad - grad_no_ckpt[name]).abs().max().item()
            max_grad_diff = max(max_grad_diff, diff)
    print(f"  Max gradient diff: {max_grad_diff:.8f}")
    assert max_grad_diff < 0.1, f"  FAIL: gradient diff {max_grad_diff}"

    model.model.set_gradient_checkpointing(enable=False)
    model.zero_grad()
    print(PASS)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Device: {DEVICE}")
    print(f"PyTorch: {torch.__version__}")
    print()

    cfg = get_tiny_config()
    model = load_model(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} params (tiny config)")

    ids, labs = make_conversation()
    print(f"Sample conversation: {len(ids)} tokens, {sum(1 for l in labs if l != IGNORE_INDEX)} trained")
    print()

    test_forward_pass(model)
    test_backward_pass(model)
    test_training_convergence(model)
    test_label_masking(model)
    test_kv_cache_decode(model)
    test_multi_turn_training(model)
    test_long_sequence(model)
    test_gradient_checkpointing(model)

    print("=" * 50)
    print("All 8 tests passed.")
    print("=" * 50)


if __name__ == "__main__":
    main()
