# Validation script for glm5-raw-decoupled-from-hf: proves data runs through correctly.
#
# Uses the same shared data as glm5-triton/validate.py (from data/sample_data.py).
#
# Run from the glm5/ root directory:
#   python glm5-raw-decoupled-from-hf/validate.py

import sys
import os
import torch

# Add repo root and this directory to path
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
this_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, repo_root)
sys.path.insert(0, this_dir)

from data.sample_data import get_sample_batch, get_tiny_config, make_conversation, IGNORE_INDEX
from model import GlmMoeDsaForCausalLM


def load_model(cfg, device="cuda"):
    """Instantiate a tiny GLM-5 model for validation."""
    model = GlmMoeDsaForCausalLM(cfg).to(device).to(torch.bfloat16)
    return model


def test_forward_pass(model, device):
    """Test 1: Forward pass produces finite loss and correct logit shape."""
    print("Test 1: Forward pass")
    input_ids, labels = get_sample_batch(batch_size=2, device=device)
    vocab_size = model.vocab_size
    batch_size, seq_len = input_ids.shape

    with torch.no_grad():
        loss, logits, kv = model(input_ids=input_ids, labels=labels)

    assert logits.shape == (batch_size, seq_len, vocab_size)
    assert torch.isfinite(loss)
    print(f"  logits: {logits.shape}, loss: {loss.item():.4f}")
    print(f"  PASS")
    print()


def test_training_loop(model, device):
    """Test 2: Loss decreases over 10 steps on the same batch."""
    print("Test 2: Training loop (loss should decrease)")
    input_ids, labels = get_sample_batch(batch_size=2, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    losses = []
    for step in range(10):
        optimizer.zero_grad()
        loss, _, _ = model(input_ids=input_ids, labels=labels)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"  Step 0: {losses[0]:.4f} -> Step 9: {losses[-1]:.4f}")
    assert losses[-1] < losses[0], "  FAIL: loss did not decrease"
    print(f"  PASS")
    print()


def test_kv_cache_decode(model, device):
    """Test 3: Autoregressive decoding with KV cache."""
    print("Test 3: KV cache decode")
    input_ids, _ = get_sample_batch(batch_size=1, device=device)

    with torch.no_grad():
        _, logits, kv = model(input_ids=input_ids, use_cache=True)
        print(f"  Prefill: logits {logits.shape}, cache len={kv.get_seq_length()}")

        for step in range(3):
            next_token = logits[:, -1:, :].argmax(dim=-1)
            _, logits, kv = model(input_ids=next_token, past_key_values=kv, use_cache=True)
            print(f"  Decode {step+1}: logits {logits.shape}, cache len={kv.get_seq_length()}")

    print(f"  PASS")
    print()


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Validating: glm5-raw-decoupled-from-hf")
    print()

    cfg = get_tiny_config()
    model = load_model(cfg, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_params:,} parameters")
    print()

    test_forward_pass(model, device)
    test_training_loop(model, device)
    test_kv_cache_decode(model, device)

    print("=" * 50)
    print("All tests passed. Raw model is functional.")
    print("=" * 50)


if __name__ == "__main__":
    main()
