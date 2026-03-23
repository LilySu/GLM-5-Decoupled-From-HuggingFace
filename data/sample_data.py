# Shared sample data for validating GLM-5 forward/backward passes.
#
# This module provides synthetic token sequences that follow GLM-5's ChatML
# format. The token IDs are fake (the model has random weights anyway) but
# the structure is correct: special token boundaries, user/assistant turns,
# and properly masked labels.
#
# Used by both:
#   - glm5-triton/validate.py
#   - glm5-raw-decoupled-from-hf/validate.py
#
# GLM-5 ChatML format:
#   <|im_start|>system\n{system_message}\n<|im_end|>\n
#   <|im_start|>user\n{user_message}\n<|im_end|>\n
#   <|im_start|>assistant\n{assistant_message}\n<|im_end|>
#
# For training, labels = -100 on everything except the assistant response,
# so the model only learns to predict what the assistant says.

import torch

# ---------------------------------------------------------------------------
# Fake special token IDs (matching ChatML structure)
# In a real model these come from the tokenizer vocabulary.
# ---------------------------------------------------------------------------
BOS_ID = 0           # <|endoftext|> / beginning of sequence
EOS_ID = 1           # <|endoftext|> / end of sequence
IM_START_ID = 10     # <|im_start|>
IM_END_ID = 11       # <|im_end|>
NEWLINE_ID = 12      # \n

# Role token IDs (in a real tokenizer these would be actual token IDs)
SYSTEM_ID = 20       # "system"
USER_ID = 21         # "user"
ASSISTANT_ID = 22    # "assistant"

# Content token IDs (arbitrary tokens representing words)
# We use a range that's valid for small test vocab sizes (256+)
CONTENT_TOKENS = list(range(100, 200))

IGNORE_INDEX = -100  # PyTorch cross_entropy ignore value


def _make_content(length):
    """Generate a sequence of fake content tokens of given length."""
    return [CONTENT_TOKENS[i % len(CONTENT_TOKENS)] for i in range(length)]


def make_chat_turn(role_id, content_length):
    """Build one ChatML turn: <|im_start|> role \\n content \\n <|im_end|> \\n

    Returns (token_ids, is_assistant) where is_assistant marks which tokens
    are part of the assistant response (for label masking).
    """
    content = _make_content(content_length)
    tokens = [IM_START_ID, role_id, NEWLINE_ID] + content + [NEWLINE_ID, IM_END_ID, NEWLINE_ID]
    is_assistant = [role_id == ASSISTANT_ID] * len(tokens)
    return tokens, is_assistant


def make_conversation(system_len=3, user_len=8, assistant_len=6):
    """Build a single-turn ChatML conversation (system + user + assistant).

    Returns:
        input_ids: list[int] — the full token sequence
        labels:    list[int] — same as input_ids but with -100 on non-assistant tokens
    """
    sys_tokens, sys_mask = make_chat_turn(SYSTEM_ID, system_len)
    usr_tokens, usr_mask = make_chat_turn(USER_ID, user_len)
    ast_tokens, ast_mask = make_chat_turn(ASSISTANT_ID, assistant_len)

    input_ids = sys_tokens + usr_tokens + ast_tokens + [EOS_ID]
    is_assistant = sys_mask + usr_mask + ast_mask + [False]

    labels = []
    for tok, is_ast in zip(input_ids, is_assistant):
        if is_ast:
            labels.append(tok)
        elif tok == EOS_ID and input_ids[-1] == EOS_ID:
            labels.append(tok)
        else:
            labels.append(IGNORE_INDEX)

    return input_ids, labels


def make_multi_turn_conversation(num_turns=3, system_len=3, user_len=8, assistant_len=6):
    """Build a multi-turn ChatML conversation.

    Multiple user/assistant exchanges after a single system prompt.
    Only assistant turns are trained on (labels != -100).

    Returns:
        input_ids: list[int]
        labels:    list[int]
    """
    all_tokens = []
    all_masks = []

    # System prompt (once)
    sys_tokens, sys_mask = make_chat_turn(SYSTEM_ID, system_len)
    all_tokens.extend(sys_tokens)
    all_masks.extend(sys_mask)

    # Multiple user/assistant turns
    for turn in range(num_turns):
        # Vary content length per turn to stress variable-length handling
        u_len = user_len + turn * 2
        a_len = assistant_len + turn * 3
        usr_tokens, usr_mask = make_chat_turn(USER_ID, u_len)
        ast_tokens, ast_mask = make_chat_turn(ASSISTANT_ID, a_len)
        all_tokens.extend(usr_tokens)
        all_masks.extend(usr_mask)
        all_tokens.extend(ast_tokens)
        all_masks.extend(ast_mask)

    all_tokens.append(EOS_ID)
    all_masks.append(False)

    labels = []
    for tok, is_ast in zip(all_tokens, all_masks):
        if is_ast:
            labels.append(tok)
        elif tok == EOS_ID and all_tokens[-1] == EOS_ID:
            labels.append(tok)
        else:
            labels.append(IGNORE_INDEX)

    return all_tokens, labels


def pad_and_batch(sequences, pad_id=0):
    """Pad a list of variable-length token lists to the same length.

    Returns:
        padded: [B, max_len] long tensor
        mask:   [B, max_len] bool tensor (True = real token, False = padding)
    """
    max_len = max(len(s) for s in sequences)
    padded = []
    mask = []
    for s in sequences:
        pad_len = max_len - len(s)
        padded.append(s + [pad_id] * pad_len)
        mask.append([True] * len(s) + [False] * pad_len)
    return padded, mask


def get_sample_batch(batch_size=2, device="cuda", dtype=torch.long):
    """Create a batch of identical single-turn conversations as tensors.

    All conversations have the same length (no padding needed).

    Returns:
        input_ids: [B, S] long tensor
        labels:    [B, S] long tensor (-100 for masked positions)
    """
    batch_input_ids = []
    batch_labels = []
    for _ in range(batch_size):
        ids, labs = make_conversation()
        batch_input_ids.append(ids)
        batch_labels.append(labs)

    return (
        torch.tensor(batch_input_ids, dtype=dtype, device=device),
        torch.tensor(batch_labels, dtype=dtype, device=device),
    )


def get_multi_turn_batch(batch_size=2, num_turns=3, device="cuda", dtype=torch.long):
    """Create a batch of multi-turn conversations, padded to equal length.

    Each conversation has a different total length (varying content per turn),
    so padding is applied. Labels are -100 on both padding and non-assistant tokens.

    Returns:
        input_ids: [B, S] long tensor (right-padded with 0)
        labels:    [B, S] long tensor (-100 on padding + non-assistant tokens)
    """
    all_ids = []
    all_labs = []
    for i in range(batch_size):
        # Vary turns per sample to create different lengths
        turns = num_turns + (i % 2)
        ids, labs = make_multi_turn_conversation(num_turns=turns)
        all_ids.append(ids)
        all_labs.append(labs)

    max_len = max(len(s) for s in all_ids)
    padded_ids = []
    padded_labs = []
    for ids, labs in zip(all_ids, all_labs):
        pad_len = max_len - len(ids)
        padded_ids.append(ids + [BOS_ID] * pad_len)
        padded_labs.append(labs + [IGNORE_INDEX] * pad_len)

    return (
        torch.tensor(padded_ids, dtype=dtype, device=device),
        torch.tensor(padded_labs, dtype=dtype, device=device),
    )


def get_long_sequence(length=256, device="cuda", dtype=torch.long):
    """Create a single long sequence to stress-test attention and memory.

    Builds a multi-turn conversation that reaches approximately `length` tokens.

    Returns:
        input_ids: [1, ~length] long tensor
        labels:    [1, ~length] long tensor
    """
    all_tokens = []
    all_masks = []

    # System prompt
    sys_tokens, sys_mask = make_chat_turn(SYSTEM_ID, 5)
    all_tokens.extend(sys_tokens)
    all_masks.extend(sys_mask)

    # Keep adding turns until we hit the target length
    turn = 0
    while len(all_tokens) < length - 20:
        u_len = 10 + (turn * 3) % 20
        a_len = 8 + (turn * 5) % 15
        usr_tokens, usr_mask = make_chat_turn(USER_ID, u_len)
        ast_tokens, ast_mask = make_chat_turn(ASSISTANT_ID, a_len)
        all_tokens.extend(usr_tokens)
        all_masks.extend(usr_mask)
        all_tokens.extend(ast_tokens)
        all_masks.extend(ast_mask)
        turn += 1

    all_tokens.append(EOS_ID)
    all_masks.append(False)

    labels = []
    for tok, is_ast in zip(all_tokens, all_masks):
        labels.append(tok if is_ast else IGNORE_INDEX)
    # Train on final EOS
    labels[-1] = EOS_ID

    return (
        torch.tensor([all_tokens], dtype=dtype, device=device),
        torch.tensor([labels], dtype=dtype, device=device),
    )


def get_tiny_config():
    """Return a minimal GLM-5 config that fits on any GPU.

    Uses the same config keys as the real model but with tiny dimensions
    so validation can run in seconds with <1GB memory.
    """
    return {
        "vocab_size": 256,
        "hidden_size": 128,
        "tie_word_embeddings": False,
        "num_hidden_layers": 2,
        "intermediate_size": 256,
        "num_attention_heads": 4,
        "num_key_value_heads": 4,
        "attention_bias": False,
        "attention_dropout": 0.0,
        "q_lora_rank": 64,
        "kv_lora_rank": 32,
        "qk_rope_head_dim": 16,
        "qk_nope_head_dim": 16,
        "qk_head_dim": 32,
        "v_head_dim": 32,
        "n_routed_experts": 4,
        "n_shared_experts": 1,
        "num_experts_per_tok": 2,
        "moe_intermediate_size": 64,
        "routed_scaling_factor": 2.5,
        "n_group": 1,
        "topk_group": 1,
        "norm_topk_prob": True,
        "index_topk": 8,
        "index_head_dim": 32,
        "index_n_heads": 2,
        "hidden_act": "silu",
        "rms_norm_eps": 1e-5,
        "max_position_embeddings": 512,
        "rope_theta": 10000.0,
        "initializer_range": 0.02,
        "pad_token_id": None,
        "bos_token_id": 0,
        "eos_token_id": 1,
        "use_cache": True,
        "mlp_layer_types": ["dense", "sparse"],
    }


# ---------------------------------------------------------------------------
# Quick self-test when run directly
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    ids, labs = make_conversation()
    print("Single-turn conversation:")
    print(f"  {len(ids)} tokens, {sum(1 for l in labs if l != IGNORE_INDEX)} trained")
    print()

    ids, labs = make_multi_turn_conversation(num_turns=3)
    print("Multi-turn conversation (3 turns):")
    print(f"  {len(ids)} tokens, {sum(1 for l in labs if l != IGNORE_INDEX)} trained")
    print()

    b_ids, b_labs = get_multi_turn_batch(batch_size=3, device="cpu")
    print(f"Multi-turn batch (3 samples, padded): {b_ids.shape}")
    print()

    l_ids, l_labs = get_long_sequence(length=256, device="cpu")
    print(f"Long sequence: {l_ids.shape}, {sum(1 for l in l_labs[0].tolist() if l != IGNORE_INDEX)} trained")
