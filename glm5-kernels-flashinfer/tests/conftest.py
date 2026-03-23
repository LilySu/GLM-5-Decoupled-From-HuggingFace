"""Shared test utilities for glm5-kernels-flashinfer tests."""

import sys
import os
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Package name for this implementation
PKG = "glm5-kernels-flashinfer"


def assert_close(name, a, b, atol=1e-5, rtol=1e-4):
    if a.shape != b.shape:
        print(f"  FAIL {name}: shape mismatch {a.shape} vs {b.shape}")
        return False
    max_diff = (a.float() - b.float()).abs().max().item()
    if torch.allclose(a.float(), b.float(), atol=atol, rtol=rtol):
        print(f"  PASS {name} (max_diff={max_diff:.2e})")
        return True
    else:
        print(f"  FAIL {name} (max_diff={max_diff:.2e}, atol={atol}, rtol={rtol})")
        return False


def make_cfg(num_layers=2):
    return {
        "vocab_size": 256, "hidden_size": 128, "tie_word_embeddings": False,
        "num_hidden_layers": num_layers, "intermediate_size": 256,
        "num_attention_heads": 4, "num_key_value_heads": 4,
        "attention_bias": False, "attention_dropout": 0.0,
        "q_lora_rank": 64, "kv_lora_rank": 32,
        "qk_rope_head_dim": 8, "qk_nope_head_dim": 24, "qk_head_dim": 32, "v_head_dim": 32,
        "n_routed_experts": 8, "n_shared_experts": 1, "num_experts_per_tok": 2,
        "moe_intermediate_size": 64, "routed_scaling_factor": 2.5,
        "n_group": 1, "topk_group": 1, "norm_topk_prob": True,
        "index_topk": 16, "index_head_dim": 16, "index_n_heads": 4,
        "hidden_act": "silu", "rms_norm_eps": 1e-5,
        "max_position_embeddings": 512, "rope_theta": 10000.0,
        "initializer_range": 0.02,
        "pad_token_id": None, "bos_token_id": 0, "eos_token_id": 1,
        "use_cache": True,
        "mlp_layer_types": ["dense"] + ["sparse"] * (num_layers - 1),
    }


def has_cuda():
    return torch.cuda.is_available()

def has_sm90():
    if not has_cuda():
        return False
    props = torch.cuda.get_device_properties(0)
    return props.major == 9 and props.minor == 0

def has_flashinfer():
    try:
        import flashinfer
        return True
    except ImportError:
        return False

def has_deep_gemm():
    try:
        import deep_gemm
        return True
    except ImportError:
        return False

def skip_no_cuda(fn):
    def wrapper(*args, **kwargs):
        if not has_cuda():
            print(f"  SKIP {fn.__name__} (no CUDA)")
            return True
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper

def skip_no_sm90(fn):
    def wrapper(*args, **kwargs):
        if not has_sm90():
            print(f"  SKIP {fn.__name__} (no SM90 GPU)")
            return True
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper

def jaccard(a, b):
    sa, sb = set(a.tolist()), set(b.tolist())
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)

def cosine_sim(a, b):
    a_f, b_f = a.float().flatten(), b.float().flatten()
    return (torch.dot(a_f, b_f) / (a_f.norm() * b_f.norm() + 1e-12)).item()


def skip_no_multi_gpu(fn):
    def wrapper(*args, **kwargs):
        if not has_cuda() or torch.cuda.device_count() < 2:
            print(f"  SKIP {fn.__name__} (need >= 2 GPUs)")
            return True
        return fn(*args, **kwargs)
    wrapper.__name__ = fn.__name__
    return wrapper


def cuda_timer_fn(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record(); fn(); end.record(); torch.cuda.synchronize()
        times.append(start.elapsed_time(end))
    return sorted(times)


def make_full_cfg():
    return {
        "vocab_size": 154880, "hidden_size": 6144, "tie_word_embeddings": False,
        "num_hidden_layers": 78, "intermediate_size": 12288,
        "num_attention_heads": 64, "num_key_value_heads": 64,
        "attention_bias": False, "attention_dropout": 0.0,
        "q_lora_rank": 2048, "kv_lora_rank": 512,
        "qk_rope_head_dim": 64, "qk_nope_head_dim": 192, "qk_head_dim": 256, "v_head_dim": 256,
        "n_routed_experts": 256, "n_shared_experts": 1, "num_experts_per_tok": 8,
        "moe_intermediate_size": 2048, "routed_scaling_factor": 2.5,
        "n_group": 1, "topk_group": 1, "norm_topk_prob": True,
        "index_topk": 2048, "index_head_dim": 128, "index_n_heads": 32,
        "hidden_act": "silu", "rms_norm_eps": 1e-5,
        "max_position_embeddings": 202752, "rope_theta": 10000.0,
        "initializer_range": 0.02,
        "pad_token_id": None, "bos_token_id": 0, "eos_token_id": 1,
        "use_cache": True,
        "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
    }
