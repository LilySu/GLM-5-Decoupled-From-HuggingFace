"""Fix dsa_indexer.py to use correct DeepGEMM fp8_mqa_logits API.

The fix: replace quantize_activations_deepgemm (which gives 2D scales)
with manual per-row scaling (which gives 1D scales as required).
"""
import os

path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "glm5-kernels-flashmla-deepgemm",
    "dsa_indexer.py",
)

txt = open(path).read()

old = """        # Quantize to FP8
        q_fp8 = q_2d.to(torch.float8_e4m3fn)
        from .fp8_utils import quantize_activations_deepgemm
        k_fp8, k_scales = quantize_activations_deepgemm(k_2d)"""

new = """        # Quantize to FP8
        # q: raw FP8 tensor (NOT tuple) — confirmed by debug_all_kernels2.py
        q_fp8 = q_2d.to(torch.float8_e4m3fn)
        # kv: (FP8 tensor, 1D scales [seq_kv]) — scales MUST be 1D
        k_fp8 = k_2d.to(torch.float8_e4m3fn)
        k_scales = k_2d.abs().amax(dim=-1).float() / 448.0  # 1D [T]"""

if old in txt:
    txt = txt.replace(old, new)
    open(path, "w").write(txt)
    print(f"Fixed: {path}")
else:
    print("Pattern not found — file may already be fixed or has different formatting.")
    print("Searching for the line manually...")
    for i, line in enumerate(txt.split("\n"), 1):
        if "quantize_activations_deepgemm" in line:
            print(f"  Line {i}: {line.strip()}")
