"""Debug: test a single decoder layer in eager mode with random weights."""
import os
import sys
import torch

os.environ["GLM5_FORCE_EAGER"] = "1"

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Create symlinks if needed
src = os.path.join(ROOT, "glm5-kernels-flashmla-deepgemm")
dst = os.path.join(ROOT, "glm5_kernels_flashmla_deepgemm")
if os.path.isdir(src) and not os.path.exists(dst):
    os.symlink(src, dst)

from glm5_kernels_flashmla_deepgemm.config import GLM_MOE_DSA_CONFIG
from glm5_kernels_flashmla_deepgemm.model import DecoderLayer
from glm5_kernels_flashmla_deepgemm.rope_partial import RotaryEmbedding

device = torch.device("cuda")

# Test 1: Dense layer
print("=== Test 1: Dense layer (eager mode) ===")
cfg = dict(GLM_MOE_DSA_CONFIG)
cfg["num_hidden_layers"] = 1
cfg["mlp_layer_types"] = ["dense"]
cfg["use_flash_mla"] = False
cfg["use_deepgemm"] = False

layer = DecoderLayer(cfg, layer_idx=0).to(device).bfloat16().eval()
rope = RotaryEmbedding(cfg).to(device)

B, S, T = 1, 128, 128
hidden = torch.randn(B, S, 6144, dtype=torch.bfloat16, device=device)
pos_ids = torch.arange(T - S, T, device=device).unsqueeze(0)
cos, sin = rope(hidden, pos_ids)
mask = torch.full((S, T), float("-inf"), device=device, dtype=torch.bfloat16)
mask = torch.triu(mask, diagonal=1).unsqueeze(0).unsqueeze(0)

with torch.no_grad():
    out = layer(hidden, attention_mask=mask, position_embeddings=(cos, sin))
print(f"  SUCCESS: output shape = {out[0].shape}")

torch.cuda.empty_cache()

# Test 2: MoE/sparse layer
print("\n=== Test 2: MoE/sparse layer (eager mode) ===")
cfg2 = dict(GLM_MOE_DSA_CONFIG)
cfg2["num_hidden_layers"] = 1
cfg2["mlp_layer_types"] = ["sparse"]
cfg2["use_flash_mla"] = False
cfg2["use_deepgemm"] = False

try:
    layer2 = DecoderLayer(cfg2, layer_idx=0).to(device).bfloat16().eval()
    rope2 = RotaryEmbedding(cfg2).to(device)
    hidden2 = torch.randn(B, S, 6144, dtype=torch.bfloat16, device=device)
    cos2, sin2 = rope2(hidden2, pos_ids)
    with torch.no_grad():
        out2 = layer2(hidden2, attention_mask=mask, position_embeddings=(cos2, sin2))
    print(f"  SUCCESS: output shape = {out2[0].shape}")
except Exception as e:
    print(f"  FAILED: {e}")

print("\n=== Done ===")
