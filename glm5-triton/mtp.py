# MTP (Multi-Token Prediction) — speculative decoding draft model.
#
# GLM-5 uses MTP with parameter sharing across 3 draft layers to predict
# multiple future tokens during inference. During training, the MTP head
# provides an auxiliary loss that improves base model quality.
#
# STATUS: Stub — NOT implemented in the raw model, NOT available as a kernel.
#
# Architecture (from paper):
#   - 3 MTP layers that SHARE parameters (single set of weights, run 3 times)
#   - Each MTP layer: embedding + transformer block + output projection
#   - Training: predict next n tokens with shared MTP layers, auxiliary loss
#   - Inference: acts as draft model for speculative decoding (4 speculative steps)
#   - Acceptance length: 2.76 tokens (vs DeepSeek-V3.2's 2.55)
#
# Key design choice: DeepSeek-V3 uses 1 MTP layer (predicts 2 tokens).
# GLM-5 shares params across 3 MTP layers, keeping memory cost identical
# to a single layer but increasing acceptance rate.
#
# Paper ref: GLM-5 (arXiv 2602.15763v2), Section 2.1 — "Multi-token Prediction
# with Parameter Sharing"
#
# To implement, you would need:
#   1. A shared transformer block (same weights applied 3 times)
#   2. An embedding projection for each draft position
#   3. An output head for each draft position
#   4. A training loss that sums CE loss across all draft positions
#   5. For inference: integration with speculative decoding loop

import torch
import torch.nn as nn


class MTPHead(nn.Module):
    """Multi-Token Prediction head with parameter sharing.

    This is a STUB. The architecture is described but not implemented.
    The raw GLM-5 model (glm5-raw-decoupled-from-hf) does not include MTP.
    """

    def __init__(self, cfg):
        super().__init__()
        self.num_mtp_layers = 1  # Single set of shared parameters
        self.num_speculative_steps = 3  # Applied 3 times with shared weights

        raise NotImplementedError(
            "MTP is not implemented. The GLM-5 paper describes Multi-Token "
            "Prediction with parameter sharing across 3 draft layers, but "
            "this component is not present in the released model weights or "
            "the raw decoupled implementation. See paper Section 2.1."
        )

    def forward(self, hidden_states, labels=None):
        raise NotImplementedError("MTP forward not implemented.")

    def speculative_decode(self, hidden_states, num_steps=4):
        """Generate draft tokens for speculative decoding.

        Would produce `num_steps` draft tokens using the shared MTP layer,
        then the main model verifies them in a single forward pass.
        """
        raise NotImplementedError("MTP speculative decoding not implemented.")
