# =============================================================================
# src/xlstm_layer.py
#
# Thin, READABLE wrapper around the OFFICIAL xLSTM mLSTM layer.
#
# Why a wrapper instead of using mLSTMLayer directly?
#   * We need one stable interface: .forward(x) for training (parallel) and
#     .step(x, state) for generation (recurrent, state carried across tokens).
#     The real layer already exposes both, so the wrapper just unifies them.
#   * We need an "identity-safe" init so that at step 0 the whole model still
#     equals the frozen base (so we don't break Qwen on insertion). See
#     init_identity() below -- it zeros the output (down) projection so the new
#     branch contributes ~0 until training teaches it to add memory.
#   * We keep the RMSNorm OUT of this file; the patcher applies it per-sublayer
#     so every sublayer (attn / xlstm / ffn) is shaped identically.
#
# We import the REAL class -- no reimplementation:
#     from xlstm.blocks.mlstm.layer import mLSTMLayer, mLSTMLayerConfig
# =============================================================================

from dataclasses import dataclass

import torch
import torch.nn as nn

# --- the real xLSTM package (do NOT reimplement mLSTM) ---
from xlstm.blocks.mlstm.layer import mLSTMLayer, mLSTMLayerConfig


@dataclass
class XLSTMLayerConfig:
    """Config for OUR wrapper. Forwards only the dims the real layer needs.

    embedding_dim : must equal the Qwen hidden size (896 for 0.5B).
    num_heads     : mLSTM heads (paper default 4; bump for more capacity).
    context_length : max sequence length the recurrent kernel unrolls for.
    proj_factor   : inner up-projection multiplier (paper default 2.0).
    conv1d_kernel : causal conv kernel (paper default 4).
    """

    embedding_dim: int = 896
    num_heads: int = 4
    context_length: int = 2048
    proj_factor: float = 2.0
    conv1d_kernel: int = 4
    bias: bool = False
    dropout: float = 0.0

    def to_real_config(self) -> mLSTMLayerConfig:
        """Build the underlying xlstm.mLSTMLayerConfig from our fields."""
        return mLSTMLayerConfig(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            context_length=self.context_length,
            proj_factor=self.proj_factor,
            conv1d_kernel_size=self.conv1d_kernel,
            bias=self.bias,
            dropout=self.dropout,
            _num_blocks=1,  # one standalone layer, not a stack
        )


class XLSTMLayer(nn.Module):
    """Our mLSTM sublayer = real xlstm.mLSTMLayer + identity-safe controls.

    forward (parallel, training):
        y = real_layer(x)            # (B, S, D) -> (B, S, D)
    step (recurrent, generation):
        y, state = real_layer.step(x, mlstm_state=..., conv_state=...)
        # state is carried across tokens and fed back on the next call.
    """

    def __init__(self, config: XLSTMLayerConfig):
        super().__init__()
        self.config = config
        # --- the only thing we instantiate: the real layer ---
        self.mlstm = mLSTMLayer(config=config.to_real_config())

    # ---- training forward: parallel over the whole sequence ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlstm(x)

    # ---- generation step: one token at a time, state carried across ----
    # The real layer already implements state carry via .step(); we just forward
    # the state dict back and return it so the caller can hold it between steps.
    def step(self, x: torch.Tensor, state=None):
        mlstm_state = state.get("mlstm_state") if state else None
        conv_state = state.get("conv_state") if state else None
        y, new_state = self.mlstm.step(
            x, mlstm_state=mlstm_state, conv_state=conv_state
        )
        # new_state = {"mlstm_state": (...), "conv_state": (...)}
        return y, new_state

    # ---- IDENTITY-SAFE INIT ----
    # Zero the DOWN projection so the mLSTM branch contributes ~0 at the start.
    # Because the patcher adds this branch residually, the model then equals the
    # frozen base at step 0 -> we don't break Qwen on insert. Training then
    # lifts the down-proj off zero to "add memory".
    def init_identity(self):
        nn.init.zeros_(self.mlstm.proj_down.weight)
        if self.mlstm.proj_down.bias is not None:
            nn.init.zeros_(self.mlstm.proj_down.bias)
        # also zero the learnable skip so the conv-skip path starts at 0
        nn.init.zeros_(self.mlstm.learnable_skip)

    # convenience: number of trainable params in this sublayer (for logging)
    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
