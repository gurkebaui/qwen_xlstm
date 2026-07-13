# =============================================================================
# src/xlstm_layer.py
#
# Thin, READABLE wrapper around the OFFICIAL xLSTM package.
#
# Supports BOTH memory cells the paper introduces:
#   * block_type="mlstm"  -> mLSTMLayer  (matrix memory, parallel-trainable,
#                                       the "training speed-up insert" Henry
#                                       found it really is)
#   * block_type="slstm"  -> sLSTMLayer  (scalar memory + memory MIXING, the
#                                       actual recurrent-long-context cell)
#
# Why a wrapper instead of using the layers directly?
#   * ONE stable interface: .forward(x) for training (parallel) and
#     .step(x, state) for generation (recurrent, state carried across tokens).
#     The real layers already expose both; we just unify the state dict so the
#     patcher (recurrent_mode, RoPE fix, generate_step) never has to know
#     which cell is inside.
#   * IDENTITY-SAFE init: zero the output projection so at step 0 the inserted
#     branch contributes ~0 and the model == frozen base. See init_identity().
#   * We import the REAL classes -- NO reimplementation:
#         from xlstm.blocks.mlstm.layer import mLSTMLayer, mLSTMLayerConfig
#         from xlstm.blocks.slstm.layer import sLSTMLayer, sLSTMLayerConfig
#
# State contract (normalized, cell-agnostic):
#     state = {"conv_state": <tensor|None>, "recurrent_state": <tensor|None>}
# mLSTM stores its matrix memory under "recurrent_state" (mapped from the
# layer's "mlstm_state"); sLSTM stores its (c, n, m, h) states there too.
# =============================================================================

from dataclasses import dataclass, field

import torch
import torch.nn as nn

# --- the real xLSTM package (do NOT reimplement the cells) ---
from xlstm.blocks.mlstm.layer import mLSTMLayer, mLSTMLayerConfig
from xlstm.blocks.slstm.layer import sLSTMLayer, sLSTMLayerConfig


@dataclass
class XLSTMLayerConfig:
    """Config for OUR wrapper. Forwards only the dims each real layer needs.

    block_type : "mlstm" or "slstm" (the paper's two cells).
    embedding_dim : must equal the Qwen hidden size (896 for 0.5B).
    num_heads     : sLSTM/mLSTM heads (paper default 4; must divide 896).
    context_length : max sequence length the mLSTM recurrent kernel unrolls
                     for. IGNORED by sLSTM (it is genuinely recurrent, no
                     unroll). Kept here so the same config works for both.
    proj_factor   : mLSTM inner up-projection multiplier (paper default 2.0).
    conv1d_kernel : causal conv kernel (paper default 4). 0 = no conv.
    bias_init     : sLSTM forget-gate bias init. "powerlaw_blockdependent"
                    (paper default) ramps the forget bias so the cell starts
                    in "remember" mode -- this is the paper's gate-init recipe.
                    Ignored by mLSTM.
    architecture  : ratio hint for the model builder ("1:0" = all sLSTM,
                    "7:1" = every 8th block sLSTM, rest mLSTM). The wrapper
                    itself only cares about block_type; the model uses this
                    to pick block_type per Qwen layer.
    """

    block_type: str = "mlstm"          # "mlstm" | "slstm"
    embedding_dim: int = 896
    num_heads: int = 4
    context_length: int = 2048
    proj_factor: float = 2.0
    conv1d_kernel: int = 4
    bias: bool = False
    dropout: float = 0.0
    # sLSTM-specific (ignored by mLSTM):
    bias_init: str = "powerlaw_blockdependent"
    # ratio hint for the model builder (not used by the wrapper directly):
    architecture: str = "1:0"
    # HYBRID context split (2026-07-13): if set, the FROZEN base attention
    # sees only local windows of `base_ctx` tokens (cheap, O(L*base_ctx)),
    # while the trainable sLSTM processes the FULL sequence (O(L) parallel
    # scan) -> global recurrent memory at a fraction of the base-attn cost.
    # None = base sees the full sequence (original behaviour).
    base_ctx: int = None

    def to_mlstm_config(self) -> mLSTMLayerConfig:
        """Build the underlying xlstm.mLSTMLayerConfig."""
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

    def to_slstm_config(self) -> sLSTMLayerConfig:
        """Build the underlying xlstm.sLSTMLayerConfig.

        Forget-gate bias is initialized via bias_init (paper recipe:
        powerlaw_blockdependent -> forget bias ramps +5..-7 so the cell
        remembers by default). _block_idx=0, _num_blocks=1 because this is
        a single standalone graft layer, not a residual stack.
        """
        return sLSTMLayerConfig(
            embedding_dim=self.embedding_dim,
            num_heads=self.num_heads,
            conv1d_kernel_size=self.conv1d_kernel,
            group_norm_weight=True,
            dropout=self.dropout,
            bias_init=self.bias_init,
            recurrent_weight_init="zeros",
            backend="vanilla",         # pure-Torch sLSTM (no CUDA compile).
                                        # The xlstm CUDA kernel fails to build
                                        # on this box (CUDA 12.1 + Fedora's
                                        # very new GCC -> host_config.h
                                        # "unterminated #if"). Vanilla is
                                        # correct, just a bit slower/step.
            _block_idx=0,
            _num_blocks=1,
        )


class XLSTMLayer(nn.Module):
    """Our xLSTM sublayer = real xlstm layer + identity-safe controls.

    Works for BOTH mLSTM and sLSTM via `config.block_type`. The recurrent
    state is normalized to {"conv_state", "recurrent_state"} so the patcher's
    generate_step / RoPE fix are cell-agnostic.

    forward (parallel, training):
        y = core(x)                 # (B, S, D) -> (B, S, D)
    step (recurrent, generation):
        y, state = core.step(x, state)   # state carried across tokens
    """

    def __init__(self, config: XLSTMLayerConfig):
        super().__init__()
        self.config = config
        self.block_type = config.block_type
        if config.block_type == "mlstm":
            self.core = mLSTMLayer(config=config.to_mlstm_config())
            self.out_proj = None  # mLSTM already has its internal down-proj
        elif config.block_type == "slstm":
            self.core = sLSTMLayer(config=config.to_slstm_config())
            # sLSTM layer outputs (B,S,D) via its own GroupNorm. We add an
            # EXTRA out_proj Linear so identity-init can zero the WHOLE
            # inserted branch (sLSTM's internal compute is non-zero even at
            # init, so we must zero this outer proj to keep model==base at
            # step 0). Same residual contract as mLSTM, just placed outside.
            self.out_proj = nn.Linear(config.embedding_dim, config.embedding_dim,
                                      bias=False)
        else:
            raise ValueError(f"unknown block_type {config.block_type}")

    # ---- training forward: parallel over the whole sequence ----
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.core(x)
        if self.out_proj is not None:
            y = self.out_proj(y)
        return y

    # ---- generation step: one token at a time, state carried across ----
    # Normalizes the cell-specific state dict into our contract
    # {"conv_state":..., "recurrent_state":...} so the patcher never cares
    # which cell is inside.
    def step(self, x: torch.Tensor, state=None):
        conv_state = state.get("conv_state") if state else None
        rec_state = state.get("recurrent_state") if state else None
        if self.block_type == "mlstm":
            y, s = self.core.step(
                x, mlstm_state=rec_state, conv_state=conv_state
            )
            # mLSTM returns {"mlstm_state":..., "conv_state":...}
            return y, {"conv_state": s.get("conv_state"),
                        "recurrent_state": s.get("mlstm_state")}
        else:  # slstm
            y, s = self.core.step(
                x, conv_state=conv_state, slstm_state=rec_state
            )
            # sLSTM returns {"conv_state":..., "slstm_state":...}
            if self.out_proj is not None:
                y = self.out_proj(y)
            return y, {"conv_state": s.get("conv_state"),
                        "recurrent_state": s.get("slstm_state")}

    # ---- IDENTITY-SAFE INIT ----
    # Zero the output projection so the xLSTM branch contributes ~0 at start.
    # Because the patcher adds this branch residually (with a small gate), the
    # model then equals the frozen base at step 0 -> we don't break Qwen on
    # insert. Training then lifts the proj off zero to "add memory".
    # NOTE: for sLSTM we do NOT zero the cell's internal biases -- the paper's
    # forget-gate powerlaw init is what makes sLSTM remember long-range, so we
    # KEEP it. We only zero our outer out_proj (which gates the whole branch).
    def init_identity(self):
        if self.block_type == "mlstm":
            nn.init.zeros_(self.core.proj_down.weight)
            if self.core.proj_down.bias is not None:
                nn.init.zeros_(self.core.proj_down.bias)
            nn.init.zeros_(self.core.learnable_skip)
        else:  # slstm
            nn.init.zeros_(self.out_proj.weight)

    # convenience: number of trainable params in this sublayer
    @property
    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
