# =============================================================================
# src/patcher.py
#
# Inserts OUR trainable mLSTM sublayer BETWEEN the attention and FFN sublayers
# of every Qwen2.5 decoder layer, WITHOUT touching the base transformer.
#
# Qwen2.5 decoder layer (HF `Qwen2DecoderLayer`) originally does:
#     residual = hidden
#     h = residual + self_attn( norm1(hidden) )      # attention sublayer
#     o = h + mlp( norm2(h) )                        # FFN sublayer
#
# We insert a THIRD sublayer (mLSTM) in the gap, each with its own RMSNorm
# and its own residual — exactly mirroring Qwen's own sublayer convention:
#
#     residual = hidden
#     h = residual + attn(      norm1(hidden) )     # FROZEN (base)
#     m = h        + xlstm(      normX(h)     )     # TRAINABLE (new)
#     o = m        + mlp(        norm2(m)     )     # FROZEN (base)
#
# Freezing:
#     * every parameter of the loaded Qwen model is set requires_grad=False
#       EXCEPT the parameters we add (the XLSTMLayer modules).
#     * the LM head also stays frozen.
#
# Recurrent state (the key gotcha):
#     * Training uses  .forward()  — the real layer runs in parallel (fast).
#     * Generation uses  .step()    — one token at a time; we keep a
#       `layer_states` dict keyed by layer index and feed it back each step,
#       resetting it per generated sequence.
#
# This module does NOT reimplement mLSTM. It only wraps the real layer
# (see xlstm_layer.py) and rewires the Qwen forward pass.
# =============================================================================

from typing import Dict, Optional

import torch
import torch.nn as nn
from transformers import Qwen2ForCausalLM, Qwen2Config

from xlstm_layer import XLSTMLayer, XLSTMLayerConfig


class XlstmQwenLayer(nn.Module):
    """One patched decoder layer = base attn + NEW xlstm + base ffn.

    We do NOT subclass Qwen2DecoderLayer (fragile across transformers versions).
    Instead we hold references to the ORIGINAL submodules (self_attn, mlp, the
    two RMSNorms) and reimplement ONLY the forward wiring so we control exactly
    where the xlstm sublayer sits and how residuals combine.
    """

    def __init__(self, base_layer, xlstm: XLSTMLayer):
        super().__init__()
        # --- keep the base submodules (already frozen by the patcher) ---
        self.input_layernorm = base_layer.input_layernorm      # norm before attn
        self.self_attn = base_layer.self_attn
        self.post_attention_layernorm = base_layer.post_attention_layernorm  # norm before ffn
        self.mlp = base_layer.mlp

        # --- our new sublayer: a THIRD norm + the xlstm block ---
        # norm dim must match hidden size
        hidden = base_layer.input_layernorm.weight.shape[0]
        self.xlstm_layernorm = nn.RMSNorm(hidden, eps=1e-6)
        self.xlstm = xlstm

    # ---- training forward (parallel; state not needed) ----
    def forward(self, hidden_states, **kwargs):
        # attention sublayer (frozen) — Qwen's own residual
        res = hidden_states
        h = res + self.self_attn(self.input_layernorm(hidden_states), **kwargs)[0]

        # NEW xlstm sublayer (trainable) — own norm + own residual
        m = h + self.xlstm(self.xlstm_layernorm(h))

        # ffn sublayer (frozen)
        o = m + self.mlp(self.post_attention_layernorm(m))
        return o

    # ---- generation step (recurrent; state carried) ----
    def step(self, hidden_states, state: Optional[dict] = None, **kwargs):
        res = hidden_states
        # attn is causal+KV-cached by HF; for a single token we call it directly
        h = res + self.self_attn(self.input_layernorm(hidden_states), **kwargs)[0]
        m, xlstm_state = self.xlstm.step(self.xlstm_layernorm(h), state=state)
        m = h + m
        o = m + self.mlp(self.post_attention_layernorm(m))
        return o, {"xlstm_state": xlstm_state}


class XlstmQwenModel(nn.Module):
    """The assembled, patched model.

    Holds the (frozen) HF backbone + a list of XLSTMLayer inserted per layer.
    Exposes:
        forward(...)             -> parallel training pass (returns logits)
        generate_step(...)       -> single-token recurrent pass (returns logits + state)
        num_trainable_params()   -> for logging / sanity
    """

    def __init__(self, model_id: str, xlstm_cfg: XLSTMLayerConfig,
                 device: str = "cuda", dtype=torch.bfloat16):
        super().__init__()
        # --- load base, FULLY FROZEN ---
        self.backbone = Qwen2ForCausalLM.from_pretrained(
            model_id, torch_dtype=dtype
        )
        self.backbone.requires_grad_(False)  # freeze EVERYTHING in the base

        # --- insert one trainable xlstm sublayer per decoder layer ---
        # the xlstm layer must match the backbone dtype (Qwen loads in bfloat16);
        # the real layer defaults to float32, so we cast it explicitly.
        self.xlstm_layers: nn.ModuleList = nn.ModuleList()
        for i, base_layer in enumerate(self.backbone.model.layers):
            xl = XLSTMLayer(xlstm_cfg)
            xl = xl.to(self.backbone.dtype)
            patched = XlstmQwenLayer(base_layer, xl)
            # base submodule params stay frozen (requires_grad False inherited);
            # the xlstm + its new norm must be TRAINABLE:
            for p in patched.xlstm.parameters():
                p.requires_grad = True
            for p in patched.xlstm_layernorm.parameters():
                p.requires_grad = True
            self.xlstm_layers.append(patched)
            # swap the layer in the backbone with our wrapped version
            self.backbone.model.layers[i] = patched

        self.config = self.backbone.config
        self.device = device

    # ---------------------------------------------------------------- #
    # TRAINING forward: delegates to the backbone (each wrapped layer runs  #
    # its parallel .forward internally). Returns model output (logits etc).  #
    # ---------------------------------------------------------------- #
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        return self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs,
        )

    # ---------------------------------------------------------------- #
    # GENERATION step: one token at a time, recurrent state carried.       #
    # `layer_states[i]` is the xlstm state for layer i. Caller resets it   #
    # to None at the start of each new sequence.                         #
    # ---------------------------------------------------------------- #
    def generate_step(self, input_ids, layer_states: Optional[Dict[int, dict]] = None, **kwargs):
        if layer_states is None:
            layer_states = {i: None for i in range(len(self.xlstm_layers))}
        # run the backbone layers manually so we can thread state through xlstm
        x = self.backbone.model.embed_tokens(input_ids)
        for i, layer in enumerate(self.backbone.model.layers):
            st = layer_states.get(i)
            x, new_st = layer.step(x, state=st, **kwargs)
            layer_states[i] = new_st
        x = self.backbone.model.norm(x)
        logits = self.backbone.lm_head(x)
        return logits, layer_states

    # ---------------------------------------------------------------- #
    # IDENTITY init: zero every xlstm down-proj so the inserted branch     #
    # contributes ~0 at step 0 -> model == frozen base. Call once after   #
    # construction, before training.                                    #
    # ---------------------------------------------------------------- #
    def init_identity(self):
        for layer in self.xlstm_layers:
            layer.xlstm.init_identity()

    def num_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def num_total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def from_config(cls, base_yaml: dict):
        # helper used by main.py / train.py later
        m = cls(
            model_id=base_yaml["model"]["name"],
            xlstm_cfg=XLSTMLayerConfig(**base_yaml["xlstm"]),
            device=base_yaml.get("device", "cuda"),
        )
        return m
