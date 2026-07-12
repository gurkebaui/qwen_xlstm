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

        # --- generation mode toggle ---
        # False (default): xLSTM runs in PARALLEL .forward()  -> training + prefill.
        # True (set during token-by-token decode): xLSTM runs in RECURRENT
        # .step() and carries state across tokens (the memory). We store the
        # carried state on the layer so the caller can read it back.
        self.recurrent_mode = False
        self._last_xlstm_state = None

    # ---- training / prefill forward (parallel; state not needed) ----
    def forward(self, hidden_states, **kwargs):
        # attention sublayer (frozen) — Qwen's own residual
        res = hidden_states
        h = res + self.self_attn(self.input_layernorm(hidden_states), **kwargs)[0]

        # NEW xlstm sublayer (trainable) — own norm + own residual
        if self.recurrent_mode:
            # decode step: one token, carry state
            m_xlstm, state = self.xlstm.step(
                self.xlstm_layernorm(h), state=self._last_xlstm_state
            )
            self._last_xlstm_state = state
        else:
            # prefill / training: whole sequence in parallel
            m_xlstm = self.xlstm(self.xlstm_layernorm(h))
        m = h + m_xlstm

        # ffn sublayer (frozen)
        o = m + self.mlp(self.post_attention_layernorm(m))
        return o

    # ---- generation: caller drives recurrent_mode; see XlstmQwenModel.generate_step ----


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
    # GENERATION: prefill in parallel, then decode token-by-token.           #
    #                                                                     #
    # We do NOT reimplement the decoder walk. Instead we let HuggingFace's   #
    # own forward do attention + RoPE + KV-cache correctly (that's exactly #
    # what crashed before: calling Qwen attention raw lost those). We only   #
    # flip the xLSTM sublayers into RECURRENT mode so THEY carry memory.    #
    #                                                                     #
    # Two things must be carried between steps (this is the part that was    #
    # previously broken):                                                       #
    #   * HF's past_key_values (the KV cache) -> attention sees the full   #
    #     prefix without recomputing it (O(n) decode, correct RoPE).       #
    #   * the xLSTM recurrent state (matrix memory + conv) -> our memory.   #
    # `generate_step` threads BOTH. `generate()` orchestrates the loop.      #
    # ---------------------------------------------------------------- #
    @torch.no_grad()
    def generate_step(self, input_ids, layer_states=None, past_key_values=None,
                    **kwargs):
        if layer_states is None:
            layer_states = {i: None for i in range(len(self.xlstm_layers))}
        # put xlstm layers into recurrent mode (state carried across tokens).
        # each layer holds its own carried state in _last_xlstm_state;
        # we harvest it back into layer_states[i] after the forward.
        for layer in self.xlstm_layers:
            layer.recurrent_mode = True
            layer._last_xlstm_state = None

        out = self.backbone(
            input_ids=input_ids,
            use_cache=True,
            past_key_values=past_key_values,
            **kwargs,
        )
        logits = out.logits
        pkv = out.past_key_values  # carry the KV cache forward

        # harvest the carried xlstm states back into the caller dict
        for i, layer in enumerate(self.xlstm_layers):
            layer_states[i] = layer._last_xlstm_state
        return logits, layer_states, pkv

    @torch.no_grad()
    def generate(self, prompt_ids: torch.Tensor, max_new_tokens: int = 8):
        """Autoregressive generate. Returns the full token ids (prompt + new).

        Correctness: prefill builds the KV cache + xlstm states from the
        whole prompt; each decode step feeds ONLY the last token back, carrying
        BOTH HF's past_key_values and the xlstm recurrent state.
        """
        self.reset_generation()
        ids = prompt_ids.to(self.device)

        # --- prefill: parallel forward over the full prompt ---
        layer_states = {i: None for i in range(len(self.xlstm_layers))}
        pkv = None
        # prefill uses parallel xlstm (recurrent_mode False) via normal forward
        out = self.backbone(input_ids=ids, use_cache=True)
        pkv = out.past_key_values
        # initialize xlstm states to None for the upcoming decode steps
        for layer in self.xlstm_layers:
            layer._last_xlstm_state = None

        # --- decode loop: one new token at a time ---
        for _ in range(max_new_tokens):
            last = ids[:, -1:]
            logits, layer_states, pkv = self.generate_step(
                last, layer_states=layer_states, past_key_values=pkv
            )
            next_tok = logits[:, -1].argmax(dim=-1, keepdim=True)
            ids = torch.cat([ids, next_tok], dim=1)
        self.reset_generation()
        return ids

    def reset_generation(self):
        """Clear recurrent state + recurrent_mode for a fresh sequence."""
        for layer in self.xlstm_layers:
            layer.recurrent_mode = False
            layer._last_xlstm_state = None

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
