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

    @staticmethod
    def _make_rotary(hidden_states, position_ids, head_dim, device, dtype):
        # Explicit, correct RoPE (cos, sin), each shape (B, S, head_dim)
        # -- which is EXACTLY what HF's apply_rotary_pos_emb expects
        # (it does cos.unsqueeze(1) -> (B,1,S,head_dim) to broadcast
        # over heads). We must NOT pre-unsqueeze here, or it becomes
        # 5D and breaks. Theta = 10000^(-2i/head_dim), angle=pos*theta.
        inv_freq = 1.0 / (10000.0 ** (
            torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
            / head_dim
        ))                                              # (head_dim/2,)
        pos = position_ids.to(torch.float32).reshape(-1)   # (S,)
        # (S, D/2) * (D/2,) -> (S, D/2)
        ang = pos[:, None] * inv_freq[None, :]
        # interleave to full head_dim: [a, a] pairs -> (S, head_dim)
        emb = torch.cat([ang, ang], dim=-1)             # (S, head_dim)
        # prepend batch dim -> (B, S, head_dim)
        emb = emb.unsqueeze(0)                             # (1, S, head_dim)
        return emb.cos().to(dtype), emb.sin().to(dtype)

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

        # --- learnable GATE on the xlstm branch (stability fix, notes #9/i) ---
        # m = h + gate * xlstm(norm(h)). Init to a SMALL NONZERO (0.1), NOT 0.
        # WHY NOT 0: with down_proj also zeroed (identity), gate=0 and down_proj=0
        # both have zero gradient at step 0 (gate grad ~ xlstm_out~0; down_proj
        # grad ~ gate*...~0) -> DEADLOCK, the xlstm never trains (confirmed:
        # gate+down_proj stayed 0.0 after 5 steps). A small nonzero gate gives
        # down_proj a real gradient from step 0, so the branch is trainable.
        # Step-0 output is still ~= base because xlstm_out~0 (down_proj=0), so
        # the identity contract holds (smoke delta=0); the gate just lets it LEARN.
        self.gate = nn.Parameter(torch.full((1,), 0.1))

        # --- generation mode toggle ---
        # False (default): xLSTM runs in PARALLEL .forward()  -> training + prefill.
        # True (set during token-by-token decode): xLSTM runs in RECURRENT
        # .step() and carries state across tokens (the memory). We store the
        # carried state on the layer so the caller can read it back.
        self.recurrent_mode = False
        self._last_xlstm_state = None

    # ---- training / prefill forward (parallel; state not needed) ----
    def forward(self, hidden_states, **kwargs):
        # --- CRITICAL: RoPE / position_embeddings plumbing ---
        # HF's Qwen2Attention.forward expects `position_embeddings`
        # (a (cos, sin) tuple) as a NAMED argument. In transformers
        # 5.5 the TOP-LEVEL model computes cos/sin via
        # `self.rotary_emb(hidden_states, position_ids)` and passes
        # them DOWN to the decoder layer. WE REPLACED the decoder
        # layer, so we must compute + forward them ourselves —
        # otherwise the attn call gets a mis-shapen cos/sin (the
        # "size of tensor a (14) must match tensor b (64)" crash
        # during token-by-token recurrent decode, where position
        # bookkeeping diverges from the parallel path).
        # Fix: build cos/sin from the backbone's OWN rotary_emb
        # (same module HF uses) using the position_ids in kwargs,
        # then pass them EXPLICITLY. We strip the conflicting
        # position_ids / position_embeddings / use_cache from the
        # kwargs we forward so they can't collide.
        pos_ids = kwargs.get("position_ids", None)
        cos_sin = None
        if pos_ids is not None:
            # explicit, correct RoPE (see _make_rotary): (cos, sin)
            # each (B, S, head_dim) -> what Qwen2Attention wants.
            hd = self.self_attn.head_dim
            cos_sin = self._make_rotary(
                hidden_states, pos_ids, hd, hidden_states.device,
                hidden_states.dtype,
            )
        # only forward what attn actually wants (not position_ids /
        # position_embeddings / use_cache which we handle ourselves)
        attn_kwargs = {
            k: v for k, v in kwargs.items()
            if k not in ("position_ids", "position_embeddings",
                            "use_cache")
        }

        # attention sublayer (frozen) — Qwen's own residual
        res = hidden_states
        attn_in = self.input_layernorm(hidden_states)
        if cos_sin is not None:
            h = res + self.self_attn(
                attn_in, position_embeddings=cos_sin, **attn_kwargs
            )[0]
        else:
            # no position_ids given (parallel prefill w/ default) ->
            # let HF compute cos/sin internally as usual
            h = res + self.self_attn(attn_in, **attn_kwargs)[0]

        # NEW xlstm sublayer (trainable) — own norm + own residual.
        # gated: m = h + gate * xlstm(norm(h)). gate starts at 0 (identity)
        # and is trained up gently (stability fix, notes #9/i).
        if self.recurrent_mode:
            # decode step: one token, carry state
            m_xlstm, state = self.xlstm.step(
                self.xlstm_layernorm(h), state=self._last_xlstm_state
            )
            self._last_xlstm_state = state
        else:
            # prefill / training: whole sequence in parallel
            m_xlstm = self.xlstm(self.xlstm_layernorm(h))
        m = h + self.gate.to(m_xlstm.dtype) * m_xlstm

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
            # give the wrapped layer a direct handle to the backbone's
            # RoPE module so its forward can recompute (cos, sin) itself
            # for token-by-token recurrent decode (see forward's RoPE fix).
            patched.rotary_emb = self.backbone.model.rotary_emb

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
                     position_ids=None, **kwargs):
        # coerce to 2D (B, S): callers may pass a 1D row
        # (ids[t:t+1]) which would break .shape[1] below.
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)
        if layer_states is None:
            layer_states = {i: None for i in range(len(self.xlstm_layers))}
        # across steps. (BUG FIX: an earlier version reset
        # `_last_xlstm_state = None` here on EVERY call, which discarded
        # the memory between tokens -> the xlstm only saw one token at a
        # time and carried NOTHING. That made the recurrent path a no-op
        # and the earlier "generate MATCH" smoke test meaningless. Now we
        # feed the previous step's state back in, so memory accumulates
        # across the whole sequence -- THIS is what the long-context
        # memory probe measures.
        for i, layer in enumerate(self.xlstm_layers):
            layer.recurrent_mode = True
            layer._last_xlstm_state = layer_states[i]   # carry, don't reset

        # --- RoPE position_ids: ALWAYS explicit, never None ---
        # If the caller didn't pass one, derive the ABSOLUTE position
        # of this single token from the KV-cache length (past_length).
        # Passing None lets HF auto-compute position_ids, which (with our
        # swapped decoder layers + cache) produces a GARBAGE shape
        # (1, hidden_size) and blows up apply_rotary_pos_emb. So we
        # compute it ourselves: a (1,1) tensor holding the absolute
        # index. Correct for both prefill (t) and decode (past_len+t).
        if position_ids is None:
            seq = input_ids.shape[1]
            past_len = 0
            if past_key_values is not None:
                # HF DynamicCache / tuple: past_length = cached key seq dim
                try:
                    past_len = past_key_values.get_seq_length()
                except Exception:
                    try:
                        past_len = past_key_values[0][0].shape[-2]
                    except Exception:
                        past_len = 0
            pos = torch.arange(past_len, past_len + seq,
                               device=input_ids.device).unsqueeze(0)  # (1, seq)
            position_ids = pos

        out = self.backbone(
            input_ids=input_ids,
            use_cache=True,
            past_key_values=past_key_values,
            position_ids=position_ids,   # explicit, correct shape (1, seq)
            **kwargs,                            # (no conflicting keys)
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
            # gate -> 0.1 (small nonzero, NOT 0): keeps step-0 ~= base
            # (down_proj=0 -> xlstm_out~0) while staying TRAINABLE (gate=0
            # would deadlock with down_proj=0, see notes #9 / gate comment).
            with torch.no_grad():
                layer.gate.fill_(0.1)

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
