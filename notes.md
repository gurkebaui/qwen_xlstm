# =============================================================================
# notes.md — architectural gotchas & decisions (read before changing generation/training)
#
# This file is for the NON-OBVIOUS stuff that will bite us later if we
# forget it. Not a changelog; not the plan. Just "here's the trap and
# how we solved it" so a future session (or Henry) doesn't re-derive it.
# =============================================================================


## 1. GENERATION PATH (src/patcher.py) — the KV-cache + RoPE trap
------------------------------------------------------------------
DATE: 2026-07-12   STATUS: FIXED (smoke_generate.py proves it)

THE TRAP:
  The first draft of `XlstmQwenLayer.step()` called Qwen's attention
  RAW:  `h = res + self.self_attn(self.input_layernorm(x), **kw)[0]`
  This is DOUBLE-broken:
    (a) CRASH: transformers>=5.x `Qwen2Attention.forward` requires
        positional args `position_embeddings` + `attention_mask` that the
        full `Qwen2DecoderLayer.forward` normally supplies. Raw call ->
        TypeError (missing 2 required positional args).
    (b) WRONG EVEN IF IT RAN: calling attention per-token this way
        loses HF's KV-cache (recomputes O(n^2) each step) AND loses
        RoPE position encoding -> garbage output for any multi-token gen.

THE FIX (what's in the code now):
  Do NOT reimplement the decoder walk. Let HF's own `self.backbone(...)`
  forward handle attention + RoPE + KV-cache. We only intercept the
  xLSTM sublayer:
    * `XlstmQwenLayer` has a `recurrent_mode` flag (default False).
    * Parallel mode (training + prefill): xlstm uses `.forward()` (whole seq).
    * Recurrent mode (decode): xlstm uses `.step(state)` and stores the
      carried state in `layer._last_xlstm_state`.
    * `XlstmQwenModel.generate_step()` flips every layer to recurrent_mode,
      calls `self.backbone(input_ids, use_cache=True,
      past_key_values=pkv)`, and threads BOTH:
        - HF's `out.past_key_values` (the KV cache) between steps, AND
        - the xlstm recurrent state (harvested from each layer).
    * `generate()` orchestrates: prefill (parallel, builds pkv + xlstm
      states) -> decode loop (one token, carrying pkv + xlstm state).

PROVEN CORRECT BY: tests/smoke_generate.py
  Identity-init model (== frozen base), then compare the first generated
  token from `generate()` vs a parallel `forward` over the prompt.
  They MUST agree (deterministic base) -> if RoPE/KV/wiring were wrong,
  they'd diverge. Currently: MATCH = True.

IF GENERATION BREAKS AGAIN LATER (e.g. after a transformers upgrade):
  - first suspect: a raw `self.self_attn(...)` call slipped back in.
    There should be NONE. All attention goes through `self.backbone(...)`.
  - second suspect: `past_key_values` not threaded (check generate_step
    returns pkv and the caller feeds it back).
  - third suspect: xlstm state not carried (check `recurrent_mode` is
    True during decode and `_last_xlstm_state` is harvested).


## 2. xLSTM STATE IS RECURRENT (not parallel like attention)
------------------------------------------------------------------
DATE: 2026-07-12   STATUS: handled by package + our step() path

  mLSTM carries matrix-memory + conv state ACROSS tokens. Attention is
  token-parallel. So a single forward pass is NOT enough for generation;
  state must be threaded (see #1). The real `xlstm` package already
  implements `.step()` with state carry — we do NOT reimplement it
  (we only wrap it in src/xlstm_layer.py). Smoke: tests/smoke_mlstm.py
  proves parallel .forward() == recurrent .step() threading.


## 3. DTYPE: xLSTM layer defaults to float32, Qwen loads bfloat16
------------------------------------------------------------------
DATE: 2026-07-12   STATUS: fixed (cosmetic warning remains)

  `XLSTMLayer` is cast to `self.backbone.dtype` (bf16) right after
  construction in the patcher. Without this, the mLSTM weights are
  float32 and you get a dtype mismatch at the first matmul.
  A benign UserWarning ("Mismatch dtype ... RMSNorm") still prints from
  the BASE model's own norms during eval — harmless (outputs bit-identical).
  Left as-is rather than silently casting (could mask a real dtype bug).


## 4. pg19 eval probe is BLOCKED on datasets>=4.8
------------------------------------------------------------------
DATE: 2026-07-12   STATUS: workaround in place (wikitext stand-in)

  pg19 is a SCRIPTED HF dataset. `datasets` 4.8.4 DROPPED support
  for scripted loaders -> `load_dataset("pg19", streaming=True)` raises
  RuntimeError "Dataset scripts are no longer supported".
  The INTENDED long-range memory probe is pg19 (Gutenberg books, ~20x
  longer than WikiText). Until we add a parquet loader or pin
  datasets<4.8, `eval.probe` defaults to "wikitext" (wikitext-103-v1,
  streams fine). The eval LOGIC is identical; only the data source differs.
  TODO: re-enable pg19 via parquet (HF hub Rallio/pg19 or local
  parquet) so we get the stronger long-range signal.


## 5. We use the REAL `xlstm` package — do NOT reimplement mLSTM
------------------------------------------------------------------
DATE: 2026-07-12   STATUS: enforced by design

  src/xlstm_layer.py wraps `xlstm.blocks.mlstm.layer.mLSTMLayer`
  (xlstm 2.0.5 + mlstm_kernels 2.0.2, both pip-installed).
  If a future session is tempted to "just write the mLSTM cell", DON'T —
  the official package is the source of truth and already has the parallel
  + recurrent (state-carry) kernels. Inspect the installed source under
  /home/henry/miniconda3/lib/python3.13/site-packages/xlstm/ before
  touching anything mLSTM-related.
