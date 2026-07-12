# =============================================================================
# notes.md — architectural gotchas & decisions (read before changing generation/training)
#
# This file is for the NON-OBVIOUS stuff that will bite us later if we
# forget it. Not a changelog; not the plan. Just "here's the trap and
# how we solved it" so a future session (or Henry) doesn't re-derive it.
#
# ---------------------------------------------------------------------------
# CONTENTS
#   1. GENERATION PATH — the KV-cache + RoPE trap (FIXED, proven)
#   2. xLSTM STATE IS RECURRENT (not parallel like attention)
#   3. DTYPE — xLSTM layer defaults float32, Qwen loads bfloat16
#   4. pg19 eval probe BLOCKED on datasets>=4.8 (wikitext stand-in)
#   5. Use the REAL `xlstm` package — do NOT reimplement mLSTM
#   6. COSMETIC / known minors (harmless, tracked so they don't surprise)
# ---------------------------------------------------------------------------
# =============================================================================


## 6. COSMETIC / known minors (harmless — tracked so they don't surprise)
------------------------------------------------------------------
DATE: 2026-07-12   STATUS: documented, NO fix needed

  (a) RMSNorm dtype warning.
      During eval (and any forward in bf16) the base model's OWN
      RMSNorms print:
        UserWarning: Mismatch dtype between input and weight:
        input dtype = c10::BFloat16, weight dtype = float, Cannot
        dispatch to fused implementation.
      Cause: the base Qwen layers were loaded in bf16 but their norm
      weights present as float in that code path. OUTPUTS ARE BIT-IDENTIAL
      (verified: identity diff = 0.0, gen match = True), so it is pure
      noise. Left as-is rather than silently casting (a cast could mask a
      REAL dtype bug later). Suppress only if it annoys: wrap the eval
      forward in torch.backends.fused_layer_norm(False) or set the base
      norms explicitly to bf16.

  (b) generate() uses GREEDY argmax (no sampling/temperature).
      tests/smoke_generate.py and the current generate() pick
      next_tok = logits[:, -1].argmax(...). Fine for the correctness
      smoke + a deterministic baseline, but REAL text generation will
      want sampling / temperature / top-p. That's a thin wrapper over
      HF's sampling — add later (don't hack it into the recurrent
      path; keep generate() returning logits and sample at the call site).
      NOTE: when we switch to sampling, the proof "generate matches
      parallel forward" no longer holds (sampling is non-deterministic),
      so keep a greedy smoke alongside any sampling change.

  (c) quick_eval loads the FROZEN base a SECOND time as a separate
      ~1GB instance to get the reference perplexity. Fine at 16GB for
      0.5B, but doubles VRAM during eval. If we eval a bigger base
      later, compute the reference once and reuse, or shard.

  (e) TRAINING SMOKE: per-step eval re-loads the 0.5B base and STALLS on
      this box (the 2nd from_pretrained call hangs ~indefinitely). The
      smoke pins eval ONCE at the end (eval_every = max_steps). Real runs
      eval every 200 steps so it's a non-issue there. If a future smoke
      hangs again, suspect quick_eval() re-instantiating the model.

  (g) CHECKPOINT SIZE: train() saves the FULL model.state_dict()
      (frozen base + xlstm), ~1.2GB per checkpoint for 0.5B. That's
      deliberate (lets us resume/load everything from one file), but at
      2000 steps with periodic saves this is several GB — keep under
      /home/henry/Documents/qwen_xlstm/checkpoints (NEVER /tmp; /tmp is
      wiped on the idle-reboot). To shrink: save only xlstm params
      (model.xlstm_layers.state_dict()) if base is unchanged.

  (h) EVAL BUG (burned the first 2000-step run's eval): quick_eval used
      patched_model.backbone as the "base" reference. But we swap
      backbone.model.layers IN-PLACE for XlstmQwenLayer during construction,
      so patched_model.backbone IS the patched model -> base==patched ->
      delta ALWAYS 0.0 (false "no change"). FIXED 2026-07-12: quick_eval
      now loads a SEPARATE fresh Qwen2ForCausalLM as the reference (cached
      per run). After the fix, the real eval on xlstm_cpt_step2000.pt showed
      base=27.5, patched=1494.8 -> the graft made ppl ~54x WORSE (see #9).
      NEVER pass patched_model.backbone as a base reference again.

  (i) TRAINING DIVERGENCE (the first real run FAILED): after 2000 steps at
      lr=5e-4 (cosine->0) the graft's ppl went 27.5 -> 1494.8. The training
      loss oscillated 1.1-8.0 the whole run = unstable, not converging.
      Root cause: LR too high + no grad clip for a from-scratch mLSTM graft
      in bf16; the xlstm output overshot and wrecked the residual stream.
      FIX BEFORE NEXT RUN: lower lr (1e-4 or 5e-5), add grad clipping
      (clip_grad_norm 1.0), consider a learnable scalar gate on the xlstm
      branch so it starts near-zero and grows slowly. ALSO: add a
      few-step validation gate (eval delta stays ~0 or improves) BEFORE
      committing to a full 2000-step run. The infrastructure (data, VRAM,
      generation, eval) all work; only the training RECIPE failed.

  (j) RUN #2 OUTCOME (2000 steps, the FIXED recipe, 2026-07-12):
      Config: seq_len=1024 x grad_accum=2 (eff 2048 tok/step),
      lr=1e-4 cosine, grad_clip=1.0, gate init 0.1, AdamW.
      Result: 0 OOMs, DONE in 986s (~16 min). Training loss dropped
      2.4 -> ~2.0-2.7 (xlstm DID train: gate 0.10->0.078,
      down_proj 0.00->0.0025). BUT perplexity delta vs frozen base was
      MONOTONICALLY NEGATIVE:
        step  20  +0.013   (warmup, ~base)
        step 200 +0.032
        step 600 +0.263
        step 1000 +0.423   (worst)
        step 2000 +0.350   (patched WORSE by +0.35 ppl)
      So the graft learned SOMETHING (real training, not deadlocked) but it
      made wikitext perplexity slightly WORSE. The "memory helps" hypothesis
      is NOT supported by this run. CAVEATS (don't over-read the fail):
        * eval is WIKITEXT ppl — a SHORT-RANGE LM probe. A recurrent
          memory sublayer is hypothesized to help LONG-RANGE/code/structure,
          which wikitext ppl can't show even if real. We never got CODE
          data (StarCoder gated) or a long-range task (pg19 blocked).
        * +0.35 ppl is SMALL (not catastrophic). Recipe is STABLE.
        * Likely needs: better eval task (code/long-range), maybe lower LR
          or more steps, or the gate/dim tuning. The plumbing is solid;
          the SCIENCE question is still open.
      Checkpoint: checkpoints/xlstm_cpt_step2000.pt (loads OK, 611.5M params).

## 9. STILL-OPEN / TODO before claiming "memory helps"
------------------------------------------------------------------
  * BETTER EVAL: wikitext ppl is the wrong probe for a memory graft.
    Need code perplexity (StarCoder-Data, gated -> needs HF auth) and/or
    a long-range task (pg19 needs a parquet loader; blocked on datasets>=4.8).
  * LR / gate / dim tuning: +0.35 ppl suggests the graft is undertrained
    or mis-scaled for this task. Try lr=5e-5, longer runs, or a
    learnable per-layer gate scaling.
  * Real CODE data (StarCoder-Data) is gated -> needs HF auth; currently
    using math (open-web-math) + long-text (fineweb-edu-dedup) + wikitext.
  * generate() uses greedy argmax (notes 6b) - fine for now.


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
