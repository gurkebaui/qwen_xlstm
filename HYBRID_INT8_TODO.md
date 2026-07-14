# HYBRID (base-256 / sLSTM-1024) + INT8-FFN EXPERIMENT — PARKED

Status: **parked, not started.** Branch created so we can roll back to `main`
cleanly. No training was launched. Revisit later when compute/time allow.

---

## THE HYPOTHESIS (why this branch exists)

We want the sLSTM graft to act as **long-range memory beyond the frozen
base transformer's context window**. The clean test:

- Train the **base attention** on SHORT local windows (base_ctx = 256).
- Train the **sLSTM** on the FULL sequence (1024 / 2048) so it learns
  to carry state across the whole rollout — i.e. remember things the
  base's 256-window attention can't see.

If that works, the sLSTM becomes a genuine "memory extension" of the
base model. That is the end goal of the whole graft (paper: sLSTM learns
to carry long-range state instead of only being trained in parallel).

This is ALREADY implemented as `base_ctx` in `src/patcher.py`
(commit `e6b03e6`): when set, the frozen base attention runs on local
`base_ctx` chunks while the sLSTM processes the full `h`. It was only
ever TIMED, never TRAINED that way.

---

## WHAT WE PROVED SO FAR (evidence, not opinion)

1. **Step time scales ~LINEARLY in L, NOT L².**
   From `runs/time_seqlen.log`:
   ```
   L=256  -> 3.25 s/step
   L=512  -> 6.97 s/step   (2.1x per doubling)
   L=1024 -> 13.3 s/step    (1.9x per doubling)
   ```
   If attention (O(L²)) dominated, doubling L would cost ~4x. It costs
   ~2x → the dominant cost is the **FFN (O(L), linear)**, not attention.
   The sLSTM block itself is only ~0.13 s/step (`bench_backends`).
   CONSEQUENCE: long-sequence training is inherently ~13 s/step at L=1024
   no matter what we do to attention. That is the compute ceiling.

2. **Hybrid alone gives NO speedup at L=1024.**
   `scripts/time_hybrid.py` (commit `e6b03e6`):
   ```
   FULL   (base_ctx=None): 12.49 s/step
   HYBRID (base_ctx=128): 12.44 s/step   <- no win
   ```
   Because the FFN still processes the full 1024 length. Chunking the
   (already-cheap) attention saves almost nothing.

3. **Quantizing the FFN to int8 DOES NOT WORK for training.**
   Tested `bitsandbytes.nn.Linear8bitLt` on this torch/cu130/sm_89:
   - FORWARD works (int8 matmul runs, no compat error).
   - BACKWARD is BROKEN: `autograd/_functions.py` `MatMul8bitLt`
     disconnects the autograd graph → gradient cannot flow THROUGH the
     FFN to reach the sLSTM (which sits upstream of the FFN in the
     residual path: `... -> mlp -> o`, grad flows `loss -> o -> mlp
     -> ... -> sLSTM`). Result: sLSTM gets ZERO gradient → can't train.
   Dead end. (Forward-only int8 is fine for inference, useless for CPT.)

4. **Current graft does NOT carry long-range memory (yet).**
   Probe of `xlstm_cpt_step3000.pt` on natural-language `emozilla/pg19`
   (recurrent roll to 16384, `scripts/probe_step60.py --probe-source
   emozilla/pg19`):
   ```
   L=  2048: base 10.86  patch 15.62  delta +4.75  (WORSE)
   L=  4096: base 11.44  patch 10.05  delta -1.38  (BETTER)
   L=  8192: base 13.53  patch 15.60  delta +2.06  (WORSE)
   L= 16384: base 15.14  patch 20.01  delta +4.87  (WORSE)
   ```
   Wins ONLY at 4096, degrades monotonically past it. Root cause: the
   graft was trained at `seq_len=256`, so the sLSTM NEVER saw sequences
   >256 during training and never learned long-range recurrence. The
   in-training `eval[pg19] -6.76` was misleading (short eval window).

---

## TODO (do this when we revisit)

- [ ] **Launch the real hybrid-CPT run** (no quantization needed):
      config `xlstm.base_ctx: 256`, sLSTM `context_length: 1024`
      (full), train ~1500 steps. Expect ~13 s/step (FFN-bound,
      unavoidable). This is the actual test of "sLSTM beyond base window".
      The sLSTM trains because the FFN stays bf16 (gradient flows).
- [ ] **Re-probe** the resulting ckpt with `scripts/recurrent_probe.py`
      (build the patched model at `context_length=2048`, recurrently
      decode to 16384) OR `scripts/probe_step60.py --probe-source
      emozilla/pg19`. Check whether delta stays negative / improves as L
      grows (the real memory signal).
- [ ] **If FFN cost is still the blocker:** options to revisit —
      (a) a smaller base model (0.5B is our floor right now),
      (b) fix the CUDA toolkit (torch cu130 vs nvcc 13.0.88 mismatch)
          to unlock fused kernels — but that's only ~5-8% and risky,
      (c) accept linear O(L) FFN cost as the price of long-seq training.

## OPEN QUESTION (the linear-compute scaling issue)

Why is the FFN the bottleneck and not the attention? Because at L=1024 on
a 0.5B model the FFN matmuls (2× per layer, hidden 896→intermediate
2432) dominate the per-token compute, and they scale O(L) not O(L²).
Quantization (int8) would cut them ~2x BUT its backward is broken here,
and the gradient MUST pass through them to train the sLSTM. So there is
no cheap way to shrink the FFN on this toolchain. Smaller model or
fused kernels are the only real levers, and both are out of scope tonight.

---

## FILES ON THIS BRANCH

- `src/patcher.py` — `base_ctx` param on `XlstmQwenLayer` + chunked
  local-window attention path in `forward()`. (main)
- `scripts/time_hybrid.py` — times FULL vs HYBRID step. (main)
- `scripts/probe_offline.py` — offline recurrent probe (local mix.jsonl,
  bypasses HF 403). (main)
- `scripts/probe_step60.py` — `emozilla/pg19` recurrent probe (works
  now; deepmind/pg19 is DEAD — HF dropped script datasets). (main + the
  uncommitted `--n-books` arg fix)
- `scripts/recurrent_probe.py` — build patched model at chosen
  `context_length`, recurrently eval to 16384. **The parked experiment's
  main script.** (new, untracked on main, committed here)
- `configs/base.yaml` — `xlstm.base_ctx: 128` default (set during dev;
  flip to 256 for the real run).

## ROLLBACK

`git checkout main` — everything here is additive; main is untouched
except this branch was cut FROM main's HEAD.
