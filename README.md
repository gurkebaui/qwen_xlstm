# qwen_xlstm

Graft a **trainable sLSTM memory sublayer** into Qwen2.5-Coder-0.5B, between
the attention and FFN sublayers of every decoder block. The base transformer
is **frozen**; only the inserted sLSTM blocks are trained (Stage 1 =
continued pretraining / CPT). Goal: the sLSTM learns to carry **long-range
state** that the frozen base (limited context window) cannot — i.e. a learned
memory extension of the model.

All xLSTM code comes from the **official `xlstm` PyPI package** (v2.0.5)
+ `mlstm_kernels` (v2.0.2). We do NOT reimplement sLSTM — we wrap
the real `sLSTMLayer` and plug it into Qwen.

---

## Architecture

```
Qwen decoder layer (frozen except the graft):
  h      = x + Attention(x)            # frozen, base_ctx-local (optional)
  m_xl   = sLSTM(norm(h))           # TRAINABLE, full-seq (parallel .forward / recurrent .step)
  m      = h + gate * m_xl           # learnable gate (init 0.1, NOT 0)
  o      = m + FFN(norm(m))          # frozen
```

- **sLSTM, not mLSTM** (decided 2026-07-13): the paper's best-at-scale
  config is `architecture: "1:0"` (every layer sLSTM). mLSTM collapsed
  past its training window; sLSTM's forget-gate + the `powerlaw_blockdependent`
  bias init let it hold long-range state.
- **Identity init**: at step 0 the graft output ≈ base (down_proj zeroed,
  gate small-nonzero) so training starts stable. Verified by smoke test.
- **Hybrid context split** (see branch `exp/hybrid-int8-parked`): the frozen
  base attention can run on LOCAL windows (`base_ctx`, e.g. 256) while the
  sLSTM processes the FULL sequence (e.g. 1024) — so the sLSTM carries
  memory beyond the base's window. Implemented but not yet trained that way.

## Layout

```
configs/
  base.yaml            # single-source-of-truth config (single-field switches, no CLI soup)
src/
  xlstm_layer.py      # thin wrapper around real xlstm.sLSTMLayer
                        #   -> identity-safe init + unified forward/step API
  patcher.py          # inserts XlstmQwenLayer into each Qwen block;
                        #   freezes base, trains only sLSTM; residual per sublayer
  train.py            # CPT loop: loads data, trains, saves ckpt, non-fatal eval
  eval.py             # recurrent_long_context_eval + quick_eval (HF-gated, see below)
scripts/
  time_seqlen.py      # times one step at 256/512/1024 -> proves ~L^2 scaling
  time_hybrid.py     # times FULL vs HYBRID (base_ctx) step
  bench_backends.py   # vanilla vs cuda xlstm backend bench
  probe_offline.py    # fully-OFFLINE recurrent probe (local mix.jsonl; bypasses HF 403)
  probe_step60.py     # recurrent probe vs frozen base (emozilla/pg19; works now)
  recurrent_probe.py   # build patched model at chosen context_length, eval to 16384
runs/                 # logs (gitignored)
checkpoints/           # .pt ckpts (gitignored)
data_cache/            # training data mix (gitignored; scripts/cache_data.py)
```

## Train

```bash
export HF_TOKEN=$(sed -n '2p' token.txt)   # gitignored token file (line2 = HF)
python3 -u src/train.py --config configs/base.yaml
# resume:  --resume checkpoints/xlstm_cpt_stepNNNN.pt  (warmup restarts)
```

Key config fields (`configs/base.yaml`):
- `xlstm.block_type: slstm`, `xlstm.architecture: "1:0"`,
  `xlstm.bias_init: powerlaw_blockdependent`
- `xlstm.context_length` — sLSTM's context (parallel-scan length)
- `xlstm.base_ctx` — **hybrid**: frozen base sees local windows of this size
  (None = base sees full seq). Untrained experiment, see branch.
- `train.seq_len` — training sequence length (the base+data length)
- `train.max_steps`, `train.grad_accum`, `train.learning_rate`

## Probe (does the graft extend context?)

```bash
# natural-language long text (emozilla/pg19 — works, huge books):
python3 -u scripts/probe_step60.py --ckpt checkpoints/xlstm_cpt_step3000.pt \
        --probe-source emozilla/pg19 --lengths 2048 4096 8192 16384

# or build the patched model at a chosen sLSTM context and decode to 16384:
python3 -u scripts/recurrent_probe.py --ckpt <ckpt> --ctx 2048
```

Lower `patch_ppl` than `base_ppl` = the graft carries long-range memory.

## Known blockers / notes (2026-07-14)

- **Step time is FFN-bound, ~linear in seq_len.** 256→3.25s, 512→6.97s,
  1024→13.3s/step (~2x per doubling = linear O(L), NOT O(L²) attention).
  The frozen FFN dominates; the sLSTM block is only ~0.13s. Long-seq
  training is inherently ~13s/step at L=1024. No cheap fix on this toolchain
  (int8-FFN backward is broken on torch cu130; fused kernels blocked by a
  nvcc-vs-torch-CUDA-header mismatch; 0.5B is the smallest model we have).
- **HF datasets 403** for gated/streaming downloads with the current token
  (pg19/wikitext probed fine when the token recovered; `deepmind/pg19` is
  DEAD — HF dropped script-based datasets). `probe_offline.py` reads local
  `data_cache/mix.jsonl` to avoid the network entirely.
- **Eval is non-fatal**: a 403 on a probe dataset logs + skips, never aborts
  the run (commit 56c7ec2). A 403 used to silently kill 1.5h runs.

## Branches

- `main` — current working code (sLSTM graft, CPT, probing).
- `exp/hybrid-int8-parked` — the "base-256 / sLSTM-1024 + int8-FFN"
  experiment, **parked** (no training launched). See `HYBRID_INT8_TODO.md`
  on that branch for the hypothesis, proven findings, and the TODO.
