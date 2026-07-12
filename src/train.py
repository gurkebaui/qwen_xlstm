# =============================================================================
# src/train.py
#
# Stage-1 CONTINUED PRETRAINING (CPT) loop for the grafted xLSTM model.
#
# What it does:
#   * builds XlstmQwenModel from configs/base.yaml
#   * identity-init (so step0 == frozen base; see notes #1/#3)
#   * FREEZES the base; trains ONLY the xlstm sublayers + their norms
#   * streams a long-text dataset (subset only for speed; full for real runs)
#   * AdamW with linear warmup -> cosine decay
#   * logs train loss; every `eval_every` steps runs quick_eval()
#     (patched vs frozen-base perplexity delta — the memory probe)
#   * saves checkpoints to paths.checkpoints (NEVER /tmp — disk rule)
#
# VRAM-SAFE: config verified by scripts/vram_probe.py
#   2048 x accum=1 -> OOM at step 1 (needs 14.65GB, only 1.17 free)
#   1024 x accum=1 -> 7.2GB   (GO)
#   1024 x accum=2 -> ~7-8GB  (GO, USED: 2048 tokens/optimizer-step)
# Effective batch = seq_len * per_device_batch * grad_accum.
# Do NOT set seq_len*grad_accum past ~2048 without re-running the probe
# AND confirming peak < 14GB (the 16GB card has no real headroom).
#
# Run a real run:
#     python src/train.py --config configs/base.yaml
# Smoke (2 steps, tiny subset) — see tests/smoke_train.py
# =============================================================================

# FRAG-OOM FIX (2026-07-12): at seq_len=2048 the 16GB card is near the
# VRAM limit; PyTorch's default allocator fragments and a 1.16GB chunk fails
# with ~1.17GB "free" (it can't find a contiguous block). expandable_segments
# lets it reuse freed segments. MUST be set BEFORE torch/CUDA initializes.
# Previously set only as a shell env var (PYTORCH_CUDA_ALLOC_CONF=...) but
# the conda/shell wrapper STRIPPED it -> the real run OOM'd at step 1 with
# PyTorch's own "try setting expandable_segments" hint. Setting it HERE in
# Python is bulletproof (can't be stripped).
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import math
import os
import sys
import time

# FRAG-OOM FIX (2026-07-12): at seq_len=2048 the 16GB card is near the
# VRAM limit; PyTorch's default allocator fragments and a 1.16GB contiguous
# alloc can fail even with ~900MB free. expandable_segments lets it grow
# segments instead of failing. Set BEFORE torch allocates anything.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/..")
from src.patcher import XlstmQwenModel
from src.xlstm_layer import XLSTMLayerConfig
from src.eval import quick_eval

try:
    import yaml
except ImportError:
    yaml = None
try:
    from omegaconf import OmegaConf
except ImportError:
    OmegaConf = None


# --- config loading ----------------------------------------------------
def load_config(path: str) -> dict:
    """Load base.yaml. Tries yaml, falls back to OmegaConf."""
    if yaml is None and OmegaConf is None:
        raise RuntimeError("need pyyaml or omegaconf to read base.yaml")
    with open(path) as f:
        if yaml is not None:
            return yaml.safe_load(f)
        return OmegaConf.load(path)


# --- data: stream a mix of long-text sources, tokenize to seq_len chunks -
def _open_source(src: str, split: str = "train"):
    """Open one configured source as a streaming HF dataset.

    `src` may be:
      * "wikitext"                       -> wikitext-103-v1 (special-cased)
      * "org/dset"                       -> HF dataset, split=train, 'text' field
      * "org/dset:config_name"           -> HF dataset with that config
    Returns (dataset_iterator, text_field_name).
    """
    from datasets import load_dataset
    if src == "wikitext":
        return load_dataset("wikitext", name="wikitext-103-v1",
                           split=split, streaming=True), "text"
    if ":" in src:
        ds_id, sub = src.split(":", 1)
        # `:sub` -> datasets `data_dir=sub` (e.g. bigcode/starcoderdata:python).
        # (We deliberately DON'T use `name=` here: starcoderdata selects its
        #  language subset via data_dir, not config name.)
        ds = load_dataset(ds_id, data_dir=sub, split=split, streaming=True)
    else:
        ds = load_dataset(src, split=split, streaming=True)
    # detect the text field (most of our sources use 'text')
    sample = next(iter(ds))
    field = "text" if "text" in sample else next(
        (k for k in sample.keys() if isinstance(sample[k], str)), "text"
    )
    return ds, field


def stream_tokens(cfg, max_docs: int = None):
    """Interleave the configured `data.sources`, yielding seq_len token chunks.

    Streaming (no full-corpus download). `max_docs=None` = unbounded
    stream (real run). Rotates through sources so the batch mix stays
    diverse (math / long-text / baseline) across the run.

    LOCAL CACHE: if data_cache/mix.jsonl exists (written by
    scripts/cache_data.py), training reads from it instead of hitting HF
    Hub live — so the run does NOT depend on the network staying up.
    """
    from transformers import AutoTokenizer

    seq_len = int(cfg["train"].get("seq_len", 2048))
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])

    cache = "data_cache/mix.jsonl"
    if os.path.exists(cache):
        print(f"[data] using local cache {cache} (no live HF stream)")
        count = 0
        buf = []
        with open(cache) as f:
            for line in f:
                row = json.loads(line)
                text = row.get("text") or ""
                if not text or len(text) < 32:
                    continue
                buf.extend(tok(text).input_ids)
                while len(buf) >= seq_len:
                    yield torch.tensor(buf[:seq_len], dtype=torch.long)
                    del buf[:seq_len]
                    count += 1
                    if max_docs is not None and count >= max_docs:
                        return
        if buf:
            buf = (buf + [tok.eos_token_id] * seq_len)[:seq_len]
            yield torch.tensor(buf, dtype=torch.long)
        return

    # --- live streaming fallback (only if no cache) ---
    sources = cfg["data"].get("sources", ["wikitext"])
    streams = []
    for src in sources:
        try:
            ds, field = _open_source(src)
            streams.append((ds, field, src))
        except Exception as e:
            print(f"[data] WARNING skipping source '{src}': {repr(e)[:120]}")
    assert streams, "no data sources could be opened"

    iters = [iter(ds) for ds, _, _ in streams]
    count = 0
    buf = []
    while True:
        progressed = False
        for si, (it, field, src) in enumerate(streams):
            try:
                row = next(iters[si])
            except StopIteration:
                continue
            progressed = True
            text = row.get(field) or ""
            if not text or len(text) < 32:
                continue
            buf.extend(tok(text).input_ids)
            while len(buf) >= seq_len:
                yield torch.tensor(buf[:seq_len], dtype=torch.long)
                del buf[:seq_len]
                count += 1
                if max_docs is not None and count >= max_docs:
                    return
        if not progressed:
            break
    if buf:
        buf = (buf + [tok.eos_token_id] * seq_len)[:seq_len]
        yield torch.tensor(buf, dtype=torch.long)

# --- optimizer with warmup + cosine -------------------------------------
def build_optimizer(model, lr: float, warmup: int, max_steps: int):
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95))

    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        prog = (step - warmup) / max(1, max_steps - warmup)
        prog = min(1.0, max(0.0, prog))
        return 0.5 * (1.0 + math.cos(math.pi * prog))  # cosine decay

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    return opt, sched


# --- main loop ----------------------------------------------------------
def train(config_path: str, smoke: bool = False, resume: str = None):
    cfg = load_config(config_path)
    device = cfg["model"].get("device", "cuda")
    dtype = getattr(torch, cfg["model"].get("dtype", "bfloat16"))
    # our XLSTMLayerConfig only takes the real-layer dims; strip plan-level
    # switches (variant/lora_on_base/insert_all_layers) that live under xlstm:.
    _dim_keys = {"embedding_dim", "num_heads", "context_length",
                 "proj_factor", "conv1d_kernel", "bias", "dropout"}
    xlstm_cfg = XLSTMLayerConfig(
        **{k: v for k, v in cfg["xlstm"].items() if k in _dim_keys}
    )
    train_cfg = cfg["train"]

    # --- model ---
    model = XlstmQwenModel(
        model_id=cfg["model"]["name"],
        xlstm_cfg=xlstm_cfg, device=device, dtype=dtype,
    ).to(device)
    if train_cfg.get("identity_init", True):
        model.init_identity()
    # RESUME: load prior weights + continue from saved step (saves
    # re-burning already-trained steps). Optimizer/LR sched are reset
    # (warmup restarts) — acceptable for CPT continuation.
    start_step = 0
    if resume:
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        if "model" in ckpt:
            model.load_state_dict(ckpt["model"])
        start_step = int(ckpt.get("step", 0))
        print(f"[train] RESUMED from {resume} (step {start_step})")
    model.train()

    n_train = model.num_trainable_params()
    n_total = model.num_total_params()
    print(f"[train] trainable={n_train:,} / total={n_total:,} "
          f"({100*n_train/n_total:.2f}% trainable)")

    # --- optimizer ---
    opt, sched = build_optimizer(
        model, float(train_cfg["learning_rate"]),
        int(train_cfg["warmup_steps"]), int(train_cfg["max_steps"]),
    )

    # --- data ---
    # smoke: tiny subset + very short seq_len so 2 docs yield >=2 chunks
    # (proves multi-step + step-2 checkpoint). VRAM trivially fine.
    max_docs = 2 if smoke else None
    seq_len = 64 if smoke else int(train_cfg.get("seq_len", 2048))
    # override the data stream's seq_len for the smoke run
    cfg["train"]["seq_len"] = seq_len
    data = stream_tokens(cfg, max_docs=max_docs)

    # --- loop ---
    max_steps = 2 if smoke else int(train_cfg["max_steps"])
    # smoke: eval ONCE at the end (per-step eval re-loads the 0.5B base and
    # stalls on this box). Real runs eval every eval_every steps.
    eval_every = max_steps if smoke else int(train_cfg.get("eval_every", 200))
    ckpt_dir = cfg["paths"]["checkpoints"]
    os.makedirs(ckpt_dir, exist_ok=True)

    step = start_step
    t0 = time.time()
    best_delta = float("inf")   # for the validation gate
    # gate fires at gate_steps (default 20) in the real run; in smoke it's
    # effectively disabled (gate only triggers on a dedicated early eval).
    gate_steps = int(train_cfg.get("gate_steps", 20))
    grad_accum = int(train_cfg.get("grad_accum", 1))
    micro = 0          # micro-step counter within an accumulation cycle
    run_loss = 0.0     # summed loss over the cycle (for logging)
    for ids in data:
        if step >= max_steps:
            break
        ids = ids.unsqueeze(0).to(device)  # (1, seq_len)
        out = model(input_ids=ids, labels=ids)
        # scale loss by accum so the effective batch grad magnitude matches
        # a single big batch (avoid lr effectively *accum)
        (out.loss / grad_accum).backward()
        run_loss += out.loss.item()
        micro += 1

        if micro < grad_accum:
            continue  # accumulate more micro-steps before stepping

        # --- end of accumulation cycle: clip + step ONCE ---
        # GRAD CLIP (stability fix, notes #9/i): a from-scratch graft in
        # bf16 can produce huge grads -> clip to keep it bounded.
        gclip = float(train_cfg.get("grad_clip", 1.0))
        torch.nn.utils.clip_grad_norm_(model.parameters(), gclip)
        opt.step()
        sched.step()
        opt.zero_grad()
        step += 1
        micro = 0

        if step % 1 == 0:
            lr = sched.get_last_lr()[0]
            print(f"[train] step {step}/{max_steps}  loss={run_loss/grad_accum:.4f}  "
                  f"lr={lr:.2e}  ({time.time()-t0:.1f}s)")
        run_loss = 0.0

        # periodic eval: patched vs frozen-base perplexity delta.
        # Also force an eval at the validation-gate step (step 20 by default)
        # so we catch divergence EARLY, before the next scheduled eval.
        force_gate = (not smoke) and step == gate_steps
        if step % eval_every == 0 or force_gate:
            from transformers import Qwen2ForCausalLM
            if "ref_base" not in locals():
                ref_base = Qwen2ForCausalLM.from_pretrained(
                    cfg["model"]["name"], torch_dtype=dtype
                ).to(device)
                ref_base.requires_grad_(False)
            eval_cfg = dict(cfg["eval"])
            eval_cfg["subsample"] = 4 if smoke else int(eval_cfg.get("subsample", 32))
            eval_cfg["seq_len"] = min(seq_len, 128) if smoke else int(eval_cfg.get("seq_len", 512))
            res = quick_eval(model, eval_cfg, device=device, base_model=ref_base)
            # quick_eval returns {probe: {base_ppl, patched_ppl, delta_ppl}}
            #   (keyed per-probe). Summarize each, pick a representative
            #   delta for the gate (wikitext if present, else first probe).
            for probe_name, pdict in res.items():
                if probe_name == "long_context":
                    continue  # long-context handled by its own print below
                if not isinstance(pdict, dict):
                    continue
                print(f"[train]   eval[{probe_name}] "
                      f"delta_ppl={pdict.get('delta_ppl', float('nan')):+.3f} "
                      f"(base={pdict.get('base_ppl', float('nan')):.2f} "
                      f"patched={pdict.get('patched_ppl', float('nan')):.2f})")
            # representative delta for the validation gate
            gate_probe = "wikitext" if "wikitext" in res else next(
                (k for k in res if isinstance(res[k], dict)), None)
            gate_delta = (res[gate_probe]["delta_ppl"]
                          if gate_probe and isinstance(res.get(gate_probe), dict)
                          else float("nan"))

            # --- 20-STEP VALIDATION GATE (notes #9/i) ---
            # Abort early if the recipe diverges: if at the gate the patched
            # model is WORSE than base by a lot, the recipe is bad -> stop
            # instead of burning a full run. Only in the real (non-smoke) run.
            if force_gate:
                if gate_delta > 5.0:   # patched >5 ppl worse than base
                    print(f"[train] VALIDATION GATE FAILED at step {step}: "
                          f"delta_ppl={gate_delta:.2f} (patched much worse). "
                          f"Aborting — recipe diverges. Fix LR/clip/gate, don't "
                          f"waste a full run.")
                    return ckpt_dir  # no checkpoint of a diverged run
                else:
                    print(f"[train] VALIDATION GATE PASSED at step {step}: "
                          f"delta_ppl={gate_delta:+.2f}. Recipe stable; "
                          f"continuing full run.")

    # --- checkpoint (NEVER /tmp) ---
    ckpt_path = os.path.join(ckpt_dir, "xlstm_cpt_step%d.pt" % step)
    torch.save({"step": step, "model": model.state_dict()}, ckpt_path)
    print(f"[train] saved checkpoint -> {ckpt_path}")
    print(f"[train] DONE in {time.time()-t0:.1f}s")
    return ckpt_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--smoke", action="store_true",
                   help="2 steps on a tiny subset (proves the loop without a real run)")
    ap.add_argument("--max-steps", type=int, default=None,
                   help="override train.max_steps (e.g. 20 for a probe run)")
    ap.add_argument("--resume", type=str, default=None,
                   help="path to a checkpoint to resume from (loads weights + step)")
    args = ap.parse_args()
    if args.max_steps is not None:
        # inject override into config before train() reads it
        import yaml
        with open(args.config) as f:
            _cfg = yaml.safe_load(f) if yaml else __import__("omegaconf").OmegaConf.load(f)
        _cfg["train"]["max_steps"] = args.max_steps
        import tempfile, os
        os.makedirs("runs", exist_ok=True)
        _p = os.path.join("runs", "probe_cfg.yaml")  # project dir, not /tmp
        with open(_p, "w") as f:
            if yaml: yaml.safe_dump(_cfg, f)
            else: __import__("omegaconf").OmegaConf.save(_cfg, _p)
        args.config = _p
    train(args.config, smoke=args.smoke, resume=args.resume)
