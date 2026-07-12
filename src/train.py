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
# VRAM-SAFE: config is verified by scripts/vram_probe.py
#   (seq_len=2048, grad_accum=1, ~13.9GB on 16GB). Do NOT bump
#   seq_len*grad_accum past ~4096 without re-running the probe.
#
# Run a real run:
#     python src/train.py --config configs/base.yaml
# Smoke (2 steps, tiny subset) — see tests/smoke_train.py
# =============================================================================

import argparse
import math
import os
import sys
import time

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


# --- data: stream a long-text dataset, tokenize to seq_len chunks -------
def stream_tokens(cfg, max_docs: int = None):
    """Yield token-id tensors of length seq_len from the configured probe.

    Streaming (no full-corpus download). `max_docs=None` = unbounded
    stream (real run). Returns an iterator of 1D token tensors.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    probe = cfg["eval"].get("probe", "wikitext")
    seq_len = int(cfg["train"].get("seq_len", 2048))
    tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])

    if probe == "wikitext":
        ds = load_dataset("wikitext", name="wikitext-103-v1",
                         split="train", streaming=True)
    else:
        ds = load_dataset(probe, split="train", streaming=True)

    count = 0
    buf = []
    for row in ds:
        text = row.get("text") or ""
        if not text or len(text) < 32:
            continue
        ids = tok(text).input_ids
        buf.extend(ids)
        while len(buf) >= seq_len:
            yield torch.tensor(buf[:seq_len], dtype=torch.long)
            del buf[:seq_len]
        count += 1
        if max_docs is not None and count >= max_docs:
            break
    if buf:  # flush remainder (pad to seq_len for a well-formed last batch)
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
def train(config_path: str, smoke: bool = False):
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

    step = 0
    t0 = time.time()
    for ids in data:
        if step >= max_steps:
            break
        ids = ids.unsqueeze(0).to(device)  # (1, seq_len)
        opt.zero_grad()
        out = model(input_ids=ids, labels=ids)
        out.loss.backward()
        opt.step()
        sched.step()
        step += 1

        if step % 1 == 0:
            lr = sched.get_last_lr()[0]
            print(f"[train] step {step}/{max_steps}  loss={out.loss.item():.4f}  "
                  f"lr={lr:.2e}  ({time.time()-t0:.1f}s)")

        # periodic eval: patched vs frozen-base perplexity delta
        if step % eval_every == 0:
            eval_cfg = dict(cfg["eval"])
            eval_cfg["subsample"] = 4 if smoke else int(eval_cfg.get("subsample", 32))
            eval_cfg["seq_len"] = min(seq_len, 128) if smoke else int(eval_cfg.get("seq_len", 512))
            res = quick_eval(model, eval_cfg, device=device)
            print(f"[train]   eval delta_ppl={res['delta_ppl']:+.3f} "
                  f"(base={res['base_ppl']:.2f} patched={res['patched_ppl']:.2f})")

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
    args = ap.parse_args()
    train(args.config, smoke=args.smoke)
