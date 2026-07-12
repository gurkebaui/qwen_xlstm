# =============================================================================
# src/eval.py
#
# FAST, SUBSET-ONLY eval harness for the experimental stage.
#
# Why this exists: full eval suites (HumanEval, GSM8K, lm_eval) take ages and
# we are still experimenting. So this runs ONLY a small subsample of data and
# answers the one question that matters right now:
#
#     "Does the grafted mLSTM change (hopefully lower) perplexity on
#      long-range text, vs the frozen base on the SAME data?"
#
# That directly probes the "memory capability" hypothesis cheaply.
#
# What it does:
#   * loads the `eval_probe` dataset (default: pg19 — long Gutenberg books,
#     the natural long-range test) via STREAMING (no 1TB download),
#   * takes only `eval.subsample` documents (default 32) — fast,
#   * truncates each to `eval.seq_len` tokens,
#   * computes perplexity = exp(mean token loss) for BOTH the frozen base and
#     the patched model on that identical subset,
#   * returns the delta (patched - base). Lower patched ppl = good.
#
# It also supports an optional `code` subset (a few code files) toggled by
# `eval.subsets` — off by default to keep it quick.
#
# IMPORTANT: this is a PROBE, not a benchmark. Numbers are comparable
# run-to-run (same subsample seed) but not publication-grade.
# =============================================================================

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from transformers import Qwen2ForCausalLM


# --- streaming loaders (no full download) ----------------------------------
def _load_text_docs(probe: str, subsample: int, seq_len: int,
                  split: str = "test") -> List[torch.Tensor]:
    """Stream a long-text dataset, take `subsample` docs truncated to `seq_len`.

    Probe-agnostic: any HF text dataset with a 'text' field works.
    Defaults: 'wikitext' (wikitext-103-v1) — a real LM corpus that
    streams cleanly on datasets>=4.8.

    NOTE on pg19: pg19 is a *scripted* dataset and datasets>=4.8 dropped
    script loaders, so it currently raises. It's still the INTENDED long-range
    probe (Gutenberg books, ~20x longer than WikiText); once we add a parquet
    loader (or pin datasets<4.8) we flip the default back to it. The eval
    LOGIC is identical either way.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if probe == "pg19":
        raise NotImplementedError(
            "pg19 needs a parquet loader or datasets<4.8 (script loaders dropped "
            "in datasets>=4.8). Use probe:'wikitext' for now; pg19 is the "
            "intended long-range probe and will be re-enabled."
        )

    if probe == "wikitext":
        ds = load_dataset("wikitext", name="wikitext-103-v1",
                         split=split, streaming=True)
    else:  # generic: assume `probe` is a HF dataset id with a 'text' field
        ds = load_dataset(probe, split=split, streaming=True)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")

    out: List[torch.Tensor] = []
    taken = 0
    for row in ds:
        text = row.get("text") or ""
        if not text or len(text) < 32:   # skip empty / tiny docs
            continue
        ids = tok(text, return_tensors="pt", truncation=True,
                  max_length=seq_len).input_ids[0]
        if ids.numel() < 8:
            continue
        out.append(ids)
        taken += 1
        if taken >= subsample:
            break
    if taken == 0:
        raise RuntimeError(
            f"no usable docs streamed from probe='{probe}' (all empty/too short?)"
        )
    return out


# --- perplexity computation ------------------------------------------------
@torch.no_grad()
def _perplexity(model, doc_ids: List[torch.Tensor], device: str,
                max_len: int) -> float:
    """Mean perplexity over docs. ppl = exp(mean token loss).

    We feed each doc in chunks of `max_len` (handling long docs) and average
    the losses weighted by token count. Labels are shifted internally by HF.
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for ids in doc_ids:
        ids = ids.to(device)
        n = ids.numel()
        # chunk to avoid OOM on very long docs; sum losses weighted by tokens
        seg_loss = 0.0
        seg_tok = 0
        for s in range(0, n, max_len):
            chunk = ids[s:s + max_len].unsqueeze(0)  # (1, L)
            if chunk.numel() < 2:
                continue
            out = model(input_ids=chunk, labels=chunk)
            # HF returns loss averaged over the chunk's tokens
            seg_loss += out.loss.item() * (chunk.numel() - 1)
            seg_tok += (chunk.numel() - 1)
        if seg_tok > 0:
            total_loss += seg_loss
            total_tokens += seg_tok
    if total_tokens == 0:
        return float("nan")
    mean_loss = total_loss / total_tokens
    return float(torch.exp(torch.tensor(mean_loss)).item())


# --- public entry ---------------------------------------------------------
def quick_eval(patched_model, cfg: Dict, device: str) -> Dict[str, float]:
    """Run the quick subset eval. Compares patched vs frozen base.

    `cfg` is the eval section of base.yaml (dict). Returns a dict of ppl
    numbers + the delta.
    """
    subsample = int(cfg.get("subsample", 32))
    seq_len = int(cfg.get("seq_len", 512))
    probe = cfg.get("probe", "wikitext")

    print(f"[eval] streaming {probe} (subset={subsample}, seq_len={seq_len}) ...")
    docs = _load_text_docs(probe=probe, subsample=subsample, seq_len=seq_len)
    print(f"[eval] got {len(docs)} docs, computing perplexity ...")

    # --- frozen base (reference) ---
    # FIX (2026-07-12): do NOT re-instantiate the 0.5B model — the 2nd
    # from_pretrained() STALLS on this box (see notes.md 6e). The patched
    # model HOLDS the frozen backbone, so we evaluate that directly.
    # It is identical to the base (backbone is frozen, untouched).
    base_model = getattr(patched_model, "backbone", None)
    if base_model is None:
        # fallback: standalone model (e.g. called on a bare Qwen)
        base = Qwen2ForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Coder-0.5B", torch_dtype=torch.bfloat16
        ).to(device)
        base.requires_grad_(False)
        base_ppl = _perplexity(base, docs, device, max_len=seq_len)
        del base
        torch.cuda.empty_cache()
    else:
        base_ppl = _perplexity(base_model, docs, device, max_len=seq_len)

    # --- patched model ---
    patched_ppl = _perplexity(patched_model, docs, device, max_len=seq_len)

    delta = patched_ppl - base_ppl
    result = {
        "base_ppl": base_ppl,
        "patched_ppl": patched_ppl,
        "delta_ppl": delta,  # <0 means patched is better
    }
    print(f"[eval] base ppl   = {base_ppl:.3f}")
    print(f"[eval] patched ppl = {patched_ppl:.3f}")
    print(f"[eval] delta       = {delta:+.3f}  "
          f"({'patched BETTER' if delta < 0 else 'patched WORSE'})")
    return result
