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
    """Perplexity = exp(mean token loss) over the eval docs.

    CORRECT aggregation: run each doc once (docs are already <= max_len),
    take HF's mean-CE loss for that doc, then weight across docs by token
    count. This matches the standard "concatenate valid tokens" ppl.

    (Earlier version chunked docs at max_len AND re-weighted, which
    over-estimated ppl ~2x — see notes.md #8. Now: one forward per doc,
    token-weighted mean loss. No double counting.)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for ids in doc_ids:
        ids = ids.to(device)
        n = ids.numel()
        if n < 2:
            continue
        chunk = ids.unsqueeze(0)  # (1, n)  — doc already <= max_len
        out = model(input_ids=chunk, labels=chunk)
        # HF returns loss = MEAN CE over the doc's n-1 shifted targets
        total_loss += out.loss.item() * (n - 1)
        total_tokens += (n - 1)
    if total_tokens == 0:
        return float("nan")
    mean_loss = total_loss / total_tokens
    return float(torch.exp(torch.tensor(mean_loss)).item())


# --- public entry ---------------------------------------------------------
def quick_eval(patched_model, cfg: Dict, device: str,
               base_model=None) -> Dict[str, float]:
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

    # --- frozen base (reference) = a CLEAN, separately-loaded Qwen ---
    # CRITICAL: do NOT use patched_model.backbone here. During construction
    # we swap the backbone's layers IN-PLACE for XlstmQwenLayer, so
    # patched_model.backbone IS the patched model, not a clean base. Using it
    # would make base==patched and always report delta=0 (this bug burned a
    # full 2000-step run's eval — see notes.md #8). Load a real fresh base.
    if base_model is None:
        base_model = Qwen2ForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Coder-0.5B", torch_dtype=torch.bfloat16
        ).to(device)
        base_model.requires_grad_(False)
        _own_base = True
    else:
        _own_base = False

    base_ppl = _perplexity(base_model, docs, device, max_len=seq_len)
    if _own_base:
        del base_model
        torch.cuda.empty_cache()

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
