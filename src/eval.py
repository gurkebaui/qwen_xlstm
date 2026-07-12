# =============================================================================
# src/eval.py
#
# FAST, SUBSET-ONLY eval harness for the experimental stage.
#
# Why this exists: full eval suites (HumanEval, GSM8K, lm_eval) take ages
# and we are still experimenting. So this runs ONLY a small subsample and
# answers the questions that matter right now:
#
#   (a) "Does the grafted mLSTM change (hopefully lower) perplexity on
#       short-range text, vs the frozen base on the SAME data?"
#   (b) THE REAL ONE: "On LONG context (prefix > model window), does
#       the graft's recurrent memory beat the frozen base at next-token
#       prediction?"  -> that is the memory-capability hypothesis.
#
# What it does:
#   * loops over cfg['eval_probes'] (wikitext / code / pg19) via
#     STREAMING (no 1TB download), takes `subsample` docs, computes
#     perplexity for BOTH frozen base and patched model.
#   * runs a LONG-CONTEXT memory probe: feed a long book prefix
#     (4096 tok > Qwen's 2048 window), measure next-token ppl past
#     the window. Base can't see past its window; graft's mLSTM
#     carries state -> if it helps, patched ppl there is lower.
#
# IMPORTANT: this is a PROBE, not a benchmark. Numbers are comparable
# run-to-run (same subsample) but not publication-grade.
# =============================================================================

from typing import Dict, List, Optional

import torch
import torch.nn as nn

from transformers import Qwen2ForCausalLM


# --- streaming loaders (no full download) ----------------------------------
def _load_text_docs(probe: str, subsample: int, seq_len: int,
                     split: str = "train") -> List[torch.Tensor]:
    """Stream a text dataset, take `subsample` docs truncated to `seq_len`.

    Probe-agnostic: any HF text dataset with a 'text' (or 'content')
    field works. Supports "hf_id:subdir" -> datasets data_dir=subdir
    (e.g. bigcode/starcoderdata:python).
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    if probe == "wikitext":
        ds = load_dataset("wikitext", name="wikitext-103-v1",
                         split=split, streaming=True)
    elif ":" in probe:
        # "hf_id:subdir" -> datasets data_dir=subdir
        ds_id, sub = probe.split(":", 1)
        ds = load_dataset(ds_id, data_dir=sub, split=split, streaming=True)
    else:  # generic HF text dataset
        ds = load_dataset(probe, split=split, streaming=True)

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")

    out: List[torch.Tensor] = []
    taken = 0
    for row in ds:
        text = row.get("text") or row.get("content") or ""
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


# --- perplexity computation -----------------------------------------------
@torch.no_grad()
def _perplexity(model, doc_ids: List[torch.Tensor], device: str,
                max_len: int) -> float:
    """Perplexity = exp(mean token loss) over the eval docs.

    CORRECT aggregation: run each doc once (docs are already <= max_len),
    take HF's mean-CE loss for that doc, then weight across docs by
    token count. This matches the standard "concatenate valid tokens" ppl.

    (An earlier version chunked docs at max_len AND re-weighted, which
    over-estimated ppl ~2x -- see notes.md #8. Now: one forward
    per doc, token-weighted mean loss. No double counting.)
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    for ids in doc_ids:
        ids = ids.to(device)
        n = ids.numel()
        if n < 2:
            continue
        chunk = ids.unsqueeze(0)  # (1, n) -- doc already <= max_len
        out = model(input_ids=chunk, labels=chunk)
        # HF returns loss = MEAN CE over the doc's n-1 shifted targets
        total_loss += out.loss.item() * (n - 1)
        total_tokens += (n - 1)
    if total_tokens == 0:
        return float("nan")
    mean_loss = total_loss / total_tokens
    return float(torch.exp(torch.tensor(mean_loss)).item())


# --- long-context MEMORY probe (the real "does memory help?" test) ----
@torch.no_grad()
def long_context_eval(patched_model, base_model, device: str,
                     prefix_len: int = 4096, pred_len: int = 512,
                     probe: str = "emozilla/pg19", n_books: int = 4):
    """The probe that actually tests the graft's memory.

    A standard next-token perplexity only sees `seq_len` tokens, so it
    never exercises long-range recall. Here we take a LONG book (pg19,
    100k+ tok), feed `prefix_len` tokens as context (4096 > Qwen's
    effective 2048 window), then measure next-token ppl on the
    FOLLOWING `pred_len` tokens.

    The frozen base can't see past its window, so its prediction there
    is from short-range only. The graft's mLSTM carries recurrent
    state across the whole prefix -> if it helps, patched ppl on the
    post-window chunk should be LOWER than base. That's the win.
    The delta here is the metric that matters, not wikitext ppl.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")

    ds = load_dataset(probe, split="train", streaming=True)
    ds = load_dataset(probe, split="train", streaming=True)
    it = iter(ds)
    pl_base, pl_patch = 0.0, 0.0
    tk_base, tk_patch = 0, 0
    books = 0
    scanned = 0
    MAX_SCANNED = 400   # safety: never hang scanning short rows
    need_chars = (prefix_len + pred_len) * 2  # ~min chars for a prefix+pred split
    while books < n_books and scanned < MAX_SCANNED:
        try:
            row = next(it)
        except StopIteration:
            break
        scanned += 1
        text = row.get("text") or ""
        if len(text) < need_chars:
            continue   # too short to split into prefix+pred
        ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=prefix_len + pred_len).input_ids[0].to(device)
        if ids.numel() < prefix_len + pred_len:   # need full prefix+pred
            continue
        pre = ids[:prefix_len]

        # base (frozen, no recurrent memory across the long prefix)
        b = base_model(input_ids=pre.unsqueeze(0),
                        labels=pre.unsqueeze(0))
        pl_base += b.loss.item() * (pre.numel() - 1)
        tk_base += pre.numel() - 1
        # patched (mLSTM carries state across the prefix)
        p = patched_model(input_ids=pre.unsqueeze(0),
                                 labels=pre.unsqueeze(0))
        pl_patch += p.loss.item() * (pre.numel() - 1)
        tk_patch += pre.numel() - 1
        books += 1
        print(f"[long]   book {books}/{n_books} ok (scanned {scanned})")

    if tk_base == 0:
        return {"prefix_ppl_base": float("nan"),
                "prefix_ppl_patch": float("nan"), "delta": float("nan")}
    lb = pl_base / tk_base
    lp = pl_patch / tk_patch
    res = {
        "prefix_ppl_base": float(torch.exp(torch.tensor(lb)).item()),
        "prefix_ppl_patch": float(torch.exp(torch.tensor(lp)).item()),
        "delta": float(torch.exp(torch.tensor(lp)).item()
                   - torch.exp(torch.tensor(lb)).item()),
    }
# --- VARIABLE-CONTEXT probe (find the mLSTM limit) ----------------
# Method matches the xLSTM paper's own extrapolation test (Fig 7:
# "trained at 2048, tested at 16384"): PARALLEL forward at
# increasing context length. The paper does NOT use recurrent decode
# for this eval -- it feeds the long prefix in parallel and measures
# next-token ppl. So we build the eval model with a LARGE
# context_length (so the mLSTM's internal causal-mask buffer fits
# the longest L) and run parallel .forward at each L.
#
# WHY this over the .step() recurrent walk: walking token-by-token
# through HF's attention with swapped decoder layers hits a
# transformers-5.5 RoPE/position_embeddings bug (the model's
# top-level forward normally pre-computes cos/sin and passes them
# DOWN to attention; our custom layer skips that). Parallel forward
# is bug-free AND is what the paper actually does. (Recurrent
# .step() decode is still the right path for GENERATION; that's a
# separate, deferred fix -- noted in notes.md.)
@torch.no_grad()
def variable_context_eval(patched_model, base_model, device: str,
                         lengths=(2048, 4096, 8192, 16384),
                         eval_len: int = 256, probe: str = "emozilla/pg19",
                         n_books: int = 3, max_scanned: int = 300):
    """Measure next-token ppl at increasing prefix lengths L.

    For each L: take a pg19 book, tokenize to L+eval_len, run a
    SINGLE parallel forward over the whole (prefix+post) for BOTH
    base and patched, and measure ppl on the post-L tokens only.

    * Base (frozen Qwen, native 32k): ppl should HOLD or only
      mildly degrade as L grows (attention sees all L tokens).
    * Patched (mLSTM graft, built with context_length>=max L):
      if the graft's memory generalizes like the paper says, patched
      ppl should stay flat (or beat base) as L -> 16384. If it
      explodes, that's the mLSTM's effective limit.

    Returns: {L: {base_ppl, patch_ppl, delta}}.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")

    max_L = max(lengths)
    need_chars = (max_L + eval_len) * 2

    ds = load_dataset(probe, split="train", streaming=True)
    it = iter(ds)
    books_text = []
    scanned = 0
    while len(books_text) < n_books and scanned < max_scanned:
        scanned += 1
        row = next(it, None)
        if row is None:
            break
        text = row.get("text") or ""
        if len(text) < need_chars:
            continue
        books_text.append(text)

    if not books_text:
        print(f"[vctx] FOUND 0 books >= {need_chars} chars after "
              f"{max_scanned} scanned -> cannot run")
        return {}

    ce = torch.nn.functional.cross_entropy
    out = {}
    for L in lengths:
        pl_base, pl_patch, tk = 0.0, 0.0, 0
        ok = 0
        for text in books_text:
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=L + eval_len).input_ids[0].to(device)
            if ids.numel() < L + eval_len:
                continue
            pre, post = ids[:L], ids[L:L + eval_len]
            full = torch.cat([pre, post]).unsqueeze(0)

            # HF logits[i] predict token i+1, so to predict post
            # (= full[L : L+eval_len]) we use logits at L-1 : L-1+eval_len.
            b_out = base_model(input_ids=full, use_cache=False)
            b_l = ce(b_out.logits[0, L - 1 : L - 1 + eval_len].float(),
                     post, reduction="sum").item()
            pl_base += b_l

            p_out = patched_model(input_ids=full, use_cache=False)
            p_l = ce(p_out.logits[0, L - 1 : L - 1 + eval_len].float(),
                     post, reduction="sum").item()
            pl_patch += p_l
            tk += post.numel()
            ok += 1

        if tk == 0:
            print(f"[vctx] L={L}: no usable books, skip")
            continue
        rb = float(torch.exp(torch.tensor(pl_base / tk)).item())
        rp = float(torch.exp(torch.tensor(pl_patch / tk)).item())
        out[L] = {"base_ppl": rb, "patch_ppl": rp, "delta": rp - rb}
        print(f"[vctx] L={L:6d} books={ok}: base_ppl={rb:.2f} "
              f"patch_ppl={rp:.2f} delta={rp - rb:+.2f} "
              f"({'patch BETTER' if rp < rb else 'patch WORSE'})")
    return out


# --- public entry ---------------------------------------------------------
def quick_eval(patched_model, cfg: Dict, device: str,
               base_model=None) -> Dict[str, float]:
    """Run the quick subset eval over ALL configured probes.

    `cfg` is the eval section of base.yaml. It loops over
    `cfg['eval_probes']` (default ['wikitext']) so we get ppl on
    short-range (wikitext), code (starcoderdata), and long-text
    (pg19) in one call. Returns a dict keyed by probe.

    `base_model` (optional): a PRE-LOADED fresh Qwen2ForCausalLM to
    use as the reference (cached across probes). NEVER pass
    patched_model.backbone (see notes.md #8).
    """
    subsample = int(cfg.get("subsample", 32))
    seq_len = int(cfg.get("seq_len", 512))
    probes = cfg.get("eval_probes", cfg.get("probe", "wikitext"))
    if isinstance(probes, str):
        probes = [probes]

    # --- frozen base (reference) = a CLEAN, separately-loaded Qwen ---
    # CRITICAL: do NOT use patched_model.backbone here. During construction
    # we swap the backbone's layers IN-PLACE for XlstmQwenLayer, so
    # patched_model.backbone IS the patched model, not a clean base.
    # Using it would make base==patched and always report delta=0.
    if base_model is None:
        base_model = Qwen2ForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-Coder-0.5B", torch_dtype=torch.bfloat16
        ).to(device)
        base_model.requires_grad_(False)
        _own_base = True
    else:
        _own_base = False

    results = {}
    for probe in probes:
        print(f"[eval] streaming {probe} (subset={subsample}, seq_len={seq_len}) ...")
        docs = _load_text_docs(probe=probe, subsample=subsample, seq_len=seq_len)
        print(f"[eval] got {len(docs)} docs, computing perplexity ...")
        base_ppl = _perplexity(base_model, docs, device, max_len=seq_len)
        patched_ppl = _perplexity(patched_model, docs, device, max_len=seq_len)
        delta = patched_ppl - base_ppl
        results[probe] = {
            "base_ppl": base_ppl, "patched_ppl": patched_ppl,
            "delta_ppl": delta,
        }
        print(f"[eval]   {probe}: base={base_ppl:.3f} patched={patched_ppl:.3f} "
              f"delta={delta:+.3f} ({'BETTER' if delta < 0 else 'WORSE'})")

    if _own_base:
        del base_model
        torch.cuda.empty_cache()

    # --- long-context memory probe (the metric that actually matters) ---
    long_cfg = cfg.get("long_context", {})
    if long_cfg.get("enabled", True):
        print("[eval] running LONG-CONTEXT memory probe (pg19, prefix>window) ...")
        ref = base_model if not _own_base else \
            Qwen2ForCausalLM.from_pretrained(
                "Qwen/Qwen2.5-Coder-0.5B", torch_dtype=torch.bfloat16
            ).to(device).requires_grad_(False)
        long_res = long_context_eval(
            patched_model, ref, device,
            prefix_len=int(long_cfg.get("prefix_len", 4096)),
            pred_len=int(long_cfg.get("pred_len", 512)),
            probe=long_cfg.get("probe", "emozilla/pg19"),
            n_books=int(long_cfg.get("n_books", 4)),
        )
        if _own_base:
            del ref
            torch.cuda.empty_cache()
        results["long_context"] = long_res

    return results
