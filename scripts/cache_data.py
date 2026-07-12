# =============================================================================
# scripts/cache_data.py
#
# Pre-download the data mix from base.yaml `data.sources` into a LOCAL file
# so the training run does NOT depend on HF Hub staying reachable for the
# whole run (streaming mid-run is flaky — transient 'Bad file descriptor'
# blips, see notes.md). Run this ONCE before launching training:
#
#     python scripts/cache_data.py --docs-per-source 2000
#
# Writes data_cache/mix.jsonl  (one text doc per line). Training's
# stream_tokens() reads this file if present, else falls back to streaming.
# The cache is gitignored (large, regenerable).
# =============================================================================

import argparse
import json
import os
import sys

sys.path.insert(0, ".")
from src.train import _open_source  # reuse the source opener


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--docs-per-source", type=int, default=2000)
    ap.add_argument("--out", default="data_cache/mix.jsonl")
    args = ap.parse_args()

    # read sources from base.yaml via a tiny inline parse (avoid yaml dep)
    import re
    srcs = []
    capture = False
    with open("configs/base.yaml") as f:
        for line in f:
            if line.strip().startswith("sources:"):
                capture = True
                continue
            if capture:
                if line.strip().startswith("- ") and "sources" not in line:
                    # strip list dash, quotes, and any trailing # comment
                    s = line.strip()[2:].strip().strip('"').strip("'")
                    s = s.split("#")[0].strip().strip('"').strip("'")
                    if s:
                        srcs.append(s)
                elif line.strip() and not line.startswith(" "):
                    break
    if not srcs:
        srcs = ["wikitext", "open-web-math/open-web-math",
                "HuggingFaceTB/smollm-corpus:fineweb-edu-dedup"]
    print(f"[cache] sources: {srcs}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    written = 0
    with open(args.out, "w") as out:
        for src in srcs:
            try:
                ds, field = _open_source(src)
            except Exception as e:
                print(f"[cache] SKIP {src}: {repr(e)[:100]}")
                continue
            n = 0
            for row in ds:
                text = row.get(field) or ""
                if len(text) < 64:
                    continue
                out.write(json.dumps({"text": text}) + "\n")
                written += 1
                n += 1
                if n >= args.docs_per_source:
                    break
            print(f"[cache] {src}: wrote {n} docs")
    print(f"[cache] DONE -> {args.out} ({written} docs total)")


if __name__ == "__main__":
    main()
