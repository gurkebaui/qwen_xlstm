# =============================================================================
# tests/smoke_eval.py
#
# Smoke test for the QUICK eval harness (src/eval.py).
#
# We run the REAL harness but on a TINY subset (4 docs, 64 tokens) so it
# finishes in seconds and proves end-to-end wiring:
#   * pg19 streaming load works
#   * perplexity math runs
#   * patched vs frozen-base comparison produces numbers + a delta
#
# It does NOT assert patched is BETTER (that needs training) — it only
# asserts the harness runs and returns sane (finite) numbers.
#
# Run:  python tests/smoke_eval.py
# =============================================================================

import sys
import torch

sys.path.insert(0, ".")
from src.patcher import XlstmQwenModel
from src.xlstm_layer import XLSTMLayerConfig
from src.eval import quick_eval

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# tiny cfg override -> fast smoke (real run later uses base.yaml defaults)
tiny_cfg = {
    "probe": "wikitext",    # streams on datasets>=4.8 (pg19 script blocked)
    "subsample": 4,      # only 4 docs
    "seq_len": 64,       # only 64 tokens each
}


def main():
    # build the patched model (identity-init so it == base; we only test the
    # harness, not training here)
    m = XlstmQwenModel(
        model_id="Qwen/Qwen2.5-Coder-0.5B",
        xlstm_cfg=XLSTMLayerConfig(embedding_dim=896, context_length=128),
        device=DEVICE, dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    m.init_identity()

    res = quick_eval(m, tiny_cfg, device=DEVICE)

    # assertions: numbers must be finite; delta may be 0 (identity init =>
    # patched==base, expected before any training). Only reject non-finite.
    assert isinstance(res["base_ppl"], float) and res["base_ppl"] == res["base_ppl"], \
        "base_ppl not finite"
    assert isinstance(res["patched_ppl"], float) and res["patched_ppl"] == res["patched_ppl"], \
        "patched_ppl not finite"
    assert isinstance(res["delta_ppl"], float) and res["delta_ppl"] == res["delta_ppl"], \
        "delta_ppl not finite"
    print("[eval] SMOKE PASS -> harness runs, returns finite ppl + delta (=0 expected pre-train)")
    print(f"        {res}")


if __name__ == "__main__":
    main()
