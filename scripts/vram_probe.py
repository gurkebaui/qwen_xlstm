# =============================================================================
# scripts/vram_probe.py
#
# Before committing to a 2000-step CPT run, KNOW the VRAM budget. A run
# that OOMs at step 3 wastes hours. This probes peak VRAM for a few
# (seq_len, batch, grad_accum) combos at identity-init (worst-case-ish:
# all 24 xlstm layers + base + grads + optimizer state live on GPU).
#
# Prints peak MB for each combo + a GO/NO-GO for the 16GB card.
# Run: python scripts/vram_probe.py
# =============================================================================

import sys
import torch

sys.path.insert(0, ".")
from src.patcher import XlstmQwenModel
from src.xlstm_layer import XLSTMLayerConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
assert DEVICE == "cuda", "this probe is only meaningful on GPU"


def probe(seq_len: int, batch: int, accum: int, ctx: int = 2048):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = XlstmQwenModel(
        model_id="Qwen/Qwen2.5-Coder-0.5B",
        xlstm_cfg=XLSTMLayerConfig(embedding_dim=896, context_length=ctx),
        device=DEVICE, dtype=torch.bfloat16,
    ).to(DEVICE).train()
    m.init_identity()

    # synthetic tokens (no dataset needed for a VRAM probe)
    ids = torch.randint(0, 1000, (batch, seq_len), device=DEVICE)

    # simulate one optimizer step (AdamW keeps 2 states per trainable param)
    opt = torch.optim.AdamW(
        [p for p in m.parameters() if p.requires_grad], lr=5e-4
    )
    opt.zero_grad()
    # grad accum micro-steps
    for _ in range(accum):
        out = m(input_ids=ids, labels=ids)
        out.loss.backward()
    opt.step()

    peak = torch.cuda.max_memory_allocated() / 1e6  # MB
    del m, opt, out, ids
    torch.cuda.empty_cache()
    return peak


if __name__ == "__main__":
    print(f"{'seq_len':>8} {'batch':>6} {'accum':>6} {'peak_MB':>9}  {'@16GB':>7}")
    print("-" * 44)
    plan = [
        (512, 1, 1),
        (1024, 1, 4),
        (2048, 1, 8),     # matches base.yaml train defaults
        (2048, 1, 4),
        (4096, 1, 8),
    ]
    LIMIT = 16 * 1024  # 16 GB in MB
    for sl, b, a in plan:
        mb = probe(sl, b, a)
        go = "GO" if mb < LIMIT * 0.9 else ("TIGHT" if mb < LIMIT else "NO-GO")
        print(f"{sl:>8} {b:>6} {a:>6} {mb:>9.0f}  {go:>7}")
    print("\n(peak excludes grad-accum's held activations scaling; if TIGHT, "
          "raise accum or lower seq_len before a real run.)")
