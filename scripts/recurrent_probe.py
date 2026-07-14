# =============================================================================
# scripts/recurrent_probe.py
#
# Headline test for the xLSTM graft: train at 2048, RECURRENTLY decode at
# 16384 with CONSTANT VRAM, and check whether the graft's long-range memory
# beats the frozen base at next-token prediction.
#
# Uses src/eval.recurrent_long_context_eval, which rolls the prefix
# token-by-token via generate_step (recurrent .step() + carries BOTH the KV
# cache AND the xlstm state) then measures ppl on the following pred_len
# tokens. The frozen base consumes the SAME prefix in parallel.
#
# Usage:
#   python3 -u scripts/recurrent_probe.py \
#       --ckpt checkpoints/xlstm_cpt_step10000.pt \
#       --lengths 2048 4096 8192 16384 --pred-len 256 --n-books 3
# =============================================================================
import argparse
import sys

import torch

sys.path.insert(0, ".")
from src.patcher import XlstmQwenModel
from src.xlstm_layer import XLSTMLayerConfig
from src.eval import recurrent_long_context_eval
from transformers import Qwen2ForCausalLM

DEVICE = "cuda"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/xlstm_cpt_step10000.pt")
    ap.add_argument("--lengths", nargs="+", type=int,
                    default=[2048, 4096, 8192, 16384])
    ap.add_argument("--pred-len", type=int, default=256)
    ap.add_argument("--n-books", type=int, default=3)
    ap.add_argument("--probe", default="emozilla/pg19")
    ap.add_argument("--max-scanned", type=int, default=300)
    ap.add_argument("--ctx", type=int, default=2048,
                    help="xlstm context_length to BUILD the eval model at")
    args = ap.parse_args()

    torch.cuda.empty_cache()
    print(f">>> building patched model (xlstm context_length={args.ctx}) ...")
    m = XlstmQwenModel(
        "Qwen/Qwen2.5-Coder-0.5B",
        XLSTMLayerConfig(embedding_dim=896, context_length=args.ctx),
        device=DEVICE, dtype=torch.bfloat16,
    ).to(DEVICE).eval()
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    m.load_state_dict(ck["model"])
    print(f">>> loaded ckpt {args.ckpt} (step={ck.get('step', '?')})")

    print(">>> building frozen base (reference) ...")
    base = Qwen2ForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-0.5B", torch_dtype=torch.bfloat16,
    ).to(DEVICE).requires_grad_(False)

    print(f">>> recurrent long-context eval: L={args.lengths}, "
          f"pred_len={args.pred_len}, n_books={args.n_books}")
    res = recurrent_long_context_eval(
        m, base, DEVICE,
        lengths=tuple(args.lengths),
        pred_len=args.pred_len,
        probe=args.probe,
        n_books=args.n_books,
        max_scanned=args.max_scanned,
    )
    print("\n=== RECURRENT LONG-CONTEXT RESULTS ===")
    for L, r in res.items():
        print(f"  L={L:6d}: base={r['base_ppl']:.2f}  "
              f"patch={r['patch_ppl']:.2f}  delta={r['delta']:+.2f}  "
              f"({'patch BETTER' if r['delta'] < 0 else 'patch WORSE'})")
    # json-friendly dump
    import json
    with open("runs/recurrent_probe_results.json", "w") as f:
        json.dump(res, f, indent=2)
    print(">>> saved runs/recurrent_probe_results.json")


if __name__ == "__main__":
    main()
