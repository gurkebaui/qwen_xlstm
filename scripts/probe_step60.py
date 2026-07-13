"""Early recurrent long-context probe for the sLSTM graft (step-60 ckpt).

Loads the frozen base + the step-60 sLSTM checkpoint, then measures
next-token perplexity at increasing prefix lengths using RECURRENT decode
(constant VRAM), vs the frozen base. This is the paper's real test:
does the sLSTM graft beat base at long context (where the prior mLSTM
graft collapsed past 4096)?

Run:
    python src/probe_step60.py --ckpt checkpoints/xlstm_cpt_step60.pt
"""
import argparse
import sys
import torch

sys.path.insert(0, ".")
from src.patcher import XlstmQwenModel
from src.xlstm_layer import XLSTMLayerConfig
from src.eval import recurrent_long_context_eval
from transformers import Qwen2ForCausalLM


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True,
                    help="path to a saved xlstm checkpoint (.pt)")
    ap.add_argument("--lengths", nargs="+", type=int,
                    default=[2048, 4096, 8192, 16384])
    ap.add_argument("--pred-len", type=int, default=256)
    ap.add_argument("--n-books", type=int, default=3)
    args = ap.parse_args()

    DEVICE = "cuda"
    DTYPE = torch.bfloat16
    MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B"

    # --- frozen base (reference) ---
    base = Qwen2ForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE
                                            ).to(DEVICE).requires_grad_(False)

    # --- patched model with the sLSTM graft, load step-60 weights ---
    xl = XLSTMLayerConfig(
        block_type="slstm", embedding_dim=896, num_heads=4,
        architecture="1:0", context_length=2048, conv1d_kernel=4,
        bias_init="powerlaw_blockdependent",
    )
    model = XlstmQwenModel(MODEL_ID, xl, device=DEVICE, dtype=DTYPE).to(DEVICE)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[probe] loaded checkpoint step {ck.get('step')} from {args.ckpt}")

    # --- run the recurrent long-context probe ---
    res = recurrent_long_context_eval(
        model, base, device=DEVICE,
        lengths=tuple(args.lengths), pred_len=args.pred_len,
        n_books=args.n_books,
    )
    print("\n=== RECURRENT LONG-CONTEXT PROBE (sLSTM step-%s) ===" % ck.get("step"))
    for L, r in res.items():
        print(f"  L={L:>6}: base={r['base_ppl']:.2f}  patch={r['patch_ppl']:.2f}  "
              f"delta={r['delta']:+.2f}  ({'patch BETTER' if r['delta'] < 0 else 'patch WORSE'})")
    print("=== done ===")


if __name__ == "__main__":
    main()
