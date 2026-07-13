"""Fully-OFFLINE recurrent long-context probe (no HF dataset download).

Reads local data_cache/mix.jsonl (the training text mix) instead of
emozilla/pg19 / wikitext (both 403 on this box -- token rejected).
Rolls the sLSTM graft RECURRENTly to increasing prefix lengths vs the
frozen base; lower patch_ppl = the graft carries long-range memory.

Run:
    python scripts/probe_offline.py --ckpt checkpoints/xlstm_cpt_step1500.pt
"""
import argparse, json, sys, os, time
import torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.patcher import XlstmQwenModel
from src.xlstm_layer import XLSTMLayerConfig
from transformers import Qwen2ForCausalLM, AutoTokenizer

DEVICE = "cuda"
DTYPE = torch.bfloat16
MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B"
MIX = "data_cache/mix.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--lengths", nargs="+", type=int,
                    default=[2048, 4096, 8192, 16384])
    ap.add_argument("--pred-len", type=int, default=256)
    ap.add_argument("--n-books", type=int, default=4)
    ap.add_argument("--max-scanned", type=int, default=12000)
    args = ap.parse_args()

    base = Qwen2ForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE
                                            ).to(DEVICE).requires_grad_(False)
    xl = XLSTMLayerConfig(
        block_type="slstm", embedding_dim=896, num_heads=4,
        architecture="1:0", context_length=2048, conv1d_kernel=4,
        bias_init="powerlaw_blockdependent",
    )
    model = XlstmQwenModel(MODEL_ID, xl, device=DEVICE, dtype=DTYPE).to(DEVICE)
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[probe-offline] loaded ckpt step {ck.get('step')} from {args.ckpt}")

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    ce = torch.nn.functional.cross_entropy

    # --- collect long-enough local docs ---
    max_L = max(args.lengths)
    need_chars = (max_L + args.pred_len) * 2
    docs = []
    scanned = 0
    with open(MIX) as f:
        for line in f:
            scanned += 1
            if len(docs) >= args.n_books:
                break
            if scanned > args.max_scanned:
                break
            try:
                d = json.loads(line)
            except Exception:
                continue
            t = d.get("text") or ""
            if len(t) >= need_chars:
                docs.append(t)
    print(f"[probe-offline] found {len(docs)} docs >= {need_chars} chars "
          f"(scanned {scanned})")
    if not docs:
        # fall back: take the longest available docs even if shorter
        lens = []
        with open(MIX) as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                lens.append(d.get("text") or "")
        lens.sort(key=len, reverse=True)
        docs = lens[:args.n_books]
        print(f"[probe-offline] FALLBACK: using {len(docs)} longest docs "
              f"(max {len(docs[0]) if docs else 0} chars)")

    out = {}
    for L in args.lengths:
        pl_base, pl_patch, tk, ok = 0.0, 0.0, 0, 0
        for text in docs:
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=L + args.pred_len).input_ids[0].to(DEVICE)
            if ids.numel() < L + args.pred_len:
                continue
            pre, post = ids[:L], ids[L:L + args.pred_len]
            full = torch.cat([pre, post])
            # patched recurrent roll
            model.reset_generation()
            ls = {i: None for i in range(len(model.xlstm_layers))}
            pkv = None
            for t in range(full.numel()):
                lg, ls, pkv = model.generate_step(
                    full[t:t + 1].unsqueeze(0),
                    layer_states=ls, past_key_values=pkv)
                if t >= L and t + 1 < full.numel():
                    pl_patch += ce(lg[0, -1:].float(),
                                   full[t + 1:t + 2]).item()
                    tk += 1
            model.reset_generation()
            # base recurrent roll (KV cache)
            bkv = None
            for t in range(full.numel()):
                ob = base(input_ids=full[t:t + 1].unsqueeze(0),
                          use_cache=True, past_key_values=bkv)
                bkv = ob.past_key_values
                if t >= L and t + 1 < full.numel():
                    pl_base += ce(ob.logits[0, -1:].float(),
                                  full[t + 1:t + 2]).item()
            del bkv
            ok += 1
        if tk == 0:
            print(f"[probe-offline] L={L}: no usable docs, skip")
            continue
        rb = float(torch.exp(torch.tensor(pl_base / tk)).item())
        rp = float(torch.exp(torch.tensor(pl_patch / tk)).item())
        out[L] = {"base_ppl": rb, "patch_ppl": rp, "delta": rp - rb}
        print(f"[probe-offline] L={L:6d} books={ok}: base_ppl={rb:.2f} "
              f"patch_ppl={rp:.2f} delta={rp - rb:+.2f} "
              f"({'patch BETTER' if rp < rb else 'patch WORSE'})",
              flush=True)
    print("\n=== OFFLINE RECURRENT LONG-CONTEXT PROBE (sLSTM step-%s) ==="
          % ck.get("step"))
    for L, r in out.items():
        print(f"  L={L:>6}: base={r['base_ppl']:.2f}  patch={r['patch_ppl']:.2f}  "
              f"delta={r['delta']:+.2f}  ({'patch BETTER' if r['delta'] < 0 else 'patch WORSE'})")
    print("=== done ===")


if __name__ == "__main__":
    main()
