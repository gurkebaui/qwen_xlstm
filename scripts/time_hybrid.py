"""
Time ONE optimizer step at L=1024 for two configs:
  (A) FULL:    base_ctx=None  (base attn sees full 1024)  -- current behaviour
  (B) HYBRID:  base_ctx=128   (base attn sees local 128 windows; sLSTM full 1024)
Confirms the hybrid is faster AND runs without error (finite loss).
Pure PyTorch (vanilla sLSTM backend) -- no nvcc needed.
"""
import time, torch, yaml, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.patcher import XlstmQwenModel
from src.xlstm_layer import XLSTMLayerConfig

def build(base_ctx):
    with open("configs/base.yaml") as f:
        cfg = yaml.safe_load(f)
    _dim_keys = {"embedding_dim", "num_heads", "context_length",
                 "proj_factor", "conv1d_kernel", "bias", "dropout", "base_ctx"}
    xcfg = XLSTMLayerConfig(**{k: v for k, v in cfg["xlstm"].items() if k in _dim_keys})
    xcfg.base_ctx = base_ctx
    torch.manual_seed(0)
    model = XlstmQwenModel(model_id=cfg["model"]["name"],
                           xlstm_cfg=xcfg, device="cuda",
                           dtype=torch.bfloat16).to("cuda")
    if cfg["train"].get("identity_init", True):
        model.init_identity()
    model.train()
    return model

def one_step(model, L):
    x = torch.randint(0, 1000, (1, L), device="cuda")
    t0 = time.perf_counter()
    out = model(input_ids=x, labels=x)
    loss = out.loss
    t_fwd = time.perf_counter() - t0
    t0 = time.perf_counter()
    loss.backward()
    t_bwd = time.perf_counter() - t0
    t0 = time.perf_counter()
    for p in model.parameters():
        if p.grad is not None: p.grad = None
    return t_fwd, t_bwd, float(loss.item())

for label, bc in (("FULL(base_ctx=None)", None), ("HYBRID(base_ctx=128)", 128)):
    model = build(bc)
    # warm
    for _ in range(2):
        one_step(model, 1024)
    torch.cuda.synchronize()
    fws, bws, ls = [], [], []
    for _ in range(3):
        f, b, l = one_step(model, 1024)
        fws.append(f); bws.append(b); ls.append(l)
    print(f"[{label}] fwd={sum(fws)/3:.3f}s bwd={sum(bws)/3:.3f}s "
          f"TOTAL={(sum(fws)+sum(bws))/3:.3f}s  loss={sum(ls)/3:.4f}", flush=True)
    del model; torch.cuda.empty_cache()
print("DONE")
