"""
Time ONE optimizer step at a given INPUT seq_len (warm-stepped 3x, then averaged).
Pure PyTorch (vanilla sLSTM backend) -- no nvcc needed.
Proves how much of the ~15s/step is the frozen base's O(L^2) attention
vs the sLSTM cell. If 256 is ~4-8x faster than 1024, trimming the base
context is the real speed lever.
"""
import time, torch, yaml, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.patcher import XlstmQwenModel
from src.xlstm_layer import XLSTMLayerConfig

def build():
    with open("configs/base.yaml") as f:
        cfg = yaml.safe_load(f)
    _dim_keys = {"embedding_dim", "num_heads", "context_length",
                 "proj_factor", "conv1d_kernel", "bias", "dropout"}
    xlstm_cfg = XLSTMLayerConfig(
        **{k: v for k, v in cfg["xlstm"].items() if k in _dim_keys})
    torch.manual_seed(0)
    model = XlstmQwenModel(model_id=cfg["model"]["name"],
                           xlstm_cfg=xlstm_cfg, device="cuda",
                           dtype=torch.bfloat16).to("cuda")
    if cfg["train"].get("identity_init", True):
        model.init_identity()
    model.train()
    return model

def one_step(model, seq_len):
    x = torch.randint(0, 1000, (1, seq_len), device="cuda")
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
    t_zero = time.perf_counter() - t0
    return t_fwd, t_bwd, t_zero

model = build()
for seq_len in (256, 512, 1024):
    for _ in range(3):                 # warm
        one_step(model, seq_len)
    torch.cuda.synchronize()
    fws, bws = [], []
    for _ in range(3):
        f, b, _ = one_step(model, seq_len)
        fws.append(f); bws.append(b)
    print(f"[seq_len={seq_len}] fwd={sum(fws)/3:.3f}s bwd={sum(bws)/3:.3f}s "
          f"TOTAL={(sum(fws)+sum(bws))/3:.3f}s", flush=True)
del model
print("DONE")
