"""Evidence-gathering: does xlstm sLSTM backend='cuda' compile+run+grad here,
and how fast vs the vanilla (current) backend? RTX 4060 Ti = sm_89.
torch cu130, nvcc 12.1 (mismatch risk -> cuda backend may fail to JIT).
"""
import sys, time, torch
sys.path.insert(0, ".")
from xlstm import sLSTMBlockConfig, sLSTMLayerConfig, sLSTMBlock

DEVICE = "cuda"; DTYPE = torch.bfloat16
EDIM, NHEAD, B, T = 896, 4, 1, 256

def make(backend):
    cfg = sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            embedding_dim=EDIM, num_heads=NHEAD, conv1d_kernel_size=4,
            bias_init="powerlaw_blockdependent", backend=backend,
        ),
        _num_blocks=1, _block_idx=0,
    )
    return sLSTMBlock(cfg).to(DEVICE).to(DTYPE)

for backend in ["vanilla", "cuda"]:
    try:
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        t0 = time.time(); m = make(backend); torch.cuda.synchronize()
        t_load = time.time() - t0
        x = torch.randn(B, T, EDIM, device=DEVICE, dtype=DTYPE, requires_grad=True)
        y = m(x); (y.float().sum()).backward(); torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(3):
            xx = torch.randn(B, T, EDIM, device=DEVICE, dtype=DTYPE)
            yy = m(xx); yy.float().sum().backward(); torch.cuda.synchronize()
        dt = (time.time() - t0) / 3
        ok = any(p.grad is not None for p in m.parameters())
        print(f"[OK] backend={backend:8s} load={t_load:.1f}s  fwd+bwd/step={dt:.4f}s  "
              f"peak={torch.cuda.max_memory_allocated()/1e9:.2f}GB  grad={ok}", flush=True)
    except Exception as e:
        print(f"[FAIL] backend={backend:8s}: {type(e).__name__}: {str(e)[:500]}", flush=True)
print("DONE")
