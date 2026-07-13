"""PROFILE one real training step to find WHERE the ~30s goes.
Times: data load, forward, backward, clip_grad_norm, opt.step, zero_grad.
Also checks the dtype-mismatch RMSNorm warning (float weights vs bf16 input
=> slow non-fused path) and per-block forward cost.
"""
import sys, time, torch
sys.path.insert(0, ".")
import yaml
from transformers import AutoTokenizer
from src.patcher import XlstmQwenModel
from src.xlstm_layer import XLSTMLayerConfig
from src.train import build_optimizer, stream_tokens, load_config

DEVICE = "cuda"; DTYPE = torch.bfloat16
cfg = load_config("configs/base.yaml")
xl = XLSTMLayerConfig(block_type="slstm", embedding_dim=896, num_heads=4,
                      architecture="1:0", context_length=2048, conv1d_kernel=4,
                      bias_init="powerlaw_blockdependent")
m = XlstmQwenModel("Qwen/Qwen2.5-Coder-0.5B", xl, device=DEVICE, dtype=DTYPE).to(DEVICE).train()
m.init_identity()
opt, sched = build_optimizer(m, 1e-4, 50, 800)
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-0.5B")

# --- dtype check: are any norm/weight tensors float (slow path)? ---
float_params = [n for n, p in m.named_parameters() if p.dtype == torch.float32]
print(f"[dtype] {len(float_params)} params are float32 (bf16 model). "
      f"examples: {float_params[:4]}")

data = stream_tokens(cfg, max_docs=None)
ids = next(data).unsqueeze(0).to(DEVICE)

# --- step 1: time the big pieces ---
torch.cuda.synchronize(); t = time.time()
# data load of the NEXT chunk (CPU tokenize)
ids2 = next(data).unsqueeze(0).to(DEVICE); t_data = time.time() - t

torch.cuda.synchronize(); t = time.time()
out = m(input_ids=ids, labels=ids); t_fwd = time.time() - t
torch.cuda.synchronize(); t = time.time()
(out.loss).backward(); t_bwd = time.time() - t
torch.cuda.synchronize(); t = time.time()
torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0); t_clip = time.time() - t
torch.cuda.synchronize(); t = time.time()
opt.step(); sched.step(); opt.zero_grad(); t_opt = time.time() - t

print(f"[step] data={t_data*1000:.1f}ms  fwd={t_fwd:.3f}s  bwd={t_bwd:.3f}s  "
      f"clip={t_clip*1000:.1f}ms  opt={t_opt*1000:.1f}ms  TOTAL={t_data+t_fwd+t_bwd+t_clip+t_opt:.2f}s")
print("DONE")
