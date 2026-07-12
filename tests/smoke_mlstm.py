# =============================================================================
# tests/smoke_mlstm.py
#
# The make-or-break smoke test for the graft. TWO assertions must hold
# before ANY training run:
#
#   (1) IDENTITY: after init_identity(), the patched model's output must
#       NUMERICALLY EQUAL the frozen base model's output on the same input.
#       Proof we did NOT break Qwen by inserting the mLSTM branch (because
#       the branch is zero-init'd and added residually).
#
#   (2) STATE CARRY: running in recurrent .step() mode and threading the
#       returned state back must give the SAME logits as a single parallel
#       .forward() over the whole sequence. Proof the mLSTM remembers
#       across tokens (the whole point of the memory sublayer).
#
# Run:  python tests/smoke_mlstm.py
# Exit non-zero on either failure so it can gate CI / training later.
# =============================================================================

import sys
import torch

from transformers import Qwen2ForCausalLM

sys.path.insert(0, ".")
from src.patcher import XlstmQwenModel
from src.xlstm_layer import XLSTMLayerConfig

MODEL_ID = "Qwen/Qwen2.5-Coder-0.5B"
DTYPE = torch.bfloat16
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

torch.manual_seed(0)
tok_input = torch.randint(0, 1000, (1, 16), device=DEVICE)  # (B, S)


def load_base():
    m = Qwen2ForCausalLM.from_pretrained(MODEL_ID, torch_dtype=DTYPE).to(DEVICE)
    m.requires_grad_(False)
    m.eval()
    return m


def test_identity():
    """Assertion (1): patched == base at identity init."""
    base = load_base()
    with torch.no_grad():
        base_out = base(input_ids=tok_input).logits

    patched = XlstmQwenModel(
        model_id=MODEL_ID,
        xlstm_cfg=XLSTMLayerConfig(embedding_dim=896, context_length=64),
        device=DEVICE, dtype=DTYPE,
    ).to(DEVICE).eval()
    patched.init_identity()  # <-- the critical zero-init

    with torch.no_grad():
        patched_out = patched(input_ids=tok_input).logits

    # only the first token's logits are comparable (attention is causal anyway,
    # but diff over the full (B,S,V) tensor is the strict test)
    diff = (base_out - patched_out).abs().max().item()
    print(f"[identity] max |base - patched| = {diff:.3e}")
    assert diff < 1e-2, f"IDENTITY FAILED: diff {diff} too large"
    print("[identity] PASS  -> insertion does not break the base at step 0")
    return patched


def test_state_carry(patched):
    """Asserton (2): the mLSTM sublayer remembers across tokens.

    Self-contained proof: feed a synthetic (B,S,D) tensor to the xlstm
    sublayer in PARALLEL mode (.forward) vs RECURRENT mode (.step, threading
    state back token-by-token). If the carried state is correct, both must
    produce the SAME output. No HF-attention involved -> no version fragility.
    """
    patched.eval()
    xl = patched.xlstm_layers[0].xlstm
    B, S, D = 1, 16, xl.config.embedding_dim

    x = torch.randn(B, S, D, device=DEVICE, dtype=DTYPE)
    with torch.no_grad():
        ref = xl(x)                              # parallel
        states = None
        rec = None
        for t in range(S):
            y, states = xl.step(x[:, t:t + 1], state=states)
            rec = y if rec is None else torch.cat([rec, y], dim=1)

    diff = (ref - rec).abs().max().item()
    print(f"[state]    max |parallel - recurrent| = {diff:.3e}")
    assert diff < 1e-2, f"STATE CARRY FAILED: diff {diff} too large"
    print("[state]    PASS  -> mLSTM memory carries across tokens")


if __name__ == "__main__":
    print(f"device={DEVICE}  model={MODEL_ID}")
    patched = test_identity()
    test_state_carry(patched)
    print("\nALL SMOKE CHECKS PASSED.")
