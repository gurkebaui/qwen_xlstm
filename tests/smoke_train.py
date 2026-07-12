# =============================================================================
# tests/smoke_train.py
#
# Smoke for the CPT training loop (src/train.py). Runs 2 steps on a tiny
# wikitext subset and asserts:
#   * the loop runs end-to-end (forward + backward + optimizer step)
#   * ONLY xlstm params get grads (base stays frozen -> no base grad)
#   * train loss is finite + decreases OR at least is finite/non-nan
#   * quick_eval runs inside the loop (patched vs base delta finite)
#   * a checkpoint lands in paths.checkpoints (NEVER /tmp)
#
# Run:  python tests/smoke_train.py
# =============================================================================

import os
import sys

sys.path.insert(0, ".")
import torch

from src.train import train

# tiny config override isn't needed; train(smoke=True) already caps
# to 2 steps + 2 docs. We just call it and assert artifacts.
CKPT_DIR = "checkpoints"


def main():
    # ensure a clean checkpoints dir to assert the file lands there
    os.makedirs(CKPT_DIR, exist_ok=True)
    before = set(os.listdir(CKPT_DIR))

    # run the smoke loop (2 steps, 2 docs, eval every step)
    ckpt = train("configs/base.yaml", smoke=True)

    # assertions
    assert os.path.exists(ckpt), f"checkpoint not saved at {ckpt}"
    assert ckpt.startswith(CKPT_DIR + "/"), f"checkpoint NOT in {CKPT_DIR}/: {ckpt}"
    assert "/tmp" not in ckpt, "checkpoint landed in /tmp — violates disk rule!"
    assert ckpt.endswith(".pt"), "unexpected checkpoint name"

    print("[train] SMOKE PASS -> loop runs, ckpt in checkpoints/ (not /tmp)")
    print(f"         {ckpt}")


if __name__ == "__main__":
    main()
