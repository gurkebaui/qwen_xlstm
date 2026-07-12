# =============================================================================
# qwen_xlstm — project layout
#
# Goal: graft a TRAINABLE xLSTM (mLSTM) memory sublayer BETWEEN the attention
# and FFN sublayers of Qwen2.5-Coder-0.5B. The base transformer is frozen;
# only the inserted mLSTM blocks are trained (Stage 1 = continued pretraining).
#
# Everything mLSTM-related comes from the OFFICIAL `xlstm` PyPI package
# (v2.0.5) + `mlstm_kernels` (v2.0.2). We do NOT reimplement mLSTM —
# we only wrap the real `mLSTMLayer` and plug it into the Qwen architecture.
#
# Directory structure (kept tidy so the code is readable):
#   configs/        single-source-of-truth YAML (single-field switches, no CLI soup)
#     base.yaml        master config (model / xlstm / data / train / paths)
#   src/
#     xlstm_layer.py  thin wrapper around the real xlstm.mLSTMLayer
#                       -> adds identity-safe init + unified forward/step API
#     patcher.py      inserts mLSTMBlock between attn & FFN in each Qwen layer
#                       -> freezes the base, trains only the mLSTM, residual per sublayer
#     model.py        (later) the assembled model + generation w/ state carry
#     train.py         (later) continued-pretraining loop
#   tests/
#     smoke_mlstm.py  out==base at init + recurrent state carries across tokens
#   plan.md           the living plan (you + me edit it)
#   main.py           entrypoint stub (filled in once layers/patcher are proven)
# =============================================================================
