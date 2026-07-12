# Fine-Tuning Plan: making the grafted xLSTM-Qwen good at
# long-context coding, reasoning, and agentic tool use
#
# Status: DRAFT plan for Stage 2-3 (Stage 1 = CPT, in progress).
# Author: Hermes (night session, 2026-07-13). Henry to review.
#
# TOC
#   1. Goal & the one hard constraint (our model != PreTrainedModel)
#   2. Stage 1 recap (CPT, done/running) — why we need SFT after
#   3. Stage 2: Supervised Fine-Tuning (SFT)
#   4. Stage 3: Reinforcement Learning (GRPO)
#   5. The long-context-via-mLSTM training trick (the actual point)
#   6. Library decision: TRL vs. do-it-ourselves (with reasons)
#   7. VRAM budget at Stage 2-3 (unfrozen base)
#   8. Open risks / things still broken
#   9. Concrete next actions (ordered)
#
# =============================================================================
# 1. GOAL & THE ONE HARD CONSTRAINT
# =============================================================================
# Goal: after CPT inserts the mLSTM, turn the model into one that is
#   (a) good at CODING (long files, repos),
#   (b) good at REASONING (math, multi-step),
#   (c) good at AGENTIC TOOL USE (function/API calling, multi-turn).
#
# The thing that makes THIS model interesting: the mLSTM sublayer should
# let it handle CONTEXT BEYOND Qwen's trained window cheaply
# (constant-state recurrence vs. attention's O(S^2)). The CPT run did NOT
# teach the model to USE that — it only taught the mLSTM to exist without
# hurting ppl (delta ~+0.1, stable, see eval notes). SFT/RL is where
# the model learns to actually lean on the mLSTM for long context.
#
# HARD CONSTRAINT (verified this session):
#   Our model is `XlstmQwenModel(nn.Module)` — NOT a
#   `transformers.PreTrainedModel`. TRL's GRPOTrainer builds a
#   `ref_model` via `AutoModelForCausalLM.from_pretrained(...)` and
#   calls `.generate()` / `.save_pretrained()`. It CANNOT load our
#   custom arch as-is. SFTTrainer is fine (it accepts `nn.Module`),
#   but GRPO is the problem child.
#
# =============================================================================
# 2. STAGE 1 RECAP (CPT) — WHY WE NEED SFT AFTER
# =============================================================================
# - Inserted one trainable mLSTM sublayer per Qwen layer, base frozen.
# - 2000-step run: stable, ppl delta +0.35 (slightly worse, NOT
#   catastrophic — your main worry is dead).
# - Overnight run: 10000 steps now training (PID 278577, log
#   runs/cpt_run.log). Eval so far (saved step2000 ckpt, variable
#   context probe, L in {1024,2048,4096}):
#       L=1024: base 4.60  patch 4.65  delta +0.05
#       L=2048: base 10.78 patch 10.92 delta +0.15
#       L=4096: base 11.50 patch 11.67 delta +0.17
#   => graft is STABLE at 2x trained context (no explosion) but not
#   yet HELPFUL. Consistent with Henry's prediction: "a slight decrease
#   is okay until we fine-tune it to use the mLSTM."
#
# =============================================================================
# 3. STAGE 2: SUPERVISED FINE-TUNING (SFT)
# =============================================================================
# What changes vs Stage 1: UNFREEZE EVERYTHING.
#   - Stage 1: base frozen, only mLSTM trainable (19% params).
#   - Stage 2: base + mLSTM + LM head all trainable. Now the base
#     can ADAPT to route long-range signal through the mLSTM.
#
# Datasets (mix, all streamable / cached):
#   - CODE: bigcode/starcoderdata (python) + a code-instruction set
#     (e.g. m-a-p/CodeFeedback, or therse/code_instructions).
#   - REASONING: open-web-math + a math-instruction set
#     (MetaMath / GSM8K as SFT pairs).
#   - AGENTIC TOOL USE:
#       * OpenBMB/ToolBench (instruction-tuning data for function
#         calling; the dataset + scripts exist, Apache-2.0).
#       * AgentInstruct (25M synthetic pairs incl. tool-use; can be
#         used for instruction tuning of any base).
#       * Optionally: a small synthetic "tool-call trajectory" set we
#         generate ourselves (cheap, on-distribution for our use case).
#   - LONG-CONTEXT (the point): long-doc QA from PG19 / long
#     GitHub repos, FORMATTED so the answer requires attending to
#     tokens > 2048 from the start of the context. This is what
#     forces the model to USE the mLSTM (see section 5).
#
# Library: SFTTrainer from TRL (it accepts nn.Module, verified).
#   We pass our `XlstmQwenModel` directly. If save/load quirks
#   appear, fall back to HF Trainer with our `train.py` loop
#   extended for unfrozen params + an SFT data collator.
#
# =============================================================================
# 4. STAGE 3: REINFORCEMENT LEARNING (GRPO)
# =============================================================================
# Why RL (not just SFT): tool use & reasoning are best sharpened by
#   reward on OUTCOME (did the tool call execute? did the test pass?),
#   which SFT (next-token CE) can't optimize directly.
#
# Algorithm: GRPO (Group Relative Policy Optimization, DeepSeekMath) —
#   the same family as the o1/o3-style post-training. No separate
#   value model needed (computes advantage from group scores).
#
# Reward signals (per trajectory):
#   - TOOL USE: did the generated call match a valid schema and
#     execute? (binary / schema-distance)
#   - CODE: did the produced function pass the held-out unit tests?
#     (execution-based reward, like RLCF / CodeR)
#   - LONG-CONTEXT RETENTION: on long-doc tasks, did it use the
#     early context correctly? (answer-accuracy on >2048 context)
#   - light KL penalty vs. the SFT model (keep it from drifting).
#
# Library constraint (CRITICAL): GRPOTrainer needs a
#   PreTrainedModel (builds ref_model via from_pretrained +
#   .generate()). Our nn.Module is NOT. Two options:
#   (A) WRAP: make `XlstmQwenModel` subclass
#       `transformers.PreTrainedModel`, implement `forward` (we
#       already have it) + `generate` (our recurrent decode) +
#       `save_pretrained`/`from_pretrained` (save backbone +
#       mLSTM state dicts). Then TRL GRPOTrainer works natively.
#       Pro: unlocks TRL's distributed GRPO, vLLM/FSDP, reward
#       hooks, for free. Con: wrapping a custom arch in
#       PreTrainedModel is fiddly (config class, weight loading).
#   (B) DO IT OURSELVES: a small GRPO loop reusing our
#       `train.py` scaffolding — sample G completions per prompt
#       (our `generate`), score with our reward fns, compute
#       group-relative advantage, PPO/GRPO policy update on the
#       mLSTM+base. Pro: full control, no TRL constraint, we
#       already have the decode + train loop. Con: we reimplement
#       GRPO (a few hundred lines, well-understood).
#
# RECOMMENDATION: do (A) the wrapper for GRPO — TRL is the
#   mature, distributed, reward-hook-friendly path, and the wrapper
#   is a one-time cost. Keep (B) as fallback if the wrapper
#   fights us. Reason we would NOT just "always use TRL": TRL
#   doesn't know about our mLSTM recurrent `.step()` decode, so
#   even wrapped, our `generate()` must drive the recurrent path
#   (and that path has a known RoPE bug — see section 8).
#
# =============================================================================
# 5. THE LONG-CONTEXT-VIA-mLSTM TRAINING TRICK (the actual point)
# =============================================================================
# To make the model USE the mLSTM for >window context, the SFT/RL
# data must contain examples where the RELEVANT info is > 2048
# tokens before the answer. Then attention (windowed at train time)
# can't solve it alone; the gradient is forced to route through
# the mLSTM recurrence.
#
# Concrete recipe:
#   - Build SFT examples: long doc (PG19 book chunk / long repo)
#     + question whose answer depends on an EARLY passage.
#   - Train with the model in RECURRENT-friendly mode for the long
#     prefix (the mLSTM .step() carries state), then CE on the
#     answer. This is exactly the variable-context eval inverted
#     into training.
#   - CRUCIAL: this needs the recurrent decode path to WORK
#     (currently buggy — section 8). Until fixed, we train in
#     parallel mode at <= context_length and only extrapolate at eval.
#
# =============================================================================
# 6. LIBRARY DECISION (Henry asked: use libs or self-implement?)
# =============================================================================
# Researched:
#   - TRL (huggingface/trl): SFTTrainer accepts nn.Module
#     (OK for Stage 2). GRPOTrainer needs PreTrainedModel
#     (blocks our custom arch unless wrapped). Mature, distributed,
#     reward hooks. NOT a fit as-is for our custom recurrent decode.
#   - ToolBench / ToolLLaMA (OpenBMB): the DATA + recipes for
#     agentic tool use. We use their DATASET, not their trainer.
#   - AgentInstruct: 25M synthetic post-training pairs (incl.
#     tool use, coding) — dataset, model-agnostic. Good SFT fuel.
#   - We do NOT reinvent SFT/GRPO from zero if a lib fits.
#
# Decision:
#   - SFT  -> TRL SFTTrainer (fits, less code, we keep our
#           data pipeline + recurrent eval).
#   - GRPO -> TRL IF we wrap as PreTrainedModel (recommended),
#           else self-implemented GRPO loop reusing train.py.
#   - We DO implement ourselves ONLY the parts libs can't touch:
#     the mLSTM recurrent training/eval path (TRL is blind to it).
#
# =============================================================================
# 7. VRAM BUDGET AT STAGE 2-3 (unfrozen base)
# =============================================================================
# Stage 1 (base frozen) used ~12 GB at 1024x2. Unfreezing the
# base ~3x's trainable params + needs grad/optimizer state for ALL
# 611M params -> expect ~2-3x VRAM. Mitigations available:
#   - gradient_checkpointing (Henry said skip earlier for Stage 1;
#     likely NEEDED at Stage 2 — re-enable).
#   - LoRA on the base (keep mLSTM full) if full-unfreeze OOMs.
#   - grad_accum (already wired) to shrink per-step activation.
#   - smaller effective batch / shorter seq during SFT if needed.
# 16 GB is tight for full-unfreeze + 4096 ctx; plan for
#   checkpointing + maybe LoRA-on-base as the realistic path.
#
# =============================================================================
# 8. OPEN RISKS / THINGS STILL BROKEN
# =============================================================================
# (a) RECURRENT DECODE RoPE BUG: walking the model token-by-token
#     via `generate_step` (which calls backbone(input_ids=last,
#     past_key_values=..., position_ids=...)) hits a transformers-5.5
#     error inside Qwen attention (apply_rotary_pos_emb shape
#     mismatch). Root: HF's Qwen2Attention.forward expects
#     `position_embeddings` (cos/sin) passed DOWN from the model's
#     top-level forward; our replaced decoder layer skips that
#     computation, so passing position_ids via **kwargs doesn't
#     reconstruct it. IMPACT: generation + recurrent training +
#     the recurrent-context eval all blocked. NOT yet fixed.
#     Fix path: either (i) compute cos/sin in XlstmQwenLayer
#     and pass position_embeddings explicitly, or (ii) stop
#     replacing the layer and instead monkey-patch its forward.
# (b) VARIABLE-CONTEXT EVAL is parallel-only (capped at L<=4096
#     by VRAM: the mLSTM causal-mask is context_length^2; a
#     16384 parallel forward OOMs at 16GB). The paper's
#     "2048->16384" claim needs recurrent decode (blocked by a)
#     OR a bigger GPU. Tonight's data stops at 4096 (2x).
# (c) EVAL DTYPE NOISE: RMSNorm "Mismatch dtype" warning
#     (input bf16, weight float) — cosmetic, but worth a cast
#     fix so ppl numbers are clean.
#
# =============================================================================
# 9. CONCRETE NEXT ACTIONS (ordered)
# =============================================================================
# 1. [now] Let CPT 10000-step run finish; capture final ppl +
#    variable-context delta. (running, PID 278577)
# 2. [next] Fix (a) the recurrent RoPE bug — unblocks generation
#    + recurrent long-context training/eval (the real test of the
#    paper's extrapolation claim). Highest-value fix.
# 3. [then] Build the PreTrainedModel wrapper (section 4A) so
#    TRL GRPO can load us; verify .generate() + save/load.
# 4. [then] Stage 2 SFT: unfreeze all, train on code+reason+
#    agentic+long-context mix (ToolBench/AgentInstruct/PG19-QA).
#    Add gradient_checkpointing; consider LoRA-on-base if OOM.
# 5. [then] Stage 3 GRPO: outcome rewards (tool-exec, test-
#    pass, long-context retention) via TRL GRPO (or self-loop).
# 6. [throughout] Keep the variable-context eval as the north
#    star: we succeed when patched_ppl at L=4096/8192 BEATS
#    base, i.e. the mLSTM is genuinely carrying long context.
