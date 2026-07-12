This will be the plan for the project. both of us shall modify and change it according to our needs.

my plan:
we will take our trusty qwen2.5 coder 0.5B model and modify it again! but a bit differently.
we will keep the compleate model intact and even keep the attention mechanism intact. we will instead squeeze in the XLSTM layer between the attention and feedforward layers. this is to give the model a memory-like capability without modifying the attention mechanism. we will of course use LoRA to adapt the model to the new XLSTM part in its layers and we shall use residual connections bridging the XLSTM layer to give the model both memory and attention capabilities and us trying not to break the model.

picture ish:

 input -> attention -> XLSTM -----------> + feedforward -> output
                    |                     ^
                    |-residual connection |

this is kinda illustrated of what we are going to do.


================================================================================
REFINED SPEC  (review pass, 2026-07-12 — Hermes)
================================================================================
The base idea is sound and a known-safe pattern: insert a recurrent/memory
sublayer between attention and FFN, freeze the base, add a residual, train only
the new path. Below is the same idea with the open questions pinned down so we
don't rediscover them mid-build.

1. xLSTM primitive = mLSTM (matrix memory, GPU-parallel; the one that gives
   the LM gains in xLSTM 2.0). Use NVIDIA's official `xlstm` PyPI package for
   the cell rather than hand-rolling it.
   (sLSTM is the scalar/heads variant — not what we want here.)

2. Layer topology = xLSTM as its OWN sublayer with its own pre-norm + residual,
   sitting between the attention sublayer and the FFN sublayer. (Not a parallel
   adapter on the attention stream — that was ambiguous in the original sketch.)

   Per decoder layer (x24 in Qwen2.5-0.5B):

     x --> [RMSNorm] --> Attention -->+
          (FROZEN)                 |
                                    +  residual add
                                    |
          h = x + Attn(Norm(x))    |
                 |                  |
                 v                  |
          h --> [RMSNorm] --> xLSTM -->+
               (NEW, trainable)     |   (mLSTM memory)
               (init ~identity/out~0)|   residual add
                                    + 
          z = h + xLSTM(Norm(h))   |
                 |                  |
                 v                  |
          z --> [RMSNorm] --> FFN -->+
               (FROZEN)            |   residual add
                                    +
          out = z + FFN(Norm(z))
                 |
                 v
            LM HEAD (frozen)

   Compact:  x -> Attn -> +res -> xLSTM -> +res -> FFN -> +res -> next layer
                  (frozen)        (NEW)         (frozen)

3. LoRA targets = xLSTM block is FULLY TRAINABLE (it's small); the FROZEN base
   (attention + FFN) gets optional LoRA so it can re-adjust to the new
   composition. (Alternative: LoRA the xLSTM in/out projections — not chosen.)

4. Dim matching: xLSTM output must equal hidden dim (896 for 0.5B) to add
   residually. Use an auto in/out linear or native-dim block.

5. Insertion scope: all 24 layers to start; reduce to a subset if VRAM bites.

6. Recurrent state: unlike attention (token-parallel), xLSTM carries state
   across the sequence. Must init/reset per sequence and during generation.
   This is a real gotcha — handle in the patcher + generate loop.

7. Safe init (serves "don't break it"): init the xLSTM output projection to
   ~zero/identity so at step 0 the model == original Qwen, then it learns to
   add memory. Strongly recommended.

8. Dataset + eval: NOT yet decided. Need (a) training corpus (code? reasoning?)
   and (b) metric (HumanEval pass@1? perplexity vs base? downstream code gen?).
   Blocking for any run.

RESOLVED Q&A (2026-07-12 — Henry)
  Q1:  mLSTM is the way to go. sLSTM added LATER as a config option.
  Q3/Q5: Confirmed — implement mLSTM layer first (full-train, all 24 layers).
        LoRA is added ONLY in a later training run, not stage 1.
  Q8:  Training mode = CONTINUED PRETRAINING (CPT), NOT from-scratch, NOT SFT.
        Goal of stage 1: teach the mLSTM block to function on a frozen base.
        -> AGREED: leave LoRA OUT of stage 1.

DATA SPACE (reasoning + math + code + agentic; from web-search 2026-07-12)
  - Code:          StarCoder-Data / The Stack, or SmolLM2 code mix (long files = long-range).
  - Math+code:     MathCoder2 corpus (19.2B-tok math+code CPT set; CPT lifts math reasoning).
  - General long:  FineWeb / Dolma; explicitly PG-19 (Gutenberg books, ~20x longer than WikiText).
  - Agentic (L1): agentic REASONING text only (tool-use docs, ReAct/tool traces).
                    Full multimodal trajectories (AgentTrek-style) deferred to a later stage.
  -> Stage-1 mix should be long-context code + math-code + some long text, so the
     memory can actually be measured. Agentic full trajectories come later.

EVAL (how we know it "works")
  - Primary:  validation loss / ppl during CPT vs a FROZEN-BASE baseline on same data.
  - Memory probe: PG-19 perplexity — does grafted mLSTM cut long-range ppl vs base?
                   (direct test of the "memory capability" hypothesis)
  - Downstream (later): HumanEval / MBPP (code), GSM8K / MATH-500 (math).

KEY IMPLEMENTATION GOTCHA (stage 1)
  - mLSTM is RECURRENT (sequential, carries state across the sequence); attention is parallel.
  - Patcher + training loop MUST carry/reset state at document boundaries (packed seqs).
  - FIRST forward pass must NUMERICALLY EQUAL the frozen base (identity-init sanity check).
    This is the make-or-break test before any training run.

NEXT STEP: implement the mLSTM layer (smoke test: out==base at init, state carries),
           THEN CPT run, THEN eval, THEN (later) optional LoRA + sLSTM option.
