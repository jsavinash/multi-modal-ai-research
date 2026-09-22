# P9 — Instruction Tuning (SFT)

| | |
|---|---|
| **Goal** | Turn the pretrained base model into something that follows a small set of **templated** instructions over retrieved context — with the prompt-span loss mask and a forgetting gate |
| **Wall-clock** | 1 evening |
| **Compute** | ~30–60 min (~2 000 steps) |
| **Prerequisites** | P8 gate green — a checkpoint with val ppl ≤ 11 |
| **Blocks** | P11 (so `engine: scratch` is worth using) |
| **Read first** | [analysis §9.5](../00_ANALYSIS.md#95-instruction-tuning-phase-9), [§9.6](../00_ANALYSIS.md#96-what-an-11-m-param-model-can-and-cannot-do-set-expectations-in-advance), [docs/02 §6.11](../02_FUNCTION_REQUIREMENTS.md#611-sft-mmarllmsftpy-sft_datapy-eval_sftpy) |

---

## Deliverables

```
src/mmar/llm/sft.py                 # SFTTrainer + dataset builder
src/mmar/llm/sft_data.py            # prompt/response pair construction and templating
src/mmar/llm/eval_sft.py            # exact-match / ROUGE-L on held-out templated prompts
configs/train_sft.yaml
tests/test_sft_mask.py  test_sft_data.py
artifacts/sft/dataset.jsonl         # generated, versioned by a hash of the builder
checkpoints/micro-sft/              # SFT checkpoint
```

## The one thing that makes SFT correct

```python
input_ids, loss_mask = tokenizer.encode_chat(turns, mask_prompt=True)
loss = cross_entropy(logits, targets, mask=loss_mask)   # loss ONLY on assistant tokens
```

`test_sft_mask.py` must assert that `loss_mask == 0` on every prompt token and `1` on every assistant
token. Training on the prompt tokens teaches the model to produce **your questions**, which destroys
multi-turn behaviour — this is the most common SFT bug and it is invisible in the loss curve.

## Recipe constraints that follow from an 11 M-param model

From analysis §9.5/§9.6 — a model this small **cannot** generalise over diverse instructions:

| Do | Don't |
|---|---|
| Short, templated tasks: `summarise`, `extract`, `answer from context` | "Be a helpful assistant" role-play |
| ~5–20 distinct instruction templates, varied in content not in form | Hundreds of free-form instruction phrasings |
| Answer verbatim-in-context questions the model can actually get right | Questions requiring reasoning or synthesis across chunks |
| ~2 000 pairs from your own ingested documents | A 100 k-sample general instruction dataset |

**This is a property of the model size, not of the recipe.** Setting this expectation up front is the
difference between a working demo and a phase you declare broken.

## Also required

- **20 % raw pretraining text mixed into every batch.** Catastrophic forgetting is real at this scale
  and the mixing is what prevents it (analysis §9.5).
- **The forgetting gate:** evaluate TinyStories val ppl after SFT; fail loudly if it exceeds
  **1.3×** the P8 value.
- **Loss 0.05** in SFT (0.0 in pretraining) — you are now over-fitting deliberately to a narrow
  distribution, which is exactly what SFT is.
- Lower LR than pretraining (`3e-4`, cosine to `3e-5`) and ~2 000 steps. More steps will overfit.

## Forbidden

- Training on prompt tokens (see above).
- Evaluating on training loss. A model that memorised 2 000 pairs has training loss ≈ 0 and answers
  nothing.
- Using a general instruction dataset. It will not work at this size and it will hide the fact.
- Skipping the forgetting gate because "the samples look fine".
- Reporting a success rate without a held-out set the model never saw.

## Tasks

1. `sft_data.py`: build pairs from your ingested store (summarise a chunk, extract a field, answer a
   question whose answer is verbatim in the chunk), plus a hand-written seed set of ~50 templates.
2. Hold out **200 prompts** the model never trains on, and write them to `tests/eval_sets/sft_holdout.jsonl`.
3. `train_sft.yaml` with the mixing ratio, LR and step count above.
4. `sft.py`: `SFTTrainer` reusing P7's `Trainer` with a masked loss and the 20 % text mixing.
5. `eval_sft.py`: exact-match + ROUGE-L on the holdout set, plus the TinyStories ppl regression check.
6. Run it. Report the holdout score and the ppl ratio honestly.

## Exit gate

```bash
make train-sft
python -m mmar.llm.eval_sft --checkpoint checkpoints/micro-sft --holdout tests/eval_sets/sft_holdout.jsonl
pytest tests/test_sft_mask.py -v
```

## Definition of Done

- [ ] `test_sft_mask.py` green: mask is 0 on prompt tokens, 1 on assistant tokens; output pasted
- [ ] Holdout score measured on 200 unseen prompts; the real number pasted with per-template breakdown
- [ ] Forgetting gate measured: SFT ppl / P8 ppl, and it is ≤ 1.3
- [ ] 10 generated answers read by a human and pasted; at least 5 must be on-task
- [ ] Peak RSS and wall-clock measured
- [ ] `docs/12_RESULTS.md` updated

---

## 📋 Copy-paste prompt

```
ROLE
You are a senior ML systems engineer. You write code that is measured, not asserted.

CONTEXT - read these first, in this order
  1. docs/00_ANALYSIS.md  section 9.5 (instruction tuning) and section 9.6 (what an 11M
     model can and cannot do - internalise this BEFORE judging your results)
  2. docs/02_FUNCTION_REQUIREMENTS.md  section 6.11 (SFT signatures), section 6.6 (Trainer),
     section 6.1 (encode_chat / loss mask)
  3. docs/01_ROADMAP.md  section 5 (Definition of Done)
  4. Existing code: src/mmar/llm/train.py, src/mmar/llm/tokenizer.py, src/mmar/llm/data.py,
     src/mmar/llm/checkpoint.py, src/mmar/rag/store.py

MACHINE
  MacBook Air M1, 8 GB unified memory, fanless. python3.11 in .venv.
  Compute budget: ~30-60 min (~2 000 steps). Peak RSS budget: 1.0 GB.

HARD CONSTRAINTS
  - Prerequisite: a P8 checkpoint with val ppl <= 11. If the checkpoint is missing, or its
    report.json shows val ppl > 11, STOP and say so. Never SFT a base model that failed P8.
  - THE LOSS MASK IS THIS PHASE: input_ids, loss_mask = tokenizer.encode_chat(turns,
    mask_prompt=True); loss = cross_entropy(logits, targets, mask=loss_mask).
    test_sft_mask.py must assert loss_mask == 0 on EVERY prompt token and 1 on EVERY
    assistant token. Training on prompt tokens teaches the model to emit YOUR QUESTIONS -
    it is the most common SFT bug and it is invisible in the loss curve.
  - Mix 20% raw pretraining text into every batch. Catastrophic forgetting is real at 11M
    params and the mixing is what prevents it (analysis 9.5).
  - Forgetting gate: TinyStories val ppl AFTER SFT / P8 val ppl must be <= 1.3. Measure it
    on the same val shard; fail loudly if it is exceeded.
  - 5-20 DISTINCT instruction templates varied in CONTENT, not phrasing. ~2 000 pairs built
    from YOUR own ingested store (summarise / extract / answer-verbatim-in-context). NOT a
    general instruction dataset - at 11M params it will not help and will hide failure.
  - Hold out 200 prompts FIRST (tests/eval_sets/sft_holdout.jsonl), written before any
    training pair exists. Report exact-match + ROUGE-L overall AND per-template on the
    holdout. Never report training loss as a success metric.
  - LR cosine 3e-4 -> 3e-5, ~2 000 steps (analysis 9.5 / phase body). Do not change lr,
    batch size or step count "just to converge faster".
  - Expect training loss to fall near 0 - you are deliberately over-fitting ~2 000 pairs.
    That is exactly why the holdout number, not the loss, is the phase result.
  - 10 generated answers pasted verbatim; a human judges >= 5 on-task. Do not infer
    quality from the loss curve.

DELIVERABLES - exactly these paths
  src/mmar/llm/sft.py
  src/mmar/llm/sft_data.py
  src/mmar/llm/eval_sft.py
  configs/train_sft.yaml
  tests/test_sft_mask.py
  tests/test_sft_data.py
  tests/eval_sets/sft_holdout.jsonl     (200 prompts; generated once, never regenerated)
  artifacts/sft/dataset.jsonl           (generated; record the builder hash in meta)
  checkpoints/micro-sft/
  docs/12_RESULTS.md                    (UPDATE with the P9 section)

KEY REQUIREMENTS
  - Implement every signature in docs/02 section 6.11: SFTPair, SFTDatasetBuilder,
    SFTTrainer, SFTReport, eval_sft(). SFTTrainer REUSES P7's Trainer - grad accumulation,
    clip_global_norm with the pre-clip norm logged every step, cosine schedule, divergence
    guard, bit-exact resume. Only the loss mask and the 20% text mixing are new.
  - sft_data.py: pairs built from your store via chunk ids - summarise a chunk, extract a
    field, answer a question whose answer is verbatim in the chunk - plus ~50 hand-written
    seed templates. Every pair carries template_id in meta for the per-template breakdown.
    Builder is deterministic: same store -> byte-identical jsonl.
  - Holdout disjointness: assert no (template_id, source_chunk_id) pair appears in BOTH
    train and holdout (test_sft_data.py).
  - eval_sft.py returns {'exact_match', 'rouge_l', 'val_ppl', 'forgetting_ratio'} in one
    call; forgetting_ratio = sft_val_ppl / base_val_ppl.
  - Checkpoint meta records: base checkpoint path, base val ppl, forgetting_ratio,
    holdout scores, dataset hash, is_sft=True (flag exists in docs/02 section 6.8).
  - Pair-quality gate BEFORE training: decode and paste 10 random pairs. If the response
    is not actually derivable verbatim from the prompt's context, the pair is wrong -
    fix the builder first.

FORBIDDEN
  - Training on prompt tokens (see HARD CONSTRAINTS).
  - Evaluating on training loss or reporting a success rate without the 200-prompt holdout.
  - A general instruction dataset (Alpaca / ShareGPT / OASST style).
  - Skipping the forgetting gate because the samples look fine.
  - Changing lr / batch / steps to converge faster.
  - Any file containing "# TODO" or "raise NotImplementedError".
  - Reporting a command as passing without pasting its raw output.

TASKS
  1. sft_data.py: builder + ~50 seed templates. Write the 200-prompt holdout FIRST, before
     any training pair, so it cannot leak.
  2. Decode and paste 10 built pairs; human-check each is answerable verbatim from context.
  3. tests/test_sft_mask.py (0 on prompt positions, 1 on assistant positions) and
     tests/test_sft_data.py (templating, determinism, holdout disjointness). COMMIT.
  4. configs/train_sft.yaml: cosine 3e-4 -> 3e-5, ~2 000 steps, 20% text mix.
  5. sft.py: SFTTrainer over P7's Trainer - masked loss + text mixing only.
  6. eval_sft.py: exact-match + ROUGE-L + forgetting ppl ratio in one call.
  7. make train-sft. Record grad_norm, loss, tok/s, RSS throughout.
  8. Run the gate; write the docs/12_RESULTS.md P9 section.

EXIT GATE - paste the complete unedited output
  make test-fast
  pytest tests/test_sft_mask.py tests/test_sft_data.py -v
  make train-sft
  python -m mmar.llm.eval_sft --checkpoint checkpoints/micro-sft \
      --holdout tests/eval_sets/sft_holdout.jsonl
  /usr/bin/time -l python -m mmar.llm.eval_sft --checkpoint checkpoints/micro-sft \
      --holdout tests/eval_sets/sft_holdout.jsonl

EVIDENCE REQUIRED IN YOUR RESPONSE
  1. make test-fast + test_sft_mask.py raw output (the actual mask assertions, not just
     "passed").
  2. The 10 decoded training pairs, pasted verbatim.
  3. The holdout score: exact-match and ROUGE-L, overall AND per-template, with the TRUE
     numbers even if below 50%. Name the failing templates and why you think they fail.
  4. The forgetting gate: SFT val ppl / P8 val ppl as a number, against the 1.3 limit.
  5. Training summary: steps, final loss, wall-clock, peak RSS vs the 1.0 GB budget.
  6. TEN generated answers from micro-sft pasted verbatim; state how many are on-task
     (target >= 5). This is a human judgement - do not infer it from the loss.
  7. Anything you could NOT verify, stated explicitly as UNVERIFIED.

Before writing code, answer in at most three lines each: what is the exit gate; which
command proves it; what is the most likely way to get this silently wrong.
```
