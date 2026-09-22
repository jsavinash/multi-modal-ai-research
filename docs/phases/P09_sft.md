# P9 — Instruction Tuning (SFT)

| | |
|---|---|
| **Goal** | Turn the pretrained base model into something that follows a small set of **templated** instructions over retrieved context — with the prompt-span loss mask and a forgetting gate |
| **Wall-clock** | 1 evening |
| **Compute** | ~30–60 min (~2 000 steps) |
| **Prerequisites** | P8 gate green — a checkpoint with val ppl ≤ 11 |
| **Blocks** | P11 (so `engine: scratch` is worth using) |
| **Read first** | [analysis §9.5](../00_ANALYSIS.md#95-instruction-tuning-phase-9), [§9.6](../00_ANALYSIS.md#96-what-an-11-m-param-model-can-and-cannot-do-set-expectations-in-advance) |

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

<!-- PROMPT -->
