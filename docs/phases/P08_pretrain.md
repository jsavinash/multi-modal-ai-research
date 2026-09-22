# P8 — Pretrain  ★ MILESTONE 3 driver

| | |
|---|---|
| **Goal** | Actually train the from-scratch model: a smoke run first, then the Chinchilla-optimal overnight run |
| **Wall-clock** | 1 evening of setup + **1–1.5 h smoke** + **3–8 h overnight** |
| **Compute** | the only expensive phase in the project |
| **Prerequisites** | P7 gate green — **`make test-parity` must pass before you start any run** |
| **Blocks** | P9, P10 |
| **Read first** | [analysis §5](../00_ANALYSIS.md#5-throughput-model--feasibility) (especially §5.2), [§9.3](../00_ANALYSIS.md#93-training-loop-the-parts-that-actually-matter), [§11](../00_ANALYSIS.md#11-data-strategy) |

---

## Deliverables

```
src/mmar/llm/train.py                     # EXTEND: shard prep CLI, run reports
scripts/prepare_tinystories.py            # download + tokenize + shard, resumable
docs/12_RESULTS.md                        # UPDATE: loss curves, ppl, TPS, wall-clock, peak RSS
artifacts/runs/<run_name>/report.json     # machine-readable TrainReport per run
checkpoints/<run_name>/                   # model + optimizer + config + meta, keep_last=3
```

## Function requirements

`Trainer`, `TrainReport`, `train_step`, `evaluate`: [docs/02 §6.6](../02_FUNCTION_REQUIREMENTS.md#66-mmar-llmtrainpy).
Data: [docs/02 §6.4](../02_FUNCTION_REQUIREMENTS.md#64-mmar-llmdata-py).

The two runs are already configured — do not invent hyperparameters:

| Config | Tokens | Steps | Expected wall-clock | Target |
|---|---|---|---|---|
| `configs/train_smoke.yaml` | 40 M | 2 442 | ~1–1.5 h | val ppl < 25 |
| `configs/train_micro.yaml` | **220 M** | 13 428 | **~3–8 h** | **val ppl 7–11** |

**220 M tokens and not 40 M.** Chinchilla says `D ≈ 20·N`; for 11 M params that is 220 M. Training
`micro` on 40 M tokens would be 5× under-trained and would produce a model that is *worse than a
smaller one trained properly* (analysis §5.2). This is the single most important number in the phase.

## The four failure signatures to watch for

| Symptom | Cause | Action |
|---|---|---|
| `grad_norm` > 5 in the first 200 steps | LR too high, or warmup too short | Stop, lower `lr` to 3e-4, restart |
| Loss plateaus around 3.5 and will not move | Weight decay applied to norms (analysis §9.3) | Fix `build_optimizer`, restart |
| Loss falls but samples are gibberish | **Tokenizer/shard fingerprint mismatch** (R4) or a mask bug | `pytest -m parity`; check `meta.json` |
| val/train gap widening past 1.5 | Data bug or the divergence guard firing correctly | Inspect the shards — `peek_batch` and decode it back to text |

## Forbidden

- Starting the full run before the smoke run has finished and produced val ppl < 25.
- Changing `lr`, `micro_batch_size` or `grad_accum_steps` between the smoke run and the full run.
  They must be identical so the smoke run actually predicts the full one.
- Raising `max_process_rss_mb` if memory is tight — lower `micro_batch_size` instead.
- Restarting from scratch because a run was interrupted. `--resume auto` exists and is bit-exact.
- Running the full run on battery. It will throttle and take twice as long.
- Touching the machine during the overnight run (checking email is fine; a video call is not).
- Reporting a perplexity number without the loss curve that produced it.

## Tasks

1. `scripts/prepare_tinystories.py`: download → split by document → train/reuse the tokenizer →
   write `uint16` shards with the fingerprint in `meta.json`. Resumable and idempotent.
2. Verify the data before training: `peek_batch` a validation batch and **decode it back to text and
   read it**. If it is not coherent English, stop and fix the data path before anything else.
3. Run `make train-smoke`. Watch `grad_norm`, loss, tok/s and RSS. Record everything.
4. Only if smoke val ppl < 25: run `make train-micro` overnight, plugged into power.
5. Write `artifacts/runs/<name>/report.json` and update `docs/12_RESULTS.md` with: loss curve
   (sampled every 500 steps), final train/val loss, val perplexity, measured tok/s, wall-clock,
   peak RSS, and the sample generations.

## Exit gate

```bash
python scripts/prepare_tinystories.py --tokens 220000000
make train-smoke
make train-micro                     # overnight
python -m mmar.cli chat --config configs/train_micro.yaml
```

## Definition of Done

- [ ] A decoded validation batch, pasted, that is recognisably English
- [ ] Smoke run completed; val ppl < 25; the number pasted
- [ ] Full run completed; **val ppl ≤ 11**; the number pasted
- [ ] The train/val loss curve, sampled, pasted (a table is fine)
- [ ] `grad_norm` never exceeded 5 (or the divergence was investigated and explained)
- [ ] **A human read of 10 generated samples.** At least 7 must be grammatical English. This cannot
      be delegated or inferred from the loss
- [ ] Measured: tok/s, wall-clock, peak RSS, and effective `eff(N)` vs the estimate band
- [ ] `docs/12_RESULTS.md` updated; `[est]` markers in the analysis doc replaced where measured
- [ ] Checkpoint resume verified: kill and `--resume auto` on the smoke run; the loss continues

---

## 📋 Copy-paste prompt

```
ROLE
You are a senior ML systems engineer running a real training job on constrained hardware. You
write code that is measured, not asserted. You never report a number you did not measure.

CONTEXT - read these first
  1. docs/00_ANALYSIS.md  section 5 whole (the throughput model AND section 5.2, the Chinchilla
     conclusion about why the `small` preset is a trap), section 9.3 (the five training-loop
     requirements), section 14 (risk register R1-R5), section 11 (data strategy)
  2. docs/02_FUNCTION_REQUIREMENTS.md  section 6.4 (data) and 6.6 (trainer)
  3. configs/train_smoke.yaml and configs/train_micro.yaml - these are ALREADY TUNED
  4. docs/01_ROADMAP.md section 6 (what to do when something fails)

MACHINE
  MacBook Air M1, 8 GB unified memory, FANLESS. Expect the last hour of a long run to be ~20%
  slower than the first. Peak RSS during training must stay under 1.2 GB.

HARD CONSTRAINTS
  - `make test-parity` MUST be green before you start ANY run. Do not skip this.
  - The smoke run (40M tokens, ~1-1.5 h, target val ppl < 25) MUST complete and pass BEFORE the
    full run (220M tokens, ~3-8 h, target val ppl 7-11) starts.
  - Do NOT change lr, micro_batch_size or grad_accum_steps between the smoke and full runs. They
    must be identical so the smoke run actually predicts the full one.
  - 220M tokens and not 40M. Chinchilla says D ~= 20*N; for 11M params that is 220M. Training
    micro on 40M tokens is 5x under-trained and would produce a model worse than a smaller one
    trained properly. This is the most important number in the phase.
  - If memory is tight, LOWER micro_batch_size. Never raise max_process_rss_mb.
  - A run that is interrupted is RESUMED with --resume auto, never restarted from scratch. The
    resume must be bit-exact in LR schedule and RNG state.
  - Run the full job on AC power. On battery it throttles and takes twice as long.

DELIVERABLES - exactly these paths
  scripts/prepare_tinystories.py        (download + tokenize + shard, idempotent, resumable)
  src/mmar/llm/train.py                 (EXTEND if needed for the run-report CLI)
  artifacts/runs/<run_name>/report.json (machine-readable TrainReport per run)
  checkpoints/<run_name>/               (model + optimizer + config + meta, keep_last=3)
  docs/12_RESULTS.md                    (UPDATE with the P8 measurements)

BEFORE YOU TRAIN - do this and report the result
  Build the shards, then peek_batch() a validation batch, DECODE IT BACK TO TEXT and print it.
  If it is not recognisably English, STOP and fix the data path. Nothing downstream can be right
  if this is wrong.

WATCH FOR THESE FOUR FAILURE SIGNATURES AND REPORT WHICH YOU SEE
  1. grad_norm > 5 in the first 200 steps -> LR too high or warmup too short. Stop, lower lr to
     3e-4, restart.
  2. Loss plateaus around 3.5 and will not move -> weight decay was applied to norm weights.
     Fix build_optimizer, restart.
  3. Loss falls but samples are gibberish -> tokenizer/shard fingerprint mismatch (risk R4) or a
     mask bug. Run pytest -m parity, check meta.json.
  4. val/train gap widening past 1.5 -> a data bug, or the divergence guard firing correctly.
     Inspect the shards and decode them.

FORBIDDEN
  - Starting the full run before the smoke run passes.
  - Changing hyperparameters between smoke and full runs.
  - Raising runtime.max_process_rss_mb. Lower micro_batch_size instead.
  - Restarting from scratch instead of --resume auto.
  - Reporting a perplexity number without the loss curve that produced it.
  - Reporting a tokens/second number you did not measure.
  - Any file containing "# TODO" or "raise NotImplementedError".
  - Reporting a command as passing without pasting its raw output.

TASKS
  1. scripts/prepare_tinystories.py: download, split by DOCUMENT, train or reuse the tokenizer,
     write uint16 shards with the tokenizer fingerprint in meta.json. Idempotent + resumable.
  2. Decode and print a validation batch. Report whether it is English.
  3. make train-smoke. Record grad_norm, loss, tok/s, RSS throughout.
  4. If and only if smoke val ppl < 25: run make train-micro overnight.
  5. Write artifacts/runs/<name>/report.json for both runs.
  6. Update docs/12_RESULTS.md.

EXIT GATE - paste the complete unedited output
  python scripts/prepare_tinystories.py --tokens 220000000
  make train-smoke
  make train-micro                            # overnight
  python -m mmar.cli chat --config configs/train_micro.yaml
  # and: kill + resume check on the smoke run

EVIDENCE REQUIRED IN YOUR RESPONSE
  1. The DECODED validation batch, printed verbatim, so a human can judge whether it is English.
  2. Smoke run summary box: final train loss, val loss, val ppl, tok/s, wall-clock, peak RSS.
  3. Full run summary box, same fields, plus the val ppl compared against the <= 11 target.
  4. The loss curve sampled every 500 steps, as a table: step, train_loss, val_loss, lr, grad_norm.
  5. TEN generated samples from the final checkpoint, verbatim. State how many of the ten are
     grammatical English. This is a human judgement - do not infer it from the loss.
  6. The resume check: the loss immediately before the kill and immediately after the resume,
     showing the trajectory continued rather than restarting.
  7. Measured tok/s and the resulting eff(N) percentage, next to the [est] band in
     docs/00_ANALYSIS.md 5.1.
  8. Anything you could NOT verify, stated explicitly as UNVERIFIED.

This phase drives MILESTONE 3. The milestone is: "I pretrained an 11M-param model on this laptop
and it writes grammatical short stories." Passing tests do not establish that. Ten readable
samples and a val perplexity do.

Before writing code, answer in at most three lines each: what is the exit gate; which command
proves it; the most likely way to get this silently wrong.
```

