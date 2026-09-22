# P7 — MLX Transformer + Parity  ★ MILESTONE 2

| | |
|---|---|
| **Goal** | The from-scratch MLX transformer, the training loop, the sampler and the KV cache — with the NumPy oracle proving the maths is right and a measured tokens/second number |
| **Wall-clock** | 2 evenings |
| **Compute** | ~13 min (parity, overfit test, throughput benchmark) |
| **Prerequisites** | P5 (tokenizer + shards) and P6 (oracle) gates green |
| **Blocks** | P8 (training cannot start until `make test-parity` is green) |
| **Read first** | [analysis §9](../00_ANALYSIS.md#9-from-scratch-llm-design), [§5.1](../00_ANALYSIS.md#51-the-training-time-equation), [docs/02 §6.3–§6.8](../02_FUNCTION_REQUIREMENTS.md#63-mmar-llmlayerspy--mmar-llmmodelpy) |

---

## Deliverables

```
src/mmar/llm/layers.py  model.py  optim.py  train.py  sample.py  checkpoint.py
benchmarks/throughput.py
tests/test_parity_ops.py  test_parity_forward.py  test_kv_cache.py
tests/test_llm_overfit.py  test_sample.py  test_train_resume.py  test_init.py
docs/12_RESULTS.md        # UPDATE with the measured TPS per preset
```

## The four tests that gate this phase

| Test | Assertion | Why it exists |
|---|---|---|
| `test_parity_ops.py` (`-m parity`) | MLX vs NumPy `< 1e-4` for softmax, rms_norm, rope, attention, swiglu, cross_entropy | A RoPE or mask bug is invisible in the loss curve |
| `test_parity_forward.py` (`-m parity`) | Full forward, MLX vs NumPy `< 1e-4`, random init **and** from a checkpoint | Some bugs only appear once weights are not near zero |
| `test_kv_cache.py` (`-m parity`) | Cached incremental decode == full forward, `atol=1e-4` | Catches the majority of KV-cache bugs in one shot |
| `test_llm_overfit.py` (`slow`) | Loss `< 0.1` on one fixed batch within 200 steps | Proves the model *can* learn at all, isolating model bugs from data bugs |

## Implementation requirements

From [docs/02 §6.3–§6.8](../02_FUNCTION_REQUIREMENTS.md#63-mmar-llmlayerspy--mmar-llmmodelpy):

- **The mask is built from `(q_len, kv_len, offset)`**, never from a `(T, T)` boolean. A `(T, T)`
  mask silently breaks when `q_len != kv_len` during cached decode, and it is the single most
  common way a from-scratch transformer "trains fine but generates noise".
- **No `mx.eval` inside the model.** Synchronisation belongs to the loops.
- **`init_model` must scale residual output projections by `(2·n_layers)**-0.5`** and use
  `std = 0.02`. This is the #1 cause of "my from-scratch model will not train" (analysis §9.1).
- **AdamW must exclude `tok_emb`, all `norm_*` weights and every 1-D tensor from weight decay.**
  Decaying norms produces a plateau that looks exactly like a data problem (analysis §9.3, point 1).
- **Loss computed in fp32** even with a bf16 forward (analysis §9.3, point 2).
- **The sampler is written out, not delegated.** A documented MLX bug in
  `mx.random.categorical` produced broken sampling of otherwise-correct logits — the model was fine
  and the sampler was not (analysis §9.4). Top-k then top-p, in that order; repetition penalty with
  a **window**; stop strings matched on the decoded tail.
- **`mx.compile` on the train step only**, and the step counter is a traced argument. A captured
  Python int silently retraces every step and loses the entire speed-up (analysis §9.3, point 4).
- **Grad norm is logged every step.** Divergence is visible ~500 steps before it is fatal (R1).

## Forbidden

- Starting any training run before `make test-parity` is green. The parity gate is the whole point of
  P6 existing.
- `mx.nn.Transformer` or any high-level transformer module, if such a thing exists.
- Skip connections implemented as `x = x + Attn(x)` without the pre-norm and residual scale.
- `mx.eval` inside `MMRTransformer.__call__` or any layer.
- Adding `compile=True` to the model forward (only the train step).
- Reporting a tokens/second number you did not measure.

## Tasks

1. `layers.py`: `RMSNorm`, `build_rope_cache`, `apply_rope`, `causal_mask`,
   `scaled_dot_product_attention`, `KVCache`, `Attention`, `SwiGLU`, `TransformerBlock`.
2. `model.py`: `MMRTransformer` with explicit `positions`, tied embeddings, `n_params()`, and
   `loss()` in fp32 over a mask.
3. `optim.py`: `cosine_schedule`, `clip_global_norm`, `init_model`, `build_optimizer`, `count_parameters`.
4. `checkpoint.py`: save/load with `config.json` + `meta.json` (including the tokenizer fingerprint
   and git SHA), `prune()`, and `CheckpointMismatch` on a config diff.
5. `sample.py`: the sampler, `generate`, streaming callback.
6. `train.py`: `Trainer` with grad accumulation, divergence guard, wall-clock budget, thermal relief,
   resumable checkpoints, and `peak_tracker`.
7. `benchmarks/throughput.py`: measure real TPS for `nano`, `micro` and `small` over 50 steps and
   write the numbers into `docs/12_RESULTS.md`, replacing the `[est]` markers.
8. The seven test files.
9. Run the overfit test to completion — it is the strongest single signal that everything is correct.

## Exit gate

```bash
make test-parity                              # ALL green
pytest tests/test_llm_overfit.py -v           # loss < 0.1
python benchmarks/throughput.py --preset micro --steps 50
python benchmarks/throughput.py --preset small --steps 50
python -m mmar.cli bench --preset micro
```

## Definition of Done

- [ ] `make test-parity` green — raw output pasted, including the max absolute difference per test
- [ ] Overfit test reaches loss < 0.1; the loss curve pasted
- [ ] KV-cache equivalence test green
- [ ] **Measured TPS for all three presets**, and the `[est]` markers in analysis §5.1 replaced with
      the measured values
- [ ] Peak RSS of a `micro` training step measured and reported
- [ ] `compile: true` vs `false` measured and both numbers recorded (proves the compile is not a
      no-op — risk R10)
- [ ] `mmar bench` prints a table of preset, params, TPS, peak RSS

---

## 📋 Copy-paste prompt

```
ROLE
You are a senior ML systems engineer. You are implementing the transformer ITSELF in Apple MLX.
You write code that is measured, not asserted. Everything you claim must have a pasted command.

CONTEXT - read these first, in this order
  1. docs/00_ANALYSIS.md  section 9.1 (architecture choices and why each one), 9.2 (forward
     pass), 9.3 (the five non-obvious training requirements), 9.4 (why parity tests), 5.1 (the
     throughput model), 14 (risk register)
  2. docs/02_FUNCTION_REQUIREMENTS.md  sections 6.3 through 6.8
  3. Existing code: src/mmar/llm/ref/*.py (the oracle you are judged against),
     src/mmar/llm/config.py, src/mmar/llm/data.py, src/mmar/llm/tokenizer.py

MACHINE
  MacBook Air M1, 8 GB unified memory, FANLESS. GPU is 8 cores, ~5.2 TFLOPS bf16, 68 GB/s.
  python3.11 in .venv with mlx installed.
  Peak RSS for a micro training step must stay under 1.2 GB.

NON-NEGOTIABLE DETAILS - each of these has silently broken a real run
  1. The causal mask is built from (q_len, kv_len, offset), NEVER from a (T, T) boolean. A
     (T, T) mask silently breaks when q_len != kv_len during cached decode. This is the most
     common way a from-scratch transformer "trains fine but generates noise".
  2. positions are EXPLICIT in the forward signature. You may derive them from cache.offset, but
     the parameter must exist - it is what makes the KV-cache equivalence test possible.
  3. init_model: std=0.02 for embeddings and linears, zeros for norms, and residual output
     projections scaled by cfg.residual_scale == (2*n_layers)**-0.5. Getting this wrong is the
     number one cause of "my from-scratch model will not train".
  4. build_optimizer: AdamW with weight decay ONLY on >=2-D parameter matrices. tok_emb, every
     norm weight and every 1-D tensor are EXCLUDED from decay. Decaying norms produces a plateau
     that looks exactly like a data problem.
  5. Loss is computed in fp32 even when the forward is bf16. softmax over 4096 bf16 logits loses
     enough precision to bias the gradient.
  6. mx.compile goes on the TRAIN STEP ONLY, and the step counter is a traced argument, never a
     captured Python int. A captured int silently retraces every step and destroys the speed-up.
  7. NO mx.eval inside the model or any layer.
  8. The sampler is written out BY HAND. Do NOT delegate to mx.random.categorical - there is a
     documented MLX bug where it produced broken sampling of otherwise-correct logits. Apply
     top-k THEN top-p. Repetition penalty uses a WINDOW, not the whole context. Stop strings are
     matched on the decoded tail, not per token.
  9. Grad norm is logged every log_interval and the running max is reported.

DELIVERABLES - exactly these paths
  src/mmar/llm/{layers,model,optim,train,sample,checkpoint}.py
  benchmarks/throughput.py
  tests/test_parity_ops.py  test_parity_forward.py  test_kv_cache.py
  tests/test_llm_overfit.py  test_sample.py  test_train_resume.py  test_init.py
  docs/12_RESULTS.md            (UPDATE: replace [est] markers with measurements)

REQUIRED BEHAVIOURS IN Trainer
  - grad accumulation over cfg.grad_accum_steps, with grads divided by it
  - clip_global_norm(grads, 1.0) every step, logging the PRE-clip norm
  - divergence guard: abort after 3 consecutive evals where val_loss > train_loss + 1.5
  - wall_clock_budget_h hard stop, writing a checkpoint BEFORE exiting
  - optional thermal relief: 25 min on / 5 min off, driven by `pmset -g therm`
  - resume must be bit-exact: continue the LR schedule and RNG state, never reset them
  - peak_tracker("train") around the whole loop, reported in TrainReport
  - evaluate() runs in eval mode, dropout off, on deterministic windows via peek_batch()

FORBIDDEN
  - Starting a training run before `make test-parity` is green.
  - mx.nn.Transformer or any high-level transformer module, if such exists.
  - Skipping the residual scale, or implementing pre-norm as post-norm.
  - compile=True on the model forward (train step only).
  - Reporting a tokens/second number you did not measure.
  - Any file containing "# TODO" or "raise NotImplementedError".
  - Reporting a command as passing without pasting its raw output.

TASKS
  1. layers.py: RMSNorm, build_rope_cache, apply_rope, causal_mask,
     scaled_dot_product_attention, KVCache, Attention, SwiGLU, TransformerBlock.
  2. model.py: MMRTransformer with explicit positions, tied embeddings, n_params(), loss() in fp32.
  3. optim.py: cosine_schedule, clip_global_norm, init_model, build_optimizer, count_parameters.
  4. checkpoint.py: save/load with config.json + meta.json (tokenizer fingerprint + git SHA),
     prune(keep_last), CheckpointMismatch with a readable diff on config mismatch.
  5. sample.py: the hand-written sampler, generate(), streaming callback.
  6. train.py: Trainer with everything listed above.
  7. benchmarks/throughput.py measuring real TPS for nano, micro and small over 50 steps.
  8. The seven test files.
  9. Run the overfit test to completion.

EXIT GATE - paste the complete unedited output
  make test-parity
  pytest tests/test_llm_overfit.py -v
  python benchmarks/throughput.py --preset nano  --steps 50
  python benchmarks/throughput.py --preset micro --steps 50
  python benchmarks/throughput.py --preset small --steps 50
  python -m mmar.cli bench --preset micro
  /usr/bin/time -l python benchmarks/throughput.py --preset micro --steps 20

EVIDENCE REQUIRED IN YOUR RESPONSE
  1. Full `make test-parity` output INCLUDING the max absolute difference printed per test. A
     parity test that only prints "passed" is not evidence - print the actual max|delta|.
  2. The overfit loss curve: step-by-step losses from step 1 to the step where loss < 0.1.
  3. Measured TPS for all three presets next to the [est] bands from docs/00_ANALYSIS.md 5.1
     (nano 30-60k, micro 8-20k, small 3-8k). Then UPDATE docs/00_ANALYSIS.md 5.1, removing the
     [est] markers and replacing them with your measured numbers.
  4. TPS with compile=true vs compile=false for micro - both numbers, to prove the compile is real.
  5. The max-RSS number from /usr/bin/time -l, against the 1.2 GB budget.
  6. Anything you could NOT verify, stated explicitly as UNVERIFIED.

This phase is MILESTONE 2: "I have a transformer with my own tokenizer, my own attention, my own
training loop, and a NumPy oracle proving the maths is right, plus a measured tokens-per-second
number." Both halves of that sentence need evidence.
```

