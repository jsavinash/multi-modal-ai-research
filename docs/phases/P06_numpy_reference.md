# P6 — NumPy Reference Transformer

| | |
|---|---|
| **Goal** | A deliberately naive, readable NumPy implementation of every primitive in the model — the numerical oracle that P7 is tested against |
| **Wall-clock** | 1 evening |
| **Compute** | none (runs on `nano` shapes in milliseconds) |
| **Prerequisites** | P5 gate green (the reference model needs a `ModelConfig`) |
| **Blocks** | P7 (the parity gate cannot exist without this) |
| **Read first** | [analysis §9.1](../00_ANALYSIS.md#91-architecture-decisions-and-their-justification), [§9.4](../00_ANALYSIS.md#94-the-numpy-oracle-why-this-project-has-parity-tests), [docs/02 §6.9](../02_FUNCTION_REQUIREMENTS.md#69-mmar-llmref--the-numpy-oracle) |

---

## Deliverables

```
src/mmar/llm/config.py                       # ModelConfig (shared by ref and MLX paths)
src/mmar/llm/ref/__init__.py  numpy_ops.py  numpy_model.py
tests/test_ref_numpy.py
tests/test_model_shapes.py
```

## Function requirements

[docs/02 §6.2, §6.9](../02_FUNCTION_REQUIREMENTS.md#62-mmar-llmconfigpy).

`numpy_ops.py` implements, in the most obvious way possible:

```python
softmax(x, axis=-1) ; rms_norm(x, w, eps) ; rope(x, positions, theta)
attention(q, k, v, *, mask, n_kv_heads) ; swiglu(x, w_gate, w_up, w_down)
cross_entropy(logits, targets, mask=None)
```

**Reimplementing GQA in `attention` is the point.** The NumPy version must expand `n_kv_heads` to
`n_heads` explicitly with `np.repeat`, because that is the readable formulation, and P7's MLX version
will instead reshape so the grouped dimension broadcasts. If the two disagree, one of them is wrong —
and `test_ref_numpy.py` plus P7's parity tests will say which.

`NumpyTransformer` loads real weights from a checkpoint so parity can be checked on a **trained**
model, not merely on random init. Some bugs only appear once weights are not near zero.

`ModelConfig.n_params()` is the closed-form count from analysis §4.1. `test_model_shapes.py` asserts
it equals the real module count per preset, i.e. **0.99 M (`nano`), 11.0 M (`micro`), 39.0 M (`small`)**
— so the documentation's arithmetic is enforced by a test rather than trusted.

## Forbidden

- Vectorising for speed. Readability is the deliverable; the `nano` preset keeps it fast enough.
- Reusing any MLX code path, even "just to be consistent". Independence is the entire value.
- Skipping the causal mask because "the loss looked fine". Write the mask explicitly and test it.
- Copying a cache/offset optimisation into the reference. It has no cache; that is deliberate.

## Tasks

1. `config.py` — `ModelConfig` with `head_dim`, `residual_scale`, `n_params()`.
2. `ref/numpy_ops.py` — the six primitives, each with a docstring stating the exact formula.
3. `ref/numpy_model.py` — `NumpyTransformer` with pre-norm blocks, GQA, SwiGLU, tied embeddings, and
   `from_checkpoint()`.
4. `test_ref_numpy.py`:
   - `softmax` against a hand-computed 3-element example.
   - `rms_norm` against hand-computed values and against `x/std * w`.
   - **Causality proof:** perturbing token `t` must not change any logit at position `< t`. Assert
     `max|delta| == 0` exactly for those positions.
   - `cross_entropy` against a hand-computed value for a 2-class case.
   - `rope` against a hand-computed rotation for a 2-dimensional head with `theta=10000`.
5. `test_model_shapes.py` — parameter counts per preset vs the analysis table.

## Exit gate

```bash
pytest tests/test_ref_numpy.py tests/test_model_shapes.py -v
```

## Definition of Done

- [ ] Every primitive has a hand-computed test, not a round-trip against itself
- [ ] Causality test proves `max|delta| == 0` exactly for positions before the perturbation
- [ ] `n_params()` matches the analysis table for all three presets; raw output pasted
- [ ] Full forward on `nano` shapes completes in under 200 ms
- [ ] Every formula in `numpy_ops.py` has a docstring stating the mathematical definition, because
      this file is the readable specification the MLX version is judged against

---

## 📋 Copy-paste prompt

```
ROLE
You are a senior ML systems engineer writing a reference implementation whose ONLY job is to
be obviously correct and readable. Speed is irrelevant. Independence is everything.

CONTEXT - read these first
  1. docs/00_ANALYSIS.md  section 9.1 (architecture decisions AND why each was chosen at this
     scale), section 9.4 (why the NumPy oracle exists), section 4.1 (the parameter formula)
  2. docs/02_FUNCTION_REQUIREMENTS.md  section 6.2 (ModelConfig) and 6.9 (the oracle)
  3. Existing code: src/mmar/llm/tokenizer.py

MACHINE
  MacBook Air M1, 8 GB unified memory. python3.11 in .venv. numpy only.

HARD CONSTRAINTS
  - NO MLX anywhere in this phase. Independence is the entire value of the reference: it is the
    oracle that P7's MLX implementation is judged against.
  - Do NOT vectorise for speed. If a loop is clearer than a broadcast, use the loop. The `nano`
    preset (d_model 128, 2 layers) keeps this fast enough for tests.
  - Do NOT copy a KV-cache or offset optimisation into the reference. The reference has no
    cache; that is deliberate, because it is the definition of "no cache".
  - Write the causal mask EXPLICITLY and test it. Do not skip it because "the loss looked fine".
  - ModelConfig.n_params() must implement the closed-form formula from docs/00_ANALYSIS.md
    section 4.1, and test_model_shapes.py must assert it equals the real module count for all
    three presets: 0.99 M (nano), 11.0 M (micro), 39.0 M (small). If your architecture drifts
    from the documented arithmetic, this test must fail.

DELIVERABLES - exactly these paths
  src/mmar/llm/config.py
  src/mmar/llm/ref/__init__.py
  src/mmar/llm/ref/numpy_ops.py
  src/mmar/llm/ref/numpy_model.py
  tests/test_ref_numpy.py
  tests/test_model_shapes.py

KEY REQUIREMENTS
  - numpy_ops.py implements exactly: softmax(x, axis=-1); rms_norm(x, w, eps=1e-5);
    rope(x, positions, theta); attention(q, k, v, *, mask, n_kv_heads);
    swiglu(x, w_gate, w_up, w_down); cross_entropy(logits, targets, mask=None).
    Every function has a docstring stating the mathematical definition - this file IS the
    readable specification that the MLX version is judged against.
  - attention() must expand n_kv_heads to n_heads explicitly with np.repeat. That is the
    readable formulation, and P7 will instead reshape so the grouped dimension broadcasts. If
    the two disagree, one of them is wrong and the parity test will say which.
  - NumpyTransformer: pre-norm blocks, RMSNorm, RoPE, GQA, SwiGLU, tied embeddings,
    residual scale (2*n_layers)**-0.5, weight init std=0.02. from_checkpoint(path, cfg) so
    parity can be checked on a TRAINED model, not just random init.
  - ModelConfig must expose head_dim, residual_scale and n_params().

TESTS THAT MUST EXIST AND MUST BE HAND-COMPUTED (not round-trips against themselves)
  - softmax against an explicit 3-element example you compute by hand in the test.
  - rms_norm against hand-computed values AND against x/std(x)*w.
  - CAUSALITY PROOF: perturb token t, assert every logit at position < t is bit-identical
    (max|delta| == 0, not "small"). This is the test that catches the most dangerous class of
    bug in the whole project.
  - cross_entropy against a hand-computed 2-class value.
  - rope against a hand-computed rotation for a 2-dimensional head with theta=10000.

FORBIDDEN
  - Importing mlx, torch, or any tensor framework.
  - Skipping the causal mask test.
  - Any file containing "# TODO" or "raise NotImplementedError".
  - Reporting a command as passing without pasting its raw output.

TASKS
  1. config.py with ModelConfig including n_params() and residual_scale.
  2. ref/numpy_ops.py - the six primitives with formula docstrings.
  3. ref/numpy_model.py - NumpyTransformer with from_checkpoint().
  4. tests/test_ref_numpy.py with the five hand-computed tests above.
  5. tests/test_model_shapes.py asserting n_params() against 0.99M / 11.0M / 39.0M.

EXIT GATE - paste the complete unedited output
  pytest tests/test_ref_numpy.py tests/test_model_shapes.py -v
  python -c "..."    # time a full forward on nano shapes; must be under 200 ms

EVIDENCE REQUIRED IN YOUR RESPONSE
  1. The full pytest -v output for both files.
  2. The causality test's assertion message, showing max|delta| for both the allowed and
     forbidden positions (forbidden must be exactly 0).
  3. The n_params() numbers for all three presets next to the documented values.
  4. The nano forward-pass wall-clock.
  5. Anything you could NOT verify, stated explicitly as UNVERIFIED.

Before writing code, answer in at most three lines each: what is the exit gate; which command
proves it; the most likely way to get this silently wrong.
```

