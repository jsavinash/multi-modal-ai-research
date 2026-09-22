# P0 — Environment & Foundation

| | |
|---|---|
| **Goal** | A reproducible 3.11 environment, a validated config system, the memory guard, the CLI skeleton and a `doctor` gate that proves the machine is ready |
| **Wall-clock** | 1 evening |
| **Compute** | none |
| **Prerequisites** | none — this is the first phase |
| **Blocks** | every subsequent phase |
| **Read first** | [analysis §3.3](../00_ANALYSIS.md#33-environment-gaps-to-close-in-phase-0), [§4](../00_ANALYSIS.md#4-memory-budget-arithmetic), [§12.3](../00_ANALYSIS.md#123-residency-policy-modelregistry--memoryguard) |

---

## Deliverables (exact paths)

```
pyproject.toml  Makefile  .python-version  .gitignore  LICENSE
src/mmar/__init__.py
configs/{default,train_smoke,train_micro,train_small}.yaml
src/mmar/errors.py
src/mmar/config.py
src/mmar/cli.py
src/mmar/utils/__init__.py  memory.py  io.py  sniff.py  log.py
tests/conftest.py  test_config.py  test_memory.py  test_sniff.py  test_architecture.py
docs/12_RESULTS.md
```

## Function requirements

Full signatures: [`docs/02_FUNCTION_REQUIREMENTS.md §0, §1`](../02_FUNCTION_REQUIREMENTS.md#0-conventions).
The non-negotiable behaviours:

- `load_settings()` resolves the `extends:` chain in `configs/*.yaml`, applies `MMAR_*` env vars, then
  explicit overrides — in that precedence order — and rejects unknown keys.
- `MemoryGuard.enforce(estimate_mb, policy)` raises `MemoryBudgetExceeded` when a load would exceed
  `runtime.max_process_rss_mb`. It must account for **already-registered** handles, not just the new one.
- `configure_mlx_limits()` must call `mx.set_cache_limit` **and** `mx.set_memory_limit`. The cache limit
  is the one people forget and the one that matters (analysis §12.3, point 5).
- `sniff_mime()` reads magic bytes; a `.txt` file containing PDF bytes must return `application/pdf`.
- `test_architecture.py` parses imports with `ast` and asserts no lower layer imports from a higher one
  (analysis §6.1). Write it now and it guards every future phase for free.
- `mmar info`: CPU/GPU cores, total and available RAM, disk free, python version, MLX version and Metal
  availability, `ffmpeg` presence. Under 2 seconds.
- `mmar doctor`: pass/fail table, exit code 1 if anything fails.

## Forbidden

- Creating placeholder packages for future phases (`mmar/ingest/__init__.py` with nothing in it, etc.).
  P0 delivers the foundation only; later phases create their own packages.
- Importing `mlx` anywhere reachable from `import mmar` or `import mmar.cli`.
- New dependencies beyond what `pyproject.toml` already declares.
- Writing to `data/`, `artifacts/` or `checkpoints/` except to create the directories.

## Dependency manifest (create `pyproject.toml` with exactly these)

| Group | Packages |
|---|---|
| core | `numpy`, `pydantic`, `pydantic-settings`, `pyyaml`, `typer`, `rich` |
| ingest | `trafilatura`, `pypdfium2`, `openpyxl`, `python-docx`, `python-pptx`, `ebooklib` |
| vision | `ocrmac`, `pyobjc-framework-Vision` |
| models | `mlx`, `mlx-lm`, `mlx-embeddings`, `mlx-whisper`, `mlx-vlm` |
| eval | `jiwer`, `rouge-score` |
| serve | `fastapi`, `uvicorn`, `sse-starlette`, `python-multipart`, `jinja2` |
| `[dev]` extra | `pytest`, `ruff`, `mypy`, `httpx` |

No phase after P0 may add a dependency without editing this table **and** `pyproject.toml` in the
same commit (R7: pin `mlx*` packages to minor versions). pytest markers registered: `parity`,
`slow`, `heavy` (docs/02 §10 marker policy).

## Tasks

1. Create the bootstrap files first — `LICENSE` (MIT) and `.gitignore` already exist; **create
   the rest**: `pyproject.toml` (the manifest above + pytest markers `parity`/`slow`/`heavy` +
   ruff and mypy config), `Makefile` (GNU make, tab-indented; every target any phase gate calls:
   `doctor info test-fast test-parity memory data-tinystories tokenize ingest index chat serve
   eval bench train-smoke train-micro train-sft`), `.python-version` (`3.11`),
   `src/mmar/__init__.py`, and the four `configs/*.yaml` wired with the `extends:` chain.
2. `uv venv --python 3.11 .venv && uv pip install -e ".[dev]"`. Record the resolved `mlx` version.
3. Ensure `ffmpeg` (`brew install ffmpeg` if absent); record `ffmpeg -version | head -1`.
4. Implement `errors.py`, then `utils/io.py`, `utils/log.py`, `utils/sniff.py`, `utils/memory.py`.
5. Implement `config.py` with the full validation list from docs/02 §1.1.
6. Implement `cli.py` with **only** `info` and `doctor`.
7. Write the four test files.
8. Run the gate, then write `docs/12_RESULTS.md` with the P0 measurements.

## Exit gate

```bash
make doctor                       # all checks PASS, exit code 0
make test-fast                    # green, < 30 s, no network
python -c "import mmar; print(mmar.__version__)"
/usr/bin/time -l .venv/bin/python -c "import mmar, mmar.config"   # report max RSS
```

## Definition of Done

- [ ] `make doctor` exits 0 with a fully green table; raw output pasted
- [ ] `make test-fast` green; raw output pasted
- [ ] `pytest tests/test_architecture.py -v` green
- [ ] Peak RSS of a bare import measured and reported
- [ ] `docs/12_RESULTS.md` created with: python version, MLX version, Metal availability, `ffmpeg`
      presence, total/available RAM, import RSS
- [ ] The README Quickstart commands verified to work exactly as written

---

## 📋 Copy-paste prompt

```
ROLE
You are a senior ML systems engineer. You write code that is measured, not asserted.

CONTEXT - read these before writing anything
  1. docs/00_ANALYSIS.md  sections 3.3, 4, 6.1, 12.3
  2. docs/02_FUNCTION_REQUIREMENTS.md  sections 0, 0.1, 1.1-1.5
  3. docs/01_ROADMAP.md  section 5 (Definition of Done)
  4. Bootstrap files: LICENSE and .gitignore already exist; you CREATE the rest in task 1 -
     pyproject.toml, Makefile, .python-version, src/mmar/__init__.py, configs/default.yaml,
     configs/train_micro.yaml, configs/train_smoke.yaml, configs/train_small.yaml

MACHINE
  MacBook Air M1, 8 GB unified memory, fanless, macOS 26.x. python3.11 via uv.
  GNU Make 3.81 (recipes must be tab-indented). ffmpeg is NOT installed.
  The default `python3` is 3.14 and has no MLX wheel - use .venv with 3.11.

HARD CONSTRAINTS
  - Peak RSS of `import mmar` plus `import mmar.config` must be UNDER 250 MB. Measure it
    with /usr/bin/time -l. Do not estimate.
  - pyproject.toml is created in task 1 with EXACTLY the DEPENDENCIES block below. After
    task 1 no phase may add a package - the manifest is closed.
  - Do not create placeholder packages for phases 1-12. P0 is foundation only.
  - Do not import mlx anywhere reachable from `import mmar` or `import mmar.cli`.

DELIVERABLES - exactly these paths
  pyproject.toml
  Makefile
  .python-version
  .gitignore
  LICENSE
  src/mmar/__init__.py
  configs/default.yaml
  configs/train_smoke.yaml
  configs/train_micro.yaml
  configs/train_small.yaml
  src/mmar/errors.py
  src/mmar/config.py
  src/mmar/cli.py
  src/mmar/utils/__init__.py
  src/mmar/utils/memory.py
  src/mmar/utils/io.py
  src/mmar/utils/sniff.py
  src/mmar/utils/log.py
  tests/conftest.py
  tests/test_config.py
  tests/test_memory.py
  tests/test_sniff.py
  tests/test_architecture.py
  docs/12_RESULTS.md

KEY REQUIREMENTS
  - load_settings: resolves `extends:` in configs/*.yaml (depth <= 3, cycle detection),
    then MMAR_* env vars, then explicit kwargs - exactly that precedence. Rejects unknown
    keys. Every validation in docs/02 section 1.1 raises pydantic ValidationError with a
    readable message, never a bare KeyError.
  - MemoryGuard.enforce(estimate_mb, policy): must account for models ALREADY registered,
    not only the candidate. Raises MemoryBudgetExceeded.
  - configure_mlx_limits(): must call BOTH mx.set_cache_limit and mx.set_memory_limit.
    Idempotent. If mlx is absent, degrade with a warning - do not crash.
  - sniff_mime(): magic bytes beat extension. A file named notes.txt containing PDF bytes
    must sniff as application/pdf. Unknown -> "application/octet-stream", never raise.
  - mmar info: cpu+GPU cores, total+available RAM, disk free, python version, MLX version,
    Metal availability, ffmpeg presence and version. Under 2 seconds, no model allocation.
  - mmar doctor: pass/fail table, exit code 1 if anything fails.
  - test_architecture.py: parse every module under src/mmar with ast and assert the layer
    rule from docs/00_ANALYSIS.md section 6.1. It will have little to check now; write it
    now so every later phase is covered automatically.

DEPENDENCIES - pyproject.toml declares exactly this manifest, no other package anywhere
  core:    numpy pydantic pydantic-settings pyyaml typer rich
  ingest:  trafilatura pypdfium2 openpyxl python-docx python-pptx ebooklib
  vision:  ocrmac pyobjc-framework-Vision
  models:  mlx mlx-lm mlx-embeddings mlx-whisper mlx-vlm
  eval:    jiwer rouge-score
  serve:   fastapi uvicorn sse-starlette python-multipart jinja2
  [dev]:   pytest ruff mypy httpx
  Also in pyproject: pytest markers parity, slow, heavy (docs/02 section 10);
  ruff + mypy config; pin mlx* packages to minor versions (risk R7).

FORBIDDEN
  - Raising runtime.max_process_rss_mb in configs/default.yaml to make something fit.
  - `except Exception: pass`. Every except must re-raise as an MmarError subclass or log at
    WARNING with the exception text and an actionable hint.
  - Any file containing "# TODO" or "raise NotImplementedError".
  - Reporting a command as passing without pasting its raw output.

TASKS
  1. Create the bootstrap files (LICENSE and .gitignore already exist - keep them):
     pyproject.toml with the DEPENDENCIES block above + pytest markers parity/slow/heavy +
     ruff + mypy config; Makefile (tab-indented) with targets: doctor info test-fast
     test-parity memory data-tinystories tokenize ingest index chat serve eval bench
     train-smoke train-micro train-sft; .python-version ("3.11"); src/mmar/__init__.py;
     configs/default.yaml + train_smoke + train_micro + train_small wired with extends:.
  2. uv venv --python 3.11 .venv ; uv pip install -e ".[dev]"
     Record: python version, mlx version, and pip freeze | grep -E "mlx|numpy|pydantic".
  3. Ensure ffmpeg (brew install ffmpeg if absent). Record ffmpeg -version | head -1.
  4. Implement errors.py with the 8-class hierarchy from docs/02 section 0.1.
  5. Implement utils/io.py, utils/log.py, utils/sniff.py, utils/memory.py.
  6. Implement config.py with the full validation list.
  7. Implement cli.py with only `info` and `doctor`.
  8. Write the four test files.
  9. Run the exit gate, then write docs/12_RESULTS.md with the measurements.

EXIT GATE - run these and paste the complete unedited output
  make doctor
  make test-fast
  python -c "import mmar; print(mmar.__version__)"
  /usr/bin/time -l .venv/bin/python -c "import mmar, mmar.config"

EVIDENCE REQUIRED IN YOUR RESPONSE
  1. Raw output of make doctor (the whole table).
  2. Raw output of make test-fast.
  3. The /usr/bin/time -l line showing maximum resident set size, with the number in MB.
  4. The contents of docs/12_RESULTS.md.
  5. Anything you could NOT verify, stated explicitly as UNVERIFIED.

Before writing code, answer in at most three lines each: what is the exit gate; which
command proves it; what is the most likely way to get this silently wrong.
```

