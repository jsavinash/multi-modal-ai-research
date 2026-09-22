# P12 — Evaluation & Certification (MILESTONE 5)

| | |
|---|---|
| **Goal** | Every performance claim in the documentation is a measurement with the command that produced it. The project is provably complete. |
| **Wall-clock** | 1 evening |
| **Compute** | ~1 h |
| **Prerequisites** | All previous phases green |
| **Blocks** | nothing — this is the last phase |
| **Read first** | [docs/01_ROADMAP.md section 5](../01_ROADMAP.md#5-definition-of-done-applies-to-every-phase), [analysis §15](../00_ANALYSIS.md#15-performance-envelope--slos) |

---

## Deliverables

```
docs/12_RESULTS.md            # Complete SLO table with measured values and commands
src/mmar/eval/__init__.py    harness.py  metrics.py
tests/test_eval.py
artifacts/eval/report.json
```

## What "certified" means

A project is certified when every number in the documentation can be traced to a command whose raw output is in `docs/12_RESULTS.md` or linked from it. Certification does not mean every SLO is met — it means every SLO is measured, and every miss is explained.

## SLO table

Every row must be filled. `[est]` markers in earlier docs must be replaced with measured values or deleted.

| SLO | Target | Measured | Command | Status |
|---|---|---|---|---|
| **Import RSS** | < 250 MB | _____ MB | `/usr/bin/time -l python -c "import mmar"` | ⬜ |
| **Peak RSS — text ingest** | < 400 MB | _____ MB | `/usr/bin/time -l mmar ingest tests/fixtures` | ⬜ |
| **Peak RSS — image ingest** | < 1.2 GB | _____ GB | `/usr/bin/time -l mmar ingest tests/fixtures/images` | ⬜ |
| **Peak RSS — audio ingest** | < 1.2 GB | _____ GB | `/usr/bin/time -l mmar ingest tests/fixtures/audio` | ⬜ |
| **Peak RSS — chat** | < 1.5 GB | _____ GB | `/usr/bin/time -l mmar chat --once "hi"` | ⬜ |
| **Peak RSS — bridge train** | < 1.2 GB | _____ GB | `/usr/bin/time -l mmar llm bridge train ...` | ⬜ |
| **Pretrain TPS (micro)** | 8 000–20 000 tok/s | _____ tok/s | `python -m mmar.llm.train --config configs/train_micro.yaml --benchmark` | ⬜ |
| **Pretrain val ppl (smoke)** | < 25 | _____ | `mmar eval --tasks ppl --checkpoint checkpoints/micro-smoke` | ⬜ |
| **Pretrain val ppl (full)** | ≤ 11 | _____ | `mmar eval --tasks ppl --checkpoint checkpoints/micro-tinystories-220m` | ⬜ |
| **SFT val ppl** | ≤ 1.3× P8 | _____ | `mmar eval --tasks ppl --checkpoint checkpoints/micro-sft` | ⬜ |
| **SFT exact/ROUGE-L** | ≥ 50 % | _____ % | `mmar eval --tasks sft --checkpoint checkpoints/micro-sft` | ⬜ |
| **Bridge caption ROUGE-L** | ≥ 0.25 | _____ | `mmar eval --tasks caption --checkpoint checkpoints/micro-vl-stage2` | ⬜ |
| **Bridge VQA accuracy** | ≥ 25 % | _____ % | `mmar eval --tasks vqa --checkpoint checkpoints/micro-vl-stage2` | ⬜ |
| **Bridge probe accuracy** | ≥ 60 % | _____ % | `mmar eval --tasks probe --checkpoint checkpoints/micro-vl-stage2` | ⬜ |
| **Audio WER (60 s clip)** | ≤ 15 % | _____ % | `mmar eval --tasks asr` | ⬜ |
| **Retrieval recall@10** | ≥ 0.8 | _____ | `mmar eval --tasks recall` | ⬜ |
| **Embedding cache hit rate** | ≥ 95 % | _____ % | `mmar eval --tasks cache_hit` | ⬜ |
| **Tokenizer compression** | ≥ 3.2 chars/token | _____ | `mmar eval --tasks tokenizer` | ⬜ |
| **MemoryGuard enforcement** | Never exceeds budget | _____ | `mmar eval --tasks memory_pressure` | ⬜ |
| **Ingest throughput (text)** | ≥ 50 files/s | _____ /s | `mmar eval --tasks ingest_throughput` | ⬜ |
| **PDF text page** | ≤ 120 ms/page | _____ ms | `mmar eval --tasks pdf_latency` | ⬜ |
| **PDF OCR page** | ≤ 900 ms/page | _____ ms | `mmar eval --tasks pdf_latency` | ⬜ |
| **Image OCR + embed** | ≤ 1.2 s/image | _____ s | `mmar eval --tasks image_latency` | ⬜ |
| **Audio ingest (1 h)** | ≤ 6 min | _____ min | `mmar eval --tasks audio_latency` | ⬜ |
| **Retrieval p95** | ≤ 60 ms | _____ ms | `mmar eval --tasks retrieval_latency` | ⬜ |
| **Chat TTFT (`micro`)** | ≤ 400 ms | _____ ms | `mmar eval --tasks ttft` | ⬜ |
| **Chat decode (`micro`)** | ≥ 60 tok/s | _____ tok/s | `mmar eval --tasks decode_tps` | ⬜ |
| **Chat decode (instruct)** | ≥ 20 tok/s | _____ tok/s | `mmar eval --tasks decode_tps` | ⬜ |
| **Vision caption** | ≤ 6 s/image | _____ s | `mmar eval --tasks caption_latency` | ⬜ |
| **Peak RSS — pretrain** | < 1.2 GB | _____ GB | `mmar eval --tasks train_rss` | ⬜ |
| **Idle server RSS** | < 250 MB | _____ MB | `mmar eval --tasks idle_rss` (5 min idle) | ⬜ |
| **Cold start → first token** | ≤ 5 s | _____ s | `mmar eval --tasks cold_start` | ⬜ |
| **Peak RSS — any request** | < 2.5 GB | _____ GB | `mmar eval --tasks request_rss` | ⬜ |

## Evaluation harness

```python
# src/mmar/eval/harness.py
class EvalHarness:
    def __init__(self, settings: Settings, store: MediaStore) -> None: ...
    def run(self, tasks: Sequence[str], *, checkpoint: Path | None = None) -> list[TaskReport]: ...
    def task_ppl(self, checkpoint: Path) -> float: ...
    def task_sft(self, checkpoint: Path) -> float: ...
    def task_caption(self, checkpoint: Path) -> float: ...
    def task_vqa(self, checkpoint: Path) -> float: ...
    def task_probe(self, checkpoint: Path) -> float: ...
    def task_asr(self) -> float: ...
    def task_recall(self) -> float: ...
    def task_cache_hit(self) -> float: ...
    def task_tokenizer(self) -> float: ...
    def task_memory_pressure(self) -> dict[str, float]: ...
    def task_ingest_throughput(self) -> float: ...
    def task_pdf_latency(self) -> float: ...
    def task_image_latency(self) -> float: ...
    def task_audio_latency(self) -> float: ...
    def task_retrieval_latency(self) -> float: ...
    def task_ttft(self, checkpoint: Path) -> float: ...
    def task_decode_tps(self, checkpoint: Path) -> float: ...
    def task_caption_latency(self, checkpoint: Path) -> float: ...
    def task_train_rss(self) -> float: ...
    def task_idle_rss(self) -> float: ...
    def task_cold_start(self) -> float: ...
    def task_request_rss(self) -> float: ...

@dataclass(slots=True, frozen=True)
class TaskReport:                 # one per task; distinct from docs/02 section 9 EvalReport
    task: str; value: float; target: float | None; unit: str
    passed: bool; command: str; raw_output: str
```

**Contract:** `TaskReport` (one per task, this file) carries the command and the raw output; the aggregate `EvalReport` (docs/02 §9) is built from a sequence of `TaskReport` objects. `docs/12_RESULTS.md` is generated from them, not written by hand. This is how the document stays honest.

## What to do when a target is missed

| Miss | Action |
|---|---|
| Pretrain val ppl > 11 | Check LR, data shard, init. Do not declare P8 done. |
| SFT exact/ROUGE-L < 50 % | Check loss mask, forgetting ratio, holdout set. Do not declare P9 done. |
| Bridge caption ROUGE-L < 0.25 | Check pool_factor, dataset quality, stage-1 convergence. Report honestly. |
| Any RSS > budget | Lower `micro_batch_size` or `max_concurrent_models`. Never raise the budget. |
| Throughput < 8 000 tok/s | Check thermal relief, batch size, compilation. Report measured value. |

**Never rerun a measurement until it passes.** The honest report says "target was X, measured Y, most likely cause is Z".

## Forbidden

- Estimating any number. Every cell in the SLO table must have a command and a raw output.
- Deleting a row because the measurement is embarrassing. A miss is evidence; hide it and the document is worthless.
- Running `make eval` once and trusting it. Each task must be run at least twice and the values must be within measurement noise.
- Declaring the project complete without reading every pasted output.

## Tasks

1. `src/mmar/eval/harness.py` with all task methods above. Each task writes its raw command output into `TaskReport.raw_output`.
2. `src/mmar/eval/metrics.py`: `compute_wer`, `compute_rouge_l`, `compute_exact_match`, `compute_recall@k`, `compute_cache_hit_rate`.
3. `tests/test_eval.py`: each metric tested against hand-computed values; the harness tested with a fake store.
4. Run every task in the SLO table. For each, paste the raw command output into `docs/12_RESULTS.md`.
5. For every miss, write a one-paragraph explanation: most likely cause, what you would change, whether it is a code bug or a design trade-off.
6. Verify the README's claims section against the measured values. Update the README if any number changed.
7. Run `tests/test_architecture.py` one final time. If it fails, the layer rule was violated somewhere and the phase is not done.

## Exit gate

```bash
make test-fast
python -m mmar.cli eval --out artifacts/eval/report.json   # writes report.json + summary box
pytest tests/test_architecture.py -v
```

**The gate passes when `docs/12_RESULTS.md` has no empty "Measured" cells and every miss has an explanation.**

## Definition of Done

- [ ] `docs/12_RESULTS.md` has 33 rows, all "Measured" cells filled, all misses explained
- [ ] `artifacts/eval/report.json` exists and is valid JSON with the same data
- [ ] `tests/test_architecture.py` green — the layer rule holds end to end
- [ ] README's claims section matches the measured values
- [ ] Every measurement in the SLO table has a command and raw output linked or inlined
- [ ] You have read every raw output and confirmed it matches the documented claim

---

## 📋 Copy-paste prompt

```
ROLE
You are a senior ML systems engineer. You write code that is measured, not asserted.

CONTEXT - read these first, in this order
  1. docs/00_ANALYSIS.md  section 15 (SLOs)
  2. docs/01_ROADMAP.md  section 5 (Definition of Done) and MILESTONE 5
  3. docs/02_FUNCTION_REQUIREMENTS.md  as needed for metric implementations
  4. Existing code: src/mmar/eval/ (if any), src/mmar/llm/eval_sft.py (P9),
     src/mmar/utils/memory.py

MACHINE
  MacBook Air M1, 8 GB unified memory, fanless. python3.11 in .venv.

HARD CONSTRAINTS
  - Every number in docs/12_RESULTS.md must come from a command whose raw output is pasted
    or linked from that file. No exceptions.
  - If a target is missed, write a one-paragraph explanation: most likely cause, what you
    would change, whether it is a bug or a trade-off. Do not rerun until it passes.
  - tests/test_architecture.py must be green. The layer rule (no higher-layer imports from
    lower layers) is the final integrity check of the whole project.
  - README claims must match measured values. If a number changed, update the README.

DELIVERABLES - exactly these paths
  src/mmar/eval/__init__.py
  src/mmar/eval/harness.py
  src/mmar/eval/metrics.py
  tests/test_eval.py
  artifacts/eval/report.json
  docs/12_RESULTS.md

KEY REQUIREMENTS
  - EvalHarness.run(tasks) returns a list of TaskReport, one per task. Each TaskReport
    carries: task name, measured value, target, unit, passed boolean, command string,
    raw_output string.
  - Metrics: compute_wer (jiwer), compute_rouge_l (rouge-score), compute_exact_match,
    compute_recall@k, compute_cache_hit_rate. All deterministic, all tested.
  - task_ppl: load the checkpoint, run evaluate() on the val shard, return val_loss and
    derived perplexity. Run twice and assert the two values are within 0.1.
  - task_sft: run eval_sft.py on the holdout set; return exact-match and ROUGE-L.
  - task_memory_pressure: run peak_tracker around a representative sequence (import,
    ingest, chat) and return {phase: peak_mb} dict.
  - docs/12_RESULTS.md: a Markdown table with all 33 SLO rows (the union of analysis section 15
    and this file's table). Each row has: SLO name,
    target, measured value, unit, pass/fail, command, raw output (inlined or linked).
    Misses have an "Explanation" paragraph below the table.
  - artifacts/eval/report.json: the same data as the Markdown table, as JSON.

FORBIDDEN
  - Any cell in the SLO table left blank or marked "TBD".
  - Rerunning a measurement until it passes. Report the first honest measurement.
  - Any file containing "# TODO" or "raise NotImplementedError".
  - Reporting a command as passing without pasting its raw output.

TASKS
  1. src/mmar/eval/metrics.py with all metric functions and hand-computed unit tests.
  2. src/mmar/eval/harness.py with all task methods and the TaskReport dataclass.
  3. tests/test_eval.py covering every metric and the harness with a fake store.
  4. Run every task in the SLO table. For each, capture raw output and write the row.
  5. For every miss, write the explanation paragraph.
  6. Cross-check README claims against measured values; update README if needed.
  7. Run tests/test_architecture.py one final time.
  8. Commit with message "P12: evaluation - certified".

EXIT GATE - paste the complete unedited output
  make test-fast
  python -m mmar.cli eval --out artifacts/eval/report.json  # writes report.json + summary box
  pytest tests/test_architecture.py -v
  git diff --stat

EVIDENCE REQUIRED IN YOUR RESPONSE
  1. make test-fast raw output.
  2. The eval report JSON (full contents).
  3. The rendered docs/12_RESULTS.md (full contents).
  4. test_architecture.py output.
  5. git diff --stat showing what changed in this phase.
  6. A list of every SLO that missed target, with the explanation paragraph verbatim.
  7. Anything you could NOT verify, stated explicitly as UNVERIFIED.

Before writing code, answer in at most three lines each: what is the exit gate; which command proves it; the most likely way to get this silently wrong.
