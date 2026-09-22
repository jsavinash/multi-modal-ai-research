# 01 — Roadmap: Phase 0 → Phase 12

> **How to use this document.** Each phase has: an objective, prerequisites, deliverables (exact
> file paths), an **exit gate** that is either pass or fail, and a link to its copy-paste agent
> prompt in `docs/phases/`. Work one phase at a time. **Do not start a phase until the previous
> phase's gate passes** — the gates exist because the failure modes on this hardware are silent
> (see §14 of the analysis), and a passing gate is the only proof you are not building on sand.

---

## 1. Phase dependency graph

```
                    ┌───────────────────────────────────────────────┐
   P0 Foundation ──►│  P1 Canonical IR + text/structured ingest      │
  (env, config,     │        │                                       │
   memory guard,    │        ├──► P2 Vision (OCR, image embed, video) │
   CLI, doctor)     │        │                                       │
                    │        └──► P3 Audio (ffmpeg, ASR)             │
                    │                 │                               │
                    │                 ▼                               │
                    │            P4 RAG + Orchestrator  ◄── MILESTONE 1: USEFUL ASSISTANT
                    └───────────────────────────────────────────────┘
                                                 │
             ┌───────────────────────────────────┘
             ▼
   P5 Tokenizer ──► P6 NumPy reference ──► P7 MLX transformer + parity
                                                   │  ◄── MILESTONE 2: VERIFIED TRANSFORMER
                                                   ▼
                                       P8 Pretrain (overnight) ──► P9 SFT
                                                   │  ◄── MILESTONE 3: OUR LLM SPEAKS
                                                   ▼
                                            P10 Multimodal bridge
                                                   │  ◄── MILESTONE 4: OUR MULTIMODAL MODEL
                                                   ▼
                                            P11 Serving ──► P12 Evaluation ◄── MILESTONE 5: CERTIFIED
```

**Parallelisable if you finish early:** P2 and P3 are independent of each other and depend only on
P1. P5, P6 and P7 form a strict chain. Nothing after P4 is on the critical path for Track D.

**The critical path is P8** — the only phase that costs real clock time. Every decision before it is
sized so that P8 is one overnight run rather than a week.

---

## 2. Master phase table

| Phase | Name | Wall-clock | Compute | Key deliverable | Exit gate |
|---|---|---|---|---|---|
| **P0** | Environment & foundation | 1 evening | — | venv + config + memory guard + CLI + `make doctor` | `make doctor` green; `pytest -q -m "not heavy"` green; peak RSS < 250 MB |
| **P1** | Canonical IR + text/structured ingest | 2 evenings | — | `mmar.ir`, `mmar.ingest` (text/code/web/pdf/office/archive) | 20 fixtures → 20 `MediaDoc`s; unknown extension yields `UNKNOWN`, not an exception; zero crashes on a mixed batch |
| **P2** | Vision ingestion | 2 evenings | ~0 | image extractor (OCR + embed), video keyframes | OCR on 5 screenshots readable by eye; image search returns the right image top-1 for 10/10 probes |
| **P3** | Audio ingestion | 1–2 evenings | ~0 | ffmpeg pipeline + ASR + timestamped chunks | WER ≤ 15 % on a 60 s clip; timestamped chunk retrieval works |
| **P4** | Embed + index + RAG **(MILESTONE 1)** | 2 evenings | ~0 | hybrid retriever + `Orchestrator` + Track-D answerer | **Ask 10 questions about your own files; 8 answers correct with correct citations** |
| **P5** | Tokenizer from scratch | 1 evening | minutes | `mmar/llm/tokenizer.py` + trained `tokenizer.json` | Round-trip lossless on 1 MB incl. emoji/CJK; ≥ 3.2 chars/token |
| **P6** | NumPy reference transformer | 1 evening | — | `mmar/llm/ref/*.py` | Softmax/attention match hand-computed values; masks provably causal |
| **P7** | MLX transformer + parity **(MILESTONE 2)** | 2 evenings | minutes | `mmar/llm/model.py`, `train.py`, `sample.py`, `benchmarks/throughput.py` | **`make test-parity` green**; overfit-1-batch loss < 0.1; KV-cache equivalence; **measured TPS recorded** |
| **P8** | Pretrain **(MILESTONE 3 driver)** | 1 evening + 3–8 h unattended | **3–8 h** | `checkpoints/micro-tinystories-220m/` | Smoke val ppl < 25, **then** full run val ppl ≤ 11 and grammatical samples |
| **P9** | Instruction tuning | 1 evening + ~1 h | ~1 h | SFT checkpoint + chat template | Templated held-out prompts ≥ 50 % exact/ROUGE-L; TinyStories ppl regression ≤ 1.3× |
| **P10** | Multimodal bridge **(MILESTONE 4)** | 2 evenings | ~1.5 h | `mmar/llm/bridge/` + `mmar-vl-micro` | Caption ROUGE-L ≥ 0.25 vs teacher; colour/count probes ≥ 60 % of 50 hand-built items |
| **P11** | Serving | 2 evenings | ~0 | FastAPI + SSE + minimal web UI | A full mixed-modality request answered over HTTP with citations and memory telemetry |
| **P12** | Evaluation & certification **(MILESTONE 5)** | 1 evening | ~1 h | `docs/12_RESULTS.md` + `make eval` report | Every SLO in analysis §15 measured and either met or explicitly explained |

**Totals:** ~18–22 evenings of work, **~7–13 h of unattended GPU time**.

---

## 3. Compute budget

| Consumer | Estimate |
|---|---|
| Tokenizer training (P5) | ~2 min (200 MB subset, vocab 4 096) |
| Overfit + parity tests (P7) | ~3 min |
| Throughput benchmark (P7) | ~10 min |
| **P8 smoke run** (40 M tokens) | **~1–1.5 h** |
| **P8 full run** (220 M tokens) | **~3–8 h** — the only overnight |
| P9 SFT (~2 000 steps) | ~30–60 min |
| P10 bridge stages 1–3 | ~1.5 h |
| P12 evaluation sweep | ~1 h |
| **Total** | **~7–13 h** |

Thermal reality: the Air has no fan. Expect the *last* hour of any long run to be ~20 % slower than
the first. The `wall_clock_budget_h` value in each training config already includes that slack.

---

## 4. Milestones

Milestones are the points where the project is *demonstrably* worth continuing. Each is independent:
if you stop after M1 you still have something useful; if you stop after M2 you still learned the
from-scratch transformer properly.

### M1 — A useful local assistant over your own files *(end of P4)*
**Claim:** "I can drop any folder of mixed documents on this and ask questions about it offline, with
citations."
**Demo:** `make ingest IN=~/Documents/research && make serve` → ask 10 questions → 8 correct.
**Why it matters:** the first point where the project pays you back, reachable with zero GPU time.

### M2 — A verified transformer you wrote *(end of P7)*
**Claim:** "I have a transformer with my own tokenizer, my own attention, my own training loop, and a
NumPy oracle proving the maths is right."
**Demo:** `make test-parity` green + `pytest tests/test_llm_overfit.py` (loss < 0.1 on one batch) +
measured tok/s printed.
**Why it matters:** "from scratch" stops being a claim and becomes evidence.

### M3 — An LLM that speaks English that you trained *(end of P9)*
**Claim:** "I pretrained and instruction-tuned an 11 M-param model on this laptop. It writes
grammatical short stories and answers templated questions."
**Demo:** `make chat` with `llm.active: micro` — coherent TinyStories-grade output plus templated
extractive QA.
**Why it matters:** the pretraining run is the hardest gate in the project; after it, the rest is
engineering.

### M4 — A multimodal model you trained *(end of P10)*
**Claim:** "I froze CLIP/SigLIP and Whisper, trained a projector into my own token space, and my own
model now describes images and consumes speech."
**Demo:** upload an image → `mmar-vl-micro` produces a short caption with correct coarse content.
**Why it matters:** the actual "multimodal LLM built from scratch" deliverable.

### M5 — Certified *(end of P12)*
**Claim:** "Every performance claim in the docs is a measurement with the command that produced it."
**Demo:** `docs/12_RESULTS.md` with the SLO table filled in.

---

## 5. Definition of Done (applies to every phase)

A phase is done when **all** of the following are true. Paste this checklist into each phase summary
as evidence; do not merely assert it.

- [ ] **Code** exists at the exact paths in the phase prompt, and imports only from lower layers.
- [ ] **Tests** exist and pass: `make test-fast` green, plus any phase gate (`make test-parity`, …).
- [ ] **Numerical claims are measured, not asserted.** Every number in a doc is either measured here or explicitly marked `[est]`.
- [ ] **Failure paths are tested.** Every new component has a test feeding it garbage that asserts graceful degradation — never an exception escaping the pipeline.
- [ ] **Memory is measured.** Peak RSS for every new code path is recorded; if it exceeds the §4.5 budget, the config default is lowered, never the budget raised.
- [ ] **Docs updated:** the phase prompt's "do not forget" list is executed — README capability matrix, the analysis doc if a design changed, `docs/12_RESULTS.md` for every measurement.
- [ ] **No new dependency** without the rationale written in the phase's notes.
- [ ] **The exit-gate command** was run and its raw output pasted into the phase summary.

---

## 6. What to do when a phase fails

| Symptom | Do this — in this order |
|---|---|
| A gate fails and you do not know why | **Stop and bisect.** Do not proceed and do not "work around" it. Gates are cheap; debugging a downstream mystery caused by skipping one is expensive |
| Loss plateaus and will not fall | Check in order: (1) weight decay applied to norms, (2) LR too high, (3) loss not in fp32, (4) the data shard is actually varying. All four are in analysis §9.3 |
| Samples are gibberish but loss is falling | Almost always the **tokenizer/shard hash mismatch** (R4) or a **causal-mask bug**. Run `pytest -m parity` |
| Memory blows past budget | `make memory`, check the MLX cache, verify `mx.set_cache_limit` was applied, then lower `micro_batch_size`. Never raise `max_process_rss_mb` |
| Throughput collapses mid-run | Thermal (R3). Pause 10 min, resume with `--resume auto --thermal-relief`. Confirm with the tok/s trend |
| A heavy model download stalls | `mmar models pull <name> --verbose`; check `MMAR_MODEL_CACHE` has space; a "hang" is usually a 1–3 GB download |
| You are tempted to skip ahead | Re-read the exit gate for the phase you would skip. If you cannot state how you would *prove* it passed later, you are not ready to skip it |

---

## 7. Explicitly deferred (Phase 13+, documented not planned)

| Idea | Why deferred | What would unblock it |
|---|---|---|
| QLoRA fine-tune of a 0.6 B model on your own chat history | Drifts from "from scratch"; Track D already covers quality | M4 done, plus a labelled preference dataset |
| Streaming microphone ASR | Different concurrency model (ring buffer, partial hypotheses) | Moonshine-base + a producer/consumer refactor of the ASR wrapper |
| Learned query resampler instead of pixel-pooling (analysis §10.3) | Another trained module; pixel-shuffle already solves the token budget | M4 done and caption quality measured as token-limited |
| Document layout understanding (tables, columns, reading order) | Needs a layout model (~0.5 GB) or the Docling pipeline | A measured need from real documents, not a hypothetical one |
| Tuned fusion weights for hybrid retrieval | The 0.35 BM25 weight is a defensible default | A labelled relevance set (≥ 200 query/doc pairs) |
| Continued pretraining on a second corpus (code, or your notes) | Sequential pretraining on 8 GB is slow; measure the payoff first | M3 done and TinyStories quality plateauing |
| GGUF / `llama.cpp` export of `micro` | Single-framework by design (ADR-1) | A concrete need to run the model outside this repo |
| CI on a macOS arm runner (ruff, mypy, `test-fast`, `-m parity`) | Gates are hand-run today, and CI has nothing to run before P7 exists | P7 gate green |

---

## 8. Phase prompt index

Each phase file contains: objective, prerequisites, exact deliverables, inline function
requirements, forbidden actions, ordered tasks, definition of done, and the **copy-paste agent
prompt**.

| Phase | File | Prompt self-contained? |
|---|---|---|
| P0 | [`phases/P00_foundation.md`](phases/P00_foundation.md) | ✅ |
| P1 | [`phases/P01_ir_and_text_ingest.md`](phases/P01_ir_and_text_ingest.md) | ✅ |
| P2 | [`phases/P02_vision_ingest.md`](phases/P02_vision_ingest.md) | ✅ |
| P3 | [`phases/P03_audio_ingest.md`](phases/P03_audio_ingest.md) | ✅ |
| P4 | [`phases/P04_embed_index_rag.md`](phases/P04_embed_index_rag.md) | ✅ |
| P5 | [`phases/P05_tokenizer.md`](phases/P05_tokenizer.md) | ✅ |
| P6 | [`phases/P06_numpy_reference.md`](phases/P06_numpy_reference.md) | ✅ |
| P7 | [`phases/P07_mlx_transformer.md`](phases/P07_mlx_transformer.md) | ✅ |
| P8 | [`phases/P08_pretrain.md`](phases/P08_pretrain.md) | ✅ |
| P9 | [`phases/P09_sft.md`](phases/P09_sft.md) | ✅ |
| P10 | [`phases/P10_multimodal_bridge.md`](phases/P10_multimodal_bridge.md) | ✅ |
| P11 | [`phases/P11_serving.md`](phases/P11_serving.md) | ✅ |
| P12 | [`phases/P12_evaluation.md`](phases/P12_evaluation.md) | ✅ |

**Before pasting any phase prompt, read [`docs/03_PROMPT_PLAYBOOK.md`](03_PROMPT_PLAYBOOK.md).** It
defines the verification protocol that stops an AI agent from reporting success it did not achieve —
on a project with silent failure modes like this one, that is the difference between a working
pipeline and three weeks of plausible-looking progress.

---

## 9. Progress tracker (copy into your issue tracker / a scratch file)

```
[ ] P0  Environment & foundation            gate: make doctor green, RSS < 250 MB
[ ] P1  Canonical IR + text/structured      gate: 20/20 fixtures, 0 crashes on mixed batch
[ ] P2  Vision ingestion                    gate: OCR legible, 10/10 image top-1 retrieval
[ ] P3  Audio ingestion                     gate: WER <= 15% on 60 s clip
[ ] P4  RAG + orchestrator       M1 ★        gate: 8/10 file questions correct + cited
[ ] P5  Tokenizer from scratch              gate: lossless round-trip, >= 3.2 chars/token
[ ] P6  NumPy reference transformer         gate: hand-computed values match, mask causal
[ ] P7  MLX transformer + parity M2 ★        gate: make test-parity green, overfit < 0.1, TPS measured
[ ] P8a Pretrain smoke (40 M tokens)        gate: val ppl < 25
[ ] P8b Pretrain full (220 M tokens) M3 ★    gate: val ppl <= 11, grammatical samples
[ ] P9  Instruction tuning                  gate: >= 50% on templated held-out prompts
[ ] P10 Multimodal bridge        M4 ★        gate: ROUGE-L >= 0.25, probes >= 60%
[ ] P11 Serving                             gate: end-to-end HTTP request with citations
[ ] P12 Evaluation & certification M5 ★      gate: every SLO measured and explained
```
