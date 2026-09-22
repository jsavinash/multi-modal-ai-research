# mmar — Multi-Modal AI Research (local, from scratch, 8 GB M1 Air)

A fully local, fully offline pipeline that ingests **any input format**, normalises it into
**one canonical representation**, and reasons over it using a **transformer language model
whose tokenizer, attention, training loop and KV cache are written from scratch** in Apple MLX.

> **Status:** 📋 *Documentation-first repository.* Phases 0–12 are fully specified but not yet
> implemented. Start at [`docs/01_ROADMAP.md`](docs/01_ROADMAP.md) then execute
> [`docs/phases/P00_foundation.md`](docs/phases/P00_foundation.md).

---

## Documentation map

| Doc | What it answers |
|---|---|
| [`docs/00_ANALYSIS.md`](docs/00_ANALYSIS.md) | **System design + deep analysis.** Hardware reality check, memory arithmetic, architecture, model-selection matrix, ADRs, risks, SLOs |
| [`docs/01_ROADMAP.md`](docs/01_ROADMAP.md) | **The roadmap.** Phases 0–12, dependencies, deliverables, compute/time budget, exit criteria |
| [`docs/02_FUNCTION_REQUIREMENTS.md`](docs/02_FUNCTION_REQUIREMENTS.md) | **Function requirements.** Every module's classes, signatures, contracts, invariants, error taxonomy |
| [`docs/03_PROMPT_PLAYBOOK.md`](docs/03_PROMPT_PLAYBOOK.md) | How to drive an AI coding agent through the phases without it lying to you |
| [`docs/phases/P00…P12.md`](docs/phases/) | **Copy-paste agent prompt per phase**, with tasks, DoD and tests |

---

## What this project actually is (read this before the hype)

There are two different things people mean by "build a multimodal LLM from scratch". This
project does **both**, in four clearly separated tracks (A–D), because conflating them is how
these projects die.

### Track A — `mmar.llm`: a genuinely from-scratch transformer (text → text)

Tokenizer (BPE, trained on your corpus), RoPE, RMSNorm, SwiGLU, GQA, KV cache, AdamW training
loop, checkpointing, sampling — **all implemented in this repo**. Sizes are honest:

| Preset | Params | Trains on M1 Air in | What it can do |
|---|---|---|---|
| `nano` | 1.0 M | ~10 s | Numeric parity tests only |
| `micro` smoke | 11 M | ~1–1.5 h | Pipeline validation; val ppl 15–25 |
| **`micro` full** | **11 M** | **~3–8 h (overnight)** | ✅ **The run to do.** Val ppl 7–11 |
| `small` | 39 M | ~70 h | ❌ Skip — see §5.2 of the analysis |

A ~1.8 M-param GPT trained on ~20 M TinyStories tokens reaches **~9.6 perplexity** and emits
grammatical prose (Goedecke, 2025). A 53 M-param model on a 16 GB M2 Pro trained at
**27k tok/s** (nanoGPT-on-MLX). Those are the real reference points this roadmap is calibrated
against — not "a local GPT-4".

> **Counter-intuitive finding from the sizing arithmetic:** a 39 M-param model needs ~780 M tokens
> to be Chinchilla-optimal, i.e. ~70 h of GPU on this machine. **Staying at 11 M and training it
> properly beats going bigger.** See [`docs/00_ANALYSIS.md §5.2`](docs/00_ANALYSIS.md#52-chinchilla-check--and-why-the-small-preset-is-a-trap).


### Track B — `mmar.perception`: make *any* format look like text

Images, audio, video, PDF, Office, HTML, spreadsheets, code, archives, URLs. Each becomes a
`MediaDoc` (canonical IR) containing extracted text + embeds + provenance. No branching on file
type anywhere downstream.

### Track C — `mmar.bridge`: the multimodal model (Track A ⊗ Track B)

LLaVA-style: **frozen** pretrained encoders (SigLIP/CLIP for vision, Whisper/Moonshine for
audio) are projected through a trained MLP into **our own from-scratch LLM's** embedding space,
then instruction-tuned. This is how real VLMs are built, and it is the only path to
"multimodal" that fits in 8 GB.

### Track D — `mmar.serve`: something genuinely useful today

A RAG assistant that ingests your files and answers questions with citations, using a small
pretrained instruct model (Qwen3-0.6B / SmolLM2-360M via MLX) as the reasoner. This works on
day one and is what you will actually use while Track A/C trains.

> **Honest claim:** Track A/C prove you own the stack end-to-end. Track D is what is useful.
> Neither is a frontier model, and this project never pretends otherwise.

---

## Why an 8 GB M1 Air changes the design

| Constraint | Measured / derived | Consequence |
|---|---|---|
| Unified memory | 8 GB shared, ~3.0–3.8 GB consumed by macOS + editor | Hard ML budget ≈ **4.6 GB peak**, sustainable ≈ 3.6 GB |
| Memory bandwidth | **68.25 GB/s** (M1) vs 200 GB/s (M2 Pro) | Decode speed ≈ bandwidth ÷ weight bytes. 4-bit 1.6 B → ~25–40 tok/s real |
| GPU | 8 cores, ~2.6 TFLOPS fp32 / ~5.2 TFLOPS bf16 | Sets the *training* ceiling: ~30 M params is the sweet spot |
| Cooling | **Fanless** | Sustained throughput ≈ 70–80 % of burst; train in cycles, not marathons |
| Python | System `python3` is **3.14** — no MLX/PyObjC wheels | Must pin **3.11** via `.python-version` + `uv` |

**The single most important architectural rule:** `max_concurrent_models: 1`. Never hold a VLM
and an ASR model resident at once. `MemoryGuard` refuses loads that would exceed the budget, and
`ModelRegistry` unloads idle heavy handles after 120 s.

---

## Format coverage

Legend: ✅ native text extraction · 🔤 OCR/text-in-image · 🧠 needs a neural model · 🚫 out of
scope (see [`docs/00_ANALYSIS.md`](docs/00_ANALYSIS.md) §16)

| Category | Extensions | Strategy | Phase |
|---|---|---|---|
| Plain text / markup | `.txt .md .rst .log .ini .tex` | decode + normalise | 1 |
| Delimited / structured | `.csv .tsv .json .jsonl .ndjson .yaml .toml .xml` | typed parse → schema + rows as text | 1 |
| Code | `.py .js .ts .go .rs .java .c .cpp .swift .sql …` | AST/regex-aware chunker | 1 |
| Web | `.html .htm .mhtml`, `http(s)://` | trafilatura → main-content text | 1 |
| PDF | `.pdf` | per-page text; image-only pages fall back to OCR | 1 → 2 |
| Office | `.docx .pptx .xlsx .xls .odt .odp .ods .epub` | python-docx/pptx/openpyxl/ebooklib | 1 |
| Images | `.png .jpg .jpeg .webp .bmp .tif .heic .gif` | Apple Vision OCR + SigLIP/CLIP embed (+ optional VLM caption) | 2 |
| Audio | `.wav .mp3 .m4a .flac .ogg .aac .opus` | ffmpeg → 16 kHz mono → Moonshine/Whisper | 3 |
| Video | `.mp4 .mov .mkv .webm .avi` | ffmpeg keyframes (scene-detect) + audio track, **sequentially** | 3 |
| Archives | `.zip .tar .gz .7z` | bounded recursive expansion | 1 → 3 |
| Unknown | anything else | `MediaDoc(modality=UNKNOWN, errors=[...])`, never a crash | 1 |

Adding a format = **one file + one registry decorator**. See
[`docs/02_FUNCTION_REQUIREMENTS.md`](docs/02_FUNCTION_REQUIREMENTS.md#extractor-registry).

---

## Quickstart (once Phase 0 is executed)

```bash
brew install uv ffmpeg
uv venv --python 3.11 .venv
uv pip install -e ".[dev]"
make doctor          # pass/fail: python, mlx, metal, ffmpeg, RAM headroom
make info            # runtime + memory profile of THIS machine
make test-fast       # inner dev loop, no model downloads
make ingest IN=~/Documents/some-mixed-folder
make chat
make serve           # http://127.0.0.1:8000
```

## Repository layout (target)

```
mmar/
├── README.md                  docs/00_ANALYSIS.md      docs/01_ROADMAP.md
├── pyproject.toml             docs/02_FUNCTION_REQUIREMENTS.md
├── Makefile                   docs/03_PROMPT_PLAYBOOK.md   docs/phases/P00…P12.md
├── configs/{default,train_micro,train_small}.yaml
├── src/mmar/
│   ├── config.py  utils/  ir/            # settings, memory guard, canonical IR
│   ├── ingest/                           # one extractor per format family
│   ├── models/                           # ModelRegistry + inference wrappers
│   ├── rag/                              # dense + BM25 hybrid retrieval
│   ├── llm/                              # ← FROM SCRATCH: tokenizer, model, train, sample
│   │   ├── ref/                          # NumPy oracle for parity tests
│   │   └── bridge/                       # projectors + multimodal training
│   ├── serve/                            # FastAPI app, orchestrator, web UI
│   └── eval/                             # metrics + benchmark harness
└── tests/                                # unit + parity + integration
```

## Time budget

| Track | Calendar time (evenings) | Compute time |
|---|---|---|
| Phase 0–4 (ingest → RAG, useful assistant) | ~7–8 evenings | ~0 |
| Phase 5–7 (tokenizer → from-scratch transformer, verified) | ~4 evenings | minutes |
| Phase 8–9 (pretrain + instruction-tune) | ~2 evenings | **~5–10 h** (unattended) |
| Phase 10–12 (multimodal bridge, serving, eval) | ~5 evenings | ~2.5–3 h |

**Total: ~18–22 evenings (~4 weeks), ~7–13 h of GPU time.** Every phase has an exit gate; no phase
requires more than one sitting of thinking.

## Licence

MIT.
