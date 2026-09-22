# 00 — System Analysis & Design

> **Read this once, end to end, before writing code.** Every number here was either measured on
> the target machine or derived from a published reference point. Estimates are marked `[est]`
> and each has a Phase gate that replaces it with a measurement.

**Target machine (verified):** MacBook Air, `MacBookAir10,1`, Apple M1 (4P+4E), **8 GB** unified
memory, macOS 26.6.2, 337 GB free disk, Python 3.11.16 & 3.12.14 available via `uv`, `uv 0.12.17`,
`make 3.81`, `ffmpeg` **not installed**.

---

## Table of contents

1. [Scope: defining "multimodal" and "from scratch"](#1-scope)
2. [Requirement → engineering translation](#2-requirement-translation)
3. [Hardware reality check](#3-hardware-reality-check)
4. [Memory budget arithmetic](#4-memory-budget-arithmetic)
5. [Throughput model & feasibility matrix](#5-throughput-model--feasibility)
6. [System architecture](#6-system-architecture)
7. [Canonical IR design (`MediaDoc`)](#7-canonical-ir-design)
8. [Per-modality model selection matrix](#8-model-selection-matrix)
9. [From-scratch LLM design](#9-from-scratch-llm-design)
10. [Multimodal bridge design](#10-multimodal-bridge-design)
11. [Data strategy](#11-data-strategy)
12. [Serving architecture](#12-serving-architecture)
13. [Architecture decision records](#13-architecture-decision-records)
14. [Risk register](#14-risk-register)
15. [Performance envelope / SLOs](#15-performance-envelope--slos)
16. [Non-goals](#16-non-goals)

---

## 1. Scope

### 1.1 What "process all possible input formats" means here

It does **not** mean a model that natively consumes 200 file extensions. No such model exists,
and any design that claims it is lying. It means:

- **Closed under a documented matrix.** Every extension listed in the README has an extractor;
  unlisted extensions degrade to `MediaDoc(modality=UNKNOWN, errors=[...])` — never a crash, never
  a silent skip.
- **Two extraction lanes per modality.** A cheap deterministic lane (parsers, macOS Vision OCR,
  ffmpeg) that is always on, and an expensive neural lane (VLM captioning, ASR, embeddings) that is
  opt-in per format so peak memory stays bounded.
- **One output type.** Everything converges on `MediaDoc`. Downstream modules (chunker, embedder,
  retriever, orchestrator) never see a file extension. This is what makes "all formats" tractable
  instead of combinatorial.

### 1.2 What "build an LLM from scratch with transformer architecture" means here

Implemented in this repository — no `transformers`/`torch` model classes, no copied checkpoint:

| Component | From scratch? | Location |
|---|---|---|
| Tokenizer (BPE training, encode, decode, specials) | ✅ | `mmar/llm/tokenizer.py` |
| Embeddings, RoPE, RMSNorm, SwiGLU, GQA, attention mask, KV cache | ✅ | `mmar/llm/model.py` |
| Init, cross-entropy loss, AdamW, LR schedule, grad clip/accum | ✅ | `mmar/llm/train.py`, `optim.py` |
| Data loading + memmap sharding | ✅ | `mmar/llm/data.py` |
| Sampling: temperature, top-k, top-p, repetition penalty, streaming | ✅ | `mmar/llm/sample.py` |
| Autograd (`value_and_grad`) | ❌ | Library primitive, like `torch.autograd` |
| Vision encoder (SigLIP / CLIP ViT) | ❌ frozen pretrained | Cannot pretrain a vision backbone on 8 GB |
| Audio encoder (Whisper / Moonshine) | ❌ frozen pretrained | Same reason |
| Multimodal projector + instruction tuning | ✅ | `mmar/llm/bridge/` |

**The line:** autograd is a library primitive; everything that is *architecture*, *training
semantics* or *inference semantics* is ours. Re-implementing autodiff would be a different project,
and pretending a 9 M-param model is GPT-4 would be dishonest.

---

## 2. Requirement → engineering translation

| You asked for | Engineering translation | Where it lives |
|---|---|---|
| "Process all possible input formats" | Extractor registry + `MediaDoc` IR + `UNKNOWN` fallback | `mmar/ingest`, `mmar/ir` |
| "LLM from scratch, transformer architecture" | MLX decoder-only LM, 0.5 M→30 M presets, NumPy oracle, KV-cache parity tests | `mmar/llm` |
| "That can process it" (multimodal) | LLaVA-style bridge: frozen encoders → MLP projector → our LLM token space | `mmar/llm/bridge` |
| "End to end, locally" | Zero network at inference; `make ingest && make chat`; FastAPI on `127.0.0.1` | `mmar/serve` |
| "Deep analysis, phase-wise execution" | This document + `docs/01_ROADMAP.md` | `docs/` |
| "Detailed prompt per phase + function requirements" | `docs/phases/P00…P12.md` + `docs/02_FUNCTION_REQUIREMENTS.md` | `docs/` |

## 3. Hardware reality check

### 3.1 The numbers that matter

| Property | Value | Why it matters |
|---|---|---|
| Unified memory | 8 GB (8,589,934,592 B) | Shared by CPU + GPU + OS. **Not** 8 GB of model budget |
| Memory bandwidth | 68.25 GB/s LPDDR4X-4266 | **Decode is bandwidth-bound:** `tok/s ≈ 0.85 × BW ÷ weight_bytes` |
| GPU | 8 cores / 128 EUs | ~2.6 TFLOPS fp32, ~5.2 TFLOPS bf16 peak |
| AMX matrix coprocessor | CPU path only, unreachable from MLX | No free CPU training win |
| Cooling | **Fanless** | Sustained ≈ 70–80 % of burst; throttles after ~10–15 min at full load |
| Disk | 337 GB free | Ample. TinyStories shards ≈ 1 GB, checkpoints ≈ 100 MB each |
| Zero-copy unified memory | Weights need no host↔device copy | Effective ~1.3–1.6× advantage vs discrete-GPU paths at equal FLOPs |

**Measured baseline `[est, verified in Phase 0]`:** idle macOS + VS Code + terminal consumes
~3.0–3.8 GB. Hence `max_process_rss_mb: 4600` and `warn_process_rss_mb: 3600` in
`configs/default.yaml`. These are not conservative guesses — crossing ~5 GB on an 8 GB Apple
Silicon machine starts the compressor and ML throughput collapses 5–20×. The failure mode is **not**
an OOM kill; it is silent, mysterious slowness. `MemoryGuard` exists to prevent exactly that.

### 3.2 Reference throughput points (published, not extrapolated)

| Source | Hardware | Model | Throughput |
|---|---|---|---|
| nanoGPT-on-MLX | M2 Pro 16 GB | 53 M params bf16, TinyStories | **27 k tok/s** |
| Goedecke 2025 | M4 MBP 24 GB (MPS) | 1.0 M params | 100 k tok/s |
| Goedecke 2025 | M4 MBP 24 GB (MPS) | 2.6 M params | 56 k tok/s |
| mlx-vlm compat report | M4 Max 128 GB | LFM2.5-VL-1.6B bf16 | 196 tok/s @ **4.4 GB peak** |
| mlx-vlm compat report | M4 Max 128 GB | SmolVLM-Instruct bf16 | 112 tok/s @ **5.5 GB peak** |

**M1 Air scaling:** ≈ ½ the GPU cores of an M2 Pro, and 68 vs 200 GB/s bandwidth. Expect
**1.8–2.4× slower** than the M2 Pro row, i.e. **11–15 k tok/s for a ~9 M-param model `[est]`**.
Phase 7 runs `benchmarks/throughput.py` and freezes the measured number into `docs/12_RESULTS.md`.

### 3.3 Environment gaps to close in Phase 0

| Gap | Impact if ignored | Fix |
|---|---|---|
| `python3` is **3.14.7** | `mlx`, `pyobjc-framework-Vision`, `tokenizers` wheels missing/untested → cryptic build errors | Pin 3.11 via `.python-version` + `uv venv --python 3.11` |
| `ffmpeg` **not installed** | No audio/video ingestion, no transcode | `brew install ffmpeg` |
| No CI / test config | Silent regressions in numerical code | `pyproject.toml` already declares pytest + ruff + mypy |
| Fanless thermal ceiling | An 8 h run silently becomes 16 h | `--thermal-relief` duty-cycle mode (Phase 8) |
| HF cache default `~/.cache/huggingface` | First model download can be 1–5 GB; a surprise here looks like a hang | `MMAR_MODEL_CACHE` env var + `mmar models pull` with a progress bar |

---

## 4. Memory budget arithmetic

Everything in `configs/default.yaml` derives from the formulas below. Do not change a number in
that file without redoing this arithmetic.

### 4.1 Parameter count (the formula to memorise)

For a decoder-only transformer with GQA, SwiGLU and tied embeddings, embedding table size `V`:

```
d_kv      = n_kv_heads * (d_model / n_heads)          # grouped-query key/value width
per_layer = (d_model*d_model + 2*d_model*d_kv + d_model*d_model)   # QKV + output proj
          + 3*d_model*d_ff                                          # SwiGLU gate/up/down
params    = n_layers * per_layer + V*d_model                        # embeddings counted once
```

The **`V*d_model` term dominates at small scale** — this is why the presets use a 4 096-token
vocabulary and not GPT-2's 50 257. A 50 257 vocab on the `micro` body costs 19.3 M params of pure
embedding table (1.7× the rest of the model) and buys nothing for a TinyStories corpus.

### 4.2 Computed preset sizes

| Preset | d_model | layers | heads / kv | d_ff | Params | bf16 weights | Train peak `[est]` |
|---|---|---|---|---|---|---|---|
| `nano` | 128 | 2 | 4 / 2 | 512 | **1.0 M** | 2 MB | ~120 MB |
| `micro` | 384 | 6 | 6 / 2 | 1 024 | **11.0 M** | 22 MB | ~0.7 GB |
| `small` | 640 | 8 | 8 / 4 | 1 728 | **39.0 M** | 78 MB | ~1.2 GB |

**The key insight of this whole design:** our from-scratch LLM is *memory-free*. At 22–78 MB of
weights it is noise next to a 1 GB encoder. Memory is consumed by the **pretrained perception
models**, never by our own model. That is why the memory plan is entirely about `max_concurrent_models: 1`.

### 4.3 Inference peak RSS

```
peak ≈ W + KV + A + F
  W  = params * bytes_per_param        # fp32 4.0 | bf16 2.0 | 8-bit 1.0 | 4-bit ~0.55
  KV = 2 * n_layers * n_kv_heads * head_dim * seq_len * batch * dtype_bytes
  A  = activations, O(batch * seq_len * d_model)   -- negligible for single-stream decode
  F  = 150-300 MB  # MLX runtime + Metal buffers + tokenizer + Python interpreter
```

Sanity check on our model: `micro` at `seq_len = 2048`, batch 1, bf16 →
`2 * 6 * 2 * 64 * 2048 * 2 = 6.3 MB` of KV cache. The KV cache is irrelevant at our scale; it
becomes relevant only if you swap in a 1.5 B-param encoder-decoder.

### 4.4 Training peak RSS

```
peak ≈ params * 14          # bf16 weights(2) + fp32 master(4) + Adam m,v (8)
      + activations         # ~ batch * seq * n_layers * 12 * d_model bytes
      + 300 MB framework
```

Worked example, `micro`: `11e6 * 14 B = 154 MB` + `8*512*6*12*384 B = 113 MB` + 300 MB ≈ **0.6 GB**.
Worked example, `small`: `39e6 * 14 = 546 MB` + `8*512*8*12*640 = 252 MB` + 300 MB ≈ **1.1 GB**.

**Conclusion: the 8 GB limit does not constrain model *size* at all for training from scratch — it
constrains model *time*.** A 100 M-param model would fit (~2.2 GB); you simply cannot push enough
tokens through it to converge in a sane number of evenings. Time, not memory, is the binding
constraint (§5).

### 4.5 Resident profiles — what may run at the same time

`max_concurrent_models: 1` is enforced by `ModelRegistry`. These are the *only* legal combinations:

| Scenario | Resident components | Projected peak RSS | Headroom vs 4.6 GB |
|---|---|---|---|
| A. Text/doc ingest | parsers + Apple Vision OCR (OS-managed) | ~0.3 GB | ✅ 4.3 GB |
| B. Image ingest | + SigLIP/CLIP embedder (0.2 GB) | ~0.6 GB | ✅ 4.0 GB |
| C. Audio ingest | + `whisper-base` ASR (~0.2 GB) | ~0.6 GB | ✅ 4.0 GB |
| D. VLM captioning | + SmolVLM-256M 4-bit (~0.7 GB) | ~1.4 GB | ✅ 3.2 GB |
| E. RAG chat (Track D) | + Qwen3-0.6B 4-bit (~0.5 GB) | ~1.1 GB | ✅ 3.5 GB |
| F. Pretrain `micro` | our LM only | ~0.7 GB | ✅ 3.9 GB |
| G. Bridge training | CLIP ViT-B/32 (0.35 GB) + our LM (0.1 GB) | ~1.3 GB | ✅ 3.3 GB |
| ❌ Never | LFM2.5-VL-1.6B bf16 (**4.4 GB**) + anything | ~5.4 GB → **swapping** | 🚫 forbidden |

**Rule derived from the table:** any model whose measured peak exceeds **2.5 GB** is rejected by
`MemoryGuard` unless `runtime.allow_oversize: true` is set explicitly. The 4.4 GB LFM2.5-VL entry
is exactly the trap this rule catches — it is *the* most memory-efficient VLM in the mlx-vlm
compatibility report and it is still unusable here.

---

## 5. Throughput model & feasibility

### 5.1 The training-time equation

```
training FLOPs   C    = 6 * N * D                     # N params, D tokens
tokens/second    TPS  = (P * eff(N)) / (6 * N)         # P = bf16 peak FLOPs
wall-clock hours H    = D / (TPS * 3600)
```

`P(bf16, M1 8-core GPU) ≈ 5.2 TFLOPS`. The non-obvious term is `eff(N)`, the achieved fraction of
peak, and it is **strongly size-dependent because small models are launch-bound, not FLOP-bound**.
Goedecke measured this directly on MPS: a 1 M-param model reached only ~6 % of peak (100 k tok/s
for 4.3 GFLOP/step), a 2.6 M model ~9 %. Efficiency climbs steeply with size as per-launch work
grows. Calibrating that curve against the published M2 Pro data point (53 M params → 27 k tok/s)
and applying a 1.8–2.4× M1 penalty gives:

| Preset | N (params) | `eff(N)` `[est]` | TPS on M1 Air `[est]` | Source of estimate |
|---|---|---|---|---|
| `nano` | 1.0 M | 4–8 % | 30 000–60 000 | M4 measurement, launch-bound floor |
| `micro` | 11.0 M | 10–18 % | **8 000–20 000** | interpolated |
| `small` | 39.0 M | 20–35 % | **3 000–8 000** | M2 Pro point, M1 penalty applied |

**Phase 7 gate:** `benchmarks/throughput.py` measures real TPS for each preset and writes
`docs/12_RESULTS.md`. Every hour estimate in the roadmap is replaced by the measured value at that
point. Assume the **low end** of each range when planning your evenings; that is why the roadmap is
written in "evenings", not hours.

### 5.2 Chinchilla check — and why the `small` preset is a trap

Chinchilla-optimal: `D ≈ 20 × N`. If you train on fewer tokens than that, you are wasting
parameters; if you train on many more, you are wasting compute on too small a model. Applying it:

| Preset | N | Optimal D (20N) | Time at low-end TPS | Verdict |
|---|---|---|---|---|
| `nano` | 1.0 M | 20 M | ~9 min | Tests and parity checks only |
| `micro` smoke | 11.0 M | 40 M (5× under-trained) | **~1.4 h** | Pipeline validation run — do this first |
| **`micro` full** | **11.0 M** | **220 M (optimal)** | **~5.1 h** | ✅ **This is the run to do** |
| `small` | 39.0 M | 780 M | **~72 h** | ❌ **Not worth it** |

**The conclusion the arithmetic forces:** on an M1 Air, `small` costs ~2 weeks of overnight runs for
a marginal quality gain over a properly-trained `micro`. **Stay at 11 M params and give it a full
pass over the corpus instead.** This contradicts the instinct to "go bigger", and it is the single
most valuable finding in this document.

TinyStories context: 2.1 M stories ≈ **~470 M tokens** available, so the 220 M-token target is a
real 47 % epoch, not a synthetic repeat. For reference, the TinyStories paper shows a 125 M-param
GPT-Neo producing fluent, coherent children's stories; an 11 M model trained to Chinchilla-optimal
similarly produces short, grammatically correct, locally coherent English with occasional drift.
Expect val perplexity **7–11** for the full run and **15–25** for the smoke run.

---

## 6. System architecture

### 6.1 Layer model and the one-directional dependency rule

```
 L6  interfaces      CLI (typer)            FastAPI + Web UI            library API
 ------------------------------------------------------------------------------------
 L5  orchestration   Orchestrator  ·  RAGPipeline  ·  ChatSession  ·  EvalHarness
 ------------------------------------------------------------------------------------
 L4  reasoning       mmar.llm (from scratch)          mmar.models (frozen encoders)
                     tokenizer/model/train/sample     captioner/transcriber/embedder
                     mmar.llm.bridge (projectors)     ModelRegistry + MemoryGuard
 ------------------------------------------------------------------------------------
 L3  retrieval       DenseIndex  ·  BM25Index  ·  Retriever  ·  Reranker
 ------------------------------------------------------------------------------------
 L2  representation  chunkers  ·  EmbeddingCache  ·  MediaStore (sqlite + files)
 ------------------------------------------------------------------------------------
 L1  ingestion       ExtractorRegistry  ·  one Extractor per format family
 ------------------------------------------------------------------------------------
 L0  foundation      config  ·  utils(memory, logging, io, hashing)  ·  ir (MediaDoc)
```

**Dependency rule (enforced by a test, `tests/test_architecture.py`):** a module may import only
from strictly *lower* layers and from its own layer's `ir`/`config`. Concretely:

- `mmar.llm` must **never** import `mmar.ingest`, `mmar.rag` or `mmar.serve`. This keeps the
  from-scratch model training script runnable on a machine with no PDF/audio dependencies.
- `mmar.ingest` must **never** import `mmar.llm`. Ingestion is text extraction, not generation.
- `mmar.serve` is the only place that knows about HTTP.
- `mmar.models` is the only place that touches MLX *inference* objects; `mmar.llm` is the only place
  that touches MLX *training* objects. They share `mmar.utils.memory` and nothing else.

Why this matters on this specific project: it is the only defence against the classic failure of
"local multimodal" repos, where ingestion, model inference and the web server all import each other
and the training script stops working the moment someone adds a PDF parser.

### 6.2 End-to-end data flow

```
 INPUT (anything)
   │
   ▼
[1] sniff: extension -> magic bytes -> MIME      (utils/sniff.py)   never trust the extension
   │
   ▼
[2] dispatch: ExtractorRegistry.resolve(mime, path) -> Extractor    (ingest/registry.py)
   │
   ▼
[3] extract: cheap lane always; neural lane if config allows        (ingest/extractors/*)
   │      text/pdf/office  -> deterministic text
   │      image           -> Apple Vision OCR text (+ optional VLM caption)
   │      audio           -> ASR transcript + segment timestamps
   │      video           -> keyframes (recursively image-extracted) + audio transcript
   │
   ▼
[4] canonicalise -> MediaDoc(assets[], chunks[], extracted{}, errors[])   (ir/document.py)
   │
   ▼
[5] chunk: strategy per modality, token-budgeted                    (rag/chunker.py)
   │
   ▼
[6] embed: text chunks + image chunks -> vectors (cached by sha256)  (rag/embed.py)
   │
   ▼
[7] persist: MediaStore (sqlite) + vector index + BM25 index        (rag/store.py)
   │
   ▼  ── ingestion ends / query begins ──
   ▼
[8] retrieve: hybrid dense+BM25 -> fuse -> rerank                   (rag/retriever.py)
   │
   ▼
[9] assemble: budget-aware context builder with citations           (llm/prompt.py)
   │
   ▼
[10] reason: from-scratch LLM  OR  bridge VLM  OR  Track-D instruct model   (serve/orchestrator.py)
   │
   ▼
[11] stream: SSE token stream + citations + latency/memory telemetry (serve/app.py)
```

Steps 1–7 are pure Python with no model resident except the cheap embedder, which is why a
mixed-format ingestion of hundreds of files never threatens the memory budget. Steps 9–11 are where
a model becomes resident, and where `max_concurrent_models: 1` applies.

---

## 7. Canonical IR design

### 7.1 Design constraints that shaped it

1. **Must be JSON-serialisable** — you will debug this by inspecting stored docs, and a
   pickle-only IR is undebuggable.
2. **Must carry provenance** — every chunk needs to answer "which file, which page, which
   timestamp, which byte range". Without this, RAG citations are impossible.
3. **Must record failure** — an `errors` list, not an exception. A 400-file ingest must not abort
   because one `.mp4` has a corrupt index.
4. **Must be flat enough for sqlite** — `MediaDoc` is a container; `Chunk` rows are the queryable
   unit. Chunks carry `doc_id`, so one table serves every modality.
5. **Must not encode modality in the schema.** Adding video support must not require a migration.

### 7.2 The types

```python
# src/mmar/ir/types.py
from __future__ import annotations
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Literal

class Modality(StrEnum):
    TEXT = "text";       IMAGE = "image";     AUDIO = "audio"
    VIDEO = "video";     DOCUMENT = "document"  # paged/structured docs: pdf, docx, xlsx
    TABULAR = "tabular"; CODE = "code";       WEB = "web"
    ARCHIVE = "archive"; UNKNOWN = "unknown"

class ExtractLane(StrEnum):
    NATIVE = "native"    # deterministic parser produced this text
    OCR = "ocr"          # macOS Vision framework read it off pixels
    ASR = "asr"          # speech model transcribed it
    VLM = "vlm"          # a vision-language model described it
    META = "meta"        # synthesised from file metadata only (last resort)

@dataclass(frozen=True, slots=True)
class Asset:
    """One physical artefact inside a source: a page, a keyframe, an audio track, a sheet."""
    asset_id: str                 # f"{doc_id}:a{index}"
    modality: Modality
    mime: str
    uri: str | None               # extracted-to-disk path, or original path for top-level
    sha256: str
    bytes_len: int
    meta: dict[str, Any] = field(default_factory=dict)  # page=3, ts_start=12.4, sheet="Q3"…

@dataclass(frozen=True, slots=True)
class Chunk:
    """The unit of retrieval. One row in the `chunks` table."""
    chunk_id: str                 # f"{doc_id}:c{index}"
    doc_id: str
    asset_id: str | None
    text: str
    lane: ExtractLane
    token_count: int
    char_span: tuple[int, int]
    time_span: tuple[float, float] | None = None   # audio/video seconds
    page: int | None = None
    embedding: list[float] | None = None           # filled by the embed stage, not by extractors
    meta: dict[str, Any] = field(default_factory=dict)

@dataclass(slots=True)
class DocStats:
    n_assets: int = 0
    n_chunks: int = 0
    n_tokens: int = 0
    n_errors: int = 0
    extract_ms: float = 0.0
    ocr_pages: int = 0
    asr_seconds: float = 0.0

@dataclass(slots=True)
class MediaDoc:
    """The canonical representation. Every extractor returns exactly this."""
    doc_id: str                   # sha256 of (source_uri + content_hash)[:16]
    source: str                   # original path or URL, for display + re-ingest
    modality: Modality
    mime: str
    schema_version: int = 1       # bump on any breaking change; migrations live in ir/migrate.py
    assets: list[Asset] = field(default_factory=list)
    chunks: list[Chunk] = field(default_factory=list)
    text: str = ""                # concatenated plain text, cheapest possible consumer view
    extracted: dict[str, Any] = field(default_factory=dict)  # typed payload: {"tables":[...]}
    errors: list[str] = field(default_factory=list)          # human-readable, never raised
    stats: DocStats = field(default_factory=DocStats)
    meta: dict[str, Any] = field(default_factory=dict)       # title, author, duration_s…
```

### 7.3 Invariants (asserted in tests, not trusted by convention)

| # | Invariant | Rationale |
|---|---|---|
| I1 | `len(chunk.chunk_id) > 0` and IDs are unique across the whole store | Retrieval joins on ID |
| I2 | `chunk.doc_id == doc.doc_id` for every chunk | Store integrity |
| I3 | `chunk.asset_id is None or asset_id ∈ {a.asset_id for a in doc.assets}` | No dangling provenance |
| I4 | `sum(c.token_count for c in chunks) == stats.n_tokens` (± rounding note in tests) | Cheap consistency check |
| I5 | A failed extraction returns a `MediaDoc` with `errors` non-empty and **does not raise** | 400-file batches must complete |
| I6 | `to_json(doc)` → `from_json(...)` is lossless for all fields except `embedding` floats (documented tolerance 1e-6) | Persistence round-trip |
| I7 | `schema_version` is monotonically increasing and every stored doc is migrated on read | Long-lived on-disk store |
| I8 | `MediaDoc` is constructible with zero arguments except `doc_id`/`source`/`modality`/`mime` | Test ergonomics |

**Why `text` *and* `chunks`:** `text` exists so the simplest possible consumer (a prompt builder, a
grep, a debug print) never has to reassemble chunks. `chunks` exists so retrieval never has to
re-chunk. Duplication is intentional and bounded (text ≈ Σ chunk text + separators).

---

## 8. Per-modality model selection matrix

Every row was chosen against one test: **does peak RSS stay under ~2.5 GB in isolation?** Models
that failed are listed at the bottom, because knowing what *not* to try is half the design.

### 8.1 Vision — encode, OCR, caption

| Job | Chosen | Size / quant | Peak RSS | Why this one |
|---|---|---|---|---|
| Text-in-image | **Apple Vision via `ocrmac`** | OS service | **~0** (OS-managed) | Free, no download, 30+ languages, ~0.5 s/page on M1. Beats Tesseract decisively and needs no model weights |
| Image embedding | **SigLIP-base-patch16-224** | 93 M, bf16 | ~0.25 GB | Better than CLIP at the same size for retrieval; 196 patch tokens |
| Image embedding (alt) | CLIP ViT-B/32 | 151 M, bf16 | ~0.35 GB | Required only if you want a *text* encoder in the same space (bridge stage) |
| Caption / VQA | **SmolVLM-256M-Instruct** 4-bit | 256 M | **~0.8 GB** | Smallest VLM whose captions are actually usable; ~112 tok/s class |
| Caption (alt) | LFM2-VL-450M 8-bit | 450 M | ~1.1 GB | Better grounding than SmolVLM-256M |
| Caption (alt, best) | SmolVLM2-2.2B-Instruct 4-bit | 2.2 B | ~1.8 GB | Noticeably better captions; still legal here, but only alone |
| ❌ **Rejected** | LFM2.5-VL-1.6B **bf16** | 1.6 B | **4.4 GB measured** | The most memory-efficient VLM in the mlx-vlm compatibility report, and still ~96 % of our budget. 4-bit variant only |
| ❌ **Rejected** | Qwen2-VL-2B-4bit | 2 B | OOM reported | Failed in the mlx-vlm report even on a 128 GB machine — a known-broken config |
| ❌ **Rejected** | Qwen3-VL-2B-Instruct | 2 B | **12 GB measured** | 2.6× our budget |

**Phase 2 gate:** captioning is **off by default** (`ingest.image.caption: false`). It is the single
most expensive optional lane, and OCR + embedding already make images searchable. You turn it on
deliberately, for a folder of images where free-form description matters.

### 8.2 Audio — speech to text

| Job | Chosen | Size | Peak RSS | Why |
|---|---|---|---|---|
| ASR (default, pass 1) | `whisper-base` | 74 M | ~0.2 GB | `ingest.audio.backend` default (P3): fast first pass over any archive |
| ASR (upgrade, pass 2) | `mlx-community/whisper-large-v3-turbo` | ~0.8 B | ~1.0–1.6 GB | Accuracy pass via `mmar transcribe --upgrade`; best accuracy/size trade on Apple Silicon, 99 languages, pure-MLX |
| ASR (bulk triage) | `whisper-tiny` | 39 M | ~0.1 GB | Pre-screening very large archives only |
| ASR (alt, English) | Moonshine-base via CPU ONNX | 61 M | ~0.3 GB | 6× smaller than Whisper-large-v3, **much lower hallucination risk**; streaming-friendly |
| ASR (alt, best English) | Parakeet-TDT-0.6b-v3 | 0.6 B | ~0.7 GB | Strongest English accuracy per benchmark; NeMo path is heavier to install |
| ❌ Rejected | `faster-whisper` large-v3 | 1.5 B | ~2 GB+ int8, CPU-only | CTranslate2 has no Apple GPU path — it burns the CPU while the GPU idles |

**Design consequence:** audio ingest is a **two-pass** design. Pass 1 with `whisper-base` produces a
searchable transcript in seconds; pass 2 with turbo re-transcribes only the files you actually
retrieve. This is what makes a 3-hour podcast archive tractable.

### 8.3 Text embeddings

| Job | Chosen | Size | Dim | Why |
|---|---|---|---|---|
| Chunk embeddings | **`intfloat/multilingual-e5-small`** | 118 M | 384 | 100+ languages covers "any input format" in the linguistic sense too; 384-dim keeps the index tiny |
| Alt (English-only, best speed) | `all-MiniLM-L6-v2` | 22 M | 384 | ~5× faster, ~0.1 GB; switch with one config line |
| ❌ Rejected | `bge-large-en-v1.5` | 335 M | 1024 | 4.5× the index size and ~0.7 GB resident for a corpus that does not justify it |

Index size sanity check: 10 000 chunks × 384 dims × 4 bytes = **15 MB**. The whole index fits in
RAM many times over, so exact numpy search beats approximate search — no FAISS, no HNSW, no
recall loss, one dependency fewer. Approximate indexing only becomes relevant above ~1 M chunks,
which is a decade of personal documents.

### 8.4 The reasoning model (three tracks, escalating capability)

| Track | Model | Peak RSS | Role |
|---|---|---|---|
| A | **our `micro` LM, 11 M params** | ~0.1 GB (inference) | The from-scratch deliverable. Free memory-wise; proves the stack |
| C | **our `micro` + CLIP/SigLIP projector** | ~0.5 GB | The multimodal deliverable. Our LLM + frozen eyes |
| D | **Qwen3-0.6B-Instruct 4-bit** (or SmolLM2-360M-Instruct 8-bit) | ~0.5 GB | The *useful* assistant over your own files, available in Phase 4 |

Track D exists so that Phases 5–10 (the from-scratch work) never block Phases 1–4 from producing
something you can use. **You will have a working personal RAG assistant weeks before you have a
model you trained yourself** — and the trained model gets plugged into the same interface via
`llm.active: micro`, which is exactly why the LLM is behind an interface in the first place.

---

## 9. From-scratch LLM design

### 9.1 Architecture decisions and their justification

Each choice is either (a) necessary at 11 M params, or (b) free at 11 M params. Nothing is chosen
"because GPT-3 did it".

| Decision | Value | Why, at this scale |
|---|---|---|
| Decoder-only | ✅ | One loss, one objective, no encoder-decoder plumbing. Completion, instruction-following and VLM token consumption are all decoder-only |
| Pre-norm **RMSNorm** | ε = 1e-5 | Pre-norm is what makes deep-but-narrow stacks trainable without warmup heroics; no mean-centring to get wrong |
| **RoPE** positional encoding | θ = 10 000 | No learned position table = fewer params (the embedding table is already ~14 % of the model) and ~2× length extrapolation |
| **SwiGLU** MLP | `d_ff ≈ 8/3·d` | ~+0.3 nats over GELU-MLP at equal params; 3 matrices instead of 2 is affordable here |
| **GQA** `n_kv_heads = n_heads/3` | 6 query / 2 kv | Equal quality, 3× smaller KV cache and K/V projections. This is what makes appending dozens of image tokens cheap in Phase 10 |
| Tied embeddings | ✅ | Saves `V·d = 1.6 M` params on `micro` — 14 % of the model. Negligible loss at 4 096 vocab; would not be at 50 257 |
| No biases anywhere | ✅ | Standard for >5 years; one fewer thing to initialise |
| std = 0.02 init, residual scaled by `1/√(2·n_layers)` | ✅ | Keeps activation variance ≈ 1 through 6–8 blocks. **This is the single most common cause of "my from-scratch model won't train"** |
| bf16 autocast + fp32 master weights | ✅ | bf16 exists for training stability; fp32 masters stop `lr=6e-4` + bf16 from stalling |
| Dropout 0.0 pretrain / 0.05 SFT | ✅ | At 220 M tokens on a 470 M-token corpus you are **under**-fitting. Pretraining dropout here is pure harm |
| No MoE / attention sinks / YaRN / MTP | ✅ | Complexity with no payoff at 11 M params and 512-token contexts |
| Context **512**, not 2 048 | ✅ | Attention is quadratic, TinyStories stories average ~200 tokens, and every wasted context token is paid 13 428 times |

### 9.2 The forward pass

```python
# src/mmar/llm/model.py  (from scratch - MX primitives only, no nn.Transformer)
class MMRTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig):
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks  = [TransformerBlock(cfg) for _ in range(cfg.n_layers)]
        self.norm_f  = RMSNorm(cfg.d_model, cfg.norm_eps)
        # lm_head is *absent* when cfg.tie_embeddings -> use tok_emb.as_linear()

    def __call__(self, tokens, cache: "KVCache | None" = None) -> mx.array:
        x = self.tok_emb(tokens)
        mask = causal_mask(tokens.shape[-1], offset=cache.offset if cache else 0)
        for i, blk in enumerate(self.blocks):
            x = blk(x, mask=mask, cache=cache, layer_idx=i)
        return self.output_proj(self.norm_f(x))

class TransformerBlock(nn.Module):
    # pre-norm residual:  x = x + scale * Attn(RMSNorm(x))
    #                     x = x + scale * MLP (RMSNorm(x))   where scale = (2*n_layers)**-0.5

class Attention(nn.Module):
    # q_proj: d_model -> n_heads    * head_dim   (no bias)
    # k_proj: d_model -> n_kv_heads * head_dim   (no bias)
    # v_proj: d_model -> n_kv_heads * head_dim   (no bias)
    # o_proj: d_model -> d_model                 (no bias)
    # RoPE applied to q,k only; SDPA takes the Metal fast path
```

### 9.3 Training loop (the parts that actually matter)

```
for step in 1..max_steps:
    lr = cosine(warmup(step, 300), max_steps, lr=6e-4, min_lr=6e-5)     # no lr=1e-3 heroics
    for micro_step in 1..grad_accum_steps:                              # 4 x 8 x 512 = 16 384 tok/step
        loss, grads = value_and_grad(model, loss_fn)(batch)             # cross-entropy, bf16 fwd
    grads = clip_global_norm(grads, 1.0)                                # non-negotiable at lr=6e-4
    optimizer.update(model, grads)                                      # AdamW, wd=0.1 on matrices only
    if step % eval_interval   == 0: val_loss = evaluate(model, val_loader, 20 batches)
    if step % ckpt_interval   == 0: checkpointer.save(model, optimizer, step)
```

Five non-obvious requirements, each of which has silently broken a real run:

1. **Weight decay must skip** `tok_emb`, every `norm_*` and all 1-D parameters. Decaying norms makes
   the loss curve plateau at a level that looks exactly like a data problem.
2. **Loss must be computed in fp32** even when the forward is bf16. `softmax` over 4 096 bf16 logits
   loses enough precision to bias the gradient.
3. **Validation must run in eval mode** with dropout disabled, and the *first* validation batch must
   be deterministic and reused, so the val curve is comparable across steps.
4. **`mx.compile` only on the train step**, and the step counter must be a traced argument, not a
   captured Python int — otherwise you silently retrace every step and lose the entire speed-up.
5. **Divergence guard:** if `val_loss > train_loss + 1.5`, abort and keep the last good checkpoint.
   At this scale a corrupt data shard shows up as a *widening* loss gap, not a spike.

### 9.4 The NumPy oracle (why this project has parity tests)

`mmar/llm/ref/` holds a deliberately naive NumPy implementation of exactly the same maths:
`softmax`, `rms_norm`, `rope`, `attention`, `swiglu`, `cross_entropy`. It exists for three reasons:

- **Correctness.** A bug in RoPE or in the causal mask is invisible in the loss curve and destroys
  generation quality. `pytest -m parity` compares MLX vs NumPy to `atol=1e-4` on random inputs for
  each primitive *and* for a full forward pass.
- **Education.** The NumPy file is the readable specification; the MLX file is the fast
  implementation. Diffing them is how you actually learn what MLX is doing.
- **Framework-bug insurance.** The nanoGPT-on-MLX write-up documents a real silent MLX bug where
  `mx.random.categorical` produced broken sampling of otherwise-fine logits. Independent
  verification is not paranoia.

**These tests are the Phase 7 exit gate. No training run starts until `make test-parity` passes.**

### 9.5 Instruction tuning (Phase 9)

After pretraining, the base model becomes an assistant via a chat template baked into the
tokenizer's special tokens (`<|user|>`, `<|assistant|>`, `<|eos|>`), trained on ~2 000 hand-built
prompts + self-generated pairs, with **loss masked to the assistant spans only**.

```
<|bos|><|user|>Summarise this note: <text><|assistant|>The note is about…<|eos|>
                      ^^^ loss-masked (weight 0) ^^^        ^^^ loss counted here ^^^
```

Design rules that matter at 11 M params:
- **Loss on assistant tokens only.** Training on the prompt teaches the model to *produce* your
  questions, which ruins multi-turn behaviour.
- **Keep instructions short and templated.** An 11 M-param model cannot generalise over long,
  varied instructions. Constrained formats (summarise, extract, answer-from-context) work; open
  "be a helpful assistant" role-play does not. This is a property of the model size, not of the
  recipe.
- **Catastrophic forgetting is real.** Mix ~20 % raw pretraining text into every SFT batch.
- **Evaluate on held-out prompts with exact-match / ROUGE-L**, not on the training loss.

### 9.6 What an 11 M-param model can and cannot do (set expectations in advance)

| Can | Cannot |
|---|---|
| Produce grammatical, locally coherent English | Follow multi-step or long-form instructions |
| Continue a story or a templated pattern reliably | Do arithmetic, logic or factual recall |
| Answer extractive questions when the answer is *verbatim in the context* | Reason over retrieved context or synthesise across documents |
| Learn a narrow, templated output format | Behave like a chat assistant in the ChatGPT sense |

**Consequence for the architecture:** the from-scratch model's real job in this project is to be the
**decoder and the token space that the multimodal bridge projects into** (Phase 10). Where genuine
reasoning quality is required, the system routes to Track D (`Qwen3-0.6B-Instruct`). Both sit behind
the same `TextGenModel` interface, so switching is one config line and zero code changes.

---

## 10. Multimodal bridge design

### 10.1 Why the LLaVA pattern is the only viable path here

Three possible architectures for a multimodal model on 8 GB:

| Approach | Verdict |
|---|---|
| Train vision + language jointly from scratch | ❌ Requires 10³–10⁴ GPU-hours. Not a "hardware is weak" problem, a "physically impossible" problem |
| Use a pretrained VLM end-to-end | ✅ Works (Track D) but teaches you nothing and is not "from scratch" |
| **Frozen pretrained encoders + trained projector into our own LLM** | ✅ **Chosen.** This is the LLaVA recipe that produced the first open GPT-4V-class models, and its trainable part is small enough to fit |

The insight: **the expensive part of multimodality is the modality encoder, not the fusion.** By
freezing CLIP/SigLIP and Whisper and only training the mapping into our token space, the trainable
parameter count drops from billions to ~1.3 M — and the "from scratch" claim stays honest, because
what we train (the fusion + the decoder) is exactly what LLaVA trains.

### 10.2 Architecture

```
  image ──► SigLIP-base/CLIP ViT-B32 (FROZEN) ──► 196 patch feats (d_v = 768)
                                                        │
                                          Projector (TRAINED, 2-layer MLP + GELU)
                                          d_v -> 4·d_model -> d_model
                                                        │
                                              196 tokens ──┐
  audio ──► Whisper encoder (FROZEN) ──► 1500 frames ──►│  Projector 2 (TRAINED)
                     subsample 6x -> 250, mean-pool 5x -> 50 tokens ──┘
                                                        │
  text  ──► our BPE tokenizer ──► N text tokens ─────────┤
                                                        ▼
                             [ text_tokens ; image_tokens ; audio_tokens ; text_tokens ]
                                                        │
                                                        ▼
                                 our MMRTransformer (TRAINED: stage 1 frozen, stage 2 LoRA)
                                                        │
                                                        ▼
                                        "<|assistant|> The image shows a …"
```

### 10.3 Token budget arithmetic (the reason the projector is shaped this way)

At `max_seq_len = 512` and `d_model = 384`, every modality competes for the same 512 slots:

| Source | Raw | After projection | Cost |
|---|---|---|---|
| CLIP ViT-B/32 image | 196 patches | 196 tokens | 38 % of context |
| Whisper audio, 30 s window | 1 500 frames | 50 tokens (6× subsample, 5× pool) | 10 % of context |
| Text prompt | — | ~100 tokens | 20 % of context |
| **Answer budget left** | | **~166 tokens** | Enough for a short caption or answer |

**Design consequence:** patch count must be reduced, not accepted. Two mechanisms, config-selectable:
1. **Pixel-shuffle / spatial pooling** on the 196 patch embeddings → 49 tokens (2×2 merge). Costs
   fine-grained detail, triples the answer budget. Default for Phase 10.
2. **Learned query resampler** (Perceiver-style, 32 queries). Better quality-per-token but another
   trained module. Documented as a Phase-10 stretch goal, not a requirement.

Without this reduction the model spends 38 % of its context on image tokens and produces one-line
answers, which is the classic failure of a naive LLaVA-from-scratch attempt.

### 10.4 Staged training recipe

| Stage | Trainable | Frozen | Data | Steps | Time `[est]` | Effect |
|---|---|---|---|---|---|---|
| **0. Pretrain** (Phase 8) | whole LM | — | TinyStories 220 M tok | 13 428 | 3–8 h | Language exists |
| **1. Align** | projector only | LLM + encoders | 30 k caption pairs | ~2 000 | 25–40 min | Image tokens become "words" the LM already understands |
| **2. Instruct** | projector + LoRA on LLM (r=8) | encoders, most of LLM | 10 k VQA pairs | ~3 000 | 45–90 min | Answers questions about the image |
| **3. Audio** | projector 2 only | everything else | 10 k audio-caption pairs | ~1 000 | 15–25 min | Speech becomes text-like tokens |

**Why stage 1 freezes the LLM:** if you train both from the start, the LM unlearns English while the
projector is still random — the gradients from garbage image tokens are pure noise. Freezing the LLM
for stage 1 is the single most important detail of the LLaVA recipe and the most commonly skipped.

**Why LoRA rather than full fine-tuning in stage 2:** at 11 M params full fine-tuning *would* fit in
memory, but it would also catastrophically forget TinyStories-level language in ~3 000 steps on
10 k VQA examples. LoRA (r=8 on `q_proj`/`v_proj`/`o_proj` only) adapts the model ~15× more slowly
and is 12× cheaper to checkpoint.

### 10.5 Data strategy for the bridge

| Need | Source | Size | Why this one |
|---|---|---|---|
| Caption pairs (stage 1) | COCO-2014 subset, 30 k images | ~5 GB images | The canonical LLaVA alignment set; captions are short and templated, which suits an 11 M LM |
| VQA pairs (stage 2) | VQA-v2 subset, 10 k QA | ~2 GB | Natural question/answer phrasing |
| Audio pairs (stage 3) | LibriSpeech `dev-clean` + `test-clean` | ~1 GB | Word-level transcripts give exact supervision cheaply |
| **Bootstrapped** | SmolVLM-256M captions over **your own** photos | unlimited | Legal, free, personal, and cheap: the 256 M teacher captions ~1 image/s |
| Optional | WebSight / screenshots | varies | Teaches UI understanding — genuinely useful for "process any format" |

**Bootstrapping is the highest-leverage move:** run SmolVLM-256M over every image in your own
library once, store the captions as `MediaDoc.chunks` with `lane=VLM`, and you have a
domain-matched training set with zero labelling cost. It is knowledge distillation from a smaller
teacher into a smaller student, which is the only kind of distillation that makes sense here.

### 10.6 Memory profile of bridge training

| Resident | Bytes |
|---|---|
| CLIP ViT-B/32 frozen, bf16 | ~0.35 GB |
| Our `micro` LM (frozen in stage 1) + projector | ~0.12 GB |
| Activations, batch 8 × (512 + 49) tokens | ~0.35 GB |
| Projector gradients + Adam (1.3 M params) | ~0.02 GB |
| Framework | ~0.3 GB |
| **Peak** | **~1.2 GB** ✅ 3.4 GB headroom |

Note the asymmetry: **bridge training is far cheaper than pretraining**, because in stage 1 the
backbone does not need any stored activations. This is why Phase 10 can be completed in a single
evening even though Phase 8 needs an overnight run.

### 10.7 Evaluation and honest expectations

| Metric | Target | Measurement |
|---|---|---|
| Caption exact-ish match (teacher-forced, ROUGE-L vs SmolVLM teacher) | ≥ 0.25 | `mmar eval --tasks caption` |
| VQA accuracy on a 500-question held-out split | ≥ 25 % (random = ~1/1000 for open vocab; near-guessing is expected) | `mmar eval --tasks vqa` |
| Colour/count/proximity probes ("is the cat left of the dog?") | qualitative, ≥ 60 % on a 50-item hand-built set | `mmar eval --tasks probe` |
| Catastrophic forgetting check: TinyStories val ppl after stage 2 | ≤ 1.3× the Phase-8 value | `mmar eval --tasks ppl` |

**Honest expectation for `mmar-vl-micro`:** it will produce short, mostly-grammatical descriptions
containing genuinely correct coarse content (object presence, dominant colour, scene type) and
unreliable detail. It is a *working end-to-end multimodal transformer you trained yourself* — which
is the goal — not a useful image assistant. For useful image understanding, route to Track D's
SmolVLM-256M, which is one config line away.

---

## 11. Data strategy

### 11.1 The pretraining corpus decision

| Dataset | Tokens | Licence | Verdict |
|---|---|---|---|
| **TinyStories v2** | ~470 M | CDLA-Sharing-1.0 | ✅ **Chosen.** Purpose-built to show tiny models can produce coherent English; short stories fully exercise the 512-token context; 470 M tokens is ~2× the Chinchilla budget for `micro` |
| WikiText-103 | 103 M | CC BY-SA | ❌ Kept as the **negative control** in Phase 8, to *demonstrate* encyclopedic text is a poor fit at 11 M params |
| FineWeb-Edu sample | 10 B+ | ODC-By | ⚠️ Legal and high quality, but web-diverse text needs ~10× the parameters. Documented as out of scope |
| Your own documents | unlimited | yours | ✅ Excellent for the **SFT** stage (§9.5) |

**Why not your own corpus for pretraining:** at 11 M params and a ~5 h budget, training on a
heterogeneous personal corpus yields a model that is bad at everything. TinyStories first proves the
machinery works; personal data then makes SFT useful. Reversing that order is the most common way
this kind of project fails.

### 11.2 Tokenizer design

| Decision | Value | Rationale |
|---|---|---|
| Algorithm | **BPE from scratch**, trained on the corpus | Byte-level; no `tokenizers` training call, no borrowed GPT-2 vocabulary |
| Vocab size | **4 096** | Embedding table = 1.6 M params = 14 % of `micro`. 8 192 doubles that for ~0.05 bits/token of gain — a measured trade-off recorded in `docs/12_RESULTS.md` |
| Pre-tokenisation | GPT-2-style regex + `bytes_to_unicode` | Handles arbitrary Unicode and never raises on emoji |
| Special tokens | `<|pad|> <|bos|> <|eos|> <|user|> <|assistant|>` | 5 tokens; the last two are what make Phase 9 and Phase 10 possible |
| Compression target | ≥ 3.2 chars/token on TinyStories | Otherwise the effective context shrinks. Asserted in tests |

**Two-stage train/subset:** training BPE on 470 M tokens takes minutes and lots of RAM. Train on a
**200 MB random subset**, then re-apply the *frozen* merges to the full corpus. Standard practice, and
what keeps tokenizer training inside the memory budget.

### 11.3 Sharding

```
data/tinystories/
├── train.txt                 # raw, ~1 GB, downloaded once
├── val.txt                   # held-out 0.5 %, split by *document*, never by line
├── train.bin                 # uint16 memmap of token ids (~440 MB for 220M tokens)
├── train.idx                 # int64 offsets of document starts, for curriculum/debug
├── val.bin  val.idx
└── meta.json                 # {"vocab_size":4096, "n_tokens":220_000_000, "tokenizer_sha256":…}
```

- **`uint16`** because `vocab_size = 4 096 < 65 536`. Halves file size and I/O bandwidth — which
  matters because disk is a shared resource and a fanless machine has no thermal headroom to spare.
- **`np.memmap`, not a `Dataset` object.** The loader reads a random `(batch, seq_len+1)` window per
  step: no shuffling machinery, no worker processes, no RAM duplication. 440 MB on disk lives in the
  page cache, costing ~0 MB of RSS.
- **Split by document, not by line.** Line-splitting TinyStories leaks half-sentences across
  train/val and produces a fake val loss of ~1.0, which has convinced many people they had solved
  language modelling.
- **`meta.json` records the tokenizer hash.** A shard/tokenizer mismatch is the most common silent
  failure: training succeeds, the loss falls, and the output is gibberish. The loader refuses to run
  on a hash mismatch.

### 11.4 Ingestion-side data (Track B)

- **Idempotent:** `doc_id = sha256(content_hash)[:16]`, so re-ingesting a folder skips unchanged
  files. `EmbeddingCache` is keyed on `sha256(content) + model_name + model_version`.
- **Bounded:** `max_file_mb: 512`, `max_archive_members: 256`, `max_archive_depth: 2`,
  `max_archive_total_mb: 2048` — in config, enforced, tested. A zip bomb should exhaust config, not
  the machine.
- **Content-addressed assets:** keyframes and page renders go to
  `artifacts/assets/<sha256[:2]>/<sha256>.<ext>`, so two sources sharing an image extract it once.
- **Store:** sqlite (`docs`, `chunks`, `assets`, `embeddings_vec`, `bm25_terms`) + `.npy` for the
  dense matrix. One file to back up, one to inspect with `sqlite3`.

---

## 12. Serving architecture

### 12.1 The API surface

| Endpoint | Method | Body | Returns | Notes |
|---|---|---|---|---|
| `/health` | GET | — | `{status, version, uptime_s}` | Never loads a model |
| `/v1/memory` | GET | — | `MemorySnapshot` | RSS, MLX active/cache, headroom, resident models |
| `/v1/models` | GET | — | loaded + available handles with `peak_mb` | Drives the UI's "will this fit?" warning |
| `/v1/ingest` | POST | multipart file(s) or `{path}` | `MediaDoc` summary + progress | Never loads a heavy model |
| `/v1/ingest/async` | POST | `{paths[]}` | `job_id` + SSE progress | Folders; survives client disconnect |
| `/v1/embed` | POST | `{texts[]}` \| `{images[]}` | vectors | |
| `/v1/transcribe` | POST | audio file | `Transcript{text, segments[]}` | Unloads the embedder first |
| `/v1/vision` | POST | image + question | `{caption, answer}` | Unloads ASR first |
| `/v1/chat` | POST | `ChatRequest` | `ChatResponse` | Main path |
| `/v1/chat/stream` | POST | `ChatRequest` | SSE `token`/`citations`/`done` | The UI uses this |
| `/v1/search` | POST | `{query, k, mode}` | `Hit[]` with provenance | Retrieval without generation |
| `/v1/llm/*` | — | — | OpenAI-compatible shim | Lets existing tools talk to our model |

### 12.2 `ChatRequest` / `ChatResponse`

```python
class ChatRequest(BaseModel):
    message: str
    doc_ids: list[str] | None = None      # restrict retrieval
    modalities: list[str] | None = None   # restrict context, e.g. ["image", "pdf"]
    images: list[str] | None = None       # ad-hoc images not yet ingested
    audio: str | None = None              # ad-hoc audio path
    engine: Literal["scratch", "bridge", "instruct"] = "scratch"   # which track reasons
    max_tokens: int = 256
    temperature: float = 0.8
    stream: bool = False

class Citation(BaseModel):
    chunk_id: str; doc_id: str; source: str
    page: int | None; time_span: tuple[float, float] | None
    score: float; snippet: str            # <=200 chars, for the UI

class ChatResponse(BaseModel):
    text: str
    citations: list[Citation]
    engine: str                           # which model actually answered
    usage: dict[str, int]                 # prompt_tokens, completion_tokens
    timings: dict[str, float]             # retrieve_ms, assemble_ms, prefill_ms, decode_ms, tps
    memory: MemorySnapshot
```

`engine` and `timings` are not decoration: they are how you discover that a query spent 900 ms
retrieving and 20 s decoding, or that the answer silently came from Track D because the config had
fallen back.

### 12.3 Residency policy (`ModelRegistry` + `MemoryGuard`)

```python
class MemoryGuard:
    def available_mb(self) -> float: ...                    # from runtime.max_process_rss_mb
    def can_load(self, estimate_mb: float) -> bool: ...
    def enforce(self, estimate_mb: float, policy: Literal["raise", "evict", "warn"]) -> None: ...

class ModelRegistry:
    """LRU, capacity 1 for anything with peak_mb > 1000."""
    def acquire(self, key: str) -> ModelHandle: ...          # context manager; evicts as needed
    def release(self, key: str) -> None: ...
    def evict_all(self) -> None: ...                          # called between pipeline stages
    def snapshot(self) -> list[ResidentModel]: ...
```

Ordering rules the registry enforces, derived from §4.5:

1. **One oversized handle at a time.** `acquire` evicts every other handle with `peak_mb > 1000`.
2. **Idle unload after 120 s** (`unload_idle_after_s`). Without this, a batch that touched audio then
   images holds both resident forever.
3. **Explicit `evict_all()` between ingestion stages.** The pipeline calls it at each boundary rather
   than trusting reference counting — Python's GC is not a memory manager on a 8 GB machine.
4. **Estimates are declared, not measured, and verified after load.** Each handle declares
   `peak_mb`; after loading, `mx.metal.get_active_memory()` is checked against the estimate and a
   >30 % overrun logs a warning and updates the estimate for future runs.
5. **`mx.set_memory_limit(4200 MB)` and `mx.set_cache_limit(512 MB)` are set at import time** in
   `mmar.utils.memory`. The cache limit matters as much as the memory limit: MLX's free-page pool
   grows by default and is a common source of "why is a 500 MB model using 2 GB?"

---

## 13. Architecture decision records

| # | Decision | Alternatives rejected | Rationale | Consequence |
|---|---|---|---|---|
| **ADR-1** | Apple **MLX** as the single tensor framework | PyTorch+MPS; JAX; llama.cpp/GGUF | MLX is built for unified memory (zero-copy weights), has first-class `mlx-lm`/`mlx-vlm`/`mlx-whisper`, and *supports training*, which GGUF does not. A published MacBook comparison found PyTorch-MPS and MLX roughly equal for tiny models, so the deciding factor is ecosystem, not raw speed | The from-scratch model is MLX-only. Weights are `.safetensors`/`.npz`, so conversion is possible but not provided |
| **ADR-2** | **One canonical IR** (`MediaDoc`) rather than per-modality types | `Document`/`Image`/`Audio` class hierarchies; LangChain-style `Document` | One type keeps the chunker, embedder, retriever and prompt builder modality-agnostic. Hierarchies would force `isinstance` branching in every downstream module | Every extractor must fit its output into the IR — lossy for exotic formats (documented in each extractor's `meta`) |
| **ADR-3** | **11 M-param `micro` as the primary model; `small` documented and discouraged** | Train `small`; train 125 M | Chinchilla: 39 M params needs 780 M tokens ≈ 70 h here. 11 M params needs 220 M tokens ≈ 5 h and captures most of the quality | The roadmap centres on one overnight run, not two weeks of them |
| **ADR-4** | **NumPy reference implementation** kept in `src/` as a test oracle | Trust MLX; test only end-to-end loss | Silent numerical bugs (RoPE, masking, `softmax` precision) never show in the loss curve. `pytest -m parity` catches them in milliseconds | ~400 extra lines to maintain. Non-negotiable at this ambition level |
| **ADR-5** | **BPE trained from scratch**, vocab 4 096 | Borrow GPT-2's 50 257 vocab; `tokenizers` BPE | A borrowed tokenizer undercuts "from scratch"; 50 257 × 384 = 19.3 M params of embedding table would dominate an 11 M-param model | Own tokenizer artifact to version; shard/tokenizer hash check required |
| **ADR-6** | **Frozen pretrained encoders + trained projector** for multimodality | End-to-end multimodal training; pretrained VLM only | LLaVA recipe: only the fusion is trainable. Trainable params drop from billions to ~1.3 M, which is what makes it fit in 8 GB | Encoder weights are downloaded, so "from scratch" applies to fusion + decoder — and that is stated plainly |
| **ADR-7** | **Exact numpy vector search**, no FAISS/HNSW | faiss-cpu; hnswlib; sqlite-vec | 10 k chunks × 384 dims = 15 MB. Exact search is ~2 ms at 100 % recall. Approximate indexing is a dependency and a recall regression for no gain | Search is O(N·d). Re-evaluation trigger documented: >500 k chunks |
| **ADR-8** | **Apple Vision (`ocrmac`)** for OCR, not Tesseract/EasyOCR | pytesseract; easyocr; docling | Zero model weights, OS-managed memory, 30+ languages, decisively better than Tesseract on real screenshots. EasyOCR is slow on CPU and its MPS path is immature | macOS-only. `backend: none` fallback keeps ingestion runnable elsewhere |
| **ADR-9** | **Three reasoning engines behind one interface** (`scratch`/`bridge`/`instruct`) | Train the method and use only it | Lets Phases 1–4 produce something useful before Phases 5–10 finish, and makes model comparison a config flag rather than a code fork | `ChatResponse.engine` must always be surfaced so fallbacks are never invisible |
| **ADR-10** | **sqlite + `.npy`** for the store, not a vector DB / Postgres | Chroma; Qdrant; LanceDB; DuckDB | One file, zero daemon, inspectable with the `sqlite3` CLI, trivially backed up. A daemon on an 8 GB machine is a memory tax with no benefit at this scale | Manual migrations in `ir/migrate.py`; `schema_version` discipline required |
| **ADR-11** | **Fanless-aware training**: wall-clock budget + optional duty cycling | Just run it | A MacBook Air throttles; an 8 h run silently becomes 14 h, and a 70 h run is a foot-gun. A hard `wall_clock_budget_h` and `--thermal-relief` make the cost visible | Runs may stop before convergence and must be resumable — they are (`--resume auto`) |
| **ADR-12** | **Documentation-first repository** (this doc set) | Start coding immediately | The three asks (all formats, from scratch, multimodal, 8 GB, local) are mutually constraining. Writing the arithmetic first produced the `small`-preset finding *before* any compute was spent | Docs must stay in sync; every phase prompt requires updating `docs/12_RESULTS.md` |

---

## 14. Risk register

| # | Risk | Likelihood | Impact | Mitigation | Early-warning signal |
|---|---|---|---|---|---|
| R1 | Training **diverges** (loss NaN or rising) at lr = 6e-4 | Medium | High (hours lost) | Warmup 300 steps, grad-clip 1.0, fp32 loss, divergence guard aborting at `val > train + 1.5`, keep-last-3 checkpoints | `grad_norm` spiking above ~5 in the first 200 steps |
| R2 | **Memory pressure / swapping** during ingest or eval | High | High | `MemoryGuard` + `max_concurrent_models: 1` + `evict_all()` between stages; `make memory` before enabling a new lane | Decode tok/s dropping >2× mid-run; `memory_pressure` reporting "warn"; MLX cache above 512 MB |
| R3 | **Thermal throttling** silently doubles wall-clock | High | Medium | `--thermal-relief` (25 min on / 5 min off), `wall_clock_budget_h`, log the tok/s trend | tok/s declining >25 % over a 30-min window |
| R4 | **Tokenizer/shard mismatch** → fluent nonsense | Medium | High | Hash in `meta.json`, loader refuses on mismatch, `round_trip_ratio` test, Phase 8a smoke run | Loss falling while samples stay gibberish; compression ratio far from 3.2 chars/token |
| R5 | **Overfitting** on the smoke run mistaken for success | Medium | Medium | Compare *val* loss; reuse a fixed deterministic val batch; assert `val ≤ train + 0.5` in the run report | val/train gap > 0.8 while both fall |
| R6 | A pretrained **encoder download is larger than expected** | High (first run) | Low | `MMAR_MODEL_CACHE`, `mmar models pull` with a progress bar and a size table; downloads only in a dedicated step, never lazily mid-ingest | A first-run "hang" that is really a 3 GB download |
| R7 | **mlx-vlm / mlx-lm API churn** breaks wrappers | High | Medium | All framework calls isolated in `mmar/models/*`; pins in `pyproject.toml`; `tests/test_models_contract.py` runs one tiny inference per adapter | Import errors after `pip install -U` |
| R8 | **Apple Vision OCR unavailable** (non-macOS or API change) | Low | Medium | `ocr.backend: none` degrades to embedded-text-only for PDFs and the `meta` lane for images. Ingestion never fails | `errors` containing `ocr_unavailable` |
| R9 | **Scope creep** — ingest 30 formats before training anything | High | High | Roadmap order is fixed; Phase 8 cannot be reached by adding extractors. Phase exit gates are binary | Phase 8 not started after 3 weeks of adding formats |
| R10 | `mx.compile` retracing → no speed-up | Medium | Low | Step counter as a traced argument; benchmark with and without compile, record both | tok/s identical for `compile: true` and `false` |
| R11 | **Zip bomb / huge file** exhausts disk or memory | Low | High | `max_file_mb`, `max_archive_members`, `max_archive_depth`, `max_archive_total_mb`; all tested with a synthetic bomb | Disk free falling during an ingest |
| R12 | Bridge model **catastrophically forgets** English | Medium | Medium | Freeze the LLM in stage 1; LoRA in stage 2; mix 20 % raw text into every SFT batch; explicit ppl regression gate (≤1.3×) | Captions degrading into repetition or losing grammar |

---

## 15. Performance envelope / SLOs

These are the numbers Phase 12 certifies. "Certified" means measured on this machine, written into
`docs/12_RESULTS.md` together with the command that produced it. **The canonical SLO list is the
union of this table and the quality/memory table in [P12](phases/P12_evaluation.md) — 33 rows; the
certification is incomplete until every row of both is filled.**

| Path | Metric | Target | Stretch | Notes |
|---|---|---|---|---|
| Ingest, text | throughput | ≥ 50 files/s | 200 files/s | Pure Python, no model |
| Ingest, PDF (text layer) | per page | ≤ 120 ms | 50 ms | pypdfium2 |
| Ingest, PDF (OCR) | per page | ≤ 900 ms | 400 ms | Apple Vision, 200 DPI render |
| Ingest, image (OCR + embed) | per image | ≤ 1.2 s | 0.5 s | |
| Ingest, audio 1 h | wall-clock | ≤ 6 min | 2 min | `whisper-base` first pass |
| Retrieval | p95 latency | ≤ 60 ms | 20 ms | 10 k chunks, exact numpy + BM25 |
| Chat, first token (`micro`) | TTFT | ≤ 400 ms | 150 ms | prompt ≤ 512 tokens |
| Chat, decode (`micro`) | tok/s | ≥ 60 | 150 | bandwidth-bound; ~22 MB of weights |
| Chat, decode (`instruct` 0.6 B 4-bit) | tok/s | ≥ 20 | 45 | ~0.4 GB of weights at 68 GB/s |
| Vision caption (SmolVLM-256M) | per image | ≤ 6 s | 2 s | |
| Peak RSS, ingest | MB | ≤ 1200 | 700 | Per §4.5 scenarios |
| Peak RSS, training `micro` | MB | ≤ 1200 | 800 | |
| Peak RSS, any single request | MB | ≤ 2500 | 1500 | The hard `MemoryGuard` threshold |
| Idle RSS, server with no model loaded | MB | ≤ 250 | 150 | Proves the residency policy works |
| Cold start (launch → ready) | seconds | ≤ 5 | 2 | No model loaded |

**The two SLOs that actually protect the user experience:** `Peak RSS, any single request
≤ 2500 MB` and `Idle RSS ≤ 250 MB`. If either regresses the machine starts swapping and *everything*
gets slow, which is far worse than one endpoint being slow.

---

## 16. Non-goals

Explicitly out of scope. Each was considered and rejected with a reason, so that a future
contributor (or a future you) does not rediscover them as "obviously missing features".

| Non-goal | Why not |
|---|---|
| Training a model competitive with any frontier LLM | Physically impossible here: ~10⁴× the compute. The goal is ownership of the stack, not benchmark parity |
| Multilingual *generation* | `micro` trains on English TinyStories. Multilingual *retrieval* is supported via `multilingual-e5-small` embeddings |
| Speech *output* (TTS) | Doubles the dependency surface for no gain in a retrieval/generation pipeline. Documented as a possible Phase 13 extension |
| Video *understanding* (temporal reasoning) | Keyframes + audio are extractable and searchable; temporal reasoning needs a video encoder that does not fit in 8 GB |
| Real-time streaming ASR (microphone) | A different concurrency model (ring buffer + partial hypotheses). Moonshine makes it feasible later; not in this roadmap |
| Fine-tuning Track D models beyond LoRA | QLoRA on a 0.6 B model is possible but drifts away from "from scratch". Documented as a Phase 13 extension |
| Distributed / multi-machine training | One machine, by definition |
| A polished production UI | The web UI is a debug surface that must display provenance, IR output and memory telemetry. Beauty is not a requirement |
| Cloud fallbacks of any kind | "Local" is a hard requirement. There is no API key in this repo and there never will be |
| Beating Apple Vision OCR with a custom OCR model | Training an OCR model needs 10³ GPU-hours and would still lose to the OS |

---

## Document status

| Item | State |
|---|---|
| Hardware figures (§3.1) | ✅ Measured on the target machine |
| Environment gaps (§3.3) | ✅ Verified (`ffmpeg` absent, Python 3.14 is the default) |
| Published reference throughputs (§3.2) | ✅ Sourced from named public reports |
| Preset parameter counts (§4.2) | ✅ Computed from the closed-form formula; asserted in `tests/test_model_shapes.py` |
| M1-Air TPS bands (§5.1) | ⚠️ **Estimates.** Replaced with measurements at the Phase 7 gate |
| Model peak-RSS column (§8) | ⚠️ Mixed: measured for LFM2.5-VL / SmolVLM, project-total for the track rows |
| Bridge metrics (§10.7) | ⚠️ Targets, not claims. Certified in Phase 12 |
| SLO table (§15) | ⚠️ Targets. Certified in Phase 12 |

**Next document:** [`docs/01_ROADMAP.md`](01_ROADMAP.md) — the phase plan that executes this design.

