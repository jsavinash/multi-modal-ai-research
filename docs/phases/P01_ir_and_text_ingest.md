# P1 — Canonical IR + Text & Structured Ingestion

| | |
|---|---|
| **Goal** | One canonical `MediaDoc` representation, plus extractors that turn every non-neural format into it without ever raising |
| **Wall-clock** | 2 evenings (splits naturally after task 5) |
| **Compute** | none |
| **Prerequisites** | P0 gate green |
| **Blocks** | P2, P3, P4 |
| **Read first** | [analysis §7](../00_ANALYSIS.md#7-canonical-ir-design), [§11.4](../00_ANALYSIS.md#114-ingestion-side-data), [§6.2](../00_ANALYSIS.md#62-end-to-end-data-flow), [docs/02 §2](../02_FUNCTION_REQUIREMENTS.md#2-l1--canonical-ir-and-ingestion) |

---

## Deliverables

```
src/mmar/ir/__init__.py  types.py  document.py  migrate.py
src/mmar/ingest/__init__.py  base.py  registry.py  pipeline.py
src/mmar/ingest/extractors/__init__.py  text.py  structured.py  code.py
                            web.py  pdf.py  office.py  archive.py  null.py
src/mmar/rag/__init__.py  chunker.py           # base chunkers (docs/02 §3.1); TemporalChunker is P3
tests/fixtures/                     # >= 20 real files, one per format family
tests/test_ir.py
tests/test_chunker.py
tests/test_extractor_contract.py
tests/test_extractors_{text,structured,code,web,pdf,office}.py
tests/test_archive.py
tests/test_registry_coverage.py
tests/test_pipeline.py
```

## Function requirements

Full signatures: [`docs/02 §2`](../02_FUNCTION_REQUIREMENTS.md#2-l1--canonical-ir-and-ingestion).

The five IR invariants (analysis §7.3) must be asserted by `assert_invariants()` and called from
`test_ir.py`. The extractor contract (docs/02 §2.2) is non-negotiable — in particular:

- **I5: extraction failure never raises.** `IngestionPipeline.ingest()` returns a `MediaDoc` with
  `errors` populated for *every* failure mode: corrupt file, permission error, file over
  `max_file_mb`, unsupported extension.
- `NullExtractor` always matches and always emits a metadata chunk, so an `UNKNOWN` doc is still
  findable by filename.
- `ExtractorRegistry.coverage()` returns `{extension: extractor_name}` for every known extension, and
  `test_registry_coverage.py` **parses the README Markdown table** and asserts the two agree.
- **`rag/chunker.py` is delivered in this phase** (docs/02 §3.1): `Chunker` protocol,
  `ChunkContext`, `get_chunker()`, `RecursiveChunker`, `FixedTokenChunker`, `CodeChunker`,
  `TabularChunker`, `SemanticChunker`, with `test_chunker.py` asserting the §3.1 invariants.
  (`TemporalChunker` is P3 and is explicitly not yours.)

## Per-format behaviour that must be implemented, not approximated

From [docs/02 §2.5](../02_FUNCTION_REQUIREMENTS.md#25-required-extractors-by-phase):

- subtitles → one chunk per cue with `time_span` populated
- CSV/XLSX → schema line + row-wise `col=val | col=val` text (never `repr(dict)`)
- code → definition-aware chunking with `meta["symbol"]`
- HTML → trafilatura main-content extraction, `<title>` in `meta`, links in `meta["links"]`
- PDF → per-page text layer first, OCR hook for pages under `min_chars_per_page`,
  `meta["page_lanes"]` recording which lane each page used
- archive → recursive expansion with **limits checked before extraction**, members re-dispatched
  through the same registry

## Forbidden

- Enabling OCR/ASR behaviour in this phase. The hook exists; P1 must pass with
  `ingest.ocr.enabled: false`. OCR is wired in P2.
- Any network access. `ingest.allow_network` defaults to false and nothing may bypass it.
- `pandas` — not in `pyproject.toml`, and stdlib `csv` + `openpyxl` cover the need.
- Choosing an extractor from the file extension alone.

## Tasks

1. `ir/types.py`, `ir/document.py`, `ir/migrate.py`; `test_ir.py` with all eight invariants and a
   synthetic v0 → v1 migration.
2. `ingest/base.py` (`Extractor` protocol + `ExtractContext`), `ingest/registry.py`,
   `ingest/extractors/null.py`, plus a contract test that feeds every registered extractor a garbage
   file and asserts a `MediaDoc` comes back rather than an exception.
3. `ingest/pipeline.py` with the five ordering guarantees from docs/02 §2.4, including
   `guard.evict_all()` at stage boundaries.
4. `rag/chunker.py` — the **base** chunkers (`RecursiveChunker`, `FixedTokenChunker`,
   `CodeChunker`, `TabularChunker`, `SemanticChunker`; `TemporalChunker` is P3) plus
   `test_chunker.py` asserting the docs/02 §3.1 invariants. The pipeline cannot emit `Chunk`s
   without this file; P3 and P4 treat it as existing code.
5. Build `tests/fixtures/` — at least 20 small real files including the hostile cases: zero-byte file;
   `.txt` containing PDF magic bytes; nested zip whose member is a PDF; CSV with embedded newlines and
   a quoted delimiter; UTF-16 text; scanned-image PDF; a file with no extension. Generate tiny ones in
   `conftest.py` if a real binary would be unreasonable, but keep at least 8 as real files.
6. Text + structured + code extractors with tests. **Commit.**
7. Web + PDF + office + archive extractors with tests, including the nested-zip case and a synthetic
   zip bomb config refuses. **Commit.**
8. `test_registry_coverage.py` and `test_pipeline.py` (whole-fixtures ingest, zero exceptions).
9. Run the gate; measure peak RSS.

## Exit gate

```bash
make test-fast
python -m mmar.cli ingest tests/fixtures
python -m mmar.cli formats
/usr/bin/time -l python -m mmar.cli ingest tests/fixtures   # report max RSS
```

**Peak RSS during the full-fixtures ingest under 400 MB.**

## Definition of Done

- [ ] All fixtures produce a `MediaDoc`; zero unhandled exceptions; raw output pasted
- [ ] `assert_invariants()` passes for every produced doc
- [ ] Hostile fixtures produce populated `errors`, not crashes
- [ ] `test_registry_coverage.py` green — README and registry agree
- [ ] Peak RSS of the full ingest measured and reported
- [ ] `mmar formats` prints the coverage table
- [ ] README format table/extension list updated if any extension was added or removed (the
      `test_registry_coverage.py` test enforces this)

---

## 📋 Copy-paste prompt

```
ROLE
You are a senior ML systems engineer. You write code that is measured, not asserted.

CONTEXT - read these first, in this order
  1. docs/00_ANALYSIS.md  section 7 (IR + invariants I1-I8), section 6.2 (data flow),
     section 11.4 (ingestion data)
  2. docs/02_FUNCTION_REQUIREMENTS.md  section 2 in full
  3. docs/01_ROADMAP.md  section 5 (Definition of Done)
  4. Existing code: src/mmar/errors.py, src/mmar/config.py, src/mmar/utils/*

MACHINE
  MacBook Air M1, 8 GB unified memory, fanless. python3.11 in .venv.
  Peak RSS for the whole-fixtures ingest must stay UNDER 400 MB. Measure it.

HARD CONSTRAINTS
  - Every extractor MUST degrade gracefully. A corrupt, empty, mislabelled or unsupported
    file produces a MediaDoc with MediaDoc.errors populated (modality UNKNOWN where
    appropriate). It must NEVER raise out of IngestionPipeline.ingest().
  - Extraction must never modify, move or delete the input path.
  - No network access. ingest.allow_network defaults to false.
  - No pandas. Use stdlib csv plus openpyxl.
  - Never choose an extractor from the file extension alone - sniff magic bytes.

DELIVERABLES - exactly these paths
  src/mmar/ir/{__init__,types,document,migrate}.py
  src/mmar/ingest/{__init__,base,registry,pipeline}.py
  src/mmar/ingest/extractors/{__init__,text,structured,code,web,pdf,office,archive,null}.py
  src/mmar/rag/{__init__,chunker}.py
  tests/fixtures/                     (>= 20 real files - see task 5)
  tests/test_ir.py
  tests/test_extractor_contract.py
  tests/test_extractors_{text,structured,code,web,pdf,office}.py
  tests/test_archive.py
  tests/test_registry_coverage.py
  tests/test_pipeline.py

KEY REQUIREMENTS
  - MediaDoc / Asset / Chunk / DocStats exactly as specified in docs/00_ANALYSIS.md
    section 7.2. JSON round-trip must be lossless (I6). schema_version plus migrate()
    wired from the start (I7).
  - assert_invariants(doc) implements I1-I8 and is called by tests.
  - Extractor protocol: name, priority, modalities, extensions, supports(),
    estimate_peak_mb(), extract(). estimate_peak_mb must be a conservative OVER-estimate.
  - ExtractorRegistry.coverage() -> {extension: extractor_name}. test_registry_coverage.py
    must PARSE the README format table Markdown and assert the two agree.
  - NullExtractor always matches: UNKNOWN modality, errors=["unsupported_format: <mime>"],
    plus one metadata chunk (filename, size, mtime) so the file is at least findable.
  - IngestionPipeline: never raises for per-file problems; calls guard.evict_all() at every
    stage boundary; idempotent via doc_id and content hash; streams progress; fails fast on
    bad CONFIGURATION (e.g. unwritable index_dir) at construction, not on file 380.
  - rag/chunker.py lands in THIS phase: Chunker protocol, ChunkContext, get_chunker(),
    RecursiveChunker, FixedTokenChunker, CodeChunker, TabularChunker, SemanticChunker
    (TemporalChunker is P3, NOT yours). Invariants from docs/02 section 3.1:
    min_chunk_tokens <= token_count <= 2*target_tokens (except a document's final chunk
    and a flagged oversize table row); overlap only on RecursiveChunker. test_chunker.py
    asserts them.
  - Format specifics that must be implemented, not approximated:
      subtitles  -> one chunk per cue with time_span populated
      csv/tsv    -> schema line plus row-wise "col=val | col=val" text (never repr(dict))
      xlsx       -> per-sheet schema plus row text, meta["sheet"]
      code       -> definition-aware chunking, meta["symbol"]
      html       -> trafilatura main content, <title> in meta, links in meta["links"]
      pdf        -> text layer first; OCR only when a page has fewer than
                    ingest.ocr.min_chars_per_page chars; meta["page_lanes"] records each
                    page's lane. OCR itself is DISABLED in P1 (ingest.ocr.enabled false) -
                    wire the hook only.
      archive    -> recursive, with depth+count+size limits enforced BEFORE extraction;
                    members re-dispatched through the SAME registry

FORBIDDEN
  - Files containing "# TODO" or "raise NotImplementedError".
  - `except Exception: pass`.
  - Reading a whole large file into memory when a streaming path exists.
  - Adding OCR or ASR behaviour in this phase (that is P2 and P3).
  - Reporting a test or command as passing without pasting its raw output.

TASKS
  1. ir/types.py, ir/document.py, ir/migrate.py; test_ir.py covering I1-I8, JSON round-trip,
     and a synthetic v0 document migration.
  2. ingest/base.py, ingest/registry.py, ingest/extractors/null.py, plus a contract test that
     feeds every registered extractor a garbage file and asserts a MediaDoc comes back.
  3. ingest/pipeline.py with the five ordering guarantees.
  4. rag/chunker.py - BASE chunkers only (RecursiveChunker, FixedTokenChunker, CodeChunker,
     TabularChunker, SemanticChunker; TemporalChunker is P3) + tests/test_chunker.py asserting
     the docs/02 section 3.1 invariants. The pipeline cannot emit Chunks without this file;
     P3 and P4 treat it as existing code.
  5. Build tests/fixtures/ with at least 20 small real files. MUST include the hostile cases:
     zero-byte file; .txt containing PDF magic bytes; nested zip whose member is a PDF; CSV
     with embedded newlines and a quoted delimiter; UTF-16 text file; a scanned-image PDF; a
     file with no extension. Generate tiny ones programmatically in conftest.py if a real
     binary would be unreasonable, but keep at least 8 as real files on disk.
  6. text/structured/code extractors plus tests. COMMIT.
  7. web/pdf/office/archive extractors plus tests, including the nested-zip case and a
     synthetic zip bomb that config refuses. COMMIT.
  8. test_registry_coverage.py and test_pipeline.py (whole-fixtures ingest, zero exceptions).
  9. Run the gate. Measure peak RSS.

EXIT GATE - paste the complete unedited output
  make test-fast
  python -m mmar.cli ingest tests/fixtures
  python -m mmar.cli formats
  /usr/bin/time -l python -m mmar.cli ingest tests/fixtures

EVIDENCE REQUIRED IN YOUR RESPONSE
  1. Raw output of make test-fast (summary line and any failures).
  2. Raw output of the full-fixtures ingest, including the IngestReport summary box.
  3. The output of `mmar formats`.
  4. The max-RSS number from /usr/bin/time -l, compared against the 400 MB budget.
  5. A table: fixture filename -> modality -> n_chunks -> lane -> errors (or "none").
  6. Anything you could NOT verify, stated explicitly as UNVERIFIED.

Before writing code, answer in at most three lines each: what is the exit gate; which
command proves it; the most likely way to get this silently wrong.
```

