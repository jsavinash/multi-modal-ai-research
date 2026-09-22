# 02 — Function Requirements

Every class and function the implementation must provide, with signatures, contracts, invariants and
error behaviour. Phase prompts reference this document by anchor; this file is the single source of
truth for interfaces.

---

## 0. Conventions

| Convention | Rule |
|---|---|
| Type hints | Mandatory on every public signature. `from __future__ import annotations` at the top of every module |
| Signature style | `X \| None`, `list[str]`, `dict[str, Any]` — PEP 604/585, no `Optional`/`List` |
| Dataclasses | `@dataclass(slots=True)`; `frozen=True` for value types (`Asset`, `Chunk`, `Hit`) |
| Paths | Accept `str \| Path`, normalise to `Path` internally, **never** mutate the input file |
| Config | Every tunable comes from a validated `Settings`. **No magic numbers in function bodies** |
| Logging | `mmar.utils.log.get_logger(__name__)`. No `print()` in library code |
| Errors | Custom hierarchy in `mmar.errors`. Library raises; the **pipeline** catches and records |
| Purity | Extractors and chunkers are pure: same bytes → same `MediaDoc`. No global mutable state |
| Determinism | Identical inputs produce identical outputs; randomness seeded from `settings.project.seed` |
| Memory | Functions that may allocate >200 MB take an explicit budget or `max_mb: float \| None` |
| Async | The pipeline is synchronous. FastAPI endpoints call sync code via `run_in_threadpool`. **No `async def` inside `mmar.llm` or `mmar.ingest`** |

### 0.1 Error taxonomy (`mmar/errors.py`)

```python
class MmarError(Exception):
    """Base class. Carries a machine-readable `code` and a human `hint`."""

class UnsupportedFormat(MmarError):        # no extractor for this MIME/extension
class CorruptInput(MmarError):             # parser raised on malformed data
class ResourceLimitExceeded(MmarError):    # max_file_mb, archive limits, token budget
class MemoryBudgetExceeded(MmarError):     # MemoryGuard refused a load
class ModelUnavailable(MmarError):         # weights missing / backend import failed
class TokenizerMismatch(MmarError):        # shard/tokenizer hash disagreement
class CheckpointMismatch(MmarError):       # config vs checkpoint architecture mismatch
class NotTrained(MmarError):               # asked to generate before any checkpoint exists
```

**Contract:** `MmarError` subclasses never escape `IngestionPipeline.ingest()` — they are caught and
converted into `MediaDoc.errors`. They **do** propagate out of `mmar.llm` and `mmar.models`, because
those are developer-facing APIs where a silent failure is worse than a stack trace.

---

## 1. L0 — Foundation

### 1.1 `mmar/config.py`

```python
class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MMAR_", env_nested_delimiter="__")
    project: ProjectSettings; runtime: RuntimeSettings; ingest: IngestSettings
    embedding: EmbeddingSettings; rag: RagSettings; llm: LlmSettings
    training: TrainingSettings; generation: GenerationSettings; serve: ServeSettings

def load_settings(config_path: str | Path | None = None, **overrides: Any) -> Settings:
    """Load YAML (resolving the `extends:` chain) then apply env vars then explicit overrides.

    Precedence: explicit overrides > MMAR_* env vars > YAML > field defaults.
    `extends: default.yaml` resolves relative to the including file, depth <= 3, with cycle
    detection. Raises MmarError on unknown keys (model_config extra="forbid").
    """

def resolve_preset(settings: Settings, name: str | None = None) -> ModelConfig:
    """Materialise `llm.presets[name or llm.active]` into a ModelConfig dataclass."""
```

**Required validations** (each raises `ValidationError`, never a bare `KeyError`):

- `runtime.warn_process_rss_mb < runtime.max_process_rss_mb`
- `llm.presets[*].d_model % n_heads == 0` and `n_heads % n_kv_heads == 0`
- `training.seq_len <= llm.presets[active].max_seq_len`
- `rag.max_context_tokens < llm.presets[active].max_seq_len`
- `training.micro_batch_size * training.grad_accum_steps * training.seq_len == training.tokens_per_step`
- `embedding.dim` matches what the configured embedder actually produces (lazy check, loud log on mismatch)

### 1.2 `mmar/utils/memory.py`

```python
@dataclass(slots=True, frozen=True)
class MemorySnapshot:
    process_rss_mb: float; peak_rss_mb: float
    mlx_active_mb: float | None; mlx_cache_mb: float | None; mlx_peak_mb: float | None
    system_available_mb: float
    pressure: Literal["normal", "warn", "critical"]
    resident_models: list[str]
    def render(self) -> str: ...        # the table printed by `make memory`
    def to_dict(self) -> dict[str, Any]: ...

def configure_mlx_limits(cache_mb: int, memory_mb: int) -> None:
    """Set mx.set_cache_limit / mx.set_memory_limit. Idempotent; safe at import time."""

def rss_mb() -> float: ...
def peak_rss_mb() -> float: ...
def snapshot() -> MemorySnapshot: ...

@contextmanager
def peak_tracker(label: str) -> Iterator[MemorySnapshot]:
    """Measure peak RSS across a block, log it, yield the final snapshot. Used by every phase gate."""

class MemoryGuard:
    def __init__(self, settings: Settings) -> None: ...
    def available_mb(self) -> float: ...
    def can_load(self, estimate_mb: float) -> bool: ...
    def enforce(self, estimate_mb: float, *, policy: Literal["raise","evict","warn"] = "raise") -> None: ...
    def register(self, key: str, mb: float) -> None: ...
    def unregister(self, key: str) -> None: ...
```

**Why it exists:** RSS is the only honest signal. On this machine a regression from 1.2 GB to 2.6 GB
peak fails no test — it just makes everything slower, and you will blame the model. `peak_tracker`
turns that into a printed number at every phase gate.

### 1.3 `mmar/utils/sniff.py`

```python
def sniff_mime(path: str | Path, *, declared: str | None = None) -> str:
    """Magic bytes > extension > declared hint. Never trusts the extension alone.
    Returns "application/octet-stream" when unknown, never raises."""

def is_probably_text(path: str | Path, probe_bytes: int = 8192) -> bool:
    """Null-byte and UTF-8 decodability heuristic."""

def looks_like_html(data: bytes) -> bool: ...
def looks_like_pdf(data: bytes) -> bool: ...
```

### 1.4 `mmar/utils/io.py`

```python
def sha256_file(path: str | Path, chunk_bytes: int = 1 << 20) -> str: ...
def sha256_bytes(data: bytes) -> str: ...
def atomic_write_text(path: str | Path, text: str, encoding: str = "utf-8") -> None: ...
def atomic_write_json(path: str | Path, payload: Any, *, indent: int = 2) -> None: ...
def read_json(path: str | Path) -> Any: ...
def ensure_dir(path: str | Path) -> Path: ...
def human_bytes(n: int | float) -> str: ...
def content_addressed_path(root: Path, digest: str, suffix: str) -> Path:
    """root/<digest[:2]>/<digest><suffix> — the on-disk layout for extracted assets."""
def decode_text(data: bytes, *, encodings: Sequence[str] = ("utf-8","utf-16","latin-1")) -> tuple[str, str]:
    """Return (text, encoding_used). Latin-1 is the never-fails fallback."""
def strip_control_chars(text: str) -> str: ...
def normalise_whitespace(text: str, *, preserve_indentation: bool = False) -> str:
    """Collapse runs of >2 newlines and trailing spaces; preserves code indentation when asked."""
```

### 1.5 `mmar/utils/log.py`

```python
def get_logger(name: str) -> logging.Logger: ...
def configure_logging(level: str = "INFO", *, rich_output: bool = True, log_file: Path | None = None) -> None: ...
def log_phase(phase: str, message: str) -> None:
    """Emits a visually distinct banner. Used at every phase boundary in the CLIs."""
```

---

## 2. L1 — Canonical IR and ingestion

### 2.1 `mmar/ir/` — types

The dataclasses are specified verbatim in [`docs/00_ANALYSIS.md §7.2`](00_ANALYSIS.md#72-the-types).
Additional required functions:

```python
# mmar/ir/document.py
def make_doc_id(source: str, content: bytes | None = None) -> str:
    """sha256(source + b'\\x00' + (content or b''))[:16]. Stable across runs; content-sensitive."""

def to_json(doc: MediaDoc) -> str: ...
def from_json(payload: str) -> MediaDoc: ...                     # migrates on read (I7)
def doc_from_text(text: str, *, source: str, modality: Modality, mime: str,
                  lane: ExtractLane = ExtractLane.NATIVE, meta: dict | None = None) -> MediaDoc: ...
def merge_docs(docs: Sequence[MediaDoc]) -> MediaDoc:
    """Combine extraction passes over the same source (e.g. PDF text pass + OCR pass).
    Precedence: NATIVE > OCR > ASR > VLM > META. Dedupes chunks by char_span."""
def truncate_to_tokens(doc: MediaDoc, budget: int, counter: "TokenCounter") -> MediaDoc: ...
def assert_invariants(doc: MediaDoc) -> None:
    """Raises AssertionError on any violation of I1..I8. Called in tests, cheap enough for dev."""

# mmar/ir/migrate.py
CURRENT_SCHEMA_VERSION: int = 1
def migrate(payload: dict[str, Any]) -> dict[str, Any]:
    """Apply migrations stepwise v_n -> v_n+1 until CURRENT. Idempotent."""
```

### 2.2 `mmar/ingest/base.py` — the extractor contract

```python
@dataclass(slots=True, frozen=True)
class ExtractContext:
    settings: Settings
    work_dir: Path               # scratch space; cleared per document
    asset_root: Path             # content-addressed output for extracted assets
    token_counter: TokenCounter  # for token_count on chunks; never loads a model
    registry: "ModelRegistry"    # for the neural lane; extractors must not hold handles
    deadline_s: float | None = None   # soft limit; long extractors check it and bail gracefully

class Extractor(Protocol):
    name: str
    priority: int                # higher wins when several match; specific > generic
    modalities: frozenset[Modality]
    extensions: frozenset[str]   # {".pdf", ".PDF"} normalised lowercase

    def supports(self, mime: str, path: Path) -> bool: ...
    def estimate_peak_mb(self, path: Path) -> float: ...   # used by MemoryGuard before running
    def extract(self, path: Path, ctx: ExtractContext) -> MediaDoc: ...
```

**Contract for every `Extractor`:**

1. `extract` **may raise** `MmarError` subclasses. That is allowed and encouraged — a specific error
   beats a generic one. The pipeline converts it to `MediaDoc.errors`.
2. `extract` must **never** modify, move, or delete the input path.
3. `extract` must populate `doc.stats.extract_ms` and set `doc.modality`/`doc.mime` correctly even on
   the failure path (return a doc with `errors` set, do not return `None`).
4. `estimate_peak_mb` must be a **conservative over-estimate**; it is the input to `MemoryGuard.enforce`.
   Returning 0 for a lane that loads a 700 MB model is a bug.
5. The neural lane is opt-in. An extractor must check `settings.ingest.<lane>.enabled` and produce
   useful output without it.

### 2.3 `mmar/ingest/registry.py`

```python
class ExtractorRegistry:
    def register(self, extractor: Extractor) -> None: ...
    def unregister(self, name: str) -> None: ...
    def resolve(self, mime: str, path: Path) -> Extractor:
        """Highest-priority extractor whose supports() is true, else the NullExtractor."""
    def list_supported(self) -> dict[str, list[str]]:
        """{extractor_name: sorted extensions}. Powers `mmar formats` and the README table."""
    def coverage(self) -> dict[str, str]:
        """{extension: extractor_name} for every known extension. Used by a test that
        asserts README's format table matches reality (documentation drift guard)."""

def default_registry(settings: Settings) -> ExtractorRegistry:
    """Build the standard registry. Ordering encodes priority:
    archive > office > pdf > image > audio > video > code > html > structured > text > null."""

class NullExtractor:
    """Always matches. Emits UNKNOWN modality + errors=['unsupported_format: <mime>'] +
    a metadata-only chunk (filename, size, mtime). Guarantees the pipeline never raises."""
```

**Why `coverage()` exists:** the README's capability matrix is a promise. A test asserts the matrix
and the registry agree, so adding an extractor without documenting it — or documenting one that does
not exist — fails CI. That is how the documentation stays true.

### 2.4 `mmar/ingest/pipeline.py`

```python
@dataclass(slots=True)
class IngestReport:
    n_files: int = 0; n_ok: int = 0; n_failed: int = 0; n_skipped_unchanged: int = 0
    total_chunks: int = 0; total_tokens: int = 0
    by_modality: dict[str, int] = field(default_factory=dict)
    by_lane: dict[str, int] = field(default_factory=dict)
    slowest: list[tuple[str, float]] = field(default_factory=list)
    peak_rss_mb: float = 0.0; elapsed_s: float = 0.0
    def render(self) -> str: ...        # the box printed by `mmar ingest`

class IngestionPipeline:
    def __init__(self, settings: Settings, registry: ExtractorRegistry,
                 store: "MediaStore", guard: "MemoryGuard") -> None: ...

    def ingest(self, source: str | Path | bytes, *, hint_mime: str | None = None,
               filename: str | None = None, recursive: bool | None = None) -> MediaDoc:
        """Single item. NEVER raises MmarError - failures land in MediaDoc.errors."""

    def ingest_many(self, sources: Iterable[str | Path], *, workers: int = 1,
                    recursive: bool | None = None,
                    on_progress: Callable[[IngestReport], None] | None = None,
                    ) -> Iterator[MediaDoc]:
        """Streams docs as they complete; aggregates the report.

        workers=1 by default: on 8 GB there is no memory for process-level parallelism and the
        neural lanes are GPU-bound anyway. workers>1 parallelises only the cheap lanes, which is
        documented behaviour rather than a promise.
        """

    def expand(self, source: str | Path, *, depth: int = 0) -> Iterator[Path]:
        """Directory walk + archive member enumeration, honouring every ingest.* limit.
        Yields real files only; archive members are materialised into ctx.work_dir."""

    def _stage_result(self, doc: MediaDoc) -> MediaDoc:
        """chunk -> embed -> persist. Called after extraction, guarded by MemoryGuard."""
```

**Ordering guarantees the pipeline must provide:**

1. **Never raise** for a per-file problem. `KeyboardInterrupt` and `SystemExit` are the only
   exceptions allowed through.
2. **`guard.evict_all()` at every stage boundary** (after each file's extraction, before embedding,
   before the next file). This is the mechanism that keeps a mixed-modality folder inside budget.
3. **Idempotent:** if `store.has_doc(doc_id)` and the content hash matches, skip and count it in
   `n_skipped_unchanged`. `--force` re-ingests.
4. **Progress is streamed,** not buffered to the end. A 400-file ingest must show a live count.
5. **One bad file never aborts a batch** — but a bad *configuration* (e.g. an unwritable `index_dir`)
   fails at construction, not on file 380.

### 2.5 Required extractors by phase

| Module | Phase | Extensions | Lane | `estimate_peak_mb` |
|---|---|---|---|---|
| `extractors/text.py` | P1 | `.txt .md .rst .log .ini .cfg .tex .srt .vtt` | NATIVE | 20 |
| `extractors/structured.py` | P1 | `.csv .tsv .json .jsonl .ndjson .yaml .yml .toml .xml` | NATIVE | 60 |
| `extractors/code.py` | P1 | ~30 source extensions | NATIVE | 30 |
| `extractors/web.py` | P1 | `.html .htm .xhtml .mhtml` + `http(s)://` | NATIVE | 120 |
| `extractors/pdf.py` | P1→P2 | `.pdf` | NATIVE + OCR fallback | 60 / +400 with OCR |
| `extractors/office.py` | P1 | `.docx .pptx .xlsx .xls .odt .odp .ods .epub` | NATIVE | 120 |
| `extractors/archive.py` | P1 | `.zip .tar .gz .tgz .bz2 .7z` | recursive | 40 + members |
| `extractors/image.py` | P2 | `.png .jpg .jpeg .webp .bmp .tif .tiff .heic .gif` | OCR + embed (+VLM) | 260 / +800 with caption |
| `extractors/audio.py` | P3 | `.wav .mp3 .m4a .flac .ogg .aac .opus .aiff` | ASR | 210 / +1400 with turbo |
| `extractors/video.py` | P3 | `.mp4 .mov .mkv .webm .avi` | keyframes + ASR | 300 / +1400 |
| `extractors/null.py` | P1 | everything else | META | 10 |

**Per-extractor requirements that must be implemented, not glossed over:**

- `text.py`: auto-detect BOM/UTF-16; subtitle files produce **one chunk per cue with `time_span`
  populated**, because that is what makes video quotable.
- `structured.py`: CSVs become a schema line plus row-wise text (`col=val | col=val`); a naive
  `repr(dict)` destroys retrieval quality. JSON is flattened to dotted paths with a depth cap of 6.
- `code.py`: chunk on top-level definitions (`def`/`class`/`fn`/`func`) via a regex fallback chain,
  never mid-function where avoidable. Sets `meta={"path":…, "symbol":…}` so citations name the symbol.
- `web.py`: `trafilatura.extract(output_format="txt")` to strip boilerplate; keep `<title>`; record
  absolute links in `meta["links"]`. Live URLs are fetched **only if `ingest.allow_network: true`** —
  default off, because "local" is a hard requirement.
- `pdf.py`: per page, use the embedded text layer; if `len(text.strip()) < min_chars_per_page`, render
  at `render_dpi` and OCR that page. Record per-page lanes in `meta["page_lanes"]` so a citation can
  honestly say "page 7 (OCR)".
- `office.py`: docx → paragraphs **plus table cells**; pptx → slide titles and bullets with
  `meta["slide"]`; xlsx → per-sheet schema plus row text with `meta["sheet"]`; epub → per chapter.
- `archive.py`: recursive with **depth and count limits enforced before extraction**, never after.
  Members are materialised into `ctx.work_dir` and re-dispatched through the *same registry*. A test
  ingests a nested zip containing a PDF and asserts the PDF text is found.
- `image.py`: OCR text as a `lane=OCR` chunk; the SigLIP/CLIP vector is stored with
  `meta={"space": "image"}` so dense retrieval never mixes text and image vectors in one matrix.
- `audio.py`: two passes — `ingest.audio.backend` for pass 1, `meta["asr_model"]` recorded;
  `mmar transcribe --upgrade <doc_id>` re-runs pass 2 with the turbo model.
- `video.py`: `scene_detect` via ffmpeg `select='gt(scene,0.4)'`, falling back to uniform sampling at
  `keyframes_per_minute` when fewer than 2 scenes are found. Audio is extracted to a temp `.wav` and
  pushed through the audio extractor **after the frame loop finishes** — sequential, not nested. That
  sequencing is the entire point of the memory policy.

---

## 3. L2 — Representation: chunking, embedding, storage

### 3.1 `mmar/rag/chunker.py`

> **Ownership:** `RecursiveChunker`, `FixedTokenChunker`, `CodeChunker`, `TabularChunker` and
> `SemanticChunker` land in **P1** (with `test_chunker.py`); `TemporalChunker` is added in **P3**.

```python
class Chunker(Protocol):
    name: str
    def chunk(self, doc: MediaDoc, ctx: ChunkContext) -> list[Chunk]: ...

@dataclass(slots=True, frozen=True)
class ChunkContext:
    target_tokens: int = 512
    overlap_tokens: int = 64
    min_chunk_tokens: int = 32
    counter: "TokenCounter" = ...        # exact token counts from our own tokenizer (P5+)
                                       # or a char/4 estimate before P5 exists
def get_chunker(modality: Modality, strategy: str, ctx: ChunkContext) -> Chunker: ...

class RecursiveChunker(Chunker):
    """Split on the largest separator that fits: paragraph -> sentence -> word -> char.
    Default for text, pdf, office, web, image-OCR."""

class FixedTokenChunker(Chunker): ...
class SemanticChunker(Chunker):
    """Embed adjacent sentences, split where cosine distance between neighbours exceeds
    a percentile threshold. Costs one embedder pass; opt-in only."""

class CodeChunker(Chunker):
    """Definition-aware. Never splits a function unless it exceeds 3x target_tokens."""

class TemporalChunker(Chunker):
    """Audio/video: merge consecutive ASR cues up to target_tokens, preserving
    (t_start, t_end) on every chunk so citations can deep-link a timestamp."""

class TabularChunker(Chunker):
    """Header + N rows per chunk; repeats the header in every chunk. Never splits a row."""
```

**Invariants:** every produced chunk satisfies `min_chunk_tokens <= token_count <= 2 * target_tokens`
(except a document's final chunk, which may be shorter, and a single oversized table row, which may
be longer and is flagged in `meta["oversize"]`). Overlap is applied only for `RecursiveChunker` and
`TemporalChunker`; overlap on code and tables produces duplicate symbols and duplicate rows.

### 3.2 `mmar/rag/embed.py`

```python
@dataclass(slots=True, frozen=True)
class EmbeddingResult:
    vectors: np.ndarray                    # (n, dim) float32
    dim: int
    model: str
    peak_mb: float

class Embedder(Protocol):
    name: str; dim: int; space: Literal["text", "image"]
    def estimate_peak_mb(self) -> float: ...
    def embed_texts(self, texts: Sequence[str], *, batch_size: int = 8) -> EmbeddingResult: ...
    def embed_images(self, paths: Sequence[Path], *, batch_size: int = 4) -> EmbeddingResult: ...

class MlxTextEmbedder(Embedder):     # mlx-embeddings, multilingual-e5-small
class ClipImageEmbedder(Embedder):   # SigLIP-base or CLIP ViT-B/32
class HashEmbedder(Embedder):
    """Dependency-free fallback: hashing trick into 384 dims. Deterministic, zero download,
    zero memory. Exists so P1-P4 run on a machine where MLX is broken. Quality is poor
    and this is stated in its docstring and in the README."""

class EmbeddingCache:
    """sqlite-backed, keyed on sha256(text|image_bytes) + model + dim.
    Hit rate on re-ingest must be ~100% for unchanged files, asserted in a test."""
    def get_many(self, keys: Sequence[str]) -> list[np.ndarray | None]: ...
    def put_many(self, items: Sequence[tuple[str, np.ndarray]]) -> None: ...

def normalise_rows(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise so cosine similarity is a dot product. Applied once at write time."""
```

**Design rules:** embeddings are **always L2-normalised at write time** so search is a plain dot
product; E5-family models require the `"query: "` / `"passage: "` prefixes and the embedder adds them
internally so no caller can forget; text and image vectors live in **separate matrices** keyed by
`space`, because mixing them silently destroys retrieval quality and is a classic unforced error.

### 3.3 `mmar/rag/store.py`

```python
class MediaStore:
    """sqlite + numpy. Tables: docs, assets, chunks, embeddings_text, embeddings_image,
    bm25_terms, jobs. One file, `indexes/mmar.sqlite3`."""

    def __init__(self, path: str | Path, *, read_only: bool = False) -> None: ...
    def migrate(self) -> None: ...
    def upsert_doc(self, doc: MediaDoc) -> None: ...            # transactional
    def upsert_chunks(self, chunks: Sequence[Chunk]) -> None: ...
    def has_doc(self, doc_id: str, *, content_hash: str | None = None) -> bool: ...
    def get_doc(self, doc_id: str) -> MediaDoc | None: ...
    def get_chunk(self, chunk_id: str) -> Chunk | None: ...
    def iter_chunks(self, *, doc_ids: Sequence[str] | None = None,
                    modalities: Sequence[Modality] | None = None,
                    lanes: Sequence[ExtractLane] | None = None,
                    batch_size: int = 512) -> Iterator[list[Chunk]]: ...
    def put_vectors(self, space: str, ids: Sequence[str], matrix: np.ndarray) -> None: ...
    def get_vectors(self, space: str) -> tuple[list[str], np.ndarray]:
        """Loads the whole matrix. At 10k x 384 this is 15 MB, which is why this API is
        acceptable and why there is no approximate index (ADR-7)."""
    def stats(self) -> StoreStats: ...
    def vacuum(self) -> None: ...
```

---

## 4. L3 — Retrieval

### 4.1 `mmar/rag/index.py`

```python
@dataclass(slots=True, frozen=True)
class Hit:
    chunk_id: str; doc_id: str; score: float
    text: str; snippet: str
    source: str; page: int | None; time_span: tuple[float, float] | None
    lane: ExtractLane; modality: Modality
    dense_score: float | None = None; bm25_score: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)

class DenseIndex:
    """Exact cosine over a normalised matrix. numpy only (ADR-7)."""
    def __init__(self, ids: list[str], matrix: np.ndarray) -> None: ...
    @classmethod
    def load(cls, store: "MediaStore", space: str = "text") -> "DenseIndex": ...
    def search(self, query: np.ndarray, k: int, *, mask: np.ndarray | None = None) -> list[tuple[str, float]]: ...
    def __len__(self) -> int: ...

class BM25Index:
    """Okapi BM25 (k1=1.2, b=0.75) with a purpose-built tokenizer, not sklearn.
    Postings live in sqlite so the index survives restarts without re-tokenising."""
    def build(self, store: "MediaStore") -> None: ...
    def load(self, store: "MediaStore") -> None: ...
    def search(self, query: str, k: int, *, mask: np.ndarray | None = None) -> list[tuple[str, float]]: ...

def reciprocal_rank_fusion(rankings: Sequence[Sequence[tuple[str, float]]], *, k: float = 60.0,
                           weights: Sequence[float] | None = None) -> list[tuple[str, float]]:
    """RRF: score(d) = sum_i w_i / (k + rank_i(d)).

    Chosen over score normalisation because dense cosine and BM25 scores are not on a
    comparable scale, and per-query min-max normalisation is unstable exactly when the
    top hit is an outlier. RRF needs no scale assumptions.
    """
```

### 4.2 `mmar/rag/retriever.py`

```python
class Retriever:
    def __init__(self, settings: Settings, store: "MediaStore", embedder_text: "Embedder",
                 dense: DenseIndex, bm25: BM25Index, guard: "MemoryGuard") -> None: ...

    def retrieve(self, query: str, *, k: int | None = None,
                 doc_ids: Sequence[str] | None = None,
                 modalities: Sequence[Modality] | None = None,
                 lane_deny: Sequence[ExtractLane] = (),
                 mode: Literal["hybrid", "dense", "bm25"] = "hybrid") -> list[Hit]:
        """1. embed the query once (the only model touch)
           2. dense top-2k and bm25 top-2k under the same mask
           3. RRF fuse (rag.bm25_weight weights the bm25 ranking)
           4. hydrate from sqlite, build snippets
           5. return exactly k, sorted by fused score"""

class Reranker(Protocol):
    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]: ...

class HeuristicReranker:
    """Cheap, no model: boosts exact-phrase, title and symbol matches; penalises lane=META.
    The default, because a cross-encoder costs 0.3 GB and ~40 ms per query."""

class CrossEncoderReranker:
    """Optional, opt-in via rag.rerank. Its memory cost is documented in the prompt."""

class ContextBuilder:
    def __init__(self, settings: Settings, counter: "TokenCounter") -> None: ...
    def build(self, hits: Sequence[Hit], *, budget_tokens: int, question: str,
              max_chunks: int | None = None) -> tuple[str, list["Citation"]]:
        """Assemble prompt context. Rules:
        - every chunk is prefixed with an ordinal citation marker [1], [2], ...
        - chunks that do not fit the budget are dropped whole, never truncated mid-chunk
        - the question is emitted last so it sits adjacent to the generation point
        - near-duplicate chunks (embedding cosine > 0.97) are deduped
        - Citation.snippet is the first 200 chars of the chunk, never a paraphrase"""
```

**Why RRF rather than weighted score fusion:** dense cosine and BM25 live on incomparable scales, and
per-query min-max normalisation is unstable precisely when the top hit is an outlier — the case you
care about most. RRF makes no scale assumptions and has one interpretable parameter.

---

## 5. L4 — Pretrained model wrappers (`mmar/models/`)

### 5.1 `mmar/models/registry.py`

```python
@dataclass(slots=True, frozen=True)
class ModelSpec:
    key: str                       # "vision.caption", "audio.asr", "embed.text"
    repo_id: str                   # HF repo id
    kind: Literal["vlm","asr","embed_text","embed_image","lm","enc_vision","enc_audio"]
    peak_mb: float                 # declared estimate; verified after first load
    quant: str | None = None       # "4bit" | "8bit" | None
    downloaded_bytes: int = 0

MODEL_CATALOG: dict[str, ModelSpec]     # analysis section 8's table, as data

class ModelHandle:
    key: str; spec: ModelSpec
    def load(self) -> Any: ...             # imports the backend lazily; raises ModelUnavailable
    def unload(self) -> None: ...          # drops the object AND calls mx.clear_cache()
    def __enter__(self) -> "ModelHandle": ...
    def __exit__(self, *exc: Any) -> None: ...

class ModelRegistry:
    def __init__(self, settings: Settings, guard: MemoryGuard) -> None: ...
    def is_available(self, key: str) -> bool: ...       # weights present in cache?
    def pull(self, key: str, *, progress: bool = True) -> Path:
        """Download with a progress bar and a pre-checked size. Raises ModelUnavailable with an
        actionable message if the disk cannot hold it."""
    def acquire(self, key: str) -> ModelHandle: ...
    def evict_all(self) -> None: ...
    def snapshot(self) -> list[ResidentModel]: ...
```

### 5.2 Backend adapters

```python
class TextGenModel(Protocol):
    name: str; peak_mb: float
    def generate(self, prompt: str, *, max_tokens: int, temperature: float, top_p: float,
                 top_k: int, stop: Sequence[str]) -> str: ...
    def stream(self, prompt: str, **kw: Any) -> Iterator[str]: ...

class ScratchTextGen(TextGenModel):     # wraps mmar.llm          - Track A
class MlxInstructTextGen(TextGenModel): # mlx-lm + Qwen3-0.6B-Instruct  - Track D
class BridgeTextGen(TextGenModel):      # wraps mmar.llm.bridge   - Track C

class VisionCaptioner(Protocol):
    def caption(self, image: Path, *, prompt: str = "Describe this image briefly.") -> str: ...
    def answer(self, image: Path, question: str) -> str: ...

class Transcriber(Protocol):
    def transcribe(self, audio: Path, *, language: str | None = None,
                   word_timestamps: bool = True) -> Transcript: ...

@dataclass(slots=True, frozen=True)
class TranscriptSegment:
    start: float; end: float; text: str; confidence: float | None = None

@dataclass(slots=True, frozen=True)
class Transcript:
    text: str; segments: list[TranscriptSegment]; language: str; model: str; duration_s: float
```

**Contract:** all adapters are **lazy**. Importing `mmar.models` must not import MLX, download
anything, or allocate a Metal buffer. Weights are resolved and buffers allocated only inside
`generate`/`caption`/`transcribe`/`embed_*`. A test asserts `import mmar.models; rss_mb() < 250`.

**Why the `Protocol`s rather than base classes:** the three text-generation engines have nothing in
common but their method signatures, and structural typing lets `ScratchTextGen`, `MlxInstructTextGen`
and `BridgeTextGen` be swapped by config without an inheritance hierarchy that would force imports of
MLX into the CLI's import path.

---

## 6. L4 — The from-scratch LLM (`mmar/llm/`)

**This is the heart of the project.** Everything in this section is implemented here: no
`transformers`, no `torch`, no copied vocabulary.

### 6.1 `mmar/llm/tokenizer.py`

```python
class BPETrainer:
    """Byte-level BPE. Trains on an iterator of strings so a 470 MB corpus never loads at once."""
    def __init__(self, vocab_size: int = 4096, special_tokens: Sequence[str] = (...),
                 min_frequency: int = 2, max_workers: int = 4) -> None: ...
    def train(self, corpus: Iterator[str], *, max_docs: int | None = None) -> BPEArtifacts: ...
    def save(self, out_dir: Path) -> Path: ...     # tokenizer.json (merges + vocab + specials)

@dataclass(slots=True, frozen=True)
class BPEArtifacts:
    merges: list[tuple[int, int]]
    vocab: dict[int, bytes]
    special_tokens: dict[str, int]

class Tokenizer:
    def __init__(self, artifacts: BPEArtifacts) -> None: ...
    @classmethod
    def load(cls, path: str | Path) -> "Tokenizer": ...
    @classmethod
    def train_new(cls, corpus: Iterator[str], *, vocab_size: int, out_dir: Path,
                  subset_bytes: int = 200 << 20, **kw: Any) -> "Tokenizer":
        """Trains merges on a bounded random subset, then applies the frozen merges to the
        full corpus in the encode path. Keeps tokenizer training inside the memory budget."""

    @property
    def vocab_size(self) -> int: ...
    @property
    def n_special(self) -> int: ...
    def encode(self, text: str, *, add_special: bool = False,
               allowed_special: Literal["none","all","none_raise"] = "all") -> list[int]: ...
    def decode(self, ids: Sequence[int], *, skip_special: bool = True) -> str: ...
    def encode_batch(self, texts: Sequence[str], **kw: Any) -> list[list[int]]: ...
    def encode_file(self, path: Path, *, chunk_chars: int = 1 << 20) -> Iterator[np.ndarray]:
        """Streams token ids for a large file without loading it. Feeds P8's shard builder."""
    def count(self, text: str) -> int: ...
    def compression_ratio(self, text: str) -> float:
        """len(text) / len(encode(text)). Must be >= 3.2 on the training corpus."""
    def encode_chat(self, turns: Sequence[tuple[str, str]], *, add_generation_prompt: bool = True,
                    mask_prompt: bool = True) -> tuple[list[int], list[int]]:
        """Return (input_ids, loss_mask) where loss_mask[i] == 1 only on assistant tokens.
        This is the function that makes Phase 9 correct."""
    def fingerprint(self) -> str:
        """sha256 over a canonical serialisation of merges+vocab+specials. Recorded in
        meta.json and in every checkpoint; mismatch raises TokenizerMismatch."""

def bytes_to_unicode() -> dict[int, str]: ...
def get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]: ...
```

**Required properties, each asserted by a test:**

- **Round-trip lossless:** `decode(encode(s)) == s` for ASCII, emoji, CJK, RTL, combining marks and
  embedded NUL bytes.
- **`encode("") == []`**, and `decode([]) == ""`.
- **Special tokens are never producible from ordinary text** — the pre-tokeniser must not be able to
  emit `<|user|>`, which would let user input forge a role boundary.
- **Determinism:** the same text always yields the same ids, on any machine.
- **`compression_ratio(corpus_sample) >= 3.2`** on the training corpus, else the vocab is too small
  and the effective context is smaller than `max_seq_len` implies.

### 6.2 `mmar/llm/config.py`

```python
@dataclass(slots=True, frozen=True)
class ModelConfig:
    vocab_size: int
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    max_seq_len: int = 512
    rope_theta: float = 10000.0
    norm_eps: float = 1e-5
    tie_embeddings: bool = True
    dropout: float = 0.0
    init_std: float = 0.02

    @property
    def head_dim(self) -> int: return self.d_model // self.n_heads
    @property
    def residual_scale(self) -> float: return (2 * self.n_layers) ** -0.5
    def n_params(self, *, exact: bool = True) -> int:
        """The closed-form count from analysis section 4.1. Asserted against the real
        module tree in tests/test_model_shapes.py - a mismatch means the architecture
        drifted from the documented arithmetic."""
    def to_dict(self) -> dict[str, Any]: ...
    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ModelConfig": ...
```

### 6.3 `mmar/llm/layers.py` + `mmar/llm/model.py`

```python
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...

def build_rope_cache(dim: int, max_seq_len: int, theta: float,
                     dtype: mx.Dtype) -> tuple[mx.array, mx.array]: ...
def apply_rope(x: mx.array, cos: mx.array, sin: mx.array, offset: int = 0) -> mx.array: ...
def causal_mask(q_len: int, kv_len: int, offset: int = 0) -> mx.array: ...
def scaled_dot_product_attention(q, k, v, mask: mx.array | None) -> mx.array:
    """Thin wrapper over mx.fast.scaled_dot_product_attention with a documented NumPy
    fallback used by ref/ and by parity tests."""

class KVCache:
    def __init__(self, n_layers: int, n_kv_heads: int, head_dim: int, max_seq_len: int,
                 dtype: mx.Dtype = mx.bfloat16) -> None: ...
    @property
    def offset(self) -> int: ...
    def update(self, layer_idx: int, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]: ...
    def reset(self) -> None: ...
    def nbytes(self) -> int: ...

class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None: ...
    def __call__(self, x, mask=None, cache: KVCache | None = None, layer_idx: int = 0) -> mx.array: ...

class SwiGLU(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None: ...
    def __call__(self, x: mx.array) -> mx.array: ...
    # gate = w1(x); up = w3(x); out = w2(silu(gate) * up)

class TransformerBlock(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None: ...
    def __call__(self, x, mask=None, cache=None, layer_idx: int = 0) -> mx.array: ...

class MMRTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None: ...
    def __call__(self, tokens: mx.array, *, cache: KVCache | None = None,
                 positions: mx.array | None = None) -> mx.array: ...
    def n_params(self) -> int: ...
    def loss(self, tokens: mx.array, targets: mx.array,
             loss_mask: mx.array | None = None) -> mx.array:
        """Cross-entropy in fp32, mean over masked positions only. Float32 is mandatory."""

class InputAdapter(Protocol):
    """What makes the bridge possible: a module that injects non-text tokens into the
    sequence at known positions. Implemented by mmar.llm.bridge.Projector."""
    def embed(self, batch: "ModalityBatch") -> tuple[mx.array, mx.array]:
        """Return (tokens[B, T, d_model], insert_mask[B, T])."""
```

**Forward-pass requirements:**

1. **`positions` are explicit.** Decoding with a KV cache must pass absolute positions so RoPE is
   applied at the right offset. Deriving positions from `cache.offset` internally is allowed, but the
   parameter must exist — it is what makes the KV-cache equivalence test possible.
2. **The mask is built from `(q_len, kv_len, offset)`**, never from a boolean of shape `(T, T)` that
   silently breaks when `q_len != kv_len` during cached decoding.
3. **`cache=None` and `cache=KVCache(...)` must give numerically identical logits** for the same
   input. Asserted to `atol=1e-4`. This single test catches the majority of KV-cache bugs.
4. **No `mx.eval` inside the model.** Synchronisation belongs to the training/inference loops.

### 6.4 `mmar/llm/data.py`

```python
@dataclass(slots=True, frozen=True)
class ShardMeta:
    vocab_size: int; n_tokens: int; tokenizer_fingerprint: str
    dataset: str; created_at: str; doc_offsets_file: str | None = None

@dataclass(slots=True)
class Batch:
    inputs: mx.array      # (B, T)   int32
    targets: mx.array     # (B, T)   int32
    mask: mx.array | None # (B, T)   float32; None during pretraining (all 1s)

class TokenShardWriter:
    """tokenize a corpus -> uint16 memmap + document offsets + meta.json (with the tokenizer hash)."""
    def __init__(self, out_dir: Path, tokenizer: Tokenizer) -> None: ...
    def add_document(self, doc_id: str, text: str, *, prepend_bos: bool = True,
                     append_eos: bool = True) -> None: ...
    def finalise(self) -> ShardMeta: ...

class MemmapDataset:
    """Random (B, T+1) windows over a uint16 memmap. Zero RAM duplication (page cache)."""
    def __init__(self, bin_path: Path, meta: ShardMeta, *, seq_len: int, batch_size: int,
                 seed: int = 1337, shuffle: bool = True) -> None: ...
    def __iter__(self) -> Iterator[Batch]: ...
    def __len__(self) -> int: ...
    def peek_batch(self, index: int = 0) -> Batch:
        """Deterministic batch, always the same windows. Used for comparable validation."""

def download_tinystories(out_dir: Path, *, variant: str = "v2") -> tuple[Path, Path]:
    """Returns (train_txt, val_txt). Split by *document*, never by line (analysis 11.3)."""
def build_shards(*, train_txt: Path, val_txt: Path, tokenizer: Tokenizer, out_dir: Path,
                 max_train_tokens: int | None = None, max_val_tokens: int | None = None) -> ShardMeta: ...
def load_shards(shard_dir: Path, *, expected_fingerprint: str | None = None) -> tuple[MemmapDataset, MemmapDataset]:
    """Raises TokenizerMismatch when expected_fingerprint disagrees with meta.json (R4)."""
```

### 6.5 `mmar/llm/optim.py`

```python
def cosine_schedule(step: int, *, warmup: int, max_steps: int, lr: float, min_lr: float) -> float: ...
def clip_global_norm(grads: dict, max_norm: float) -> tuple[dict, float]:
    """Returns (clipped_grads, pre_clip_norm). The norm must be logged every step -
    it is the earliest signal of divergence (risk R1)."""
def init_model(model: mx.nn.Module, cfg: ModelConfig, *, seed: int) -> mx.nn.Module:
    """std=0.02 for linear/embedding weights, zeros for norms, and residual output
    projections scaled by cfg.residual_scale. Getting this wrong is the #1 cause of
    'my from-scratch model will not train'."""
def build_optimizer(model: mx.nn.Module, cfg: TrainingSettings) -> tuple[optim.Optimizer, Any]:
    """AdamW with weight decay applied ONLY to >=2-D parameter matrices.
    tok_emb, all norm weights and every 1-D tensor are excluded (analysis 9.3, point 1)."""
def count_parameters(model: mx.nn.Module) -> dict[str, int]:
    """{'total': …, 'embedding': …, 'attention': …, 'mlp': …, 'norms': …}"""
```

### 6.6 `mmar/llm/train.py`

```python
@dataclass(slots=True)
class TrainReport:
    steps_completed: int; tokens_seen: int
    train_loss: float; val_loss: float; val_ppl: float; best_val_loss: float
    tokens_per_second: float; grad_norm_last: float; grad_norm_max: float
    peak_rss_mb: float; wall_clock_s: float
    aborted_reason: str | None          # "diverged" | "wall_clock_budget" | "nan"
    checkpoint_path: Path | None
    def render(self) -> str: ...        # the run summary printed at the end

class Trainer:
    def __init__(self, settings: Settings, *, resume: Literal["auto","never"] | Path = "never",
                 thermal_relief: bool = False, watch_memory: bool = False) -> None: ...
    def train(self) -> TrainReport: ...
    @staticmethod
    def train_step(model, optimizer, batch, *, cfg: TrainingSettings) -> tuple[mx.array, dict, dict]:
        """Pure, compiled with mx.compile. Returns (loss, grads, aux) where aux carries
        grad_norm. The step counter is an argument, never captured - see analysis 9.3 point 4."""
    def evaluate(self, model, val_iter: MemmapDataset, *, batches: int) -> float:
        """Eval mode, dropout off, deterministic windows only."""
```

**Enforced behaviours:**

| Behaviour | Why it is mandatory |
|---|---|
| `grad_norm` logged every `log_interval`, and the running max reported | Divergence (R1) is visible ~500 steps before it is fatal |
| Divergence guard: abort when `val_loss > train_loss + 1.5` for 3 consecutive evals | Distinguishes a data bug from under-training |
| `wall_clock_budget_h` hard stop, checkpoint written first | Fanless hardware makes an unbounded run a foot-gun (ADR-11) |
| `peak_tracker("train")` around the whole loop, reported in `TrainReport` | Memory regressions must fail the phase gate, not be discovered in production |
| Thermal relief: 25 min on / 5 min off, driven by `pmset -g therm` | Sustained throttling costs more than the pause |
| Resume must be **bit-exact**: `--resume auto` continues the LR schedule and RNG state | A resume that resets the LR schedule silently destroys a run |

### 6.7 `mmar/llm/sample.py`

```python
@dataclass(slots=True)
class SamplingParams:
    max_tokens: int = 256; temperature: float = 0.8
    top_k: int = 40; top_p: float = 0.95
    repetition_penalty: float = 1.1; repetition_window: int = 64
    stop: tuple[str, ...] = ("<|eos|>", "<|user|>")
    seed: int | None = None

def sample_ids(model, prompt_ids: Sequence[int], *, params: SamplingParams,
               cache: KVCache | None = None, tokenizer: Tokenizer) -> Iterator[int]:
    """Yields one token id at a time using the KV cache. Greedy when temperature == 0."""

def generate(model, tokenizer: Tokenizer, prompt: str, *, params: SamplingParams,
             on_token: Callable[[str], None] | None = None) -> str: ...

def beam_search(model, tokenizer, prompt: str, *, beam_width: int = 4, **kw: Any) -> list[tuple[str, float]]:
    """Optional. Documented as usually worse than sampling for creative text and better for
    templated extraction tasks, which is exactly what Phase 9 trains for."""
```

**Sampling requirements:**

1. **Never use a framework sampling helper without verifying it.** A documented MLX bug in
   `mx.random.categorical` produced broken sampling of otherwise-correct logits — the model was fine
   and the sampler was not. `tests/test_sample.py` asserts that with `temperature=0` the output
   equals `argmax`, and that with a fixed seed the output is reproducible.
2. **Top-k and top-p compose in that order** (mask, then nucleus), and the implementation is written
   out rather than delegated, so behaviour cannot change under you.
3. **Repetition penalty uses a window**, not the whole context. Penalising tokens from 400 steps ago
   is how small models get stuck in punctuation loops.
4. **Stop strings are matched on the decoded tail**, not per token, because a stop string can span a
   token boundary.

### 6.8 `mmar/llm/checkpoint.py`

```python
@dataclass(slots=True, frozen=True)
class CheckpointInfo:
    path: Path; step: int; val_loss: float; config: ModelConfig
    tokenizer_fingerprint: str; created_at: str; is_sft: bool; is_bridge: bool

class Checkpointer:
    def __init__(self, dir: Path, *, keep_last: int = 3) -> None: ...
    def save(self, model: mx.nn.Module, optimizer: Any | None, *, step: int, val_loss: float,
             cfg: ModelConfig, tokenizer_fingerprint: str, meta: dict[str, Any] | None = None) -> Path: ...
    def latest(self) -> CheckpointInfo | None: ...
    def load(self, path: Path, *, strict: bool = True) -> tuple[mx.nn.Module, Any | None, CheckpointInfo]:
        """Raises CheckpointMismatch with a readable diff when the stored ModelConfig
        does not match the requested one. Never silently load a mismatched architecture."""
    def prune(self) -> list[Path]: ...
```

**Checkpoint contents:** `model.safetensors` (bf16), `optimizer.safetensors` (fp32 moments),
`config.json` (ModelConfig), `meta.json` (step, val_loss, tokenizer fingerprint, git SHA, timestamps).
Storing the git SHA costs nothing and has saved countless debugging sessions.

### 6.9 `mmar/llm/ref/` — the NumPy oracle

```python
# mmar/llm/ref/numpy_ops.py  - deliberately naive, readable, correct. Not fast.
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray: ...
def rms_norm(x: np.ndarray, weight: np.ndarray, eps: float = 1e-5) -> np.ndarray: ...
def rope(x: np.ndarray, positions: np.ndarray, theta: float) -> np.ndarray: ...
def attention(q, k, v, *, mask: np.ndarray | None, n_kv_heads: int) -> np.ndarray: ...
def swiglu(x, w_gate, w_up, w_down) -> np.ndarray: ...
def cross_entropy(logits: np.ndarray, targets: np.ndarray, mask: np.ndarray | None = None) -> float: ...

# mmar/llm/ref/numpy_model.py
class NumpyTransformer:
    """Same architecture, same weights, no MLX. Forwards an (1, T) sequence and returns logits.
    Loaded from a checkpoint so parity can be checked on a *trained* model, not just random init."""
    def __init__(self, cfg: ModelConfig, weights: dict[str, np.ndarray]) -> None: ...
    def __call__(self, tokens: np.ndarray) -> np.ndarray: ...
```

**The parity contract:** for every test in `pytest -m parity`,
`abs(mlx_value - numpy_value).max() < 1e-4`, on inputs drawn from a fixed seed. Both on random init
and on a real checkpoint, because some bugs only appear once weights are not near-zero.

### 6.10 `mmar/llm/prompt.py`

```python
def format_qa(context: str, question: str, *, style: Literal["scratch", "instruct"] = "scratch") -> str:
    """The scratch style is deliberately terse and templated, because an 11 M-param model
    cannot follow prose instructions (analysis 9.5/9.6). The instruct style matches the
    Track-D model's chat template."""

def build_rag_prompt(context: str, question: str, citations: Sequence["Citation"],
                     *, engine: str, budget_tokens: int, counter: "TokenCounter") -> str: ...

def extract_citation_markers(text: str) -> list[int]:
    """Parse '[1]', '[2,3]' out of a generated answer so the response can report which
    sources the model actually used, not merely which ones were offered."""
```

---

### 6.11 SFT (`mmar/llm/sft.py`, `sft_data.py`, `eval_sft.py`)

```python
# mmar/llm/sft_data.py
@dataclass(slots=True, frozen=True)
class SFTPair:
    prompt: str; response: str; template_id: str; source_chunk_id: str | None

class SFTDatasetBuilder:
    """Build prompt/response pairs from the ingested store + a seed template set.
    Deterministic: same store + same settings -> byte-identical dataset.jsonl."""
    def __init__(self, settings: Settings, store: MediaStore, tokenizer: Tokenizer) -> None: ...
    def build(self, *, n_pairs: int, holdout_path: Path, n_holdout: int = 200) -> Path:
        """Writes the 200-prompt holdout FIRST (it must never leak into the train split),
        then artifacts/sft/dataset.jsonl. Returns the train-set path."""

# mmar/llm/sft.py
@dataclass(slots=True)
class SFTReport:
    steps_completed: int; train_loss: float; val_loss: float
    holdout_exact_match: float; holdout_rouge_l: float
    base_val_ppl: float; sft_val_ppl: float; forgetting_ratio: float
    tokens_per_second: float; peak_rss_mb: float; wall_clock_s: float
    def render(self) -> str: ...

class SFTTrainer:
    """Reuses llm.train.Trainer. The ONLY deltas: assistant-only loss mask and the
    text-mixing fraction. Everything else (grad clip + pre-clip norm logging,
    cosine schedule, divergence guard, bit-exact resume) comes from P7."""
    def __init__(self, settings: Settings, *, base_checkpoint: Path, dataset: Path,
                 holdout: Path, resume: Literal["auto", "never"] | Path = "never") -> None: ...
    def train(self) -> SFTReport: ...

# mmar/llm/eval_sft.py
def eval_sft(checkpoint: Path, holdout: Path, *, settings: Settings | None = None) -> dict[str, float]:
    """Returns {'exact_match': ..., 'rouge_l': ..., 'val_ppl': ..., 'forgetting_ratio': ...}.
    forgetting_ratio = sft_val_ppl / base_val_ppl; gate at 1.3 (risk R12)."""
```

**Invariants:** `loss_mask[i] == 1` only on assistant tokens (asserted by `test_sft_mask.py`);
holdout pairs are disjoint from train pairs by `template_id` **and** `source_chunk_id`;
20 % of every batch is raw pretraining text; `forgetting_ratio <= 1.3` or the phase fails.



```python
# mmar/llm/bridge/projector.py
class Projector(mx.nn.Module):
    """2-layer MLP + GELU, d_in -> 4*d_model -> d_model. The only trainable fusion module."""
    def __init__(self, d_in: int, d_model: int, *, n_layers: int = 2, dropout: float = 0.0) -> None: ...
    def __call__(self, features: mx.array) -> mx.array: ...

class PatchPooler(mx.nn.Module):
    """196 patch tokens -> 49 by 2x2 spatial merge (pixel-shuffle). Parameter-free.
    pool_factor is the TOKEN-COUNT divisor: 1 -> 196, 2 -> 98, 4 -> 49 tokens
    (on a 14x14 grid, 4 = 2x2 merge -> (7, 7)). Default 4 = analysis 10.3 default."""
    def __init__(self, pool_factor: int = 4, grid_hw: tuple[int, int] = (14, 14)) -> None: ...
    def __call__(self, patches: mx.array) -> mx.array: ...

# mmar/llm/bridge/encoders.py
class FrozenVisionEncoder(Protocol):
    d_out: int; n_patches: int
    def encode(self, images: Sequence[Path | np.ndarray]) -> mx.array: ...
class ClipVisionEncoder(FrozenVisionEncoder):
    """CLIP ViT-B/32 or SigLIP-base. Always eval mode, always no_grad. Raises
    MemoryBudgetExceeded via the registry if it cannot be resident."""

class FrozenAudioEncoder(Protocol):
    d_out: int
    def encode(self, audio: Path) -> mx.array: ...
class WhisperAudioEncoder(FrozenAudioEncoder):
    """Encoder half only. 1500 frames -> subsample 6x -> 250 -> mean-pool 5x -> 50 tokens."""

# mmar/llm/bridge/dataset.py
@dataclass(slots=True)
class MultimodalSample:
    image_paths: list[Path]; audio_paths: list[Path]
    turns: list[tuple[str, str]]          # [("user","<image> Describe this."), ("assistant", "A cat…")]
    source: str                            # "coco" | "vqa" | "librispeech" | "bootstrapped"

class MultimodalDataset:
    def __init__(self, samples: Sequence[MultimodalSample], tokenizer: Tokenizer,
                 *, seq_len: int, include_text_only_fraction: float = 0.2) -> None:
        """The text-only fraction matters: 20% raw text in every batch is what prevents
        catastrophic forgetting during stage 2 (risk R12)."""
    def __iter__(self) -> Iterator["ModalityBatch"]: ...

@dataclass(slots=True)
class ModalityBatch:
    input_ids: mx.array          # (B, T) text tokens, with a placeholder run at image positions
    targets: mx.array
    loss_mask: mx.array
    modality_tokens: mx.array | None   # (B, M, d_model) projected encoder outputs
    modality_mask: mx.array | None     # (B, T) which positions are replaced by modality tokens

# mmar/llm/bridge/train.py
@dataclass(slots=True)
class BridgeStageReport:
    stage: Literal[1, 2, 3]; steps: int; loss_start: float; loss_end: float
    trainable_params: int; peak_rss_mb: float; wall_clock_s: float

class BridgeTrainer:
    def __init__(self, settings: Settings, base_checkpoint: Path, *,
                 vision_repo: str, audio_repo: str | None = None) -> None: ...

    def stage1_align(self, dataset: MultimodalDataset, *, steps: int) -> BridgeStageReport:
        """Train the projector ONLY. LLM weights and all encoders frozen with mx.stop_gradient.
        LR 1e-3 (high - the projector starts random and the target space is fixed)."""

    def stage2_instruct(self, dataset: MultimodalDataset, *, steps: int) -> BridgeStageReport:
        """Unfreeze the projector + LoRA (r=8) on q_proj/v_proj/o_proj. LR 2e-4 cosine.
        20% text-only batches mixed in."""

    def stage3_audio(self, dataset: MultimodalDataset, *, steps: int) -> BridgeStageReport:
        """Train Projector 2 only, everything else frozen. LR 1e-3."""

    def merge_lora(self, out_path: Path) -> Path:
        """W+A merged back into the base weights so inference needs no LoRA code path."""

def freeze(module: mx.nn.Module) -> mx.nn.Module: ...
def apply_lora(model: mx.nn.Module, *, rank: int = 8, targets: Sequence[str] = ("q_proj","v_proj","o_proj")) -> mx.nn.Module: ...
def count_trainable(model: mx.nn.Module) -> int: ...
```

**Non-negotiable requirements:**

1. **Stage 1 must freeze the LLM.** Training both from the start makes the LM unlearn English while
   the projector emits noise (analysis §10.4).
2. **Encoders are always in eval mode and never receive gradients.** Asserted:
   `all(not p.requires_grad for p in encoder.parameters())`.
3. **Token budget is enforced before the forward pass.** If
   `n_text + n_image + n_audio > max_seq_len`, drop samples — raising mid-batch wastes the step.
4. **Every stage checkpoints separately** so stage 2 failure does not cost stage 1's work.
5. **The forgetting gate runs automatically** at the end of stage 2: evaluate TinyStories val ppl and
   fail loudly if it exceeds 1.3× the base value (risk R12).

---

## 8. L5/L6 — Orchestration, serving, CLI

### 8.1 `mmar/serve/orchestrator.py`

```python
class Orchestrator:
    """The only place that knows how to combine retrieval, modality handling and generation."""

    def __init__(self, settings: Settings, store: MediaStore, retriever: Retriever,
                 pipeline: IngestionPipeline, registry: ModelRegistry, guard: MemoryGuard) -> None: ...

    def chat(self, req: ChatRequest) -> ChatResponse:
        """Full path. Never raises MmarError - degrades to a response with `error` set."""

    def stream_chat(self, req: ChatRequest) -> Iterator[ChatEvent]:
        """Yields ('citations', ...) once, then ('token', str) repeatedly, then ('done', meta)."""

    def search(self, query: str, *, k: int, mode: str) -> list[Hit]: ...

    def ingest_and_answer(self, file: Path, question: str) -> ChatResponse:
        """Ad-hoc upload: ingest -> retrieve -> answer, without persisting unless asked."""

    def _select_engine(self, req: ChatRequest) -> TextGenModel:
        """Routing rules, in order:
        1. req.engine if explicitly set and available
        2. `bridge` if an image/audio was supplied and a bridge checkpoint exists
        3. `instruct` if the question needs synthesis across >2 chunks (heuristic on the
           assembled context), because `micro` cannot do that (analysis 9.6)
        4. `scratch` otherwise - the from-scratch model is the DEFAULT, not a fallback.
        The chosen engine is always reported in ChatResponse.engine."""

    def _residency_plan(self, req: ChatRequest) -> list[tuple[str, str]]:
        """Ordered (action, model_key) plan, e.g. [('acquire','audio.asr'),
        ('evict','audio.asr'), ('acquire','embed.text'), ('acquire','llm.instruct')].
        Encodes the max_concurrent_models: 1 rule as a plan instead of scattered unload calls."""
```

### 8.2 `mmar/serve/app.py`

```python
app: FastAPI
def create_app(settings: Settings | None = None) -> FastAPI: ...

# Dependency-injected and lazily built at startup, so create_app() alone allocates nothing.
def get_orchestrator() -> Orchestrator: ...
```

Requirements:

- **No model load on import or on `/health`.** Asserted by a test that imports the app, hits
  `/health`, and checks RSS < 250 MB.
- **SSE for `/v1/chat/stream`** via `sse-starlette`, with `event: token|citations|done` and a 15 s
  heartbeat so a long prefill does not look like a hang.
- **Uploads stream to a temp file** and are deleted after ingestion unless `persist=true`.
  `max_upload_mb` is enforced by a content-length check **and** a running byte count.
- **Errors are structured:** `{"error": {"code": "unsupported_format", "hint": …, "detail": …}}` with a
  status from a fixed map (`UnsupportedFormat` → 415, `ResourceLimitExceeded` → 413,
  `MemoryBudgetExceeded` → 503 with `Retry-After`).
- **Every response carries `timings` and `memory`** so the UI shows real numbers and a performance
  regression is visible in normal use rather than in a benchmark.

### 8.3 `mmar/serve/web/`

A single static HTML page (no build step, no node): chat pane, drag-drop zone, provenance panel with
citations and page/timestamp deep links, and a live memory bar fed by `/v1/memory`. Deliberately a
**debug surface**: it must display the extracted `MediaDoc` — chunk count, lanes used, errors — so
ingestion bugs are visible while chatting.

### 8.4 `mmar/cli.py`

```python
app = typer.Typer(help="mmar - local multi-modal AI research")

@app.command()  def info(verbose: bool = False) -> None: ...          # environment + memory profile
@app.command()  def doctor() -> None: ...                             # pass/fail gate table
@app.command()  def formats() -> None: ...                            # registry coverage vs README
@app.command()  def ingest(path: str, *, force: bool = False, dry_run: bool = False,
                           only: list[str] | None = None, recursive: bool = True) -> None: ...
@app.command()  def embed(*, rebuild: bool = False) -> None: ...
@app.command()  def index(action: Literal["build","stats","vacuum"]) -> None: ...
@app.command()  def search(query: str, *, k: int = 10, mode: str = "hybrid") -> None: ...
@app.command()  def chat(*, engine: str = "scratch", system: str | None = None) -> None: ...
@app.command()  def tokenize(action: Literal["train","test","compress"]) -> None: ...
@app.command()  def train(*, config: Path, resume: str = "never", thermal_relief: bool = False,
                          watch_memory: bool = False, dry_run: bool = False) -> None: ...
@app.command()  def bench(*, preset: str = "micro", steps: int = 50) -> None: ...   # TPS measurement
@app.command()  def models(action: Literal["list","pull","rm","verify"], name: str | None = None) -> None: ...
@app.command()  def eval(*, tasks: list[str] | None = None, out: Path | None = None) -> None: ...
@app.command()  def serve(*, host: str, port: int, reload: bool = False) -> None: ...
```

**CLI requirements:** every command ends with a `rich` summary box; every long-running command
supports `--dry-run`; `info` and `doctor` are the first things anyone runs, so they must be fast
(< 2 s) and must never download or allocate a model.

---

## 9. L5 — Evaluation (`mmar/eval/`)

```python
@dataclass(slots=True)
class MetricResult:
    name: str; value: float; unit: str; target: float | None
    passed: bool | None; n: int; notes: str = ""

@dataclass(slots=True)
class EvalReport:
    results: dict[str, MetricResult]
    peak_rss_mb: float; wall_clock_s: float; git_sha: str; machine: dict[str, Any]
    def render(self) -> str: ...      # the table that goes into docs/12_RESULTS.md
    def to_json(self) -> str: ...

def eval_perplexity(checkpoint: Path, *, split: str = "val", batches: int = 50) -> MetricResult: ...

def eval_generation_quality(checkpoint: Path, *, n_prompts: int = 20) -> MetricResult:
    """Distinct-n plus a repetition rate. Catches the classic 'loss looks fine, output is a
    loop' failure that perplexity alone cannot see."""

def eval_caption_quality(bridge_checkpoint: Path, pairs: Path, *, metric: str = "rougeL") -> MetricResult: ...
def eval_vqa(bridge_checkpoint: Path, questions: Path) -> MetricResult: ...

def eval_probes(bridge_checkpoint: Path, probes: Path) -> MetricResult:
    """Hand-built colour/count/proximity probes — the only honest test of whether the bridge
    learned anything about the image at all."""

def eval_asr(asr: Transcriber, clips: Path) -> MetricResult: ...     # WER via jiwer

def eval_retrieval(retriever: Retriever, labelled_pairs: Path, *, k: int = 5) -> list[MetricResult]:
    """recall@k, MRR, nDCG@k."""

def eval_latency(orch: Orchestrator, queries: Sequence[str]) -> list[MetricResult]:
    """TTFT, tok/s, retrieve p95, total p95."""

def eval_memory(orch: Orchestrator) -> MetricResult:
    """Peak RSS across a mixed workload — the SLO from analysis section 15."""

def eval_forgetting(base_ckpt: Path, tuned_ckpt: Path) -> MetricResult:
    """ppl ratio after SFT/bridge. Gate at 1.3x (risk R12)."""
```

`make eval` runs all of them and writes `artifacts/eval/report.json` plus a Markdown table ready to
paste into `docs/12_RESULTS.md`. **The report carries `git_sha` and a machine fingerprint**, so a
number without provenance is impossible.

---

## 10. Test requirements matrix

| Test file | Marker | What it proves | Phase |
|---|---|---|---|
| `test_config.py` | — | `extends:` chains, env precedence, every validation in §1.1 fires | P0 |
| `test_memory.py` | — | `MemoryGuard` refuses an oversized load; `peak_tracker` measures; MLX limits applied | P0 |
| `test_sniff.py` | — | Magic bytes beat a lying extension; unknown → `octet-stream`, never raises | P0 |
| `test_architecture.py` | — | **Import-layer rule** (§6.1 of the analysis) — no upward imports | P0, run always |
| `test_ir.py` | — | Invariants I1–I8; JSON round-trip; migration from a synthetic v0 doc | P1 |
| `test_extractors_*.py` | — | One fixture per format; each asserts text is found **and** provenance is right | P1–P3 |
| `test_extractor_contract.py` | — | Every registered extractor returns a `MediaDoc` for a garbage file, never raises | P1 |
| `test_registry_coverage.py` | — | **README format table == `registry.coverage()`** | P1 |
| `test_archive.py` | — | Nested zip containing a PDF yields the PDF's text; a zip bomb is refused by config | P1 |
| `test_chunker.py` | — | Token bounds per chunker; no mid-function splits; table headers repeated | P1 |
| `test_embed_cache.py` | — | Re-ingest hits the cache ~100 %; vectors are L2-normalised | P4 |
| `test_retrieval.py` | — | RRF ranks a known-best chunk first on a 20-doc labelled set; filters work | P4 |
| `test_tokenizer_roundtrip.py` | — | Lossless on ASCII/emoji/CJK/RTL/NUL; specials un-forgeable; compression ≥ 3.2 | P5 |
| `test_tokenizer_chat.py` | — | `encode_chat` mask is 0 on prompt tokens, 1 on assistant tokens | P5/P9 |
| `test_ref_numpy.py` | — | Hand-computed softmax/attention values; mask is provably causal | P6 |
| `test_model_shapes.py` | — | `cfg.n_params()` == real parameter count, per preset | P7 |
| `test_parity_*.py` | `parity` | MLX vs NumPy < 1e-4 for every primitive **and** a full forward | P7 |
| `test_kv_cache.py` | `parity` | Cached incremental decode == full forward, `atol=1e-4` | P7 |
| `test_llm_overfit.py` | `slow` | Loss < 0.1 on one fixed batch in ≤ 200 steps (the model *can* learn) | P7 |
| `test_shard_meta.py` | — | `load_shards` raises `TokenizerMismatch` on a fingerprint mismatch | P8 |
| `test_train_resume.py` | `slow` | A resumed run matches the un-interrupted LR and loss trajectory | P8 |
| `test_sft_mask.py` | — | Loss is computed on assistant tokens only | P9 |
| `test_bridge_freeze.py` | — | Encoders and the LLM have `requires_grad=False` in stage 1 | P10 |
| `test_bridge_budget.py` | — | Over-budget samples are dropped before the forward pass | P10 |
| `test_api.py` | — | Every endpoint's happy path and error path; `/health` allocates nothing | P11 |
| `test_serve_memory.py` | `slow` | `import mmar.serve.app; /health` → RSS < 250 MB | P11 |
| `test_eval_harness.py` | `slow` | Report contains `git_sha`, all metrics, and a rendered table | P12 |
| `test_models_contract.py` | `heavy` | One tiny inference per adapter (5 tokens) — catches framework API drift | P11+ |

**Marker policy:** `make test-fast` runs everything except `slow`/`heavy`/`parity` and must finish in
under 30 s with no network access. That is the inner loop; the full suite runs at phase gates.

---

## 11. Documentation-sync requirements

These are the mechanisms that stop this document set from becoming fiction:

| Mechanism | Enforced by |
|---|---|
| README format table == registry coverage | `test_registry_coverage.py` fails on drift |
| Preset parameter counts == code | `test_model_shapes.py` fails on drift |
| Every SLO is measured with a recorded command | `EvalReport` requires `git_sha`; P12 checklist |
| `[est]` markers are replaced by measurements | Phase prompts require the `[est]` tag be removed when the number is measured |
| Design changes update the analysis doc | Definition of Done item in `docs/01_ROADMAP.md §5` |

