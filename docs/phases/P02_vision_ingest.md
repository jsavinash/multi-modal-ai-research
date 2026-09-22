# P2 — Vision Ingestion

| | |
|---|---|
| **Goal** | Images become searchable text + vectors via Apple Vision OCR and SigLIP/CLIP embeddings; video becomes keyframes + audio — all within a 1.2 GB peak |
| **Wall-clock** | 2 evenings |
| **Compute** | ~0 (Apple Vision is an OS service; embeddings are small) |
| **Prerequisites** | P1 gate green; `ocrmac` + `pyobjc-framework-Vision` installed |
| **Blocks** | P4 image retrieval, P10 bridge |
| **Read first** | [analysis §8.1](../00_ANALYSIS.md#81-vision--encode-ocr-caption), [§4.5](../00_ANALYSIS.md#45-resident-profiles--what-may-run-at-the-same-time), [docs/02 §2.5](../02_FUNCTION_REQUIREMENTS.md#25-required-extractors-by-phase) |

---

## Deliverables

```
src/mmar/ingest/extractors/image.py   video.py
src/mmar/ingest/media/ffmpeg.py          # probe, keyframes, audio extract, scene detection
src/mmar/models/__init__.py  registry.py  vision.py  embed_image.py
tests/test_extractors_image.py  test_extractors_video.py  test_ffmpeg.py  test_models_lazy.py
tests/fixtures/images/   (>= 6 imgs: screenshot, photo, rotated photo, low-res, code screenshot, photo with no text)
tests/fixtures/video/    (1 short clip with speech AND >= 2 scene changes)
```

## Function requirements

`Extractor` protocol and `estimate_peak_mb` semantics: [docs/02 §2.2](../02_FUNCTION_REQUIREMENTS.md#22-mmar-ingestbasepy--the-extractor-contract).
Registry / handle / guard interaction: [docs/02 §5.1](../02_FUNCTION_REQUIREMENTS.md#51-mmar-modelsregistrypy).

Non-negotiable behaviours:

- **`import mmar.models` allocates nothing.** It must not import `mlx`, not download, not create a
  Metal buffer. `test_models_lazy.py` asserts `rss_mb() < 250` after importing it.
- **OCR needs no model download.** Apple Vision is an OS service; `ocrmac` only bridges to it. If the
  import fails, fall back to a `lane=META` metadata chunk with `errors=["ocr_unavailable"]` — never
  raise, and never silently return an empty doc with no explanation.
- **`estimate_peak_mb` must be honest.** `ImageExtractor` declares ≥ 800 MB with captioning and
  ≥ 260 MB without. That value is what `MemoryGuard.enforce` sees *before* anything is loaded.
- **Image vectors carry `meta={"space": "image"}`** and live in the image matrix, never the text matrix.
- **`video.py` is sequential, not nested:** all keyframes first, then release the image lane, then push
  the audio track through the audio extractor. Never hold a VLM and an ASR model at once.
- **Captioning is opt-in** (`ingest.image.caption: false`) and must still be implemented, because the
  P10 bridge needs a bootstrapping path that captions thousands of your own images.

## ffmpeg wrapper requirements

- Detect `ffmpeg`/`ffprobe` once; if absent raise `ModelUnavailable` carrying the exact
  `brew install ffmpeg` hint.
- `probe(path) -> MediaInfo` from `ffprobe -print_format json`: duration, fps, resolution, streams,
  and whether an audio stream exists.
- `extract_keyframes(path, out_dir, *, per_minute, max_frames, scene_detect) -> list[FrameInfo]` uses
  `select='gt(scene,0.4)'` and **falls back to uniform sampling when fewer than 2 scenes are found** —
  a single-shot lecture video has no scene changes and would otherwise yield one frame.
- `extract_audio(path, out_wav, *, sample_rate=16000, mono=True)`.
- Every invocation has a timeout; a non-zero exit raises `CorruptInput` with stderr captured into the
  doc's `errors`.

## Forbidden

- Loading a VLM to caption unless `ingest.image.caption` is true.
- Holding two heavy models resident. All model access goes through `ModelRegistry.acquire()`.
- Writing frames as `.npy` — they must be real image files under the content-addressed asset root.
- Transcoding video to disk "to keep ffmpeg simple". Extract what you need and move on.
- Adding `opencv-python` (~90 MB) when ffmpeg already does scene detection.

## Tasks

1. `mmar/models/registry.py` with `MODEL_CATALOG` as data, `MemoryGuard` integration and the lazy-import
   contract. Write `test_models_lazy.py` **first** — it guards every later phase.
2. `mmar/ingest/media/ffmpeg.py` + `test_ffmpeg.py`, including the no-scene-changes fallback.
3. `extractors/image.py`: `ocrmac` OCR → `lane=OCR` chunk; optional captioning; SigLIP embedding via
   `ClipImageEncoder`; resize to `max_edge_px`; **EXIF rotation handled** (a rotated phone photo must
   OCR correctly — the most common real-world failure).
4. `extractors/video.py`: keyframes → recursive image extraction → then audio transcode → then ASR
   (call the ASR adapter through `ModelRegistry.acquire("audio.asr")`; P3 implements the adapter).
5. Fixtures: screenshot with text, rotated photo, low-res photo, code screenshot, a photo with no text,
   and one short video with speech plus ≥ 2 scene changes.
6. Wire `pdf.py`'s OCR hook now that OCR exists; re-run P1's PDF fixture including a scanned page.
7. Run the gate; measure image-ingest and video-ingest peak RSS separately.

## Exit gate

```bash
make test-fast
python -m mmar.cli ingest tests/fixtures/images
python -m mmar.cli ingest tests/fixtures/video
/usr/bin/time -l python -m mmar.cli ingest tests/fixtures/images
/usr/bin/time -l python -m mmar.cli ingest tests/fixtures/video
/usr/bin/time -l .venv/bin/python -c "import mmar.models"
```

Plus **a human read of the OCR output**: the screenshot's text must be correct, and the rotated
photo's OCR must be correct.

## Definition of Done

- [ ] Images produce `lane=OCR` chunks; image vectors carry `meta["space"] == "image"`
- [ ] The rotated photo OCRs correctly (EXIF handled)
- [ ] Video produces keyframes as real image assets, plus a transcript once P3 lands
- [ ] `import mmar.models` peaks under 250 MB; raw output pasted
- [ ] Image and video ingest peak RSS measured separately, reported against the 1.2 GB budget
- [ ] With `ingest.image.caption: false` no VLM is loaded — evidenced by the measured RSS
- [ ] Missing ffmpeg yields `errors`, not an exception
- [ ] README extension list updated if anything changed

## 📋 Copy-paste prompt

```
ROLE
You are a senior ML systems engineer. You write code that is measured, not asserted.

CONTEXT - read these first
  1. docs/00_ANALYSIS.md  section 8.1 (vision model selection), section 4.5 (resident
     profiles), section 12.3 (residency policy)
  2. docs/02_FUNCTION_REQUIREMENTS.md  section 2.2 (extractor contract), 2.5 (per-extractor
     requirements), 3.2 (embedders), 5.1 (registry)
  3. docs/01_ROADMAP.md  section 5
  4. Existing code: src/mmar/ingest/{base,registry,pipeline}.py, extractors/{image,pdf}.py,
     src/mmar/utils/memory.py

MACHINE
  MacBook Air M1, 8 GB unified memory, FANLESS, macOS 26.x. python3.11 in .venv.
  Peak RSS budget for this phase: 1.2 GB. The whole machine leaves ~4.6 GB for ML.

HARD CONSTRAINTS
  - `import mmar.models` must allocate nothing: no mlx import, no download, no Metal buffer.
    Prove it with /usr/bin/time -l and report the number.
  - OCR uses the macOS Vision framework via ocrmac and needs NO model download. If the import
    fails, degrade to a lane=META metadata chunk with errors=["ocr_unavailable"]. Never raise
    and never return an empty doc with no explanation.
  - estimate_peak_mb() must be a conservative OVER-estimate, because MemoryGuard enforces it
    BEFORE anything loads. >= 800 MB with captioning, >= 260 MB without.
  - Image embeddings go to the "image" space with meta["space"]="image". Never mix text and
    image vectors in one matrix.
  - Video extraction is SEQUENTIAL: all keyframes first, release the image lane, then the
    audio track. Never hold a VLM and an ASR model at the same time.

DELIVERABLES - exactly these paths
  src/mmar/ingest/extractors/image.py
  src/mmar/ingest/extractors/video.py
  src/mmar/ingest/media/__init__.py
  src/mmar/ingest/media/ffmpeg.py
  src/mmar/models/__init__.py
  src/mmar/models/registry.py
  src/mmar/models/vision.py
  src/mmar/models/embed_image.py
  tests/test_extractors_image.py
  tests/test_extractors_video.py
  tests/test_ffmpeg.py
  tests/test_models_lazy.py
  tests/fixtures/images/      (>= 6 images)
  tests/fixtures/video/       (1 short clip)
  src/mmar/ingest/extractors/pdf.py      (MODIFY: wire the OCR hook only)

KEY REQUIREMENTS
  - ModelRegistry: MODEL_CATALOG as data (key, repo_id, kind, peak_mb, quant). acquire() is a
    context manager that calls MemoryGuard.enforce(spec.peak_mb) FIRST, evicts every other
    handle with peak_mb > 1000, loads lazily, then compares actual mx.metal active memory to
    the declared estimate and warns on a >30% overrun. unload() drops the object AND calls
    mx.clear_cache().
  - ffmpeg wrapper: probe() parsed from `ffprobe -print_format json`; extract_keyframes() using
    select='gt(scene,0.4)' WITH A UNIFORM-SAMPLING FALLBACK when fewer than 2 scenes are found
    (a single-shot lecture has no scene changes); extract_audio() to 16 kHz mono. Timeouts on
    every call; a non-zero exit raises CorruptInput with stderr captured into doc.errors.
  - image.py: EXIF rotation applied BEFORE OCR (test with a rotated fixture - this is the most
    common real-world failure); resize to ingest.image.max_edge_px; ocrmac OCR producing one
    chunk with lane=OCR and confidence in meta; captioning ONLY when ingest.image.caption is
    true; SigLIP embedding carrying meta["space"]="image".
  - video.py: keyframes written as real image assets under the content-addressed root, each
    recursively extracted by ImageExtractor with meta["ts"] set; afterwards the audio track is
    extracted and passed to the audio extractor through the registry.
  - Fixtures you must create: a screenshot with readable text; a ROTATED photo containing text;
    a low-resolution photo with text; a screenshot of source code; a photo containing no text
    at all; and one short video with speech and at least two scene changes. Generate the video
    with ffmpeg if you cannot source one (concat two colour clips plus a sine tone).

FORBIDDEN
  - Loading a VLM unless ingest.image.caption is true.
  - Two heavy models resident at once.
  - Writing frames as .npy instead of real image files.
  - Adding opencv-python or any dependency not already in pyproject.toml.
  - Any file containing "# TODO" or "raise NotImplementedError".
  - Reporting a command as passing without pasting its raw output.

TASKS
  1. src/mmar/models/registry.py + test_models_lazy.py FIRST. Prove the lazy contract before
     writing anything else - every later phase depends on it.
  2. mmar/ingest/media/ffmpeg.py + test_ffmpeg.py including the scene-detection fallback.
  3. extractors/image.py + tests (the rotated fixture is mandatory).
  4. extractors/video.py + tests. Reach the ASR call through ModelRegistry.acquire("audio.asr");
     P3 implements that adapter.
  5. Create all fixtures.
  6. MODIFY extractors/pdf.py to OCR pages under min_chars_per_page, and re-run the P1 PDF test
     with a scanned page included.
  7. Run the gate. Measure image and video peak RSS separately.

EXIT GATE - paste the complete unedited output
  make test-fast
  python -m mmar.cli ingest tests/fixtures/images
  python -m mmar.cli ingest tests/fixtures/video
  /usr/bin/time -l python -m mmar.cli ingest tests/fixtures/images
  /usr/bin/time -l python -m mmar.cli ingest tests/fixtures/video
  /usr/bin/time -l .venv/bin/python -c "import mmar.models"

EVIDENCE REQUIRED IN YOUR RESPONSE
  1. make test-fast raw output.
  2. The OCR text from the screenshot fixture AND from the rotated fixture, verbatim, so a human
     can judge legibility and correctness.
  3. RSS numbers for: import mmar.models; image ingest; video ingest. Compare each to its budget.
  4. Proof that no VLM was loaded when captioning is false (the RSS number is that proof).
  5. Anything you could NOT verify, stated explicitly as UNVERIFIED.

Before writing code, answer in at most three lines each: what is the exit gate; which command
proves it; the most likely way to get this silently wrong.
```
