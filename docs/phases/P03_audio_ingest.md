# P3 — Audio Ingestion

| | |
|---|---|
| **Goal** | Any audio or video becomes a timestamped transcript you can retrieve, with a cheap first pass and an accurate second pass |
| **Wall-clock** | 1–2 evenings |
| **Compute** | ~0 (ASR on 1 h of audio ≈ 4–6 min with `whisper-base`) |
| **Prerequisites** | P1 (chunker, `TemporalChunker`) and P2 (ffmpeg wrapper) gates green |
| **Blocks** | P4 (transcript retrieval), P10 stage 3 |
| **Read first** | [analysis §8.2](../00_ANALYSIS.md#82-audio--speech-to-text), [§4.5](../00_ANALYSIS.md#45-resident-profiles--what-may-run-at-the-same-time) |

---

## Deliverables

```
src/mmar/ingest/extractors/audio.py
src/mmar/models/asr.py                    # Transcriber adapter over mlx-whisper / Moonshine
src/mmar/rag/chunker.py                   # TemporalChunker lands here in this phase
tests/test_extractors_audio.py  test_asr.py  test_chunker_temporal.py
tests/fixtures/audio/                     # 60 s clean speech (WAV) + a 30 s mp3 + a silent WAV
```

## Function requirements

`Transcript` / `TranscriptSegment` and the two-pass design: [docs/02 §5.2](../02_FUNCTION_REQUIREMENTS.md#52-backend-adapters).
`TemporalChunker`: [docs/02 §3.1](../02_FUNCTION_REQUIREMENTS.md#31-mmar-ragchunkerpy).

Non-negotiable behaviours:

- **Two passes are the design, not an optimisation.** Pass 1 uses `ingest.audio.backend` (default
  `whisper-base`) and writes `meta["asr_model"]`. `mmar transcribe --upgrade <doc_id>` re-runs pass 2
  with `whisper-large-v3-turbo`. This is what makes a 3-hour podcast archive tractable (analysis §8.2).
- **Every chunk carries `time_span=(start, end)`.** A citation must be able to deep-link a timestamp.
  `test_chunker_temporal.py` asserts no chunk spans more than 2× `target_tokens` of ASR text and that
  spans are monotonically increasing and non-overlapping except for the configured overlap.
- **`estimate_peak_mb()` is model-dependent**: ~210 MB for `whisper-base`, ~1400 MB for
  `large-v3-turbo`. `MemoryGuard` must see the real number for the configured backend.
- **A silent or music-only file must not hallucinate text.** Whisper is known to invent sentences in
  silence. Gate on segment `no_speech_prob` / the mean logprob and emit `errors=["no_speech_detected"]`
  rather than a fabricated transcript. This is the single most important correctness requirement of
  this phase.
- **Resampling goes through ffmpeg to 16 kHz mono.** Never decode audio in Python.

## Forbidden

- `faster-whisper` / CTranslate2 — no Apple GPU path (analysis §8.2 explicitly rejects it).
- Growing a transcript in memory for a multi-hour file. Stream segments to the chunker.
- Returning a transcript with zero segments and no `errors` entry. Silence must be reported.
- Loading the turbo model during ingestion when the configured backend is `whisper-base`.

## Tasks

1. `models/asr.py` with the `Transcriber` protocol implementations and the lazy-load contract.
2. `TemporalChunker` in `rag/chunker.py` with its tests, plus `Counter` handling: before P5 the
  token counter is `chars/4`; after P5 it is the real tokenizer. Both paths must work.
3. `extractors/audio.py`: ffmpeg → 16 kHz mono temp WAV → ASR → per-segment text → `TemporalChunker`.
4. Fixtures: 60 s of clean speech (record yourself, or synthesise), a 30 s `.mp3` (transcoded by
   ffmpeg to prove the container path), and a 30 s silent WAV that must produce
   `errors=["no_speech_detected"]` and no text.
5. Wire `video.py` to call the audio extractor for its audio track (replacing P2's stub).
6. Run the gate; measure peak RSS and compute WER on the 60 s clip if you have a reference transcript.

## Exit gate

```bash
make test-fast
python -m mmar.cli ingest tests/fixtures/audio
python -m mmar.cli search "the first words spoken in the clip"
/usr/bin/time -l python -m mmar.cli ingest tests/fixtures/audio
```

Plus: **WER ≤ 15 % on the 60 s clean clip** (compute with `jiwer`; if you have no reference
transcript, hand-write one — it takes ten minutes and it is the only way to know the ASR works).

## Definition of Done

- [ ] Every transcript chunk carries `time_span`; spans are ordered and bounded
- [ ] The silent fixture yields `errors=["no_speech_detected"]` and **no** fabricated text
- [ ] WER on the 60 s clip measured with `jiwer` and reported against the 15 % target
- [ ] Peak RSS measured and reported against the 1.2 GB budget
- [ ] `video.py` now produces a real transcript through the audio extractor
- [ ] Two-pass upgrade path works: `mmar transcribe --upgrade <doc_id>` re-transcribes and updates
      `meta["asr_model"]`

---

## 📋 Copy-paste prompt

```
ROLE
You are a senior ML systems engineer. You write code that is measured, not asserted.

CONTEXT - read these first
  1. docs/00_ANALYSIS.md  section 8.2 (ASR model selection and the rejection of
     faster-whisper), section 4.5 (resident profiles)
  2. docs/02_FUNCTION_REQUIREMENTS.md  section 3.1 (chunkers), section 5.2 (Transcriber
     protocol and Transcript/TranscriptSegment)
  3. Existing code: src/mmar/ingest/extractors/video.py (has a stubbed ASR call),
     src/mmar/ingest/media/ffmpeg.py, src/mmar/models/registry.py, src/mmar/rag/chunker.py

MACHINE
  MacBook Air M1, 8 GB unified memory, fanless. python3.11 in .venv.
  Peak RSS budget: 1.2 GB for the default backend, 1.6 GB when large-v3-turbo is configured.

HARD CONSTRAINTS
  - ASR goes through mlx-whisper (GPU) or Moonshine (CPU). NEVER faster-whisper/CTranslate2 -
    it has no Apple GPU path and burns the CPU while the GPU idles.
  - A silent or music-only file MUST NOT produce fabricated text. Whisper hallucinates in
    silence; gate on the segment no_speech_prob / mean logprob and emit
    errors=["no_speech_detected"] with an empty transcript instead. This is the single most
    important correctness requirement of this phase.
  - Every chunk must carry time_span=(start, end) so a citation can deep-link a timestamp.
  - estimate_peak_mb() must reflect the CONFIGURED backend: ~210 MB for whisper-base,
    ~1400 MB for large-v3-turbo. MemoryGuard enforces it before loading.
  - Decoding and resampling go through ffmpeg to 16 kHz mono. Never decode audio in Python.
  - Stream segments; never accumulate a multi-hour transcript in memory.

DELIVERABLES - exactly these paths
  src/mmar/ingest/extractors/audio.py
  src/mmar/models/asr.py
  src/mmar/rag/chunker.py            (ADD TemporalChunker; keep the existing chunkers)
  tests/test_extractors_audio.py
  tests/test_asr.py
  tests/test_chunker_temporal.py
  tests/fixtures/audio/              (60 s clean speech WAV, 30 s mp3, 30 s silent WAV)
  src/mmar/ingest/extractors/video.py   (MODIFY: replace the stubbed ASR call)

KEY REQUIREMENTS
  - Two-pass design, and it is the DESIGN not an optimisation:
      pass 1 uses ingest.audio.backend (default whisper-base), fast, writes meta["asr_model"]
      pass 2 is `mmar transcribe --upgrade <doc_id>`, re-runs with large-v3-turbo and updates
             meta["asr_model"] in place
    This is what makes a multi-hour archive tractable.
  - TemporalChunker merges consecutive ASR cues up to target_tokens, preserving (t_start,
    t_end) on EVERY chunk. Invariants to assert in tests: no chunk exceeds 2x target_tokens of
    ASR text; spans are monotonically increasing; only the configured overlap is duplicated.
  - The token counter must work in both modes: chars/4 before P5 exists, and the real
    tokenizer after. Both paths tested.
  - auto.py must set lane=ExtractLane.ASR on every chunk and record language + model in meta.

FORBIDDEN
  - faster-whisper or ctranslate2.
  - Returning an empty transcript with no errors entry.
  - Loading the turbo model during ingestion when the configured backend is whisper-base.
  - Any file containing "# TODO" or "raise NotImplementedError".
  - Reporting a command as passing without pasting its raw output.

TASKS
  1. src/mmar/models/asr.py with the Transcriber protocol + lazy-load contract.
  2. TemporalChunker in src/mmar/rag/chunker.py + tests.
  3. src/mmar/ingest/extractors/audio.py.
  4. Fixtures: 60 s of clean speech (record it yourself), a 30 s .mp3 produced by ffmpeg from
     that WAV (proves the container path), and a 30 s silent WAV.
  5. MODIFY video.py to route its audio track through the audio extractor.
  6. Hand-write a reference transcript for the 60 s clip and compute WER with jiwer.
  7. Run the gate and measure peak RSS.

EXIT GATE - paste the complete unedited output
  make test-fast
  python -m mmar.cli ingest tests/fixtures/audio
  python -m mmar.cli search "the first words spoken in the clip"
  /usr/bin/time -l python -m mmar.cli ingest tests/fixtures/audio
  python -c "import jiwer; ..."   # compute WER against your hand-written reference

EVIDENCE REQUIRED IN YOUR RESPONSE
  1. make test-fast raw output.
  2. The full transcript of the 60 s clip, verbatim, plus your reference transcript, plus the
     WER number. A human must be able to judge whether the ASR works.
  3. The output for the SILENT fixture, showing errors=["no_speech_detected"] and no text.
  4. The RSS number compared against the 1.2 GB budget.
  5. Anything you could NOT verify, stated explicitly as UNVERIFIED.

Before writing code, answer in at most three lines each: what is the exit gate; which command
proves it; the most likely way to get this silently wrong.
```

