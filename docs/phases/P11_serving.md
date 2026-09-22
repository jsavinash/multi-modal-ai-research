# P11 — Serving (MILESTONE 5 enabler)

| | |
|---|---|
| **Goal** | A FastAPI server that answers mixed-modality questions over HTTP with SSE streaming, citations, and live memory telemetry — the only phase that knows about HTTP |
| **Wall-clock** | 2 evenings |
| **Compute** | ~0 (no training) |
| **Prerequisites** | P4 gate green (RAG + orchestrator + Track D); P10 strongly recommended so `engine: bridge` is exercised |
| **Blocks** | P12 (eval harness calls the API), end-user usage |
| **Read first** | [analysis §12](../00_ANALYSIS.md#12-serving-architecture), [§6.1](../00_ANALYSIS.md#61-layer-model-and-the-one-directional-dependency-rule), [docs/02 §5.2](../02_FUNCTION_REQUIREMENTS.md#52-backend-adapters) |

---

## Deliverables

```
src/mmar/serve/__init__.py  app.py  schemas.py  api.py
src/mmar/serve/orchestrator.py           # EXISTS from P4 - call it, never rewrite it
src/mmar/serve/web/__init__.py  index.html  static/
tests/test_serve.py  test_api.py  test_sse.py
```

## Function requirements

Full signatures: [`docs/02 §5.2`](../02_FUNCTION_REQUIREMENTS.md#52-backend-adapters), [analysis §12.2](../00_ANALYSIS.md#122-chatrequest--chatresponse).

Non-negotiable behaviours:

- **`mmar.serve` is the only place that knows about HTTP.** `mmar.llm` and `mmar.ingest` must never import `fastapi`, `starlette`, or `uvicorn`. The test in `tests/test_architecture.py` enforces this.
- **All endpoints call sync code via `run_in_threadpool`.** There must be no `async def` inside `mmar.llm`, `mmar.ingest`, or `mmar.rag`. The async boundary is the API layer only.
- **Memory telemetry is in every response.** `ChatResponse` carries `MemorySnapshot` so the UI can show "this answer used 1.2 GB; 3.4 GB available" in real time.
- **Model residency follows `max_concurrent_models: 1`.** `/v1/transcribe` unloads the embedder first; `/v1/vision` unloads ASR first. `ModelRegistry.evict_all()` is called at every modality boundary.
- **SSE streaming emits three event types:** `token` (a decoded string fragment), `citations` (a list of `Citation` objects emitted once at the first retrieval), `done` (final `ChatResponse` as JSON). The UI reconstructs the stream from these.
- **`/v1/llm/*` is an OpenAI-compatible shim** so existing tools (langchain, llama-index) can talk to our model without changes. It does not need to be complete; `/v1/chat/completions` is the minimum.

## API surface

| Endpoint | Method | Body | Returns | Notes |
|---|---|---|---|---|
| `/health` | GET | — | `{status, version, uptime_s}` | Never loads a model |
| `/v1/memory` | GET | — | `MemorySnapshot` | RSS, MLX active/cache, headroom, resident models |
| `/v1/models` | GET | — | loaded + available handles with `peak_mb` | Drives the UI's "will this fit?" warning |
| `/v1/ingest` | POST | multipart files or `{path}` | `MediaDoc` summary + progress | Never loads a heavy model |
| `/v1/ingest/async` | POST | `{paths[]}` | `job_id` + SSE progress | Folders; survives client disconnect |
| `/v1/embed` | POST | `{texts[]}` \| `{images[]}` | vectors | |
| `/v1/transcribe` | POST | audio file | `Transcript{text, segments[]}` | Unloads the embedder first |
| `/v1/vision` | POST | image + question | `{caption, answer}` | Unloads ASR first |
| `/v1/chat` | POST | `ChatRequest` | `ChatResponse` | Main path |
| `/v1/chat/stream` | POST | `ChatRequest` | SSE `token`/`citations`/`done` | The UI uses this |
| `/v1/search` | POST | `{query, k, mode}` | `Hit[]` with provenance | Retrieval without generation |
| `/v1/llm/chat/completions` | POST | OpenAI-style | OpenAI-compatible chunked JSON | Shim for existing tools |

## Request/Response schemas

```python
class ChatRequest(BaseModel):
    message: str
    doc_ids: list[str] | None = None
    modalities: list[str] | None = None
    images: list[str] | None = None
    audio: str | None = None
    engine: Literal["scratch", "bridge", "instruct"] = "scratch"
    max_tokens: int = 256
    temperature: float = 0.8
    stream: bool = False

class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    source: str
    page: int | None
    time_span: tuple[float, float] | None
    score: float
    snippet: str  # <= 200 chars, verbatim

class ChatResponse(BaseModel):
    text: str
    citations: list[Citation]
    engine: str
    usage: dict[str, int]  # prompt_tokens, completion_tokens
    memory: MemorySnapshot
    timings: dict[str, float]  # retrieve_ms, decode_ms, total_ms
```

## Forbidden

- `async def` inside `mmar.llm`, `mmar.ingest`, or `mmar.rag`.
- Returning a generated answer without `Citation` objects.
- Loading a heavy model without checking `MemoryGuard` first.
- Using WebSockets. SSE is sufficient and simpler.
- Adding any dependency beyond `fastapi`, `uvicorn`, `sse-starlette`, `python-multipart`, and `jinja2`.

## Tasks

1. `serve/schemas.py`: all request/response models above, plus `MemorySnapshot` schema.
2. `serve/api.py` (endpoint glue): one function per endpoint that translates HTTP inputs into calls on the **existing** `mmar/serve/orchestrator.py` (P4) and back. No business logic here — and do not move or rewrite P4's module.
3. `serve/app.py`: FastAPI app, middleware for memory telemetry, error handlers that convert `MmarError` to JSON API errors with actionable hints.
4. SSE implementation for `/v1/chat/stream`: `token`, `citations`, `done` events. `test_sse.py` asserts event order and content.
5. `/v1/ingest/async` with a job store in sqlite (`jobs` table, already in `MediaStore`). A disconnected client can poll `/v1/jobs/{job_id}`.
6. Minimal web UI in `serve/web/`: a single `index.html` + `static/` with a chat pane, a citation drawer, and a memory indicator. No build step, no npm, no framework — vanilla JS + one CSS file.
7. OpenAI shim at `/v1/llm/chat/completions`: accepts `model`, `messages`, `stream`, and returns the OpenAI chunked-JSON format. Only the `scratch` engine is supported here.
8. Run the gate; test with curl and with the browser.

## Exit gate

```bash
make serve &
SERVER_PID=$!
sleep 2
curl -s http://127.0.0.1:8000/health
curl -s -X POST http://127.0.0.1:8000/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "summarise the document about Q3", "engine": "instruct"}'
curl -s -X POST http://127.0.0.1:8000/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "what is the revenue?", "engine": "instruct", "stream": true}' \
  --max-time 30
pytest tests/test_serve.py tests/test_api.py tests/test_sse.py -v
kill $SERVER_PID
```

**Honest check:** open `http://127.0.0.1:8000` in a browser, ask a question about one of your own files, and confirm you get an answer with clickable citations and a memory indicator that updates.

## Definition of Done

- [ ] All 12 endpoints respond with 200 on valid input and appropriate error codes on invalid input
- [ ] `/v1/chat/stream` emits `token`, `citations`, `done` in that order; `test_sse.py` green
- [ ] Every `ChatResponse` carries `memory` and `timings` fields populated from real measurements
- [ ] The web UI loads in a browser, sends a message, displays the answer with citations, and shows memory pressure
- [ ] `/v1/llm/chat/completions` returns a syntactically valid OpenAI response that `curl` can parse
- [ ] `tests/test_architecture.py` still green: `mmar.serve` is the only module that imports `fastapi`
- [ ] Peak RSS of a `/v1/chat` request measured and reported against the 1.5 GB budget

---

## 📋 Copy-paste prompt

```
ROLE
You are a senior ML systems engineer. You write code that is measured, not asserted.

CONTEXT - read these first, in this order
  1. docs/00_ANALYSIS.md  section 12 (serving architecture), section 6.1 (layer rule)
  2. docs/02_FUNCTION_REQUIREMENTS.md  section 5.2 (TextGenModel protocol)
  3. docs/01_ROADMAP.md  section 5 (Definition of Done)
  4. Existing code: src/mmar/serve/orchestrator.py (from P4), src/mmar/rag/retriever.py,
     src/mmar/models/textgen.py, src/mmar/utils/memory.py

MACHINE
  MacBook Air M1, 8 GB unified memory, fanless. python3.11 in .venv.
  Peak RSS budget: 1.5 GB for a /v1/chat request.

HARD CONSTRAINTS
  - mmar.serve is the ONLY module that may import fastapi/starlette/uvicorn. Tests in
    tests/test_architecture.py enforce this - if they fail, the phase is not done.
  - All business logic in mmar.serve must call sync code via run_in_threadpool. No async
    def inside mmar.llm, mmar.ingest, or mmar.rag.
  - Every ChatResponse must carry MemorySnapshot and timings. This is not optional - the
    UI and the eval harness both depend on it.
  - max_concurrent_models: 1 is enforced by ModelRegistry. /v1/transcribe unloads the
    embedder first; /v1/vision unloads ASR first.
  - SSE emits exactly three event types: token, citations, done. No other events.
  - The web UI is one index.html + one CSS file + one JS file. No build step, no npm.

DELIVERABLES - exactly these paths
  src/mmar/serve/__init__.py
  src/mmar/serve/app.py
  src/mmar/serve/schemas.py
  src/mmar/serve/api.py
  src/mmar/serve/web/__init__.py
  src/mmar/serve/web/index.html
  src/mmar/serve/web/static/app.js
  src/mmar/serve/web/static/style.css
  tests/test_serve.py
  tests/test_api.py
  tests/test_sse.py

KEY REQUIREMENTS
  - schemas.py: ChatRequest, ChatResponse, Citation, MemorySnapshot, Transcript, and all
    endpoint request/response models. Use pydantic BaseModel; no dataclasses here because
    FastAPI needs pydantic for OpenAPI.
  - api.py: one function per endpoint. It translates HTTP inputs to calls on the EXISTING
    P4 orchestrator (src/mmar/serve/orchestrator.py) and returns. No business logic, no
    chunking, no embedding here, and no rewrite of the P4 module.
  - app.py: FastAPI with CORS, exception handlers converting MmarError to JSON with
    {"error": {"code": "...", "hint": "..."}}, middleware recording per-request RSS delta.
  - SSE: /v1/chat/stream uses sse-starlette. Events are: {"type": "token", "text": "..."},
    {"type": "citations", "items": [...]}, {"type": "done", "response": {...}}.
  - /v1/ingest/async: job_id returned immediately; progress via SSE at /v1/jobs/{job_id}/events.
    Poll /v1/jobs/{job_id} for status.
  - /v1/llm/chat/completions: OpenAI-compatible. Accepts model, messages, stream. Returns
    chunks with id, object, created, model, choices[]. Only scratch engine.
  - Web UI: single page. Chat pane with message history; citation drawer showing source,
    page, snippet; memory indicator (green/yellow/red bar). No external JS dependencies.
  - Error responses: every endpoint returns {"error": {"code": MmarError.code, "hint": "..."}}
    with the appropriate HTTP status. Never a bare 500 with no body.

FORBIDDEN
  - async def inside mmar.llm, mmar.ingest, mmar.rag.
  - WebSockets. SSE is sufficient.
  - Any dependency beyond fastapi, uvicorn, sse-starlette, python-multipart, jinja2.
  - Any file containing "# TODO" or "raise NotImplementedError".
  - Reporting a command as passing without pasting its raw output.

TASKS
  1. schemas.py - all request/response models.
  2. serve/api.py - endpoint glue functions (call, do not modify, the P4 orchestrator).
  3. app.py - FastAPI app, middleware, exception handlers, SSE.
  4. test_serve.py + test_api.py + test_sse.py.
  5. Web UI (index.html + app.js + style.css).
  6. OpenAI shim at /v1/llm/chat/completions.
  7. /v1/ingest/async with job store.
  8. Run the gate; test with curl and browser.

EXIT GATE - paste the complete unedited output
  make test-fast
  pytest tests/test_serve.py tests/test_api.py tests/test_sse.py -v
  curl -s http://127.0.0.1:8000/health
  curl -s -X POST http://127.0.0.1:8000/v1/chat \
    -H "Content-Type: application/json" \
    -d '{"message": "summarise the document about Q3", "engine": "instruct"}'
  # start a stream and show the first 5 events
  curl -N -X POST http://127.0.0.1:8000/v1/chat/stream \
    -H "Content-Type: application/json" \
    -d '{"message": "what is the revenue?", "engine": "instruct", "stream": true}' \
    --max-time 15 | head -20

EVIDENCE REQUIRED IN YOUR RESPONSE
  1. make test-fast raw output.
  2. /health JSON response verbatim.
  3. /v1/chat JSON response verbatim, including the memory and timings fields.
  4. First 10 SSE events from /v1/chat/stream verbatim.
  5. Browser screenshot or description of the UI loading and producing a cited answer.
  6. Peak RSS of a /v1/chat request measured with /usr/bin/time -l, compared to the 1.5 GB budget.
  7. Anything you could NOT verify, stated explicitly as UNVERIFIED.

Before writing code, answer in at most three lines each: what is the exit gate; which command proves it; the most likely way to get this silently wrong.
