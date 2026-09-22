# P4 — Embeddings, Index, Retrieval, RAG  ★ MILESTONE 1

| | |
|---|---|
| **Goal** | A working local assistant over your own files: ingest anything, ask questions, get answers with citations. **This is the first point the project pays you back** |
| **Wall-clock** | 2 evenings |
| **Compute** | ~0 |
| **Prerequisites** | P1 (P2/P3 strongly recommended — without them you can only ask about text) |
| **Blocks** | everything after it consumes the same store |
| **Read first** | [analysis §8.3](../00_ANALYSIS.md#83-text-embeddings), [§12](../00_ANALYSIS.md#12-serving-architecture), [docs/02 §3–§4](../02_FUNCTION_REQUIREMENTS.md#3-l2--representation-chunking-embedding-storage) |

---

## Deliverables

```
src/mmar/rag/__init__.py  store.py  embed.py  index.py  retriever.py
src/mmar/serve/__init__.py  orchestrator.py
src/mmar/llm/__init__.py  prompt.py
src/mmar/models/textgen.py                # MlxInstructTextGen (Track D)
tests/test_store.py  test_embed_cache.py  test_retrieval.py  test_context_builder.py
tests/test_orchestrator.py
tests/eval_sets/file_qa.yaml              # 10 questions you know the answers to
```

## Function requirements

[`docs/02 §3, §4, §5.2, §8.1`](../02_FUNCTION_REQUIREMENTS.md#3-l2--representation-chunking-embedding-storage).

Non-negotiable behaviours:

- **`MediaStore` is transactional.** A failed ingest leaves no partial doc. `upsert_doc` +
  `upsert_chunks` + `put_vectors` commit together or not at all.
- **`EmbeddingCache` keys on `sha256(content) + model + dim`.** Re-ingesting an unchanged folder must
  produce a **cache hit rate ≥ 95 %**, asserted in `test_embed_cache.py`.
- **Vectors are L2-normalised at write time**, so search is a plain dot product. A test asserts
  `abs(norm(v) - 1) < 1e-5` for every stored vector.
- **Text and image vectors never share a matrix** (`space` key).
- **RRF is used, not score normalisation** (analysis §4 of the function spec): dense cosine and BM25
  are on incomparable scales and per-query min-max normalisation is unstable exactly when the top hit
  is an outlier.
- **`ContextBuilder` drops whole chunks**, never truncates mid-chunk, prefixes each with `[n]`, and
  emits the question last. `Citation.snippet` is the first 200 chars of the chunk, never a paraphrase.
- **`Orchestrator._select_engine` reports the engine it actually used** in `ChatResponse.engine`. A
  silent fallback is a bug, not a convenience.
- **`HashEmbedder` must exist** as a dependency-free fallback so P4 runs even if MLX is broken. Its
  docstring and the README must state that its retrieval quality is poor.

## Forbidden

- FAISS, HNSW, Chroma, LanceDB, or any approximate index. 10 k chunks × 384 dims = 15 MB; exact search
  is ~2 ms at 100 % recall (ADR-7).
- A cross-encoder reranker by default. `HeuristicReranker` is the default; the cross-encoder is opt-in
  because it costs 0.3 GB and ~40 ms per query.
- Generating an answer without citations. Every response path must carry `Citation` objects.
- `print()` anywhere in `mmar/rag` or `mmar/serve`.
- Claiming the assistant "works" without the 10-question evaluation in the gate.

## Tasks

1. `rag/store.py` with the sqlite schema, migrations, transactions and `stats()`.
2. `rag/embed.py` with `MlxTextEmbedder` (multilingual-e5-small, with the `query:`/`passage:`
   prefixes added internally) + `HashEmbedder` + `EmbeddingCache`.
3. `rag/index.py`: `DenseIndex`, `BM25Index` (postings in sqlite), `reciprocal_rank_fusion`.
4. `rag/retriever.py`: `Retriever`, `HeuristicReranker`, `ContextBuilder`.
5. `models/textgen.py`: `MlxInstructTextGen` over `mlx-lm` + `Qwen3-0.6B-Instruct-4bit`, with a
   memory check before load and `stream()` implemented over the KV cache.
6. `llm/prompt.py`: `format_qa`, `build_rag_prompt`, `extract_citation_markers`.
7. `serve/orchestrator.py`: `chat()`, `stream_chat()`, `search()`, `_select_engine()`,
   `_residency_plan()`.
8. Write `tests/eval_sets/file_qa.yaml` — **10 questions about 10 real files of yours**, with the
   expected answer substring and expected source. This is the milestone artefact. Run it and report
   the score honestly.

## Exit gate

```bash
make ingest IN=<a folder of your own mixed files>
make index
make chat                      # or: python -m mmar.cli search "..."
python -m pytest tests/test_orchestrator.py tests/test_retrieval.py -v
```

Plus the honour test: **of the 10 questions in `file_qa.yaml`, how many are answered correctly with
the correct citation?** Target ≥ 8/10. Report the real number, including which ones failed and why.

## Definition of Done

- [ ] 10-question evaluation run; score reported honestly with the failures named
- [ ] Every answer carries citations with `source`, and `page`/`time_span` where applicable
- [ ] Embedding cache hit rate ≥ 95 % on a second `make ingest` of the same folder; number pasted
- [ ] Vectors verified L2-normalised; test green
- [ ] `ChatResponse.timings` shows retrieve_ms and decode_ms for every query
- [ ] Peak RSS for ingest and for chat measured and reported
- [ ] `HashEmbedder` path exercised by a test so P4 is provably runnable without MLX

---

## 📋 Copy-paste prompt

```
ROLE
You are a senior ML systems engineer. You write code that is measured, not asserted.

CONTEXT - read these first
  1. docs/00_ANALYSIS.md  section 8.3 (why exact numpy search, no FAISS), section 12
     (serving architecture), section 4.5 (resident profiles)
  2. docs/02_FUNCTION_REQUIREMENTS.md  section 3 (store/embed/chunk), section 4 (retrieval),
     section 5.2 (TextGenModel), section 8.1 (Orchestrator)
  3. docs/01_ROADMAP.md  section 5 and the MILESTONE 1 definition
  4. Existing code: src/mmar/rag/chunker.py, src/mmar/models/registry.py,
     src/mmar/utils/memory.py, src/mmar/config.py

MACHINE
  MacBook Air M1, 8 GB unified memory, fanless. python3.11 in .venv.
  Peak RSS budget: 1.2 GB for ingest+index, 1.5 GB for chat.

HARD CONSTRAINTS
  - NO FAISS, HNSW, Chroma, LanceDB or any approximate index. 10k chunks x 384 dims is 15 MB
    and exact search is ~2 ms at 100% recall. Approximate indexing is a dependency and a recall
    regression for no gain (ADR-7).
  - Embeddings are L2-normalised AT WRITE TIME so search is a plain dot product. Assert it.
  - Text and image vectors live in SEPARATE matrices keyed by `space`. Assert it.
  - Fusion is reciprocal rank fusion, NOT weighted score normalisation. Dense cosine and BM25
    are on incomparable scales and per-query min-max normalisation is unstable exactly when the
    top hit is an outlier.
  - ContextBuilder drops whole chunks when over budget; never truncates mid-chunk. Every chunk
    is prefixed with an ordinal marker [n], and the question is emitted LAST.
  - The Orchestrator must report, in ChatResponse.engine, which engine ACTUALLY answered. A
    silent fallback is a bug.
  - HashEmbedder must exist as a dependency-free fallback so P4 runs even if MLX is broken. Its
    docstring must state that its retrieval quality is poor.

DELIVERABLES - exactly these paths
  src/mmar/rag/{__init__,store,embed,index,retriever}.py
  src/mmar/serve/{__init__,orchestrator}.py
  src/mmar/llm/{__init__,prompt}.py
  src/mmar/models/textgen.py
  tests/test_store.py
  tests/test_embed_cache.py
  tests/test_retrieval.py
  tests/test_context_builder.py
  tests/test_orchestrator.py
  tests/eval_sets/file_qa.yaml

KEY REQUIREMENTS
  - MediaStore: sqlite at indexes/mmar.sqlite3, tables docs/assets/chunks/embeddings_text/
    embeddings_image/bm25_terms/jobs. upsert_doc + upsert_chunks + put_vectors commit together;
    a failed ingest leaves NO partial doc. has_doc() is content-hash aware. get_vectors(space)
    returning the whole matrix is acceptable AND must be documented as such.
  - EmbeddingCache: sqlite, keyed sha256(content)+model+dim. A second ingest of an unchanged
    folder must show >= 95% hit rate - assert it and report the number.
  - MlxTextEmbedder: intfloat/multilingual-e5-small via mlx-embeddings. It must add the
    "query: " / "passage: " prefixes INTERNALLY so no caller can forget them.
  - BM25Index: Okapi BM25 k1=1.2 b=0.75 with your own tokenizer; postings in sqlite so the
    index survives a restart without re-tokenising.
  - Retriever.retrieve(): embed the query once; dense top-2k and bm25 top-2k under the same
    mask; RRF weighted by rag.bm25_weight; hydrate; return exactly k.
  - ContextBuilder.build(): whole-chunk dropping, [n] markers, question last, near-duplicate
    dedupe at cosine > 0.97, Citation.snippet = first 200 chars verbatim.
  - MlxInstructTextGen: mlx-lm over Qwen3-0.6B-Instruct-4bit. The model is OPTIONAL - if the
    weights are absent, raise ModelUnavailable whose hint contains the exact
    `mmar models pull <name>` command.
  - Orchestrator._select_engine routing order: explicit request > bridge if an image/audio was
    supplied > instruct if the assembled context needs synthesis across >2 chunks > scratch.
    scratch is the DEFAULT, not a fallback.

FORBIDDEN
  - FAISS / HNSW / any vector DB.
  - A cross-encoder reranker as the default. HeuristicReranker is the default.
  - Any answer path without Citation objects.
  - print() anywhere in mmar/rag or mmar/serve.
  - Any file containing "# TODO" or "raise NotImplementedError".
  - Reporting a command as passing without pasting its raw output.

TASKS
  1. rag/store.py with schema, migrations, transactions, stats().
  2. rag/embed.py: MlxTextEmbedder + HashEmbedder + EmbeddingCache.
  3. rag/index.py: DenseIndex, BM25Index, reciprocal_rank_fusion.
  4. rag/retriever.py: Retriever, HeuristicReranker, ContextBuilder.
  5. models/textgen.py: MlxInstructTextGen with a memory check before load, and stream().
  6. llm/prompt.py: format_qa, build_rag_prompt, extract_citation_markers.
  7. serve/orchestrator.py: chat, stream_chat, search, _select_engine, _residency_plan.
  8. tests/eval_sets/file_qa.yaml - 10 questions about 10 REAL files of yours, each with the
     expected answer substring and the expected source. Then RUN it and report the score.

EXIT GATE - paste the complete unedited output
  make ingest IN=<a folder of your own mixed files>
  make index
  make chat                          # or: python -m mmar.cli search "..."
  pytest tests/test_orchestrator.py tests/test_retrieval.py -v
  /usr/bin/time -l python -m mmar.cli ingest <folder>
  # then run the 10-question eval and report the score

EVIDENCE REQUIRED IN YOUR RESPONSE
  1. make test-fast plus the retrieval/orchestrator test output.
  2. The 10-question evaluation result: question, expected, actual answer, answer correct?,
     citation correct? Report the TRUE score even if it is 3/10, and name which ones failed
     and why.
  3. The embedding cache hit rate on a second ingest of the same folder.
  4. RSS numbers for ingest and for chat, against their budgets.
  5. Anything you could NOT verify, stated explicitly as UNVERIFIED.

This phase is MILESTONE 1. Do not declare it complete on the strength of passing tests alone -
the milestone is "8 of 10 real questions about my own files answered correctly with the right
citation". Report that number.

Before writing code, answer in at most three lines each: what is the exit gate; which command
proves it; the most likely way to get this silently wrong.
```

