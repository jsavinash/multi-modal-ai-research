# mmar - Multi-Modal AI Research (Apple Silicon / 8 GB)
# GNU Make 3.81 compatible (macOS default). Recipes MUST be tab-indented.
# Do not add anything here that requires >4.6 GB RSS; see docs/00_ANALYSIS.md.

SHELL      := /bin/bash
PY         := python3.11
VENV       := .venv
PYV        := $(VENV)/bin/python
PIP        := $(VENV)/bin/pip
CONFIG     ?= configs/default.yaml
VENV_SIZE_LIMIT_MB := 4600

.PHONY: help venv install install-perception install-serve install-dev info doctor \
        test test-fast test-parity lint fmt typecheck \
        data-tinystories tokenize train-micro train-small resume chat \
        ingest embed index serve eval clean nuke memory

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
venv: ## Create the 3.11 virtualenv (3.14 has no MLX/PyObjC wheels)
	$(PY) -m venv $(VENV)
	$(PIP) install -U pip
	@echo "venv ready -> $(VENV)"

install: venv ## Phase 0: core + dev tooling only (smallest safe footprint)
	$(PIP) install -e ".[dev]"
	@$(MAKE) --no-print-directory doctor

install-perception: ## Phases 1-3: text/docs/OCR + frozen encoder backends
	$(PIP) install -e ".[text,docs,ocr-macos,media]"
	$(PIP) install -e ".[mlx,perception]"

install-serve: ## Phase 11: FastAPI serving stack
	$(PIP) install -e ".[serve]"

info: ## Print the runtime + memory profile of this machine
	$(PYV) -m mmar.cli info

doctor: ## Verify the toolchain and print a pass/fail table
	$(PYV) -m mmar.cli doctor

memory: ## Live memory pressure: RSS of this process + system memory_pressure
	@$(PYV) -c "from mmar.utils.memory import snapshot; print(snapshot().render())"

# --------------------------------------------------------------------------
# Quality gates
# --------------------------------------------------------------------------
test: ## Full test suite
	$(VENV)/bin/pytest

test-fast: ## Skip slow/heavy/parity tests - the inner development loop
	$(VENV)/bin/pytest -m "not slow and not heavy and not parity"

test-parity: ## NumPy reference vs MLX numerical parity (Phase 7 gate)
	$(VENV)/bin/pytest -m parity

lint:
	$(VENV)/bin/ruff check src tests

fmt:
	$(VENV)/bin/ruff check --fix src tests
	$(VENV)/bin/ruff format src tests

typecheck:
	$(VENV)/bin/mypy

# --------------------------------------------------------------------------
# Data + tokenizer (Phases 5, 8)
# --------------------------------------------------------------------------
data-tinystories: ## Download + shard TinyStories into memory-mapped token shards
	$(PYV) -m mmar.llm.data --dataset tinystories --out data/tinystories

tokenize: ## Train the BPE tokenizer from scratch (Phase 5)
	$(PYV) -m mmar.llm.tokenizer train --corpus data/tinystories/train.txt \
		--vocab-size 4096 --out artifacts/tokenizer

# --------------------------------------------------------------------------
# From-scratch training (Phases 8, 9, 10)
# --------------------------------------------------------------------------
train-smoke: ## Phase 8a: ~40M-token validation run (~1-1.5 h). Do this FIRST
	$(PYV) -m mmar.llm.train --config configs/train_smoke.yaml

train-micro: ## Phase 8b: THE REAL RUN. 220M tokens, Chinchilla-optimal (~3-8 h, overnight)
	$(PYV) -m mmar.llm.train --config configs/train_micro.yaml

train-small: ## NOT RECOMMENDED: 39M params needs ~780M tokens (~70 h). See docs/00_ANALYSIS.md 5.2
	$(PYV) -m mmar.llm.train --config configs/train_small.yaml

resume: ## Resume the latest checkpoint for the active preset
	$(PYV) -m mmar.llm.train --config configs/train_micro.yaml --resume auto


chat: ## Interactive loop against the local from-scratch model
	$(PYV) -m mmar.cli chat --config $(CONFIG)

# --------------------------------------------------------------------------
# Ingestion + retrieval (Phases 1-4)
# --------------------------------------------------------------------------
ingest: ## IN=path/to/file-or-dir - normalise anything into MediaDocs
	@test -n "$(IN)" || (echo "usage: make ingest IN=<path>" && exit 2)
	$(PYV) -m mmar.cli ingest "$(IN)" --config $(CONFIG)

embed: ## Re-embed everything already ingested
	$(PYV) -m mmar.cli embed --config $(CONFIG)

index: ## Build the hybrid (dense + BM25) retrieval index
	$(PYV) -m mmar.cli index build --config $(CONFIG)

# --------------------------------------------------------------------------
# Serving + evaluation (Phases 11-12)
# --------------------------------------------------------------------------
serve: ## Run the local FastAPI server on 127.0.0.1:8000
	$(PYV) -m uvicorn mmar.serve.app:app --host 127.0.0.1 --port 8000

eval: ## Phase 12 harness: ppl, VQA, WER, recall@k, latency, peak RSS
	$(PYV) -m mmar.cli eval --config $(CONFIG) --out artifacts/eval/report.json

# --------------------------------------------------------------------------
# Housekeeping
# --------------------------------------------------------------------------
clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__

nuke: ## DESTRUCTIVE: drop venv, caches and all generated artifacts
	rm -rf $(VENV) data artifacts indexes checkpoints logs
