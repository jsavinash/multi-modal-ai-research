# P10 — Multimodal Bridge (MILESTONE 4)

| | |
|---|---|
| **Goal** | Freeze pretrained vision/audio encoders, train a projector into your from-scratch LLM's token space, and run the staged alignment → instruct → audio recipe so your own model can describe images and consume speech |
| **Wall-clock** | 2 evenings |
| **Compute** | ~1.5 h unattended |
| **Prerequisites** | P8 gate green (pretrained `micro` checkpoint) and P9 gate green (SFT checkpoint); P2/P3 strongly recommended so your own images/audio are already ingested |
| **Blocks** | P11 (`engine: bridge` is a first-class path), P12 multimodal eval |
| **Read first** | [analysis §10](../00_ANALYSIS.md#10-multimodal-bridge-design), [§9.5](../00_ANALYSIS.md#95-instruction-tuning-phase-9), [§9.6](../00_ANALYSIS.md#96-what-an-11-m-param-model-can-and-cannot-do-set-expectations-in-advance) |

---

## Deliverables

```
src/mmar/llm/bridge/__init__.py  projector.py  encoders.py  dataset.py  train.py
src/mmar/models/vision.py        audio.py     # frozen-encoder adapters
tests/test_bridge_projector.py  test_encoders.py  test_bridge_dataset.py  test_bridge_train.py
artifacts/bridge/coco_captions.jsonl      # bootstrapped + downloaded caption pairs
artifacts/bridge/vqa_pairs.jsonl          # VQA pairs
artifacts/bridge/audio_captions.jsonl     # audio-caption pairs
tests/eval_sets/bridge_probes.yaml        # 50 hand-built colour/count/proximity items
checkpoints/micro-vl-stage1/              # projector-only align checkpoint
checkpoints/micro-vl-stage2/              # projector + LoRA instruct checkpoint
checkpoints/micro-vl-stage3/              # audio projector checkpoint
```

## Function requirements

Full signatures: [`docs/02 §7`](../02_FUNCTION_REQUIREMENTS.md#7-l4--the-multimodal-bridge-mmarllmbridge).

Non-negotiable behaviours:

- **`Projector` is a 2-layer MLP + GELU: d_in → 4·d_model → d_model.** Nothing else is trainable in stage 1. A more complex projector is not free at 11 M params and the literature does not support it.
- **`PatchPooler` is parameter-free 2×2 spatial merge:** 196 patch embeddings → 49 tokens by reshaping to (14, 14, d_v) → (7, 7, 4·d_v) → (49, d_v). Config-selectable `pool_factor` is the **token-count divisor**: 1 → 196, 2 → 98, 4 → 49 tokens; default **4** (this 2×2 merge, analysis §10.3).
- **Frozen encoders are always eval mode and always `no_grad`.** They are never moved to the GPU by the bridge code; the registry owns residency and enforces `max_concurrent_models: 1`.
- **Stage 1 freezes the LLM.** Training both projector and LM from the start lets the LM unlearn English while the projector is still random — gradients from garbage image tokens are pure noise. This is the single most important detail of the LLaVA recipe.
- **Audio token budget is enforced:** Whisper encoder produces ~1500 frames; subsample 6× → 250 frames; mean-pool 5× → 50 tokens. `meta["audio_tokens"]` must be recorded.
- **The forgetting gate is evaluated after every stage:** TinyStories val ppl must stay ≤ 1.3× the P8 baseline.
- **20 % raw pretraining text is mixed into every stage-2 and stage-3 batch** to prevent catastrophic forgetting. `MultimodalDataset` has `include_text_only_fraction` exactly for this.

## Architecture

```
  image ──► SigLIP-base/CLIP ViT-B/32 (FROZEN) ──► 196 patch feats (d_v = 768)
                                                         │
                                           Projector (TRAINED, 2-layer MLP + GELU)
                                           d_v -> 4*d_model -> d_model
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

## Token budget arithmetic

At `max_seq_len = 512` and `d_model = 384`:

| Source | Raw | After projection | Cost |
|---|---|---|---|
| CLIP ViT-B/32 image | 196 patches | 49 tokens (pool_factor=4) | ~10 % of context |
| Whisper audio, 30 s window | 1 500 frames | 50 tokens | ~10 % of context |
| Text prompt | — | ~100 tokens | ~20 % of context |
| **Answer budget left** | | **~313 tokens** | Enough for a short caption or answer |

**Why pool_factor=4 is the default:** without reduction the model spends 38 % of context on image tokens and produces one-line answers. pool_factor=4 triples the answer budget while retaining enough spatial detail for coarse captions.

## Staged training recipe

| Stage | Trainable | Frozen | Data | Steps | LR | Time `[est]` | Effect |
|---|---|---|---|---|---|---|---|
| **0. Pretrain** | whole LM | — | TinyStories 220 M tok | 13 428 | 6e-4 | 3–8 h | Language exists |
| **1. Align** | projector only | LLM + encoders | 30 k caption pairs | ~2 000 | 1e-4 | 25–40 min | Image tokens become "words" the LM already understands |
| **2. Instruct** | projector + LoRA(r=8 on q/k/v/o) | encoders, most of LLM | 10 k VQA pairs | ~3 000 | 3e-5 | 45–90 min | Answers questions about the image |
| **3. Audio** | projector 2 only | everything else | 10 k audio-caption pairs | ~1 000 | 1e-4 | 15–25 min | Speech becomes text-like tokens |

**LoRA configuration (stage 2 only):**
- Rank r=8, alpha=16, dropout=0.05
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj` in every `TransformerBlock`
- ~0.5 M additional trainable params; fits inside the 1.2 GB bridge-training budget

## Data strategy

| Need | Source | Size | Why |
|---|---|---|---|
| Caption pairs (stage 1) | COCO-2014 subset, 30 k images | ~5 GB images | Canonical LLaVA alignment set; captions are short and templated |
| VQA pairs (stage 2) | VQA-v2 subset, 10 k QA | ~2 GB | Natural question/answer phrasing |
| Audio pairs (stage 3) | LibriSpeech `dev-clean` + `test-clean` | ~1 GB | Word-level transcripts give exact supervision |
| Bootstrapped | SmolVLM-256M captions over **your own** photos | unlimited | Legal, free, personal, cheap (~1 image/s) |
| Probes | Hand-built `tests/eval_sets/bridge_probes.yaml` | 50 items | Colour, count, proximity questions |

**Bootstrapping is the highest-leverage move:** run SmolVLM-256M over every image in your own library once, store the captions as `MediaDoc.chunks` with `lane=VLM`, and you have a domain-matched training set with zero labelling cost.

## Evaluation and honest expectations

| Metric | Target | Measurement |
|---|---|---|
| Caption exact-ish match (teacher-forced, ROUGE-L vs SmolVLM teacher) | ≥ 0.25 | `mmar eval --tasks caption` |
| VQA accuracy on a 500-question held-out split | ≥ 25 % | `mmar eval --tasks vqa` |
| Colour/count/proximity probes ("is the cat left of the dog?") | ≥ 60 % on 50 hand-built items | `mmar eval --tasks probe` |
| Catastrophic forgetting check: TinyStories val ppl after stage 2 | ≤ 1.3× the Phase-8 value | `mmar eval --tasks ppl` |

**Honest expectation for `mmar-vl-micro`:** it will produce short, mostly-grammatical descriptions containing genuinely correct coarse content (object presence, dominant colour, scene type) and unreliable detail. It is a working end-to-end multimodal transformer you trained yourself — which is the goal — not a useful image assistant. For useful image understanding, route to Track D's SmolVLM-256M.

## Forbidden

- Training the LLM and projector simultaneously from random init (stage 1 must freeze the LLM).
- Loading a VLM for training data generation; SmolVLM-256M is used only at inference time over your own images.
- Skipping the forgetting gate. If TinyStories val ppl regresses more than 1.3×, the stage is not done.
- Full fine-tuning in stage 2. LoRA only; full fine-tuning would catastrophically forget in ~3 000 steps on 10 k VQA examples.
- A cross-attention projector. The LLaVA MLP projector is what the literature validates at this scale.

## Tasks

1. `llm/bridge/projector.py`: `Projector` (2-layer MLP + GELU) and `PatchPooler` (parameter-free spatial merge). Tests assert output shapes and that PatchPooler is deterministic and parameter-free.
2. `llm/bridge/encoders.py`: `FrozenVisionEncoder` and `FrozenAudioEncoder` protocols, plus `ClipVisionEncoder` and `WhisperAudioEncoder` implementations that always run in eval mode and `no_grad`.
3. `llm/bridge/dataset.py`: `MultimodalDataset` and `ModalityBatch`. Tests assert the 20 % text-only fraction, correct input_ids/targets/loss_mask assembly, and that image/audio token positions are masked in the loss.
4. `llm/bridge/train.py`: `BridgeTrainer` implementing the four-stage recipe. Tests assert stage-1 LLM weights are frozen, stage-2 LoRA adapter is the only trainable LLM module, and the forgetting gate fires correctly.
5. Build the three dataset files (`coco_captions.jsonl`, `vqa_pairs.jsonl`, `audio_captions.jsonl`) and the probe file (`bridge_probes.yaml`).
6. Run stage 1 (align), measure ROUGE-L vs the SmolVLM teacher captions, and assert ≥ 0.25 on a 200-item held-out split.
7. Run stage 2 (instruct), measure VQA accuracy and probe accuracy, and assert forgetting gate ≤ 1.3×.
8. Run stage 3 (audio), measure caption quality on a 100-item held-out split.
9. Run the full gate; write `docs/12_RESULTS.md` bridge section.

## Exit gate

```bash
make test-fast
python -m mmar.llm.bridge.train --stage 1 --config configs/train_bridge_stage1.yaml
python -m mmar.llm.eval_bridge --checkpoint checkpoints/micro-vl-stage2 --tasks caption,vqa,probe
pytest tests/test_bridge_train.py -v
```

**Honest score:** report the real ROUGE-L, real VQA accuracy, real probe accuracy, and real forgetting ratio. If any number is below target, name the most likely cause (dataset size, model capacity, training instability) rather than rerunning until it passes.

## Definition of Done

- [ ] `Projector` and `PatchPooler` implemented and tested; output shapes verified against token budget arithmetic
- [ ] Frozen encoders load without allocating Metal buffers until `encode()` is called; `test_encoders.py` asserts this
- [ ] Stage 1 completes with projector-only training; LLM weights verified unchanged via checksum
- [ ] Stage 2 uses LoRA only; adapter params counted and reported
- [ ] Forgetting gate measured after stages 2 and 3; real ratio pasted
- [ ] Caption ROUGE-L, VQA accuracy, and probe accuracy measured on held-out splits; real numbers pasted
- [ ] 10 generated captions read by a human and pasted; at least 5 must contain correct coarse content
- [ ] Peak RSS measured for each stage and reported against the 1.2 GB budget
- [ ] `docs/12_RESULTS.md` updated with bridge measurements

---

## 📋 Copy-paste prompt

```
ROLE
You are a senior ML systems engineer. You write code that is measured, not asserted.

CONTEXT - read these first, in this order
  1. docs/00_ANALYSIS.md  section 10 (multimodal bridge design), section 9.5 (SFT),
     section 4.5 (resident profiles), section 10.3 (token budget arithmetic)
  2. docs/02_FUNCTION_REQUIREMENTS.md  section 7 (bridge)
  3. docs/01_ROADMAP.md  section 5 (Definition of Done) and MILESTONE 4
  4. Existing code: src/mmar/llm/{model,layers,train,sample,checkpoint}.py,
     src/mmar/llm/tokenizer.py, src/mmar/models/registry.py, src/mmar/rag/chunker.py

MACHINE
  MacBook Air M1, 8 GB unified memory, fanless. python3.11 in .venv.
  Peak RSS budget: 1.2 GB for bridge training (all stages combined).

HARD CONSTRAINTS
  - The LLM MUST be frozen during stage 1. Training both from random init lets the LM
    unlearn English while the projector is still random. This is the LLaVA recipe and
    it is not optional.
  - All frozen encoders are eval mode + no_grad. They do not allocate Metal buffers until
    encode() is called.
  - PatchPooler is parameter-free. If you are tempted to make it learnable, stop and re-read
    analysis section 10.3.
  - 20% raw pretraining text in every stage-2 and stage-3 batch. The MultimodalDataset
    include_text_only_fraction parameter enforces this.
  - The forgetting gate must be checked after stages 2 and 3. TinyStories val ppl must stay
    within 1.3x of the P8 baseline.
  - LoRA in stage 2 only: r=8, alpha=16, target q/k/v/o projections. Full fine-tuning
    would catastrophically forget in ~3000 steps.

DELIVERABLES - exactly these paths
  src/mmar/llm/bridge/__init__.py
  src/mmar/llm/bridge/projector.py
  src/mmar/llm/bridge/encoders.py
  src/mmar/llm/bridge/dataset.py
  src/mmar/llm/bridge/train.py
  src/mmar/models/vision.py
  src/mmar/models/audio.py
  tests/test_bridge_projector.py
  tests/test_encoders.py
  tests/test_bridge_dataset.py
  tests/test_bridge_train.py
  tests/eval_sets/bridge_probes.yaml
  artifacts/bridge/coco_captions.jsonl
  artifacts/bridge/vqa_pairs.jsonl
  artifacts/bridge/audio_captions.jsonl
  checkpoints/micro-vl-stage1/
  checkpoints/micro-vl-stage2/
  checkpoints/micro-vl-stage3/

KEY REQUIREMENTS
  - Projector: mx.nn.Module, 2 linear layers + GELU, d_in -> 4*d_model -> d_model.
    Projector 1 (images): d_in = 768 (CLIP ViT-B/32 / SigLIP-base patch width).
    Projector 2 (audio): d_in = encoder.d_out of the CONFIGURED ASR backend - 768 for
    whisper-base, 1280 for large-v3-turbo. Assert d_in == encoder.d_out at construction;
    a mismatch must raise immediately, never surface as a matmul error mid-training.
  - PatchPooler: pool_factor is the token-count divisor {1, 2, 4}; default 4.
    At 4: reshape (B, 196, d_v) -> (B, 14, 14, d_v) -> (B, 7, 7, 4*d_v) -> (B, 49, d_v).
    At 1: identity (196 tokens); at 2: merge pairs -> 98 tokens. All three divide a 14x14
    grid evenly - never pad, never drop edge patches.
  - Frozen encoders: acquire() through ModelRegistry; encode() under no_grad; eval mode.
    The registry enforces max_concurrent_models: 1.
  - ModalityBatch: input_ids (B, T) with placeholder tokens at image/audio positions;
    modality_tokens (B, M, d_model) projected encoder outputs; modality_mask (B, T) boolean.
  - BridgeTrainer: four-stage loop, each stage with its own config, data file, and step count.
    Saves checkpoints with is_bridge=True and stage number in meta.json.
  - Caption pairs: {"image_path": ..., "caption": ...}. Build from COCO subset plus your
    own images captioned by SmolVLM-256M.
  - VQA pairs: {"image_path": ..., "question": ..., "answer": ...}. Build from VQA-v2 subset.
  - Audio pairs: {"audio_path": ..., "caption": ...}. Build from LibriSpeech transcripts.
  - Probe file: list of {"image_path": ..., "question": ..., "expected_keywords": [...]}.
    50 items covering colour, count, and spatial proximity.

FORBIDDEN
  - Training the LLM in stage 1.
  - Loading a VLM to generate training captions; SmolVLM-256M is inference-only.
  - Skipping the forgetting gate.
  - Any file containing "# TODO" or "raise NotImplementedError".
  - Reporting a command as passing without pasting its raw output.

TASKS
  1. Projector + PatchPooler with tests verifying shapes and parameter counts.
  2. Frozen encoder adapters with tests asserting eval mode + no_grad + lazy allocation.
  3. MultimodalDataset with tests for the 20% text-only fraction and correct loss masking.
  4. BridgeTrainer with tests for each stage's frozen/trainable split and the forgetting gate.
  5. Build the three dataset files and the probe file.
  6. Run stage 1 (align), measure ROUGE-L on 200-item held-out split, report honestly.
  7. Run stage 2 (instruct), measure VQA accuracy + probe accuracy + forgetting ratio.
  8. Run stage 3 (audio), measure caption quality on 100-item held-out split.
  9. Run the gate; write docs/12_RESULTS.md bridge section.

EXIT GATE - paste the complete unedited output
  make test-fast
  pytest tests/test_bridge_train.py -v
  python -m mmar.llm.eval_bridge --checkpoint checkpoints/micro-vl-stage2 --tasks caption,vqa,probe

EVIDENCE REQUIRED IN YOUR RESPONSE
  1. make test-fast raw output.
  2. Stage-1 ROUGE-L on the 200-item held-out split, with per-bucket breakdown.
  3. Stage-2 VQA accuracy, probe accuracy, and forgetting ratio.
  4. Stage-3 audio caption quality.
  5. Peak RSS for each stage, compared against the 1.2 GB budget.
  6. 10 generated captions pasted verbatim; at least 5 must have correct coarse content.
  7. Anything you could NOT verify, stated explicitly as UNVERIFIED.

Before writing code, answer in at most three lines each: what is the exit gate; which command proves it; the most likely way to get this silently wrong.
