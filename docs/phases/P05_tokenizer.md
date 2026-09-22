# P5 — Tokenizer From Scratch

| | |
|---|---|
| **Goal** | A byte-level BPE tokenizer trained on the corpus, with no borrowed vocabulary, that round-trips losslessly and hits ≥ 3.2 chars/token |
| **Wall-clock** | 1 evening |
| **Compute** | ~2 min (200 MB subset, vocab 4 096) |
| **Prerequisites** | P0 gate green. The TinyStories corpus downloaded (`make data-tinystories`) |
| **Blocks** | P7, P8 |
| **Read first** | [analysis §4.1](../00_ANALYSIS.md#41-parameter-count-the-formula-to-memorise), [§11.2](../00_ANALYSIS.md#112-tokenizer-design), [docs/02 §6.1](../02_FUNCTION_REQUIREMENTS.md#61-mmar-llmtokenizerpy) |

---

## Deliverables

```
src/mmar/llm/__init__.py  tokenizer.py  bpe_train.py  data.py
tests/test_tokenizer_roundtrip.py  test_tokenizer_chat.py  test_tokenizer_special.py
artifacts/tokenizer/tokenizer.json      (generated, gitignored)
```

## Function requirements

[docs/02 §6.1](../02_FUNCTION_REQUIREMENTS.md#61-mmar-llmtokenizerpy) — implement every signature
there. The five properties, each asserted by a test:

1. **Round-trip lossless:** `decode(encode(s)) == s` for ASCII, emoji, CJK, RTL, combining marks and
   embedded NUL bytes. Not "mostly" — exactly.
2. **`encode("") == []`** and `decode([]) == ""`.
3. **Special tokens are un-forgeable from ordinary text.** The pre-tokeniser must not be able to
   produce `<|user|>`, because that would let user input forge a role boundary in P9/P10.
4. **Determinism:** identical text yields identical ids on any machine and any run.
5. **`compression_ratio(corpus_sample) >= 3.2`** on the training corpus, else the effective context
   is smaller than `max_seq_len` claims.

## Why these specific decisions

| Decision | Reason (analysis §11.2) |
|---|---|
| Vocab **4 096**, not 50 257 | The embedding table is `V·d`. At 50 257 × 384 it would be 19.3 M params, i.e. 1.7× the rest of the `micro` model |
| BPE **implemented here**, not `tokenizers` | A borrowed tokenizer undercuts "from scratch" (ADR-5) and `tokenizers` training would be a black box in the one component where readability matters most |
| GPT-2-style regex + `bytes_to_unicode` | Handles arbitrary Unicode without raising, and is the only formulation that is byte-level *and* regex-pre-tokenised |
| Train merges on a **200 MB subset**, then apply frozen merges to the whole corpus | Tokenizer training over 470 M tokens would take many minutes and hold large frequency tables. The merge list is nearly identical either way |
| **uint16** ids in shards | `vocab_size = 4 096 < 65 536`, so this halves shard size and disk bandwidth |

## Forbidden

- `transformers.AutoTokenizer`, `tokenizers.Tokenizer`, `sentencepiece`, or any pretrained vocab.
- Regenerating merges on the full corpus "for better quality" — measure it if you disagree, but do
  not do it silently.
- A round-trip test restricted to ASCII. The Unicode cases are the point.
- Storing ids as `int64` in the shard writer.
- Training the tokenizer before the corpus is actually on disk (a partial-corpus tokenizer silently
  produces a vocabulary that cannot encode the rest of it).

## Tasks

1. `bpe_train.py`: `bytes_to_unicode`, `get_pairs`, the merge loop, and `save()`.
2. `tokenizer.py`: `Tokenizer` with `encode`, `decode`, `encode_batch`, `encode_file`, `count`,
   `compression_ratio`, `encode_chat`, `fingerprint`.
3. `data.py`: `download_tinystories` (split by **document**), `TokenShardWriter`, `ShardMeta`,
   `MemmapDataset`, `build_shards`, `load_shards`. The shard `meta.json` must record the tokenizer
   fingerprint and `load_shards` must raise `TokenizerMismatch` on disagreement (risk R4).
4. The three test files.
5. Train the real tokenizer and record: vocab size, compression ratio, merge count, train time, peak RSS.

## Exit gate

```bash
make data-tinystories
make tokenize
python -m mmar.cli tokenize compress
python -m mmar.cli tokenize test --string "café 🎉 日本語 مرحبا"
make test-fast
```

## Definition of Done

- [ ] Round-trip lossless for ASCII, emoji, CJK, RTL, combining marks, embedded NUL — raw test output pasted
- [ ] Compression ratio ≥ 3.2 measured on the corpus; the number pasted
- [ ] `<|user|>` / `<|assistant|>` NOT producible from ordinary text; test green
- [ ] `encode_chat` returns a loss mask that is 0 on prompt tokens and 1 on assistant tokens
- [ ] Tokenizer training time and peak RSS measured and reported
- [ ] Shard files written as `uint16`; shard sizes reported
- [ ] `docs/12_RESULTS.md` updated with: vocab size, merges, compression ratio, train time, peak RSS

---

## 📋 Copy-paste prompt

```
ROLE
You are a senior ML systems engineer implementing a tokenizer from scratch. You write code
that is measured, not asserted.

CONTEXT - read these first
  1. docs/00_ANALYSIS.md  section 11.2 (tokenizer design decisions), section 11.3 (sharding),
     section 4.1 (why vocab 4096)
  2. docs/02_FUNCTION_REQUIREMENTS.md  section 6.1 (the full Tokenizer API)
  3. Existing code: configs/default.yaml (llm.tokenizer block), src/mmar/utils/io.py

MACHINE
  MacBook Air M1, 8 GB unified memory. python3.11 in .venv.
  Peak RSS for tokenizer training must stay under 1.5 GB. Measure it.

HARD CONSTRAINTS
  - BPE implemented HERE. No transformers.AutoTokenizer, no tokenizers.Tokenizer, no
    sentencepiece, no pretrained vocabulary. The merge loop is 30 lines and it is the
    deliverable.
  - Train merges on a BOUNDED RANDOM SUBSET (default 200 MB), then apply the FROZEN merges to
    the full corpus in the encode path. Training merges over 470 M tokens would take minutes
    and hold large frequency tables.
  - Special tokens MUST NOT be producible from ordinary text. A user must not be able to type
    something that encodes to <|user|> - that would let input forge a role boundary in P9/P10.
    Write an explicit test for this.
  - Round-trip must be EXACTLY lossless for ASCII, emoji, CJK, RTL, combining marks and an
    embedded NUL byte. Not "mostly". decode(encode(s)) == s.
  - Shard ids are uint16 (vocab 4096 < 65536). Halves both file size and disk bandwidth.
  - Train/val split is BY DOCUMENT, never by line. Line-splitting TinyStories leaks
    half-sentences across the split and produces a fake val loss around 1.0.

DELIVERABLES - exactly these paths
  src/mmar/llm/__init__.py
  src/mmar/llm/tokenizer.py
  src/mmar/llm/bpe_train.py
  src/mmar/llm/data.py
  tests/test_tokenizer_roundtrip.py
  tests/test_tokenizer_chat.py
  tests/test_tokenizer_special.py
  artifacts/tokenizer/tokenizer.json          (generated)
  docs/12_RESULTS.md                          (UPDATE with the P5 measurements)

KEY REQUIREMENTS
  - Implement every signature in docs/02_FUNCTION_REQUIREMENTS.md section 6.1: BPETrainer,
    BPEArtifacts, Tokenizer(load/train_new/encode/decode/encode_batch/encode_file/count/
    compression_ratio/encode_chat/fingerprint), bytes_to_unicode, get_pairs.
  - fingerprint() = sha256 over a canonical serialisation of merges+vocab+specials. It is
    recorded in the shard meta.json and in every checkpoint. load_shards() must RAISE
    TokenizerMismatch on disagreement - this is risk R4, the single most common silent failure
    in this project (training succeeds, loss falls, output is gibberish).
  - encode_chat(turns, add_generation_prompt, mask_prompt) returns (input_ids, loss_mask) where
    loss_mask[i] == 1 ONLY on assistant tokens. This one function is what makes Phase 9 correct.
  - data.py: download_tinystories (split by document), TokenShardWriter, ShardMeta,
    MemmapDataset (np.memmap random (B, T+1) windows, zero RAM duplication),
    build_shards(max_train_tokens, max_val_tokens), load_shards(expected_fingerprint).
  - Pre-tokenisation uses the GPT-2 regex with bytes_to_unicode so arbitrary Unicode never raises.

FORBIDDEN
  - Any pretrained tokenizer or vocabulary.
  - A round-trip test restricted to ASCII.
  - int64 ids in the shard writer.
  - Training the tokenizer on a partial corpus (it silently produces a vocab that cannot encode
    the rest of it).
  - Any file containing "# TODO" or "raise NotImplementedError".
  - Reporting a command as passing without pasting its raw output.

TASKS
  1. bpe_train.py: bytes_to_unicode, get_pairs, the merge loop, save().
  2. tokenizer.py: the full Tokenizer API.
  3. data.py: downloader, shard writer, memmap dataset, fingerprint checking.
  4. The three test files, including the exact-Unicode round-trip and the un-forgeable-special
     test.
  5. make data-tinystories ; make tokenize
  6. Record: vocab size, merge count, compression ratio, training wall-clock, peak RSS.

EXIT GATE - paste the complete unedited output
  make data-tinystories
  make tokenize
  python -m mmar.cli tokenize compress
  python -m mmar.cli tokenize test --string "café 🎉 日本語 مرحبا"
  make test-fast
  /usr/bin/time -l .venv/bin/python -m mmar.cli tokenize train --corpus data/tinystories/train.txt

EVIDENCE REQUIRED IN YOUR RESPONSE
  1. make test-fast raw output, and the specific round-trip test output.
  2. The compression ratio number from `mmar tokenize compress`, compared against the 3.2 target.
  3. The encode/decode line for the Unicode probe string above, showing MATCH.
  4. The tokenizer training wall-clock and the max-RSS number.
  5. The shard file sizes and their dtype.
  6. Anything you could NOT verify, stated explicitly as UNVERIFIED.

Before writing code, answer in at most three lines each: what is the exit gate; which command
proves it; the most likely way to get this silently wrong.
```

