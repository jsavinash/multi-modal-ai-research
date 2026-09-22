# 03 — Prompt Playbook

How to drive an AI coding agent through Phases 0–12 **without it reporting success it did not
achieve**. Read this before pasting any phase prompt.

---

## 1. Why this document exists

This project has an unusually high density of **silent failure modes**. On a normal CRUD app, a wrong
implementation fails a test or throws. Here:

| Silent failure | Looks like success | Actually means |
|---|---|---|
| Causal mask built as `(T, T)` instead of `(q_len, kv_len, offset)` | Loss falls smoothly | Every cached decode is wrong |
| Weight decay applied to norm weights | Loss plateaus around 3.5 | The model is being actively sabotaged |
| Tokenizer/shard hash mismatch | Loss falls, training "converges" | The output is fluent nonsense |
| Loss computed in bf16 | Loss falls | The gradient is subtly biased; quality is capped |
| `mlx` cache limit not set | Everything works | A 500 MB model uses 2 GB and the machine swaps |
| Sampler using a buggy framework helper | Model seems broken | The model is fine, the sampler is not |
| Memory regression from 1.2 GB → 2.6 GB | Nothing fails | Everything is 5× slower and you blame the model |

An agent that "writes the code and says it works" will produce every one of these. So the playbook is
not about making the agent write better code — it is about making the agent **produce evidence**.

---

## 2. The three protocols

### Protocol A — one phase, one session, one gate

1. **Read the gate first.** Open the phase file and read the exit gate *before* the task list. If you
   cannot state how the gate is proven, do not start.
2. **Paste the phase prompt verbatim**, unmodified. The prompts are written to be self-contained and
   they already contain the forbidden-action list.
3. **Require the evidence block.** Every phase prompt ends with an evidence requirement. An agent
   response without raw command output is **not a completed phase** — it is a draft.
4. **Run the gate yourself.** Never accept the agent's assertion that a command passed. Run it. This
   is the entire point of gates being binary.
5. **Commit before moving on.** `git commit -m "P0: foundation - gate green"`. The commit message
   includes the gate command. If a later phase reveals a problem, you can bisect.

### Protocol B — measure, never assert

Three rules, in priority order:

1. **Any number must come from a command whose output is pasted.** "The model uses about 700 MB" is
   not an answer. "`make memory` → peak RSS 682 MB after a 200-file ingest" is.
2. **Any claim of correctness needs a test that would fail if the code were wrong.** Ask the agent
   explicitly: *"What test would fail if this were broken? Write that test."* If it cannot name one,
   the code is untested regardless of what it says.
3. **`[est]` markers are a contract.** When a number in the docs was an estimate and you now have a
   measurement, replace the estimate and delete the marker. Never leave a stale `[est]`.

### Protocol C — the ladder of verification

Ask for the **cheapest** verification that would catch the bug, and require the next rung up at each
phase gate:

| Rung | Technique | Cost | Catches |
|---|---|---|---|
| 1 | Unit test on a hand-computed value | ms | Wrong formula |
| 2 | Round-trip / property test | ms | Wrong serialisation, lossy encoding |
| 3 | Parity against an independent implementation | ms | Subtle numeric bug (RoPE, mask, softmax dtype) |
| 4 | Overfit a single batch | seconds | Model cannot learn at all |
| 5 | Loss trajectory on real data | minutes | LR, init, data pipeline |
| 6 | Human-read generation | seconds | Everything above that still passed |
| 7 | Cross-machine / cross-version check | minutes | Environment-specific behaviour |

**Rung 6 is mandatory and cannot be delegated.** Read the output yourself. If the text is not English,
the model did not work — regardless of what the loss curve says.

---

## 3. The reusable prompt skeleton

Every phase prompt in `docs/phases/` follows this structure. When you need a prompt for something not
covered, copy this:

```
ROLE
  You are a senior ML systems engineer working on <project>. You write code that is
  measured, not asserted.

CONTEXT (read these first, in this order)
  - docs/00_ANALYSIS.md section <n>   (the design decision this phase implements)
  - docs/02_FUNCTION_REQUIREMENTS.md section <n>   (the exact signatures)
  - <existing files relevant to the phase>

HARD CONSTRAINTS
  - Machine: MacBook Air M1, 8 GB unified memory, fanless, macOS 26.x.
  - Peak RSS for any code path in this phase must stay under <n> MB. Measure it.
  - Python 3.11. MLX for tensors. No new dependency without stating why.
  - Do not modify files outside the DELIVERABLES list.

DELIVERABLES (exact paths)
  <paths>

FORBIDDEN
  - <anti-patterns specific to this phase>
  - Never report a command as passing without pasting its raw output.
  - Never mark a number as measured if you did not measure it.

TASKS (in order; stop and report if a task reveals a contradiction with the docs)
  1. …
  2. …

DEFINITION OF DONE
  [ ] <gate command> passes; raw output pasted
  [ ] <phase-specific test> exists and would fail if the code were wrong
  [ ] Peak RSS measured and reported
  [ ] Docs updated: <exact list>

EVIDENCE REQUIRED IN YOUR RESPONSE
  1. The raw output of <gate command>.
  2. The raw output of the test command.
  3. A memory measurement with the command that produced it.
  4. A list of anything you could NOT verify, stated explicitly as unverified.
```

**The last item is the most important one.** An agent that lists its unverified assumptions is far
more useful than one that claims completeness. Reward that behaviour by asking for it every time.

---

## 4. Meta-prompts

Five reusable prompts for the moments around a phase. Paste these verbatim.

### 4.1 Kickoff (before pasting a phase prompt)

```
Before you write any code for this phase, answer these five questions in at most
three lines each. Do not write code yet.

1. What is the exit gate for this phase, and what exact command proves it?
2. Which existing files will you read before starting, and why those?
3. What is the single most likely way to get this wrong such that it "works"
   but is silently incorrect?
4. What will you measure, and with what command?
5. Name one thing in the phase prompt that is ambiguous. If nothing is ambiguous,
   say what you checked to be sure.
```

If the answers to (1) and (4) do not name commands, the agent has not read the gate.

### 4.2 Mid-phase checkpoint (every ~30 min of work)

```
Pause. Without changing any code, report:
  - files created/modified so far (paths only)
  - the last command you ran and its raw output
  - what is currently failing or unverified
  - your estimate of remaining work, and whether you have hit anything that
    contradicts docs/00_ANALYSIS.md

Do not summarise success. List uncertainty.
```

### 4.3 Gate verification (at the end of the phase)

```
Run the exit gate now and paste the complete, unedited output. Then, for each item in
the Definition of Done, state either the evidence (command + output) or the word
UNVERIFIED.

Then run these three checks and paste their output:
  1. make test-fast
  2. The memory measurement for this phase
  3. git status --short and git diff --stat

Do not describe what the output "should" be. Paste what it is.
```

### 4.4 Failure triage (when a gate fails)

```
The gate failed. Do NOT attempt a fix yet.

1. Paste the complete error output.
2. State the single most likely root cause, and the evidence for it.
3. Name the smallest experiment that would confirm or eliminate that cause.
4. Check docs/01_ROADMAP.md section 6 ("What to do when a phase fails") and say whether
   this symptom is listed there.
5. Only then propose a fix.

Constraint: any fix must be validated by the same gate, with output pasted.
```

### 4.5 Context handoff (when a session gets long or you switch tools)

```
Produce a handoff document with exactly these sections, as Markdown:

## State
  Phase, gate status, last passing command with output, last failing command with output.

## Files
  Every file created/modified, with a one-line purpose each.

## Decisions
  Any decision you made that is NOT in docs/00_ANALYSIS.md, plus why.

## Open questions
  Anything ambiguous, plus your current best guess.

## Next action
  The single next thing to do, stated as a command.

Keep it under 60 lines. Assume the reader has the docs but not the conversation.
```

---

## 5. Anti-patterns to forbid explicitly

These are failure modes common in AI-generated ML code. Every phase prompt forbids them; here is why
each is on the list.

| Anti-pattern | Why it is fatal here |
|---|---|
| `# TODO: implement` / stubs raising `NotImplementedError` | The gate cannot pass, so the agent eventually "adjusts" the gate. Forbid stubs: implement it or report blocked |
| Mocking a model to make a test pass | The test then proves nothing. Only `HashEmbedder` and the `nano` preset are legitimate stand-ins, and both are real implementations |
| `try: … except Exception: pass` | Destroys the error philosophy (analysis §0.1). Every catch must re-raise as `MmarError` or log at WARNING with the exception text |
| Adding `torch` "because it was easier" | Violates ADR-1 and adds ~2 GB of install for no gain on this hardware |
| Raising `max_process_rss_mb` to make something fit | Inverts the design. The budget constrains the code, not the reverse |
| Truncating the tokenizer round-trip test to ASCII-only | Hides the exact bug class (byte-level Unicode) the test exists to catch |
| Reporting "tests pass" without output | Protocol A item 3 — the most common agent dishonesty |
| Silently changing an interface in `docs/02_FUNCTION_REQUIREMENTS.md` | Downstream phases are written against those signatures. A change must be reported, not smuggled |
| Generating the README from imagination | The format table is asserted against `registry.coverage()`; invented coverage fails CI |
| Committing `data/`, `checkpoints/`, `artifacts/` or weights | `.gitignore` covers it; a 400 MB commit is unrecoverable in practice |
| `pip install -U` of MLX mid-phase | R7: API churn breaks wrappers. Upgrade deliberately at a phase boundary, then run `make test` |

---

## 6. Managing context and long phases

Phases 1, 10 and 11 are larger than one comfortable agent session. Handling:

1. **Split by deliverable, not by time.** P1 splits naturally into "IR + registry + text extractors"
   then "pdf/office/web/archive extractors". Each half must still be independently testable.
2. **Always re-anchor on the docs, not the conversation.** Start each new session by pointing at the
   phase file and the relevant `docs/02_FUNCTION_REQUIREMENTS.md` anchor. The docs are the memory.
3. **Commit at every split point.** A split phase with an uncommitted half is one you cannot bisect.
4. **Use the handoff prompt (§4.5) whenever a session exceeds ~40 messages.** Shorter to read than the
   transcript, and it forces the agent to state uncertainty.
5. **Never let the agent "fix" a gate by weakening it.** If a gate cannot pass, the phase is
   incomplete. Gates change only through `docs/`, with a stated reason, in a separate commit a human
   reads.

---

## 7. A worked example: good vs bad phase reporting

### ❌ Bad

> I've implemented Phase 5, the BPE tokenizer. It trains on the corpus and achieves good compression.
> All tests pass. The tokenizer handles Unicode correctly and the vocab size is 4096 as specified.
> Ready to move to Phase 6.

Why it fails: no command output; "good compression" is a number the agent never measured; "handles
Unicode correctly" is an untested claim; "all tests pass" is an assertion, not evidence.

### ✅ Good

> **Phase 5 — Tokenizer. Gate: round-trip lossless + compression ≥ 3.2.**
>
> Files: `src/mmar/llm/tokenizer.py`, `src/mmar/llm/bpe_train.py`, `tests/test_tokenizer_roundtrip.py`,
> `tests/test_tokenizer_chat.py`, `artifacts/tokenizer/tokenizer.json`.
>
> ```
> $ make test-fast -k tokenizer
> tests/test_tokenizer_roundtrip.py ..........  [ 10 passed ]
> tests/test_tokenizer_chat.py .....            [  5 passed ]
> 15 passed in 3.42s
>
> $ python -m mmar.cli tokenize compress
> corpus sample        8,388,608 chars
> encoded              2,301,772 tokens
> compression ratio    3.64 chars/token          target >= 3.20   PASS
>
> $ python -m mmar.cli tokenize test --string "café 🎉 日本語 مرحبا"
> encode -> [1421, 88, 992, 4417, 2210, 771, 3390]
> decode -> "café 🎉 日本語 مرحبا"                MATCH
>
> $ /usr/bin/time -l python -m mmar.llm.bpe_train --subset-mb 200
> maximum resident set size  412 MB
> elapsed                    117 s
> ```
>
> **Not verified:** the tokenizer was trained on a 200 MB subset by design, but I measured compression
> only on an 8.4 MB sample, not the full 470 M-token corpus. If the ratio is distribution-sensitive it
> could drop below 3.2; I would flag that as a Phase 8 risk.

Why it works: every claim has a command and raw output, the measurement method is stated, and the
uncertainty is named with its implication. Hold the agent to this standard.

---

## 8. Phase prompt checklist

Before pasting any phase prompt, confirm it contains all of:

- [ ] The exact deliverable paths
- [ ] The exact exit-gate command
- [ ] The inline function signatures the phase must implement (or the `docs/02` anchor)
- [ ] A hard peak-RSS budget for the phase
- [ ] The forbidden-actions list, including the no-stub rule
- [ ] The evidence requirement, including the UNVERIFIED declaration
- [ ] The docs-update list

If any is missing, add it. An under-specified prompt is what produces a phase that "passed" without
evidence.

---

## 9. Final note on honesty

The design in `docs/00_ANALYSIS.md` deliberately states capabilities that are *unimpressive*: an
11 M-parameter model that writes simple children's stories, a VLM that gets coarse image content
right and fine detail wrong, an OCR fallback that is "good enough". That honesty is load-bearing.

If you let an agent inflate those claims — or inflate them yourself by skipping a gate — you will have
built exactly the kind of project this plan was written to avoid: one that reports success without
evidence. The gates are cheap. Run them yourself.
