# Eval report: Ledgerly support assistant (RAG)

> **MOCK RUN.** TF-IDF retrieval, an extractive stand-in generator and a heuristic judge. These numbers only prove the harness runs. Run `make all` with a Gemini key for real results.

## Recommendation

**Do not ship yet.** Reason: the judge did not clear the calibration bar, so its scores cannot support a launch decision.

Closest run: `baseline`. Gates still failing: False abstain.

### Launch gates (set before any results were seen)

| Metric | Gate | `baseline` |
|---|---|---|
| Faithful | >= 90% | 100% pass |
| Relevant | >= 85% | 90% pass |
| Unanswerable safe | >= 80% | 100% pass |
| False abstain | <= 10% | 15% FAIL |
| Recall@k | >= 85% | 92% pass |

## Judge calibration

`heuristic` vs 15 hand labels. Pass/fail kappa: faithfulness 0.30, relevance 0.04 (bar 0.6). Judge NOT trusted. Details in [calibration.md](sample_mock_calibration.md).

## Experiments (one variable changed at a time)

Baseline: chunk size 800, overlap 100, `gemini-embedding-001`, prompt `v1_basic`, top-k 4, generator `gemini-3.1-flash-lite`, judge `gemini-3.5-flash`. 49 golden questions per run. Deltas are faithfulness pass rate vs baseline with 95% paired bootstrap CIs.

### Experiment 1: chunk size

| Run | Recall@k | MRR | Faithful | Relevant | Unanswerable safe | False abstain | p50 latency | Gen tokens | Faithful vs baseline |
|---|---|---|---|---|---|---|---|---|---|
| `chunk_size=400` | 83% | 0.86 | 100% | 95% | 100% | 12% | 0.00s | 0 | +0 pp [+0 pp, +0 pp] |
| `baseline` | 92% | 0.89 | 100% | 90% | 100% | 15% | 0.00s | 0 | baseline |
| `chunk_size=1600` | 93% | 0.90 | 100% | 92% | 100% | 15% | 0.00s | 0 | +0 pp [+0 pp, +0 pp] |

Best on faithfulness (ties broken by relevance): `chunk_size=400`.

### Experiment 2: embedding model

| Run | Recall@k | MRR | Faithful | Relevant | Unanswerable safe | False abstain | p50 latency | Gen tokens | Faithful vs baseline |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 92% | 0.89 | 100% | 90% | 100% | 15% | 0.00s | 0 | baseline |
| `embedding=bge-small-en-v1.5` | 92% | 0.89 | 100% | 90% | 100% | 15% | 0.00s | 0 | +0 pp [+0 pp, +0 pp] |
| `embedding=tfidf` | 92% | 0.89 | 100% | 90% | 100% | 15% | 0.00s | 0 | +0 pp [+0 pp, +0 pp] |

Best on faithfulness (ties broken by relevance): `baseline`.

### Experiment 3: prompt

| Run | Recall@k | MRR | Faithful | Relevant | Unanswerable safe | False abstain | p50 latency | Gen tokens | Faithful vs baseline |
|---|---|---|---|---|---|---|---|---|---|
| `baseline` | 92% | 0.89 | 100% | 90% | 100% | 15% | 0.00s | 0 | baseline |
| `prompt=v2_grounded` | 92% | 0.89 | 100% | 90% | 100% | 15% | 0.00s | 0 | +0 pp [+0 pp, +0 pp] |
| `prompt=v3_grounded_clarify` | 92% | 0.89 | 100% | 90% | 100% | 15% | 0.00s | 0 | +0 pp [+0 pp, +0 pp] |

Best on faithfulness (ties broken by relevance): `baseline`.

### Confirmation run: winners combined

`combined` = baseline with chunk_size=400. Faithful 100%, relevant 95%, unanswerable safe 100% (+0 pp faithfulness vs baseline, CI +0 pp to +0 pp).

## By question type (`baseline`)

| Type | n | Recall@k | Faithful | Relevant | Correct behavior |
|---|---|---|---|---|---|
| single_hop | 25 | 100% | 100% | 92% | 88% |
| multi_hop | 8 | 71% | 100% | 88% | 100% |
| ambiguous | 7 | 86% | 100% | 86% | 57% |
| unanswerable | 9 | n/a | 100% | 100% | 100% |

Correct behavior: answers for answerable and ambiguous questions; no invented facts for unanswerable ones.

## Lowest-scoring answers (`baseline`)

| ID | Type | Faithful | Relevant | Unsupported claims or judge note |
|---|---|---|---|---|
| Q06 | single_hop | 5 | 3 | min sentence overlap 1.00 |
| Q13 | single_hop | 5 | 3 | min sentence overlap 1.00 |
| Q27 | multi_hop | 5 | 3 | min sentence overlap 1.00 |
| Q36 | ambiguous | 5 | 3 | min sentence overlap 1.00 |
| Q03 | single_hop | 5 | 4 | abstention |

## Method

- Corpus: 59 synthetic help center articles for a fictional invoicing product, Ledgerly (`data/corpus`).
- Golden set: 49 questions with source doc IDs: single-hop, multi-hop, ambiguous (answer depends on plan or method) and unanswerable (`data/golden`).
- Retrieval: recall@k and MRR at the document level, scored against the golden source IDs.
- Faithfulness and relevance: LLM judge on a 1 to 5 rubric, pass is 4 or higher (`src/rageval/judge.py`).
- Uncertainty: 95% paired bootstrap CIs over questions. With about 50 questions, differences under 10 pp are often not distinguishable from noise.

## Limitations

- Synthetic corpus and a single labeler. Real tickets would be messier.
- About 50 questions limits statistical power; treat small deltas as directional.
- The judge model is from the same provider as the generator, which can bias scores. Calibration checks this only on 15 items.
- Latency includes API network time and varies by run.
