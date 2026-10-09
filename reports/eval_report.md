# Eval report: Ledgerly support assistant (RAG)

## Recommendation

**Do not ship yet.** Reason: the judge did not clear the calibration bar, so its scores cannot support a launch decision.

8 of 8 runs clear all five gates on the judge's scores, but those scores are not verified until the judge clears calibration.

### Launch gates (set before any results were seen)

| Metric | Gate | `baseline` |
|---|---|---|
| Faithful | >= 90% | 98% pass |
| Relevant | >= 85% | 100% pass |
| Unanswerable safe | >= 80% | 89% pass |
| False abstain | <= 10% | 0% pass |
| Recall@k | >= 85% | 97% pass |

## Judge calibration

`gemini-3.5-flash-lite` vs 15 hand labels. Pass/fail kappa: faithfulness 0.86, relevance 0.38 (bar 0.6). Judge NOT trusted. Details in [calibration.md](calibration.md).

## Experiments (one variable changed at a time)

Baseline: chunk size 800, overlap 100, `gemini-embedding-001`, prompt `v1_basic`, top-k 4, generator `gemini-3.1-flash-lite`, judge `gemini-3.5-flash-lite`. 49 golden questions per run. Deltas are faithfulness pass rate vs baseline with 95% paired bootstrap CIs.

### Experiment 1: chunk size

| Run | Recall@k | MRR | Faithful | Relevant | Unanswerable safe | False abstain | Faithful vs baseline |
|---|---|---|---|---|---|---|---|
| `chunk_size=400` | 98% | 0.96 | 98% | 100% | 100% | 0% | +0 pp [-6 pp, +6 pp] |
| `baseline` | 97% | 0.96 | 98% | 100% | 89% | 0% | baseline |
| `chunk_size=1600` | 97% | 0.96 | 100% | 100% | 100% | 0% | +2 pp [+0 pp, +6 pp] |

Best on faithfulness (ties broken by relevance): `chunk_size=1600`.

### Experiment 2: embedding model

| Run | Recall@k | MRR | Faithful | Relevant | Unanswerable safe | False abstain | Faithful vs baseline |
|---|---|---|---|---|---|---|---|
| `baseline` | 97% | 0.96 | 98% | 100% | 89% | 0% | baseline |
| `embedding=bge-small-en-v1.5` | 95% | 0.94 | 98% | 100% | 100% | 0% | +0 pp [-6 pp, +6 pp] |
| `embedding=tfidf` | 92% | 0.89 | 96% | 98% | 100% | 0% | -2 pp [-10 pp, +4 pp] |

Best on faithfulness (ties broken by relevance): `baseline`.

### Experiment 3: prompt

| Run | Recall@k | MRR | Faithful | Relevant | Unanswerable safe | False abstain | Faithful vs baseline |
|---|---|---|---|---|---|---|---|
| `baseline` | 97% | 0.96 | 98% | 100% | 89% | 0% | baseline |
| `prompt=v2_grounded` | 97% | 0.96 | 100% | 98% | 100% | 2% | +2 pp [+0 pp, +6 pp] |
| `prompt=v3_grounded_clarify` | 97% | 0.96 | 100% | 100% | 100% | 0% | +2 pp [+0 pp, +6 pp] |

Best on faithfulness (ties broken by relevance): `prompt=v3_grounded_clarify`.

### Confirmation run: winners combined

`combined` = baseline with chunk_size=1600, prompt=v3_grounded_clarify. Faithful 98%, relevant 100%, unanswerable safe 100% (+0 pp faithfulness vs baseline, CI -6 pp to +6 pp).

## By question type (`baseline`)

| Type | n | Recall@k | Faithful | Relevant | Correct behavior |
|---|---|---|---|---|---|
| single_hop | 25 | 100% | 100% | 100% | 100% |
| multi_hop | 8 | 83% | 100% | 100% | 100% |
| ambiguous | 7 | 100% | 100% | 100% | 100% |
| unanswerable | 9 | n/a | 89% | 100% | 89% |

Correct behavior: answers for answerable and ambiguous questions; no invented facts for unanswerable ones.

## Lowest-scoring answers (`baseline`)

| ID | Type | Faithful | Relevant | Unsupported claims or judge note |
|---|---|---|---|---|
| Q43 | unanswerable | 2 | 5 | You would need to contact Ledgerly's sales team to discuss specific costs for your organization. |
| Q41 | unanswerable | 5 | 4 | Every factual claim made in the answer is fully supported by the provided context. |
| Q47 | unanswerable | 5 | 4 | Every factual claim in the answer is directly supported by the provided context. |
| Q49 | unanswerable | 5 | 4 | The answer correctly states that the information is not available in the context. |
| Q01 | single_hop | 5 | 5 | Every factual claim in the answer is directly supported by the context. |

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
- Latency and token cost are not compared across runs: responses are cached and reused between runs, and free-tier rate limits add waits, so neither number is comparable here.
