# RAG Eval Framework: should this support bot ship?

**[View the case study site](https://claude.ai/artifact/HXoaNEXqsJBtq8QyeZJPBc)**

An evaluation harness for a retrieval-augmented support assistant. It answers one product question with evidence: **which configuration is safe to launch, and how do we know?**

| | |
|---|---|
| **Assistant under test** | Support bot for Ledgerly, a fictional invoicing app (59 synthetic help articles) |
| **Stack** | Python, LangChain, FAISS, Gemini (generator, judge, embeddings) |
| **Golden set** | 49 questions with source doc IDs: 25 single-hop, 8 multi-hop, 7 ambiguous, 9 unanswerable |
| **Judges** | LLM-as-judge rubrics for faithfulness and relevance, calibrated against 15 hand-labeled answers |
| **Experiments** | Chunk size, embedding model, prompt variant. One variable at a time, then a confirmation run |
| **Decision rule** | 5 launch gates set before any results were seen |
| **Output** | [`reports/eval_report.md`](reports/eval_report.md): results tables, CIs, and a ship or no-ship call |

## Results

Run `make all` to generate the report. Until then, see the format in [`reports/sample_mock_report.md`](reports/sample_mock_report.md), produced offline with stand-in components (its numbers are not meaningful).

<!-- After your run, replace this block with 3 lines from reports/eval_report.md:
**Recommendation:** Ship `<run>` (faithfulness X% vs Y% baseline, +Z pp, 95% CI ...).
**Biggest lever:** ...
**Judge agreement with humans:** kappa ... -->

## What it measures

| Metric | Question it answers | Launch gate |
|---|---|---|
| Faithfulness pass rate | Is every claim backed by the retrieved docs? | 90% or higher |
| Relevance pass rate | Does the answer address what was asked? | 85% or higher |
| Unanswerable safe rate | When the docs lack the answer, does it avoid inventing one? | 80% or higher |
| False abstention rate | How often does it refuse a question it could answer? | 10% or lower |
| Retrieval recall@k | Are the right docs in the top-k chunks? | 85% or higher |

Also reported: MRR, key-fact recall (string match against the golden answer), p50 and p95 latency, token usage, and per question type breakdowns. Differences between runs come with 95% paired bootstrap confidence intervals.

## How it works

```
data/corpus (59 docs) -> chunk -> embed -> FAISS index
                                              |
golden question -> retrieve top-k -> Gemini generator -> answer
                                              |
                     judge: faithfulness (answer vs retrieved context)
                     judge: relevance + behavior (answered, abstained, clarified)
                                              |
                     metrics -> gates -> eval_report.md
```

1. **Calibrate the judge first.** `make calibrate` scores 15 hand-labeled answers (including planted errors: wrong numbers, outdated prices, invented features, a false refusal). The judge is trusted only if pass/fail Cohen's kappa is 0.6 or higher on both rubrics. If it is not, the report refuses to recommend a launch.
2. **Run the experiments.** Each run changes one field of the baseline (`configs/experiments.yaml`).
3. **Confirm.** `make combine` runs one extra config that combines each experiment's winner, since one-at-a-time winners do not always add up.
4. **Decide.** The report recommends the passing config with the highest faithfulness, or says no config should ship and which gates fail.

## Quick start

```bash
git clone https://github.com/deekshithlanka/rag-eval-framework && cd rag-eval-framework
python -m venv .venv && source .venv/bin/activate
make install
cp .env.example .env            # add your GOOGLE_API_KEY from aistudio.google.com/apikey

make test                       # offline unit tests
make smoke                      # offline end-to-end run, no API key needed
make all                        # calibrate, run 7 configs, confirmation run, report
```

A full run is about 400 generator calls and 800 judge calls. Responses are cached in `.cache/`, so reruns are free. If you hit rate limits, lower `requests_per_minute` in the config or test with `python -m rageval.run --limit 10`.

## Repo map

```
configs/experiments.yaml      baseline, experiments, launch gates
data/corpus/                  59 help center articles (synthetic, Markdown)
data/golden/golden_set.jsonl  49 questions, source IDs, reference answers, key facts
data/calibration/             15 answers with human faithfulness and relevance labels
src/rageval/
  corpus.py      load docs, paragraph-aware chunking
  retrieval.py   FAISS (Gemini or BGE embeddings) and a TF-IDF baseline
  prompts.py     3 generator prompts: basic, grounded, grounded + clarify
  pipeline.py    retrieve then generate
  judge.py       rubrics, LLM judge, offline heuristic judge
  metrics.py     scoring, aggregation, gates
  stats.py       bootstrap CIs, Cohen's kappa
  calibrate.py   judge vs human labels
  run.py         experiment runner
  report.py      writes reports/eval_report.md
tests/           offline tests
```

## Design choices

- **Gates before results.** Thresholds live in the config and were set first, so the decision cannot be tuned to the data.
- **Judge model differs from the generator** (`gemini-3.1-flash-lite` judges `gemini-2.5-flash-lite`) to reduce self-preference.
- **Hard cases on purpose.** The corpus has an outdated price page that conflicts with the current one, plan-dependent limits, and look-alike numbers (120 vs 600 API requests per minute) to test whether the bot grounds answers or pattern-matches.
- **Unanswerable does not mean refuse.** "Enterprise pricing is custom" is a safe answer; "$199 a month" is not. The safe rate counts either a refusal or a fully supported answer.
- **Lexical baseline.** TF-IDF is included in the embedding experiment so gains from neural embeddings are measured against something cheap.

## Limitations

Synthetic corpus, one labeler, and about 50 questions. Treat deltas under about 10 pp as directional. The calibration labels in `data/calibration` were drafted to test known failure modes; review them before relying on the kappa.
