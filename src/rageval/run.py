"""Run the experiment plan.

    python -m rageval.run                      # all runs with Gemini
    python -m rageval.run --only baseline      # one run
    python -m rageval.run --mock               # offline smoke run, no API key
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from .config import ROOT, ExperimentPlan, RunConfig, load_plan
from .corpus import chunk_docs, load_corpus
from .judge import HeuristicJudge, LLMJudge
from .metrics import behavior_correct, check_gates, key_fact_recall, retrieval_scores, summarize
from .pipeline import ExtractiveMockGenerator, RAGPipeline
from .retrieval import build_retriever


def load_golden(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


class Factory:
    """Builds and reuses chunks, indexes and model clients across runs."""

    def __init__(self, plan: ExperimentPlan, mock: bool):
        self.plan = plan
        self.mock = mock
        self.docs = load_corpus(ROOT / "data/corpus")
        self._chunks: dict = {}
        self._retrievers: dict = {}
        self._chats: dict = {}

    def chunks(self, cfg: RunConfig):
        key = (cfg.chunk_size, cfg.chunk_overlap)
        if key not in self._chunks:
            self._chunks[key] = chunk_docs(self.docs, cfg.chunk_size, cfg.chunk_overlap)
        return self._chunks[key]

    def retriever(self, cfg: RunConfig):
        embedding = "tfidf" if self.mock else cfg.embedding
        key = (embedding, cfg.chunk_size, cfg.chunk_overlap)
        if key not in self._retrievers:
            self._retrievers[key] = build_retriever(self.chunks(cfg), embedding, cfg.index_key())
        return self._retrievers[key]

    def chat(self, model: str, temperature):
        from .llm import GeminiChat

        key = (model, temperature)
        if key not in self._chats:
            self._chats[key] = GeminiChat(model, temperature, rpm=self.plan.rpm)
        return self._chats[key]

    def generator(self, cfg: RunConfig):
        return ExtractiveMockGenerator() if self.mock else self.chat(cfg.generator_model, cfg.temperature)

    def judge(self, cfg: RunConfig):
        return HeuristicJudge() if self.mock else LLMJudge(self.chat(cfg.judge_model, None))


def combined_config(plan: ExperimentPlan, summary: pd.DataFrame) -> RunConfig:
    """Take the faithfulness winner of each experiment and apply all of them to the baseline."""
    from dataclasses import replace

    s = summary.set_index("run")
    changes = {}
    for exp_runs in plan.experiments.values():
        present = [r for r in exp_runs if r.name in s.index]
        if not present:
            continue
        best = max(present, key=lambda r: (s.loc[r.name, "faithfulness_pass_rate"], s.loc[r.name, "relevance_pass_rate"]))
        if best.name != "baseline":
            field, _, _ = best.name.partition("=")
            changes[field] = getattr(best, field)
    return replace(plan.baseline, name="combined", **changes)


def evaluate_one(q: dict, pipeline: RAGPipeline, judge) -> dict:
    out = pipeline.answer(q["question"])
    faith = judge.faithfulness(q["question"], out["context"], out["answer"])
    rel = judge.relevance(q["question"], out["answer"])
    record = {
        "id": q["id"],
        "type": q["type"],
        "question": q["question"],
        "expected_sources": q["expected_sources"],
        **out,
        **retrieval_scores(q["expected_sources"] if q["type"] != "unanswerable" else [], out["retrieved_docs"]),
        "faithfulness": faith.score,
        "faithfulness_reasoning": faith.reasoning,
        "unsupported_claims": faith.unsupported,
        "relevance": rel.score,
        "relevance_reasoning": rel.reasoning,
        "behavior": rel.behavior,
        "key_fact_recall": key_fact_recall(q["key_facts"], out["answer"]),
        "judge_error": faith.error or rel.error,
    }
    record["behavior_correct"] = behavior_correct(q["type"], rel.behavior, faith.score)
    return record


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/experiments.yaml"))
    ap.add_argument("--golden", default=str(ROOT / "data/golden/golden_set.jsonl"))
    ap.add_argument("--out", default=str(ROOT / "results"))
    ap.add_argument("--only", help="comma-separated run names")
    ap.add_argument("--limit", type=int, help="first N questions only (for a cheap test)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--mock", action="store_true", help="offline: TF-IDF retrieval, extractive generator, heuristic judge")
    ap.add_argument("--combine", action="store_true",
                    help="after the experiments: run one confirmation config that combines each experiment's winner")
    args = ap.parse_args(argv)

    plan = load_plan(args.config)
    golden = load_golden(Path(args.golden))[: args.limit]
    out_dir = Path(args.out)
    previous = pd.read_csv(out_dir / "summary.csv") if (args.combine and (out_dir / "summary.csv").exists()) else None

    if args.combine:
        if previous is None:
            raise SystemExit("Run the experiments first; --combine reads results/summary.csv")
        runs = [combined_config(plan, previous)]
    else:
        runs = plan.all_runs()
        if args.only:
            wanted = set(args.only.split(","))
            runs = [r for r in runs if r.name in wanted]

    (out_dir / "runs").mkdir(parents=True, exist_ok=True)
    factory = Factory(plan, args.mock)
    rows = []
    for cfg in runs:
        print(f"> {cfg.name}: {len(golden)} questions")
        pipeline = RAGPipeline(cfg, factory.retriever(cfg), factory.generator(cfg))
        judge = factory.judge(cfg)
        with ThreadPoolExecutor(max_workers=1 if args.mock else args.workers) as pool:
            records = list(pool.map(lambda q: evaluate_one(q, pipeline, judge), golden))
        (out_dir / "runs" / f"{cfg.name}.jsonl").write_text(
            "\n".join(json.dumps(r) for r in records) + "\n"
        )
        summary = summarize(records)
        gates = check_gates(summary, plan.gates)
        rows.append({"run": cfg.name, **summary, "passes_all_gates": all(gates.values())})
        print(
            f"  faithfulness pass {summary['faithfulness_pass_rate']:.0%} | relevance pass "
            f"{summary['relevance_pass_rate']:.0%} | recall@k {summary['retrieval_recall_at_k']:.0%} | "
            f"unanswerable safe {summary['unanswerable_safe_rate']:.0%}"
        )

    new = pd.DataFrame(rows)
    if previous is not None:
        new = pd.concat([previous[~previous["run"].isin(new["run"])], new], ignore_index=True)
        old_meta = json.loads((out_dir / "plan.json").read_text())
        old_meta["runs"].update({r.name: r.to_dict() for r in runs})
        (out_dir / "plan.json").write_text(json.dumps(old_meta, indent=2))
        new.to_csv(out_dir / "summary.csv", index=False)
        print(f"Added {[r.name for r in runs]} to {out_dir / 'summary.csv'}")
        return
    new.to_csv(out_dir / "summary.csv", index=False)
    meta = {
        "mock": args.mock,
        "n_questions": len(golden),
        "baseline": asdict(plan.baseline),
        "experiments": {k: [r.name for r in v] for k, v in plan.experiments.items()},
        "gates": asdict(plan.gates),
        "prices_per_million": plan.prices_per_million,
        "runs": {r.name: r.to_dict() for r in runs},
    }
    (out_dir / "plan.json").write_text(json.dumps(meta, indent=2))
    print(f"Wrote {out_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()
