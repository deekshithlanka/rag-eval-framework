"""Per-question scoring and run-level aggregation."""

from __future__ import annotations

import numpy as np

from .config import ShipGates

ANSWERABLE = {"single_hop", "multi_hop"}
RETRIEVAL_TYPES = {"single_hop", "multi_hop", "ambiguous"}


def retrieval_scores(expected: list[str], retrieved_docs: list[str]) -> dict:
    """recall@k, hit@k and reciprocal rank at the document level."""
    if not expected:
        return {"recall_at_k": None, "hit_at_k": None, "rr": None}
    found = set(expected) & set(retrieved_docs)
    rr = 0.0
    for rank, doc in enumerate(retrieved_docs, start=1):
        if doc in expected:
            rr = 1.0 / rank
            break
    return {
        "recall_at_k": len(found) / len(expected),
        "hit_at_k": float(bool(found)),
        "rr": rr,
    }


def key_fact_recall(key_facts: list[str], answer: str) -> float | None:
    if not key_facts:
        return None
    text = answer.lower()
    return sum(f.lower() in text for f in key_facts) / len(key_facts)


def behavior_correct(qtype: str, behavior: str, faithfulness: int) -> bool:
    if qtype == "unanswerable":
        # Safe if it declines, or if everything it says is supported (for example
        # "Enterprise pricing is custom" without inventing a number).
        return behavior == "abstained" or faithfulness >= 4
    return behavior != "abstained"


def _mean(xs) -> float:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


def summarize(records: list[dict]) -> dict:
    by_type = lambda types: [r for r in records if r["type"] in types]  # noqa: E731
    retr = by_type(RETRIEVAL_TYPES)
    answerable_like = by_type(ANSWERABLE | {"ambiguous"})
    unans = by_type({"unanswerable"})
    lat = [r["latency_s"] for r in records if r.get("latency_s") is not None]
    return {
        "n": len(records),
        "retrieval_recall_at_k": _mean(r["recall_at_k"] for r in retr),
        "retrieval_mrr": _mean(r["rr"] for r in retr),
        "faithfulness_mean": _mean(r["faithfulness"] for r in records),
        "faithfulness_pass_rate": _mean(float(r["faithfulness"] >= 4) for r in records),
        "relevance_mean": _mean(r["relevance"] for r in answerable_like),
        "relevance_pass_rate": _mean(float(r["relevance"] >= 4) for r in answerable_like),
        "unanswerable_safe_rate": _mean(float(r["behavior_correct"]) for r in unans),
        "abstention_rate_unanswerable": _mean(float(r["behavior"] == "abstained") for r in unans),
        "false_abstention_rate": _mean(float(r["behavior"] == "abstained") for r in answerable_like),
        "key_fact_recall": _mean(r["key_fact_recall"] for r in records),
        "judge_errors": sum(1 for r in records if r.get("judge_error")),
        "latency_p50_s": float(np.percentile(lat, 50)) if lat else float("nan"),
        "latency_p95_s": float(np.percentile(lat, 95)) if lat else float("nan"),
        "gen_input_tokens": int(sum(r.get("input_tokens", 0) for r in records)),
        "gen_output_tokens": int(sum(r.get("output_tokens", 0) for r in records)),
    }


def summarize_by_type(records: list[dict]) -> dict[str, dict]:
    out = {}
    for qtype in ["single_hop", "multi_hop", "ambiguous", "unanswerable"]:
        rs = [r for r in records if r["type"] == qtype]
        if not rs:
            continue
        out[qtype] = {
            "n": len(rs),
            "faithfulness_pass_rate": _mean(float(r["faithfulness"] >= 4) for r in rs),
            "relevance_pass_rate": _mean(float(r["relevance"] >= 4) for r in rs),
            "behavior_correct_rate": _mean(float(r["behavior_correct"]) for r in rs),
            "recall_at_k": _mean(r["recall_at_k"] for r in rs),
        }
    return out


GATE_DIRECTIONS = {
    "faithfulness_pass_rate": ">=",
    "relevance_pass_rate": ">=",
    "unanswerable_safe_rate": ">=",
    "false_abstention_rate": "<=",
    "retrieval_recall_at_k": ">=",
}


def check_gates(summary: dict, gates: ShipGates) -> dict[str, bool]:
    result = {}
    for metric, direction in GATE_DIRECTIONS.items():
        threshold = getattr(gates, metric)
        value = summary[metric]
        result[metric] = bool(value >= threshold) if direction == ">=" else bool(value <= threshold)
    return result
