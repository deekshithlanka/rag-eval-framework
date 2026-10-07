"""Offline tests. No API key or network needed."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rageval.config import load_plan  # noqa: E402
from rageval.corpus import chunk_docs, chunk_text, load_corpus  # noqa: E402
from rageval.judge import HeuristicJudge, detect_behavior, parse_json  # noqa: E402
from rageval.metrics import behavior_correct, key_fact_recall, retrieval_scores  # noqa: E402
from rageval.retrieval import TfidfRetriever  # noqa: E402
from rageval.stats import bootstrap_ci, cohen_kappa, paired_bootstrap_diff  # noqa: E402

DOCS = load_corpus(ROOT / "data/corpus")
GOLDEN = [json.loads(l) for l in (ROOT / "data/golden/golden_set.jsonl").read_text().splitlines()]
CALIB = [json.loads(l) for l in (ROOT / "data/calibration/calibration_set.jsonl").read_text().splitlines()]


def test_corpus_size_and_ids_unique():
    assert 50 <= len(DOCS) <= 100
    assert len({d.id for d in DOCS}) == len(DOCS)


def test_golden_set_shape():
    assert len(GOLDEN) >= 40
    types = {q["type"] for q in GOLDEN}
    assert types == {"single_hop", "multi_hop", "ambiguous", "unanswerable"}
    ids = {d.id for d in DOCS}
    for q in GOLDEN:
        assert set(q["expected_sources"]) <= ids, q["id"]
        if q["type"] != "unanswerable":
            assert q["expected_sources"], q["id"]


def test_calibration_set_shape():
    assert len(CALIB) == 15
    ids = {d.id for d in DOCS}
    for c in CALIB:
        assert set(c["context_doc_ids"]) <= ids
        assert 1 <= c["human"]["faithfulness"] <= 5
        assert c["human"]["behavior"] in {"answered", "abstained", "clarified"}


def test_chunks_respect_size_and_cover_text():
    for size in (400, 800, 1600):
        chunks = chunk_docs(DOCS, size, 100)
        assert all(len(c.text) <= size for c in chunks)
        for doc in DOCS[:10]:
            joined = " ".join(c.text for c in chunks if c.doc_id == doc.id)
            for para in doc.text.split("\n\n"):
                assert para.split(". ")[0][:40] in joined


def test_smaller_chunks_make_more_chunks():
    n = [len(chunk_docs(DOCS, s, 100)) for s in (400, 800, 1600)]
    assert n[0] > n[1] >= n[2]


def test_chunk_text_rejects_bad_overlap():
    try:
        chunk_text("abc", 100, 100)
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_tfidf_retrieves_obvious_doc():
    r = TfidfRetriever(chunk_docs(DOCS, 800, 100))
    top = [c.doc_id for c, _ in r.search("API rate limit requests per minute", 3)]
    assert "INT-07" in top


def test_retrieval_scores():
    s = retrieval_scores(["A", "B"], ["C", "B", "A"])
    assert s["recall_at_k"] == 1.0 and s["rr"] == 0.5
    assert retrieval_scores([], ["A"])["recall_at_k"] is None


def test_key_fact_recall_and_behavior():
    assert key_fact_recall(["$24", "$288"], "It is $24 a month") == 0.5
    assert detect_behavior("I couldn't find that in the Ledgerly help center.") == "abstained"
    assert behavior_correct("unanswerable", "answered", 5)
    assert not behavior_correct("unanswerable", "answered", 2)
    assert not behavior_correct("single_hop", "abstained", 5)


def test_parse_json_handles_fences():
    assert parse_json('```json\n{"score": 4}\n```')["score"] == 4


def test_heuristic_judge_flags_contradiction():
    j = HeuristicJudge()
    ctx = "Ledgerly does not support SMS codes for 2FA."
    good = j.faithfulness("q", ctx, "Ledgerly does not support SMS codes for 2FA.")
    bad = j.faithfulness("q", ctx, "Yes, pick text message delivery during onboarding setup.")
    assert good.score > bad.score


def test_kappa_and_bootstrap():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert abs(cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-9
    m, lo, hi = bootstrap_ci([1, 1, 1, 0])
    assert lo <= m <= hi
    d, _, _ = paired_bootstrap_diff([0, 0, 1], [1, 1, 1])
    assert d > 0


def test_plan_varies_one_field_per_run():
    plan = load_plan(ROOT / "configs/experiments.yaml")
    base = plan.baseline.to_dict()
    for runs in plan.experiments.values():
        for run in runs:
            diff = {k for k, v in run.to_dict().items() if k != "name" and v != base[k]}
            assert len(diff) <= 1, (run.name, diff)
    assert len(plan.experiments) >= 3
