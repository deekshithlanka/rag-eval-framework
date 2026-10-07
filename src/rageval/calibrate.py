"""Check the LLM judge against human labels before trusting it.

    python -m rageval.calibrate          # Gemini judge
    python -m rageval.calibrate --mock   # heuristic judge, offline

Writes reports/calibration.md and reports/calibration.json. The judge is
trusted for the experiments only if it clears the agreement bar below.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import ROOT, load_plan
from .corpus import load_corpus
from .judge import PASS_THRESHOLD, RUBRIC_VERSION, HeuristicJudge, LLMJudge
from .stats import agreement, cohen_kappa

# Pre-registered bar: pass/fail kappa >= 0.6 ("substantial") on both rubrics.
KAPPA_BAR = 0.6


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(ROOT / "configs/experiments.yaml"))
    ap.add_argument("--labels", default=str(ROOT / "data/calibration/calibration_set.jsonl"))
    ap.add_argument("--mock", action="store_true")
    args = ap.parse_args(argv)

    plan = load_plan(args.config)
    docs = {d.id: d for d in load_corpus(ROOT / "data/corpus")}
    items = [json.loads(line) for line in Path(args.labels).read_text().splitlines() if line.strip()]

    if args.mock:
        judge = HeuristicJudge()
    else:
        from .llm import GeminiChat

        judge = LLMJudge(GeminiChat(plan.baseline.judge_model, None, rpm=plan.rpm))

    rows = []
    for it in items:
        context = "\n\n---\n\n".join(f"[{d}] {docs[d].title}\n{docs[d].text}" for d in it["context_doc_ids"])
        f = judge.faithfulness(it["question"], context, it["answer"])
        r = judge.relevance(it["question"], it["answer"], context)
        rows.append({
            "id": it["id"],
            "human_f": it["human"]["faithfulness"], "judge_f": f.score,
            "human_r": it["human"]["relevance"], "judge_r": r.score,
            "human_b": it["human"]["behavior"], "judge_b": r.behavior,
            "judge_f_reason": f.reasoning or f.error, "judge_r_reason": r.reasoning or r.error,
            "note": it.get("label_note", ""),
        })

    def stats(h, j):
        hp = [x >= PASS_THRESHOLD for x in h]
        jp = [x >= PASS_THRESHOLD for x in j]
        return {
            "exact_agreement": agreement(h, j, 0),
            "within_1_agreement": agreement(h, j, 1),
            "pass_fail_agreement": sum(a == b for a, b in zip(hp, jp)) / len(hp),
            "pass_fail_kappa": cohen_kappa(hp, jp, labels=[False, True]),
            "weighted_kappa": cohen_kappa(h, j, weights="quadratic", labels=[1, 2, 3, 4, 5]),
        }

    result = {
        "mock": args.mock,
        "rubric_version": RUBRIC_VERSION,
        "judge_model": "heuristic" if args.mock else plan.baseline.judge_model,
        "n": len(rows),
        "labelers": sorted({it.get("labeler", "unknown") for it in items}),
        "faithfulness": stats([r["human_f"] for r in rows], [r["judge_f"] for r in rows]),
        "relevance": stats([r["human_r"] for r in rows], [r["judge_r"] for r in rows]),
        "behavior_agreement": sum(r["human_b"] == r["judge_b"] for r in rows) / len(rows),
        "kappa_bar": KAPPA_BAR,
        "rows": rows,
    }
    result["judge_trusted"] = (
        result["faithfulness"]["pass_fail_kappa"] >= KAPPA_BAR
        and result["relevance"]["pass_fail_kappa"] >= KAPPA_BAR
    )

    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "calibration.json").write_text(json.dumps(result, indent=2))
    (out / "calibration.md").write_text(render_markdown(result))
    f, r = result["faithfulness"], result["relevance"]
    print(f"Faithfulness: pass/fail kappa {f['pass_fail_kappa']:.2f}, exact {f['exact_agreement']:.0%}")
    print(f"Relevance:    pass/fail kappa {r['pass_fail_kappa']:.2f}, exact {r['exact_agreement']:.0%}")
    print("Judge trusted" if result["judge_trusted"] else "Judge NOT trusted: revise the rubric and rerun")


def render_markdown(res: dict) -> str:
    lines = [
        "# Judge calibration",
        "",
        f"Judge: `{res['judge_model']}`, rubric `{res['rubric_version']}`, {res['n']} hand-labeled examples.",
        f"Bar: pass/fail Cohen's kappa of {res['kappa_bar']} or higher on both rubrics.",
        f"Result: **{'trusted' if res['judge_trusted'] else 'not trusted'}**.",
        "",
    ]
    if res["mock"]:
        lines += ["> MOCK RUN: heuristic judge, not the LLM judge. Numbers only show the harness works.", ""]
    if "draft" in res["labelers"]:
        lines += ["> Labels are still marked `draft`. Review them before relying on these numbers.", ""]
    lines += [
        "| Rubric | Exact agreement | Within 1 point | Pass/fail agreement | Pass/fail kappa | Weighted kappa |",
        "|---|---|---|---|---|---|",
    ]
    for name in ["faithfulness", "relevance"]:
        s = res[name]
        lines.append(
            f"| {name.title()} | {s['exact_agreement']:.0%} | {s['within_1_agreement']:.0%} | "
            f"{s['pass_fail_agreement']:.0%} | {s['pass_fail_kappa']:.2f} | {s['weighted_kappa']:.2f} |"
        )
    lines += ["", f"Behavior label agreement (answered, abstained, clarified): {res['behavior_agreement']:.0%}", ""]
    lines += ["## Disagreements (pass/fail flips)", "", "| ID | Rubric | Human | Judge | Label note | Judge reasoning |", "|---|---|---|---|---|---|"]
    flips = 0
    for row in res["rows"]:
        for rub, h, j, why in [("F", row["human_f"], row["judge_f"], row["judge_f_reason"]),
                               ("R", row["human_r"], row["judge_r"], row["judge_r_reason"])]:
            if (h >= PASS_THRESHOLD) != (j >= PASS_THRESHOLD):
                flips += 1
                lines.append(f"| {row['id']} | {rub} | {h} | {j} | {row['note']} | {(why or '').replace('|', '/')} |")
    if not flips:
        lines.append("| none | | | | | |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
