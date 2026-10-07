"""Build reports/eval_report.md from results/ and reports/calibration.json.

    python -m rageval.report
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd

from .config import ROOT, ShipGates
from .metrics import GATE_DIRECTIONS, check_gates, summarize_by_type
from .stats import paired_bootstrap_diff

METRIC_LABELS = {
    "retrieval_recall_at_k": "Recall@k",
    "faithfulness_pass_rate": "Faithful",
    "relevance_pass_rate": "Relevant",
    "unanswerable_safe_rate": "Unanswerable safe",
    "false_abstention_rate": "False abstain",
}


def pct(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.0%}"


def pp(x) -> str:
    return f"{x * 100:+.0f} pp"


def changed_fields(meta: dict) -> str:
    base, comb = meta["baseline"], meta["runs"]["combined"]
    diffs = [f"{k}={v}" for k, v in comb.items() if k != "name" and base.get(k) != v]
    return ", ".join(diffs) or "no changes (baseline won every experiment)"


def load_runs(results: Path) -> dict[str, list[dict]]:
    runs = {}
    for path in sorted((results / "runs").glob("*.jsonl")):
        runs[path.stem] = [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
    return runs


def paired(runs, base: str, other: str, field: str):
    a = {r["id"]: float(r[field] >= 4) for r in runs[base]}
    b = {r["id"]: float(r[field] >= 4) for r in runs[other]}
    ids = sorted(set(a) & set(b))
    return paired_bootstrap_diff([a[i] for i in ids], [b[i] for i in ids])


def cost(row, prices: dict, model: str) -> float | None:
    p = prices.get(model)
    if not p:
        return None
    return row["gen_input_tokens"] / 1e6 * p["input"] + row["gen_output_tokens"] / 1e6 * p["output"]


def choose(summary: pd.DataFrame, judge_trusted: bool) -> tuple[str | None, str]:
    passing = summary[summary["passes_all_gates"]]
    if not judge_trusted:
        return None, "the judge did not clear the calibration bar, so its scores cannot support a launch decision"
    if passing.empty:
        return None, "no configuration cleared every pre-registered gate"
    best = passing.sort_values(
        ["faithfulness_pass_rate", "relevance_pass_rate", "gen_input_tokens"], ascending=[False, False, True]
    ).iloc[0]
    return str(best["run"]), "it clears every gate with the highest faithfulness pass rate among passing runs"


def build(results: Path = ROOT / "results", reports: Path = ROOT / "reports") -> str:
    summary = pd.read_csv(results / "summary.csv")
    meta = json.loads((results / "plan.json").read_text())
    runs = load_runs(results)
    gates = ShipGates(**meta["gates"])
    calib_path = reports / "calibration.json"
    calib = json.loads(calib_path.read_text()) if calib_path.exists() else None
    judge_trusted = bool(calib and calib["judge_trusted"])
    by_name = summary.set_index("run")
    base = "baseline"
    pick, why = choose(summary, judge_trusted)
    n = meta["n_questions"]

    L: list[str] = ["# Eval report: Ledgerly support assistant (RAG)", ""]
    if meta["mock"]:
        L += ["> **MOCK RUN.** TF-IDF retrieval, an extractive stand-in generator and a heuristic judge. "
              "These numbers only prove the harness runs. Run `make all` with a Gemini key for real results.", ""]

    # 1. Decision
    L += ["## Recommendation", ""]
    if pick:
        b, w = by_name.loc[base], by_name.loc[pick]
        L += [f"**Ship `{pick}`.** Chosen because {why}.", ""]
        if pick != base:
            d, lo, hi = paired(runs, base, pick, "faithfulness")
            L += [f"Versus baseline: faithfulness pass rate {pct(b.faithfulness_pass_rate)} to "
                  f"{pct(w.faithfulness_pass_rate)} ({pp(d)}, 95% CI {pp(lo)} to {pp(hi)}, paired bootstrap, n={n}).", ""]
    else:
        L += [f"**Do not ship yet.** Reason: {why}.", ""]
        best = summary.sort_values("faithfulness_pass_rate", ascending=False).iloc[0]
        failing = [METRIC_LABELS[k] for k, ok in check_gates(best.to_dict(), gates).items() if not ok]
        L += [f"Closest run: `{best['run']}`. Gates still failing: {', '.join(failing) or 'none'}.", ""]

    # 2. Gate table
    show = [base] + ([pick] if pick and pick != base else [])
    L += ["### Launch gates (set before any results were seen)", "",
          "| Metric | Gate | " + " | ".join(f"`{s}`" for s in show) + " |",
          "|---|---|" + "---|" * len(show)]
    for metric, direction in GATE_DIRECTIONS.items():
        cells = []
        for s in show:
            v = by_name.loc[s, metric]
            ok = v >= getattr(gates, metric) if direction == ">=" else v <= getattr(gates, metric)
            cells.append(f"{pct(v)} {'pass' if ok else 'FAIL'}")
        L.append(f"| {METRIC_LABELS[metric]} | {direction} {getattr(gates, metric):.0%} | " + " | ".join(cells) + " |")
    L.append("")

    # 3. Judge
    L += ["## Judge calibration", ""]
    if calib:
        f, r = calib["faithfulness"], calib["relevance"]
        L += [f"`{calib['judge_model']}` vs {calib['n']} hand labels. Pass/fail kappa: faithfulness "
              f"{f['pass_fail_kappa']:.2f}, relevance {r['pass_fail_kappa']:.2f} (bar {calib['kappa_bar']}). "
              f"Judge {'trusted' if judge_trusted else 'NOT trusted'}. Details in [calibration.md](calibration.md).", ""]
    else:
        L += ["Not run. Run `make calibrate` first; without it no recommendation is made.", ""]

    # 4. Experiments
    L += ["## Experiments (one variable changed at a time)", "",
          f"Baseline: chunk size {meta['baseline']['chunk_size']}, overlap {meta['baseline']['chunk_overlap']}, "
          f"`{meta['baseline']['embedding']}`, prompt `{meta['baseline']['prompt']}`, top-k {meta['baseline']['top_k']}, "
          f"generator `{meta['baseline']['generator_model']}`, judge `{meta['baseline']['judge_model']}`. "
          f"{n} golden questions per run. Deltas are faithfulness pass rate vs baseline with 95% paired bootstrap CIs.", ""]
    prices = meta.get("prices_per_million") or {}
    for exp_name, run_names in meta["experiments"].items():
        present = [r for r in run_names if r in by_name.index]
        if not present:
            continue
        L += [f"### {exp_name}", "",
              "| Run | Recall@k | MRR | Faithful | Relevant | Unanswerable safe | False abstain | p50 latency | Gen tokens | Faithful vs baseline |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for rn in present:
            row = by_name.loc[rn]
            if rn == base:
                delta = "baseline"
            else:
                d, lo, hi = paired(runs, base, rn, "faithfulness")
                delta = f"{pp(d)} [{pp(lo)}, {pp(hi)}]"
            tokens = int(row.gen_input_tokens + row.gen_output_tokens)
            c = cost(row, prices, meta["runs"][rn]["generator_model"])
            tok = f"{tokens:,}" + (f" (${c:.3f})" if c is not None else "")
            L.append(f"| `{rn}` | {pct(row.retrieval_recall_at_k)} | {row.retrieval_mrr:.2f} | {pct(row.faithfulness_pass_rate)} | "
                     f"{pct(row.relevance_pass_rate)} | {pct(row.unanswerable_safe_rate)} | {pct(row.false_abstention_rate)} | "
                     f"{row.latency_p50_s:.2f}s | {tok} | {delta} |")
        winner = max(present, key=lambda r: (by_name.loc[r, "faithfulness_pass_rate"], by_name.loc[r, "relevance_pass_rate"]))
        L += ["", f"Best on faithfulness (ties broken by relevance): `{winner}`.", ""]

    if "combined" in by_name.index:
        row = by_name.loc["combined"]
        d, lo, hi = paired(runs, base, "combined", "faithfulness")
        L += ["### Confirmation run: winners combined", "",
              f"`combined` = baseline with {changed_fields(meta)}. Faithful {pct(row.faithfulness_pass_rate)}, relevant "
              f"{pct(row.relevance_pass_rate)}, unanswerable safe {pct(row.unanswerable_safe_rate)} "
              f"({pp(d)} faithfulness vs baseline, CI {pp(lo)} to {pp(hi)}).", ""]

    # 5. Breakdown by question type
    focus = pick or base
    L += [f"## By question type (`{focus}`)", "",
          "| Type | n | Recall@k | Faithful | Relevant | Correct behavior |", "|---|---|---|---|---|---|"]
    for qtype, s in summarize_by_type(runs[focus]).items():
        L.append(f"| {qtype} | {s['n']} | {pct(s['recall_at_k'])} | {pct(s['faithfulness_pass_rate'])} | "
                 f"{pct(s['relevance_pass_rate'])} | {pct(s['behavior_correct_rate'])} |")
    L += ["", "Correct behavior: answers for answerable and ambiguous questions; no invented facts for unanswerable ones.", ""]

    # 6. Failures
    worst = sorted(runs[focus], key=lambda r: (r["faithfulness"], r["relevance"]))[:5]
    L += [f"## Lowest-scoring answers (`{focus}`)", "", "| ID | Type | Faithful | Relevant | Unsupported claims or judge note |", "|---|---|---|---|---|"]
    for r in worst:
        note = "; ".join(r.get("unsupported_claims") or []) or r.get("faithfulness_reasoning") or ""
        L.append(f"| {r['id']} | {r['type']} | {r['faithfulness']} | {r['relevance']} | {note.replace('|', '/')[:160]} |")
    L.append("")

    # 7. Method and limits
    L += ["## Method", "",
          "- Corpus: 59 synthetic help center articles for a fictional invoicing product, Ledgerly (`data/corpus`).",
          f"- Golden set: {n} questions with source doc IDs: single-hop, multi-hop, ambiguous (answer depends on plan or method) and unanswerable (`data/golden`).",
          "- Retrieval: recall@k and MRR at the document level, scored against the golden source IDs.",
          "- Faithfulness and relevance: LLM judge on a 1 to 5 rubric, pass is 4 or higher (`src/rageval/judge.py`).",
          "- Uncertainty: 95% paired bootstrap CIs over questions. With about 50 questions, differences under 10 pp are often not distinguishable from noise.",
          "", "## Limitations", "",
          "- Synthetic corpus and a single labeler. Real tickets would be messier.",
          "- About 50 questions limits statistical power; treat small deltas as directional.",
          "- The judge model is from the same provider as the generator, which can bias scores. Calibration checks this only on 15 items.",
          "- Latency includes API network time and varies by run.", ""]
    text = "\n".join(L)
    (reports / "eval_report.md").write_text(text)
    return text


def main() -> None:
    build()
    print("Wrote reports/eval_report.md")


if __name__ == "__main__":
    main()
