"""LLM-as-judge rubrics for faithfulness and relevance.

Each rubric returns a 1 to 5 score. A score of 4 or 5 counts as a pass.
Rubric text is versioned so calibration changes are traceable in git.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from .llm import parse_json

RUBRIC_VERSION = "v2"
# v2 changes, from v1 calibration disagreements (see reports/rubric_changelog.md):
#   faithfulness: incomplete or off-topic answers are not penalized (C08);
#                 recompute arithmetic explicitly (C10); abstentions always score 5 (C13)
#   relevance:    the judge now sees the context, so a false "I couldn't find that"
#                 scores 1 (C13)
PASS_THRESHOLD = 4

FAITHFULNESS_SYSTEM = """You are a strict evaluator of a customer support answer.
Judge FAITHFULNESS: is every factual claim in the ANSWER supported by the CONTEXT?

Steps:
1. List each factual claim in the answer (numbers, prices, limits, features, steps, yes/no statements).
2. For each claim, decide: SUPPORTED (stated in or directly computable from the context) or UNSUPPORTED (missing from or contradicted by the context).
3. Score using the scale below. Judge only support, not helpfulness or style.
   Do NOT lower the score because the answer is incomplete, off-topic or skips part of the question.
   That is relevance, which is scored separately.

Scale:
5 = every claim is supported, or the answer only says the information is not available.
4 = all key claims supported; one minor unsupported detail that would not mislead the customer.
3 = a mix: some claims supported, at least one unsupported claim that could mislead.
2 = the main claim is supported but an important added claim is unsupported or contradicted.
1 = the main claim is unsupported or contradicts the context.

Arithmetic: for every number the answer derives (totals, sums, percentages), recompute it yourself from
the context values and show the calculation in the claim. If your result differs from the answer's number,
that claim is UNSUPPORTED, even if every input number is correct.
Abstentions: if the answer only says it could not find the information (with or without suggesting
support), score 5. Wrongly declining is measured by a separate metric, not by faithfulness.
Dated facts: a fact the context marks as old or replaced is unsupported if the answer presents it as current.

Return only JSON: {"claims": [{"claim": "...", "supported": true}], "score": 1-5, "reasoning": "one sentence"}"""

FAITHFULNESS_USER = """CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}"""

RELEVANCE_SYSTEM = """You are evaluating a customer support answer.
Judge RELEVANCE: does the ANSWER address what the customer actually asked?
Do not judge factual accuracy. An answer with wrong facts that directly addresses the question is still relevant.
Use the CONTEXT only to check whether the answer was available.

Scale:
5 = directly and completely addresses every part of the question.
4 = addresses the main point; a minor part is thin or there is some unneeded detail.
3 = partially addresses the question; a requested part is missing.
2 = mostly off target, or declines to answer a clear, ordinary support question.
1 = does not address the question at all.

Special cases:
- If the question depends on unstated details (for example the customer's plan) and the answer covers the cases or asks a clarifying question, that is fully relevant.
- If the answer says the information is not available and the CONTEXT does not contain it, score 4.
- If the answer says the information is not available but the CONTEXT does contain it, score 1.

Also classify the answer's behavior:
"answered" = gives an answer; "abstained" = says it cannot find or does not know; "clarified" = mainly asks a clarifying question.

Return only JSON: {"score": 1-5, "behavior": "answered|abstained|clarified", "reasoning": "one sentence"}"""

RELEVANCE_USER = """CONTEXT:
{context}

QUESTION: {question}

ANSWER: {answer}"""


@dataclass
class Verdict:
    score: int
    reasoning: str = ""
    behavior: str | None = None
    unsupported: list[str] = field(default_factory=list)
    error: str | None = None

    @property
    def passed(self) -> bool:
        return self.score >= PASS_THRESHOLD


def _clamp(v) -> int:
    return max(1, min(5, int(round(float(v)))))


class LLMJudge:
    def __init__(self, chat):
        self.chat = chat
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def _call(self, system: str, user: str) -> dict:
        res = self.chat.complete(system, user)
        if not res.cached:
            self.usage["input_tokens"] += res.input_tokens
            self.usage["output_tokens"] += res.output_tokens
        return parse_json(res.text)

    def faithfulness(self, question: str, context: str, answer: str) -> Verdict:
        try:
            data = self._call(FAITHFULNESS_SYSTEM, FAITHFULNESS_USER.format(
                context=context, question=question, answer=answer))
            unsupported = [c.get("claim", "") for c in data.get("claims", []) if not c.get("supported", True)]
            return Verdict(_clamp(data["score"]), data.get("reasoning", ""), unsupported=unsupported)
        except (ValueError, KeyError, TypeError) as e:  # malformed judge output is recorded; API errors still raise
            return Verdict(1, error=f"judge_error: {e}")

    def relevance(self, question: str, answer: str, context: str = "") -> Verdict:
        try:
            data = self._call(RELEVANCE_SYSTEM, RELEVANCE_USER.format(
                context=context, question=question, answer=answer))
            behavior = data.get("behavior", "answered")
            if behavior not in {"answered", "abstained", "clarified"}:
                behavior = "answered"
            return Verdict(_clamp(data["score"]), data.get("reasoning", ""), behavior=behavior)
        except (ValueError, KeyError, TypeError) as e:
            return Verdict(1, behavior=detect_behavior(answer), error=f"judge_error: {e}")


# ---------------------------------------------------------------------------
# Heuristic judge for --mock runs and tests. Not a substitute for the LLM judge.

_ABSTAIN = re.compile(
    r"couldn't find|could not find|not (?:in|covered in) the (?:ledgerly )?help center|"
    r"don't have (?:that|this|any) information|do not have (?:that|this) information|no information",
    re.I,
)
_CLARIFY = re.compile(r"which (?:plan|one|payment method|integration)[^?]*\?|could you (?:tell|confirm)[^?]*\?", re.I)
_WORD = re.compile(r"[a-z0-9$%.,]+")


def detect_behavior(answer: str) -> str:
    if _ABSTAIN.search(answer):
        return "abstained"
    if _CLARIFY.search(answer) and len(answer) < 200:
        return "clarified"
    return "answered"


def _tokens(text: str) -> set[str]:
    return {t.strip(".,") for t in _WORD.findall(text.lower()) if len(t.strip(".,")) > 2}


class HeuristicJudge:
    usage = {"input_tokens": 0, "output_tokens": 0}

    def faithfulness(self, question: str, context: str, answer: str) -> Verdict:
        if detect_behavior(answer) == "abstained":
            return Verdict(5, "abstention")
        ctx = _tokens(context)
        sents = [s for s in re.split(r"(?<=[.!?])\s+", answer) if s.strip()]
        support = [len(_tokens(s) & ctx) / max(len(_tokens(s)), 1) for s in sents] or [0.0]
        worst = min(support)
        score = 5 if worst >= 0.8 else 4 if worst >= 0.65 else 3 if worst >= 0.5 else 2 if worst >= 0.3 else 1
        return Verdict(score, f"min sentence overlap {worst:.2f}")

    def relevance(self, question: str, answer: str, context: str = "") -> Verdict:
        behavior = detect_behavior(answer)
        if behavior == "abstained":
            return Verdict(4, "abstention", behavior=behavior)
        overlap = len(_tokens(question) & _tokens(answer)) / max(len(_tokens(question)), 1)
        score = 5 if overlap >= 0.5 else 4 if overlap >= 0.35 else 3 if overlap >= 0.2 else 2
        return Verdict(score, f"question overlap {overlap:.2f}", behavior=behavior)
