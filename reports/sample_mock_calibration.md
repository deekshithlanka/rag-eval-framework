# Judge calibration

Judge: `heuristic`, rubric `v1`, 15 hand-labeled examples.
Bar: pass/fail Cohen's kappa of 0.6 or higher on both rubrics.
Result: **not trusted**.

> MOCK RUN: heuristic judge, not the LLM judge. Numbers only show the harness works.

> Labels are still marked `draft`. Review them before relying on these numbers.

| Rubric | Exact agreement | Within 1 point | Pass/fail agreement | Pass/fail kappa | Weighted kappa |
|---|---|---|---|---|---|
| Faithfulness | 47% | 73% | 67% | 0.30 | 0.34 |
| Relevance | 33% | 67% | 53% | 0.04 | 0.05 |

Behavior label agreement (answered, abstained, clarified): 100%

## Disagreements (pass/fail flips)

| ID | Rubric | Human | Judge | Label note | Judge reasoning |
|---|---|---|---|---|---|
| C02 | R | 5 | 3 | Both numbers match BIL-01. | question overlap 0.25 |
| C03 | F | 1 | 5 | Uses the pre-July 2026 price from BIL-06 as if current. Context mentions $20 but only as the old price, so the claim as stated is unsupported. | min sentence overlap 1.00 |
| C03 | R | 4 | 3 | Uses the pre-July 2026 price from BIL-06 as if current. Context mentions $20 but only as the old price, so the claim as stated is unsupported. | question overlap 0.25 |
| C04 | F | 1 | 4 | 600 is the Enterprise limit; Scale is 120. | min sentence overlap 0.78 |
| C05 | F | 1 | 4 | Directly contradicts the context. | min sentence overlap 0.77 |
| C07 | R | 5 | 2 | Fully supported. | question overlap 0.08 |
| C09 | R | 5 | 3 | Correct arithmetic from supported numbers. | question overlap 0.33 |
| C10 | F | 3 | 4 | Inputs supported, arithmetic wrong. Final number unsupported. | min sentence overlap 0.70 |
| C10 | R | 4 | 3 | Inputs supported, arithmetic wrong. Final number unsupported. | question overlap 0.33 |
| C12 | F | 1 | 5 | Invented price. Enterprise is custom pricing. | min sentence overlap 0.91 |
| C12 | R | 3 | 5 | Invented price. Enterprise is custom pricing. | question overlap 0.50 |
| C13 | R | 1 | 4 | False abstention. The answer (365 days) is in the context. | abstention |
