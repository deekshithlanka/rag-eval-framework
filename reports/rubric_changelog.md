# Rubric changelog

## v1 to v2

**v1 calibration** (judge `gemini-3.1-flash-lite`, 15 hand-labeled examples):

| Rubric | Exact agreement | Pass/fail agreement | Pass/fail kappa | Bar |
|---|---|---|---|---|
| Faithfulness | 80% | 80% | 0.60 | 0.6, met |
| Relevance | 40% | 80% | 0.33 | 0.6, missed |

Result: judge not trusted. Six pass/fail disagreements, reviewed one by one:

| ID | Rubric | Human | Judge | Cause | Fix in v2 |
|---|---|---|---|---|---|
| C08 | Faithfulness | 5 | 3 | Judge penalized an accurate but off-topic answer, mixing relevance into faithfulness | Rubric: never lower faithfulness for incomplete or off-topic answers |
| C10 | Faithfulness | 3 | 5 | Judge missed a wrong total ($43 instead of $45) | Rubric: recompute every derived number and show the calculation |
| C13 | Faithfulness | 5 | 1 | Judge scored "I couldn't find that" as an unsupported claim | Rubric: abstentions always score 5; wrong refusals are measured by the false abstention metric |
| C13 | Relevance | 1 | 4 | Relevance judge could not see the context, so it could not tell the refusal was wrong | Judge now sees the context; a refusal when the answer is in the context scores 1 |
| C12 | Relevance | 3 | 5 | **Label error.** The answer invents a price but directly addresses the question. The rubric scores relevance without regard to accuracy | Relabeled to 5. Faithfulness already scores this answer 1 |
| C15 | Relevance | 3 | 5 | **Label error.** Same as C12: invented steps, but on-topic | Relabeled to 5 |

**Caveat:** the same 15 examples were used to find these problems and will be used to re-check v2, so v2 agreement may be optimistic. A held-out set of new labeled examples is the right next step.

## v2

Results: run `make calibrate` and record them here.
