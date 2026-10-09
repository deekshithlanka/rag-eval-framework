# Judge calibration

Judge: `gemini-3.5-flash-lite`, rubric `v2`, 15 hand-labeled examples.
Bar: pass/fail Cohen's kappa of 0.6 or higher on both rubrics.
Result: **not trusted**.

| Rubric | Exact agreement | Within 1 point | Pass/fail agreement | Pass/fail kappa | Weighted kappa |
|---|---|---|---|---|---|
| Faithfulness | 87% | 93% | 93% | 0.86 | 0.95 |
| Relevance | 47% | 73% | 73% | 0.38 | 0.33 |

Behavior label agreement (answered, abstained, clarified): 100%

## Disagreements (pass/fail flips)

| ID | Rubric | Human | Judge | Label note | Judge reasoning |
|---|---|---|---|---|---|
| C04 | R | 4 | 1 | 600 is the Enterprise limit; Scale is 120. | The answer contradicts the context by stating the Scale plan has 600 requests per minute when the context specifies 120. |
| C05 | R | 4 | 1 | Directly contradicts the context. | The answer directly contradicts the provided context which states that SMS codes are not supported. |
| C10 | F | 3 | 5 | Inputs supported, arithmetic wrong. Final number unsupported. | Every factual claim and the arithmetic ($29 + 2 * $8 = $43) are directly supported by the context. |
| C12 | R | 5 | 1 | Invented price. Enterprise is custom pricing. Relevance relabeled 3 to 5 in v2: the rubric scores relevance without regard to accuracy, and the answer directly addresses the question. Accuracy is caught by faithfulness. | The answer provides a specific price of $199 per month, whereas the context states that the Enterprise plan has custom pricing. |
| C15 | R | 5 | 1 | Invented integration and steps. Relevance relabeled 3 to 5 in v2: the rubric scores relevance without regard to accuracy, and the answer directly addresses the question. Accuracy is caught by faithfulness. | The answer provides instructions for connecting to Shopify, but Shopify is not mentioned in the context as an available integration. |
