# Findings

Hand-written analysis of the full run. Numbers come from [`eval_report.md`](eval_report.md) and the per-question files in [`../results/runs`](../results/runs).

## Decision

**Do not ship yet.** All 8 configurations clear the five launch gates on the judge's scores, but the judge is only trusted on one of its two rubrics:

| Rubric | Agreement with 15 human labels (pass/fail kappa) | Bar | Trusted? |
|---|---|---|---|
| Faithfulness | 0.86 | 0.6 | Yes |
| Relevance | 0.38 | 0.6 | No |

So the faithfulness results below can be used. The relevance results cannot, which is why relevance reads 98 to 100% for every configuration.

## What the experiments show

1. **The grounded prompts stopped the only invented fact, then over-corrected.**
   - The baseline prompt was 89% safe on unanswerable questions. On Q43 it said Enterprise pricing is "custom," which is correct, then told the customer to contact a "sales team" the docs never mention.
   - Both grounded prompts (v2, v3) reached 100% safe.
   - The cost: v2 answered a FreshBooks question correctly, then added "I couldn't find that in the Ledgerly help center" anyway (Q25). The prompt's rule to "reply exactly" with the refusal line made it refuse even when it had the answer.
   - The combined run declined a question about Scale's uptime (Q46) that the baseline had answered correctly.
   - **Product takeaway:** a grounding prompt trades made-up facts for false refusals, so both have to be tracked.
2. **Retrieval misses turn into wrong answers.**
   - TF-IDF retrieval had the lowest recall (92% vs 97% for Gemini embeddings).
   - On Q31 it missed the plan page and told an EU customer on Scale they could store data in the EU. That is wrong: only Enterprise can.
   - Neural embeddings earn their cost here.
3. **Chunk size barely matters on this corpus.**
   - Faithfulness was 98 to 100% for chunk sizes of 400, 800 and 1600 characters, and every confidence interval includes zero.
   - The one real effect: at 400 characters, a supporting sentence for Q10 landed in a chunk that was not retrieved, so the answer stated something its retrieved context did not contain.
4. **Multi-hop questions are the weak spot for retrieval:** 83% recall, compared with 100% for single-hop.
5. **The test set is too easy to separate the options.**
   - With 49 questions and scores near 100%, most differences are 0 to 2 points with intervals of about plus or minus 6 points.
   - The harness works, but it needs harder and more varied questions to rank configurations with confidence.

## What calibration revealed about the judge

- **v1 to v2** (see [`rubric_changelog.md`](rubric_changelog.md)): faithfulness agreement rose from 0.60 to 0.86 after the rubric stopped penalizing off-topic answers and refusals, and after fixing two of my own labeling errors.
- **The v2 relevance change backfired.**
  - I gave the relevance judge the retrieved context so it could catch wrong refusals.
  - Instead, it began scoring factually wrong but on-topic answers as irrelevant (C04, C05, C12, C15 all scored 1), mixing accuracy into relevance.
  - That is most of the remaining relevance disagreement.
- **The judge made an arithmetic mistake.** On C10 it computed $29 + 2 x $8 as $43 and passed an answer with the same error. A small judge model is not a reliable calculator.

## Next steps

1. **Rubric v3 for relevance.** Judge relevance without the context. Run a separate yes/no check, with context, only when the bot declines, to catch wrong refusals. Recheck it on new held-out labels, not the same 15.
2. **More failing examples for calibration.** Only 2 of 15 labels fail on relevance, which makes kappa unstable.
3. **Harder golden questions.** Add more multi-hop questions and near-miss numbers so configurations separate.
4. **Fix the refusal wording in the grounded prompt** so it never adds the refusal line after a real answer.
