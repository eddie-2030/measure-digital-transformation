# Assistant Evaluation Scorecard (people-analytics AI assistant)

Use once before launch and then monthly. Fill one column per question type. Treat the assistant like any measurement instrument: a labeled test set, retrieval scored separately from generation, and an automated judge checked against human grading before any accuracy number is reported.

**Assistant / version:** ______  **Eval date:** ______  **Test-set version:** ______  **Owner:** ______

## 1. Test set (is it a fair sample of real use?)

| Check | Value | OK? |
|---|---|---|
| Number of questions (aim for 300 or more) | | |
| Mix by type matches real usage logs (policy / metric / multi-hop %) | | |
| Includes questions with no source (the right answer is "I don't know") | | |
| Near-duplicates grouped into clusters | | |
| Refreshed since the last policy or data change (date) | | |

## 2. Retrieval (did it fetch the right sources?)

| Metric | Policy lookup | Computed metric | Multi-hop | Target |
|---|---|---|---|---|
| Recall@5 (share of gold docs in top 5) | | | | ≥ 0.90 |
| Hit rate (all gold docs in top 5) | | | | ≥ 0.85 |
| MRR (how high the first gold doc ranks) | | | | ≥ 0.70 |

## 3. Generation (did it stick to the sources?)

| Metric | Policy lookup | Computed metric | Multi-hop | Target |
|---|---|---|---|---|
| Groundedness (share of claims supported by retrieved text) | | | | ≥ 0.90 |
| Share of answers ≥ 80% grounded | | | | ≥ 0.85 |
| Abstains correctly on no-source questions | | | | ≥ 0.80 |

## 4. Judge validation (can we trust the automated grader?)

Human grades come from a **random** sample (not hand-picked). Report the counts.

| Item | Value |
|---|---|
| Human-graded questions (n) | |
| Truly correct / truly wrong in that sample | / |
| Judge sensitivity = P(judge says correct, given truly correct) | |
| Judge specificity = P(judge says wrong, given truly wrong) | |
| Sensitivity + specificity − 1 (should be well above 0; under 0.5 means a weak judge) | |
| Enough grades to estimate sens/spec per type (at least 30 correct and 30 wrong per type)? If not, use pooled values and say so | Yes / No |

## 5. Accuracy, corrected for judge error

Corrected accuracy = (Judge-rated accuracy + Specificity − 1) / (Sensitivity + Specificity − 1), clipped to 0–1.

| | Thumbs-up rate (satisfaction only) | Raw judge accuracy | Corrected accuracy | 95% CI (bootstrap by question) | Human-only accuracy |
|---|---|---|---|---|---|
| All | | | | | |
| Policy lookup | | | | | |
| Computed metric | | | | | |
| Multi-hop | | | | | |

## 6. Failure decomposition (what to fix?)

Share of wrong answers (human-graded) by where they broke:

| Cause | Policy lookup | Computed metric | Multi-hop | Fix |
|---|---|---|---|---|
| Retrieval miss | | | | Indexing, chunking, metadata filters, query rewriting |
| Ungrounded generation | | | | Prompting, citation enforcement, model choice |
| Grounded but wrong | | | | Calculation tools, governed metric layer |
| No source; should abstain | | | | Abstention threshold, coverage gaps |

## 7. Decision

| Question type | Corrected accuracy | Decision (deploy with citations / route to governed metrics / hold) | Re-test date |
|---|---|---|---|
| Policy lookup | | | |
| Computed metric | | | |
| Multi-hop | | | |

Example rule: corrected accuracy ≥ 0.80 → deploy with citations shown; 0.60–0.80 → route through governed metric definitions or the analytics team; < 0.60 → hold until retrieval improves. Re-run the scorecard monthly and after every policy, data, or model change.
