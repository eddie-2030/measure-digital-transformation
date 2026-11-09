# J01 · The AI-Native People Analytics Team

**Can we trust AI-generated answers about our workforce?**

Companion code for the *Measure Digital Transformation* edition J01 (Dr. Eddie Lin). Synthetic data, Python, and R.

## The business question

A people-analytics team has launched an AI assistant that answers HR and workforce questions ("What is the parental-leave policy in EMEA?", "What was attrition in APAC Sales last quarter?") using retrieval-augmented generation (RAG). Leaders want one number: *how accurate is it?* The dashboard says users give it a thumbs-up 90% of the time and an automated LLM judge rates it 84% accurate. Should we believe either number?

## Measurement strategy

Treat the assistant like any measurement instrument:

1. **A labeled test set.** 600 questions in a realistic mix (50% policy lookups, 30% computed metrics, 20% multi-hop / cross-source), each with its truly relevant ("gold") documents. Twelve have no source at all, so the right answer is "I don't know".
2. **Score retrieval and generation separately.** A RAG answer can fail in two places:
   - **Retrieval:** the right document was never fetched. Measured by *recall@5* (share of gold documents in the top 5), *hit rate* (all gold documents fetched), and *MRR* (how high the first gold document ranks).
   - **Generation:** the document was fetched, but the answer did not stick to it. Measured by *groundedness* (share of the answer's claims supported by the retrieved text).
3. **Validate the judge.** An LLM judge is itself an instrument with error. Compare it with expert grades on a **random** sample of 150 questions. *Sensitivity* is how often it calls truly correct answers correct. *Specificity* is how often it calls truly wrong answers wrong.
4. **Correct the headline for judge error** with the Rogan–Gladen formula:

   `Corrected accuracy = (Judge-rated accuracy + Specificity − 1) / (Sensitivity + Specificity − 1)`

   The 95% interval comes from a bootstrap that resamples *questions* (near-duplicate paraphrases are resampled together), so it includes uncertainty in both the judge rate and the judge's error rates.
5. **Decompose the failures.** For each wrong answer, the trace shows where it broke: retrieval miss, ungrounded generation, grounded but still wrong (for example, an arithmetic error), or answered when no source existed.

## The method in plain words

A lenient judge works like a smoke alarm that never misses a fire but also goes off for burnt toast. If you count alarms, you overcount fires. Once you know how often it misses (sensitivity) and how often it raises a false alarm (1 − specificity), you can work back to the true rate. That is the Rogan–Gladen correction, borrowed from diagnostic testing in epidemiology.

## How to run (about one minute)

Requirements: Python 3 with numpy, pandas, matplotlib, and R (base R only, no packages).

```bash
python data/generate_data.py      # writes data/*.csv (seed 20261114; identical every run)
python python/analysis.py         # Steps 1-4, figures, outputs/results.json
Rscript r/analysis.R              # the same steps in base R -> outputs/r_results.csv
python python/check_agreement.py  # confirms Python = R and adds "python"/"r" values to results.json
```

## What's in the folder

| Path | What it is |
|---|---|
| `data/generate_data.py` | Seeded generator. Its docstring documents the data-generating process and every true parameter |
| `data/rag_eval_records.csv` | 600 evaluation records (one per test question) |
| `data/corpus.csv` | The 300 synthetic documents |
| `data/true_parameters.csv` | True values, used for the ground-truth check |
| `data/data_dictionary.md` | Column definitions |
| `python/analysis.py` | Step 1 naive → Step 2 correct method → Step 3 ground-truth check → Step 4 decision |
| `r/analysis.R` | The same analysis in base R |
| `python/check_agreement.py` | Python-versus-R comparison |
| `outputs/results.json` | Every number the article may quote, with rounding, meaning, and Python and R values |
| `outputs/fig1_judge_overstates_accuracy.png` | Naive versus corrected versus true accuracy, by question type |
| `outputs/fig2_failure_decomposition.png` | Where wrong answers broke, by question type |
| `templates/assistant_evaluation_scorecard.md` | A scorecard to fill in for your own assistant |
| `templates/test_set_template.csv` | Test-set layout: question_id, question_type, gold_doc_ids, gold_answer, retrieved_doc_ids, answer, judge_label, human_label |

## What to look for in the output

- **Step 1 (naive).** Thumbs-up 0.903 and raw judge "accuracy" 0.843. Neither is accuracy. In the human-graded sample, wrong answers still get a thumbs-up 68% of the time.
- **Step 2 (correct method).** Retrieval falls from recall@5 of 0.942 on policy lookups to 0.777 on multi-hop questions. Only 54% of multi-hop questions retrieve *all* the sources they need. The judge's specificity is 0.513: it passes about half of the wrong answers. Corrected accuracy is **0.735 (95% CI 0.62–0.82)**, compared with 0.843 raw. By type: 0.902 for policy lookup, 0.680 for computed metrics, and 0.389 for multi-hop. 76% of the wrong multi-hop answers are retrieval misses.
- **Step 3 (ground truth).** True accuracy in the test set is 0.728. The raw judge is 11.5 points too high; the corrected estimate is 0.6 points off, and its CI contains the truth. Per type, the CIs contain the truth, but the multi-hop point estimate is 12 points below its true 0.508 and its CI is wide (0.09–0.58), because the correction there rests on 120 questions and a weak judge.
- **Step 4 (decision).** Deploy for policy lookups with citations shown. Route computed metrics through governed metric definitions or the analytics team. Hold multi-hop questions until retrieval improves. Re-run the evaluation monthly.

## Honest caveats (built into the code)

- **Pooled judge error rates.** No question type has 30 or more truly correct *and* 30 or more truly wrong human grades, so per-type sensitivity and specificity would be too noisy. Per-type corrected accuracy therefore uses the **pooled** sensitivity and specificity. This assumes the judge errs at the same rate on every question type, which is true by construction here; check it in your data as the human sample grows.
- **With this judge, the correction adds no precision.** With sensitivity + specificity − 1 ≈ 0.48, the corrected CI (0.62–0.82, about 20 points wide) is *wider* than the CI from the 150 human grades alone. Compared like for like, using the same clustered bootstrap, the human-only interval is 0.66–0.81 in Python and 0.67–0.80 in R, about 15 points wide; its Wald interval, 0.663–0.804, is similar. Correction removes the judge's bias, but the extra uncertainty from estimating the judge's error rates outweighs what the 573 judge verdicts add. The judge earns its keep only once its specificity improves, or as a cheap monitor between human-graded rounds. Grading more questions narrows both intervals.
- **Bootstrap intervals depend on the random-number generator.** Python and R agree exactly on every point estimate (104 compared, 0 mismatches). CI bounds can differ by about 0.01 because numpy and R draw different bootstrap samples; both are recorded in `results.json`.
- **How the seed was chosen.** The seed was picked from a scan of 40 seeds as a *typical* draw. Across those seeds, the corrected accuracy ranged from about 0.61 to 0.79 around a true value near 0.71, which is why the interval matters.
- **No real LLM was called.** Retrieval results, answers, claim-support counts, judge verdicts, thumbs and human grades are simulated records with known probabilities (see `data/generate_data.py`).

## Synthetic-data disclaimer

All data in this folder are synthetic, generated to illustrate the method. They do not come from Meta, any client, or any real organization. The goal is to show how a business question, a measurement strategy, the statistics, and the code connect end to end.

## Connect to the article

The article asks whether leaders can trust AI-generated answers about their workforce. This package shows the answer is "measure it first". The two naive numbers a team usually has (90% thumbs-up, 84% judge "accuracy") both overstate an assistant that is truly about 73% accurate, and the gap is concentrated in multi-hop questions, where the right documents are never retrieved. The same discipline from the C01 edition on auditing AI agents applies to the analytics team's own tools: a labeled test set, a human gold standard, and an error-corrected metric, reported by question type with an honest interval.
