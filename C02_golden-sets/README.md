# C02 · Golden Sets: Building Ground Truth You Can Trust

Companion code for the *Measure Digital Transformation* newsletter by Dr. Eddie Lin.

> All data in this folder are synthetic, generated to illustrate the method. They do not come from Meta, any client, or any real organization. The goal is to show how a business question, a measurement strategy, the statistics, and the code connect end to end.

## The business question

*How do we know what the right answer is, when the people who label our AI's work disagree?*

A company audits its AI customer-support agent. To build a **golden set** (the answer key every evaluation, audit and oversight metric is scored against), 1,000 AI-generated replies are labeled **error** (a policy, factual or tone error that needs correction) or **OK** by 15 labelers of different quality: 3 senior QA specialists, 8 typical labelers (one a new hire with 7 labels), 3 lenient "rubber-stampers" who rarely flag anything (and, being fast, give 39% of all labels), and 1 labeler who is close to guessing. Each reply has 3 to 5 labelers assigned; after skipped labels, 18 replies ended up with only 2 labels (kept in the analysis).

## The measurement strategy

1. **Overlap by design.** Every item is labeled by several randomly assigned labelers, so each labeler shares items with many colleagues. The overlap is what makes their error patterns estimable.
2. **Model the labelers, not just the labels.** Fit a latent class model (Dawid-Skene) that estimates each labeler's confusion matrix and each item's probability of being an error.
3. **Spend the expert budget where the answer is uncertain.** Send the items whose probability is closest to 0.5 to an expert adjudicator (budget: 100 items, 10%).
4. **Report the uncertainty with the key**: per-item posterior, per-labeler sensitivity and specificity, and which items were adjudicated (`templates/golden_set_protocol.md`).

## The method in plain words

- **Majority vote** treats every labeler as equally good and every mistake as symmetric. When some labelers almost never say "error," a real error seen by two of them and one careful colleague is outvoted 2 to 1. The key quietly inherits the lenient labelers' blind spot, and the error rate looks lower than it is.
- **Dawid-Skene** gives every labeler a hidden confusion matrix: **sensitivity** (how often they say "error" when it truly is one) and **specificity** (how often they say "OK" when it truly is OK). We never see the truth, but the pattern of agreements across overlapping items reveals who is reliable, the way you work out which witnesses to trust from how their stories overlap. An EM algorithm alternates two steps until nothing changes:
  - *E-step (score the items):* `P(error | labels) ∝ prevalence × Π_j P(label from labeler j | error)`, compared with `(1 − prevalence) × Π_j P(label from labeler j | OK)`.
  - *M-step (score the labelers):* re-estimate the prevalence and each labeler's sensitivity and specificity, weighting each item by its current probability of being an error.

  Each vote then counts by how informative it is. A rubber-stamper's "error" is strong evidence (they almost never say it falsely); their "OK" is weak evidence (they say it to most real errors too). A near-random labeler's votes count for almost nothing.
- **Implementation details (identical in Python and R):** binary Dawid-Skene, written from scratch; initialized from the majority-vote key; one pseudo-count added to each confusion-matrix cell (keeps estimates away from 0 and 1, which matters for the new hire); stops when no item's probability moves by more than 1e-9 (81 iterations here); an item enters the key as "error" when its probability exceeds 0.5. Majority-vote ties (2-2) go to "OK"; ranking ties are broken by a pre-drawn random order.

## How to run (a few seconds in total)

From this folder:

```bash
python3 data/generate_data.py   # writes the synthetic CSVs (seed 20261109)
python3 python/analysis.py      # Steps 0-4; writes outputs/results.json, two figures, templates/adjudication_queue.csv
Rscript r/analysis.R            # same steps in base R; checks every number against Python and fills the "r" values
```

Dependencies: Python 3 with numpy, pandas, matplotlib. R ≥ 4.0 with base R plus **jsonlite** (`install.packages("jsonlite")`), used only to read and write JSON. No Dawid-Skene package is needed in either language.

## Folder contents

| Path | What it is |
|---|---|
| `data/generate_data.py` | Seeded generator; documents the data-generating process and the true parameters |
| `data/labels_long.csv` | One row per item-labeler assignment (with skipped labels and a duplicate row) |
| `data/items.csv` | The 1,000 replies and an automated AI judge's flag (used in Step 4) |
| `data/expert_adjudication_draws.csv` | Pre-drawn expert labels (98% accurate) and a random order, shared by Python and R |
| `data/ground_truth_items.csv` | Hidden truth per item: true label, ambiguous flag |
| `data/ground_truth_labelers.csv` | Hidden truth per labeler: sensitivity, specificity, workload |
| `data/data_dictionary.md` | Columns, types, units, known data issues |
| `python/analysis.py` | Step 0 cleaning → Step 1 majority vote → Step 2 Dawid-Skene → Step 3 truth check and adjudication → Step 4 decision |
| `r/analysis.R` | The same steps and numbers in base R |
| `outputs/results_python_full_precision.json` | Unrounded Python values (R compares 165 values against these) |
| `outputs/results.json` | Every number the article may quote: name, value, rounding, meaning, `python` and `r` |
| `outputs/fig_1_golden_set_errors.png` | Wrong labels in the golden set under five ways of building it |
| `outputs/fig_2_labeler_sensitivity.png` | Each labeler's estimated vs. true sensitivity |
| `templates/golden_set_protocol.md` | One-page golden-set protocol |
| `templates/adjudication_queue.csv` | Adjudication queue format, filled with the first 12 items from this example |

## What to look for in the output

- **Step 0 (cleaning):** 3,860 rows → 1 duplicate dropped, 52 skipped labels removed → 3,807 labels (3.8 per item). True prevalence: **19.5%** (195 errors).
- **Step 1 (majority vote, the naive key):** raw pairwise agreement is **78%**, which sounds reassuring. Yet the key catches only **62%** of real errors, gets **98** of 1,000 labels wrong (75 of them missed errors), and puts the AI agent's error rate at **14.3%** instead of 19.5%.
- **Step 2 (Dawid-Skene):** catches **76%** of real errors, **78** wrong labels, error rate **17.9%** (the model's expected rate, the average probability, is **20.1%**). It recovers each labeler's real catch rate within **3 percentage points** on average (correlation **0.98**); scoring labelers against majority vote misses by **14 points**, because the key shares the errors it is judging. The rubber-stampers come out at about 0.5 sensitivity, plainly below everyone else.
- **Step 3 (truth check and adjudication, 100 expert reviews each):**
  - Dawid-Skene + the **100 most uncertain items**: **41** wrong labels, catches **88%** of real errors. 37 of the 100 items it chose were wrong in the key, and all 37 were fixed.
  - Dawid-Skene + **100 random items**: **76** wrong. Only 3 of the 100 were wrong to begin with; the expert fixed all 3 but, being 98% accurate, introduced 1 new mistake, a net gain of 2. This random draw was unlucky: on average a random 100 would fix a net of about 6 (95% range 0 to 12; a net of 2 or fewer happens about 12% of the time). The targeted 100 would be expected to fix about 35; the realized 37 was slightly lucky.
  - Majority vote + the **100 closest votes**: **52** wrong. Targeting uncertainty helps any key; the model's uncertainty is the better guide.
  - Majority vote + 100 random items: 95 wrong.
  - **63%** of the final key's remaining mistakes sit on the 10% of items that are genuinely ambiguous, where every labeler struggles at once.
- **Step 4 (decision):** the AI agent's measured error rate depends on the answer key: **14.3%** (majority vote), **17.9%** (Dawid-Skene), **19.0%** (Dawid-Skene + targeted adjudication), against a true **19.5%**. Scoring an automated AI judge against the keys: its precision looks like **48%** against majority vote, **69%** against the final key, and is **78%** against the truth. A good judge would look like it cries wolf half the time, because the answer key, not the judge, is missing errors.

## Pitfalls the code makes visible

1. **Conditional independence.** Dawid-Skene assumes labelers err independently given the truth. On the 100 ambiguous items everyone errs together, so the model can be confidently wrong there. That is why 63% of the remaining mistakes sit on them, why adjudication and a sharp guideline still matter, and why the protocol tags ambiguous items.
2. **Enough overlap.** Each labeler needs enough items (the new hire with 7 labels is not estimable) and each item needs 3+ labels.
3. **Consensus is not policy.** The model finds what the labelers collectively mean by "error." Anchor it with an expert-verified seed set and re-check after any policy change. And if most labelers were worse than chance, the model could flip the classes: check the estimated confusion matrices for face validity.

## Connect to the article

The article "Golden Sets: Building Ground Truth You Can Trust" argues that every AI evaluation is only as good as the way its answer key was decided. This package shows each claim with numbers you can reproduce: majority vote looks fine (78% raw agreement) yet misses 38% of real errors because lenient labelers dominate the vote; Dawid-Skene learns each labeler's error pattern from the overlap alone and recovers more of them; and the same 100 expert reviews remove 37 wrong labels when targeted at the uncertain items versus a net 2 when spent at random. It is the step behind C01 "Who Audits the Agents?", which took a golden set as given. Next: D02 asks which questions actually discriminate, with item response theory.
