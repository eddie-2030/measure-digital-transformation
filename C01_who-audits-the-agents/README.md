# C01 Who Audits the Agents? Human Oversight as a Measurable Capability

Companion code for the *Measure Digital Transformation* edition "Who Audits the Agents?".

> All data in this folder are synthetic, generated to illustrate the method. They do not come from Meta, any client, or any real organization. The goal is to show how a business question, a measurement strategy, the statistics, and the code connect end to end.

## The business question

"Every AI agent decision we care about has a human reviewer behind it. How do we know that oversight is actually working?"

## The measurement strategy

Treat human oversight as a capability you measure, not a box you tick. Measure it on three things:

| Dimension | Question | Metric in this package |
|---|---|---|
| **Consistency** | Do reviewers agree with each other beyond chance? | Krippendorff's alpha (all reviewers), Cohen's kappa (reviewer pairs) |
| **Correctness** | Do they agree with verified ground truth? | Sensitivity (error-catch rate), specificity, and errors missed by every reviewer, on a golden set |
| **Coverage** | Are they looking at the cases that matter? | Double-review rate, overall and for high-value decisions; golden-set size |

## The scenario (synthetic)

An AI agent approves or denies customer refund requests. Each of 1,500 live decisions goes to one of three review sites (A, B, C; 4 reviewers each), where two reviewers drawn at random label the AI decision "Correct" or "Error" (3% go to a third reviewer). The true AI error rate is 8%. Sites A and B are well calibrated. Site C misses most true errors and labels almost everything "Correct".

An expert panel verified 300 past decisions (100 per site, 25% of them true errors) and the review tool seeded them blind into the queues. This is the golden set. It is enriched with errors on purpose: a random sample of 300 would hold only about 24 errors, too few to measure a catch rate. Enrichment does not bias sensitivity or specificity, because each is computed within one kind of truth.

## The method in plain words

1. **Naive (Step 1): raw percent agreement.** The share of reviewer pairs that gave the same label. When 92% of AI decisions are correct, two reviewers who say "Correct" to nearly everything will agree about 90% of the time without looking closely.
2. **Correct (Step 2): chance-corrected agreement.**
   kappa = (observed agreement - chance agreement) / (1 - chance agreement).
   Cohen's kappa works for one pair of reviewers who rated the same cases. Here reviewers only see their own site's queue, so 48 of the 66 possible pairs share no cases. Krippendorff's alpha gives one chance-corrected number for any number of reviewers with missing ratings. 1 = perfect, 0 = no better than chance.
3. **Check (Step 3): agreement is not accuracy.** Two reviewers can agree and both be wrong. On the golden set we measure what share of their reviews of true errors each site flags, and compare every estimate with the true parameters used to simulate the data (`data/true_parameters.csv`).
4. **Decide (Step 4).** Translate the golden-set results into AI errors that pass review unflagged per 1,000 decisions, and fill an Oversight Scorecard with targets.

## How to run (about one minute)

From this folder:

```bash
python3 data/generate_data.py    # optional: regenerates data/*.csv (seed 20261101)
Rscript r/analysis.R             # base R; writes outputs/r_headline.csv
python3 python/analysis.py       # writes outputs/results.json, figures, scorecard example
```

Run R before Python so `results.json` records both languages' numbers. Python needs numpy, pandas, and matplotlib. R needs base R only (no packages). Krippendorff's alpha and kappa are written out in both languages so you can read every step. If the optional `krippendorff` Python package is installed, the script cross-checks against it (it matches to 4 decimals).

## What is in the folder

| Path | What it is |
|---|---|
| `data/generate_data.py` | Seeded generator; the data-generating process and true parameters are documented at the top |
| `data/oversight_reviews.csv` | 3,655 review rows (long format), with a few messy rows on purpose |
| `data/true_parameters.csv` | Ground truth: error rate and every reviewer's true sensitivity and specificity |
| `data/data_dictionary.md` | Columns, types, meanings, units |
| `python/analysis.py` | Steps 1-4, figures, `results.json` |
| `r/analysis.R` | The same steps and numbers in base R |
| `outputs/results.json` | Every number the article may quote, with rounding, meaning, and Python and R values |
| `outputs/fig_1_agreement_vs_alpha.png` | Raw agreement vs. alpha, by site |
| `outputs/fig_2_error_catch_by_site.png` | Share of reviews of true errors flagged, by site, against the golden set |
| `outputs/scorecard_example.csv` | The scorecard filled with this synthetic data |
| `templates/oversight_scorecard.csv` | Blank Oversight Scorecard: metrics, definitions, targets, red flags |

## What to look for in the output

- **Raw agreement is high everywhere (90-92%), including Site C.** On that number alone, all three sites look the same.
- **Alpha separates them.** Overall alpha is 0.46; Sites A and B are about 0.57 (95% CIs 0.44-0.70 and 0.44-0.68); Site C is about zero (-0.03, CI -0.06 to 0.02). Site C's reviewers agree no more than chance would predict.
- **Even good sites sit at or below 0.67.** With an 8% error rate, chance agreement is already very high, so a few disagreements, including false alarms on correct decisions, pull alpha down sharply. This is the prevalence problem: read alpha together with the golden-set results, not alone.
- **Pooled numbers hide the weak site.** An overall alpha of 0.46 reads as one middling operation; it is really two middling sites pooled with one whose agreement is no better than chance.
- **The golden set shows what agreement cannot.** Sites A and B's reviewers flag 94% and 84% of their reviews of true errors (47 of 50 and 42 of 50 reviews); Site C's flag 14% (7 of 49). At Site C, both reviewers said "Correct" on 71% of true errors.
- **The per-1,000 estimates run high in this seed.** The live error rate is estimated from Sites A and B by correcting their flag rate with their golden-set sensitivity and specificity. Golden-set specificity happens to run above the true value here (0.98-0.99 against about 0.95), so the estimated live error rate is 10.5% (case-level bootstrap 95% CI about 8% to 13%) against a true 7.9%. Every errors-per-1,000 figure inherits that roughly 1.3x overestimate: Site C is estimated at 74 per 1,000 (95% CI about 48 to 103) against a simulated true value of 59. Quote the estimate together with its interval, or say it is an estimate.
- **Intervals are case-level.** Each golden case is reviewed about twice, so the roughly 50 reviews per site are not independent. Confidence intervals for sensitivity and for errors per 1,000 come from a bootstrap that resamples cases (2,000 resamples), not reviews. Site A missed no golden errors with both reviewers (0 of 25), so its per-1,000 interval collapses to 0-0; the true rate is small but not zero.
- **Every estimate is checked against the truth** printed next to it in Step 3. Expect sampling noise: the golden set is small, and specificity on it happens to run a little above the true value in this seed.

## Connect to the article

The article argues that human oversight of AI agents is a capability you can measure on consistency, correctness, and coverage, and that a high "reviewers agree" number can hide oversight that is not working. This package produces the article's chart (raw agreement vs. alpha by site) and its numbers: a 91% agreement headline, an alpha of 0.46 overall and near zero at Site C, and a site whose reviewers flag only 14% of their reviews of expert-verified errors. The Oversight Scorecard in `templates/` is the practical takeaway: three dimensions, nine metrics, targets, and red flags you can adopt for your own review operation. It extends A01's rung 3 ("quality next to speed") from AI outputs to the humans who check them.
