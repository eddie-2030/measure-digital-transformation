# D01 · Measuring AI Supervision Skills: A Psychometric Playbook

**How do we assess judgment, evaluation, and oversight skills, and can we trust the score?**

Companion code for the *Measure Digital Transformation* edition D01 (Dr. Eddie Lin). Synthetic data, Python, and R.
Article code folder: https://github.com/eddie-2030/measure-digital-transformation/tree/main/D01_measuring-ai-supervision-skills

## The business question

About 600 employees who review or supervise AI outputs (content reviewers, operations analysts, claims and support leads) in four business units take a 16-item "AI Supervision Skills" assessment. Leaders want to use the total score for training, staffing, and a certification with a pass mark. Before they do: does the score measure consistently, does it measure the skill we care about, and is it precise enough to pass or fail one person?

The assessment has two kinds of items:

- **12 scenario items** (S01–S12, 0–4 points each): the person reviews an AI output and must spot the error, decide whether to escalate, or verify a claim against a source.
- **4 self-report confidence items** (C01–C04, 0–10): near-duplicate statements such as "I feel comfortable checking AI output".

A few weeks later, everyone also completes an independent **work sample**: an audit of about 40 AI outputs with seeded errors. The share of errors caught is the yardstick (the *criterion*) for real supervision performance.

## Measurement strategy

A score is only as useful as two pieces of evidence behind it:

1. **Reliability:** does the score measure consistently, and how precise is one person's score?
2. **Validity:** does it measure the skill we care about? The *Standards for Educational and Psychological Testing* (AERA, APA, NCME, 2014) name several sources of validity evidence. This package covers three: **content** (do the scenarios cover the real supervision tasks? a review step, in the report card), **internal structure** (do the items hang together as intended?), and **relations to other variables** (does the score predict the audit?). The code quantifies the last two.

## The method in plain words

- **Reliability** is the share of the differences between people's scores that reflects real skill differences rather than noise (0 = all noise, 1 = no noise).
- **Cronbach's alpha** estimates reliability from how strongly the items move together. It assumes every item reflects the skill equally strongly.
- **McDonald's omega** drops that assumption: a one-factor model estimates how strongly each item reflects the skill (its *loading*), and omega is computed from those loadings. When the items are unequal, omega is the better estimate (McNeish, 2018).
- **Item-rest correlation** is the correlation between one item and the sum of the other items. A good item rises with the rest of the test. Near zero: the item measures something else, or nothing. Negative: the scoring key is probably wrong.
- **Standard error of measurement (SEM)** turns reliability into score points:

  `SEM = SD × √(1 − reliability); a person's score is roughly their true score ± 2 × SEM`

  People whose score is within ±1.96 SEM of a cutoff cannot be confidently classified as pass or fail.
- **Spearman-Brown** projects how reliability would change if you added items as good as the current ones: `k_new = k × [target × (1 − rel)] / [rel × (1 − target)]`.
- **Validity coefficient**: the correlation between the score and the audit, with a 95% interval (Fisher z). The gain of one score over another uses Zou's (2007) interval for two correlations that share the same criterion.

Rules of thumb (conventions, not laws; e.g., Nunnally & Bernstein, 1994): about 0.70+ for group-level or research use, about 0.90+ for high-stakes decisions about individuals.

Not taught here (later editions): factor analysis in depth (D03), item response theory (D02), measurement invariance across groups (D08), self-assessment calibration (D09).

## How to run (about one minute)

Requirements: Python 3 with numpy, pandas, scipy, statsmodels, matplotlib; R (base R only, no packages).

```bash
python data/generate_data.py      # writes data/ai_supervision_assessment.csv + ground-truth files (seed 20261104)
python python/analysis.py         # Steps 0-4, figures, outputs/results.json, outputs/python_item_statistics.csv
Rscript r/analysis.R              # the same steps in base R -> outputs/r_results.csv, outputs/r_item_statistics.csv
python python/check_agreement.py  # confirms Python = R on all 80 numbers and adds the "r" values to results.json

# optional: the reusable item-statistics tool on any csv (here: the 12 scenario items, unscreened)
python templates/item_stats.py data/ai_supervision_assessment.csv --items S01 S02 S03 S04 S05 S06 S07 S08 S09 S10 S11 S12 --out outputs/example_item_stats_python.csv
Rscript templates/item_stats.R data/ai_supervision_assessment.csv S01,S02,S03,S04,S05,S06,S07,S08,S09,S10,S11,S12 outputs/example_item_stats_r.csv
```

Every script finds its files relative to its own location, so you can run it from any folder. Omega uses `statsmodels.multivariate.factor.Factor(method="ml")` in Python and `factanal()` in R (one-factor maximum likelihood); across all 80 numbers, Python and R differ by less than 0.000001 before rounding.

**Missing-data rule (both languages):** reliability and item statistics use pairwise-complete covariances; a person's scale score is prorated (mean of answered items × number of items) if at least 75% of the scale's items were answered. **Speeder rule:** exclude anyone who finished in under 8 minutes or gave the same answer to every scenario item.

## What's in the folder

| Path | What it is |
|---|---|
| `data/generate_data.py` | Seeded generator; its docstring documents the data-generating process and every true parameter |
| `data/ai_supervision_assessment.csv` | 612 test records (600 people + 12 speeders): items, completion time, audit results |
| `data/true_parameters.csv` | Generating values and the true (population) reliability and validity of each score |
| `data/true_person_values.csv` | Each person's true skill and true score (used only for the Step 3 checks) |
| `data/data_dictionary.md` | Column definitions, scoring, and the missing-data and speeder rules |
| `python/analysis.py` | Step 0 screen → Step 1 naive → Step 2 correct analysis → Step 3 ground truth → Step 4 decision; figures |
| `r/analysis.R` | The same analysis in base R |
| `python/check_agreement.py` | Python-versus-R comparison; writes both values into `results.json` |
| `outputs/results.json` | Every number the article may quote, with rounding, meaning, and Python and R values |
| `outputs/python_item_statistics.csv`, `outputs/r_item_statistics.csv` | Item statistics within the 16-item total and within the 12 scenario items |
| `outputs/fig1_consistent_is_not_valid.png` | Reliability vs. validity of the confidence block, the naive total, and the cleaned scale |
| `outputs/fig2_too_close_to_call.png` | Score distribution with the cutoff and the ±1.96 SEM band |
| `templates/assessment_quality_report_card.md` | One-page report card to fill in before a score is used for decisions |
| `templates/item_statistics_template.csv` | Item statistics template: item, n, mean, sd, item_rest_r, alpha_if_deleted, decision |
| `templates/item_stats.py`, `templates/item_stats.R` | Reusable function / script that fills the template for any item-response csv |

## What to look for in the output

- **Step 0 (screening).** 12 of 612 records are speeders (under 8 minutes, straight-lined answers); 600 people remain. 3.1% of answers are blank; 9 unit labels are fixed. Speeders matter: on the unscreened file, the 12 scenario items' alpha is 0.76 instead of 0.74, because identical answers look "consistent".
- **Step 1 (naive).** Alpha for all 16 items is **0.82**. The usual verdict: "reliable, certify people on the total." Note that the 12 skill scenarios alone have alpha **0.74**: the four confidence items *raise* alpha.
- **Step 2 (correct analysis):**
  - **Item analysis** finds the two flawed scenarios: **S06** (item-rest r **0.03**, an ambiguous scenario) and **S11** (**−0.20**, a mis-keyed answer). Deleting either one *raises* alpha (to 0.77 and 0.79). The kept items range from 0.35 to 0.55.
  - **Item analysis alone does not catch the confidence block.** Inside the 16-item total, the confidence items have the *highest* item-rest correlations (0.72 and up). The structure check does: confidence items correlate **0.83** with each other but only **0.17** with the scenarios (scenario items correlate 0.31 with each other). They form their own cluster.
  - **Reliability, scale by scale:** cleaned 10-item scenario scale alpha **0.82**, omega **0.82** (loadings 0.40 to 0.63); confidence scale alpha **0.95**, omega **0.95**. Dropping two items made the scale *more* reliable (0.74 → 0.82).
  - **Validity (correlation with the audit, 95% CI, n = 569):** cleaned scenario scale **0.58 (0.52–0.63)**; naive 16-item total **0.44 (0.37–0.50)**; confidence **0.22 (0.14–0.30)**. The cleaned scale's gain over the naive total is **0.14 (0.08–0.19)**. The most consistent score (confidence, 0.95) is the least valid.
  - **Precision for individuals:** SD 6.8 points, SEM **2.9** points. At the proposed cutoff of 26/40 (27% pass), the band runs from 20.3 to 31.7, and **46%** of people fall inside it: their pass/fail call is not confident.
  - **Spearman-Brown:** reaching 0.90 takes about **20** items as good as the kept ones (**10 more**). This uses omega as reported, rounded to 0.82 (19.8 items, rounded up to 20); with unrounded omega (0.818) the projection is 20.02, which rounds up to 21, so quote "about 20". Rewriting S06 and S11 as good items gets to about 0.85.
  - **Reliability depends on the group:** alpha of the same 10 items is 0.84 in Trust & Safety Review and Claims, 0.80 in Operations, and **0.73** in Customer Support, where scores spread less (SD 5.6 vs. 6.7–7.3).
- **Step 3 (ground truth).** The two flagged items are exactly the two built to be flawed, and the screen finds exactly the 12 speeders. Estimated vs. true reliability: cleaned scale 0.82 vs. **0.81**, confidence 0.95 vs. 0.95, 12 scenario items 0.74 vs. 0.75. Estimated vs. true validity: 0.58 vs. **0.57** (cleaned), 0.44 vs. 0.44 (naive), 0.22 vs. 0.21 (confidence). The naive total's alpha (0.82) *understates* its true reliability (**0.91**): it reliably measures a mix of skill and confidence. Comparing observed with true scores, **13%** of people get the wrong pass/fail call at the cutoff; **96%** of them sit inside the SEM band (26% misclassified inside the band vs. 1% outside).
- **Step 4 (decision).** Fix or drop S06 and S11. Report confidence separately (useful for change management, not as a skill). Use the cleaned scenario score now for team-level training needs (reliability 0.82 clears the group-level bar). Do not certify individuals on it yet: add about 10 good scenarios to approach 0.90, or send everyone inside the band to a second assessment.
- **Side note: attenuation.** The audit is not perfectly reliable either (split-half 0.92). Correcting for unreliability in both measures gives 0.66 (true value 0.66). Use the observed 0.58 for decisions; the corrected value shows the ceiling the test is up against.

## Honest caveats (built into the code)

- **Alpha is a lower bound when items are unequal or the test is not one-dimensional.** Here the cleaned scale's alpha and omega are both 0.82, because the loadings are not very different. For the naive total, alpha (0.82) is far below the true reliability (0.91), because the test mixes two traits. A high alpha did not mean "one skill", and a lower alpha would not have meant "unreliable".
- **Omega here is one-factor omega for each scale.** We fit it only to scales we expect to be one-dimensional. Testing that formally is factor analysis (edition D03).
- **The SEM is one number for everyone.** In reality precision varies along the score range (item response theory, edition D02, models that). The SEM band is a practical, conservative screen.
- **Shares near the cutoff depend on where the cutoff sits.** A cutoff in the thick of the distribution puts more people in the band. 26/40 is a proposed cutoff for illustration.
- **One sample, one seed.** Estimates move from sample to sample; the true values in `data/true_parameters.csv` do not. Change `SEED` in the generator to see how much.
- **The validity coefficients are higher than many real-world ones**, because the audit was built to be a clean, reliable work sample. Real criteria (manager ratings, error rates in production) are noisier and give lower correlations.
- **Prorating assumes skipped items are missing at random** (true here by construction). If people skip the hard items, prorating flatters them.

## Synthetic-data disclaimer

All data in this folder are synthetic, generated to illustrate the method. They do not come from Meta, any client, or any real organization. The goal is to show how a business question, a measurement strategy, the statistics, and the code connect end to end.

## Connect to the article

The article, *Measuring AI Supervision Skills: How to Know Whether Your Assessment Deserves Trust*, asks how to assess judgment, evaluation, and oversight skills, the core of supervising AI. This package shows why a single alpha of 0.82 is not an answer. The same test produces a confidence score that is the most consistent (0.95) and the least valid (0.22), a naive total that predicts real error-catching worse than a shorter, cleaned scale (0.44 vs. 0.58), and two flawed scenarios that an item analysis finds in minutes. Even the cleaned scale, good enough for team-level training decisions, leaves 46% of people too close to a pass mark to call. The C01 edition measured whether reviewers agree with each other; this one measures how well a test measures each person's skill, and what that score can and cannot carry.
