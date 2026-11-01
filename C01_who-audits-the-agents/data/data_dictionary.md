# Data dictionary

All data in this folder are synthetic, generated to illustrate the method. They do not come from Meta, any client, or any real organization. The goal is to show how a business question, a measurement strategy, the statistics, and the code connect end to end.

## `oversight_reviews.csv`

One row per human review of an AI refund decision (long format). 3,655 rows: 1,500 live AI decisions plus 300 expert-verified golden cases, each reviewed by 2 reviewers (3% by 3), with a few messy rows left in on purpose.

| Column | Type | Meaning | Units / values |
|---|---|---|---|
| `review_id` | string | Unique id of the review row | `V00001`... |
| `decision_id` | string | The AI decision being reviewed | `D0001`...`D1800` |
| `site` | string | Review site that owns the decision's queue | `A`, `B`, `C` |
| `decision_date` | date | Date the AI made the decision | ISO date, Sep-Oct 2026 |
| `ai_decision` | string | What the AI agent decided | `approve`, `deny` |
| `refund_amount_usd` | float | Refund amount requested | US dollars |
| `reviewer_id` | string | Human reviewer (4 per site: A = R01-R04, B = R05-R08, C = R09-R12) | `R01`...`R12` |
| `review_order` | integer | 1st, 2nd, or 3rd (escalation) review of this decision | 1-3 |
| `review_label` | string | Reviewer's verdict on the AI decision, as typed (messy) | `Correct`, `Error`; also variants such as `correct `, `ERR`; blank, `Skipped`, `Unsure` = no usable rating |
| `review_seconds` | integer | Time the reviewer spent on the case | seconds |
| `case_source` | string | `production` = live decision; `golden_seeded` = expert-verified case seeded blind into the queue | `production`, `golden_seeded` |
| `golden_set` | integer | 1 if the case is in the golden set | 0 / 1 |
| `gold_label` | string | Expert-adjudicated truth (golden set only; blank otherwise) | `Correct`, `Error`, blank |

Known messy cases (handled in the analysis scripts):
- 8 reviews were double-submitted (same decision, reviewer, and label; different `review_id`). Keep one.
- 98 reviews have no usable label (blank, `Skipped`, `Unsure`). Treat as missing; Krippendorff's alpha handles this.
- About 1.5% of labels have inconsistent case or spacing. Trim and lower-case before mapping.

Analysis rules: agreement statistics (percent agreement, kappa, alpha) use `case_source == "production"` rows only, so they reflect the real AI error rate. Correctness statistics (sensitivity, specificity, missed-by-all) use the golden set only. The golden set is enriched to 25% errors on purpose; it must not be used to estimate the live error rate.

## `true_parameters.csv`

The ground truth used by `generate_data.py`, so the methods can be checked.

| Column | Type | Meaning |
|---|---|---|
| `level` | string | `population` (design error rate), `realized` (error rate actually drawn), `design` (golden-set error mix), `site_design` (site means used to draw reviewers), `site_reviewer_mean` (average of the 4 reviewers actually drawn), `reviewer` |
| `id` | string | Site letter, reviewer id, or `all` / `production` / `golden` |
| `site` | string | `A`, `B`, `C`, or `all` |
| `parameter` | string | `prevalence`, `error_rate_production`, `error_rate_golden_set`, `sensitivity`, `specificity` |
| `value` | float | Value of the parameter (proportion) |

Sensitivity = probability the reviewer labels a truly wrong AI decision "Error". Specificity = probability the reviewer labels a truly correct AI decision "Correct".
