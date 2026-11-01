# B06 · Don't Launch Everywhere at Once: Who Benefits from AI?

Companion code for the *Measure Digital Transformation* edition on heterogeneous treatment effects, by Dr. Eddie Lin.

> All data in this folder are synthetic, generated to illustrate the method. They do not come from Meta, any client, or any real organization. The goal is to show how a business question, a measurement strategy, the statistics, and the code connect end to end.

## The business question

A randomized pilot of a GenAI assistant raised output by about 7% on average, and the result is "significant". Should we launch it everywhere? **Does the average hide groups that are left behind?**

## Measurement strategy

- **Design:** 2,450 knowledge and support workers in 120 teams across 4 regions. Within every team, about half got access to the assistant at random. Randomization is what lets us read *differences in the effect* between groups as causal.
- **Outcome:** weekly quality-adjusted tasks resolved over the 8-week pilot, adjusted for the same measure in the 8 weeks before.
- **Pre-specified moderators:** work language (English vs. non-English team) and tenure (novice = under 12 months). Both were written down before the analysis, with a reason.
- **Candidate noise:** region, role family, shift and remote work. By construction, none of them changes the effect.
- **Uncertainty:** people in a team share a manager and a queue, so standard errors are **clustered by team**.

## The method in plain words

1. **Naive approach (Step 1).** Report one average. Then slice the data 14 ways (each region, role, shift…) and ask "is the effect different here?" with no correction. The average hides a group with no measurable gain, and the slicing "finds" a pattern that is pure noise.
2. **Pre-specified interaction model (Step 2a).** One regression in which the AI effect may differ by language and by tenure:

   `Outcome = b0 + b1·AI + b2·NonEnglish + b3·(AI × NonEnglish) + b4·Novice + b5·(AI × Novice) + controls`

   Here **b3** is how much the AI effect differs for non-English teams, and **b5** how much it differs for novices. We also estimate the effect inside each subgroup, with 95% intervals.
3. **ML discovery step (Step 2b).** When there are many possible moderators, a causal-ML learner searches for them. We use a **DR-learner**: for each person, build a "pseudo-outcome" whose average in any group equals that group's treatment effect, then let a random forest predict it from the covariates. The trees split where the *effect* differs, not where output differs. Everything is cross-fitted by team, so each person's prediction comes from a model that never saw their team. We then check that the people predicted to benefit most really do benefit more.
4. **Ground-truth check (Step 3).** Because the data are synthetic, we know every employee's true effect, and we grade each method against it.
5. **Decision (Step 4).** A phased-rollout grid built on rules fixed in advance: launch where the lower 95% bound clears the smallest worthwhile gain (+1 task/week), hold and fix where the upper bound falls short, and re-test where the evidence is unclear.

## How to run (about 1 minute)

From this folder:

```bash
python3 data/generate_data.py      # writes data/ai_pilot_heterogeneity.csv (seed 20261153)
python3 python/analysis.py         # Steps 1-4 in Python; writes outputs/results.json, headline_python.csv, figures
Rscript r/analysis.R               # the same steps in R, reading the SAME csv; writes outputs/headline_r.csv, ml_r.csv, rollout_grid_r.csv
python3 python/merge_results.py    # puts the R numbers next to the Python ones in results.json and checks they agree
```

`python/analysis.py` runs the merge automatically if the R outputs already exist, so running R before Python also works. The merge step exits with an error if any headline number differs between Python and R at its stated rounding.

**Python:** numpy, pandas, statsmodels, scikit-learn, matplotlib (scipy comes with statsmodels).
**R:** base R plus two packages: `sandwich` (team-clustered standard errors) and `rpart` (regression trees, shipped with every R installation as a recommended package). We meant to use `grf` (causal forests) in R, but it could not be installed in the build environment (CRAN unreachable). The R ML step is therefore the same DR-learner as in Python, with a small bagged-tree forest written in base R plus `rpart`. An optional `grf` block is commented at the end of Step 2b in `r/analysis.R` if you have it.

## Files

| Path | What it is |
|---|---|
| `data/generate_data.py` | Seeded generator; the docstring documents the data-generating process and true parameters |
| `data/ai_pilot_heterogeneity.csv` | The synthetic pilot (one row per employee) |
| `data/data_dictionary.md` | Columns, types, units, true values |
| `python/analysis.py` | Steps 1 to 4, figures, results.json |
| `python/merge_results.py` | Merges the R numbers and checks Python–R agreement |
| `r/analysis.R` | Steps 1 to 4 in R |
| `outputs/results.json` | Every number the article may quote: `headline` (both languages, with rounding and meaning), `ml_step`, `ground_truth_check`, `naive_slicing_*`, decision grids, `agreement` |
| `outputs/fig_1_subgroup_effects.png` | Article chart: subgroup effects with 95% CIs vs. true effects and the overall average |
| `outputs/fig_2_cate_variable_importance.png` | What the ML step found (Python forest) |
| `templates/heterogeneity_analysis_plan.md` | Pre-registration template: moderators, minimum subgroup size, decision rules |
| `templates/rollout_decision_grid.csv` | Phased-rollout grid with the synthetic example rows filled in |

## What to look for in the output

- **The average looks fine:** +3.00 tasks/week, **+7.2%** (95% CI 6.0% to 8.4%), p < 0.001. On its own, this says "launch everywhere".
- **Non-English teams show no measurable gain:** 0.0% (95% CI −1.6% to +1.6%; true effect about +0.9%). English-language teams get +9.7%.
- **Novices gain most:** +11.7% for tenure under 12 months vs. +5.6% for experienced staff.
- **Interaction model:** b3 (AI × NonEnglish) = −4.18 tasks/week [−5.01, −3.35]; b5 (AI × Novice) = +2.65 [1.74, 3.55]. Both 95% CIs contain the true values (−3.6 and +2.2).
- **Naive slicing misleads:** of 14 uncorrected slices, *remote workers* look like they benefit less (−0.93 tasks/week, p = 0.006). Remote work has no true effect. After Holm correction p = 0.077 and the flag disappears, while language and tenure survive. Across 200 simulated pilots, at least one of the 8 pure-noise slices was a false alarm 24% of the time uncorrected and 2% with Holm.
- **The ML step finds the right drivers:** work language (39%) and tenure (30%) top the importance ranking in Python; R gives 39% and 30%. Each noise covariate gets 3 to 5%. Baseline output ranks third (12 to 13%) because novices start lower, so it partly stands in for tenure. Predicted effects correlate 0.87 with the true individual effects in both languages, and the predicted high-benefit half really did gain more on held-out data (+4.1 vs. +1.9 tasks/week in Python).
- **Ground truth:** all 9 subgroup 95% CIs contain the true subgroup effect.
- **Decision:** launch to English-language teams now (novices first, as they gain most). Hold experienced staff on non-English teams (−0.88 [−1.81, 0.04]) for fixes and localization. Re-test non-English novices (+1.83 [0.60, 3.07]): promising but uncertain.

Exact values are in `outputs/results.json`. If you change the seed, every number changes.

## Connect to the article

The article asks a leader's question: *does the average hide groups that are left behind?* This package answers it step by step. The headline +7% is real, but it averages a strong gain for English-language teams and novices with no measurable gain for non-English teams. The interaction model, which you can explain in one formula, recovers that pattern with honest, team-clustered intervals. The ML step confirms that no other big driver is hiding, and the noise covariate that "looked significant" shows why you write your moderators down before the data arrive. The rollout grid turns the evidence into a phased launch: go where the gain is clear, fix where it is not, and re-test. That is the same call as pushing back on a full launch when non-English quality lags.

The previous edition ([link to A01 edition], "Adoption Is Not Impact") asked whether a rollout worked *on average*. This one asks *for whom*.
