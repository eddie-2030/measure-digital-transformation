# F01 The Broken Bottom Rung: Entry-Level Work in the AI Era

Companion code for the *Measure Digital Transformation* newsletter (Dr. Eddie Lin). Method: **cohort comparison on a tenure clock + career sequence analysis**, with matching Python and R code.

> All data in this folder are synthetic, generated to illustrate the method. They do not come from Meta, any client, or any real organization. The goal is to show how a business question, a measurement strategy, the statistics, and the code connect end to end.

## The business question

*What happens to skill-building when AI does junior work?* Are new hires still climbing toward independent, expert work, or has AI removed the rungs they used to learn on?

## The measurement strategy

1. **Measure development, not just output.** Code each person's month as a work state: **R** routine junior tasks, **S** supervised complex ("stretch") work, **I** independent complex work, **X** left. In real life these come from task logs, assignment levels and review sign-offs.
2. **Use the tenure clock.** Line everyone up by months since hire, not by calendar month, and compare cohorts at the same tenure month.
3. **Handle right truncation and leavers.** Recent cohorts have fewer observed months. A cohort enters a "month 18" metric only if the person has been employed long enough to reach month 18 by the data cut (decided by the calendar, not by what happened to them). Leavers stay in as X.
4. **Look at whole sequences.** Compare the full 18-month paths, group them into trajectory types, and compare the type mix across cohorts. Use a job family with little AI change as a contrast.

## The method in plain words

- **Chronogram (state distribution by tenure month):** at each month since hire, the % of people in R, S, I and X. One stacked chart per cohort group.
- **Time in state:** mean months in S (stretch work) during the first 12 months.
- **Sequence distance (Hamming):**
  `distance(A, B) = number of months in which person A and person B were in different work states`
- **Clustering into trajectory types:** k-medoids (PAM) with k = 3 on the Hamming distances of the first 18 months. A *medoid* is the most typical real person in a group. The groups are named from their medoid's path: most months in X = "early leaver", most months in I = "climber", the remaining group = "plateau".
- **Uncertainty:** bootstrap 95% intervals (1,000 resamples of people within each family and period; cluster labels held fixed).
- **Ground-truth check:** the generator knows each person's true type, so we can score the clustering (adjusted Rand index and % correctly typed).

## Data and folders

```
data/generate_data.py      seeded generator (seed 20261105); documents the true parameters
data/junior_careers.csv    900 synthetic hires x up to 24 months (19,560 person-months)
data/ground_truth.csv      true trajectory type per person (for the check only)
data/data_dictionary.md    columns, types, meanings, units
python/analysis.py         Steps 1-4 + figures; writes outputs/results_python.csv and results.json
python/check_agreement.py  confirms Python and R agree; rebuilds outputs/results.json
r/analysis.R               the same steps and numbers in base R; writes outputs/results_r.csv
outputs/results.json       every quotable number (python and r values, rounding, meaning)
outputs/fig1_output_up_development_down.png
outputs/fig2_state_distribution_tenure_clock_high.png
templates/junior_development_scorecard.md
templates/person_state_records_template.csv
templates/tenure_state_summary.py / .R   state distribution + time in state for any such csv
```

Scenario: a mid-size services company hires 900 entry-level staff in 6 half-year cohorts (2023H1 to 2025H2) into two job families. In the HIGH-exposure family (operations and support analysts) a GenAI assistant took over most routine tasks from July 2024. The LOW-exposure family (field-service coordinators) saw little AI change. Data cut: October 2026. Messy on purpose: about 2% blank months, non-standard state codes, a few "active after leaving" errors, a hiring-mix shift (post-AI cohorts are smaller and more selective), and team-level noise.

## How to run (about 5 seconds)

Requirements: Python 3 with numpy, pandas, scikit-learn, matplotlib; R 4.x (base R only, no packages).

```bash
python data/generate_data.py      # optional: the csv files are already included
python python/analysis.py
Rscript r/analysis.R
python python/check_agreement.py  # prints AGREEMENT OK when every number matches
```

Each script finds its files relative to its own location, so it also runs from inside its own folder (`cd r && Rscript analysis.R`).

To use the template on your own data: `python templates/tenure_state_summary.py your_records.csv 12` (or `Rscript templates/tenure_state_summary.R your_records.csv 12`).

## What to look for in the output

1. **Naive view (Step 1):** in the HIGH family, post-AI cohorts close about 31% more tasks per month in their first 6 months (34 to 44). It looks like "ramp-up is solved."
2. **Tenure-clock view (Step 2):** in the same family, months in supervised stretch work during the first year fall from 5.24 to 4.23, and the share working independently at month 18 falls from 62% to 42%. The plateau trajectory type rises from 23% to 45% (+22 points, 95% interval 13 to 32).
3. **Timing matters (month 12 vs month 18):** a month-12 snapshot reverses the story. At month 12, more post-AI hires in the HIGH family are already independent: 29% vs 20% (+9 points, 95% interval 1 to 17). The fastest climbers now leave routine work sooner, because the assistant absorbs it. The gap opens later: by month 18 the plateaued group has not climbed, and the share falls from 62% to 42%. In the LOW family, month 12 moves little (21% to 17%; -4 points, 95% interval -12 to 5). This is why you should compare whole sequences rather than any single month.
4. **Contrast:** the LOW family barely moves (plateau share 25% to 23%; independent at month 18, 61% to 60%). The HIGH-minus-LOW gap in the plateau shift is +24 points (95% interval 10 to 37).
5. **Ground truth (Step 3):** the clustering recovers the true types very well (adjusted Rand index 0.98; 99% correctly typed), and the estimated plateau shift (+22) is close to the true shift for the same people (+23; +20 built into the generator).
6. **Python vs R:** `check_agreement.py` should print `AGREEMENT OK`. Every number is in `outputs/results.json` with both values.

Exact numbers live in `outputs/results.json`.

## Design choices you should know about

- **Why k-medoids and not hierarchical clustering?** Hamming distances are whole numbers, so many pairs are tied, and tie handling in `scipy` and `stats::hclust` can differ. PAM with integer costs and "lowest index wins" tie-breaking gives identical clusters in both languages. Hierarchical clustering (e.g., Ward's method) on the same distance matrix is a fine alternative for exploration.
- **Why a home-made random number generator for the bootstrap?** Python and R have different default generators. A tiny Park-Miller generator, written identically in both scripts, makes the bootstrap intervals match exactly.
- **Missing months** are filled with the previous month's state (a missing first month = R). **X is absorbing**: an active state after leaving is treated as a coding error.
- **Cluster labels are held fixed in the bootstrap** (we resample people, not re-cluster), so the intervals ignore clustering uncertainty.
- **Optional pointer:** for richer sequence tools in R (optimal-matching distances, sequence plots), see the TraMineR package (Gabadinho et al., 2011). It is not needed here.

## Caveats

- **Descriptive, not causal.** Cohorts differ in more than AI exposure: hiring markets, onboarding and managers change too. The LOW-family contrast strengthens the AI explanation but does not prove it. A causal design (e.g., difference-in-differences, see edition A01) is a separate step.
- **Right truncation.** The latest cohorts (hired after May 2025) are not yet observed at month 18 and are excluded from month-18 metrics and from the clustering, not imputed.
- **State definitions drive everything.** Coding R/S/I from real work data needs care, and the trajectory types depend on the distance and the number of clusters chosen.
- The synthetic states are cleaner than real data, so the clustering recovers the truth unusually well.

## Connect to the article

The article asks whether AI is removing the bottom rung of the career ladder. This package shows why a dashboard of early output can say "never better" while development quietly stalls: put everyone on the same tenure clock, follow whole career sequences, keep leavers and truncation in view, and compare against a job family where AI changed little. Use `templates/junior_development_scorecard.md` to report these development metrics for every cohort each quarter.
