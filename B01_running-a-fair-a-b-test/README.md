# B01 · Running a Fair A/B Test Inside Your Company

Companion code for the *Measure Digital Transformation* newsletter by Dr. Eddie Lin.

> All data in this folder are synthetic, generated to illustrate the method. They do not come from Meta, any client, or any real organization. The goal is to show how a business question, a measurement strategy, the statistics, and the code connect end to end.

## The business question

*How do we test an AI tool on employees responsibly, and get an answer we can trust?*

A support organization (1,200 analysts, 40 teams, 4 sites) wants to know whether an AI drafting assistant increases **cases resolved per employee per week** without lowering **QA quality**.

## The measurement strategy

1. **Randomize who gets access**, stratified (blocked) by site so both groups are balanced. The control group is a waitlist: they get the tool when the pilot ends.
2. **Fix one primary outcome and the analysis plan in advance** (see `templates/experiment_plan_template.md`).
3. **Measure a 4-week baseline** of the same outcome before launch.
4. **Analyze as randomized** (intention-to-treat) with **regression adjustment for the baseline** (ANCOVA, the idea behind CUPED):

```
cases_post = a + b × assigned + c × cases_pre + site effects + error      (b = effect of giving access)
```

## The method in plain words

- **Why randomize?** Volunteers differ: they were already better and more motivated. Comparing them with everyone else mixes the tool's effect with who they are. A lottery makes the two groups alike on everything, measured or not.
- **Why adjust for the baseline?** Most of the week-to-week differences between people are stable (some people simply resolve more cases). Subtracting out what each person's baseline predicts removes that noise. Because the baseline was measured before randomization, this cannot bias the estimate; it only tightens it. The variance shrinks by about 1 − ρ², where ρ is the pre/post correlation (here ρ ≈ 0.7, so about half the variance goes away).
- **Intention-to-treat (ITT):** compare people as randomized, whether or not they used the tool. ITT answers the rollout question ("what happens if we give everyone access?"). Effect among users ≈ ITT ÷ uptake.
- **Do not adjust for anything measured after launch** (e.g., hours of AI use). It breaks the randomization.

## How to run (about 5 seconds in total)

From this folder:

```bash
python3 data/generate_data.py   # writes the synthetic CSVs (seed 20261108)
python3 python/analysis.py      # Steps 1-4, writes outputs/results.json and the two figures
Rscript r/analysis.R            # same steps in R; fills the "r" values and checks agreement
```

Dependencies: Python 3 with numpy, pandas, statsmodels (and scipy, which statsmodels installs), matplotlib. R ≥ 4.0 with base R plus **jsonlite** (`install.packages("jsonlite")`).

## Folder contents

| Path | What it is |
|---|---|
| `data/generate_data.py` | Seeded generator; documents the data-generating process and the true parameters |
| `data/randomized_pilot.csv` | The stratified randomized pilot (with a duplicate row, an outage week and leavers) |
| `data/volunteer_pilot.csv` | The same organization running an opt-in pilot |
| `data/ground_truth.csv` | Each employee's hidden traits and true effects |
| `data/rerandomizations.csv` | 500 alternative stratified assignments for the simulation check |
| `data/data_dictionary.md` | Columns, types, units, known data issues |
| `python/analysis.py` | Step 0 cleaning → Step 1 naive → Step 2 correct → Step 3 truth check → Step 4 decision |
| `r/analysis.R` | The same steps and numbers in base R |
| `outputs/results_python_full_precision.json` | Unrounded Python values (R compares against these) |
| `outputs/results.json` | Every number the article may quote: name, value, rounding, meaning, `python` and `r` |
| `outputs/fig_1_estimates.png` | Estimates with 95% CIs vs the true effect |
| `outputs/fig_2_simulation.png` | 500 re-randomizations: simple difference vs ANCOVA |
| `templates/experiment_plan_template.md` | One-page internal experiment plan (with worked example) |
| `templates/stratified_randomize.py` | Stratified randomization helper for your own roster |
| `templates/example_assignment.csv` | Example output of the helper |

## What to look for in the output

- **Step 1 (naive):** the volunteer comparison says the tool adds about **+7.0 cases/week**; the volunteers' true gain is **+2.3**. Volunteers were already 3.6 cases/week ahead before the tool existed. Adjusting for the baseline still leaves +4.3, because motivation is not in the data.
- **Step 2 (correct):** randomized, simple difference **+1.5 [0.5, 2.5]**; ANCOVA **+1.7 [1.0, 2.4]**. The ANCOVA standard error is **28% smaller**, equivalent to running the unadjusted analysis on about **2,250 employees instead of 1,183**. Uptake was 80%, so the effect among users is about +2.1. Quality: +0.1 [−0.2, +0.4], inside the −1 point guardrail.
- **Pitfalls:** adjusting for AI hours (a post-treatment variable) flips the answer to **−1.6**; comparing users with non-users gives +3.0, inflated by who chooses to use the tool.
- **Step 3 (truth check):** the true effect of access is +2.0. Across 500 re-randomizations both randomized estimators are unbiased (bias ≈ 0.0); ANCOVA's spread is 27% smaller; 95% CIs cover the truth 96.0% (simple) and 94.8% (ANCOVA) of the time.
- **Step 4 (decision):** under the pre-registered rule (ITT CI above 0 and quality non-inferior), the decision is **roll out**.

One realized pilot is one draw: this one landed at +1.7 against a truth of +2.0, well inside its CI. That is exactly what a 95% CI promises, and it is why the simulation check matters.

## Messy cases, handled explicitly (same rules in both scripts)

1. Exact duplicate rows are dropped (1 row).
2. A team-week where every recorded value is 0 is a data-feed outage and is set to missing (36 employee-weeks), not counted as zero work.
3. Leavers keep the weeks they worked; anyone with fewer than 4 valid pilot weeks is excluded (17 people). Attrition is reported by arm (1.5% vs 1.3%), so you can see it is not related to the tool.

## Connect to the article

The article "Running a Fair A/B Test Inside Your Company" argues that a fair internal experiment is designed before it is analyzed. This package shows each claim with numbers you can reproduce: the volunteer pilot overstates the gain about threefold; a stratified lottery plus a baseline-adjusted analysis recovers the true effect with a tighter interval at the same headcount; and the experiment plan template covers the "fair" part (telling employees, a waitlist, no individual consequences, privacy, HR or works-council review). It is the causal design that A02 said time-savings claims need; when you cannot randomize, see A01 (difference-in-differences). When colleagues share the tool, randomize teams instead (B02).
