# Data dictionary: `ai_pilot_heterogeneity.csv`

One row per employee in a randomized pilot of a GenAI assistant (2,450 employees, 120 teams, 4 regions). Written by `data/generate_data.py` (seed 20261153).

All data in this folder are synthetic, generated to illustrate the method. They do not come from Meta, any client, or any real organization. The goal is to show how a business question, a measurement strategy, the statistics, and the code connect end to end.

| Column | Type | Meaning | Units / values |
|---|---|---|---|
| `employee_id` | string | Employee identifier | `E00001` to `E02450` |
| `team_id` | string | Team identifier. Randomization happened within team, and standard errors are clustered on it | `T001` to `T120` (team size 10 to 30) |
| `region` | string | Region of the team | Americas, EMEA, APAC, LATAM |
| `work_language` | string | Primary working language of the team (team-level). `non_english` in the scripts = anything except English | English, Spanish, Portuguese, German, French, Japanese |
| `role_family` | string | Job family. **Noise covariate: no true effect modification** | Tier-1 support, Tier-2 support, Operations analyst, Specialist |
| `shift` | string | Usual shift. **Noise covariate** | Day, Evening, Night |
| `remote` | integer | Works remotely. **Noise covariate** | 0 = no, 1 = yes |
| `tenure_months` | float | Tenure at pilot start. Missing for about 5% of employees (missing completely at random). `novice` in the scripts = tenure < 12 months | months (0.5 to 240); blank = missing |
| `ai_access` | integer | Randomly assigned access to the AI assistant (about half of each team) | 0 = control, 1 = access |
| `baseline_tasks` | float | Pre-pilot weekly quality-adjusted tasks resolved (8-week average) | tasks per week |
| `outcome_tasks` | float | Pilot-period weekly quality-adjusted tasks resolved (8-week average). Contains 10 data-entry outliers (4-week totals keyed in as weekly averages); the scripts winsorize at the 0.5th and 99.5th percentiles | tasks per week |
| `true_effect` | float | **Ground truth, only available because the data are synthetic.** This employee's true effect of AI access on `outcome_tasks`. Never available in a real pilot; used only in Step 3 to grade the methods | tasks per week |

## True data-generating values

- True individual effect: 3.4 + 2.2 x novice - 3.6 x non-English + N(0, 0.6) tasks per week.
- Expected effect by segment: English and experienced +3.4, English and novice +5.6, non-English and experienced -0.2, non-English and novice +2.0.
- True interaction coefficients: AI x NonEnglish = -3.6, AI x Novice = +2.2. Region, role family, shift and remote: 0.
- Control-group mean outcome is about 41.5 tasks per week, so +3.0 tasks is about +7%.
