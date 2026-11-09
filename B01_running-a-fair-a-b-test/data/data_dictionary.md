# Data dictionary (all data SYNTHETIC)

All files are produced by `data/generate_data.py` (seed 20261108). One row = one employee unless noted.

## randomized_pilot.csv (1,201 rows: 1,200 employees + 1 duplicate row)

| Column | Type | Meaning | Units |
|---|---|---|---|
| employee_id | string | Employee identifier (E0001–E1200) | – |
| site | string | Site A, B, C or D | – |
| team | string | Team code (site letter + number, 40 teams) | – |
| stratum | string | Randomization block (= site) | – |
| assigned | int 0/1 | 1 = randomized to get the AI drafting assistant; 0 = waitlist control | – |
| used_tool | int 0/1 | 1 = actually used the tool during the pilot (only possible if assigned) | – |
| cases_pre_w1 … cases_pre_w4 | int | Cases resolved in each of the 4 baseline weeks | cases/week |
| cases_post_w1 … cases_post_w8 | int, may be blank | Cases resolved in each of the 8 pilot weeks; blank after an employee leaves | cases/week |
| ai_hours_per_week | float | Average hours per week using the tool, from tool logs (0 for non-users). **Post-treatment variable – do not adjust for it** | hours/week |
| quality_pre | float | QA audit quality score, baseline | points (0–100) |
| quality_post | float, may be blank | QA audit quality score, pilot period; blank if the employee left in week 4 or earlier | points (0–100) |
| left_week | int, may be blank | Pilot week in which the employee left; weeks from this one on are blank | week 1–8 |

## volunteer_pilot.csv (1,200 rows)

Same employees, the pilot run as an opt-in program. Same columns as above except: no `stratum`/`assigned`; instead

| Column | Type | Meaning |
|---|---|---|
| volunteered | int 0/1 | 1 = opted in and was given the tool |
| used_tool | int 0/1 | 1 = volunteered and actually used it |

## ground_truth.csv (1,200 rows) – known only because the data are simulated

| Column | Type | Meaning | Units |
|---|---|---|---|
| employee_id | string | Employee identifier | – |
| ability | float | Latent stable performance (part of the baseline level) | cases/week |
| motivation | float | Latent motivation (standardized); drives opting in, using the tool and improving anyway | SD units |
| complier | int 0/1 | 1 = would use the tool if given access (exactly 80% of employees) | – |
| volunteer_probability | float | True probability of opting in to the volunteer pilot | 0–1 |
| true_effect_if_used | float | True effect of using the tool = 2.5 | cases/week |
| true_quality_effect_if_used | float | True effect on quality = 0 | points |

## rerandomizations.csv (1,200 rows × 501 columns)

`employee_id` plus `r001` … `r500`: 500 alternative assignments (0/1) drawn with the same stratified procedure. Used to check bias, standard errors and 95% CI coverage.

## Known data issues (deliberate, handled identically in both scripts)

1. **Duplicate row:** employee E0457 appears twice in `randomized_pilot.csv` (export glitch) → exact duplicates dropped.
2. **Data-feed outage:** team C04 shows 0 cases for every member in pilot week 5 (team B07 in week 3 of the volunteer pilot) → any team-week where every recorded value is 0 is set to missing, not treated as zero work.
3. **Leavers:** about 3% of employees leave mid-pilot → the outcome is the mean of valid pilot weeks; employees with fewer than 4 valid weeks are excluded, and attrition is reported by arm.

## True parameters (summary)

- Baseline mean ≈ 40 cases/week; weekly noise SD 9; pre/post correlation ≈ 0.7.
- True effect of using the tool: +2.5 cases/week (≈ 6%). Uptake 80%. True effect of giving access (ITT): +2.0.
- True effect on quality: 0.
- Volunteers: more able and more motivated (opt-in probability rises with both).
