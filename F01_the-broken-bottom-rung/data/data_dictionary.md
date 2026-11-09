# Data dictionary (synthetic data)

All data in this folder are synthetic, generated to illustrate the method. They do not come from Meta, any client, or any real organization.

## junior_careers.csv: person-month panel (19,560 rows, 900 people)

One row per person per observed tenure month, from the hire month up to the data cut (October 2026) or month 24, whichever comes first. Rows continue after someone leaves (state `X`), so you can tell "left" apart from "not observed yet" (right truncation).

| Column | Type | Meaning | Units / values |
|---|---|---|---|
| person_id | string | Synthetic employee ID | P0001 to P0900 |
| cohort | string | Half-year hiring class | 2023H1, 2023H2, 2024H1 (pre-AI); 2024H2, 2025H1, 2025H2 (post-AI) |
| hire_month | string | Calendar month of hire | YYYY-MM |
| job_family | string | HIGH = AI-exposed operations/support analyst roles (GenAI assistant from Jul 2024); LOW = field-service coordinator roles with little AI change | HIGH, LOW |
| team_id | string | Team (nesting unit; adds a small output effect) | H01 to H06, L01 to L06 |
| selection_score | float | Hiring assessment (standardized). Post-AI cohorts are slightly more selective | z-score |
| calendar_month | string | Calendar month of the record | YYYY-MM |
| tenure_month | integer | Months since hire (month of hire = 1). This is the "tenure clock" | 1 to 24 |
| state | string | Work state that month. R = routine junior tasks; S = supervised complex ("stretch") work; I = independent complex work; X = left the company (absorbing). Messy on purpose: about 2% blank; a few non-standard codes ("SUP", "Indep", "routine", "LEFT", lower case, extra spaces); 6 rows show an active state after the person left | R, S, I, X |
| tasks_closed | float | Tasks closed that month (the "early output" metric). Blank when the person had left or the month is missing | tasks per month |
| skill_score_m18 | float | Month-18 skill assessment, repeated on every row of the person. Blank if not yet observed at month 18, already left, or not recorded (5%) | 0 to 100 |

## ground_truth.csv: known by construction (never available in real life)

| Column | Type | Meaning | Units / values |
|---|---|---|---|
| person_id | string | Links to junior_careers.csv | P0001 to P0900 |
| true_type | string | Latent trajectory type the generator used | climber, plateau, early_leaver |

True type shares (climber / plateau / early leaver): HIGH pre-AI 65/20/15, HIGH post-AI 45/40/15, LOW pre-AI 63/22/15, LOW post-AI 61/24/15. See `generate_data.py` for every parameter.
