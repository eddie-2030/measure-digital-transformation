# Junior Development Scorecard

A one-page quarterly view of whether new hires are still climbing, not only producing. Compare cohorts on the **tenure clock** (months since hire), never on the calendar. Fill it from monthly person-state records (`person_state_records_template.csv`) with `tenure_state_summary.py` or `tenure_state_summary.R`.

**Job family:** ____________  **Quarter:** ________  **Data cut (last month observed):** ________  **Owner:** ____________

## 1. State definitions (agree on these before you measure)

| State | Definition in our work data (task logs, assignment level, review sign-off) | Source system |
|---|---|---|
| R: routine | | |
| S: supervised complex (stretch) | | |
| I: independent complex | | |
| X: left | | |

## 2. Development metrics by cohort (same tenure month for every cohort)

Only include a cohort in a month-N column if **everyone** in it has been employed long enough to reach month N, or left before it. Count leavers as X; never drop them.

| Cohort | Hires | Early output, tasks/month, months 1-6 | Months in S, months 1-12 | % in I at month 12 | % in I at month 18 | % left by month 12 | Month-18 skill score |
|---|---|---|---|---|---|---|---|
| | | | | | | | |
| | | | | | | | |
| | | | | | | | |

Guide: early output is a **speed** metric. Months in S and % in I are **development** metrics. If output rises while months in S or % in I falls, treat that as a warning sign, not a win. Read month 12 together with month 18: a cohort can look more independent at month 12 (its fastest climbers move up sooner) and still fall behind by month 18 if a larger group stays in routine work.

## 3. Trajectory mix (optional, once 18 months are observed)

| Cohort | % climbers | % plateaued | % early leavers | Method notes (distance, number of clusters) |
|---|---|---|---|---|
| | | | | |

## 4. Comparison group

Which job family saw little AI change? Did its cohorts shift too? ____________________

If both families shifted, look first at shared causes (hiring market, onboarding changes, managers) before crediting or blaming AI.

## 5. Learning-rich work protected this quarter

- [ ] Deliberate stretch assignments: number per junior per month: ____
- [ ] AI-off practice blocks on core skills: ____ hours per month
- [ ] "Review the AI" tasks used as training, with feedback from a senior: ____
- [ ] Named coach for every hire in months 1-12

## 6. Caveats to state on every report

- Cohort differences are descriptive, not causal: hiring, onboarding and markets change too.
- Recent cohorts are observed for fewer months (right truncation); compare them only at months they have reached.
- Results depend on how states are coded from real work data, and trajectory types depend on the distance and number of clusters chosen.
