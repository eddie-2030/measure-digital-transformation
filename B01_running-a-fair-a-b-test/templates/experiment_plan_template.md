# Internal Experiment Plan (one page)

*Fill this in and get it signed off BEFORE anyone is randomized. Date-stamp it and store it where the analyst cannot quietly edit it later. The worked example in brackets comes from the synthetic B01 pilot.*

| Item | Your plan |
|---|---|
| **Pilot name / owner** | [AI drafting assistant pilot; owner: Support Ops + People Analytics] |
| **Decision this pilot informs** | [Give all 1,200 analysts access, or not] |
| **Question** | [Does giving access to the assistant increase cases resolved per analyst, without lowering quality?] |
| **Primary outcome (one only)** | [Cases resolved per employee per week, mean of the 8 pilot weeks] |
| **Guardrail(s)** | [QA quality score must not fall by more than 1.0 point (non-inferiority margin)] |
| **Secondary outcomes (descriptive only)** | [Tool usage, AI hours/week, employee survey on workload] |
| **Unit of randomization** | [Individual employee. If colleagues share drafts or prompts, randomize teams instead, see B02] |
| **Strata (blocks)** | [Site (4); optionally baseline-performance tercile within site] |
| **Allocation** | [50/50 within each stratum; seed written here: ______ ; helper: `templates/stratified_randomize.py`] |
| **Baseline (pre-period)** | [4 weeks of the primary outcome and the guardrail before launch] |
| **Duration** | [8 weeks; no early stopping, no interim "peeks" that change the plan] |
| **Sample size / precision** | [~1,200 employees. Expected 95% CI half-width about 0.7 cases/week with ANCOVA (about 1.0 without)] |
| **Analysis model (primary)** | [ANCOVA, intention-to-treat: `outcome_post = a + b × assigned + c × outcome_pre + stratum effects + error`; b with a 95% CI] |
| **What we will NOT adjust for** | [Anything measured after launch: AI hours, usage, satisfaction with the tool] |
| **Noncompliance** | [Analyze as randomized (ITT); report uptake; effect among users = ITT / uptake, labeled as such] |
| **Missing data rules** | [Duplicates removed; team-weeks with an outage (all zeros) set to missing; need ≥ 4 valid pilot weeks; attrition reported by arm] |
| **Decision rule** | [Roll out if the ITT 95% CI lower bound > 0 AND the quality CI lower bound > −1.0. Otherwise iterate or stop] |
| **Employee communication** | [Everyone is told a pilot is running, why, for how long, and how groups were chosen (a lottery within each site)] |
| **Fairness: waitlist** | [Control group gets access when the pilot ends, whatever the result] |
| **No individual consequences** | [Pilot data are not used for any individual performance rating or decision] |
| **Privacy** | [Results reported only in aggregate (minimum group size, e.g. 10); individual-level data access limited to the analyst; retention date: ______] |
| **Review** | [HR / works council / employee representatives reviewed on: ______ (where required). This plan is good practice, not legal advice] |
| **Sign-off** | [Business owner ______ · Analyst ______ · HR ______ · Date ______] |

**After the pilot:** report the primary result exactly as specified above first, then anything exploratory, clearly labeled as such.
