# Data dictionary

All data in this folder are synthetic, generated to illustrate the method. They do not come from Meta, any client, or any real organization. The goal is to show how a business question, a measurement strategy, the statistics, and the code connect end to end.

## `support_team_weeks.csv`

One row per customer-support team per week (a panel). 60 teams (6 regions x 10 teams) x 24 weeks = 1,440 team-weeks by design; the extract holds 1,421 rows because some team-weeks are missing and a few are duplicated (see messy cases below). Teams are nested in regions. The AI assistant launched in three waves of 20 teams: wave 1 in week 7, wave 2 in week 13, wave 3 in week 19.

| Column | Type | Meaning | Units / values |
|---|---|---|---|
| `team_id` | string | Support team | `T01`...`T60` |
| `region` | string | Region the team belongs to, as typed in the extract (messy) | `Americas-East`, `Americas-West`, `EMEA-North`, `EMEA-South`, `APAC-North`, `APAC-South`; some rows in other case or with spaces |
| `wave` | integer | Rollout wave (set by trainer availability, not by performance) | 1, 2, 3 |
| `launch_week` | integer | Week the team got the assistant | 7, 13, 19 |
| `week` | integer | Week of observation | 1-24 |
| `week_start` | date | Monday of that week | ISO date, 2026-03-02 to 2026-08-10 |
| `n_agents` | integer | Agents on the team | 7-16 |
| `novice_share` | float | Share of agents with under 6 months tenure | proportion, 0.10-0.50 |
| `agent_hours` | float | Logged agent-hours that week | hours; blank = export failure |
| `tickets_resolved` | integer | Tickets resolved that week | count |
| `reopened_tickets` | integer | Of those, tickets reopened within 7 days | count |
| `ai_active_agents` | integer | Agents with at least one assistant session that week | count |
| `tickets_per_agent_hour` | float | `tickets_resolved / agent_hours` (rung 3 outcome) | tickets per agent-hour, 4 decimals; blank if hours blank |
| `reopen_rate` | float | `reopened_tickets / tickets_resolved` (quality) | proportion |
| `ai_weekly_active_share` | float | `ai_active_agents / n_agents` (rung 1 usage) | proportion |
| `post_launch` | integer | 1 if `week >= launch_week` | 0 / 1 |

Known messy cases (handled in both analysis scripts):
- 4 team-weeks appear twice (same `team_id` and `week`). Keep one.
- 3 team-weeks have a blank `agent_hours` (time-tracking export failure). Drop them: the outcome cannot be computed.
- 26 team-weeks are missing or unusable after cleaning (holidays, data outages, blank hours). The panel is unbalanced; the methods handle that.
- About 2% of `region` labels differ in case or have spaces (`" EMEA-NORTH"`, `"emea-north "`). Trim and upper-case.
- 3 teams show 1 active assistant user in the week before launch (early-access testers). This is real-world noise in usage logs; it does not affect the outcome analysis.

Analysis rules: the scripts recompute the outcome as `tickets_resolved / agent_hours` (rather than reading the rounded column) so Python and R use identical numbers. The unit of clustering for standard errors is the team.

## `true_parameters.csv`

The ground truth used to simulate the panel, in long format (`level`, `id`, `parameter`, `value`).

| `level` | What it holds |
|---|---|
| `global` | Common trend (+0.05 per week), established effect (0.38 by design), ramp shares (0.20, 0.50, 0.80 in launch week and the next two weeks; 1.0 after), heterogeneity slope by novice share, true effect on reopen rate (0), base rate, seed |
| `design_truth` | Average true effect on the full design (before messy rows): wave 1 weeks 7-12, waves 1-2 up to week 18, all treated team-weeks |
| `team` | Each team's established effect `tau_established` (0.38 x (1 + 0.6 x (novice_share - 0.30))) |
| `region` | Each region's level effect |

The analysis scripts rebuild the true effect of every team-week as `tau_established x ramp share` and average it over exactly the cells each estimator uses, so every estimate is checked against the right truth.
