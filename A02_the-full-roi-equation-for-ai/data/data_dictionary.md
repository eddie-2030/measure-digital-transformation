# Data dictionary — A02 The Full ROI Equation for AI Agents

All data in this folder are synthetic, generated to illustrate the method. They do not come from Meta, any client, or any real organization. The goal is to show how a business question, a measurement strategy, the statistics, and the code connect end to end.

All files are written by `data/generate_data.py` (seed 20261107; Monte Carlo uniforms use seed 20261108).

## helpdesk_tickets.csv — ticket log, weeks 1–26 (one row per ticket, plus a few duplicates)

| column | type | meaning | units |
|---|---|---|---|
| ticket_id | string | Ticket identifier (`T000001`…). About 1.2% of rows are exact duplicates from a double export; drop them. | — |
| week | integer | Week of the pilot, 1–26 | week |
| created_date | date (YYYY-MM-DD) | Date the ticket was opened (weekdays only; week 1 starts 2026-03-02) | date |
| queue | string | `access` (passwords, permissions), `it_software` (apps, devices), `hr_policy` (leave, benefits, policy questions). Difficulty rises in that order. | — |
| routed_to_agent | 0/1 | 1 = the ticket was routed to the AI agent first | flag |
| outcome | string | `human_only` (never routed), `resolved_by_agent`, `escalated` (agent handed it to a human), `reopened` (agent closed it, employee reopened it, a human fixed it = rework). Blank for ~2% of routed tickets (logging gap). | — |
| agent_cost_usd | float | Compute/token cost of the agent's work on this ticket. Blank when not routed. | USD |
| human_minutes | float | Minutes a human spent on the ticket. 0 for agent-resolved tickets; includes the handoff/rework time for escalated and reopened tickets. Blank when the outcome is missing and for ~1% of human-only tickets. | minutes |
| qa_sampled | 0/1 | 1 = this agent-resolved ticket was in the 12% QA review sample | flag |
| qa_error_found | 0/1 | 1 = the QA reviewer found an error in the agent's answer. Blank when not sampled. | flag |

## cost_assumptions.csv — inputs that are not in the logs (ranges from finance/ops)

| column | type | meaning |
|---|---|---|
| parameter | string | Input name (licence, build, monitoring, loaded labour cost, remediation cost per error, review share, review minutes, steady-state routing share for months 7–12) |
| low / most_likely / high | float | The range a finance or ops owner gives. Used as a triangular distribution; `most_likely` is the point estimate. |
| unit | string | Units of the parameter |
| distribution | string | `fixed` or `triangular` |
| source | string | Who would normally supply the number |
| notes | string | What is included and why the range is skewed |

## ground_truth.csv — the true values used to generate the data

| column | type | meaning |
|---|---|---|
| parameter | string | True rates per queue (`p_escalated_*`, `p_reopened_*`, `p_resolved_*`, `p_error_*`), base minutes, extra minutes, run costs, true cost inputs, the true steady-state routing share, expected agent tickets in 12 months, and the true 12-month benefit, variable cost, fixed cost, **net value** and **ROI** |
| true_value | float | Value of the parameter (USD, share, minutes or tickets) |

In real life you never see this file. It exists so Step 3 can check that the method recovers the truth.

## mc_uniforms.csv — shared random numbers for the Monte Carlo

10,000 rows × 25 columns of uniform(0, 1) numbers, one column per uncertain input (`u_esc_<queue>`, `u_rw_<queue>`, `u_err_<queue>`, `u_minutes_1..5` for the five regression coefficients, `u_volume`, `u_run_<queue>`, and one per triangular cost input). Python and R each convert them into draws with the same inverse-CDF formulas, so both languages produce identical simulation results.
