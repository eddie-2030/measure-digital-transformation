# A01 Adoption Is Not Impact: How to Prove a GenAI Rollout Actually Worked

Companion code for the *Measure Digital Transformation* edition "Adoption Is Not Impact".

> All data in this folder are synthetic, generated to illustrate the method. They do not come from Meta, any client, or any real organization. The goal is to show how a business question, a measurement strategy, the statistics, and the code connect end to end.

## The business question

"We rolled out an AI assistant to our support teams and productivity went up. How much of that rise did the assistant cause, and is it worth scaling?"

## The measurement strategy

Climb the ladder, and pair every speed metric with a quality metric:

| Rung | What it shows | Metric in this package | Evidence it needs |
|---|---|---|---|
| 1 Usage | People use the tool | Weekly active share of agents | Descriptive |
| 2 Behavior | Work is done differently | (template only) draft-use rate, QA compliance | Descriptive vs. baseline |
| 3 Work outcomes | The work itself improved | **Tickets resolved per agent-hour**, paired with **reopen rate within 7 days** | **A comparison group** |
| 4 Business value | It pays off | (template only) cost per resolved ticket, CSAT | Comparison group x finance |

Usage climbs to about 80% of agents, but usage is not impact. Impact sits at rung 3, and rung 3 needs a comparison group: teams that have not received the assistant yet.

## The scenario (synthetic)

60 customer-support teams (6 regions x 10 teams, 7-16 agents each) are observed weekly for 24 weeks. The assistant launches in three waves of 20 teams: week 7, week 13, week 19. Within each region, the launch order was set by trainer availability, not by performance. Every team improves by about +0.05 tickets per agent-hour per week (seasonality and ordinary learning), with or without the assistant. The assistant's true effect builds over the first three weeks (20%, 50%, 80% of the full effect) and then holds at about +0.38 tickets per agent-hour (slightly more for teams with more novice agents). Averaged over a team's first six weeks, the true effect is about +0.29. The true effect on the reopen rate is zero. The full data-generating process is documented at the top of `data/generate_data.py`.

## The method in plain words

1. **Naive (Step 1): before-after.** Wave-1 teams' average in weeks 7-12 minus weeks 1-6. It credits the assistant with everything that changed, including the trend that lifted every team.
2. **Correct (Step 2): difference-in-differences (DiD).** Subtract the change that comparison teams (waves 2 and 3, not launched yet) went through over the same weeks. What is left is the assistant's effect, as long as both groups would have moved in parallel without it. Standard errors are clustered by team, because a team's weeks are not independent observations.
3. **Check (Step 3).**
   - *Parallel pre-trends:* before launch, did wave-1 and comparison teams move in parallel?
   - *Staggered rollout:* with all three waves, the textbook regression is two-way fixed effects (TWFE: team effects, week effects, one "has the assistant" coefficient). When the effect grows over the first weeks, TWFE quietly uses teams that already have the assistant as controls and gets the answer wrong. The fix is to estimate one clean DiD per launch wave and week (a group-time ATT, in the style of Callaway & Sant'Anna), always against teams not yet launched, and then average them.
   - *Truth:* every estimate is compared with the true effect the data were simulated with, averaged over exactly the same team-weeks.
   - *Monte Carlo:* the Python script reruns the whole design on 30 seeds to show the results are not a lucky draw.
4. **Decide (Step 4).** Translate the effect into extra tickets per 1,000 agent-hours, and check that quality held.

## How to run (about 15 seconds)

From this folder:

```bash
python3 data/generate_data.py    # optional: regenerates data/*.csv (seed 20261105)
Rscript r/analysis.R             # base R; writes outputs/r_headline.csv
python3 python/analysis.py       # writes outputs/results.json, figures, article chart data
```

Run R before Python so `results.json` records both languages' numbers (`python` and `r` keys; `python_r_all_headline_match` must be `true`). Python needs numpy, pandas, statsmodels, matplotlib (and scipy, which statsmodels installs, for one chi-square p-value). R needs base R only, no packages: the clustered standard errors and the group-time estimator are written out by hand, so you can read every step.

## What is in the folder

| Path | What it is |
|---|---|
| `data/generate_data.py` | Seeded generator; the data-generating process and true parameters are documented at the top |
| `data/support_team_weeks.csv` | 1,421 team-week rows, with a few messy rows on purpose |
| `data/true_parameters.csv` | Ground truth: trend, ramp, every team's established effect, design-level true ATTs |
| `data/data_dictionary.md` | Columns, types, meanings, units, messy cases |
| `python/analysis.py` | Steps 0-4, Monte Carlo, figures, `results.json`, article chart data |
| `r/analysis.R` | The same steps and numbers in base R |
| `outputs/results.json` | Every number the article may quote, with rounding, meaning, and Python and R values |
| `outputs/article_chart_data.csv` | Weekly means, wave 1 vs. comparison teams, weeks 1-12 (for the article's chart) |
| `outputs/fig_1_pilot_vs_comparison.png` | Wave 1 vs. comparison teams: before-after gain vs. DiD effect |
| `outputs/fig_2_event_study_vs_truth.png` | Effect by weeks since launch, group-time estimator vs. truth vs. naive TWFE |
| `templates/ladder_scorecard.csv` | Ladder Scorecard: 4 rungs x a primary (volume or speed) metric and its paired quality metric, data source, targets, evidence needed |
| `templates/ladder_scorecard.pdf` | One-page printable Ladder Scorecard for leaders: one metric and its quality pair per rung, the evidence each rung needs, and three questions to ask your analysts |

## What to look for in the output

- **The before-after gain is +0.60 tickets per agent-hour** (2.60 to 3.20, team-clustered SE 0.017). It is precise, and precisely wrong.
- **Comparison teams gained +0.30 over the same weeks without the assistant.** Half the apparent gain would have happened anyway.
- **The 2x2 DiD effect is +0.30** (team-clustered SE 0.020; 95% CI 0.26 to 0.34). The interval excludes the naive +0.60. The true average effect over those weeks is +0.29. That is about +12% over the wave-1 baseline.
- **Pre-trends are consistent with parallel trends.** Before launch, wave-1 and comparison slopes differ by -0.007 per week (SE 0.008); placebo estimates for the six pre-launch weeks are all within 0.03 of zero (joint test of the leads at weeks -6 to -2, p = 0.65). A pre-trend test can only fail to find a difference; it cannot prove there is none.
- **Naive TWFE on the staggered rollout gives +0.28** (95% CI 0.25 to 0.30), against a true +0.32 over the cells the group-time estimator can identify (+0.33 over all treated team-weeks). It understates the effect either way, and its narrow interval excludes both.
- **The group-time estimator gives +0.32** (95% CI 0.29 to 0.36), true +0.32. Its event-time profile follows the true ramp: about +0.09 in launch week, +0.20 a week later and +0.34 in the third week (true +0.08, +0.19, +0.31), then about +0.38 once established (true +0.38).
- **This is not a lucky seed.** Across 30 simulated rollouts, TWFE is below the truth every time (by -0.05 on average, -0.03 to -0.08), while the group-time estimator is off by +0.001 on average (SD 0.02). The naive gain averages +0.59, the 2x2 DiD +0.29.
- **Quality held.** The reopen-rate DiD is +0.09 percentage points (95% CI -0.22 to +0.40) on a 7.8% baseline. Usage reached 72% of agents in wave 1's first six weeks and 81% once established; neither number measures impact.
- **The business case halves.** The naive story claims 601 extra tickets per 1,000 agent-hours; the evidence supports 302. Once established, the effect is worth about 8,900 extra tickets a week across the 23,500 agent-hours of all 60 teams.

## Notes on the method

- **Base period.** Callaway & Sant'Anna's default compares each week with the single week before launch. Here each team's base is its average over all its pre-launch weeks. The identifying assumption (parallel trends with not-yet-treated teams) is the same, and with noisy weekly data it is noticeably more precise. One consequence: a group's placebo leads sum to about zero by construction, so the joint lead test uses event times -6 to -2.
- **Wave 3 has no effect estimate.** After week 18 every team has the assistant, so nobody is left to compare with. Wave 3 serves only as a control. The overall group-time ATT covers waves 1 and 2 up to week 18, and the truth it is checked against covers the same cells. If you keep a holdout group that never gets the tool (or gets it last, much later), you can measure later waves too.
- **No anticipation and no spillovers.** Teams do not change behavior before their own launch (the 3 early-access testers are a mild violation; e = -1 in Fig 2 is near zero), and comparison teams do not benefit from the assistant indirectly, for example through shared macros or agents moving between teams. Spillovers would shrink the DiD toward zero.
- **TWFE has no single well-defined target.** When effects build up over time, the TWFE coefficient is a weighted mix of comparisons, some of which use already-launched teams as controls (and it also uses weeks 19-24, which the group-time ATT excludes). That is why it is checked against two truths, the identified cells (+0.32) and all treated team-weeks (+0.33), and misses both. That ambiguity is the point of the comparison.
- **Standard errors.** Regressions use CR1 team-clustered SEs (G/(G-1) x (N-1)/(N-K), the Stata and statsmodels default). Group-time ATTs use team-level influence functions (group sizes treated as fixed). All CIs use the normal approximation (1.96).

## Connect to the article

The article argues that adoption is not impact: usage dashboards show that people use a GenAI tool, but only a comparison group shows what the tool changed. This package produces the article's chart data (`outputs/article_chart_data.csv`: wave-1 teams vs. comparison teams, weeks 1-12, launch at week 7) and its numbers: a +0.60 before-after gain, +0.30 that comparison teams gained anyway, and a DiD effect of +0.30, so half the apparent gain would have happened anyway. It then shows the next trap for staggered rollouts: the standard two-way fixed effects regression understates an effect that builds over time, and a group-time estimator recovers it. The Ladder Scorecard in `templates/` is the practical takeaway: one primary (volume or speed) metric and one paired quality metric for each rung (Usage, Behavior, Work outcomes, Business value), with the evidence each rung needs.
