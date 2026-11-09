# Heterogeneity analysis plan (template)

Fill this in **before** the pilot data are unblinded, then file it with the experiment record. Words in *[brackets]* are prompts; the right-hand examples come from the B06 synthetic pilot.

## 1. Decision this analysis informs

- Decision: *[e.g., launch the GenAI assistant to all support teams, or phase it]*
- Decision owner and date: *[name, role, date]*
- What would change the decision: *[e.g., a segment where the gain cannot reach the minimum worthwhile effect]*

## 2. Primary outcome and minimum worthwhile effect

| Item | Your plan | B06 example |
|---|---|---|
| Primary outcome | | Weekly quality-adjusted tasks resolved, 8-week pilot average |
| Unit of randomization | | Individual, within team (about 50% per team) |
| Clustering unit for standard errors | | Team |
| Minimum worthwhile effect (MWE) | | +1.0 task/week (about +2.4%) |
| Data-cleaning rules (fixed in advance) | | Winsorize outcome and baseline at the 0.5th and 99.5th percentiles; complete cases for tenure models |

## 3. Pre-specified moderators (keep this list short: 2 to 4)

| Moderator | Definition and cut-point | Why we expect a different effect (mechanism) | Expected direction | Source of the hypothesis |
|---|---|---|---|---|
| Work language | Team's primary language is not English | Model quality is lower in other languages | Smaller effect | Regional focus groups, vendor benchmarks |
| Tenure | Novice = tenure < 12 months | AI spreads the know-how of experienced staff to new staff | Larger effect for novices | Published field experiments |
| *[add]* | | | | |

Everything not on this list is **exploratory**. Report it as such and confirm it in a new test before acting on it.

## 4. Minimum subgroup size

- Smallest segment we will report on its own: *[e.g., 150 people across at least 10 teams]*.
- Rough check: with half of a subgroup of n people treated, the smallest effect detectable at 80% power (5% two-sided) is about 5.6 x σ / sqrt(n), where σ is the outcome SD left after adjusting for baseline (B06: σ ≈ 4 tasks). So n = 400 detects about 1.1 tasks/week, while n = 165 (non-English novices) detects only about 1.75. Add a team-clustering penalty if treatment varies by team.
- Segments below the minimum are pooled or flagged "insufficient evidence", never declared "no effect".

## 5. Analysis steps (in this order)

1. **Overall effect (ATE)** with team-clustered 95% CI. Report it, but do not decide on it alone.
2. **Pre-specified interaction model:** `Outcome = b0 + b1·AI + b2·Moderator + b3·(AI × Moderator) + controls`, with team-clustered SEs. One model holding all pre-specified moderators.
3. **Subgroup effects with 95% CIs** for each pre-specified segment and their crossings (for example language × tenure), shown on one chart.
4. **Exploratory ML step** (causal forest or DR-learner, cross-fitted by team): rank candidate moderators, check calibration on held-out data (do predicted high-benefit people actually benefit more?).
5. **Multiple-testing control** for any exploratory slicing: Holm (family-wise) or Benjamini-Hochberg (false discovery rate). State the number of tests.

## 6. Decision rules for phasing (fixed in advance)

| Evidence for a segment | Decision |
|---|---|
| Lower 95% bound ≥ MWE | **Launch** in wave 1 |
| Upper 95% bound < MWE | **Hold**: diagnose (focus groups, error review), fix or localize, then re-test |
| CI straddles the MWE | **Targeted re-test**: extend the pilot or enlarge the sample in that segment |
| Any segment with a credible harm (upper bound < 0) | Stop for that segment and escalate |

## 7. Guardrails and interpretation

- Moderators are **markers, not causes**: "non-English" stands in for model quality in that language, which is what you fix.
- Heterogeneity estimates are causal only because access was randomized. Do not read the same pattern from usage logs.
- Report every pre-specified subgroup, including the dull ones, with intervals.
- Quality and well-being guardrails: *[e.g., error rate, escalations, overtime]*. These are checked per segment too.

## 8. Sign-off

| Role | Name | Date |
|---|---|---|
| Analyst | | |
| Business owner | | |
| Reviewer (independent) | | |
