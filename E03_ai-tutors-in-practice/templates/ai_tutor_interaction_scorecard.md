# AI Tutor Interaction Metrics: Cohort Scorecard

Use one scorecard per cohort (course run, business unit, or tutor version) each month. Report HOW learners use the tutor, not only HOW MUCH. Compute the numbers with `tutor_sequence_tools.py` or `tutor_sequence_tools.R` from an event log in the format of `tutor_event_log_template.csv`.

Cohort: ____________  Period: ____________  Tutor version: ____________  Owner: ____________

## 1. Coverage and data quality (read this first)

| Metric | This period | Last period | Note |
|---|---|---|---|
| Learners with logs / enrolled | | | Low coverage = biased picture |
| Learners with pre- and post-test | | | Report missing outcomes; do not impute silently |
| Sessions ending without a quit event (%) | | | Logouts and time-outs; high values = logging problem |
| Unknown or unreadable action codes (%) | | | |
| Action-coding agreement (two raters, sample of 100 events) | | | Kappa or % agreement; recheck after any tutor update |

## 2. Volume (context only, not a success metric)

| Metric | This period | Last period |
|---|---|---|
| Messages per learner | | |
| Minutes per learner | | |
| Sessions per learner | | |

In the companion example, volume predicted learning gains barely better than a coin flip (held-out AUC 0.55).

## 3. Learning-rich interaction metrics (the core of the scorecard)

| Metric | Definition | This period | Last period | Direction |
|---|---|---|---|---|
| Hint-to-attempt rate | P(attempt next given a hint) | | | Higher is better |
| Answer-to-re-attempt rate | P(attempt next given a full answer) | | | Higher is better |
| Hint-to-answer rate | P(full answer next given a hint) | | | Lower is better |
| Answer-to-copy loop rate | P(copy/paste next given a full answer) | | | Lower is better |
| Share of actions that are full-answer requests (%) | A / all actions | | | Context |
| Sequence log-score (median, held-out) | Sum of log P_gain/P_nogain over a learner's steps, chains fit on a previous cohort | | | Higher is better |

In the companion example, learners who gained 10+ points had a hint-to-attempt rate of 0.48 vs 0.25 and an answer-to-copy rate of 0.29 vs 0.49.

## 4. Outcomes

| Metric | This period | Last period |
|---|---|---|
| Mean gain, post minus pre (points) | | |
| Learners gaining 10+ points (%) | | |
| Gain gap, top vs bottom quartile of the sequence log-score (points) | | |

## 5. Interpretation guardrails

- Sequence patterns are **predictive and descriptive, not causal**. Prior skill and motivation shape both how people use the tutor and how much they learn (in the example, the answer-to-copy rate is 0.49 in the lowest pre-test quartile vs 0.27 in the highest). To claim that a tutor design causes learning, run an experiment (for example, "hints before answers" vs. the current tutor, randomized by team).
- The action codes are a **measurement decision**. Write down what counts as a hint vs. an answer, and check rater agreement before trusting trends.
- Evaluate any predictive model on **held-out learners** (or a later cohort), never on the learners used to fit it. Start with the first-order Markov chain; adopt a richer model (second-order, RNN, transformer) only if it beats that baseline on held-out data.
- **Privacy and consent**: tell learners what is logged and why; report cohorts, not individuals, unless there is a clear and agreed purpose.

## 6. Decisions this period

| Observation | Decision | Owner | Test or follow-up date |
|---|---|---|---|
| | | | |
