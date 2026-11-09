# E03 AI Tutors in Practice: What Interaction Data Reveals

Companion code for the *Measure Digital Transformation* newsletter (Dr. Eddie Lin). Method: **sequence models, from Markov chains toward neural models**, evaluated on held-out learners, with matching Python and R code.

> All data in this folder are synthetic, generated to illustrate the method. They do not come from Meta, any client, or any real organization. The goal is to show how a business question, a measurement strategy, the statistics, and the code connect end to end.

## The business question

*How do learners actually use AI tutors, and does it help?* Most tutor dashboards count usage: messages, minutes, sessions. This package shows why those counts say almost nothing about learning, and how the **order** of what learners do with the tutor does.

## The measurement strategy

1. **Code the logs as actions.** Each tutor event becomes one token: **T** attempts the problem, **H** asks for a hint or explanation, **A** asks for the full answer, **C** copies/pastes the AI answer, **V** verifies own work, **Q** quits the session. Every session starts in an implicit START state.
2. **Define the outcome before modeling.** A learner "gained" if post-test minus pre-test is 10 points or more. Learners with no post-test are reported and kept out of the gain analysis, not imputed.
3. **Evaluate on held-out learners.** The generator writes a `split` column (70% train, 30% test, by learner). Every model is fit on training learners and scored on test learners, identically in Python and R.
4. **Climb a ladder of models**, and only keep a richer model if it beats the simpler one on held-out learners.

## The method in plain words

- **First-order Markov chain:** a table of P(next action | current action), estimated by counting transitions (plus 0.5 in every cell so no probability is zero). Fit one chain on training learners who gained and one on those who did not.
- **Log-score:** how much more a learner's whole sequence looks like the "gained" chain than the "did not gain" chain:
  `log-score(sequence) = sum over steps of log [ P_gain(next | current) / P_nogain(next | current) ]`
- **The ladder** (held-out AUC: the chance that a randomly chosen learner who gained scores above one who did not; 0.50 = coin flip):
  - (a) usage volume: logistic regression on messages, minutes, sessions;
  - (b) bag of actions: logistic regression on the share of each action type, order ignored;
  - (c) first-order Markov log-score: order matters;
  - (d) second-order Markov log-score: the "richer model" (two steps of memory), in both languages;
  - (e) **Python only:** a small neural network (scikit-learn MLP, 16 hidden units) on 1-, 2- and 3-gram shares, a light stand-in for a neural sequence model. It is not reproduced in R and is labeled `python_only` in `results.json`. A recurrent network (GRU/LSTM) or transformer would need PyTorch or similar, which this package deliberately avoids.
- **Uncertainty:** 95% bootstrap intervals (1,000 resamples of learners). A tiny Park-Miller random number generator, written identically in both scripts, makes the intervals match exactly.
- **Ground-truth check:** the generator knows each learner's true usage style and its true transition matrix. We compare (i) chains estimated with the true labels and (ii) chains recovered **without labels** by a 3-style mixture of Markov chains (EM algorithm), against the truth.

## Data and folders

```
data/generate_data.py             seeded generator (seed 20261106); documents every true parameter
data/tutor_events.csv             96,866 logged actions (learner_id, session_id, step, action, timestamp)
data/learners.csv                 1,200 learners: team, business unit, pre/post scores, train/test split
data/ground_truth.csv             true style per learner (for the check only)
data/ground_truth_transitions.csv true transition matrices, long form
data/ground_truth_params.json     all true parameters, incl. realized style shares
data/data_dictionary.md           columns, types, meanings, units
python/analysis.py                Steps 1-4 + figures; writes outputs/results_python.csv
python/check_agreement.py         confirms Python and R agree; builds outputs/results.json
r/analysis.R                      the same steps and numbers in base R; writes outputs/results_r.csv
outputs/results.json              every quotable number (python and r values, rounding, meaning, agree flag)
outputs/fig1_order_beats_volume_auc.png
outputs/fig2_two_loops_by_gain_group.png
templates/ai_tutor_interaction_scorecard.md
templates/tutor_event_log_template.csv   (+ tutor_labels_template.csv)
templates/tutor_sequence_tools.py / .R   transition matrix, scorecard metrics, log-score for any such log
```

Scenario: a company runs an 8-week AI upskilling course ("data skills for analysts") with an AI tutor: 1,200 learners in 60 teams (each with its own instructor) across 3 business units, a common pre-test and post-test (0-100). Each learner has a latent usage style (about 40% scaffolded, 35% answer-seeking, 25% light/mixed), and each style has its own transition matrix. Lower prior skill makes answer-seeking more likely, so style and prior skill are confounded. Messy on purpose: very short logs, sessions cut by logouts, about 2% mis-coded actions, lower-case and unreadable codes, instructor effects, and 18 learners with no post-test.

## How to run (about 15 seconds)

Requirements: Python 3 with numpy, pandas, scipy, statsmodels, scikit-learn, matplotlib; R 4.x (base R only, no packages).

```bash
python data/generate_data.py      # optional: the csv files are already included
python python/analysis.py
Rscript r/analysis.R
python python/check_agreement.py  # prints AGREEMENT OK when every number matches
```

Each script finds its files relative to its own location, so it runs from any folder.

To use the template on your own logs: `python templates/tutor_sequence_tools.py your_events.csv [your_labels.csv]` (or `Rscript templates/tutor_sequence_tools.R ...`). The labels file has `learner_id,label` with 1 = good outcome.

## What to look for in the output

1. **Naive view (Step 1):** messages and gains are uncorrelated (r = -0.03), and learners who gained sent about as many messages as those who did not (70.0 vs 72.3). A usage dashboard cannot tell them apart.
2. **The ladder (Step 2, held-out AUC):** usage volume 0.55 (95% interval 0.49 to 0.61, so barely above a coin flip); bag of actions 0.67; first-order Markov 0.78 (0.74 to 0.83). Adding order adds 0.12 AUC over the bag of actions (0.07 to 0.16). The richer second-order chain scores 0.79, a gain of 0.00 (0.00 to 0.01) over first-order, and the Python-only neural net on n-grams scores 0.72. Richer models add nothing here because the true process is first-order.
3. **What the score means in points:** held-out learners in the top quartile of the Markov log-score gained 11.5 points more than those in the bottom quartile (16.8 vs 5.4; 95% interval 9.5 to 13.5).
4. **The two loops:** P(copy | full answer) is 0.49 for learners who did not gain vs 0.29 for those who did; P(attempt | hint) is 0.48 vs 0.25. Learners who gained turn hints and answers into new attempts; the rest loop hint, answer, copy.
5. **Confounding:** the answer-to-copy rate is 0.49 in the lowest pre-test quartile vs 0.27 in the highest. Usage style tracks prior skill, so these patterns are not causes.
6. **Ground truth (Step 3):** transition probabilities are recovered with a mean absolute error of 0.007, both with the true labels and with the unlabeled 3-style mixture, which puts 95% of learners in their true style (estimated shares 41/38/21% vs true 40/37/23%).
7. **Python vs R:** `check_agreement.py` should print `AGREEMENT OK` (48 numbers and all their intervals). Every number is in `outputs/results.json` with both values.

Exact numbers live in `outputs/results.json`.

## Design choices you should know about

- **Order, not mix, by design.** The true transition matrices give scaffolded and answer-seeking learners broadly similar action mixes (both ask for hints and answers, both attempt and verify); most of the difference is in what follows what. That is why the bag of actions does only moderately well. In real data, styles may also differ in mix, and then the bag of actions will do better; that is exactly why you run the ladder on held-out learners instead of assuming.
- **Volume was built to be nearly uninformative**: scaffolded and answer-seeking learners have the same number of sessions on average, and answer-seekers send about 10% more messages.
- **Smoothing:** 0.5 is added to every transition count. With very sparse second-order contexts, the smoothing choice matters; keep it fixed and report it.
- **Log-score is a sum over steps**, so longer logs get more extreme scores. That is fine for ranking (AUC); if you need calibrated probabilities, fit a logistic regression on the score in a separate calibration sample.
- **Cleaning:** codes are trimmed and upper-cased; unreadable codes are dropped (294 here) and the transition skips over them; sessions without Q simply stop (no END transition).
- **Mixture of chains (EM)** starts from two transparent rules (hint-to-attempt rate at least 0.5 = scaffolded-like; answer-to-copy rate at least 0.4 = answer-seeking-like; otherwise light) and runs exactly 200 iterations, so Python and R take identical paths.
- **Quartile groups are fixed in the bootstrap** (we resample learners, not re-rank them).

## Caveats

- **Predictive and descriptive, not causal.** Prior skill and motivation drive both usage style and gains. To test whether a tutor design (for example, hints before answers, an "attempt first" prompt) improves learning, run an experiment (see edition B01).
- **The action coding is a measurement decision.** What counts as a hint vs. an answer must be defined and checked for rater agreement (see edition C01). Here 2% of codes are deliberately wrong; real logs may be worse.
- **Privacy and consent.** Log learner interactions only with a clear purpose and notice; report cohorts, not individuals.
- **Evaluate on held-out learners**, never on the learners used to fit the chains. Scoring the training learners would flatter every model.
- The synthetic process is exactly first-order, so a first-order chain is the right model by construction. With long sequences, many action types, or long-range dependencies (for example, what a learner did last week), recurrent or transformer models can earn their place, but only if they beat the Markov baseline on held-out learners.

## Connect to the article

The article asks how learners really use AI tutors, and whether it helps. This package shows why a usage dashboard can report heavy, enthusiastic use while learning stalls: two groups send the same number of messages, but one turns hints into new attempts and the other loops hint, answer, copy. An interpretable first-order Markov chain finds that difference on held-out learners; a richer model adds nothing here. Use `templates/ai_tutor_interaction_scorecard.md` to report the learning-rich interaction metrics for every cohort, and test tutor designs with an experiment before claiming they cause learning.
