"""
check_agreement.py -- confirms that python/analysis.py and r/analysis.R give the same numbers,
then builds outputs/results.json (every number the article may quote, with both values).

Run after both analyses:  python python/check_agreement.py
Prints AGREEMENT OK when every number matches at its stated rounding.
"""
import json
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")
DATA = os.path.join(ROOT, "data")

# One-line meaning for every number. HEADLINE marks the numbers the article is most likely to quote.
MEANING = {
    "n_learners": "Synthetic learners in the course (60 teams, 3 business units).",
    "n_events": "Logged tutor actions (rows in tutor_events.csv), before cleaning.",
    "n_unreadable_tokens_dropped": "Unreadable action codes ('?') dropped during cleaning.",
    "n_missing_post_test": "Learners with no post-test: kept in the logs and the style mixture, excluded from gain analyses (not imputed).",
    "n_short_logs_under_5_actions": "Learners with fewer than 5 logged actions (very short logs, mostly early drop-outs).",
    "n_train_with_post": "Training learners with a post-test (used to fit every model).",
    "n_test_with_post": "Held-out learners with a post-test (used only to evaluate).",
    "share_gained": "Percent of learners (with a post-test) who gained 10+ points.",
    "r_messages_gain": "Naive: correlation between number of tutor messages and gain (points).",
    "r_minutes_gain": "Naive: correlation between minutes with the tutor and gain (points).",
    "mean_messages_gained": "Mean messages per learner among those who gained 10+ points.",
    "mean_messages_not_gained": "Mean messages per learner among those who did not gain 10+ points.",
    "auc_volume": "Held-out AUC, rung (a): logistic regression on messages, minutes, sessions (0.50 = coin flip).",
    "auc_bag": "Held-out AUC, rung (b): logistic regression on shares of each action type, order ignored.",
    "auc_markov1": "Held-out AUC, rung (c): first-order Markov log-score (gain chain vs no-gain chain).",
    "auc_markov2": "Held-out AUC, rung (d): second-order Markov log-score (two steps of memory; richer model).",
    "auc_diff_m1_minus_bag": "AUC gain from adding ORDER: first-order Markov minus bag of actions.",
    "auc_diff_m1_minus_volume": "AUC gain of first-order Markov over usage volume.",
    "auc_diff_m2_minus_m1": "AUC gain of the richer second-order chain over the first-order chain (about zero: the true process is first-order).",
    "gain_top_quartile_markov": "Mean gain (points) of held-out learners in the top quartile of the Markov log-score.",
    "gain_bottom_quartile_markov": "Mean gain (points) of held-out learners in the bottom quartile of the Markov log-score.",
    "gain_diff_top_vs_bottom_quartile": "Gain gap (points), top minus bottom quartile of the Markov log-score, held-out learners.",
    "p_AC_gained": "P(copy | full answer) among learners who gained 10+ points.",
    "p_AC_not": "P(copy | full answer) among learners who did not gain.",
    "p_HT_gained": "P(attempt | hint) among learners who gained.",
    "p_HT_not": "P(attempt | hint) among learners who did not gain.",
    "p_HA_gained": "P(full answer | hint) among learners who gained.",
    "p_HA_not": "P(full answer | hint) among learners who did not gain.",
    "p_AT_gained": "P(re-attempt | full answer) among learners who gained.",
    "p_AT_not": "P(re-attempt | full answer) among learners who did not gain.",
    "p_TV_gained": "P(verify | attempt) among learners who gained (similar across groups: verifying alone is not the signal).",
    "p_TV_not": "P(verify | attempt) among learners who did not gain.",
    "p_AC_not_minus_gained": "Answer->copy loop rate, not-gained minus gained (probability points).",
    "p_HT_gained_minus_not": "Hint->attempt rate, gained minus not-gained (probability points).",
    "p_AC_low_prior_quartile": "P(copy | full answer) in the lowest pre-test quartile (confounding: style tracks prior skill).",
    "p_AC_high_prior_quartile": "P(copy | full answer) in the highest pre-test quartile.",
    "mae_transition_true_labels": "Ground truth: mean absolute error of estimated vs true transition probabilities, chains fit with the true style labels.",
    "mae_transition_em": "Ground truth: same error for chains recovered WITHOUT labels by a 3-style mixture of Markov chains (EM).",
    "style_accuracy_em": "Ground truth: percent of learners the unlabeled mixture assigns to their true style.",
    "mean_gain_true_scaffolded": "Mean observed gain (points) for learners whose TRUE style is scaffolded.",
    "mean_gain_true_answer_seeking": "Mean observed gain (points) for learners whose TRUE style is answer-seeking.",
    "mean_gain_true_light_mixed": "Mean observed gain (points) for learners whose TRUE style is light/mixed.",
}
for s in ["scaffolded", "answer_seeking", "light_mixed"]:
    MEANING[f"style_share_true_{s}"] = f"Ground truth: percent of learners whose true style is {s} (realized in this sample)."
    MEANING[f"style_share_em_{s}"] = f"Percent of learners the unlabeled mixture estimates as {s}."

HEADLINE = {"r_messages_gain", "mean_messages_gained", "mean_messages_not_gained", "auc_volume", "auc_bag",
            "auc_markov1", "auc_markov2", "auc_diff_m1_minus_bag", "auc_diff_m2_minus_m1",
            "gain_diff_top_vs_bottom_quartile", "p_AC_gained", "p_AC_not", "p_HT_gained", "p_HT_not",
            "p_AC_not_minus_gained", "p_HT_gained_minus_not", "p_AC_low_prior_quartile", "p_AC_high_prior_quartile", "mae_transition_em", "style_accuracy_em",
            "share_gained", "n_missing_post_test"}


def fmt(v, d):
    v = float(v)
    return int(round(v)) if d == 0 else round(v, d) + 0.0   # + 0.0 turns -0.0 into 0.0


py = pd.read_csv(os.path.join(OUT, "results_python.csv")).set_index("name")
r = pd.read_csv(os.path.join(OUT, "results_r.csv")).set_index("name")
missing = sorted(set(py.index) ^ set(r.index))
results, bad = {}, []
for name in py.index:
    if name.endswith("_lo") or name.endswith("_hi"):
        continue
    d = int(py.loc[name, "digits"])
    entry = {"value": fmt(py.loc[name, "value"], d), "python": fmt(py.loc[name, "value"], d),
             "r": fmt(r.loc[name, "value"], d) if name in r.index else None, "rounding": d,
             "headline": name in HEADLINE, "meaning": MEANING.get(name, "")}
    entry["agree"] = bool(entry["python"] == entry["r"])
    for side in ("lo", "hi"):
        k = f"{name}_{side}"
        if k in py.index:
            entry[f"ci95_{side}"] = {"python": fmt(py.loc[k, "value"], d),
                                     "r": fmt(r.loc[k, "value"], d) if k in r.index else None}
            entry["agree"] = bool(entry["agree"] and entry[f"ci95_{side}"]["python"] == entry[f"ci95_{side}"]["r"])
    if not entry["agree"]:
        bad.append(name)
    results[name] = entry

with open(os.path.join(OUT, "results_python_only.json")) as f:
    python_only = json.load(f)
with open(os.path.join(DATA, "ground_truth_params.json")) as f:
    truth = json.load(f)

payload = {
    "edition": "E03_ai-tutors-in-practice",
    "disclaimer": ("All data in this folder are synthetic, generated to illustrate the method. They do not come from "
                   "Meta, any client, or any real organization. The goal is to show how a business question, a "
                   "measurement strategy, the statistics, and the code connect end to end."),
    "notes": {
        "outcome": "gained = post-test minus pre-test >= 10 points",
        "rounding": "AUC and probabilities 2 decimals; score points 1 decimal; shares whole percent; MAE 3 decimals",
        "intervals": "95% percentile bootstrap, 1,000 resamples of learners, Park-Miller RNG seed 20261106 shared by Python and R",
        "split": "train/test by learner, from the 'split' column in data/learners.csv (identical in both languages)",
    },
    "agreement": {"all_agree": bool(not bad and not missing), "n_numbers": len(results), "mismatches": bad,
                  "missing_in_one_language": missing},
    "results": results,
    "python_only": python_only,
    "ground_truth": {
        "style_shares_target": truth["style_shares_target"],
        "style_shares_realized": truth["style_shares_realized"],
        "gain_model": truth["gain_model"],
        "true_transition_matrices": "data/ground_truth_transitions.csv (also data/ground_truth_params.json)",
    },
}
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(payload, f, indent=2)

if bad or missing:
    print("AGREEMENT FAILED:", bad, missing)
    sys.exit(1)
print(f"AGREEMENT OK: {len(results)} numbers (and their intervals) match between Python and R. "
      "Wrote outputs/results.json")
