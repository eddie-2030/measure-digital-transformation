"""C02 Golden Sets: majority vote vs. Dawid-Skene, plus targeted adjudication.

Run from the package folder (or anywhere):  python3 python/analysis.py
Reads  ../data/*.csv   Writes ../outputs/results.json, results_python_full_precision.json,
fig_1_*.png, fig_2_*.png and ../templates/adjudication_queue.csv (worked example).

Steps
  0. Clean the label file (duplicate row, skipped labels).
  1. Naive answer key: majority vote, and why it misleads.
  2. Correct answer key: Dawid-Skene latent class model fitted by EM (from scratch).
  3. Check against the hidden truth, and spend a 100-item expert budget three ways.
  4. Decision implication: the AI agent's measured error rate under each answer key.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA, OUT, TPL = ROOT / "data", ROOT / "outputs", ROOT / "templates"
OUT.mkdir(exist_ok=True)

BUDGET = 100          # expert adjudication budget (10% of items)
PSEUDO = 1.0          # pseudo-count added to each confusion-matrix cell (keeps estimates off 0 and 1)
TOL = 1e-9            # EM stops when no item's posterior moves by more than this
MAX_ITER = 1000

# ------------------------------------------------------------------ Step 0: load and clean
raw = pd.read_csv(DATA / "labels_long.csv")
n_raw = len(raw)
dedup = raw.drop_duplicates()                       # rule 1: drop exact duplicate rows
n_dup = n_raw - len(dedup)
lab = dedup[dedup["label"].notna()].copy()          # rule 2: a skipped (blank) label is not a vote
n_skipped = len(dedup) - len(lab)
lab["label"] = lab["label"].astype(int)

items = pd.read_csv(DATA / "items.csv")
draws = pd.read_csv(DATA / "expert_adjudication_draws.csv")
truth = pd.read_csv(DATA / "ground_truth_items.csv")       # hidden truth: used ONLY in checks
true_lab = pd.read_csv(DATA / "ground_truth_labelers.csv")

item_ids = items["item_id"].to_numpy()
labeler_ids = np.sort(lab["labeler_id"].unique())
I, J = len(item_ids), len(labeler_ids)
ii = pd.Index(item_ids).get_indexer(lab["item_id"])
jj = pd.Index(labeler_ids).get_indexer(lab["labeler_id"])
y = lab["label"].to_numpy()

n_err_votes = np.bincount(ii, weights=y, minlength=I)
n_votes = np.bincount(ii, minlength=I).astype(float)
n_ok_votes = n_votes - n_err_votes
y_true = truth.set_index("item_id").loc[item_ids, "true_label"].to_numpy()
ambig = truth.set_index("item_id").loc[item_ids, "ambiguous"].to_numpy()
rand_order = draws.set_index("item_id").loc[item_ids, "random_order"].to_numpy()
expert = draws.set_index("item_id").loc[item_ids, "expert_label"].to_numpy()
judge = items["ai_judge_flag"].to_numpy()


def key_metrics(key):
    """Compare an answer key with the hidden truth."""
    return dict(
        wrong=int((key != y_true).sum()),
        missed_errors=int(((key == 0) & (y_true == 1)).sum()),
        false_alarms=int(((key == 1) & (y_true == 0)).sum()),
        accuracy=float((key == y_true).mean()),
        sensitivity=float(key[y_true == 1].mean()),      # share of real errors the key calls errors
        prevalence=float(key.mean()),
        wrong_on_ambiguous=int(((key != y_true) & (ambig == 1)).sum()),
    )


# ------------------------------------------------------------------ Step 1: majority vote
# An item is "error" when error votes outnumber OK votes. Ties go to "OK"
# (the common default: "not enough agreement to call it an error").
mv = (n_err_votes > n_ok_votes).astype(int)
mv_m = key_metrics(mv)
n_ties = int((n_err_votes == n_ok_votes).sum())

# Raw pairwise agreement: across all pairs of labels on the same item, how often do they match?
pairs = n_err_votes * (n_err_votes - 1) / 2 + n_ok_votes * (n_ok_votes - 1) / 2
all_pairs = n_votes * (n_votes - 1) / 2
pairwise_agreement = float(pairs.sum() / all_pairs.sum())


# ------------------------------------------------------------------ Step 2: Dawid-Skene EM
def dawid_skene(ii, jj, y, init_T):
    """Binary Dawid-Skene. T[i] = P(item i is truly an error | all labels).

    M-step: given T, estimate prevalence and each labeler's confusion matrix
            (sensitivity, specificity), with PSEUDO counts in every cell.
    E-step: given those, P(error | labels) is proportional to
            prevalence * product over labelers of P(their label | error),
            compared with (1 - prevalence) * product of P(their label | OK).
    """
    T = init_T.astype(float).copy()
    for it in range(1, MAX_ITER + 1):
        # M-step
        prev = T.mean()
        tp = np.bincount(jj, weights=T[ii] * y, minlength=J)
        pos = np.bincount(jj, weights=T[ii], minlength=J)
        tn = np.bincount(jj, weights=(1 - T[ii]) * (1 - y), minlength=J)
        neg = np.bincount(jj, weights=1 - T[ii], minlength=J)
        sens = (tp + PSEUDO) / (pos + 2 * PSEUDO)
        spec = (tn + PSEUDO) / (neg + 2 * PSEUDO)
        # E-step, on the log scale: each vote adds its labeler's log-likelihood ratio
        la = np.log(prev) + np.bincount(ii, weights=y * np.log(sens[jj]) + (1 - y) * np.log(1 - sens[jj]), minlength=I)
        lb = np.log(1 - prev) + np.bincount(ii, weights=y * np.log(1 - spec[jj]) + (1 - y) * np.log(spec[jj]), minlength=I)
        T_new = 1 / (1 + np.exp(lb - la))
        delta = np.max(np.abs(T_new - T))
        T = T_new
        if delta < TOL:
            break
    loglik = float(np.sum(np.logaddexp(la, lb)))
    return dict(T=T, prev=prev, sens=sens, spec=spec, iterations=it, loglik=loglik)


ds = dawid_skene(ii, jj, y, init_T=mv)       # initialized from the majority-vote key
post = ds["T"]
ds_key = (post > 0.5).astype(int)
ds_m = key_metrics(ds_key)

# Labelers: estimated vs. true (generating) parameters
tl = true_lab.set_index("labeler_id").loc[labeler_ids]
n_items_lab = np.bincount(jj, minlength=J)
# "Measured against majority vote": what you'd conclude about each labeler if MV were the truth
mv_ref_sens = np.bincount(jj, weights=mv[ii] * y, minlength=J) / np.bincount(jj, weights=mv[ii], minlength=J)
# Realized sensitivity against the hidden truth on the items each labeler actually saw
real_sens = np.bincount(jj, weights=y_true[ii] * y, minlength=J) / np.bincount(jj, weights=y_true[ii], minlength=J)
real_spec = (np.bincount(jj, weights=(1 - y_true[ii]) * (1 - y), minlength=J)
             / np.bincount(jj, weights=1 - y_true[ii], minlength=J))
# "True" catch rate used for checking = realized sensitivity against the hidden truth on the items
# each labeler actually labeled. It sits below the generating parameter because 10% of items are
# ambiguous (everyone is worse there). We also report the error vs. the generating parameters.
big = n_items_lab >= 30                   # MAE over labelers with enough items to be estimable
mae_sens_ds = float(np.mean(np.abs(ds["sens"][big] - real_sens[big])))
mae_spec_ds = float(np.mean(np.abs(ds["spec"][big] - real_spec[big])))
mae_sens_mv = float(np.mean(np.abs(mv_ref_sens[big] - real_sens[big])))
mae_sens_ds_vs_param = float(np.mean(np.abs(ds["sens"][big] - tl["true_sensitivity"].to_numpy()[big])))
corr_sens_ds = float(np.corrcoef(ds["sens"][big], real_sens[big])[0, 1])
ds_expected_prev = float(post.mean())     # expected prevalence = average posterior (no 0.5 cut-off)
lenient = tl["group"].to_numpy() == "lenient"
lenient_share_of_labels = float(np.isin(lab["labeler_id"], labeler_ids[lenient]).mean())

# ------------------------------------------------------------------ Step 3: adjudication, same budget
def adjudicate(key, chosen):
    k = key.copy()
    k[chosen] = expert[chosen]           # the expert's (pre-drawn, 98%-accurate) label replaces the key
    return k


def pick(score):
    """Indices of the BUDGET items with the smallest score; ties broken by the pre-drawn random order."""
    return np.lexsort((rand_order, np.round(score, 10)))[:BUDGET]


pick_targeted = pick(np.abs(post - 0.5))                                 # most uncertain under DS
pick_random = pick(rand_order.astype(float))                             # random items
pick_margin = pick(np.abs(n_err_votes - n_ok_votes) / n_votes)           # narrowest MV margin

keys = {
    "majority_vote": mv,
    "mv_plus_random_adjudication": adjudicate(mv, pick_random),
    "mv_plus_margin_adjudication": adjudicate(mv, pick_margin),
    "dawid_skene": ds_key,
    "ds_plus_random_adjudication": adjudicate(ds_key, pick_random),
    "ds_plus_targeted_adjudication": adjudicate(ds_key, pick_targeted),
}
km = {k: key_metrics(v) for k, v in keys.items()}
targeted_fixed = int(((ds_key != y_true) & (keys["ds_plus_targeted_adjudication"] == y_true)).sum())
random_fixed = int(((ds_key != y_true) & (keys["ds_plus_random_adjudication"] == y_true)).sum())
targeted_wrong_before = int((ds_key[pick_targeted] != y_true[pick_targeted]).sum())
random_wrong_before = int((ds_key[pick_random] != y_true[pick_random]).sum())
final = keys["ds_plus_targeted_adjudication"]
final_wrong_ambig_share = float(km["ds_plus_targeted_adjudication"]["wrong_on_ambiguous"]
                                / km["ds_plus_targeted_adjudication"]["wrong"])
# How lucky or unlucky was the single random draw? Exact distribution, no simulation needed.
# W = number of wrong DS labels among 100 random items ~ Hypergeometric(1000 items, 78 wrong, 100 drawn).
# Given W, a fresh 98%-accurate expert fixes F ~ Binomial(W, 0.98) and breaks M ~ Binomial(100 - W, 0.02).
# Net fixed = F - M. Its distribution is the mixture over W of the difference of the two binomials.
from math import comb
EXPERT_ACC = 0.98
n_wrong_ds = int((ds_key != y_true).sum())


def binom_pmf(n, p):
    return np.array([comb(n, k) * p ** k * (1 - p) ** (n - k) for k in range(n + 1)])


net_pmf = {}
for w in range(0, min(BUDGET, n_wrong_ds) + 1):
    pw = comb(n_wrong_ds, w) * comb(I - n_wrong_ds, BUDGET - w) / comb(I, BUDGET)
    if pw == 0:
        continue
    pf, pm = binom_pmf(w, EXPERT_ACC), binom_pmf(BUDGET - w, 1 - EXPERT_ACC)
    for f in range(w + 1):
        for m in range(BUDGET - w + 1):
            net_pmf[f - m] = net_pmf.get(f - m, 0.0) + pw * pf[f] * pm[m]
net_vals = np.array(sorted(net_pmf)); net_p = np.array([net_pmf[v] for v in net_vals])
net_cdf = np.cumsum(net_p)
random_expected_net_fixed = float(BUDGET * (n_wrong_ds / I * EXPERT_ACC - (1 - n_wrong_ds / I) * (1 - EXPERT_ACC)))
random_net_fixed_lo95 = int(net_vals[np.argmax(net_cdf >= 0.025)])
random_net_fixed_hi95 = int(net_vals[np.argmax(net_cdf >= 0.975)])
random_p_net_le_2 = float(net_p[net_vals <= 2].sum())
# Expected net for the targeted 100, had the expert been a fresh 98% draw on each item
targeted_expected_net_fixed = float(targeted_wrong_before * EXPERT_ACC - (BUDGET - targeted_wrong_before) * (1 - EXPERT_ACC))
mean_post_ambig = float(np.mean(np.minimum(post, 1 - post)[ambig == 1]))
mean_post_clear = float(np.mean(np.minimum(post, 1 - post)[ambig == 0]))

# ------------------------------------------------------------------ Step 4: decision implication
def judge_scores(key):
    flagged = judge == 1
    return dict(recall=float(judge[key == 1].mean()), precision=float(key[flagged].mean()))


agent_rate = {"majority_vote": mv.mean(), "dawid_skene": ds_key.mean(),
              "ds_plus_targeted_adjudication": final.mean(), "truth": y_true.mean()}
judge_by_key = {"majority_vote": judge_scores(mv), "dawid_skene": judge_scores(ds_key),
                "ds_plus_targeted_adjudication": judge_scores(final), "truth": judge_scores(y_true)}

# ------------------------------------------------------------------ results
full = {
    "n_label_rows_raw": n_raw, "n_duplicate_rows_dropped": n_dup, "n_skipped_labels": n_skipped,
    "n_labels_used": len(lab), "n_items": I, "n_labelers": J,
    "n_items_under_3_labels": int((n_votes < 3).sum()),
    "mean_labels_per_item": float(n_votes.mean()),
    "lenient_share_of_labels": lenient_share_of_labels,
    "true_prevalence": float(y_true.mean()), "n_true_errors": int(y_true.sum()),
    "n_ambiguous": int(ambig.sum()),
    "pairwise_agreement": pairwise_agreement, "mv_ties": n_ties,
    "ds_iterations": ds["iterations"], "ds_loglik": ds["loglik"], "ds_prevalence_param": float(ds["prev"]),
    "mae_sens_ds": mae_sens_ds, "mae_spec_ds": mae_spec_ds, "mae_sens_mv_referenced": mae_sens_mv,
    "mae_sens_ds_vs_param": mae_sens_ds_vs_param, "corr_sens_ds": corr_sens_ds, "ds_expected_prevalence": ds_expected_prev,
    "targeted_items_wrong_before": targeted_wrong_before, "random_items_wrong_before": random_wrong_before,
    "targeted_fixed": targeted_fixed, "random_fixed": random_fixed,
    "final_wrong_ambiguous_share": final_wrong_ambig_share,
    "mean_uncertainty_ambiguous": mean_post_ambig, "mean_uncertainty_clear": mean_post_clear,
    "random_expected_net_fixed": random_expected_net_fixed, "random_net_fixed_lo95": random_net_fixed_lo95,
    "random_net_fixed_hi95": random_net_fixed_hi95, "random_p_net_le_2": random_p_net_le_2,
    "targeted_expected_net_fixed": targeted_expected_net_fixed, "random_net_pmf_total": float(net_p.sum()),
}
for k, m in km.items():
    for mk, mv_ in m.items():
        full[f"{k}__{mk}"] = mv_
for k, v in agent_rate.items():
    full[f"agent_error_rate__{k}"] = float(v)
for k, v in judge_by_key.items():
    full[f"judge_recall__{k}"] = v["recall"]
    full[f"judge_precision__{k}"] = v["precision"]
for j, lid in enumerate(labeler_ids):
    full[f"labeler__{lid}__ds_sens"] = float(ds["sens"][j])
    full[f"labeler__{lid}__ds_spec"] = float(ds["spec"][j])
    full[f"labeler__{lid}__mv_ref_sens"] = float(mv_ref_sens[j])
    full[f"labeler__{lid}__n_labels"] = int(n_items_lab[j])
    full[f"labeler__{lid}__realized_sens"] = float(real_sens[j])
json.dump(full, open(OUT / "results_python_full_precision.json", "w"), indent=1)

# Headline numbers the article may quote. "digits" is the rounding the article must use
# (for a percentage: digits of the percentage, e.g. 0.1953 -> 19.5%).
H = [
    ("true_prevalence", full["true_prevalence"], "pct", 1, "True share of the 1,000 AI replies that contain an error (hidden truth)"),
    ("n_true_errors", full["n_true_errors"], "count", 0, "Number of truly erroneous replies"),
    ("lenient_share_of_labels", lenient_share_of_labels, "pct", 0, "Share of all labels given by the 3 lenient rubber-stampers"),
    ("pairwise_agreement", pairwise_agreement, "pct", 0, "Raw agreement between pairs of labelers on the same item (looks reassuring)"),
    ("mv_prevalence", km["majority_vote"]["prevalence"], "pct", 1, "Error rate according to the majority-vote answer key"),
    ("mv_sensitivity", km["majority_vote"]["sensitivity"], "pct", 0, "Share of real errors the majority-vote key labels as errors"),
    ("mv_wrong", km["majority_vote"]["wrong"], "count", 0, "Golden-set labels that are wrong under majority vote (of 1,000)"),
    ("mv_missed_errors", km["majority_vote"]["missed_errors"], "count", 0, "Real errors the majority-vote key calls OK"),
    ("mv_accuracy", km["majority_vote"]["accuracy"], "pct", 1, "Majority-vote key accuracy against the truth"),
    ("ds_prevalence", km["dawid_skene"]["prevalence"], "pct", 1, "Error rate according to the Dawid-Skene answer key"),
    ("ds_sensitivity", km["dawid_skene"]["sensitivity"], "pct", 0, "Share of real errors the Dawid-Skene key labels as errors"),
    ("ds_expected_prevalence", ds_expected_prev, "pct", 1, "Dawid-Skene expected error rate (average posterior probability, no 0.5 cut-off)"),
    ("ds_wrong", km["dawid_skene"]["wrong"], "count", 0, "Golden-set labels that are wrong under Dawid-Skene (of 1,000)"),
    ("ds_missed_errors", km["dawid_skene"]["missed_errors"], "count", 0, "Real errors the Dawid-Skene key calls OK"),
    ("ds_accuracy", km["dawid_skene"]["accuracy"], "pct", 1, "Dawid-Skene key accuracy against the truth"),
    ("ds_iterations", ds["iterations"], "count", 0, "EM iterations to converge (tolerance 1e-9 on posteriors)"),
    ("mae_sens_ds", mae_sens_ds, "pp", 0, "Mean absolute gap between DS-estimated and true (realized) labeler sensitivity, 14 labelers with 30+ labels, percentage points"),
    ("mae_sens_mv_referenced", mae_sens_mv, "pp", 0, "Same gap when each labeler is scored against the majority-vote key, percentage points"),
    ("corr_sens_ds", corr_sens_ds, "num", 2, "Correlation between DS-estimated and true (realized) labeler sensitivity"),
    ("ds_targeted_wrong", km["ds_plus_targeted_adjudication"]["wrong"], "count", 0, "Wrong labels after DS + adjudicating the 100 most uncertain items"),
    ("ds_random_wrong", km["ds_plus_random_adjudication"]["wrong"], "count", 0, "Wrong labels after DS + adjudicating 100 random items"),
    ("random_expected_net_fixed", random_expected_net_fixed, "count", 0, "Expected net labels fixed by a random 100-item expert budget (exact: 100 x (7.8% x 0.98 - 92.2% x 0.02)); display 'about 6'"),
    ("random_net_fixed_lo95", random_net_fixed_lo95, "count", 0, "Lower end of the 95% range of net fixes from a random 100 (exact hypergeometric x binomial distribution)"),
    ("random_net_fixed_hi95", random_net_fixed_hi95, "count", 0, "Upper end of the same 95% range"),
    ("random_p_net_le_2", random_p_net_le_2, "pct", 0, "Probability that a random 100 nets 2 or fewer fixes (how unlucky this draw was)"),
    ("targeted_expected_net_fixed", targeted_expected_net_fixed, "count", 0, "Expected net fixes for the targeted 100 with a fresh 98% expert (37 x 0.98 - 63 x 0.02); realized was 37"),
    ("mv_margin_wrong", km["mv_plus_margin_adjudication"]["wrong"], "count", 0, "Wrong labels after MV + adjudicating the 100 narrowest-margin items"),
    ("mv_random_wrong", km["mv_plus_random_adjudication"]["wrong"], "count", 0, "Wrong labels after MV + adjudicating 100 random items"),
    ("targeted_items_wrong_before", targeted_wrong_before, "count", 0, "Of the 100 targeted items, how many the DS key had wrong (the budget's hit rate)"),
    ("random_items_wrong_before", random_wrong_before, "count", 0, "Of the 100 random items, how many the DS key had wrong"),
    ("ds_targeted_sensitivity", km["ds_plus_targeted_adjudication"]["sensitivity"], "pct", 0, "Share of real errors the final key (DS + targeted) labels as errors"),
    ("ds_targeted_prevalence", km["ds_plus_targeted_adjudication"]["prevalence"], "pct", 1, "Error rate according to the final key (DS + targeted)"),
    ("final_wrong_ambiguous_share", final_wrong_ambig_share, "pct", 0, "Share of the final key's remaining wrong labels that sit on the 10% ambiguous items"),
    ("mv_ties", n_ties, "count", 0, "Items with a tied majority vote (2-2), assigned 'OK' by the tie rule"),
    ("n_items_under_3_labels", full["n_items_under_3_labels"], "count", 0, "Items left with fewer than 3 valid labels after skips"),
    ("judge_precision_mv", judge_by_key["majority_vote"]["precision"], "pct", 0, "AI judge's measured precision when scored against the majority-vote key"),
    ("judge_precision_ds_targeted", judge_by_key["ds_plus_targeted_adjudication"]["precision"], "pct", 0, "AI judge's measured precision against the final key (DS + targeted)"),
    ("judge_precision_truth", judge_by_key["truth"]["precision"], "pct", 0, "AI judge's precision against the hidden truth"),
]
results = {
    "package": "C02_golden-sets", "seed": 20261109,
    "note": "pct values are proportions; quote them as percentages with 'digits' decimals. pp = percentage points. "
            "'value' is the Python value; 'python' and 'r' are the rounded values from each language.",
    "results": [],
}


def rnd(v, unit, d):
    x = v * 100 if unit in ("pct", "pp") else v
    return int(round(x)) if d == 0 else round(x, d)


for name, v, unit, d, meaning in H:
    results["results"].append(dict(name=name, value=v, unit=unit, digits=d,
                                   display=f"{rnd(v, unit, d)}{'%' if unit == 'pct' else (' pp' if unit == 'pp' else '')}",
                                   meaning=meaning, python=rnd(v, unit, d), r=None))
json.dump(results, open(OUT / "results.json", "w"), indent=1)

# ------------------------------------------------------------------ worked adjudication queue (template example)
q = pick_targeted[:12]
lab_sorted = lab.sort_values(["item_id", "labeler_id"])
summ = lab_sorted.groupby("item_id").apply(
    lambda d: " ".join(f"{a}={'E' if b else 'OK'}" for a, b in zip(d["labeler_id"], d["label"])), include_groups=False)
queue = pd.DataFrame({
    "item_id": item_ids[q], "posterior_error_prob": np.round(post[q], 3),
    "n_labels": n_votes[q].astype(int),
    "labels_summary": [f"{int(n_err_votes[k])} error / {int(n_ok_votes[k])} OK: {summ[item_ids[k]]}" for k in q],
    "reason": "DS posterior near 0.5 (most uncertain)", "adjudicator": "", "final_label": "", "notes": "",
})
queue.to_csv(TPL / "adjudication_queue.csv", index=False)

# ------------------------------------------------------------------ figures (phone-readable: big fonts, few marks)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 15, "axes.spines.top": False,
                     "axes.spines.right": False, "text.parse_math": False})
GREY, BLUE, ORANGE, GREEN, INK = "#9aa3ad", "#2a78d6", "#eb6834", "#2e9e6a", "#111111"

order = [("majority_vote", "Majority vote", GREY), ("mv_plus_margin_adjudication", "Majority vote\n+ 100 closest votes", GREY),
         ("dawid_skene", "Dawid-Skene", BLUE), ("ds_plus_random_adjudication", "Dawid-Skene\n+ 100 random", BLUE),
         ("ds_plus_targeted_adjudication", "Dawid-Skene\n+ 100 most uncertain", GREEN)]
fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
vals = [km[k]["wrong"] for k, _, _ in order]
miss = [km[k]["missed_errors"] for k, _, _ in order]
ypos = np.arange(len(order))[::-1]
ax.barh(ypos, vals, color=[c for _, _, c in order], height=0.62)
for yp, v, m in zip(ypos, vals, miss):
    ax.text(v + 2, yp, f"{v} wrong  ({m} missed errors)", va="center", fontsize=14, color=INK)
ax.set_yticks(ypos, [lbl for _, lbl, _ in order], fontsize=14)
ax.set_xlim(0, max(vals) * 1.85)
ax.set_xlabel("Wrong labels in a 1,000-item golden set", fontsize=14)
fig.suptitle(f"Dawid-Skene plus {BUDGET} targeted expert checks\ncuts golden-set errors from "
             f"{km['majority_vote']['wrong']} to {km['ds_plus_targeted_adjudication']['wrong']}",
             x=0.03, ha="left", fontsize=19, fontweight="bold")
ax.set_title("Same expert budget (100 items) for every adjudication bar. Synthetic data.", loc="left",
             fontsize=12, color="#555555")
fig.tight_layout()
fig.savefig(OUT / "fig_1_golden_set_errors.png", facecolor="white")
plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 9), dpi=150)
ts = real_sens
grp = tl["group"].to_numpy()
ax.plot([0.3, 1], [0.3, 1], color="#cccccc", lw=2, zorder=0)
ax.text(0.97, 0.93, "perfect\nestimate", ha="right", va="top", fontsize=12, color="#888888")
for j in range(J):
    if not big[j]:
        continue
    ax.plot([ts[j], ts[j]], [mv_ref_sens[j], ds["sens"][j]], color="#dddddd", lw=1.5, zorder=1)
ax.scatter(ts[big], mv_ref_sens[big], s=110, facecolors="white", edgecolors=GREY, linewidths=2.5,
           label="Scored against majority vote", zorder=2)
ax.scatter(ts[big], ds["sens"][big], s=110, color=BLUE, label="Dawid-Skene estimate", zorder=3)
for j in np.where(big & lenient)[0]:
    ax.annotate(labeler_ids[j], (ts[j], mv_ref_sens[j]), xytext=(8, 0), textcoords="offset points",
                fontsize=12, color=ORANGE, va="center", fontweight="bold")
ax.text(0.455, 0.40, "rubber-\nstampers", fontsize=13, color=ORANGE, ha="center")
for j in np.where(tl["group"].to_numpy() == "near_random")[0]:
    ax.annotate(f"{labeler_ids[j]} (near-random)", (ts[j], mv_ref_sens[j]), xytext=(0, 16),
                textcoords="offset points", fontsize=12, color="#555555", va="bottom", ha="center")
ax.set_xlim(0.3, 1); ax.set_ylim(0.3, 1)
ax.set_xlabel("True sensitivity (share of real errors the labeler flagged)", fontsize=14)
ax.set_ylabel("Estimated sensitivity", fontsize=14)
ax.legend(loc="lower right", fontsize=13, frameon=False)
fig.suptitle("Dawid-Skene tracks each labeler's real catch rate;\nscoring against majority vote inflates everyone's",
             x=0.03, ha="left", fontsize=19, fontweight="bold")
ax.set_title("Each dot is a labeler with 30+ labels. Synthetic data.", loc="left", fontsize=12, color="#555555")
fig.tight_layout()
fig.savefig(OUT / "fig_2_labeler_sensitivity.png", facecolor="white")
plt.close(fig)

# ------------------------------------------------------------------ console report
print(f"Step 0  {n_raw} rows -> dropped {n_dup} duplicate, {n_skipped} skipped -> {len(lab)} labels; "
      f"{int((n_votes < 3).sum())} items left with <3 labels")
print(f"        true prevalence {y_true.mean():.3f}; lenient labelers give {lenient_share_of_labels:.1%} of labels")
print(f"Step 1  pairwise agreement {pairwise_agreement:.1%}; ties {n_ties}")
for k in km:
    m = km[k]
    print(f"        {k:32s} wrong {m['wrong']:3d} missed {m['missed_errors']:3d} FA {m['false_alarms']:3d} "
          f"sens {m['sensitivity']:.3f} prev {m['prevalence']:.3f} acc {m['accuracy']:.3f} ambig-wrong {m['wrong_on_ambiguous']}")
print(f"Step 2  DS converged in {ds['iterations']} iterations, prevalence param {ds['prev']:.3f}; "
      f"MAE sens DS {mae_sens_ds:.3f} vs MV-referenced {mae_sens_mv:.3f}; MAE spec DS {mae_spec_ds:.3f}; "
      f"vs params {mae_sens_ds_vs_param:.3f}; corr {corr_sens_ds:.3f}; expected prev {ds_expected_prev:.3f}")
for j, lid in enumerate(labeler_ids):
    print(f"        {lid} n={n_items_lab[j]:3d} param sens {tl['true_sensitivity'].iloc[j]:.2f} ds {ds['sens'][j]:.3f} mv-ref {mv_ref_sens[j]:.3f} "
          f"realized {real_sens[j]:.3f} | true spec {tl['true_specificity'].iloc[j]:.2f} ds {ds['spec'][j]:.3f}")
print(f"        random 100: expected net {random_expected_net_fixed:.2f}, 95% range {random_net_fixed_lo95} to "
      f"{random_net_fixed_hi95}, P(net<=2) {random_p_net_le_2:.3f}; targeted expected {targeted_expected_net_fixed:.2f}")
print(f"Step 3  targeted items wrong before {targeted_wrong_before}, random {random_wrong_before}; "
      f"fixed {targeted_fixed} vs {random_fixed}; final wrong on ambiguous {final_wrong_ambig_share:.1%}")
print(f"        mean min(p,1-p): ambiguous {mean_post_ambig:.3f} clear {mean_post_clear:.3f}")
print("Step 4  agent error rate:", {k: round(float(v), 4) for k, v in agent_rate.items()})
print("        judge:", judge_by_key)
