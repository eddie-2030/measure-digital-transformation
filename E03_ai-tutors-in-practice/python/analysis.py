"""
analysis.py -- E03 "AI Tutors in Practice: What Interaction Data Reveals"
Measure Digital Transformation (Dr. Eddie Lin). Synthetic data only.

Question: How do learners actually use the AI tutor, and does it help?
Outcome:  "gained" = post-test minus pre-test >= 10 points.

Step 1  Naive: usage volume (messages, minutes, sessions) vs learning gain.
Step 2  The ladder, evaluated on HELD-OUT learners (split column written by the generator):
          (a) volume features -> logistic regression
          (b) bag of actions (shares of T/H/A/C, order ignored) -> logistic regression
          (c) first-order Markov log-score (order matters)
          (d) second-order Markov log-score (the "richer model", same idea with 2 steps of memory)
          (e) PYTHON ONLY: a small neural network (MLP) on 1-3-gram shares, a light stand-in for
              a neural sequence model. Not reproduced in R.
        plus key differences with bootstrap 95% intervals (shared Park-Miller RNG, identical in R).
Step 3  Ground-truth check: transition matrices estimated with the true styles, and a 3-style
        mixture of Markov chains (EM) that recovers the latent styles without labels.
Step 4  Decision implication: the "learning-rich interaction" metrics to put on a dashboard.

Run from anywhere:  python python/analysis.py
Writes outputs/results_python.csv, outputs/fig1_*.png, outputs/fig2_*.png.
Then run r/analysis.R and python/check_agreement.py to build outputs/results.json.
"""
import json
import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

ACTIONS = ["T", "H", "A", "C", "V"]           # Q (quit) ends a session
ROWS = ["S", "T", "H", "A", "C", "V"]         # S = START of a session
COLS = ["T", "H", "A", "C", "V", "Q"]
SMOOTH = 0.5                                  # pseudo-count added to every transition cell
B = 1000                                      # bootstrap resamples
SEED = 20261106
RESULTS = []                                  # (name, value, rounding)


def record(name, value, digits):
    RESULTS.append((name, float(value), digits))


# --------------------------------------------------------------------------------------
# Shared tools (written identically in r/analysis.R)
# --------------------------------------------------------------------------------------
class ParkMiller:
    """Minimal-standard LCG: x <- 16807 * x mod (2^31 - 1). Same stream in Python and R."""
    M = 2147483647

    def __init__(self, seed):
        self.x = seed % self.M

    def index(self, n, k):
        """k random integers in 0..n-1 (sampling with replacement)."""
        out = np.empty(k, dtype=np.int64)
        for i in range(k):
            self.x = (16807 * self.x) % self.M
            out[i] = int(np.floor(self.x / self.M * n))
        return out


def auc(y, score):
    """Area under the ROC curve via ranks (Mann-Whitney), ties get average ranks."""
    y = np.asarray(y)
    r = rankdata(score)
    n1 = y.sum()
    n0 = len(y) - n1
    return (r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def chain(counts):
    """Row-normalized transition matrix from a (rows x cols) count matrix, with smoothing."""
    c = counts + SMOOTH
    return c / c.sum(axis=1, keepdims=True)


def pct_interval(x):
    """Percentile 95% interval: 26th and 975th of the 1,000 sorted values (same rule in R)."""
    s = np.sort(x)
    return s[int(np.floor(0.025 * len(s)))], s[int(np.ceil(0.975 * len(s))) - 1]


# --------------------------------------------------------------------------------------
# Load and clean
# --------------------------------------------------------------------------------------
ev = pd.read_csv(os.path.join(DATA, "tutor_events.csv"), dtype=str, keep_default_na=False)
lr = pd.read_csv(os.path.join(DATA, "learners.csv"))
n_events_raw = len(ev)

# 1) Normalize codes (" h" -> "H"); 2) drop unreadable tokens and count them.
ev["tok"] = ev["action"].str.strip().str.upper()
n_unreadable = int((~ev["tok"].isin(COLS)).sum())
ev = ev[ev["tok"].isin(COLS)].copy()
ev["step"] = ev["step"].astype(int)
ev = ev.sort_values(["learner_id", "session_id", "step"]).reset_index(drop=True)
# 3) Anything logged after a Q in the same session is ignored (none in this file, but be safe).
after_q = ev.groupby("session_id")["tok"].transform(lambda s: (s == "Q").cumsum().shift(fill_value=0) > 0)
ev = ev[~after_q].copy()
# 4) Previous tokens within each session (START = "S").
ev["p1"] = ev.groupby("session_id")["tok"].shift(1, fill_value="S")
ev["p2"] = ev.groupby("session_id")["tok"].shift(2, fill_value="S")
ev["ts"] = pd.to_datetime(ev["timestamp"])

learner_ids = lr["learner_id"].tolist()
idx = {l: i for i, l in enumerate(learner_ids)}
n = len(learner_ids)
li = ev["learner_id"].map(idx).to_numpy()

# Per-learner transition counts: first order (6 x 6 = 36 cells), second order (36 contexts x 6).
r1 = ev["p1"].map({s: i for i, s in enumerate(ROWS)}).to_numpy()
r2 = ev["p2"].map({s: i for i, s in enumerate(ROWS)}).to_numpy()
cc = ev["tok"].map({s: i for i, s in enumerate(COLS)}).to_numpy()
N1 = np.zeros((n, 6 * 6))
np.add.at(N1, (li, r1 * 6 + cc), 1)
N2 = np.zeros((n, 36 * 6))
np.add.at(N2, (li, (r2 * 6 + r1) * 6 + cc), 1)

# Volume features.
lr["messages"] = np.bincount(li, weights=(ev["tok"] != "Q").to_numpy(), minlength=n)
sess = ev.groupby(["learner_id", "session_id"])["ts"].agg(["min", "max"])
sess["minutes"] = (sess["max"] - sess["min"]).dt.total_seconds() / 60
per = sess.groupby("learner_id")["minutes"].agg(["sum", "size"])
lr["minutes"] = lr["learner_id"].map(per["sum"]).fillna(0).to_numpy()
lr["sessions"] = lr["learner_id"].map(per["size"]).fillna(0).to_numpy()
# Bag of actions: shares of T, H, A, C among the five action types (V is the reference share).
act_counts = np.zeros((n, 5))
mask = ev["tok"].isin(ACTIONS).to_numpy()
np.add.at(act_counts, (li[mask], ev.loc[mask, "tok"].map({a: i for i, a in enumerate(ACTIONS)}).to_numpy()), 1)
shares = act_counts / np.maximum(act_counts.sum(1, keepdims=True), 1)
for j, a in enumerate(ACTIONS):
    lr[f"share_{a}"] = shares[:, j]

lr["gain"] = lr["post_score"] - lr["pre_score"]
n_missing_post = int(lr["post_score"].isna().sum())
lr["gained"] = (lr["gain"] >= 10).astype(float)
lr.loc[lr["post_score"].isna(), "gained"] = np.nan
has_post = lr["post_score"].notna().to_numpy()
is_train = (lr["split"] == "train").to_numpy() & has_post
is_test = (lr["split"] == "test").to_numpy() & has_post

record("n_learners", n, 0)
record("n_events", n_events_raw, 0)
record("n_unreadable_tokens_dropped", n_unreadable, 0)
record("n_missing_post_test", n_missing_post, 0)
record("n_short_logs_under_5_actions", (lr["messages"] < 5).sum(), 0)
record("n_train_with_post", is_train.sum(), 0)
record("n_test_with_post", is_test.sum(), 0)
record("share_gained", lr.loc[has_post, "gained"].mean() * 100, 0)
print(f"{n} learners, {n_events_raw:,} events; dropped {n_unreadable} unreadable tokens; "
      f"{n_missing_post} learners without a post-test are kept out of the gain analysis (not imputed).")

# --------------------------------------------------------------------------------------
# Step 1 -- naive: does usage volume track learning?
# --------------------------------------------------------------------------------------
d = lr[has_post]
record("r_messages_gain", np.corrcoef(d["messages"], d["gain"])[0, 1], 2)
record("r_minutes_gain", np.corrcoef(d["minutes"], d["gain"])[0, 1], 2)
record("mean_messages_gained", d.loc[d.gained == 1, "messages"].mean(), 1)
record("mean_messages_not_gained", d.loc[d.gained == 0, "messages"].mean(), 1)
print("\nStep 1 (naive): r(messages, gain) = %.2f; r(minutes, gain) = %.2f" %
      (RESULTS[-4][1], RESULTS[-3][1]))
print("  mean messages: gained %.1f vs not gained %.1f -> a usage dashboard cannot tell them apart."
      % (RESULTS[-2][1], RESULTS[-1][1]))

# --------------------------------------------------------------------------------------
# Step 2 -- the ladder on held-out learners
# --------------------------------------------------------------------------------------
y_tr = lr.loc[is_train, "gained"].to_numpy()
y_te = lr.loc[is_test, "gained"].to_numpy()


def logit_scores(cols):
    """Unpenalized logistic regression fit on train, linear predictor on test (same as R glm)."""
    Xtr = sm.add_constant(lr.loc[is_train, cols].to_numpy(), has_constant="add")
    Xte = sm.add_constant(lr.loc[is_test, cols].to_numpy(), has_constant="add")
    fit = sm.Logit(y_tr, Xtr).fit(disp=0, method="newton", maxiter=100, tol=1e-12)
    return Xte @ fit.params


def markov_logratio(Ncounts, n_rows, train_mask, y):
    """log P_gain(next|current) - log P_nogain(next|current), estimated on training learners."""
    g = Ncounts[train_mask][y == 1].sum(0).reshape(n_rows, 6)
    ng = Ncounts[train_mask][y == 0].sum(0).reshape(n_rows, 6)
    return np.log(chain(g)) - np.log(chain(ng)), chain(g), chain(ng)


# log-score(sequence) = sum over steps of log[ P_gain(next | current) / P_nogain(next | current) ]
LR1, P_gain_tr, P_nogain_tr = markov_logratio(N1, 6, is_train, y_tr)
LR2, _, _ = markov_logratio(N2, 36, is_train, y_tr)
lr["markov1_score"] = N1 @ LR1.ravel()
lr["markov2_score"] = N2 @ LR2.ravel()

scores_te = {
    "volume": logit_scores(["messages", "minutes", "sessions"]),
    "bag": logit_scores(["share_T", "share_H", "share_A", "share_C"]),
    "markov1": lr.loc[is_test, "markov1_score"].to_numpy(),
    "markov2": lr.loc[is_test, "markov2_score"].to_numpy(),
}

# (e) PYTHON ONLY: small MLP on 1-, 2- and 3-gram shares (fixed random_state).
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

grams = {}
for k in (1, 2, 3):
    toks = ev.groupby("session_id")["tok"].apply(list)
    for sid, seq in toks.items():
        seq = ["S"] + seq
        lid = idx[sid.split("_")[0]]
        for j in range(len(seq) - k + 1):
            g = "".join(seq[j:j + k])
            grams.setdefault(g, np.zeros(n))[lid] += 1
G = pd.DataFrame(grams)
G = G.loc[:, (G > 0).sum() >= 20]                       # keep n-grams seen for >= 20 learners
G = G.div(G.sum(axis=1).replace(0, 1), axis=0).to_numpy()
sc = StandardScaler().fit(G[is_train])
mlp = MLPClassifier(hidden_layer_sizes=(16,), alpha=1.0, max_iter=2000, random_state=SEED)
mlp.fit(sc.transform(G[is_train]), y_tr)
auc_mlp = auc(y_te, mlp.predict_proba(sc.transform(G[is_test]))[:, 1])

print("\nStep 2 (held-out AUC for predicting a gain of 10+ points; 0.50 = coin flip)")
labels = {"volume": "(a) usage volume", "bag": "(b) bag of actions", "markov1": "(c) 1st-order Markov",
          "markov2": "(d) 2nd-order Markov"}
auc_point = {}
for k, s in scores_te.items():
    auc_point[k] = auc(y_te, s)
    print(f"  {labels[k]:<24} AUC = {auc_point[k]:.2f}")
print(f"  (e) MLP on n-grams [python_only] AUC = {auc_mlp:.2f}")

# Bootstrap (resample held-out learners) for AUCs and the score-quartile gain gap.
te = lr[is_test].reset_index(drop=True)
order = np.lexsort((te["learner_id"].to_numpy(), te["markov1_score"].to_numpy()))  # ties: by id
q = len(te) // 4
quart = np.zeros(len(te), dtype=int)                    # 1 = bottom quartile, 4 = top quartile
quart[order[:q]] = 1
quart[order[-q:]] = 4
gain_te = te["gain"].to_numpy()
record("gain_top_quartile_markov", gain_te[quart == 4].mean(), 1)
record("gain_bottom_quartile_markov", gain_te[quart == 1].mean(), 1)
diff_q = gain_te[quart == 4].mean() - gain_te[quart == 1].mean()

rng = ParkMiller(SEED)
boot = {k: np.empty(B) for k in list(scores_te) + ["m1_minus_bag", "m1_minus_volume", "m2_minus_m1", "qdiff"]}
for b in range(B):
    ix = rng.index(len(te), len(te))
    yb = y_te[ix]
    a = {k: auc(yb, s[ix]) for k, s in scores_te.items()}
    for k in scores_te:
        boot[k][b] = a[k]
    boot["m1_minus_bag"][b] = a["markov1"] - a["bag"]
    boot["m1_minus_volume"][b] = a["markov1"] - a["volume"]
    boot["m2_minus_m1"][b] = a["markov2"] - a["markov1"]
    gb, qb = gain_te[ix], quart[ix]
    boot["qdiff"][b] = gb[qb == 4].mean() - gb[qb == 1].mean()

for k in scores_te:
    lo, hi = pct_interval(boot[k])
    record(f"auc_{k}", auc_point[k], 2)
    record(f"auc_{k}_lo", lo, 2)
    record(f"auc_{k}_hi", hi, 2)
for k, pt in [("m1_minus_bag", auc_point["markov1"] - auc_point["bag"]),
              ("m1_minus_volume", auc_point["markov1"] - auc_point["volume"]),
              ("m2_minus_m1", auc_point["markov2"] - auc_point["markov1"])]:
    lo, hi = pct_interval(boot[k])
    record(f"auc_diff_{k}", pt, 2)
    record(f"auc_diff_{k}_lo", lo, 2)
    record(f"auc_diff_{k}_hi", hi, 2)
lo, hi = pct_interval(boot["qdiff"])
record("gain_diff_top_vs_bottom_quartile", diff_q, 1)
record("gain_diff_top_vs_bottom_quartile_lo", lo, 1)
record("gain_diff_top_vs_bottom_quartile_hi", hi, 1)
print(f"  Held-out learners in the top quartile of the Markov log-score gained {diff_q:.1f} points more "
      f"than the bottom quartile (95% interval {lo:.1f} to {hi:.1f}).")

# Key transitions by gain group (all learners with a post-test), bootstrap by learner.
def key_transitions(Nsub, ysub):
    g = chain(Nsub[ysub == 1].sum(0).reshape(6, 6))
    ng = chain(Nsub[ysub == 0].sum(0).reshape(6, 6))
    cell = lambda P, r, c: P[ROWS.index(r), COLS.index(c)]
    return {"AC_gained": cell(g, "A", "C"), "AC_not": cell(ng, "A", "C"),
            "HT_gained": cell(g, "H", "T"), "HT_not": cell(ng, "H", "T"),
            "HA_gained": cell(g, "H", "A"), "HA_not": cell(ng, "H", "A"),
            "AT_gained": cell(g, "A", "T"), "AT_not": cell(ng, "A", "T"),
            "TV_gained": cell(g, "T", "V"), "TV_not": cell(ng, "T", "V")}


N_post = N1[has_post]
y_post = lr.loc[has_post, "gained"].to_numpy()
kt = key_transitions(N_post, y_post)
bt = {"AC": np.empty(B), "HT": np.empty(B)}
for b in range(B):
    ix = rng.index(len(y_post), len(y_post))
    k2 = key_transitions(N_post[ix], y_post[ix])
    bt["AC"][b] = k2["AC_not"] - k2["AC_gained"]
    bt["HT"][b] = k2["HT_gained"] - k2["HT_not"]
for k, v in kt.items():
    record(f"p_{k}", v, 2)
for nm, pt, key in [("p_AC_not_minus_gained", kt["AC_not"] - kt["AC_gained"], "AC"),
                    ("p_HT_gained_minus_not", kt["HT_gained"] - kt["HT_not"], "HT")]:
    lo, hi = pct_interval(bt[key])
    record(nm, pt, 2)
    record(nm + "_lo", lo, 2)
    record(nm + "_hi", hi, 2)
print(f"  P(copy | answer): not gained {kt['AC_not']:.2f} vs gained {kt['AC_gained']:.2f};"
      f"  P(attempt | hint): gained {kt['HT_gained']:.2f} vs not gained {kt['HT_not']:.2f}")

# Confounding check: the answer->copy loop is more common among low prior-skill learners.
pre = lr["pre_score"].to_numpy()
ordp = np.lexsort((np.arange(n), pre))
qp = n // 4
low_p, high_p = ordp[:qp], ordp[-qp:]
p_ac = lambda rows: chain(N1[rows].sum(0).reshape(6, 6))[ROWS.index("A"), COLS.index("C")]
record("p_AC_low_prior_quartile", p_ac(low_p), 2)
record("p_AC_high_prior_quartile", p_ac(high_p), 2)
print(f"  Confounding: P(copy | answer) is {p_ac(low_p):.2f} in the lowest pre-test quartile vs "
      f"{p_ac(high_p):.2f} in the highest -> style tracks prior skill; patterns are not causes.")

# --------------------------------------------------------------------------------------
# Step 3 -- ground-truth check
# --------------------------------------------------------------------------------------
with open(os.path.join(DATA, "ground_truth_params.json")) as f:
    truth = json.load(f)
gt = pd.read_csv(os.path.join(DATA, "ground_truth.csv"))
STY = truth["styles"]
TRUE = {s: np.array(truth["true_transition_matrices"][s]) for s in STY}
true_style = gt.set_index("learner_id").loc[learner_ids, "true_style"].to_numpy()

# 3a) Estimate each style's chain using the TRUE labels: pure estimation + mis-coding error.
mae_true_labels = np.mean([np.abs(chain(N1[true_style == s].sum(0).reshape(6, 6)) - TRUE[s]).mean() for s in STY])
record("mae_transition_true_labels", mae_true_labels, 3)

# 3b) Mixture of 3 Markov chains fit by EM, WITHOUT labels. Deterministic start from two
#     interpretable rules, then exactly 200 EM iterations (same in R).
def rate(r, c):
    num = N1[:, ROWS.index(r) * 6 + COLS.index(c)]
    den = N1[:, ROWS.index(r) * 6:(ROWS.index(r) + 1) * 6].sum(1)
    return np.where(den > 0, num / np.maximum(den, 1), 0.0)


init = np.full(n, 2)                                   # 2 = light/mixed
init[rate("H", "T") >= 0.5] = 0                        # 0 = scaffolded-like
init[rate("A", "C") >= 0.4] = 1                        # 1 = answer-seeking-like
R = np.zeros((n, 3))
R[np.arange(n), init] = 1.0
for it in range(200):
    pi = R.mean(0)
    logP = [np.log(chain((R[:, [k]] * N1).sum(0).reshape(6, 6))).ravel() for k in range(3)]
    ll = np.column_stack([np.log(pi[k]) + N1 @ logP[k] for k in range(3)])
    ll -= ll.max(1, keepdims=True)
    R = np.exp(ll)
    R /= R.sum(1, keepdims=True)
em_label = np.array(STY)[R.argmax(1)]
P_em = {s: chain((R[:, [k]] * N1).sum(0).reshape(6, 6)) for k, s in enumerate(STY)}
mae_em = np.mean([np.abs(P_em[s] - TRUE[s]).mean() for s in STY])
record("mae_transition_em", mae_em, 3)
record("style_accuracy_em", np.mean(em_label == true_style) * 100, 0)
for k, s in enumerate(STY):
    record(f"style_share_true_{s}", np.mean(true_style == s) * 100, 0)
    record(f"style_share_em_{s}", R[:, k].mean() * 100, 0)
print("\nStep 3 (ground truth)")
print(f"  mean absolute error of transition probabilities: {mae_true_labels:.3f} with true labels, "
      f"{mae_em:.3f} from the unlabeled 3-style mixture")
print(f"  mixture assigns {np.mean(em_label == true_style) * 100:.0f}% of learners to their true style; shares "
      + ", ".join(f"{s} {R[:, k].mean() * 100:.0f}% (true {np.mean(true_style == s) * 100:.0f}%)"
                  for k, s in enumerate(STY)))
# True styles vs gains (built in): mean gain by true style, among learners with a post-test.
for s in STY:
    record(f"mean_gain_true_{s}", lr.loc[has_post & (true_style == s), "gain"].mean(), 1)

# --------------------------------------------------------------------------------------
# Step 4 -- decision implication (printed; numbers above feed the scorecard template)
# --------------------------------------------------------------------------------------
print("\nStep 4 (decision): log action TYPES, not only volume; report P(attempt | hint) and "
      "P(copy | answer) by cohort; test tutor designs (hints before answers) with an experiment.")

# --------------------------------------------------------------------------------------
# Python-only rung + write results
# --------------------------------------------------------------------------------------
pd.DataFrame(RESULTS, columns=["name", "value", "digits"]).to_csv(os.path.join(OUT, "results_python.csv"), index=False)
with open(os.path.join(OUT, "results_python_only.json"), "w") as f:
    json.dump({"auc_ngram_mlp": {"value": round(auc_mlp, 2), "rounding": 2, "python_only": True,
                                 "meaning": "Held-out AUC of a small neural net (MLP, 16 hidden units) on 1-3-gram "
                                            "shares; stand-in for a neural sequence model; Python only."}}, f, indent=2)

# --------------------------------------------------------------------------------------
# Figures (phone-readable: 1500 px wide, large fonts, the finding as the title)
# --------------------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 20, "axes.titlesize": 24, "axes.labelsize": 20, "font.family": "DejaVu Sans"})
res = {r[0]: r[1] for r in RESULTS}

# Figure 1: held-out AUC by rung.
names = ["Usage volume\n(messages, minutes,\nsessions)", "Bag of actions\n(no order)", "1st-order Markov\n(order matters)",
         "2nd-order Markov\n(richer model)", "Neural net on n-grams\n(Python only)"]
vals = [res["auc_volume"], res["auc_bag"], res["auc_markov1"], res["auc_markov2"], auc_mlp]
los = [res["auc_volume_lo"], res["auc_bag_lo"], res["auc_markov1_lo"], res["auc_markov2_lo"], np.nan]
his = [res["auc_volume_hi"], res["auc_bag_hi"], res["auc_markov1_hi"], res["auc_markov2_hi"], np.nan]
colors = ["#9aa3ad", "#7fa7c9", "#1f5f99", "#1f5f99", "#b9c7d6"]
fig, ax = plt.subplots(figsize=(15, 9.5), dpi=100)
yy = np.arange(len(names))[::-1]
ax.barh(yy, np.array(vals) - 0.5, left=0.5, color=colors, height=0.62)
for i in range(4):
    ax.plot([los[i], his[i]], [yy[i], yy[i]], color="#222222", lw=3)
for i, v in enumerate(vals):
    x_text = (his[i] if not np.isnan(his[i]) else v) + 0.012
    ax.text(x_text, yy[i], f"{v:.2f}", va="center", fontsize=22, fontweight="bold")
ax.axvline(0.5, color="#555555", ls="--", lw=2)
ax.text(0.503, yy[0] + 0.48, "coin flip", fontsize=17, color="#555555", va="bottom")
ax.set_yticks(yy)
ax.set_yticklabels(names)
ax.set_xlim(0.45, 0.95)
ax.set_ylim(-0.6, len(names) - 0.2)
ax.set_xlabel("AUC on held-out learners (predicting a gain of 10+ points)")
fig.suptitle("Counting usage barely predicts learning;\nthe order of tutor actions does", fontsize=28,
             fontweight="bold", x=0.02, ha="left", y=0.985)
fig.text(0.02, 0.02, "Synthetic data, 1,200 learners; black lines = 95% bootstrap intervals.", fontsize=16,
         color="#555555")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.subplots_adjust(left=0.25, right=0.97, top=0.83, bottom=0.16)
fig.savefig(os.path.join(OUT, "fig1_order_beats_volume_auc.png"))
plt.close(fig)

# Figure 2: the two key loops by gain group.
trans = [("H", "T", "Hint →\nattempt"), ("A", "T", "Answer →\nre-attempt"),
         ("H", "A", "Hint →\nfull answer"), ("A", "C", "Answer →\ncopy")]
Pg = chain(N_post[y_post == 1].sum(0).reshape(6, 6))
Pn = chain(N_post[y_post == 0].sum(0).reshape(6, 6))
gv = [Pg[ROWS.index(a), COLS.index(b)] for a, b, _ in trans]
nv = [Pn[ROWS.index(a), COLS.index(b)] for a, b, _ in trans]
fig, ax = plt.subplots(figsize=(15, 9.5), dpi=100)
x = np.arange(len(trans))
w = 0.38
b1 = ax.bar(x - w / 2, gv, w, color="#1f5f99", label="Gained 10+ points")
b2 = ax.bar(x + w / 2, nv, w, color="#d9822b", label="Did not gain")
for bars in (b1, b2):
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{bar.get_height():.2f}",
                ha="center", fontsize=20, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels([t[2] for t in trans])
ax.set_ylabel("P(next action | current action)")
ax.set_ylim(0, max(gv + nv) + 0.16)
ax.legend(frameon=False, loc="upper left", fontsize=20, ncol=2)
fig.suptitle("Learners who gained turn hints and answers into new attempts;\nthe rest loop hint → answer → copy",
             fontsize=28, fontweight="bold", x=0.02, ha="left", y=0.985)
fig.text(0.02, 0.02, f"Synthetic data; first-order transition probabilities, {int(has_post.sum()):,} learners with a "
         "post-test.", fontsize=16, color="#555555")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
fig.subplots_adjust(left=0.09, right=0.98, top=0.83, bottom=0.2)
fig.savefig(os.path.join(OUT, "fig2_two_loops_by_gain_group.png"))
plt.close(fig)
print("\nWrote outputs/results_python.csv, outputs/results_python_only.json and 2 figures.")
