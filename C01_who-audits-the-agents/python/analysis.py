"""
analysis.py -- C01 "Who Audits the Agents? Human Oversight as a Measurable Capability"

Run from the code folder root:
    python3 data/generate_data.py      # (optional) regenerate the synthetic data
    Rscript r/analysis.R               # writes outputs/r_headline.csv
    python3 python/analysis.py         # writes results.json, figures, scorecard example

The business question: every AI decision has a human reviewer behind it. Is that
oversight actually working? We measure it on three things:
  Consistency  -- do reviewers agree with each other beyond chance?
  Correctness  -- do they agree with expert-verified ground truth (golden set)?
  Coverage     -- are the decisions that matter actually double-reviewed and checked?

Steps
  Step 1  The naive approach: raw percent agreement (and why it misleads).
  Step 2  The correct method: chance-corrected agreement (Cohen's kappa for a pair,
          Krippendorff's alpha for many reviewers with missing ratings).
  Step 3  Check against ground truth: sensitivity / specificity on the golden set,
          compared with the true parameters used to simulate the data.
  Step 4  Decision implication: how many errors slip through per site, and the scorecard.

Dependencies: numpy, pandas, matplotlib (standard library json/pathlib).
"""
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "oversight_reviews.csv"
TRUE = ROOT / "data" / "true_parameters.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
SITES = ["A", "B", "C"]
BOOT_SEED = 7
N_BOOT = 1000

# =============================================================================
# Step 0. Load and clean the review log
# =============================================================================
raw = pd.read_csv(DATA, dtype={"review_label": str, "gold_label": str}, keep_default_na=False)


def clean_label(x):
    """Map messy free-text labels to 'Correct' / 'Error'; anything else is missing."""
    x = str(x).strip().lower()
    if x.startswith("corr"):
        return "Correct"
    if x.startswith("err"):
        return "Error"
    return np.nan  # blank, 'Skipped', 'Unsure' -> no usable rating


df = raw.copy()
df["label"] = df["review_label"].map(clean_label)
n_rows_raw = len(df)
# The same reviewer should rate a decision once; drop double-submitted rows.
df = df.drop_duplicates(subset=["decision_id", "reviewer_id"], keep="first")
n_duplicates = n_rows_raw - len(df)
n_missing = int(df["label"].isna().sum())
df = df.dropna(subset=["label"])
df["is_error"] = (df["label"] == "Error").astype(int)
df["gold"] = df["gold_label"].replace("", np.nan)
# Golden-set cases were seeded blind into the queues with an enriched 25% error mix.
# Agreement statistics use live production cases only, so they reflect the real
# (low) AI error rate; the golden set is used only to check correctness.
gold = df[df["case_source"] == "golden_seeded"].copy()
df = df[df["case_source"] == "production"].copy()
print(f"Rows: {n_rows_raw} raw, {n_duplicates} duplicates dropped, {n_missing} unusable labels dropped.")


# =============================================================================
# Helper: Krippendorff's alpha for nominal data, written out so it can be read.
# =============================================================================
def krippendorff_alpha_nominal(data, unit="decision_id", value="label"):
    """
    Krippendorff's alpha for nominal data with any number of raters and missing ratings.

    1. Keep only units (decisions) with at least 2 ratings ("pairable" units).
    2. Build the coincidence matrix o[c, k]: within each unit with m ratings, every
       ordered pair of different ratings (i != j) adds 1 / (m - 1) to o[c_i, c_j].
    3. Observed disagreement D_o = share of pairable values that sit off the diagonal.
       Expected disagreement D_e = disagreement if the same values were paired at random.
    4. alpha = 1 - D_o / D_e  (1 = perfect, 0 = no better than chance, < 0 = worse).
    Returns alpha and the observed agreement (1 - D_o), which is the "percent agreement".
    """
    cats = sorted(data[value].unique())
    idx = {c: i for i, c in enumerate(cats)}
    o = np.zeros((len(cats), len(cats)))
    for _, g in data.groupby(unit):
        vals = g[value].tolist()
        m = len(vals)
        if m < 2:
            continue
        for i in range(m):
            for j in range(m):
                if i != j:
                    o[idx[vals[i]], idx[vals[j]]] += 1.0 / (m - 1)
    n = o.sum()
    n_c = o.sum(axis=1)
    d_o = 1 - np.trace(o) / n
    d_e = 1 - (n_c * (n_c - 1)).sum() / (n * (n - 1))
    return 1 - d_o / d_e, 1 - d_o


def alpha_bootstrap_ci(data, n_boot=N_BOOT, seed=BOOT_SEED):
    """95% percentile CI for alpha, resampling decisions (units) with replacement."""
    rng = np.random.default_rng(seed)
    groups = {u: g[["label"]].to_numpy().ravel() for u, g in data.groupby("decision_id")}
    groups = {u: v for u, v in groups.items() if len(v) >= 2}
    keys = np.array(list(groups))
    # Precompute each unit's contribution to the 2x2 coincidence matrix (Correct, Error).
    contrib = np.zeros((len(keys), 4))
    for k, u in enumerate(keys):
        v = (groups[u] == "Error").astype(int)
        m = len(v)
        e = v.sum()
        c = m - e
        # o_cc, o_ce, o_ec, o_ee contributions
        contrib[k] = [c * (c - 1), c * e, e * c, e * (e - 1)]
        contrib[k] /= (m - 1)
    stats = []
    for _ in range(n_boot):
        s = contrib[rng.integers(0, len(keys), len(keys))].sum(axis=0)
        o = s.reshape(2, 2)
        n = o.sum()
        n_c = o.sum(axis=1)
        d_o = 1 - np.trace(o) / n
        d_e = 1 - (n_c * (n_c - 1)).sum() / (n * (n - 1))
        stats.append(1 - d_o / d_e)
    return np.percentile(stats, [2.5, 97.5])


def cohen_kappa(a, b):
    """Cohen's kappa for two raters on the same cases (labels 'Correct'/'Error')."""
    a, b = np.asarray(a), np.asarray(b)
    p_o = np.mean(a == b)
    p_e = sum(np.mean(a == c) * np.mean(b == c) for c in ["Correct", "Error"])
    return (p_o - p_e) / (1 - p_e) if p_e < 1 else np.nan, p_o, p_e


# =============================================================================
# Step 1. The naive approach: raw percent agreement
# =============================================================================
# "Reviewers agree X% of the time." With only ~8% of AI decisions being errors,
# two reviewers who both say "Correct" almost every time will agree most of the
# time even if neither one is looking carefully. Raw agreement rewards that.
alpha_all, agree_all = krippendorff_alpha_nominal(df)
site_stats = {}
for s in SITES:
    a, po = krippendorff_alpha_nominal(df[df.site == s])
    site_stats[s] = {"alpha": a, "agree": po}
flag_rate = df.groupby("site")["is_error"].mean()
print(f"\nStep 1  Raw percent agreement, all sites: {agree_all:.1%}")
for s in SITES:
    print(f"        site {s}: {site_stats[s]['agree']:.1%} agreement; reviewers flag {flag_rate[s]:.1%} of decisions as errors")


# =============================================================================
# Step 2. The correct method: chance-corrected agreement
# =============================================================================
# kappa = (observed agreement - chance agreement) / (1 - chance agreement)
# Chance agreement is what two reviewers with the same flag rates would reach by
# labelling at random. When almost everything is "Correct", chance agreement is
# already ~85-90%, so 90% observed agreement is barely better than chance.

# 2a. Cohen's kappa for every reviewer pair that shares cases.
#     Reviewers only work their own site's queue, so most pairs share nothing.
pairs = []
wide = df.pivot_table(index="decision_id", columns="reviewer_id", values="label", aggfunc="first")
reviewer_site = df.drop_duplicates("reviewer_id").set_index("reviewer_id")["site"]
all_reviewers = sorted(reviewer_site.index)
for r1, r2 in combinations(all_reviewers, 2):
    both = wide[[r1, r2]].dropna()
    if len(both) == 0:
        continue
    k, po, pe = cohen_kappa(both[r1], both[r2])
    pairs.append({"r1": r1, "r2": r2, "site": reviewer_site[r1], "n_shared": len(both),
                  "kappa": k, "agree": po, "chance": pe})
pairs = pd.DataFrame(pairs)
n_possible_pairs = len(all_reviewers) * (len(all_reviewers) - 1) // 2
n_pairs_sharing = len(pairs)

# The "one frequent pair": the pair with the most shared decisions (ties -> first by id).
top = pairs.sort_values(["n_shared", "r1", "r2"], ascending=[False, True, True]).iloc[0]
mean_kappa = pairs["kappa"].mean()
mean_kappa_site = pairs.groupby("site")["kappa"].mean()
print(f"\nStep 2  {n_pairs_sharing} of {n_possible_pairs} reviewer pairs share any decisions.")
print(f"        Most frequent pair {top.r1}-{top.r2} (site {top.site}, n={top.n_shared}): "
      f"agreement {top.agree:.1%}, chance {top.chance:.1%}, kappa {top.kappa:.2f}")
print(f"        Average pairwise kappa: {mean_kappa:.2f} "
      + ", ".join(f"site {s} {mean_kappa_site[s]:.2f}" for s in SITES))

# 2b. Krippendorff's alpha: one number for all reviewers, missing ratings allowed.
alpha_ci = {"all": alpha_bootstrap_ci(df)}
for s in SITES:
    alpha_ci[s] = alpha_bootstrap_ci(df[df.site == s])
print(f"        Krippendorff's alpha, all sites: {alpha_all:.2f} "
      f"(95% CI {alpha_ci['all'][0]:.2f} to {alpha_ci['all'][1]:.2f})")
for s in SITES:
    print(f"        site {s}: alpha {site_stats[s]['alpha']:.2f} "
          f"(95% CI {alpha_ci[s][0]:.2f} to {alpha_ci[s][1]:.2f})")

# =============================================================================
# Step 3. Check against ground truth (golden set + true simulation parameters)
# =============================================================================
# Agreement is necessary but not sufficient: two reviewers can agree and both be
# wrong. On the golden set, compare each review with the expert label.
#   sensitivity = share of true errors the reviewer flagged ("error-catch rate")
#   specificity = share of truly correct decisions the reviewer passed
gold_site = {}
for s in SITES:
    g = gold[gold.site == s]
    te = g[g.gold == "Error"]
    tc = g[g.gold == "Correct"]
    k_sens, n_sens = int(te.is_error.sum()), len(te)
    k_spec, n_spec = int((tc.is_error == 0).sum()), len(tc)
    # Missed by both: true errors where every reviewer (2+) said "Correct".
    per_case = te.groupby("decision_id")["is_error"].agg(["sum", "count"])
    per_case = per_case[per_case["count"] >= 2]
    k_miss, n_miss = int((per_case["sum"] == 0).sum()), len(per_case)
    gold_site[s] = {
        "sens": k_sens / n_sens, "sens_k": k_sens, "sens_n": n_sens,
        "spec": k_spec / n_spec, "spec_k": k_spec, "spec_n": n_spec,
        "miss_both": k_miss / n_miss if n_miss else np.nan, "miss_k": k_miss, "miss_n": n_miss,
        "n_gold_cases": g.decision_id.nunique(),
        "n_gold_errors": te.decision_id.nunique(),
    }
gold_cases = gold.drop_duplicates("decision_id")
gold_error_rate = (gold_cases["gold"] == "Error").mean()
te_all = gold[gold.gold == "Error"].groupby("decision_id")["is_error"].agg(["sum", "count"])
te_all = te_all[te_all["count"] >= 2]
miss_all_k, miss_all_n = int((te_all["sum"] == 0).sum()), len(te_all)
sens_all_k = int(gold[gold.gold == "Error"].is_error.sum())
sens_all_n = int((gold.gold == "Error").sum())

# True values implied by the simulation parameters (what a perfect, infinite
# golden set would show). For alpha we compute the expected agreement implied by
# the reviewers' true sensitivity / specificity and the true error rate.
tp = pd.read_csv(TRUE)
prev = tp.query("parameter == 'prevalence'")["value"].iloc[0]
rv = tp.query("level == 'reviewer'").pivot(index="id", columns="parameter", values="value")
rv["site"] = tp.query("level == 'reviewer'").drop_duplicates("id").set_index("id")["site"]


def implied_truth(sub):
    """Expected agreement, alpha, and miss-by-both for a site's reviewers, with pairs drawn uniformly."""
    po, mb = [], []
    for i, j in combinations(sub.index, 2):
        si, sj, ci, cj = sub.at[i, "sensitivity"], sub.at[j, "sensitivity"], sub.at[i, "specificity"], sub.at[j, "specificity"]
        po.append(prev * (si * sj + (1 - si) * (1 - sj)) + (1 - prev) * (ci * cj + (1 - ci) * (1 - cj)))
        mb.append((1 - si) * (1 - sj))
    q = np.mean(prev * sub["sensitivity"] + (1 - prev) * (1 - sub["specificity"]))  # flag rate
    pe = q**2 + (1 - q) ** 2
    return {"agree": np.mean(po), "alpha": 1 - (1 - np.mean(po)) / (1 - pe), "miss_both": np.mean(mb),
            "sens": sub["sensitivity"].mean(), "spec": sub["specificity"].mean()}


truth = {s: implied_truth(rv[rv.site == s]) for s in SITES}

# The golden set is enriched, so it cannot tell us the live AI error rate directly.
# Estimate it from the live flag rate of the well-calibrated sites (A and B),
# correcting for their measured sensitivity and specificity (Rogan-Gladen):
#   error rate = (flag rate + specificity - 1) / (sensitivity + specificity - 1)
ab_flag = df[df.site.isin(["A", "B"])]["is_error"].mean()
ab_gold = gold[gold.site.isin(["A", "B"])]
ab_sens = ab_gold[ab_gold.gold == "Error"]["is_error"].mean()
ab_spec = 1 - ab_gold[ab_gold.gold == "Correct"]["is_error"].mean()
est_prev = (ab_flag + ab_spec - 1) / (ab_sens + ab_spec - 1)
# A reviewer who stamps every decision "Correct" would be right (1 - error rate)
# of the time while catching nothing.
approve_all_accuracy = 1 - est_prev
print(f"\nStep 3  Estimated live AI error rate: {est_prev:.1%} (true {prev:.0%}). A reviewer who says")
print(f"        'Correct' to everything would be right {approve_all_accuracy:.0%} of the time and catch nothing.")
print("        Golden set (expert labels) vs. reviewers, with the true simulated values:")
for s in SITES:
    gs = gold_site[s]
    print(f"        site {s}: caught {gs['sens_k']}/{gs['sens_n']} error reviews = {gs['sens']:.0%} "
          f"(true {truth[s]['sens']:.0%}); specificity {gs['spec']:.0%} (true {truth[s]['spec']:.0%}); "
          f"both reviewers missed {gs['miss_k']}/{gs['miss_n']} true errors (true rate {truth[s]['miss_both']:.0%}); "
          f"alpha {site_stats[s]['alpha']:.2f} (true {truth[s]['alpha']:.2f})")

# Optional cross-check against the 'krippendorff' package if it happens to be installed.
try:
    import krippendorff  # not required

    mat = wide.T.replace({"Correct": 0, "Error": 1}).astype(float).to_numpy()
    pkg_alpha = krippendorff.alpha(reliability_data=mat, level_of_measurement="nominal")
    print(f"        (cross-check: krippendorff package alpha = {pkg_alpha:.4f}, ours = {alpha_all:.4f})")
except Exception:
    pkg_alpha = None

# =============================================================================
# Step 4. Decision implication
# =============================================================================
# Errors that slip through = AI error rate x share of true errors that every reviewer missed.
# We use the estimated live error rate (the AI is the same everywhere) and each
# site's missed-by-both rate from the golden set. Expressed per 1,000 AI decisions.
slip = {s: 1000 * est_prev * gold_site[s]["miss_both"] for s in SITES}
slip_true = {s: 1000 * prev * truth[s]["miss_both"] for s in SITES}
print("\nStep 4  Estimated AI errors that pass review unflagged, per 1,000 decisions:")
for s in SITES:
    print(f"        site {s}: {slip[s]:.0f} (true {slip_true[s]:.0f})")


# Case-level bootstrap for the golden-set numbers and the per-1,000 estimates.
# Reviews are clustered within cases (each golden case is reviewed twice), so we
# resample CASES, not reviews: golden cases within each site, and live Site A+B
# decisions for the flag rate. Each resample recomputes sensitivity by site, the
# A+B-corrected live error rate, missed-by-both by site, and errors per 1,000.
def case_level_bootstrap(n_boot=2000, seed=11):
    rng = np.random.default_rng(seed)
    # live A+B decisions: number of reviews and number flagged 'Error' per decision
    live = df[df.site.isin(["A", "B"])].groupby("decision_id")["is_error"].agg(["sum", "count"]).to_numpy()
    # golden cases: site, truth, reviews, flags
    g = gold.groupby("decision_id").agg(site=("site", "first"), err=("gold", lambda x: x.iloc[0] == "Error"),
                                        k=("is_error", "sum"), n=("is_error", "size"))
    by_site = {s: g[g.site == s][["err", "k", "n"]].to_numpy(dtype=float) for s in SITES}
    keys = ["est_prev"] + [f"sens_{s}" for s in SITES] + [f"slip_{s}" for s in SITES]
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        lv = live[rng.integers(0, len(live), len(live))]
        flag = lv[:, 0].sum() / lv[:, 1].sum()
        rs = {s: by_site[s][rng.integers(0, len(by_site[s]), len(by_site[s]))] for s in SITES}
        ab = np.vstack([rs["A"], rs["B"]])
        e, c = ab[ab[:, 0] == 1], ab[ab[:, 0] == 0]
        sens_ab = e[:, 1].sum() / e[:, 2].sum()
        spec_ab = 1 - c[:, 1].sum() / c[:, 2].sum()
        p = (flag + spec_ab - 1) / (sens_ab + spec_ab - 1)
        draws["est_prev"].append(p)
        for s in SITES:
            te = rs[s][rs[s][:, 0] == 1]
            draws[f"sens_{s}"].append(te[:, 1].sum() / te[:, 2].sum())
            te2 = te[te[:, 2] >= 2]
            draws[f"slip_{s}"].append(1000 * p * np.mean(te2[:, 1] == 0))
    return {k: np.percentile(v, [2.5, 97.5]) for k, v in draws.items()}


boot = case_level_bootstrap()
print(f"        Case-level bootstrap 95% CIs: live error rate {boot['est_prev'][0]:.1%} to {boot['est_prev'][1]:.1%}; "
      + "; ".join(f"site {s} per 1,000 {boot[f'slip_{s}'][0]:.0f} to {boot[f'slip_{s}'][1]:.0f}" for s in SITES))

# Coverage: is every decision double-reviewed, and do high-value refunds get it?
cov = df.groupby("decision_id").agg(site=("site", "first"), n=("label", "size"),
                                    amount=("refund_amount_usd", "first"))
all_dec = raw[raw.case_source == "production"].drop_duplicates("decision_id")[["decision_id", "site", "refund_amount_usd"]].set_index("decision_id")
cov = all_dec.join(cov[["n"]]).fillna({"n": 0})
hv_cut = cov["refund_amount_usd"].quantile(0.90)
coverage = {}
for s in SITES:
    c = cov[cov.site == s]
    coverage[s] = {
        "double_review_rate": (c.n >= 2).mean(),
        "high_value_double_review_rate": (c[c.refund_amount_usd >= hv_cut].n >= 2).mean(),
        "gold_cases": gold_site[s]["n_gold_cases"],
        "gold_errors": gold_site[s]["n_gold_errors"],
    }
print(f"        Double-review rate: " + ", ".join(f"site {s} {coverage[s]['double_review_rate']:.1%}" for s in SITES))

# =============================================================================
# Figures (phone-readable: wide, large type, finding in the title)
# =============================================================================
GREY_DARK, GREY, GREY_LIGHT, BLUE = "#333333", "#9a9a9a", "#d9d9d9", "#2f6fd0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 17, "axes.edgecolor": GREY,
                     "axes.labelcolor": GREY_DARK, "xtick.color": GREY_DARK, "ytick.color": GREY_DARK})

# Figure 1: raw agreement vs. alpha by site
fig, ax = plt.subplots(figsize=(12, 7.2), dpi=120)
labels = ["All sites", "Site A", "Site B", "Site C"]
agree_vals = [agree_all] + [site_stats[s]["agree"] for s in SITES]
alpha_vals = [alpha_all] + [site_stats[s]["alpha"] for s in SITES]
x = np.arange(len(labels))
w = 0.38
b1 = ax.bar(x - w / 2, agree_vals, w, color=GREY_LIGHT, edgecolor=GREY, label="Raw agreement")
b2 = ax.bar(x + w / 2, alpha_vals, w, color=BLUE, label="Krippendorff's alpha (chance-corrected, 95% CI)")
ci_keys = ["all"] + SITES
ax.errorbar(x + w / 2, alpha_vals,
            yerr=[[alpha_vals[i] - alpha_ci[k][0] for i, k in enumerate(ci_keys)],
                  [alpha_ci[k][1] - alpha_vals[i] for i, k in enumerate(ci_keys)]],
            fmt="none", ecolor=GREY_DARK, capsize=6, lw=1.5)
for rect, v in zip(b1, agree_vals):
    ax.text(rect.get_x() + rect.get_width() / 2, v + 0.015, f"{v:.0%}", ha="center", va="bottom",
            color=GREY_DARK, fontsize=17)
for rect, v, k in zip(b2, alpha_vals, ci_keys):
    ax.text(rect.get_x() + rect.get_width() / 2, max(alpha_ci[k][1], 0) + 0.015, f"{v:.2f}", ha="center", va="bottom",
            color=BLUE, fontsize=17, fontweight="bold")
ax.set_xticks(x, labels)
ax.set_ylim(-0.1, 1.12)
ax.axhline(0, color=GREY, lw=1)
ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
ax.axhline(0.667, color=GREY, ls="--", lw=1.2)
ax.text(3.62, 0.68, "alpha 0.67\nminimum", color=GREY_DARK, fontsize=13, ha="right", va="bottom")
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=15)
fig.suptitle(f"Reviewers agree {agree_all:.0%} of the time, but beyond chance\n"
             f"agreement is weak overall (alpha {alpha_all:.2f}) and near zero at Site C ({site_stats['C']['alpha']:.2f})",
             fontsize=20, fontweight="bold", color=GREY_DARK, x=0.04, ha="left", y=0.98)
fig.text(0.04, 0.005, "Synthetic data: 1,500 AI refund decisions, 12 reviewers, 3 sites.",
         fontsize=12, color=GREY)
fig.subplots_adjust(left=0.08, right=0.97, top=0.83, bottom=0.18)
fig.savefig(OUT / "fig_1_agreement_vs_alpha.png")
plt.close(fig)

# Figure 2: error-catch rate (sensitivity) by site against the golden set
fig, ax = plt.subplots(figsize=(12, 6.6), dpi=120)
sens = [gold_site[s]["sens"] for s in SITES]
lo = [boot[f"sens_{s}"][0] for s in SITES]   # case-level bootstrap CI
hi = [boot[f"sens_{s}"][1] for s in SITES]
y = np.arange(len(SITES))[::-1]
colors = [GREY, GREY, BLUE]
ax.barh(y, sens, color=colors, height=0.55)
ax.errorbar(sens, y, xerr=[np.array(sens) - lo, np.array(hi) - sens], fmt="none", ecolor=GREY_DARK,
            capsize=6, lw=1.5)
for yi, s, v in zip(y, SITES, sens):
    gs = gold_site[s]
    ax.text(boot[f"sens_{s}"][1] + 0.02, yi, f"{v:.0%}  ({gs['sens_k']} of {gs['sens_n']} reviews)",
            va="center", fontsize=17, color=BLUE if s == "C" else GREY_DARK,
            fontweight="bold" if s == "C" else "normal")
ax.set_yticks(y, [f"Site {s}" for s in SITES])
ax.set_xlim(0, 1.45)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0], ["0%", "25%", "50%", "75%", "100%"])
ax.set_xlabel("Share of reviews of true AI errors labelled 'Error' (golden set, 95% CI)")
ax.spines[["top", "right"]].set_visible(False)
fig.suptitle(f"On expert-verified cases, Site C reviewers flagged {gold_site['C']['sens']:.0%} of their\n"
             f"reviews of true errors; Sites A and B flagged {gold_site['A']['sens']:.0%} and {gold_site['B']['sens']:.0%}",
             fontsize=20, fontweight="bold", color=GREY_DARK, x=0.04, ha="left", y=0.98)
fig.text(0.04, 0.005, f"Synthetic data. Golden set: 300 expert-verified cases seeded blind into queues "
         f"({int((gold_cases.gold == 'Error').sum())} true errors, 25 per site,\neach reviewed about twice). "
         f"Intervals: case-level bootstrap, because the two reviews of a case are not independent.", fontsize=12, color=GREY)
fig.subplots_adjust(left=0.1, right=0.97, top=0.82, bottom=0.23)
fig.savefig(OUT / "fig_2_error_catch_by_site.png")
plt.close(fig)

# =============================================================================
# results.json (every number the article may quote), with R values merged in
# =============================================================================
r_path = OUT / "r_headline.csv"
r_vals = pd.read_csv(r_path).set_index("name")["value"].to_dict() if r_path.exists() else {}


def rnd(v, k):
    """Round for reporting; whole numbers become int and -0.0 becomes 0.0."""
    v = round(float(v), k) + 0.0
    return int(v) if k == 0 else v


def item(name, value, rounding, meaning, unit="proportion", headline=False):
    d = {"name": name, "value": rnd(value, rounding), "rounding": rounding, "unit": unit,
         "meaning": meaning}
    if headline:
        d["python"] = rnd(value, rounding)
        rv_ = r_vals.get(name)
        d["r"] = None if rv_ is None or pd.isna(rv_) else rnd(rv_, rounding)
        d["python_r_match"] = d["r"] is not None and abs(d["python"] - d["r"]) < 10 ** (-rounding) / 2 + 1e-12
    return d


def ci_item(name, value, meaning, rounding=2, unit="alpha"):
    """Bootstrap bounds: Python and R use different random number generators, so the
    bounds can differ slightly; both are recorded, no exact match is expected."""
    d = item(name, value, rounding, meaning + " Python and R bootstrap RNGs differ, so bounds may differ slightly.", unit)
    d["python"] = d["value"]
    rv_ = r_vals.get("r_" + name)
    d["r"] = None if rv_ is None else rnd(rv_, rounding)
    return d


res = []
H = True
res += [
    item("percent_agreement_all", agree_all, 3, "Raw share of reviewer pairs on the same decision that gave the same label, all sites.", headline=H),
    *[item(f"percent_agreement_site_{s}", site_stats[s]["agree"], 3, f"Raw agreement among Site {s} reviewers.", headline=H) for s in SITES],
    *[item(f"reviewer_flag_rate_site_{s}", flag_rate[s], 3, f"Share of Site {s} reviews labelled 'Error'.") for s in SITES],
    item("kappa_top_pair", top.kappa, 2, f"Cohen's kappa for the most frequent pair, {top.r1}-{top.r2} (Site {top.site}, {top.n_shared} shared decisions).", "kappa", headline=H),
    item("kappa_top_pair_agreement", top.agree, 3, f"Raw agreement of pair {top.r1}-{top.r2}.", headline=H),
    item("kappa_top_pair_chance_agreement", top.chance, 3, f"Agreement pair {top.r1}-{top.r2} would reach by chance given their flag rates.", headline=H),
    item("kappa_top_pair_n_shared", top.n_shared, 0, "Decisions shared by the most frequent pair.", "count"),
    item("kappa_mean_pairwise_all", mean_kappa, 2, f"Average Cohen's kappa over the {n_pairs_sharing} reviewer pairs that share decisions.", "kappa", headline=H),
    *[item(f"kappa_mean_pairwise_site_{s}", mean_kappa_site[s], 2, f"Average pairwise kappa within Site {s}.", "kappa", headline=H) for s in SITES],
    item("reviewer_pairs_possible", n_possible_pairs, 0, "Possible reviewer pairs among 12 reviewers.", "count"),
    item("reviewer_pairs_sharing_cases", n_pairs_sharing, 0, "Reviewer pairs that share at least one decision (the rest share none, so pairwise kappa cannot be computed).", "count"),
    item("alpha_all", alpha_all, 2, "Krippendorff's alpha (nominal), all reviewers and sites; 1 = perfect, 0 = chance.", "alpha", headline=H),
    ci_item("alpha_all_ci_low", alpha_ci["all"][0], "Bootstrap 95% CI lower bound for overall alpha (1,000 resamples of decisions)."),
    ci_item("alpha_all_ci_high", alpha_ci["all"][1], "Bootstrap 95% CI upper bound for overall alpha."),
]
for s in SITES:
    res += [
        item(f"alpha_site_{s}", site_stats[s]["alpha"], 2, f"Krippendorff's alpha within Site {s}.", "alpha", headline=H),
        ci_item(f"alpha_site_{s}_ci_low", alpha_ci[s][0], f"Bootstrap 95% CI lower bound, Site {s} alpha."),
        ci_item(f"alpha_site_{s}_ci_high", alpha_ci[s][1], f"Bootstrap 95% CI upper bound, Site {s} alpha."),
    ]
res += [
    item("gold_set_cases", len(gold_cases), 0, "Expert-adjudicated decisions seeded blind into review queues (100 per site).", "count"),
    item("gold_set_true_errors", int((gold_cases.gold == "Error").sum()), 0, "True AI errors in the golden set (enriched to 25% by design).", "count"),
    item("estimated_live_error_rate", est_prev, 3, "Live AI error rate estimated from Sites A+B flag rate corrected for their gold sensitivity/specificity (true value 0.08; realized 0.079). Overestimated in this seed because golden-set specificity (0.98-0.99) runs above the true ~0.95.", headline=H),
    ci_item("estimated_live_error_rate_ci_low", boot["est_prev"][0], "95% CI lower bound, live error rate: case-level bootstrap.", 3, "proportion"),
    ci_item("estimated_live_error_rate_ci_high", boot["est_prev"][1], "95% CI upper bound, live error rate: case-level bootstrap.", 3, "proportion"),
    item("approve_everything_accuracy", approve_all_accuracy, 3, "Share of live decisions a reviewer who says 'Correct' to everything would get right, while catching zero errors.", headline=H),
    item("sensitivity_all", sens_all_k / sens_all_n, 2, "Share of true-error reviews (golden set) labelled 'Error', all sites.", headline=H),
]
for s in SITES:
    gs = gold_site[s]
    res += [
        item(f"sensitivity_site_{s}", gs["sens"], 2, f"Site {s} error-catch rate: share of reviews of true errors labelled 'Error' ({gs['sens_k']} of {gs['sens_n']}).", headline=H),
        ci_item(f"sensitivity_site_{s}_ci_low", boot[f"sens_{s}"][0], f"95% CI lower bound, Site {s} sensitivity: case-level bootstrap (2,000 resamples of golden cases; reviews are clustered within cases).", 2, "proportion"),
        ci_item(f"sensitivity_site_{s}_ci_high", boot[f"sens_{s}"][1], f"95% CI upper bound, Site {s} sensitivity: case-level bootstrap.", 2, "proportion"),
        item(f"specificity_site_{s}", gs["spec"], 2, f"Site {s}: share of reviews of truly correct decisions labelled 'Correct' ({gs['spec_k']} of {gs['spec_n']}).", headline=H),
        item(f"missed_by_both_site_{s}", gs["miss_both"], 2, f"Site {s}: share of golden-set true errors where every reviewer said 'Correct' ({gs['miss_k']} of {gs['miss_n']}).", headline=H),
        item(f"errors_slipping_per_1000_site_{s}", slip[s], 0, f"Estimated AI errors per 1,000 decisions that pass Site {s} review unflagged (estimated live error rate x missed-by-both rate). Biased upward in this seed because the live error rate is overestimated (true value in true_parameters.site_implied_errors_slipping_per_1000).", "per 1,000 decisions", headline=H),
        ci_item(f"errors_slipping_per_1000_site_{s}_ci_low", boot[f"slip_{s}"][0], f"95% CI lower bound, Site {s} errors per 1,000: case-level bootstrap.", 0, "per 1,000 decisions"),
        ci_item(f"errors_slipping_per_1000_site_{s}_ci_high", boot[f"slip_{s}"][1], f"95% CI upper bound, Site {s} errors per 1,000: case-level bootstrap."
                + (" With 0 golden errors missed by both reviewers, every resample gives 0, so the interval collapses to 0-0; the true rate is small but not zero." if gold_site[s]["miss_k"] == 0 else ""), 0, "per 1,000 decisions"),
        item(f"double_review_rate_site_{s}", coverage[s]["double_review_rate"], 3, f"Coverage: share of Site {s} decisions with 2+ usable reviews."),
        item(f"high_value_double_review_rate_site_{s}", coverage[s]["high_value_double_review_rate"], 3, f"Coverage: share of Site {s} top-10%-by-amount refunds with 2+ usable reviews."),
        item(f"gold_errors_site_{s}", gs["n_gold_errors"], 0, f"True errors in Site {s}'s golden set (small: precision is limited).", "count"),
    ]
res += [
    item("missed_by_both_all", miss_all_k / miss_all_n, 2, f"All sites: share of golden-set true errors missed by every reviewer ({miss_all_k} of {miss_all_n}).", headline=H),
    item("rows_raw", n_rows_raw, 0, "Review rows in the raw log.", "count"),
    item("rows_duplicate_dropped", n_duplicates, 0, "Double-submitted review rows dropped.", "count"),
    item("rows_unusable_label_dropped", n_missing, 0, "Reviews with blank/'Skipped'/'Unsure' labels dropped.", "count"),
]

true_params = {
    "prevalence_design": prev,
    "site_design": {s: {"sensitivity": float(tp.query(f"level=='site_design' and site=='{s}' and parameter=='sensitivity'")['value'].iloc[0]),
                        "specificity": float(tp.query(f"level=='site_design' and site=='{s}' and parameter=='specificity'")['value'].iloc[0])}
                    for s in SITES},
    "site_implied": {s: {k: round(float(v), 3) for k, v in truth[s].items()} for s in SITES},
    "site_implied_errors_slipping_per_1000": {s: round(slip_true[s], 1) for s in SITES},
    "reviewers": {i: {"site": rv.at[i, "site"], "sensitivity": float(rv.at[i, "sensitivity"]),
                      "specificity": float(rv.at[i, "specificity"])} for i in rv.index},
    "note": "site_implied = what an infinitely large sample would show, computed from the 4 reviewers' true sensitivity/specificity and the true 8% error rate (reviewer pairs drawn uniformly within site).",
}

headline = [d for d in res if "python_r_match" in d]
results = {
    "topic": "C01 Who Audits the Agents? Human Oversight as a Measurable Capability",
    "data": "Synthetic. 1,500 live AI refund decisions (true error rate 8%), 3 sites x 4 reviewers, 2 reviewers per decision (3% escalated to 3), plus a golden set of 300 expert-verified cases (25% errors) seeded blind into the queues. Seed 20261101.",
    "how_to_reproduce": "python3 data/generate_data.py; Rscript r/analysis.R; python3 python/analysis.py",
    "python_r_all_headline_match": bool(r_vals) and all(d["python_r_match"] for d in headline),
    "results": res,
    "true_parameters": true_params,
    "krippendorff_package_crosscheck_alpha_all": None if pkg_alpha is None else round(float(pkg_alpha), 4),
    "figures": ["outputs/fig_1_agreement_vs_alpha.png", "outputs/fig_2_error_catch_by_site.png"],
}
with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

# Example scorecard filled with this synthetic data (the blank template lives in templates/).
targets = pd.read_csv(ROOT / "templates" / "oversight_scorecard.csv")
lookup = {
    "krippendorff_alpha": {s: site_stats[s]["alpha"] for s in SITES},
    "percent_agreement": {s: site_stats[s]["agree"] for s in SITES},
    "mean_pairwise_kappa": {s: mean_kappa_site[s] for s in SITES},
    "sensitivity_vs_gold": {s: gold_site[s]["sens"] for s in SITES},
    "specificity_vs_gold": {s: gold_site[s]["spec"] for s in SITES},
    "missed_by_all_reviewers": {s: gold_site[s]["miss_both"] for s in SITES},
    "double_review_rate": {s: coverage[s]["double_review_rate"] for s in SITES},
    "high_value_double_review_rate": {s: coverage[s]["high_value_double_review_rate"] for s in SITES},
    "golden_set_true_errors": {s: coverage[s]["gold_errors"] for s in SITES},
}
rows = []
for t in targets.itertuples(index=False):
    for s in SITES:
        v = lookup.get(t.metric_id, {}).get(s, np.nan)
        if pd.isna(v):
            status = "n/a"
        elif t.metric_id == "krippendorff_alpha":
            # Alpha is read together with the golden-set catch rate (see template note):
            # with rare errors, low alpha alone does not mean the reviewers are poor.
            sens_ok = gold_site[s]["sens"] >= targets.set_index("metric_id").at["sensitivity_vs_gold", "target"]
            status = ("meets target" if v >= t.target else "tentative" if v >= t.red_flag else
                      "investigate: sensitivity meets target, low alpha likely reflects rare errors" if sens_ok else
                      "investigate: sensitivity also below target")
        elif t.direction == "higher_is_better":
            status = "meets target" if v >= t.target else ("red flag" if v < t.red_flag else "watch")
        elif t.direction == "lower_is_better":
            status = "meets target" if v <= t.target else ("red flag" if v > t.red_flag else "watch")
        else:
            status = "context only"
        rows.append({"dimension": t.dimension, "metric_id": t.metric_id, "site": s,
                     "value": round(float(v), 3) if not pd.isna(v) else None,
                     "target": t.target, "red_flag": t.red_flag, "status": status})
pd.DataFrame(rows).to_csv(OUT / "scorecard_example.csv", index=False)

print(f"\nWrote outputs/results.json, 2 figures, outputs/scorecard_example.csv.")
if r_vals:
    bad = [d["name"] for d in headline if not d["python_r_match"]]
    print("Python vs R headline numbers: " + ("all match." if not bad else f"MISMATCH in {bad}"))
else:
    print("R results not found (run `Rscript r/analysis.R` first to fill the 'r' keys).")
