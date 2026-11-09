"""
D01 - Measuring AI Supervision Skills: reliability and validity evidence for a skills test.

Run from anywhere:  python python/analysis.py
Reads  data/ai_supervision_assessment.csv  (+ the ground-truth files, used only in Step 3)
Writes outputs/results.json, outputs/python_item_statistics.csv, outputs/fig1_*.png, outputs/fig2_*.png

Steps
  0. Screen the data (speeders, label typos, missing responses).
  1. The naive approach: one Cronbach's alpha on all 16 items -> "reliable, certify people".
  2. The correct approach: item analysis, separate scales, alpha AND omega, validity against an
     independent work sample, the standard error of measurement (SEM) around the cutoff,
     and the Spearman-Brown projection.
  3. Check every estimate against the known ground truth.
  4. The decision implication.

Missing-data rule (identical in R):
  * Reliability and item statistics use the PAIRWISE-complete covariance matrix of the items
    (every available answer is used; nobody is dropped for one skipped item).
  * A person's scale score is PRORATED: mean of answered items x number of items, if the person
    answered at least 75% of that scale's items (ceil(0.75 k)); otherwise the score is missing.
"""
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.multivariate.factor import Factor

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)

S_ITEMS = [f"S{i:02d}" for i in range(1, 13)]   # scenario items, scored 0-4
C_ITEMS = [f"C{i:02d}" for i in range(1, 5)]    # self-report confidence, 0-10
CUTOFF = 26           # proposed certification cutoff on the cleaned 10-item scenario score (0-40)
TARGET_HIGH_STAKES = 0.90
SPEED_MIN = 8         # minutes; nobody can read 16 scenarios faster than this
FLAG_R = 0.20         # item-rest correlation below this -> drop/fix; 0.20-0.30 -> revise
CANON_UNITS = ["Trust & Safety Review", "Operations", "Claims", "Customer Support"]

R = {}  # every number the article may quote: name -> (value, rounding, meaning)


def put(name, value, rounding, meaning):
    R[name] = (value, rounding, meaning)


# ------------------------------------------------------------------------------------------
# Small, readable building blocks (each is one formula)
# ------------------------------------------------------------------------------------------
def cov_pairwise(df):
    """Pairwise-complete covariance matrix (same as R's cov(use = 'pairwise.complete.obs'))."""
    return df.cov()


def alpha_from_cov(c):
    """Cronbach's alpha = k/(k-1) * (1 - sum of item variances / variance of the total)."""
    c = np.asarray(c)
    k = c.shape[0]
    return k / (k - 1) * (1 - np.trace(c) / c.sum())


def omega_from_cov(c, nobs):
    """McDonald's omega (total) from a one-factor maximum-likelihood model.

    Omega lets each item reflect the skill with a different strength (loading).
    omega = (sum of loadings)^2 / [(sum of loadings)^2 + sum of unique variances],
    with loadings and unique variances in the items' raw units, so it describes the
    unit-weighted sum score that people actually receive.
    """
    c = np.asarray(c)
    sd = np.sqrt(np.diag(c))
    corr = c / np.outer(sd, sd)
    fa = Factor(corr=corr, n_factor=1, method="ml", nobs=nobs).fit()
    lam = np.real(np.asarray(fa.loadings)).ravel()  # statsmodels may return a complex dtype with zero imaginary part
    if lam.sum() < 0:          # the sign of a factor is arbitrary; make "more skill" positive
        lam = -lam
    lam_raw = lam * sd
    uniq_raw = (1 - lam**2) * sd**2
    return lam_raw.sum() ** 2 / (lam_raw.sum() ** 2 + uniq_raw.sum()), lam


def item_stats(c, items):
    """Item-rest correlation and alpha-if-item-deleted, computed from the covariance matrix.

    item-rest r: correlation between the item and the sum of the OTHER items. A good item
    goes up when the rest of the test goes up. Near zero = the item measures something else
    (or nothing); negative = it is probably mis-keyed.
    """
    c = np.asarray(c)
    rows = []
    for i, it in enumerate(items):
        rest = [j for j in range(len(items)) if j != i]
        cov_item_rest = c[i, rest].sum()
        var_rest = c[np.ix_(rest, rest)].sum()
        r_ir = cov_item_rest / math.sqrt(c[i, i] * var_rest)
        rows.append({"item": it, "item_rest_r": r_ir,
                     "alpha_if_deleted": alpha_from_cov(c[np.ix_(rest, rest)])})
    return pd.DataFrame(rows)


def decision(r_ir, alpha_del, alpha_all):
    if r_ir < FLAG_R:
        return "drop" if r_ir >= 0 else "drop (check key)"
    if r_ir < 0.30 or alpha_del > alpha_all:
        return "revise"
    return "keep"


def prorated(df, items):
    """Prorated sum score; missing if fewer than ceil(0.75 k) items were answered."""
    k = len(items)
    answered = df[items].notna().sum(axis=1)
    score = df[items].mean(axis=1) * k
    return score.where(answered >= math.ceil(0.75 * k))


def fisher_ci(r, n):
    z, se = np.arctanh(r), 1 / math.sqrt(n - 3)
    return math.tanh(z - 1.96 * se), math.tanh(z + 1.96 * se)


def zou_diff_ci(r1, r2, r12, n):
    """95% CI for r1 - r2 when both correlations share the criterion (Zou, 2007)."""
    l1, u1 = fisher_ci(r1, n)
    l2, u2 = fisher_ci(r2, n)
    c = ((r12 - r1 * r2 / 2) * (1 - r1**2 - r2**2 - r12**2) + r12**3) / ((1 - r1**2) * (1 - r2**2))
    d = r1 - r2
    lo = d - math.sqrt((r1 - l1) ** 2 + (u2 - r2) ** 2 - 2 * c * (r1 - l1) * (u2 - r2))
    hi = d + math.sqrt((u1 - r1) ** 2 + (r2 - l2) ** 2 - 2 * c * (u1 - r1) * (r2 - l2))
    return d, lo, hi


def spearman_brown_items(rel, k, target):
    """Items needed to reach `target` if we add items as good as the current ones."""
    factor = target * (1 - rel) / (rel * (1 - target))
    return math.ceil(k * factor)


# ------------------------------------------------------------------------------------------
# Step 0. Screen the data
# ------------------------------------------------------------------------------------------
raw = pd.read_csv(DATA / "ai_supervision_assessment.csv")
put("n_records", len(raw), 0, "Rows in the raw file")

# Fix business-unit typos: match on trimmed lower-case text.
lookup = {u.lower(): u for u in CANON_UNITS}
key = raw["business_unit"].str.strip().str.lower()
put("n_unit_labels_fixed", int((raw["business_unit"] != key.map(lookup)).sum()), 0,
    "Business-unit labels corrected (case or spaces)")
raw["business_unit"] = key.map(lookup)

# Speeder rule: finished in under 8 minutes OR gave the same answer to every answered scenario item.
answered_s = raw[S_ITEMS].notna().sum(axis=1)
straight = (raw[S_ITEMS].nunique(axis=1) == 1) & (answered_s >= 9)
speeder = (raw["completion_minutes"] < SPEED_MIN) | straight
put("n_speeders_flagged", int(speeder.sum()), 0, "Records excluded as speeders (<8 min or straight-lined)")
df = raw.loc[~speeder].reset_index(drop=True)
n = len(df)
put("n_analyzed", n, 0, "Test takers kept for analysis")
put("pct_item_responses_missing", 100 * df[S_ITEMS + C_ITEMS].isna().to_numpy().mean(), 1,
    "Percent of item responses left blank (handled pairwise / by prorating)")

# Criterion: share of seeded errors caught in the independent audit (two parallel sets a, b).
df["audit_pct_caught"] = (df["audit_caught_a"] + df["audit_caught_b"]) / (df["audit_seeded_a"] + df["audit_seeded_b"])
df["audit_a"] = df["audit_caught_a"] / df["audit_seeded_a"]
df["audit_b"] = df["audit_caught_b"] / df["audit_seeded_b"]
put("n_with_audit", int(df["audit_pct_caught"].notna().sum()), 0, "Test takers with the audit (criterion) score")

SCALES = {"naive16": S_ITEMS + C_ITEMS, "scenario12": S_ITEMS}

# ------------------------------------------------------------------------------------------
# Step 1. The naive approach: one alpha for all 16 items
# ------------------------------------------------------------------------------------------
cov16 = cov_pairwise(df[S_ITEMS + C_ITEMS])
alpha16 = alpha_from_cov(cov16)
put("alpha_naive16", alpha16, 2, "Cronbach's alpha of all 16 items (the naive 'the test is reliable' number)")
put("alpha_scenario12", alpha_from_cov(cov_pairwise(df[S_ITEMS])), 2,
    "Alpha of the 12 scenario items alone, before fixing anything")
print(f"Step 1 (naive): alpha for all 16 items = {alpha16:.2f}. Looks 'reliable'. "
      f"The 12 skill scenarios alone: {R['alpha_scenario12'][0]:.2f}. The confidence items push alpha up.")

# ------------------------------------------------------------------------------------------
# Step 2a. Item analysis on the 12 scenario items (and on all 16, to show what it can't see)
# ------------------------------------------------------------------------------------------
cov12 = cov_pairwise(df[S_ITEMS])
alpha12 = alpha_from_cov(cov12)
ist = item_stats(cov12, S_ITEMS)
ist.insert(1, "n", df[S_ITEMS].notna().sum().to_numpy())
ist.insert(2, "mean", df[S_ITEMS].mean().to_numpy())
ist["decision"] = [decision(r, a, alpha12) for r, a in zip(ist["item_rest_r"], ist["alpha_if_deleted"])]
ist16 = item_stats(cov16, S_ITEMS + C_ITEMS).rename(columns={"item_rest_r": "item_rest_r_in_16",
                                                            "alpha_if_deleted": "alpha_if_deleted_in_16"})
table = ist16.merge(ist, on="item", how="left")
table["n"] = table["n"].astype("Int64")
table.round(3).to_csv(OUT / "python_item_statistics.csv", index=False)

flagged = ist.loc[ist["item_rest_r"] < FLAG_R, "item"].tolist()
CLEAN = [s for s in S_ITEMS if s not in flagged]
SCALES["scenario10"] = CLEAN
SCALES["conf4"] = C_ITEMS
put("n_items_flagged", len(flagged), 0, "Scenario items with item-rest r < 0.20")
put("items_flagged", ";".join(flagged), None, "Which items were flagged")
for it in ["S06", "S11"]:
    row = ist.set_index("item").loc[it]
    put(f"item_rest_r_{it}", row["item_rest_r"], 2, f"Item-rest correlation of {it} within the 12 scenario items")
    put(f"alpha_if_deleted_{it}", row["alpha_if_deleted"], 2, f"Alpha of the 12-item scenario scale if {it} is deleted")
good = ist[~ist["item"].isin(flagged)]
put("item_rest_r_min_kept", good["item_rest_r"].min(), 2, "Lowest item-rest r among the 10 kept scenario items")
put("item_rest_r_max_kept", good["item_rest_r"].max(), 2, "Highest item-rest r among the 10 kept scenario items")
conf_in16 = ist16[ist16["item"].isin(C_ITEMS)]["item_rest_r_in_16"]
put("item_rest_r_conf_min_in16", conf_in16.min(), 2,
    "Lowest item-rest r of a confidence item inside the 16-item total (they pass the usual screen)")

# Step 2b. Internal structure: do the items hang together the way we intended?
corr16 = cov16.to_numpy() / np.outer(np.sqrt(np.diag(cov16)), np.sqrt(np.diag(cov16)))
idx = {it: i for i, it in enumerate(S_ITEMS + C_ITEMS)}


def mean_r(a, b):
    vals = [corr16[idx[x], idx[y]] for x in a for y in b if x != y]
    return float(np.mean(vals))


put("mean_r_within_scenario10", mean_r(CLEAN, CLEAN), 2, "Average correlation between two kept scenario items")
put("mean_r_within_conf4", mean_r(C_ITEMS, C_ITEMS), 2, "Average correlation between two confidence items (near-duplicates)")
put("mean_r_between_blocks", mean_r(CLEAN, C_ITEMS), 2, "Average correlation between a scenario item and a confidence item")

# Step 2c. Reliability of each scale, separately: alpha and omega
for name in ["scenario12", "scenario10", "conf4"]:
    c = cov_pairwise(df[SCALES[name]])
    nobs = int(df[SCALES[name]].dropna().shape[0])  # only used for the ML fit's sample-size term
    om, lam = omega_from_cov(c, nobs)
    if name != "scenario12":
        put(f"alpha_{name}", alpha_from_cov(c), 2, f"Cronbach's alpha of the {name} scale")
    put(f"omega_{name}", om, 2, f"McDonald's omega (one-factor ML) of the {name} scale")
    if name == "scenario10":
        loadings10 = lam
put("loading_min_scenario10", loadings10.min(), 2, "Smallest standardized loading among the kept scenario items")
put("loading_max_scenario10", loadings10.max(), 2, "Largest standardized loading among the kept scenario items")

# Step 2d. Validity: does the score predict the independent audit of AI outputs?
for name, items in SCALES.items():
    df[f"score_{name}"] = prorated(df, items)
vcols = [f"score_{s}" for s in SCALES] + ["audit_pct_caught", "audit_a", "audit_b"]
v = df.dropna(subset=vcols)
nv = len(v)
put("n_validity_sample", nv, 0, "People with all scale scores and the audit (validity sample)")
for name in SCALES:
    r = v[f"score_{name}"].corr(v["audit_pct_caught"])
    lo, hi = fisher_ci(r, nv)
    put(f"validity_{name}", r, 2, f"Correlation of the {name} score with % of seeded errors caught")
    put(f"validity_{name}_ci_low", lo, 2, "95% CI lower bound (Fisher z)")
    put(f"validity_{name}_ci_high", hi, 2, "95% CI upper bound (Fisher z)")
r12 = v["score_scenario10"].corr(v["score_naive16"])
d, lo, hi = zou_diff_ci(R["validity_scenario10"][0], R["validity_naive16"][0], r12, nv)
put("validity_gain_scenario10_vs_naive16", d, 2, "Validity of the cleaned scale minus validity of the naive total")
put("validity_gain_ci_low", lo, 2, "95% CI lower bound of the gain (Zou 2007, overlapping correlations)")
put("validity_gain_ci_high", hi, 2, "95% CI upper bound of the gain")

# Side note: the audit itself is not perfectly reliable, which shrinks every validity coefficient.
r_ab = v["audit_a"].corr(v["audit_b"])
rel_crit = 2 * r_ab / (1 + r_ab)  # Spearman-Brown: two halves -> full 40-error audit
put("criterion_reliability", rel_crit, 2, "Split-half reliability of the audit score (Spearman-Brown corrected)")
# Downstream formulas use the omega AS REPORTED (2 decimals), so readers can reproduce them by hand.
OM10 = round(R["omega_scenario10"][0], 2)
put("validity_scenario10_corrected", R["validity_scenario10"][0] / math.sqrt(OM10 * rel_crit), 2,
    "Validity corrected for unreliability in both the test and the audit (side note, not for decisions)")

# Step 2e. Precision for individuals: SEM and the band around the cutoff
s10 = df["score_scenario10"].dropna()
sd10 = s10.std(ddof=1)
sem = sd10 * math.sqrt(1 - OM10)
band = 1.96 * sem
put("mean_scenario10", s10.mean(), 1, "Mean cleaned scenario score (0-40)")
put("sd_scenario10", sd10, 1, "SD of the cleaned scenario score (0-40)")
put("sem_scenario10", sem, 1, "Standard error of measurement = SD x sqrt(1 - omega), in score points")
put("cutoff_scenario10", CUTOFF, 0, "Proposed certification cutoff (pass if score >= 26 of 40)")
put("band_low", CUTOFF - band, 1, "Lower edge of the cutoff +/- 1.96 SEM band")
put("band_high", CUTOFF + band, 1, "Upper edge of the cutoff +/- 1.96 SEM band")
put("pct_pass", 100 * (s10 >= CUTOFF).mean(), 0, "Percent who pass at the cutoff")
in_band = (s10 - CUTOFF).abs() < band
put("pct_within_band", 100 * in_band.mean(), 0,
    "Percent whose score is within 1.96 SEM of the cutoff (cannot be confidently classified)")
put("pct_within_1sem", 100 * ((s10 - CUTOFF).abs() < sem).mean(), 0, "Percent within 1 SEM of the cutoff")

# Step 2f. Spearman-Brown: how long would the test need to be for high-stakes use?
need = spearman_brown_items(OM10, len(CLEAN), TARGET_HIGH_STAKES)
put("sb_items_needed_090", need, 0, "Scenario items needed for reliability 0.90 (Spearman-Brown, items as good as the kept ones)")
put("sb_items_to_add_090", need - len(CLEAN), 0, "Good scenario items to add to reach 0.90")
f = 12 / 10
sb12 = f * OM10 / (1 + (f - 1) * OM10)
put("sb_reliability_if_two_good_items", sb12, 2, "Projected reliability if S06 and S11 were rewritten as good items (12 items)")

# Step 2g. Reliability is a property of scores in a group: check each business unit
for u in CANON_UNITS:
    sub = df[df["business_unit"] == u]
    tag = u.lower().replace(" & ", "_").replace(" ", "_")
    put(f"alpha_scenario10_{tag}", alpha_from_cov(cov_pairwise(sub[CLEAN])), 2, f"Alpha of the cleaned scale in {u}")
    put(f"sd_scenario10_{tag}", sub["score_scenario10"].std(ddof=1), 1, f"Score SD in {u}")

# ------------------------------------------------------------------------------------------
# Step 3. Check against the ground truth (possible only because the data are synthetic)
# ------------------------------------------------------------------------------------------
truth = pd.read_csv(DATA / "true_parameters.csv").set_index("parameter")["value"]
persons = pd.read_csv(DATA / "true_person_values.csv")
true_flawed = truth["flawed_items"].split(";")
put("flagged_items_match_truth", int(sorted(flagged) == sorted(true_flawed)), 0, "1 = flagged items are exactly the flawed ones")
spe = raw.loc[speeder, "person_id"]
put("speeders_match_truth", int(set(spe) == set(persons.loc[persons["is_speeder"] == 1, "person_id"])), 0,
    "1 = the screening rule found exactly the speeders")
for name in ["naive16", "scenario12", "scenario10", "conf4"]:
    put(f"true_reliability_{name}", float(truth[f"true_reliability_{name}"]), 2, f"TRUE population reliability of {name}")
    put(f"true_validity_{name}", float(truth[f"true_validity_{name}"]), 2, f"TRUE population validity of {name}")
put("true_corr_truescore10_truecriterion", float(truth["true_corr_truescore10_truecriterion"]), 2,
    "TRUE correlation the attenuation-corrected validity estimates")

# Classification: who would be misclassified at the cutoff?
m = df[["person_id", "score_scenario10"]].dropna().merge(persons, on="person_id")
obs_pass, true_pass = m["score_scenario10"] >= CUTOFF, m["true_score_scenario10"] >= CUTOFF
mis = obs_pass != true_pass
inb = (m["score_scenario10"] - CUTOFF).abs() < band
put("pct_misclassified", 100 * mis.mean(), 0, "Percent whose pass/fail differs from their TRUE-score pass/fail")
put("pct_misclassified_in_band", 100 * mis[inb].mean(), 0, "Misclassification rate inside the SEM band")
put("pct_misclassified_outside_band", 100 * mis[~inb].mean(), 0, "Misclassification rate outside the SEM band")
put("pct_of_misclassified_inside_band", 100 * inb[mis].mean(), 0, "Share of all misclassified people who sit inside the band")

# ------------------------------------------------------------------------------------------
# Step 4. Decision implication (printed; the article carries the argument)
# ------------------------------------------------------------------------------------------
g = lambda k: R[k][0]  # noqa: E731
print(f"""
Step 2: S06 (item-rest r {g('item_rest_r_S06'):.2f}) and S11 ({g('item_rest_r_S11'):.2f}) are flawed.
  Confidence items look fine inside the 16-item total (item-rest r >= {g('item_rest_r_conf_min_in16'):.2f}) but
  form their own cluster (r within {g('mean_r_within_conf4'):.2f} vs. {g('mean_r_between_blocks'):.2f} with scenarios).
  Reliability:  scenario10 alpha {g('alpha_scenario10'):.2f} / omega {g('omega_scenario10'):.2f};  confidence alpha {g('alpha_conf4'):.2f}
  Validity (r with audit): naive16 {g('validity_naive16'):.2f} [{g('validity_naive16_ci_low'):.2f}, {g('validity_naive16_ci_high'):.2f}], \
scenario10 {g('validity_scenario10'):.2f} [{g('validity_scenario10_ci_low'):.2f}, {g('validity_scenario10_ci_high'):.2f}], \
confidence {g('validity_conf4'):.2f} [{g('validity_conf4_ci_low'):.2f}, {g('validity_conf4_ci_high'):.2f}]
  Gain of cleaned over naive: {g('validity_gain_scenario10_vs_naive16'):.2f} [{g('validity_gain_ci_low'):.2f}, {g('validity_gain_ci_high'):.2f}]
  SEM {g('sem_scenario10'):.1f} points; {g('pct_within_band'):.0f}% within +/-1.96 SEM of the cutoff {CUTOFF}.
  Spearman-Brown: {g('sb_items_needed_090')} items for 0.90 ({g('sb_items_to_add_090')} more).
Step 3: true reliability scenario10 {g('true_reliability_scenario10'):.2f}, true validity {g('true_validity_scenario10'):.2f};
  {g('pct_misclassified'):.0f}% misclassified at the cutoff, {g('pct_of_misclassified_inside_band'):.0f}% of them inside the band.
Step 4: drop/fix S06 and S11; report confidence separately (change management, not skill); use the
  scenario score for team-level training needs now; for individual certification, add items toward
  0.90 or send everyone inside the band to a second assessment.""")

# ------------------------------------------------------------------------------------------
# results.json (R values are merged in by python/check_agreement.py)
# ------------------------------------------------------------------------------------------
HEADLINE = ["alpha_naive16", "alpha_scenario12", "alpha_scenario10", "omega_scenario10", "alpha_conf4", "omega_conf4",
            "item_rest_r_S06", "item_rest_r_S11", "item_rest_r_min_kept", "item_rest_r_conf_min_in16",
            "mean_r_within_conf4", "mean_r_between_blocks",
            "validity_naive16", "validity_naive16_ci_low", "validity_naive16_ci_high",
            "validity_scenario12",
            "validity_scenario10", "validity_scenario10_ci_low", "validity_scenario10_ci_high",
            "validity_conf4", "validity_conf4_ci_low", "validity_conf4_ci_high",
            "validity_gain_scenario10_vs_naive16", "validity_gain_ci_low", "validity_gain_ci_high",
            "sem_scenario10", "pct_within_band", "pct_misclassified", "sb_items_needed_090", "sb_items_to_add_090"]


def rounded(val, rd):
    if rd is None:
        return val
    return int(round(val)) if rd == 0 else round(float(val), rd)


results = {
    "edition": "D01 Measuring AI Supervision Skills",
    "seed": 20261104,
    "note": "value = Python value at the stated rounding; 'r' is filled by python/check_agreement.py",
    "headline": HEADLINE,
    "numbers": {k: {"value": rounded(v, rd), "rounding": rd, "meaning": mn, "python": rounded(v, rd),
                    "python_unrounded": (float(v) if rd is not None else v)}
                for k, (v, rd, mn) in R.items()},
}
(OUT / "results.json").write_text(json.dumps(results, indent=2))

# ------------------------------------------------------------------------------------------
# Figures (phone-readable: 1600 px wide, large type, the finding as the title)
# ------------------------------------------------------------------------------------------
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 22, "font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})

# fig1: reliability vs. validity for the three scores
labels = ["Self-reported\nconfidence (4 items)", "Naive total\n(all 16 items)", "Cleaned skill\nscenarios (10 items)"]
keys = ["conf4", "naive16", "scenario10"]
rel = [g("alpha_conf4"), g("alpha_naive16"), g("alpha_scenario10")]
val = [g(f"validity_{k}") for k in keys]
lo_ = [g(f"validity_{k}_ci_low") for k in keys]
hi_ = [g(f"validity_{k}_ci_high") for k in keys]
fig, ax = plt.subplots(figsize=(16, 10), dpi=100)
y = np.arange(3)
h = 0.36
ax.barh(y + h / 2 + 0.01, rel, h, color=BLUE, label="Reliability (Cronbach's alpha)")
ax.barh(y - h / 2 - 0.01, val, h, color=ORANGE, label="Validity (r with audit of AI outputs)")
ax.errorbar(val, y - h / 2 - 0.01, xerr=[np.subtract(val, lo_), np.subtract(hi_, val)], fmt="none",
            ecolor=INK, elinewidth=2.5, capsize=7)
for i in range(3):
    ax.text(rel[i] + 0.015, y[i] + h / 2, f"{rel[i]:.2f}", va="center", fontsize=24, color=INK, fontweight="bold")
    ax.text(hi_[i] + 0.015, y[i] - h / 2, f"{val[i]:.2f}", va="center", fontsize=24, color=INK, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=22)
ax.set_xlim(0, 1.08)
ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax.grid(axis="x", color=GRID)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(loc="upper left", bbox_to_anchor=(-0.02, -0.07), ncol=2, frameon=False, fontsize=20)
fig.text(0.02, 0.955, "The most consistent score is the least valid", fontsize=34, fontweight="bold", color=INK)
fig.text(0.02, 0.905, f"Reliability vs. validity of three scores from one test. Synthetic data, n = {nv}; "
         "whiskers = 95% CI.", fontsize=19, color=MUTED)
fig.subplots_adjust(left=0.23, right=0.97, top=0.86, bottom=0.17)
fig.savefig(OUT / "fig1_consistent_is_not_valid.png")
plt.close(fig)

# fig2: score distribution near the cutoff with the SEM band
fig, ax = plt.subplots(figsize=(16, 10), dpi=100)
bins = np.arange(-0.5, 41.5, 1)
counts, edges = np.histogram(s10, bins=bins)
centers = (edges[:-1] + edges[1:]) / 2
inside = np.abs(centers - CUTOFF) < band
ax.axvspan(CUTOFF - band, CUTOFF + band, color=ORANGE, alpha=0.12, lw=0)
ax.bar(centers[~inside], counts[~inside], width=0.85, color=BLUE, label="Clearly below or clearly above")
ax.bar(centers[inside], counts[inside], width=0.85, color=ORANGE, label="Too close to call (within ±1.96 SEM)")
ax.vlines(CUTOFF, 0, counts.max() * 1.14, color=INK, lw=3, ls="--")
ax.text(CUTOFF + 0.4, counts.max() * 1.05, f"Cutoff {CUTOFF}", fontsize=22, color=INK, fontweight="bold")
ax.text(CUTOFF - band + 0.3, counts.max() * 1.20, f"Band: cutoff ±{band:.1f} points", fontsize=20, color=MUTED)
ax.set_xlabel("Cleaned skill score (10 scenarios, 0-40 points)", fontsize=22)
ax.set_ylabel("People", fontsize=22)
ax.set_ylim(0, counts.max() * 1.30)
ax.set_xlim(-0.5, 40.5)
ax.grid(axis="y", color=GRID)
ax.set_axisbelow(True)
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.legend(loc="upper left", frameon=False, fontsize=20)
fig.text(0.02, 0.955, f"{g('pct_within_band'):.0f}% of people score too close to the cutoff to call",
         fontsize=34, fontweight="bold", color=INK)
fig.text(0.02, 0.905, f"Reliability (omega) {g('omega_scenario10'):.2f}, SEM = {sem:.1f} points. "
         f"Synthetic data, n = {len(s10)}.", fontsize=19, color=MUTED)
fig.subplots_adjust(left=0.08, right=0.97, top=0.86, bottom=0.12)
fig.savefig(OUT / "fig2_too_close_to_call.png")
plt.close(fig)
print("\nWrote outputs/results.json, outputs/python_item_statistics.csv, and 2 figures.")
