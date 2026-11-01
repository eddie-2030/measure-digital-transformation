"""
analysis.py -- B06 "Don't Launch Everywhere at Once: Who Benefits from AI?"

Run from anywhere (paths are resolved from this file):
    python3 data/generate_data.py      # 1. make the synthetic data
    python3 python/analysis.py         # 2. this script
    Rscript r/analysis.R               # 3. the same analysis in R
    python3 python/merge_results.py    # 4. merge R numbers into results.json + check agreement

Steps in this script
    Step 1  The naive approach: one overall average, then slicing the data
            one covariate at a time with no correction. Why both mislead.
    Step 2  The correct method: (a) a pre-specified AI x moderator interaction
            model with team-clustered standard errors, plus subgroup effects with
            95% CIs; (b) an ML "discovery" step: a doubly robust CATE learner
            (DR-learner) with a random forest, cross-fitted by team.
    Step 3  Check against ground truth (known only because data are synthetic).
    Step 4  Decision implication: a phased rollout grid.

Dependencies: numpy, pandas, statsmodels, scikit-learn, matplotlib (+ scipy,
which statsmodels already installs).

All data are synthetic, generated to illustrate the method.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "ai_pilot_heterogeneity.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT / "data"))
sys.path.insert(0, str(ROOT / "python"))

SEED = 20261153            # same seed as the generator; also seeds the forest
NOVICE_CUTOFF = 12         # months, pre-specified
MIN_EFFECT = 1.0           # pre-specified smallest effect worth launching for (tasks/week)
N_FOLDS = 5                # cross-fitting folds (assigned by team, see make_folds)


# -----------------------------------------------------------------------------
# 0. Load and prepare
# -----------------------------------------------------------------------------
def prepare(d: pd.DataFrame) -> pd.DataFrame:
    """Same cleaning rules in Python and R:
    * winsorize outcome and baseline at the 0.5th/99.5th percentiles
      (catches the 4-week totals keyed in as weekly averages);
    * novice = tenure < 12 months; left missing when tenure is missing
      (models that use tenure run on complete cases; tenure is missing at random);
    * non_english = team works in a language other than English.
    """
    d = d.copy()
    for c in ["outcome_tasks", "baseline_tasks"]:
        lo, hi = np.percentile(d[c], [0.5, 99.5])   # numpy 'linear' == R quantile type 7
        d[c + "_w"] = d[c].clip(lo, hi)
    d["non_english"] = (d["work_language"] != "English").astype(int)
    d["novice"] = np.where(d["tenure_months"].isna(), np.nan,
                           (d["tenure_months"] < NOVICE_CUTOFF).astype(float))
    return d


def ols_cluster(formula: str, d: pd.DataFrame) -> pd.DataFrame:
    """OLS with standard errors clustered by team.

    Why cluster? People in the same team share a manager, queue and workload, so
    their outcomes move together. Treating 2,400 people as 2,400 independent
    data points would overstate our certainty. Clustering by team fixes that.

    CIs and p-values use a t distribution with (number of teams - 1) degrees of
    freedom -- the same convention as the R script (sandwich::vcovCL, HC1).
    """
    groups = pd.factorize(d["team_id"])[0]
    m = smf.ols(formula, data=d).fit(cov_type="cluster", cov_kwds={"groups": groups})
    G = len(np.unique(groups))
    tq = stats.t.ppf(0.975, G - 1)
    tab = pd.DataFrame({"est": m.params, "se": m.bse})
    tab["lo"] = tab.est - tq * tab.se
    tab["hi"] = tab.est + tq * tab.se
    tab["p"] = 2 * stats.t.sf(np.abs(tab.est / tab.se), G - 1)
    tab.attrs["n"] = int(m.nobs)
    return tab


def holm(p: pd.Series) -> pd.Series:
    """Holm step-down adjustment (identical to R's p.adjust(method = 'holm'))."""
    k = len(p)
    order = p.sort_values()
    adj = np.minimum(1, np.maximum.accumulate(order.values * (k - np.arange(k))))
    return pd.Series(adj, index=order.index).reindex(p.index)


df = prepare(pd.read_csv(DATA))
N, G = len(df), df.team_id.nunique()
control_mean = df.loc[df.ai_access == 0, "outcome_tasks_w"].mean()
pct = lambda x: 100 * x / control_mean   # express effects as % of the control mean
n_outliers_capped = int((df.outcome_tasks != df.outcome_tasks_w).sum())
print(f"{N:,} employees, {G} teams; control mean = {control_mean:.2f} tasks/week; "
      f"{df.tenure_months.isna().sum()} missing tenure; {n_outliers_capped} outcome values capped")

CONTROLS = "baseline_tasks_w + C(region)"   # pre-period outcome + region (precision only)

# -----------------------------------------------------------------------------
# STEP 1. The naive approach
# -----------------------------------------------------------------------------
# 1a. One average for everyone. It is positive and "significant" -> "launch everywhere".
ate = ols_cluster(f"outcome_tasks_w ~ ai_access + {CONTROLS}", df).loc["ai_access"]
print(f"\n[Step 1a] Overall ATE = {ate.est:.2f} tasks/week ({pct(ate.est):.1f}%), "
      f"95% CI [{ate.lo:.2f}, {ate.hi:.2f}], p = {ate.p:.4f}")

# 1b. Slice one covariate level at a time and test "is the AI effect different
#     here than in everyone else?" (an AI x slice interaction), no correction.
#     14 tests: 2 real moderators, 4 region slices, 8 pure-noise slices.
slices = {}
for r in ["Americas", "EMEA", "APAC", "LATAM"]:
    slices[f"Region: {r}"] = ("region", (df.region == r).astype(float))
slices["Language: non-English"] = ("work_language", df.non_english.astype(float))
slices["Tenure: novice (<12 mo)"] = ("tenure", df.novice)
for r in ["Tier-1 support", "Tier-2 support", "Operations analyst", "Specialist"]:
    slices[f"Role: {r}"] = ("role_family", (df.role_family == r).astype(float))
for r in ["Day", "Evening", "Night"]:
    slices[f"Shift: {r}"] = ("shift", (df["shift"] == r).astype(float))
slices["Remote worker"] = ("remote", df.remote.astype(float))

TRUE_MODERATORS = {"work_language", "tenure"}
rows = []
for name, (var, s) in slices.items():
    d = df.assign(s=s).dropna(subset=["s"])
    t = ols_cluster("outcome_tasks_w ~ ai_access * s + baseline_tasks_w", d).loc["ai_access:s"]
    rows.append(dict(slice=name, variable=var, true_moderator=var in TRUE_MODERATORS,
                     n_in_slice=int(d.s.sum()), diff_in_effect=t.est,
                     ci_low=t.lo, ci_high=t.hi, p_raw=t.p))
naive = pd.DataFrame(rows)
naive["p_holm"] = holm(naive.set_index("slice").p_raw).values
naive["flag_raw"] = naive.p_raw < 0.05
naive["flag_holm"] = naive.p_holm < 0.05
false_pos = naive[naive.flag_raw & ~naive.true_moderator]
print("\n[Step 1b] Naive slicing (difference in AI effect, slice vs. rest):")
print(naive[["slice", "diff_in_effect", "p_raw", "p_holm"]].round(3).to_string(index=False))
print(f"  Flagged at p<.05 uncorrected: {naive.flag_raw.sum()} slices "
      f"({len(false_pos)} of them have NO true moderation). After Holm: {naive.flag_holm.sum()}.")

# -----------------------------------------------------------------------------
# STEP 2a. Pre-specified interaction model (the workhorse)
#   Outcome = b0 + b1*AI + b2*NonEnglish + b3*(AI x NonEnglish)
#           + b4*Novice + b5*(AI x Novice) + controls
#   b3 = how much the AI effect differs for non-English teams
#   b5 = how much the AI effect differs for novices
# Complete cases on tenure (tenure is missing completely at random here).
# -----------------------------------------------------------------------------
cc = df.dropna(subset=["novice"]).copy()
inter = ols_cluster(
    f"outcome_tasks_w ~ ai_access * non_english + ai_access * novice + {CONTROLS}", cc)
b1, b3, b5 = inter.loc["ai_access"], inter.loc["ai_access:non_english"], inter.loc["ai_access:novice"]
print(f"\n[Step 2a] Interaction model (n = {inter.attrs['n']}):")
for lab, r in [("b1 AI (English, experienced)", b1), ("b3 AI x NonEnglish", b3), ("b5 AI x Novice", b5)]:
    print(f"  {lab:30s} {r.est:6.2f}  [{r.lo:6.2f}, {r.hi:6.2f}]  p = {r.p:.4f}")

# Subgroup effects: the same simple model run inside each subgroup.
groups_def = {
    "all": ("All employees", df, None),
    "english": ("English-language teams", df[df.non_english == 0], None),
    "non_english": ("Non-English teams", df[df.non_english == 1], None),
    "novice": ("Novices (<12 months)", cc[cc.novice == 1], None),
    "experienced": ("Experienced (12+ months)", cc[cc.novice == 0], None),
    "english_novice": ("English, novice", cc[(cc.non_english == 0) & (cc.novice == 1)], None),
    "english_experienced": ("English, experienced", cc[(cc.non_english == 0) & (cc.novice == 0)], None),
    "non_english_novice": ("Non-English, novice", cc[(cc.non_english == 1) & (cc.novice == 1)], None),
    "non_english_experienced": ("Non-English, experienced", cc[(cc.non_english == 1) & (cc.novice == 0)], None),
}
sub = []
for key, (label, d, _) in groups_def.items():
    r = ols_cluster(f"outcome_tasks_w ~ ai_access + {CONTROLS}", d).loc["ai_access"]
    sub.append(dict(key=key, label=label, n=len(d), n_teams=d.team_id.nunique(),
                    est=r.est, lo=r.lo, hi=r.hi, p=r.p,
                    true=d.true_effect.mean()))   # ground truth: mean true effect in the group
sub = pd.DataFrame(sub).set_index("key")
sub["covered"] = (sub.lo <= sub.true) & (sub.true <= sub.hi)
print("\n[Step 2a] Subgroup effects (tasks/week) vs. truth:")
print(sub[["label", "n", "est", "lo", "hi", "p", "true", "covered"]].round(2).to_string())

# -----------------------------------------------------------------------------
# STEP 2b. ML discovery step: a doubly robust CATE learner ("DR-learner")
#
# Intuition. A normal model predicts the OUTCOME. We want to predict the
# TREATMENT EFFECT, which we never see for one person. The trick: build a
# "pseudo-outcome" for each person whose average, in any group, equals that
# group's treatment effect, then let a random forest predict it from the
# covariates. Trees split where the *effect* differs, not where output differs.
#
#   pseudo_i = mu1(X) - mu0(X) + W(Y - mu1(X))/e - (1-W)(Y - mu0(X))/(1-e)
#
#   mu1, mu0 : outcome models for treated / control (plain linear regression,
#              fit on OTHER teams' data = cross-fitting, so no one grades
#              their own homework)
#   e        : probability of AI access, known by design (team's treated share)
#
# The forest is also cross-fitted: each person's CATE is predicted by a forest
# that never saw their team. Folds are fixed by team number so Python and R use
# identical folds (and identical pseudo-outcomes); only the forest differs.
# -----------------------------------------------------------------------------
def design_matrix(d: pd.DataFrame) -> pd.DataFrame:
    tenure_med = d.tenure_months.median()
    X = pd.DataFrame({
        "tenure_months": d.tenure_months.fillna(tenure_med),
        "tenure_missing": d.tenure_months.isna().astype(int),
        "non_english": d.non_english,
        "remote": d.remote,
        "baseline_tasks": d.baseline_tasks_w,
    })
    for var, levels in [("region", ["Americas", "EMEA", "APAC", "LATAM"]),
                        ("role_family", ["Tier-1 support", "Tier-2 support", "Operations analyst", "Specialist"]),
                        ("shift", ["Day", "Evening", "Night"])]:
        for lv in levels:
            X[f"{var}={lv}"] = (d[var] == lv).astype(int)
    return X


# Map each column back to the business variable it came from (for importance).
def variable_of(col: str) -> str:
    if col.startswith("tenure"):
        return "Tenure"
    if col == "non_english":
        return "Work language"
    if col == "baseline_tasks":
        return "Baseline output"
    if col == "remote":
        return "Remote"
    return {"region": "Region", "role_family": "Role family", "shift": "Shift"}[col.split("=")[0]]


def make_folds(d):
    # Team T001 -> fold 1, T002 -> fold 2, ... T006 -> fold 1 (same rule in R).
    return (d.team_id.str[1:].astype(int) - 1) % N_FOLDS


def dr_pseudo_outcome(d, X, folds):
    y, w = d.outcome_tasks_w.to_numpy(), d.ai_access.to_numpy()
    e = d.groupby("team_id").ai_access.transform("mean").to_numpy()   # design propensity
    mu0, mu1 = np.zeros(len(d)), np.zeros(len(d))
    Xn = X.to_numpy(float)
    for k in range(N_FOLDS):
        tr, te = folds != k, folds == k
        m0 = LinearRegression().fit(Xn[tr & (w == 0)], y[tr & (w == 0)])
        m1 = LinearRegression().fit(Xn[tr & (w == 1)], y[tr & (w == 1)])
        mu0[te], mu1[te] = m0.predict(Xn[te]), m1.predict(Xn[te])
    return mu1 - mu0 + w * (y - mu1) / e - (1 - w) * (y - mu0) / (1 - e)


def forest():
    # Big leaves (min 60 people): we want stable group-level effects, not
    # individual-level noise. max_features adds the "random" in random forest.
    return RandomForestRegressor(n_estimators=500, min_samples_leaf=60, max_features=0.33,
                                 random_state=SEED, n_jobs=-1)


X = design_matrix(df)
folds = make_folds(df).to_numpy()
pseudo = dr_pseudo_outcome(df, X, folds)
df["dr_pseudo"] = pseudo
cate = np.zeros(N)
for k in range(N_FOLDS):
    tr, te = folds != k, folds == k
    cate[te] = forest().fit(X[tr], pseudo[tr]).predict(X[te])
df["cate_hat"] = cate

full_rf = forest().fit(X, pseudo)
imp = (pd.Series(full_rf.feature_importances_, index=X.columns)
       .groupby(variable_of).sum().sort_values(ascending=False))
imp = imp / imp.sum()
print("\n[Step 2b] DR-learner + random forest: share of heterogeneity splits by variable")
print(imp.round(3).to_string())

# Calibration on held-out predictions: do people the forest predicts will benefit
# more actually benefit more? Split at the median predicted CATE and estimate the
# real (randomized) effect in each half.
df["pred_half"] = np.where(df.cate_hat >= np.median(df.cate_hat), "high", "low")
cal = {h: ols_cluster(f"outcome_tasks_w ~ ai_access + {CONTROLS}", df[df.pred_half == h]).loc["ai_access"]
       for h in ["high", "low"]}
q25 = df.cate_hat <= np.quantile(df.cate_hat, 0.25)
ml_py = {
    "learner": "DR-learner (cross-fitted linear outcome models, design propensity) + scikit-learn RandomForestRegressor (500 trees, min_samples_leaf=60), cross-fitted by team",
    "dr_ate_tasks": round(float(pseudo.mean()), 2),
    "variable_importance_share": {k: round(float(v), 3) for k, v in imp.items()},
    "top2_moderators": list(imp.index[:2]),
    "mean_cate_by_group_tasks": {
        "english": round(float(df.loc[df.non_english == 0, "cate_hat"].mean()), 2),
        "non_english": round(float(df.loc[df.non_english == 1, "cate_hat"].mean()), 2),
        "novice": round(float(cc.join(df.cate_hat).query("novice == 1").cate_hat.mean()), 2),
        "experienced": round(float(cc.join(df.cate_hat).query("novice == 0").cate_hat.mean()), 2),
    },
    "corr_cate_vs_true_effect": round(float(np.corrcoef(df.cate_hat, df.true_effect)[0, 1]), 2),
    "calibration_high_half_effect_tasks": [round(float(cal["high"][c]), 2) for c in ["est", "lo", "hi"]],
    "calibration_low_half_effect_tasks": [round(float(cal["low"][c]), 2) for c in ["est", "lo", "hi"]],
    "share_non_english_in_bottom_quartile_cate": round(float(df.loc[q25, "non_english"].mean()), 2),
}
print(f"  corr(predicted CATE, true effect) = {ml_py['corr_cate_vs_true_effect']}; "
      f"high-half effect {ml_py['calibration_high_half_effect_tasks']}, "
      f"low-half effect {ml_py['calibration_low_half_effect_tasks']}")

# -----------------------------------------------------------------------------
# STEP 3. Ground-truth check
# -----------------------------------------------------------------------------
TRUE_B3, TRUE_B5 = -3.6, 2.2
truth = {
    "true_ate_tasks": round(float(df.true_effect[df.ai_access == 1].mean()), 2),
    "true_ate_tasks_all_employees": round(float(df.true_effect.mean()), 2),
    "true_interaction_ai_x_nonenglish": TRUE_B3,
    "true_interaction_ai_x_novice": TRUE_B5,
    "interaction_ai_x_nonenglish_ci_covers_truth": bool(b3.lo <= TRUE_B3 <= b3.hi),
    "interaction_ai_x_novice_ci_covers_truth": bool(b5.lo <= TRUE_B5 <= b5.hi),
    "subgroups_true_vs_estimated_tasks": {
        k: {"true": round(r.true, 2) + 0.0, "estimated": round(r.est, 2) + 0.0, "ci_covers_truth": bool(r.covered)}
        for k, r in sub.iterrows()},
    "subgroup_ci_coverage": f"{int(sub.covered.sum())} of {len(sub)}",
    "noise_covariates_true_moderation": 0.0,
}
print(f"\n[Step 3] Truth: interaction CIs cover truth? b3 {truth['interaction_ai_x_nonenglish_ci_covers_truth']}, "
      f"b5 {truth['interaction_ai_x_novice_ci_covers_truth']}; subgroup CI coverage {truth['subgroup_ci_coverage']}")

# -----------------------------------------------------------------------------
# STEP 4. Decision implication: phase the rollout by evidence
# The rollout unit is a (language x tenure) segment. Rules, set BEFORE looking:
#   Launch     : lower 95% bound >= MIN_EFFECT (confident the gain is worthwhile)
#   Hold & fix : upper 95% bound <  MIN_EFFECT (cannot reach a worthwhile gain)
#   Re-test    : anything else (promising but uncertain -> fix, extend, re-test)
# -----------------------------------------------------------------------------
def decide(r):
    if r.lo >= MIN_EFFECT:
        return "Launch now (wave 1)"
    if r.hi < MIN_EFFECT:
        return "Hold: fix/localize, then re-test"
    return "Targeted re-test (wave 2 candidate)"


cells = ["english_novice", "english_experienced", "non_english_novice", "non_english_experienced"]
grid = [dict(segment=sub.loc[k, "label"], n=int(sub.loc[k, "n"]),
             effect_tasks=round(sub.loc[k, "est"], 2), ci_low=round(sub.loc[k, "lo"], 2),
             ci_high=round(sub.loc[k, "hi"], 2), effect_pct=round(pct(sub.loc[k, "est"]), 1),
             decision=decide(sub.loc[k]))
        for k in cells]
print("\n[Step 4] Rollout grid (rule: launch if CI lower >= "
      f"{MIN_EFFECT} task/week; hold if CI upper < {MIN_EFFECT}; else re-test):")
print(pd.DataFrame(grid).to_string(index=False))

# -----------------------------------------------------------------------------
# Supplement (Python only): how often does naive slicing produce a false alarm?
# Re-generate the pilot 200 times with new seeds; count pilots in which at least
# one of the 8 pure-noise slices is "significant" at p < .05 without correction.
# -----------------------------------------------------------------------------
from generate_data import generate  # noqa: E402

N_SIM = 200
hits_raw = hits_holm = 0
for s in range(1, N_SIM + 1):
    d = prepare(generate(SEED + 1000 + s))
    ps = []
    for var, lv in [("role_family", "Tier-1 support"), ("role_family", "Tier-2 support"),
                    ("role_family", "Operations analyst"), ("role_family", "Specialist"),
                    ("shift", "Day"), ("shift", "Evening"), ("shift", "Night"), ("remote", 1)]:
        dd = d.assign(s=(d[var] == lv).astype(float))
        ps.append(ols_cluster("outcome_tasks_w ~ ai_access * s + baseline_tasks_w", dd).loc["ai_access:s", "p"])
    ps = np.array(ps)
    hits_raw += (ps < 0.05).any()
    hits_holm += (holm(pd.Series(ps)) < 0.05).any()
sim = {"n_replications": N_SIM,
       "share_pilots_with_false_alarm_uncorrected": round(hits_raw / N_SIM, 2),
       "share_pilots_with_false_alarm_holm": round(hits_holm / N_SIM, 2),
       "meaning": "Share of simulated pilots (new seeds, same design) where at least one of 8 pure-noise slices looks 'significant' at p<.05 (Holm-corrected within the 8 noise tests in the second line). Python only."}
print(f"\n[Supplement] False alarm in {sim['share_pilots_with_false_alarm_uncorrected']:.0%} of "
      f"{N_SIM} simulated pilots (uncorrected) vs {sim['share_pilots_with_false_alarm_holm']:.0%} with Holm.")

# -----------------------------------------------------------------------------
# Write outputs: headline numbers (flat csv, compared with R) + results.json
# -----------------------------------------------------------------------------
H = []   # (name, value, decimals, unit, meaning)


def add(name, value, dec, unit, meaning):
    H.append((name, float(value), dec, unit, meaning))


add("n_employees", N, 0, "people", "Employees in the pilot")
add("n_teams", G, 0, "teams", "Teams (randomization happened within each team)")
add("n_complete_tenure", len(cc), 0, "people", "Employees with known tenure (used in tenure models)")
add("control_mean_tasks", control_mean, 2, "tasks/week", "Average weekly quality-adjusted tasks in the control group (denominator for %)")
add("ate_tasks", ate.est, 2, "tasks/week", "Overall average effect of AI access (adjusted for baseline and region)")
add("ate_ci_low", ate.lo, 2, "tasks/week", "Lower 95% bound, team-clustered")
add("ate_ci_high", ate.hi, 2, "tasks/week", "Upper 95% bound, team-clustered")
add("ate_p", ate.p, 4, "p-value", "p-value of the overall effect")
add("ate_pct", pct(ate.est), 1, "%", "Overall effect as % of the control mean")
add("ate_pct_ci_low", pct(ate.lo), 1, "%", "Lower 95% bound in %")
add("ate_pct_ci_high", pct(ate.hi), 1, "%", "Upper 95% bound in %")
for nm, r, txt in [("b1_ai_english_experienced", b1, "AI effect for English-language, experienced employees (b1)"),
                   ("b3_ai_x_nonenglish", b3, "How much the AI effect differs for non-English teams (b3)"),
                   ("b5_ai_x_novice", b5, "How much the AI effect differs for novices, tenure < 12 months (b5)")]:
    add(nm, r.est, 2, "tasks/week", txt)
    add(nm + "_ci_low", r.lo, 2, "tasks/week", "Lower 95% bound, team-clustered")
    add(nm + "_ci_high", r.hi, 2, "tasks/week", "Upper 95% bound, team-clustered")
    add(nm + "_p", r.p, 4, "p-value", "p-value")
for k, r in sub.iterrows():
    add(f"sub_{k}_n", r.n, 0, "people", f"Sample size: {r.label}")
    add(f"sub_{k}_tasks", r.est, 2, "tasks/week", f"Estimated AI effect: {r.label}")
    add(f"sub_{k}_ci_low", r.lo, 2, "tasks/week", "Lower 95% bound, team-clustered")
    add(f"sub_{k}_ci_high", r.hi, 2, "tasks/week", "Upper 95% bound, team-clustered")
    add(f"sub_{k}_p", r.p, 4, "p-value", "p-value")
    add(f"sub_{k}_pct", pct(r.est), 1, "%", f"Estimated AI effect as % of control mean: {r.label}")
    add(f"sub_{k}_true_tasks", r.true, 2, "tasks/week", f"TRUE average effect (synthetic ground truth): {r.label}")
for _, r in naive.iterrows():
    key = "slice_" + "".join(ch if ch.isalnum() else "_" for ch in r.slice.lower()).strip("_").replace("__", "_")
    add(key + "_diff", r.diff_in_effect, 2, "tasks/week", f"Naive slicing: AI effect in '{r.slice}' minus rest")
    add(key + "_p_raw", r.p_raw, 4, "p-value", "Uncorrected p-value")
    add(key + "_p_holm", r.p_holm, 4, "p-value", "Holm-corrected p-value (14 tests)")
add("naive_n_flagged_raw", naive.flag_raw.sum(), 0, "slices", "Slices flagged at p<.05 without correction (of 14)")
add("naive_n_false_flagged_raw", len(false_pos), 0, "slices", "Flagged slices with NO true moderation")
add("naive_n_flagged_holm", naive.flag_holm.sum(), 0, "slices", "Slices still flagged after Holm correction")
add("dr_ate_tasks", pseudo.mean(), 2, "tasks/week", "Average of the DR pseudo-outcomes (= doubly robust ATE; identical folds in Python and R)")

headline = pd.DataFrame(H, columns=["name", "value", "decimals", "unit", "meaning"])
headline.to_csv(OUT / "headline_python.csv", index=False)

fp = false_pos.iloc[0] if len(false_pos) else None
results = {
    "meta": {
        "edition": "B06 - Don't Launch Everywhere at Once: Who Benefits from AI?",
        "disclaimer": "All data in this folder are synthetic, generated to illustrate the method. They do not come from Meta, any client, or any real organization. The goal is to show how a business question, a measurement strategy, the statistics, and the code connect end to end.",
        "seed": SEED, "data_file": "data/ai_pilot_heterogeneity.csv",
        "outcome": "Weekly quality-adjusted tasks resolved during the pilot (winsorized at 0.5/99.5 pct)",
        "inference": "OLS, controls = baseline output + region; SEs clustered by team (HC1 / statsmodels cluster); 95% CIs and p-values use t with (teams - 1) df",
        "novice_definition": "tenure < 12 months (pre-specified); tenure models use complete cases",
        "pct_definition": "effect / control-group mean x 100 (same denominator for every subgroup)",
        "min_effect_for_launch_tasks": MIN_EFFECT,
        "outcome_values_capped": n_outliers_capped,
        "how_built": "python/analysis.py writes headline_python.csv and this file; r/analysis.R writes headline_r.csv and ml_r.csv; python/merge_results.py fills every 'r' value and the 'agreement' block.",
    },
    "headline": {r.name: {"python": (int(round(r.value)) if r.decimals == 0 else round(r.value, int(r.decimals)) + 0.0),
                          "r": None,
                          "decimals": int(r.decimals), "unit": r.unit, "meaning": r.meaning}
                 for r in headline.itertuples()},
    "naive_slicing_false_positive_example": None if fp is None else {
        "slice": fp.slice, "diff_in_effect_tasks": round(fp.diff_in_effect, 2),
        "p_raw": round(fp.p_raw, 4), "p_holm": round(fp.p_holm, 4),
        "meaning": "A covariate with NO true effect modification looks 'significant' when slicing 14 ways without correction; it disappears after Holm correction, while language and tenure survive."},
    "naive_slicing_table": naive.round(4).to_dict(orient="records"),
    "ml_step": {"python": ml_py, "r": None, "qualitative_agreement": None,
                "note": "ML-step numbers are expected to differ across languages (different forest implementations); compare rankings and group averages, not decimals."},
    "ground_truth_check": truth,
    "rollout_decision_grid_python": grid,
    "decision_rule": f"Launch if the 95% CI lower bound >= {MIN_EFFECT} task/week (the pre-specified smallest worthwhile gain); hold and fix if the upper bound < {MIN_EFFECT}; otherwise targeted re-test.",
    "simulation_false_alarm_rate": sim,
    "agreement": {"status": "pending: run Rscript r/analysis.R then python3 python/merge_results.py"},
}
with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2)

# -----------------------------------------------------------------------------
# Figures (finding as title; >= 1600 px wide; large type for phones)
# -----------------------------------------------------------------------------
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": INK})

# --- Figure 1: subgroup effects vs truth vs the average ---------------------
order = ["all", "english", "non_english", "novice", "experienced"]
f1 = sub.loc[order].copy()
for c in ["est", "lo", "hi", "true"]:
    f1[c] = pct(f1[c])
labels = ["All employees", "English-language teams", "Non-English teams",
          "Novices (<12 months)", "Experienced (12+ months)"]
fig, ax = plt.subplots(figsize=(16, 9.6), dpi=110)
ypos = np.arange(len(order))[::-1].astype(float)
ypos[0] += 0.35   # small gap under the "All employees" row
ax.axvline(0, color=MUTED, lw=1.5)
ax.axvline(f1.loc["all", "est"], color=MUTED, lw=2, ls=(0, (5, 4)))
ax.text(f1.loc["all", "est"] + 0.25, ypos[0] + 0.48, f"Average: +{f1.loc['all', 'est']:.1f}%",
        color=MUTED, fontsize=20, va="center")
for y, (k, r) in zip(ypos, f1.iterrows()):
    col = ORANGE if k == "non_english" else BLUE
    ax.plot([r.lo, r.hi], [y, y], color=col, lw=5, solid_capstyle="round", zorder=2)
    ax.plot(r.est, y, "o", ms=17, color=col, mec="white", mew=2.5, zorder=3)
    ax.plot(r.true, y, "D", ms=13, mfc="none", mec=INK, mew=2.5, zorder=4)
    v1 = round(r.est, 1) + 0.0   # + 0.0 avoids "-0.0"; zero is shown unsigned
    ax.text(19.3, y, f"{v1:.1f}%" if v1 == 0 else f"{v1:+.1f}%", fontsize=24, fontweight="bold", color=INK,
            va="center", ha="right")
ax.set_yticks(ypos)
ax.set_yticklabels(labels, fontsize=24)
ax.set_xlim(-4.5, 19.5)
ax.set_ylim(-0.7, ypos[0] + 0.9)
ax.set_xlabel("Effect of AI access on weekly quality-adjusted tasks (% of control mean)", fontsize=20, labelpad=12)
ax.tick_params(axis="x", labelsize=20)
ax.tick_params(axis="y", length=0)
ax.xaxis.grid(True, color=GRID, lw=1)
ax.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
fig.suptitle(f"A +{pct(ate.est):.0f}% average hides teams with no measurable gain",
             x=0.015, ha="left", y=0.985, fontsize=34, fontweight="bold", color=INK)
fig.text(0.015, 0.895, "Dots = estimate with 95% CI (team-clustered)     ◇ = true effect (known: synthetic data)",
         fontsize=19, color=MUTED, ha="left")
fig.text(0.015, 0.015, "Synthetic data: randomized pilot, 2,450 employees in 120 teams. Measure Digital Transformation, B06.",
         fontsize=15, color=MUTED, ha="left")
fig.subplots_adjust(left=0.29, right=0.98, top=0.85, bottom=0.15)
fig.savefig(OUT / "fig_1_subgroup_effects.png", facecolor="white")
plt.close(fig)

# --- Figure 2: what the ML step found ----------------------------------------
fig, ax = plt.subplots(figsize=(16, 9.6), dpi=110)
iv = imp.sort_values()
true_vars = {"Work language", "Tenure"}
cols = [BLUE if v in true_vars else "#b7b6b0" for v in iv.index]
ax.barh(np.arange(len(iv)), 100 * iv.values, color=cols, height=0.62)
for i, v in enumerate(iv.values):
    ax.text(100 * v + 0.8, i, f"{100 * v:.0f}%", va="center", fontsize=21, color=INK)
ax.set_yticks(np.arange(len(iv)))
ax.set_yticklabels(iv.index, fontsize=24)
ax.set_xlabel("Share of the forest's effect-splitting importance (%)", fontsize=20, labelpad=12)
ax.tick_params(axis="x", labelsize=20)
ax.tick_params(axis="y", length=0)
ax.set_xlim(0, 100 * iv.max() * 1.18)
ax.xaxis.grid(True, color=GRID, lw=1)
ax.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
fig.suptitle(f"The ML step points to {iv.index[-1].lower()} and {iv.index[-2].lower()}",
             x=0.015, ha="left", y=0.985, fontsize=34, fontweight="bold", color=INK)
fig.text(0.015, 0.895, "Causal ML (DR-learner + random forest): which variables the effect differs by. "
         "Blue = true moderators.", fontsize=19, color=MUTED, ha="left")
fig.text(0.015, 0.015, "Synthetic data. Python/scikit-learn version; the R version (bagged rpart trees) ranks the same top two.",
         fontsize=15, color=MUTED, ha="left")
fig.subplots_adjust(left=0.22, right=0.97, top=0.85, bottom=0.15)
fig.savefig(OUT / "fig_2_cate_variable_importance.png", facecolor="white")
plt.close(fig)
print(f"\nWrote {OUT/'results.json'}, headline_python.csv and 2 figures.")

# If the R results already exist (e.g. R was run first), merge them now.
if (OUT / "headline_r.csv").exists():
    from merge_results import merge  # noqa: E402
    merge()
