"""
analysis.py  --  B01 "Running a Fair A/B Test Inside Your Company"   (SYNTHETIC data)

Run from the package root, after data/generate_data.py:
    python3 python/analysis.py

Steps
  0. Clean the data the same way the R script does (duplicate, outage week, leavers).
  1. The naive approach: compare volunteers with non-volunteers. Why it misleads.
  2. The correct method: stratified randomization, analyzed as randomized (ITT),
     first by a simple difference in means, then by ANCOVA:
         cases_post = a + b * assigned + c * cases_pre + stratum effects + error
     b is the effect of GIVING ACCESS to the tool.
     Plus: uptake, effect among users, the quality guardrail, and two pitfalls
     (comparing users with non-users; adjusting for a post-treatment variable).
  3. The check against ground truth: the true effect, and 500 re-randomizations
     showing bias, standard error and 95% CI coverage of both estimators.
  4. The decision implication, using a rule written down before the pilot.

Outputs: outputs/results.json (every number the article may quote) and
         outputs/fig_1_estimates.png, outputs/fig_2_simulation.png
"""

import json
import os

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats  # scipy ships with statsmodels; used only for the t quantile

DATA, OUT = "data", "outputs"
os.makedirs(OUT, exist_ok=True)

MIN_POST_WEEKS = 4          # pre-registered: need >= 4 valid post weeks to have an outcome
QUALITY_MARGIN = -1.0       # pre-registered guardrail: quality may not drop by more than 1 point
MIN_WORTHWHILE = 0.0        # pre-registered: roll out only if the CI lower bound is above this

results = {}                # name -> raw python value (rounded when written)


# =============================================================================
# Step 0. Cleaning -- explicit, identical in Python and R
# =============================================================================
def clean(df):
    """Return (analysis frame, log). The same four rules are coded in r/analysis.R."""
    log = {"n_rows_raw": len(df)}
    # Rule 1: drop exact duplicate rows (an export glitch), then insist one row per person.
    df = df.drop_duplicates().reset_index(drop=True)
    log["n_duplicates_removed"] = log["n_rows_raw"] - len(df)
    assert df["employee_id"].is_unique, "conflicting rows for the same employee"

    post_cols = [f"cases_post_w{w}" for w in range(1, 9)]
    pre_cols = [f"cases_pre_w{w}" for w in range(1, 5)]
    post = df[post_cols].astype(float).to_numpy()

    # Rule 2: a team-week where EVERY recorded value is 0 is a data-feed outage, not a
    # week of zero work. Set those cells to missing (do not count them as zeros).
    n_outage = 0
    for team, idx in df.groupby("team").indices.items():
        for w in range(post.shape[1]):
            vals = post[idx, w]
            obs = ~np.isnan(vals)
            if obs.sum() > 0 and np.all(vals[obs] == 0):
                post[idx[obs], w] = np.nan
                n_outage += int(obs.sum())
    log["n_outage_cells_set_missing"] = n_outage

    # Rule 3: the outcome is the mean of the valid post weeks; leavers keep the weeks
    # they worked, but need at least MIN_POST_WEEKS valid weeks to be analyzed.
    n_valid = (~np.isnan(post)).sum(axis=1)
    df["cases_post"] = np.nansum(post, axis=1) / np.where(n_valid > 0, n_valid, np.nan)
    df["n_valid_post_weeks"] = n_valid
    df["cases_pre"] = df[pre_cols].mean(axis=1)
    log["n_leavers"] = int(df["left_week"].notna().sum())
    keep = n_valid >= MIN_POST_WEEKS
    log["n_excluded_short_post"] = int((~keep).sum())
    return df[keep].reset_index(drop=True), df, log


def ols(formula, data, term):
    """Estimate, SE and t-based 95% CI for one coefficient (classical OLS SEs, as in R's lm)."""
    fit = smf.ols(formula, data=data).fit()
    lo, hi = fit.conf_int(alpha=0.05).loc[term]
    return {"est": fit.params[term], "se": fit.bse[term], "lo": lo, "hi": hi, "fit": fit}


vol_raw = pd.read_csv(os.path.join(DATA, "volunteer_pilot.csv"))
rnd_raw = pd.read_csv(os.path.join(DATA, "randomized_pilot.csv"))
truth = pd.read_csv(os.path.join(DATA, "ground_truth.csv"))
rerand = pd.read_csv(os.path.join(DATA, "rerandomizations.csv"))

vol, _, vol_log = clean(vol_raw)
rnd, rnd_all, rnd_log = clean(rnd_raw)
for k, v in rnd_log.items():
    results[k] = v
results["n_analysis"] = len(rnd)
results["n_analysis_treated"] = int(rnd["assigned"].sum())
results["n_analysis_control"] = int((1 - rnd["assigned"]).sum())
# attrition by arm: should be similar if leaving is unrelated to the tool
for arm, name in [(1, "treated"), (0, "control")]:
    a = rnd_all[rnd_all["assigned"] == arm]
    results[f"attrition_{name}_pct"] = 100 * (a["n_valid_post_weeks"] < MIN_POST_WEEKS).mean()

print("Step 0 - cleaning (randomized pilot):", rnd_log, "| analyzed:", len(rnd))

# =============================================================================
# Step 1. Naive: the volunteer pilot
# =============================================================================
# Volunteers chose themselves. They were better performers BEFORE the tool existed and
# they are more motivated, so they would have improved anyway.
results["vol_share_volunteered_pct"] = 100 * vol["volunteered"].mean()
results["vol_baseline_gap"] = (vol.loc[vol.volunteered == 1, "cases_pre"].mean()
                               - vol.loc[vol.volunteered == 0, "cases_pre"].mean())
v_naive = ols("cases_post ~ volunteered", vol, "volunteered")
results["vol_naive_estimate"] = v_naive["est"]
results["vol_naive_ci_lo"] = v_naive["lo"]
results["vol_naive_ci_hi"] = v_naive["hi"]
# Even adjusting for the baseline does not rescue it: motivation is not in the data.
v_anc = ols("cases_post ~ volunteered + cases_pre + C(site)", vol, "volunteered")
results["vol_ancova_estimate"] = v_anc["est"]
# What the tool truly did for these volunteers (only knowable in a simulation):
vt = vol.merge(truth, on="employee_id")
results["vol_true_effect"] = (vt["true_effect_if_used"] * vt["used_tool"])[vt.volunteered == 1].mean()
results["vol_overstatement_ratio"] = results["vol_naive_estimate"] / results["vol_true_effect"]
print(f"Step 1 - volunteer naive: {v_naive['est']:.2f} cases/week "
      f"(truth for volunteers {results['vol_true_effect']:.2f})")

# =============================================================================
# Step 2. Correct: stratified randomized pilot, analyzed as randomized (ITT)
# =============================================================================
t, c = rnd[rnd.assigned == 1], rnd[rnd.assigned == 0]
# Balance check: randomization should make the arms alike at baseline.
results["balance_pre_diff"] = t["cases_pre"].mean() - c["cases_pre"].mean()
pooled_sd = np.sqrt((t["cases_pre"].var() + c["cases_pre"].var()) / 2)
results["balance_pre_std_diff"] = results["balance_pre_diff"] / pooled_sd
results["control_pre_mean"] = c["cases_pre"].mean()
results["control_post_mean"] = c["cases_post"].mean()
results["rho_pre_post"] = np.corrcoef(c["cases_pre"], c["cases_post"])[0, 1]

# (a) simple difference in post means: unbiased, but noisy
simple = ols("cases_post ~ assigned", rnd, "assigned")
# (b) ANCOVA: same comparison, adjusting for the baseline and the strata
ancova = ols("cases_post ~ assigned + cases_pre + C(stratum)", rnd, "assigned")
for nm, r in [("simple", simple), ("ancova", ancova)]:
    results[f"{nm}_estimate"] = r["est"]
    results[f"{nm}_se"] = r["se"]
    results[f"{nm}_ci_lo"] = r["lo"]
    results[f"{nm}_ci_hi"] = r["hi"]
# Sensitivity (printed only, NOT in results.json): heteroskedasticity-robust HC2 SE for the
# headline ANCOVA, as Lin (2013) recommends. If it is close to the classical SE, the CI stands.
hc2 = smf.ols("cases_post ~ assigned + cases_pre + C(stratum)", data=rnd).fit(cov_type="HC2", use_t=True)
hc2_lo, hc2_hi = hc2.conf_int(alpha=0.05).loc["assigned"]
print(f"Sensitivity - ANCOVA with HC2 robust SE: {hc2.params['assigned']:.2f}, SE {hc2.bse['assigned']:.3f}, "
      f"95% CI [{hc2_lo:.2f}, {hc2_hi:.2f}] (classical SE {ancova['se']:.3f})")
results["simple_ci_width"] = simple["hi"] - simple["lo"]
results["ancova_ci_width"] = ancova["hi"] - ancova["lo"]
results["se_reduction_pct"] = 100 * (1 - ancova["se"] / simple["se"])
results["variance_ratio"] = (ancova["se"] / simple["se"]) ** 2
results["theory_variance_factor"] = 1 - results["rho_pre_post"] ** 2
# Equivalent sample size: how many employees an unadjusted analysis would need
# to be as precise as the ANCOVA on the employees we actually have.
results["equivalent_n_unadjusted"] = len(rnd) / results["variance_ratio"]
results["itt_pct_lift"] = 100 * ancova["est"] / results["control_post_mean"]

# Noncompliance: not everyone given access used it. ITT = effect of OFFERING the tool.
results["uptake_pct"] = 100 * t["used_tool"].mean()
# Effect among users (Wald / instrumental-variable logic): ITT / uptake.
# Valid only because nobody in the control arm could use the tool.
results["effect_among_users"] = ancova["est"] / t["used_tool"].mean()

# Pitfall A: compare users with everyone else (breaks randomization: users self-select)
as_treated = ols("cases_post ~ used_tool + cases_pre + C(stratum)", rnd, "used_tool")
results["as_treated_estimate"] = as_treated["est"]
# Pitfall B: adjust for a post-treatment variable (hours of AI use)
post_adj = ols("cases_post ~ assigned + cases_pre + C(stratum) + ai_hours_per_week",
               rnd, "assigned")
results["posttreat_adjusted_estimate"] = post_adj["est"]
results["posttreat_adjusted_ci_lo"] = post_adj["lo"]
results["posttreat_adjusted_ci_hi"] = post_adj["hi"]

# Guardrail: quality must not get worse (non-inferiority margin of -1 point)
q = ols("quality_post ~ assigned + quality_pre + C(stratum)",
        rnd.dropna(subset=["quality_post"]), "assigned")
results["quality_estimate"] = q["est"]
results["quality_ci_lo"] = q["lo"]
results["quality_ci_hi"] = q["hi"]
results["quality_noninferior"] = int(q["lo"] > QUALITY_MARGIN)

print(f"Step 2 - simple diff {simple['est']:.2f} [{simple['lo']:.2f}, {simple['hi']:.2f}]; "
      f"ANCOVA {ancova['est']:.2f} [{ancova['lo']:.2f}, {ancova['hi']:.2f}]; "
      f"SE -{results['se_reduction_pct']:.0f}%")

# =============================================================================
# Step 3. Check against the ground truth
# =============================================================================
rt = rnd.merge(truth, on="employee_id")
results["true_effect_if_used"] = float(truth["true_effect_if_used"].iloc[0])
results["true_itt_design"] = results["true_effect_if_used"] * truth["complier"].mean()
# The true ITT for the employees we analyzed: effect x share who would use it if offered
results["true_itt_sample"] = (rt["true_effect_if_used"] * rt["complier"]).mean()

# 500 re-randomizations. Because the data are synthetic we know BOTH potential outcomes:
#   without access: y0 = observed post - effect * used
#   with access:    y1 = y0 + effect * complier
# We then re-draw the assignment 500 times with the same stratified procedure and
# re-estimate. Vectorized with the Frisch-Waugh-Lovell trick: residualize the outcome
# and the assignment on the controls, then b = sum(z~ * y~) / sum(z~^2).
tau = (rt["true_effect_if_used"] * rt["complier"]).to_numpy()
y0 = (rt["cases_post"] - rt["true_effect_if_used"] * rt["used_tool"]).to_numpy()
Z = rerand.set_index("employee_id").loc[rt["employee_id"]].to_numpy().astype(float)
Y = y0[:, None] + tau[:, None] * Z
n = len(rt)
truth_sim = results["true_itt_sample"]


def sim_fit(W):
    proj = lambda M: M - W @ np.linalg.solve(W.T @ W, W.T @ M)
    Zt, Yt = proj(Z), proj(Y)
    szz = (Zt ** 2).sum(axis=0)
    b = (Zt * Yt).sum(axis=0) / szz
    df_res = n - W.shape[1] - 1
    rss = ((Yt - Zt * b) ** 2).sum(axis=0)
    se = np.sqrt(rss / df_res / szz)
    tcrit = stats.t.ppf(0.975, df_res)
    cover = np.abs(b - truth_sim) <= tcrit * se
    return b, se, cover


strata_dummies = pd.get_dummies(rt["stratum"], drop_first=True).to_numpy(float)
W_simple = np.ones((n, 1))
W_ancova = np.column_stack([np.ones(n), rt["cases_pre"].to_numpy(), strata_dummies])
sim = {}
for nm, W in [("simple", W_simple), ("ancova", W_ancova)]:
    b, se, cover = sim_fit(W)
    sim[nm] = b
    results[f"sim_{nm}_mean_estimate"] = b.mean()
    results[f"sim_{nm}_bias"] = b.mean() - truth_sim
    results[f"sim_{nm}_empirical_sd"] = b.std(ddof=1)
    results[f"sim_{nm}_mean_se"] = se.mean()
    results[f"sim_{nm}_coverage_pct"] = 100 * cover.mean()
results["sim_n_reps"] = Z.shape[1]
results["sim_sd_ratio"] = results["sim_ancova_empirical_sd"] / results["sim_simple_empirical_sd"]
print(f"Step 3 - truth {truth_sim:.2f}; sim bias simple {results['sim_simple_bias']:+.3f}, "
      f"ANCOVA {results['sim_ancova_bias']:+.3f}; coverage "
      f"{results['sim_simple_coverage_pct']:.1f}% / {results['sim_ancova_coverage_pct']:.1f}%")

# =============================================================================
# Step 4. Decision implication (rule fixed in the experiment plan before the pilot)
# =============================================================================
# Roll out if the ITT CI lower bound > MIN_WORTHWHILE AND quality is non-inferior.
results["decision_rollout"] = int(ancova["lo"] > MIN_WORTHWHILE and results["quality_noninferior"] == 1)
results["weekly_cases_gain_1200"] = ancova["est"] * 1200
print("Step 4 - decision: roll out" if results["decision_rollout"] else "Step 4 - decision: do not roll out")

# =============================================================================
# results.json -- name, value, rounding, one-line meaning, python (and r, filled by R)
# =============================================================================
SPEC = [  # (name, decimals, meaning)
    ("n_rows_raw", 0, "Rows in randomized_pilot.csv before cleaning"),
    ("n_duplicates_removed", 0, "Exact duplicate rows removed"),
    ("n_outage_cells_set_missing", 0, "Employee-weeks set to missing because a team's data feed was out (all zeros)"),
    ("n_leavers", 0, "Employees who left during the 8-week pilot"),
    ("n_excluded_short_post", 0, "Employees excluded for having fewer than 4 valid post weeks"),
    ("n_analysis", 0, "Employees in the randomized analysis sample"),
    ("n_analysis_treated", 0, "Analyzed employees assigned to the tool"),
    ("n_analysis_control", 0, "Analyzed employees assigned to control (waitlist)"),
    ("attrition_treated_pct", 1, "Percent of the tool arm with no usable outcome"),
    ("attrition_control_pct", 1, "Percent of the control arm with no usable outcome"),
    ("vol_share_volunteered_pct", 1, "Percent of employees who opted in to the volunteer pilot"),
    ("vol_baseline_gap", 2, "Volunteers' pre-period lead over non-volunteers (cases/week), before any tool"),
    ("vol_naive_estimate", 2, "Naive volunteer comparison: post-period cases/week, opt-in minus non-opt-in"),
    ("vol_naive_ci_lo", 2, "Lower 95% CI bound of the naive volunteer comparison"),
    ("vol_naive_ci_hi", 2, "Upper 95% CI bound of the naive volunteer comparison"),
    ("vol_ancova_estimate", 2, "Volunteer comparison adjusted for baseline and site: still biased (motivation unobserved)"),
    ("vol_true_effect", 2, "True average effect of the tool on the volunteers (cases/week)"),
    ("vol_overstatement_ratio", 1, "Naive volunteer estimate divided by the volunteers' true effect"),
    ("balance_pre_diff", 2, "Randomized arms: baseline difference in cases/week (tool minus control)"),
    ("balance_pre_std_diff", 3, "Randomized arms: standardized baseline difference"),
    ("control_pre_mean", 1, "Control arm mean cases/week in the pre-period"),
    ("control_post_mean", 1, "Control arm mean cases/week in the pilot period"),
    ("rho_pre_post", 2, "Correlation of pre- and post-period cases/week (control arm)"),
    ("simple_estimate", 2, "Randomized pilot, simple difference in post means (ITT, cases/week)"),
    ("simple_se", 2, "Standard error of the simple difference"),
    ("simple_ci_lo", 2, "Lower 95% CI bound, simple difference"),
    ("simple_ci_hi", 2, "Upper 95% CI bound, simple difference"),
    ("simple_ci_width", 2, "Width of the 95% CI, simple difference"),
    ("ancova_estimate", 2, "HEADLINE: randomized pilot, ANCOVA (pre-period + strata) ITT effect, cases/week"),
    ("ancova_se", 2, "Standard error of the ANCOVA estimate"),
    ("ancova_ci_lo", 2, "Lower 95% CI bound, ANCOVA"),
    ("ancova_ci_hi", 2, "Upper 95% CI bound, ANCOVA"),
    ("ancova_ci_width", 2, "Width of the 95% CI, ANCOVA"),
    ("se_reduction_pct", 0, "Percent smaller standard error from ANCOVA vs simple difference"),
    ("variance_ratio", 2, "Variance of ANCOVA estimate relative to simple difference"),
    ("theory_variance_factor", 2, "Theory: 1 - rho^2, the expected variance ratio"),
    ("equivalent_n_unadjusted", -1, "Employees an unadjusted analysis would need to match ANCOVA's precision (to nearest 10)"),
    ("itt_pct_lift", 1, "ANCOVA ITT as percent of control post-period mean"),
    ("uptake_pct", 1, "Percent of employees given access who actually used the tool"),
    ("effect_among_users", 2, "Effect among users = ITT / uptake (cases/week)"),
    ("as_treated_estimate", 2, "Pitfall: users vs everyone else (ANCOVA) - biased by self-selection into use"),
    ("posttreat_adjusted_estimate", 2, "Pitfall: ANCOVA that also adjusts for AI hours (post-treatment) - biased"),
    ("posttreat_adjusted_ci_lo", 2, "Lower 95% CI bound of the post-treatment-adjusted estimate"),
    ("posttreat_adjusted_ci_hi", 2, "Upper 95% CI bound of the post-treatment-adjusted estimate"),
    ("quality_estimate", 2, "Guardrail: ANCOVA effect on QA quality score (points, 0-100)"),
    ("quality_ci_lo", 2, "Lower 95% CI bound, quality"),
    ("quality_ci_hi", 2, "Upper 95% CI bound, quality"),
    ("quality_noninferior", 0, "1 if the quality CI lower bound is above the -1 point margin"),
    ("true_effect_if_used", 2, "TRUTH: effect of the tool on an employee who uses it (cases/week)"),
    ("true_itt_design", 2, "TRUTH: effect of giving access = 2.5 x 80% uptake"),
    ("true_itt_sample", 2, "TRUTH: effect of giving access for the analyzed employees"),
    ("sim_n_reps", 0, "Number of re-randomizations in the simulation check"),
    ("sim_simple_mean_estimate", 2, "Simulation: average simple-difference estimate"),
    ("sim_simple_bias", 2, "Simulation: bias of the simple difference"),
    ("sim_simple_empirical_sd", 2, "Simulation: spread (SD) of the simple-difference estimates"),
    ("sim_simple_mean_se", 2, "Simulation: average reported SE, simple difference"),
    ("sim_simple_coverage_pct", 1, "Simulation: percent of 95% CIs covering the truth, simple difference"),
    ("sim_ancova_mean_estimate", 2, "Simulation: average ANCOVA estimate"),
    ("sim_ancova_bias", 2, "Simulation: bias of ANCOVA"),
    ("sim_ancova_empirical_sd", 2, "Simulation: spread (SD) of the ANCOVA estimates"),
    ("sim_ancova_mean_se", 2, "Simulation: average reported SE, ANCOVA"),
    ("sim_ancova_coverage_pct", 1, "Simulation: percent of 95% CIs covering the truth, ANCOVA"),
    ("sim_sd_ratio", 2, "Simulation: ANCOVA spread relative to simple-difference spread"),
    ("decision_rollout", 0, "1 = pre-registered rule says roll out (ITT CI above 0 and quality non-inferior)"),
    ("weekly_cases_gain_1200", -1, "ITT x 1,200 employees: extra cases/week if everyone is given access (to nearest 10)"),
]


def rnd_val(x, d):
    v = round(float(x), d) + 0.0   # +0.0 turns -0.0 into 0.0
    return int(v) if d <= 0 else v


out = {"edition": "B01", "data": "synthetic", "units": "cases resolved per employee per week unless noted",
       "results": [{"name": nm, "value": rnd_val(results[nm], d), "rounding": d, "meaning": m,
                    "python": rnd_val(results[nm], d), "r": None} for nm, d, m in SPEC]}
assert set(results) == {s[0] for s in SPEC}, set(results) ^ {s[0] for s in SPEC}
with open(os.path.join(OUT, "results.json"), "w") as f:
    json.dump(out, f, indent=2)
# full-precision copy, so the R script can show how closely it matches
with open(os.path.join(OUT, "results_python_full_precision.json"), "w") as f:
    json.dump({k: float(v) for k, v in results.items()}, f, indent=2)

# =============================================================================
# Figures (phone-readable; finding as the title; labeled synthetic)
# =============================================================================
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
plt.rcParams.update({"font.size": 15, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK, "figure.facecolor": SURF,
                     "axes.facecolor": SURF, "font.family": "DejaVu Sans"})

# ---- Figure 1: estimates with 95% CIs vs the truth
rows = [("Volunteer pilot\nopt-in vs. not", v_naive, ORANGE),
        ("Randomized + adjusted\nfor AI hours (wrong)", post_adj, ORANGE),
        ("Randomized\nsimple difference", simple, BLUE),
        ("Randomized\nANCOVA (baseline + strata)", ancova, BLUE)]
fig, ax = plt.subplots(figsize=(8, 8.4), dpi=200)
ypos = np.arange(len(rows))[::-1]
for y, (lab, r, col) in zip(ypos, rows):
    ax.plot([r["lo"], r["hi"]], [y, y], color=col, lw=4, solid_capstyle="round")
    ax.plot(r["est"], y, "o", ms=13, color=col, mec=SURF, mew=2)
    ax.text(r["hi"] + 0.25, y, f"{r['est']:+.1f}", va="center", ha="left", fontsize=16,
            color=INK, fontweight="bold")
ax.axvline(results["true_itt_sample"], color=INK, lw=2, ls="--")
ax.text(results["true_itt_sample"] + 0.1, len(rows) - 0.45,
        f"True effect of access: +{results['true_itt_sample']:.1f}", fontsize=13.5, color=INK, va="bottom")
ax.axvline(0, color=GRAY, lw=1)
ax.set_yticks(ypos, [r[0] for r in rows], fontsize=14.5)
ax.set_xlim(min(-1.5, post_adj["lo"] - 0.5), v_naive["hi"] + 1.6)
ax.set_ylim(-0.6, len(rows) - 0.1)
ax.set_xlabel("Effect on cases resolved per employee per week\n(estimate and 95% CI)", fontsize=14)
ax.grid(axis="x", color=GRID, lw=1)
ax.set_axisbelow(True)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
fig.suptitle("Volunteer pilots overstate the gain;\nrandomize and adjust for the baseline",
             fontsize=19, fontweight="bold", color=INK, x=0.03, ha="left", y=0.985)
fig.text(0.03, 0.885, f"Volunteers truly gained +{results['vol_true_effect']:.1f}; "
         f"the naive comparison says +{v_naive['est']:.1f}", fontsize=13, color=INK2)
fig.text(0.03, 0.012, "SYNTHETIC DATA - illustration only (1,200 simulated employees). "
         "B01, Measure Digital Transformation", fontsize=10.5, color=INK2)
fig.subplots_adjust(left=0.38, right=0.97, top=0.84, bottom=0.14)
fig.savefig(os.path.join(OUT, "fig_1_estimates.png"))
plt.close(fig)

# ---- Figure 2: the 500 re-randomizations, simple vs ANCOVA (two stacked panels, shared x)
fig, axes = plt.subplots(2, 1, figsize=(8, 8.4), dpi=200, sharex=True, sharey=True)
lo_x = min(sim["simple"].min(), sim["ancova"].min()) - 0.05
hi_x = max(sim["simple"].max(), sim["ancova"].max()) + 0.05
bins = np.linspace(lo_x, hi_x, 34)
panels = [("simple", "Simple difference", GRAY), ("ancova", "ANCOVA (baseline + strata)", BLUE)]
for ax, (key, lab, col) in zip(axes, panels):
    b = sim[key]
    ax.hist(b, bins=bins, color=col, edgecolor=SURF, lw=1.5)
    ax.axvline(truth_sim, color=INK, lw=2, ls="--")
    lo95, hi95 = np.percentile(b, [2.5, 97.5])
    ax.set_title(f"{lab}: middle 95% of estimates {lo95:.1f} to {hi95:.1f}",
                 loc="left", fontsize=14.5, color=INK, pad=8)
    ax.grid(axis="y", color=GRID, lw=1)
    ax.set_axisbelow(True)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    ax.set_ylabel("Re-randomizations", fontsize=13)
axes[0].text(truth_sim + 0.04, axes[0].get_ylim()[1] * 0.92, f"Truth +{truth_sim:.1f}",
             fontsize=14, color=INK, ha="left", va="top")
axes[1].set_xlabel("Estimated effect of access (cases/week),\n500 re-randomizations of the same employees",
                   fontsize=14)
fig.suptitle(f"Same pilot, same headcount: baseline\nadjustment cuts the noise by "
             f"{100 * (1 - results['sim_sd_ratio']):.0f}%",
             fontsize=19, fontweight="bold", color=INK, x=0.03, ha="left", y=0.985)
fig.text(0.03, 0.865, f"Both centered on the truth; their 95% CIs cover it "
         f"{results['sim_simple_coverage_pct']:.0f}% and {results['sim_ancova_coverage_pct']:.0f}% of the time",
         fontsize=13, color=INK2)
fig.text(0.03, 0.012, "SYNTHETIC DATA - illustration only. B01, Measure Digital Transformation",
         fontsize=10.5, color=INK2)
fig.subplots_adjust(left=0.12, right=0.97, top=0.80, bottom=0.14, hspace=0.35)
fig.savefig(os.path.join(OUT, "fig_2_simulation.png"))
plt.close(fig)
print("Wrote outputs/results.json, outputs/fig_1_estimates.png, outputs/fig_2_simulation.png")
