"""
analysis.py -- A01 "Adoption Is Not Impact: How to Prove a GenAI Rollout Actually Worked"

Run from the code folder root:
    python3 data/generate_data.py      # (optional) regenerate the synthetic data
    Rscript r/analysis.R               # writes outputs/r_headline.csv
    python3 python/analysis.py         # writes results.json, figures, article chart data

The business question: "We rolled out an AI assistant to our support teams and
productivity went up. How much of that rise did the assistant cause?"

Measurement strategy: the outcome is a rung-3 work outcome (tickets resolved per
agent-hour), paired with a quality metric (reopen rate). Usage (weekly active share)
shows adoption but is not evidence of impact. The rollout came in three waves
(weeks 7, 13, 19), so teams that have not launched yet are a natural comparison group.

Steps
  Step 0  Clean the extract.
  Step 1  The naive approach: before-after for wave 1, and why it misleads.
  Step 2  The correct method: classic 2x2 difference-in-differences with
          team-clustered standard errors.
  Step 3  Checks: (a) parallel pre-trends, (b) staggered adoption: naive two-way
          fixed effects vs. a group-time (Callaway & Sant'Anna style) estimator,
          (c) every estimate against the true effect, (d) a 30-seed Monte Carlo.
  Step 4  Decision implication: what the effect is worth and whether quality held.

Dependencies: numpy, pandas, statsmodels, matplotlib (scipy, a statsmodels dependency,
for one chi-square p-value).
"""
import importlib.util
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import matplotlib

matplotlib.use("Agg")
sys.dont_write_bytecode = True   # importing the generator for the Monte Carlo leaves no __pycache__
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "support_team_weeks.csv"
TRUE = ROOT / "data" / "true_parameters.csv"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)
Z = stats.norm.ppf(0.975)        # normal critical value for 95% CIs
GROUPS = (7, 13)                 # launch weeks with identified effects (wave 3 = control only)
LAST_WEEK_IDENTIFIED = 18        # after week 18 no team is untreated


# =============================================================================
# Step 0. Clean the extract
# =============================================================================
def clean(raw):
    """Fix region labels, drop double-loaded team-weeks and rows with no hours."""
    d = raw.copy()
    d["region"] = d["region"].str.strip().str.upper()
    dup = d.duplicated(["team_id", "week"])
    d = d[~dup]
    hours = pd.to_numeric(d["agent_hours"], errors="coerce")
    blank = hours.isna() | (hours <= 0)
    d = d[~blank].copy()
    d["agent_hours"] = hours[~blank]
    d = d.sort_values(["team_id", "week"]).reset_index(drop=True)
    # Outcomes recomputed from counts so Python and R use identical numbers.
    d["y"] = d.tickets_resolved / d.agent_hours            # tickets per agent-hour
    d["reopen"] = d.reopened_tickets / d.tickets_resolved   # quality
    d["usage"] = d.ai_active_agents / d.n_agents            # weekly active share
    d["D"] = (d.week >= d.launch_week).astype(int)          # has the assistant
    return d, int(dup.sum()), int(blank.sum())


def cluster_fit(formula, data):
    """OLS with team-clustered SEs (CR1: G/(G-1) * (N-1)/(N-K), the same as r/analysis.R)."""
    groups = pd.factorize(data["team_id"])[0]
    return smf.ols(formula, data).fit(cov_type="cluster", cov_kwds={"groups": groups})


def true_effect_fn(tp, launch):
    """Return f(team, week) = true effect of the assistant, from data/true_parameters.csv."""
    g = tp[tp.level == "global"].set_index("parameter")["value"].astype(float)
    ramp = [g["ramp_share_e0"], g["ramp_share_e1"], g["ramp_share_e2"]]
    tau = tp[tp.level == "team"].set_index("id")["value"].astype(float)

    def f(team, week):
        e = np.asarray(week) - launch.loc[team].to_numpy()
        share = np.where(e < 0, 0.0, np.where(e < 3, np.take(ramp, np.clip(e, 0, 2)), 1.0))
        return tau.loc[team].to_numpy() * share
    return f


def group_time_att(d, true_effect=None):
    """Callaway & Sant'Anna style group-time ATTs with not-yet-treated controls.

    For launch group g and week t, each team's base B_i is its AVERAGE over the
    pre-launch weeks 1..g-1 (the C&S default uses the single week g-1; the average
    rests on the same parallel-trends assumption but is far less noisy weekly):
        ATT(g,t) = mean_{i in g}(Y_it - B_i) - mean_{j in controls}(Y_jt - B_j)
    Controls: teams not yet treated at t (launch week > t); for pre-launch weeks
    (placebo leads), teams launching after g. Teams missing week t drop from that cell.
    Returns the cell table and the influence-function matrix Psi (teams x cells):
    an aggregate with cell weights w has team-clustered SE sqrt(sum_i (Psi w)_i^2).
    """
    Y = d.pivot(index="team_id", columns="week", values="y")
    G = d.groupby("team_id")["launch_week"].first().loc[Y.index]
    n_g = G.value_counts()
    cells, psis = [], []
    for g in GROUPS:
        B = Y.loc[:, 1:g - 1].mean(axis=1)                  # pre-launch average per team
        for t in range(1, LAST_WEEK_IDENTIFIED + 1):
            ctrl = (G > t) if t >= g else (G > g)
            has_t = Y[t].notna()
            tr = Y.index[(G == g) & has_t]
            co = Y.index[ctrl & has_t]
            dT = Y.loc[tr, t] - B[tr]
            dC = Y.loc[co, t] - B[co]
            psi = pd.Series(0.0, index=Y.index)
            psi[tr] += (dT - dT.mean()) / len(tr)          # treated team's influence
            psi[co] -= (dC - dC.mean()) / len(co)          # control team's influence
            psis.append(psi.to_numpy())
            cells.append({"g": g, "t": t, "e": t - g, "att": dT.mean() - dC.mean(),
                          "n_treated": len(tr), "n_control": len(co), "w_g": n_g[g],
                          "true_att": (np.mean(true_effect(tr, np.full(len(tr), t)))
                                       if true_effect is not None else np.nan)})
    return pd.DataFrame(cells), np.column_stack(psis)


def aggregate(cells, Psi, sel):
    """Weighted average of the selected cells (weights proportional to group size)."""
    w = np.where(sel, cells.w_g, 0.0)
    w = w / w.sum()
    psi = Psi @ w
    return {"est": float(w @ cells.att), "se": float(np.sqrt(np.sum(psi ** 2))),
            "truth": float(w @ cells.true_att)}


raw = pd.read_csv(DATA)
d, n_duplicates, n_blank_hours = clean(raw)
n_teams = d.team_id.nunique()
tp = pd.read_csv(TRUE)
launch = d.groupby("team_id")["launch_week"].first()
true_effect = true_effect_fn(tp, launch)
print(f"Step 0  {len(raw)} raw rows -> {len(d)} clean team-weeks "
      f"({n_duplicates} duplicates, {n_blank_hours} blank hours, "
      f"{n_teams * 24 - len(d)} team-weeks missing from the extract)")

# =============================================================================
# Step 1. Naive before-after for wave 1 (weeks 1-12)
#   mean(weeks 7-12) - mean(weeks 1-6). It credits the assistant with everything that
#   changed, including the trend that lifted every team.
# =============================================================================
s12 = d[d.week <= 12].copy()
s12["treat"] = (s12.wave == 1).astype(int)
s12["post"] = (s12.week >= 7).astype(int)
cell = s12.groupby(["treat", "post"])["y"].mean()
w1_pre, w1_post, c_pre, c_post = cell[1, 0], cell[1, 1], cell[0, 0], cell[0, 1]
naive = w1_post - w1_pre
comparison_change = c_post - c_pre
fit_naive = cluster_fit("y ~ post", s12[s12.treat == 1])
print(f"Step 1  naive before-after, wave 1: {naive:+.3f} (SE {fit_naive.bse['post']:.3f})")
print(f"        comparison teams (waves 2-3) changed by {comparison_change:+.3f} without the assistant")

# =============================================================================
# Step 2. Classic 2x2 DiD: y = a + b*treat + c*post + DiD*(treat x post)
# =============================================================================
fit_did = cluster_fit("y ~ treat * post", s12)
did, did_se = fit_did.params["treat:post"], fit_did.bse["treat:post"]
print(f"Step 2  2x2 DiD: {did:+.3f} (team-clustered SE {did_se:.3f}, "
      f"95% CI {did - Z * did_se:.3f} to {did + Z * did_se:.3f})")
fit_reopen = cluster_fit("reopen ~ treat * post", s12)
did_reopen, did_reopen_se = fit_reopen.params["treat:post"], fit_reopen.bse["treat:post"]
print(f"        reopen-rate DiD: {did_reopen:+.4f} (SE {did_reopen_se:.4f})")

# =============================================================================
# Step 3a. Parallel pre-trends: same weekly slope before launch?
# =============================================================================
fit_pt = cluster_fit("y ~ treat * week", s12[s12.week <= 6])
pt, pt_se = fit_pt.params["treat:week"], fit_pt.bse["treat:week"]
print(f"Step 3a pre-trend slope difference: {pt:+.4f} per week (SE {pt_se:.4f})")

# =============================================================================
# Step 3b. Staggered adoption over 24 weeks
#   (a) naive two-way fixed effects: y = team FE + week FE + beta * D
#   (b) group-time ATTs aggregated to an overall ATT and an event-time profile
# =============================================================================
fit_tw = cluster_fit("y ~ D + C(team_id) + C(week)", d)
twfe, twfe_se = fit_tw.params["D"], fit_tw.bse["D"]
print(f"Step 3b naive TWFE: {twfe:+.3f} (SE {twfe_se:.3f})")

cells, Psi = group_time_att(d, true_effect)
cs_all = aggregate(cells, Psi, cells.e >= 0)
cs_e3 = aggregate(cells, Psi, cells.e >= 3)
cs_w1_6 = aggregate(cells, Psi, (cells.g == 7) & cells.e.between(0, 5))
ES_E = list(range(-6, 12))
es = pd.DataFrame([{"e": e, **aggregate(cells, Psi, cells.e == e)} for e in ES_E])
# Joint test of placebo leads e = -6..-2 (e = -1 left out: with an average base the
# leads of a group sum to about zero by construction).
W = np.column_stack([np.where(cells.e == e, cells.w_g, 0.0) / np.where(cells.e == e, cells.w_g, 0.0).sum()
                     for e in range(-6, -1)])
lead_est = W.T @ cells.att.to_numpy()
V_leads = (Psi @ W).T @ (Psi @ W)
lead_chi2 = float(lead_est @ np.linalg.solve(V_leads, lead_est))
lead_p = float(stats.chi2.sf(lead_chi2, df=len(lead_est)))
print(f"        group-time ATT overall: {cs_all['est']:+.3f} (SE {cs_all['se']:.3f}); "
      f"established (3+ weeks): {cs_e3['est']:+.3f}; leads chi2 {lead_chi2:.2f}, p = {lead_p:.3f}")

# =============================================================================
# Step 3c. Check against the truth
# =============================================================================
w1r = d[(d.wave == 1) & d.week.between(7, 12)]
true_w1_6 = float(np.mean(true_effect(w1r.team_id, w1r.week)))
trr = d[d.D == 1]
true_all_treated = float(np.mean(true_effect(trr.team_id, trr.week)))
true_cs = cs_all["truth"]
check = pd.DataFrame([
    ("Naive before-after (wave 1)", naive, true_w1_6),
    ("Classic 2x2 DiD (wave 1)", did, true_w1_6),
    ("Naive TWFE (all 24 weeks)", twfe, true_cs),
    ("Group-time ATT, overall", cs_all["est"], true_cs),
    ("Group-time ATT, 3+ weeks after launch", cs_e3["est"], cs_e3["truth"]),
], columns=["estimator", "estimate", "truth"])
check["bias"] = check.estimate - check.truth
print("Step 3c check against the truth\n" + check.round(3).to_string(index=False))
print(f"        (true ATT over all treated team-weeks incl. weeks 19-24: {true_all_treated:.3f})")


# =============================================================================
# Step 3d. Monte Carlo: is this seed typical? Rerun the full design on 30 seeds.
# =============================================================================
def fast_twfe(dd):
    """TWFE coefficient on D (point estimate only), via least squares with dummies."""
    X = pd.get_dummies(dd[["team_id"]].astype(str), drop_first=False).astype(float)
    X = X.join(pd.get_dummies(dd.week.astype(str), prefix="w", drop_first=True).astype(float))
    X.insert(0, "D", dd.D.astype(float))
    return float(np.linalg.lstsq(X.to_numpy(), dd.y.to_numpy(), rcond=None)[0][0])


spec = importlib.util.spec_from_file_location("gen", ROOT / "data" / "generate_data.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)
mc = []
for seed in range(20261101, 20261131):
    raw_s, tp_s = gen.simulate(seed)
    ds, _, _ = clean(raw_s)
    te = true_effect_fn(tp_s, ds.groupby("team_id")["launch_week"].first())
    s = ds[ds.week <= 12]
    a = s[s.wave == 1]
    b = s[s.wave > 1]
    nv = a[a.week >= 7].y.mean() - a[a.week < 7].y.mean()
    dd_ = nv - (b[b.week >= 7].y.mean() - b[b.week < 7].y.mean())
    c_s, P_s = group_time_att(ds, te)
    agg = aggregate(c_s, P_s, c_s.e >= 0)
    t6 = float(np.mean(te(a[a.week >= 7].team_id, a[a.week >= 7].week)))
    mc.append({"seed": seed, "naive": nv, "did": dd_, "true_w1": t6, "twfe": fast_twfe(ds),
               "cs": agg["est"], "true_cs": agg["truth"]})
mc = pd.DataFrame(mc)
mc_summary = {
    "n_seeds": len(mc),
    "seeds": "20261101-20261130",
    "naive_mean": mc.naive.mean(), "did_mean": mc.did.mean(),
    "did_bias_mean": (mc.did - mc.true_w1).mean(),
    "twfe_bias_mean": (mc.twfe - mc.true_cs).mean(),
    "twfe_bias_min": (mc.twfe - mc.true_cs).min(), "twfe_bias_max": (mc.twfe - mc.true_cs).max(),
    "twfe_below_truth_share": float((mc.twfe < mc.true_cs).mean()),
    "cs_bias_mean": (mc.cs - mc.true_cs).mean(),
    "cs_bias_sd": (mc.cs - mc.true_cs).std(),
    "cs_closer_than_twfe_share": float(((mc.cs - mc.true_cs).abs() < (mc.twfe - mc.true_cs).abs()).mean()),
}
print(f"Step 3d Monte Carlo over {len(mc)} seeds: TWFE bias {mc_summary['twfe_bias_mean']:+.3f} on average "
      f"(below truth in {mc_summary['twfe_below_truth_share']:.0%}), group-time ATT bias "
      f"{mc_summary['cs_bias_mean']:+.3f}; naive averages {mc_summary['naive_mean']:+.2f}, DiD {mc_summary['did_mean']:+.2f}")

# =============================================================================
# Step 4. Decision implication
# =============================================================================
hours_per_week = d.agent_hours.mean() * n_teams      # org-wide agent-hours per week
usage_w1 = d.loc[(d.wave == 1) & d.week.between(7, 12), "usage"].mean()
usage_est = d.loc[(d.D == 1) & (d.week - d.launch_week >= 3), "usage"].mean()
reopen_base = s12.loc[(s12.treat == 1) & (s12.post == 0), "reopen"].mean()
print(f"Step 4  per 1,000 agent-hours the naive story claims {1000 * naive:+.0f} extra tickets; "
      f"the DiD supports {1000 * did:+.0f}. Reopen-rate DiD {100 * did_reopen:+.2f} pp "
      f"(95% CI {100 * (did_reopen - Z * did_reopen_se):+.2f} to {100 * (did_reopen + Z * did_reopen_se):+.2f}).")

# =============================================================================
# Article chart data: weekly means, wave 1 vs. comparison (waves 2-3), weeks 1-12
# =============================================================================
wk = s12.groupby(["week", "treat"]).agg(mean=("y", "mean"), n=("y", "size")).unstack("treat")
chart = pd.DataFrame({
    "week": wk.index,
    "period": np.where(wk.index >= 7, "after launch", "before launch"),
    "wave1_mean_tickets_per_agent_hour": wk[("mean", 1)].round(3).to_numpy(),
    "comparison_mean_tickets_per_agent_hour": wk[("mean", 0)].round(3).to_numpy(),
    "wave1_without_ai_counterfactual": np.where(
        wk.index >= 7, (wk[("mean", 0)] + (w1_pre - c_pre)).round(3), np.nan),
    "wave1_n_teams": wk[("n", 1)].to_numpy(),
    "comparison_n_teams": wk[("n", 0)].to_numpy(),
})
chart.to_csv(OUT / "article_chart_data.csv", index=False)

# =============================================================================
# Figures: phone-readable, neutral greys with one blue accent
# =============================================================================
GREY_DARK, GREY, GREY_LIGHT, BLUE = "#333333", "#8a8a8a", "#d9d9d9", "#2f6fd0"
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 17, "axes.edgecolor": GREY,
                     "axes.labelcolor": GREY_DARK, "xtick.color": GREY_DARK, "ytick.color": GREY_DARK})

# Figure 1: pilot (wave 1) vs. comparison teams, with naive gain vs. DiD estimate
fig, ax = plt.subplots(figsize=(12, 7.4), dpi=120)
ax.axvspan(6.5, 12.6, color="#f2f2f2", zorder=0)
ax.axvline(6.5, color=GREY, lw=1.2, ls=":")
y_lo = min(chart.comparison_mean_tickets_per_agent_hour.min(), chart.wave1_mean_tickets_per_agent_hour.min()) - 0.12
y_hi = chart.wave1_mean_tickets_per_agent_hour.max() + 0.08
ax.text(6.65, y_lo + 0.03, "AI assistant launches\nfor wave 1 (week 7)", fontsize=14, color=GREY_DARK, va="bottom")
ax.plot(chart.week, chart.comparison_mean_tickets_per_agent_hour, color=GREY, lw=3, marker="o", ms=7,
        label="Comparison teams (waves 2-3, not yet launched)")
ax.plot(chart.week, chart.wave1_mean_tickets_per_agent_hour, color=BLUE, lw=3.5, marker="o", ms=8,
        label="Wave-1 teams (assistant from week 7)")
cf = chart[chart.week >= 7]
ax.plot(cf.week, cf.wave1_without_ai_counterfactual, color=BLUE, lw=2, ls="--", alpha=0.6,
        label="Wave 1 without the assistant (parallel-trends counterfactual)")
# Brackets: before-after gain splits into "would have happened anyway" + "AI effect".
xb = 12.35
ax.plot([xb, xb], [w1_pre, w1_pre + comparison_change], color=GREY, lw=3, solid_capstyle="butt")
ax.plot([xb, xb], [w1_pre + comparison_change, w1_post], color=BLUE, lw=3, solid_capstyle="butt")
ax.hlines([w1_pre, w1_pre + comparison_change, w1_post], xb - 0.12, xb + 0.12, color=GREY_DARK, lw=1.5)
ax.text(xb + 0.2, w1_pre + comparison_change / 2, f"+{comparison_change:.2f}\nwould have\nhappened\nanyway",
        fontsize=13, color=GREY_DARK, va="center")
ax.text(xb + 0.2, w1_pre + comparison_change + did / 2, f"+{did:.2f}\nAI effect\n(DiD)",
        fontsize=13, color=BLUE, va="center", fontweight="bold")
ax.hlines(w1_pre, 0.8, 10.3, color=BLUE, lw=1, ls=(0, (1, 2)))
ax.hlines(w1_post, 6.6, 10.3, color=BLUE, lw=1, ls=(0, (1, 2)))
ax.text(xb - 0.2, w1_pre, f"before avg {w1_pre:.2f}", fontsize=13, color=BLUE, va="center", ha="right")
ax.text(xb - 0.2, w1_post, f"after avg {w1_post:.2f}", fontsize=13, color=BLUE, va="center", ha="right")
ax.set_xlim(0.6, 13.6)
ax.set_xticks(range(1, 13))
ax.set_xlabel("Week")
ax.set_ylabel("Tickets resolved per agent-hour")
ax.set_ylim(y_lo, y_hi)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=1, frameon=False, fontsize=14)
fig.suptitle(f"Wave-1 teams gained +{naive:.2f} tickets per agent-hour after launch,\n"
             f"but +{comparison_change:.2f} would have happened anyway: the AI effect is +{did:.2f}",
             fontsize=20, fontweight="bold", color=GREY_DARK, x=0.04, ha="left", y=0.98)
fig.text(0.04, 0.008, f"Synthetic data: 60 support teams, weekly. DiD 95% CI {did - Z * did_se:.2f} to "
         f"{did + Z * did_se:.2f} (team-clustered). True effect over weeks 7-12: {true_w1_6:.2f}.",
         fontsize=12, color=GREY)
fig.subplots_adjust(left=0.09, right=0.97, top=0.85, bottom=0.30)
fig.savefig(OUT / "fig_1_pilot_vs_comparison.png")
plt.close(fig)

# Figure 2: event-study profile from the group-time estimator vs. the true effect
fig, ax = plt.subplots(figsize=(12, 7.4), dpi=120)
ax.axhline(0, color=GREY, lw=1)
ax.axvline(-0.5, color=GREY, lw=1.2, ls=":")
ax.step(es.e, es.truth, where="mid", color=GREY_DARK, lw=2.5, label="True effect (from the simulation)")
ax.errorbar(es.e, es.est, yerr=Z * es.se, fmt="o", color=BLUE, ecolor=BLUE, elinewidth=1.8, capsize=4,
            ms=9, label="Group-time estimator (95% CI)")
ax.hlines(twfe, -0.4, 11.4, color=GREY, lw=2.5, ls="--", label=f"Naive TWFE: a single number, +{twfe:.2f}")
ax.text(11.4, twfe - 0.03, f"TWFE +{twfe:.2f}", color=GREY_DARK, fontsize=14, ha="right", va="top")
ax.text(-6, 0.47, "Before launch:\nestimates near zero\n(parallel pre-trends)", fontsize=14, color=GREY_DARK, va="top")
ax.text(5.6, 0.12, "Weeks 6-11: wave 1 only", fontsize=13, color=GREY, ha="left")
ax.set_xticks(ES_E)
ax.set_xlabel("Weeks since the team's launch")
ax.set_ylabel("Effect on tickets per agent-hour")
ax.set_ylim(-0.15, 0.55)
ax.set_xlim(-6.6, 11.6)
ax.spines[["top", "right"]].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2, frameon=False, fontsize=14)
fig.suptitle(f"The staggered estimator recovers the true effect (+{cs_all['est']:.2f} vs. true +{true_cs:.2f});\n"
             f"naive two-way fixed effects understates it (+{twfe:.2f})",
             fontsize=20, fontweight="bold", color=GREY_DARK, x=0.04, ha="left", y=0.98)
fig.text(0.04, 0.008, "Synthetic data: 60 support teams, 3 rollout waves (weeks 7, 13, 19). Controls: teams not "
         "yet launched.\nWave 3 has no untreated comparison after week 18, so it serves only as a control.",
         fontsize=12, color=GREY)
fig.subplots_adjust(left=0.1, right=0.97, top=0.85, bottom=0.27)
fig.savefig(OUT / "fig_2_event_study_vs_truth.png")
plt.close(fig)

# =============================================================================
# results.json (every number the article may quote), with R values merged in
# =============================================================================
r_path = OUT / "r_headline.csv"
r_vals = pd.read_csv(r_path).set_index("name")["value"].to_dict() if r_path.exists() else {}


def rnd(v, k):
    v = round(float(v), k) + 0.0
    return int(v) if k == 0 else v


def item(name, value, rounding, meaning, unit="tickets per agent-hour", both=True):
    """One reported number. both=True: computed in Python and R, which must agree."""
    d_ = {"name": name, "value": rnd(value, rounding), "rounding": rounding, "unit": unit, "meaning": meaning}
    if both:
        d_["python"] = rnd(value, rounding)
        rv = r_vals.get(name)
        d_["r"] = None if rv is None or pd.isna(rv) else rnd(rv, rounding)
        d_["python_r_match"] = d_["r"] is not None and abs(d_["python"] - d_["r"]) < 10 ** (-rounding) / 2 + 1e-12
    return d_


T = "tickets per agent-hour"
res = [
    item("n_rows_raw", len(raw), 0, "Rows in the raw extract.", "rows"),
    item("n_duplicates", n_duplicates, 0, "Double-loaded team-weeks removed.", "rows"),
    item("n_blank_hours", n_blank_hours, 0, "Team-weeks dropped because agent_hours was blank.", "rows"),
    item("n_rows_clean", len(d), 0, "Clean team-weeks analysed (60 teams x 24 weeks minus gaps).", "rows"),
    item("n_missing_teamweeks", n_teams * 24 - len(d), 0, "Team-weeks absent or unusable (holidays, outages, blank hours).", "rows"),
    item("wave1_pre_mean", w1_pre, 3, "Wave-1 mean, weeks 1-6 (before launch)."),
    item("wave1_post_mean", w1_post, 3, "Wave-1 mean, weeks 7-12 (after launch)."),
    item("comparison_pre_mean", c_pre, 3, "Comparison (waves 2-3) mean, weeks 1-6."),
    item("comparison_post_mean", c_post, 3, "Comparison (waves 2-3) mean, weeks 7-12 (still without the assistant)."),
    item("naive_before_after_wave1", naive, 3, "NAIVE: wave-1 mean after minus before (weeks 7-12 vs 1-6). Overstates the effect about 2x."),
    item("naive_before_after_se", fit_naive.bse["post"], 3, "Team-clustered SE of the naive gain (precise, but precisely wrong)."),
    item("comparison_change", comparison_change, 3, "Change in comparison teams over the same weeks: what would have happened anyway."),
    item("did_2x2", did, 3, "Classic 2x2 DiD: wave 1 vs not-yet-treated teams, weeks 1-12. The article's AI effect."),
    item("did_2x2_se", did_se, 3, "Team-clustered (CR1) standard error of the 2x2 DiD."),
    item("did_2x2_ci_low", did - Z * did_se, 3, "95% CI lower bound of the 2x2 DiD (normal approximation)."),
    item("did_2x2_ci_high", did + Z * did_se, 3, "95% CI upper bound of the 2x2 DiD; excludes the naive +0.60."),
    item("naive_to_did_ratio", naive / did, 2, "Naive gain divided by the DiD effect (about 2: half the gain was not the AI).", "ratio"),
    item("did_share_of_naive", did / naive, 3, "Share of the naive gain that the assistant caused.", "proportion"),
    item("did_pct_of_baseline", did / w1_pre, 3, "DiD effect relative to wave-1 pre-launch level.", "proportion"),
    item("pretrend_slope_diff", pt, 4, "Parallel-trends check: wave-1 minus comparison weekly slope, weeks 1-6 (should be ~0).", "tickets per agent-hour per week"),
    item("pretrend_slope_diff_se", pt_se, 4, "Team-clustered SE of the pre-trend slope difference.", "tickets per agent-hour per week"),
    item("twfe_att", twfe, 3, "NAIVE staggered: two-way fixed effects coefficient on 'has assistant', all 24 weeks. Biased low."),
    item("twfe_se", twfe_se, 3, "Team-clustered SE of the TWFE coefficient (precise but biased)."),
    item("twfe_ci_low", twfe - Z * twfe_se, 3, "95% CI lower bound, TWFE."),
    item("twfe_ci_high", twfe + Z * twfe_se, 3, "95% CI upper bound, TWFE; excludes the true effect over the group-time cells (0.318) and over all treated team-weeks (0.335)."),
    item("cs_att_overall", cs_all["est"], 3, "Group-time (Callaway & Sant'Anna style) overall ATT, waves 1-2, weeks up to 18."),
    item("cs_att_overall_se", cs_all["se"], 3, "Team-clustered SE (influence function) of the overall group-time ATT."),
    item("cs_att_overall_ci_low", cs_all["est"] - Z * cs_all["se"], 3, "95% CI lower bound, overall group-time ATT."),
    item("cs_att_overall_ci_high", cs_all["est"] + Z * cs_all["se"], 3, "95% CI upper bound, overall group-time ATT."),
    item("cs_att_established", cs_e3["est"], 3, "Group-time ATT 3+ weeks after launch: the established effect."),
    item("cs_att_established_se", cs_e3["se"], 3, "SE of the established effect."),
    item("cs_att_wave1_weeks7to12", cs_w1_6["est"], 3, "Group-time ATT for wave 1, weeks 7-12 (same target as the 2x2 DiD)."),
    item("cs_leads_chi2", lead_chi2, 2, "Joint chi-square of placebo leads, event times -6 to -2 (5 df).", "chi-square"),
    item("cs_leads_p", lead_p, 3, "p-value of the placebo-lead test (large = no sign of diverging pre-trends).", "p-value"),
    item("true_att_wave1_weeks7to12", true_w1_6, 3, "TRUTH: average true effect, wave 1, weeks 7-12 (target of naive and 2x2 DiD)."),
    item("true_att_cs_cells", true_cs, 3, "TRUTH: average true effect over the cells the group-time ATT covers (target of TWFE and group-time ATT)."),
    item("true_att_established", cs_e3["truth"], 3, "TRUTH: average true effect 3+ weeks after launch (same cells as cs_att_established)."),
    item("true_att_all_treated", true_all_treated, 3, "TRUTH: average true effect over every treated team-week incl. weeks 19-24."),
    item("bias_naive", naive - true_w1_6, 3, "Naive before-after minus truth."),
    item("bias_did_2x2", did - true_w1_6, 3, "2x2 DiD minus truth."),
    item("bias_twfe_vs_cs_cells", twfe - true_cs, 3, "TWFE minus truth (same cells as the group-time ATT)."),
    item("bias_twfe_vs_all_treated", twfe - true_all_treated, 3, "TWFE minus truth over all treated team-weeks."),
    item("bias_cs", cs_all["est"] - true_cs, 3, "Group-time ATT minus truth."),
    item("did_reopen_rate", did_reopen, 4, "Quality: 2x2 DiD on reopen rate (true effect 0). Speed rose, quality held.", "proportion"),
    item("did_reopen_rate_se", did_reopen_se, 4, "Team-clustered SE, reopen-rate DiD.", "proportion"),
    item("did_reopen_rate_ci_low", did_reopen - Z * did_reopen_se, 4, "95% CI lower bound, reopen-rate DiD.", "proportion"),
    item("did_reopen_rate_ci_high", did_reopen + Z * did_reopen_se, 4, "95% CI upper bound, reopen-rate DiD.", "proportion"),
    item("reopen_rate_baseline_wave1", reopen_base, 4, "Wave-1 reopen rate before launch.", "proportion"),
    item("usage_share_wave1_weeks7to12", usage_w1, 3, "Rung 1: wave-1 weekly active share of agents, weeks 7-12.", "proportion"),
    item("usage_share_established", usage_est, 3, "Rung 1: weekly active share 3+ weeks after launch, all launched teams.", "proportion"),
    item("agent_hours_per_week_org", hours_per_week, 0, "Org-wide agent-hours per week (60 teams).", "hours"),
    item("extra_tickets_per_1000_hours_naive", 1000 * naive, 0, "What the naive story claims: extra tickets per 1,000 agent-hours.", "tickets"),
    item("extra_tickets_per_1000_hours_did", 1000 * did, 0, "What the evidence supports: extra tickets per 1,000 agent-hours (DiD).", "tickets"),
    item("extra_tickets_per_week_org_established", cs_e3["est"] * hours_per_week, 0, "Extra tickets per week org-wide once established (established effect x weekly hours).", "tickets"),
]
for r_ in es.itertuples(index=False):
    tag = f"m{-r_.e}" if r_.e < 0 else str(r_.e)
    res.append(item(f"es_e{tag}", r_.est, 3, f"Event-study estimate, {r_.e} weeks since launch."))
    res.append(item(f"es_se_e{tag}", r_.se, 3, f"SE of the event-study estimate at {r_.e}."))
    res.append(item(f"es_true_e{tag}", r_.truth, 3, f"True effect at {r_.e} weeks since launch (same teams)."))
for k, v in mc_summary.items():
    if k in ("n_seeds", "seeds"):
        continue
    res.append(item(f"mc_{k}", v, 3, f"Monte Carlo (Python only, 30 seeds): {k.replace('_', ' ')}.",
                    "proportion" if "share" in k else T, both=False))

headline = [x for x in res if "python_r_match" in x]
true_params = tp[tp.level.isin(["global", "design_truth"])].to_dict(orient="records")
results = {
    "topic": "A01 Adoption Is Not Impact: How to Prove a GenAI Rollout Actually Worked",
    "data": ("Synthetic. 60 customer-support teams (6 regions x 10), observed weekly for 24 weeks; AI assistant "
             "rolled out in 3 waves of 20 teams (weeks 7, 13, 19). Common trend +0.05/week. True effect ramps "
             "to an established ~0.38 tickets per agent-hour over 3 weeks (about 0.29 averaged over the first "
             "6 weeks). True effect on reopen rate = 0. Seed 20261105."),
    "how_to_reproduce": "python3 data/generate_data.py; Rscript r/analysis.R; python3 python/analysis.py",
    "python_r_all_headline_match": bool(r_vals) and all(x["python_r_match"] for x in headline),
    "ci_method": "95% CIs use the normal approximation (1.96) with team-clustered SEs. CR1 for regressions; influence-function SEs for group-time ATTs (group sizes treated as fixed).",
    "results": res,
    "check_against_truth": check.round(3).to_dict(orient="records"),
    "monte_carlo": {k: (round(v, 3) if isinstance(v, float) else v) for k, v in mc_summary.items()},
    "true_parameters": true_params,
    "figures": ["outputs/fig_1_pilot_vs_comparison.png", "outputs/fig_2_event_study_vs_truth.png"],
    "article_chart_data": "outputs/article_chart_data.csv",
}
with open(OUT / "results.json", "w") as f:
    json.dump(results, f, indent=2, default=float)

print("\nWrote outputs/results.json, 2 figures, outputs/article_chart_data.csv.")
if r_vals:
    bad = [x["name"] for x in headline if not x["python_r_match"]]
    print("Python vs R numbers: " + ("all match." if not bad else f"MISMATCH in {bad}"))
else:
    print("R results not found (run `Rscript r/analysis.R` first to fill the 'r' keys).")
