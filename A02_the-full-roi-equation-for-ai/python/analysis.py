"""
analysis.py -- A02 "The Full ROI Equation for AI Agents" (Python version)

Business question
    What does our AI help-desk agent really cost, and what does it return over
    its first 12 months? How likely is it that we lose money?

Steps
    Step 1  The naive "vendor" ROI and why it misleads
    Step 2  The full cost-benefit model, with Monte Carlo uncertainty
    Step 3  Check against the ground truth (possible only with synthetic data)
    Step 4  Decision implication: P(ROI < 0), median, intervals, drivers,
            break-even

The per-ticket equation (one line, for every ticket routed to the agent):

    net value per agent ticket =
          p(resolved) x human minutes avoided x $/min        (benefit)
        - agent run cost
        - $/min x [p(escalated) x handoff minutes + p(reopened) x rework minutes]
        - p(resolved) x p(error) x remediation cost
        - review share x review minutes x $/min

    12-month net value = sum over queues of (agent tickets x net per ticket)
                         - licence - build - monitoring

    ROI = net value / total cost

Run from the edition folder:   python python/analysis.py
Data are synthetic (see data/generate_data.py).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats          # installed with statsmodels (beta / normal quantiles)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)

QUEUES = ["access", "it_software", "hr_policy"]
LOGGED_WEEKS, HORIZON_WEEKS = 26, 52
RESULTS = []                     # every number the article may quote


def record(name, value, digits, unit, meaning, headline=False):
    """Store a result. `digits` is the rounding used everywhere (Python, R,
    article): 2 -> 0.01, 0 -> whole units, -3 -> nearest thousand."""
    RESULTS.append({"name": name, "value": float(round(value, digits)) + 0.0,  # +0.0 avoids "-0.0"
                    "exact": float(value), "digits": digits, "unit": unit,
                    "meaning": meaning, "headline": headline})


def tri_ppf(u, low, mode, high):
    """Inverse CDF of a triangular(low, mode, high) distribution."""
    u = np.asarray(u, dtype=float)
    if high == low:                                   # a fixed input
        return np.full_like(u, low)
    cut = (mode - low) / (high - low)
    left = low + np.sqrt(u * (high - low) * (mode - low))
    right = high - np.sqrt((1 - u) * (high - low) * (high - mode))
    return np.where(u < cut, left, right)


# =============================================================================
# 0. Load and clean the log
# =============================================================================
raw = pd.read_csv(DATA / "helpdesk_tickets.csv")
costs = pd.read_csv(DATA / "cost_assumptions.csv").set_index("parameter")
truth = pd.read_csv(DATA / "ground_truth.csv").set_index("parameter")["true_value"]
U = pd.read_csv(DATA / "mc_uniforms.csv")

n_raw = len(raw)
log = raw.drop_duplicates()                       # messy case: double export
n_dupes = n_raw - len(log)
assert log.ticket_id.is_unique, "duplicates that are not exact copies"
agent = log[log.routed_to_agent == 1]
n_missing_outcome = int(agent.outcome.isna().sum())
agent_known = agent.dropna(subset=["outcome"])    # assume missing at random

print("=" * 72)
print("DATA CHECK")
print(f"  rows in file: {n_raw:,}   exact duplicates removed: {n_dupes:,}")
print(f"  unique tickets: {len(log):,}   routed to agent: {len(agent):,}")
print(f"  routed tickets with missing outcome (excluded from rates): {n_missing_outcome}")

weekly = log.groupby("week").agg(tickets=("ticket_id", "size"),
                                 routed=("routed_to_agent", "sum"))
weekly["share"] = weekly.routed / weekly.tickets
vol_mean = weekly.tickets.mean()                  # tickets per week, all queues
vol_se = weekly.tickets.std(ddof=1) / np.sqrt(LOGGED_WEEKS)
last4 = log[log.week > LOGGED_WEEKS - 4]
last4_share = last4.routed_to_agent.mean()
agent_last4 = last4[last4.routed_to_agent == 1]
mix = agent_last4.queue.value_counts(normalize=True).reindex(QUEUES)
h1_agent = agent.queue.value_counts().reindex(QUEUES)   # agent tickets, weeks 1-26
h1_share = agent.shape[0] / len(log)

# =============================================================================
# STEP 1. The naive "vendor" ROI
# =============================================================================
# Vendor logic: "the agent already handles X% of tickets; every one of them is
# a ticket a human no longer works; multiply by labour cost; divide by licence."
wage_ml = costs.loc["loaded_hourly_cost", "most_likely"]
licence_12m = 12 * costs.loc["license_fee_per_month", "most_likely"]
human_only = log[(log.outcome == "human_only")].dropna(subset=["human_minutes"])
base_simple = human_only.groupby("queue").human_minutes.mean().reindex(QUEUES)
naive_tickets = HORIZON_WEEKS * vol_mean * last4_share
naive_minutes = float((mix * base_simple).sum())
naive_gross = naive_tickets * naive_minutes * wage_ml / 60
naive_net = naive_gross - licence_12m
naive_roi = naive_net / licence_12m

print("\n" + "=" * 72)
print("STEP 1  Naive (vendor-style) ROI")
print(f"  share routed to agent, last 4 weeks: {last4_share:.1%}")
print(f"  'automated' tickets per year: {naive_tickets:,.0f}  x {naive_minutes:.1f} min"
      f"  x ${wage_ml:.0f}/h = ${naive_gross:,.0f} gross")
print(f"  licence: ${licence_12m:,.0f}  ->  naive ROI = {naive_roi:.2f}x")
print("  Why it misleads: it treats every routed ticket as fully automated, assumes")
print("  the current adoption level held all year, and leaves out escalations,")
print("  rework, review, error remediation, build and monitoring costs.")

record("naive_share_routed_last4", last4_share, 3, "share",
       "Share of tickets routed to the agent in the last 4 logged weeks")
record("naive_gross_savings_12m", naive_gross, -3, "USD",
       "Vendor-style gross savings: routed tickets x baseline minutes x labour cost")
record("naive_net_value_12m", naive_net, -3, "USD",
       "Vendor-style net value: gross savings minus 12 months of licence", True)
record("naive_roi", naive_roi, 2, "x",
       "Vendor-style ROI = (gross savings - licence) / licence", True)

# =============================================================================
# STEP 2. The full model
# =============================================================================
# 2a. Outcome rates per queue from the log (Dirichlet posterior, flat prior)
cnt = (agent_known.groupby(["queue", "outcome"]).size().unstack(fill_value=0)
       .reindex(QUEUES))
n_res, n_esc, n_rw = cnt["resolved_by_agent"], cnt["escalated"], cnt["reopened"]
n_tot = n_res + n_esc + n_rw
p_esc_hat, p_rw_hat = n_esc / n_tot, n_rw / n_tot

# 2b. Silent-error rate from the QA sample of resolved tickets (beta posterior)
qa = agent_known[(agent_known.outcome == "resolved_by_agent") & (agent_known.qa_sampled == 1)]
qa_n = qa.groupby("queue").size().reindex(QUEUES)
qa_e = qa.groupby("queue").qa_error_found.sum().reindex(QUEUES)
p_err_hat = qa_e / qa_n

# 2c. Human minutes: OLS on all human-handled tickets.
#     minutes = base(queue) + extra_esc * escalated + extra_rw * reopened
hh = log[log.outcome.isin(["human_only", "escalated", "reopened"])].dropna(
    subset=["human_minutes"])
X = pd.DataFrame({q: (hh.queue == q).astype(float) for q in QUEUES})
X["escalated"] = (hh.outcome == "escalated").astype(float)
X["reopened"] = (hh.outcome == "reopened").astype(float)
ols = sm.OLS(hh.human_minutes.values, X.values).fit()
coef, vcov = ols.params, ols.cov_params()
coef_names = QUEUES + ["extra_esc", "extra_rw"]

# 2d. Agent compute cost per ticket (mean and standard error per queue)
run = agent.groupby("queue").agent_cost_usd.agg(["mean", "std", "size"]).reindex(QUEUES)
run_se = run["std"] / np.sqrt(run["size"])


def model(p):
    """12-month net value for a dict of inputs (scalars or arrays of draws).

    Ticket counts: weeks 1-26 are the agent tickets actually logged; weeks
    27-52 are projected = 26 x weekly volume x steady-state routing share,
    split across queues with the last-4-week agent mix.
    Returns a dict of totals (benefit, each cost line, net, ROI).
    """
    w = p["wage"] / 60.0
    out = {k: 0.0 for k in ["n_agent", "gross_all", "benefit", "unresolved",
                            "handoff", "remediation", "review", "run"]}
    for i, q in enumerate(QUEUES):
        n_q = h1_agent[q] + (HORIZON_WEEKS - LOGGED_WEEKS) * p["volume"] * p["s2"] * mix[q]
        p_res = 1 - p["p_esc"][i] - p["p_rw"][i]
        base = p["base"][i]
        out["n_agent"] = out["n_agent"] + n_q
        out["gross_all"] = out["gross_all"] + n_q * base * w
        out["benefit"] = out["benefit"] + n_q * p_res * base * w
        out["unresolved"] = out["unresolved"] + n_q * (1 - p_res) * base * w
        out["handoff"] = out["handoff"] + n_q * w * (p["p_esc"][i] * p["extra_esc"]
                                                     + p["p_rw"][i] * p["extra_rw"])
        out["remediation"] = out["remediation"] + n_q * p_res * p["p_err"][i] * p["c_rem"]
        out["review"] = out["review"] + n_q * p["review_share"] * p["review_min"] * w
        out["run"] = out["run"] + n_q * p["run"][i]
    out["licence"] = 12 * p["licence"] + 0 * out["benefit"]
    out["build_monitor"] = p["build"] + 12 * p["monitor"]
    out["var_cost"] = out["handoff"] + out["remediation"] + out["review"] + out["run"]
    out["total_cost"] = out["var_cost"] + out["licence"] + out["build_monitor"]
    out["net"] = out["benefit"] - out["total_cost"]
    out["roi"] = out["net"] / out["total_cost"]
    return out


def mode(name):
    return costs.loc[name, "most_likely"]


# Point estimate: rates and minutes from the log, cost inputs at "most likely"
point = {
    "p_esc": p_esc_hat.values, "p_rw": p_rw_hat.values, "p_err": p_err_hat.values,
    "base": coef[:3], "extra_esc": coef[3], "extra_rw": coef[4],
    "run": run["mean"].values, "volume": vol_mean,
    "wage": mode("loaded_hourly_cost"), "c_rem": mode("remediation_cost_per_error"),
    "review_share": mode("review_share"), "review_min": mode("review_minutes"),
    "build": mode("build_cost"), "monitor": mode("monitoring_cost_per_month"),
    "licence": mode("license_fee_per_month"), "s2": mode("steady_state_routing_share"),
}
pt = model(point)

# 2e. Monte Carlo: turn the shared uniforms into 10,000 input scenarios
def tri(name, u):
    r = costs.loc[name]
    return tri_ppf(u, r["low"], r["most_likely"], r["high"])


n_mc = len(U)
# Dirichlet(1+n_res, 1+n_esc, 1+n_rw) drawn as two betas ("stick breaking"):
#   p_esc ~ Beta(1+n_esc, 2+n_res+n_rw);  p_rw/(1-p_esc) ~ Beta(1+n_rw, 1+n_res)
mc_esc = np.column_stack([stats.beta.ppf(U[f"u_esc_{q}"], 1 + n_esc[q], 2 + n_res[q] + n_rw[q])
                          for q in QUEUES])
mc_rw = np.column_stack([(1 - mc_esc[:, i]) * stats.beta.ppf(U[f"u_rw_{q}"], 1 + n_rw[q], 1 + n_res[q])
                         for i, q in enumerate(QUEUES)])
mc_err = np.column_stack([stats.beta.ppf(U[f"u_err_{q}"], 1 + qa_e[q], 1 + qa_n[q] - qa_e[q])
                          for q in QUEUES])
# OLS coefficients: multivariate normal via Cholesky (keeps their correlation)
Z = stats.norm.ppf(U[[f"u_minutes_{k}" for k in range(1, 6)]].values)
L = np.linalg.cholesky(vcov)
mc_coef = coef + Z @ L.T
draws = {
    "p_esc": mc_esc.T, "p_rw": mc_rw.T, "p_err": mc_err.T,
    "base": mc_coef[:, :3].T, "extra_esc": mc_coef[:, 3], "extra_rw": mc_coef[:, 4],
    "run": np.array([run["mean"][q] + run_se[q] * stats.norm.ppf(U[f"u_run_{q}"])
                     for q in QUEUES]),
    "volume": vol_mean + vol_se * stats.norm.ppf(U["u_volume"].values),
    "wage": tri("loaded_hourly_cost", U["u_loaded_hourly_cost"]),
    "c_rem": tri("remediation_cost_per_error", U["u_remediation_cost_per_error"]),
    "review_share": tri("review_share", U["u_review_share"]),
    "review_min": tri("review_minutes", U["u_review_minutes"]),
    "build": tri("build_cost", U["u_build_cost"]),
    "monitor": tri("monitoring_cost_per_month", U["u_monitoring_cost_per_month"]),
    "licence": mode("license_fee_per_month"),
    "s2": tri("steady_state_routing_share", U["u_steady_state_routing_share"]),
}
mc = model(draws)
net, roi = mc["net"], mc["roi"]
q = lambda x, p: float(np.quantile(x, p))      # numpy "linear" = R type 7

print("\n" + "=" * 72)
print("STEP 2  Full cost-benefit model")
print("  Estimated from the log (95% interval from the posterior / OLS):")
for i, qq in enumerate(QUEUES):
    print(f"   {qq:12s} p(esc)={p_esc_hat[qq]:.3f}  p(reopen)={p_rw_hat[qq]:.3f}"
          f"  p(error|resolved)={p_err_hat[qq]:.3f} ({int(qa_e[qq])}/{int(qa_n[qq])} QA)"
          f"  base min={coef[i]:.1f}  run=${run['mean'][qq]:.3f}")
print(f"   extra minutes: escalated +{coef[3]:.2f}, reopened +{coef[4]:.2f}")
print(f"  Point estimate (inputs at most-likely): net ${pt['net']:,.0f}, ROI {pt['roi']:.2f}x")
print(f"  Monte Carlo ({n_mc:,} draws): median net ${q(net, .5):,.0f}, "
      f"80% [{q(net, .1):,.0f}, {q(net, .9):,.0f}], P(ROI<0) = {np.mean(roi < 0):.1%}")

for i, qq in enumerate(QUEUES):
    record(f"p_escalated_{qq}", p_esc_hat[qq], 3, "share", f"Estimated escalation rate, {qq} queue")
    record(f"p_reopened_{qq}", p_rw_hat[qq], 3, "share", f"Estimated rework (reopen) rate, {qq} queue")
    record(f"p_error_{qq}", p_err_hat[qq], 3, "share",
           f"Estimated silent-error rate among resolved tickets (QA sample), {qq} queue")
    record(f"base_minutes_{qq}", coef[i], 1, "minutes", f"Estimated human minutes per ticket, {qq} queue")
record("extra_minutes_escalated", coef[3], 2, "minutes",
       "Extra human minutes on an escalated ticket vs a human-only ticket")
record("extra_minutes_reopened", coef[4], 2, "minutes",
       "Extra human minutes on a reopened ticket vs a human-only ticket")
record("p_resolved_overall", n_res.sum() / n_tot.sum(), 3, "share",
       "Share of agent tickets truly resolved (not escalated, not reopened), all queues")
record("agent_tickets_12m_point", pt["n_agent"], -2, "tickets",
       "Agent tickets in 12 months: 26 weeks logged + 26 weeks projected at most-likely adoption")
record("h1_share_routed", h1_share, 3, "share", "Share of tickets routed to the agent over weeks 1-26 (ramp)")

# Bridge from vendor gross to full net (point estimate) -- for a waterfall
bridge = [
    ("bridge_vendor_gross", naive_gross, "Vendor gross savings"),
    ("bridge_adoption_ramp", pt["gross_all"] - naive_gross,
     "Adoption ramp: fewer agent tickets than 'current share all year'"),
    ("bridge_not_resolved", -pt["unresolved"], "Escalated/reopened tickets save no human time"),
    ("bridge_handoff_rework", -pt["handoff"], "Extra human minutes on escalations and rework"),
    ("bridge_review", -pt["review"], "Human review (oversight) time"),
    ("bridge_remediation", -pt["remediation"], "Fixing silent errors"),
    ("bridge_run_cost", -pt["run"], "Agent compute cost"),
    ("bridge_licence", -pt["licence"], "Licence, 12 months"),
    ("bridge_build_monitoring", -pt["build_monitor"], "One-off build + 12 months monitoring"),
]
for name, val, meaning in bridge:
    record(name, val, -3, "USD", "Bridge, point estimate: " + meaning)
assert abs(sum(v for _, v, _ in bridge) - pt["net"]) < 1e-6

record("full_net_value_12m_point", pt["net"], -3, "USD",
       "Full-model 12-month net value with all inputs at estimate / most likely", True)
record("full_roi_point", pt["roi"], 2, "x", "Full-model ROI with inputs at estimate / most likely", True)
record("full_total_cost_12m_point", pt["total_cost"], -3, "USD", "Full-model total 12-month cost (point)")
record("full_benefit_12m_point", pt["benefit"], -3, "USD", "Full-model 12-month benefit (point)")
record("net_per_agent_ticket_point", pt["net"] / pt["n_agent"], 2, "USD",
       "Net value per agent ticket after all costs (point)")

# =============================================================================
# STEP 3. Check against the ground truth
# =============================================================================
true_net, true_roi = truth["net_value_12m"], truth["roi_12m"]
lo80, hi80, lo90, hi90 = q(net, .1), q(net, .9), q(net, .05), q(net, .95)
print("\n" + "=" * 72)
print("STEP 3  Check against ground truth (synthetic data only)")
cover = 0
for i, qq in enumerate(QUEUES):
    for lab, est, draws_i in [("escalated", p_esc_hat[qq], mc_esc[:, i]),
                              ("reopened", p_rw_hat[qq], mc_rw[:, i]),
                              ("error", p_err_hat[qq], mc_err[:, i])]:
        t = truth[f"p_{lab}_{qq}"]
        lo, hi = q(draws_i, .025), q(draws_i, .975)
        cover += lo <= t <= hi
        print(f"   {qq:12s} p({lab:9s}) est {est:.3f}  95% [{lo:.3f}, {hi:.3f}]  true {t:.3f}")
print(f"   true rates inside their 95% interval: {cover} of 9")
print(f"   extra minutes est {coef[3]:.2f}/{coef[4]:.2f}  true "
      f"{truth['extra_minutes_escalated']:.2f}/{truth['extra_minutes_reopened']:.2f}")
pct_true = float(np.mean(net <= true_net))
print(f"  True net value ${true_net:,.0f} (ROI {true_roi:.2f}x); "
      f"inside 80% interval: {lo80 <= true_net <= hi80}; inside 90%: {lo90 <= true_net <= hi90}")
print(f"  Naive ROI overstates the true ROI by {naive_roi / true_roi:.1f}x")

record("true_net_value_12m", true_net, -3, "USD", "Ground-truth 12-month net value (from the DGP)", True)
record("true_roi", true_roi, 2, "x", "Ground-truth 12-month ROI (from the DGP)", True)
record("true_inside_80pct_interval", float(lo80 <= true_net <= hi80), 0, "1=yes",
       "Is the true net value inside the simulated 80% interval?")
record("true_inside_90pct_interval", float(lo90 <= true_net <= hi90), 0, "1=yes",
       "Is the true net value inside the simulated 90% interval?")
record("true_net_percentile_in_mc", pct_true, 3, "share",
       "Where the true net value sits in the simulated distribution (0.5 = median)")
record("rates_covered_by_95pct", cover, 0, "of 9",
       "True escalation/rework/error rates inside their 95% posterior interval")
record("naive_overstatement_factor", naive_roi / true_roi, 1, "x",
       "Naive ROI divided by true ROI")

# =============================================================================
# STEP 4. Decision implication
# =============================================================================
# 4a. One-at-a-time sensitivity (tornado): move one input group to its own
#     10th and 90th percentile, everything else at the point estimate.
def with_(**kw):
    d = dict(point)
    d.update(kw)
    return model(d)["net"]


def tri_at(name, u):
    return float(tri(name, np.array([u]))[0])


def norm_at(m, s, u):
    return m + s * stats.norm.ppf(u)


se = np.sqrt(np.diag(vcov))
groups = {
    "Adoption (share routed)": lambda u: with_(s2=tri_at("steady_state_routing_share", u)),
    "Remediation cost per error": lambda u: with_(c_rem=tri_at("remediation_cost_per_error", u)),
    "Human review share": lambda u: with_(review_share=tri_at("review_share", u)),
    "Build cost (one-off)": lambda u: with_(build=tri_at("build_cost", u)),
    "Loaded labour cost": lambda u: with_(wage=tri_at("loaded_hourly_cost", u)),
    "Review minutes per review": lambda u: with_(review_min=tri_at("review_minutes", u)),
    "Monitoring cost": lambda u: with_(monitor=tri_at("monitoring_cost_per_month", u)),
    "Silent-error rates": lambda u: with_(p_err=np.array(
        [stats.beta.ppf(u, 1 + qa_e[qq], 1 + qa_n[qq] - qa_e[qq]) for qq in QUEUES])),
    "Escalation rates": lambda u: with_(p_esc=np.array(
        [stats.beta.ppf(u, 1 + n_esc[qq], 2 + n_res[qq] + n_rw[qq]) for qq in QUEUES])),
    "Rework rates": lambda u: with_(p_rw=np.array(
        [stats.beta.ppf(u, 1 + n_rw[qq], 2 + n_res[qq] + n_esc[qq]) for qq in QUEUES])),
    "Handoff & rework minutes": lambda u: with_(extra_esc=norm_at(coef[3], se[3], u),
                                                extra_rw=norm_at(coef[4], se[4], u)),
    "Baseline human minutes": lambda u: with_(base=np.array([norm_at(coef[i], se[i], u) for i in range(3)])),
    "Ticket volume": lambda u: with_(volume=norm_at(vol_mean, vol_se, u)),
    "Agent run cost": lambda u: with_(run=np.array([norm_at(run["mean"][qq], run_se[qq], u) for qq in QUEUES])),
}
tor = pd.DataFrame([{"input": k, "low_case": float(f(0.10)), "high_case": float(f(0.90))}
                    for k, f in groups.items()])
tor["swing"] = (tor.high_case - tor.low_case).abs()
tor = tor.sort_values("swing", ascending=False).reset_index(drop=True)
tor.to_csv(OUT / "tornado_python.csv", index=False)

# 4b. Break-even conditions (the model is linear in each of these inputs)
def breakeven(key):
    a, b = with_(**{key: 0.0}), with_(**{key: 1.0})
    return -a / (b - a)


be_s2 = breakeven("s2")
be_rem = breakeven("c_rem")
be_review = breakeven("review_share")

print("\n" + "=" * 72)
print("STEP 4  Decision implication")
print(f"  P(ROI < 0) = {np.mean(roi < 0):.1%}")
print(f"  12-month net value: median ${q(net, .5):,.0f}; 80% [{lo80:,.0f}, {hi80:,.0f}];"
      f" 90% [{lo90:,.0f}, {hi90:,.0f}]")
print(f"  ROI: median {q(roi, .5):.2f}x; 80% [{q(roi, .1):.2f}, {q(roi, .9):.2f}]")
print("  Top drivers of the spread (net value at P10 / P90 of the input):")
for _, r in tor.head(5).iterrows():
    print(f"   {r.input:40s} {r.low_case:>10,.0f} {r.high_case:>10,.0f}  swing {r.swing:,.0f}")
print(f"  Break-even steady-state adoption: {be_s2:.1%} of tickets routed (months 7-12)")
print(f"  Break-even remediation cost: ${be_rem:,.0f} per error")
print(f"  Break-even review share: {be_review:.1%} of agent tickets")

record("p_roi_negative", float(np.mean(roi < 0)), 3, "probability",
       "Monte Carlo probability that 12-month ROI is below zero (a loss)", True)
record("mc_median_net_value_12m", q(net, .5), -3, "USD", "Median simulated 12-month net value", True)
record("mc_mean_net_value_12m", float(np.mean(net)), -3, "USD", "Mean simulated 12-month net value")
record("mc_net_p10", lo80, -3, "USD", "10th percentile of net value (lower end of 80% interval)", True)
record("mc_net_p90", hi80, -3, "USD", "90th percentile of net value (upper end of 80% interval)", True)
record("mc_net_p05", lo90, -3, "USD", "5th percentile of net value (lower end of 90% interval)", True)
record("mc_net_p95", hi90, -3, "USD", "95th percentile of net value (upper end of 90% interval)", True)
record("mc_median_roi", q(roi, .5), 2, "x", "Median simulated 12-month ROI", True)
record("mc_roi_p10", q(roi, .1), 2, "x", "10th percentile of ROI", True)
record("mc_roi_p90", q(roi, .9), 2, "x", "90th percentile of ROI", True)
record("mc_roi_p05", q(roi, .05), 2, "x", "5th percentile of ROI")
record("mc_roi_p95", q(roi, .95), 2, "x", "95th percentile of ROI")
record("p_roi_above_0_5", float(np.mean(roi > 0.5)), 3, "probability",
       "Probability ROI exceeds 0.5x")
record("breakeven_steady_state_share", be_s2, 3, "share",
       "Steady-state routing share (months 7-12) at which point-estimate net value is zero", True)
record("breakeven_remediation_cost", be_rem, 0, "USD/error",
       "Remediation cost per error at which point-estimate net value is zero")
record("breakeven_review_share", be_review, 3, "share",
       "Human review share at which point-estimate net value is zero (above 1 = review alone cannot erase the value)")
for i, r in tor.head(3).iterrows():
    record(f"tornado_rank{i + 1}_swing", r.swing, -3, "USD",
           f"Rank {i + 1} sensitivity driver: {r.input} (net value swing P10->P90)")
    record(f"tornado_rank{i + 1}_low_case", r.low_case, -3, "USD",
           f"Net value with {r.input} at its 10th percentile")
    record(f"tornado_rank{i + 1}_high_case", r.high_case, -3, "USD",
           f"Net value with {r.input} at its 90th percentile")

# =============================================================================
# Figures (phone-readable: big fonts, finding as the title)
# =============================================================================
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
BLUE, ORANGE = "#2a78d6", "#eb6834"
plt.rcParams.update({"text.usetex": False, "font.size": 17, "axes.edgecolor": INK2, "axes.labelcolor": INK,
                     "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})


def k(x):
    # "\\$" stops matplotlib from reading two dollar signs as a maths formula
    return f"{'-' if x < 0 else '+'}\\${abs(x) / 1000:,.0f}k"


# Figure 1: distribution of 12-month net value with the vendor number marked
fig, ax = plt.subplots(figsize=(12, 7.5), dpi=150, facecolor=SURF)
ax.set_facecolor(SURF)
bins = np.arange(np.floor(net.min() / 10_000) * 10_000, naive_net + 30_000, 10_000)
hist, edges = np.histogram(net, bins=bins)
share = hist / len(net) * 100
centers = (edges[:-1] + edges[1:]) / 2
ax.bar(centers / 1000, share, width=8.6, color=np.where(centers < 0, ORANGE, BLUE),
       edgecolor=SURF, linewidth=0)
ax.axvspan(lo80 / 1000, hi80 / 1000, color=BLUE, alpha=0.08, zorder=0)
ax.axvline(q(net, .5) / 1000, color=INK, lw=2)
ax.axvline(naive_net / 1000, color=INK2, lw=2, ls="--")
ax.axvline(0, color=INK2, lw=1)
ymax = share.max()
ax.plot([lo80 / 1000, hi80 / 1000], [ymax * 1.08] * 2, color=BLUE, lw=3)
ax.text(hi80 / 1000 + 6, ymax * 1.08, f"80% of scenarios: {k(lo80)} to {k(hi80)}", color=BLUE,
        fontsize=16, va="center")
ax.text(hi80 / 1000 + 6, ymax * 0.93, f"Median {k(q(net, .5))}", color=INK, fontsize=17,
        fontweight="bold")
ax.text(naive_net / 1000 - 6, ymax * 0.62, f"Vendor math\n{k(naive_net)}", ha="right",
        color=INK2, fontsize=17)
ax.text(-8, ymax * 0.35, f"Loss\n{np.mean(roi < 0):.0%}", ha="right", color=ORANGE,
        fontsize=17, fontweight="bold")
ax.set_ylim(0, ymax * 1.16)
ax.set_xlabel("12-month net value (\\$ thousands)")
ax.set_ylabel("% of 10,000 scenarios")
for s in ["top", "right"]:
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.8)
ax.set_axisbelow(True)
pct_loss = int(round(100 * np.mean(roi < 0)))
article = "an" if pct_loss in (8, 11, 18) or 80 <= pct_loss < 90 else "a"
fig.suptitle(f"Vendor math says {k(naive_net)}. The full model says {k(q(net, .5))},\n"
             f"with {article} {np.mean(roi < 0):.0%} chance of losing money",
             x=0.06, ha="left", fontsize=22, fontweight="bold", color=INK, y=0.975)
fig.text(0.06, 0.835, "AI help-desk agent, 12-month net value across 10,000 simulated scenarios. "
         "Synthetic data.", fontsize=14, color=INK2)
fig.subplots_adjust(left=0.1, right=0.97, top=0.80, bottom=0.12)
fig.savefig(OUT / "fig_1_net_value_distribution.png", facecolor=SURF)
plt.close(fig)

# Figure 2: tornado of the top 8 drivers
top = tor.head(8).iloc[::-1]
fig, ax = plt.subplots(figsize=(12, 7.5), dpi=150, facecolor=SURF)
ax.set_facecolor(SURF)
y = np.arange(len(top))
base_net = pt["net"] / 1000
for yi, (_, r) in zip(y, top.iterrows()):
    lo, hi = sorted([r.low_case / 1000, r.high_case / 1000])
    ax.barh(yi, hi - base_net, left=base_net, height=0.62, color=BLUE)
    ax.barh(yi, lo - base_net, left=base_net, height=0.62, color=ORANGE)
    ax.text(hi + 3, yi, k(hi * 1000), va="center", fontsize=14, color=INK2)
    ax.text(lo - 3, yi, k(lo * 1000), va="center", ha="right", fontsize=14, color=INK2)
ax.axvline(base_net, color=INK, lw=1.5)
ax.axvline(0, color=INK2, lw=1, ls=":")
ax.set_yticks(y, top.input, fontsize=16, color=INK)
ax.set_xlabel("12-month net value (\\$ thousands)")
xmin = min(top[["low_case", "high_case"]].min().min() / 1000, 0) - 40
xmax = top[["low_case", "high_case"]].max().max() / 1000 + 40
ax.set_xlim(xmin, xmax)
for s in ["top", "right", "left"]:
    ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.grid(axis="x", color=GRID, lw=0.8)
ax.set_axisbelow(True)
d1, d2 = tor.input[0].split(" (")[0], tor.input[1].split(" (")[0]
n_known = int(n_tot.sum())
fig.suptitle(f"{d1} and {d2.lower()} drive the risk.\n"
             f"Escalation rates, measured on {n_known:,} tickets, barely move it",
             x=0.03, ha="left", fontsize=22, fontweight="bold", color=INK, y=0.975)
fig.text(0.03, 0.835, f"Each input moved alone to its 10th / 90th percentile. Orange = worse case, "
         f"blue = better case.\nCentre line = point estimate {k(pt['net'])}. Synthetic data.",
         fontsize=14, color=INK2, va="top")
fig.subplots_adjust(left=0.34, right=0.97, top=0.74, bottom=0.12)
fig.savefig(OUT / "fig_2_tornado_sensitivity.png", facecolor=SURF)
plt.close(fig)

# =============================================================================
# Save results
# =============================================================================
res = pd.DataFrame(RESULTS)
res.to_csv(OUT / "results_long.csv", index=False)       # read by r/analysis.R
pd.DataFrame({"net": net, "roi": roi}).describe().to_csv(OUT / "mc_summary_python.csv")
json_out = {
    "edition": "A02 The Full ROI Equation for AI Agents",
    "note": "Synthetic data. 'value' is rounded to 'digits' (2 -> 0.01, -3 -> nearest 1,000). "
            "Python wrote this file; r/analysis.R adds the 'r' key and agreement check.",
    "results": [{"name": r["name"], "value": r["value"], "digits": r["digits"],
                 "unit": r["unit"], "meaning": r["meaning"], "headline": r["headline"],
                 "python": r["value"]} for r in RESULTS],
}
(OUT / "results.json").write_text(json.dumps(json_out, indent=2))
print(f"\nWrote {len(RESULTS)} results, 2 figures and tornado table to {OUT}")
