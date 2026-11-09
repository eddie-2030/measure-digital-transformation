"""
generate_data.py -- synthetic data for A02 "The Full ROI Equation for AI Agents"

All data written by this script are synthetic. They do not come from Meta, any
client, or any real organization.

Scenario
--------
A company pilots an internal AI help-desk agent that answers employee HR/IT
tickets. Over 26 logged weeks, a growing share of tickets is routed to the
agent (the adoption ramp). For each ticket the log records the queue, whether
the agent handled it, what happened (resolved / escalated to a human /
resolved-then-reopened), what the agent's compute cost, how many minutes a
human spent, and, for a random QA sample of agent-resolved tickets, whether a
reviewer found an error in the agent's answer.

The decision question: over the first 12 months (52 weeks) of the programme,
what is the net value of the agent, and how likely is it to lose money?

Files written (all into this data/ folder)
------------------------------------------
helpdesk_tickets.csv   ticket-level log, weeks 1-26 (with a few messy cases)
cost_assumptions.csv   the cost inputs a finance partner would supply, as
                       low / most-likely / high ranges (NOT in the logs)
ground_truth.csv       the true parameters used to generate the data, plus the
                       true 12-month net value and ROI (for the Step 3 check)
mc_uniforms.csv        10,000 x 25 uniform(0,1) random numbers. Both the Python
                       and the R analysis turn these into Monte Carlo draws, so
                       the two languages produce identical simulation results.

Data-generating process (DGP) and TRUE parameters
-------------------------------------------------
Random seed: SEED = 20261107 (log) and SEED + 1 (Monte Carlo uniforms).

Queues (organisational nesting: tickets within weeks within queues)
    queue        weekly volume  base human   max routing
                 (Poisson mean) minutes      share (ramp)
    access            480          10           0.85
    it_software       340          22           0.72
    hr_policy         200          32           0.55

Adoption ramp: share routed to the agent in week w (true, for w = 1..52) is
    share_q(w) = max_share_q / (1 + exp(-(w - 9) / 3))   (logistic S-curve)

Agent outcomes for routed tickets (multinomial, true):
    queue        resolved  escalated  reopened (rework)
    access         0.84      0.10       0.06
    it_software    0.68      0.22       0.10
    hr_policy      0.56      0.34       0.10

Silent errors: an agent-"resolved" ticket can still carry a wrong answer.
True error rate among resolved tickets: access 0.012, it_software 0.025,
hr_policy 0.050. A QA reviewer samples 12% of resolved tickets and (we
assume) detects every error in the sample.

Human handling minutes (additive model, true):
    human-only ticket : base_q x LogNormal noise (mean 1, sigma 0.45)
    escalated ticket  : same base draw + extra handoff minutes,
                        Gamma(shape 2, mean 6.0)  -> escalations take LONGER
    reopened ticket   : same base draw + extra rework minutes,
                        Gamma(shape 2, mean 8.0)
    agent-resolved    : 0 human minutes
Agent compute cost per routed ticket: LogNormal with mean $0.25 / $0.40 /
$0.55 (access / it_software / hr_policy), sigma 0.6.

True costs (not visible in the log; the analyst only sees ranges):
    loaded labour cost            $54 per hour
    remediation cost per error    $85
    human review share            22% of agent tickets reviewed
    review minutes per review     4.0
    build / integration (one-off) $28,000
    monitoring & upkeep           $1,000 per month
    licence / platform fee        $8,000 per month (contract, known exactly)

Messy cases (deliberate, so the analysis has to clean them)
    * ~1.2% of rows are exact duplicates (a double export from the ticketing tool)
    * ~2% of routed tickets have a missing outcome (logging gap, missing
      completely at random); their minutes are also missing
    * ~1% of human-only tickets have missing human minutes

Run:  python data/generate_data.py      (from the edition folder)
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261107
N_MC = 10_000                       # Monte Carlo draws used by both analyses
HERE = Path(__file__).resolve().parent

# ----------------------------------------------------------------------------
# 1. True parameters (the ground truth the analysis should recover)
# ----------------------------------------------------------------------------
QUEUES = ["access", "it_software", "hr_policy"]
WEEKLY_VOLUME = {"access": 480, "it_software": 340, "hr_policy": 200}
BASE_MINUTES = {"access": 10.0, "it_software": 22.0, "hr_policy": 32.0}
MAX_SHARE = {"access": 0.85, "it_software": 0.72, "hr_policy": 0.55}
RAMP_MID, RAMP_SCALE = 9.0, 3.0

P_ESC = {"access": 0.10, "it_software": 0.22, "hr_policy": 0.34}
P_RW = {"access": 0.06, "it_software": 0.10, "hr_policy": 0.10}
P_ERR = {"access": 0.012, "it_software": 0.025, "hr_policy": 0.050}
RUN_COST = {"access": 0.25, "it_software": 0.40, "hr_policy": 0.55}

EXTRA_ESC_MIN = 6.0                 # extra human minutes on an escalated ticket
EXTRA_RW_MIN = 8.0                  # extra human minutes on a reopened ticket
QA_RATE = 0.12

TRUE_COSTS = {
    "loaded_hourly_cost": 54.0,
    "remediation_cost_per_error": 120.0,
    "review_share": 0.26,
    "review_minutes": 4.5,
    "build_cost": 28_000.0,
    "monitoring_cost_per_month": 1_000.0,
    "license_fee_per_month": 8_000.0,
}

N_WEEKS_LOGGED = 26
HORIZON_WEEKS = 52
START_DATE = pd.Timestamp("2026-03-02")      # a Monday


def routing_share(queue, week):
    """True adoption ramp: logistic S-curve toward the queue's max share."""
    return MAX_SHARE[queue] / (1.0 + np.exp(-(week - RAMP_MID) / RAMP_SCALE))


def true_net_value():
    """True 12-month net value, computed from the true parameters only.

    Uses EXPECTED ticket counts (volume x true ramp, summed over 52 weeks), so
    it is a fixed number, not a random draw.
    """
    w = TRUE_COSTS["loaded_hourly_cost"] / 60.0          # $ per minute
    weeks = np.arange(1, HORIZON_WEEKS + 1)
    benefit = var_cost = 0.0
    n_agent_total = 0.0
    for q in QUEUES:
        n_q = (WEEKLY_VOLUME[q] * routing_share(q, weeks)).sum()
        n_agent_total += n_q
        p_res = 1 - P_ESC[q] - P_RW[q]
        benefit += n_q * p_res * BASE_MINUTES[q] * w
        var_cost += n_q * (
            RUN_COST[q]
            + w * (P_ESC[q] * EXTRA_ESC_MIN + P_RW[q] * EXTRA_RW_MIN)
            + p_res * P_ERR[q] * TRUE_COSTS["remediation_cost_per_error"]
            + TRUE_COSTS["review_share"] * TRUE_COSTS["review_minutes"] * w
        )
    fixed = (12 * TRUE_COSTS["license_fee_per_month"] + TRUE_COSTS["build_cost"]
             + 12 * TRUE_COSTS["monitoring_cost_per_month"])
    net = benefit - var_cost - fixed
    total_vol = sum(WEEKLY_VOLUME.values())
    h2_share = sum(WEEKLY_VOLUME[q] * routing_share(q, weeks[26:]).mean()
                   for q in QUEUES) / total_vol
    return {"net": net, "roi": net / (var_cost + fixed), "benefit": benefit,
            "var_cost": var_cost, "fixed": fixed, "n_agent": n_agent_total,
            "h2_share": h2_share}


# ----------------------------------------------------------------------------
# 2. Simulate the 26-week ticket log
# ----------------------------------------------------------------------------
def simulate_log(rng):
    rows = []
    for week in range(1, N_WEEKS_LOGGED + 1):
        for q in QUEUES:
            n = rng.poisson(WEEKLY_VOLUME[q])
            routed = rng.random(n) < routing_share(q, week)
            # Base human effort for this ticket (used if a human touches it).
            noise = rng.lognormal(mean=-0.45**2 / 2, sigma=0.45, size=n)
            base = BASE_MINUTES[q] * noise
            # Outcome for routed tickets.
            u = rng.random(n)
            outcome = np.where(~routed, "human_only",
                       np.where(u < P_ESC[q], "escalated",
                        np.where(u < P_ESC[q] + P_RW[q], "reopened",
                                 "resolved_by_agent")))
            extra_esc = rng.gamma(2.0, EXTRA_ESC_MIN / 2.0, size=n)
            extra_rw = rng.gamma(2.0, EXTRA_RW_MIN / 2.0, size=n)
            minutes = np.select(
                [outcome == "human_only", outcome == "escalated",
                 outcome == "reopened"],
                [base, base + extra_esc, base + extra_rw], default=0.0)
            cost = rng.lognormal(np.log(RUN_COST[q]) - 0.6**2 / 2, 0.6, size=n)
            agent_cost = np.where(routed, cost, np.nan)
            # QA sample of agent-resolved tickets; error detected if present.
            is_res = outcome == "resolved_by_agent"
            qa = is_res & (rng.random(n) < QA_RATE)
            has_err = rng.random(n) < P_ERR[q]
            qa_err = np.where(qa, has_err.astype(float), np.nan)
            day = rng.integers(0, 5, size=n)          # Mon-Fri
            for i in range(n):
                rows.append({
                    "week": week,
                    "created_date": (START_DATE + pd.Timedelta(days=7 * (week - 1)
                                     + int(day[i]))).date().isoformat(),
                    "queue": q,
                    "routed_to_agent": int(routed[i]),
                    "outcome": outcome[i],
                    "agent_cost_usd": round(agent_cost[i], 4) if routed[i] else np.nan,
                    "human_minutes": round(minutes[i], 1),
                    "qa_sampled": int(qa[i]),
                    "qa_error_found": qa_err[i],
                })
    df = pd.DataFrame(rows)
    df.insert(0, "ticket_id", [f"T{i:06d}" for i in range(1, len(df) + 1)])

    # --- messy case 1: missing outcomes on ~2% of routed tickets -------------
    routed_idx = df.index[df.routed_to_agent == 1]
    miss = rng.choice(routed_idx, size=int(0.02 * len(routed_idx)), replace=False)
    df.loc[miss, ["outcome", "human_minutes", "qa_error_found"]] = np.nan
    df.loc[miss, "qa_sampled"] = 0
    # --- messy case 2: missing minutes on ~1% of human-only tickets ----------
    human_idx = df.index[df.outcome == "human_only"]
    miss_h = rng.choice(human_idx, size=int(0.01 * len(human_idx)), replace=False)
    df.loc[miss_h, "human_minutes"] = np.nan
    # --- messy case 3: ~1.2% exact duplicate rows (double export) ------------
    dup = rng.choice(df.index, size=int(0.012 * len(df)), replace=False)
    df = pd.concat([df, df.loc[dup]]).sort_values(["ticket_id"], kind="stable")
    return df.reset_index(drop=True)


# ----------------------------------------------------------------------------
# 3. Cost assumptions (what finance / ops would give you, as ranges)
# ----------------------------------------------------------------------------
def cost_assumptions():
    rows = [
        # parameter, low, most_likely, high, unit, distribution, source, notes
        ("license_fee_per_month", 8000, 8000, 8000, "USD/month", "fixed",
         "Vendor contract", "Known exactly; contract price"),
        ("build_cost", 20000, 28000, 45000, "USD one-off", "triangular",
         "IT integration estimate", "Integration, SSO, knowledge-base prep; overruns skew high"),
        ("monitoring_cost_per_month", 600, 1000, 1800, "USD/month", "triangular",
         "Platform team", "Prompt/knowledge upkeep, dashboards, incident triage"),
        ("loaded_hourly_cost", 50, 54, 58, "USD/hour", "triangular",
         "Finance (HR/IT service desk)", "Salary + benefits + overhead for help-desk staff"),
        ("remediation_cost_per_error", 40, 80, 250, "USD/error", "triangular",
         "HR/IT ops leads", "Fixing a wrong answer: rework, follow-up, occasional compliance case"),
        ("review_share", 0.10, 0.20, 0.50, "share of agent tickets", "triangular",
         "Oversight policy (draft)", "Share of agent tickets a human reviews; compliance may push it up"),
        ("review_minutes", 3, 4, 7, "minutes/review", "triangular",
         "QA team time study", "Minutes per human review of an agent answer"),
        ("steady_state_routing_share", 0.35, 0.72, 0.85, "share of all tickets", "triangular",
         "Product owner + log trend", "Average share routed to the agent in months 7-12; low = adoption stalls"),
    ]
    return pd.DataFrame(rows, columns=["parameter", "low", "most_likely", "high",
                                       "unit", "distribution", "source", "notes"])


# ----------------------------------------------------------------------------
# 4. Ground truth table
# ----------------------------------------------------------------------------
def ground_truth():
    t = true_net_value()
    rows = []
    for q in QUEUES:
        rows += [(f"p_escalated_{q}", P_ESC[q]), (f"p_reopened_{q}", P_RW[q]),
                 (f"p_resolved_{q}", 1 - P_ESC[q] - P_RW[q]),
                 (f"p_error_{q}", P_ERR[q]), (f"base_minutes_{q}", BASE_MINUTES[q]),
                 (f"run_cost_{q}", RUN_COST[q])]
    rows += [("extra_minutes_escalated", EXTRA_ESC_MIN),
             ("extra_minutes_reopened", EXTRA_RW_MIN)]
    rows += list(TRUE_COSTS.items())
    rows += [("steady_state_routing_share", t["h2_share"]),
             ("agent_tickets_12m", t["n_agent"]),
             ("benefit_12m", t["benefit"]), ("variable_cost_12m", t["var_cost"]),
             ("fixed_cost_12m", t["fixed"]),
             ("net_value_12m", t["net"]), ("roi_12m", t["roi"])]
    return pd.DataFrame(rows, columns=["parameter", "true_value"])


# ----------------------------------------------------------------------------
# 5. Monte Carlo uniforms (shared by Python and R)
# ----------------------------------------------------------------------------
# One column per uncertain input. Each language converts a uniform u into a
# draw with the inverse CDF (beta, normal or triangular), which is exactly the
# same maths in both languages -> identical simulation results.
MC_COLUMNS = (
    [f"u_esc_{q}" for q in QUEUES] + [f"u_rw_{q}" for q in QUEUES]
    + [f"u_err_{q}" for q in QUEUES]
    + [f"u_minutes_{k}" for k in range(1, 6)]      # 5 OLS coefficients
    + ["u_volume"] + [f"u_run_{q}" for q in QUEUES]
    + ["u_build_cost", "u_monitoring_cost_per_month", "u_loaded_hourly_cost",
       "u_remediation_cost_per_error", "u_review_share", "u_review_minutes",
       "u_steady_state_routing_share"]
)


def mc_uniforms(rng):
    u = rng.random((N_MC, len(MC_COLUMNS)))
    return pd.DataFrame(np.round(u, 8), columns=MC_COLUMNS)


if __name__ == "__main__":
    rng = np.random.default_rng(SEED)
    log = simulate_log(rng)
    log.to_csv(HERE / "helpdesk_tickets.csv", index=False)
    cost_assumptions().to_csv(HERE / "cost_assumptions.csv", index=False)
    gt = ground_truth()
    gt.to_csv(HERE / "ground_truth.csv", index=False)
    mc_uniforms(np.random.default_rng(SEED + 1)).to_csv(HERE / "mc_uniforms.csv",
                                                        index=False)
    t = true_net_value()
    print(f"Wrote {len(log):,} log rows ({log.ticket_id.nunique():,} unique tickets)")
    print(f"True 12-month net value: ${t['net']:,.0f}   true ROI: {t['roi']:.2f}")
