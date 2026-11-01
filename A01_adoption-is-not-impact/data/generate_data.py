"""
generate_data.py -- synthetic data for A01 "Adoption Is Not Impact:
How to Prove a GenAI Rollout Actually Worked"

Run from the code folder root:   python3 data/generate_data.py
Writes:
  data/support_team_weeks.csv   one row per customer-support team per week (panel)
  data/true_parameters.csv      the ground truth used to simulate the panel

SCENARIO
A customer-support organization rolls out an AI assistant to 60 support teams
(6 regions x 10 teams) in three waves:
    wave 1 launches in week 7, wave 2 in week 13, wave 3 in week 19.
Teams are observed weekly for 24 weeks. Within each region, the order in which
teams launch was set by trainer availability (an operational constraint), drawn at
random here. Waves were NOT chosen by past performance, so the waves differ only
by chance in their average level, and they share the same underlying trend.
There is no never-treated group: by week 19 every team has the assistant.

DATA-GENERATING PROCESS (the ground truth the analysis is checked against)
Outcome (rung 3, work outcome): tickets resolved per agent-hour, for team i in week t:

    rate_it = BASE_RATE + region_r + team_i            (stable level differences)
              + TREND * (t - 1)                         (common trend, everyone)
              + week_shock_t                            (common week-to-week noise)
              + tau_i * RAMP(t - launch_week_i)         (true AI effect, after launch)
              + e_it                                    (team-week noise)

  * BASE_RATE = 2.40 tickets per agent-hour.
  * region_r ~ Normal(0, 0.10); team_i = -0.60 * (novice_share_i - 0.30) + Normal(0, 0.22).
  * TREND = +0.05 per week: seasonality plus ordinary learning that lifts every team,
    with or without the assistant. Over the 6 weeks between the before and after
    windows of wave 1 this alone adds about +0.30.
  * week_shock_t ~ Normal(0, 0.03), shared by all teams; e_it ~ Normal(0, 0.08).
  * Then the team's tickets are drawn as Poisson(rate_it * agent_hours_it), which adds
    realistic count noise. The csv stores tickets and hours; the outcome is their ratio.

True effect of the assistant (dynamic, ramps up after launch):
  * Established (steady-state) effect tau_i = TAU_SS * (1 + HET * (novice_share_i - 0.30)),
    with TAU_SS = 0.38 and HET = 0.60: teams with more novice agents gain slightly more
    (a +/- 12% spread, a small hook for edition B06). Novice share is independent of wave.
  * RAMP(e) for e = weeks since launch: e=0 -> 0.20, e=1 -> 0.50, e=2 -> 0.80, e>=3 -> 1.00.
    The effect builds over the first three weeks as agents learn the tool.
  * So the average true effect over a team's first 6 weeks is about 0.38 * 0.75 = 0.29,
    which is what the 6-week before/after windows of wave 1 can measure: the "+0.30" of
    the article. Over all treated team-weeks the average true effect is about 0.33.
  * Because the effect grows with time since launch, a naive two-way fixed effects
    regression on the staggered rollout uses already-treated teams (whose effect is
    still growing) as controls and is pulled below the true average effect.

Quality (rung 3, paired with speed): reopen rate = share of resolved tickets reopened
within 7 days. True rate ~ 8% (team SD 1.2 pp, team-week swings SD 0.6 pp) with a
tiny common improvement (-0.02 pp per week). TRUE AI EFFECT ON REOPEN RATE = 0: speed rose, quality held.

Usage (rung 1): weekly active share of agents who used the assistant. Zero before
launch; after launch it ramps 0.45, 0.65, 0.75, then ~0.80 (team-specific level, noisy).
Usage is NOT impact: it is reported to show adoption, not to estimate the effect.

Messy cases, as in real operational extracts:
  * about 1.5% of team-weeks are missing entirely (holidays, data outages)
  * 3 team-weeks have a blank agent_hours (time-tracking export failure); drop them
  * 4 team-weeks appear twice (exact duplicate rows); keep one
  * about 2% of region labels have inconsistent case or trailing spaces ("emea-north ")
  * 3 teams show 1 active AI user in the week before launch (early-access testers)

The seed is fixed (SEED = 20261105), so the csv is reproducible.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261105
N_WEEKS = 24
REGIONS = ["Americas-East", "Americas-West", "EMEA-North", "EMEA-South", "APAC-North", "APAC-South"]
TEAMS_PER_REGION = 10
WAVE_LAUNCH = {1: 7, 2: 13, 3: 19}
# Teams per wave within each region (rows) -> 20 teams per wave in total.
WAVE_COUNTS = [(3, 3, 4), (3, 4, 3), (4, 3, 3), (3, 3, 4), (3, 4, 3), (4, 3, 3)]

BASE_RATE = 2.40          # tickets per agent-hour, week 1, average team
REGION_SD = 0.10
TEAM_SD = 0.22
NOVICE_SLOPE = -0.60      # teams with more novices resolve fewer tickets per hour
TREND = 0.05              # common change per week (everyone)
WEEK_SHOCK_SD = 0.03
NOISE_SD = 0.08
TAU_SS = 0.38             # established true effect (tickets per agent-hour)
HET = 0.60                # effect heterogeneity by novice share (small)
NOVICE_REF = 0.30         # reference novice share
RAMP = [0.20, 0.50, 0.80]  # share of established effect in launch week, +1, +2; 1.0 after

REOPEN_BASE = 0.080
REOPEN_TEAM_SD = 0.012
REOPEN_TREND = -0.0002
REOPEN_WEEK_SD = 0.006   # week-to-week swings in a team's reopen rate (beyond binomial noise)
REOPEN_EFFECT = 0.0       # true AI effect on reopen rate
USAGE_RAMP = [0.45, 0.65, 0.75, 0.80]

P_MISSING_WEEK = 0.015
N_BLANK_HOURS = 3
N_DUPLICATES = 4
P_MESSY_REGION = 0.02

HERE = Path(__file__).resolve().parent


def ramp_share(e):
    """Share of the established effect reached e weeks after launch (e < 0: not launched)."""
    if e < 0:
        return 0.0
    return RAMP[e] if e < len(RAMP) else 1.0


def simulate(seed=SEED):
    """Simulate one rollout. Returns (panel as written to csv, true_parameters table).
    python/analysis.py also calls this with other seeds for a small Monte Carlo check."""
    rng = np.random.default_rng(seed)
    rng_q = np.random.default_rng(seed + 1)     # separate stream for the quality metric (reopens)

    # ---- Teams: region, wave, size, novice share --------------------------------
    teams = []
    k = 0
    for r, region in enumerate(REGIONS):
        waves = np.repeat([1, 2, 3], WAVE_COUNTS[r])
        rng.shuffle(waves)                       # trainer schedule, not performance
        for w in waves:
            k += 1
            teams.append({"team_id": f"T{k:02d}", "region": region, "wave": int(w),
                          "launch_week": WAVE_LAUNCH[int(w)]})
    teams = pd.DataFrame(teams)
    n = len(teams)
    teams["n_agents"] = rng.integers(7, 17, size=n)
    teams["novice_share"] = np.round(rng.uniform(0.10, 0.50, size=n), 2)
    region_eff = dict(zip(REGIONS, rng.normal(0, REGION_SD, len(REGIONS))))
    teams["team_level"] = (NOVICE_SLOPE * (teams.novice_share - NOVICE_REF)
                           + rng.normal(0, TEAM_SD, n))
    teams["tau_ss"] = TAU_SS * (1 + HET * (teams.novice_share - NOVICE_REF))
    teams["reopen_level"] = REOPEN_BASE + rng.normal(0, REOPEN_TEAM_SD, n)
    teams["usage_level"] = np.clip(rng.normal(1.0, 0.08, n), 0.8, 1.15)

    week_shock = rng.normal(0, WEEK_SHOCK_SD, N_WEEKS + 1)
    week_start = pd.date_range("2026-03-02", periods=N_WEEKS, freq="7D")

    # ---- Team-week panel ---------------------------------------------------------
    rows = []
    for t_ in teams.itertuples(index=False):
        for t in range(1, N_WEEKS + 1):
            e = t - t_.launch_week
            effect = t_.tau_ss * ramp_share(e)
            rate = (BASE_RATE + region_eff[t_.region] + t_.team_level + TREND * (t - 1)
                    + week_shock[t] + effect + rng.normal(0, NOISE_SD))
            hours = round(float(t_.n_agents * rng.normal(34, 2.5)), 1)
            tickets = int(rng.poisson(max(rate, 0.2) * hours))
            p_reopen = np.clip(t_.reopen_level + REOPEN_TREND * (t - 1) + REOPEN_EFFECT * (e >= 0)
                               + rng_q.normal(0, REOPEN_WEEK_SD), 0.01, 0.3)
            reopened = int(rng_q.binomial(tickets, p_reopen))
            if e >= 0:
                u = USAGE_RAMP[min(e, 3)] * t_.usage_level + rng.normal(0, 0.05)
                active = int(np.clip(round(u * t_.n_agents), 0, t_.n_agents))
            else:
                active = 0
            rows.append({
                "team_id": t_.team_id, "region": t_.region, "wave": t_.wave,
                "launch_week": t_.launch_week, "week": t,
                "week_start": week_start[t - 1].date().isoformat(),
                "n_agents": int(t_.n_agents), "novice_share": t_.novice_share,
                "agent_hours": hours, "tickets_resolved": tickets,
                "reopened_tickets": reopened, "ai_active_agents": active,
                "true_effect": effect,           # kept aside for true_parameters, not written
            })
    df = pd.DataFrame(rows)

    # ---- Ground truth (computed on the full design, before messy rows) ----------
    true_rows = [
        ("global", "all", "trend_per_week", TREND),
        ("global", "all", "tau_established_design", TAU_SS),
        ("global", "all", "tau_established_mean_teams", float(teams.tau_ss.mean())),
        ("global", "all", "ramp_share_e0", RAMP[0]),
        ("global", "all", "ramp_share_e1", RAMP[1]),
        ("global", "all", "ramp_share_e2", RAMP[2]),
        ("global", "all", "ramp_share_e3plus", 1.0),
        ("global", "all", "het_novice_slope", HET),
        ("global", "all", "novice_reference", NOVICE_REF),
        ("global", "all", "reopen_rate_true_effect", REOPEN_EFFECT),
        ("global", "all", "base_rate", BASE_RATE),
        ("global", "all", "seed", seed),
    ]
    tr = df[df.week >= df.launch_week]
    w1 = tr[(tr.wave == 1) & (tr.week <= 12)]
    cs = tr[(tr.week <= 18) & (tr.wave < 3)]
    true_rows += [
        ("design_truth", "wave1_weeks7to12", "att", float(w1.true_effect.mean())),
        ("design_truth", "waves1and2_weeks_upto18", "att", float(cs.true_effect.mean())),
        ("design_truth", "all_treated_teamweeks", "att", float(tr.true_effect.mean())),
    ]
    for t_ in teams.itertuples(index=False):
        true_rows.append(("team", t_.team_id, "tau_established", float(t_.tau_ss)))
    for r, v in region_eff.items():
        true_rows.append(("region", r, "region_effect", float(v)))
    tp = pd.DataFrame(true_rows, columns=["level", "id", "parameter", "value"])
    tp["value"] = pd.Series([int(v) if float(v).is_integer() and abs(v) > 1 else round(float(v), 6)
                             for v in tp["value"]], dtype=object)

    # ---- Messy cases -------------------------------------------------------------
    df = df.drop(columns="true_effect")
    drop = rng.random(len(df)) < P_MISSING_WEEK
    df = df[~drop].reset_index(drop=True)
    blank = rng.choice(len(df), N_BLANK_HOURS, replace=False)
    df["agent_hours"] = df["agent_hours"].astype(object)
    df.loc[blank, "agent_hours"] = np.nan
    messy = rng.random(len(df)) < P_MESSY_REGION
    df.loc[messy, "region"] = [s.lower() + " " if i % 2 else " " + s.upper()
                               for i, s in enumerate(df.loc[messy, "region"])]
    # Early-access testers: 1 active user in the week before launch for 3 teams.
    for tid in rng.choice(teams.team_id, 3, replace=False):
        lw = int(teams.loc[teams.team_id == tid, "launch_week"].iloc[0])
        df.loc[(df.team_id == tid) & (df.week == lw - 1), "ai_active_agents"] = 1
    dups = df.iloc[rng.choice(len(df), N_DUPLICATES, replace=False)]
    df = pd.concat([df, dups]).sort_values(["team_id", "week"], kind="stable").reset_index(drop=True)

    # Derived columns, as an operations extract would provide them.
    hrs = pd.to_numeric(df.agent_hours)
    df["tickets_per_agent_hour"] = (df.tickets_resolved / hrs).round(4)
    df["reopen_rate"] = (df.reopened_tickets / df.tickets_resolved).round(4)
    df["ai_weekly_active_share"] = (df.ai_active_agents / df.n_agents).round(4)
    df["post_launch"] = (df.week >= df.launch_week).astype(int)
    return df, tp


def main():
    df, tp = simulate(SEED)
    tp.to_csv(HERE / "true_parameters.csv", index=False)
    df.to_csv(HERE / "support_team_weeks.csv", index=False)
    print(f"Wrote {len(df)} rows to data/support_team_weeks.csv and data/true_parameters.csv")
    print(tp[tp.level.isin(["global", "design_truth"])].to_string(index=False))


if __name__ == "__main__":
    main()
