"""
generate_data.py -- synthetic data for F01 "The Broken Bottom Rung"

ALL DATA PRODUCED HERE ARE SYNTHETIC. They do not come from Meta, any client,
or any real organization.

Run:  python data/generate_data.py      (from the package root)
  or  python generate_data.py           (from the data/ folder)

Writes (next to this script):
  junior_careers.csv   person-month panel (what an analyst would really have)
  ground_truth.csv     each person's TRUE trajectory type (never visible in real
                       life; used only to check whether the method recovers it)

Seed: 20261105 (fixed, so every run produces the identical files).

--------------------------------------------------------------------------------
DATA-GENERATING PROCESS (the "truth" the analysis tries to recover)
--------------------------------------------------------------------------------
Company: a mid-size services firm hiring entry-level staff into two job families
  HIGH = operations / customer-support analysts. A GenAI assistant is deployed
         from July 2024 (calendar) and absorbs most routine tasks.
  LOW  = field-service coordinators. Little AI change.
Each family has 6 teams (team_id); teams add a small output random effect.

Cohorts (half-year hiring classes), hire month uniform inside the half-year:
  pre-AI : 2023H1, 2023H2, 2024H1      post-AI: 2024H2, 2025H1, 2025H2
  Head counts per cohort: HIGH 95 pre / 75 post, LOW 70 pre / 60 post (N = 900).
  Post-AI cohorts are smaller and a bit more selective:
  selection_score z ~ N(0, 1) pre, N(0.3, 0.9) post (both families).

Data cut: October 2026 is the last observed calendar month. A person is observed
for tenure months 1 .. min(24, months from hire through Oct 2026). Hires after
May 2025 are therefore NOT observed through tenure month 18 (right truncation).

Monthly work states:
  R = routine junior tasks, S = supervised complex work ("stretch" work),
  I = independent complex work, X = left the company (absorbing).

TRUE trajectory type (drawn once per person; shares by family x period):
                    climber  plateau  early_leaver
  HIGH, pre-AI       0.65     0.20      0.15
  HIGH, post-AI      0.45     0.40      0.15      <- the true shift (+20 pts plateau)
  LOW,  pre-AI       0.63     0.22      0.15
  LOW,  post-AI      0.61     0.24      0.15      <- small shift (+2 pts plateau)
Type is independent of selection_score by construction.

State paths by type:
  climber      : R for r_len months, then S for s_len months, then I.
                 r_len ~ U{3,4,5}; in HIGH post-AI cohorts r_len ~ U{2,3}
                 (the assistant absorbs routine work, so routine phase is shorter).
                 s_len ~ U{6,...,11}.
  plateau      : starts in R; from tenure month 4, each month R->S with prob p_s
                 and S->R with prob 0.5; never reaches I in 24 months.
                 p_s = 0.15, except HIGH post-AI cohorts p_s = 0.08
                 (fewer stretch assignments once AI does the routine work).
  early_leaver : leaves (X) at tenure month U{4,...,12}; before that behaves
                 like a plateau person.
  Climbers and plateau people also face a small late-attrition hazard of 0.6% per
  month from tenure month 13 on.

Output metric tasks_closed (active months only):
  mean = base[state] * ramp(t) * exp(person_re + team_re) * ai(calendar, state)
         * exp(0.04 * selection_z);  observed count ~ Poisson(mean)
  base: R 42, S 28, I 34 tasks/month;  ramp(t) = 0.6 + 0.4 * min(t, 6) / 6
  ai multiplier, HIGH family from calendar 2024-07 on: R 1.45, S 1.15, I 1.15;
  LOW family from 2024-07 on: 1.03 for all states.  (So 2024H1 HIGH hires get a
  partial boost in their first 6 months, as would happen in real life.)
  person_re ~ N(0, 0.15), team_re ~ N(0, 0.05)

Month-18 skill assessment (only if active and observed at tenure month 18):
  55 + 18*[state at 18 == I] + 1.2*(months in S, months 1-18) + 3*selection_z
  + N(0, 6), rounded, clipped to 0..100; 5% randomly not recorded.

Messy realism (the analysis must clean these):
  * ~2% of active person-months have a blank state (and blank tasks_closed).
  * ~2% of people have inconsistently coded states on some rows:
    "r", " S", "SUP", "Indep", "routine", "LEFT" etc.
  * 6 people have a coding error where an active state appears after X.
--------------------------------------------------------------------------------
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261105
HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(SEED)

COHORTS = ["2023H1", "2023H2", "2024H1", "2024H2", "2025H1", "2025H2"]
POST = {"2024H2", "2025H1", "2025H2"}
SIZES = {("HIGH", "pre"): 95, ("HIGH", "post"): 75, ("LOW", "pre"): 70, ("LOW", "post"): 60}
TYPE_SHARES = {  # climber, plateau, early_leaver
    ("HIGH", "pre"): [0.65, 0.20, 0.15],
    ("HIGH", "post"): [0.45, 0.40, 0.15],
    ("LOW", "pre"): [0.63, 0.22, 0.15],
    ("LOW", "post"): [0.61, 0.24, 0.15],
}
TYPES = ["climber", "plateau", "early_leaver"]
CUT_YEAR, CUT_MONTH = 2026, 10
MAX_T = 24
BASE = {"R": 42.0, "S": 28.0, "I": 34.0}


def months_between(y1, m1, y2, m2):
    """Number of calendar months from (y1, m1) to (y2, m2), inclusive."""
    return (y2 - y1) * 12 + (m2 - m1) + 1


def simulate_states(ptype, family, period):
    """Return the 24-month TRUE state path for one person."""
    states = []
    if ptype == "climber":
        r_len = rng.integers(2, 4) if (family == "HIGH" and period == "post") else rng.integers(3, 6)
        s_len = rng.integers(6, 12)
        for t in range(1, MAX_T + 1):
            states.append("R" if t <= r_len else ("S" if t <= r_len + s_len else "I"))
    else:
        p_s = 0.08 if (family == "HIGH" and period == "post") else 0.15
        leave_at = rng.integers(4, 13) if ptype == "early_leaver" else 10**6
        cur = "R"
        for t in range(1, MAX_T + 1):
            if t >= leave_at:
                cur = "X"
            elif t >= 4:
                if cur == "R" and rng.random() < p_s:
                    cur = "S"
                elif cur == "S" and rng.random() < 0.5:
                    cur = "R"
            states.append(cur)
    # late attrition for stayers
    if ptype != "early_leaver":
        for t in range(13, MAX_T + 1):
            if rng.random() < 0.006:
                states[t - 1:] = ["X"] * (MAX_T - t + 1)
                break
    return states


def main():
    team_re = {(f, k): rng.normal(0, 0.05) for f in ["HIGH", "LOW"] for k in range(1, 7)}
    rows, truth = [], []
    pid = 0
    for cohort in COHORTS:
        period = "post" if cohort in POST else "pre"
        year = int(cohort[:4])
        first_month = 1 if cohort.endswith("H1") else 7
        for family in ["HIGH", "LOW"]:
            n = SIZES[(family, period)]
            for _ in range(n):
                pid += 1
                person_id = f"P{pid:04d}"
                hire_m = first_month + int(rng.integers(0, 6))
                team = int(rng.integers(1, 7))
                sel = rng.normal(0.3, 0.9) if period == "post" else rng.normal(0, 1)
                ptype = TYPES[rng.choice(3, p=TYPE_SHARES[(family, period)])]
                states = simulate_states(ptype, family, period)
                n_obs = min(MAX_T, months_between(year, hire_m, CUT_YEAR, CUT_MONTH))
                p_re = rng.normal(0, 0.15)

                # month-18 skill score (if active and observed at month 18)
                score = np.nan
                if n_obs >= 18 and states[17] != "X" and rng.random() > 0.05:
                    s_months = sum(s == "S" for s in states[:18])
                    score = 55 + 18 * (states[17] == "I") + 1.2 * s_months + 3 * sel + rng.normal(0, 6)
                    score = float(np.clip(round(score), 0, 100))

                truth.append({"person_id": person_id, "true_type": ptype})
                for t in range(1, n_obs + 1):
                    cal_index = (year * 12 + hire_m - 1) + (t - 1)
                    cy, cm = divmod(cal_index, 12)
                    cm += 1
                    st = states[t - 1]
                    tasks = np.nan
                    if st != "X":
                        ai = 1.0
                        if (cy, cm) >= (2024, 7):
                            ai = {"R": 1.45, "S": 1.15, "I": 1.15}[st] if family == "HIGH" else 1.03
                        ramp = 0.6 + 0.4 * min(t, 6) / 6
                        mu = BASE[st] * ramp * np.exp(p_re + team_re[(family, team)]) * ai * np.exp(0.04 * sel)
                        tasks = float(rng.poisson(mu))
                    rows.append({
                        "person_id": person_id, "cohort": cohort, "hire_month": f"{year}-{hire_m:02d}",
                        "job_family": family, "team_id": f"{family[0]}{team:02d}",
                        "selection_score": round(float(sel), 2),
                        "calendar_month": f"{cy}-{cm:02d}", "tenure_month": t,
                        "state": st, "tasks_closed": tasks, "skill_score_m18": score,
                    })

    df = pd.DataFrame(rows)

    # ---- messy realism --------------------------------------------------------
    active = df.index[df["state"] != "X"].to_numpy()
    blank = rng.choice(active, size=int(0.02 * len(active)), replace=False)
    df.loc[blank, "state"] = ""
    df.loc[blank, "tasks_closed"] = np.nan

    variants = {"R": ["r", "routine", " R"], "S": ["s", "SUP", " S"], "I": ["i", "Indep", "I "], "X": ["x", "LEFT"]}
    people = df["person_id"].unique()
    messy_people = rng.choice(people, size=int(0.02 * len(people)), replace=False)
    for p in messy_people:
        idx = df.index[(df["person_id"] == p) & (df["state"] != "")].to_numpy()
        for i in rng.choice(idx, size=min(4, len(idx)), replace=False):
            v = variants[df.at[i, "state"]]
            df.at[i, "state"] = v[int(rng.integers(0, len(v)))]

    # coding errors: an active state recorded after the person left
    leavers = df.loc[df["state"] == "X", "person_id"].unique()
    for p in rng.choice(leavers, size=6, replace=False):
        idx = df.index[(df["person_id"] == p) & (df["state"] == "X")].to_numpy()
        if len(idx) >= 2:
            df.at[idx[-1], "state"] = "R"  # spurious "back to routine" after leaving

    df.to_csv(HERE / "junior_careers.csv", index=False)
    pd.DataFrame(truth).to_csv(HERE / "ground_truth.csv", index=False)
    print(f"wrote {len(df):,} person-months for {df.person_id.nunique()} people -> {HERE / 'junior_careers.csv'}")


if __name__ == "__main__":
    main()
