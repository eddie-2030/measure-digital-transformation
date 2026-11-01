"""
generate_data.py -- synthetic randomized pilot of a GenAI assistant (B06)

Run from the package root:   python3 data/generate_data.py
Writes:                      data/ai_pilot_heterogeneity.csv

ALL DATA ARE SYNTHETIC. They do not come from Meta, any client, or any real
organization. Because we generate the data ourselves, we know the true effect
of the AI assistant for every employee, so every method can be checked against
the truth.

-------------------------------------------------------------------------------
DATA-GENERATING PROCESS (the "ground truth")
-------------------------------------------------------------------------------
Organization
  * 120 teams in 4 regions (Americas, EMEA, APAC, LATAM); team size 10-30
    (unequal), about 2,400 employees in total.
  * Each team works in one primary language. About a quarter of employees sit
    in non-English teams (Spanish, Portuguese, German, French, Japanese). Non-English
    teams exist in every region, somewhat more often outside the Americas.
  * Noise covariates with NO true effect modification: role_family (4 levels),
    shift (3 levels), remote (0/1).

Randomization
  * Within every team, half the members (rounded at random for odd sizes) get
    access to the AI assistant (ai_access = 1). Assignment is random, so any
    difference in outcomes between arms is caused by the assistant.

Outcome: weekly quality-adjusted tasks resolved
  * baseline_tasks  = pre-pilot weekly average (8 weeks before the pilot)
  * outcome_tasks   = pilot-period weekly average (8 weeks of the pilot)
  * Both share a person "skill" term, a team random effect (sd 3.5), a region
    level shift and a lower level for newer staff, plus measurement noise.

TRUE INDIVIDUAL EFFECT of AI access (tasks per week):
      true_effect = 3.4                      # English team, experienced (12+ mo)
                  + 2.2 * novice             # tenure < 12 months gain more
                  - 3.6 * non_english        # assistant is weaker in other languages
                  + N(0, 0.6)                # small idiosyncratic variation
  So, in expectation:
      English, experienced      +3.4      English, novice      +5.6
      Non-English, experienced  -0.2      Non-English, novice  +2.0
  True interaction coefficients: AI x NonEnglish = -3.6 ; AI x Novice = +2.2.
  role_family, shift, remote and region have ZERO true effect modification.
  With ~26% non-English and ~25% novices, the true average effect is about
  +3.0 tasks/week, roughly +7% of the control-group mean (~42 tasks/week).

Messy realism (on purpose)
  * tenure_months missing for ~5% of employees (missing completely at random).
  * 10 outcome values recorded as a 4-week TOTAL instead of a weekly average
    (about 4x too large) -- a classic data-entry outlier.
  * Unequal team sizes and unequal subgroup sizes.

The column `true_effect` exists only because the data are synthetic. In a real
pilot you never observe it; here we use it to grade the methods.

SEED: 20261153 (fixed; change it and every number in outputs/ will change).
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261153

# True parameters (documented above; reused by the analysis scripts for checks)
TAU_BASE = 3.4          # English, experienced
TAU_NOVICE = 2.2        # extra gain for tenure < 12 months
TAU_NONENG = -3.6       # shift for non-English teams
TAU_SD = 0.6            # idiosyncratic sd of the individual effect
NOVICE_CUTOFF = 12      # months


def generate(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    # ---- Teams ---------------------------------------------------------------
    n_teams = 120
    regions = np.array(["Americas", "EMEA", "APAC", "LATAM"])
    team_region = rng.choice(regions, size=n_teams, p=[0.32, 0.28, 0.22, 0.18])
    # Probability a team works in a non-English language, by region.
    p_noneng = {"Americas": 0.18, "EMEA": 0.30, "APAC": 0.32, "LATAM": 0.38}
    lang_by_region = {
        "Americas": ["Spanish", "French"],
        "EMEA": ["German", "French"],
        "APAC": ["Japanese"],
        "LATAM": ["Spanish", "Portuguese"],
    }
    team_lang = []
    for r in team_region:
        if rng.random() < p_noneng[r]:
            team_lang.append(rng.choice(lang_by_region[r]))
        else:
            team_lang.append("English")
    team_size = rng.integers(10, 31, size=n_teams)       # unequal sizes
    team_re = rng.normal(0, 3.5, size=n_teams)            # team random effect
    region_shift = {"Americas": 0.0, "EMEA": 1.0, "APAC": -1.0, "LATAM": -0.5}

    rows = []
    emp = 0
    for t in range(n_teams):
        n = team_size[t]
        # Randomize within team: half get access (random rounding if n is odd).
        n_treat = n // 2 + (rng.random() < 0.5 if n % 2 else 0)
        w = np.zeros(n, dtype=int)
        w[rng.choice(n, size=n_treat, replace=False)] = 1
        for i in range(n):
            emp += 1
            rows.append(dict(
                employee_id=f"E{emp:05d}",
                team_id=f"T{t + 1:03d}",
                region=team_region[t],
                work_language=team_lang[t],
                ai_access=w[i],
                _team_re=team_re[t],
            ))
    df = pd.DataFrame(rows)
    N = len(df)

    # ---- Individuals ---------------------------------------------------------
    # Tenure (months): right-skewed, ~25% under 12 months.
    tenure = np.exp(rng.normal(np.log(20), 0.85, size=N))
    tenure = np.clip(tenure, 0.5, 240).round(1)
    df["role_family"] = rng.choice(
        ["Tier-1 support", "Tier-2 support", "Operations analyst", "Specialist"],
        size=N, p=[0.40, 0.25, 0.20, 0.15])
    df["shift"] = rng.choice(["Day", "Evening", "Night"], size=N, p=[0.55, 0.30, 0.15])
    df["remote"] = (rng.random(N) < 0.45).astype(int)

    novice = (tenure < NOVICE_CUTOFF).astype(int)
    non_english = (df["work_language"] != "English").astype(int).to_numpy()

    # ---- Outcomes --------------------------------------------------------------
    skill = rng.normal(0, 5.0, size=N)
    reg = df["region"].map(region_shift).to_numpy()
    newness = -5.0 * np.exp(-tenure / 10.0)   # newer staff resolve fewer tasks
    learning = 1.5 * novice                    # ...but improve during the pilot

    baseline = 42 + reg + df["_team_re"].to_numpy() + skill + newness + rng.normal(0, 3.0, N)
    true_effect = (TAU_BASE + TAU_NOVICE * novice + TAU_NONENG * non_english
                   + rng.normal(0, TAU_SD, N))
    outcome = (43 + reg + df["_team_re"].to_numpy() + skill + newness + learning
               + rng.normal(0, 3.5, N) + df["ai_access"].to_numpy() * true_effect)

    df["tenure_months"] = tenure
    df["baseline_tasks"] = baseline.round(1)
    df["outcome_tasks"] = outcome.round(1)
    df["true_effect"] = true_effect.round(3)

    # ---- Messiness -------------------------------------------------------------
    miss = rng.random(N) < 0.05
    df.loc[miss, "tenure_months"] = np.nan                   # missing tenure
    out_idx = rng.choice(N, size=10, replace=False)          # 4-week totals
    df.loc[out_idx, "outcome_tasks"] = (df.loc[out_idx, "outcome_tasks"] * 4).round(1)

    cols = ["employee_id", "team_id", "region", "work_language", "role_family",
            "shift", "remote", "tenure_months", "ai_access", "baseline_tasks",
            "outcome_tasks", "true_effect"]
    return df[cols]


if __name__ == "__main__":
    out = Path(__file__).resolve().parent / "ai_pilot_heterogeneity.csv"
    d = generate(SEED)
    d.to_csv(out, index=False)
    print(f"Wrote {out} with {len(d):,} employees in {d.team_id.nunique()} teams "
          f"(seed {SEED}).")
