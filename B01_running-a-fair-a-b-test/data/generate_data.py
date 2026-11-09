"""
generate_data.py  --  B01 "Running a Fair A/B Test Inside Your Company"

Writes four SYNTHETIC files into data/:
  randomized_pilot.csv     the pilot run the fair way (stratified randomization)
  volunteer_pilot.csv      the same organization, pilot run as an opt-in (volunteer) program
  ground_truth.csv         the hidden truth for each employee (only a simulation can give you this)
  rerandomizations.csv     500 alternative stratified randomizations the pilot COULD have drawn
                           (used by the analysis scripts to check bias, SE and CI coverage)

Run from the package root:   python3 data/generate_data.py
Random seed: 20261108 (fixed, so every run gives identical files).

----------------------------------------------------------------------------------
The story the data tell
----------------------------------------------------------------------------------
1,200 customer-support analysts work in 40 teams across 4 sites. The company pilots an
AI drafting assistant for 8 weeks, after a 4-week baseline ("pre-period").

Primary outcome: cases resolved per employee per week.
Guardrail:       QA quality score (0-100, from audited cases). The tool must not make it worse.

----------------------------------------------------------------------------------
Data-generating process (all TRUE parameters are set here)
----------------------------------------------------------------------------------
For employee i in team t at site s, week w:

  base_i      = 40 + site_s + team_t + ability_i
                site_s ~ N(0, 2^2), team_t ~ N(0, 2.5^2), ability_i ~ N(0, 6^2)
  motivation_i ~ N(0, 1), correlated 0.4 with ability (motivated people are a bit better)

  pre-period  cases_iw = base_i + week_w + e_iw                  e_iw ~ N(0, 9^2)
  post-period cases_iw = base_i + drift + change_i + week_w + TRUE_EFFECT * used_i + e_iw
                drift    = +1.0 (everybody gets a little better over time)
                change_i = 1.5 * motivation_i + N(0, 3^2)   (motivated people improve anyway)

  TRUE_EFFECT = +2.5 cases/week for an employee who actually USES the tool (about +6%)
  Uptake: exactly 80% of employees are "would-use-if-given-access" (compliers);
          the more motivated are more likely to be compliers.
  So the TRUE effect of GIVING ACCESS (intention-to-treat, ITT) = 2.5 x 0.80 = +2.0 cases/week.

  Pre/post correlation of the employee means is about 0.7 by construction.

  Quality: stable_q_i = 85 + 0.25*ability_i + N(0, 3^2)
           quality_pre  = stable_q_i + N(0, 2^2)
           quality_post = stable_q_i + 0.3*motivation_i + N(0, 2^2) + TRUE_QUALITY_EFFECT*used_i
           TRUE_QUALITY_EFFECT = 0 (the tool neither helps nor hurts quality)

  AI hours per week (logged by the tool, users only) = 3 + 1.2*motivation + N(0, 1), >= 0.5.
  This is a POST-treatment variable: it exists only because of the assignment.

Volunteer pilot: same employees, fresh weekly noise. P(opt in) = logistic(-1.3 + 1.3*motivation
  + 0.06*ability). Volunteers are more able AND more motivated -- the selection bias.

Randomized pilot: stratified (blocked) randomization within site (4 strata),
  half of each site to the tool. Users = assigned AND complier.

----------------------------------------------------------------------------------
Messy cases (deliberate; the analysis scripts handle them explicitly)
----------------------------------------------------------------------------------
  * Leavers: about 3% of employees leave mid-pilot; their weeks from `left_week` on are blank.
  * Duplicate row: one employee appears twice in randomized_pilot.csv (an export glitch).
  * Outage: one team's data feed failed for one post week; every member shows 0 cases that week.
"""

import os
import numpy as np
import pandas as pd

SEED = 20261108
rng = np.random.default_rng(SEED)
HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- true parameters ----------------
N_EMP, N_SITES, TEAMS_PER_SITE = 1200, 4, 10
N_PRE, N_POST = 4, 8
BASE_MEAN, SD_SITE, SD_TEAM, SD_ABILITY = 40.0, 2.0, 2.5, 6.0
SD_WEEK_NOISE = 9.0
DRIFT, MOTIVATION_GROWTH, SD_CHANGE = 1.0, 1.5, 3.0
TRUE_EFFECT_IF_USED = 2.5
UPTAKE_SHARE = 0.80
TRUE_QUALITY_EFFECT = 0.0
LEAVE_RATE = 0.03
N_RERAND = 500

SITES = ["A", "B", "C", "D"]

# ---------------- organization: sites, teams, employees ----------------
team_ids = [f"{s}{k:02d}" for s in SITES for k in range(1, TEAMS_PER_SITE + 1)]
team_site = np.repeat(np.arange(N_SITES), TEAMS_PER_SITE)
team_share = rng.dirichlet(np.full(len(team_ids), 30.0))
team_of_emp = np.sort(rng.choice(len(team_ids), size=N_EMP, p=team_share))
site_of_emp = team_site[team_of_emp]

site_eff = rng.normal(0, SD_SITE, N_SITES)
team_eff = rng.normal(0, SD_TEAM, len(team_ids))
z1, z2 = rng.normal(size=N_EMP), rng.normal(size=N_EMP)
ability = SD_ABILITY * z1
motivation = 0.4 * z1 + np.sqrt(1 - 0.4 ** 2) * z2
base = BASE_MEAN + site_eff[site_of_emp] + team_eff[team_of_emp] + ability

# compliers: exactly 80% would use the tool if given access; motivated people more likely
comply_score = 1.5 * motivation + rng.logistic(size=N_EMP)
complier = (comply_score >= np.quantile(comply_score, 1 - UPTAKE_SHARE)).astype(int)

stable_q = 85 + 0.25 * ability + rng.normal(0, 3, N_EMP)
emp_ids = np.array([f"E{i:04d}" for i in range(1, N_EMP + 1)])


def stratified_assign(strata, rng):
    """Half of each stratum to treatment (an odd stratum's extra person goes by coin flip)."""
    z = np.zeros(len(strata), dtype=int)
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        idx = rng.permutation(idx)
        n_t = len(idx) // 2 + (rng.random() < 0.5 if len(idx) % 2 else 0)
        z[idx[:n_t]] = 1
    return z


def simulate_pilot(used_rule, outage_team, outage_week):
    """Simulate weekly cases & quality for one pilot design. Returns a DataFrame + `used`."""
    week_pre = rng.normal(0, 1, N_PRE)
    week_post = rng.normal(0, 1, N_POST)
    change = MOTIVATION_GROWTH * motivation + rng.normal(0, SD_CHANGE, N_EMP)
    pre = base[:, None] + week_pre[None, :] + rng.normal(0, SD_WEEK_NOISE, (N_EMP, N_PRE))
    pre_mean = pre.mean(axis=1)
    assigned, used = used_rule(pre_mean)
    post = (base[:, None] + DRIFT + change[:, None] + week_post[None, :]
            + TRUE_EFFECT_IF_USED * used[:, None]
            + rng.normal(0, SD_WEEK_NOISE, (N_EMP, N_POST)))
    pre = np.clip(np.round(pre), 0, None)
    post = np.clip(np.round(post), 0, None)

    q_pre = np.clip(stable_q + rng.normal(0, 2, N_EMP), 0, 100).round(1)
    q_post = np.clip(stable_q + 0.3 * motivation + rng.normal(0, 2, N_EMP)
                     + TRUE_QUALITY_EFFECT * used, 0, 100).round(1)
    hours = np.where(used == 1,
                     np.maximum(0.5, 3 + 1.2 * motivation + rng.normal(0, 1, N_EMP)), 0.0).round(1)

    # messy case 1: leavers (weeks from left_week onwards are missing)
    leaves = rng.random(N_EMP) < LEAVE_RATE
    left_week = np.where(leaves, rng.integers(1, N_POST + 1, N_EMP), 0)
    post = post.astype(float)
    for i in np.flatnonzero(leaves):
        post[i, left_week[i] - 1:] = np.nan
        if left_week[i] <= 4:           # left before any quality audit of the pilot period
            q_post[i] = np.nan
    # messy case 3: data-feed outage -> the whole team shows 0 that week
    in_team = team_of_emp == team_ids.index(outage_team)
    post[in_team & ~np.isnan(post[:, outage_week - 1]), outage_week - 1] = 0.0

    df = pd.DataFrame({"employee_id": emp_ids,
                       "site": np.array(SITES)[site_of_emp],
                       "team": np.array(team_ids)[team_of_emp]})
    return df, pre, post, pre_mean, assigned, used, hours, q_pre, q_post, left_week


def to_frame(df, pre, post, hours, q_pre, q_post, left_week):
    for w in range(N_PRE):
        df[f"cases_pre_w{w + 1}"] = pre[:, w].astype(int)
    for w in range(N_POST):
        df[f"cases_post_w{w + 1}"] = pd.array(post[:, w], dtype="Int64")
    df["ai_hours_per_week"] = hours
    df["quality_pre"] = q_pre
    df["quality_post"] = q_post
    df["left_week"] = pd.array(np.where(left_week > 0, left_week, np.nan), dtype="Int64")
    return df


# ================= Design 1: the volunteer (opt-in) pilot =================
vol_p = 1 / (1 + np.exp(-(-1.3 + 1.3 * motivation + 0.06 * ability)))
volunteered = (rng.random(N_EMP) < vol_p).astype(int)
df, pre, post, pre_mean, a, used_v, hours, q_pre, q_post, lw = simulate_pilot(
    lambda pm: (volunteered, volunteered * complier), outage_team="B07", outage_week=3)
df["volunteered"] = volunteered
df["used_tool"] = used_v
vol = to_frame(df, pre, post, hours, q_pre, q_post, lw)

# ================= Design 2: the stratified randomized pilot =================
store = {}


def randomized_rule(pm):
    # Stratified (blocked) randomization: half of each SITE gets the tool, so every site
    # is represented equally in both arms. (The template also lets you add a baseline
    # tercile as a second stratifier; we keep one here so the lesson stays simple.)
    strata = np.array(SITES)[site_of_emp]
    z = stratified_assign(strata, rng)
    store.update(strata=strata)
    return z, z * complier


df, pre, post, pre_mean, assigned, used_r, hours, q_pre, q_post, lw = simulate_pilot(
    randomized_rule, outage_team="C04", outage_week=5)
df["stratum"] = store["strata"]
df["assigned"] = assigned
df["used_tool"] = used_r
rnd = to_frame(df, pre, post, hours, q_pre, q_post, lw)

# messy case 2: a duplicated export row
dup_idx = 456
rnd = pd.concat([rnd.iloc[:dup_idx + 1], rnd.iloc[[dup_idx]], rnd.iloc[dup_idx + 1:]],
                ignore_index=True)

# ================= 500 alternative randomizations (same procedure) =================
rr = {"employee_id": emp_ids}
for r in range(1, N_RERAND + 1):
    rr[f"r{r:03d}"] = stratified_assign(store["strata"], rng)
rerand = pd.DataFrame(rr)

# ================= ground truth =================
truth = pd.DataFrame({
    "employee_id": emp_ids,
    "ability": ability.round(3),
    "motivation": motivation.round(3),
    "complier": complier,
    "volunteer_probability": vol_p.round(4),
    "true_effect_if_used": TRUE_EFFECT_IF_USED,
    "true_quality_effect_if_used": TRUE_QUALITY_EFFECT,
})

vol.to_csv(os.path.join(HERE, "volunteer_pilot.csv"), index=False)
rnd.to_csv(os.path.join(HERE, "randomized_pilot.csv"), index=False)
truth.to_csv(os.path.join(HERE, "ground_truth.csv"), index=False)
rerand.to_csv(os.path.join(HERE, "rerandomizations.csv"), index=False)

print(f"volunteer_pilot.csv  : {len(vol)} rows, {volunteered.sum()} volunteers")
print(f"randomized_pilot.csv : {len(rnd)} rows (incl. 1 duplicate), {assigned.sum()} assigned")
print(f"ground_truth.csv     : true effect if used = {TRUE_EFFECT_IF_USED}, "
      f"uptake = {complier.mean():.2f}, true ITT = {TRUE_EFFECT_IF_USED * complier.mean():.2f}")
print(f"rerandomizations.csv : {N_RERAND} alternative stratified assignments")
