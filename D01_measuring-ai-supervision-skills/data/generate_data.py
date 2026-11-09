"""
Seeded generator for the D01 synthetic data: an "AI Supervision Skills" assessment.

Run:  python data/generate_data.py
Writes (next to this file):
  ai_supervision_assessment.csv  - what an analyst would receive (one row per test taker)
  true_parameters.csv            - every generating value plus population ("true") reliability and validity
  true_person_values.csv         - each person's true latent skill and true score (used ONLY in Step 3)

SEED = 20261104 (the production date of the edition). Change it to see how much the
estimates move from sample to sample; the true values in true_parameters.csv do not change.

------------------------------------------------------------------------------------------
DATA-GENERATING PROCESS (the ground truth the analysis is checked against)
------------------------------------------------------------------------------------------
People. 612 records in 4 business units. 600 genuine test takers + 12 "speeders" (finished in
under 6 minutes and gave the same answer to every item). The analysis must find and drop them.

Two latent traits per genuine person:
  theta = AI supervision skill. theta = unit_mean + unit_sd * z, z ~ N(0, 1).
          Unit means / SDs are in UNITS below. Customer Support has a narrower spread
          (sd 0.70): the same test is less reliable there (reliability depends on the group).
  A     = AI confidence (how comfortable people feel checking AI output).
          A = RHO * z + sqrt(1 - RHO^2) * u, u ~ N(0, 1).  RHO = 0.35: confidence is only
          modestly related to real skill.

Items. Every item is a continuous response y* = loading * trait + sqrt(1 - loading^2) * e,
e ~ N(0, 1), cut into ordered categories at fixed thresholds (an "ordinal congeneric" model).
  S01-S12 scenario items (spot the error, decide to escalate, verify a claim), scored 0-4,
          driven by theta with UNEQUAL loadings (LOAD_S). Two items are flawed by design:
            S06 ambiguous scenario: loading 0.05 (answers barely depend on skill)
            S11 mis-keyed answer:   loading -0.22 (as scored, skilled people lose points)
  C01-C04 self-report confidence items, 0-10 rating (0 = not at all, 10 = completely confident),
          near-duplicate wording,
          driven by A with very high loadings (LOAD_C, 0.90-0.94).

Criterion (independent work sample, about 3 weeks later). Each person audits about 40 AI
outputs containing seeded errors, delivered as two parallel sets (a and b, 19-21 errors each).
  P(catch an error) = logistic(0.30 + 0.85 * theta + 0.90 * d), d ~ N(0, 1) (day-to-day noise).
  Caught errors ~ Binomial(seeded, P). Confidence A has NO direct effect on the criterion.

Messy realism. 3% of item responses missing at random; 4% of people missing the audit
(left before it); 9 business-unit labels typed inconsistently; 2 very slow completions (kept).

Population truth. Reliability and validity are properties of scores in a population, so the
"true" values are computed on 1,500,000 simulated people from the same model (no missing data,
no speeders): true reliability of a sum score = Var(true score) / Var(observed score), where the
true score is the expected item score given the latent traits (closed form under the model).
"""
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

SEED = 20261104
HERE = Path(__file__).resolve().parent

UNITS = {  # name: (n genuine people, theta mean, theta sd)
    "Trust & Safety Review": (180, 0.20, 1.00),
    "Operations": (150, 0.00, 1.00),
    "Claims": (140, -0.10, 1.05),
    "Customer Support": (130, -0.15, 0.70),
}
N_SPEEDERS = 12
RHO = 0.35  # true correlation between z (skill, standardized) and confidence A

S_ITEMS = [f"S{i:02d}" for i in range(1, 13)]
C_ITEMS = [f"C{i:02d}" for i in range(1, 5)]
LOAD_S = np.array([0.67, 0.60, 0.53, 0.63, 0.48, 0.05, 0.58, 0.65, 0.50, 0.61, -0.22, 0.56])
SHIFT_S = np.array([-0.4, 0.0, 0.2, -0.2, 0.3, 0.0, 0.1, 0.4, -0.3, 0.0, 0.1, 0.2])  # item difficulty
CUTS_S = np.array([-1.4, -0.5, 0.35, 1.2])  # 4 cuts -> scores 0..4
LOAD_C = np.array([0.92, 0.94, 0.90, 0.93])
CUTS_C = np.linspace(-1.8, 1.6, 10)  # 10 cuts -> 0..10 (slightly skewed: people rate themselves high)
FLAWED = ["S06", "S11"]
CLEAN_S = [s for s in S_ITEMS if s not in FLAWED]
CRIT_B0, CRIT_B1, CRIT_NOISE = 0.30, 0.85, 0.90
P_MISSING_ITEM, P_MISSING_AUDIT = 0.03, 0.04


def draw_people(rng, units):
    """Latent traits for genuine test takers."""
    rows = []
    for unit, (n, mu, sd) in units.items():
        z = rng.standard_normal(n)
        rows.append(pd.DataFrame({"unit": unit, "z": z, "theta": mu + sd * z}))
    df = pd.concat(rows, ignore_index=True)
    df["A"] = RHO * df["z"] + np.sqrt(1 - RHO**2) * rng.standard_normal(len(df))
    return df


def ordinal(rng, trait, loading, cuts):
    """Ordinal congeneric item: count how many thresholds the continuous response passes."""
    ystar = loading * trait + np.sqrt(1 - loading**2) * rng.standard_normal(len(trait))
    return np.searchsorted(cuts, ystar)  # 0..len(cuts)


def expected_item(trait, loading, cuts):
    """E[item score | trait] = sum over thresholds of P(y* > cut). Used for true scores."""
    sd = np.sqrt(1 - loading**2)
    return sum(norm.cdf((loading * trait - c) / sd) for c in cuts)


def simulate_items(rng, people):
    out = {}
    for j, s in enumerate(S_ITEMS):
        out[s] = ordinal(rng, people["theta"].to_numpy(), LOAD_S[j], CUTS_S + SHIFT_S[j])
    for j, c in enumerate(C_ITEMS):
        out[c] = ordinal(rng, people["A"].to_numpy(), LOAD_C[j], CUTS_C)
    return pd.DataFrame(out)


def true_scores(people):
    th, a = people["theta"].to_numpy(), people["A"].to_numpy()
    t = {s: expected_item(th, LOAD_S[j], CUTS_S + SHIFT_S[j]) for j, s in enumerate(S_ITEMS)}
    t.update({c: expected_item(a, LOAD_C[j], CUTS_C) for j, c in enumerate(C_ITEMS)})
    return pd.DataFrame(t)


def criterion_probability(rng, theta):
    return 1 / (1 + np.exp(-(CRIT_B0 + CRIT_B1 * theta + CRIT_NOISE * rng.standard_normal(len(theta)))))


def population_truth():
    """True reliability and validity of each score, from 1,500,000 simulated people."""
    rng = np.random.default_rng(1)
    big = {u: (v[0] * 2500, v[1], v[2]) for u, v in UNITS.items()}  # 600 x 2500 = 1.5M, same unit mix
    pop = draw_people(rng, big)
    X, T = simulate_items(rng, pop), true_scores(pop)
    p = criterion_probability(rng, pop["theta"].to_numpy())
    crit = rng.binomial(40, p) / 40  # a 40-error audit
    scales = {"naive16": S_ITEMS + C_ITEMS, "scenario12": S_ITEMS, "scenario10": CLEAN_S, "conf4": C_ITEMS}
    rows = []
    for name, items in scales.items():
        x, t = X[items].sum(axis=1), T[items].sum(axis=1)
        rows.append((f"true_reliability_{name}", np.var(t) / np.var(x),
                     f"Population reliability of the {name} sum score: Var(true score)/Var(observed)"))
        rows.append((f"true_validity_{name}", np.corrcoef(x, crit)[0, 1],
                     f"Population correlation of the {name} sum score with the 40-error audit score"))
    rows += [
        ("true_corr_theta_criterion", np.corrcoef(pop["theta"], crit)[0, 1], "Correlation of true skill with the audit score"),
        ("true_corr_confidence_criterion", np.corrcoef(pop["A"], crit)[0, 1], "Correlation of true confidence with the audit score"),
        ("true_corr_truescore10_truecriterion", np.corrcoef(T[CLEAN_S].sum(axis=1), p)[0, 1],
         "Correlation of the scenario10 true score with the true catch rate (target of the attenuation correction)"),
        ("true_reliability_audit_score", np.var(p) / np.var(crit), "Population reliability of the 40-error audit score"),
    ]
    # population SD of the scenario10 score and the cut score used for certification
    rows.append(("pop_sd_scenario10", np.std(X[CLEAN_S].sum(axis=1)), "Population SD of the scenario10 sum score"))
    return rows


def main():
    rng = np.random.default_rng(SEED)

    # ---- genuine test takers -------------------------------------------------------------
    people = draw_people(rng, UNITS)
    items = simulate_items(rng, people)
    tscore = true_scores(people)
    n = len(people)

    # ---- missing item responses (3%, at random) ------------------------------------------
    item_cols = S_ITEMS + C_ITEMS
    mask = rng.random((n, len(item_cols))) < P_MISSING_ITEM
    items = items.astype(float).mask(mask)

    df = pd.DataFrame({"business_unit": people["unit"]})
    roles = {"Trust & Safety Review": ["Content reviewer", "Policy specialist", "Team lead"],
             "Operations": ["Operations analyst", "Workflow lead", "Team lead"],
             "Claims": ["Claims examiner", "Claims lead", "Team lead"],
             "Customer Support": ["Support agent", "Support lead", "Team lead"]}
    df["role"] = [rng.choice(roles[u], p=[0.6, 0.25, 0.15]) for u in people["unit"]]
    df["completion_minutes"] = np.round(np.clip(rng.lognormal(np.log(28), 0.3, n), 9, None), 1)
    df = pd.concat([df, items], axis=1)

    # ---- criterion: two parallel audit sets ----------------------------------------------
    p = criterion_probability(rng, people["theta"].to_numpy())
    for half in ["a", "b"]:
        seeded = rng.integers(19, 22, n)
        df[f"audit_seeded_{half}"] = seeded
        df[f"audit_caught_{half}"] = rng.binomial(seeded, p)
    miss_audit = rng.random(n) < P_MISSING_AUDIT
    for col in ["audit_seeded_a", "audit_caught_a", "audit_seeded_b", "audit_caught_b"]:
        df[col] = df[col].astype("Int64")
        df.loc[miss_audit, col] = pd.NA

    truth = pd.DataFrame({"theta": people["theta"], "confidence_factor": people["A"],
                          "true_catch_rate": p,
                          "true_score_scenario10": tscore[CLEAN_S].sum(axis=1),
                          "true_score_conf4": tscore[C_ITEMS].sum(axis=1)})

    # ---- speeders: very fast, straight-lined -----------------------------------------------
    sp = pd.DataFrame({"business_unit": rng.choice(list(UNITS), N_SPEEDERS),
                       "role": "Content reviewer"})
    sp["role"] = [roles[u][0] for u in sp["business_unit"]]
    sp["completion_minutes"] = np.round(rng.uniform(2.5, 5.9, N_SPEEDERS), 1)
    for k in range(N_SPEEDERS):
        s_val, c_val = rng.integers(0, 5), rng.integers(6, 11)
        for s in S_ITEMS:
            sp.loc[k, s] = s_val
        for c in C_ITEMS:
            sp.loc[k, c] = c_val
    sp_p = rng.uniform(0.3, 0.8, N_SPEEDERS)
    for half in ["a", "b"]:
        seeded = rng.integers(19, 22, N_SPEEDERS)
        sp[f"audit_seeded_{half}"] = pd.array(seeded, dtype="Int64")
        sp[f"audit_caught_{half}"] = pd.array(rng.binomial(seeded, sp_p), dtype="Int64")
    sp_truth = pd.DataFrame({c: np.nan for c in truth.columns}, index=range(N_SPEEDERS))

    df = pd.concat([df, sp], ignore_index=True)
    truth = pd.concat([truth, sp_truth], ignore_index=True)
    truth.insert(0, "is_speeder", [0] * n + [1] * N_SPEEDERS)

    # ---- shuffle rows, assign ids ----------------------------------------------------------
    order = rng.permutation(len(df))
    df, truth = df.iloc[order].reset_index(drop=True), truth.iloc[order].reset_index(drop=True)
    ids = [f"P{i:04d}" for i in range(1, len(df) + 1)]
    df.insert(0, "person_id", ids)
    truth.insert(0, "person_id", ids)

    # ---- messy labels and two very slow (but genuine) sessions -------------------------------
    messy = {"Operations": "operations ", "Customer Support": "Customer support", "Claims": "CLAIMS"}
    genuine_idx = truth.index[truth["is_speeder"] == 0].to_numpy()
    for i in rng.choice(genuine_idx, 9, replace=False):
        u = df.loc[i, "business_unit"]
        df.loc[i, "business_unit"] = messy.get(u, "trust & safety review")
    for i in rng.choice(genuine_idx, 2, replace=False):
        df.loc[i, "completion_minutes"] = float(rng.integers(95, 140))

    for c in item_cols:
        df[c] = df[c].astype("Int64")
    df.to_csv(HERE / "ai_supervision_assessment.csv", index=False)
    truth.round(6).to_csv(HERE / "true_person_values.csv", index=False)

    # ---- true parameters -------------------------------------------------------------------
    rows = [("seed", SEED, "Random seed for the sample"), ("rho_skill_confidence", RHO,
            "True correlation between standardized skill z and confidence A")]
    rows += [(f"loading_{s}", LOAD_S[j], "Generating loading of scenario item on skill"
              + (" (FLAWED: ambiguous)" if s == "S06" else " (FLAWED: mis-keyed)" if s == "S11" else ""))
             for j, s in enumerate(S_ITEMS)]
    rows += [(f"loading_{c}", LOAD_C[j], "Generating loading of confidence item on confidence A")
             for j, c in enumerate(C_ITEMS)]
    rows += [("criterion_b0", CRIT_B0, "Audit model intercept (logit)"),
             ("criterion_b1", CRIT_B1, "Audit model slope on theta (logit)"),
             ("criterion_noise_sd", CRIT_NOISE, "SD of person-level noise in the audit (logit)"),
             ("flawed_items", "S06;S11", "Items that are flawed by design")]
    rows += population_truth()
    tp = pd.DataFrame(rows, columns=["parameter", "value", "meaning"])
    tp["value"] = [round(v, 4) if isinstance(v, float) else v for v in tp["value"]]
    tp.to_csv(HERE / "true_parameters.csv", index=False)
    print(f"Wrote {len(df)} rows ({N_SPEEDERS} speeders) to ai_supervision_assessment.csv")
    print(tp.tail(13).to_string(index=False))


if __name__ == "__main__":
    main()
