"""
generate_data.py -- synthetic data for C01 "Who Audits the Agents?"

Run from the code folder root:   python3 data/generate_data.py
Writes:
  data/oversight_reviews.csv    one row per human review of an AI decision (long format)
  data/true_parameters.csv      the ground truth used to simulate the reviews

SCENARIO
An AI agent decides customer refund requests (approve or deny). Every AI decision
is routed to one of three review sites (A, B, C). At that site, two reviewers are
drawn at random from the site's four reviewers, and each labels the AI decision
"Correct" or "Error". A small share of cases is escalated to a third reviewer.

GOLDEN SET. An expert panel adjudicated 300 past AI decisions (100 per site) and
the review tool seeded them, blind, into each site's queue. Reviewers cannot tell
them apart from live cases. Because AI errors are rare, a purely random sample of
300 would hold only ~24 errors (~8 per site), too few to estimate a catch rate.
So the golden set is enriched: 25% of seeded cases are true errors (25 per site).
Enrichment does not bias sensitivity or specificity, which are computed
separately within true errors and true non-errors. Golden rows carry gold_label
and case_source = "golden_seeded"; live rows have case_source = "production".
Agreement statistics (percent agreement, kappa, alpha) use production rows only,
so they reflect the real 8% error rate.

DATA-GENERATING PROCESS (the ground truth the analysis is checked against)
1. Each production AI decision is truly an error with probability PREVALENCE = 0.08,
   independent of site, amount, and decision type (low prevalence by design).
   Golden-set cases are exactly 25% errors (25 of 100 per site).
2. Each reviewer r has a true sensitivity (P[says Error | truly Error]) and a true
   specificity (P[says Correct | truly Correct]). They are drawn around the site
   means below with modest reviewer-level variation on the logit scale (SD 0.30):
        site   mean sensitivity   mean specificity
        A      0.90               0.96
        B      0.85               0.95
        C      0.15               0.94
   Sites A and B are well calibrated. Site C misses most true errors and labels
   almost everything "Correct". Its "Error" flags are close to random noise.
3. Each review is an independent Bernoulli draw given the truth and the
   reviewer's sensitivity/specificity (conditional independence of reviewers).
4. Messy cases, as in real review logs:
   - about 2.5% of reviews are missing (blank, "Skipped", or "Unsure")
   - about 1.5% of labels have inconsistent spelling or spacing ("correct ", "ERR")
   - 3% of decisions were escalated to a third reviewer from the same site
   - 8 reviews were logged twice (same decision, reviewer, and label; new review_id)
Because reviewers only see cases from their own site, 48 of the 66 possible
reviewer pairs share no cases at all, and each within-site pair shares only
about 80-110 live decisions. A full reviewer-by-reviewer kappa matrix is impossible, which is
why Krippendorff's alpha (which handles many raters and missing ratings) is used.

The seed is fixed (SEED = 20261101), so the csv is reproducible.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261101
N_DECISIONS = 1500
PREVALENCE = 0.08                  # true AI error rate
SITE_PARAMS = {                    # site mean sensitivity, specificity
    "A": (0.90, 0.96),
    "B": (0.85, 0.95),
    "C": (0.15, 0.94),
}
REVIEWERS_PER_SITE = 4
REVIEWER_SD_LOGIT = 0.30           # reviewer-level variation around the site mean
GOLDEN_PER_SITE = 100              # expert-adjudicated cases seeded per site
GOLDEN_ERRORS_PER_SITE = 25        # true errors among them (enriched to 25%)
P_ESCALATE = 0.03                  # share of decisions given a third reviewer
P_MISSING = 0.025                  # share of reviews with no usable label
P_MESSY_LABEL = 0.015              # share of labels with inconsistent formatting
N_DUPLICATES = 8                   # exact duplicate rows in the log

HERE = Path(__file__).resolve().parent


def logit(p):
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


def main():
    rng = np.random.default_rng(SEED)

    # --- Reviewers and their true accuracy -------------------------------------
    reviewers = []
    k = 1
    for site, (sens, spec) in SITE_PARAMS.items():
        for _ in range(REVIEWERS_PER_SITE):
            reviewers.append({
                "reviewer_id": f"R{k:02d}",
                "site": site,
                "sensitivity": expit(logit(sens) + rng.normal(0, REVIEWER_SD_LOGIT)),
                "specificity": expit(logit(spec) + rng.normal(0, REVIEWER_SD_LOGIT)),
            })
            k += 1
    rev = pd.DataFrame(reviewers)

    # --- AI decisions ------------------------------------------------------------
    def make_decisions(n, sites, start):
        return pd.DataFrame({
            "decision_id": [f"D{i:04d}" for i in range(start, start + n)],
            "site": sites,
            "decision_date": pd.Timestamp("2026-09-01")
            + pd.to_timedelta(rng.integers(0, 61, n), unit="D"),
            "ai_decision": np.where(rng.random(n) < 0.70, "approve", "deny"),
            "refund_amount_usd": np.round(rng.lognormal(4.0, 0.9, n), 2),
        })

    # Production decisions: 500 per site, true error with probability PREVALENCE.
    sites = np.repeat(list(SITE_PARAMS), N_DECISIONS // len(SITE_PARAMS))
    rng.shuffle(sites)
    prod = make_decisions(N_DECISIONS, sites, 1)
    prod["true_error"] = (rng.random(N_DECISIONS) < PREVALENCE).astype(int)
    prod["case_source"] = "production"

    # Golden set: 100 expert-adjudicated decisions per site, 25 of them true errors,
    # seeded blind into the queue.
    g_sites = np.repeat(list(SITE_PARAMS), GOLDEN_PER_SITE)
    gold = make_decisions(len(g_sites), g_sites, N_DECISIONS + 1)
    gold["true_error"] = np.concatenate([
        rng.permutation([1] * GOLDEN_ERRORS_PER_SITE + [0] * (GOLDEN_PER_SITE - GOLDEN_ERRORS_PER_SITE))
        for _ in SITE_PARAMS])
    gold["case_source"] = "golden_seeded"
    dec = pd.concat([prod, gold], ignore_index=True)
    # shuffle IDs so golden cases are not identifiable by number
    dec["decision_id"] = rng.permutation(dec["decision_id"].to_numpy())
    dec = dec.sort_values("decision_id").reset_index(drop=True)
    dec["golden_set"] = (dec["case_source"] == "golden_seeded").astype(int)

    # --- Reviews -----------------------------------------------------------------
    rows = []
    for d in dec.itertuples(index=False):
        pool = rev[rev["site"] == d.site]
        n_rev = 3 if rng.random() < P_ESCALATE else 2
        chosen = pool.iloc[rng.choice(len(pool), n_rev, replace=False)]
        for order, r in enumerate(chosen.itertuples(index=False), start=1):
            p_error = r.sensitivity if d.true_error else 1 - r.specificity
            label = "Error" if rng.random() < p_error else "Correct"
            rows.append({
                "review_id": None,
                "decision_id": d.decision_id,
                "site": d.site,
                "decision_date": d.decision_date.date().isoformat(),
                "ai_decision": d.ai_decision,
                "refund_amount_usd": d.refund_amount_usd,
                "reviewer_id": r.reviewer_id,
                "review_order": order,
                "review_label": label,
                # review time in seconds: right-skewed, independent of the label
                "review_seconds": int(np.clip(rng.lognormal(np.log(75), 0.5), 8, 900)),
                "case_source": d.case_source,
                "golden_set": d.golden_set,
                "gold_label": ("Error" if d.true_error else "Correct") if d.golden_set else "",
            })
    out = pd.DataFrame(rows)

    # --- Messy cases ---------------------------------------------------------------
    n = len(out)
    miss = rng.random(n) < P_MISSING
    out.loc[miss, "review_label"] = rng.choice(["", "Skipped", "Unsure"], miss.sum())
    messy = (~miss) & (rng.random(n) < P_MESSY_LABEL)
    variants = {"Correct": ["correct", " Correct", "CORRECT", "correct "],
                "Error": ["error", "ERR", "Error ", "err"]}
    out.loc[messy, "review_label"] = [rng.choice(variants[v]) for v in out.loc[messy, "review_label"]]
    dups = out.iloc[rng.choice(n, N_DUPLICATES, replace=False)]
    out = pd.concat([out, dups]).sort_values(["decision_id", "review_order"], kind="stable")
    out["review_id"] = [f"V{i:05d}" for i in range(1, len(out) + 1)]
    # duplicates are logged twice with the same content but a new review_id,
    # as happens when a review tool double-submits
    out.to_csv(HERE / "oversight_reviews.csv", index=False)

    # --- True parameters (ground truth) --------------------------------------------
    tp = [{"level": "population", "id": "all", "site": "all", "parameter": "prevalence",
           "value": PREVALENCE}]
    tp.append({"level": "realized", "id": "production", "site": "all",
               "parameter": "error_rate_production", "value": prod["true_error"].mean()})
    tp.append({"level": "design", "id": "golden", "site": "all",
               "parameter": "error_rate_golden_set", "value": GOLDEN_ERRORS_PER_SITE / GOLDEN_PER_SITE})
    for site, (sens, spec) in SITE_PARAMS.items():
        s = rev[rev.site == site]
        tp += [
            {"level": "site_design", "id": site, "site": site, "parameter": "sensitivity", "value": sens},
            {"level": "site_design", "id": site, "site": site, "parameter": "specificity", "value": spec},
            # the average of this site's 4 reviewers (what a golden set should recover)
            {"level": "site_reviewer_mean", "id": site, "site": site, "parameter": "sensitivity",
             "value": s.sensitivity.mean()},
            {"level": "site_reviewer_mean", "id": site, "site": site, "parameter": "specificity",
             "value": s.specificity.mean()},
        ]
    for r in rev.itertuples(index=False):
        tp += [
            {"level": "reviewer", "id": r.reviewer_id, "site": r.site, "parameter": "sensitivity", "value": r.sensitivity},
            {"level": "reviewer", "id": r.reviewer_id, "site": r.site, "parameter": "specificity", "value": r.specificity},
        ]
    tp = pd.DataFrame(tp)
    tp["value"] = tp["value"].round(4)
    tp.to_csv(HERE / "true_parameters.csv", index=False)

    print(f"Wrote {len(out)} review rows: {N_DECISIONS} production decisions "
          f"({prod.true_error.sum()} true AI errors) + {len(gold)} seeded golden cases "
          f"({gold.true_error.sum()} true errors).")


if __name__ == "__main__":
    main()
