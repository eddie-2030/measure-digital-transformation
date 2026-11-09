"""
G01 · Defensible by Design: synthetic hiring-funnel data for an adverse impact audit.

Run from anywhere:
    python data/generate_data.py            # writes data/applications.csv and data/true_parameters.csv
    python data/generate_data.py --scan     # (optional) shows how the seed was chosen

SEED = 20261272 (fixed; the output is identical on every run).

------------------------------------------------------------------------------
THE STORY
------------------------------------------------------------------------------
A company uses an AI resume screen in one hiring cycle (June to September 2026).
About 6,000 people apply to three job families. Each application moves through
three decision stages:

    AI screen pass  ->  interview pass  ->  offer

Applicants self-identify sex (Woman / Man) and age band (Under 40 / 40+), or
decline to answer. The audit compares women with men and 40+ with under 40.

------------------------------------------------------------------------------
DATA-GENERATING PROCESS (ground truth by construction)
------------------------------------------------------------------------------
1. Applicants per job family (unique people):
       Software Engineering (SWE)   1,950
       Customer Operations  (CO)    3,600   (the largest pool)
       Finance              (FIN)     450   (a small pool: about 68 women)
2. Group shares within each family (sex and age are drawn independently):
       share women:   SWE 0.30   CO 0.45   FIN 0.15
       share 40+:     SWE 0.25   CO 0.35   FIN 0.40
3. Stage pass probabilities for the REFERENCE groups (men, under 40).
   Interview and offer probabilities are conditional on reaching that stage.
                       AI screen   interview   offer
       SWE               0.40        0.45      0.50
       CO                0.60        0.50      0.60
       FIN               0.30        0.40      0.55
4. Group multipliers on those probabilities (the TRUE impact ratios per stage):
       Sex (women / men): 1.00 at every stage in every family (true parity).
       Age (40+ / under 40):
                       AI screen   interview   offer
       SWE               0.70        1.15      1.10   <- real AI-stage adverse impact
       CO                0.91        1.00      1.00   <- small but real gap, large pool
       FIN               1.00        1.00      1.00
   The SWE story: the screen leans on a feature correlated with age (years
   since graduation / since last role change). Human interviewers, who see the
   experience, partly offset it later, so the SWE applicant-to-offer ratio is
   0.70 x 1.15 x 1.10 = 0.886 and passes the four-fifths rule.
5. Requisition nesting: each family has several requisitions (job openings).
   Each requisition gets its own multiplicative factor on each stage's pass
   probability, lognormal with mean 1 and sd(log) 0.10. Groups are spread
   evenly across requisitions, so the factors add noise but do not change
   the true ratios.
6. Withdrawals (independent of group): 5% of AI-screen passers withdraw before
   the interview; 4% of interview passers withdraw before the offer.
7. Messy cases:
     - Sex not disclosed: 6% ("Prefer not to say" or blank), at random.
     - Age not disclosed: 7% ("Prefer not to say" or blank), at random.
     - 72 applicants applied twice; the earlier application is also in the
       file (same applicant_id, earlier date). Keep the latest.
     - About 1% of job-family labels are typed inconsistently
       ("software engineering", "Customer Ops", "Finance " ...).

True impact ratios at the AI stage are therefore:
     40+ vs under 40:  SWE 0.70, CO 0.91, FIN 1.00
     women vs men:     1.00 in every family
true_parameters.csv lists every stage, the applicant-to-offer ratios, and the
expected pooled (company-wide) ratios implied by the design shares.

How the seed was chosen: the Finance women four-fifths flag happens by chance
(about 1 time in 6 under true parity, see the Monte Carlo in the analysis), so
`--scan` checks the 1,000 seeds 20261103 to 20262102 and we kept the FIRST one
whose realized data show all four teaching cases (see scan_ok below): 20261272.
9 of the 1,000 seeds pass every check; the Finance women flag is the rare one.
The seed is a teaching choice, not a typical draw for the Finance cell; the
README says so.
"""
import os
import sys
import numpy as np
import pandas as pd

SEED = 20261272          # chosen by --scan (see docstring)
SCAN_START = 20261103    # the edition date; the scan starts here
HERE = os.path.dirname(os.path.abspath(__file__))

FAMILIES = ["Software Engineering", "Customer Operations", "Finance"]
N_APPLICANTS = {"Software Engineering": 1950, "Customer Operations": 3600, "Finance": 450}
N_REQS = {"Software Engineering": 24, "Customer Operations": 18, "Finance": 9}
SHARE_WOMEN = {"Software Engineering": 0.30, "Customer Operations": 0.45, "Finance": 0.15}
SHARE_40PLUS = {"Software Engineering": 0.25, "Customer Operations": 0.35, "Finance": 0.40}
# Reference-group pass probabilities: (AI screen, interview | AI pass, offer | interview pass)
BASE = {"Software Engineering": (0.40, 0.45, 0.50),
        "Customer Operations": (0.60, 0.50, 0.60),
        "Finance": (0.30, 0.40, 0.55)}
# True multipliers for 40+ relative to under 40, per stage
AGE_MULT = {"Software Engineering": (0.70, 1.15, 1.10),
            "Customer Operations": (0.91, 1.00, 1.00),
            "Finance": (1.00, 1.00, 1.00)}
SEX_MULT = (1.00, 1.00, 1.00)            # women vs men: parity everywhere
REQ_SD = 0.10                            # sd of log requisition factor
W_BEFORE_INTERVIEW, W_BEFORE_OFFER = 0.05, 0.04
P_MISS_SEX, P_MISS_AGE = 0.06, 0.07
N_DUPLICATES = 72
LABEL_VARIANTS = {"Software Engineering": ["software engineering", "Software Eng.", "Software Engineering "],
                  "Customer Operations": ["Customer Ops", "customer operations", " Customer Operations"],
                  "Finance": ["Finance ", "finance", "FINANCE"]}
STAGES = ["ai_screen", "interview", "offer"]


def generate(seed=SEED):
    rng = np.random.default_rng(seed)
    rows = []
    pid = 0
    for fam in FAMILIES:
        n = N_APPLICANTS[fam]
        # requisition factors, one per stage, mean 1 on the probability scale
        z = rng.normal(0, REQ_SD, size=(N_REQS[fam], 3))
        req_factor = np.exp(z - REQ_SD ** 2 / 2)
        req = rng.integers(0, N_REQS[fam], size=n)
        woman = rng.random(n) < SHARE_WOMEN[fam]
        older = rng.random(n) < SHARE_40PLUS[fam]
        u = rng.random((n, 3))
        wd = rng.random((n, 2))
        for i in range(n):
            pid += 1
            p = []
            for s in range(3):
                pr = BASE[fam][s] * req_factor[req[i], s]
                pr *= AGE_MULT[fam][s] if older[i] else 1.0
                pr *= SEX_MULT[s] if woman[i] else 1.0
                p.append(min(pr, 0.98))
            ai = int(u[i, 0] < p[0])
            interview, offer, withdrew = "", "", ""
            if ai:
                if wd[i, 0] < W_BEFORE_INTERVIEW:
                    withdrew = "before_interview"
                else:
                    interview = int(u[i, 1] < p[1])
                    if interview:
                        if wd[i, 1] < W_BEFORE_OFFER:
                            withdrew = "before_offer"
                        else:
                            offer = int(u[i, 2] < p[2])
            rows.append(dict(applicant_id=f"P{pid:05d}", job_family=fam,
                             requisition_id=f"{fam[:3].upper()}-{req[i] + 1:03d}",
                             sex="Woman" if woman[i] else "Man",
                             age_band="40+" if older[i] else "Under 40",
                             ai_screen_pass=ai, interview_pass=interview, offer=offer,
                             withdrew=withdrew))
    df = pd.DataFrame(rows)
    n = len(df)
    # application dates across the cycle
    start = np.datetime64("2026-06-01")
    df["application_date"] = start + rng.integers(40, 122, size=n).astype("timedelta64[D]")
    # missing / declined self-identification (at random)
    for col, pm in [("sex", P_MISS_SEX), ("age_band", P_MISS_AGE)]:
        miss = rng.random(n) < pm
        declined = rng.random(n) < 0.6
        df.loc[miss & declined, col] = "Prefer not to say"
        df.loc[miss & ~declined, col] = ""
    # duplicates: an EARLIER application by the same person, with its own outcome
    dup_idx = rng.choice(n, size=N_DUPLICATES, replace=False)
    dups = df.loc[dup_idx].copy()
    dups["application_date"] = dups["application_date"] - rng.integers(5, 40, size=N_DUPLICATES).astype("timedelta64[D]")
    dups["ai_screen_pass"] = (rng.random(N_DUPLICATES) < 0.3).astype(int)
    dups["interview_pass"] = ""
    dups["offer"] = ""
    dups["withdrew"] = np.where(dups["ai_screen_pass"] == 1, "before_interview", "")
    df = pd.concat([df, dups], ignore_index=True)
    # inconsistent job-family labels (~1%)
    bad = rng.random(len(df)) < 0.01
    for i in np.where(bad)[0]:
        fam = df.at[i, "job_family"]
        df.at[i, "job_family"] = LABEL_VARIANTS[fam][rng.integers(0, 3)]
    # shuffle rows and assign application ids in date order
    df = df.sample(frac=1, random_state=seed % (2 ** 32)).sort_values("application_date", kind="stable")
    df.insert(0, "application_id", [f"APP{k:06d}" for k in range(1, len(df) + 1)])
    df["application_date"] = pd.to_datetime(df["application_date"]).dt.strftime("%Y-%m-%d")
    cols = ["application_id", "applicant_id", "application_date", "job_family", "requisition_id",
            "sex", "age_band", "ai_screen_pass", "interview_pass", "offer", "withdrew"]
    return df[cols].reset_index(drop=True)


def true_parameters():
    """True selection rates and impact ratios implied by the design (expected values)."""
    out = []
    # family-level, stage-level ratios (exact by construction)
    for fam in FAMILIES:
        for comp, mult in [("age", AGE_MULT[fam]), ("sex", SEX_MULT)]:
            ref = list(BASE[fam])
            foc = [ref[s] * mult[s] for s in range(3)]
            for s, st in enumerate(STAGES):
                out.append(dict(stage=st, pool=fam, comparison=comp, true_rate_focal=foc[s],
                                true_rate_reference=ref[s], true_ir=mult[s]))
            # applicant -> offer (withdrawals count as not selected; same for both groups)
            keep = (1 - W_BEFORE_INTERVIEW) * (1 - W_BEFORE_OFFER)
            out.append(dict(stage="cumulative", pool=fam, comparison=comp,
                            true_rate_focal=np.prod(foc) * keep, true_rate_reference=np.prod(ref) * keep,
                            true_ir=np.prod(mult)))
    # pooled (company-wide) expected ratios, weighting families by design volumes/shares
    for comp in ["age", "sex"]:
        for st_i, st in enumerate(STAGES + ["cumulative"]):
            num_f = den_f = num_r = den_r = 0.0
            for fam in FAMILIES:
                n = N_APPLICANTS[fam]
                share = SHARE_40PLUS[fam] if comp == "age" else SHARE_WOMEN[fam]
                other = SHARE_WOMEN[fam] if comp == "age" else SHARE_40PLUS[fam]
                for grp_focal, nshare in [(True, share), (False, 1 - share)]:
                    # average the other characteristic's multipliers (parity for sex, so only age matters)
                    probs = []
                    for oth, w in [(True, other), (False, 1 - other)]:
                        older = grp_focal if comp == "age" else oth
                        m = AGE_MULT[fam] if older else (1, 1, 1)
                        probs.append((w, [BASE[fam][s] * m[s] for s in range(3)]))
                    # mass reaching and passing each stage
                    reach = pas = 0.0
                    for w, pr in probs:
                        if st == "ai_screen":
                            r, p_ = 1.0, pr[0]
                        elif st == "interview":
                            r, p_ = pr[0] * (1 - W_BEFORE_INTERVIEW), pr[0] * (1 - W_BEFORE_INTERVIEW) * pr[1]
                        elif st == "offer":
                            r = pr[0] * (1 - W_BEFORE_INTERVIEW) * pr[1] * (1 - W_BEFORE_OFFER)
                            p_ = r * pr[2]
                        else:
                            r, p_ = 1.0, pr[0] * pr[1] * pr[2] * (1 - W_BEFORE_INTERVIEW) * (1 - W_BEFORE_OFFER)
                        reach += w * r
                        pas += w * p_
                    if grp_focal:
                        num_f += n * nshare * pas; den_f += n * nshare * reach
                    else:
                        num_r += n * nshare * pas; den_r += n * nshare * reach
            rf, rr = num_f / den_f, num_r / den_r
            out.append(dict(stage=st, pool="All families", comparison=comp, true_rate_focal=rf,
                            true_rate_reference=rr, true_ir=rf / rr))
    tp = pd.DataFrame(out)
    tp["focal_group"] = np.where(tp["comparison"] == "age", "40+", "Woman")
    tp["reference_group"] = np.where(tp["comparison"] == "age", "Under 40", "Man")

    def label(r):
        if r < 0.80:
            return "real adverse impact"
        if r < 0.97:
            return "small real gap"
        if r > 1.03:
            return "focal group favored"
        return "parity"
    tp["true_category"] = tp["true_ir"].apply(label)
    for c in ["true_rate_focal", "true_rate_reference", "true_ir"]:
        tp[c] = tp[c].round(4)
    return tp[["stage", "pool", "comparison", "focal_group", "reference_group",
               "true_rate_focal", "true_rate_reference", "true_ir", "true_category"]]


def scan_ok(df):
    """Teaching-case checks used only by --scan to pick the seed (same cleaning and
    stage definitions as the analysis scripts)."""
    from scipy.stats import fisher_exact
    d = df.sort_values(["applicant_id", "application_date", "application_id"]).drop_duplicates("applicant_id", keep="last").copy()
    d["job_family"] = d["job_family"].str.strip().str.lower().map(
        {"software engineering": "Software Engineering", "software eng.": "Software Engineering",
         "customer operations": "Customer Operations", "customer ops": "Customer Operations", "finance": "Finance"})
    for c in ["interview_pass", "offer"]:
        d[c] = pd.to_numeric(d[c].replace("", np.nan))

    def cell(pool, comp, stage="ai_screen"):
        x = d if pool == "ALL" else d[d.job_family == pool]
        if stage == "interview":
            x = x[x.interview_pass.notna()]; sel = x.interview_pass == 1
        elif stage == "offer":
            x = x[x.offer.notna()]; sel = x.offer == 1
        elif stage == "cumulative":
            sel = x.offer == 1
        else:
            sel = x.ai_screen_pass == 1
        col, f, r = ("age_band", "40+", "Under 40") if comp == "age" else ("sex", "Woman", "Man")
        a, n1 = int(sel[x[col] == f].sum()), int((x[col] == f).sum())
        c, n2 = int(sel[x[col] == r].sum()), int((x[col] == r).sum())
        ir = (a / n1) / (c / n2) if a > 0 and c > 0 else np.nan
        p = fisher_exact([[a, n1 - a], [c, n2 - c]])[1]
        return ir, p
    ok = []
    ir, p = cell("Software Engineering", "age"); ok.append(ir < 0.78 and p < 0.001)
    ir, p = cell("Finance", "sex"); ok.append(0.70 <= ir < 0.80 and p >= 0.20)
    ir, p = cell("Customer Operations", "age"); ok.append(0.86 <= ir <= 0.94 and p < 0.01)
    for pool, comp in [("Software Engineering", "sex"), ("Customer Operations", "sex"), ("Finance", "age")]:
        ir, p = cell(pool, comp); ok.append(min(ir, 1 / ir) >= 0.85 and p >= 0.10)
    ir, p = cell("ALL", "age"); ok.append(ir >= 0.85)
    ir, p = cell("ALL", "age", "cumulative"); ok.append(ir >= 0.85)
    ir, p = cell("Software Engineering", "age", "cumulative"); ok.append(ir >= 0.82)
    # no other cell may reach an "act" verdict (four-fifths flag AND p < .05)
    act_elsewhere = False
    for st in ["ai_screen", "interview", "offer", "cumulative"]:
        for pool in ["Software Engineering", "Customer Operations", "Finance", "ALL"]:
            for comp in ["age", "sex"]:
                if (st, pool, comp) == ("ai_screen", "Software Engineering", "age"):
                    continue
                ir, p = cell(pool, comp, st)
                if np.isfinite(ir) and min(ir, 1 / ir) < 0.80 and p < 0.05:
                    act_elsewhere = True
    ok.append(not act_elsewhere)
    return all(ok), ok


if __name__ == "__main__":
    if "--scan" in sys.argv:
        hits = []
        for s in range(SCAN_START, SCAN_START + 1000):
            good, ok = scan_ok(generate(s))
            pass
            if good:
                hits.append(s)
        print("seeds meeting all teaching checks:", hits)
        sys.exit(0)
    df = generate(SEED)
    df.to_csv(os.path.join(HERE, "applications.csv"), index=False)
    true_parameters().to_csv(os.path.join(HERE, "true_parameters.csv"), index=False)
    print(f"Wrote {len(df)} application rows ({df.applicant_id.nunique()} unique applicants) "
          f"to data/applications.csv (seed {SEED}); true values in data/true_parameters.csv")
