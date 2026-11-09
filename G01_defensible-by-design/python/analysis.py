"""
G01 · Defensible by Design: an adverse impact audit of an AI resume screen.

Run from the package root or from anywhere (paths are resolved from this file):
    python python/analysis.py

Steps
  0. Clean the file: fix job-family labels, keep each applicant's latest
     application, count undisclosed sex/age and withdrawals.
  1. NAIVE: one company-wide four-fifths check (final offers, and the AI screen
     pooled across job families). It passes.
  2. CORRECT: for every stage x job family x comparison, compute selection
     rates, the impact ratio (IR), a 95% interval, a Fisher exact p-value, and
     the shortfall; then read the four-fifths rule and the test TOGETHER.
  3. GROUND TRUTH: compare with the true ratios from the generator, and show
     (exactly, and by Monte Carlo) how often the four-fifths rule flags a group
     by chance when there is no impact at all.
  4. DECISION: what the audit pack should say.

Writes: outputs/python_results.csv, outputs/python_false_alarm.csv,
        outputs/results.json, outputs/fig1_*.png, outputs/fig2_*.png
"""
import json
import os

import numpy as np
import pandas as pd
from scipy.stats import binom, fisher_exact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "outputs")
os.makedirs(OUT, exist_ok=True)

FOUR_FIFTHS = 0.80
ALPHA = 0.05
Z95 = 1.959963984540054
MC_SEED = 20261272
MC_REPS = 5000
MC_SIZES = [25, 50, 100, 250, 1000]
MC_BASE_RATE = 0.30

FAMILY_FIX = {"software engineering": "Software Engineering", "software eng.": "Software Engineering",
              "customer operations": "Customer Operations", "customer ops": "Customer Operations",
              "finance": "Finance"}
POOLS = ["Software Engineering", "Customer Operations", "Finance", "All families"]
STAGES = ["ai_screen", "interview", "offer", "cumulative"]
COMPARISONS = {"age": ("age_band", "40+", "Under 40"), "sex": ("sex", "Woman", "Man")}


def rnd(x, k):
    """Round half away from zero (same rule as the R script), so both agree exactly."""
    return float(np.sign(x) * np.floor(abs(x) * 10 ** k + 0.5) / 10 ** k)


# ---------------------------------------------------------------------------
# Step 0. Load and clean
# ---------------------------------------------------------------------------
raw = pd.read_csv(os.path.join(DATA, "applications.csv"), dtype=str, keep_default_na=False)
n_rows = len(raw)
raw["job_family"] = raw["job_family"].str.strip().str.lower().map(FAMILY_FIX)
assert raw["job_family"].notna().all(), "unknown job-family label"
n_label_fixed = int((pd.read_csv(os.path.join(DATA, "applications.csv"), dtype=str)["job_family"]
                    != raw["job_family"]).sum())
# Duplicates: the same person applied twice. Keep the LATEST application (ties broken by id).
d = (raw.sort_values(["applicant_id", "application_date", "application_id"])
        .drop_duplicates("applicant_id", keep="last").reset_index(drop=True))
n_dupes_removed = n_rows - len(d)
for c in ["ai_screen_pass", "interview_pass", "offer"]:
    d[c] = pd.to_numeric(d[c].replace("", np.nan))
# Undisclosed self-identification ("Prefer not to say" or blank) is excluded from
# THAT comparison only (an applicant with unknown age still counts in the sex comparison).
n_sex_missing = int((~d["sex"].isin(["Woman", "Man"])).sum())
n_age_missing = int((~d["age_band"].isin(["40+", "Under 40"])).sum())
n_withdrew_int = int((d["withdrew"] == "before_interview").sum())
n_withdrew_off = int((d["withdrew"] == "before_offer").sum())
print(f"Step 0. {n_rows} rows -> {len(d)} applicants after removing {n_dupes_removed} earlier duplicate "
      f"applications; {n_label_fixed} job-family labels fixed.")
print(f"        Sex not disclosed: {n_sex_missing}; age not disclosed: {n_age_missing} "
      f"(excluded from that comparison only). Withdrew: {n_withdrew_int} before interview, "
      f"{n_withdrew_off} before offer.")


def stage_frame(x, stage):
    """Who was at risk of selection at a stage (denominator) and who was selected.
    ai_screen : everyone; selected = passed the AI screen
    interview : AI passers who did not withdraw first; selected = passed interview
    offer     : interview passers who did not withdraw first; selected = offer
    cumulative: everyone (withdrawals count as not selected); selected = offer
    """
    if stage == "ai_screen":
        return x, x["ai_screen_pass"] == 1
    if stage == "interview":
        y = x[x["interview_pass"].notna()]
        return y, y["interview_pass"] == 1
    if stage == "offer":
        y = x[x["offer"].notna()]
        return y, y["offer"] == 1
    return x, x["offer"] == 1


def katz_ci(a, n1, c, n2):
    """95% CI for the ratio of two proportions (a/n1)/(c/n2), log scale (Katz).
    se(log IR) = sqrt(1/a - 1/n1 + 1/c - 1/n2). If a cell is 0, add 0.5 to every cell."""
    if a == 0 or c == 0:
        a, c, n1, n2 = a + 0.5, c + 0.5, n1 + 1, n2 + 1
    ir = (a / n1) / (c / n2)
    se = np.sqrt(1 / a - 1 / n1 + 1 / c - 1 / n2)
    return ir * np.exp(-Z95 * se), ir * np.exp(Z95 * se)


def audit_cell(a, n1, c, n2):
    """All the numbers for one comparison. a/n1 = focal selected/applicants, c/n2 = reference."""
    r1, r2 = a / n1, c / n2
    ir = r1 / r2
    lo, hi = katz_ci(a, n1, c, n2)
    p = fisher_exact([[a, n1 - a], [c, n2 - c]], alternative="two-sided")[1]
    # Four-fifths compares the LOWER-rate group with the HIGHER-rate group, in either direction.
    focal_lower = r1 <= r2
    adverse_ir = ir if focal_lower else 1 / ir
    # Exact integer test of "lower rate / higher rate < 4/5" (avoids floating-point ties at 0.80)
    flag = 5 * min(a * n2, c * n1) < 4 * max(a * n2, c * n1)
    # Shortfall: extra selections the lower group needs to match the higher group's rate
    if focal_lower:
        short = r2 * n1 - a
        short80 = max(0.0, FOUR_FIFTHS * r2 * n1 - a)
    else:
        short = r1 * n2 - c
        short80 = max(0.0, FOUR_FIFTHS * r1 * n2 - c)
    sig = p < ALPHA
    if flag and sig:
        verdict = "Adverse impact: act"
    elif flag:
        verdict = "Inconclusive: four-fifths flag, not significant"
    elif sig:
        verdict = "Small but real gap: monitor"
    else:
        verdict = "No evidence of impact"
    return dict(rate_focal=r1, rate_reference=r2, ir=ir, ir_lo=lo, ir_hi=hi, diff=r1 - r2,
                fisher_p=p, adverse_ir=adverse_ir, focal_is_lower=bool(focal_lower),
                four_fifths_flag=bool(flag), significant=bool(sig),
                shortfall=short, shortfall_to_080=short80, verdict=verdict)


def build_table(d):
    rows = []
    for stage in STAGES:
        for pool in POOLS:
            x = d if pool == "All families" else d[d["job_family"] == pool]
            y, sel = stage_frame(x, stage)
            for comp, (col, focal, ref) in COMPARISONS.items():
                f, r = y[col] == focal, y[col] == ref
                a, n1, c, n2 = int(sel[f].sum()), int(f.sum()), int(sel[r].sum()), int(r.sum())
                rows.append(dict(stage=stage, pool=pool, comparison=comp, focal_group=focal,
                                 reference_group=ref, n_focal=n1, sel_focal=a, n_reference=n2,
                                 sel_reference=c, **audit_cell(a, n1, c, n2)))
    t = pd.DataFrame(rows)
    t["lower_group"] = np.where(t["focal_is_lower"], t["focal_group"], t["reference_group"])
    # Pre-registered PRIMARY cells: the AI screen (the tool we deployed) within each job family.
    t["primary"] = (t["stage"] == "ai_screen") & (t["pool"] != "All families")
    return t


tab = build_table(d)


def cell(stage, pool, comp):
    return tab[(tab.stage == stage) & (tab.pool == pool) & (tab.comparison == comp)].iloc[0]


# ---------------------------------------------------------------------------
# Step 1. The naive audit: one company-wide number
# ---------------------------------------------------------------------------
naive_offer = cell("cumulative", "All families", "age")
naive_ai = cell("ai_screen", "All families", "age")
naive_ai_sex = cell("ai_screen", "All families", "sex")
print("\nStep 1. NAIVE company-wide four-fifths checks (40+ vs under 40):")
print(f"        Final offers (applicant -> offer): IR = {naive_offer.ir:.2f}  -> passes four-fifths")
print(f"        AI screen pooled over families:   IR = {naive_ai.ir:.2f}  -> passes four-fifths")
print("        Conclusion a naive audit would write: 'the tool is fine'. Both numbers hide where it is not.")

# ---------------------------------------------------------------------------
# Step 2. The correct audit: stage x job family, both rules, interval, shortfall
# ---------------------------------------------------------------------------
pd.set_option("display.width", 200)
show = tab[tab.stage == "ai_screen"][["pool", "comparison", "n_focal", "n_reference", "rate_focal",
                                       "rate_reference", "ir", "ir_lo", "ir_hi", "fisher_p",
                                       "shortfall", "verdict"]]
print("\nStep 2. AI screen, by job family (IR = focal / reference; focal = 40+ or women):")
print(show.round(3).to_string(index=False))
n_cells = len(tab)
n_flag = int(tab.four_fifths_flag.sum())
n_sig = int(tab.significant.sum())
n_act = int((tab.verdict == "Adverse impact: act").sum())
n_disagree = int((tab.four_fifths_flag != tab.significant).sum())
print(f"        Across all {n_cells} cells (4 stages x 4 pools x 2 comparisons): {n_flag} four-fifths flags, "
      f"{n_sig} significant, {n_act} both; the two rules disagree in {n_disagree} cells.")

# ---------------------------------------------------------------------------
# Step 3. Ground truth: true ratios, interval coverage, and the false-alarm rate
# ---------------------------------------------------------------------------
truth = pd.read_csv(os.path.join(DATA, "true_parameters.csv"))
tab = tab.merge(truth[["stage", "pool", "comparison", "true_ir", "true_category"]],
                on=["stage", "pool", "comparison"], how="left")
tab["true_adverse"] = np.minimum(tab.true_ir, 1 / tab.true_ir) < FOUR_FIFTHS
tab["ci_covers_truth"] = (tab.ir_lo <= tab.true_ir) & (tab.true_ir <= tab.ir_hi)
n_true_adverse = int(tab.true_adverse.sum())
ff_false = int((tab.four_fifths_flag & ~tab.true_adverse).sum())
act_false = int(((tab.verdict == "Adverse impact: act") & ~tab.true_adverse).sum())
act_true = int(((tab.verdict == "Adverse impact: act") & tab.true_adverse).sum())
coverage = int(tab.ci_covers_truth.sum())
print(f"\nStep 3. Ground truth: {n_true_adverse} of {n_cells} cells have a true IR below 0.80. "
      f"The four-fifths rule flagged {n_flag} cells, {ff_false} of them false alarms.")
print(f"        'Act' (flag AND significant) caught {act_true} true case(s) with {act_false} false alarm(s). "
      f"95% intervals contain the true IR in {coverage} of {n_cells} cells.")


def exact_flag_rate(n1, n2, p):
    """Exact probability that the four-fifths rule flags EITHER group when both truly
    have selection rate p (true parity). Enumerates every possible pair of outcomes."""
    x = np.arange(n1 + 1)[:, None]
    y = np.arange(n2 + 1)[None, :]
    flag = 5 * np.minimum(x * n2, y * n1) < 4 * np.maximum(x * n2, y * n1)
    w = binom.pmf(np.arange(n1 + 1), n1, p)[:, None] * binom.pmf(np.arange(n2 + 1), n2, p)[None, :]
    return float((w * flag).sum())


def exact_focal_flag_rate(n1, n2, p):
    """Same, but only flags where the FOCAL group (n1) is the lower one."""
    x = np.arange(n1 + 1)[:, None]
    y = np.arange(n2 + 1)[None, :]
    flag = 5 * x * n2 < 4 * y * n1
    w = binom.pmf(np.arange(n1 + 1), n1, p)[:, None] * binom.pmf(np.arange(n2 + 1), n2, p)[None, :]
    return float((w * flag).sum())


# Monte Carlo under true parity: equal groups of n, both with selection rate 30%.
rng = np.random.default_rng(MC_SEED)
mc_rows = []
for n in MC_SIZES:
    xa = rng.binomial(n, MC_BASE_RATE, MC_REPS)
    xb = rng.binomial(n, MC_BASE_RATE, MC_REPS)
    flag = 5 * np.minimum(xa, xb) < 4 * np.maximum(xa, xb)
    pv = np.array([fisher_exact([[a, n - a], [b, n - b]])[1] for a, b in zip(xa, xb)])
    mc_rows.append(dict(n_per_group=n, exact_four_fifths_flag_rate=exact_flag_rate(n, n, MC_BASE_RATE),
                        mc_four_fifths_flag_rate=flag.mean(), mc_fisher_sig_rate=(pv < ALPHA).mean(),
                        mc_both_rate=(flag & (pv < ALPHA)).mean()))
mc = pd.DataFrame(mc_rows)
# a finer exact curve for the figure
curve_n = [10, 15, 20, 25, 30, 40, 50, 60, 75, 100, 125, 150, 200, 250, 300, 400, 500, 600, 750, 1000]
curve = pd.DataFrame(dict(n_per_group=curve_n,
                          exact_four_fifths_flag_rate=[exact_flag_rate(n, n, MC_BASE_RATE) for n in curve_n]))
mc.to_csv(os.path.join(OUT, "python_false_alarm.csv"), index=False)
print("\n        Under TRUE parity (both groups selected at 30%), how often does each rule raise a flag?")
print(mc.round(3).to_string(index=False))

fin = cell("ai_screen", "Finance", "sex")
fin_p_both = exact_flag_rate(int(fin.n_focal), int(fin.n_reference), float(fin.rate_reference))
fin_p_women = exact_focal_flag_rate(int(fin.n_focal), int(fin.n_reference), float(fin.rate_reference))
print(f"        Finance: with {fin.n_focal} women and {fin.n_reference} men at a {fin.rate_reference:.0%} pass rate, "
      f"parity still produces a four-fifths flag {fin_p_both:.0%} of the time ({fin_p_women:.0%} against women).")

# ---------------------------------------------------------------------------
# Step 4. Decision implication
# ---------------------------------------------------------------------------
swe = cell("ai_screen", "Software Engineering", "age")
co = cell("ai_screen", "Customer Operations", "age")
swe_cum = cell("cumulative", "Software Engineering", "age")
print("\nStep 4. What the audit pack should say:")
print(f"  ACT    SWE, AI screen, 40+: IR {swe.ir:.2f} (95% CI {swe.ir_lo:.2f}-{swe.ir_hi:.2f}), p {swe.fisher_p:.1e}; "
      f"{swe.shortfall:.0f} more 40+ passes needed for equal rates. Pause/adjust the screen for this family and "
      "investigate the age-correlated feature, even though the family's applicant-to-offer IR is "
      f"{swe_cum.ir:.2f}.")
print(f"  MONITOR CO, AI screen, 40+: IR {co.ir:.2f} (95% CI {co.ir_lo:.2f}-{co.ir_hi:.2f}), p {co.fisher_p:.4f}; "
      f"passes four-fifths but is real; shortfall {co.shortfall:.0f}.")
print(f"  MORE DATA Finance, AI screen, women: IR {fin.ir:.2f} (95% CI {fin.ir_lo:.2f}-{fin.ir_hi:.2f}), "
      f"p {fin.fisher_p:.2f}; a four-fifths flag that the data cannot confirm. Do not call it a finding; "
      "pool more cycles or pre-set a minimum n.")
print("  ALWAYS  Report both rules with intervals, pre-register the primary cells (AI stage x family), "
      "and treat any IR as evidence for review, not a legal conclusion.")

# ---------------------------------------------------------------------------
# Save tables and results.json
# ---------------------------------------------------------------------------
cols = ["stage", "pool", "comparison", "focal_group", "reference_group", "n_focal", "sel_focal",
        "n_reference", "sel_reference", "rate_focal", "rate_reference", "ir", "ir_lo", "ir_hi",
        "diff", "fisher_p", "adverse_ir", "lower_group", "four_fifths_flag", "significant",
        "shortfall", "shortfall_to_080", "verdict", "primary", "true_ir", "true_category",
        "ci_covers_truth"]
tab[cols].to_csv(os.path.join(OUT, "python_results.csv"), index=False, float_format="%.10g")

R = {}


def put(name, value, digits, meaning):
    R[name] = dict(value=rnd(value, digits) if digits is not None else value,
                   rounding=f"{digits} decimals" if digits is not None else "integer",
                   meaning=meaning, python=rnd(value, digits) if digits is not None else value)


def put_sig(name, value, sig, meaning):
    v = float(f"{value:.{sig}g}")
    R[name] = dict(value=v, rounding=f"{sig} significant figures", meaning=meaning, python=v)


put("n_application_rows", n_rows, None, "Rows in the raw file (applications, including duplicates)")
put("n_applicants", len(d), None, "Unique applicants after keeping each person's latest application")
put("n_duplicates_removed", n_dupes_removed, None, "Earlier duplicate applications removed")
put("n_labels_fixed", n_label_fixed, None, "Job-family labels typed inconsistently and corrected")
put("n_sex_not_disclosed", n_sex_missing, None, "Applicants excluded from the women-vs-men comparison")
put("n_age_not_disclosed", n_age_missing, None, "Applicants excluded from the 40+ comparison")
put("n_withdrew_before_interview", n_withdrew_int, None, "AI-screen passers who withdrew before interview")
put("n_withdrew_before_offer", n_withdrew_off, None, "Interview passers who withdrew before an offer")
put("naive_offer_ir_age", naive_offer.ir, 2, "NAIVE: company-wide applicant-to-offer IR, 40+ vs under 40 (passes)")
put("naive_ai_ir_age", naive_ai.ir, 2, "NAIVE: AI-screen IR pooled over families, 40+ vs under 40 (passes)")
put("naive_ai_ir_sex", naive_ai_sex.ir, 2, "AI-screen IR pooled over families, women vs men (women higher: mix of families)")
for key, c_, label in [("swe_ai_age", swe, "Software Engineering, AI screen, 40+ vs under 40"),
                       ("co_ai_age", co, "Customer Operations, AI screen, 40+ vs under 40"),
                       ("fin_ai_sex", fin, "Finance, AI screen, women vs men")]:
    put(f"{key}_n_focal", int(c_.n_focal), None, f"{label}: focal-group applicants")
    put(f"{key}_n_reference", int(c_.n_reference), None, f"{label}: reference-group applicants")
    put(f"{key}_rate_focal", c_.rate_focal, 3, f"{label}: focal selection rate")
    put(f"{key}_rate_reference", c_.rate_reference, 3, f"{label}: reference selection rate")
    put(f"{key}_ir", c_.ir, 2, f"{label}: impact ratio")
    put(f"{key}_ir_lo", c_.ir_lo, 2, f"{label}: 95% CI lower bound (Katz log method)")
    put(f"{key}_ir_hi", c_.ir_hi, 2, f"{label}: 95% CI upper bound")
    put_sig(f"{key}_fisher_p", c_.fisher_p, 3, f"{label}: Fisher exact two-sided p-value")
    put(f"{key}_shortfall", c_.shortfall, 0, f"{label}: extra focal selections needed to equalize rates")
    put(f"{key}_true_ir", float(c_.true_ir), 2, f"{label}: TRUE impact ratio from the generator")
put("swe_ai_age_shortfall_to_080", swe.shortfall_to_080, 0, "SWE AI screen 40+: extra passes needed just to reach IR 0.80")
put("swe_cum_age_ir", swe_cum.ir, 2, "Software Engineering, applicant-to-offer IR, 40+ (looks acceptable)")
put("swe_interview_age_ir", cell("interview", "Software Engineering", "age").ir, 2,
    "Software Engineering, interview stage IR, 40+ vs under 40 (humans partly offset the AI screen)")
put("fin_parity_flag_prob", fin_p_both, 2, "Finance: exact chance of a four-fifths flag (either group) under true parity at these group sizes")
put("fin_parity_flag_prob_women", fin_p_women, 2, "Finance: exact chance women are flagged under true parity")
put("n_cells", n_cells, None, "Audit cells: 4 stages x 4 pools (3 families + all) x 2 comparisons")
put("n_four_fifths_flags", n_flag, None, "Cells flagged by the four-fifths rule")
put("n_significant", n_sig, None, "Cells with Fisher p < 0.05")
put("n_act", n_act, None, "Cells both flagged and significant ('act')")
put("n_rules_disagree", n_disagree, None, "Cells where the four-fifths rule and the significance test disagree")
put("n_true_adverse", n_true_adverse, None, "Cells whose TRUE IR is below 0.80")
put("n_four_fifths_false_alarms", ff_false, None, "Four-fifths flags in cells with no true adverse impact")
put("n_act_false_alarms", act_false, None, "'Act' verdicts in cells with no true adverse impact")
put("n_ci_cover_truth", coverage, None, "Cells whose 95% CI contains the true IR")
for _, r_ in mc.iterrows():
    n = int(r_.n_per_group)
    put(f"exact_ff_flag_rate_n{n}", r_.exact_four_fifths_flag_rate, 3,
        f"Exact four-fifths false-alarm rate under parity, {n} per group, 30% base rate")
    put(f"mc_ff_flag_rate_n{n}", r_.mc_four_fifths_flag_rate, 3,
        f"Monte Carlo ({MC_REPS} reps) four-fifths false-alarm rate, {n} per group (RNG-dependent)")
    put(f"mc_fisher_sig_rate_n{n}", r_.mc_fisher_sig_rate, 3,
        f"Monte Carlo Fisher p<0.05 rate under parity, {n} per group (RNG-dependent; Fisher is conservative)")
results = dict(edition="G01", title="Defensible by Design", seed_data=20261272, seed_monte_carlo=MC_SEED,
               monte_carlo_reps=MC_REPS, ci_method="Katz log-ratio Wald: log(IR) +/- 1.96*sqrt(1/a-1/n1+1/c-1/n2)",
               p_value_method="Fisher exact, two-sided", results=R)
with open(os.path.join(OUT, "results.json"), "w") as fh:
    json.dump(results, fh, indent=2)
curve.to_csv(os.path.join(OUT, "python_false_alarm_curve.csv"), index=False)

# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

INK, MUTED, GRID = "#0b0b0b", "#5f5e5a", "#e1e0d9"
VCOL = {"Adverse impact: act": "#d03b3b", "Small but real gap: monitor": "#ec835a",
        "Inconclusive: four-fifths flag, not significant": "#fab219", "No evidence of impact": "#8a8984"}
VSHORT = {"Adverse impact: act": "ACT", "Small but real gap: monitor": "MONITOR",
          "Inconclusive: four-fifths flag, not significant": "INCONCLUSIVE",
          "No evidence of impact": "no evidence"}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13})


def pfmt(p):
    return "p<.001" if p < 0.001 else f"p={p:.2f}".replace("0.", ".")


# Fig 1: impact ratios with intervals
from matplotlib.transforms import blended_transform_factory
rows = [("Applicant \u2192 offer", cell("cumulative", "All families", "age")),
        ("AI screen", cell("ai_screen", "All families", "age")),
        ("Software Eng.: 40+", swe), ("Software Eng.: women", cell("ai_screen", "Software Engineering", "sex")),
        ("Customer Ops: 40+", co), ("Customer Ops: women", cell("ai_screen", "Customer Operations", "sex")),
        ("Finance: 40+", cell("ai_screen", "Finance", "age")), ("Finance: women", fin)]
fig, ax = plt.subplots(figsize=(8, 8.8), dpi=200)
ypos = [8.9, 7.9, 5.6, 4.6, 3.6, 2.6, 1.6, 0.6]
right_col = blended_transform_factory(ax.transAxes, ax.transData)   # x in axes units, y in data units
left_col = blended_transform_factory(fig.transFigure, ax.transData)
for (lab, c_), y in zip(rows, ypos):
    col = VCOL[c_.verdict]
    ax.plot([c_.ir_lo, c_.ir_hi], [y, y], color=col, lw=3, solid_capstyle="round", zorder=2)
    ax.scatter([c_.ir], [y], s=130, color=col, edgecolor="white", linewidth=1.5, zorder=4)
    ax.scatter([c_.true_ir], [y], s=80, marker="D", facecolor="none", edgecolor=INK, linewidth=1.6, zorder=5)
    ax.text(1.03, y + 0.17, f"{c_.ir:.2f}  {pfmt(c_.fisher_p)}", va="center", ha="left", fontsize=12.5,
            color=INK, transform=right_col)
    ax.text(1.03, y - 0.22, VSHORT[c_.verdict], va="center", ha="left", fontsize=11, color=MUTED,
            fontweight="bold", transform=right_col)
ax.set_yticks(ypos)
ax.set_yticklabels([r[0] for r in rows], fontsize=13)
ax.axvline(0.80, color=INK, lw=1.4, ls="--", zorder=1)
ax.axvline(1.00, color="#c3c2b7", lw=1.2, zorder=1)
ax.text(0.79, 9.75, "four-fifths\nline 0.80", ha="right", va="center", fontsize=11, color=INK)
ax.text(1.01, 9.75, "parity\n1.00", ha="left", va="center", fontsize=11, color=MUTED)
ax.text(0.03, 9.6, "NAIVE: company-wide, 40+", fontsize=11.5, color=MUTED, fontweight="bold",
        va="center", ha="left", transform=left_col)
ax.text(0.03, 6.5, "CORRECT: AI screen, by job family", fontsize=11.5, color=MUTED, fontweight="bold",
        va="center", ha="left", transform=left_col)
ax.set_xlim(0.40, 1.60)
ax.set_ylim(0.1, 10.3)
ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6])
ax.set_xlabel("Impact ratio (focal rate / reference rate)\nDot = estimate, bar = 95% interval",
              fontsize=12, color=INK)
for s_ in ["top", "right", "left"]:
    ax.spines[s_].set_visible(False)
ax.spines["bottom"].set_color("#c3c2b7")
ax.tick_params(axis="y", length=0)
handles = [Line2D([], [], marker="o", color=VCOL[k], lw=3, markersize=9,
                  label=VSHORT[k].capitalize() if k != "No evidence of impact" else "No evidence")
           for k in ["Adverse impact: act", "Small but real gap: monitor",
                     "Inconclusive: four-fifths flag, not significant", "No evidence of impact"]]
handles.append(Line2D([], [], marker="D", color="none", markeredgecolor=INK, markerfacecolor="none",
                      markersize=8, label="True ratio (synthetic truth)"))
fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=11.5,
           bbox_to_anchor=(0.5, 0.0), handletextpad=0.4, columnspacing=1.2)
fig.suptitle("Company-wide, the AI screen passes the four-fifths\nrule. By job family, it fails 40+ applicants "
             "in\nSoftware Engineering", x=0.03, ha="left", y=0.985, fontsize=17.5, fontweight="bold", color=INK)
fig.text(0.03, 0.845, f"Synthetic data: {len(d):,} applicants, 3 job families. Focal groups: 40+, women.",
         fontsize=11, color=MUTED, ha="left")
fig.subplots_adjust(left=0.30, right=0.76, top=0.80, bottom=0.185)
fig.savefig(os.path.join(OUT, "fig1_ai_screen_fails_40plus_in_one_family.png"), facecolor="white")
plt.close(fig)

# Fig 2: false alarms under true parity
fig, ax = plt.subplots(figsize=(8, 6.8), dpi=200)
ax.plot(curve.n_per_group, curve.exact_four_fifths_flag_rate, color="#2a78d6", lw=2.5, zorder=3)
ax.scatter(mc.n_per_group, mc.exact_four_fifths_flag_rate, s=70, color="#2a78d6", edgecolor="white",
           linewidth=1.5, zorder=4)
ax.plot(mc.n_per_group, mc.mc_fisher_sig_rate, color="#8a8984", lw=2, ls="--", marker="o", markersize=7,
        markeredgecolor="white", zorder=3)
for _, r_ in mc.iterrows():
    v = r_.exact_four_fifths_flag_rate
    if v >= 0.01:
        ax.annotate(f"{v:.0%}", (r_.n_per_group, v), textcoords="offset points",
                    xytext=(0, 11), ha="center", fontsize=13, fontweight="bold", color=INK)
    else:   # the 1,000 point sits near 0: label it to the right, clear of the Fisher line
        ax.annotate("<1%", (r_.n_per_group, v), textcoords="offset points",
                    xytext=(9, 2), ha="left", va="bottom", fontsize=13, fontweight="bold", color=INK)
ax.set_xscale("log")
ax.set_xticks(MC_SIZES)
ax.set_xticklabels([str(n) for n in MC_SIZES])
ax.minorticks_off()
ax.set_xlim(18, 1800)
ax.set_ylim(0, 0.70)
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
ax.set_xlabel("Applicants per group (both groups the same size, log scale)", fontsize=12.5)
ax.set_ylabel("Share of audits that raise a flag", fontsize=12.5)
for s_ in ["top", "right"]:
    ax.spines[s_].set_visible(False)
ax.grid(axis="y", color=GRID, lw=0.8, zorder=0)
ax.legend(handles=[Line2D([], [], color="#2a78d6", lw=2.5, marker="o", label="Four-fifths flag (exact)"),
                   Line2D([], [], color="#8a8984", lw=2, ls="--", marker="o",
                          label=f"Fisher p < .05 (simulated, {MC_REPS:,} runs)")],
          loc="upper right", frameon=False, fontsize=12)
r50 = mc.loc[mc.n_per_group == 50, "exact_four_fifths_flag_rate"].iloc[0]
fig.suptitle(f"With no real impact, the four-fifths rule still\n"
             f"flags {r50:.0%} of audits with 50 people per group.\n"
             f"At 1,000 per group, it almost never does",
             x=0.03, ha="left", y=0.985, fontsize=17, fontweight="bold")
fig.text(0.03, 0.80, "Both groups truly selected at 30% (true parity). Synthetic simulation.",
         fontsize=11, color=MUTED, ha="left")
fig.subplots_adjust(left=0.13, right=0.96, top=0.765, bottom=0.12)
fig.savefig(os.path.join(OUT, "fig2_four_fifths_false_alarms_by_group_size.png"), facecolor="white")
plt.close(fig)
print("\nWrote outputs/python_results.csv, outputs/python_false_alarm.csv, outputs/results.json, and 2 figures.")
