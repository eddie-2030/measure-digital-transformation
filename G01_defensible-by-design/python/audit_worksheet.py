"""
Run an adverse impact audit on YOUR counts, using the worksheet template.

    python python/audit_worksheet.py                                   # the synthetic example
    python python/audit_worksheet.py my_counts.csv --min-n 30          # your own file

Input: a csv with columns stage, job_family, group_variable, group, applicants, selected
(see templates/adverse_impact_audit_worksheet.csv). Any number of groups per
group_variable is allowed. Within each stage x job_family x group_variable, every
group is compared with the HIGHEST-rate group (the Uniform Guidelines convention).

Output (printed and saved next to the input as <name>_audit.csv; the bundled
example is saved to outputs/worksheet_example_audit.csv): selection rates,
impact ratio with a 95% Katz interval, Fisher exact p-value (two-sided), shortfall,
and a verdict that reads the four-fifths rule and the test together. Groups with
fewer than --min-n applicants are marked "below minimum n" and get no verdict.

This produces evidence for review, not a legal conclusion.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
Z95 = 1.959963984540054

ap = argparse.ArgumentParser()
ap.add_argument("path", nargs="?", default=os.path.join(ROOT, "templates", "adverse_impact_audit_worksheet.csv"))
ap.add_argument("--min-n", type=int, default=30, help="pre-set minimum applicants per group (default 30)")
args = ap.parse_args()

w = pd.read_csv(args.path)
need = {"stage", "job_family", "group_variable", "group", "applicants", "selected"}
missing = need - set(w.columns)
assert not missing, f"worksheet is missing columns: {missing}"
assert (w.selected <= w.applicants).all(), "selected cannot exceed applicants"
w["rate"] = w.selected / w.applicants

out = []
for key, g in w.groupby(["stage", "job_family", "group_variable"], sort=False):
    ref = g.loc[g.rate.idxmax()]                      # the highest-rate group is the reference
    for _, f in g.iterrows():
        if f.group == ref.group:
            continue
        a, n1, c, n2 = int(f.selected), int(f.applicants), int(ref.selected), int(ref.applicants)
        aa, cc, m1, m2 = (a, c, n1, n2) if a > 0 and c > 0 else (a + .5, c + .5, n1 + 1, n2 + 1)
        ir = (a / n1) / (c / n2)
        se = np.sqrt(1 / aa - 1 / m1 + 1 / cc - 1 / m2)
        ir_c = (aa / m1) / (cc / m2)
        p = fisher_exact([[a, n1 - a], [c, n2 - c]])[1]
        flag = 5 * a * n2 < 4 * c * n1                 # exact test of IR < 0.80
        if min(n1, n2) < args.min_n:
            verdict = "below minimum n: collect more data"
        elif flag and p < 0.05:
            verdict = "Adverse impact: act"
        elif flag:
            verdict = "Inconclusive: four-fifths flag, not significant"
        elif p < 0.05:
            verdict = "Small but real gap: monitor"
        else:
            verdict = "No evidence of impact"
        out.append(dict(stage=key[0], job_family=key[1], group_variable=key[2], group=f.group,
                        reference_group=ref.group, applicants=n1, selected=a, rate=a / n1,
                        reference_rate=c / n2, impact_ratio=ir, ci_low=ir_c * np.exp(-Z95 * se),
                        ci_high=ir_c * np.exp(Z95 * se), fisher_p=p, four_fifths_flag=flag,
                        shortfall=c / n2 * n1 - a, verdict=verdict))
res = pd.DataFrame(out)
DEFAULT = os.path.join(ROOT, "templates", "adverse_impact_audit_worksheet.csv")
dest = (os.path.join(ROOT, "outputs", "worksheet_example_audit.csv")
        if os.path.abspath(args.path) == DEFAULT else os.path.splitext(args.path)[0] + "_audit.csv")
res.to_csv(dest, index=False, float_format="%.4f")
pd.set_option("display.width", 220)
print(res.drop(columns=["reference_rate"]).round(3).to_string(index=False))
print(f"\nSaved {dest}. Evidence for review, not a legal conclusion.")
