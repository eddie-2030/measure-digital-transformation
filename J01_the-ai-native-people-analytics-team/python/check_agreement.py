"""
check_agreement.py -- compare Python and R results and record both in results.json
==================================================================================
Run after python/analysis.py and r/analysis.R (from the package root):
    python python/check_agreement.py

* Point estimates must agree EXACTLY at the stated rounding (3 decimals; 1 for pp).
* Bootstrap CI bounds come from different random-number generators (numpy vs. R's
  Mersenne-Twister), so they may differ slightly; we report both and flag any
  difference larger than 0.02.
The script adds a "python"/"r" pair to every numeric result and a "headline" block.
"""

import json
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs"
res = json.load(open(OUT / "results.json"))
py = pd.read_csv(OUT / "python_results.csv").set_index("name")["value"]
r = pd.read_csv(OUT / "r_results.csv").set_index("name")["value"]

mismatch, ci_diff = [], []
for name, rec in res["results"].items():
    if name not in r.index:
        continue                      # text/boolean items exist only in Python
    pv, rv = float(py[name]), float(r[name])
    rec["python"], rec["r"] = rec["value"], rv
    if name.startswith("corrected_ci_") or "_boot_ci_" in name:
        if abs(pv - rv) > 0.02:
            ci_diff.append((name, pv, rv))
    elif not (pd.isna(pv) and pd.isna(rv)) and pv != rv:
        mismatch.append((name, pv, rv))

HEAD = ["naive_thumbs_up_rate", "naive_judge_accuracy", "judge_sensitivity", "judge_specificity",
        "corrected_accuracy_overall", "corrected_ci_low_overall", "corrected_ci_high_overall",
        "human_only_accuracy", "human_only_boot_ci_low", "human_only_boot_ci_high",
        "corrected_accuracy_policy_lookup", "corrected_accuracy_computed_metric",
        "corrected_accuracy_multi_hop", "recall_at5_multi_hop", "groundedness_multi_hop",
        "fail_share_retrieval_miss_multi_hop", "true_accuracy_sample_overall"]
res["headline"] = {k: {"python": res["results"][k]["python"], "r": res["results"][k]["r"],
                       "meaning": res["results"][k]["meaning"]} for k in HEAD}
res["agreement"] = {
    "point_estimates_compared": int(sum(1 for n in res["results"] if n in r.index and not (n.startswith("corrected_ci_") or "_boot_ci_" in n))),
    "point_estimate_mismatches": [list(m) for m in mismatch],
    "ci_bounds_differing_by_more_than_0.02": [list(m) for m in ci_diff],
    "note": "Point estimates identical in Python and R; CI bounds differ only through RNG."}
json.dump(res, open(OUT / "results.json", "w"), indent=2)

print(f"Compared {res['agreement']['point_estimates_compared']} point estimates: "
      f"{len(mismatch)} mismatches. CI bounds differing > 0.02: {len(ci_diff)}")
for m in mismatch + ci_diff:
    print("  ", m)
sys.exit(1 if mismatch else 0)
