"""
Compare Python and R, number by number, and record the R values in outputs/results.json.

Run after python/analysis.py and r/analysis.R:  python python/check_agreement.py
Every number must agree at its stated rounding (coefficients 2 decimals, shares whole percent).
Ground-truth values (true_*) are read from the same file by both languages, so they agree trivially.
"""
import json
import sys
from pathlib import Path

import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs"
res = json.loads((OUT / "results.json").read_text())
rvals = pd.read_csv(OUT / "r_results.csv", dtype=str).set_index("name")["value"]

mismatches, compared, max_raw_diff = [], 0, 0.0
for name, entry in res["numbers"].items():
    if name not in rvals.index:
        mismatches.append((name, "missing in R"))
        continue
    rd = entry["rounding"]
    if rd is None:                                  # text value, e.g. the flagged items
        r_val = rvals[name]
        ok = r_val == entry["python"]
    else:
        raw = float(rvals[name])
        r_val = int(round(raw)) if rd == 0 else round(raw, rd)
        ok = r_val == entry["python"]
        max_raw_diff = max(max_raw_diff, abs(raw - entry["python_unrounded"]))
    entry["r"] = r_val
    entry["agree"] = bool(ok)
    compared += 1
    if not ok:
        mismatches.append((name, f"python {entry['python']} vs r {r_val}"))

res["agreement"] = {"numbers_compared": compared, "mismatches": len(mismatches),
                    "max_abs_difference_unrounded": max_raw_diff}
res["note"] = "value = the number to quote; 'python' and 'r' are each language's value at the stated rounding"
(OUT / "results.json").write_text(json.dumps(res, indent=2))

print(f"Compared {compared} numbers. Largest unrounded difference: {max_raw_diff:.2e}")
if mismatches:
    for m in mismatches:
        print("MISMATCH:", *m)
    sys.exit(1)
print("Python and R agree on every number at the stated rounding.")
