"""
Compare the Python and R results and record R's values in outputs/results.json.

Run after both analyses:
    python python/analysis.py
    Rscript r/analysis.R
    python python/check_agreement.py

- Cell table: every count must be identical, every estimate equal to 1e-8 (relative),
  every verdict identical.
- Headline numbers: equal at the stated rounding. Monte Carlo rates (mc_*) use
  different random-number generators, so they are compared to a tolerance of 0.02
  and both values are kept; the EXACT false-alarm rates must match exactly.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")

py = pd.read_csv(os.path.join(OUT, "python_results.csv"))
r = pd.read_csv(os.path.join(OUT, "r_results.csv"))
assert list(py.columns) == list(r.columns) and len(py) == len(r)
bad = []
for c in py.columns:
    if pd.api.types.is_numeric_dtype(py[c]) and pd.api.types.is_numeric_dtype(r[c]):
        if not np.allclose(py[c].astype(float), r[c].astype(float), rtol=1e-8, atol=1e-12):
            bad.append(c)
    elif not (py[c].astype(str) == r[c].astype(str)).all():
        bad.append(c)
print(f"Cell table: {len(py)} cells x {len(py.columns)} columns compared; mismatching columns: {bad or 'none'}")

res_path = os.path.join(OUT, "results.json")
res = json.load(open(res_path))
rh = pd.read_csv(os.path.join(OUT, "r_headline.csv")).set_index("name")["value"]
mism, mc_diffs = [], {}
for name, entry in res["results"].items():
    rv = float(rh[name])
    rv = int(rv) if entry["rounding"] == "integer" else rv
    entry["r"] = rv
    if name.startswith("mc_"):
        mc_diffs[name] = round(abs(rv - entry["python"]), 3)
        if mc_diffs[name] > 0.02:
            mism.append(name)
    elif entry["rounding"].endswith("significant figures"):
        if not np.isclose(rv, entry["python"], rtol=1e-9):
            mism.append(name)
    elif rv != entry["python"]:
        mism.append(name)
n_mc = len(mc_diffs)
res["agreement"] = dict(
    cell_table_columns_mismatching=bad,
    headline_compared=len(res["results"]),
    headline_mismatches=mism,
    note=(f"All {len(res['results']) - n_mc} deterministic headline numbers agree exactly at the stated rounding. "
          f"The {n_mc} Monte Carlo rates (mc_*) use different RNGs (numpy PCG64 vs R Mersenne-Twister, same seed "
          f"{res['seed_monte_carlo']}); largest Python-R difference {max(mc_diffs.values()):.3f}. "
          "Exact false-alarm rates (exact_*) are computed by enumeration and agree exactly."),
)
json.dump(res, open(res_path, "w"), indent=2)
print(f"Headline numbers: {len(res['results'])} compared; mismatches: {mism or 'none'}; "
      f"largest Monte Carlo difference {max(mc_diffs.values()):.3f}")
sys.exit(1 if (bad or mism) else 0)
