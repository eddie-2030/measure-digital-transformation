"""
check_agreement.py -- confirm Python and R produce the same numbers, and build
outputs/results.json (every quotable number, with python and r values).

Run AFTER python/analysis.py and r/analysis.R:
    python python/check_agreement.py
Exit code 0 = every number agrees to its stated rounding; 1 = mismatch or missing.
"""
from pathlib import Path
import json
import math
import sys

import pandas as pd

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent.parent / "outputs"


def merge(write=True, verbose=True):
    py = pd.read_csv(OUT / "results_python.csv")
    r_path = OUT / "results_r.csv"
    r = pd.read_csv(r_path).set_index("name")["value"].to_dict() if r_path.exists() else {}
    items, problems = [], []
    for row in py.itertuples(index=False):
        digits = int(row.rounding)
        pv = round(float(row.value), digits) + 0.0
        rv = r.get(row.name)
        rv = None if rv is None or (isinstance(rv, float) and math.isnan(rv)) else round(float(rv), digits) + 0.0
        agree = None if rv is None else abs(pv - rv) < 1e-9
        if rv is None or not agree:
            problems.append((row.name, pv, rv))
        val = int(pv) if digits == 0 else pv
        items.append({
            "name": row.name, "value": val, "rounding": f"{digits} decimals" if digits else "whole number",
            "meaning": row.meaning, "headline": bool(row.headline),
            "python": val, "r": (None if rv is None else (int(rv) if digits == 0 else rv)), "agree": agree,
        })
    extra = sorted(set(r) - set(py.name))
    doc = {
        "edition": "F01 The Broken Bottom Rung",
        "note": "All numbers come from SYNTHETIC data generated to illustrate the method (seed 20261105). "
                "Cohort differences are descriptive, not causal.",
        "units": "shares and differences of shares are in percent / percentage points; months are months; tasks are tasks per month",
        "all_agree": (len(problems) == 0 and not extra and bool(r)),
        "results": items,
    }
    if write:
        (OUT / "results.json").write_text(json.dumps(doc, indent=2))
    if verbose:
        n_head = sum(i["headline"] for i in items)
        print(f"{len(items)} numbers checked ({n_head} headline). Mismatches/missing: {len(problems)}; R-only names: {len(extra)}")
        for p in problems:
            print("  MISMATCH", p)
        for e in extra:
            print("  R-ONLY", e)
    return len(problems) == 0 and not extra and bool(r)


if __name__ == "__main__":
    ok = merge(write=True, verbose=True)
    print("AGREEMENT OK" if ok else "AGREEMENT FAILED")
    sys.exit(0 if ok else 1)
