"""
stratified_randomize.py  --  a stratified (blocked) randomization helper for internal pilots.

What it does
  Reads a roster CSV (one row per employee), splits people into strata (e.g. site, or
  site x baseline-performance tercile), and assigns half of each stratum to the tool and
  half to the waitlist/control. Writes an assignment file you can archive with the plan.

Usage (from the package root)
  python3 templates/stratified_randomize.py ROSTER.csv OUT.csv --strata site \
          [--baseline cases_pre] [--seed 12345] [--id employee_id]

  --strata    one or more roster columns to block on (e.g. --strata site team_size_band)
  --baseline  optional numeric column; adds its tercile (within the other strata) as a stratum
  --seed      write it in the experiment plan BEFORE you run this; never re-run to "get a
              better draw" -- that quietly undoes randomization

Example (uses the synthetic pilot data in this folder):
  python3 templates/stratified_randomize.py data/randomized_pilot.csv templates/example_assignment.csv \
          --strata site --baseline cases_pre_w1 --seed 2026

Output columns: employee_id, stratum, arm (tool / waitlist), seed, randomized_on
"""

import argparse
import datetime as dt

import numpy as np
import pandas as pd


def stratified_assign(df, strata_cols, baseline=None, seed=12345, id_col="employee_id"):
    rng = np.random.default_rng(seed)
    df = df.drop_duplicates(subset=id_col).copy()
    key = df[strata_cols].astype(str).agg("-".join, axis=1)
    if baseline:
        # tercile of the baseline WITHIN each of the other strata
        terc = df.groupby(key)[baseline].transform(
            lambda x: pd.qcut(x.rank(method="first"), 3, labels=["T1", "T2", "T3"]).astype(str))
        key = key + "-" + terc
    df["stratum"] = key
    df["arm"] = "waitlist"
    # sort first so the result depends only on the seed, not on the roster's row order
    df = df.sort_values(["stratum", id_col]).reset_index(drop=True)
    for _, idx in df.groupby("stratum").indices.items():
        idx = rng.permutation(idx)
        n_tool = len(idx) // 2 + (int(rng.random() < 0.5) if len(idx) % 2 else 0)
        df.loc[idx[:n_tool], "arm"] = "tool"
    df["seed"] = seed
    df["randomized_on"] = dt.date.today().isoformat()
    return df[[id_col, "stratum", "arm", "seed", "randomized_on"]]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roster")
    ap.add_argument("out")
    ap.add_argument("--strata", nargs="+", required=True)
    ap.add_argument("--baseline", default=None)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--id", default="employee_id")
    a = ap.parse_args()
    out = stratified_assign(pd.read_csv(a.roster), a.strata, a.baseline, a.seed, a.id)
    out.to_csv(a.out, index=False)
    # balance table: the arms should be about equal in every stratum
    print(pd.crosstab(out["stratum"], out["arm"], margins=True))
    print(f"Wrote {a.out}")
