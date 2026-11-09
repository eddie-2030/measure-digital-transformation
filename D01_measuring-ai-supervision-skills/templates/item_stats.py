"""
Reusable item statistics for any item-response csv (one row per person, one column per item).

Usage:
  python templates/item_stats.py responses.csv                      # all numeric columns are items
  python templates/item_stats.py responses.csv --items S01 S02 S03   # choose the items of ONE scale
  python templates/item_stats.py responses.csv --out my_item_stats.csv

Or import:  from item_stats import item_statistics; table, summary = item_statistics(df, items)

Output columns match templates/item_statistics_template.csv:
  item, n, mean, sd, item_rest_r, alpha_if_deleted, decision
Rules (edit to taste, and write them in your report card):
  item_rest_r < 0.20  -> drop  ("drop (check key)" if negative: the scoring key may be reversed)
  0.20 to 0.30, or alpha goes UP when the item is deleted -> revise
  otherwise -> keep
Missing answers: pairwise-complete covariances (nobody is dropped for one blank answer).
Run it on one intended scale at a time; alpha assumes the items measure one thing.
Dependencies: numpy, pandas.
"""
import argparse
import math

import numpy as np
import pandas as pd


def _alpha(c):
    k = c.shape[0]
    return k / (k - 1) * (1 - np.trace(c) / c.sum())


def item_statistics(df, items=None, drop_below=0.20, revise_below=0.30):
    items = list(items) if items else df.select_dtypes("number").columns.tolist()
    c = df[items].cov().to_numpy()  # pairwise-complete
    a = _alpha(c)
    rows = []
    for i, it in enumerate(items):
        rest = [j for j in range(len(items)) if j != i]
        r_ir = c[i, rest].sum() / math.sqrt(c[i, i] * c[np.ix_(rest, rest)].sum())
        a_del = _alpha(c[np.ix_(rest, rest)])
        if r_ir < drop_below:
            dec = "drop" if r_ir >= 0 else "drop (check key)"
        elif r_ir < revise_below or a_del > a:
            dec = "revise"
        else:
            dec = "keep"
        rows.append({"item": it, "n": int(df[it].notna().sum()), "mean": df[it].mean(), "sd": df[it].std(),
                     "item_rest_r": r_ir, "alpha_if_deleted": a_del, "decision": dec})
    table = pd.DataFrame(rows)
    summary = {"k_items": len(items), "alpha": a, "n_people": int(df[items].notna().any(axis=1).sum())}
    return table, summary


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("csv")
    p.add_argument("--items", nargs="*")
    p.add_argument("--out", default="item_statistics.csv")
    a = p.parse_args()
    table, summary = item_statistics(pd.read_csv(a.csv), a.items)
    table.round(3).to_csv(a.out, index=False)
    print(table.round(2).to_string(index=False))
    print(f"\n{summary['k_items']} items, {summary['n_people']} people, Cronbach's alpha = {summary['alpha']:.2f}")
    print(f"Wrote {a.out}")
