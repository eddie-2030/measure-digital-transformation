"""
tenure_state_summary.py -- tenure-aligned state distribution and time-in-state
summary for ANY csv of monthly person-state records.

Required columns: person_id, cohort, job_family, tenure_month, state
  state codes: R = routine, S = supervised complex (stretch), I = independent
  complex, X = left. Blank = missing month (filled with the previous state).
  One row per person per observed month, INCLUDING months after leaving (X)
  up to your data cut -- that is how right truncation is handled correctly.

Usage:
    python templates/tenure_state_summary.py my_records.csv [first_n_months]
    (no arguments -> runs on person_state_records_template.csv)
Or import:  from tenure_state_summary import state_distribution, time_in_state
"""
from pathlib import Path
import sys

import pandas as pd

STATES = ["R", "S", "I", "X"]


def clean_states(df):
    """Standardize codes, carry the previous state into blank months, make X absorbing."""
    df = df.sort_values(["person_id", "tenure_month"]).copy()
    s = df["state"].fillna("").astype(str).str.strip().str.upper().replace({"LEFT": "X"})
    s = s.str[:1]                                                            # 'SUP' -> 'S', 'Indep' -> 'I'
    s = s.where(s.isin(STATES))                                              # unknown/blank -> NaN
    df["state"] = s.groupby(df["person_id"]).ffill().fillna("R")
    left = (df["state"] == "X").groupby(df["person_id"]).cummax()
    df.loc[left, "state"] = "X"
    return df


def state_distribution(df, by=("cohort", "job_family")):
    """% of people in each state at each tenure month, per group.
    Denominator = people observed at that tenure month (leavers stay in as X)."""
    df = clean_states(df)
    keys = list(by) + ["tenure_month"]
    tab = (df.groupby(keys)["state"].value_counts(normalize=True).mul(100)
             .unstack(fill_value=0).reindex(columns=STATES, fill_value=0).round(1))
    tab["n_observed"] = df.groupby(keys)["person_id"].nunique()
    return tab.reset_index()


def time_in_state(df, first_n_months=12, by=("cohort", "job_family")):
    """Mean months spent in each state during tenure months 1..first_n_months,
    among people observed for at least first_n_months (right truncation handled)."""
    df = clean_states(df)
    n_obs = df.groupby("person_id")["tenure_month"].max()
    keep = n_obs.index[n_obs >= first_n_months]
    sub = df[df.person_id.isin(keep) & (df.tenure_month <= first_n_months)]
    if sub.empty:                                    # guard: nobody observed long enough
        print(f"No one is observed for {first_n_months} months yet; lower first_n_months.")
        return pd.DataFrame()
    per = (pd.crosstab(sub.person_id, sub.state).reindex(columns=STATES, fill_value=0)
             .join(sub.groupby("person_id")[list(by)].first()))
    out = per.groupby(list(by))[STATES].mean().round(2).add_prefix("months_in_")
    out["n_people"] = per.groupby(list(by)).size()
    out[f"pct_in_I_at_month_{first_n_months}"] = (
        sub[sub.tenure_month == first_n_months].groupby(list(by))["state"].apply(lambda s: (s == "I").mean() * 100).round(0))
    return out.reset_index()


if __name__ == "__main__":
    sys.dont_write_bytecode = True
    here = Path(__file__).resolve().parent
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else here / "person_state_records_template.csv"
    months = int(sys.argv[2]) if len(sys.argv) > 2 else (12 if len(sys.argv) > 1 else 6)
    data = pd.read_csv(path, dtype={"state": str}, keep_default_na=False)
    pd.set_option("display.width", 160)
    print("State distribution by tenure month (%):")
    print(state_distribution(data).to_string(index=False))
    print(f"\nTime in state, months 1-{months}:")
    print(time_in_state(data, months).to_string(index=False))
