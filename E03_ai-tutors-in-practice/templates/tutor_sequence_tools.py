"""
tutor_sequence_tools.py -- reusable functions for AI-tutor event logs (E03 companion template).

Input: an event log with columns learner_id, session_id, step, action, timestamp
       (see tutor_event_log_template.csv). Actions: T attempt, H hint, A full answer,
       C copy/paste answer, V verify, Q quit (end of session). A session may end without Q.
Optional: a labels file with columns learner_id, label (1 = good outcome, 0 = not).

Command line:
    python tutor_sequence_tools.py events.csv                  # transition matrix + scorecard metrics
    python tutor_sequence_tools.py events.csv labels.csv       # + two chains and a log-score per learner
                                                                 #   (writes <events>_log_scores.csv)
Only numpy and pandas are needed. For an honest test of the log-score, fit the chains on some
learners and score OTHER learners (fit_chains on a training subset, log_scores on the rest).
"""
import sys

import numpy as np
import pandas as pd

ROWS = ["S", "T", "H", "A", "C", "V"]   # S = start of session
COLS = ["T", "H", "A", "C", "V", "Q"]


def load_events(path):
    """Read and clean an event log: normalize codes, drop unknown codes and anything after Q."""
    ev = pd.read_csv(path, dtype=str, keep_default_na=False)
    ev["tok"] = ev["action"].str.strip().str.upper()
    dropped = int((~ev["tok"].isin(COLS)).sum())
    ev = ev[ev["tok"].isin(COLS)].copy()
    ev["step"] = ev["step"].astype(int)
    ev = ev.sort_values(["learner_id", "session_id", "step"]).reset_index(drop=True)
    after_q = ev.groupby("session_id")["tok"].transform(lambda s: (s == "Q").cumsum().shift(fill_value=0) > 0)
    ev = ev[~after_q].copy()
    ev["prev"] = ev.groupby("session_id")["tok"].shift(1, fill_value="S")
    if dropped:
        print(f"note: dropped {dropped} rows with unknown action codes")
    return ev


def transition_counts(ev):
    """Learner x 36 matrix of first-order transition counts (row-major: from-state x next-token)."""
    ids = sorted(ev["learner_id"].unique())
    li = ev["learner_id"].map({l: i for i, l in enumerate(ids)}).to_numpy()
    cell = ev["prev"].map(ROWS.index).to_numpy() * 6 + ev["tok"].map(COLS.index).to_numpy()
    N = np.zeros((len(ids), 36))
    np.add.at(N, (li, cell), 1)
    return ids, N


def transition_matrix(counts36, smooth=0.5):
    """P(next | current) from summed counts, with a small pseudo-count so no cell is zero."""
    c = np.asarray(counts36, dtype=float).reshape(6, 6) + smooth
    return pd.DataFrame(c / c.sum(1, keepdims=True), index=ROWS, columns=COLS)


def fit_chains(ids, N, labels):
    """Two chains: learners with label 1 and with label 0. labels: dict learner_id -> 0/1."""
    y = np.array([labels.get(i, np.nan) for i in ids], dtype=float)
    return transition_matrix(N[y == 1].sum(0)), transition_matrix(N[y == 0].sum(0))


def log_scores(N, P_good, P_bad):
    """log-score = sum over steps of log[ P_good(next | current) / P_bad(next | current) ]."""
    lr = (np.log(P_good.to_numpy()) - np.log(P_bad.to_numpy())).ravel()
    return N @ lr


def scorecard_metrics(ev):
    """The 'learning-rich interaction' metrics from the scorecard, pooled over the file."""
    P = transition_matrix(transition_counts(ev)[1].sum(0))
    acts = ev[ev["tok"] != "Q"]
    return {
        "learners": ev["learner_id"].nunique(),
        "sessions": ev["session_id"].nunique(),
        "messages_per_learner": round(len(acts) / max(ev["learner_id"].nunique(), 1), 1),
        "P(attempt | hint)": round(P.loc["H", "T"], 2),
        "P(re-attempt | answer)": round(P.loc["A", "T"], 2),
        "P(answer | hint)": round(P.loc["H", "A"], 2),
        "P(copy | answer)": round(P.loc["A", "C"], 2),
        "share of actions that are full-answer requests (%)": round(100 * (acts["tok"] == "A").mean()),
        "sessions ending without Q (%)": round(100 * (1 - ev.groupby("session_id")["tok"].apply(lambda s: (s == "Q").any()).mean())),
    }


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    ev = load_events(sys.argv[1])
    ids, N = transition_counts(ev)
    print("Pooled transition matrix P(next | current):")
    print(transition_matrix(N.sum(0)).round(2).to_string())
    print("\nScorecard metrics:")
    for k, v in scorecard_metrics(ev).items():
        print(f"  {k}: {v}")
    if len(sys.argv) > 2:
        lab = pd.read_csv(sys.argv[2])
        labels = dict(zip(lab["learner_id"], lab["label"]))
        P_good, P_bad = fit_chains(ids, N, labels)
        out = pd.DataFrame({"learner_id": ids, "log_score": np.round(log_scores(N, P_good, P_bad), 3)})
        path = sys.argv[1].rsplit(".", 1)[0] + "_log_scores.csv"
        out.to_csv(path, index=False)
        print(f"\nWrote {path} (higher = sequence looks more like the label-1 group).")
