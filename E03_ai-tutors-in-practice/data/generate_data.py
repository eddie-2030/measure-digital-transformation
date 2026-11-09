"""
generate_data.py -- synthetic AI-tutor interaction logs for edition E03
"AI Tutors in Practice: What Interaction Data Reveals" (Measure Digital Transformation).

All data are synthetic, generated to illustrate the method. They do not come from
Meta, any client, or any real organization.

Run:  python data/generate_data.py      (works from any folder; paths are relative to this file)
Seed: 20261106 (fixed). Re-running gives byte-identical csv files.

SCENARIO
--------
A company runs an 8-week AI upskilling course ("data skills for analysts") with an AI tutor.
1,200 learners sit in 60 teams (each team has its own course instructor) across 3 business units.
Everyone takes a pre-test and a post-test (0-100) on a common assessment. The tutor logs every
learner action as a token, with a timestamp.

ACTION ALPHABET (6 tokens, plus START, which is implicit at the beginning of every session)
    T = attempts the problem (submits own work to the tutor / runs own code)
    H = asks for a hint or an explanation
    A = asks for the full answer / worked solution
    C = copies or pastes the AI answer into the submission
    V = checks or verifies own work with the tutor
    Q = quits / ends the session normally (the END token).
        A session cut by a logout or time-out simply stops without a Q.

DATA-GENERATING PROCESS (the ground truth the analysis tries to recover)
-----------------------------------------------------------------------
1. Prior skill. pre = 55 + BU shift (-2, 0, +2) + Normal(0, 13), clipped to [5, 95], rounded.

2. Latent usage style (never observed by the analyst). Multinomial logit on z = (pre - 55) / 13:
       log-odds(answer_seeking vs scaffolded) = -0.18 - 0.9 z
       log-odds(light_mixed    vs scaffolded) = -0.41 - 0.3 z
   This targets population shares of about 40% scaffolded, 35% answer-seeking, 25% light/mixed,
   and makes LOW prior skill -> MORE answer-seeking (so style is confounded with prior skill).

3. Each style has its own FIRST-ORDER transition matrix P(next | current). Rows are the current
   state (START, T, H, A, C, V), columns the next token (T, H, A, C, V, Q). See TRUE_P below.
   Scaffolded learners run "attempt -> hint -> attempt -> verify" loops and turn answers into new
   attempts (A -> T). Answer-seekers run "hint -> answer -> copy -> answer -> copy" loops.
   Light/mixed learners do a bit of everything and end sessions quickly.
   DESIGN CHOICE: the matrices are tuned so that scaffolded and answer-seeking learners use every
   action type in broadly similar proportions (expected shares T/H/A/C/V: scaffolded
   34/17/18/13/18%, answer-seeking 26/14/24/20/16%). Most of the difference is in the ORDER,
   which is the ground truth the article is about ("order, not volume or mix, carries the signal").
   Expected session length (actions before Q): scaffolded 7.6, answer-seeking 8.6, light 3.8.

4. Volume. Sessions per learner = 1 + Poisson(m - 1) with m = 11 (scaffolded), 11 (answer-seeking),
   8 (light/mixed). Total messages are therefore about equal for scaffolded and answer-seeking
   learners (answer-seekers about 10% higher) and lower for light users. Dwell time after each action
   is lognormal (median seconds: T 110, H 45, A 120, C 30, V 60; sd of log 0.6), so minutes are
   also similar across the two main styles.

5. Learning gain (post - pre), in score points:
       gain = 2 + style effect + 0.15 * (55 - pre) + instructor effect + Normal(0, 6)
       style effect: scaffolded +14, light/mixed +8, answer-seeking +3
       instructor (team) effect ~ Normal(0, 3), one per team
   post = clip(round(pre + gain), 0, 100). "Gained" (the outcome we predict) = post - pre >= 10.

6. Messy realism (deliberate):
   - about 2% of learners drop out after one very short session (1-3 actions, no Q);
   - 5% of sessions are truncated by a logout (no Q at the end);
   - 2% of non-Q tokens are MIS-CODED: replaced by a random different action token;
   - 0.5% of tokens are written in lower case or with stray spaces (" h", "a ");
   - 0.3% of tokens are unreadable ("?"); the analysis drops them and reports the count;
   - 18 learners (1.5%) have no post-test (post_score is blank); they are reported, not silently dropped;
   - team-level instructor effects on the gain.

7. Train/test split by LEARNER: a random 30% of learners (fixed by the seed) are "test".
   Both the Python and R analyses read this column, so they use identical sets.

Outputs (in this folder):
   tutor_events.csv         one row per logged action (the event log)
   learners.csv             one row per learner (team, BU, pre/post scores, split)
   ground_truth.csv         true style and noise-free expected gain per learner (for the check only)
   ground_truth_params.json all true parameters above, plus the realized style shares
   ground_truth_transitions.csv  the three true transition matrices in long form (for R and spreadsheets)
"""
import json
import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

SEED = 20261106
HERE = os.path.dirname(os.path.abspath(__file__))

STYLES = ["scaffolded", "answer_seeking", "light_mixed"]
ROWS = ["START", "T", "H", "A", "C", "V"]       # current state
COLS = ["T", "H", "A", "C", "V", "Q"]           # next token

# True first-order transition matrices; each row sums to 1.
TRUE_P = {
    # Scaffolded: attempt -> hint -> attempt -> verify; after verifying, compare with the worked
    # solution (V -> A) or paste the verified final version (V -> C); after an answer, re-attempt (A -> T).
    "scaffolded": [
        # T    H    A    C    V    Q
        [.55, .20, .10, .03, .10, .02],   # START
        [.06, .28, .18, .05, .31, .12],   # T
        [.64, .04, .15, .07, .06, .04],   # H
        [.48, .10, .04, .16, .14, .08],   # A
        [.28, .12, .14, .04, .12, .30],   # C
        [.10, .05, .32, .33, .04, .16],   # V
    ],
    # Answer-seeking: hint -> answer -> copy -> answer -> copy loops; runs the pasted answer (C -> T)
    # and "verifies" it (T -> V). Uses every action type, but in a different ORDER.
    "answer_seeking": [
        [.34, .32, .26, .02, .04, .02],   # START
        [.12, .14, .18, .06, .40, .10],   # T
        [.12, .06, .62, .10, .06, .04],   # H
        [.12, .06, .04, .62, .08, .08],   # A
        [.55, .08, .16, .03, .06, .12],   # C
        [.25, .20, .20, .06, .04, .25],   # V
    ],
    # Light / mixed: a bit of everything, short sessions (high Q).
    "light_mixed": [
        [.45, .20, .20, .05, .08, .02],   # START
        [.15, .20, .20, .05, .15, .25],   # T
        [.35, .08, .25, .07, .10, .15],   # H
        [.25, .08, .07, .25, .10, .25],   # A
        [.20, .08, .20, .07, .10, .35],   # C
        [.25, .10, .10, .10, .10, .35],   # V
    ],
}
STYLE_LOGIT = {"answer_seeking": (-0.18, -0.9), "light_mixed": (-0.41, -0.3)}  # (intercept, slope on z)
TARGET_SHARES = {"scaffolded": 0.40, "answer_seeking": 0.35, "light_mixed": 0.25}
SESSIONS_MEAN = {"scaffolded": 11, "answer_seeking": 11, "light_mixed": 8}
STYLE_EFFECT = {"scaffolded": 14.0, "light_mixed": 8.0, "answer_seeking": 3.0}
GAIN_INTERCEPT, GAIN_PRIOR_SLOPE, GAIN_NOISE_SD, TEAM_SD = 2.0, 0.15, 6.0, 3.0
GAIN_THRESHOLD = 10
DWELL_MEDIAN_SEC = {"T": 110, "H": 45, "A": 120, "C": 30, "V": 60, "Q": 0}

N_LEARNERS, N_TEAMS = 1200, 60
BUS = ["Finance & Risk", "Operations", "Sales & Marketing"]
BU_SHIFT = [-2.0, 0.0, 2.0]
P_EARLY_DROP, P_TRUNCATE, P_MISCODE, P_MESSY_CASE, P_UNREADABLE = 0.02, 0.05, 0.02, 0.005, 0.003
N_NO_POST = 18
TEST_SHARE = 0.30
COURSE_START = datetime(2026, 9, 7)


def main():
    rng = np.random.default_rng(SEED)

    # --- learners, teams, business units -------------------------------------------------
    team_bu = np.repeat(np.arange(3), N_TEAMS // 3)                 # 20 teams per BU
    team_w = rng.dirichlet(np.full(N_TEAMS, 8.0))                    # unequal team sizes
    team = rng.choice(N_TEAMS, size=N_LEARNERS, p=team_w)
    bu = team_bu[team]
    team_effect = rng.normal(0, TEAM_SD, N_TEAMS)

    pre = np.clip(np.round(55 + np.array(BU_SHIFT)[bu] + rng.normal(0, 13, N_LEARNERS)), 5, 95)
    z = (pre - 55) / 13

    # --- latent style ----------------------------------------------------------------------
    logits = np.column_stack([np.zeros(N_LEARNERS),
                              STYLE_LOGIT["answer_seeking"][0] + STYLE_LOGIT["answer_seeking"][1] * z,
                              STYLE_LOGIT["light_mixed"][0] + STYLE_LOGIT["light_mixed"][1] * z])
    probs = np.exp(logits)
    probs /= probs.sum(1, keepdims=True)
    u = rng.random(N_LEARNERS)
    style_idx = (u[:, None] > np.cumsum(probs, 1)).sum(1)
    style = np.array(STYLES)[style_idx]

    # --- gain --------------------------------------------------------------------------------
    eff = np.array([STYLE_EFFECT[s] for s in style])
    expected_gain = GAIN_INTERCEPT + eff + GAIN_PRIOR_SLOPE * (55 - pre) + team_effect[team]
    gain = expected_gain + rng.normal(0, GAIN_NOISE_SD, N_LEARNERS)
    post = np.clip(np.round(pre + gain), 0, 100)
    post_out = post.astype(object)
    no_post = rng.choice(N_LEARNERS, N_NO_POST, replace=False)
    post_out[no_post] = None

    split = np.array(["train"] * N_LEARNERS, dtype=object)
    split[rng.permutation(N_LEARNERS)[: int(round(TEST_SHARE * N_LEARNERS))]] = "test"

    # --- event logs --------------------------------------------------------------------------
    P = {s: np.array(TRUE_P[s]) for s in STYLES}
    for s in STYLES:
        assert np.allclose(P[s].sum(1), 1.0), s
    early_drop = rng.random(N_LEARNERS) < P_EARLY_DROP
    rows = []
    for i in range(N_LEARNERS):
        lid = f"L{i + 1:04d}"
        s = style[i]
        n_sess = 1 if early_drop[i] else 1 + rng.poisson(SESSIONS_MEAN[s] - 1)
        days = np.sort(rng.integers(0, 56, n_sess))
        for k in range(n_sess):
            # simulate one session from START until Q (cap 60 actions)
            seq, cur = [], 0
            while len(seq) < 60:
                nxt = COLS[rng.choice(6, p=P[s][cur])]
                seq.append(nxt)
                if nxt == "Q":
                    break
                cur = ROWS.index(nxt)
            if early_drop[i]:
                seq = [t for t in seq if t != "Q"][: rng.integers(1, 4)] or ["T"]
            elif rng.random() < P_TRUNCATE and len(seq) > 1:
                seq = seq[: rng.integers(1, len(seq))]                 # logout: no Q
            # observation layer: mis-coding and messy codes
            obs = []
            for t in seq:
                if t != "Q" and rng.random() < P_MISCODE:
                    t = rng.choice([x for x in "THACV" if x != t])
                r = rng.random()
                if r < P_UNREADABLE:
                    t = "?"
                elif r < P_UNREADABLE + P_MESSY_CASE:
                    t = [t.lower(), " " + t, t + " "][rng.integers(0, 3)]
                obs.append(t)
            ts = COURSE_START + timedelta(days=int(days[k]), hours=int(rng.integers(8, 20)),
                                          minutes=int(rng.integers(0, 60)))
            for j, t in enumerate(obs):
                rows.append((lid, f"{lid}_S{k + 1:02d}", j + 1, t, ts.strftime("%Y-%m-%dT%H:%M:%S")))
                base = t.strip().upper()
                med = DWELL_MEDIAN_SEC.get(base, 30)
                if med > 0:
                    ts = ts + timedelta(seconds=int(round(med * np.exp(rng.normal(0, 0.6)))))

    events = pd.DataFrame(rows, columns=["learner_id", "session_id", "step", "action", "timestamp"])
    learners = pd.DataFrame({
        "learner_id": [f"L{i + 1:04d}" for i in range(N_LEARNERS)],
        "team_id": [f"TM{t + 1:02d}" for t in team],
        "business_unit": np.array(BUS)[bu],
        "pre_score": pre.astype(int),
        "post_score": pd.array([None if p is None else int(p) for p in post_out], dtype="Int64"),
        "split": split,
    })
    truth = pd.DataFrame({
        "learner_id": learners.learner_id,
        "true_style": style,
        "expected_gain": np.round(expected_gain, 2),
        "early_dropout": early_drop.astype(int),
    })

    events.to_csv(os.path.join(HERE, "tutor_events.csv"), index=False)
    learners.to_csv(os.path.join(HERE, "learners.csv"), index=False)
    truth.to_csv(os.path.join(HERE, "ground_truth.csv"), index=False)
    long = [(s, r, c, TRUE_P[s][i][j]) for s in STYLES for i, r in enumerate(ROWS) for j, c in enumerate(COLS)]
    pd.DataFrame(long, columns=["style", "from_state", "to_token", "true_prob"]).to_csv(
        os.path.join(HERE, "ground_truth_transitions.csv"), index=False)

    realized = {s: round(float(np.mean(style == s)), 4) for s in STYLES}
    params = {
        "seed": SEED,
        "n_learners": N_LEARNERS, "n_teams": N_TEAMS, "business_units": BUS,
        "styles": STYLES,
        "style_shares_target": TARGET_SHARES,
        "style_shares_realized": realized,
        "style_logit_vs_scaffolded": {k: {"intercept": v[0], "slope_on_z_pre": v[1]} for k, v in STYLE_LOGIT.items()},
        "transition_rows": ROWS, "transition_cols": COLS,
        "true_transition_matrices": TRUE_P,
        "sessions_mean": SESSIONS_MEAN,
        "dwell_median_seconds": DWELL_MEDIAN_SEC,
        "gain_model": {"formula": "gain = 2 + style_effect + 0.15*(55 - pre) + team_effect + N(0, 6^2)",
                       "style_effect": STYLE_EFFECT, "team_effect_sd": TEAM_SD,
                       "noise_sd": GAIN_NOISE_SD, "gained_definition": "post - pre >= 10"},
        "messiness": {"early_dropout_share": P_EARLY_DROP, "truncated_session_share": P_TRUNCATE,
                      "miscoded_token_share": P_MISCODE, "messy_case_share": P_MESSY_CASE,
                      "unreadable_token_share": P_UNREADABLE, "n_missing_post_test": N_NO_POST},
        "test_share": TEST_SHARE,
    }
    with open(os.path.join(HERE, "ground_truth_params.json"), "w") as f:
        json.dump(params, f, indent=2)
    print(f"wrote {len(events):,} events for {N_LEARNERS} learners; realized style shares {realized}")


if __name__ == "__main__":
    main()
