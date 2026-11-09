"""
analysis.py -- F01 "The Broken Bottom Rung": cohort comparison on a tenure clock
plus career sequence analysis.  (All data are synthetic.)

Run from the package root or from python/:
    python python/analysis.py        or      cd python && python analysis.py

Steps
  0. Load and clean the person-month panel (state codes, blanks, X-after-X errors).
  1. NAIVE: average tasks closed per month in months 1-6, pre- vs post-AI cohorts.
  2. CORRECT: align everyone on the tenure clock (months since hire) and look at
     whole sequences:
       2a  state distribution by tenure month (the "chronogram", Figure 2)
       2b  months in supervised complex work (S) during the first 12 months
       2c  share in independent work (I) at tenure month 18
       2d  Hamming distance between 18-month sequences + k-medoids (PAM) into
           3 trajectory types; type shares pre vs post, by job family
       2e  bootstrap 95% intervals for the key pre-to-post differences
  3. CHECK against ground truth (adjusted Rand index, % correctly classified,
     true vs estimated shift in plateau share).
  4. DECISION implication (printed).

Writes outputs/results_python.csv, outputs/fig1_*.png, outputs/fig2_*.png and
(by calling check_agreement.merge) outputs/results.json.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score

sys.dont_write_bytecode = True  # keep the package free of __pycache__ folders

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

STATES = ["R", "S", "I", "X"]
GROUPS = [("HIGH", "pre"), ("HIGH", "post"), ("LOW", "pre"), ("LOW", "post")]
POST_COHORTS = {"2024H2", "2025H1", "2025H2"}
SEQ_LEN = 18          # sequence window used for distances and clustering
B = 1000              # bootstrap replicates
BOOT_SEED = 20261105  # seed of the shared Park-Miller generator (same in R)

results = []  # (name, value, rounding, headline, meaning)


def rec(name, value, digits, meaning, headline=False):
    """Record a quotable number, rounded to `digits` decimals."""
    # (+ 0.0 turns a rounded -0.0 into 0.0)
    results.append({"name": name, "value": round(float(value), digits) + 0.0, "rounding": digits,
                    "headline": headline, "meaning": meaning})


# ---------------------------------------------------------------------------
# Step 0. Load and clean
# ---------------------------------------------------------------------------
raw = pd.read_csv(DATA / "junior_careers.csv", keep_default_na=False, na_values={"tasks_closed": [""], "skill_score_m18": [""]})
truth = pd.read_csv(DATA / "ground_truth.csv")

# Map messy codes to the 4 canonical states. Anything empty = missing month.
CODE_MAP = {"R": "R", "ROUTINE": "R", "S": "S", "SUP": "S", "SUPERVISED": "S",
            "I": "I", "INDEP": "I", "INDEPENDENT": "I", "X": "X", "LEFT": "X"}
df = raw.sort_values(["person_id", "tenure_month"]).reset_index(drop=True)
code = df["state"].str.strip().str.upper()
n_recoded = int(((code != "") & (df["state"] != code)).sum())
df["state_clean"] = code.map(CODE_MAP)          # "" -> NaN (missing)
n_blank = int(df["state_clean"].isna().sum())

# Missing months: carry the previous month's state forward (a person rarely
# changes work level for one month and back). A missing first month -> "R",
# because everyone starts in routine work.
# X is absorbing: once someone has left, every later month is X (fixes coding
# errors where an active state appears after leaving).
clean, n_after_x = [], 0
for _, g in df.groupby("person_id", sort=True):
    prev, left = None, False
    for s in g["state_clean"]:
        if isinstance(s, float) or s is None:   # NaN
            s = prev if prev is not None else "R"
        if left and s != "X":
            n_after_x += 1
            s = "X"
        if s == "X":
            left = True
        clean.append(s)
        prev = s
df["state_clean"] = clean
df["period"] = np.where(df["cohort"].isin(POST_COHORTS), "post", "pre")

# One row per person: family, period, months observed (right truncation!).
people = (df.groupby("person_id", sort=True)
            .agg(job_family=("job_family", "first"), period=("period", "first"),
                 cohort=("cohort", "first"), n_obs=("tenure_month", "max"),
                 skill18=("skill_score_m18", "first"))
            .reset_index())
wide = df.pivot(index="person_id", columns="tenure_month", values="state_clean").sort_index()

rec("n_people", len(people), 0, "Synthetic entry-level hires in the dataset (6 half-year cohorts, 2 job families)")
rec("n_person_months", len(df), 0, "Person-month records in the panel")
rec("n_blank_months_imputed", n_blank, 0, "Months with a blank state, filled by carrying the previous state forward")
rec("n_recoded_rows", n_recoded, 0, "Rows with inconsistent state codes (e.g. 'SUP', 'Indep') mapped to R/S/I/X")
rec("n_active_after_left_fixed", n_after_x, 0, "Rows showing an active state after the person had left, recoded to X")
for f, p in GROUPS:
    rec(f"n_people_{f.lower()}_{p}", ((people.job_family == f) & (people.period == p)).sum(), 0,
        f"Hires in the {f} family, {p}-AI cohorts")


def gkey(f, p):
    return f"{f.lower()}_{p}"


# ---------------------------------------------------------------------------
# Step 1. NAIVE: early output by cohort period
# ---------------------------------------------------------------------------
# Average tasks closed per active month in tenure months 1-6 (person mean, then
# mean over people). This is what an "AI makes juniors productive" slide shows.
early = df[(df.tenure_month <= 6) & (df.state_clean != "X") & df.tasks_closed.notna()]
pm = early.groupby("person_id")["tasks_closed"].mean().rename("tasks16")
people = people.merge(pm, on="person_id", how="left")
naive = {}
for f, p in GROUPS:
    naive[(f, p)] = people.loc[(people.job_family == f) & (people.period == p), "tasks16"].mean()
    rec(f"naive_tasks_m1_6_{gkey(f, p)}", naive[(f, p)], 1,
        f"NAIVE: mean tasks closed per month in months 1-6, {f} family, {p}-AI cohorts", headline=True)
for f in ["HIGH", "LOW"]:
    rec(f"naive_tasks_pct_change_{f.lower()}", 100 * (naive[(f, "post")] / naive[(f, "pre")] - 1), 0,
        f"NAIVE: % change in early output, post- vs pre-AI cohorts, {f} family", headline=True)

# ---------------------------------------------------------------------------
# Step 2. CORRECT: tenure clock + sequences
# ---------------------------------------------------------------------------
# 2a. Chronogram: share of people in each state at each tenure month, among the
#     people OBSERVED at that tenure month (leavers stay in as X).
def chronogram(f, p):
    ids = people.loc[(people.job_family == f) & (people.period == p), "person_id"]
    sub = df[df.person_id.isin(ids)]
    tab = pd.crosstab(sub.tenure_month, sub.state_clean, normalize="index")
    return tab.reindex(columns=STATES, fill_value=0.0)


chrono = {g: chronogram(*g) for g in GROUPS}

# Person-level development measures
seq = wide.loc[people.person_id]
s12 = (seq.loc[:, 1:12] == "S").sum(axis=1).to_numpy()
st18 = seq[18].to_numpy() if 18 in seq.columns else np.array([None] * len(seq))
people["s_months_12"] = s12
people["in_I_18"] = (st18 == "I").astype(float)
people["in_I_12"] = (seq[12].to_numpy() == "I").astype(float)
people["obs12"] = people.n_obs >= 12
people["obs18"] = people.n_obs >= 18   # observed THROUGH month 18 (by calendar, not by state)

# 2b/2c point estimates
for f, p in GROUPS:
    m = (people.job_family == f) & (people.period == p)
    rec(f"s_months_first12_{gkey(f, p)}", people.loc[m & people.obs12, "s_months_12"].mean(), 2,
        f"Mean months in supervised complex work (S) in months 1-12, {f}, {p}-AI (people observed 12+ months, leavers included)", headline=True)
    # Month 12 too: a single snapshot month can tell the opposite story (see README)
    rec(f"share_I_m12_{gkey(f, p)}", 100 * people.loc[m & people.obs12, "in_I_12"].mean(), 0,
        f"% in independent complex work (I) at tenure month 12, {f}, {p}-AI (people observed 12+ months; leavers count as not-I)", headline=True)
    rec(f"n_obs12_{gkey(f, p)}", (m & people.obs12).sum(), 0, f"People observed through month 12, {f}, {p}-AI")
    rec(f"share_I_m18_{gkey(f, p)}", 100 * people.loc[m & people.obs18, "in_I_18"].mean(), 0,
        f"% in independent complex work (I) at tenure month 18, {f}, {p}-AI (people observed through month 18; leavers count as not-I)", headline=True)
    rec(f"n_obs18_{gkey(f, p)}", (m & people.obs18).sum(), 0, f"People observed through month 18, {f}, {p}-AI")
    rec(f"skill_m18_{gkey(f, p)}", people.loc[m & people.obs18, "skill18"].mean(), 1,
        f"Mean month-18 skill assessment score (0-100) among those assessed, {f}, {p}-AI")

# 2d. Sequence distances + clustering on the complete-observation subset
cl = people[people.obs18].reset_index(drop=True)            # sorted by person_id
S = wide.loc[cl.person_id, 1:SEQ_LEN].to_numpy()
n = len(cl)
# Hamming distance: number of months in which two people were in different states
D = np.zeros((n, n), dtype=np.int64)
for t in range(SEQ_LEN):
    D += (S[:, t][:, None] != S[:, t][None, :])


def pam(D, k):
    """k-medoids (PAM: BUILD + SWAP). Integer costs; ties -> lowest index.
    Deterministic, and implemented line-for-line the same way in R."""
    n = D.shape[0]
    med = [int(np.argmin(D.sum(axis=1)))]                       # BUILD, first medoid
    while len(med) < k:
        dn = D[:, med].min(axis=1)
        gain = np.maximum(dn[None, :] - D, 0).sum(axis=1)       # gain[c]
        gain[med] = -1
        med.append(int(np.argmax(gain)))
    cost = D[:, med].min(axis=1).sum()
    while True:                                                 # SWAP
        best_cost, best_i, best_h = cost, -1, -1
        for i in range(k):
            others = [m for j, m in enumerate(med) if j != i]
            d_other = D[:, others].min(axis=1)
            c = np.minimum(D, d_other[None, :]).sum(axis=1)      # cost if medoid i -> h
            c[med] = np.iinfo(np.int64).max
            h = int(np.argmin(c))
            if c[h] < best_cost:
                best_cost, best_i, best_h = c[h], i, h
        if best_i < 0:
            break
        med[best_i] = best_h
        cost = best_cost
    med = sorted(med)
    assign = np.argmin(D[:, med], axis=1)                       # ties -> first medoid
    return med, assign, int(cost)


med, assign, pam_cost = pam(D, 3)
# Name the clusters from their medoid (the most typical sequence):
#   most X months -> early_leaver; of the other two, more I months -> climber.
med_seq = S[med]
x_cnt = (med_seq == "X").sum(axis=1)
i_cnt = (med_seq == "I").sum(axis=1)
leaver = int(np.argmax(x_cnt))
rest = [j for j in range(3) if j != leaver]
climber = rest[0] if i_cnt[rest[0]] > i_cnt[rest[1]] else rest[1]
plateau = rest[1] if climber == rest[0] else rest[0]
names = {leaver: "early_leaver", climber: "climber", plateau: "plateau"}
cl["cluster"] = [names[a] for a in assign]
rec("pam_total_distance", pam_cost, 0, "Sum of Hamming distances from each person to their cluster medoid (identical in Python and R)")
rec("n_clustered", n, 0, "People clustered (observed through tenure month 18)")
for j, nm in names.items():
    rec(f"medoid_{nm}_I_months", i_cnt[j], 0, f"Months in I (of 18) in the medoid sequence of the '{nm}' cluster")
    rec(f"medoid_{nm}_S_months", (med_seq[j] == "S").sum(), 0, f"Months in S (of 18) in the medoid sequence of the '{nm}' cluster")

TYPE_NAMES = ["climber", "plateau", "early_leaver"]
for f, p in GROUPS:
    m = (cl.job_family == f) & (cl.period == p)
    for t in TYPE_NAMES:
        rec(f"share_{t}_{gkey(f, p)}", 100 * (cl.loc[m, "cluster"] == t).mean(), 0,
            f"% of people in the recovered '{t}' trajectory type, {f}, {p}-AI", headline=True)


# 2e. Bootstrap (persons resampled within each family x period group; cluster
#     labels held fixed). Uses a tiny Park-Miller generator so Python and R draw
#     exactly the same resamples.
class ParkMiller:
    M = 2147483647

    def __init__(self, seed):
        self.state = seed

    def draw(self, n_pick, n_pool):
        out = np.empty(n_pick, dtype=np.int64)
        s = self.state
        for k in range(n_pick):
            s = (16807 * s) % self.M
            out[k] = (s * n_pool) // self.M        # 0-based index in [0, n_pool)
        self.state = s
        return out


def group_arrays(f, p):
    a = cl[(cl.job_family == f) & (cl.period == p)]
    b = people[(people.job_family == f) & (people.period == p) & people.obs12]
    return {"plateau": (a.cluster == "plateau").to_numpy(float), "I18": a.in_I_18.to_numpy(float),
            "S12": b.s_months_12.to_numpy(float), "I12": b.in_I_12.to_numpy(float)}


arr = {g: group_arrays(*g) for g in GROUPS}


def stats_from(est):
    """est[(f,p)] = dict of group means -> the bootstrapped differences."""
    out = {}
    for f in ["HIGH", "LOW"]:
        fl = f.lower()
        out[f"diff_plateau_{fl}"] = 100 * (est[(f, "post")]["plateau"] - est[(f, "pre")]["plateau"])
        out[f"diff_share_I_m18_{fl}"] = 100 * (est[(f, "post")]["I18"] - est[(f, "pre")]["I18"])
        out[f"diff_s_months_first12_{fl}"] = est[(f, "post")]["S12"] - est[(f, "pre")]["S12"]
        out[f"diff_share_I_m12_{fl}"] = 100 * (est[(f, "post")]["I12"] - est[(f, "pre")]["I12"])
    out["contrast_plateau"] = out["diff_plateau_high"] - out["diff_plateau_low"]
    out["contrast_share_I_m18"] = out["diff_share_I_m18_high"] - out["diff_share_I_m18_low"]
    out["contrast_s_months_first12"] = out["diff_s_months_first12_high"] - out["diff_s_months_first12_low"]
    out["contrast_share_I_m12"] = out["diff_share_I_m12_high"] - out["diff_share_I_m12_low"]
    return out


point = stats_from({g: {k: v.mean() for k, v in arr[g].items()} for g in GROUPS})
rng = ParkMiller(BOOT_SEED)
boot = {k: np.empty(B) for k in point}
for b in range(B):
    est = {}
    for g in GROUPS:
        ia = rng.draw(len(arr[g]["plateau"]), len(arr[g]["plateau"]))
        ib = rng.draw(len(arr[g]["S12"]), len(arr[g]["S12"]))
        est[g] = {"plateau": arr[g]["plateau"][ia].mean(), "I18": arr[g]["I18"][ia].mean(),
                  "S12": arr[g]["S12"][ib].mean(), "I12": arr[g]["I12"][ib].mean()}
    for k, v in stats_from(est).items():
        boot[k][b] = v

MEANING = {
    "diff_plateau": "Change in % plateau trajectory type, post- minus pre-AI cohorts (percentage points)",
    "diff_share_I_m18": "Change in % in independent work at month 18, post- minus pre-AI (percentage points)",
    "diff_share_I_m12": "Change in % in independent work at month 12, post- minus pre-AI (percentage points)",
    "diff_s_months_first12": "Change in mean months in supervised complex work (months 1-12), post- minus pre-AI",
    "contrast": "HIGH-family change minus LOW-family change (descriptive contrast, not a causal effect)",
}
for k, v in point.items():
    digits = 2 if "s_months" in k else 0
    base = "contrast" if k.startswith("contrast") else k.rsplit("_", 1)[0]
    fam = "" if k.startswith("contrast") else f", {k.rsplit('_', 1)[1].upper()} family"
    what = MEANING[base] + (f" [{k.replace('contrast_', '')}]" if k.startswith("contrast") else fam)
    lo, hi = np.quantile(boot[k], [0.025, 0.975])
    rec(k, v, digits, what, headline=True)
    rec(k + "_ci_low", lo, digits, "Bootstrap 95% interval, lower bound: " + k, headline=True)
    rec(k + "_ci_high", hi, digits, "Bootstrap 95% interval, upper bound: " + k, headline=True)

# ---------------------------------------------------------------------------
# Step 3. Check against ground truth
# ---------------------------------------------------------------------------
cl = cl.merge(truth, on="person_id", how="left")
ari = adjusted_rand_score(cl.true_type, cl.cluster)
rec("ari_recovered_vs_true", ari, 2, "Adjusted Rand index, recovered vs true trajectory type (1 = perfect, 0 = chance)", headline=True)
rec("pct_correct_type", 100 * (cl.true_type == cl.cluster).mean(), 0,
    "% of clustered people whose recovered type equals their true type", headline=True)
for f in ["HIGH", "LOW"]:
    tp = {p: 100 * (cl.loc[(cl.job_family == f) & (cl.period == p), "true_type"] == "plateau").mean() for p in ["pre", "post"]}
    rec(f"true_plateau_shift_{f.lower()}", tp["post"] - tp["pre"], 0,
        f"TRUE change in % plateau type (same clustered people), {f} family, percentage points", headline=True)
rec("design_plateau_shift_high", 20, 0, "Plateau-share shift built into the data generator, HIGH family (20 -> 40%)")
rec("design_plateau_shift_low", 2, 0, "Plateau-share shift built into the data generator, LOW family (22 -> 24%)")

# ---------------------------------------------------------------------------
# Step 4. Decision implication (printed)
# ---------------------------------------------------------------------------
R = {r["name"]: r["value"] for r in results}
print("\n=== F01 The Broken Bottom Rung (synthetic data) ===")
print(f"Cleaning: {n_blank} blank months imputed, {n_recoded} codes recoded, {n_after_x} post-exit rows fixed")
print("\nStep 1 NAIVE - tasks/month, months 1-6:")
for f in ["HIGH", "LOW"]:
    fl = f.lower()
    print(f"  {f}: pre {R[f'naive_tasks_m1_6_{fl}_pre']}  post {R[f'naive_tasks_m1_6_{fl}_post']}"
          f"  ({R[f'naive_tasks_pct_change_{fl}']:+.0f}%)")
print("\nStep 2 CORRECT - tenure clock and sequences:")
for f in ["HIGH", "LOW"]:
    fl = f.lower()
    print(f"  {f}: S months (1-12) {R[f's_months_first12_{fl}_pre']} -> {R[f's_months_first12_{fl}_post']};"
          f" % in I at m18 {R[f'share_I_m18_{fl}_pre']:.0f} -> {R[f'share_I_m18_{fl}_post']:.0f};"
          f" % plateau {R[f'share_plateau_{fl}_pre']:.0f} -> {R[f'share_plateau_{fl}_post']:.0f}"
          f" (diff {R[f'diff_plateau_{fl}']:+.0f}, 95% CI {R[f'diff_plateau_{fl}_ci_low']:.0f} to {R[f'diff_plateau_{fl}_ci_high']:.0f})")
print(f"  Contrast (HIGH shift - LOW shift) in plateau share: {R['contrast_plateau']:+.0f} pts "
      f"[{R['contrast_plateau_ci_low']:.0f}, {R['contrast_plateau_ci_high']:.0f}]")
print(f"\nStep 3 CHECK: ARI {R['ari_recovered_vs_true']:.2f}, {R['pct_correct_type']:.0f}% correctly typed;"
      f" true plateau shift HIGH {R['true_plateau_shift_high']:+.0f} vs estimated {R['diff_plateau_high']:+.0f}")
print("\nStep 4 DECISION: early output says 'ramp-up solved'; the sequences say fewer juniors in the")
print("  high-AI family get stretch work and reach independent work. Track time-in-S and share-in-I by")
print("  tenure month for every cohort, quarterly, and protect learning-rich work in junior roles.")
print("  (Descriptive cohort comparison, not a causal estimate.)")

pd.DataFrame(results).to_csv(OUT / "results_python.csv", index=False)

# ---------------------------------------------------------------------------
# Figures (Python only; R reproduces the numbers)
# ---------------------------------------------------------------------------
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
PRE, POSTC = "#a3a29c", "#2a78d6"
STATE_COL = {"R": "#b7d3f6", "S": "#5598e7", "I": "#104281", "X": "#d6d5cf"}
STATE_LAB = {"R": "Routine (R)", "S": "Supervised complex (S)", "I": "Independent (I)", "X": "Left (X)"}
plt.rcParams.update({"font.size": 15, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "font.family": "DejaVu Sans"})

# Figure 1: early output up, independent work at month 18 down (HIGH), flat (LOW)
fig, axes = plt.subplots(1, 2, figsize=(12, 7), dpi=150)
panels = [("Early output\ntasks closed per month, months 1-6", "naive_tasks_m1_6", "{:.0f}"),
          ("Development\n% in independent work at month 18", "share_I_m18", "{:.0f}%")]
x = np.arange(2)
w = 0.36
for ax, (ttl, key, fmt) in zip(axes, panels):
    vpre = [R[f"{key}_{f}_pre"] for f in ["high", "low"]]
    vpost = [R[f"{key}_{f}_post"] for f in ["high", "low"]]
    b1 = ax.bar(x - w / 2 - 0.01, vpre, w, color=PRE, label="Pre-AI cohorts (2023H1-2024H1)")
    b2 = ax.bar(x + w / 2 + 0.01, vpost, w, color=POSTC, label="Post-AI cohorts (2024H2-2025H2)")
    for bars in (b1, b2):
        for r_ in bars:
            ax.text(r_.get_x() + r_.get_width() / 2, r_.get_height() + max(vpre + vpost) * 0.015,
                    fmt.format(r_.get_height()), ha="center", va="bottom", fontsize=16, color=INK, fontweight="bold")
    ax.set_xticks(x, ["High AI exposure", "Low AI exposure"], fontsize=15, color=INK)
    ax.set_title(ttl, fontsize=16, color=INK, loc="left")
    ax.set_ylim(0, max(vpre + vpost) * 1.18)
    ax.yaxis.grid(True, color=GRID, lw=1)
    ax.set_axisbelow(True)
    for s in ["top", "right", "left"]:
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", length=0)
hp, lp = R["naive_tasks_pct_change_high"], R["diff_share_I_m18_high"]
fig.suptitle(f"In high-AI roles, new hires close {hp:.0f}% more tasks early,\n"
             f"yet {abs(lp):.0f} points fewer reach independent work by month 18",
             x=0.02, ha="left", fontsize=21, fontweight="bold", color=INK, y=0.99)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=14, bbox_to_anchor=(0.5, 0.075))
n18 = {k: int(R[f"n_obs18_{k}"]) for k in ["high_pre", "high_post", "low_pre", "low_post"]}
fig.text(0.02, 0.012, "SYNTHETIC DATA - generated to illustrate the method; not from any real organization.\n"
         f"Month-18 share: hires observed through month 18 (pre n = {n18['high_pre']} / post n = {n18['high_post']} high; "
         f"{n18['low_pre']} / {n18['low_post']} low).", fontsize=12, color=INK2)
fig.subplots_adjust(top=0.78, bottom=0.22, wspace=0.25, left=0.05, right=0.98)
fig.savefig(OUT / "fig1_output_up_development_down.png", facecolor="white")
plt.close(fig)

# Figure 2: chronogram, HIGH family, pre vs post, on the tenure clock
fig, axes = plt.subplots(1, 2, figsize=(12, 7.5), dpi=150, sharey=True)
for ax, p, lab in zip(axes, ["pre", "post"], ["Pre-AI cohorts", "Post-AI cohorts"]):
    tab = chrono[("HIGH", p)]
    tab = tab[tab.index <= SEQ_LEN]   # months 1-18: the window used throughout
    t = tab.index.to_numpy()
    ax.stackplot(t, [tab[s].to_numpy() * 100 for s in STATES], colors=[STATE_COL[s] for s in STATES],
                 edgecolor="white", linewidth=1.5)
    i18 = R[f"share_I_m18_high_{p}"]
    ax.set_title(f"{lab}: {i18:.0f}% independent at month 18", fontsize=16, color=INK, loc="left")
    ax.set_xlim(1, t.max())
    ax.set_ylim(0, 100)
    ax.set_xticks([1, 6, 12, 18])
    ax.set_xlabel("Months since hire (tenure clock)", fontsize=14)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
axes[0].set_ylabel("% of hires observed at that month", fontsize=14)
handles = [plt.Rectangle((0, 0), 1, 1, color=STATE_COL[s]) for s in STATES]
fig.legend(handles, [STATE_LAB[s] for s in STATES], loc="lower center", ncol=4, frameon=False, fontsize=13,
           bbox_to_anchor=(0.5, 0.055))
fig.suptitle("High-AI roles: more post-AI hires stay stuck in routine\n"
             "work, and fewer reach independent work by month 18",
             x=0.02, ha="left", fontsize=21, fontweight="bold", color=INK, y=0.99)
fig.text(0.02, 0.012, "SYNTHETIC DATA - generated to illustrate the method; not from any real organization.\n"
         "Each month counts the hires observed at that tenure month; leavers stay in as 'Left'.", fontsize=12, color=INK2)
fig.subplots_adjust(top=0.8, bottom=0.22, wspace=0.08, left=0.08, right=0.98)
fig.savefig(OUT / "fig2_state_distribution_tenure_clock_high.png", facecolor="white")
plt.close(fig)

# Merge with R results (if present) into outputs/results.json
sys.path.insert(0, str(HERE))
from check_agreement import merge  # noqa: E402

merge(write=True, verbose=False)
print(f"\nWrote {OUT / 'results_python.csv'}, figures, and {OUT / 'results.json'}")
