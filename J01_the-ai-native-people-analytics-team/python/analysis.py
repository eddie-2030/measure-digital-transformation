"""
analysis.py -- Can we trust AI-generated answers about our workforce?
=====================================================================
Edition J01, "The AI-Native People Analytics Team" (Measure Digital Transformation).

We evaluate a people-analytics RAG assistant the way we would evaluate any
measurement instrument:

  Step 1  NAIVE: report thumbs-up rate or raw LLM-judge "accuracy" as one number.
  Step 2  CORRECT: (a) retrieval quality (recall@5, hit rate, MRR),
                   (b) groundedness (share of claims supported by retrieved text),
                   (c) validate the LLM judge on a random human-graded subsample
                       (sensitivity, specificity),
                   (d) correct the judge's accuracy for its errors (Rogan-Gladen),
                       with a bootstrap CI that resamples QUESTIONS (clusters),
                   (e) decompose failures: retrieval miss vs. generation.
  Step 3  GROUND TRUTH: compare estimates with the true values from the generator.
  Step 4  DECISION: what to deploy, route, or hold, by question type.

All records are simulated (no real LLM calls). Run from the package root:
    python data/generate_data.py
    python python/analysis.py
Outputs: outputs/results.json, outputs/python_results.csv, outputs/fig1_*.png, outputs/fig2_*.png
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA, OUT = ROOT / "data", ROOT / "outputs"
OUT.mkdir(exist_ok=True)

TYPES = ["policy_lookup", "computed_metric", "multi_hop"]
LABEL = {"policy_lookup": "Policy lookup", "computed_metric": "Computed metric",
         "multi_hop": "Multi-hop", "overall": "All questions"}
K = 5                      # retrieval depth we evaluate (top-5 documents)
GROUNDED_THRESHOLD = 0.8   # an answer counts as "grounded" if >= 80% of its claims are supported
MIN_PER_CELL = 30          # need >= 30 truly-correct AND >= 30 truly-wrong human grades in a
                           # question type before estimating that type's own sens/spec
B = 2000                   # bootstrap replicates
BOOT_SEED = 2026
DEPLOY_AT, ROUTE_AT = 0.80, 0.60   # decision thresholds on corrected accuracy

results = {}               # name -> dict(value, rounding, meaning)


def put(name, value, meaning, digits=3):
    """Store a number for results.json (rounded to `digits` decimals)."""
    v = None if value is None or (isinstance(value, float) and np.isnan(value)) else round(float(value), digits)
    results[name] = {"value": v, "rounding": f"{digits} decimals", "meaning": meaning}
    return v


def rogan_gladen(judge_rate, sens, spec):
    """Corrected accuracy = (judge rate + spec - 1) / (sens + spec - 1), clipped to [0, 1].
    If the judge is no better than chance (sens + spec <= 1) the correction is undefined."""
    denom = sens + spec - 1
    if denom <= 0:
        return np.nan
    return float(np.clip((judge_rate + spec - 1) / denom, 0, 1))


# ------------------------------------------------------------------ load & clean
df = pd.read_csv(DATA / "rag_eval_records.csv", keep_default_na=True)
truth = pd.read_csv(DATA / "true_parameters.csv").set_index("name")["value"]


def parse_ids(s):
    """'D012; d045 ' -> ['D012', 'D045'].  Logged IDs are sometimes lower-case or padded."""
    if pd.isna(s) or str(s).strip() == "":
        return []
    return [x.strip().upper() for x in str(s).split(";") if x.strip()]


df["gold"] = df.gold_doc_ids.apply(parse_ids)
df["retr"] = df.retrieved_doc_ids.apply(lambda s: parse_ids(s)[:K])


def recall_at_k(gold, retr):
    """Share of the truly relevant documents that appear in the top-k list."""
    return np.nan if not gold else sum(g in retr for g in gold) / len(gold)


def reciprocal_rank(gold, retr):
    """1 / rank of the FIRST relevant document in the list (0 if none is retrieved)."""
    if not gold:
        return np.nan
    for i, d in enumerate(retr):
        if d in gold:
            return 1.0 / (i + 1)
    return 0.0


df["recall5"] = [recall_at_k(g, r) for g, r in zip(df.gold, df.retr)]
df["full_hit"] = (df.recall5 == 1).astype(float).where(df.answerable == 1)
df["rr"] = [reciprocal_rank(g, r) for g, r in zip(df.gold, df.retr)]
# groundedness of an answer = supported claims / claims (undefined for "I don't know")
df["groundedness"] = (df.n_claims_supported / df.n_claims).where(df.n_claims > 0)

print(f"Loaded {len(df)} questions in {df.cluster_id.nunique()} clusters "
      f"({(df.duplicate_of.notna()).sum()} near-duplicates); "
      f"{(df.answerable == 0).sum()} have no source in the corpus.")

# ================================================================ STEP 1: NAIVE
# The dashboard most teams build: thumbs-up rate and the judge's raw "accuracy".
rated = df.dropna(subset=["thumbs_up"])
judged = df.dropna(subset=["judge_label"])
put("naive_thumbs_up_rate", rated.thumbs_up.mean(),
    "Share of rated answers with a thumbs-up (satisfaction, not correctness)")
put("naive_n_rated", len(rated), "Number of answers users rated", 0)
put("naive_judge_accuracy", judged.judge_label.mean(),
    "Raw LLM-judge 'accuracy': share of judged answers the judge called correct")
put("naive_n_judged", len(judged), "Answers with a judge verdict (others timed out)", 0)
for t in TYPES:
    put(f"naive_judge_accuracy_{t}", judged.loc[judged.question_type == t, "judge_label"].mean(),
        f"Raw judge accuracy, {t}")
    put(f"naive_thumbs_up_rate_{t}", rated.loc[rated.question_type == t, "thumbs_up"].mean(),
        f"Thumbs-up rate, {t}")
# Why it misleads: thumbs-up barely depends on correctness ...
h_r = df[(df.human_graded == 1) & df.thumbs_up.notna()]
put("thumbs_up_rate_when_correct_human", h_r.loc[h_r.human_label == 1, "thumbs_up"].mean(),
    "Thumbs-up rate among human-graded CORRECT answers")
put("thumbs_up_rate_when_wrong_human", h_r.loc[h_r.human_label == 0, "thumbs_up"].mean(),
    "Thumbs-up rate among human-graded WRONG answers (users like fluent wrong answers too)")
print(f"\nSTEP 1 (naive): thumbs-up {results['naive_thumbs_up_rate']['value']:.3f}, "
      f"raw judge accuracy {results['naive_judge_accuracy']['value']:.3f}")

# ======================================================= STEP 2a: RETRIEVAL
ans = df[df.answerable == 1]
for t in TYPES + ["overall"]:
    s = ans if t == "overall" else ans[ans.question_type == t]
    put(f"recall_at5_{t}", s.recall5.mean(), f"Mean recall@5 (share of gold docs retrieved), {t}")
    put(f"hit_rate_{t}", s.full_hit.mean(), f"Share of questions with ALL gold docs in top 5, {t}")
    put(f"mrr_{t}", s.rr.mean(), f"Mean reciprocal rank of first gold doc, {t}")

# ======================================================= STEP 2b: GROUNDEDNESS
answered = df[df.n_claims > 0]
for t in TYPES + ["overall"]:
    s = answered if t == "overall" else answered[answered.question_type == t]
    put(f"groundedness_{t}", s.groundedness.mean(), f"Mean share of answer claims supported by retrieved text, {t}")
    put(f"share_grounded_{t}", (s.groundedness >= GROUNDED_THRESHOLD).mean(),
        f"Share of answers with >= 80% of claims supported, {t}")
abst = df[(df.answerable == 0)]
put("abstain_rate_unanswerable", abst.abstained.mean(),
    "Share of no-source questions where the assistant correctly said 'I don't know'")
put("n_unanswerable", len(abst), "Questions with no relevant document in the corpus", 0)

# ======================================================= STEP 2c: VALIDATE THE JUDGE
# Human experts graded a RANDOM sample of questions. Treat their grade as the gold
# standard and ask: how often does the judge agree with the truth in each direction?
hv = df[(df.human_graded == 1) & df.judge_label.notna()]


def sens_spec(d):
    pos, neg = d[d.human_label == 1], d[d.human_label == 0]
    sens = (pos.judge_label == 1).mean() if len(pos) else np.nan   # P(judge correct | truly correct)
    spec = (neg.judge_label == 0).mean() if len(neg) else np.nan   # P(judge wrong | truly wrong)
    return sens, spec, len(pos), len(neg)


SENS, SPEC, n_pos, n_neg = sens_spec(hv)
put("n_human_graded", int((df.human_graded == 1).sum()), "Questions in the random human-graded subsample", 0)
put("n_human_graded_with_judge", len(hv), "Human-graded questions that also have a judge verdict", 0)
put("judge_sensitivity", SENS, "P(judge says correct | human says correct)")
put("judge_specificity", SPEC, "P(judge says wrong | human says wrong); low = lenient judge")
put("n_human_correct", n_pos, "Human-graded answers that were truly correct", 0)
put("n_human_wrong", n_neg, "Human-graded answers that were truly wrong", 0)
per_type_ok = {}
for t in TYPES:
    se, sp, npos, nneg = sens_spec(hv[hv.question_type == t])
    per_type_ok[t] = (npos >= MIN_PER_CELL) and (nneg >= MIN_PER_CELL)
    put(f"n_human_correct_{t}", npos, f"Human-graded truly correct, {t}", 0)
    put(f"n_human_wrong_{t}", nneg, f"Human-graded truly wrong, {t}", 0)
print(f"\nSTEP 2c: judge sensitivity {SENS:.3f}, specificity {SPEC:.3f} "
      f"(n = {n_pos} correct, {n_neg} wrong human grades)")
print("  Per-type sens/spec usable (>= 30 per cell)?", per_type_ok,
      "-> using POOLED sens/spec for every type (assumes judge error does not vary by type)")

# Human-only estimate: the 150 human grades alone also estimate accuracy.
h_all = df[df.human_graded == 1]
p_h = h_all.human_label.mean()
se_h = np.sqrt(p_h * (1 - p_h) / len(h_all))
put("human_only_accuracy", p_h, "Accuracy in the random human-graded subsample alone")
put("human_only_ci_low", p_h - 1.96 * se_h, "95% Wald CI lower bound, human-only accuracy")
put("human_only_ci_high", p_h + 1.96 * se_h, "95% Wald CI upper bound, human-only accuracy")

# ======================================================= STEP 2d: CORRECT THE JUDGE
corr = {}
corr["overall"] = rogan_gladen(judged.judge_label.mean(), SENS, SPEC)
for t in TYPES:
    corr[t] = rogan_gladen(judged.loc[judged.question_type == t, "judge_label"].mean(), SENS, SPEC)
for t in TYPES + ["overall"]:
    put(f"corrected_accuracy_{t}", corr[t],
        f"Rogan-Gladen corrected accuracy, {t} (pooled sens/spec)")

# Cluster bootstrap. Near-duplicate questions share a cluster_id, so we resample
# CLUSTERS (questions), not rows. Each replicate re-estimates the judge rate AND the
# judge's sens/spec (because the human subsample is resampled too), so the CI
# carries both sources of uncertainty. Trick for speed: pre-compute per-cluster counts
# and multiply by how many times each cluster is drawn.
clusters = df.cluster_id.unique()
cidx = pd.Index(clusters)
d = df.assign(c=cidx.get_indexer(df.cluster_id))
j = d.judge_label.notna()
cols = {"J_n": j, "J_1": j & (d.judge_label == 1)}
for t in TYPES:
    cols[f"J_n_{t}"] = j & (d.question_type == t)
    cols[f"J_1_{t}"] = j & (d.question_type == t) & (d.judge_label == 1)
hj = j & (d.human_graded == 1)
cols["H_pos"] = hj & (d.human_label == 1)
cols["H_pos_j1"] = hj & (d.human_label == 1) & (d.judge_label == 1)
cols["H_neg"] = hj & (d.human_label == 0)
cols["H_neg_j0"] = hj & (d.human_label == 0) & (d.judge_label == 0)
# all human-graded questions (with or without a judge verdict), for a human-only interval
cols["Hall_n"] = d.human_graded == 1
cols["Hall_1"] = (d.human_graded == 1) & (d.human_label == 1)
C = pd.DataFrame({k: v.fillna(False).astype(int) for k, v in cols.items()}).groupby(d.c).sum()
C = C.reindex(range(len(clusters)), fill_value=0)

rng = np.random.default_rng(BOOT_SEED)
boot = {t: np.empty(B) for t in TYPES + ["overall"]}
boot_h = np.empty(B)   # human-only accuracy, same resamples
for b in range(B):
    w = np.bincount(rng.integers(0, len(clusters), len(clusters)), minlength=len(clusters))
    S = w @ C.to_numpy()
    s = dict(zip(C.columns, S))
    se_b, sp_b = s["H_pos_j1"] / s["H_pos"], s["H_neg_j0"] / s["H_neg"]
    boot["overall"][b] = rogan_gladen(s["J_1"] / s["J_n"], se_b, sp_b)
    boot_h[b] = s["Hall_1"] / s["Hall_n"]
    for t in TYPES:
        boot[t][b] = rogan_gladen(s[f"J_1_{t}"] / s[f"J_n_{t}"], se_b, sp_b)
for t in TYPES + ["overall"]:
    lo, hi = np.nanpercentile(boot[t], [2.5, 97.5])
    put(f"corrected_ci_low_{t}", lo, f"95% cluster-bootstrap CI lower bound, corrected accuracy, {t} (B=2000; RNG-dependent)", 2)
    put(f"corrected_ci_high_{t}", hi, f"95% cluster-bootstrap CI upper bound, corrected accuracy, {t} (B=2000; RNG-dependent)", 2)
# Like-for-like comparison: the human-only accuracy with the SAME clustered bootstrap.
lo_h, hi_h = np.nanpercentile(boot_h, [2.5, 97.5])
put("human_only_boot_ci_low", lo_h, "95% cluster-bootstrap CI lower bound, human-only accuracy (B=2000; RNG-dependent)", 2)
put("human_only_boot_ci_high", hi_h, "95% cluster-bootstrap CI upper bound, human-only accuracy (B=2000; RNG-dependent)", 2)
print(f"STEP 2d: corrected accuracy {corr['overall']:.3f} "
      f"[{results['corrected_ci_low_overall']['value']:.2f}, {results['corrected_ci_high_overall']['value']:.2f}]")
for t in TYPES:
    print(f"   {t:16s} raw judge {results[f'naive_judge_accuracy_{t}']['value']:.3f} -> corrected {corr[t]:.3f}")

# ======================================================= STEP 2e: FAILURE DECOMPOSITION
# For each WRONG answer, use the observable trace to say WHERE it broke:
#   no_source         : nothing relevant exists, yet the assistant answered (should abstain)
#   retrieval_miss    : at least one gold document was not in the top 5
#   ungrounded_gen    : all gold docs retrieved, but < 80% of claims supported by them
#   grounded_but_wrong: all gold docs retrieved and claims supported, still wrong
#                       (e.g., arithmetic or reasoning error)
# Correctness comes from the HUMAN grades (the judge is too lenient to trust here).
def cause(r):
    if r.answerable == 0:
        return "no_source"
    if r.recall5 < 1:
        return "retrieval_miss"
    if r.abstained == 1 or r.groundedness < GROUNDED_THRESHOLD:
        return "ungrounded_gen"
    return "grounded_but_wrong"


CAUSES = ["retrieval_miss", "ungrounded_gen", "grounded_but_wrong", "no_source"]
df["cause"] = df.apply(cause, axis=1)
wrong_h = df[(df.human_graded == 1) & (df.human_label == 0)]
put("n_wrong_human", len(wrong_h), "Wrong answers in the human-graded subsample", 0)
decomp = {}
for t in TYPES + ["overall"]:
    s = wrong_h if t == "overall" else wrong_h[wrong_h.question_type == t]
    put(f"n_wrong_human_{t}", len(s), f"Wrong human-graded answers, {t}", 0)
    # NOTE: per-type counts are small (single digits to ~20); read shares as indicative.
    decomp[t] = {}
    for c in CAUSES:
        decomp[t][c] = (s.cause == c).mean() if len(s) else np.nan
        put(f"fail_share_{c}_{t}", decomp[t][c], f"Share of wrong answers caused by {c}, {t} (human-graded)")
# Observable on ALL questions without any grading: how often retrieval missed.
for t in TYPES:
    s = ans[ans.question_type == t]
    put(f"retrieval_miss_rate_{t}", 1 - s.full_hit.mean(), f"Share of answerable questions with a retrieval miss, {t}")

# ======================================================= STEP 3: GROUND TRUTH
# Only possible with synthetic data: the generator kept the true correctness.
put("true_accuracy_sample_overall", df.true_correct.mean(), "TRUE accuracy of the 600 test questions (oracle)")
for t in TYPES:
    put(f"true_accuracy_sample_{t}", df.loc[df.question_type == t, "true_correct"].mean(),
        f"TRUE accuracy in the test set, {t} (oracle)")
for k in ["true_accuracy_overall", "true_accuracy_policy_lookup", "true_accuracy_computed_metric",
          "true_accuracy_multi_hop", "true_recall_at5_overall", "true_groundedness_overall",
          "true_recall_at5_multi_hop", "true_groundedness_multi_hop",
          "judge_sensitivity", "judge_specificity"]:
    put(f"population_{k}", truth[k], f"Generator's expected value (100k-question simulation): {k}")
wrong_all = df[df.true_correct == 0]
for c in CAUSES:
    put(f"true_fail_share_{c}_overall", (wrong_all.cause == c).mean(),
        f"Share of ALL truly wrong answers caused by {c} (oracle, 600 questions)")
    put(f"true_fail_share_{c}_multi_hop", (wrong_all[wrong_all.question_type == 'multi_hop'].cause == c).mean(),
        f"Share of truly wrong multi-hop answers caused by {c} (oracle)")
cov = results["corrected_ci_low_overall"]["value"] <= df.true_correct.mean() <= results["corrected_ci_high_overall"]["value"]
results["ci_covers_true_overall"] = {"value": bool(cov), "rounding": "boolean",
                                     "meaning": "Does the corrected-accuracy CI contain the true test-set accuracy?"}
for t in TYPES:
    tv = df.loc[df.question_type == t, "true_correct"].mean()
    results[f"ci_covers_true_{t}"] = {
        "value": bool(results[f"corrected_ci_low_{t}"]["value"] <= tv <= results[f"corrected_ci_high_{t}"]["value"]),
        "rounding": "boolean", "meaning": f"Does the corrected CI contain the true test-set accuracy, {t}?"}
    put(f"corrected_error_pp_{t}", 100 * (corr[t] - tv), f"Corrected minus true accuracy, {t}, percentage points", 1)
put("naive_judge_error_pp", 100 * (judged.judge_label.mean() - df.true_correct.mean()),
    "Raw judge accuracy minus true accuracy, percentage points", 1)
put("corrected_error_pp", 100 * (corr["overall"] - df.true_correct.mean()),
    "Corrected accuracy minus true accuracy, percentage points", 1)
print(f"\nSTEP 3: true accuracy {df.true_correct.mean():.3f}; raw judge off by "
      f"{results['naive_judge_error_pp']['value']:+.1f} pp; corrected off by {results['corrected_error_pp']['value']:+.1f} pp; "
      f"CI covers truth: {cov}")

# ======================================================= STEP 4: DECISION
decision = {}
for t in TYPES:
    a = corr[t]
    decision[t] = ("deploy with citations shown" if a >= DEPLOY_AT else
                   "route through governed metric definitions / analytics team" if a >= ROUTE_AT else
                   "hold until retrieval improves")
    results[f"decision_{t}"] = {"value": decision[t], "rounding": "text",
                                "meaning": f"Rule: corrected >= {DEPLOY_AT} deploy; >= {ROUTE_AT} route; else hold"}
print("\nSTEP 4: decisions by question type")
for t in TYPES:
    print(f"   {t:16s} corrected {corr[t]:.3f} -> {decision[t]}")

# ======================================================= FIGURES
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
GRAY_L, GRAY_D, BLUE = "#c3c2b7", "#8a8983", "#2a78d6"
SERIES = {"retrieval_miss": "#2a78d6", "ungrounded_gen": "#eb6834",
          "grounded_but_wrong": "#1baf7a", "no_source": "#eda100"}
plt.rcParams.update({"font.size": 15, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK, "font.family": "DejaVu Sans"})

# Figure 1: naive vs corrected vs true accuracy, by question type
groups = ["overall"] + TYPES
y = np.arange(len(groups))[::-1] * 1.0
fig, ax = plt.subplots(figsize=(8, 7.2), dpi=200)
h = 0.24
thumbs = [results["naive_thumbs_up_rate"]["value"]] + [results[f"naive_thumbs_up_rate_{t}"]["value"] for t in TYPES]
rawj = [results["naive_judge_accuracy"]["value"]] + [results[f"naive_judge_accuracy_{t}"]["value"] for t in TYPES]
cor = [corr[g] for g in groups]
lo = [results[f"corrected_ci_low_{g}"]["value"] for g in groups]
hi = [results[f"corrected_ci_high_{g}"]["value"] for g in groups]
tru = [df.true_correct.mean()] + [df.loc[df.question_type == t, "true_correct"].mean() for t in TYPES]
ax.barh(y + h, thumbs, h * 0.92, color=GRAY_L, label="Thumbs-up rate (naive)")
ax.barh(y, rawj, h * 0.92, color=GRAY_D, label="Raw judge 'accuracy' (naive)")
ax.barh(y - h, cor, h * 0.92, color=BLUE, label="Corrected accuracy, 95% CI")
ax.errorbar(cor, y - h, xerr=[np.array(cor) - lo, np.array(hi) - cor], fmt="none",
            ecolor=INK, elinewidth=1.6, capsize=4)
ax.scatter(tru, y - h, marker="D", s=70, color="white", edgecolor=INK, linewidth=2,
           zorder=5, label="True accuracy (synthetic truth)")
for yy, v in zip(y, thumbs):
    ax.text(v + 0.01, yy + h, f"{v:.0%}", va="center", fontsize=13, color=INK2)
for yy, v in zip(y, rawj):
    ax.text(v + 0.01, yy, f"{v:.0%}", va="center", fontsize=13, color=INK2)
for yy, v, hv_ in zip(y, cor, hi):
    ax.text(hv_ + 0.015, yy - h, f"{v:.0%}", va="center", fontsize=13, color=INK, fontweight="bold")
ax.set_yticks(y)
ax.set_yticklabels([LABEL[g] for g in groups], fontsize=15)
ax.set_xlim(0, 1.08)
ax.set_xticks([0, .25, .5, .75, 1])
ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.grid(axis="x", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for sp_ in ["top", "right"]:
    ax.spines[sp_].set_visible(False)
ax.legend(loc="upper center", bbox_to_anchor=(0.45, -0.07), ncol=2, frameon=False, fontsize=12)
fig.suptitle(f"The judge says {rawj[0]:.0%}. Corrected, the assistant\nis {cor[0]:.0%} accurate; "
             f"multi-hop is the weak spot", x=0.02, ha="left", fontsize=18,
             fontweight="bold", color=INK, y=0.985)
ax.set_title("Synthetic data: 600 test questions, 150 human-graded", loc="left", fontsize=12.5, color=INK2, pad=8)
fig.tight_layout()
fig.savefig(OUT / "fig1_judge_overstates_accuracy.png", facecolor="white")
plt.close(fig)

# Figure 2: failure decomposition of wrong answers (human-graded), by question type
fig, ax = plt.subplots(figsize=(8, 6.4), dpi=200)
NAMES = {"retrieval_miss": "Retrieval miss", "ungrounded_gen": "Ungrounded generation",
         "grounded_but_wrong": "Grounded but wrong", "no_source": "No source; should abstain"}
y2 = np.arange(len(groups))[::-1]
left = np.zeros(len(groups))
for c in CAUSES:
    vals = np.array([decomp[g][c] for g in groups])
    ax.barh(y2, vals, 0.6, left=left, color=SERIES[c], edgecolor="white", linewidth=2, label=NAMES[c])
    for yy, l_, v in zip(y2, left, vals):
        if v >= 0.12:
            ax.text(l_ + v / 2, yy, f"{v:.0%}", ha="center", va="center", fontsize=13,
                    color="white" if c in ("retrieval_miss",) else INK, fontweight="bold")
    left += vals
ax.set_yticks(y2)
ax.set_yticklabels([f"{LABEL[g]}\n(n={int(results[f'n_wrong_human_{g}']['value'])} wrong)" for g in groups], fontsize=13.5)
ax.set_xlim(0, 1)
ax.set_xticks([0, .25, .5, .75, 1])
ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
for sp_ in ["top", "right", "left"]:
    ax.spines[sp_].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.legend(loc="upper center", bbox_to_anchor=(0.42, -0.08), ncol=2, frameon=False, fontsize=12)
rm = decomp["multi_hop"]["retrieval_miss"]
fig.suptitle(f"{rm:.0%} of wrong multi-hop answers never had the\nright documents: fix retrieval before the prompt",
             x=0.02, ha="left", fontsize=18, fontweight="bold", color=INK, y=0.985)
ax.set_title("Wrong answers by where they broke (synthetic, human-graded)", loc="left",
             fontsize=12.5, color=INK2, pad=8)
fig.tight_layout()
fig.savefig(OUT / "fig2_failure_decomposition.png", facecolor="white")
plt.close(fig)

# ======================================================= WRITE OUTPUTS
pd.DataFrame([{"name": k, "value": v["value"]} for k, v in results.items()]).to_csv(
    OUT / "python_results.csv", index=False)
payload = {
    "edition": "J01_the-ai-native-people-analytics-team",
    "note": "Synthetic data. Point estimates must match the R script exactly at the stated rounding; "
            "bootstrap CIs depend on each language's RNG and may differ slightly. "
            "Run python/check_agreement.py after r/analysis.R to add the 'r' values.",
    "assumptions": [
        "Per-type corrected accuracy uses POOLED judge sensitivity/specificity: no question type has "
        ">= 30 truly-correct and >= 30 truly-wrong human grades, so per-type sens/spec would be too noisy. "
        "This assumes the judge's error rates do not differ by question type (true by construction here; "
        "check it in real data as the human sample grows).",
        "Corrected values are clipped to [0, 1].",
        "Missing judge verdicts (timeouts) are treated as missing at random and excluded.",
        "Bootstrap resamples question clusters (near-duplicates share a cluster), B = 2000, percentile CI.",
        "Failure causes use observable traces (gold vs. retrieved docs, claim support); correctness from human grades."],
    "results": results,
}
with open(OUT / "results.json", "w") as f:
    json.dump(payload, f, indent=2)
print(f"\nWrote {len(results)} results to {OUT / 'results.json'} and two figures.")
