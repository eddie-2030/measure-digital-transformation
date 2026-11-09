"""Generate the synthetic golden-set labeling data for C02 (Golden Sets).

Run from anywhere:  python3 data/generate_data.py
Seed: 20261109 (fixed). Every number in the package flows from this seed.

The scenario
------------
A company audits its AI customer-support agent. 1,000 AI-generated replies
("items") must be labeled "error" (1: a policy, factual or tone error that
needs correction) or "OK" (0). Fifteen labelers of different quality label
an incomplete, overlapping design: each item gets 3, 4 or 5 labels from
randomly chosen labelers. Nobody sees the truth; we keep it in
ground_truth_*.csv so the methods can be checked.

Data-generating process (all TRUE parameters are written to ground_truth_labelers.csv)
---------------------------------------------------------------------------------------
1. Items. 10% of items (100) are "ambiguous" (genuinely borderline cases).
   P(true error) = 0.40 for ambiguous items and 0.178 for clear items, so the
   overall true prevalence is about 20%.
2. Labelers (15) each have a hidden confusion matrix:
     sensitivity  = P(says error | truly error)
     specificity  = P(says OK    | truly OK)
     S01-S03  senior QA specialists   sens 0.91-0.93, spec 0.96-0.97, low workload (expensive)
     T01-T08  typical labelers        sens 0.75-0.85, spec 0.88-0.93, normal workload
              (T08 is a new hire with very few items)
     L01-L03  lenient "rubber-stampers" sens 0.35-0.45, spec 0.98, HIGH workload
              (fast because they rarely flag anything)
     R01      near-random labeler     sens 0.55, spec 0.55
3. Assignment. Number of labels per item: 3 / 4 / 5 with probability
   0.40 / 0.35 / 0.25. Labelers are drawn without replacement with
   probability proportional to their workload weight, so rubber-stampers
   appear on many items (this is what makes majority vote miss errors).
4. Labels. On clear items each labeler answers independently given the
   truth, using their sensitivity/specificity. On ambiguous items every
   labeler's accuracy is shrunk 75% of the way toward a coin flip
   (sens' = 0.5 + 0.25*(sens - 0.5), same for spec). This deliberately
   breaks Dawid-Skene's conditional-independence assumption on 10% of items,
   to show the method's limit.
5. Messiness: about 1.5% of assignments are skipped (label left blank);
   one exact duplicate row is appended; T08 labels only a handful of items.
6. Expert adjudicator: for every item we pre-draw the label a near-perfect
   expert would give if asked (98% accurate on every item, including ambiguous
   ones). Both the Python and the R script read these draws from
   expert_adjudication_draws.csv, so the adjudication step is identical.
   The same file holds a pre-drawn random order used for (a) the
   "random adjudication" comparison and (b) breaking ties.
7. AI judge: an automated LLM-as-judge flags each reply (true sensitivity
   0.80, specificity 0.95). Used only in Step 4 to show how the answer key
   changes a downstream metric.

Files written (next to this script)
-----------------------------------
labels_long.csv                   one row per (item, labeler) assignment  -> the analysis input
items.csv                         item list with the AI judge's flag      -> the analysis input
expert_adjudication_draws.csv     pre-drawn expert labels + random order  -> the analysis input
ground_truth_items.csv            true label and ambiguity flag per item  (hidden truth)
ground_truth_labelers.csv         true sensitivity/specificity/workload   (hidden truth)
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261109
HERE = Path(__file__).resolve().parent
rng = np.random.default_rng(SEED)

N_ITEMS = 1000
N_AMBIG = 100
P_ERR_AMBIG = 0.40
P_ERR_CLEAR = 0.178
AMBIG_SHRINK = 0.25          # ambiguous items keep only 25% of a labeler's skill above chance
P_SKIP = 0.015
EXPERT_ACC = 0.98
JUDGE_SENS, JUDGE_SPEC = 0.80, 0.95

# ---- labelers: id, group, true sensitivity, true specificity, workload weight
labelers = pd.DataFrame(
    [
        ("S01", "senior", 0.93, 0.97, 0.5),
        ("S02", "senior", 0.92, 0.96, 0.5),
        ("S03", "senior", 0.91, 0.96, 0.5),
        ("T01", "typical", 0.85, 0.93, 1.0),
        ("T02", "typical", 0.83, 0.92, 1.0),
        ("T03", "typical", 0.82, 0.90, 1.0),
        ("T04", "typical", 0.80, 0.91, 1.0),
        ("T05", "typical", 0.78, 0.89, 1.0),
        ("T06", "typical", 0.77, 0.93, 1.0),
        ("T07", "typical", 0.75, 0.88, 1.0),
        ("T08", "typical", 0.80, 0.90, 0.04),   # new hire, very few items
        ("L01", "lenient", 0.45, 0.98, 2.6),
        ("L02", "lenient", 0.40, 0.98, 2.6),
        ("L03", "lenient", 0.35, 0.98, 2.6),
        ("R01", "near_random", 0.55, 0.55, 1.0),
    ],
    columns=["labeler_id", "group", "true_sensitivity", "true_specificity", "workload_weight"],
)
J = len(labelers)

# ---- 1. items and their hidden truth
item_ids = np.array([f"I{i:04d}" for i in range(1, N_ITEMS + 1)])
ambiguous = np.zeros(N_ITEMS, dtype=int)
ambiguous[rng.choice(N_ITEMS, N_AMBIG, replace=False)] = 1
p_err = np.where(ambiguous == 1, P_ERR_AMBIG, P_ERR_CLEAR)
true_label = (rng.random(N_ITEMS) < p_err).astype(int)

# ---- 3. overlapping, incomplete assignment
n_labels = rng.choice([3, 4, 5], size=N_ITEMS, p=[0.40, 0.35, 0.25])
w = labelers["workload_weight"].to_numpy()
rows = []
for i in range(N_ITEMS):
    chosen = rng.choice(J, size=n_labels[i], replace=False, p=w / w.sum())
    for j in chosen:
        sens = labelers.at[j, "true_sensitivity"]
        spec = labelers.at[j, "true_specificity"]
        if ambiguous[i]:
            sens = 0.5 + AMBIG_SHRINK * (sens - 0.5)
            spec = 0.5 + AMBIG_SHRINK * (spec - 0.5)
        p_say_error = sens if true_label[i] == 1 else 1 - spec
        label = int(rng.random() < p_say_error)
        rows.append((item_ids[i], labelers.at[j, "labeler_id"], label))

labels = pd.DataFrame(rows, columns=["item_id", "labeler_id", "label"])
labels["label"] = labels["label"].astype("Int64")

# ---- 5. messiness: skipped labels, a duplicate row
skip = rng.random(len(labels)) < P_SKIP
labels.loc[skip, "label"] = pd.NA
dup = labels[labels["label"].notna()].iloc[[int(rng.integers(0, 500))]]
labels = pd.concat([labels, dup], ignore_index=True)
# label timestamps (minutes into the labeling sprint) for realism; not used by the method
labels["label_minute"] = np.sort(rng.integers(0, 60 * 24 * 10, len(labels)))
labels.loc[labels.index[-1], "label_minute"] = labels.loc[dup.index[0], "label_minute"]  # exact duplicate

# ---- 6. expert draws + random order (shared by Python and R)
expert_correct = rng.random(N_ITEMS) < EXPERT_ACC
expert_label = np.where(expert_correct, true_label, 1 - true_label)
random_order = rng.permutation(N_ITEMS) + 1

# ---- 7. AI judge flags
u = rng.random(N_ITEMS)
judge_flag = np.where(true_label == 1, u < JUDGE_SENS, u < 1 - JUDGE_SPEC).astype(int)

# ---- write files
labels.to_csv(HERE / "labels_long.csv", index=False)
pd.DataFrame({"item_id": item_ids, "ai_judge_flag": judge_flag}).to_csv(HERE / "items.csv", index=False)
pd.DataFrame({"item_id": item_ids, "expert_label": expert_label, "random_order": random_order}).to_csv(
    HERE / "expert_adjudication_draws.csv", index=False)
pd.DataFrame({"item_id": item_ids, "true_label": true_label, "ambiguous": ambiguous}).to_csv(
    HERE / "ground_truth_items.csv", index=False)
labelers.to_csv(HERE / "ground_truth_labelers.csv", index=False)

print(f"seed {SEED}: {N_ITEMS} items, {len(labels)} label rows "
      f"({int(labels['label'].isna().sum())} skipped, 1 duplicate), "
      f"true prevalence {true_label.mean():.3f}, ambiguous {ambiguous.sum()}")
print(labels["labeler_id"].value_counts().sort_index().to_string())
