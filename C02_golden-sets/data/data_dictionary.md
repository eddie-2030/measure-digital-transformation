# Data dictionary: C02 Golden Sets

All files are synthetic, written by `generate_data.py` (seed 20261109).

## labels_long.csv (analysis input; one row per item-labeler assignment)

| Column | Type | Meaning | Units / values |
|---|---|---|---|
| item_id | string | AI-generated customer reply being labeled | I0001 to I1000 |
| labeler_id | string | Annotator who labeled the item | S01-S03 senior QA, T01-T08 typical, L01-L03 lenient, R01 near-random (the prefix is for the reader; the methods never use it) |
| label | integer, may be blank | The labeler's verdict | 1 = error (policy, factual or tone error needing correction), 0 = OK, blank = skipped |
| label_minute | integer | When the label was given, minutes since the labeling sprint started | minutes (0 to 14,399); not used by the methods |

Known data issues (handled the same way in Python and R):
1. One exact duplicate row (3,860 rows → 3,859). Dropped.
2. 52 skipped labels (blank). Not counted as votes. After skips, 18 items have fewer than 3 valid labels; they are kept.
3. T08 is a new hire with only 7 labels: too few to estimate a confusion matrix well. The pseudo-counts in Dawid-Skene keep the estimate sensible, and T08 is excluded from the labeler-accuracy checks (minimum 30 labels).
4. The design is incomplete: each item has 3, 4 or 5 labels from different labelers; most labeler pairs overlap on only a few items.

## items.csv (analysis input)

| Column | Type | Meaning | Units / values |
|---|---|---|---|
| item_id | string | AI reply | I0001 to I1000 |
| ai_judge_flag | integer | Verdict of an automated LLM-as-judge on the same reply (used in Step 4 only) | 1 = flagged as error, 0 = not flagged |

## expert_adjudication_draws.csv (analysis input, pre-drawn so Python and R use identical draws)

| Column | Type | Meaning | Units / values |
|---|---|---|---|
| item_id | string | AI reply | I0001 to I1000 |
| expert_label | integer | The label a near-perfect expert adjudicator gives *if* this item is sent to adjudication (98% accurate) | 1 = error, 0 = OK |
| random_order | integer | A random permutation: items with random_order 1-100 form the "random adjudication" sample; also breaks ties when ranking items | 1 to 1000 |

## ground_truth_items.csv (hidden truth; used only to check the methods)

| Column | Type | Meaning | Units / values |
|---|---|---|---|
| item_id | string | AI reply | I0001 to I1000 |
| true_label | integer | The reply's true status | 1 = error, 0 = OK (195 errors, 19.5%) |
| ambiguous | integer | Genuinely borderline item on which every labeler is much less accurate | 1 = ambiguous (100 items), 0 = clear |

## ground_truth_labelers.csv (hidden truth: the generating parameters)

| Column | Type | Meaning | Units / values |
|---|---|---|---|
| labeler_id | string | Annotator | as above |
| group | string | Profile | senior, typical, lenient, near_random |
| true_sensitivity | float | P(says error \| truly error) on clear items | proportion |
| true_specificity | float | P(says OK \| truly OK) on clear items | proportion |
| workload_weight | float | Relative chance of being assigned to an item | lenient labelers 2.6, typical 1.0, senior 0.5, T08 0.04 |

On ambiguous items each labeler's sensitivity and specificity are shrunk 75% of the way toward 0.5. The analysis therefore checks the Dawid-Skene estimates against each labeler's *realized* sensitivity (share of the true errors they actually saw that they flagged), which is lower than the clear-item parameter.
