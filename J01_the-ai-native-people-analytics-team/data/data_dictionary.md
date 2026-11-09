# Data dictionary (J01, synthetic)

All files are written by `data/generate_data.py` (seed 20261114). No real LLM was called: retrieval results, answers, judge verdicts, thumbs and human grades are simulated records.

## rag_eval_records.csv: one row per test question (600 rows)

| Column | Type | Meaning | Units / values |
|---|---|---|---|
| question_id | string | Test-question ID | Q0001–Q0600 |
| cluster_id | string | Question cluster. Near-duplicate paraphrases share their original's ID. Bootstrap resamples clusters. | question_id of the original |
| duplicate_of | string | Original question if this row is a paraphrase, else empty | question_id or empty (24 filled) |
| question_type | category | Kind of question | policy_lookup, computed_metric, multi_hop |
| region | category | Region the question is about | AMER, EMEA, APAC, LATAM |
| business_unit | category | Business unit the question is about | Sales, Engineering, Operations, Corporate |
| question_text | string | Templated question wording | text |
| answerable | 0/1 | 1 if at least one relevant document exists in the corpus; 0 means the correct answer is "I don't know" | 12 rows = 0 |
| gold_doc_ids | string | Truly relevant documents, separated by `;` | doc IDs; empty if unanswerable |
| retrieved_doc_ids | string | Top-5 documents the retriever returned, in rank order, separated by `;`. **8 rows are messy** (lower-case, extra spaces): clean them before matching | doc IDs |
| answer | string | Placeholder for the generated answer | text |
| abstained | 0/1 | 1 if the assistant said "I don't know" | |
| n_claims | integer | Factual claims in the answer (0 if abstained) | count, 0–7 |
| n_claims_supported | integer | Claims supported by the retrieved text (as a claim checker would label them) | count |
| judge_label | 0/1/NA | Automated LLM-judge verdict: 1 = "correct". NA = judge call failed (about 4%) | |
| thumbs_up | 0/1/NA | User feedback: 1 = thumbs up. NA = user did not rate (about 45%) | |
| human_graded | 0/1 | 1 if the question is in the random human-graded subsample | 150 rows = 1 |
| human_label | 0/1/NA | Expert grade (gold standard): 1 = correct. NA if not graded | |
| true_grounded | 0/1 | ORACLE: answer was generated from the retrieved context. Known only because the data are synthetic | |
| true_correct | 0/1 | ORACLE: answer is truly correct. Used only for the ground-truth check (Step 3) | |

Derived in the analysis scripts: `recall5` (share of gold docs in the top 5), `full_hit` (all gold docs retrieved), `rr` (1 / rank of the first gold doc; 0 if none), `groundedness` (n_claims_supported / n_claims), `cause` (where a wrong answer broke).

## corpus.csv: the 300 documents

| Column | Type | Meaning |
|---|---|---|
| doc_id | string | D001–D300 |
| doc_type | category | policy (150), metric_definition (60), dashboard_extract (90) |
| region | category | AMER, EMEA, APAC, LATAM, or GLOBAL (metric definitions) |
| business_unit | category | Sales, Engineering, Operations, Corporate |
| title | string | Synthetic title |

## true_parameters.csv: the generator's true values

| Column | Type | Meaning |
|---|---|---|
| name | string | Parameter or expected metric (for example true_accuracy_multi_hop, judge_specificity) |
| value | float | Its value. Expected metrics come from a 100,000-question simulation of the same process (seed 20261115) |
| meaning | string | One-line description |
