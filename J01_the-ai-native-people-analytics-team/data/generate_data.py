"""
generate_data.py -- synthetic evaluation records for a people-analytics RAG assistant
====================================================================================

Edition J01, "The AI-Native People Analytics Team" (Measure Digital Transformation).

WHAT THIS SCRIPT SIMULATES
--------------------------
A people-analytics team has built an AI assistant that answers HR / workforce
questions ("What is the parental-leave policy in EMEA?", "What was attrition in
APAC Sales last quarter?") using Retrieval-Augmented Generation (RAG):

    question --> RETRIEVER fetches top-k documents --> LLM GENERATES an answer
                 from those documents

The team built a labeled TEST SET of 600 questions. For each question we record
what an evaluation harness would log: the gold (truly relevant) documents, the
documents the retriever returned, how many answer claims were supported by the
retrieved text, an automated LLM-judge verdict, a user thumbs-up, and -- for a
random subsample of 150 questions -- a human expert's grade.

NO REAL LLM IS CALLED. Retrieval results, answers, judge labels, thumbs and human
grades are all simulated as labeled records with the probabilities below, so we
know the ground truth by construction and can check every method against it.

DATA-GENERATING PROCESS (true parameters; change them and re-run)
-----------------------------------------------------------------
Seed: 20261114 (numpy default_rng). Same seed -> byte-identical CSVs.
  Seed choice, stated openly: we scanned seeds 20261100-20261139 and kept one whose
  600-question sample is TYPICAL of the population (realized accuracy, judge rate
  and judge sensitivity/specificity all close to their expected values). Across
  those 40 seeds the Rogan-Gladen corrected accuracy ranged from about 0.61 to 0.79
  -- a reminder that with 150 human grades the confidence interval is wide.

Corpus: 300 documents = 150 policy pages, 60 metric definitions, 90 dashboard
extracts, each tagged with a region and business unit.

Questions (600, unequal mix):
  policy_lookup    300 (50%)  1 gold doc  (a policy page)
  computed_metric  180 (30%)  2 gold docs (a metric definition + a dashboard extract)
  multi_hop        120 (20%)  2 or 3 gold docs from different sources (60% / 40%)
  12 questions (6 / 4 / 2) have NO relevant document in the corpus; the right
  answer is "I don't know".

Step 1, retrieval (k = 5). Each gold document lands in the top 5 independently with
  probability  P_DOC[type] x REGION_MULT[region]
     P_DOC       policy 0.95, metric 0.91, multi-hop 0.80
     REGION_MULT AMER 1.00, EMEA 0.98, APAC 0.93, LATAM 0.96
  Retrieved gold docs get ranks drawn with weights [.50,.20,.13,.10,.07] for
  ranks 1..5; remaining slots are filled with random distractor documents.
  A "full hit" means every gold document was retrieved (recall@5 = 1).

Step 2, generation.
  If full hit:   answer is grounded with P_GROUNDED = policy .94, metric .86, multi .80
                 grounded   -> correct with P_CORRECT_GROUNDED = .96, .92, .90
                 ungrounded -> correct with 0.20
  If not a full hit (retrieval miss):
                 abstains ("I don't know") with 0.15 -> counted wrong (answer existed)
                 otherwise answers fluently; ungrounded with 0.90; correct with 0.12
  Unanswerable:  abstains with 0.40 (= correct); otherwise a fluent, ungrounded,
                 wrong answer.
  Claims: an answer makes n_claims = 2 + Poisson(2) (max 7) factual claims; each claim
  is supported by the retrieved text with prob .95 (grounded) or .35 (ungrounded).
  (In real life a claim-level groundedness checker produces these counts.)

Step 3, observers of correctness.
  LLM judge (all questions): says "correct" with P = 0.96 if truly correct
          (sensitivity) and with P = 0.50 if truly wrong (specificity = 0.50,
          a LENIENT judge). 4% of judge calls fail (missing at random).
  Thumbs-up: 55% of users rate; P(up | correct) = .92, P(up | wrong) = .80.
  Human grading: a simple random sample of 150 questions is graded by an expert;
          human_label equals the true correctness (gold standard).

Messy realism: 24 near-duplicate (paraphrased) questions that share a cluster_id
  with their original and usually (85%) reproduce its result; 12 unanswerable
  questions; regional variation in retrieval; missing judge outputs; unrated
  thumbs; 8 rows whose retrieved_doc_ids were logged with lower-case / extra
  spaces and must be cleaned.

Oracle columns (true_correct, true_grounded) exist only because the data are
synthetic. A real team never sees them -- we keep them for the ground-truth check.

The script also simulates a large population (100,000 questions, seed + 1, no
duplicates) with the same process and writes the expected ("true") metrics to
data/true_parameters.csv.

Run:  python data/generate_data.py      (from the package root or anywhere)
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261114  # see note in the docstring on how this seed was chosen
HERE = Path(__file__).resolve().parent

# ---------------------------- true parameters ---------------------------------
TYPES = ["policy_lookup", "computed_metric", "multi_hop"]
N_BY_TYPE = {"policy_lookup": 300, "computed_metric": 180, "multi_hop": 120}
N_UNANSWERABLE = {"policy_lookup": 6, "computed_metric": 4, "multi_hop": 2}
REGIONS = ["AMER", "EMEA", "APAC", "LATAM"]
REGION_P = [0.35, 0.30, 0.25, 0.10]
REGION_MULT = {"AMER": 1.00, "EMEA": 0.98, "APAC": 0.93, "LATAM": 0.96}
BUS = ["Sales", "Engineering", "Operations", "Corporate"]
K = 5
RANK_W = np.array([0.50, 0.20, 0.13, 0.10, 0.07])

P_DOC = {"policy_lookup": 0.95, "computed_metric": 0.91, "multi_hop": 0.80}
P_GROUNDED = {"policy_lookup": 0.94, "computed_metric": 0.86, "multi_hop": 0.80}
P_CORRECT_GROUNDED = {"policy_lookup": 0.96, "computed_metric": 0.92, "multi_hop": 0.90}
P_CORRECT_UNGROUNDED_HIT = 0.20
P_ABSTAIN_MISS = 0.15
P_UNGROUNDED_MISS = 0.90
P_CORRECT_MISS = 0.12
P_ABSTAIN_UNANSWERABLE = 0.40
P_CLAIM_SUPPORTED = {True: 0.95, False: 0.35}

JUDGE_SENS = 0.96
JUDGE_SPEC = 0.50
P_JUDGE_MISSING = 0.04
P_RATED = 0.55
P_UP_CORRECT, P_UP_WRONG = 0.92, 0.80
N_HUMAN = 150
N_DUPLICATES = 24
P_DUP_SAME_RESULT = 0.85
N_MESSY_IDS = 8
GROUNDED_THRESHOLD = 0.8  # used by the analysis to call an answer "grounded"


def make_corpus(rng):
    """300 documents with type, region and business unit."""
    rows = []
    spec = [("policy", 150), ("metric_definition", 60), ("dashboard_extract", 90)]
    i = 1
    for dtype, n in spec:
        for _ in range(n):
            region = "GLOBAL" if dtype == "metric_definition" else rng.choice(REGIONS, p=REGION_P)
            rows.append({"doc_id": f"D{i:03d}", "doc_type": dtype, "region": region,
                         "business_unit": rng.choice(BUS)})
            i += 1
    corpus = pd.DataFrame(rows)
    topic = {"policy": ["Parental leave", "Remote work", "Promotion cycle", "Overtime",
                        "Relocation", "Learning budget", "Sick leave", "Code of conduct"],
             "metric_definition": ["Attrition rate", "Time to fill", "Internal mobility rate",
                                   "Span of control", "Absence rate", "Regretted attrition"],
             "dashboard_extract": ["Headcount", "Attrition", "Hiring funnel", "Engagement",
                                   "Absence", "Mobility"]}
    corpus["title"] = [f"{rng.choice(topic[t])} - {r} - {b}"
                       for t, r, b in zip(corpus.doc_type, corpus.region, corpus.business_unit)]
    return corpus


def draw_gold(qtype, rng, pools):
    """Pick the truly relevant documents for one question."""
    if qtype == "policy_lookup":
        return list(rng.choice(pools["policy"], 1))
    if qtype == "computed_metric":
        return [rng.choice(pools["metric_definition"]), rng.choice(pools["dashboard_extract"])]
    n = 2 if rng.random() < 0.60 else 3
    srcs = rng.choice(["policy", "metric_definition", "dashboard_extract"], n, replace=False)
    return [rng.choice(pools[s]) for s in srcs]


def simulate_answer(qtype, region, gold, answerable, rng, all_ids):
    """Simulate retrieval + generation for one question. Returns a dict."""
    # ---- retrieval: which gold docs make the top-k, and at what rank
    if answerable:
        p = P_DOC[qtype] * REGION_MULT[region]
        got = [g for g in gold if rng.random() < p]
    else:
        got = []
    ranks = (rng.choice(np.arange(1, K + 1), len(got), replace=False, p=RANK_W)
             if got else np.array([], dtype=int))
    slots = [None] * K
    for g, r in zip(got, ranks):
        slots[r - 1] = g
    distract = [d for d in rng.permutation(all_ids)[:20] if d not in gold]
    for j in range(K):
        if slots[j] is None:
            slots[j] = distract.pop(0)
    full_hit = answerable and len(got) == len(gold)

    # ---- generation
    if not answerable:
        abstained = rng.random() < P_ABSTAIN_UNANSWERABLE
        grounded = False
        correct = abstained
    elif full_hit:
        abstained = False
        grounded = rng.random() < P_GROUNDED[qtype]
        pc = P_CORRECT_GROUNDED[qtype] if grounded else P_CORRECT_UNGROUNDED_HIT
        correct = rng.random() < pc
    else:
        abstained = rng.random() < P_ABSTAIN_MISS
        if abstained:
            grounded, correct = False, False
        else:
            grounded = rng.random() >= P_UNGROUNDED_MISS
            correct = rng.random() < P_CORRECT_MISS

    if abstained:
        n_claims, n_sup = 0, 0
    else:
        n_claims = int(min(2 + rng.poisson(2), 7))
        n_sup = int(rng.binomial(n_claims, P_CLAIM_SUPPORTED[bool(grounded)]))
    return {"retrieved": slots, "abstained": int(abstained), "n_claims": n_claims,
            "n_claims_supported": n_sup, "true_grounded": int(grounded),
            "true_correct": int(correct)}


def observers(correct, rng):
    """Judge label and thumbs for one answer, given its true correctness."""
    judge = None
    if rng.random() >= P_JUDGE_MISSING:
        judge = int(rng.random() < (JUDGE_SENS if correct else 1 - JUDGE_SPEC))
    thumbs = None
    if rng.random() < P_RATED:
        thumbs = int(rng.random() < (P_UP_CORRECT if correct else P_UP_WRONG))
    return judge, thumbs


QTEXT = {"policy_lookup": "What is the {t} policy for {r} {b} employees?",
         "computed_metric": "What was the {t} in {r} {b} last quarter?",
         "multi_hop": "How does {t} in {r} {b} compare with policy targets and last year?"}
QTOPIC = {"policy_lookup": ["parental leave", "remote work", "overtime", "relocation", "sick leave"],
          "computed_metric": ["attrition rate", "time to fill", "internal mobility rate", "absence rate"],
          "multi_hop": ["regretted attrition", "promotion rate", "hiring velocity"]}


def build_test_set(rng, corpus):
    pools = {t: corpus.loc[corpus.doc_type == t, "doc_id"].to_numpy()
             for t in ["policy", "metric_definition", "dashboard_extract"]}
    all_ids = corpus.doc_id.to_numpy()

    # question types in a shuffled order; mark unanswerable ones
    qtypes = np.concatenate([[t] * N_BY_TYPE[t] for t in TYPES])
    qtypes = rng.permutation(qtypes)
    n = len(qtypes)
    answerable = np.ones(n, dtype=int)
    for t in TYPES:
        idx = np.flatnonzero(qtypes == t)
        answerable[rng.choice(idx, N_UNANSWERABLE[t], replace=False)] = 0

    # which rows are near-duplicates of an earlier question of the same type
    dup_rows = set(rng.choice(np.arange(50, n), N_DUPLICATES, replace=False).tolist())

    rows = []
    for i in range(n):
        qid = f"Q{i + 1:04d}"
        t = qtypes[i]
        if i in dup_rows:
            # paraphrase of an earlier, non-duplicate question of the same type
            cands = [r for r in rows if r["question_type"] == t and r["duplicate_of"] == ""]
            orig = cands[rng.integers(len(cands))]
            region, bu = orig["region"], orig["business_unit"]
            gold, ans = orig["_gold"], orig["answerable"]
            if rng.random() < P_DUP_SAME_RESULT:
                sim = {k: orig[k] for k in ["abstained", "n_claims", "n_claims_supported",
                                             "true_grounded", "true_correct"]}
                sim["retrieved"] = orig["_retrieved"]
            else:
                sim = simulate_answer(t, region, gold, ans, rng, all_ids)
            text = "Quick check: " + orig["question_text"][0].lower() + orig["question_text"][1:]
            cluster, dup_of = orig["cluster_id"], orig["question_id"]
        else:
            region = rng.choice(REGIONS, p=REGION_P)
            bu = rng.choice(BUS)
            ans = int(answerable[i])
            gold = draw_gold(t, rng, pools) if ans else []
            sim = simulate_answer(t, region, gold, ans, rng, all_ids)
            text = QTEXT[t].format(t=rng.choice(QTOPIC[t]), r=region, b=bu)
            if not ans:
                text = text.replace("employees?", "contractors on secondment?").replace(
                    "last quarter?", "for the 2019 legacy org?").replace(
                    "last year?", "a pre-merger baseline?")
            cluster, dup_of = qid, ""
        judge, thumbs = observers(sim["true_correct"], rng)
        if sim["abstained"]:
            answer = "I don't know - no source found."
        else:
            answer = f"[simulated answer with {sim['n_claims']} claims, citing {sim['retrieved'][0]}]"
        rows.append({
            "question_id": qid, "cluster_id": cluster, "duplicate_of": dup_of,
            "question_type": t, "region": region, "business_unit": bu,
            "question_text": text, "answerable": ans,
            "gold_doc_ids": ";".join(gold), "retrieved_doc_ids": ";".join(sim["retrieved"]),
            "answer": answer, "abstained": sim["abstained"],
            "n_claims": sim["n_claims"], "n_claims_supported": sim["n_claims_supported"],
            "judge_label": judge, "thumbs_up": thumbs,
            "true_grounded": sim["true_grounded"], "true_correct": sim["true_correct"],
            "_gold": gold, "_retrieved": sim["retrieved"]})

    df = pd.DataFrame(rows).drop(columns=["_gold", "_retrieved"])

    # random human-graded subsample (simple random sample of questions)
    hum = rng.choice(n, N_HUMAN, replace=False)
    df["human_graded"] = 0
    df.loc[hum, "human_graded"] = 1
    df["human_label"] = np.where(df.human_graded == 1, df.true_correct, np.nan)

    # messy logging: lower-case IDs and stray spaces in a few rows
    messy = rng.choice(n, N_MESSY_IDS, replace=False)
    for i in messy:
        ids = df.at[i, "retrieved_doc_ids"].split(";")
        df.at[i, "retrieved_doc_ids"] = " ; ".join(x.lower() for x in ids)

    for c in ["judge_label", "thumbs_up", "human_label"]:
        df[c] = df[c].astype("Int64")
    cols = ["question_id", "cluster_id", "duplicate_of", "question_type", "region",
            "business_unit", "question_text", "answerable", "gold_doc_ids",
            "retrieved_doc_ids", "answer", "abstained", "n_claims", "n_claims_supported",
            "judge_label", "thumbs_up", "human_graded", "human_label",
            "true_grounded", "true_correct"]
    return df[cols]


def population_truth(rng, corpus, n_pop=100_000):
    """Expected metrics of the data-generating process (large simulation)."""
    pools = {t: corpus.loc[corpus.doc_type == t, "doc_id"].to_numpy()
             for t in ["policy", "metric_definition", "dashboard_extract"]}
    all_ids = corpus.doc_id.to_numpy()
    share = {t: N_BY_TYPE[t] / 600 for t in TYPES}
    p_unans = {t: N_UNANSWERABLE[t] / N_BY_TYPE[t] for t in TYPES}
    recs = []
    for t in TYPES:
        for _ in range(int(n_pop * share[t])):
            region = rng.choice(REGIONS, p=REGION_P)
            ans = int(rng.random() >= p_unans[t])
            gold = draw_gold(t, rng, pools) if ans else []
            s = simulate_answer(t, region, gold, ans, rng, all_ids)
            got = [g for g in gold if g in s["retrieved"]]
            rr = 0.0
            for j, d in enumerate(s["retrieved"]):
                if d in gold:
                    rr = 1 / (j + 1)
                    break
            recs.append((t, ans, len(got) / len(gold) if ans else np.nan,
                         int(ans and len(got) == len(gold)), rr if ans else np.nan,
                         s["n_claims_supported"] / s["n_claims"] if s["n_claims"] else np.nan,
                         s["true_correct"], s["abstained"]))
    p = pd.DataFrame(recs, columns=["type", "answerable", "recall", "hit", "rr",
                                    "grnd", "correct", "abstained"])
    out = []

    def add(name, value, meaning):
        out.append({"name": name, "value": round(float(value), 4), "meaning": meaning})

    add("true_accuracy_overall", p.correct.mean(), "Expected end-to-end accuracy, all questions")
    for t in TYPES:
        q = p[p.type == t]
        a = q[q.answerable == 1]
        add(f"true_accuracy_{t}", q.correct.mean(), f"Expected accuracy, {t}")
        add(f"true_recall_at5_{t}", a.recall.mean(), f"Expected recall@5, {t} (answerable)")
        add(f"true_hit_rate_{t}", a.hit.mean(), f"Expected full-retrieval rate, {t}")
        add(f"true_mrr_{t}", a.rr.mean(), f"Expected MRR, {t}")
        add(f"true_groundedness_{t}", q.grnd.mean(), f"Expected claim-support rate, {t}")
    a = p[p.answerable == 1]
    add("true_recall_at5_overall", a.recall.mean(), "Expected recall@5, answerable questions")
    add("true_mrr_overall", a.rr.mean(), "Expected MRR, answerable questions")
    add("true_groundedness_overall", p.grnd.mean(), "Expected claim-support rate, answered")
    add("true_judge_rate_overall", JUDGE_SENS * p.correct.mean() + (1 - JUDGE_SPEC) * (1 - p.correct.mean()),
        "Expected share the judge calls correct")
    add("judge_sensitivity", JUDGE_SENS, "True judge sensitivity")
    add("judge_specificity", JUDGE_SPEC, "True judge specificity")
    add("p_up_correct", P_UP_CORRECT, "P(thumbs up | correct)")
    add("p_up_wrong", P_UP_WRONG, "P(thumbs up | wrong)")
    return pd.DataFrame(out)


def main():
    rng = np.random.default_rng(SEED)
    corpus = make_corpus(rng)
    df = build_test_set(rng, corpus)
    corpus.to_csv(HERE / "corpus.csv", index=False)
    df.to_csv(HERE / "rag_eval_records.csv", index=False)
    truth = population_truth(np.random.default_rng(SEED + 1), corpus)
    truth.to_csv(HERE / "true_parameters.csv", index=False)
    print(f"Wrote {len(corpus)} documents, {len(df)} evaluation records, "
          f"{len(truth)} true parameters to {HERE}")
    print(f"Realized true accuracy in the 600-question test set: {df.true_correct.mean():.3f}")


if __name__ == "__main__":
    main()
