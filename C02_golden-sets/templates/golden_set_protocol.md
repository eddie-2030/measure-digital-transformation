# Golden Set Protocol (one page)

Companion template for *Golden Sets: Building Ground Truth You Can Trust* (Measure Digital Transformation, C02). Fill in the brackets. Keep the completed protocol next to the golden set: it is the documentation an auditor will ask for.

| Field | Your entry | Guidance |
|---|---|---|
| **Golden set name and version** | [e.g., Support-agent replies v3, 2026-Q4] | Version every change; never overwrite a released key. |
| **Owner** | [name, team] | One accountable owner, not "the labeling team." |
| **Decision it supports** | [e.g., monthly AI-agent error rate; go/no-go for new model] | If no decision uses it, do not build it. |
| **Construct and labeling guideline** | [link] | Define "error" in writing with 5-10 worked examples per category (policy, factual, tone) and explicit borderline rules. Most label noise is guideline noise. |
| **Item sampling** | [e.g., 1,000 replies, stratified by product line and channel; 20% enriched with replies flagged by users] | Random sampling estimates the error rate; enrichment finds rare errors. If you enrich, keep the sampling weights so the rate can be re-weighted. |
| **Labeler pool** | [e.g., 3 senior QA, 8 trained labelers] | Record each labeler's training date and guideline version. |
| **Overlap design** | [e.g., each item 3-5 labels, labelers randomly assigned] | Randomize assignment; every labeler needs ~100+ items and overlap with several colleagues, or their error pattern cannot be estimated. |
| **Minimum labels per item** | [3] | Items that fall below the minimum after skips go back into the queue. |
| **Seed set (anchor)** | [e.g., 50 items labeled by policy owners] | Expert-verified items, mixed in blind. Anchors the model to the *policy*, not just the labelers' consensus. |
| **Aggregation model** | Dawid-Skene (latent class), EM, initialized from majority vote, pseudo-count 1, stop when posteriors move < 1e-9 | Keep majority vote as a comparison line in every report. |
| **Adjudication rule** | Send items with posterior P(error) in [0.2, 0.8], most uncertain first | Plus: any item where a senior disagrees with the model; any seed-set miss. |
| **Adjudication budget** | [e.g., 10% of items per cycle; 1 senior adjudicator, second opinion on 10%] | Queue file: `adjudication_queue.csv`. Adjudicators see the item, not the labelers' names. |
| **Quality checks** | [ ] Labeler confusion matrices reviewed  [ ] No labeler worse than chance  [ ] Seed-set accuracy ≥ [95%]  [ ] Prevalence vs. majority vote compared | A labeler with sensitivity far below peers triggers retraining, not silent down-weighting forever. |
| **Known limits** | [e.g., ~10% genuinely ambiguous items] | Ambiguous items violate the model's independence assumption; tag them, adjudicate them, and consider a third label "ambiguous." |
| **Refresh cadence** | [e.g., quarterly, and after any policy or model change] | A policy change can make an old key wrong overnight. Re-label the seed set first. |
| **Documentation fields per item** | item_id, labels (labeler, label, time), posterior_error_prob, final_label, label_source (model / adjudicated / seed), adjudicator, guideline_version, notes | Store per-labeler sensitivity and specificity for each release. |
| **Release sign-off** | [name, date] | |
