# Measure Digital Transformation: companion code

Companion code for the LinkedIn newsletter *Measure Digital Transformation* by Dr. Eddie Lin. Each folder pairs one edition's leadership question with a measurement strategy, the statistics behind it, and runnable Python and R code on a synthetic dataset.

All data in this repository are synthetic, generated to illustrate the methods. They do not come from Meta, any client, or any real organization.

## Editions

| Edition | Folder | Method |
|---|---|---|
| Adoption Is Not Impact: How to Prove a GenAI Rollout Actually Worked | [A01_adoption-is-not-impact](A01_adoption-is-not-impact) | Difference-in-differences; staggered-adoption (group-time) estimators |
| Who Audits the Agents? Human Oversight as a Measurable Capability | [C01_who-audits-the-agents](C01_who-audits-the-agents) | Chance-corrected agreement (Cohen's kappa, Krippendorff's alpha); golden-set accuracy |
| Don't Launch Everywhere at Once: Who Benefits from AI? | [B06_don-t-launch-everywhere-at-once](B06_don-t-launch-everywhere-at-once) | Heterogeneous effects: pre-specified interactions, DR-learner with random forest, Holm correction |
| The AI-Native People Analytics Team | [J01_the-ai-native-people-analytics-team](J01_the-ai-native-people-analytics-team) | RAG evaluation: retrieval accuracy and groundedness |

Each folder's README explains how to run it: generate the data with `python3 data/generate_data.py`, then run the Python and R analysis scripts.
