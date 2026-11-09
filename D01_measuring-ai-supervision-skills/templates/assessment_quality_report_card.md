# Assessment Quality Report Card

*Fill this in before an assessment score is used for any decision about people. One card per scale (a set of items meant to measure one skill). The "Example" column holds the synthetic D01 numbers; replace them with yours.*

**Assessment:** ______________________  **Scale:** ______________________  **Version / date:** __________
**Intended use (tick one):** ☐ Research / program evaluation ☐ Team-level training needs ☐ Individual feedback ☐ Individual high-stakes (certification, staffing, pay)
**Population tested:** who, how many, which units / languages: ______________________

## 1. Data screening

| Check | Rule used | Result | Example (D01) |
|---|---|---|---|
| Speeders / careless responders | e.g., < 8 minutes or identical answer to every item | __ excluded | 12 of 612 excluded |
| Missing answers | % blank; how handled | __% | 3.1%; pairwise covariances, scores prorated if ≥ 75% answered |
| Label / data fixes | | | 9 unit labels fixed |

## 2. Content evidence (do the items cover the real job?)

| Question | Answer |
|---|---|
| Which real supervision tasks does each item sample? (map items to a task list) | |
| Which important tasks have no item? | |
| Who reviewed the items (subject-matter experts, how many)? | |

## 3. Internal structure and item quality

Attach the item statistics table (`item_statistics_template.csv`, produced by `item_stats.py` or `item_stats.R`).

| Indicator | Rule of thumb | Result | Example (D01, scenario items) |
|---|---|---|---|
| Items with item-rest r < 0.20 | 0; negative = check the scoring key | | S06 (0.03), S11 (−0.20, mis-keyed) |
| Items whose deletion raises alpha | 0 | | S06, S11 |
| Self-report or "attitude" items mixed into a skill score? | Report them as a separate scale | | 4 confidence items: r = 0.83 among themselves, 0.17 with scenarios |

## 4. Reliability (consistency and precision)

| Indicator | Rule of thumb (a convention, not a law) | Result | Example (D01, cleaned 10 items) |
|---|---|---|---|
| Cronbach's alpha | ≥ 0.70 group use; ≥ 0.90 individual high-stakes | | 0.82 |
| McDonald's omega | same | | 0.82 |
| Standard error of measurement: SEM = SD × √(1 − reliability) | smaller than the differences you act on | | 2.9 points on a 0–40 scale |
| Share of people within ±1.96 SEM of the cutoff | as low as possible; these need a second look | | 46% |
| Reliability in each subgroup (unit, language, role) | each meets the bar for its use | | 0.73 to 0.84 (lowest: Customer Support) |
| Items needed for 0.90 (Spearman-Brown) | | | 20 (10 more good items) |

## 5. Relations to other variables (does the score predict real performance?)

| Indicator | Result | Example (D01) |
|---|---|---|
| Criterion used (independent of the test, measured later) | | % of seeded errors caught in an audit of ~40 AI outputs, 3 weeks later |
| Criterion reliability | | 0.92 (split-half) |
| Validity r with 95% CI: proposed score | | 0.58 (0.52–0.63) |
| Validity r with 95% CI: any alternative score (e.g., all items, self-report) | | all 16 items 0.44 (0.37–0.50); confidence 0.22 (0.14–0.30) |

## 6. Decision

| Use | Allowed now? | Conditions |
|---|---|---|
| Team-level training needs | ☐ yes ☐ no | |
| Individual feedback / development | ☐ yes ☐ no | report a score band (score ± 2 SEM), not a single number |
| Individual certification / staffing | ☐ yes ☐ no | reliability ≥ 0.90, or a second assessment for everyone inside the SEM band |

**Items to fix or drop:** __________ **Next review date:** __________ **Owner:** __________

*Recheck sections 3–5 for every new population, language, or major item change: reliability and validity belong to scores in a group, not to the test itself.*
