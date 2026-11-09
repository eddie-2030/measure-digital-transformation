# Data dictionary

All data are synthetic (see `generate_data.py` for the data-generating process and seed).

## `applications.csv` (one row per application; 6,072 rows, 6,000 unique applicants)

| Column | Type | Meaning | Units / values |
|---|---|---|---|
| `application_id` | string | Unique id of the application, assigned in date order | `APP000001` … |
| `applicant_id` | string | The person. 72 people appear twice (an earlier and a later application); keep the latest | `P00001` … |
| `application_date` | date | Date the application was submitted | ISO `YYYY-MM-DD`, 2026-06 to 2026-09 |
| `job_family` | string | Comparable applicant pool. About 1% of labels are typed inconsistently and must be normalized | Software Engineering, Customer Operations, Finance (plus variants such as `Customer Ops`, `software engineering`, `Finance `) |
| `requisition_id` | string | The job opening (requisitions are nested within job family) | e.g. `SOF-007`, `CUS-012`, `FIN-003` |
| `sex` | string | Self-identified sex | `Woman`, `Man`, `Prefer not to say`, or blank (not disclosed) |
| `age_band` | string | Self-identified age band | `Under 40`, `40+`, `Prefer not to say`, or blank (not disclosed) |
| `ai_screen_pass` | integer | 1 if the AI resume screen advanced the application | 0 / 1 |
| `interview_pass` | integer or blank | 1 if the interview loop passed the applicant. Blank if the applicant did not reach the interview (failed the AI screen or withdrew) | 0 / 1 / blank |
| `offer` | integer or blank | 1 if an offer was made. Blank if the applicant did not reach the offer decision | 0 / 1 / blank |
| `withdrew` | string | Stage before which the applicant withdrew, if any | blank, `before_interview`, `before_offer` |

How the analysis uses these columns:

- **Undisclosed sex or age** (`Prefer not to say` or blank) is excluded from *that* comparison only, and the count is reported (363 sex, 463 age).
- **Stage denominators.** AI screen: every applicant. Interview: AI-screen passers who did not withdraw. Offer: interview passers who did not withdraw. Applicant-to-offer ("cumulative"): every applicant, with withdrawals counted as not selected.

## `true_parameters.csv` (the ground truth, from the design)

| Column | Type | Meaning |
|---|---|---|
| `stage` | string | `ai_screen`, `interview`, `offer` (each conditional on reaching it), or `cumulative` (applicant to offer) |
| `pool` | string | Job family, or `All families` (pooled) |
| `comparison` | string | `age` (40+ vs Under 40) or `sex` (Woman vs Man) |
| `focal_group`, `reference_group` | string | The two groups compared (IR = focal rate / reference rate) |
| `true_rate_focal`, `true_rate_reference` | number | Expected selection rates implied by the design (proportions, 0 to 1) |
| `true_ir` | number | True impact ratio. Family rows are exact by construction; `All families` rows are the expected pooled ratios given the design's group shares |
| `true_category` | string | `real adverse impact` (< 0.80), `small real gap` (0.80 to 0.97), `parity`, or `focal group favored` (> 1.03) |
