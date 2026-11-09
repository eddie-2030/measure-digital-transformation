# G01 · Defensible by Design: Measuring AI Used in People Decisions

**Would our AI hiring or promotion tool survive an audit?**

Companion code for the *Measure Digital Transformation* edition G01 (Dr. Eddie Lin). Synthetic data, Python, and R.

## The business question

A company uses an AI resume screen for about 6,000 applicants in one hiring cycle, across three job families: Software Engineering, Customer Operations, and Finance. Every application moves through three decisions: **AI screen → interview → offer**. Leaders want to know whether the tool would survive an adverse impact audit for two comparisons: women vs. men, and applicants aged 40+ vs. under 40. The usual answer is one company-wide number. Here, that number says "passes". Is that the right answer?

## Measurement strategy

Treat the audit as a measurement exercise, not a single ratio:

1. **Measure each decision stage separately.** A fair final outcome can hide an unfair AI step, and the AI step is the one you deployed.
2. **Measure within comparable applicant pools (job family)** as well as company-wide. Pools differ in pass rates and in their mix of groups, so pooling can hide a gap or create one.
3. **For every cell (stage × pool × comparison) report four things:** the impact ratio, a significance test, a 95% interval, and the shortfall.
4. **Pre-register the primary cells** (here: the AI screen within each job family, 6 cells) and a minimum sample size, because 32 cells will produce some flags by chance.
5. **Check against ground truth.** The data are synthetic, so we know the true impact ratios and can see which flags are real.

## The method in plain words

The **selection rate** of a group is the share of its applicants who were selected. The **impact ratio (IR)** compares two groups:

`Impact ratio = (selected_focal / applicants_focal) / (selected_reference / applicants_reference); flag if < 0.80`

The **four-fifths rule** (US Uniform Guidelines on Employee Selection Procedures, 1978) treats a ratio below 0.80 as evidence of adverse impact. It is a rule of thumb about *size*. It ignores sample size: in a small group, a few applicants either way can push the ratio below 0.80. A **significance test** (here Fisher's exact test) asks the opposite question: could a gap this large arise by chance alone? It ignores size: in a very large pool, even a small gap is "significant". So report both, and read them together:

| Four-fifths flag | Significant (p < .05) | Verdict |
|---|---|---|
| yes | yes | **Adverse impact: act** |
| yes | no | **Inconclusive**: usually too few applicants. Not a finding |
| no | yes | **Small but real gap: monitor** |
| no | no | **No evidence of impact** (check the interval width) |

Two more numbers make the result usable. The **95% interval** for the impact ratio (Katz log method: `exp(log(IR) ± 1.96 × sqrt(1/a − 1/n1 + 1/c − 1/n2))`, where a and c are the focal and reference selections and n1 and n2 their applicants) shows how precisely the ratio is known. The **shortfall** is the number of additional selections the lower-rate group would need to match the higher group's rate, a number leaders can picture.

In this package, IR is always focal ÷ reference (focal = 40+ or women). A ratio above 1 means the focal group is selected at a higher rate; the four-fifths check then compares the reference group with the focal group (lower rate ÷ higher rate), as the Uniform Guidelines do.

## How to run (about one minute)

Requirements: Python 3 with numpy, pandas, scipy, matplotlib; R (base R only, no packages).

```bash
python data/generate_data.py        # writes data/applications.csv and data/true_parameters.csv (seed 20261272)
python python/analysis.py           # Steps 1-4, figures, outputs/results.json, outputs/python_results.csv
Rscript r/analysis.R                # the same steps in base R -> outputs/r_results.csv, outputs/r_headline.csv
python python/check_agreement.py    # confirms Python = R and adds the "r" values to results.json
python python/audit_worksheet.py    # optional: the audit on the worksheet template (or your own csv)
```

Every script finds its files relative to its own location, so you can run it from any folder.

## What's in the folder

| Path | What it is |
|---|---|
| `data/generate_data.py` | Seeded generator. Its docstring documents the data-generating process, every true parameter, and how the seed was chosen (`--scan`) |
| `data/applications.csv` | 6,072 applications (6,000 people; 72 applied twice) |
| `data/true_parameters.csv` | True selection rates and impact ratios for all 32 cells |
| `data/data_dictionary.md` | Column definitions and how the analysis uses them |
| `python/analysis.py` | Step 1 naive → Step 2 correct audit → Step 3 ground truth and false-alarm simulation → Step 4 decision |
| `r/analysis.R` | The same analysis in base R (`fisher.test`, `dbinom`, `rbinom`) |
| `python/check_agreement.py` | Python-versus-R comparison; writes both values into `results.json` |
| `python/audit_worksheet.py` | Runs the audit on any worksheet csv (any number of groups; flags groups below a minimum n) |
| `outputs/results.json` | Every number the article may quote, with rounding, meaning, and Python and R values |
| `outputs/python_results.csv`, `outputs/r_results.csv` | The full 32-cell audit table (identical in both languages) |
| `outputs/python_false_alarm.csv`, `outputs/r_false_alarm.csv` | Four-fifths false-alarm rates under true parity, exact and simulated |
| `outputs/python_false_alarm_curve.csv` | The exact false-alarm curve plotted in fig2 (20 group sizes) |
| `outputs/r_headline.csv` | R's headline numbers, merged into `results.json` by `check_agreement.py` |
| `outputs/worksheet_example_audit.csv` | Output of `audit_worksheet.py` on the template's synthetic example |
| `outputs/fig1_ai_screen_fails_40plus_in_one_family.png` | Impact ratios with 95% intervals: company-wide vs. AI screen by job family |
| `outputs/fig2_four_fifths_false_alarms_by_group_size.png` | How often the four-fifths rule flags by chance, by group size |
| `templates/adverse_impact_audit_worksheet.csv` | Input template: `stage, job_family, group_variable, group, applicants, selected` (filled with the synthetic example) |
| `templates/adverse_impact_audit_checklist.md` | Pre-registration checklist, decision rules, and a reporting template |

## What to look for in the output

- **Step 0 (cleaning).** 6,072 rows become 6,000 applicants after dropping 72 earlier duplicate applications; 68 job-family labels are fixed. 363 applicants did not disclose sex and 463 did not disclose age; each is left out of that comparison only.
- **Step 1 (naive).** One company-wide check of final offers gives an impact ratio of **0.92** for applicants 40+. The AI screen pooled across families gives **0.89**. Both pass the four-fifths rule, and a naive audit would stop here: "the tool is fine."
- **Step 2 (correct audit), AI screen by job family:**
  - **Software Engineering, 40+: IR 0.64 (95% CI 0.54–0.75), p < 0.001.** 27.0% of 40+ applicants pass the screen vs. 42.5% under 40. That is a shortfall of **65** passes (29 just to reach 0.80). Both rules agree: **act**. The family's applicant-to-offer ratio is 0.82, because interviewers advance a larger share of the 40+ applicants who get through. A check of final outcomes would miss it.
  - **Finance, women: IR 0.72 (95% CI 0.43–1.22), p = 0.26.** Only 55 women applied. The four-fifths rule flags it, but the interval runs from a large gap to women ahead. **Inconclusive**: not a finding. Under true parity, groups of this size get a four-fifths flag 32% of the time (18% with women as the lower group).
  - **Customer Operations, 40+: IR 0.90 (95% CI 0.84–0.95), p < 0.001.** With 3,300+ applicants with a disclosed age, a 6-point gap is clearly real but passes four-fifths. **Monitor**; the shortfall is 78 passes.
  - The other three AI-stage cells show no evidence of impact.
- **Across all 32 cells** (4 stages × 4 pools × 2 comparisons), the four-fifths rule raises **6 flags**; only **1** is a true adverse impact. The other 5 are all in Finance's small groups. Requiring both rules gives **1 "act" verdict and 0 false alarms**. One cell with true parity (women at the Customer Operations interview stage, p = 0.009) is significant by chance: what you should expect when you test 32 cells.
- **Pooling can also create a gap.** Company-wide, women pass the AI screen at a higher rate than men (IR 1.08, p = 0.003), although the screen treats them identically within every family. Women are a larger share of Customer Operations, where pass rates are highest.
- **Step 3 (ground truth).** The true AI-screen ratios are 0.70 (Software Engineering, 40+), 0.91 (Customer Operations, 40+), and 1.00 everywhere else. The 95% intervals contain the true ratio in 30 of 32 cells. Under true parity at a 30% selection rate, the four-fifths rule flags **60%** of audits with 25 people per group, **46%** at 50, **30%** at 100, **10%** at 250, and **0.1%** at 1,000. The Fisher test stays at or below 5% at every size.
- **Step 4 (decision).** Pause or adjust the AI screen for Software Engineering and investigate the feature correlated with age (years since graduation or since the last role change). Monitor Customer Operations each cycle. Do not act on the Finance flag; pool more cycles. Pre-register primary cells and a minimum n, and report both rules with intervals in the audit pack. The next step after a confirmed impact is a job-relatedness (validity) review, not a verdict.

## Honest caveats (built into the code)

- **The seed was chosen to show the teaching cases.** A small-group four-fifths flag under true parity happens by chance (32% of the time for Finance's group sizes, in either direction), so the generator's `--scan` checks 1,000 seeds and keeps the first whose realized data show all four cases cleanly (9 of 1,000 qualify). The *rates* in the Monte Carlo are what to generalize, not this one draw.
- **The four-fifths rule is evidence for review, not a legal conclusion.** This package is not legal advice.
- **The Fisher test is conservative.** Under parity it rejects less than 5% of the time with small groups (2–3% at 25–50 per group). The 95% interval is the better guide to what the data can and cannot rule out.
- **Monte Carlo rates depend on the random-number generator.** Python and R agree exactly on every point estimate, p-value, interval, and exact false-alarm rate (60 headline numbers, 32 × 27 table entries). The simulated rates (`mc_*`) differ by at most 0.011 because numpy and R draw different samples; both are recorded in `results.json`.
- **Requisition nesting.** Each requisition has its own pass-rate factor, which adds realistic noise between openings. The audit treats applicants within a job family as one pool, as most audits do. If requisitions differ a lot in group mix, audit them, too.
- **Binary comparisons only.** Sex and age bands are simplified to two categories to teach the method. `audit_worksheet.py` handles any number of groups, comparing each with the highest-rate group.

## Synthetic-data disclaimer

All data in this folder are synthetic, generated to illustrate the method. They do not come from Meta, any client, or any real organization. The goal is to show how a business question, a measurement strategy, the statistics, and the code connect end to end.

## Connect to the article

The article asks whether an AI hiring tool would survive an audit. This package shows why the answer depends on where you look and which rules you use. One company-wide ratio of 0.92 says "pass" for a screen that, in one job family, advances applicants 40+ at 64% of the rate of younger applicants. Stage-by-stage, family-by-family audits find it. Reading the four-fifths rule together with a significance test and an interval separates the real problem from a small-sample false alarm in Finance, and from a small but real gap in Customer Operations. As in the B06 edition on phased rollouts, the average hides who is left behind. The difference here is that the subgroup is a protected group, and the decision is a person's job application.
