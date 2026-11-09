# Adverse impact audit: checklist and reporting template

Use this with `adverse_impact_audit_worksheet.csv` (one row per stage × job family × group: `stage, job_family, group_variable, group, applicants, selected`). The rows shipped in the worksheet are the **synthetic** example from this package; replace them with your own counts and run `python python/audit_worksheet.py your_file.csv --min-n 30`.

This template supports a measurement review. It is not legal advice; involve employment counsel before drawing legal conclusions.

## A. Before you look at the data (pre-register)

- [ ] **Decision stages listed**, with the AI-assisted stage(s) named: e.g. AI screen → interview → offer.
- [ ] **Comparable pools defined** (job family, level, or requisition group) and frozen before analysis.
- [ ] **Group variables and categories defined** (e.g. sex: women, men; age: 40+, under 40), including how undisclosed self-identification is handled (recommended: exclude from that comparison and report the count).
- [ ] **Primary cells named.** Recommended: the AI-assisted stage within each pool, for each group variable. Everything else is secondary (reported, but a flag there is a lead to check, not a finding).
- [ ] **Minimum sample size set** (e.g. ≥ 30 applicants per group per cell; below it, report counts only and "collect more data").
- [ ] **Decision rules written down** (section C), including what happens when the four-fifths rule and the significance test disagree.
- [ ] **Denominators defined per stage**: who was at risk of selection, and how withdrawals and duplicate applications are treated (recommended: keep each person's latest application).

## B. Data checks

- [ ] Duplicates removed (count: ___ ).
- [ ] Pool labels normalized (count fixed: ___ ).
- [ ] Undisclosed self-identification counted per group variable (sex: ___ , age: ___ ). Is the share similar across pools and stages? If not, note it as a limitation.
- [ ] Withdrawals counted per stage (___ before interview, ___ before offer).
- [ ] Stage counts reconcile: applicants ≥ AI passes ≥ interview passes ≥ offers, per pool.

## C. For every cell, report all of these

| Item | Formula / rule |
|---|---|
| Selection rate per group | selected / applicants |
| Impact ratio (IR) | rate of the lower-rate group / rate of the highest-rate group |
| Four-fifths flag | IR < 0.80 |
| Significance | Fisher exact test, two-sided (p < 0.05) |
| 95% interval for IR | Katz: exp( log(IR) ± 1.96 × sqrt(1/a − 1/n1 + 1/c − 1/n2) ) |
| Shortfall | (higher group's rate × lower group's applicants) − lower group's selections |

**Verdict (read both rules together):**

| Four-fifths flag | p < 0.05 | Verdict | What to do |
|---|---|---|---|
| Yes | Yes | **Adverse impact: act** | Escalate. Pause or adjust the step in that pool; investigate the features driving it; proceed to a job-relatedness (validity) review. |
| Yes | No | **Inconclusive** | Not a finding. Usually a small group. Report the interval, pool more cycles, re-test at the pre-set minimum n. |
| No | Yes | **Small but real gap: monitor** | Usually a large pool. Track it each cycle; report the shortfall; check whether it is growing. |
| No | No | **No evidence of impact** | Report the interval anyway: a wide interval means "not enough data", not "fair". |

## D. Reporting template (one block per primary cell)

```
Stage:              ____________   (AI-assisted? yes / no)
Pool:               ____________
Comparison:         ____________ vs ____________
Applicants:         focal ____   reference ____
Selected:           focal ____   reference ____
Selection rates:    focal ____%  reference ____%
Impact ratio:       ____  (95% CI ____ to ____)
Fisher exact p:     ____
Four-fifths flag:   yes / no
Shortfall:          ____ selections (to equal rates); ____ (to reach IR 0.80)
Verdict:            act / inconclusive / monitor / no evidence
Next step & owner:  ______________________   due ________
```

## E. Summary for leaders (one paragraph per audit)

1. How many cells were tested, and how many were primary.
2. Which primary cells were **act**, **monitor**, **inconclusive** (with the reason, usually sample size).
3. The pooled (company-wide) and applicant-to-offer numbers, labelled as context only: they can hide a problem at the AI step or inside one pool.
4. How many four-fifths flags you would expect by chance given the group sizes (see `outputs/fig2_four_fifths_false_alarms_by_group_size.png`).
5. Actions, owners, and the date of the next audit.
