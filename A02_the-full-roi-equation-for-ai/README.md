# A02 · The Full ROI Equation for AI Agents

Companion code for the *Measure Digital Transformation* newsletter edition "The Full ROI Equation for AI Agents" by Dr. Eddie Lin.

> All data in this folder are synthetic, generated to illustrate the method. They do not come from Meta, any client, or any real organization. The goal is to show how a business question, a measurement strategy, the statistics, and the code connect end to end.

## The business question

**What does an AI agent really cost, and what does it return?** A company has run an internal AI help-desk agent for 26 weeks. It answers employee tickets in three queues: access, IT software and HR policy. The vendor's slide says the agent "already handles 75% of tickets," so the ROI is about 5×. The CFO wants to know what the first 12 months will really be worth, and how likely it is that the agent loses money.

## The measurement strategy

1. **Unit economics from the logs.** Every routed ticket can end three ways: the agent resolves it, the agent escalates it to a human, or the agent closes it and the employee reopens it (rework). The ticket log tells us how often each happens per queue. It also tells us how many human minutes a normal ticket, an escalated ticket and a reopened ticket take. A QA sample of resolved tickets tells us how often a "resolved" answer was actually wrong (a silent error).
2. **Costs the logs cannot see** (licence, build, monitoring, loaded labour cost, remediation cost per error, human review share and minutes, future adoption) come from finance and operations owners as **ranges** (low / most likely / high), not single numbers.
3. **Monte Carlo.** We draw 10,000 plausible combinations of all the inputs and compute the 12-month net value for each. The result is a distribution: a median, an interval and a probability of loss.
4. **Ground-truth check.** Because the data are synthetic, we know the true parameters and the true net value. We check that the method recovers them.

## The method in plain words

The equation for one ticket the agent handles:

```
net value per agent ticket =
      p(resolved) x human minutes avoided x cost per minute     (benefit)
    - agent run cost
    - cost per minute x [p(escalated) x extra handoff minutes + p(reopened) x extra rework minutes]
    - p(resolved) x p(silent error) x remediation cost
    - review share x review minutes x cost per minute

12-month net value = sum over queues (agent tickets x net value per ticket) - licence - build - monitoring
ROI = net value / total cost
```

- **Rates** (resolved, escalated, reopened) get a Dirichlet posterior per queue, which is the "beta distribution" idea for three outcomes. The silent-error rate gets a beta posterior from the QA sample. Small samples give wide distributions, as they should.
- **Human minutes** come from one regression on all human-handled tickets: `minutes = base(queue) + extra × escalated + extra × reopened`. The coefficients are drawn jointly from their estimated covariance.
- **Cost inputs** use triangular distributions built from each owner's low / most likely / high.
- **Agent tickets in 12 months** = the 26 weeks actually logged + 26 projected weeks at an uncertain steady-state routing share (the adoption ramp).
- **Sensitivity (tornado):** we move one input at a time to its 10th and 90th percentile and see how far the net value moves.
- **Break-even:** the value of an input that makes the point-estimate net value exactly zero.

## How to run (about one minute)

From this folder:

```bash
python data/generate_data.py    # writes the four csv files in data/
python python/analysis.py       # Steps 1-4, figures, outputs/results.json
Rscript r/analysis.R            # same steps in R; adds "r" values and the agreement check to results.json
```

Python 3.10+ with **numpy, pandas, statsmodels, matplotlib** (scipy, which installs with statsmodels, supplies the beta and normal quantiles; scikit-learn is not needed for this edition). R 4.x, **base R only, no packages**.

**How Python and R agree exactly.** Monte Carlo results normally differ between languages because their random number generators differ. Here, `generate_data.py` writes the random numbers themselves (`data/mc_uniforms.csv`, 10,000 × 25 uniforms). Both scripts turn them into draws with the same inverse-CDF formulas (`qbeta`/`scipy.stats.beta.ppf`, `qnorm`/`norm.ppf`, a closed-form triangular inverse) and the same Cholesky factor. Quantiles use the same definition (R type 7 = numpy default). The R script compares all 67 results with Python's: all are identical at the stated rounding, and the largest unrounded difference is about 5e-9.

## What to look for in the output

| | Python | R |
|---|---|---|
| Naive (vendor) ROI | 5.30× (net +$509k) | 5.30× (net +$509k) |
| Full model, point estimate (inputs at "most likely") | 0.39× (net +$96k) | 0.39× (net +$96k) |
| Monte Carlo median ROI / net value | 0.15× / +$42k | 0.15× / +$42k |
| 80% interval, net value | −$1k to +$83k | −$1k to +$83k |
| 90% interval, net value | −$14k to +$93k | −$14k to +$93k |
| P(ROI < 0) | 10.7% | 10.7% |
| True net value / ROI (ground truth) | +$63k / 0.22× | +$63k / 0.22× |
| Break-even steady-state adoption | 21.7% of tickets | 21.7% of tickets |

1. **Step 1:** the vendor-style ROI is about 5×. It treats every routed ticket as fully automated, assumes today's adoption level held for the whole year, and ignores escalations, rework, review, errors, build and monitoring.
2. **Step 2:** the bridge from vendor gross savings ($605k) to full net value ($96k at the point estimate). The biggest cuts are escalated or reopened tickets that save no human time (−$150k), the adoption ramp (−$112k), and the licence (−$96k). Handoff and rework minutes (−$47k), build and monitoring (−$40k), remediation (−$30k) and review (−$23k) follow.
3. **Step 2, uncertainty:** the Monte Carlo median (+$42k) sits well below the point estimate (+$96k). Most cost ranges have a long upside tail (remediation, review share, build overruns) and adoption can stall, so "everything at its most likely value" is an optimistic scenario.
4. **Step 3:** all 9 true outcome and error rates fall inside their 95% intervals. The extra minutes for an escalation and a reopen are recovered (5.9 vs 6.0 and 7.8 vs 8.0). The true net value (+$63k) falls inside the 80% interval, at about the 74th percentile of the simulation.
5. **Step 4:** the decision numbers are a roughly 1-in-9 chance of losing money and a median ROI of about 0.15×. Adoption and the remediation cost per error are the top two drivers, followed by the silent-error rate (QA sample of only 125 HR tickets) and the review share. The escalation rates, measured on about 13,000 tickets, move the result least. Adoption must stay above about 22% of tickets in months 7–12 to break even (it was 75% in the last four logged weeks).

## Files

```
README.md
data/generate_data.py           seeded generator; DGP and true parameters documented at the top
data/helpdesk_tickets.csv       26-week ticket log (with duplicates and missing values)
data/cost_assumptions.csv       low / most likely / high cost inputs
data/ground_truth.csv           true parameters, true 12-month net value and ROI
data/mc_uniforms.csv            shared Monte Carlo random numbers
data/data_dictionary.md
python/analysis.py              Steps 1-4 + figures + results.json
r/analysis.R                    same steps in base R + agreement check
outputs/results.json            every quotable number, with python and r values
outputs/results_long.csv        same numbers, flat (Python)
outputs/results_r.csv           R values (unrounded)
outputs/tornado_python.csv      sensitivity table (also tornado_r.csv)
outputs/mc_summary_python.csv   describe() of the 10,000 simulated net values and ROIs
outputs/fig_1_net_value_distribution.png
outputs/fig_2_tornado_sensitivity.png
templates/roi_inputs_template.csv  fill-in sheet for your own agent business case
```

## Pitfalls this example is built to show

- **Escalations are not neutral.** An escalated ticket costs more human time than a fresh one, because someone has to read the agent's transcript first.
- **Correlated inputs.** This model treats inputs as independent. In reality, low adoption and a high review share can arrive together, for example after a visible error. Positive correlation between bad outcomes widens the loss tail, so treat P(loss) here as a likely lower estimate.
- **Hours saved that are never redeployed.** The model values freed minutes at the full loaded cost. If freed time is not redeployed, multiply the benefit by a realization share (the template has a row for it).
- **A decision model is not a causal estimate.** The simulation shows what the inputs imply. It does not prove the agent caused anything. Causal evidence on time saved comes from designs like A01's difference-in-differences.

## Connect to the article

The edition argues that the vendor ROI ("tasks automated × labour cost ÷ licence fee") answers the wrong question. A CFO needs a range and a probability of losing money. This folder is the worked example behind that argument. Step 1 reproduces the roughly 5× vendor number. Step 2 rebuilds it from the per-ticket equation in the article's *Under the Hood* section and runs 10,000 scenarios. Step 3 shows the method lands on the known truth. Step 4 turns the result into the numbers a decision needs: P(ROI < 0), the median and interval, the inputs worth measuring better, and the break-even adoption level. `fig_1` is the chart in the article. Use `templates/roi_inputs_template.csv` to collect the ranges for your own agent business case.

GitHub: https://github.com/eddie-2030/measure-digital-transformation/tree/main/A02_the-full-roi-equation-for-ai
