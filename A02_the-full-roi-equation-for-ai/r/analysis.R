# analysis.R -- A02 "The Full ROI Equation for AI Agents" (R version)
#
# Same four steps and the same numbers as python/analysis.py, in base R only
# (no packages). Run from the edition folder:   Rscript r/analysis.R
#
#   Step 1  The naive "vendor" ROI and why it misleads
#   Step 2  The full cost-benefit model, with Monte Carlo uncertainty
#   Step 3  Check against the ground truth (possible only with synthetic data)
#   Step 4  Decision implication: P(ROI < 0), median, intervals, drivers,
#           break-even
#
# How R and Python agree exactly: both read the same 10,000 x 25 uniform
# random numbers from data/mc_uniforms.csv and turn them into draws with the
# same inverse-CDF maths (qbeta, qnorm, triangular). So even the Monte Carlo
# summaries match, not just the deterministic numbers.
#
# If outputs/results_long.csv (written by Python) exists, this script adds an
# "r" value to every result, checks agreement at the stated rounding, and
# rewrites outputs/results.json with both "python" and "r" keys.
#
# Data are synthetic (see data/generate_data.py).

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
ROOT <- if (length(file_arg)) normalizePath(file.path(dirname(file_arg), "..")) else getwd()
DATA <- file.path(ROOT, "data"); OUT <- file.path(ROOT, "outputs")
dir.create(OUT, showWarnings = FALSE)

QUEUES <- c("access", "it_software", "hr_policy")
LOGGED_WEEKS <- 26; HORIZON_WEEKS <- 52
R_RES <- c()                                  # name -> unrounded value
rec <- function(name, value) R_RES[[name]] <<- as.numeric(value)

tri_ppf <- function(u, low, mode, high) {
  # Inverse CDF of a triangular(low, mode, high) distribution
  if (high == low) return(rep(low, length(u)))
  cut <- (mode - low) / (high - low)
  ifelse(u < cut, low + sqrt(u * (high - low) * (mode - low)),
         high - sqrt((1 - u) * (high - low) * (high - mode)))
}
q <- function(x, p) unname(quantile(x, p, type = 7))   # same as numpy default

# =============================================================================
# 0. Load and clean the log
# =============================================================================
raw <- read.csv(file.path(DATA, "helpdesk_tickets.csv"), na.strings = c("", "NA"),
                stringsAsFactors = FALSE)
costs <- read.csv(file.path(DATA, "cost_assumptions.csv"), stringsAsFactors = FALSE)
rownames(costs) <- costs$parameter
gt <- read.csv(file.path(DATA, "ground_truth.csv"), stringsAsFactors = FALSE)
truth <- setNames(gt$true_value, gt$parameter)
U <- read.csv(file.path(DATA, "mc_uniforms.csv"))

log <- raw[!duplicated(raw), ]                       # messy case: double export
stopifnot(!any(duplicated(log$ticket_id)))
agent <- log[log$routed_to_agent == 1, ]
agent_known <- agent[!is.na(agent$outcome), ]       # assume missing at random
cat(strrep("=", 72), "\nDATA CHECK\n")
cat(sprintf("  rows in file: %d   exact duplicates removed: %d\n", nrow(raw), nrow(raw) - nrow(log)))
cat(sprintf("  unique tickets: %d   routed: %d   missing outcome: %d\n",
            nrow(log), nrow(agent), sum(is.na(agent$outcome))))

weekly_n <- as.numeric(table(log$week))
vol_mean <- mean(weekly_n)
vol_se <- sd(weekly_n) / sqrt(LOGGED_WEEKS)
last4 <- log[log$week > LOGGED_WEEKS - 4, ]
last4_share <- mean(last4$routed_to_agent)
a4 <- last4[last4$routed_to_agent == 1, ]
mix <- sapply(QUEUES, function(qq) mean(a4$queue == qq))
h1_agent <- sapply(QUEUES, function(qq) sum(agent$queue == qq))
h1_share <- nrow(agent) / nrow(log)
mode_of <- function(p) costs[p, "most_likely"]

# =============================================================================
# STEP 1. Naive "vendor" ROI
# =============================================================================
wage_ml <- mode_of("loaded_hourly_cost")
licence_12m <- 12 * mode_of("license_fee_per_month")
ho <- log[!is.na(log$outcome) & log$outcome == "human_only" & !is.na(log$human_minutes), ]
base_simple <- sapply(QUEUES, function(qq) mean(ho$human_minutes[ho$queue == qq]))
naive_tickets <- HORIZON_WEEKS * vol_mean * last4_share
naive_minutes <- sum(mix * base_simple)
naive_gross <- naive_tickets * naive_minutes * wage_ml / 60
naive_net <- naive_gross - licence_12m
naive_roi <- naive_net / licence_12m
cat(strrep("=", 72), "\nSTEP 1  Naive (vendor-style) ROI\n")
cat(sprintf("  last-4-week share routed %.3f; gross $%s; naive ROI %.2fx\n",
            last4_share, format(round(naive_gross), big.mark = ","), naive_roi))
rec("naive_share_routed_last4", last4_share); rec("naive_gross_savings_12m", naive_gross)
rec("naive_net_value_12m", naive_net); rec("naive_roi", naive_roi)

# =============================================================================
# STEP 2. Full model
# =============================================================================
cnt <- function(o) sapply(QUEUES, function(qq) sum(agent_known$queue == qq & agent_known$outcome == o))
n_res <- cnt("resolved_by_agent"); n_esc <- cnt("escalated"); n_rw <- cnt("reopened")
n_tot <- n_res + n_esc + n_rw
p_esc_hat <- n_esc / n_tot; p_rw_hat <- n_rw / n_tot

qa <- agent_known[agent_known$outcome == "resolved_by_agent" & agent_known$qa_sampled == 1, ]
qa_n <- sapply(QUEUES, function(qq) sum(qa$queue == qq))
qa_e <- sapply(QUEUES, function(qq) sum(qa$qa_error_found[qa$queue == qq]))
p_err_hat <- qa_e / qa_n

# Human minutes: minutes = base(queue) + extra_esc * escalated + extra_rw * reopened
hh <- log[!is.na(log$outcome) & log$outcome %in% c("human_only", "escalated", "reopened") &
          !is.na(log$human_minutes), ]
X <- cbind(sapply(QUEUES, function(qq) as.numeric(hh$queue == qq)),
           escalated = as.numeric(hh$outcome == "escalated"),
           reopened = as.numeric(hh$outcome == "reopened"))
fit <- lm(hh$human_minutes ~ 0 + X)
coef <- unname(coef(fit)); V <- unname(vcov(fit))

run_mean <- sapply(QUEUES, function(qq) mean(agent$agent_cost_usd[agent$queue == qq]))
run_se <- sapply(QUEUES, function(qq) { x <- agent$agent_cost_usd[agent$queue == qq]; sd(x) / sqrt(length(x)) })

model <- function(p) {
  # 12-month net value for a list of inputs (scalars or vectors of draws).
  # Weeks 1-26: agent tickets actually logged. Weeks 27-52: projected =
  # 26 x weekly volume x steady-state routing share x last-4-week queue mix.
  w <- p$wage / 60
  o <- list(n_agent = 0, gross_all = 0, benefit = 0, unresolved = 0, handoff = 0,
            remediation = 0, review = 0, run = 0)
  for (i in seq_along(QUEUES)) {
    n_q <- h1_agent[i] + (HORIZON_WEEKS - LOGGED_WEEKS) * p$volume * p$s2 * mix[i]
    pe <- p$p_esc[[i]]; pr <- p$p_rw[[i]]; p_res <- 1 - pe - pr; base <- p$base[[i]]
    o$n_agent <- o$n_agent + n_q
    o$gross_all <- o$gross_all + n_q * base * w
    o$benefit <- o$benefit + n_q * p_res * base * w
    o$unresolved <- o$unresolved + n_q * (1 - p_res) * base * w
    o$handoff <- o$handoff + n_q * w * (pe * p$extra_esc + pr * p$extra_rw)
    o$remediation <- o$remediation + n_q * p_res * p$p_err[[i]] * p$c_rem
    o$review <- o$review + n_q * p$review_share * p$review_min * w
    o$run <- o$run + n_q * p$run[[i]]
  }
  o$licence <- 12 * p$licence
  o$build_monitor <- p$build + 12 * p$monitor
  o$var_cost <- o$handoff + o$remediation + o$review + o$run
  o$total_cost <- o$var_cost + o$licence + o$build_monitor
  o$net <- o$benefit - o$total_cost
  o$roi <- o$net / o$total_cost
  lapply(o, unname)
}

point <- list(
  p_esc = as.list(p_esc_hat), p_rw = as.list(p_rw_hat), p_err = as.list(p_err_hat),
  base = as.list(coef[1:3]), extra_esc = coef[4], extra_rw = coef[5],
  run = as.list(run_mean), volume = vol_mean,
  wage = mode_of("loaded_hourly_cost"), c_rem = mode_of("remediation_cost_per_error"),
  review_share = mode_of("review_share"), review_min = mode_of("review_minutes"),
  build = mode_of("build_cost"), monitor = mode_of("monitoring_cost_per_month"),
  licence = mode_of("license_fee_per_month"), s2 = mode_of("steady_state_routing_share"))
pt <- model(point)

# Monte Carlo from the shared uniforms
tri <- function(name, u) tri_ppf(u, costs[name, "low"], costs[name, "most_likely"], costs[name, "high"])
mc_esc <- sapply(QUEUES, function(qq) qbeta(U[[paste0("u_esc_", qq)]], 1 + n_esc[qq], 2 + n_res[qq] + n_rw[qq]))
mc_rw <- sapply(seq_along(QUEUES), function(i) { qq <- QUEUES[i]
  (1 - mc_esc[, i]) * qbeta(U[[paste0("u_rw_", qq)]], 1 + n_rw[qq], 1 + n_res[qq]) })
mc_err <- sapply(QUEUES, function(qq) qbeta(U[[paste0("u_err_", qq)]], 1 + qa_e[qq], 1 + qa_n[qq] - qa_e[qq]))
Z <- qnorm(as.matrix(U[, paste0("u_minutes_", 1:5)]))
mc_coef <- sweep(Z %*% chol(V), 2, coef, "+")       # chol(V) = t(L): same as Python
draws <- list(
  p_esc = lapply(1:3, function(i) mc_esc[, i]), p_rw = lapply(1:3, function(i) mc_rw[, i]),
  p_err = lapply(1:3, function(i) mc_err[, i]),
  base = lapply(1:3, function(i) mc_coef[, i]), extra_esc = mc_coef[, 4], extra_rw = mc_coef[, 5],
  run = lapply(seq_along(QUEUES), function(i) run_mean[i] + run_se[i] * qnorm(U[[paste0("u_run_", QUEUES[i])]])),
  volume = vol_mean + vol_se * qnorm(U$u_volume),
  wage = tri("loaded_hourly_cost", U$u_loaded_hourly_cost),
  c_rem = tri("remediation_cost_per_error", U$u_remediation_cost_per_error),
  review_share = tri("review_share", U$u_review_share),
  review_min = tri("review_minutes", U$u_review_minutes),
  build = tri("build_cost", U$u_build_cost),
  monitor = tri("monitoring_cost_per_month", U$u_monitoring_cost_per_month),
  licence = mode_of("license_fee_per_month"),
  s2 = tri("steady_state_routing_share", U$u_steady_state_routing_share))
mc <- model(draws)
net <- mc$net; roi <- mc$roi

cat(strrep("=", 72), "\nSTEP 2  Full cost-benefit model\n")
cat(sprintf("  point estimate: net $%s, ROI %.2fx\n", format(round(pt$net), big.mark = ","), pt$roi))
cat(sprintf("  Monte Carlo: median net $%s, P(ROI<0) = %.1f%%\n",
            format(round(q(net, .5)), big.mark = ","), 100 * mean(roi < 0)))

for (i in seq_along(QUEUES)) {
  qq <- QUEUES[i]
  rec(paste0("p_escalated_", qq), p_esc_hat[i]); rec(paste0("p_reopened_", qq), p_rw_hat[i])
  rec(paste0("p_error_", qq), p_err_hat[i]); rec(paste0("base_minutes_", qq), coef[i])
}
rec("extra_minutes_escalated", coef[4]); rec("extra_minutes_reopened", coef[5])
rec("p_resolved_overall", sum(n_res) / sum(n_tot))
rec("agent_tickets_12m_point", pt$n_agent); rec("h1_share_routed", h1_share)
rec("bridge_vendor_gross", naive_gross)
rec("bridge_adoption_ramp", pt$gross_all - naive_gross)
rec("bridge_not_resolved", -pt$unresolved); rec("bridge_handoff_rework", -pt$handoff)
rec("bridge_review", -pt$review); rec("bridge_remediation", -pt$remediation)
rec("bridge_run_cost", -pt$run); rec("bridge_licence", -pt$licence)
rec("bridge_build_monitoring", -pt$build_monitor)
rec("full_net_value_12m_point", pt$net); rec("full_roi_point", pt$roi)
rec("full_total_cost_12m_point", pt$total_cost); rec("full_benefit_12m_point", pt$benefit)
rec("net_per_agent_ticket_point", pt$net / pt$n_agent)

# =============================================================================
# STEP 3. Ground truth check
# =============================================================================
true_net <- truth[["net_value_12m"]]; true_roi <- truth[["roi_12m"]]
lo80 <- q(net, .1); hi80 <- q(net, .9); lo90 <- q(net, .05); hi90 <- q(net, .95)
cover <- 0
for (i in seq_along(QUEUES)) {
  for (lab in c("escalated", "reopened", "error")) {
    d <- switch(lab, escalated = mc_esc[, i], reopened = mc_rw[, i], error = mc_err[, i])
    t <- truth[[paste0("p_", lab, "_", QUEUES[i])]]
    cover <- cover + (q(d, .025) <= t && t <= q(d, .975))
  }
}
cat(strrep("=", 72), "\nSTEP 3  Ground truth\n")
cat(sprintf("  true net $%s inside 80%%: %s, 90%%: %s; rates covered %d of 9\n",
            format(round(true_net), big.mark = ","), lo80 <= true_net & true_net <= hi80,
            lo90 <= true_net & true_net <= hi90, cover))
rec("true_net_value_12m", true_net); rec("true_roi", true_roi)
rec("true_inside_80pct_interval", as.numeric(lo80 <= true_net & true_net <= hi80))
rec("true_inside_90pct_interval", as.numeric(lo90 <= true_net & true_net <= hi90))
rec("true_net_percentile_in_mc", mean(net <= true_net))
rec("rates_covered_by_95pct", cover)
rec("naive_overstatement_factor", naive_roi / true_roi)

# =============================================================================
# STEP 4. Decision implication
# =============================================================================
with_ <- function(...) { d <- point; kv <- list(...); for (k in names(kv)) d[[k]] <- kv[[k]]; model(d)$net }
tri_at <- function(name, u) tri(name, u)
se <- sqrt(diag(V))
groups <- list(
  "Adoption (share routed)" = function(u) with_(s2 = tri_at("steady_state_routing_share", u)),
  "Remediation cost per error" = function(u) with_(c_rem = tri_at("remediation_cost_per_error", u)),
  "Human review share" = function(u) with_(review_share = tri_at("review_share", u)),
  "Build cost (one-off)" = function(u) with_(build = tri_at("build_cost", u)),
  "Loaded labour cost" = function(u) with_(wage = tri_at("loaded_hourly_cost", u)),
  "Review minutes per review" = function(u) with_(review_min = tri_at("review_minutes", u)),
  "Monitoring cost" = function(u) with_(monitor = tri_at("monitoring_cost_per_month", u)),
  "Silent-error rates" = function(u) with_(p_err = as.list(qbeta(u, 1 + qa_e, 1 + qa_n - qa_e))),
  "Escalation rates" = function(u) with_(p_esc = as.list(qbeta(u, 1 + n_esc, 2 + n_res + n_rw))),
  "Rework rates" = function(u) with_(p_rw = as.list(qbeta(u, 1 + n_rw, 2 + n_res + n_esc))),
  "Handoff & rework minutes" = function(u) with_(extra_esc = coef[4] + se[4] * qnorm(u),
                                                 extra_rw = coef[5] + se[5] * qnorm(u)),
  "Baseline human minutes" = function(u) with_(base = as.list(coef[1:3] + se[1:3] * qnorm(u))),
  "Ticket volume" = function(u) with_(volume = vol_mean + vol_se * qnorm(u)),
  "Agent run cost" = function(u) with_(run = as.list(run_mean + run_se * qnorm(u))))
tor <- data.frame(input = names(groups),
                  low_case = sapply(groups, function(f) f(0.10)),
                  high_case = sapply(groups, function(f) f(0.90)), row.names = NULL)
tor$swing <- abs(tor$high_case - tor$low_case)
tor <- tor[order(-tor$swing), ]; rownames(tor) <- NULL
write.csv(tor, file.path(OUT, "tornado_r.csv"), row.names = FALSE)

breakeven <- function(key) { a <- do.call(with_, setNames(list(0), key)); b <- do.call(with_, setNames(list(1), key)); -a / (b - a) }
be_s2 <- breakeven("s2"); be_rem <- breakeven("c_rem"); be_review <- breakeven("review_share")

cat(strrep("=", 72), "\nSTEP 4  Decision implication\n")
cat(sprintf("  P(ROI<0) %.1f%%; median net $%s; 80%% [%s, %s]; median ROI %.2f\n",
            100 * mean(roi < 0), format(round(q(net, .5)), big.mark = ","),
            format(round(lo80), big.mark = ","), format(round(hi80), big.mark = ","), q(roi, .5)))
print(head(tor, 5))
cat(sprintf("  break-even adoption %.3f, remediation $%.0f, review share %.3f\n", be_s2, be_rem, be_review))

rec("p_roi_negative", mean(roi < 0)); rec("mc_median_net_value_12m", q(net, .5))
rec("mc_mean_net_value_12m", mean(net))
rec("mc_net_p10", lo80); rec("mc_net_p90", hi80); rec("mc_net_p05", lo90); rec("mc_net_p95", hi90)
rec("mc_median_roi", q(roi, .5)); rec("mc_roi_p10", q(roi, .1)); rec("mc_roi_p90", q(roi, .9))
rec("mc_roi_p05", q(roi, .05)); rec("mc_roi_p95", q(roi, .95))
rec("p_roi_above_0_5", mean(roi > 0.5))
rec("breakeven_steady_state_share", be_s2); rec("breakeven_remediation_cost", be_rem)
rec("breakeven_review_share", be_review)
for (i in 1:3) {
  rec(sprintf("tornado_rank%d_swing", i), tor$swing[i])
  rec(sprintf("tornado_rank%d_low_case", i), tor$low_case[i])
  rec(sprintf("tornado_rank%d_high_case", i), tor$high_case[i])
}

# =============================================================================
# Compare with Python and write results.json (base R JSON writer)
# =============================================================================
r_df <- data.frame(name = names(R_RES), r_exact = unname(unlist(R_RES)))
write.csv(r_df, file.path(OUT, "results_r.csv"), row.names = FALSE)

py_path <- file.path(OUT, "results_long.csv")
if (!file.exists(py_path)) {
  cat("\nPython results not found; wrote outputs/results_r.csv only. Run python/analysis.py first to compare.\n")
  quit(status = 0)
}
py <- read.csv(py_path, stringsAsFactors = FALSE)
stopifnot(setequal(py$name, r_df$name))
py$r_exact <- r_df$r_exact[match(py$name, r_df$name)]
py$r <- round(py$r_exact, py$digits) + 0          # + 0 avoids "-0"
py$agree <- abs(py$r - py$value) < 1e-9
py$abs_diff_exact <- abs(py$r_exact - py$exact)
tor_py <- read.csv(file.path(OUT, "tornado_python.csv"), stringsAsFactors = FALSE)

cat(strrep("=", 72), "\nAGREEMENT CHECK (Python vs R)\n")
cat(sprintf("  %d of %d results identical at the stated rounding\n", sum(py$agree), nrow(py)))
cat(sprintf("  largest unrounded difference: %.3g (%s)\n", max(py$abs_diff_exact),
            py$name[which.max(py$abs_diff_exact)]))
cat(sprintf("  tornado order identical: %s\n", identical(tor_py$input, tor$input)))
if (!all(py$agree)) print(py[!py$agree, c("name", "value", "r", "exact", "r_exact")])

esc <- function(s) paste0('"', gsub('"', '\\\\"', gsub("\\\\", "\\\\\\\\", s)), '"')
num <- function(x) if (is.na(x)) "null" else format(x, digits = 15, scientific = FALSE, trim = TRUE)
items <- vapply(seq_len(nrow(py)), function(i) {
  r <- py[i, ]
  paste0('    {"name": ', esc(r$name), ', "value": ', num(r$value), ', "digits": ', r$digits,
         ', "unit": ', esc(r$unit), ', "meaning": ', esc(r$meaning),
         ', "headline": ', tolower(as.character(r$headline)),
         ', "python": ', num(r$value), ', "r": ', num(r$r),
         ', "agree": ', tolower(as.character(r$agree)), '}')
}, "")
json <- paste0(
  '{\n  "edition": "A02 The Full ROI Equation for AI Agents",\n',
  '  "note": "Synthetic data. value = python = r, rounded to digits (2 -> 0.01, -3 -> nearest 1,000). ',
  'Both languages read the same Monte Carlo uniforms (data/mc_uniforms.csv), so simulation results match exactly.",\n',
  '  "agreement": {"n_results": ', nrow(py), ', "n_agree": ', sum(py$agree),
  ', "max_abs_diff_unrounded": ', num(max(py$abs_diff_exact)),
  ', "tornado_order_identical": ', tolower(identical(tor_py$input, tor$input)), '},\n',
  '  "results": [\n', paste(items, collapse = ",\n"), '\n  ]\n}\n')
writeLines(json, file.path(OUT, "results.json"))
cat("  wrote outputs/results.json with python and r keys\n")
