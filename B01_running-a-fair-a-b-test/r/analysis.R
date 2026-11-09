# analysis.R  --  B01 "Running a Fair A/B Test Inside Your Company"   (SYNTHETIC data)
#
# The same four steps and the same numbers as python/analysis.py, in base R
# (plus jsonlite to read and write outputs/results.json).
#
# Run from the package root, AFTER python3 python/analysis.py:
#     Rscript r/analysis.R
# The script recomputes every result, writes its value into the "r" key of
# outputs/results.json, and stops with an error if any number disagrees with
# Python at the stated rounding.

suppressMessages(library(jsonlite))

MIN_POST_WEEKS <- 4      # pre-registered: need >= 4 valid post weeks to have an outcome
QUALITY_MARGIN <- -1.0   # pre-registered guardrail: quality may not drop by more than 1 point
MIN_WORTHWHILE <- 0.0    # pre-registered: roll out only if the CI lower bound is above this

res <- list()            # name -> value

# =============================================================================
# Step 0. Cleaning -- explicit, identical to the Python script
# =============================================================================
clean <- function(df) {
  log <- list(n_rows_raw = nrow(df))
  # Rule 1: drop exact duplicate rows (an export glitch); then one row per person.
  df <- df[!duplicated(df), ]
  rownames(df) <- NULL
  log$n_duplicates_removed <- log$n_rows_raw - nrow(df)
  stopifnot(!any(duplicated(df$employee_id)))

  post_cols <- paste0("cases_post_w", 1:8)
  pre_cols  <- paste0("cases_pre_w", 1:4)
  post <- as.matrix(df[, post_cols]) * 1.0

  # Rule 2: a team-week where EVERY recorded value is 0 is a data-feed outage,
  # not a week of zero work -> set those cells to missing.
  n_outage <- 0
  for (tm in unique(df$team)) {
    idx <- which(df$team == tm)
    for (w in seq_len(ncol(post))) {
      vals <- post[idx, w]
      obs <- !is.na(vals)
      if (sum(obs) > 0 && all(vals[obs] == 0)) {
        post[idx[obs], w] <- NA
        n_outage <- n_outage + sum(obs)
      }
    }
  }
  log$n_outage_cells_set_missing <- n_outage

  # Rule 3: outcome = mean of valid post weeks; need >= MIN_POST_WEEKS of them.
  n_valid <- rowSums(!is.na(post))
  df$cases_post <- ifelse(n_valid > 0, rowSums(post, na.rm = TRUE) / pmax(n_valid, 1), NA)
  df$n_valid_post_weeks <- n_valid
  df$cases_pre <- rowMeans(df[, pre_cols])
  log$n_leavers <- sum(!is.na(df$left_week))
  keep <- n_valid >= MIN_POST_WEEKS
  log$n_excluded_short_post <- sum(!keep)
  list(analysis = df[keep, ], all = df, log = log)
}

ols <- function(formula, data, term) {
  # estimate, classical SE and t-based 95% CI for one coefficient
  fit <- lm(formula, data = data)
  ci <- confint(fit, term, level = 0.95)
  list(est = unname(coef(fit)[term]),
       se = unname(summary(fit)$coefficients[term, "Std. Error"]),
       lo = ci[1, 1], hi = ci[1, 2])
}

vol_raw <- read.csv("data/volunteer_pilot.csv", stringsAsFactors = FALSE)
rnd_raw <- read.csv("data/randomized_pilot.csv", stringsAsFactors = FALSE)
truth   <- read.csv("data/ground_truth.csv", stringsAsFactors = FALSE)
rerand  <- read.csv("data/rerandomizations.csv", stringsAsFactors = FALSE)

vol <- clean(vol_raw)$analysis
cr  <- clean(rnd_raw)
rnd <- cr$analysis
rnd_all <- cr$all
res <- c(res, cr$log)
res$n_analysis <- nrow(rnd)
res$n_analysis_treated <- sum(rnd$assigned)
res$n_analysis_control <- sum(1 - rnd$assigned)
res$attrition_treated_pct <- 100 * mean(rnd_all$n_valid_post_weeks[rnd_all$assigned == 1] < MIN_POST_WEEKS)
res$attrition_control_pct <- 100 * mean(rnd_all$n_valid_post_weeks[rnd_all$assigned == 0] < MIN_POST_WEEKS)
cat(sprintf("Step 0 - analyzed %d of %d rows (duplicates %d, outage cells %d, excluded %d)\n",
            res$n_analysis, res$n_rows_raw, res$n_duplicates_removed,
            res$n_outage_cells_set_missing, res$n_excluded_short_post))

# =============================================================================
# Step 1. Naive: the volunteer pilot
# =============================================================================
res$vol_share_volunteered_pct <- 100 * mean(vol$volunteered)
res$vol_baseline_gap <- mean(vol$cases_pre[vol$volunteered == 1]) - mean(vol$cases_pre[vol$volunteered == 0])
v_naive <- ols(cases_post ~ volunteered, vol, "volunteered")
res$vol_naive_estimate <- v_naive$est
res$vol_naive_ci_lo <- v_naive$lo
res$vol_naive_ci_hi <- v_naive$hi
# adjusting for the baseline does not rescue a self-selected comparison
res$vol_ancova_estimate <- ols(cases_post ~ volunteered + cases_pre + factor(site), vol, "volunteered")$est
vt <- merge(vol, truth, by = "employee_id")
res$vol_true_effect <- mean((vt$true_effect_if_used * vt$used_tool)[vt$volunteered == 1])
res$vol_overstatement_ratio <- res$vol_naive_estimate / res$vol_true_effect
cat(sprintf("Step 1 - volunteer naive: %.2f (truth for volunteers %.2f)\n",
            v_naive$est, res$vol_true_effect))

# =============================================================================
# Step 2. Correct: stratified randomized pilot, analyzed as randomized (ITT)
# =============================================================================
tr <- rnd[rnd$assigned == 1, ]
co <- rnd[rnd$assigned == 0, ]
res$balance_pre_diff <- mean(tr$cases_pre) - mean(co$cases_pre)
res$balance_pre_std_diff <- res$balance_pre_diff / sqrt((var(tr$cases_pre) + var(co$cases_pre)) / 2)
res$control_pre_mean <- mean(co$cases_pre)
res$control_post_mean <- mean(co$cases_post)
res$rho_pre_post <- cor(co$cases_pre, co$cases_post)

simple <- ols(cases_post ~ assigned, rnd, "assigned")
ancova <- ols(cases_post ~ assigned + cases_pre + factor(stratum), rnd, "assigned")
for (nm in c("simple", "ancova")) {
  r <- get(nm)
  res[[paste0(nm, "_estimate")]] <- r$est
  res[[paste0(nm, "_se")]] <- r$se
  res[[paste0(nm, "_ci_lo")]] <- r$lo
  res[[paste0(nm, "_ci_hi")]] <- r$hi
}
# Sensitivity (printed only, NOT in results.json): heteroskedasticity-robust HC2 SE for the
# headline ANCOVA, as Lin (2013) recommends. Computed in base R: sandwich with e^2 / (1 - h).
fit_a <- lm(cases_post ~ assigned + cases_pre + factor(stratum), data = rnd)
X <- model.matrix(fit_a); e <- residuals(fit_a); h <- hatvalues(fit_a)
bread <- solve(crossprod(X))
V_hc2 <- bread %*% crossprod(X * (e^2 / (1 - h)), X) %*% bread
se_hc2 <- sqrt(V_hc2["assigned", "assigned"])
tq <- qt(0.975, fit_a$df.residual)
cat(sprintf("Sensitivity - ANCOVA with HC2 robust SE: %.2f, SE %.3f, 95%% CI [%.2f, %.2f] (classical SE %.3f)\n",
            ancova$est, se_hc2, ancova$est - tq * se_hc2, ancova$est + tq * se_hc2, ancova$se))
res$simple_ci_width <- simple$hi - simple$lo
res$ancova_ci_width <- ancova$hi - ancova$lo
res$se_reduction_pct <- 100 * (1 - ancova$se / simple$se)
res$variance_ratio <- (ancova$se / simple$se)^2
res$theory_variance_factor <- 1 - res$rho_pre_post^2
res$equivalent_n_unadjusted <- nrow(rnd) / res$variance_ratio
res$itt_pct_lift <- 100 * ancova$est / res$control_post_mean

res$uptake_pct <- 100 * mean(tr$used_tool)
res$effect_among_users <- ancova$est / mean(tr$used_tool)   # ITT / uptake

res$as_treated_estimate <- ols(cases_post ~ used_tool + cases_pre + factor(stratum), rnd, "used_tool")$est
post_adj <- ols(cases_post ~ assigned + cases_pre + factor(stratum) + ai_hours_per_week, rnd, "assigned")
res$posttreat_adjusted_estimate <- post_adj$est
res$posttreat_adjusted_ci_lo <- post_adj$lo
res$posttreat_adjusted_ci_hi <- post_adj$hi

q <- ols(quality_post ~ assigned + quality_pre + factor(stratum),
         rnd[!is.na(rnd$quality_post), ], "assigned")
res$quality_estimate <- q$est
res$quality_ci_lo <- q$lo
res$quality_ci_hi <- q$hi
res$quality_noninferior <- as.integer(q$lo > QUALITY_MARGIN)
cat(sprintf("Step 2 - simple %.2f [%.2f, %.2f]; ANCOVA %.2f [%.2f, %.2f]\n",
            simple$est, simple$lo, simple$hi, ancova$est, ancova$lo, ancova$hi))

# =============================================================================
# Step 3. Check against the ground truth (500 re-randomizations)
# =============================================================================
rt <- merge(rnd, truth, by = "employee_id")
rt <- rt[order(rt$employee_id), ]
res$true_effect_if_used <- truth$true_effect_if_used[1]
res$true_itt_design <- res$true_effect_if_used * mean(truth$complier)
res$true_itt_sample <- mean(rt$true_effect_if_used * rt$complier)
truth_sim <- res$true_itt_sample

tau <- rt$true_effect_if_used * rt$complier
y0  <- rt$cases_post - rt$true_effect_if_used * rt$used_tool     # outcome without access
Z   <- as.matrix(rerand[match(rt$employee_id, rerand$employee_id), -1]) * 1.0
Y   <- y0 + tau * Z                                               # n x 500 outcomes
n   <- nrow(rt)

sim_fit <- function(W) {
  # Frisch-Waugh-Lovell: residualize assignment and outcome on the controls W
  proj <- function(M) M - W %*% solve(crossprod(W), crossprod(W, M))
  Zt <- proj(Z); Yt <- proj(Y)
  szz <- colSums(Zt^2)
  b <- colSums(Zt * Yt) / szz
  df_res <- n - ncol(W) - 1
  rss <- colSums((Yt - sweep(Zt, 2, b, "*"))^2)
  se <- sqrt(rss / df_res / szz)
  cover <- abs(b - truth_sim) <= qt(0.975, df_res) * se
  list(b = b, se = se, cover = cover)
}
W_simple <- matrix(1, n, 1)
W_ancova <- model.matrix(~ cases_pre + factor(stratum), data = rt)
for (nm in c("simple", "ancova")) {
  s <- sim_fit(if (nm == "simple") W_simple else W_ancova)
  res[[paste0("sim_", nm, "_mean_estimate")]] <- mean(s$b)
  res[[paste0("sim_", nm, "_bias")]] <- mean(s$b) - truth_sim
  res[[paste0("sim_", nm, "_empirical_sd")]] <- sd(s$b)
  res[[paste0("sim_", nm, "_mean_se")]] <- mean(s$se)
  res[[paste0("sim_", nm, "_coverage_pct")]] <- 100 * mean(s$cover)
}
res$sim_n_reps <- ncol(Z)
res$sim_sd_ratio <- res$sim_ancova_empirical_sd / res$sim_simple_empirical_sd
cat(sprintf("Step 3 - truth %.2f; bias simple %+.3f, ANCOVA %+.3f; coverage %.1f%% / %.1f%%\n",
            truth_sim, res$sim_simple_bias, res$sim_ancova_bias,
            res$sim_simple_coverage_pct, res$sim_ancova_coverage_pct))

# =============================================================================
# Step 4. Decision implication (rule fixed before the pilot)
# =============================================================================
res$decision_rollout <- as.integer(ancova$lo > MIN_WORTHWHILE && res$quality_noninferior == 1)
res$weekly_cases_gain_1200 <- ancova$est * 1200
cat(if (res$decision_rollout == 1) "Step 4 - decision: roll out\n" else "Step 4 - decision: do not roll out\n")

# =============================================================================
# Write R values into results.json and confirm agreement with Python
# =============================================================================
js <- fromJSON("outputs/results.json", simplifyVector = FALSE)
mismatch <- character(0)
for (i in seq_along(js$results)) {
  item <- js$results[[i]]
  nm <- item$name
  if (is.null(res[[nm]])) stop("R did not compute ", nm)
  d <- item$rounding
  # both languages round the same doubles, so equal inputs give equal rounded values
  rv <- round(as.numeric(res[[nm]]), d) + 0
  js$results[[i]]$r <- if (d <= 0) as.integer(rv) else rv
  if (is.null(item$python) || abs(rv - item$python) > 1e-9) mismatch <- c(mismatch, nm)
}
write(toJSON(js, auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null"),
      "outputs/results.json")
if (length(mismatch) > 0) {
  stop("Python and R disagree on: ", paste(mismatch, collapse = ", "))
}
cat(sprintf("All %d results agree with Python at the stated rounding. Wrote outputs/results.json\n",
            length(js$results)))

# Full-precision comparison (how close the two languages really are)
if (file.exists("outputs/results_python_full_precision.json")) {
  pf <- fromJSON("outputs/results_python_full_precision.json")
  diffs <- sapply(names(pf), function(nm) abs(pf[[nm]] - as.numeric(res[[nm]])))
  cat(sprintf("Largest full-precision difference vs Python: %.2e (%s)\n",
              max(diffs), names(diffs)[which.max(diffs)]))
}
