# analysis.R -- C01 "Who Audits the Agents? Human Oversight as a Measurable Capability"
#
# Run from the code folder root:   Rscript r/analysis.R
# Base R only (no packages). Mirrors python/analysis.py step by step and writes
# outputs/r_headline.csv, which python/analysis.py merges into outputs/results.json.
#
#   Step 1  Naive: raw percent agreement (and why it misleads)
#   Step 2  Correct: Cohen's kappa (pairs) and Krippendorff's alpha (many raters, missing data)
#   Step 3  Check against ground truth: golden set and the true simulation parameters
#   Step 4  Decision implication: errors that slip through review, per site

sites <- c("A", "B", "C")

# ---------------------------------------------------------------------------
# Step 0. Load and clean the review log
# ---------------------------------------------------------------------------
raw <- read.csv("data/oversight_reviews.csv", colClasses = "character", na.strings = character(0))
raw$refund_amount_usd <- as.numeric(raw$refund_amount_usd)

clean_label <- function(x) {
  # Map messy labels to "Correct"/"Error"; blank, "Skipped", "Unsure" become NA.
  x <- tolower(trimws(x))
  ifelse(startsWith(x, "corr"), "Correct", ifelse(startsWith(x, "err"), "Error", NA))
}
df <- raw
df$label <- clean_label(df$review_label)
n_rows_raw <- nrow(df)
df <- df[!duplicated(df[, c("decision_id", "reviewer_id")]), ]   # double-submitted reviews
n_duplicates <- n_rows_raw - nrow(df)
n_missing <- sum(is.na(df$label))
df <- df[!is.na(df$label), ]
df$is_error <- as.integer(df$label == "Error")
gold <- df[df$case_source == "golden_seeded", ]   # expert-verified, seeded blind
df <- df[df$case_source == "production", ]        # live decisions: agreement stats use these
cat(sprintf("Rows: %d raw, %d duplicates dropped, %d unusable labels dropped.\n",
            n_rows_raw, n_duplicates, n_missing))

# ---------------------------------------------------------------------------
# Krippendorff's alpha (nominal), written out in base R
# ---------------------------------------------------------------------------
# Coincidence matrix: in a decision with m ratings, each ordered pair of different
# ratings adds 1/(m-1). Observed disagreement = off-diagonal share; expected
# disagreement = what random pairing of the same values would give.
# alpha = 1 - observed / expected.
kalpha <- function(d) {
  cats <- c("Correct", "Error")
  o <- matrix(0, 2, 2, dimnames = list(cats, cats))
  for (vals in split(d$label, d$decision_id)) {
    m <- length(vals)
    if (m < 2) next
    for (i in seq_len(m)) for (j in seq_len(m)) if (i != j)
      o[vals[i], vals[j]] <- o[vals[i], vals[j]] + 1 / (m - 1)
  }
  n <- sum(o); n_c <- rowSums(o)
  d_o <- 1 - sum(diag(o)) / n
  d_e <- 1 - sum(n_c * (n_c - 1)) / (n * (n - 1))
  c(alpha = 1 - d_o / d_e, agree = 1 - d_o)
}

# Bootstrap 95% CI (resampling decisions). Uses R's RNG, so the interval differs
# slightly from Python's; point estimates match exactly.
kalpha_ci <- function(d, n_boot = 1000, seed = 7) {
  set.seed(seed)
  g <- split(d$is_error, d$decision_id)
  g <- g[lengths(g) >= 2]
  m <- lengths(g); e <- vapply(g, sum, numeric(1)); c_ <- m - e
  contrib <- cbind(c_ * (c_ - 1), c_ * e, e * c_, e * (e - 1)) / (m - 1)
  stats <- replicate(n_boot, {
    s <- colSums(contrib[sample.int(nrow(contrib), replace = TRUE), , drop = FALSE])
    n <- sum(s); n_c <- c(s[1] + s[2], s[3] + s[4])
    d_o <- 1 - (s[1] + s[4]) / n
    d_e <- 1 - sum(n_c * (n_c - 1)) / (n * (n - 1))
    1 - d_o / d_e
  })
  quantile(stats, c(0.025, 0.975), names = FALSE)
}

cohen_kappa <- function(a, b) {
  p_o <- mean(a == b)
  p_e <- sum(sapply(c("Correct", "Error"), function(k) mean(a == k) * mean(b == k)))
  c(kappa = if (p_e < 1) (p_o - p_e) / (1 - p_e) else NA, agree = p_o, chance = p_e)
}

# ---------------------------------------------------------------------------
# Step 1. Naive: raw percent agreement
# ---------------------------------------------------------------------------
all_s <- kalpha(df)
site_s <- sapply(sites, function(s) kalpha(df[df$site == s, ]))
cat(sprintf("\nStep 1  Raw percent agreement, all sites: %.1f%%\n", 100 * all_s["agree"]))
for (s in sites) cat(sprintf("        site %s: %.1f%% agreement; reviewers flag %.1f%% as errors\n",
                             s, 100 * site_s["agree", s], 100 * mean(df$is_error[df$site == s])))

# ---------------------------------------------------------------------------
# Step 2. Chance-corrected agreement
# ---------------------------------------------------------------------------
reviewers <- sort(unique(df$reviewer_id))
rev_site <- tapply(df$site, df$reviewer_id, `[`, 1)
pairs <- NULL
for (i in seq_along(reviewers)) for (j in seq_along(reviewers)) if (i < j) {
  a <- df[df$reviewer_id == reviewers[i], c("decision_id", "label")]
  b <- df[df$reviewer_id == reviewers[j], c("decision_id", "label")]
  ab <- merge(a, b, by = "decision_id")
  if (nrow(ab) == 0) next
  k <- cohen_kappa(ab$label.x, ab$label.y)
  pairs <- rbind(pairs, data.frame(r1 = reviewers[i], r2 = reviewers[j], site = rev_site[[reviewers[i]]],
                                   n_shared = nrow(ab), kappa = k[["kappa"]], agree = k[["agree"]],
                                   chance = k[["chance"]]))
}
top <- pairs[order(-pairs$n_shared, pairs$r1, pairs$r2), ][1, ]
mean_kappa <- mean(pairs$kappa, na.rm = TRUE)
mean_kappa_site <- tapply(pairs$kappa, pairs$site, mean, na.rm = TRUE)
cat(sprintf("\nStep 2  %d of %d reviewer pairs share any decisions.\n", nrow(pairs),
            length(reviewers) * (length(reviewers) - 1) / 2))
cat(sprintf("        Most frequent pair %s-%s (site %s, n=%d): agreement %.1f%%, chance %.1f%%, kappa %.2f\n",
            top$r1, top$r2, top$site, top$n_shared, 100 * top$agree, 100 * top$chance, top$kappa))
cat(sprintf("        Average pairwise kappa: %.2f (A %.2f, B %.2f, C %.2f)\n", mean_kappa,
            mean_kappa_site["A"], mean_kappa_site["B"], mean_kappa_site["C"]))
ci_all <- kalpha_ci(df)
cat(sprintf("        Krippendorff's alpha, all sites: %.2f (95%% CI %.2f to %.2f)\n",
            all_s["alpha"], ci_all[1], ci_all[2]))
ci_site <- list()
for (s in sites) {
  ci_site[[s]] <- kalpha_ci(df[df$site == s, ])
  cat(sprintf("        site %s: alpha %.2f (95%% CI %.2f to %.2f)\n", s, site_s["alpha", s],
              ci_site[[s]][1], ci_site[[s]][2]))
}

# ---------------------------------------------------------------------------
# Step 3. Check against ground truth
# ---------------------------------------------------------------------------
tp <- read.csv("data/true_parameters.csv")
true_prev <- tp$value[tp$parameter == "prevalence"]
ab <- gold[gold$site %in% c("A", "B"), ]
ab_flag <- mean(df$is_error[df$site %in% c("A", "B")])
ab_sens <- mean(ab$is_error[ab$gold_label == "Error"])
ab_spec <- 1 - mean(ab$is_error[ab$gold_label == "Correct"])
est_prev <- (ab_flag + ab_spec - 1) / (ab_sens + ab_spec - 1)   # Rogan-Gladen correction
approve_all_accuracy <- 1 - est_prev
cat(sprintf("\nStep 3  Estimated live AI error rate %.1f%% (true %.0f%%); 'approve everything' is right %.0f%% of the time.\n",
            100 * est_prev, 100 * true_prev, 100 * approve_all_accuracy))

out <- list()
miss_case <- function(g) {      # true errors with 2+ reviews where every reviewer said Correct
  te <- g[g$gold_label == "Error", ]
  agg <- aggregate(is_error ~ decision_id, te, function(x) c(sum = sum(x), n = length(x)))
  agg <- data.frame(decision_id = agg$decision_id, s = agg$is_error[, "sum"], n = agg$is_error[, "n"])
  agg <- agg[agg$n >= 2, ]
  c(k = sum(agg$s == 0), n = nrow(agg))
}
for (s in sites) {
  g <- gold[gold$site == s, ]
  te <- g[g$gold_label == "Error", ]; tc <- g[g$gold_label == "Correct", ]
  ks <- sum(te$is_error); ns <- nrow(te)
  kc <- sum(tc$is_error == 0); nc <- nrow(tc)
  mb <- miss_case(g)
  tsens <- tp$value[tp$level == "site_reviewer_mean" & tp$site == s & tp$parameter == "sensitivity"]
  cat(sprintf("        site %s: caught %d/%d = %.0f%% (true %.0f%%); specificity %.0f%%; both missed %d/%d true errors\n",
              s, ks, ns, 100 * ks / ns, 100 * tsens, 100 * kc / nc, mb["k"], mb["n"]))
  out[[paste0("sensitivity_site_", s)]] <- ks / ns
  out[[paste0("specificity_site_", s)]] <- kc / nc
  out[[paste0("missed_by_both_site_", s)]] <- mb[["k"]] / mb[["n"]]
  # Step 4 input: errors per 1,000 live decisions that every reviewer misses
  out[[paste0("errors_slipping_per_1000_site_", s)]] <- 1000 * est_prev * mb[["k"]] / mb[["n"]]
}
mb_all <- miss_case(gold)
sens_all <- mean(gold$is_error[gold$gold_label == "Error"])

# ---------------------------------------------------------------------------
# Step 4. Decision implication
# ---------------------------------------------------------------------------
cat("\nStep 4  Estimated AI errors passing review unflagged, per 1,000 decisions:\n")
for (s in sites) cat(sprintf("        site %s: %.0f\n", s, out[[paste0("errors_slipping_per_1000_site_", s)]]))

# Case-level bootstrap. Each golden case is reviewed about twice, so reviews are
# clustered within cases; resample CASES (golden cases within site, live A+B
# decisions) and recompute sensitivity, the corrected live error rate, and errors
# per 1,000. R's RNG differs from Python's, so bounds differ slightly.
set.seed(11)
n_boot <- 2000
live <- aggregate(is_error ~ decision_id, df[df$site %in% c("A", "B"), ],
                  function(x) c(k = sum(x), n = length(x)))
live <- live$is_error
gc <- aggregate(is_error ~ decision_id + site + gold_label, gold, function(x) c(k = sum(x), n = length(x)))
gc <- data.frame(site = gc$site, err = gc$gold_label == "Error", k = gc$is_error[, "k"], n = gc$is_error[, "n"])
by_site <- split(gc, gc$site)
bs <- replicate(n_boot, {
  lv <- live[sample.int(nrow(live), replace = TRUE), ]
  flag <- sum(lv[, "k"]) / sum(lv[, "n"])
  rs <- lapply(by_site, function(d) d[sample.int(nrow(d), replace = TRUE), ])
  ab <- rbind(rs$A, rs$B)
  sens_ab <- sum(ab$k[ab$err]) / sum(ab$n[ab$err])
  spec_ab <- 1 - sum(ab$k[!ab$err]) / sum(ab$n[!ab$err])
  p <- (flag + spec_ab - 1) / (sens_ab + spec_ab - 1)
  sens <- sapply(sites, function(s) { d <- rs[[s]][rs[[s]]$err, ]; sum(d$k) / sum(d$n) })
  slip <- sapply(sites, function(s) { d <- rs[[s]][rs[[s]]$err & rs[[s]]$n >= 2, ]; 1000 * p * mean(d$k == 0) })
  c(est_prev = p, setNames(sens, paste0("sens_", sites)), setNames(slip, paste0("slip_", sites)))
})
bci <- apply(bs, 1, quantile, probs = c(0.025, 0.975), names = FALSE)
cat(sprintf("        Case-level bootstrap 95%% CI, live error rate: %.1f%% to %.1f%%\n",
            100 * bci[1, "est_prev"], 100 * bci[2, "est_prev"]))
for (s in sites) cat(sprintf("        site %s: sensitivity %.2f to %.2f; per 1,000 %.0f to %.0f\n", s,
                             bci[1, paste0("sens_", s)], bci[2, paste0("sens_", s)],
                             bci[1, paste0("slip_", s)], bci[2, paste0("slip_", s)]))

# ---------------------------------------------------------------------------
# Write headline numbers for results.json (names match python/analysis.py)
# ---------------------------------------------------------------------------
h <- c(
  percent_agreement_all = all_s[["agree"]],
  setNames(site_s["agree", ], paste0("percent_agreement_site_", sites)),
  kappa_top_pair = top$kappa, kappa_top_pair_agreement = top$agree,
  kappa_top_pair_chance_agreement = top$chance,
  kappa_mean_pairwise_all = mean_kappa,
  setNames(as.numeric(mean_kappa_site[sites]), paste0("kappa_mean_pairwise_site_", sites)),
  alpha_all = all_s[["alpha"]],
  setNames(site_s["alpha", ], paste0("alpha_site_", sites)),
  estimated_live_error_rate = est_prev, approve_everything_accuracy = approve_all_accuracy,
  sensitivity_all = sens_all,
  unlist(out),
  missed_by_both_all = mb_all[["k"]] / mb_all[["n"]],
  r_alpha_all_ci_low = ci_all[1], r_alpha_all_ci_high = ci_all[2],
  setNames(sapply(ci_site, `[`, 1), paste0("r_alpha_site_", sites, "_ci_low")),
  setNames(sapply(ci_site, `[`, 2), paste0("r_alpha_site_", sites, "_ci_high")),
  r_estimated_live_error_rate_ci_low = unname(bci[1, "est_prev"]), r_estimated_live_error_rate_ci_high = unname(bci[2, "est_prev"]),
  setNames(bci[1, paste0("sens_", sites)], paste0("r_sensitivity_site_", sites, "_ci_low")),
  setNames(bci[2, paste0("sens_", sites)], paste0("r_sensitivity_site_", sites, "_ci_high")),
  setNames(bci[1, paste0("slip_", sites)], paste0("r_errors_slipping_per_1000_site_", sites, "_ci_low")),
  setNames(bci[2, paste0("slip_", sites)], paste0("r_errors_slipping_per_1000_site_", sites, "_ci_high"))
)
dir.create("outputs", showWarnings = FALSE)
write.csv(data.frame(name = names(h), value = signif(unname(h), 10)), "outputs/r_headline.csv", row.names = FALSE)
cat("\nWrote outputs/r_headline.csv (run python3 python/analysis.py to merge into results.json).\n")
