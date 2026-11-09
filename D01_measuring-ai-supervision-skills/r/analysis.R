# D01 - Measuring AI Supervision Skills: reliability and validity evidence for a skills test.
# Base R only (factanal() for the one-factor maximum-likelihood loadings behind omega).
#
# Run from the package folder or anywhere:  Rscript r/analysis.R
# Reads  data/ai_supervision_assessment.csv (+ ground-truth files, used only in Step 3)
# Writes outputs/r_results.csv (every number, unrounded) and outputs/r_item_statistics.csv
# python/check_agreement.py then compares these with Python and writes the "r" values
# into outputs/results.json.
#
# Missing-data rule (identical in Python):
#   * Reliability and item statistics use the PAIRWISE-complete covariance matrix.
#   * Scale scores are PRORATED (mean of answered items x k) if at least ceil(0.75 k) items
#     were answered; otherwise missing.

args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
root <- if (length(file_arg)) normalizePath(file.path(dirname(file_arg), "..")) else getwd()
data_dir <- file.path(root, "data")
out_dir <- file.path(root, "outputs")
dir.create(out_dir, showWarnings = FALSE)

S_ITEMS <- sprintf("S%02d", 1:12)
C_ITEMS <- sprintf("C%02d", 1:4)
CUTOFF <- 26; TARGET_HIGH_STAKES <- 0.90; SPEED_MIN <- 8; FLAG_R <- 0.20
CANON_UNITS <- c("Trust & Safety Review", "Operations", "Claims", "Customer Support")

res <- list()
put <- function(name, value) res[[name]] <<- value

# ---- building blocks (one formula each) -----------------------------------------------------
cov_pairwise <- function(d) cov(d, use = "pairwise.complete.obs")

alpha_from_cov <- function(C) {           # k/(k-1) * (1 - sum of item variances / total variance)
  k <- ncol(C); k / (k - 1) * (1 - sum(diag(C)) / sum(C))
}

omega_from_cov <- function(C, nobs) {     # one-factor ML loadings -> omega for the raw sum score
  sd <- sqrt(diag(C))
  fa <- factanal(covmat = cov2cor(C), factors = 1, n.obs = nobs)
  lam <- as.numeric(fa$loadings)
  if (sum(lam) < 0) lam <- -lam            # factor sign is arbitrary
  lam_raw <- lam * sd; uniq_raw <- (1 - lam^2) * sd^2
  list(omega = sum(lam_raw)^2 / (sum(lam_raw)^2 + sum(uniq_raw)), loadings = lam)
}

item_stats <- function(C, items) {        # item-rest r and alpha-if-deleted from the covariance matrix
  out <- data.frame(item = items, item_rest_r = NA_real_, alpha_if_deleted = NA_real_)
  for (i in seq_along(items)) {
    rest <- setdiff(seq_along(items), i)
    out$item_rest_r[i] <- sum(C[i, rest]) / sqrt(C[i, i] * sum(C[rest, rest]))
    out$alpha_if_deleted[i] <- alpha_from_cov(C[rest, rest, drop = FALSE])
  }
  out
}

decision <- function(r_ir, a_del, a_all) {
  ifelse(r_ir < FLAG_R, ifelse(r_ir >= 0, "drop", "drop (check key)"),
         ifelse(r_ir < 0.30 | a_del > a_all, "revise", "keep"))
}

prorated <- function(d, items) {
  k <- length(items); answered <- rowSums(!is.na(d[, items]))
  s <- rowMeans(d[, items], na.rm = TRUE) * k
  s[answered < ceiling(0.75 * k)] <- NA
  s
}

fisher_ci <- function(r, n) tanh(atanh(r) + c(-1, 1) * 1.96 / sqrt(n - 3))

zou_diff_ci <- function(r1, r2, r12, n) {   # Zou (2007): overlapping correlations sharing a criterion
  a <- fisher_ci(r1, n); b <- fisher_ci(r2, n)
  c <- ((r12 - r1 * r2 / 2) * (1 - r1^2 - r2^2 - r12^2) + r12^3) / ((1 - r1^2) * (1 - r2^2))
  d <- r1 - r2
  c(d,
    d - sqrt((r1 - a[1])^2 + (b[2] - r2)^2 - 2 * c * (r1 - a[1]) * (b[2] - r2)),
    d + sqrt((a[2] - r1)^2 + (r2 - b[1])^2 - 2 * c * (a[2] - r1) * (r2 - b[1])))
}

spearman_brown_items <- function(rel, k, target) ceiling(k * target * (1 - rel) / (rel * (1 - target)))

# ---- Step 0. Screen ---------------------------------------------------------------------------
raw <- read.csv(file.path(data_dir, "ai_supervision_assessment.csv"), stringsAsFactors = FALSE,
                check.names = FALSE)
put("n_records", nrow(raw))
key <- tolower(trimws(raw$business_unit))
fixed <- CANON_UNITS[match(key, tolower(CANON_UNITS))]
put("n_unit_labels_fixed", sum(raw$business_unit != fixed))
raw$business_unit <- fixed

answered_s <- rowSums(!is.na(raw[, S_ITEMS]))
n_distinct <- apply(raw[, S_ITEMS], 1, function(x) length(unique(x[!is.na(x)])))
speeder <- raw$completion_minutes < SPEED_MIN | (n_distinct == 1 & answered_s >= 9)
put("n_speeders_flagged", sum(speeder))
df <- raw[!speeder, ]; rownames(df) <- NULL
put("n_analyzed", nrow(df))
put("pct_item_responses_missing", 100 * mean(is.na(df[, c(S_ITEMS, C_ITEMS)])))
df$audit_pct_caught <- (df$audit_caught_a + df$audit_caught_b) / (df$audit_seeded_a + df$audit_seeded_b)
df$audit_a <- df$audit_caught_a / df$audit_seeded_a
df$audit_b <- df$audit_caught_b / df$audit_seeded_b
put("n_with_audit", sum(!is.na(df$audit_pct_caught)))

# ---- Step 1. Naive: one alpha for all 16 items ---------------------------------------------------
cov16 <- cov_pairwise(df[, c(S_ITEMS, C_ITEMS)])
put("alpha_naive16", alpha_from_cov(cov16))
put("alpha_scenario12", alpha_from_cov(cov_pairwise(df[, S_ITEMS])))
cat(sprintf("Step 1 (naive): alpha for all 16 items = %.2f; 12 scenario items alone = %.2f\n",
            res$alpha_naive16, res$alpha_scenario12))

# ---- Step 2a. Item analysis -----------------------------------------------------------------------
cov12 <- cov_pairwise(df[, S_ITEMS])
alpha12 <- alpha_from_cov(cov12)
ist <- item_stats(cov12, S_ITEMS)
ist$n <- colSums(!is.na(df[, S_ITEMS])); ist$mean <- colMeans(df[, S_ITEMS], na.rm = TRUE)
ist$decision <- decision(ist$item_rest_r, ist$alpha_if_deleted, alpha12)
ist16 <- item_stats(cov16, c(S_ITEMS, C_ITEMS))
names(ist16) <- c("item", "item_rest_r_in_16", "alpha_if_deleted_in_16")
tab <- merge(ist16, ist[, c("item", "n", "mean", "item_rest_r", "alpha_if_deleted", "decision")],
             by = "item", all.x = TRUE, sort = FALSE)
num <- sapply(tab, is.numeric); tab[num] <- lapply(tab[num], round, 3)
write.csv(tab, file.path(out_dir, "r_item_statistics.csv"), row.names = FALSE, na = "")

flagged <- ist$item[ist$item_rest_r < FLAG_R]
CLEAN <- setdiff(S_ITEMS, flagged)
SCALES <- list(naive16 = c(S_ITEMS, C_ITEMS), scenario12 = S_ITEMS, scenario10 = CLEAN, conf4 = C_ITEMS)
put("n_items_flagged", length(flagged))
put("items_flagged", paste(flagged, collapse = ";"))
for (it in c("S06", "S11")) {
  put(paste0("item_rest_r_", it), ist$item_rest_r[ist$item == it])
  put(paste0("alpha_if_deleted_", it), ist$alpha_if_deleted[ist$item == it])
}
good <- ist[!(ist$item %in% flagged), ]
put("item_rest_r_min_kept", min(good$item_rest_r))
put("item_rest_r_max_kept", max(good$item_rest_r))
put("item_rest_r_conf_min_in16", min(ist16$item_rest_r_in_16[ist16$item %in% C_ITEMS]))

# ---- Step 2b. Internal structure -----------------------------------------------------------------
corr16 <- cov2cor(cov16)
mean_r <- function(a, b) {
  v <- c(); for (x in a) for (y in b) if (x != y) v <- c(v, corr16[x, y]); mean(v)
}
put("mean_r_within_scenario10", mean_r(CLEAN, CLEAN))
put("mean_r_within_conf4", mean_r(C_ITEMS, C_ITEMS))
put("mean_r_between_blocks", mean_r(CLEAN, C_ITEMS))

# ---- Step 2c. Reliability: alpha and omega per scale ----------------------------------------------
for (nm in c("scenario12", "scenario10", "conf4")) {
  C <- cov_pairwise(df[, SCALES[[nm]]])
  nobs <- sum(complete.cases(df[, SCALES[[nm]]]))
  om <- omega_from_cov(C, nobs)
  if (nm != "scenario12") put(paste0("alpha_", nm), alpha_from_cov(C))
  put(paste0("omega_", nm), om$omega)
  if (nm == "scenario10") loadings10 <- om$loadings
}
put("loading_min_scenario10", min(loadings10))
put("loading_max_scenario10", max(loadings10))

# ---- Step 2d. Validity --------------------------------------------------------------------------
for (nm in names(SCALES)) df[[paste0("score_", nm)]] <- prorated(df, SCALES[[nm]])
vcols <- c(paste0("score_", names(SCALES)), "audit_pct_caught", "audit_a", "audit_b")
v <- df[complete.cases(df[, vcols]), ]
nv <- nrow(v)
put("n_validity_sample", nv)
for (nm in names(SCALES)) {
  r <- cor(v[[paste0("score_", nm)]], v$audit_pct_caught)
  ci <- fisher_ci(r, nv)
  put(paste0("validity_", nm), r)
  put(paste0("validity_", nm, "_ci_low"), ci[1])
  put(paste0("validity_", nm, "_ci_high"), ci[2])
}
z <- zou_diff_ci(res$validity_scenario10, res$validity_naive16,
                 cor(v$score_scenario10, v$score_naive16), nv)
put("validity_gain_scenario10_vs_naive16", z[1])
put("validity_gain_ci_low", z[2]); put("validity_gain_ci_high", z[3])
r_ab <- cor(v$audit_a, v$audit_b)
rel_crit <- 2 * r_ab / (1 + r_ab)
put("criterion_reliability", rel_crit)
# Python divides by the omega rounded to 2 decimals (the quoted value); do the same here.
put("validity_scenario10_corrected", res$validity_scenario10 / sqrt(round(res$omega_scenario10, 2) * rel_crit))

# ---- Step 2e. SEM and the band around the cutoff ---------------------------------------------------
s10 <- df$score_scenario10[!is.na(df$score_scenario10)]
sd10 <- sd(s10)
sem <- sd10 * sqrt(1 - round(res$omega_scenario10, 2))   # SEM uses the reported (2-decimal) omega
band <- 1.96 * sem
put("mean_scenario10", mean(s10)); put("sd_scenario10", sd10); put("sem_scenario10", sem)
put("cutoff_scenario10", CUTOFF)
put("band_low", CUTOFF - band); put("band_high", CUTOFF + band)
put("pct_pass", 100 * mean(s10 >= CUTOFF))
put("pct_within_band", 100 * mean(abs(s10 - CUTOFF) < band))
put("pct_within_1sem", 100 * mean(abs(s10 - CUTOFF) < sem))

# ---- Step 2f. Spearman-Brown ------------------------------------------------------------------------
om10 <- round(res$omega_scenario10, 2)
need <- spearman_brown_items(om10, length(CLEAN), TARGET_HIGH_STAKES)
put("sb_items_needed_090", need); put("sb_items_to_add_090", need - length(CLEAN))
f <- 12 / 10
put("sb_reliability_if_two_good_items", f * om10 / (1 + (f - 1) * om10))

# ---- Step 2g. Reliability by business unit -------------------------------------------------------
for (u in CANON_UNITS) {
  sub <- df[df$business_unit == u, ]
  tag <- gsub(" ", "_", gsub(" & ", "_", tolower(u)))
  put(paste0("alpha_scenario10_", tag), alpha_from_cov(cov_pairwise(sub[, CLEAN])))
  put(paste0("sd_scenario10_", tag), sd(sub$score_scenario10, na.rm = TRUE))
}

# ---- Step 3. Ground truth ----------------------------------------------------------------------
truth <- read.csv(file.path(data_dir, "true_parameters.csv"), stringsAsFactors = FALSE)
tv <- setNames(truth$value, truth$parameter)
persons <- read.csv(file.path(data_dir, "true_person_values.csv"), stringsAsFactors = FALSE)
put("flagged_items_match_truth", as.integer(identical(sort(flagged), sort(strsplit(tv[["flawed_items"]], ";")[[1]]))))
put("speeders_match_truth", as.integer(setequal(raw$person_id[speeder], persons$person_id[persons$is_speeder == 1])))
for (nm in c("naive16", "scenario12", "scenario10", "conf4")) {
  put(paste0("true_reliability_", nm), as.numeric(tv[[paste0("true_reliability_", nm)]]))
  put(paste0("true_validity_", nm), as.numeric(tv[[paste0("true_validity_", nm)]]))
}
put("true_corr_truescore10_truecriterion", as.numeric(tv[["true_corr_truescore10_truecriterion"]]))
m <- merge(df[!is.na(df$score_scenario10), c("person_id", "score_scenario10")], persons, by = "person_id")
mis <- (m$score_scenario10 >= CUTOFF) != (m$true_score_scenario10 >= CUTOFF)
inb <- abs(m$score_scenario10 - CUTOFF) < band
put("pct_misclassified", 100 * mean(mis))
put("pct_misclassified_in_band", 100 * mean(mis[inb]))
put("pct_misclassified_outside_band", 100 * mean(mis[!inb]))
put("pct_of_misclassified_inside_band", 100 * mean(inb[mis]))

# ---- Step 4. Decision (printed) -------------------------------------------------------------------
cat(sprintf(paste0(
  "Step 2: S06 item-rest r %.2f, S11 %.2f -> flagged: %s\n",
  "  scenario10 alpha %.2f / omega %.2f; confidence alpha %.2f\n",
  "  validity: naive16 %.2f [%.2f, %.2f], scenario10 %.2f [%.2f, %.2f], confidence %.2f [%.2f, %.2f]\n",
  "  gain %.2f [%.2f, %.2f]; SEM %.1f; %.0f%% within +/-1.96 SEM of %d; %d items for 0.90\n",
  "Step 3: %.0f%% misclassified, %.0f%% of them inside the band\n",
  "Step 4: fix S06/S11, report confidence separately, team-level use now, second assessment inside the band.\n"),
  res$item_rest_r_S06, res$item_rest_r_S11, res$items_flagged,
  res$alpha_scenario10, res$omega_scenario10, res$alpha_conf4,
  res$validity_naive16, res$validity_naive16_ci_low, res$validity_naive16_ci_high,
  res$validity_scenario10, res$validity_scenario10_ci_low, res$validity_scenario10_ci_high,
  res$validity_conf4, res$validity_conf4_ci_low, res$validity_conf4_ci_high,
  res$validity_gain_scenario10_vs_naive16, res$validity_gain_ci_low, res$validity_gain_ci_high,
  res$sem_scenario10, res$pct_within_band, CUTOFF, as.integer(res$sb_items_needed_090),
  res$pct_misclassified, res$pct_of_misclassified_inside_band))

out <- data.frame(name = names(res),
                  value = vapply(res, function(x) if (is.numeric(x)) format(x, digits = 15) else as.character(x), ""))
write.csv(out, file.path(out_dir, "r_results.csv"), row.names = FALSE)
cat("Wrote outputs/r_results.csv and outputs/r_item_statistics.csv\n")
