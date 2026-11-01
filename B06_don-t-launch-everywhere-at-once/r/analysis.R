# =============================================================================
# analysis.R -- B06 "Don't Launch Everywhere at Once: Who Benefits from AI?"
#
# Run from the package root (after python3 data/generate_data.py):
#     Rscript r/analysis.R
# Then merge the R numbers into outputs/results.json and check agreement:
#     python3 python/merge_results.py
#
# Packages: base R + sandwich (cluster-robust SEs) + rpart (regression trees;
# ships with every R install as a "recommended" package). grf (causal forests)
# could not be installed in the build environment, so the ML step is a
# DR-learner with a bagged-tree forest written here in base R + rpart. If you
# have grf, see the optional block at the end of Step 2b.
#
# Reads the SAME csv the Python generator wrote (one dataset, one seed).
# All data are synthetic, generated to illustrate the method.
# =============================================================================

suppressPackageStartupMessages({
  library(sandwich)
  library(rpart)
})

# --- Locate the package root robustly (works from any working directory) -----
args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
ROOT <- if (length(file_arg)) normalizePath(file.path(dirname(file_arg), "..")) else getwd()
OUT <- file.path(ROOT, "outputs")
dir.create(OUT, showWarnings = FALSE)

SEED <- 20261153
NOVICE_CUTOFF <- 12   # months, pre-specified
MIN_EFFECT <- 1.0     # smallest gain worth launching for (tasks/week)
N_FOLDS <- 5

# -----------------------------------------------------------------------------
# 0. Load and prepare (identical rules to python/analysis.py)
# -----------------------------------------------------------------------------
df <- read.csv(file.path(ROOT, "data", "ai_pilot_heterogeneity.csv"), stringsAsFactors = FALSE)
winsor <- function(x) {
  q <- quantile(x, c(0.005, 0.995), type = 7)   # type 7 == numpy 'linear'
  pmin(pmax(x, q[1]), q[2])
}
df$outcome_tasks_w  <- winsor(df$outcome_tasks)
df$baseline_tasks_w <- winsor(df$baseline_tasks)
df$non_english <- as.integer(df$work_language != "English")
df$novice <- ifelse(is.na(df$tenure_months), NA, as.numeric(df$tenure_months < NOVICE_CUTOFF))

N <- nrow(df); G <- length(unique(df$team_id))
control_mean <- mean(df$outcome_tasks_w[df$ai_access == 0])
pct <- function(x) 100 * x / control_mean
cat(sprintf("%d employees, %d teams; control mean = %.2f tasks/week\n", N, G, control_mean))

# OLS with team-clustered SEs (HC1 small-sample correction, like statsmodels).
# CIs / p-values use t with (teams - 1) degrees of freedom.
ols_cluster <- function(formula, d, term) {
  m <- lm(formula, data = d)
  V <- vcovCL(m, cluster = d$team_id, type = "HC1")   # cluster by team
  g <- length(unique(d$team_id))
  est <- coef(m)[term]; se <- sqrt(V[term, term])
  tq <- qt(0.975, g - 1)
  c(est = unname(est), se = se, lo = unname(est - tq * se), hi = unname(est + tq * se),
    p = unname(2 * pt(abs(est / se), g - 1, lower.tail = FALSE)), n = nobs(m))
}
CONTROLS <- "baseline_tasks_w + factor(region)"
f_simple <- as.formula(paste("outcome_tasks_w ~ ai_access +", CONTROLS))

# -----------------------------------------------------------------------------
# STEP 1. Naive approach
# -----------------------------------------------------------------------------
# 1a. One average for everyone
ate <- ols_cluster(f_simple, df, "ai_access")
cat(sprintf("\n[Step 1a] Overall ATE = %.2f (%.1f%%), 95%% CI [%.2f, %.2f]\n",
            ate["est"], pct(ate["est"]), ate["lo"], ate["hi"]))

# 1b. Slice 14 ways, test AI x slice, no correction
slices <- list(
  "Region: Americas" = as.numeric(df$region == "Americas"),
  "Region: EMEA" = as.numeric(df$region == "EMEA"),
  "Region: APAC" = as.numeric(df$region == "APAC"),
  "Region: LATAM" = as.numeric(df$region == "LATAM"),
  "Language: non-English" = as.numeric(df$non_english),
  "Tenure: novice (<12 mo)" = df$novice,
  "Role: Tier-1 support" = as.numeric(df$role_family == "Tier-1 support"),
  "Role: Tier-2 support" = as.numeric(df$role_family == "Tier-2 support"),
  "Role: Operations analyst" = as.numeric(df$role_family == "Operations analyst"),
  "Role: Specialist" = as.numeric(df$role_family == "Specialist"),
  "Shift: Day" = as.numeric(df$shift == "Day"),
  "Shift: Evening" = as.numeric(df$shift == "Evening"),
  "Shift: Night" = as.numeric(df$shift == "Night"),
  "Remote worker" = as.numeric(df$remote)
)
naive <- do.call(rbind, lapply(names(slices), function(nm) {
  d <- df; d$s <- slices[[nm]]; d <- d[!is.na(d$s), ]
  r <- ols_cluster(outcome_tasks_w ~ ai_access * s + baseline_tasks_w, d, "ai_access:s")
  data.frame(slice = nm, diff = r["est"], p_raw = r["p"], row.names = NULL)
}))
naive$p_holm <- p.adjust(naive$p_raw, method = "holm")
true_mod <- naive$slice %in% c("Language: non-English", "Tenure: novice (<12 mo)")
cat("\n[Step 1b] Naive slicing:\n"); print(transform(naive, diff = round(diff, 3),
    p_raw = round(p_raw, 3), p_holm = round(p_holm, 3)), row.names = FALSE)

# -----------------------------------------------------------------------------
# STEP 2a. Pre-specified interaction model (complete cases on tenure)
#   Outcome = b0 + b1*AI + b2*NonEnglish + b3*(AI x NonEnglish)
#           + b4*Novice + b5*(AI x Novice) + controls
# -----------------------------------------------------------------------------
cc <- df[!is.na(df$novice), ]
f_int <- as.formula(paste("outcome_tasks_w ~ ai_access * non_english + ai_access * novice +", CONTROLS))
b1 <- ols_cluster(f_int, cc, "ai_access")
b3 <- ols_cluster(f_int, cc, "ai_access:non_english")
b5 <- ols_cluster(f_int, cc, "ai_access:novice")
cat(sprintf("\n[Step 2a] b1 %.2f [%.2f, %.2f]; b3 %.2f [%.2f, %.2f]; b5 %.2f [%.2f, %.2f]\n",
            b1["est"], b1["lo"], b1["hi"], b3["est"], b3["lo"], b3["hi"], b5["est"], b5["lo"], b5["hi"]))

groups <- list(
  all = df, english = df[df$non_english == 0, ], non_english = df[df$non_english == 1, ],
  novice = cc[cc$novice == 1, ], experienced = cc[cc$novice == 0, ],
  english_novice = cc[cc$non_english == 0 & cc$novice == 1, ],
  english_experienced = cc[cc$non_english == 0 & cc$novice == 0, ],
  non_english_novice = cc[cc$non_english == 1 & cc$novice == 1, ],
  non_english_experienced = cc[cc$non_english == 1 & cc$novice == 0, ]
)
sub <- do.call(rbind, lapply(names(groups), function(k) {
  d <- groups[[k]]; r <- ols_cluster(f_simple, d, "ai_access")
  data.frame(key = k, n = nrow(d), est = r["est"], lo = r["lo"], hi = r["hi"], p = r["p"],
             true = mean(d$true_effect), row.names = NULL)
}))
cat("\n[Step 2a] Subgroup effects vs truth:\n"); print(transform(sub, est = round(est, 2),
    lo = round(lo, 2), hi = round(hi, 2), p = round(p, 4), true = round(true, 2)), row.names = FALSE)

# -----------------------------------------------------------------------------
# STEP 2b. ML discovery: DR-learner + a bagged-tree forest (base R + rpart)
#   pseudo_i = mu1(X) - mu0(X) + W(Y - mu1)/e - (1-W)(Y - mu0)/(1-e)
#   mu0, mu1: linear outcome models fit on OTHER teams (cross-fitting);
#   e: design propensity (team's treated share). Same folds as Python, so the
#   pseudo-outcomes are identical; only the forest implementation differs.
# -----------------------------------------------------------------------------
df$tenure_imp <- ifelse(is.na(df$tenure_months), median(df$tenure_months, na.rm = TRUE), df$tenure_months)
df$tenure_missing <- as.integer(is.na(df$tenure_months))
df$region_f <- factor(df$region); df$role_f <- factor(df$role_family); df$shift_f <- factor(df$shift)
folds <- (as.integer(substr(df$team_id, 2, 4)) - 1) %% N_FOLDS
e <- ave(df$ai_access, df$team_id)

f_mu <- outcome_tasks_w ~ tenure_imp + tenure_missing + non_english + remote + baseline_tasks_w +
  region_f + role_f + shift_f
mu0 <- mu1 <- numeric(N)
for (k in 0:(N_FOLDS - 1)) {
  tr <- folds != k; te <- folds == k
  m0 <- lm(f_mu, data = df[tr & df$ai_access == 0, ])
  m1 <- lm(f_mu, data = df[tr & df$ai_access == 1, ])
  mu0[te] <- predict(m0, df[te, ]); mu1[te] <- predict(m1, df[te, ])
}
w <- df$ai_access; y <- df$outcome_tasks_w
df$pseudo <- mu1 - mu0 + w * (y - mu1) / e - (1 - w) * (y - mu0) / (1 - e)

# A small random forest: B regression trees, each grown on a half-sample of
# teams' members (subsampling) with a random 5 of the 8 candidate variables.
# Big leaves (>= 60 people) keep estimates at the group level.
FEATS <- c("tenure_imp", "tenure_missing", "non_english", "remote", "baseline_tasks_w",
           "region_f", "role_f", "shift_f")
VARNAME <- c(tenure_imp = "Tenure", tenure_missing = "Tenure", non_english = "Work language",
             remote = "Remote", baseline_tasks_w = "Baseline output", region_f = "Region",
             role_f = "Role family", shift_f = "Shift")
grow_forest <- function(d, B = 300, seed = SEED) {
  set.seed(seed)
  lapply(seq_len(B), function(b) {
    idx <- sample(nrow(d), floor(nrow(d) / 2))
    vars <- sample(FEATS, 5)
    rpart(as.formula(paste("pseudo ~", paste(vars, collapse = "+"))), data = d[idx, ],
          method = "anova",
          control = rpart.control(minbucket = 60, cp = 0, maxdepth = 6, xval = 0,
                                  maxsurrogate = 0, maxcompete = 0))
  })
}
predict_forest <- function(trees, newd) rowMeans(sapply(trees, predict, newdata = newd))

df$cate_hat <- NA_real_
for (k in 0:(N_FOLDS - 1)) {
  trees <- grow_forest(df[folds != k, ], seed = SEED + k)
  df$cate_hat[folds == k] <- predict_forest(trees, df[folds == k, ])
}
full_trees <- grow_forest(df)
vi <- unlist(lapply(full_trees, function(t) t$variable.importance))
imp <- tapply(vi, VARNAME[names(vi)], sum)
imp <- sort(imp / sum(imp), decreasing = TRUE)
cat("\n[Step 2b] Share of importance by variable (R forest):\n"); print(round(imp, 3))

med <- median(df$cate_hat)
cal_hi <- ols_cluster(f_simple, df[df$cate_hat >= med, ], "ai_access")
cal_lo <- ols_cluster(f_simple, df[df$cate_hat < med, ], "ai_access")
q25 <- df$cate_hat <= quantile(df$cate_hat, 0.25, type = 7)
ccx <- df[!is.na(df$novice), ]
ml <- c(
  dr_ate_tasks = mean(df$pseudo),
  corr_cate_vs_true_effect = cor(df$cate_hat, df$true_effect),
  mean_cate_english = mean(df$cate_hat[df$non_english == 0]),
  mean_cate_non_english = mean(df$cate_hat[df$non_english == 1]),
  mean_cate_novice = mean(ccx$cate_hat[ccx$novice == 1]),
  mean_cate_experienced = mean(ccx$cate_hat[ccx$novice == 0]),
  calib_high_est = cal_hi[["est"]], calib_high_lo = cal_hi[["lo"]], calib_high_hi = cal_hi[["hi"]],
  calib_low_est = cal_lo[["est"]], calib_low_lo = cal_lo[["lo"]], calib_low_hi = cal_lo[["hi"]],
  share_non_english_in_bottom_quartile_cate = mean(df$non_english[q25])
)
ml <- c(ml, setNames(as.numeric(imp), paste0("importance:", names(imp))))
cat(sprintf("  corr(CATE, truth) = %.2f; high half %.2f, low half %.2f\n",
            ml["corr_cate_vs_true_effect"], ml["calib_high_est"], ml["calib_low_est"]))

# Optional, if you have grf installed (not used to produce the shipped numbers):
#   library(grf)
#   X <- model.matrix(~ tenure_imp + tenure_missing + non_english + remote + baseline_tasks_w +
#                       region_f + role_f + shift_f, df)[, -1]
#   cf <- causal_forest(X, df$outcome_tasks_w, df$ai_access, W.hat = e,
#                       clusters = as.integer(factor(df$team_id)), seed = SEED)
#   variable_importance(cf); test_calibration(cf)

# -----------------------------------------------------------------------------
# STEP 3. Ground truth (printed; the Python script records the full check)
# -----------------------------------------------------------------------------
cat(sprintf("\n[Step 3] b3 CI covers -3.6: %s; b5 CI covers 2.2: %s; subgroup coverage %d of %d\n",
            b3["lo"] <= -3.6 & -3.6 <= b3["hi"], b5["lo"] <= 2.2 & 2.2 <= b5["hi"],
            sum(sub$lo <= sub$true & sub$true <= sub$hi), nrow(sub)))

# -----------------------------------------------------------------------------
# STEP 4. Rollout grid (same pre-specified rule as Python)
# -----------------------------------------------------------------------------
decide <- function(lo, hi) ifelse(lo >= MIN_EFFECT, "Launch now (wave 1)",
                           ifelse(hi < MIN_EFFECT, "Hold: fix/localize, then re-test",
                                  "Targeted re-test (wave 2 candidate)"))
cells <- sub[sub$key %in% c("english_novice", "english_experienced", "non_english_novice",
                            "non_english_experienced"), ]
cells$decision <- decide(cells$lo, cells$hi)
cat("\n[Step 4] Rollout grid:\n"); print(cells[, c("key", "n", "est", "lo", "hi", "decision")], row.names = FALSE)

# -----------------------------------------------------------------------------
# Write headline numbers (full precision; rounding happens in the merge step)
# -----------------------------------------------------------------------------
H <- c(n_employees = N, n_teams = G, n_complete_tenure = nrow(cc), control_mean_tasks = control_mean,
       ate_tasks = ate[["est"]], ate_ci_low = ate[["lo"]], ate_ci_high = ate[["hi"]], ate_p = ate[["p"]],
       ate_pct = pct(ate[["est"]]), ate_pct_ci_low = pct(ate[["lo"]]), ate_pct_ci_high = pct(ate[["hi"]]))
for (x in list(list("b1_ai_english_experienced", b1), list("b3_ai_x_nonenglish", b3),
               list("b5_ai_x_novice", b5))) {
  r <- x[[2]]; nm <- x[[1]]
  H[nm] <- r[["est"]]; H[paste0(nm, "_ci_low")] <- r[["lo"]]
  H[paste0(nm, "_ci_high")] <- r[["hi"]]; H[paste0(nm, "_p")] <- r[["p"]]
}
for (i in seq_len(nrow(sub))) {
  k <- sub$key[i]
  H[paste0("sub_", k, "_n")] <- sub$n[i]; H[paste0("sub_", k, "_tasks")] <- sub$est[i]
  H[paste0("sub_", k, "_ci_low")] <- sub$lo[i]; H[paste0("sub_", k, "_ci_high")] <- sub$hi[i]
  H[paste0("sub_", k, "_p")] <- sub$p[i]; H[paste0("sub_", k, "_pct")] <- pct(sub$est[i])
  H[paste0("sub_", k, "_true_tasks")] <- sub$true[i]
}
slice_key <- function(s) {
  k <- gsub("[^a-z0-9]", "_", tolower(s)); k <- gsub("^_+|_+$", "", k)
  gsub("__", "_", k, fixed = TRUE)
}
for (i in seq_len(nrow(naive))) {
  k <- paste0("slice_", slice_key(naive$slice[i]))
  H[paste0(k, "_diff")] <- naive$diff[i]; H[paste0(k, "_p_raw")] <- naive$p_raw[i]
  H[paste0(k, "_p_holm")] <- naive$p_holm[i]
}
H["naive_n_flagged_raw"] <- sum(naive$p_raw < 0.05)
H["naive_n_false_flagged_raw"] <- sum(naive$p_raw < 0.05 & !true_mod)
H["naive_n_flagged_holm"] <- sum(naive$p_holm < 0.05)
H["dr_ate_tasks"] <- mean(df$pseudo)

write.csv(data.frame(name = names(H), value = sprintf("%.12g", H)),
          file.path(OUT, "headline_r.csv"), row.names = FALSE)
write.csv(data.frame(key = names(ml), value = sprintf("%.12g", ml)),
          file.path(OUT, "ml_r.csv"), row.names = FALSE)
write.csv(cells[, c("key", "n", "est", "lo", "hi", "decision")],
          file.path(OUT, "rollout_grid_r.csv"), row.names = FALSE)
cat("\nWrote outputs/headline_r.csv, ml_r.csv, rollout_grid_r.csv.\n",
    "Next: python3 python/merge_results.py\n")
