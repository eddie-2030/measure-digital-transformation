# analysis.R -- F01 "The Broken Bottom Rung": cohort comparison on a tenure clock
# plus career sequence analysis, in base R (no packages). All data are synthetic.
#
# Run from the package root or from r/:
#   Rscript r/analysis.R        or      cd r && Rscript analysis.R
#
# Same steps and the same numbers as python/analysis.py:
#   0 clean, 1 naive early output, 2 tenure clock + sequences (Hamming distance,
#   k-medoids into 3 trajectory types, bootstrap), 3 ground-truth check, 4 decision.
# Writes outputs/results_r.csv. Run python/check_agreement.py afterwards.
# (Figures are drawn by the Python script. For richer sequence tools in R, the
#  TraMineR package is an optional pointer; it is NOT needed here.)

# ---- locate the package root relative to this script ----------------------
script_dir <- local({
  a <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", a[grep("^--file=", a)])
  if (length(f) == 1) dirname(normalizePath(f))
  else if (!is.null(sys.frames()[[1]]$ofile)) dirname(normalizePath(sys.frames()[[1]]$ofile))
  else getwd()
})
root <- normalizePath(file.path(script_dir, ".."))
data_dir <- file.path(root, "data")
out_dir <- file.path(root, "outputs")
dir.create(out_dir, showWarnings = FALSE)

STATES <- c("R", "S", "I", "X")
GROUPS <- list(c("HIGH", "pre"), c("HIGH", "post"), c("LOW", "pre"), c("LOW", "post"))
POST_COHORTS <- c("2024H2", "2025H1", "2025H2")
SEQ_LEN <- 18
B <- 1000
BOOT_SEED <- 20261105

res <- data.frame(name = character(), value = numeric(), stringsAsFactors = FALSE)
rec <- function(name, value) res[nrow(res) + 1, ] <<- list(name, as.numeric(value))
gkey <- function(f, p) paste0(tolower(f), "_", p)

# ---------------------------------------------------------------------------
# Step 0. Load and clean
# ---------------------------------------------------------------------------
raw <- read.csv(file.path(data_dir, "junior_careers.csv"), stringsAsFactors = FALSE,
                colClasses = c(state = "character"), na.strings = "NA")
raw$tasks_closed <- suppressWarnings(as.numeric(raw$tasks_closed))
raw$skill_score_m18 <- suppressWarnings(as.numeric(raw$skill_score_m18))
truth <- read.csv(file.path(data_dir, "ground_truth.csv"), stringsAsFactors = FALSE)

CODE_MAP <- c(R = "R", ROUTINE = "R", S = "S", SUP = "S", SUPERVISED = "S",
              I = "I", INDEP = "I", INDEPENDENT = "I", X = "X", LEFT = "X")
df <- raw[order(raw$person_id, raw$tenure_month), ]
rownames(df) <- NULL
code <- toupper(trimws(df$state))
n_recoded <- sum(code != "" & df$state != code)
state_clean <- unname(CODE_MAP[code])          # "" or unknown -> NA (missing)
n_blank <- sum(is.na(state_clean))

# Missing month -> carry previous state forward (first month -> "R");
# X is absorbing (active codes after leaving are recoded to X).
n_after_x <- 0
starts <- !duplicated(df$person_id)
prev <- NA; left <- FALSE
for (k in seq_len(nrow(df))) {
  if (starts[k]) { prev <- NA; left <- FALSE }
  s <- state_clean[k]
  if (is.na(s)) s <- if (is.na(prev)) "R" else prev
  if (left && s != "X") { n_after_x <- n_after_x + 1; s <- "X" }
  if (s == "X") left <- TRUE
  state_clean[k] <- s
  prev <- s
}
df$state_clean <- state_clean
df$period <- ifelse(df$cohort %in% POST_COHORTS, "post", "pre")

ids <- sort(unique(df$person_id))
first <- df[starts, ]
people <- data.frame(person_id = first$person_id, job_family = first$job_family,
                     period = first$period, cohort = first$cohort,
                     n_obs = as.numeric(tapply(df$tenure_month, df$person_id, max)[first$person_id]),
                     skill18 = first$skill_score_m18, stringsAsFactors = FALSE)
people <- people[order(people$person_id), ]
# wide matrix: rows = people (sorted), columns = tenure months 1..24
wide <- matrix(NA_character_, nrow = length(ids), ncol = 24, dimnames = list(ids, NULL))
wide[cbind(match(df$person_id, ids), df$tenure_month)] <- df$state_clean

rec("n_people", nrow(people))
rec("n_person_months", nrow(df))
rec("n_blank_months_imputed", n_blank)
rec("n_recoded_rows", n_recoded)
rec("n_active_after_left_fixed", n_after_x)
for (g in GROUPS) rec(paste0("n_people_", gkey(g[1], g[2])), sum(people$job_family == g[1] & people$period == g[2]))

# ---------------------------------------------------------------------------
# Step 1. NAIVE: early output (tasks per month, months 1-6)
# ---------------------------------------------------------------------------
e <- df[df$tenure_month <= 6 & df$state_clean != "X" & !is.na(df$tasks_closed), ]
pm <- tapply(e$tasks_closed, e$person_id, mean)
people$tasks16 <- as.numeric(pm[people$person_id])
naive <- list()
for (g in GROUPS) {
  v <- mean(people$tasks16[people$job_family == g[1] & people$period == g[2]], na.rm = TRUE)
  naive[[gkey(g[1], g[2])]] <- v
  rec(paste0("naive_tasks_m1_6_", gkey(g[1], g[2])), v)
}
for (f in c("high", "low"))
  rec(paste0("naive_tasks_pct_change_", f), 100 * (naive[[paste0(f, "_post")]] / naive[[paste0(f, "_pre")]] - 1))

# ---------------------------------------------------------------------------
# Step 2. CORRECT: tenure clock + sequences
# ---------------------------------------------------------------------------
W <- wide[people$person_id, , drop = FALSE]
people$s_months_12 <- rowSums(W[, 1:12] == "S", na.rm = TRUE)
people$in_I_18 <- as.numeric(!is.na(W[, 18]) & W[, 18] == "I")
people$in_I_12 <- as.numeric(!is.na(W[, 12]) & W[, 12] == "I")
people$obs12 <- people$n_obs >= 12
people$obs18 <- people$n_obs >= 18     # observed THROUGH month 18 (by calendar)

for (g in GROUPS) {
  m <- people$job_family == g[1] & people$period == g[2]
  k <- gkey(g[1], g[2])
  rec(paste0("s_months_first12_", k), mean(people$s_months_12[m & people$obs12]))
  rec(paste0("share_I_m12_", k), 100 * mean(people$in_I_12[m & people$obs12]))
  rec(paste0("n_obs12_", k), sum(m & people$obs12))
  rec(paste0("share_I_m18_", k), 100 * mean(people$in_I_18[m & people$obs18]))
  rec(paste0("n_obs18_", k), sum(m & people$obs18))
  rec(paste0("skill_m18_", k), mean(people$skill18[m & people$obs18], na.rm = TRUE))
}

# 2d. Hamming distance on the first 18 months + k-medoids (PAM)
cl <- people[people$obs18, ]
S <- W[cl$person_id, 1:SEQ_LEN]
n <- nrow(S)
D <- matrix(0, n, n)
for (t in 1:SEQ_LEN) D <- D + outer(S[, t], S[, t], "!=")

pam_fit <- function(D, k) {
  # BUILD + SWAP with integer costs; ties -> lowest index (which.min/which.max)
  n <- nrow(D)
  med <- which.min(rowSums(D))
  while (length(med) < k) {
    dn <- apply(D[, med, drop = FALSE], 1, min)
    gain <- rowSums(pmax(matrix(dn, n, n, byrow = TRUE) - D, 0))
    gain[med] <- -1
    med <- c(med, which.max(gain))
  }
  cost <- sum(apply(D[, med, drop = FALSE], 1, min))
  repeat {
    best_cost <- cost; best_i <- 0; best_h <- 0
    for (i in seq_len(k)) {
      d_other <- apply(D[, med[-i], drop = FALSE], 1, min)
      cc <- rowSums(pmin(D, matrix(d_other, n, n, byrow = TRUE)))
      cc[med] <- Inf
      h <- which.min(cc)
      if (cc[h] < best_cost) { best_cost <- cc[h]; best_i <- i; best_h <- h }
    }
    if (best_i == 0) break
    med[best_i] <- best_h
    cost <- best_cost
  }
  med <- sort(med)
  list(med = med, assign = apply(D[, med, drop = FALSE], 1, which.min), cost = cost)
}

fit <- pam_fit(D, 3)
med_seq <- S[fit$med, , drop = FALSE]
x_cnt <- rowSums(med_seq == "X"); i_cnt <- rowSums(med_seq == "I")
leaver <- which.max(x_cnt)
rest <- setdiff(1:3, leaver)
climber <- if (i_cnt[rest[1]] > i_cnt[rest[2]]) rest[1] else rest[2]
plateau <- setdiff(rest, climber)
nm <- character(3); nm[leaver] <- "early_leaver"; nm[climber] <- "climber"; nm[plateau] <- "plateau"
cl$cluster <- nm[fit$assign]
rec("pam_total_distance", fit$cost)
rec("n_clustered", n)
for (j in 1:3) {
  rec(paste0("medoid_", nm[j], "_I_months"), i_cnt[j])
  rec(paste0("medoid_", nm[j], "_S_months"), sum(med_seq[j, ] == "S"))
}
for (g in GROUPS) {
  m <- cl$job_family == g[1] & cl$period == g[2]
  for (tp in c("climber", "plateau", "early_leaver"))
    rec(paste0("share_", tp, "_", gkey(g[1], g[2])), 100 * mean(cl$cluster[m] == tp))
}

# 2e. Bootstrap with the same Park-Miller generator as Python
pm_state <- BOOT_SEED
draw <- function(n_pick, n_pool) {
  out <- numeric(n_pick); s <- pm_state
  for (k in seq_len(n_pick)) {
    s <- (16807 * s) %% 2147483647
    out[k] <- floor(s * n_pool / 2147483647) + 1   # 1-based index
  }
  pm_state <<- s
  out
}
arr <- lapply(GROUPS, function(g) {
  a <- cl[cl$job_family == g[1] & cl$period == g[2], ]
  b <- people[people$job_family == g[1] & people$period == g[2] & people$obs12, ]
  list(plateau = as.numeric(a$cluster == "plateau"), I18 = a$in_I_18, S12 = b$s_months_12, I12 = b$in_I_12)
})
stats_from <- function(est) {   # est: list in GROUPS order of c(plateau, I18, S12, I12)
  o <- c(diff_plateau_high = 100 * (est[[2]]["plateau"] - est[[1]]["plateau"]),
         diff_share_I_m18_high = 100 * (est[[2]]["I18"] - est[[1]]["I18"]),
         diff_s_months_first12_high = est[[2]]["S12"] - est[[1]]["S12"],
         diff_share_I_m12_high = 100 * (est[[2]]["I12"] - est[[1]]["I12"]),
         diff_plateau_low = 100 * (est[[4]]["plateau"] - est[[3]]["plateau"]),
         diff_share_I_m18_low = 100 * (est[[4]]["I18"] - est[[3]]["I18"]),
         diff_s_months_first12_low = est[[4]]["S12"] - est[[3]]["S12"],
         diff_share_I_m12_low = 100 * (est[[4]]["I12"] - est[[3]]["I12"]))
  names(o) <- sub("\\..*$", "", names(o))
  c(o, contrast_plateau = unname(o["diff_plateau_high"] - o["diff_plateau_low"]),
    contrast_share_I_m18 = unname(o["diff_share_I_m18_high"] - o["diff_share_I_m18_low"]),
    contrast_s_months_first12 = unname(o["diff_s_months_first12_high"] - o["diff_s_months_first12_low"]),
    contrast_share_I_m12 = unname(o["diff_share_I_m12_high"] - o["diff_share_I_m12_low"]))
}
point <- stats_from(lapply(arr, function(a) c(plateau = mean(a$plateau), I18 = mean(a$I18), S12 = mean(a$S12), I12 = mean(a$I12))))
boot <- matrix(NA, B, length(point), dimnames = list(NULL, names(point)))
for (b in 1:B) {
  est <- vector("list", 4)
  for (gi in 1:4) {
    a <- arr[[gi]]
    ia <- draw(length(a$plateau), length(a$plateau))
    ib <- draw(length(a$S12), length(a$S12))
    est[[gi]] <- c(plateau = mean(a$plateau[ia]), I18 = mean(a$I18[ia]), S12 = mean(a$S12[ib]), I12 = mean(a$I12[ib]))
  }
  boot[b, ] <- stats_from(est)
}
for (k in names(point)) {
  q <- quantile(boot[, k], c(0.025, 0.975), type = 7, names = FALSE)
  rec(k, point[k]); rec(paste0(k, "_ci_low"), q[1]); rec(paste0(k, "_ci_high"), q[2])
}

# ---------------------------------------------------------------------------
# Step 3. Ground-truth check
# ---------------------------------------------------------------------------
cl$true_type <- truth$true_type[match(cl$person_id, truth$person_id)]
ari <- function(a, b) {             # adjusted Rand index (Hubert & Arabie 1985)
  tab <- table(a, b); c2 <- function(x) x * (x - 1) / 2
  s_ij <- sum(c2(tab)); s_a <- sum(c2(rowSums(tab))); s_b <- sum(c2(colSums(tab)))
  expct <- s_a * s_b / c2(sum(tab))
  (s_ij - expct) / ((s_a + s_b) / 2 - expct)
}
rec("ari_recovered_vs_true", ari(cl$true_type, cl$cluster))
rec("pct_correct_type", 100 * mean(cl$true_type == cl$cluster))
for (f in c("HIGH", "LOW")) {
  tp <- sapply(c("pre", "post"), function(p) 100 * mean(cl$true_type[cl$job_family == f & cl$period == p] == "plateau"))
  rec(paste0("true_plateau_shift_", tolower(f)), tp["post"] - tp["pre"])
}
rec("design_plateau_shift_high", 20)
rec("design_plateau_shift_low", 2)

# ---------------------------------------------------------------------------
# Step 4. Decision implication
# ---------------------------------------------------------------------------
v <- setNames(res$value, res$name)
cat("\n=== F01 The Broken Bottom Rung (synthetic data), R ===\n")
for (f in c("high", "low")) {
  cat(sprintf("%s: naive tasks m1-6 %.1f -> %.1f (%+.0f%%) | S months %.2f -> %.2f | %% I at m18 %.0f -> %.0f | %% plateau %.0f -> %.0f\n",
              toupper(f), v[paste0("naive_tasks_m1_6_", f, "_pre")], v[paste0("naive_tasks_m1_6_", f, "_post")],
              v[paste0("naive_tasks_pct_change_", f)], v[paste0("s_months_first12_", f, "_pre")],
              v[paste0("s_months_first12_", f, "_post")], v[paste0("share_I_m18_", f, "_pre")],
              v[paste0("share_I_m18_", f, "_post")], v[paste0("share_plateau_", f, "_pre")],
              v[paste0("share_plateau_", f, "_post")]))
}
cat(sprintf("Contrast plateau shift HIGH - LOW: %+.0f pts [%.0f, %.0f]; ARI %.2f; %.0f%% correctly typed\n",
            v["contrast_plateau"], v["contrast_plateau_ci_low"], v["contrast_plateau_ci_high"],
            v["ari_recovered_vs_true"], v["pct_correct_type"]))
cat("Decision: track time in stretch work and share reaching independent work by tenure month,\n",
    "not only early output. (Descriptive cohort comparison, not causal.)\n", sep = "")

write.csv(res, file.path(out_dir, "results_r.csv"), row.names = FALSE)
cat("Wrote", file.path(out_dir, "results_r.csv"), "\n")
