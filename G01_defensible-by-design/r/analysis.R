# G01 · Defensible by Design: an adverse impact audit of an AI resume screen (base R).
#
# Run from the package root or from anywhere:
#     Rscript r/analysis.R
#
# The same Steps 0-4 as python/analysis.py, with the same definitions, so every
# point estimate matches. Only the Monte Carlo draws differ (R's random-number
# generator is not numpy's); the EXACT false-alarm rates match exactly.
#
# Writes: outputs/r_results.csv (the cell table), outputs/r_headline.csv
#         (every headline number, same names as results.json), outputs/r_false_alarm.csv

# --- locate the package root from this script's path --------------------------
args <- commandArgs(trailingOnly = FALSE)
file_arg <- sub("^--file=", "", args[grep("^--file=", args)])
script_dir <- if (length(file_arg)) dirname(normalizePath(file_arg)) else getwd()
ROOT <- normalizePath(file.path(script_dir, ".."))
DATA <- file.path(ROOT, "data")
OUT <- file.path(ROOT, "outputs")
dir.create(OUT, showWarnings = FALSE)

FOUR_FIFTHS <- 0.80
ALPHA <- 0.05
Z95 <- qnorm(0.975)
MC_SEED <- 20261272
MC_REPS <- 5000
MC_SIZES <- c(25, 50, 100, 250, 1000)
MC_BASE_RATE <- 0.30
POOLS <- c("Software Engineering", "Customer Operations", "Finance", "All families")
STAGES <- c("ai_screen", "interview", "offer", "cumulative")
COMPS <- list(age = c("age_band", "40+", "Under 40"), sex = c("sex", "Woman", "Man"))
FAMILY_FIX <- c("software engineering" = "Software Engineering", "software eng." = "Software Engineering",
                "customer operations" = "Customer Operations", "customer ops" = "Customer Operations",
                "finance" = "Finance")

# round half away from zero (same rule as the Python script)
rnd <- function(x, k) sign(x) * floor(abs(x) * 10^k + 0.5) / 10^k

# --- Step 0. Load and clean ------------------------------------------------------
raw <- read.csv(file.path(DATA, "applications.csv"), colClasses = "character",
                na.strings = character(0))
n_rows <- nrow(raw)
fixed <- unname(FAMILY_FIX[tolower(trimws(raw$job_family))])
stopifnot(!anyNA(fixed))
n_label_fixed <- sum(fixed != raw$job_family)
raw$job_family <- fixed
# Duplicates: keep each applicant's LATEST application (ties broken by application id)
raw <- raw[order(raw$applicant_id, raw$application_date, raw$application_id, method = "radix"), ]
d <- raw[!duplicated(raw$applicant_id, fromLast = TRUE), ]
n_dupes_removed <- n_rows - nrow(d)
for (cc in c("ai_screen_pass", "interview_pass", "offer")) d[[cc]] <- suppressWarnings(as.numeric(d[[cc]]))
# Undisclosed self-ID is excluded from THAT comparison only
n_sex_missing <- sum(!d$sex %in% c("Woman", "Man"))
n_age_missing <- sum(!d$age_band %in% c("40+", "Under 40"))
n_withdrew_int <- sum(d$withdrew == "before_interview")
n_withdrew_off <- sum(d$withdrew == "before_offer")
cat(sprintf("Step 0. %d rows -> %d applicants (%d duplicates removed, %d labels fixed); sex not disclosed %d, age not disclosed %d.\n",
            n_rows, nrow(d), n_dupes_removed, n_label_fixed, n_sex_missing, n_age_missing))

# Who is at risk of selection at each stage, and who was selected
stage_frame <- function(x, stage) {
  if (stage == "ai_screen") return(list(x = x, sel = x$ai_screen_pass == 1))
  if (stage == "interview") { y <- x[!is.na(x$interview_pass), ]; return(list(x = y, sel = y$interview_pass == 1)) }
  if (stage == "offer") { y <- x[!is.na(x$offer), ]; return(list(x = y, sel = y$offer == 1)) }
  list(x = x, sel = !is.na(x$offer) & x$offer == 1)   # cumulative: withdrawals count as not selected
}

# 95% CI for the impact ratio (Katz log method); 0.5 added to every cell if a count is 0
katz_ci <- function(a, n1, c, n2) {
  if (a == 0 || c == 0) { a <- a + 0.5; c <- c + 0.5; n1 <- n1 + 1; n2 <- n2 + 1 }
  ir <- (a / n1) / (c / n2)
  se <- sqrt(1 / a - 1 / n1 + 1 / c - 1 / n2)
  c(ir * exp(-Z95 * se), ir * exp(Z95 * se))
}

audit_cell <- function(a, n1, c, n2) {
  r1 <- a / n1; r2 <- c / n2; ir <- r1 / r2
  ci <- katz_ci(a, n1, c, n2)
  p <- fisher.test(matrix(c(a, n1 - a, c, n2 - c), nrow = 2, byrow = TRUE))$p.value
  focal_lower <- r1 <= r2
  adverse_ir <- if (focal_lower) ir else 1 / ir
  flag <- 5 * min(a * n2, c * n1) < 4 * max(a * n2, c * n1)      # exact "lower/higher < 4/5"
  if (focal_lower) { short <- r2 * n1 - a; short80 <- max(0, FOUR_FIFTHS * r2 * n1 - a) }
  else { short <- r1 * n2 - c; short80 <- max(0, FOUR_FIFTHS * r1 * n2 - c) }
  sig <- p < ALPHA
  verdict <- if (flag && sig) "Adverse impact: act" else if (flag) "Inconclusive: four-fifths flag, not significant" else
    if (sig) "Small but real gap: monitor" else "No evidence of impact"
  data.frame(rate_focal = r1, rate_reference = r2, ir = ir, ir_lo = ci[1], ir_hi = ci[2], diff = r1 - r2,
             fisher_p = p, adverse_ir = adverse_ir, focal_is_lower = focal_lower, four_fifths_flag = flag,
             significant = sig, shortfall = short, shortfall_to_080 = short80, verdict = verdict)
}

rows <- list()
for (stage in STAGES) for (pool in POOLS) {
  x <- if (pool == "All families") d else d[d$job_family == pool, ]
  sf <- stage_frame(x, stage)
  for (comp in names(COMPS)) {
    v <- COMPS[[comp]]
    f <- sf$x[[v[1]]] == v[2]; r <- sf$x[[v[1]]] == v[3]
    a <- sum(sf$sel[f]); n1 <- sum(f); c <- sum(sf$sel[r]); n2 <- sum(r)
    rows[[length(rows) + 1]] <- cbind(data.frame(stage = stage, pool = pool, comparison = comp,
      focal_group = v[2], reference_group = v[3], n_focal = n1, sel_focal = a, n_reference = n2,
      sel_reference = c), audit_cell(a, n1, c, n2))
  }
}
tab <- do.call(rbind, rows)
tab$lower_group <- ifelse(tab$focal_is_lower, tab$focal_group, tab$reference_group)
tab$primary <- tab$stage == "ai_screen" & tab$pool != "All families"
cell <- function(stage, pool, comp) tab[tab$stage == stage & tab$pool == pool & tab$comparison == comp, ]

# --- Step 1. Naive company-wide checks -------------------------------------------
naive_offer <- cell("cumulative", "All families", "age")
naive_ai <- cell("ai_screen", "All families", "age")
naive_ai_sex <- cell("ai_screen", "All families", "sex")
cat(sprintf("Step 1. NAIVE: company-wide offers IR (40+) = %.2f; pooled AI screen IR (40+) = %.2f. Both pass.\n",
            naive_offer$ir, naive_ai$ir))

# --- Step 2. Correct audit ---------------------------------------------------------
cat("Step 2. AI screen by job family:\n")
print(format(tab[tab$stage == "ai_screen", c("pool", "comparison", "n_focal", "n_reference", "ir", "ir_lo",
                                             "ir_hi", "fisher_p", "shortfall", "verdict")], digits = 3), row.names = FALSE)
n_cells <- nrow(tab); n_flag <- sum(tab$four_fifths_flag); n_sig <- sum(tab$significant)
n_act <- sum(tab$verdict == "Adverse impact: act"); n_disagree <- sum(tab$four_fifths_flag != tab$significant)

# --- Step 3. Ground truth -----------------------------------------------------------
truth <- read.csv(file.path(DATA, "true_parameters.csv"), stringsAsFactors = FALSE)
key <- function(df) paste(df$stage, df$pool, df$comparison)
tab$true_ir <- truth$true_ir[match(key(tab), key(truth))]
tab$true_category <- truth$true_category[match(key(tab), key(truth))]
tab$ci_covers_truth <- tab$ir_lo <= tab$true_ir & tab$true_ir <= tab$ir_hi
true_adverse <- pmin(tab$true_ir, 1 / tab$true_ir) < FOUR_FIFTHS
n_true_adverse <- sum(true_adverse)
ff_false <- sum(tab$four_fifths_flag & !true_adverse)
act_false <- sum(tab$verdict == "Adverse impact: act" & !true_adverse)
coverage <- sum(tab$ci_covers_truth)
cat(sprintf("Step 3. %d four-fifths flags (%d false alarms); 'act' false alarms %d; CI covers truth in %d of %d cells.\n",
            n_flag, ff_false, act_false, coverage, n_cells))

# Exact chance of a four-fifths flag (either group) when both truly select at rate p
exact_flag_rate <- function(n1, n2, p, focal_only = FALSE) {
  x <- 0:n1; y <- 0:n2
  X <- outer(x * n2, rep(1, n2 + 1)); Y <- outer(rep(1, n1 + 1), y * n1)
  flag <- if (focal_only) 5 * X < 4 * Y else 5 * pmin(X, Y) < 4 * pmax(X, Y)
  w <- outer(dbinom(x, n1, p), dbinom(y, n2, p))
  sum(w * flag)
}
set.seed(MC_SEED)
mc <- do.call(rbind, lapply(MC_SIZES, function(n) {
  xa <- rbinom(MC_REPS, n, MC_BASE_RATE); xb <- rbinom(MC_REPS, n, MC_BASE_RATE)
  flag <- 5 * pmin(xa, xb) < 4 * pmax(xa, xb)
  pv <- mapply(function(a, b) fisher.test(matrix(c(a, n - a, b, n - b), 2, byrow = TRUE))$p.value, xa, xb)
  data.frame(n_per_group = n, exact_four_fifths_flag_rate = exact_flag_rate(n, n, MC_BASE_RATE),
             mc_four_fifths_flag_rate = mean(flag), mc_fisher_sig_rate = mean(pv < ALPHA),
             mc_both_rate = mean(flag & pv < ALPHA))
}))
write.csv(mc, file.path(OUT, "r_false_alarm.csv"), row.names = FALSE)
print(format(mc, digits = 3), row.names = FALSE)
fin <- cell("ai_screen", "Finance", "sex")
fin_p_both <- exact_flag_rate(fin$n_focal, fin$n_reference, fin$rate_reference)
fin_p_women <- exact_flag_rate(fin$n_focal, fin$n_reference, fin$rate_reference, focal_only = TRUE)
cat(sprintf("        Finance women: parity flags %.0f%% of the time (%.0f%% against women).\n", 100 * fin_p_both, 100 * fin_p_women))

# --- Step 4. Decision -------------------------------------------------------------
swe <- cell("ai_screen", "Software Engineering", "age")
co <- cell("ai_screen", "Customer Operations", "age")
swe_cum <- cell("cumulative", "Software Engineering", "age")
cat(sprintf("Step 4. ACT: SWE 40+ IR %.2f (%.2f-%.2f), shortfall %.0f. MONITOR: CO 40+ IR %.2f, p %.4f. MORE DATA: Finance women IR %.2f, p %.2f.\n",
            swe$ir, swe$ir_lo, swe$ir_hi, swe$shortfall, co$ir, co$fisher_p, fin$ir, fin$fisher_p))

# --- Save ---------------------------------------------------------------------------
cols <- c("stage", "pool", "comparison", "focal_group", "reference_group", "n_focal", "sel_focal",
          "n_reference", "sel_reference", "rate_focal", "rate_reference", "ir", "ir_lo", "ir_hi",
          "diff", "fisher_p", "adverse_ir", "lower_group", "four_fifths_flag", "significant",
          "shortfall", "shortfall_to_080", "verdict", "primary", "true_ir", "true_category", "ci_covers_truth")
out <- tab[, cols]
for (cc in names(out)) if (is.logical(out[[cc]])) out[[cc]] <- ifelse(out[[cc]], "True", "False")
for (cc in names(out)) if (is.double(out[[cc]])) out[[cc]] <- formatC(out[[cc]], digits = 10, format = "g")
write.csv(out, file.path(OUT, "r_results.csv"), row.names = FALSE)

H <- list()
put <- function(name, value, digits = NULL) H[[name]] <<- if (is.null(digits)) value else rnd(value, digits)
put_sig <- function(name, value, sig) H[[name]] <<- signif(value, sig)
put("n_application_rows", n_rows); put("n_applicants", nrow(d)); put("n_duplicates_removed", n_dupes_removed)
put("n_labels_fixed", n_label_fixed); put("n_sex_not_disclosed", n_sex_missing); put("n_age_not_disclosed", n_age_missing)
put("n_withdrew_before_interview", n_withdrew_int); put("n_withdrew_before_offer", n_withdrew_off)
put("naive_offer_ir_age", naive_offer$ir, 2); put("naive_ai_ir_age", naive_ai$ir, 2); put("naive_ai_ir_sex", naive_ai_sex$ir, 2)
for (k in list(list("swe_ai_age", swe), list("co_ai_age", co), list("fin_ai_sex", fin))) {
  key_ <- k[[1]]; c_ <- k[[2]]
  put(paste0(key_, "_n_focal"), c_$n_focal); put(paste0(key_, "_n_reference"), c_$n_reference)
  put(paste0(key_, "_rate_focal"), c_$rate_focal, 3); put(paste0(key_, "_rate_reference"), c_$rate_reference, 3)
  put(paste0(key_, "_ir"), c_$ir, 2); put(paste0(key_, "_ir_lo"), c_$ir_lo, 2); put(paste0(key_, "_ir_hi"), c_$ir_hi, 2)
  put_sig(paste0(key_, "_fisher_p"), c_$fisher_p, 3); put(paste0(key_, "_shortfall"), c_$shortfall, 0)
  put(paste0(key_, "_true_ir"), c_$true_ir, 2)
}
put("swe_ai_age_shortfall_to_080", swe$shortfall_to_080, 0)
put("swe_cum_age_ir", swe_cum$ir, 2)
put("swe_interview_age_ir", cell("interview", "Software Engineering", "age")$ir, 2)
put("fin_parity_flag_prob", fin_p_both, 2); put("fin_parity_flag_prob_women", fin_p_women, 2)
put("n_cells", n_cells); put("n_four_fifths_flags", n_flag); put("n_significant", n_sig); put("n_act", n_act)
put("n_rules_disagree", n_disagree); put("n_true_adverse", n_true_adverse)
put("n_four_fifths_false_alarms", ff_false); put("n_act_false_alarms", act_false); put("n_ci_cover_truth", coverage)
for (i in seq_len(nrow(mc))) {
  n <- mc$n_per_group[i]
  put(paste0("exact_ff_flag_rate_n", n), mc$exact_four_fifths_flag_rate[i], 3)
  put(paste0("mc_ff_flag_rate_n", n), mc$mc_four_fifths_flag_rate[i], 3)
  put(paste0("mc_fisher_sig_rate_n", n), mc$mc_fisher_sig_rate[i], 3)
}
write.csv(data.frame(name = names(H), value = formatC(unlist(H), digits = 12, format = "g")),
          file.path(OUT, "r_headline.csv"), row.names = FALSE)
cat("Wrote outputs/r_results.csv, outputs/r_headline.csv, outputs/r_false_alarm.csv\n")
