# C02 Golden Sets: the same four steps as python/analysis.py, in base R (+ jsonlite for JSON).
#
# Run from the package folder:  Rscript r/analysis.R
# Run python/analysis.py FIRST: this script reads outputs/results_python_full_precision.json,
# checks that every number matches Python (to 1e-6), stops with an error if not, and fills
# the "r" values in outputs/results.json.
#
# Dawid-Skene is implemented from scratch (no packages). Same data files, same initialization
# (majority vote), same pseudo-counts, same stopping rule and same tie-breaking as Python.

suppressMessages(library(jsonlite))

args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grep("^--file=", args)])
HERE <- if (length(script)) dirname(normalizePath(script)) else getwd()
ROOT <- dirname(HERE)
DATA <- file.path(ROOT, "data"); OUT <- file.path(ROOT, "outputs")

BUDGET <- 100; PSEUDO <- 1; TOL <- 1e-9; MAX_ITER <- 1000

# ---------------------------------------------------------------- Step 0: load and clean
raw <- read.csv(file.path(DATA, "labels_long.csv"), stringsAsFactors = FALSE)
n_raw <- nrow(raw)
dedup <- raw[!duplicated(raw), ]                 # rule 1: drop exact duplicate rows
n_dup <- n_raw - nrow(dedup)
lab <- dedup[!is.na(dedup$label), ]              # rule 2: a skipped (blank) label is not a vote
n_skipped <- nrow(dedup) - nrow(lab)

items <- read.csv(file.path(DATA, "items.csv"), stringsAsFactors = FALSE)
draws <- read.csv(file.path(DATA, "expert_adjudication_draws.csv"), stringsAsFactors = FALSE)
truth <- read.csv(file.path(DATA, "ground_truth_items.csv"), stringsAsFactors = FALSE)
true_lab <- read.csv(file.path(DATA, "ground_truth_labelers.csv"), stringsAsFactors = FALSE)

item_ids <- items$item_id
labeler_ids <- sort(unique(lab$labeler_id))
I <- length(item_ids); J <- length(labeler_ids)
ii <- match(lab$item_id, item_ids)
jj <- match(lab$labeler_id, labeler_ids)
y <- lab$label

# sum of w by group index g (like numpy.bincount with weights)
bsum <- function(g, w, n) { out <- numeric(n); s <- tapply(w, g, sum); out[as.integer(names(s))] <- s; out }

n_err_votes <- bsum(ii, y, I)
n_votes <- bsum(ii, rep(1, length(y)), I)
n_ok_votes <- n_votes - n_err_votes
y_true <- truth$true_label[match(item_ids, truth$item_id)]
ambig <- truth$ambiguous[match(item_ids, truth$item_id)]
rand_order <- draws$random_order[match(item_ids, draws$item_id)]
expert <- draws$expert_label[match(item_ids, draws$item_id)]
judge <- items$ai_judge_flag

key_metrics <- function(key) list(
  wrong = sum(key != y_true),
  missed_errors = sum(key == 0 & y_true == 1),
  false_alarms = sum(key == 1 & y_true == 0),
  accuracy = mean(key == y_true),
  sensitivity = mean(key[y_true == 1]),
  prevalence = mean(key),
  wrong_on_ambiguous = sum(key != y_true & ambig == 1))

# ---------------------------------------------------------------- Step 1: majority vote (ties -> OK)
mv <- as.integer(n_err_votes > n_ok_votes)
n_ties <- sum(n_err_votes == n_ok_votes)
pairs <- n_err_votes * (n_err_votes - 1) / 2 + n_ok_votes * (n_ok_votes - 1) / 2
pairwise_agreement <- sum(pairs) / sum(n_votes * (n_votes - 1) / 2)

# ---------------------------------------------------------------- Step 2: Dawid-Skene EM
dawid_skene <- function(init_T) {
  T <- as.numeric(init_T)
  for (it in 1:MAX_ITER) {
    # M-step: prevalence and each labeler's sensitivity / specificity (with pseudo-counts)
    prev <- mean(T)
    sens <- (bsum(jj, T[ii] * y, J) + PSEUDO) / (bsum(jj, T[ii], J) + 2 * PSEUDO)
    spec <- (bsum(jj, (1 - T[ii]) * (1 - y), J) + PSEUDO) / (bsum(jj, 1 - T[ii], J) + 2 * PSEUDO)
    # E-step: log P(labels, error) and log P(labels, OK) for each item
    la <- log(prev) + bsum(ii, y * log(sens[jj]) + (1 - y) * log(1 - sens[jj]), I)
    lb <- log(1 - prev) + bsum(ii, y * log(1 - spec[jj]) + (1 - y) * log(spec[jj]), I)
    T_new <- 1 / (1 + exp(lb - la))
    delta <- max(abs(T_new - T))
    T <- T_new
    if (delta < TOL) break
  }
  m <- pmax(la, lb)
  list(T = T, prev = prev, sens = sens, spec = spec, iterations = it,
       loglik = sum(m + log(exp(la - m) + exp(lb - m))))
}
ds <- dawid_skene(mv)
post <- ds$T
ds_key <- as.integer(post > 0.5)

tl <- true_lab[match(labeler_ids, true_lab$labeler_id), ]
n_items_lab <- bsum(jj, rep(1, length(y)), J)
mv_ref_sens <- bsum(jj, mv[ii] * y, J) / bsum(jj, mv[ii], J)
real_sens <- bsum(jj, y_true[ii] * y, J) / bsum(jj, y_true[ii], J)
real_spec <- bsum(jj, (1 - y_true[ii]) * (1 - y), J) / bsum(jj, 1 - y_true[ii], J)
big <- n_items_lab >= 30
lenient <- tl$group == "lenient"

# ---------------------------------------------------------------- Step 3: adjudication, same budget
adjudicate <- function(key, chosen) { key[chosen] <- expert[chosen]; key }
pick <- function(score) order(round(score, 10), rand_order)[1:BUDGET]
pick_targeted <- pick(abs(post - 0.5))
pick_random <- pick(as.numeric(rand_order))
pick_margin <- pick(abs(n_err_votes - n_ok_votes) / n_votes)
keys <- list(
  majority_vote = mv,
  mv_plus_random_adjudication = adjudicate(mv, pick_random),
  mv_plus_margin_adjudication = adjudicate(mv, pick_margin),
  dawid_skene = ds_key,
  ds_plus_random_adjudication = adjudicate(ds_key, pick_random),
  ds_plus_targeted_adjudication = adjudicate(ds_key, pick_targeted))
km <- lapply(keys, key_metrics)
final <- keys$ds_plus_targeted_adjudication

# How lucky or unlucky was the single random draw? Exact distribution (same as Python):
# W ~ Hypergeometric(1000, 78 wrong, 100 drawn); net = Binomial(W, 0.98) - Binomial(100 - W, 0.02).
EXPERT_ACC <- 0.98
n_wrong_ds <- sum(ds_key != y_true)
net_pmf <- numeric(0)
for (w in 0:min(BUDGET, n_wrong_ds)) {
  pw <- dhyper(w, n_wrong_ds, I - n_wrong_ds, BUDGET)
  if (pw == 0) next
  pf <- dbinom(0:w, w, EXPERT_ACC); pm <- dbinom(0:(BUDGET - w), BUDGET - w, 1 - EXPERT_ACC)
  joint <- outer(pf, pm) * pw                     # rows f = 0..w, cols m = 0..(BUDGET - w)
  nets <- outer(0:w, 0:(BUDGET - w), "-")
  s <- tapply(as.vector(joint), as.vector(nets), sum)
  for (k in names(s)) net_pmf[k] <- (if (is.na(net_pmf[k])) 0 else net_pmf[k]) + s[[k]]
}
net_vals <- as.integer(names(net_pmf)); o <- order(net_vals)
net_vals <- net_vals[o]; net_p <- as.numeric(net_pmf[o]); net_cdf <- cumsum(net_p)
random_expected_net_fixed <- BUDGET * (n_wrong_ds / I * EXPERT_ACC - (1 - n_wrong_ds / I) * (1 - EXPERT_ACC))
random_net_fixed_lo95 <- net_vals[which(net_cdf >= 0.025)[1]]
random_net_fixed_hi95 <- net_vals[which(net_cdf >= 0.975)[1]]
random_p_net_le_2 <- sum(net_p[net_vals <= 2])
tw <- sum(ds_key[pick_targeted] != y_true[pick_targeted])
targeted_expected_net_fixed <- tw * EXPERT_ACC - (BUDGET - tw) * (1 - EXPERT_ACC)

# ---------------------------------------------------------------- Step 4: decision implication
judge_scores <- function(key) list(recall = mean(judge[key == 1]), precision = mean(key[judge == 1]))
agent_rate <- list(majority_vote = mean(mv), dawid_skene = mean(ds_key),
                   ds_plus_targeted_adjudication = mean(final), truth = mean(y_true))
judge_by_key <- list(majority_vote = judge_scores(mv), dawid_skene = judge_scores(ds_key),
                     ds_plus_targeted_adjudication = judge_scores(final), truth = judge_scores(y_true))

# ---------------------------------------------------------------- collect, compare with Python
full <- list(
  n_label_rows_raw = n_raw, n_duplicate_rows_dropped = n_dup, n_skipped_labels = n_skipped,
  n_labels_used = nrow(lab), n_items = I, n_labelers = J,
  n_items_under_3_labels = sum(n_votes < 3), mean_labels_per_item = mean(n_votes),
  lenient_share_of_labels = mean(lab$labeler_id %in% labeler_ids[lenient]),
  true_prevalence = mean(y_true), n_true_errors = sum(y_true), n_ambiguous = sum(ambig),
  pairwise_agreement = pairwise_agreement, mv_ties = n_ties,
  ds_iterations = ds$iterations, ds_loglik = ds$loglik, ds_prevalence_param = ds$prev,
  mae_sens_ds = mean(abs(ds$sens[big] - real_sens[big])),
  mae_spec_ds = mean(abs(ds$spec[big] - real_spec[big])),
  mae_sens_mv_referenced = mean(abs(mv_ref_sens[big] - real_sens[big])),
  mae_sens_ds_vs_param = mean(abs(ds$sens[big] - tl$true_sensitivity[big])),
  corr_sens_ds = cor(ds$sens[big], real_sens[big]),
  ds_expected_prevalence = mean(post),
  targeted_items_wrong_before = sum(ds_key[pick_targeted] != y_true[pick_targeted]),
  random_items_wrong_before = sum(ds_key[pick_random] != y_true[pick_random]),
  targeted_fixed = sum(ds_key != y_true & final == y_true),
  random_fixed = sum(ds_key != y_true & keys$ds_plus_random_adjudication == y_true),
  final_wrong_ambiguous_share = km$ds_plus_targeted_adjudication$wrong_on_ambiguous / km$ds_plus_targeted_adjudication$wrong,
  mean_uncertainty_ambiguous = mean(pmin(post, 1 - post)[ambig == 1]),
  mean_uncertainty_clear = mean(pmin(post, 1 - post)[ambig == 0]),
  random_expected_net_fixed = random_expected_net_fixed, random_net_fixed_lo95 = random_net_fixed_lo95,
  random_net_fixed_hi95 = random_net_fixed_hi95, random_p_net_le_2 = random_p_net_le_2,
  targeted_expected_net_fixed = targeted_expected_net_fixed, random_net_pmf_total = sum(net_p))
for (k in names(km)) for (mk in names(km[[k]])) full[[paste0(k, "__", mk)]] <- km[[k]][[mk]]
for (k in names(agent_rate)) full[[paste0("agent_error_rate__", k)]] <- agent_rate[[k]]
for (k in names(judge_by_key)) {
  full[[paste0("judge_recall__", k)]] <- judge_by_key[[k]]$recall
  full[[paste0("judge_precision__", k)]] <- judge_by_key[[k]]$precision
}
for (j in seq_len(J)) {
  lid <- labeler_ids[j]
  full[[paste0("labeler__", lid, "__ds_sens")]] <- ds$sens[j]
  full[[paste0("labeler__", lid, "__ds_spec")]] <- ds$spec[j]
  full[[paste0("labeler__", lid, "__mv_ref_sens")]] <- mv_ref_sens[j]
  full[[paste0("labeler__", lid, "__n_labels")]] <- n_items_lab[j]
  full[[paste0("labeler__", lid, "__realized_sens")]] <- real_sens[j]
}

py <- fromJSON(file.path(OUT, "results_python_full_precision.json"))
missing <- setdiff(names(py), names(full))
if (length(missing)) stop("R is missing values: ", paste(missing, collapse = ", "))
diffs <- sapply(names(py), function(k) abs(as.numeric(full[[k]]) - as.numeric(py[[k]])))
cat(sprintf("Checked %d values against Python: max abs difference %.2e\n", length(diffs), max(diffs)))
if (max(diffs) > 1e-6) {
  print(diffs[diffs > 1e-6]); stop("R and Python disagree beyond 1e-6")
}

# ---------------------------------------------------------------- fill the 'r' values in results.json
res <- fromJSON(file.path(OUT, "results.json"), simplifyVector = FALSE)
r_name <- c(mv_prevalence = "majority_vote__prevalence", mv_sensitivity = "majority_vote__sensitivity",
            mv_wrong = "majority_vote__wrong", mv_missed_errors = "majority_vote__missed_errors",
            mv_accuracy = "majority_vote__accuracy", ds_prevalence = "dawid_skene__prevalence",
            ds_sensitivity = "dawid_skene__sensitivity", ds_wrong = "dawid_skene__wrong",
            ds_missed_errors = "dawid_skene__missed_errors", ds_accuracy = "dawid_skene__accuracy",
            ds_targeted_wrong = "ds_plus_targeted_adjudication__wrong",
            ds_random_wrong = "ds_plus_random_adjudication__wrong",
            mv_margin_wrong = "mv_plus_margin_adjudication__wrong",
            mv_random_wrong = "mv_plus_random_adjudication__wrong",
            ds_targeted_sensitivity = "ds_plus_targeted_adjudication__sensitivity",
            ds_targeted_prevalence = "ds_plus_targeted_adjudication__prevalence",
            judge_precision_mv = "judge_precision__majority_vote",
            judge_precision_ds_targeted = "judge_precision__ds_plus_targeted_adjudication",
            judge_precision_truth = "judge_precision__truth")
n_bad <- 0
for (k in seq_along(res$results)) {
  e <- res$results[[k]]
  src <- if (e$name %in% names(r_name)) r_name[[e$name]] else e$name
  v <- as.numeric(full[[src]])
  x <- if (e$unit %in% c("pct", "pp")) v * 100 else v
  rv <- round(x, e$digits)
  res$results[[k]]$r <- if (e$digits == 0) as.integer(rv) else rv
  if (abs(rv - as.numeric(e$python)) > 1e-9) {
    n_bad <- n_bad + 1
    cat("MISMATCH:", e$name, "python", e$python, "r", rv, "\n")
  }
}
if (n_bad > 0) stop(n_bad, " headline numbers disagree after rounding")
res$r_check <- sprintf("R reproduced all %d headline numbers at the stated rounding; max abs difference on %d full-precision values %.1e",
                       length(res$results), length(diffs), max(diffs))
writeLines(toJSON(res, auto_unbox = TRUE, pretty = TRUE, digits = NA, null = "null"),
           file.path(OUT, "results.json"))

cat(sprintf("Step 1  majority vote: %d wrong, sensitivity %.1f%%, prevalence %.1f%% (true %.1f%%)\n",
            km$majority_vote$wrong, 100 * km$majority_vote$sensitivity, 100 * km$majority_vote$prevalence, 100 * mean(y_true)))
cat(sprintf("Step 2  Dawid-Skene: %d wrong, sensitivity %.1f%%, prevalence %.1f%% (%d iterations)\n",
            km$dawid_skene$wrong, 100 * km$dawid_skene$sensitivity, 100 * km$dawid_skene$prevalence, ds$iterations))
cat(sprintf("Step 3  + 100 targeted: %d wrong; + 100 random: %d wrong; MV + 100 closest: %d wrong\n",
            km$ds_plus_targeted_adjudication$wrong, km$ds_plus_random_adjudication$wrong, km$mv_plus_margin_adjudication$wrong))
cat(sprintf("Step 4  AI agent error rate: MV %.1f%%, DS %.1f%%, DS+targeted %.1f%%, truth %.1f%%\n",
            100 * agent_rate$majority_vote, 100 * agent_rate$dawid_skene,
            100 * agent_rate$ds_plus_targeted_adjudication, 100 * agent_rate$truth))
cat("All headline numbers agree with Python at the stated rounding.\n")
