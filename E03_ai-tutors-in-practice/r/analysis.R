# analysis.R -- E03 "AI Tutors in Practice: What Interaction Data Reveals"
# Measure Digital Transformation (Dr. Eddie Lin). Synthetic data only. Base R, no packages.
#
# The same four steps and the same numbers as python/analysis.py:
#   Step 1  naive: usage volume vs learning gain
#   Step 2  the ladder on held-out learners: (a) volume, (b) bag of actions, (c) 1st-order Markov
#           log-score, (d) 2nd-order Markov log-score; bootstrap 95% intervals with the same
#           hand-written Park-Miller random number generator as Python, so intervals match exactly.
#           (The Python-only neural-net rung is not reproduced here.)
#   Step 3  ground-truth check: true-label chains and a 3-style mixture of Markov chains (EM)
#   Step 4  decision implication
#
# Run from anywhere:  Rscript r/analysis.R      Writes outputs/results_r.csv

get_script_dir <- function() {
  args <- commandArgs(trailingOnly = FALSE)
  f <- sub("^--file=", "", args[grep("^--file=", args)])
  if (length(f) == 1) return(dirname(normalizePath(f)))
  if (!is.null(sys.frames()[[1]]$ofile)) return(dirname(normalizePath(sys.frames()[[1]]$ofile)))
  getwd()
}
ROOT <- dirname(get_script_dir())
DATA <- file.path(ROOT, "data")
OUT <- file.path(ROOT, "outputs")
dir.create(OUT, showWarnings = FALSE)

ACTIONS <- c("T", "H", "A", "C", "V")
ROWS <- c("S", "T", "H", "A", "C", "V")   # S = START of a session
COLS <- c("T", "H", "A", "C", "V", "Q")
SMOOTH <- 0.5
B <- 1000
SEED <- 20261106
RESULTS <- data.frame(name = character(), value = numeric(), digits = integer())
record <- function(name, value, digits) {
  RESULTS[nrow(RESULTS) + 1, ] <<- list(name, as.numeric(value), as.integer(digits))
}

# ------------------------------------------------------------------------------------
# Shared tools (identical logic in Python)
# ------------------------------------------------------------------------------------
pm_state <- SEED %% 2147483647
pm_index <- function(n, k) {
  # Park-Miller minimal standard LCG; returns k integers in 1..n (with replacement).
  out <- integer(k)
  for (i in seq_len(k)) {
    pm_state <<- (16807 * pm_state) %% 2147483647
    out[i] <- floor(pm_state / 2147483647 * n) + 1
  }
  out
}
auc <- function(y, score) {
  r <- rank(score)                       # ties get average ranks
  n1 <- sum(y == 1); n0 <- sum(y == 0)
  (sum(r[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0)
}
chain <- function(counts) {              # row-normalized transition matrix with smoothing
  c <- counts + SMOOTH
  c / rowSums(c)
}
as_mat <- function(v, n_rows) matrix(v, nrow = n_rows, ncol = 6, byrow = TRUE)   # row-major
flat <- function(M) as.vector(t(M))                                               # row-major
pct_interval <- function(x) {
  s <- sort(x)
  c(s[floor(0.025 * length(s)) + 1], s[ceiling(0.975 * length(s))])
}

# ------------------------------------------------------------------------------------
# Load and clean
# ------------------------------------------------------------------------------------
ev <- read.csv(file.path(DATA, "tutor_events.csv"), colClasses = "character",
               na.strings = character(0), strip.white = FALSE)
lr <- read.csv(file.path(DATA, "learners.csv"), stringsAsFactors = FALSE)
n_events_raw <- nrow(ev)

ev$tok <- toupper(trimws(ev$action))
n_unreadable <- sum(!(ev$tok %in% COLS))
ev <- ev[ev$tok %in% COLS, ]
ev$step <- as.integer(ev$step)
ev <- ev[order(ev$learner_id, ev$session_id, ev$step, method = "radix"), ]
rownames(ev) <- NULL
# Drop anything logged after a Q in the same session.
q_before <- ave(as.integer(ev$tok == "Q"), ev$session_id, FUN = function(z) c(0, head(cumsum(z), -1)))
ev <- ev[q_before == 0, ]
# Previous tokens within each session.
first <- c(TRUE, ev$session_id[-1] != ev$session_id[-nrow(ev)])
second <- c(FALSE, first[-length(first)])
ev$p1 <- c("S", ev$tok[-nrow(ev)]); ev$p1[first] <- "S"
ev$p2 <- c("S", "S", ev$tok[-c(nrow(ev) - 1, nrow(ev))]); ev$p2[first | second] <- "S"
ev$ts <- as.POSIXct(ev$timestamp, format = "%Y-%m-%dT%H:%M:%S", tz = "UTC")

ids <- lr$learner_id
n <- length(ids)
li <- match(ev$learner_id, ids)
r1 <- match(ev$p1, ROWS) - 1; r2 <- match(ev$p2, ROWS) - 1; cc <- match(ev$tok, COLS) - 1
N1 <- matrix(tabulate((li - 1) * 36 + r1 * 6 + cc + 1, n * 36), nrow = n, byrow = TRUE)
N2 <- matrix(tabulate((li - 1) * 216 + (r2 * 6 + r1) * 6 + cc + 1, n * 216), nrow = n, byrow = TRUE)

lr$messages <- tabulate(li[ev$tok != "Q"], n)
s_min <- tapply(as.numeric(ev$ts), ev$session_id, min)
s_max <- tapply(as.numeric(ev$ts), ev$session_id, max)
s_lid <- sub("_.*$", "", names(s_min))
lr$minutes <- as.numeric(tapply((s_max - s_min) / 60, factor(s_lid, levels = ids), sum))
lr$minutes[is.na(lr$minutes)] <- 0
lr$sessions <- as.numeric(table(factor(s_lid, levels = ids)))
act <- matrix(tabulate((li[ev$tok %in% ACTIONS] - 1) * 5 + match(ev$tok[ev$tok %in% ACTIONS], ACTIONS), n * 5),
              nrow = n, byrow = TRUE)
shares <- act / pmax(rowSums(act), 1)
for (j in 1:5) lr[[paste0("share_", ACTIONS[j])]] <- shares[, j]

lr$gain <- lr$post_score - lr$pre_score
n_missing_post <- sum(is.na(lr$post_score))
lr$gained <- ifelse(is.na(lr$post_score), NA, as.numeric(lr$gain >= 10))
has_post <- !is.na(lr$post_score)
is_train <- lr$split == "train" & has_post
is_test <- lr$split == "test" & has_post

record("n_learners", n, 0)
record("n_events", n_events_raw, 0)
record("n_unreadable_tokens_dropped", n_unreadable, 0)
record("n_missing_post_test", n_missing_post, 0)
record("n_short_logs_under_5_actions", sum(lr$messages < 5), 0)
record("n_train_with_post", sum(is_train), 0)
record("n_test_with_post", sum(is_test), 0)
record("share_gained", mean(lr$gained[has_post]) * 100, 0)
cat(sprintf("%d learners, %d events; dropped %d unreadable tokens; %d learners without a post-test kept out of the gain analysis.\n",
            n, n_events_raw, n_unreadable, n_missing_post))

# ------------------------------------------------------------------------------------
# Step 1 -- naive
# ------------------------------------------------------------------------------------
d <- lr[has_post, ]
record("r_messages_gain", cor(d$messages, d$gain), 2)
record("r_minutes_gain", cor(d$minutes, d$gain), 2)
record("mean_messages_gained", mean(d$messages[d$gained == 1]), 1)
record("mean_messages_not_gained", mean(d$messages[d$gained == 0]), 1)
cat(sprintf("\nStep 1 (naive): r(messages, gain) = %.2f; r(minutes, gain) = %.2f\n",
            cor(d$messages, d$gain), cor(d$minutes, d$gain)))

# ------------------------------------------------------------------------------------
# Step 2 -- the ladder on held-out learners
# ------------------------------------------------------------------------------------
y_tr <- lr$gained[is_train]
y_te <- lr$gained[is_test]
logit_scores <- function(cols) {
  tr <- lr[is_train, c(cols, "gained")]
  fit <- glm(gained ~ ., data = tr, family = binomial(), control = glm.control(epsilon = 1e-14, maxit = 100))
  as.vector(cbind(1, as.matrix(lr[is_test, cols])) %*% coef(fit))
}
markov_logratio <- function(Ncounts, n_rows, train_mask, y) {
  g <- as_mat(colSums(Ncounts[train_mask, ][y == 1, ]), n_rows)
  ng <- as_mat(colSums(Ncounts[train_mask, ][y == 0, ]), n_rows)
  log(chain(g)) - log(chain(ng))
}
# log-score(sequence) = sum over steps of log[ P_gain(next | current) / P_nogain(next | current) ]
LR1 <- markov_logratio(N1, 6, is_train, y_tr)
LR2 <- markov_logratio(N2, 36, is_train, y_tr)
lr$markov1_score <- as.vector(N1 %*% flat(LR1))
lr$markov2_score <- as.vector(N2 %*% flat(LR2))

scores_te <- list(
  volume = logit_scores(c("messages", "minutes", "sessions")),
  bag = logit_scores(c("share_T", "share_H", "share_A", "share_C")),
  markov1 = lr$markov1_score[is_test],
  markov2 = lr$markov2_score[is_test]
)
auc_point <- sapply(scores_te, function(s) auc(y_te, s))
cat("\nStep 2 (held-out AUC)\n"); print(round(auc_point, 2))

te <- lr[is_test, ]
ord <- order(te$markov1_score, te$learner_id, method = "radix")
q <- nrow(te) %/% 4
quart <- integer(nrow(te))
quart[ord[1:q]] <- 1
quart[ord[(nrow(te) - q + 1):nrow(te)]] <- 4
gain_te <- te$gain
record("gain_top_quartile_markov", mean(gain_te[quart == 4]), 1)
record("gain_bottom_quartile_markov", mean(gain_te[quart == 1]), 1)
diff_q <- mean(gain_te[quart == 4]) - mean(gain_te[quart == 1])

boot <- matrix(NA_real_, B, 8,
               dimnames = list(NULL, c(names(scores_te), "m1_minus_bag", "m1_minus_volume", "m2_minus_m1", "qdiff")))
for (b in 1:B) {
  ix <- pm_index(nrow(te), nrow(te))
  yb <- y_te[ix]
  a <- sapply(scores_te, function(s) auc(yb, s[ix]))
  boot[b, 1:4] <- a
  boot[b, "m1_minus_bag"] <- a["markov1"] - a["bag"]
  boot[b, "m1_minus_volume"] <- a["markov1"] - a["volume"]
  boot[b, "m2_minus_m1"] <- a["markov2"] - a["markov1"]
  gb <- gain_te[ix]; qb <- quart[ix]
  boot[b, "qdiff"] <- mean(gb[qb == 4]) - mean(gb[qb == 1])
}
for (k in names(scores_te)) {
  ci <- pct_interval(boot[, k])
  record(paste0("auc_", k), auc_point[[k]], 2)
  record(paste0("auc_", k, "_lo"), ci[1], 2)
  record(paste0("auc_", k, "_hi"), ci[2], 2)
}
pts <- c(m1_minus_bag = auc_point[["markov1"]] - auc_point[["bag"]],
         m1_minus_volume = auc_point[["markov1"]] - auc_point[["volume"]],
         m2_minus_m1 = auc_point[["markov2"]] - auc_point[["markov1"]])
for (k in names(pts)) {
  ci <- pct_interval(boot[, k])
  record(paste0("auc_diff_", k), pts[[k]], 2)
  record(paste0("auc_diff_", k, "_lo"), ci[1], 2)
  record(paste0("auc_diff_", k, "_hi"), ci[2], 2)
}
ci <- pct_interval(boot[, "qdiff"])
record("gain_diff_top_vs_bottom_quartile", diff_q, 1)
record("gain_diff_top_vs_bottom_quartile_lo", ci[1], 1)
record("gain_diff_top_vs_bottom_quartile_hi", ci[2], 1)
cat(sprintf("  Top vs bottom quartile of the Markov log-score: %.1f points (95%% interval %.1f to %.1f)\n",
            diff_q, ci[1], ci[2]))

key_transitions <- function(Nsub, ysub) {
  g <- chain(as_mat(colSums(Nsub[ysub == 1, , drop = FALSE]), 6))
  ng <- chain(as_mat(colSums(Nsub[ysub == 0, , drop = FALSE]), 6))
  cell <- function(P, r, c) P[match(r, ROWS), match(c, COLS)]
  c(AC_gained = cell(g, "A", "C"), AC_not = cell(ng, "A", "C"),
    HT_gained = cell(g, "H", "T"), HT_not = cell(ng, "H", "T"),
    HA_gained = cell(g, "H", "A"), HA_not = cell(ng, "H", "A"),
    AT_gained = cell(g, "A", "T"), AT_not = cell(ng, "A", "T"),
    TV_gained = cell(g, "T", "V"), TV_not = cell(ng, "T", "V"))
}
N_post <- N1[has_post, ]
y_post <- lr$gained[has_post]
kt <- key_transitions(N_post, y_post)
bt <- matrix(NA_real_, B, 2, dimnames = list(NULL, c("AC", "HT")))
for (b in 1:B) {
  ix <- pm_index(length(y_post), length(y_post))
  k2 <- key_transitions(N_post[ix, ], y_post[ix])
  bt[b, "AC"] <- k2[["AC_not"]] - k2[["AC_gained"]]
  bt[b, "HT"] <- k2[["HT_gained"]] - k2[["HT_not"]]
}
for (k in names(kt)) record(paste0("p_", k), kt[[k]], 2)
ci <- pct_interval(bt[, "AC"])
record("p_AC_not_minus_gained", kt[["AC_not"]] - kt[["AC_gained"]], 2)
record("p_AC_not_minus_gained_lo", ci[1], 2); record("p_AC_not_minus_gained_hi", ci[2], 2)
ci <- pct_interval(bt[, "HT"])
record("p_HT_gained_minus_not", kt[["HT_gained"]] - kt[["HT_not"]], 2)
record("p_HT_gained_minus_not_lo", ci[1], 2); record("p_HT_gained_minus_not_hi", ci[2], 2)

ordp <- order(lr$pre_score, seq_len(n), method = "radix")
qp <- n %/% 4
p_ac <- function(rows) chain(as_mat(colSums(N1[rows, ]), 6))[match("A", ROWS), match("C", COLS)]
record("p_AC_low_prior_quartile", p_ac(ordp[1:qp]), 2)
record("p_AC_high_prior_quartile", p_ac(ordp[(n - qp + 1):n]), 2)

# ------------------------------------------------------------------------------------
# Step 3 -- ground-truth check
# ------------------------------------------------------------------------------------
gtt <- read.csv(file.path(DATA, "ground_truth_transitions.csv"), stringsAsFactors = FALSE)
gt <- read.csv(file.path(DATA, "ground_truth.csv"), stringsAsFactors = FALSE)
STY <- c("scaffolded", "answer_seeking", "light_mixed")
TRUE_P <- lapply(setNames(STY, STY), function(s) as_mat(gtt$true_prob[gtt$style == s], 6))
true_style <- gt$true_style[match(ids, gt$learner_id)]

mae_true <- mean(sapply(STY, function(s) mean(abs(chain(as_mat(colSums(N1[true_style == s, ]), 6)) - TRUE_P[[s]]))))
record("mae_transition_true_labels", mae_true, 3)

rate <- function(r, c) {
  ri <- match(r, ROWS) - 1
  num <- N1[, ri * 6 + match(c, COLS)]
  den <- rowSums(N1[, (ri * 6 + 1):(ri * 6 + 6)])
  ifelse(den > 0, num / pmax(den, 1), 0)
}
init <- rep(3, n)
init[rate("H", "T") >= 0.5] <- 1
init[rate("A", "C") >= 0.4] <- 2
R <- matrix(0, n, 3); R[cbind(1:n, init)] <- 1
for (it in 1:200) {
  pi_k <- colMeans(R)
  ll <- sapply(1:3, function(k) log(pi_k[k]) + as.vector(N1 %*% flat(log(chain(as_mat(colSums(R[, k] * N1), 6))))))
  ll <- ll - apply(ll, 1, max)
  R <- exp(ll); R <- R / rowSums(R)
}
em_label <- STY[apply(R, 1, which.max)]
mae_em <- mean(sapply(1:3, function(k) mean(abs(chain(as_mat(colSums(R[, k] * N1), 6)) - TRUE_P[[STY[k]]]))))
record("mae_transition_em", mae_em, 3)
record("style_accuracy_em", mean(em_label == true_style) * 100, 0)
for (k in 1:3) {
  record(paste0("style_share_true_", STY[k]), mean(true_style == STY[k]) * 100, 0)
  record(paste0("style_share_em_", STY[k]), mean(R[, k]) * 100, 0)
}
for (s in STY) record(paste0("mean_gain_true_", s), mean(lr$gain[has_post & true_style == s]), 1)
cat(sprintf("\nStep 3: MAE true labels %.3f, EM %.3f; EM style accuracy %.0f%%\n",
            mae_true, mae_em, mean(em_label == true_style) * 100))

# ------------------------------------------------------------------------------------
# Step 4 -- decision implication
# ------------------------------------------------------------------------------------
cat("\nStep 4: log action TYPES, not only volume; report P(attempt | hint) and P(copy | answer) by cohort;\n",
    "test tutor designs (hints before answers) with an experiment.\n")

write.csv(RESULTS, file.path(OUT, "results_r.csv"), row.names = FALSE)
cat("Wrote outputs/results_r.csv\n")
