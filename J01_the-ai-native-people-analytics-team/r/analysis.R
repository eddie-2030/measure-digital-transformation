# analysis.R -- Can we trust AI-generated answers about our workforce?
# =====================================================================
# Edition J01, "The AI-Native People Analytics Team" (Measure Digital Transformation).
# Base R only. Same four steps and the same point estimates as python/analysis.py:
#   Step 1 naive (thumbs-up rate, raw judge "accuracy")
#   Step 2 correct method (retrieval recall@5 / MRR, groundedness, judge validation,
#          Rogan-Gladen correction with a cluster bootstrap, failure decomposition)
#   Step 3 ground-truth check     Step 4 decision
# It reads the CSV written by data/generate_data.py (all records simulated; no LLM calls).
# Run from the package root:   Rscript r/analysis.R
# Writes outputs/r_results.csv (name,value); python/check_agreement.py merges it into results.json.

args <- commandArgs(trailingOnly = FALSE)
script <- sub("^--file=", "", args[grep("^--file=", args)])
ROOT <- if (length(script)) normalizePath(file.path(dirname(script), "..")) else getwd()
DATA <- file.path(ROOT, "data"); OUT <- file.path(ROOT, "outputs")
dir.create(OUT, showWarnings = FALSE)

TYPES <- c("policy_lookup", "computed_metric", "multi_hop")
K <- 5; GROUNDED_THRESHOLD <- 0.8; MIN_PER_CELL <- 30
B <- 2000; BOOT_SEED <- 2026; DEPLOY_AT <- 0.80; ROUTE_AT <- 0.60

res <- list()                                   # name -> value
put <- function(name, value, digits = 3) {      # round like Python's round()
  res[[name]] <<- if (is.na(value)) NA else round(value, digits)
  invisible(res[[name]])
}
rogan_gladen <- function(judge_rate, sens, spec) {
  # Corrected accuracy = (judge rate + spec - 1) / (sens + spec - 1), clipped to [0, 1]
  denom <- sens + spec - 1
  if (is.na(denom) || denom <= 0) return(NA_real_)
  min(max((judge_rate + spec - 1) / denom, 0), 1)
}

# ------------------------------------------------------------------ load & clean
df <- read.csv(file.path(DATA, "rag_eval_records.csv"), stringsAsFactors = FALSE,
               na.strings = c("NA", ""))
truth <- read.csv(file.path(DATA, "true_parameters.csv"), stringsAsFactors = FALSE)
tv <- setNames(truth$value, truth$name)

parse_ids <- function(s) {   # 'D012; d045 ' -> c("D012", "D045")
  if (is.na(s) || trimws(s) == "") return(character(0))
  x <- toupper(trimws(strsplit(s, ";", fixed = TRUE)[[1]]))
  x[x != ""]
}
gold <- lapply(df$gold_doc_ids, parse_ids)
retr <- lapply(df$retrieved_doc_ids, function(s) head(parse_ids(s), K))

df$recall5 <- mapply(function(g, r) if (!length(g)) NA else mean(g %in% r), gold, retr)
df$rr <- mapply(function(g, r) {
  if (!length(g)) return(NA)
  hit <- which(r %in% g)
  if (length(hit)) 1 / hit[1] else 0
}, gold, retr)
df$full_hit <- ifelse(df$answerable == 1, as.numeric(df$recall5 == 1), NA)
df$groundedness <- ifelse(df$n_claims > 0, df$n_claims_supported / df$n_claims, NA)
cat(sprintf("Loaded %d questions in %d clusters; %d have no source in the corpus.\n",
            nrow(df), length(unique(df$cluster_id)), sum(df$answerable == 0)))

# ================================================================ STEP 1: NAIVE
rated <- df[!is.na(df$thumbs_up), ]; judged <- df[!is.na(df$judge_label), ]
put("naive_thumbs_up_rate", mean(rated$thumbs_up))
put("naive_n_rated", nrow(rated), 0)
put("naive_judge_accuracy", mean(judged$judge_label))
put("naive_n_judged", nrow(judged), 0)
for (t in TYPES) {
  put(paste0("naive_judge_accuracy_", t), mean(judged$judge_label[judged$question_type == t]))
  put(paste0("naive_thumbs_up_rate_", t), mean(rated$thumbs_up[rated$question_type == t]))
}
hr <- df[df$human_graded == 1 & !is.na(df$thumbs_up), ]
put("thumbs_up_rate_when_correct_human", mean(hr$thumbs_up[hr$human_label == 1]))
put("thumbs_up_rate_when_wrong_human", mean(hr$thumbs_up[hr$human_label == 0]))
cat(sprintf("\nSTEP 1 (naive): thumbs-up %.3f, raw judge accuracy %.3f\n",
            res$naive_thumbs_up_rate, res$naive_judge_accuracy))

# ======================================================= STEP 2a: RETRIEVAL
ans <- df[df$answerable == 1, ]
for (t in c(TYPES, "overall")) {
  s <- if (t == "overall") ans else ans[ans$question_type == t, ]
  put(paste0("recall_at5_", t), mean(s$recall5))
  put(paste0("hit_rate_", t), mean(s$full_hit))
  put(paste0("mrr_", t), mean(s$rr))
}
# ======================================================= STEP 2b: GROUNDEDNESS
answered <- df[df$n_claims > 0, ]
for (t in c(TYPES, "overall")) {
  s <- if (t == "overall") answered else answered[answered$question_type == t, ]
  put(paste0("groundedness_", t), mean(s$groundedness))
  put(paste0("share_grounded_", t), mean(s$groundedness >= GROUNDED_THRESHOLD))
}
put("abstain_rate_unanswerable", mean(df$abstained[df$answerable == 0]))
put("n_unanswerable", sum(df$answerable == 0), 0)

# ======================================================= STEP 2c: VALIDATE THE JUDGE
hv <- df[df$human_graded == 1 & !is.na(df$judge_label), ]
sens_spec <- function(d) {
  pos <- d[d$human_label == 1, ]; neg <- d[d$human_label == 0, ]
  c(sens = mean(pos$judge_label == 1), spec = mean(neg$judge_label == 0),
    npos = nrow(pos), nneg = nrow(neg))
}
ss <- sens_spec(hv); SENS <- ss[["sens"]]; SPEC <- ss[["spec"]]
put("n_human_graded", sum(df$human_graded == 1), 0)
put("n_human_graded_with_judge", nrow(hv), 0)
put("judge_sensitivity", SENS); put("judge_specificity", SPEC)
put("n_human_correct", ss[["npos"]], 0); put("n_human_wrong", ss[["nneg"]], 0)
for (t in TYPES) {
  st <- sens_spec(hv[hv$question_type == t, ])
  put(paste0("n_human_correct_", t), st[["npos"]], 0)
  put(paste0("n_human_wrong_", t), st[["nneg"]], 0)
  # fewer than 30 in a cell -> per-type sens/spec too noisy -> use pooled values
}
cat(sprintf("\nSTEP 2c: judge sensitivity %.3f, specificity %.3f (pooled, used for all types)\n", SENS, SPEC))

h_all <- df[df$human_graded == 1, ]
p_h <- mean(h_all$human_label); se_h <- sqrt(p_h * (1 - p_h) / nrow(h_all))
put("human_only_accuracy", p_h)
put("human_only_ci_low", p_h - 1.96 * se_h); put("human_only_ci_high", p_h + 1.96 * se_h)

# ======================================================= STEP 2d: CORRECT THE JUDGE
corr <- c(overall = rogan_gladen(mean(judged$judge_label), SENS, SPEC))
for (t in TYPES) corr[t] <- rogan_gladen(mean(judged$judge_label[judged$question_type == t]), SENS, SPEC)
for (t in c(TYPES, "overall")) put(paste0("corrected_accuracy_", t), corr[[t]])

# Cluster bootstrap: per-cluster counts, then weight clusters by how often each is drawn.
cl <- factor(df$cluster_id, levels = unique(df$cluster_id))
j <- !is.na(df$judge_label); j1 <- j & df$judge_label %in% 1
hj <- j & df$human_graded == 1
ind <- data.frame(J_n = j, J_1 = j1)
for (t in TYPES) {
  ind[[paste0("J_n_", t)]] <- j & df$question_type == t
  ind[[paste0("J_1_", t)]] <- j1 & df$question_type == t
}
ind$H_pos <- hj & df$human_label %in% 1
ind$H_pos_j1 <- ind$H_pos & j1
ind$H_neg <- hj & df$human_label %in% 0
ind$H_neg_j0 <- ind$H_neg & df$judge_label %in% 0
# all human-graded questions (with or without a judge verdict), for a human-only interval
ind$Hall_n <- df$human_graded == 1
ind$Hall_1 <- df$human_graded == 1 & df$human_label %in% 1
C <- rowsum(as.matrix(ind) * 1, cl)          # clusters x counts
nc <- nrow(C)
set.seed(BOOT_SEED)
boot <- matrix(NA_real_, B, 4, dimnames = list(NULL, c(TYPES, "overall")))
boot_h <- numeric(B)   # human-only accuracy, same resamples
for (b in 1:B) {
  w <- tabulate(sample.int(nc, nc, replace = TRUE), nbins = nc)
  S <- colSums(C * w)
  se_b <- S[["H_pos_j1"]] / S[["H_pos"]]; sp_b <- S[["H_neg_j0"]] / S[["H_neg"]]
  boot[b, "overall"] <- rogan_gladen(S[["J_1"]] / S[["J_n"]], se_b, sp_b)
  boot_h[b] <- S[["Hall_1"]] / S[["Hall_n"]]
  for (t in TYPES) boot[b, t] <- rogan_gladen(S[[paste0("J_1_", t)]] / S[[paste0("J_n_", t)]], se_b, sp_b)
}
for (t in c(TYPES, "overall")) {
  q <- quantile(boot[, t], c(0.025, 0.975), na.rm = TRUE, type = 7)
  put(paste0("corrected_ci_low_", t), q[[1]], 2); put(paste0("corrected_ci_high_", t), q[[2]], 2)
}
# Like-for-like comparison: human-only accuracy with the SAME clustered bootstrap
qh <- quantile(boot_h, c(0.025, 0.975), type = 7)
put("human_only_boot_ci_low", qh[[1]], 2); put("human_only_boot_ci_high", qh[[2]], 2)
cat(sprintf("STEP 2d: corrected accuracy %.3f [%.2f, %.2f]\n", corr[["overall"]],
            res$corrected_ci_low_overall, res$corrected_ci_high_overall))

# ======================================================= STEP 2e: FAILURE DECOMPOSITION
CAUSES <- c("retrieval_miss", "ungrounded_gen", "grounded_but_wrong", "no_source")
df$cause <- ifelse(df$answerable == 0, "no_source",
            ifelse(df$recall5 < 1, "retrieval_miss",
            ifelse(df$abstained == 1 | df$groundedness < GROUNDED_THRESHOLD, "ungrounded_gen",
                   "grounded_but_wrong")))
wrong_h <- df[df$human_graded == 1 & df$human_label %in% 0, ]
put("n_wrong_human", nrow(wrong_h), 0)
for (t in c(TYPES, "overall")) {
  s <- if (t == "overall") wrong_h else wrong_h[wrong_h$question_type == t, ]
  put(paste0("n_wrong_human_", t), nrow(s), 0)
  for (c_ in CAUSES) put(paste0("fail_share_", c_, "_", t), mean(s$cause == c_))
}
for (t in TYPES) put(paste0("retrieval_miss_rate_", t), 1 - mean(ans$full_hit[ans$question_type == t]))

# ======================================================= STEP 3: GROUND TRUTH
put("true_accuracy_sample_overall", mean(df$true_correct))
for (t in TYPES) put(paste0("true_accuracy_sample_", t), mean(df$true_correct[df$question_type == t]))
for (k in c("true_accuracy_overall", "true_accuracy_policy_lookup", "true_accuracy_computed_metric",
            "true_accuracy_multi_hop", "true_recall_at5_overall", "true_groundedness_overall",
            "true_recall_at5_multi_hop", "true_groundedness_multi_hop",
            "judge_sensitivity", "judge_specificity"))
  put(paste0("population_", k), tv[[k]])
wrong_all <- df[df$true_correct == 0, ]
for (c_ in CAUSES) {
  put(paste0("true_fail_share_", c_, "_overall"), mean(wrong_all$cause == c_))
  put(paste0("true_fail_share_", c_, "_multi_hop"), mean(wrong_all$cause[wrong_all$question_type == "multi_hop"] == c_))
}
put("naive_judge_error_pp", 100 * (mean(judged$judge_label) - mean(df$true_correct)), 1)
put("corrected_error_pp", 100 * (corr[["overall"]] - mean(df$true_correct)), 1)
for (t in TYPES)
  put(paste0("corrected_error_pp_", t), 100 * (corr[[t]] - mean(df$true_correct[df$question_type == t])), 1)
cat(sprintf("\nSTEP 3: true accuracy %.3f; raw judge off by %+.1f pp; corrected off by %+.1f pp\n",
            mean(df$true_correct), res$naive_judge_error_pp, res$corrected_error_pp))

# ======================================================= STEP 4: DECISION
cat("\nSTEP 4: decisions by question type\n")
for (t in TYPES) {
  a <- corr[[t]]
  d <- if (a >= DEPLOY_AT) "deploy with citations shown" else if (a >= ROUTE_AT)
    "route through governed metric definitions / analytics team" else "hold until retrieval improves"
  cat(sprintf("   %-16s corrected %.3f -> %s\n", t, a, d))
}

out <- data.frame(name = names(res), value = unlist(res, use.names = FALSE))
write.csv(out, file.path(OUT, "r_results.csv"), row.names = FALSE)
cat(sprintf("\nWrote %d results to outputs/r_results.csv\n", nrow(out)))
