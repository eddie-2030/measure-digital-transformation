# tutor_sequence_tools.R -- reusable base-R functions for AI-tutor event logs (E03 companion template).
#
# Input: an event log with columns learner_id, session_id, step, action, timestamp
#        (see tutor_event_log_template.csv). Actions: T attempt, H hint, A full answer,
#        C copy/paste answer, V verify, Q quit (end of session). A session may end without Q.
# Optional: a labels file with columns learner_id, label (1 = good outcome, 0 = not).
#
# Command line:
#   Rscript tutor_sequence_tools.R events.csv              # transition matrix + scorecard metrics
#   Rscript tutor_sequence_tools.R events.csv labels.csv   # + two chains and a log-score per learner
# Or in R: source("tutor_sequence_tools.R") and call the functions.
# For an honest test, fit the chains on some learners and score OTHER learners.

ROWS <- c("S", "T", "H", "A", "C", "V")   # S = start of session
COLS <- c("T", "H", "A", "C", "V", "Q")

load_events <- function(path) {
  ev <- read.csv(path, colClasses = "character", na.strings = character(0))
  ev$tok <- toupper(trimws(ev$action))
  dropped <- sum(!(ev$tok %in% COLS))
  ev <- ev[ev$tok %in% COLS, ]
  ev$step <- as.integer(ev$step)
  ev <- ev[order(ev$learner_id, ev$session_id, ev$step, method = "radix"), ]
  q_before <- ave(as.integer(ev$tok == "Q"), ev$session_id, FUN = function(z) c(0, head(cumsum(z), -1)))
  ev <- ev[q_before == 0, ]
  first <- c(TRUE, ev$session_id[-1] != ev$session_id[-nrow(ev)])
  ev$prev <- c("S", ev$tok[-nrow(ev)]); ev$prev[first] <- "S"
  if (dropped > 0) message("note: dropped ", dropped, " rows with unknown action codes")
  ev
}

transition_counts <- function(ev) {
  # learner x 36 matrix of first-order counts (row-major: from-state x next-token)
  ids <- sort(unique(ev$learner_id))
  li <- match(ev$learner_id, ids)
  cell <- (match(ev$prev, ROWS) - 1) * 6 + match(ev$tok, COLS)
  N <- matrix(tabulate((li - 1) * 36 + cell, length(ids) * 36), nrow = length(ids), byrow = TRUE)
  list(ids = ids, N = N)
}

transition_matrix <- function(counts36, smooth = 0.5) {
  c <- matrix(counts36, 6, 6, byrow = TRUE) + smooth
  P <- c / rowSums(c)
  dimnames(P) <- list(ROWS, COLS)
  P
}

fit_chains <- function(ids, N, labels) {
  # labels: named numeric vector, names = learner_id, values 0/1
  y <- labels[ids]
  list(good = transition_matrix(colSums(N[which(y == 1), , drop = FALSE])),
       bad = transition_matrix(colSums(N[which(y == 0), , drop = FALSE])))
}

log_scores <- function(N, P_good, P_bad) {
  # log-score = sum over steps of log[ P_good(next | current) / P_bad(next | current) ]
  as.vector(N %*% as.vector(t(log(P_good) - log(P_bad))))
}

scorecard_metrics <- function(ev) {
  P <- transition_matrix(colSums(transition_counts(ev)$N))
  acts <- ev[ev$tok != "Q", ]
  ends_q <- tapply(ev$tok == "Q", ev$session_id, any)
  list(learners = length(unique(ev$learner_id)),
       sessions = length(unique(ev$session_id)),
       messages_per_learner = round(nrow(acts) / length(unique(ev$learner_id)), 1),
       `P(attempt | hint)` = round(P["H", "T"], 2),
       `P(re-attempt | answer)` = round(P["A", "T"], 2),
       `P(answer | hint)` = round(P["H", "A"], 2),
       `P(copy | answer)` = round(P["A", "C"], 2),
       `share of actions that are full-answer requests (%)` = round(100 * mean(acts$tok == "A")),
       `sessions ending without Q (%)` = round(100 * (1 - mean(ends_q))))
}

if (sys.nframe() == 0) {
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 1) stop("usage: Rscript tutor_sequence_tools.R events.csv [labels.csv]")
  ev <- load_events(args[1])
  tc <- transition_counts(ev)
  cat("Pooled transition matrix P(next | current):\n")
  print(round(transition_matrix(colSums(tc$N)), 2))
  cat("\nScorecard metrics:\n")
  m <- scorecard_metrics(ev)
  for (k in names(m)) cat("  ", k, ": ", m[[k]], "\n", sep = "")
  if (length(args) > 1) {
    lab <- read.csv(args[2], stringsAsFactors = FALSE)
    ch <- fit_chains(tc$ids, tc$N, setNames(lab$label, lab$learner_id))
    out <- data.frame(learner_id = tc$ids, log_score = round(log_scores(tc$N, ch$good, ch$bad), 3))
    path <- paste0(sub("\\.[^.]*$", "", args[1]), "_log_scores_R.csv")
    write.csv(out, path, row.names = FALSE)
    cat("\nWrote ", path, " (higher = sequence looks more like the label-1 group).\n", sep = "")
  }
}
