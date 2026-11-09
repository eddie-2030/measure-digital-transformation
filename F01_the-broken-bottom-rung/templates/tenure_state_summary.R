# tenure_state_summary.R -- tenure-aligned state distribution and time-in-state
# summary for ANY csv of monthly person-state records (base R only).
#
# Required columns: person_id, cohort, job_family, tenure_month, state
#   R = routine, S = supervised complex (stretch), I = independent complex, X = left.
#   Blank = missing month (filled with the previous state). Keep rows for months
#   after leaving (X) up to your data cut -- that handles right truncation.
#
# Usage:  Rscript templates/tenure_state_summary.R my_records.csv [first_n_months]
#         (no arguments -> runs on person_state_records_template.csv)
# Or:     source("templates/tenure_state_summary.R"); state_distribution(df); time_in_state(df, 12)

STATES <- c("R", "S", "I", "X")

clean_states <- function(df) {
  df <- df[order(df$person_id, df$tenure_month), ]
  s <- toupper(trimws(ifelse(is.na(df$state), "", df$state)))
  s[s == "LEFT"] <- "X"
  s <- substr(s, 1, 1)                            # 'SUP' -> 'S', 'Indep' -> 'I'
  s[!s %in% STATES] <- NA
  for (k in seq_along(s)) {                       # carry forward; first month -> R; X absorbing
    same <- k > 1 && df$person_id[k] == df$person_id[k - 1]
    if (is.na(s[k])) s[k] <- if (same) s[k - 1] else "R"
    if (same && s[k - 1] == "X") s[k] <- "X"
  }
  df$state <- s
  df
}

state_distribution <- function(df, by = c("cohort", "job_family")) {
  df <- clean_states(df)
  keys <- df[, c(by, "tenure_month")]
  ind <- as.data.frame(sapply(STATES, function(st) as.numeric(df$state == st)))
  out <- aggregate(ind, keys, function(x) round(mean(x) * 100, 1))
  out$n_observed <- aggregate(df$person_id, keys, function(x) length(unique(x)))$x
  out <- out[do.call(order, out[, c(by, "tenure_month")]), ]
  rownames(out) <- NULL
  out
}

time_in_state <- function(df, first_n_months = 12, by = c("cohort", "job_family")) {
  df <- clean_states(df)
  n_obs <- tapply(df$tenure_month, df$person_id, max)
  keep <- names(n_obs)[n_obs >= first_n_months]
  sub <- df[df$person_id %in% keep & df$tenure_month <= first_n_months, ]
  if (nrow(sub) == 0) {                           # guard: nobody observed long enough
    message(sprintf("No one is observed for %d months yet; lower first_n_months.", first_n_months))
    return(data.frame())
  }
  per <- as.data.frame.matrix(table(sub$person_id, factor(sub$state, levels = STATES)))
  first <- sub[!duplicated(sub$person_id), c("person_id", by)]
  per <- cbind(first[match(rownames(per), first$person_id), by, drop = FALSE], per)
  out <- aggregate(per[, STATES], per[, by, drop = FALSE], function(x) round(mean(x), 2))
  names(out)[names(out) %in% STATES] <- paste0("months_in_", STATES)
  out$n_people <- aggregate(per$R, per[, by, drop = FALSE], length)$x
  last <- sub[sub$tenure_month == first_n_months, ]
  pi <- aggregate(last$state == "I", last[, by, drop = FALSE], function(x) round(mean(x) * 100))
  out[[paste0("pct_in_I_at_month_", first_n_months)]] <- pi$x[match(do.call(paste, out[, by]), do.call(paste, pi[, by]))]
  out
}

if (!interactive() && sys.nframe() == 0) {
  a <- commandArgs(trailingOnly = FALSE)
  here <- dirname(normalizePath(sub("^--file=", "", a[grep("^--file=", a)])))
  args <- commandArgs(trailingOnly = TRUE)
  path <- if (length(args) >= 1) args[1] else file.path(here, "person_state_records_template.csv")
  months <- if (length(args) >= 2) as.integer(args[2]) else if (length(args) >= 1) 12 else 6
  d <- read.csv(path, stringsAsFactors = FALSE, colClasses = c(state = "character"))
  cat("State distribution by tenure month (%):\n"); print(state_distribution(d), row.names = FALSE)
  cat(sprintf("\nTime in state, months 1-%d:\n", months)); print(time_in_state(d, months), row.names = FALSE)
}
