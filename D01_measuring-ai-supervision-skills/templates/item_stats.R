# Reusable item statistics for any item-response csv (one row per person, one column per item). Base R.
#
# Usage:
#   Rscript templates/item_stats.R responses.csv                        # all numeric columns are items
#   Rscript templates/item_stats.R responses.csv S01,S02,S03            # the items of ONE scale
#   Rscript templates/item_stats.R responses.csv S01,S02,S03 out.csv
# Or: source("templates/item_stats.R"); res <- item_statistics(df, items)
#
# Output columns match templates/item_statistics_template.csv:
#   item, n, mean, sd, item_rest_r, alpha_if_deleted, decision
# Rules: item_rest_r < 0.20 -> drop ("drop (check key)" if negative); 0.20-0.30, or alpha rises
# when the item is deleted -> revise; otherwise keep. Missing answers: pairwise-complete covariances.

item_statistics <- function(df, items = NULL, drop_below = 0.20, revise_below = 0.30) {
  if (is.null(items)) items <- names(df)[sapply(df, is.numeric)]
  C <- cov(df[, items], use = "pairwise.complete.obs")
  alpha <- function(C) { k <- ncol(C); k / (k - 1) * (1 - sum(diag(C)) / sum(C)) }
  a <- alpha(C)
  out <- data.frame(item = items, n = colSums(!is.na(df[, items])),
                    mean = colMeans(df[, items], na.rm = TRUE),
                    sd = apply(df[, items], 2, sd, na.rm = TRUE),
                    item_rest_r = NA_real_, alpha_if_deleted = NA_real_, decision = NA_character_,
                    row.names = NULL)
  for (i in seq_along(items)) {
    rest <- setdiff(seq_along(items), i)
    r_ir <- sum(C[i, rest]) / sqrt(C[i, i] * sum(C[rest, rest]))
    a_del <- alpha(C[rest, rest, drop = FALSE])
    out$item_rest_r[i] <- r_ir
    out$alpha_if_deleted[i] <- a_del
    out$decision[i] <- if (r_ir < drop_below) { if (r_ir >= 0) "drop" else "drop (check key)" }
                       else if (r_ir < revise_below || a_del > a) "revise" else "keep"
  }
  list(table = out, alpha = a, k_items = length(items))
}

if (sys.nframe() == 0L) {   # run as a script
  args <- commandArgs(trailingOnly = TRUE)
  if (length(args) < 1) stop("usage: Rscript item_stats.R responses.csv [S01,S02,...] [out.csv]")
  df <- read.csv(args[1], check.names = FALSE)
  items <- if (length(args) >= 2 && nzchar(args[2])) strsplit(args[2], ",")[[1]] else NULL
  out_file <- if (length(args) >= 3) args[3] else "item_statistics.csv"
  res <- item_statistics(df, items)
  tab <- res$table; num <- sapply(tab, is.numeric); tab[num] <- lapply(tab[num], round, 3)
  write.csv(tab, out_file, row.names = FALSE)
  print(tab, row.names = FALSE)
  cat(sprintf("\n%d items, Cronbach's alpha = %.2f\nWrote %s\n", res$k_items, res$alpha, out_file))
}
