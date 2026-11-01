# analysis.R -- A01 "Adoption Is Not Impact: How to Prove a GenAI Rollout Actually Worked"
#
# Run from the code folder root:
#     Rscript r/analysis.R          # writes outputs/r_headline.csv
#     python3 python/analysis.py    # then merges these numbers into outputs/results.json
#
# Base R only (no packages). Team-clustered standard errors and the Callaway & Sant'Anna
# style group-time ATT are written out by hand so you can read every step.
#
# Steps
#   Step 0  Clean the extract (duplicates, blank hours, messy region labels).
#   Step 1  The naive before-after for wave 1 (and why it misleads).
#   Step 2  Classic 2x2 difference-in-differences: wave 1 vs. not-yet-treated teams.
#   Step 3  Checks: parallel pre-trends; staggered adoption with naive TWFE vs. a
#           group-time (Callaway & Sant'Anna style) estimator; comparison with the truth.
#   Step 4  Decision implication: what the effect is worth, and whether quality held.

d <- read.csv("data/support_team_weeks.csv", stringsAsFactors = FALSE)

# =============================================================================
# Step 0. Clean the extract
# =============================================================================
n_rows_raw <- nrow(d)
d$region <- toupper(trimws(d$region))                 # " emea-north " -> "EMEA-NORTH"
dup <- duplicated(d[, c("team_id", "week")])          # double-loaded team-weeks
n_duplicates <- sum(dup)
d <- d[!dup, ]
blank <- is.na(d$agent_hours) | d$agent_hours <= 0   # failed time-tracking export
n_blank_hours <- sum(blank)
d <- d[!blank, ]
d <- d[order(d$team_id, d$week), ]
n_rows_clean <- nrow(d)
n_teams <- length(unique(d$team_id))
n_missing_teamweeks <- n_teams * 24 - n_rows_clean

# Outcomes, recomputed from counts so every language gets identical numbers.
d$y <- d$tickets_resolved / d$agent_hours           # tickets resolved per agent-hour
d$reopen <- d$reopened_tickets / d$tickets_resolved  # reopen rate (quality)
d$usage <- d$ai_active_agents / d$n_agents           # weekly active share (usage)
d$D <- as.integer(d$week >= d$launch_week)           # 1 once the team has the assistant

# -----------------------------------------------------------------------------
# Helper: OLS with team-clustered standard errors (CR1, the Stata / statsmodels default)
#   V = c * (X'X)^-1 [ sum_g (X_g' u_g)(X_g' u_g)' ] (X'X)^-1,
#   c = G/(G-1) * (N-1)/(N-K).
# Clustering by team lets each team's weekly errors be correlated with each other.
# -----------------------------------------------------------------------------
cluster_ols <- function(y, X, cl) {
  XtX_inv <- solve(crossprod(X))
  b <- drop(XtX_inv %*% crossprod(X, y))
  u <- drop(y - X %*% b)
  S <- rowsum(X * u, cl)                       # one row per cluster: X_g' u_g
  N <- nrow(X); K <- ncol(X); G <- nrow(S)
  V <- (G / (G - 1)) * ((N - 1) / (N - K)) * XtX_inv %*% crossprod(S) %*% XtX_inv
  list(b = b, se = sqrt(diag(V)), V = V, G = G, N = N)
}
z <- qnorm(0.975)   # normal critical value for 95% CIs (same as statsmodels' default)

# =============================================================================
# Step 1. Naive before-after for wave 1, weeks 1-12
#   mean(weeks 7-12) - mean(weeks 1-6) among wave-1 team-weeks.
#   It attributes ALL change to the assistant, including the common trend.
# =============================================================================
s12 <- d[d$week <= 12, ]
s12$treat <- as.integer(s12$wave == 1)
s12$post <- as.integer(s12$week >= 7)
m <- function(x) mean(x)
w1_pre <- m(s12$y[s12$treat == 1 & s12$post == 0]); w1_post <- m(s12$y[s12$treat == 1 & s12$post == 1])
c_pre <- m(s12$y[s12$treat == 0 & s12$post == 0]);  c_post <- m(s12$y[s12$treat == 0 & s12$post == 1])
naive <- w1_post - w1_pre
comparison_change <- c_post - c_pre
w1 <- s12[s12$treat == 1, ]
fit_naive <- cluster_ols(w1$y, cbind(1, w1$post), w1$team_id)
cat(sprintf("Step 1  naive before-after, wave 1: %+.3f (SE %.3f)\n", naive, fit_naive$se[2]))
cat(sprintf("        comparison teams (waves 2-3) changed by %+.3f without the assistant\n", comparison_change))

# =============================================================================
# Step 2. Classic 2x2 DiD: y = a + b*treat + c*post + DiD*(treat x post)
#   Saturated in the four cells, so DiD = (w1_post - w1_pre) - (c_post - c_pre).
# =============================================================================
X2 <- cbind(1, s12$treat, s12$post, s12$treat * s12$post)
fit_did <- cluster_ols(s12$y, X2, s12$team_id)
did <- fit_did$b[4]; did_se <- fit_did$se[4]
cat(sprintf("Step 2  2x2 DiD: %+.3f (team-clustered SE %.3f, 95%% CI %.3f to %.3f)\n",
            did, did_se, did - z * did_se, did + z * did_se))

# Same design for quality (reopen rate) and for the effect in % of the wave-1 baseline.
fit_reopen <- cluster_ols(s12$reopen, X2, s12$team_id)
did_reopen <- fit_reopen$b[4]; did_reopen_se <- fit_reopen$se[4]
cat(sprintf("        reopen-rate DiD: %+.4f (SE %.4f)\n", did_reopen, did_reopen_se))

# =============================================================================
# Step 3a. Parallel pre-trends (weeks 1-6): do wave-1 and comparison teams
#   have the same weekly slope before launch?  y = a + b*treat + c*week + d*treat*week
# =============================================================================
pre <- s12[s12$week <= 6, ]
fit_pt <- cluster_ols(pre$y, cbind(1, pre$treat, pre$week, pre$treat * pre$week), pre$team_id)
pretrend_diff <- fit_pt$b[4]; pretrend_se <- fit_pt$se[4]
cat(sprintf("Step 3a pre-trend slope difference (wave 1 - comparison): %+.4f per week (SE %.4f)\n",
            pretrend_diff, pretrend_se))

# =============================================================================
# Step 3b. Staggered adoption, all 24 weeks
#   (a) Naive two-way fixed effects: y = team FE + week FE + beta * D
# =============================================================================
Xtw <- model.matrix(~ D + factor(team_id) + factor(week), data = d)
fit_tw <- cluster_ols(d$y, Xtw, d$team_id)
twfe <- fit_tw$b[["D"]]; twfe_se <- fit_tw$se[["D"]]
cat(sprintf("Step 3b naive TWFE: %+.3f (SE %.3f)\n", twfe, twfe_se))

# -----------------------------------------------------------------------------
#   (b) Group-time ATT (Callaway & Sant'Anna style), not-yet-treated controls.
#   For launch group g and week t, each team's base is its AVERAGE over the pre-launch
#   weeks 1..g-1 (B_i). Using the whole pre-period rather than the single week g-1
#   (the Callaway & Sant'Anna default) keeps the same identifying assumption (parallel
#   trends) but is much less noisy with weekly data:
#     ATT(g,t) = mean over group g of [Y_t - B]  -  mean over controls of [Y_t - B]
#   Controls = teams not yet treated at week t (launch week > t), so they are untreated
#   in week t and in every base week. For pre-launch weeks (placebo leads, t < g) the
#   controls are the teams launching after g. Only teams observed in week t enter.
#   Wave 3 (g = 19) has no not-yet-treated controls after week 18, so its effects are
#   not identified: it serves only as a control. Cells therefore run to week 18.
# -----------------------------------------------------------------------------
Y <- tapply(d$y, list(d$team_id, d$week), identity)            # team x week matrix (NA = missing)
G_team <- tapply(d$launch_week, d$team_id, function(x) x[1])
teams <- rownames(Y)
n_g <- table(G_team)                                           # group sizes (20 each)

# True effects for the same cells, from the generator's ground truth.
tp <- read.csv("data/true_parameters.csv", stringsAsFactors = FALSE)
gp <- function(p) as.numeric(tp$value[tp$level == "global" & tp$parameter == p])
ramp <- c(gp("ramp_share_e0"), gp("ramp_share_e1"), gp("ramp_share_e2"))
tau_team <- setNames(as.numeric(tp$value[tp$level == "team"]), tp$id[tp$level == "team"])
true_effect <- function(team, week) {
  e <- week - G_team[team]
  ifelse(e < 0, 0, tau_team[team] * ifelse(e < 3, ramp[pmin(e, 2) + 1], 1))
}

cells <- list(); Psi <- NULL
for (g in c(7, 13)) {
  B <- rowMeans(Y[, 1:(g - 1), drop = FALSE], na.rm = TRUE)   # each team's pre-launch average
  for (t in 1:18) {
    ctrl_rule <- if (t >= g) G_team > t else G_team > g
    tr <- teams[G_team == g & !is.na(Y[, t])]
    co <- teams[ctrl_rule & !is.na(Y[, t])]
    dT <- Y[tr, t] - B[tr]; dC <- Y[co, t] - B[co]
    att <- mean(dT) - mean(dC)
    # Influence function of ATT(g,t) for each team (used for team-clustered SEs):
    #   treated team i: (dY_i - mean_T)/n_T ; control team j: -(dY_j - mean_C)/n_C
    psi <- setNames(numeric(length(teams)), teams)
    psi[tr] <- (dT - mean(dT)) / length(tr)
    psi[co] <- psi[co] - (dC - mean(dC)) / length(co)
    Psi <- cbind(Psi, psi)
    cells[[length(cells) + 1]] <- data.frame(g = g, t = t, e = t - g, att = att,
      n_treated = length(tr), n_control = length(co),
      true_att = mean(true_effect(tr, t)))
  }
}
cells <- do.call(rbind, cells)
cells$w_g <- as.numeric(n_g[as.character(cells$g)])

# Aggregate cells with weights proportional to group size; SE = sqrt(sum_i psi_i^2).
aggregate_cells <- function(sel) {
  w <- ifelse(sel, cells$w_g, 0); w <- w / sum(w)
  psi <- drop(Psi %*% w)
  c(est = sum(w * cells$att), se = sqrt(sum(psi^2)), truth = sum(w * cells$true_att))
}
cs_all <- aggregate_cells(cells$e >= 0)                  # overall ATT (all identified post cells)
cs_e3 <- aggregate_cells(cells$e >= 3)                   # established effect (3+ weeks after launch)
cs_w1_6 <- aggregate_cells(cells$g == 7 & cells$e %in% 0:5)  # wave 1, weeks 7-12
es_e <- -6:11                                     # event times shown (both waves have -6..5)
es <- t(sapply(es_e, function(e) aggregate_cells(cells$e == e)))
es <- data.frame(e = es_e, es)
# Joint test of the placebo leads e = -6..-2 (are pre-launch differences all zero?).
# e = -1 is left out: with an average base, a group's leads sum to about zero by construction.
W <- sapply(-6:-2, function(e) { w <- ifelse(cells$e == e, cells$w_g, 0); w / sum(w) })
lead_est <- drop(t(W) %*% cells$att)
V_leads <- crossprod(Psi %*% W)
lead_chi2 <- drop(t(lead_est) %*% solve(V_leads) %*% lead_est)
lead_p <- pchisq(lead_chi2, df = length(lead_est), lower.tail = FALSE)
cat(sprintf("        group-time ATT, overall: %+.3f (SE %.3f); established (3+ wks): %+.3f\n",
            cs_all["est"], cs_all["se"], cs_e3["est"]))
cat(sprintf("        placebo leads e=-6..-2: joint chi2 = %.2f, p = %.3f\n", lead_chi2, lead_p))

# =============================================================================
# Step 3c. Check against the truth
# =============================================================================
w1_rows <- d[d$wave == 1 & d$week %in% 7:12, ]
true_w1_6 <- mean(true_effect(w1_rows$team_id, w1_rows$week))     # what naive and 2x2 target
trt_rows <- d[d$D == 1, ]
true_all_treated <- mean(true_effect(trt_rows$team_id, trt_rows$week))
true_cs <- cs_all[["truth"]]
cat(sprintf("Step 3c truth: wave 1 weeks 7-12 %.3f | identified cells %.3f | all treated team-weeks %.3f\n",
            true_w1_6, true_cs, true_all_treated))

# =============================================================================
# Step 4. Decision implication
# =============================================================================
# Org-wide agent-hours per week = average team-week hours x number of teams.
hours_per_week <- mean(d$agent_hours) * n_teams
usage_w1 <- mean(d$usage[d$wave == 1 & d$week %in% 7:12])
usage_est <- mean(d$usage[d$D == 1 & (d$week - d$launch_week) >= 3])

h <- c(
  n_rows_raw = n_rows_raw, n_duplicates = n_duplicates, n_blank_hours = n_blank_hours,
  n_rows_clean = n_rows_clean, n_missing_teamweeks = n_missing_teamweeks,
  wave1_pre_mean = w1_pre, wave1_post_mean = w1_post,
  comparison_pre_mean = c_pre, comparison_post_mean = c_post,
  naive_before_after_wave1 = naive, naive_before_after_se = fit_naive$se[[2]],
  comparison_change = comparison_change,
  did_2x2 = did, did_2x2_se = did_se, did_2x2_ci_low = did - z * did_se, did_2x2_ci_high = did + z * did_se,
  naive_to_did_ratio = naive / did, did_share_of_naive = did / naive,
  did_pct_of_baseline = did / w1_pre,
  pretrend_slope_diff = pretrend_diff, pretrend_slope_diff_se = pretrend_se,
  twfe_att = twfe, twfe_se = twfe_se,
  twfe_ci_low = twfe - z * twfe_se, twfe_ci_high = twfe + z * twfe_se,
  cs_att_overall = cs_all[["est"]], cs_att_overall_se = cs_all[["se"]],
  cs_att_overall_ci_low = cs_all[["est"]] - z * cs_all[["se"]],
  cs_att_overall_ci_high = cs_all[["est"]] + z * cs_all[["se"]],
  cs_att_established = cs_e3[["est"]], cs_att_established_se = cs_e3[["se"]],
  cs_att_wave1_weeks7to12 = cs_w1_6[["est"]],
  cs_leads_chi2 = lead_chi2, cs_leads_p = lead_p,
  true_att_wave1_weeks7to12 = true_w1_6, true_att_cs_cells = true_cs,
  true_att_established = cs_e3[["truth"]], true_att_all_treated = true_all_treated,
  bias_naive = naive - true_w1_6, bias_did_2x2 = did - true_w1_6,
  bias_twfe_vs_cs_cells = twfe - true_cs, bias_twfe_vs_all_treated = twfe - true_all_treated,
  bias_cs = cs_all[["est"]] - true_cs,
  did_reopen_rate = did_reopen, did_reopen_rate_se = did_reopen_se,
  did_reopen_rate_ci_low = did_reopen - z * did_reopen_se, did_reopen_rate_ci_high = did_reopen + z * did_reopen_se,
  reopen_rate_baseline_wave1 = mean(s12$reopen[s12$treat == 1 & s12$post == 0]),
  usage_share_wave1_weeks7to12 = usage_w1, usage_share_established = usage_est,
  agent_hours_per_week_org = hours_per_week,
  extra_tickets_per_1000_hours_naive = 1000 * naive,
  extra_tickets_per_1000_hours_did = 1000 * did,
  extra_tickets_per_week_org_established = cs_e3[["est"]] * hours_per_week,
  setNames(es$est, paste0("es_e", ifelse(es$e < 0, paste0("m", -es$e), es$e))),
  setNames(es$se, paste0("es_se_e", ifelse(es$e < 0, paste0("m", -es$e), es$e))),
  setNames(es$truth, paste0("es_true_e", ifelse(es$e < 0, paste0("m", -es$e), es$e)))
)
cat(sprintf("Step 4  per 1,000 agent-hours: naive claims %+.0f tickets, DiD supports %+.0f; reopen-rate DiD %+.4f\n",
            1000 * naive, 1000 * did, did_reopen))
dir.create("outputs", showWarnings = FALSE)
write.csv(data.frame(name = names(h), value = signif(unname(h), 10)), "outputs/r_headline.csv", row.names = FALSE)
cat("\nWrote outputs/r_headline.csv (run python3 python/analysis.py to merge into results.json).\n")
