# Regenerate validation/reference/credibility_hierarchical_actuar.csv from
# the R package actuar.
#
#   install.packages("actuar")
#   Rscript validation/scripts/r_credibility_hierarchical.R
#
# actuar (GPL) is used only to produce reference values. The reference for
# prospicio_bayes::credibility::credibility_premiums: given the structure
# parameters, the credibility premiums of the normal model are those of
# Buhlmann-Straub (one level) and of Buhlmann-Gisler's hierarchical model
# (more levels). Rows, on Hachemeister's data (validation/reference/
# hachemeister.csv):
#
# - Buhlmann-Straub, actuar::cm(~state) with the Buhlmann-Gisler estimator;
# - the hierarchy of actuar's own example, states 1 and 3 in cohort 1 and
#   2, 4 and 5 in cohort 2, actuar::cm(~cohort + cohort:state), with the
#   Buhlmann-Gisler, Ohlsson and iterative estimators.
#
# Each case records the structure parameters (within_variance s^2 and
# between_variance by level, 0 the outermost) that the Rust test plugs in,
# and the collective mean, each cohort's and state's credibility factor and
# premium, which it checks.

suppressMessages(library(actuar))
data(hachemeister)

out <- "validation/reference/credibility_hierarchical_actuar.csv"
source <- paste0("R actuar ", as.character(packageVersion("actuar")))
cohorts <- c(1, 2, 1, 2, 2)
X <- cbind(cohort = cohorts, hachemeister)

rows <- list()
add <- function(case, quantity, arg, value, rel = 1e-10) {
  rows[[length(rows) + 1]] <<- data.frame(
    distribution = "hierarchical_credibility", params = case, quantity = quantity,
    arg = arg, expected = sprintf("%.17g", value), abs_tol = 0, rel_tol = rel,
    source = source)
}

# One level.
fit <- cm(~state, data = hachemeister, ratios = ratio.1:ratio.12,
          weights = weight.1:weight.12, method = "Buhlmann-Gisler")
case <- "levels=none;estimator=Buhlmann-Gisler"
add(case, "within_variance", "", fit$unbiased[["state"]], 0)
add(case, "between_variance", 0, fit$unbiased[["portfolio"]], 0)
add(case, "collective_mean", "", fit$means$portfolio)
for (i in 1:5) {
  add(case, "credibility", i, fit$cred[i])
  add(case, "premium", i, predict(fit)[i])
}

# Two levels.
for (method in c("Buhlmann-Gisler", "Ohlsson", "iterative")) {
  fit <- cm(~cohort + cohort:state, data = X, ratios = ratio.1:ratio.12,
            weights = weight.1:weight.12, method = method)
  v <- if (method == "iterative") fit$iterative else fit$unbiased
  case <- paste0("levels=cohort;estimator=", method)
  add(case, "within_variance", "", v[["state"]], 0)
  add(case, "between_variance", 0, v[["portfolio"]], 0)
  add(case, "between_variance", 1, v[["cohort"]], 0)
  add(case, "collective_mean", "", fit$means$portfolio)
  p <- predict(fit)
  for (k in 1:2) {
    add(case, "cohort_credibility", k, fit$cred$cohort[k])
    add(case, "cohort_premium", k, p$cohort[k])
  }
  for (i in 1:5) {
    add(case, "credibility", i, fit$cred$state[i])
    add(case, "premium", i, p$state[i])
  }
}

write.csv(do.call(rbind, rows), out, row.names = FALSE, quote = FALSE)
cat("wrote", out, "\n")
