# Regenerate validation/data/loo_loglik.csv and
# validation/reference/elpd_loo.csv.
#
#   Rscript validation/scripts/r_loo.R
#
# A pointwise log-likelihood matrix (500 posterior draws x 30 observations)
# from a normal model with unknown mean and scale, fitted by exact
# conjugate draws, with two outliers so some observations have heavy
# importance-ratio tails (large Pareto k). The R package loo computes
# PSIS-LOO and WAIC from it with r_eff = 1 (independent draws). Records
# elpd_loo, p_loo, looic and their standard errors, the pointwise
# elpd_loo and Pareto k, and the same for WAIC.
#
# loo is GPL-3: this script only produces reference values.

library(loo)
set.seed(20261004)
n <- 30
y <- c(rnorm(n - 2, 1, 1), 6, -5)
S <- 500
# Normal-inverse-gamma posterior under a flat prior.
sigma2 <- (n - 1) * var(y) / rchisq(S, n - 1)
mu <- rnorm(S, mean(y), sqrt(sigma2 / n))
ll <- sapply(y, function(yi) dnorm(yi, mu, sqrt(sigma2), log = TRUE))
write.csv(ll, "validation/data/loo_loglik.csv", row.names = FALSE)

source <- paste("loo", packageVersion("loo"))
rows <- list()
add <- function(quantity, index, value, rel, abs = 1e-10) {
  rows[[length(rows) + 1]] <<- data.frame(
    quantity = quantity, index = index, expected = sprintf("%.17g", value),
    abs_tol = abs, rel_tol = rel, source = source
  )
}
l <- loo(ll, r_eff = rep(1, n))
est <- l$estimates
add("elpd_loo", "", est["elpd_loo", "Estimate"], 1e-9)
add("p_loo", "", est["p_loo", "Estimate"], 1e-8)
add("looic", "", est["looic", "Estimate"], 1e-9)
add("se_elpd_loo", "", est["elpd_loo", "SE"], 1e-8)
for (i in seq_len(n)) {
  add("pointwise_elpd_loo", i - 1, l$pointwise[i, "elpd_loo"], 1e-9)
  add("pareto_k", i - 1, l$diagnostics$pareto_k[i], 1e-6, 1e-8)
}
w <- suppressWarnings(waic(ll))
west <- w$estimates
add("elpd_waic", "", west["elpd_waic", "Estimate"], 1e-12)
add("p_waic", "", west["p_waic", "Estimate"], 1e-12)
add("waic", "", west["waic", "Estimate"], 1e-12)
add("se_elpd_waic", "", west["elpd_waic", "SE"], 1e-12)
lppd <- sum(apply(ll, 2, function(x) log(mean(exp(x - max(x)))) + max(x)))
add("lppd", "", lppd, 1e-13)
write.csv(do.call(rbind, rows), "validation/reference/elpd_loo.csv", row.names = FALSE,
          quote = FALSE)
cat("wrote validation/data/loo_loglik.csv and validation/reference/elpd_loo.csv\n")
