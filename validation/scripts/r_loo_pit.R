# Regenerate validation/data/loo_pit_yrep.csv, validation/data/loo_pit_y.csv
# and validation/reference/loo_pit.csv.
#
#   Rscript validation/scripts/r_loo_pit.R
#
# The reference for prospicio_bayes::ppc::loo_pit. It rebuilds the normal
# model of validation/scripts/r_loo.R (the same seed and draws, so the
# log-likelihood is validation/data/loo_loglik.csv, checked here), then
# draws one replicated dataset per posterior draw, y_rep[s, i] ~
# N(mu_s, sigma_s), and computes each observation's leave-one-out PIT as
# loo's E_loo of the indicator y_rep <= y_i under the PSIS weights for
# leaving y_i out (r_eff = 1), the definition of rstantools::loo_pit. The
# data are continuous, so there are no ties to randomize. The tolerance,
# 1e-9, is that of the pointwise elpd_loo in validation/reference/elpd_loo.csv:
# the Pareto fits agree to about 1e-8 and the CSVs carry 15 digits.
#
# loo is GPL-3: this script only produces reference values.

library(loo)
set.seed(20261004)
n <- 30
y <- c(rnorm(n - 2, 1, 1), 6, -5)
S <- 500
sigma2 <- (n - 1) * var(y) / rchisq(S, n - 1)
mu <- rnorm(S, mean(y), sqrt(sigma2 / n))
ll <- sapply(y, function(yi) dnorm(yi, mu, sqrt(sigma2), log = TRUE))
committed <- as.matrix(read.csv("validation/data/loo_loglik.csv"))
stopifnot(max(abs(committed - ll)) < 1e-12)

yrep <- matrix(rnorm(S * n, mu, sqrt(sigma2)), S, n)
write.csv(yrep, "validation/data/loo_pit_yrep.csv", row.names = FALSE)
write.csv(data.frame(y = y), "validation/data/loo_pit_y.csv", row.names = FALSE)

ps <- psis(-ll, r_eff = rep(1, n))
below <- sweep(yrep, 2, y, "<=") * 1
pit <- E_loo(below, ps, type = "mean", log_ratios = -ll)$value

source <- paste("loo", packageVersion("loo"))
rows <- data.frame(quantity = "loo_pit", index = seq_len(n) - 1,
                   expected = sprintf("%.17g", pit), abs_tol = 1e-9, rel_tol = 0,
                   source = source)
write.csv(rows, "validation/reference/loo_pit.csv", row.names = FALSE, quote = FALSE)
cat("wrote validation/data/loo_pit_yrep.csv, loo_pit_y.csv and validation/reference/loo_pit.csv\n")
