# Regenerate validation/data/mcmc_chains.csv and
# validation/reference/mcmc_posterior.csv.
#
#   Rscript validation/scripts/r_mcmc_diagnostics.R
#
# Four chains of 1001 draws (odd, so splitting drops the middle draw) for
# five quantities, simulated with a fixed seed and written to the data file
# so Rust reads the same draws: independent normals, a strongly
# autocorrelated AR(1), a set with one chain shifted (should fail R-hat), a
# heavy-tailed set with one chain on a wider scale (caught by the folded
# R-hat), and a discrete set with many ties. The reference values are the
# R package posterior's rhat, ess_bulk, ess_tail, ess_mean and mcse_mean,
# the estimators of Vehtari et al. (2021).

library(posterior)
set.seed(20261003)
n <- 1001; m <- 4
ar1 <- function(phi, shift = 0) {
  x <- numeric(n); x[1] <- rnorm(1)
  for (i in 2:n) x[i] <- phi * x[i - 1] + rnorm(1)
  x + shift
}
draws <- list(
  iid = replicate(m, rnorm(n)),
  ar_09 = replicate(m, ar1(0.9)),
  shifted = cbind(ar1(0.5), ar1(0.5), ar1(0.5), ar1(0.5, 1.5)),
  scale = cbind(rt(n, 3), rt(n, 3), rt(n, 3), 4 * rt(n, 3)),
  ties = replicate(m, rpois(n, 2))
)

long <- do.call(rbind, lapply(seq_len(m), function(j) {
  data.frame(chain = j - 1, iteration = seq_len(n) - 1,
             iid = draws$iid[, j], ar_09 = draws$ar_09[, j],
             shifted = draws$shifted[, j], scale = draws$scale[, j],
             ties = draws$ties[, j])
}))
fmt <- function(x) sprintf("%.17g", x)
long[] <- lapply(long, function(c) if (is.double(c)) fmt(c) else c)
write.csv(long, "validation/data/mcmc_chains.csv", row.names = FALSE, quote = FALSE)

source <- paste("posterior", packageVersion("posterior"))
rows <- list()
for (v in names(draws)) {
  x <- draws[[v]]
  vals <- c(rhat = rhat(x), ess_bulk = ess_bulk(x), ess_tail = ess_tail(x),
            ess_mean = ess_mean(x), mcse_mean = mcse_mean(x))
  for (q in names(vals)) {
    rows[[length(rows) + 1]] <- data.frame(variable = v, quantity = q, expected = fmt(vals[[q]]),
                                           abs_tol = 1e-12, rel_tol = 1e-10, source = source)
  }
}
write.csv(do.call(rbind, rows), "validation/reference/mcmc_posterior.csv",
          row.names = FALSE, quote = FALSE)
