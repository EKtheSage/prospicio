# Regenerate validation/reference/mvn_mvtnorm.csv from R mvtnorm.
#
#   Rscript validation/scripts/r_mvn.R
#
# Rows: P(lower < X < upper) for X ~ N(0, R) in 3 to 10 dimensions by
# mvtnorm::pmvnorm with GenzBretz(maxpts = 2e7, abseps = 1e-8), against
# prospicio_math::mvn::mvn_probability. R is a random correlation matrix
# (a seeded normal matrix's cross-product, scaled to unit diagonal), the
# limits seeded; lower limits are -Inf in about half the rows. `matrix` and
# the limits are ";"-joined, row-major; `error` is pmvnorm's estimate.

library(mvtnorm)

out <- "validation/reference/mvn_mvtnorm.csv"
source <- paste0("R mvtnorm ", as.character(packageVersion("mvtnorm")))
set.seed(565)
num <- function(x) ifelse(is.infinite(x), ifelse(x > 0, "inf", "-inf"), sprintf("%.17g", x))
rows <- list()
for (d in 3:10) {
  for (k in 1:4) {
    z <- matrix(rnorm(d * (d + 2)), d + 2, d)
    s <- crossprod(z)
    r <- cov2cor(s)
    if (k == 4) r <- 0.9 * r + 0.1 * diag(d)  # a different off-diagonal scale
    upper <- round(rnorm(d, 0.5, 1), 2)
    lower <- if (k %% 2 == 0) upper - round(runif(d, 0.5, 3), 2) else rep(-Inf, d)
    p <- pmvnorm(lower, upper, sigma = r, algorithm = GenzBretz(maxpts = 2e7, abseps = 1e-8, releps = 0))
    rows[[length(rows) + 1]] <- data.frame(
      dim = d,
      matrix = paste(num(as.vector(t(r))), collapse = ";"),
      lower = paste(num(lower), collapse = ";"),
      upper = paste(num(upper), collapse = ";"),
      expected = sprintf("%.17g", as.numeric(p)),
      error = sprintf("%.3g", attr(p, "error")),
      source = source
    )
  }
}
write.csv(do.call(rbind, rows), out, row.names = FALSE, quote = FALSE)
