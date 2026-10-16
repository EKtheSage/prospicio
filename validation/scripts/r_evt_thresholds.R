# Regenerate validation/reference/evt_thresholds_r.csv from the R package
# ismev.
#
#   install.packages("ismev")
#   Rscript validation/scripts/r_evt_thresholds.R
#
# Threshold diagnostics (prospicio_prob::evt::threshold_diagnostics) on two
# data sets the Rust test recomputes: the quantiles at (i - 0.5) / n of a
# Burr (heavy tail, xi -> 1/3) and of a Weibull with shape 2 (light tail,
# xi -> 0 from below). Rows:
#
# - mean excess and its standard error, as ismev::mrl.plot computes its
#   bands (exact arithmetic, 1e-12);
# - the GPD fit at each threshold from ismev::gpd.fit (Nelder-Mead and a
#   numerical Hessian, so loose tolerances): shape, scale, modified scale
#   and the standard errors gpd.fitrange plots;
# - the same quantities at the exact estimate (the root of the profile
#   score in theta = xi / beta by uniroot) with the observed information
#   from R's symbolic deriv(), at tight tolerances.
#
# The package (GPL) is used only to produce reference values.

suppressMessages(library(ismev))

out <- "validation/reference/evt_thresholds_r.csv"
source <- paste0("R ismev ", as.character(packageVersion("ismev")))

cases <- list(
  list(name = "burr", n = 1000, a = 2, c = 1.5, u = c(0.5, 1, 1.5, 2, 3)),
  list(name = "weibull", n = 1000, k = 2, u = c(0.5, 1, 1.5, 2))
)

data_of <- function(cs) {
  p <- (seq_len(cs$n) - 0.5) / cs$n
  if (cs$name == "burr") ((1 - p)^(-1 / cs$a) - 1)^(1 / cs$c) else (-log1p(-p))^(1 / cs$k)
}

params_of <- function(cs) {
  if (cs$name == "burr") sprintf("data=burr;n=%d;a=%g;c=%g", cs$n, cs$a, cs$c)
  else sprintf("data=weibull;n=%d;k=%g", cs$n, cs$k)
}

loglik <- deriv(~ -log(b) - (1 + 1 / x) * log(1 + x * y / b), c("x", "b"),
                function(x, b, y) {}, hessian = TRUE)

exact_fit <- function(y, th0) {
  n <- length(y)
  xi_of <- function(th) mean(log1p(th * y))
  score <- function(th) {
    dxi <- mean(y / (1 + th * y))
    -dxi / xi_of(th) + 1 / th - dxi
  }
  # Bracket the root of the score around ismev's estimate th0.
  lo <- -1 / max(y)
  w <- abs(th0) * 0.1
  a <- max(th0 - w, lo + (th0 - lo) / 2); b <- th0 + w
  while (score(a) * score(b) > 0) {
    w <- 2 * w
    a <- max(th0 - w, lo + (a - lo) / 2); b <- th0 + w
  }
  th <- uniroot(score, c(a, b), tol = 1e-15)$root
  xi <- xi_of(th); beta <- xi / th
  h <- attr(loglik(xi, beta, y), "hessian")
  info <- -matrix(c(sum(h[, 1, 1]), sum(h[, 1, 2]), sum(h[, 2, 1]), sum(h[, 2, 2])), 2)
  list(xi = xi, beta = beta, cov = solve(info))
}

rows <- list()
add <- function(params, quantity, u, value, rel_tol) {
  rows[[length(rows) + 1]] <<- data.frame(
    distribution = "evt_thresholds", params = params, quantity = quantity,
    arg = sprintf("%.17g", u), expected = sprintf("%.17g", value),
    abs_tol = 1e-300, rel_tol = rel_tol, source = source
  )
}

for (cs in cases) {
  x <- data_of(cs)
  params <- params_of(cs)
  for (u in cs$u) {
    y <- x[x > u] - u
    add(params, "n_above", u, length(y), 0)
    add(params, "mean_excess", u, mean(y), 1e-12)
    add(params, "mean_excess_se", u, sqrt(var(y)) / sqrt(length(y)), 1e-12)

    z <- gpd.fit(x, u, show = FALSE)
    d <- matrix(c(1, -u), ncol = 1)
    add(params, "ismev_xi", u, z$mle[2], 2e-3)
    add(params, "ismev_beta", u, z$mle[1], 1e-3)
    add(params, "ismev_modified_scale", u, z$mle[1] - z$mle[2] * u, 1e-3)
    add(params, "ismev_xi_se", u, z$se[2], 2e-2)
    add(params, "ismev_modified_scale_se", u, sqrt(t(d) %*% z$cov %*% d), 2e-2)

    e <- exact_fit(y, z$mle[2] / z$mle[1])
    # R's MLE parameter order is (scale, shape); ours is (shape, scale).
    add(params, "xi", u, e$xi, 1e-9)
    add(params, "beta", u, e$beta, 1e-9)
    add(params, "modified_scale", u, e$beta - e$xi * u, 1e-9)
    add(params, "xi_se", u, sqrt(e$cov[1, 1]), 1e-8)
    add(params, "modified_scale_se", u,
        sqrt(e$cov[2, 2] - 2 * u * e$cov[1, 2] + u^2 * e$cov[1, 1]), 1e-8)
  }
}

write.csv(do.call(rbind, rows), out, row.names = FALSE, quote = FALSE)
message("wrote ", out)
