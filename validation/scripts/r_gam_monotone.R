# Regenerate validation/reference/gam_monotone_scam.csv from the R package
# scam.
#
#   install.packages("scam")
#   Rscript validation/scripts/r_gam_monotone.R
#
# Monotone P-splines (prospicio_glm::gam::PSpline::increasing and
# ::decreasing) against scam's bs = "mpi" and "mpd" (k = 10). The data,
# which the Rust test recomputes: for i = 1..300, x = ((37 i) mod 300 +
# 0.5) / 300, a noise e = ((7919 i) mod 13) / 6 - 1, the rising Gaussian
# response 1 + 2 plogis(10 (x - 0.5)) + 0.3 e, and the falling count
# max(0, round(exp(2.5 - 2 plogis(12 (x - 0.4))) + 1.5 e)).
#
# scam scales each penalty (S = c DᵀD); prospicio's is DᵀD, so a fixed
# smoothing parameter sp in scam is lambda = c sp here, recorded in the
# "lambda" rows. Rows per case: deviance, total edf, the GCV or UBRE score,
# the scale, fitted values at a few rows, and the smooth's coefficients on
# the response scale (coefficients.t, the increments exp(gamma)).
#
# scam selects the smoothing parameter by BFGS on the score and stops a
# little short of its minimum (the GCV case, about 1e-4 in edf). The
# "exact_" cases refine it: optimize() over log lambda on scam's own fits
# at fixed sp, to 1e-10, then the same quantities at that sp, compared at
# 1e-6; scam's own selections are compared loosely.
#
# The package (GPL) is used only to produce reference values.

suppressMessages(library(scam))

n <- 300
i <- seq_len(n)
x <- ((37 * i) %% 300 + 0.5) / 300
e <- ((7919 * i) %% 13) / 6 - 1
d <- data.frame(
  x = x,
  rise = 1 + 2 * plogis(10 * (x - 0.5)) + 0.3 * e,
  fall = pmax(0, round(exp(2.5 - 2 * plogis(12 * (x - 0.4))) + 1.5 * e))
)

source <- paste("scam", packageVersion("scam"))
rows <- list()
add <- function(case, quantity, arg, value, rel) {
  rows[[length(rows) + 1]] <<- data.frame(
    case = case, quantity = quantity, arg = arg, expected = sprintf("%.17g", value),
    abs_tol = 1e-10, rel_tol = rel, source = source
  )
}

cases <- list(
  mpi_gaussian_fixed = scam(rise ~ s(x, bs = "mpi", k = 10), data = d, sp = 0.5),
  mpi_gaussian_gcv = scam(rise ~ s(x, bs = "mpi", k = 10), data = d),
  mpd_poisson_fixed = scam(fall ~ s(x, bs = "mpd", k = 10), data = d, family = poisson(),
                           sp = 2),
  mpd_poisson_ubre = scam(fall ~ s(x, bs = "mpd", k = 10), data = d, family = poisson())
)
# Refine each selected case: minimize scam's score over lambda at fixed sp.
for (name in c("mpi_gaussian_gcv", "mpd_poisson_ubre")) {
  m <- cases[[name]]
  c <- m$smooth[[1]]$S[[1]][1, 1]
  refit <- function(lambda) update(m, sp = lambda / c)
  score <- function(f) {
    edf <- sum(f$edf)
    if (f$scale.estimated) n * f$deviance / (n - edf)^2 else f$deviance / n + 2 * edf / n - 1
  }
  o <- optimize(function(l) score(refit(exp(l))), log(m$sp[[1]] * c) + c(-1, 1), tol = 1e-10)
  best <- refit(exp(o$minimum))
  best$gcv.ubre <- o$objective
  cases[[paste0("exact_", name)]] <- best
}
for (name in names(cases)) {
  m <- cases[[name]]
  fixed <- endsWith(name, "fixed") || startsWith(name, "exact_")
  # Fixed smoothing: the penalized estimate is unique, so tight. Selected:
  # scam's BFGS on the score stops near, not at, its minimum.
  exact <- startsWith(name, "exact_")
  tol <- if (exact) 1e-6 else if (fixed) 1e-7 else 1e-3
  lambda_tol <- if (exact) 1e-4 else if (fixed) 1e-12 else 1e-2
  add(name, "lambda", "", m$sp[[1]] * m$smooth[[1]]$S[[1]][1, 1], lambda_tol)
  add(name, "deviance", "", m$deviance, tol)
  # The score is flat at its minimum: edf there is pinned to about 1e-5.
  add(name, "edf", "", sum(m$edf), if (exact) 1e-5 else tol)
  # scam reports no score at fixed smoothing parameters.
  if (!fixed || exact) add(name, "score", "", m$gcv.ubre, if (exact) 1e-9 else tol)
  add(name, "scale", "", m$sig2, tol)
  for (r in c(1, 100, 300)) add(name, "fitted", r - 1, fitted(m)[r], tol)
  ct <- m$coefficients.t
  for (j in 2:length(ct)) add(name, "coef", j - 1, ct[[j]], if (exact) 1e-4 else if (fixed) 1e-6 else 1e-2)
}
out <- do.call(rbind, rows)
write.csv(out, "validation/reference/gam_monotone_scam.csv", row.names = FALSE, quote = FALSE)
message("wrote ", nrow(out), " rows")
