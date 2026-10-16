# Regenerate validation/reference/gam_cyclic_mgcv.csv.
#
#   Rscript validation/scripts/r_gam_cyclic.R
#
# Cyclic P-splines (prospicio_glm::gam::PSpline::cyclic) against mgcv's
# bs = "cp": k = 10 cubic cyclic B-splines with a second-order cyclic
# difference penalty. The data, which the Rust test recomputes: for
# i = 1..300, t = ((37 i) mod 300 + 0.5) / 25 in (0, 12), a noise
# e = ((7919 i) mod 13) / 6 - 1, the Gaussian response
# 2 + sin(2 pi t / 12) + 0.5 cos(4 pi t / 12) + 0.4 e and the count
# max(0, round(exp(0.5 + 0.6 sin(2 pi t / 12)) + 1.2 e + 1.5)).
#
# Rows: the raw basis (before the centering constraint) at a few points and
# the penalty matrix, from smoothCon with the period (0, 12), exact to
# 1e-12; then fits by GCV and REML with the period, and by GCV over the
# data's range (no knots given), with the same quantities as r_gam.R.

library(mgcv)

n <- 300
i <- seq_len(n)
t <- ((37 * i) %% 300 + 0.5) / 25
e <- ((7919 * i) %% 13) / 6 - 1
d <- data.frame(
  t = t,
  gauss = 2 + sin(2 * pi * t / 12) + 0.5 * cos(4 * pi * t / 12) + 0.4 * e,
  count = pmax(0, round(exp(0.5 + 0.6 * sin(2 * pi * t / 12)) + 1.2 * e + 1.5))
)
period <- list(t = c(0, 12))

source <- paste("mgcv", packageVersion("mgcv"))
rows <- list()
add <- function(case, quantity, arg, value, rel) {
  rows[[length(rows) + 1]] <<- data.frame(
    case = case, quantity = quantity, arg = arg, expected = sprintf("%.17g", value),
    abs_tol = 1e-12, rel_tol = rel, source = source
  )
}

sm <- smoothCon(s(t, bs = "cp", k = 10), data = d, knots = period,
                absorb.cons = FALSE, scale.penalty = FALSE)[[1]]
for (x in c(0, 0.7, 3.3, 11.2, 12)) {
  b <- PredictMat(sm, data.frame(t = x))
  for (j in seq_len(ncol(b))) add("basis", "raw", sprintf("%g;%d", x, j - 1), b[1, j], 1e-12)
}
S <- sm$S[[1]]
for (a in seq_len(nrow(S))) for (b in seq_len(ncol(S))) {
  add("basis", "penalty", sprintf("%d;%d", a - 1, b - 1), S[a, b], 1e-12)
}

cases <- list(
  cyclic_gaussian = gam(gauss ~ s(t, bs = "cp", k = 10), data = d, knots = period,
                        family = gaussian(), method = "GCV.Cp"),
  cyclic_range_gaussian = gam(gauss ~ s(t, bs = "cp", k = 10), data = d,
                              family = gaussian(), method = "GCV.Cp"),
  reml_cyclic_gaussian = gam(gauss ~ s(t, bs = "cp", k = 10), data = d, knots = period,
                             family = gaussian(), method = "REML"),
  reml_cyclic_poisson = gam(count ~ s(t, bs = "cp", k = 10), data = d, knots = period,
                            family = poisson(), method = "REML")
)
for (name in names(cases)) {
  m <- cases[[name]]
  reml <- startsWith(name, "reml")
  add(name, "deviance", "", deviance(m), 1e-6)
  add(name, "edf", "", sum(m$edf), 1e-5)
  add(name, "scale", "", m$sig2, 1e-6)
  add(name, "score", "", m$gcv.ubre, 1e-6)
  if (reml) add(name, "reml_scale", "", m$reml.scale, 1e-6)
  for (r in c(1, 100, 300)) add(name, "fitted", r - 1, fitted(m)[r], 1e-6)
  # The mean at the period's ends, equal by construction.
  if (name != "cyclic_range_gaussian") {
    p <- predict(m, data.frame(t = c(0, 12)), type = "response")
    add(name, "predict", "0", p[1], 1e-6)
    add(name, "predict", "12", p[2], 1e-6)
  }
}
out <- do.call(rbind, rows)
write.csv(out, "validation/reference/gam_cyclic_mgcv.csv", row.names = FALSE, quote = FALSE)
message("wrote ", nrow(out), " rows")
