# Regenerate validation/data/gam_surface.csv and
# validation/reference/gam_tensor_mgcv.csv.
#
#   Rscript validation/scripts/r_gam_tensor.R
#
# Tensor product smooths with mgcv: te(x, z, bs = "ps", k = c(5, 5),
# np = FALSE), cubic B-spline margins with second-order difference
# penalties and no marginal reparameterization, so the penalties are
# P_x (x) I and I (x) P_z on the B-spline coefficients, as prospicio builds
# them. A smooth surface plus noise (Gaussian, by REML and by GCV), Poisson
# counts on the surface (REML), and a P-spline of a third variable next to
# the tensor (REML). The fit (deviance, edf, scale, score, fitted values)
# does not depend on how mgcv scales the penalties or which basis of the
# centering constraint it takes. With two smoothing parameters GCV is flat
# near its minimum: mgcv stops just above prospicio's score (by about
# 1e-9 relative), with fitted values up to 1e-4 apart, so that case's fit
# carries a looser tolerance.

library(mgcv)
set.seed(20261014)
n <- 300
d <- data.frame(x = runif(n), z = runif(n), w = runif(n))
f <- sin(3 * d$x) * cos(2 * d$z)
d$y <- f + 0.2 * rnorm(n)
d$counts <- rpois(n, exp(0.5 + f))
d$y2 <- f + 0.5 * sin(5 * d$w) + 0.2 * rnorm(n)
write.csv(d, "validation/data/gam_surface.csv", row.names = FALSE, quote = FALSE)

cases <- list(
  te_gaussian_reml = gam(y ~ te(x, z, bs = "ps", k = c(5, 5), np = FALSE), data = d,
                         method = "REML"),
  te_gaussian_gcv = gam(y ~ te(x, z, bs = "ps", k = c(5, 5), np = FALSE), data = d,
                        method = "GCV.Cp"),
  te_poisson_reml = gam(counts ~ te(x, z, bs = "ps", k = c(5, 5), np = FALSE), data = d,
                        family = poisson(), method = "REML"),
  s_te_reml = gam(y2 ~ s(w, bs = "ps", k = 8) + te(x, z, bs = "ps", k = c(5, 5), np = FALSE),
                  data = d, method = "REML")
)

source <- paste("mgcv", packageVersion("mgcv"))
rows <- list()
add <- function(case, quantity, arg, value, rel) {
  rows[[length(rows) + 1]] <<- data.frame(
    case = case, quantity = quantity, arg = arg, expected = sprintf("%.17g", value),
    abs_tol = 1e-12, rel_tol = rel, source = source
  )
}
for (name in names(cases)) {
  m <- cases[[name]]
  add(name, "deviance", "", deviance(m), 1e-5)
  add(name, "edf", "", sum(m$edf), 1e-4)
  add(name, "scale", "", m$sig2, 1e-5)
  add(name, "score", "", m$gcv.ubre, 1e-6)
  if (!is.null(m$reml.scale) && grepl("reml", name)) add(name, "reml_scale", "", m$reml.scale, 1e-5)
  for (i in c(1, 50, 200, 300)) {
    add(name, "fitted", i - 1, fitted(m)[i], if (name == "te_gaussian_gcv") 2e-4 else 1e-5)
  }
}
out <- do.call(rbind, rows)
write.csv(out, "validation/reference/gam_tensor_mgcv.csv", row.names = FALSE, quote = FALSE)
