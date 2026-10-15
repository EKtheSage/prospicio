# Regenerate validation/reference/gam_mgcv.csv.
#
#   Rscript validation/scripts/r_gam.R
#
# GAMs on validation/data/glm_policies.csv with mgcv: a P-spline smooth of
# age (bs = "ps", k = 10: cubic B-splines, second-order difference penalty)
# plus region in treatment coding, smoothing chosen by GCV.Cp (GCV when the
# scale is estimated, UBRE when it is known). Records the deviance, total
# effective degrees of freedom, scale, the GCV/UBRE score, the region[B]
# coefficient and fitted values at a few rows. These do not depend on how
# the penalty is scaled, so they compare across implementations.
#
# Age is nearly linear in the first three cases: the best smoothing
# parameter is effectively infinite and the score is flat there, so mgcv
# stops at a large finite one while other optimizers go further. Those
# cases carry looser tolerances (scores still agree to about 3e-6); the
# two "wavy" cases, with 9 to 11 effective degrees of freedom, are the
# tight check of the smoothing selection.
#
# REML (the "reml_" cases): mgcv's Newton iteration stops once the REML
# score settles to about 3e-7 relative. Where the score is flat in the
# smoothing parameters (the nearly linear cases, and the two smooths
# searched jointly) prospicio goes on to a lower score, so those cases
# compare the score to 1e-6 and the fit more loosely; the test also checks
# that prospicio's score is no higher than mgcv's. The wavy single-smooth
# cases agree tightly.

library(mgcv)
d <- read.csv("validation/data/glm_policies.csv")
d$region <- factor(d$region)

cases <- list(
  gaussian_identity = gam(gauss ~ region + s(age, bs = "ps", k = 10), data = d,
                          family = gaussian(), method = "GCV.Cp"),
  poisson_log = gam(claims ~ region + s(age, bs = "ps", k = 10) + offset(log(exposure)),
                    data = d, family = poisson(), method = "GCV.Cp"),
  gamma_log = gam(severity ~ region + s(age, bs = "ps", k = 10), data = d,
                  family = Gamma(link = "log"), method = "GCV.Cp"),
  wavy_gaussian = gam(wavy ~ region + s(age, bs = "ps", k = 12), data = d,
                      family = gaussian(), method = "GCV.Cp"),
  wavy_poisson = gam(wavy_claims ~ region + s(age, bs = "ps", k = 12) + offset(log(exposure)),
                     data = d, family = poisson(), method = "GCV.Cp"),
  # REML: the same models, a gamma whose log link is not canonical (so the
  # Laplace approximation's Hessian uses the observed weights, and the
  # scale is estimated through the gamma's saturated log-likelihood), and
  # two smooths searched jointly.
  reml_gaussian = gam(gauss ~ region + s(age, bs = "ps", k = 10), data = d,
                      family = gaussian(), method = "REML"),
  reml_poisson = gam(claims ~ region + s(age, bs = "ps", k = 10) + offset(log(exposure)),
                     data = d, family = poisson(), method = "REML"),
  reml_gamma = gam(severity ~ region + s(age, bs = "ps", k = 10), data = d,
                   family = Gamma(link = "log"), method = "REML"),
  reml_wavy_gaussian = gam(wavy ~ region + s(age, bs = "ps", k = 12), data = d,
                           family = gaussian(), method = "REML"),
  reml_wavy_poisson = gam(wavy_claims ~ region + s(age, bs = "ps", k = 12) +
                            offset(log(exposure)), data = d, family = poisson(), method = "REML"),
  reml_two_smooths = gam(wavy ~ region + s(age, bs = "ps", k = 12) +
                           s(exposure, bs = "ps", k = 8), data = d, family = gaussian(),
                         method = "REML")
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
  reml <- startsWith(name, "reml")
  flat <- !grepl("wavy|two", name)
  # tolerances: deviance, edf, scale, score, coef, fitted
  tol <- if (reml && flat) {
    c(3e-4, 1e-2, 3e-4, 1e-6, 1e-3, 3e-4)
  } else if (reml && name == "reml_two_smooths") {
    c(1e-5, 2e-4, 1e-5, 1e-6, 1e-4, 1e-5)
  } else if (flat) {
    c(1e-5, 1e-3, 1e-5, 1e-5, 1e-4, 2e-5)
  } else {
    c(1e-6, 1e-5, 1e-6, 1e-6, 1e-5, 1e-6)
  }
  add(name, "deviance", "", deviance(m), tol[1])
  add(name, "edf", "", sum(m$edf), tol[2])
  add(name, "scale", "", m$sig2, tol[3])
  add(name, "score", "", m$gcv.ubre, tol[4])
  add(name, "coef", "region[B]", coef(m)[["regionB"]], tol[5])
  if (reml) add(name, "reml_scale", "", m$reml.scale, tol[3])
  for (i in c(1, 100, 300, 600)) {
    add(name, "fitted", i - 1, fitted(m)[i], tol[6])
  }
}
out <- do.call(rbind, rows)
write.csv(out, "validation/reference/gam_mgcv.csv", row.names = FALSE, quote = FALSE)
