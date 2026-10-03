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
                     data = d, family = poisson(), method = "GCV.Cp")
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
  flat <- !startsWith(name, "wavy")
  add(name, "deviance", "", deviance(m), if (flat) 1e-5 else 1e-6)
  add(name, "edf", "", sum(m$edf), if (flat) 1e-3 else 1e-5)
  add(name, "scale", "", m$sig2, if (flat) 1e-5 else 1e-6)
  add(name, "score", "", m$gcv.ubre, if (flat) 1e-5 else 1e-6)
  add(name, "coef", "region[B]", coef(m)[["regionB"]], if (flat) 1e-4 else 1e-5)
  for (i in c(1, 100, 300, 600)) {
    add(name, "fitted", i - 1, fitted(m)[i], if (flat) 2e-5 else 1e-6)
  }
}
out <- do.call(rbind, rows)
write.csv(out, "validation/reference/gam_mgcv.csv", row.names = FALSE, quote = FALSE)
