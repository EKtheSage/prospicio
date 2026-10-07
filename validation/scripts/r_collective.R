# Regenerate validation/reference/collective_r.csv from the R package
# Pareto.
#
#   R CMD INSTALL path/to/a/clone/of/github.com/ulrichriegel/Pareto
#   Rscript validation/scripts/r_collective.R
#
# The package (GPL) is used only to produce reference values; no code from
# it is used in this repository (docs/design/pareto.md). Rows: the
# collective model (prospicio_aggregate::CollectiveModel) as the package's
# PPP_Model (piecewise Pareto severity) and PGP_Model (generalized Pareto),
# with Panjer-class claim counts by dispersion: layer mean, layer variance
# and excess frequency. Binomial cases use a whole number of trials
# (FQ / (1 - dispersion)), so no dispersion adjustment applies. Variances
# are compared at 1e-8 for the reason given in r_pareto.R.

library(Pareto)

out <- "validation/reference/collective_r.csv"
source <- paste0("R Pareto ", as.character(packageVersion("Pareto")))

fmt <- function(v) paste(vapply(v, format, "", scientific = FALSE), collapse = "|")

models <- list(
  list(kind = "ppp", FQ = 2, t = c(1000, 2000, 3000), alpha = c(1, 1.5, 2.5), dispersion = 1),
  list(kind = "ppp", FQ = 2, t = c(1000, 2000, 3000), alpha = c(1, 1.5, 2.5), dispersion = 2.5),
  list(kind = "ppp", FQ = 2, t = c(1000, 2000, 3000), alpha = c(1, 1.5, 2.5), dispersion = 0.5),
  list(kind = "ppp", FQ = 0.7, t = c(500, 5000), alpha = c(2, 1.2), dispersion = 1.8),
  list(kind = "pgp", FQ = 3, t = 1000, alpha_ini = 2, alpha_tail = 1.5, dispersion = 1),
  list(kind = "pgp", FQ = 3, t = 1000, alpha_ini = 0.8, alpha_tail = 3, dispersion = 0.25)
)
layers <- list(c(1000, 1000), c(4000, 1000), c(2500, 2500), c(1e5, 5000), c(Inf, 10000))
xs <- c(500, 1000, 2500, 1e4)

rows <- list()
add <- function(params, quantity, arg, arg2, value, rel_tol) {
  rows[[length(rows) + 1]] <<- data.frame(
    params = params, quantity = quantity,
    arg = arg, arg2 = arg2, expected = sprintf("%.17g", value),
    abs_tol = 1e-300, rel_tol = rel_tol, source = source
  )
}

for (m in models) {
  if (m$kind == "ppp") {
    model <- PPP_Model(m$FQ, m$t, m$alpha, dispersion = m$dispersion)
    params <- paste0("model=ppp;FQ=", m$FQ, ";t=", fmt(m$t), ";alpha=", fmt(m$alpha),
                     ";dispersion=", m$dispersion)
    tail_alpha <- m$alpha[length(m$alpha)]
  } else {
    model <- PGP_Model(m$FQ, m$t, m$alpha_ini, m$alpha_tail, dispersion = m$dispersion)
    params <- paste0("model=pgp;FQ=", m$FQ, ";t=", m$t, ";alpha_ini=", m$alpha_ini,
                     ";alpha_tail=", m$alpha_tail, ";dispersion=", m$dispersion)
    tail_alpha <- m$alpha_tail
  }
  for (x in xs) add(params, "excess_frequency", x, "", Excess_Frequency(model, x), 1e-13)
  for (l in layers) {
    unlimited <- is.infinite(l[1])
    if (unlimited && tail_alpha <= 1) next
    cover <- if (unlimited) "inf" else format(l[1], scientific = FALSE)
    att <- format(l[2], scientific = FALSE)
    add(params, "layer_mean", cover, att, Layer_Mean(model, l[1], l[2]), 1e-12)
    if (unlimited && tail_alpha <= 2) next
    add(params, "layer_variance", cover, att, Layer_Var(model, l[1], l[2]), 1e-8)
  }
}

write.csv(do.call(rbind, rows), out, row.names = FALSE, quote = FALSE)
message("wrote ", out)
