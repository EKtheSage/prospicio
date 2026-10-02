# Regenerate validation/reference/local_pareto_r.csv from the R package
# LocalPareto.
#
#   R CMD INSTALL path/to/a/clone/of/github.com/ulrichriegel/LocalPareto
#   Rscript validation/scripts/r_local_pareto.R
#
# The package (GPL) is used only to produce reference values; no code from
# it is used in this repository (docs/design/pareto.md). Rows: the
# log-affine local Pareto (act_prob::LogAffinePareto) distribution
# function, quantile, layer mean, layer second moment and layer variance.
# The package computes layer moments with stats::integrate (relative
# tolerance about 1e-4), so those rows are a coarse check at 1e-6 and rows
# where its integration fails are left out; layer_moments_mpmath.csv is the
# precision check. The distribution function and quantile are closed forms.

library(LocalPareto)

out <- "validation/reference/local_pareto_r.csv"
source <- paste0("R LocalPareto ", as.character(packageVersion("LocalPareto")))

cases <- list(
  list(t = 1000, alpha_0 = 1.5, gamma = 0.3),
  list(t = 1000, alpha_0 = 0.6, gamma = 1.2),
  list(t = 500, alpha_0 = 2.5, gamma = 0.05),
  list(t = 1e6, alpha_0 = 1.2, delta = 0.5)
)
xs <- c(500, 1000, 1700, 5000, 1e5, 3e6)
ps <- c(0.001, 0.1, 0.5, 0.9, 0.999)

rows <- list()
add <- function(params, quantity, arg, arg2, value, rel_tol) {
  rows[[length(rows) + 1]] <<- data.frame(
    params = params, quantity = quantity, arg = arg, arg2 = arg2,
    expected = sprintf("%.17g", value), abs_tol = 1e-300, rel_tol = rel_tol, source = source
  )
}

for (cs in cases) {
  par <- if (is.null(cs$delta)) paste0(";gamma=", cs$gamma) else paste0(";delta=", cs$delta)
  params <- paste0("t=", format(cs$t, scientific = FALSE), ";alpha_0=", cs$alpha_0, par)
  f <- function(fun, ...) fun(..., t = cs$t, alpha_0 = cs$alpha_0, gamma = cs$gamma, delta = cs$delta)
  for (x in xs) add(params, "cdf", format(x, scientific = FALSE), "", f(pLALocPareto, x), 1e-13)
  for (p in ps) add(params, "quantile", p, "", f(qLALocPareto, p), 1e-12)
  layers <- list(c(500, 0), c(1000, cs$t), c(4 * cs$t, cs$t), c(10 * cs$t, 5 * cs$t), c(Inf, 2 * cs$t))
  for (l in layers) {
    cover <- if (is.infinite(l[1])) "inf" else format(l[1], scientific = FALSE)
    att <- format(l[2], scientific = FALSE)
    for (q in list(list("layer", LALocPareto_Layer_Mean),
                   list("layer_second_moment", LALocPareto_Layer_SM),
                   list("layer_variance", LALocPareto_Layer_Var))) {
      v <- tryCatch(f(q[[2]], l[1], l[2]), error = function(e) NA)
      if (!is.na(v)) add(params, q[[1]], cover, att, v, 1e-6)
    }
  }
}

write.csv(do.call(rbind, rows), out, row.names = FALSE, quote = FALSE)
message("wrote ", out)
