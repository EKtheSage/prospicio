# Regenerate validation/reference/pareto_r.csv from the R package Pareto.
#
#   R CMD INSTALL path/to/a/clone/of/github.com/ulrichriegel/Pareto
#   Rscript validation/scripts/r_pareto.R
#
# The package (GPL) is used only to produce reference values; no code from
# it is used in this repository (docs/design/pareto.md). Rows: Pareto
# distribution function and quantile, layer mean, layer second moment and
# layer variance, with and without truncation.
#
# The package's second-moment formula cancels for high layers (about 1e-9
# relative on 1m xs 200k at alpha 3.5, against the 30-digit integral in
# layer_moments_mpmath.csv), so second moments and variances are compared at
# 1e-8 here; layer_moments_mpmath.csv is the precision check.

library(Pareto)

out <- "validation/reference/pareto_r.csv"
source <- paste0("R Pareto ", as.character(packageVersion("Pareto")))

cases <- list(
  list(t = 1000, alpha = 0.8, truncation = NULL),
  list(t = 1000, alpha = 1, truncation = NULL),
  list(t = 1000, alpha = 2, truncation = NULL),
  list(t = 1000, alpha = 3.5, truncation = NULL),
  list(t = 500, alpha = 1.2, truncation = 20000),
  list(t = 500, alpha = 2, truncation = 8000)
)
layers <- list(c(500, 0), c(4000, 1000), c(5000, 5000), c(1e6, 2e5), c(Inf, 3000))
xs <- c(500, 1000, 1500, 5000, 1e5)
ps <- c(0, 0.1, 0.5, 0.9, 0.999)

rows <- list()
add <- function(params, quantity, arg, arg2, value, rel_tol) {
  rows[[length(rows) + 1]] <<- data.frame(
    distribution = "pareto", params = params, quantity = quantity,
    arg = arg, arg2 = arg2, expected = sprintf("%.17g", value),
    abs_tol = 1e-300, rel_tol = rel_tol, source = source
  )
}

for (cs in cases) {
  params <- paste0("t=", cs$t, ";alpha=", cs$alpha,
                   if (is.null(cs$truncation)) "" else paste0(";truncation=", cs$truncation))
  for (x in xs) add(params, "cdf", x, "", pPareto(x, cs$t, cs$alpha, truncation = cs$truncation), 1e-13)
  for (p in ps) add(params, "quantile", p, "", qPareto(p, cs$t, cs$alpha, truncation = cs$truncation), 1e-13)
  for (l in layers) {
    unlimited <- is.infinite(l[1]) && is.null(cs$truncation)
    if (unlimited && cs$alpha <= 1) next
    cover <- if (is.infinite(l[1])) "inf" else format(l[1], scientific = FALSE)
    att <- format(l[2], scientific = FALSE)
    add(params, "layer", cover, att,
        Pareto_Layer_Mean(l[1], l[2], cs$alpha, t = cs$t, truncation = cs$truncation), 1e-12)
    if (unlimited && cs$alpha <= 2) next
    add(params, "layer_second_moment", cover, att,
        Pareto_Layer_SM(l[1], l[2], cs$alpha, t = cs$t, truncation = cs$truncation), 1e-8)
    add(params, "layer_variance", cover, att,
        Pareto_Layer_Var(l[1], l[2], cs$alpha, t = cs$t, truncation = cs$truncation), 1e-8)
  }
}

write.csv(do.call(rbind, rows), out, row.names = FALSE, quote = FALSE)
message("wrote ", out)
