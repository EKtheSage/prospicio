# Regenerate validation/reference/piecewise_pareto_r.csv from the R package
# Pareto.
#
#   R CMD INSTALL path/to/a/clone/of/github.com/ulrichriegel/Pareto
#   Rscript validation/scripts/r_piecewise_pareto.R
#
# The package (GPL) is used only to produce reference values; no code from
# it is used in this repository (docs/design/pareto.md). Rows: piecewise
# Pareto distribution function, quantile, layer mean, layer second moment
# and layer variance, untruncated and with both truncation types. Second
# moments and variances are compared at 1e-8 for the reason given in
# r_pareto.R.

library(Pareto)

out <- "validation/reference/piecewise_pareto_r.csv"
source <- paste0("R Pareto ", as.character(packageVersion("Pareto")))

cases <- list(
  list(t = c(1000, 2000, 3000), alpha = c(1, 1.5, 2), truncation = NULL, type = "lp"),
  list(t = c(1000, 2000, 3000), alpha = c(1, 1.5, 2), truncation = 5000, type = "lp"),
  list(t = c(1000, 2000, 3000), alpha = c(1, 1.5, 2), truncation = 5000, type = "wd"),
  list(t = c(1000, 1500, 4000, 10000), alpha = c(2.5, 0.4, 3, 1.2), truncation = NULL, type = "lp"),
  list(t = c(1000, 1500, 4000, 10000), alpha = c(2.5, 0.4, 3, 1.2), truncation = 1e6, type = "wd"),
  list(t = c(200, 1000), alpha = c(1, 3), truncation = NULL, type = "lp")
)
layers <- list(c(500, 0), c(1000, 1000), c(4000, 1000), c(2500, 2500),
               c(1e5, 5000), c(Inf, 3000))
xs <- c(500, 1000, 1700, 2500, 3500, 4999, 12000)
ps <- c(0.001, 0.1, 0.5, 0.7, 0.9, 0.999)

fmt <- function(v) paste(vapply(v, format, "", scientific = FALSE), collapse = "|")

rows <- list()
add <- function(params, quantity, arg, arg2, value, rel_tol) {
  rows[[length(rows) + 1]] <<- data.frame(
    distribution = "piecewise_pareto", params = params, quantity = quantity,
    arg = arg, arg2 = arg2, expected = sprintf("%.17g", value),
    abs_tol = 1e-300, rel_tol = rel_tol, source = source
  )
}

for (cs in cases) {
  params <- paste0("t=", fmt(cs$t), ";alpha=", fmt(cs$alpha),
                   if (is.null(cs$truncation)) "" else
                     paste0(";truncation=", format(cs$truncation, scientific = FALSE),
                            ";type=", cs$type))
  tail_alpha <- cs$alpha[length(cs$alpha)]
  pp <- function(f, ...) f(..., t = cs$t, alpha = cs$alpha, truncation = cs$truncation,
                           truncation_type = cs$type)
  for (x in xs) add(params, "cdf", x, "", pPiecewisePareto(x, cs$t, cs$alpha,
    truncation = cs$truncation, truncation_type = cs$type), 1e-13)
  for (p in ps) add(params, "quantile", p, "", qPiecewisePareto(p, cs$t, cs$alpha,
    truncation = cs$truncation, truncation_type = cs$type), 1e-13)
  for (l in layers) {
    unlimited <- is.infinite(l[1]) && is.null(cs$truncation)
    if (unlimited && tail_alpha <= 1) next
    cover <- if (is.infinite(l[1])) "inf" else format(l[1], scientific = FALSE)
    att <- format(l[2], scientific = FALSE)
    add(params, "layer", cover, att, pp(PiecewisePareto_Layer_Mean, l[1], l[2]), 1e-12)
    if (unlimited && tail_alpha <= 2) next
    add(params, "layer_second_moment", cover, att, pp(PiecewisePareto_Layer_SM, l[1], l[2]), 1e-8)
    add(params, "layer_variance", cover, att, pp(PiecewisePareto_Layer_Var, l[1], l[2]), 1e-8)
  }
}

write.csv(do.call(rbind, rows), out, row.names = FALSE, quote = FALSE)
message("wrote ", out)
