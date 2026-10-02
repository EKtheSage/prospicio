# Regenerate validation/reference/gen_pareto_r.csv from the R package
# Pareto.
#
#   R CMD INSTALL path/to/a/clone/of/github.com/ulrichriegel/Pareto
#   Rscript validation/scripts/r_gen_pareto.R
#
# The package (GPL) is used only to produce reference values; no code from
# it is used in this repository (docs/design/pareto.md). Rows: Riegel's
# generalized Pareto (act_prob::evt::Gpd::riegel) distribution function,
# quantile, layer mean, layer second moment and layer variance. Second
# moments and variances are compared at 1e-8 for the reason given in
# r_pareto.R.

library(Pareto)

out <- "validation/reference/gen_pareto_r.csv"
source <- paste0("R Pareto ", as.character(packageVersion("Pareto")))

cases <- list(
  list(t = 1000, alpha_ini = 2, alpha_tail = 1.5),
  list(t = 1000, alpha_ini = 0.5, alpha_tail = 3),
  list(t = 1000, alpha_ini = 4, alpha_tail = 0.9),
  list(t = 500, alpha_ini = 1.2, alpha_tail = 2.5)
)
layers <- list(c(500, 0), c(1000, 1000), c(4000, 1000), c(5000, 5000),
               c(1e6, 2e5), c(Inf, 3000))
xs <- c(500, 1000, 1700, 5000, 1e5)
ps <- c(0.001, 0.1, 0.5, 0.9, 0.999)

rows <- list()
add <- function(params, quantity, arg, arg2, value, rel_tol) {
  rows[[length(rows) + 1]] <<- data.frame(
    distribution = "gen_pareto", params = params, quantity = quantity,
    arg = arg, arg2 = arg2, expected = sprintf("%.17g", value),
    abs_tol = 1e-300, rel_tol = rel_tol, source = source
  )
}

for (cs in cases) {
  params <- paste0("t=", cs$t, ";alpha_ini=", cs$alpha_ini, ";alpha_tail=", cs$alpha_tail)
  for (x in xs) add(params, "cdf", x, "", pGenPareto(x, cs$t, cs$alpha_ini, cs$alpha_tail), 1e-13)
  for (p in ps) add(params, "quantile", p, "", qGenPareto(p, cs$t, cs$alpha_ini, cs$alpha_tail), 1e-13)
  for (l in layers) {
    unlimited <- is.infinite(l[1])
    if (unlimited && cs$alpha_tail <= 1) next
    cover <- if (unlimited) "inf" else format(l[1], scientific = FALSE)
    att <- format(l[2], scientific = FALSE)
    add(params, "layer", cover, att,
        GenPareto_Layer_Mean(l[1], l[2], cs$t, cs$alpha_ini, cs$alpha_tail), 1e-12)
    if (unlimited && cs$alpha_tail <= 2) next
    add(params, "layer_second_moment", cover, att,
        GenPareto_Layer_SM(l[1], l[2], cs$t, cs$alpha_ini, cs$alpha_tail), 1e-8)
    add(params, "layer_variance", cover, att,
        GenPareto_Layer_Var(l[1], l[2], cs$t, cs$alpha_ini, cs$alpha_tail), 1e-8)
  }
}

write.csv(do.call(rbind, rows), out, row.names = FALSE, quote = FALSE)
message("wrote ", out)
