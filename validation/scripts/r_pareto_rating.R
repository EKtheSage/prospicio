# Regenerate validation/reference/pareto_rating_r.csv from the R package
# Pareto.
#
#   R CMD INSTALL path/to/a/clone/of/github.com/ulrichriegel/Pareto
#   Rscript validation/scripts/r_pareto_rating.R
#
# The package (GPL) is used only to produce reference values; no code from
# it is used in this repository (docs/design/pareto.md). Rows: Pareto
# extrapolation between layers and the alpha implied by two layers, a
# frequency and a layer, or two frequencies (act_pricing::layer), with and
# without truncation. The package's solvers run at tolerance 1e-14; its
# alphas are compared at 1e-9, and at 1e-8 for alpha = 1, where the layer
# integral is logarithmic and the package's solver returns about 1 ± 5e-9
# (the exact answer is 1: the inputs were extrapolated at alpha = 1).

library(Pareto)

out <- "validation/reference/pareto_rating_r.csv"
source <- paste0("R Pareto ", as.character(packageVersion("Pareto")))
tol <- 1e-14

rows <- list()
add <- function(quantity, params, value, rel_tol) {
  rows[[length(rows) + 1]] <<- data.frame(
    quantity = quantity, params = params, expected = sprintf("%.17g", value),
    abs_tol = 1e-300, rel_tol = rel_tol, source = source
  )
}
num <- function(x) format(x, scientific = FALSE)
trunc_param <- function(tr) if (is.null(tr)) "" else paste0(";truncation=", num(tr))

layer_pairs <- list(
  c(1000, 1000, 3000, 2000),
  c(500, 500, Inf, 5000),
  c(2000, 1000, 4000, 2000)
)
for (tr in list(NULL, 50000)) {
  for (lp in layer_pairs) {
    if (!is.null(tr) && is.infinite(lp[3])) next
    for (alpha in c(0.7, 1, 1.6, 2, 3.5)) {
      if (is.infinite(lp[3]) && alpha <= 1) next
      params <- paste0("c1=", num(lp[1]), ";a1=", num(lp[2]), ";c2=", num(lp[3]),
                       ";a2=", num(lp[4]), ";alpha=", alpha, trunc_param(tr))
      e2 <- Pareto_Extrapolation(lp[1], lp[2], lp[3], lp[4], alpha, ExpLoss_1 = 1,
                                 truncation = tr)
      add("extrapolation", params, e2, 1e-12)
      params <- paste0("c1=", num(lp[1]), ";a1=", num(lp[2]), ";e1=1;c2=", num(lp[3]),
                       ";a2=", num(lp[4]), ";e2=", sprintf("%.17g", e2), trunc_param(tr))
      add("alpha_between_layers", params,
          Pareto_Find_Alpha_btw_Layers(lp[1], lp[2], 1, lp[3], lp[4], e2,
                                       tolerance = tol, truncation = tr),
          if (alpha == 1) 1e-8 else 1e-9)
    }
  }
  for (case in list(c(800, 3, 1000, 1000, 1200), c(5000, 0.5, 1000, 1000, 900),
                    c(1000, 0.5, 1e4, 2000, 300))) {
    params <- paste0("threshold=", num(case[1]), ";frequency=", case[2], ";c=", num(case[3]),
                     ";a=", num(case[4]), ";e=", num(case[5]), trunc_param(tr))
    add("alpha_between_frequency_and_layer", params,
        Pareto_Find_Alpha_btw_FQ_Layer(case[1], case[2], case[3], case[4], case[5],
                                       tolerance = tol, truncation = tr), 1e-9)
  }
  for (case in list(c(1000, 4, 2000, 1), c(1000, 2, 5000, 0.3), c(3000, 0.2, 1000, 5))) {
    params <- paste0("t1=", num(case[1]), ";f1=", case[2], ";t2=", num(case[3]),
                     ";f2=", case[4], trunc_param(tr))
    add("alpha_between_frequencies", params,
        Pareto_Find_Alpha_btw_FQs(case[1], case[2], case[3], case[4],
                                  tolerance = tol, truncation = tr), 1e-9)
  }
}

write.csv(do.call(rbind, rows), out, row.names = FALSE, quote = FALSE)
message("wrote ", out)
