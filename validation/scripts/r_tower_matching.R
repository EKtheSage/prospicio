# Regenerate validation/reference/tower_matching_r.csv from the R package
# Pareto.
#
#   R CMD INSTALL path/to/a/clone/of/github.com/ulrichriegel/Pareto
#   Rscript validation/scripts/r_tower_matching.R
#
# The package (GPL) is used only to produce reference values; no code from
# it is used in this repository (docs/design/pareto.md). Rows: the
# thresholds, alphas and frequency of the piecewise Pareto model that
# PiecewisePareto_Match_Layer_Losses fits to a tower, under the rule that
# minimizes the ratio of the two alphas in each layer
# (act_pricing::tower::SelectionRule::MinimizeAlphaRatio). The package
# minimizes with a tolerance of about 1e-8 in the free thresholds, so rows
# are compared at 1e-6. (Its rule without minimization is not the
# paper's midpoint, so it is not a reference for SelectionRule::Midpoint.)
#
# Only towers where the package's result is the stated minimum are used.
# On the tower 5m xs 5m, 15m xs 10m, Inf xs 25m with losses 2.4m, 1.5m,
# 1.2m and a frequency of 1 above 5m, the package's middle layer has
# |ln(alpha_lower / alpha_upper)| = 0.0703 where 0.0645 is attainable (and
# at that scale its model misses the middle layer's loss by 1.5e-5); that
# tower is checked in act-pricing's unit tests instead. With every
# frequency derived, the lowest layer is matched by one Pareto piece; the
# package merges it there (its alphas agree to its merge tolerance), as
# act_pricing does exactly.

library(Pareto)

out <- "validation/reference/tower_matching_r.csv"
source <- paste0("R Pareto ", as.character(packageVersion("Pareto")))

fmt <- function(v) paste(vapply(v, function(x) if (is.na(x)) "NA" else format(x, scientific = FALSE), ""), collapse = "|")

towers <- list(
  # Riegel (2018), Table 3 with f_1 = 0.25 (Example 4).
  list(a = c(1000, 1500, 2000, 2500, 3000, 5000, 10000),
       e = c(100, 90, 50, 40, 100, 50, 50), f = c(0.25, NA, NA, NA, NA, NA, NA)),
  # The same tower with every frequency derived.
  list(a = c(1000, 1500, 2000, 2500, 3000, 5000, 10000),
       e = c(100, 90, 50, 40, 100, 50, 50), f = rep(NA, 7)),
  # Every frequency given.
  list(a = c(1000, 1500, 2000, 2500, 3000, 5000, 10000),
       e = c(100, 90, 50, 40, 100, 50, 50), f = c(0.3, 0.19, 0.15, 0.09, 0.06, 0.02, 0.008))
)

rows <- list()
add <- function(params, quantity, index, value) {
  rows[[length(rows) + 1]] <<- data.frame(
    params = params, quantity = quantity, index = index,
    expected = sprintf("%.17g", value), abs_tol = 1e-300, rel_tol = 1e-6, source = source
  )
}

for (tw in towers) {
  m <- PiecewisePareto_Match_Layer_Losses(tw$a, tw$e, Frequencies = tw$f, minimize_ratios = TRUE,
                                          tolerance = 1e-12)
  stopifnot(m$Status == 0)
  params <- paste0("a=", fmt(tw$a), ";e=", fmt(tw$e), ";f=", fmt(tw$f))
  add(params, "frequency", 0, m$FQ)
  for (j in seq_along(m$t)) add(params, "threshold", j - 1, m$t[j])
  for (j in seq_along(m$alpha)) add(params, "alpha", j - 1, m$alpha[j])
}

write.csv(do.call(rbind, rows), out, row.names = FALSE, quote = FALSE)
message("wrote ", out)
