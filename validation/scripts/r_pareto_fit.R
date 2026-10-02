# Regenerate validation/reference/pareto_fit_r.csv from the R package
# Pareto.
#
#   R CMD INSTALL path/to/a/clone/of/github.com/ulrichriegel/Pareto
#   Rscript validation/scripts/r_pareto_fit.R
#
# The package (GPL) is used only to produce reference values; no code from
# it is used in this repository (docs/design/pareto.md). Rows: maximum
# likelihood alphas of the Pareto and the piecewise Pareto
# (act_prob::Pareto::fit, act_prob::PiecewisePareto::fit) on fixed data
# sets with reporting thresholds, censoring, weights and truncation. The
# untruncated estimates are closed forms; the package solves truncated
# ones numerically, so those rows are compared at 1e-6, and at 1e-5 where
# the estimate sits at the lower bound 0.001 (the package stops at about
# 0.0010000053).

library(Pareto)

out <- "validation/reference/pareto_fit_r.csv"
source <- paste0("R Pareto ", as.character(packageVersion("Pareto")))

losses <- c(1100, 1300, 1750, 2000, 2600, 3500, 4100, 5200, 7000, 9000, 12000, 18000, 25000, 40000)
reporting <- c(1000, 1200, 1000, 1500, 1000, 1000, 3000, 1000, 1000, 5000, 1000, 1000, 1000, 1000)
censored <- c(FALSE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, FALSE, FALSE, TRUE, FALSE, FALSE, TRUE)
weights <- c(1, 2, 1, 1, 0.5, 1, 1, 3, 1, 1, 1, 1, 2, 1)

cases <- list(
  list(r = NULL, c = NULL, w = NULL),
  list(r = reporting, c = NULL, w = NULL),
  list(r = reporting, c = censored, w = NULL),
  list(r = reporting, c = censored, w = weights)
)

fmt <- function(v) paste(vapply(v, function(x) format(x, scientific = FALSE), ""), collapse = "|")
rows <- list()
add <- function(model, params, index, value, rel_tol) {
  rows[[length(rows) + 1]] <<- data.frame(
    model = model, params = params, index = index,
    expected = sprintf("%.17g", value), abs_tol = 1e-300, rel_tol = rel_tol, source = source
  )
}

for (cs in cases) {
  data <- paste0("r=", if (is.null(cs$r)) "" else fmt(cs$r),
                 ";c=", if (is.null(cs$c)) "" else fmt(as.integer(cs$c)),
                 ";w=", if (is.null(cs$w)) "" else fmt(cs$w))
  for (tr in list(NULL, 60000)) {
    tol <- if (is.null(tr)) 1e-13 else 1e-6
    trp <- if (is.null(tr)) "" else paste0(";truncation=", format(tr, scientific = FALSE))
    a <- Pareto_ML_Estimator_Alpha(losses, 1000, truncation = tr, reporting_thresholds = cs$r,
                                   is.censored = cs$c, weights = cs$w)
    add("pareto", paste0("t=1000;", data, trp), 0, a, tol)
    t <- c(1000, 2500, 8000)
    a <- PiecewisePareto_ML_Estimator_Alpha(losses, t, truncation = tr, truncation_type = "lp",
                                            reporting_thresholds = cs$r, is.censored = cs$c,
                                            weights = cs$w)
    for (j in seq_along(a)) add("piecewise_pareto", paste0("t=", fmt(t), ";", data, trp), j - 1, a[j],
                                if (a[j] < 0.0011) 1e-5 else tol)
  }
}

write.csv(do.call(rbind, rows), out, row.names = FALSE, quote = FALSE)
message("wrote ", out)
