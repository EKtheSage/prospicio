# Regenerate validation/reference/pml_curve_r.csv from the R package
# Pareto.
#
#   R CMD INSTALL path/to/a/clone/of/github.com/ulrichriegel/Pareto
#   Rscript validation/scripts/r_pml_curve.R
#
# The package (GPL) is used only to produce reference values; no code from
# it is used in this repository (docs/design/pareto.md). Rows: the
# frequency, thresholds and alphas of the model through a PML curve
# (act_pricing::tower::fit_pml_curve), all closed forms.

library(Pareto)

out <- "validation/reference/pml_curve_r.csv"
source <- paste0("R Pareto ", as.character(packageVersion("Pareto")))

fmt <- function(v) paste(vapply(v, function(x) format(x, scientific = FALSE), ""), collapse = "|")
curves <- list(
  list(rp = c(5, 20, 50, 200), x = c(2e6, 5e6, 9e6, 2e7), tail = 1.8, tr = NULL),
  list(rp = c(10, 40, 100), x = c(1e6, 2e6, 3e6), tail = 2, tr = 5e7),
  list(rp = c(2, 10, 25, 100, 250), x = c(5e5, 1.5e6, 3e6, 6e6, 1.2e7), tail = 1.2, tr = NULL)
)

rows <- list()
add <- function(params, quantity, index, value) {
  rows[[length(rows) + 1]] <<- data.frame(
    params = params, quantity = quantity, index = index,
    expected = sprintf("%.17g", value), abs_tol = 1e-300, rel_tol = 1e-13, source = source
  )
}
for (cv in curves) {
  m <- Fit_PML_Curve(cv$rp, cv$x, tail_alpha = cv$tail, truncation = cv$tr)
  params <- paste0("rp=", fmt(cv$rp), ";x=", fmt(cv$x), ";tail=", cv$tail,
                   if (is.null(cv$tr)) "" else paste0(";truncation=", format(cv$tr, scientific = FALSE)))
  add(params, "frequency", 0, m$FQ)
  for (j in seq_along(m$t)) add(params, "threshold", j - 1, m$t[j])
  for (j in seq_along(m$alpha)) add(params, "alpha", j - 1, m$alpha[j])
}

write.csv(do.call(rbind, rows), out, row.names = FALSE, quote = FALSE)
message("wrote ", out)
