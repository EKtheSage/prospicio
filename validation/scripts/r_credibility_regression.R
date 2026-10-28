# Regenerate validation/reference/credibility_regression_actuar.csv from the
# R package actuar: Hachemeister's regression credibility model
# (prospicio_pricing::credibility::Hachemeister) on Hachemeister's data
# (validation/reference/hachemeister.csv, written by r_credibility.R).
#
#   install.packages("actuar")
#   Rscript validation/scripts/r_credibility_regression.R
#
# actuar (GPL) is used only to produce reference values. Each fit is
# actuar::cm(~ state, ..., regformula = ~ time, regdata = data.frame(time =
# 1:12)) with
#
# - adj=no: the design [1, t] at the origin (actuar's hache.origin, always
#   Hachemeister's iterative structure estimator). On these data the
#   between covariance has a correlation of 0.9998 and the iteration has no
#   stable fixed point: it stops at tol = sqrt(eps) after 47 steps, and run
#   on it wanders by about 1e-5 (3,000 steps at tol = 0). Round-off decides
#   the last steps, so these rows are checked to 1e-6;
# - adj=yes;estimator=unbiased and adj=yes;estimator=iterative: the intercept
#   at the collective weights' barycentre (adj.intercept = TRUE), where the
#   credibility matrices are diagonal;
# - adj=no;missing=...: the origin fit with cells missing.
#
# Quantities, with `arg` an index read as digits (1-based):
# - collective (arg j) and coefficient (arg 10 i + j): coefficients in the
#   original basis [1, t], so actuar's adjusted fits are mapped back by its
#   transition matrix;
# - credibility (arg 100 i + 10 j + k): Z_i[j, k] in the basis fitted (the
#   original design, or the orthogonalised one when adjusted; a diagonal is
#   the same whatever the signs of the orthogonal basis, so only diagonals
#   are written for adjusted fits);
# - between (arg 10 j + k, diagonals only when adjusted) and within;
# - premium (arg i): each state's prediction at t = 13.

library(actuar)
data(hachemeister)

out <- "validation/reference/credibility_regression_actuar.csv"
source <- paste0("R actuar ", as.character(packageVersion("actuar")))

rows <- list()
add <- function(case, quantity, arg, value, rel) {
  if (startsWith(case, "adj=no")) rel <- 1e-6
  rows[[length(rows) + 1]] <<- data.frame(
    distribution = "hachemeister", params = case, quantity = quantity, arg = arg,
    expected = sprintf("%.17g", value), abs_tol = 0, rel_tol = rel, source = source)
}

fit <- function(adj, method = "Buhlmann-Gisler", missing = NULL) {
  d <- hachemeister
  for (cell in missing) {
    d[cell[1], 1 + cell[2]] <- NA
    d[cell[1], 13 + cell[2]] <- NA
  }
  cm(~state, d, regformula = ~time, regdata = data.frame(time = 1:12),
     ratios = ratio.1:ratio.12, weights = weight.1:weight.12,
     adj.intercept = adj, method = method)
}

record <- function(case, f, adj) {
  R <- if (adj) f$transition else diag(2)
  coll <- solve(R, f$means[[1]])
  for (j in 1:2) add(case, "collective", j, coll[j], 1e-9)
  for (i in 1:5) {
    b <- solve(R, coef(f$adj.models[[i]]))
    for (j in 1:2) add(case, "coefficient", 10 * i + j, b[j], 1e-9)
    for (j in 1:2) for (k in 1:2) {
      if (adj && j != k) next
      add(case, "credibility", 100 * i + 10 * j + k, f$cred[j, k, i], 1e-9)
    }
  }
  est <- if (is.null(f$iterative)) f$unbiased else f$iterative
  for (j in 1:2) for (k in 1:2) {
    if (adj && j != k) next
    add(case, "between", 10 * j + k, est[[1]][j, k], 1e-9)
  }
  add(case, "within", "", est[[2]], 1e-12)
  p <- predict(f, newdata = data.frame(time = 13))
  for (i in 1:5) add(case, "premium", i, p[i], 1e-10)
}

record("adj=no", fit(FALSE), FALSE)
record("adj=yes;estimator=unbiased", fit(TRUE, "Buhlmann-Gisler"), TRUE)
record("adj=yes;estimator=iterative", fit(TRUE, "iterative"), TRUE)
missing <- list(c(1, 12), c(3, 1), c(3, 2), c(5, 7))
record("adj=no;missing=1.12|3.1|3.2|5.7", fit(FALSE, missing = missing), FALSE)

write.csv(do.call(rbind, rows), out, row.names = FALSE, quote = FALSE)
