# Accumulated local effects (#547): references for python/tests/test_models.py
# and R/prospicio/tests/test-models.R from ALEPlot 1.1 (Apley and Zhu; archived
# on CRAN, installed from its archive with its dependency yaImpute).
#
# A seeded frame of 300 rows, written to validation/data/explain_frame.csv,
# with correlated numerics x1 and x2, a numeric x3 and a factor g of four
# levels whose mix depends on x1, and a fixed prediction function with
# interactions (explain_f below, the same in the tests):
#   exp(0.3 x1) + x1 x2 + c(a = 0, b = 0.5, c = -0.3, d = 1)[g] + 0.5 x3^2
#     + 0.4 x3 (g == "d")
# ALE of x1 (K = 10), x3 (K = 40) and g (ALEPlot's own order of levels),
# written to validation/reference/ale_aleplot.csv.
#
# Run from the repository root:
#   Rscript validation/scripts/r_ale.R
suppressPackageStartupMessages(library(ALEPlot))

set.seed(547)
n <- 300
x1 <- round(rnorm(n), 4)
x2 <- round(0.6 * x1 + 0.8 * rnorm(n), 4)
x3 <- round(runif(n, -1, 2), 4)
g <- cut(x1 + rnorm(n, sd = 0.7), c(-Inf, -0.7, 0, 0.7, Inf), labels = c("a", "b", "c", "d"))
frame <- data.frame(x1 = x1, x2 = x2, x3 = x3, g = g)
write.csv(frame, "validation/data/explain_frame.csv", row.names = FALSE, quote = FALSE)

explain_f <- function(X.model, newdata) {
  with(newdata, exp(0.3 * x1) + x1 * x2 + c(a = 0, b = 0.5, c = -0.3, d = 1)[as.character(g)] +
         0.5 * x3^2 + 0.4 * x3 * (as.character(g) == "d"))
}

rows <- list()
pdf(NULL)
for (case in list(list("x1", 10), list("x3", 40), list("g", NA))) {
  j <- match(case[[1]], names(frame))
  a <- if (is.na(case[[2]])) ALEPlot(frame, NULL, explain_f, J = j) else
    ALEPlot(frame, NULL, explain_f, J = j, K = case[[2]])
  for (i in seq_along(a$x.values)) {
    rows[[length(rows) + 1]] <- data.frame(
      feature = case[[1]], bins = if (is.na(case[[2]])) "" else case[[2]], index = i,
      value = as.character(a$x.values[i]), effect = sprintf("%.15g", a$f.values[i]),
      source = "ALEPlot 1.1"
    )
  }
}
out <- do.call(rbind, rows)
con <- file("validation/reference/ale_aleplot.csv", "w")
writeLines(sprintf("# %s, ALEPlot %s", R.version.string, packageVersion("ALEPlot")), con)
write.csv(out, con, row.names = FALSE, quote = FALSE)
close(con)
cat(nrow(out), "rows\n")
