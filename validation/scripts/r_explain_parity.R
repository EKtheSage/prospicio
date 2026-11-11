# Cross-language parity of the explanation tools (#837, step 1): one frame,
# one prediction function and one reference that python/tests/test_models.py
# and R/prospicio/tests/test-models.R both read.
#
# The frame (validation/data/explain_factor_frame.csv) has a factor h of five
# levels, 20 rows each, a categorical z whose mix differs by level, and a
# numeric x with the same 20 values in every level. ALEPlot orders h's levels
# by classical scaling of the levels' distances, here half the L1 distance of
# z's conditional frequencies (x adds 0). The counts were chosen so that the
# doubly centred matrix -J D^2 J / 2 has a negative eigenvalue (-0.0675)
# larger in magnitude than its second and third positive ones (0.1765,
# 0.1303, 0.0287): an ordering that took the eigenvalue of largest magnitude,
# or stopped before the leading vector separated, would order h otherwise
# than ALEPlot's cmdscale.
#
# The prediction function (explain_g below, the same in the tests):
#   0.5 x + c(p = 0, q = 0.4, r = -0.2, s = 0.9, t = 0.3)[h]
#     + c(u = 0, v = 0.2, w = -0.1, y = 0.5)[z] + 0.3 x (h == "s")
# References: ALE of h and of x (K = 10) from ALEPlot 1.1; partial dependence
# and ICE of x at -0.5, 0, 0.5, 1 from the function itself (rows 1, 37 and 81
# for ICE).
#
# ALEPlot 1.1 is archived on CRAN; install it from the archive with its
# dependency yaImpute. Run from the repository root:
#   Rscript validation/scripts/r_explain_parity.R
suppressPackageStartupMessages(library(ALEPlot))

levels_h <- c("p", "q", "r", "s", "t")
counts <- rbind(c(6, 3, 6, 5), c(6, 3, 1, 10), c(0, 10, 5, 5), c(5, 10, 1, 4), c(0, 9, 1, 10))
cats <- c("u", "v", "w", "y")
x_values <- round(seq(-1, 0.9, by = 0.1), 1)
rows <- lapply(seq_along(levels_h), function(i) {
  data.frame(h = levels_h[i], z = rep(cats, counts[i, ]), x = x_values)
})
frame <- do.call(rbind, rows)
frame$h <- factor(frame$h, levels = levels_h)
frame$z <- factor(frame$z, levels = cats)
frame <- frame[, c("x", "z", "h")]
write.csv(frame, "validation/data/explain_factor_frame.csv", row.names = FALSE, quote = FALSE)

explain_g <- function(X.model, newdata) {
  h <- as.character(newdata$h)
  z <- as.character(newdata$z)
  0.5 * newdata$x + c(p = 0, q = 0.4, r = -0.2, s = 0.9, t = 0.3)[h] +
    c(u = 0, v = 0.2, w = -0.1, y = 0.5)[z] + 0.3 * newdata$x * (h == "s")
}

out <- list()
add <- function(tool, feature, index, value, expected, source) {
  out[[length(out) + 1]] <<- data.frame(
    tool = tool, feature = feature, index = index, value = value,
    expected = sprintf("%.15g", unname(expected)), source = source
  )
}
pdf(NULL)
a <- ALEPlot(frame, NULL, explain_g, J = 3)
for (i in seq_along(a$x.values)) add("ale", "h", i, as.character(a$x.values[i]), a$f.values[i], "ALEPlot 1.1")
a <- ALEPlot(frame, NULL, explain_g, J = 1, K = 10)
for (i in seq_along(a$x.values)) add("ale", "x", i, sprintf("%.15g", a$x.values[i]), a$f.values[i], "ALEPlot 1.1")
grid <- c(-0.5, 0, 0.5, 1)
for (k in seq_along(grid)) {
  set <- frame
  set$x <- grid[k]
  pred <- explain_g(NULL, set)
  add("pd", "x", k, sprintf("%.15g", grid[k]), mean(pred), "the function")
  for (r in c(1, 37, 81)) add(paste0("ice_row", r), "x", k, sprintf("%.15g", grid[k]), pred[r], "the function")
}
res <- do.call(rbind, out)
con <- file("validation/reference/explain_parity.csv", "w")
writeLines(sprintf("# %s, ALEPlot %s", R.version.string, packageVersion("ALEPlot")), con)
write.csv(res, con, row.names = FALSE, quote = FALSE)
close(con)
cat(nrow(res), "rows\n")
