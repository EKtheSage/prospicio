suppressMessages(library(prospicio))

# A growth curve fitted to a benchmark pattern (`benchmark_curve()`,
# docs/design/reserving-v02.md, decision 20). The Rust core is checked
# against scipy's least_squares in validation/tests/reserving_benchmark_curve.rs;
# these tests check the binding's arguments, the same reference values and
# the pattern in the chain ladder and the Bornhuetter-Ferguson method.

near <- function(a, b, rel = 1e-12) {
  a <- unname(a)
  b <- unname(b)
  if (length(a) != length(b) || !all(abs(a - b) <= rel * pmax(abs(b), 1e-300))) {
    stop(sprintf("got %s, want %s (rel %g)", paste(format(a, digits = 17), collapse = " "),
                 paste(format(b, digits = 17), collapse = " "), rel), call. = FALSE)
  }
}

expect_error_like <- function(expr, pattern) {
  msg <- tryCatch({
    expr
    NULL
  }, error = conditionMessage)
  if (is.null(msg)) stop(sprintf("expected an error mentioning %s", shQuote(pattern)), call. = FALSE)
  if (!grepl(pattern, msg, fixed = TRUE)) {
    stop(sprintf("error %s does not mention %s", shQuote(msg), shQuote(pattern)), call. = FALSE)
  }
}

ages <- seq(12, 120, by = 12)
bench <- stats::setNames(c(0.18, 0.47, 0.66, 0.78, 0.855, 0.90, 0.93, 0.951, 0.965, 0.975), ages)
w <- stats::setNames(c(1, 2, 2, 2, 1, 1, 1, 0.5, 0.5, 0.5), ages)

# The reference values of scipy.optimize.least_squares
# (validation/scripts/reserving_benchmark_curve_scipy.py).
fit <- benchmark_curve(bench)
near(fit@omega, 2.09195678319105, 1e-7)
near(fit@theta, 25.5547370205669, 1e-7)
near(fit@sum_of_squares, 0.000775555676218485, 1e-9)
fit <- benchmark_curve(bench, curve = "weibull", max_age = 240, origin_width = 12)
near(fit@omega, 1.00079725545721, 1e-7)
near(fit@theta, 28.350802255576, 1e-7)
near(fit@sum_of_squares, 0.000447194375150825, 1e-9)
stopifnot(fit@origin_width == 12, fit@max_age == 240, fit@curve == "weibull")
fit <- benchmark_curve(bench, weights = w, origin_width = 12)
near(fit@omega, 1.6187953142168, 1e-7)
near(fit@theta, 18.9910413059938, 1e-7)
near(fit@sum_of_squares, 0.00480838215217264, 1e-9)
near(fit@weights, w)
near(fit@residuals, fit@observed - fit@fitted)
stopifnot(identical(names(fit@residuals), as.character(ages)))

# Factors to ultimate, and weights naming only some ages.
fit <- benchmark_curve(bench)
by_cdf <- benchmark_curve(1 / bench, kind = "cdf")
near(by_cdf@omega, fit@omega, 1e-7)
stopifnot(is.null(fit@origin_width), is.infinite(fit@max_age))
partial <- benchmark_curve(bench, weights = c("24" = 2))
near(partial@weights, c(1, 2, rep(1, 8)))

# Values at any age, and the pattern in the chain ladder and the BF.
near(benchmark_cdf(fit, c(12, 30)), 1 / benchmark_percent_developed(fit, c(12, 30)))
rows <- list(c(110, 290, 370, 420, 440, 452), c(95, 300, 390, 425, 447), c(130, 320, 410, 470),
             c(105, 305, 395), c(120, 310), 125)
long <- data.frame(year = rep(2020:2025, lengths(rows)),
                   age = unlist(lapply(lengths(rows), function(n) 12 * seq_len(n))),
                   paid = unlist(rows), premium = 800)
tri <- triangle(long, "year", "age", c("paid", "premium"))
clark <- clark_ldf(tri, "paid", curve = "weibull")
from_clark <- benchmark_curve(stats::setNames(growth(clark, 12 * 1:8), 12 * 1:8),
                              curve = "weibull", origin_width = 12)
near(from_clark@omega, clark@omega, 1e-7)
near(from_clark@theta, clark@theta, 1e-7)
sel <- benchmark_selected(from_clark, 12 * 1:6)
stopifnot(sel$ldf_kind == "percent_developed", sel$ldf_weight == 1,
          identical(names(sel$ldf), as.character(12 * 1:6)))
cl <- do.call(chain_ladder, c(list(tri, "paid"), sel))
near(cl@ultimate, clark@ultimate, 1e-7)
bf <- do.call(bornhuetter_ferguson, c(list(tri, "paid", "premium", apriori = 0.6), sel))
latest <- vapply(rows, function(r) r[length(r)], 0)
near(bf@ultimate, latest + 0.6 * 800 *
       (1 - benchmark_percent_developed(from_clark, 12 * rev(seq_along(rows)))), 1e-9)

# Errors.
expect_error_like(benchmark_curve(unname(bench)), "named numeric vector")
expect_error_like(benchmark_curve(bench, kind = "ldf"), "should be one of")
expect_error_like(benchmark_curve(bench, weights = c("18" = 1)), "not an age of the pattern")
expect_error_like(benchmark_curve(bench["12"]), "at least two ages")
expect_error_like(benchmark_curve(bench, max_age = 60), "max_age")
expect_error_like(benchmark_selected(fit, c(12, 12)), "given twice")
expect_error_like(benchmark_selected(fit, 12.5), "whole")
expect_error_like(benchmark_cdf(clark, 12), "benchmark_curve_fit")
cat("prospicio R benchmark curve tests passed\n")
