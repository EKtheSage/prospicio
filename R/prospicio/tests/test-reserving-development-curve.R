suppressMessages(library(prospicio))
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# Development curves over every age (`development_curve()`,
# docs/design/reserving-v02.md, decision 31, issue #556). The Rust core is
# checked against R's lm and nls in validation/tests/reserving_curves.rs;
# these tests check the binding's arguments, the same reference rows and
# the fitted pattern in the chain ladder.



# Tests run from the repository root (CI and `cargo xtask r`).
read_long <- function(name) utils::read.csv(validation("data", paste0(name, ".csv")), comment.char = "#")
load_triangle <- function(name) triangle(read_long(name), "origin", "development", "value")
tris <- list(raa = load_triangle("raa"), genins = load_triangle("genins"))
shapes <- c("exponential", "inverse_power", "hoerl", "sherman", "loglogistic", "weibull")

# Every row of R's lm and nls (validation/scripts/reserving_curves_r.R).
ref <- utils::read.csv(validation("reference", "reserving_curves_r.csv"), comment.char = "#",
                       stringsAsFactors = FALSE)
fits <- list()
for (i in seq_len(nrow(ref))) {
  row <- ref[i, ]
  key <- paste(row$dataset, row$method)
  if (is.null(fits[[key]])) {
    fits[[key]] <- development_curve(tris[[row$dataset]], shape = row$method)
  }
  fit <- fits[[key]]
  q <- row$quantity
  got <- if (startsWith(q, "param_")) {
    fit@parameters[[sub("param_", "", q)]]
  } else if (q == "fitted") {
    fit@fitted[[row$arg + 1]]
  } else if (q == "tail") {
    development_curve_tail(fit)
  } else {
    S7::prop(fit, q)
  }
  near(got, row$expected, row$rel_tol)
}

# The fit as the chain ladder's pattern and tail.
raa <- tris$raa
fit <- development_curve(raa, shape = "sherman", keep_observed_until = 48)
stopifnot(
  fit@shape == "sherman", fit@keep_observed_until == 48, fit@attachment == 3,
  identical(names(fit@parameters), c("a", "b", "c")),
  identical(names(fit@fitted), as.character(seq(12, 108, by = 12))),
  all(fit@in_fit), fit@n_points == 9, fit@n_parameters == 3,
  identical(unname(fit@ldf[1:3]), unname(fit@observed[1:3])),
  identical(unname(fit@ldf[4:9]), unname(fit@fitted[4:9]))
)
near(fit@observed, chain_ladder(raa)@ldf, 1e-15)
near(fit@sum_of_squares, sum(fit@residuals^2), 1e-12)
sel <- development_curve_selected(fit)
stopifnot(identical(names(sel$ldf), as.character(seq(48, 108, by = 12))),
          sel$ldf_kind == "ldf", sel$ldf_weight == 1)
cl <- do.call(chain_ladder, c(list(raa, tail = development_curve_tail(fit)), sel))
near(cl@ldf, fit@ldf, 1e-15)
near(cl@tail, development_curve_tail(fit), 1e-12)
mack_curve <- do.call(mack, c(list(raa), sel))
stopifnot(mack_curve@total_parameter_risk < mack(raa)@total_parameter_risk)
invisible(capture.output(print(fit)))

# One row per shape, on the same factors.
table <- development_curve_compare(tris$genins)
stopifnot(
  is.data.frame(table), identical(table$shape, shapes),
  identical(table$n_parameters, c(2, 2, 3, 3, 2, 2)), all(table$n_points == 9),
  table$sum_of_squares[3] < min(table$sum_of_squares[1:2])
)
some <- development_curve_compare(tris$genins, shapes = c("hoerl", "sherman"), fit_period = c(24, NA))
stopifnot(identical(some$shape, c("hoerl", "sherman")), all(some$n_points == 8))

# Settings and errors.
fit <- development_curve(raa, shape = "hoerl", fit_period = c(24, 96), average = "simple")
stopifnot(identical(unname(fit@in_fit), c(FALSE, rep(TRUE, 6), FALSE, FALSE)),
          identical(fit@fit_period, c(24, 96)), is.null(fit@keep_observed_until))
expect_error_like(development_curve(raa, shape = "cubic"), "should be one of")
expect_error_like(development_curve(raa, keep_observed_until = 120), "past the last factor")
expect_error_like(development_curve(raa, shape = "sherman", fit_period = c(84, NA)), "four factors")
expect_error_like(development_curve_compare(raa, shapes = "cubic"), "shapes must be")

# tail_curve() takes every shape, with the curve's own tail.
for (s in shapes) {
  stopifnot(tail_curve(s)@curve == s)
  near(chain_ladder(raa, tail = tail_curve(s))@tail,
       development_curve_tail(development_curve(raa, shape = s)), 1e-12)
}

cat("development curve tests passed\n")
