suppressMessages(library(prospicio))

# bayes_new_year(), the posterior predictive of a new underwriting year
# (docs/design/reserving-bayes.md, decision 27, issue #598). The exact
# distributions are checked in validation/tests/reserving_bayes_new_year.rs;
# these tests check the binding on Meyers' group 353 Commercial Auto triangle.

expect_error_like <- function(expr, pattern) {
  msg <- tryCatch({
    expr
    NULL
  }, error = conditionMessage)
  if (is.null(msg)) stop("expected an error", call. = FALSE)
  if (!grepl(pattern, msg, fixed = TRUE)) {
    stop(sprintf("error %s does not mention %s", shQuote(msg), shQuote(pattern)), call. = FALSE)
  }
}

# Tests run from the repository root (CI and `cargo xtask r`).
rows <- utils::read.csv(file.path("validation", "data", "clrd_comauto_353.csv"),
                        comment.char = "#")
rows <- rows[rows$origin - 1988 + rows$development %/% 12 <= 10, ]
tri <- triangle(rows, "origin", "development", c("paid", "incurred", "premium"))

draws_of <- function(pd) as.vector(draw_matrix(pd))

# Without a common level the new year is the prior, Gamma(1 / cv^2, apriori).
fit <- bayes_bf_normalised(tri, "paid", "premium", apriori = 0.7, cv = 0.25, process = "none",
                           n_sims = 200, seed = 1)
year <- bayes_new_year(fit, premium = 5000, n_sims = 40000, seed = 3)
lr <- draws_of(year@loss_ratio)
stopifnot(
  S7::S7_inherits(year, new_year_fit),
  identical(year@origin, "1998"),
  year@premium == 5000,
  length(lr) == 40000,
  abs(mean(lr) / 0.7 - 1) < 0.005,
  abs(sd(lr) / (0.7 * 0.25) - 1) < 0.02,
  isTRUE(all.equal(draws_of(year@ultimate), 5000 * lr, tolerance = 1e-12)),
  identical(draws_of(bayes_new_year(fit, 5000, n_sims = 40000, seed = 3)@loss_ratio), lr),
  !identical(draws_of(bayes_new_year(fit, 5000, n_sims = 40000, seed = 4)@loss_ratio), lr)
)

# The patterns pair up with the loss ratio and give the monthly emergence.
p <- pattern_draws(year)
m <- monthly_pattern_draws(year, 120, monotone = "running_max")
stopifnot(nrow(p) == 40000, all(p[, ncol(p)] == 1), dim(m) == c(40000, 120), all(m[, 120] == 1))

# Meyers: the exchangeable level follows the triangle, Meyers' own prior does not.
mfit <- meyers(tri, "paid", "premium", n_sims = 200, seed = 1)
ex <- bayes_new_year(mfit, 5000, n_sims = 20000, seed = 2)
pr <- bayes_new_year(mfit, 5000, n_sims = 20000, seed = 2, level = "prior")
stopifnot(
  identical(ex@origin, "1998"),
  sd(log(draws_of(ex@loss_ratio))) < 0.5,
  sd(log(draws_of(pr@loss_ratio))) > 3
)

# Errors.
flat <- bayes_bf_normalised(tri, "paid", "premium", cv = Inf, n_sims = 200, seed = 1)
expect_error_like(bayes_new_year(flat, 5000), "finite cv")
expect_error_like(bayes_new_year(fit, 0), "premium")
expect_error_like(bayes_new_year(tri, 5000), "bayes_bf_normalised_fit or a meyers_fit")
