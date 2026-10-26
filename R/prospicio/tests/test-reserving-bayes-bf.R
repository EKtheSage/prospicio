suppressMessages(library(prospicio))

# bayes_bf(), the Bayesian Bornhuetter-Ferguson (docs/design/reserving-bayes.md,
# decision 9). The parity with the same model in Stan and with England,
# Verrall and Wuthrich (2012) is validation/tests/reserving_bayes_bf.rs; these
# tests check the binding on EVW's triangle with its prior ultimates as the
# exposure.

near <- function(a, b, rel) {
  if (!all(abs(a - b) <= rel * pmax(abs(b), 1e-300))) {
    stop(sprintf("got %s, want %s (rel %g)", format(a, digits = 17), format(b, digits = 17), rel),
         call. = FALSE)
  }
}

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

# The warnings `expr` raises, and its value.
with_warnings <- function(expr) {
  found <- character()
  value <- withCallingHandlers(expr, warning = function(w) {
    found <<- c(found, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = found)
}

# Tests run from the repository root (CI and `cargo xtask r`).
evw <- utils::read.csv(file.path("validation", "data", "evw2012.csv"), comment.char = "#")
tri <- triangle(evw, "origin", "development", c("paid", "prior"))

# A fit with the default chains (4 x 1,000 draws), which pass every
# diagnostic on EVW's triangle, so no warning.
run <- with_warnings(bayes_bf(tri, "paid", "prior", n_sims = 2000, seed = 3))
fit <- run$value
stopifnot(
  length(run$warnings) == 0,
  S7::S7_inherits(fit, bayes_bf_fit),
  S7::S7_inherits(fit@chain_ladder, chain_ladder_fit),
  identical(fit@origins, as.character(2001:2010)),
  identical(fit@development, seq(12L, 120L, by = 12L)),
  fit@exposure[1] == 11653101,
  identical(fit@prior_ultimate, fit@exposure)
)
# EVW (2012) print phi = 14,714.
near(fit@scale, 14714, 1e-3)

r <- fit@reserves
stopifnot(
  S7::S7_inherits(r, predictive_distribution),
  r@n_sims == 2000,
  identical(names(r@keys), "origin"), identical(r@keys$origin, fit@origins)
)
# EVW print 6,145,526 at a = 100 (cv 0.1). Over seeds 0 to 29 the mean sat
# +0.26% above it with a seed-to-seed sd of 0.15%: 1.5% is the offset plus
# 8 sds.
near(mean(r), 6145526, 0.015)

d <- fit@diagnostics
stopifnot(
  identical(names(d), c("max_rhat", "min_ess_bulk", "min_ess_tail", "divergences", "warnings")),
  d$divergences == 0, length(d$warnings) == 0
)

s <- summary(fit)
stopifnot(
  identical(names(s), c("parameter", "mean", "sd", "q05", "q50", "q95", "rhat", "ess_bulk",
                        "ess_tail")),
  nrow(s) == 20, s$parameter[1] == "alpha[1]", s$parameter[11] == "beta[1]",
  s$parameter[20] == "beta[10]",
  identical(s, fit@summary),
  identical(dim(fit@draws), c(4000L, 20L)),
  identical(colnames(fit@draws), s$parameter)
)

long <- as.data.frame(fit)
totals <- totals_frame(fit)
stopifnot(
  identical(names(long), c("origin", "latest", "ultimate", "reserve", "exposure",
                           "prior_ultimate", "ulr", "mean", "std_dev")),
  identical(long$ulr, long$ultimate / long$exposure),
  nrow(long) == 10, nrow(totals) == 1,
  all(c("exposure", "prior_ultimate", "scale") %in% names(totals)),
  nrow(development_frame(fit)) == 10,
  S7::S7_inherits(segment(fit), bayes_bf_fit)
)
near(sum(long$mean), mean(r), 1e-9)
near(totals$mean, mean(r), 1e-9)
expect_error_like(segment(fit, lob = "auto"), "lob")

loo <- bayes_loo(fit)
stopifnot(
  identical(names(loo$estimates), c("elpd", "se", "p", "ic")),
  length(loo$pointwise) == 55, length(loo$pareto_k) == 55,
  is.finite(loo$estimates[["elpd"]])
)
invisible(capture.output(print(fit)))

# The two limits of the prior's strength. A flat prior (cv = Inf) is the
# Bayesian chain ladder (EVW's 6,049,398 at a = 0, and bayes_odp()'s within
# Monte Carlo error); a very tight one reaches EVW's closed-form limit,
# 6,644,047. Over seeds 0 to 29 each mean sat within 0.7% of its target,
# with sds of 0.15% to 0.23%.
flat <- suppressWarnings(bayes_bf(tri, "paid", "prior", cv = Inf, development_sd = 100,
                                  n_sims = 2000, seed = 1))
odp <- suppressWarnings(bayes_odp(tri, "paid", n_sims = 2000, seed = 1))
tight <- suppressWarnings(bayes_bf(tri, "paid", "prior", cv = 0.001, n_sims = 2000, seed = 1))
mid <- suppressWarnings(bayes_bf(tri, "paid", "prior", cv = 0.1, n_sims = 2000, seed = 1))
near(mean(flat@reserves), mean(odp@reserves), 0.02)
near(mean(flat@reserves), 6049398, 0.015)
near(mean(tight@reserves), 6644047, 0.015)
stopifnot(mean(flat@reserves) < mean(mid@reserves), mean(mid@reserves) < mean(tight@reserves))

# One seed replays the fit; another changes it.
quick <- function(seed) {
  suppressWarnings(bayes_bf(tri, "paid", "prior", chains = 2, tune = 200, draws = 200,
                            n_sims = 500, seed = seed))
}
a <- quick(7)
b <- quick(7)
c8 <- quick(8)
stopifnot(
  identical(a@draws, b@draws),
  identical(draw_matrix(a@reserves), draw_matrix(b@reserves)),
  !identical(a@draws, c8@draws),
  !identical(draw_matrix(a@reserves), draw_matrix(c8@reserves))
)

# Failed diagnostics warn but return the fit: 2 x 10 kept draws cannot reach
# a bulk ESS of 200 (100 per chain).
short <- with_warnings(bayes_bf(tri, "paid", "prior", chains = 2, tune = 50, draws = 10,
                                n_sims = 100, seed = 1))
stopifnot(
  length(short$warnings) == 1,
  grepl("bayes_bf: the MCMC diagnostics failed", short$warnings, fixed = TRUE),
  grepl("bulk ESS", short$warnings, fixed = TRUE),
  short$value@reserves@n_sims == 100
)

# Errors.
expect_error_like(bayes_bf(tri, "paid", "prior", n_sims = 0), "n_sims must be positive")
expect_error_like(bayes_bf(tri, "paid", "prior", process = "lognormal"), "should be one of")
expect_error_like(bayes_bf(tri, "paid", "prior", seed = -1),
                  "seed must be a non-negative whole number")
expect_error_like(bayes_bf(tri, "paid", "prior", apriori = 0, n_sims = 100), "apriori")
expect_error_like(bayes_bf(tri, "paid", "prior", cv = 0, n_sims = 100), "cv")
expect_error_like(bayes_bf(tri, "paid", "prior", development_sd = Inf, n_sims = 100),
                  "development_sd")
expect_error_like(bayes_bf(tri, "paid", "missing"), "missing")
expect_error_like(bayes_bf(tri, "paid"), "exposure")
# The youngest origin's increments sum to -5: a flat prior has no log-linear
# mean there, an informative one still does, and PSIS-LOO has no density at
# a negative increment.
negative <- triangle(data.frame(year = c(2020, 2020, 2020, 2021, 2021, 2022),
                                age = c(12, 24, 36, 12, 24, 12),
                                paid = c(100, 150, 160, 110, 170, -5), premium = 200),
                     "year", "age", c("paid", "premium"))
expect_error_like(bayes_bf(negative, "paid", "premium", cv = Inf, n_sims = 100), "origin 2022")
ok <- suppressWarnings(bayes_bf(negative, "paid", "premium", apriori = 0.8, n_sims = 100))
stopifnot(ok@reserves@n_sims == 100)
expect_error_like(bayes_loo(ok), "non-negative increments")

cat("prospicio R Bayesian Bornhuetter-Ferguson tests passed\n")
