suppressMessages(library(prospicio))

# bayes_bf_normalised(), the Bayesian Bornhuetter-Ferguson with a normalised
# pattern (docs/design/reserving-bayes.md, decision 23). The limits, the
# contrast with bayes_bf() and the one-year view are checked in
# validation/tests/reserving_bayes_bf_normalised.rs; these tests check the
# binding on EVW's triangle with its prior ultimates as the exposure.

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
run <- with_warnings(bayes_bf_normalised(tri, "paid", "prior", n_sims = 2000, seed = 3))
fit <- run$value
stopifnot(
  length(run$warnings) == 0,
  S7::S7_inherits(fit, bayes_bf_normalised_fit),
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

# The shares sum to 1, the first age's the largest; one expected ultimate
# per origin.
stopifnot(
  length(fit@pattern) == 10, abs(sum(fit@pattern) - 1) < 1e-9, all(fit@pattern > 0),
  fit@pattern[1] > fit@pattern[10], length(fit@expected_ultimate) == 10
)

d <- fit@diagnostics
stopifnot(
  identical(names(d), c("max_rhat", "min_ess_bulk", "min_ess_tail", "divergences", "warnings")),
  d$divergences == 0, length(d$warnings) == 0
)

s <- summary(fit)
stopifnot(
  identical(names(s), c("parameter", "mean", "sd", "q05", "q50", "q95", "rhat", "ess_bulk",
                        "ess_tail")),
  # Ten alphas and nine betas: the first age's share is the corner.
  nrow(s) == 19, s$parameter[1] == "alpha[1]", s$parameter[11] == "beta[2]",
  s$parameter[19] == "beta[10]",
  identical(s, fit@summary),
  identical(dim(fit@draws), c(4000L, 19L)),
  identical(colnames(fit@draws), s$parameter)
)

long <- as.data.frame(fit)
totals <- totals_frame(fit)
stopifnot(
  identical(names(long), c("origin", "latest", "ultimate", "reserve", "exposure",
                           "prior_ultimate", "mean", "std_dev")),
  nrow(long) == 10, nrow(totals) == 1,
  all(c("exposure", "prior_ultimate", "scale") %in% names(totals)),
  nrow(development_frame(fit)) == 10,
  S7::S7_inherits(segment(fit), bayes_bf_normalised_fit)
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

# The limits of the prior's strength and its reach. A flat prior (cv = Inf)
# is the Bayesian chain ladder (EVW's 6,049,398 at a = 0, and bayes_odp()'s
# within Monte Carlo error); a very tight one reaches the shares
# S / (M + lambda) given the prior ultimates, whose reserve on this triangle
# is 8,987,282 (the classical Bornhuetter-Ferguson's is 7,356,578: the
# pattern follows the ultimates). The prior's level moves the reserve here,
# where bayes_bf()'s does not.
flat <- suppressWarnings(bayes_bf_normalised(tri, "paid", "prior", cv = Inf, development_sd = 100,
                                             n_sims = 2000, seed = 1))
odp <- suppressWarnings(bayes_odp(tri, "paid", n_sims = 2000, seed = 1))
tight <- suppressWarnings(bayes_bf_normalised(tri, "paid", "prior", cv = 0.001,
                                              development_sd = 100, n_sims = 2000, seed = 1))
mid <- suppressWarnings(bayes_bf_normalised(tri, "paid", "prior", cv = 0.1, n_sims = 2000, seed = 1))
near(mean(flat@reserves), mean(odp@reserves), 0.02)
near(mean(flat@reserves), 6049398, 0.015)
# The posterior mean of the shares sits 2.5% below their mode.
near(mean(tight@reserves), 8987282, 0.04)
stopifnot(mean(flat@reserves) < mean(mid@reserves), mean(mid@reserves) < mean(tight@reserves))
level <- function(f, apriori) {
  mean(suppressWarnings(f(tri, "paid", "prior", apriori = apriori, cv = 0.05, n_sims = 2000,
                          seed = 1))@reserves)
}
stopifnot(
  level(bayes_bf_normalised, 1.2) > 1.5 * level(bayes_bf_normalised, 0.6),
  level(bayes_bf, 1.2) < 1.1 * level(bayes_bf, 0.6)
)

# One seed replays the fit; another changes it.
quick <- function(seed) {
  suppressWarnings(bayes_bf_normalised(tri, "paid", "prior", chains = 2, tune = 200, draws = 200,
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
short <- with_warnings(bayes_bf_normalised(tri, "paid", "prior", chains = 2, tune = 50,
                                           draws = 10, n_sims = 100, seed = 1))
stopifnot(
  length(short$warnings) == 1,
  grepl("bayes_bf_normalised: the MCMC diagnostics failed", short$warnings, fixed = TRUE),
  grepl("bulk ESS", short$warnings, fixed = TRUE),
  short$value@reserves@n_sims == 100
)

# The one-year view: the same class as bayes_bf_one_year()'s, with the new
# lifetime fit.
view <- suppressWarnings(bayes_bf_normalised_one_year(
  tri, "paid", "prior", apriori = 0.9, cv = 0.2, tune = 300, draws = 300, n_sims = 1000,
  seed = 3, refit = "reweight_only"))
lifetime <- suppressWarnings(bayes_bf_normalised(tri, "paid", "prior", apriori = 0.9, cv = 0.2,
                                                 tune = 300, draws = 300, n_sims = 1000, seed = 3))
stopifnot(
  S7::S7_inherits(view, bayes_one_year_fit),
  S7::S7_inherits(view@fit, bayes_bf_normalised_fit),
  identical(view@fit@draws, lifetime@draws),
  identical(draw_matrix(view@fit@reserves), draw_matrix(lifetime@reserves)),
  length(view@pareto_k) == 1000, length(view@refitted) == 0,
  variance(view@cdr) < variance(view@fit@reserves),
  all(c("prior_ultimate", "opening_reserve") %in% names(as.data.frame(view)))
)
expect_error_like(bayes_bf_normalised_one_year(tri, "paid", "prior", n_sims = 10,
                                               k_threshold = NaN), "k_threshold")

# Errors.
expect_error_like(bayes_bf_normalised(tri, "paid", "prior", n_sims = 0), "n_sims must be positive")
expect_error_like(bayes_bf_normalised(tri, "paid", "prior", process = "lognormal"),
                  "should be one of")
expect_error_like(bayes_bf_normalised(tri, "paid", "prior", seed = -1),
                  "seed must be a non-negative whole number")
expect_error_like(bayes_bf_normalised(tri, "paid", "prior", apriori = 0, n_sims = 100), "apriori")
expect_error_like(bayes_bf_normalised(tri, "paid", "prior", cv = 0, n_sims = 100), "cv")
expect_error_like(bayes_bf_normalised(tri, "paid", "prior", development_sd = Inf, n_sims = 100),
                  "development_sd")
expect_error_like(bayes_bf_normalised(tri, "paid", "missing"), "missing")
expect_error_like(bayes_bf_normalised(tri, "paid"), "exposure")
# The youngest origin's increments sum to -5: a flat prior has no log-linear
# mean there, an informative one still does, and PSIS-LOO has no density at
# a negative increment.
negative <- triangle(data.frame(year = c(2020, 2020, 2020, 2021, 2021, 2022),
                                age = c(12, 24, 36, 12, 24, 12),
                                paid = c(100, 150, 160, 110, 170, -5), premium = 200),
                     "year", "age", c("paid", "premium"))
expect_error_like(bayes_bf_normalised(negative, "paid", "premium", cv = Inf, n_sims = 100),
                  "origin 2022")
ok <- suppressWarnings(bayes_bf_normalised(negative, "paid", "premium", apriori = 0.8,
                                           n_sims = 100))
stopifnot(ok@reserves@n_sims == 100)
expect_error_like(bayes_loo(ok), "non-negative increments")

cat("prospicio R Bayesian Bornhuetter-Ferguson (normalised pattern) tests passed\n")
