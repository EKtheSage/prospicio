suppressMessages(library(prospicio))

# bayes_odp(), the Bayesian ODP chain ladder (docs/design/reserving-bayes.md).
# The parity with the same model in Stan is validation/tests/reserving_bayes.rs;
# these tests check the binding.

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
read_long <- function(name) {
  utils::read.csv(file.path("validation", "data", paste0(name, ".csv")), comment.char = "#")
}
genins <- triangle(read_long("genins"), "origin", "development", "value")

# A fit with the default chains (4 x 1,000 draws), which pass every
# diagnostic on GenIns, so no warning.
run <- with_warnings(bayes_odp(genins, n_sims = 2000, seed = 3))
fit <- run$value
stopifnot(
  length(run$warnings) == 0,
  S7::S7_inherits(fit, bayes_odp_fit),
  S7::S7_inherits(fit@chain_ladder, chain_ladder_fit),
  identical(fit@origins, as.character(2001:2010)),
  identical(fit@development, seq(12L, 120L, by = 12L))
)
# The plug-in scale is the ODP bootstrap's.
near(fit@scale, odp_bootstrap(genins, n_sims = 10)@scale, 1e-12)
near(fit@scale, 52601.3615, 1e-8)

r <- fit@reserves
stopifnot(
  S7::S7_inherits(r, predictive_distribution),
  r@n_sims == 2000,
  identical(names(r@keys), "origin"), identical(r@keys$origin, fit@origins)
)
# Near the chain ladder's total (+0.7% with the defaults).
near(mean(r), fit@chain_ladder@total_reserve, 0.05)

d <- fit@diagnostics
stopifnot(
  identical(names(d), c("max_rhat", "min_ess_bulk", "min_ess_tail", "divergences", "warnings")),
  d$max_rhat < 1.01, d$min_ess_bulk >= 400, d$divergences == 0, length(d$warnings) == 0
)

s <- summary(fit)
stopifnot(
  identical(names(s), c("parameter", "mean", "sd", "q05", "q50", "q95", "rhat", "ess_bulk",
                        "ess_tail")),
  nrow(s) == 19, s$parameter[1] == "c", s$parameter[19] == "b[10]",
  identical(s, fit@summary),
  abs(s$mean[1] - 12.49) < 0.1,
  identical(dim(fit@draws), c(4000L, 19L)),
  identical(colnames(fit@draws), s$parameter)
)

long <- as.data.frame(fit)
totals <- totals_frame(fit)
stopifnot(
  identical(names(long), c("origin", "latest", "ultimate", "reserve", "mean", "std_dev")),
  nrow(long) == 10, nrow(totals) == 1, "scale" %in% names(totals),
  nrow(development_frame(fit)) == 10,
  S7::S7_inherits(segment(fit), bayes_odp_fit)
)
near(sum(long$mean), mean(r), 1e-9)
near(totals$mean, mean(r), 1e-9)
expect_error_like(segment(fit, lob = "auto"), "lob")

loo <- bayes_loo(fit)
stopifnot(
  identical(names(loo$estimates), c("elpd", "se", "p", "ic")),
  length(loo$pointwise) == 55, length(loo$pareto_k) == 55,
  abs(loo$estimates[["elpd"]] + 750) < 10
)
invisible(capture.output(print(fit)))

# One seed replays the fit; another changes it.
quick <- function(seed) {
  suppressWarnings(bayes_odp(genins, chains = 2, tune = 200, draws = 200, n_sims = 500,
                             seed = seed))
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
short <- with_warnings(bayes_odp(genins, chains = 2, tune = 50, draws = 10, n_sims = 100,
                                 seed = 1))
stopifnot(
  length(short$warnings) == 1,
  grepl("MCMC diagnostics failed", short$warnings, fixed = TRUE),
  grepl("bulk ESS", short$warnings, fixed = TRUE),
  length(short$value@diagnostics$warnings) > 0,
  short$value@reserves@n_sims == 100
)

# Errors.
expect_error_like(bayes_odp(genins, n_sims = 0), "n_sims must be positive")
expect_error_like(bayes_odp(genins, process = "lognormal"), "should be one of")
expect_error_like(bayes_odp(genins, seed = -1), "seed must be a non-negative whole number")
expect_error_like(bayes_odp(genins, origin_sd = -1, n_sims = 100), "origin_sd")
expect_error_like(bayes_odp(genins, "missing"), "missing")
negative <- triangle(data.frame(year = c(2020, 2020, 2020, 2021, 2021, 2022),
                                age = c(12, 24, 36, 12, 24, 12),
                                paid = c(100, 150, 160, 110, 170, -5)),
                     "year", "age", "paid")
expect_error_like(bayes_odp(negative, n_sims = 100), "origin 2022")
two <- triangle(data.frame(lob = rep(c("auto", "home"), each = 3), year = c(2020, 2020, 2021),
                           age = c(12, 24, 12), paid = c(100, 150, 200, 10, 20, 30)),
                "year", "age", "paid", keys = "lob")
msg <- tryCatch({
  bayes_odp(two, n_sims = 100)
  NULL
}, error = conditionMessage)
stopifnot(!is.null(msg))
# RAA has a negative increment, where the PSIS-LOO density does not exist.
raa <- triangle(read_long("raa"), "origin", "development", "value")
raa_fit <- suppressWarnings(bayes_odp(raa, chains = 2, tune = 200, draws = 200, n_sims = 100))
expect_error_like(bayes_loo(raa_fit), "non-negative increments")

cat("prospicio R Bayesian reserving tests passed\n")
