suppressMessages(library(prospicio))

# diagonal_backtest() and the model specs (docs/design/reserving-bayes.md,
# decision 21). The scoring itself is tested in Rust (backtest.rs); these
# tests check the binding: the tables, the errors and the diagnostics.

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
read_long <- function(name) {
  utils::read.csv(file.path("validation", "data", paste0(name, ".csv")), comment.char = "#")
}
raa <- triangle(read_long("raa"), "origin", "development", "value")

metrics <- c("crps", "coverage", "actual_vs_expected", "total_crps", "mean_log_density",
             "total_percentile")

# The ODP GLM on RAA, two held-out diagonals.
bt <- diagonal_backtest(raa, "value", list(odp_glm_spec()), n_diagonals = 2, n_sims = 400,
                        seed = 3)
stopifnot(
  S7::S7_inherits(bt, backtest),
  identical(bt@models, "odp"),
  identical(bt@metrics, metrics),
  bt@n_splits == 2L,
  identical(bt@diagonals, c(8L, 9L)),
  identical(names(bt@scores), c("model", "diagonal", "metric", "value")),
  nrow(bt@scores) == 2 * length(metrics),
  identical(bt@scores$metric, rep(metrics, 2)),
  identical(bt@scores$diagonal, rep(c(8L, 9L), each = length(metrics))),
  identical(names(bt@mean), c("model", metrics)),
  nrow(bt@mean) == 1,
  length(bt@excluded) == 2, all(bt@excluded >= 0),
  length(bt@scored_rows) == 2,
  identical(lengths(bt@scored_rows), c(7L, 8L)),
  identical(names(bt@log_densities), "odp"),
  length(bt@log_densities$odp) == sum(lengths(bt@scored_rows)),
  all(is.finite(bt@scores$value)),
  identical(names(bt@diagnostics),
            c("model", "diagonal", "max_rhat", "min_ess_bulk", "min_ess_tail", "divergences",
              "n_warnings")),
  nrow(bt@diagnostics) == 2,
  all(is.na(bt@diagnostics[, -(1:2)])),
  identical(names(bt@warnings), c("model", "diagonal", "message")),
  nrow(bt@warnings) == 0
)
# The mean table is the mean of the scores over the diagonals.
for (m in metrics) {
  near <- mean(bt@scores$value[bt@scores$metric == m])
  stopifnot(abs(bt@mean[[m]] - near) <= 1e-12 * max(1, abs(near)))
}
# Coverage and the percentile of the total are shares.
shares <- bt@scores$value[bt@scores$metric %in% c("coverage", "total_percentile")]
stopifnot(all(shares >= 0 & shares <= 1))
stopifnot(S7::S7_inherits(print(bt), backtest))

# The same seed gives the same scores, another seed other draws.
again <- diagonal_backtest(raa, "value", list(odp_glm_spec()), n_diagonals = 2, n_sims = 400,
                           seed = 3)
other <- diagonal_backtest(raa, "value", list(odp_glm_spec()), n_diagonals = 2, n_sims = 400,
                           seed = 4)
stopifnot(identical(again@scores, bt@scores), !identical(other@scores$value, bt@scores$value))

# Default names, and two models side by side: the GLM and a Bayesian ODP at tiny settings.
small <- bayes_odp_spec(chains = 1, tune = 100, draws = 100)
two <- diagonal_backtest(raa, "value", list(odp_glm_spec(), small), n_diagonals = 1,
                         n_sims = 100, seed = 1)
stopifnot(
  identical(two@models, c("odp", "bayes_odp")),
  nrow(two@scores) == 2 * length(metrics),
  nrow(two@diagnostics) == 2,
  identical(two@diagnostics$model, c("odp", "bayes_odp")),
  is.na(two@diagnostics$max_rhat[1]),
  is.finite(two@diagnostics$max_rhat[2]),
  two@diagnostics$divergences[2] >= 0,
  is.numeric(two@diagnostics$n_warnings),
  all(is.finite(two@scores$value[two@scores$model == "bayes_odp"])),
  length(two@log_densities$bayes_odp) == length(two@log_densities$odp)
)
# One chain of 100 draws cannot pass the ESS thresholds: the split is scored all the same,
# and the messages are reported, not raised.
stopifnot(
  two@diagnostics$n_warnings[2] > 0,
  nrow(two@warnings) == two@diagnostics$n_warnings[2],
  all(two@warnings$model == "bayes_odp"),
  all(nzchar(two@warnings$message))
)

# Meyers' model with a premium from a column of the triangle, and as a vector.
long <- read_long("clrd_comauto_353")
long <- long[long$origin - 1988 + long$development %/% 12 <= 10, ]
tri <- triangle(long, "origin", "development", c("paid", "premium"))
crc <- meyers_spec(chains = 1, tune = 100, draws = 100)
mb <- diagonal_backtest(tri, "paid", list(crc), n_diagonals = 1, premium = "premium",
                        n_sims = 100, seed = 2)
stopifnot(
  identical(mb@models, "meyers_crc"),
  all(is.finite(mb@scores$value)),
  is.finite(mb@diagnostics$max_rhat),
  nrow(mb@warnings) == mb@diagnostics$n_warnings
)
premium <- latest_diagonal(tri)[1, "premium", ]
mv <- diagonal_backtest(tri, "paid", list(a = crc), n_diagonals = 1, premium = unname(premium),
                        n_sims = 100, seed = 2)
stopifnot(identical(mv@models, "a"), identical(mv@scores$value, mb@scores$value))

# The specs check their arguments as bayes_odp() and meyers() do.
expect_error_like(bayes_odp_spec(chains = "a"), "chains must be a single number")
expect_error_like(bayes_odp_spec(process = "poisson"), "should be one of")
expect_error_like(meyers_spec(logelr_sd = c(1, 2)), "logelr_sd must be a single number")
expect_error_like(meyers_spec(model = "x"), "should be one of")
stopifnot(S7::S7_inherits(bayes_odp_spec(), bayes_odp_spec),
          bayes_odp_spec()@draws == 1000, meyers_spec()@draws == 2000)
print(bayes_odp_spec())
print(meyers_spec(model = "csr"))
print(odp_glm_spec())

# Errors.
expect_error_like(diagonal_backtest(raa, "value", list(), n_diagonals = 1), "non-empty list")
expect_error_like(diagonal_backtest(raa, "value", list(1), n_diagonals = 1), "every entry of models")
expect_error_like(diagonal_backtest(raa, "value", list(a = odp_glm_spec(), "x"), n_diagonals = 1),
                  "every entry of models")
expect_error_like(diagonal_backtest(raa, "value", odp_glm_spec(), n_diagonals = 1), "non-empty list")
expect_error_like(diagonal_backtest(raa, "value", list(a = odp_glm_spec(), a = odp_glm_spec()),
                                    n_diagonals = 1), 'two models are named "a"')
expect_error_like(diagonal_backtest(raa, "value", list(odp_glm_spec(), odp_glm_spec()),
                                    n_diagonals = 1), 'two models are named "odp"')
expect_error_like(diagonal_backtest(raa, "value", list(odp_glm_spec()), n_diagonals = 1,
                                    interval = 1.5), "interval")
expect_error_like(diagonal_backtest(raa, "value", list(odp_glm_spec()), n_diagonals = 0), "n_test = 0")
expect_error_like(diagonal_backtest(raa, "nope", list(odp_glm_spec()), n_diagonals = 1), "nope")
expect_error_like(diagonal_backtest(raa, "value", list(odp_glm_spec()), n_diagonals = 1,
                                    premium = "missing"), "premium must be")
expect_error_like(diagonal_backtest(raa, "value", list(odp_glm_spec()), n_diagonals = 1,
                                    premium = c(1, 2)), "expected 10")

cat("prospicio R diagonal backtest tests passed", "\n")
