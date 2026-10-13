suppressMessages(library(prospicio))

# meyers(), Meyers' CRC, CSR and CAY models (docs/design/reserving-bayes.md,
# decision 7). The parity with Meyers' own Stan code and his printed numbers
# is validation/tests/reserving_meyers.rs and reserving_meyers_cay.rs; these
# tests check the binding on his group 353 Commercial Auto data.

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

# Tests run from the repository root (CI and `cargo xtask r`). Meyers fits
# the upper triangle: 1988 + w at ages up to 120 - 12 w.
long <- utils::read.csv(file.path("validation", "data", "clrd_comauto_353.csv"),
                        comment.char = "#")
long <- long[long$origin - 1988 + long$development %/% 12 <= 10, ]
tri <- triangle(long, "origin", "development", c("paid", "incurred", "premium"))

# The default chains (4 x 2,000 draws) pass every diagnostic on group 353.
run <- with_warnings(meyers(tri, "paid", "premium", n_sims = 2000))
fit <- run$value
stopifnot(
  length(run$warnings) == 0,
  S7::S7_inherits(fit, meyers_fit),
  fit@model == "crc",
  S7::S7_inherits(fit@chain_ladder, chain_ladder_fit),
  identical(fit@origins, as.character(1988:1997)),
  identical(fit@development, seq(12L, 120L, by = 12L)),
  identical(fit@premium[1:3], c(5812, 4908, 5454))
)

u <- fit@ultimates
r <- fit@reserves
stopifnot(
  S7::S7_inherits(u, predictive_distribution), S7::S7_inherits(r, predictive_distribution),
  u@n_sims == 2000, r@n_sims == 2000,
  identical(names(r@keys), "origin"), identical(r@keys$origin, fit@origins),
  all(draw_matrix(u)[, 1] == 3912)  # 1988 is closed and keeps its value
)
# Meyers (2019, Table 5.2): CRC paid total ultimate 40,121.
near(mean(u), 40121, 0.02)
near(mean(r), mean(u) - sum(as.data.frame(fit)$latest), 1e-9)

d <- fit@diagnostics
stopifnot(
  identical(names(d), c("max_rhat", "min_ess_bulk", "min_ess_tail", "divergences", "warnings")),
  d$max_rhat < 1.01, d$divergences == 0, length(d$warnings) == 0
)

s <- summary(fit)
stopifnot(
  identical(names(s), c("parameter", "mean", "sd", "q05", "q50", "q95", "rhat", "ess_bulk",
                        "ess_tail")),
  nrow(s) == 29, s$parameter[1] == "logelr", s$parameter[29] == "sigma[10]",
  !("gamma" %in% s$parameter),
  identical(s, fit@summary),
  identical(dim(fit@draws), c(8000L, 29L)),
  identical(colnames(fit@draws), s$parameter)
)

frame <- as.data.frame(fit)
totals <- totals_frame(fit)
stopifnot(
  identical(names(frame)[1:4], c("origin", "latest", "ultimate", "reserve")),
  all(c("premium", "mean", "std_dev", "ultimate_mean", "ultimate_std_dev") %in% names(frame)),
  nrow(frame) == 10, nrow(totals) == 1,
  nrow(development_frame(fit)) == 10,
  S7::S7_inherits(segment(fit), meyers_fit)
)
near(sum(frame$mean), mean(r), 1e-9)
near(totals$ultimate_mean, mean(u), 1e-9)
near(totals$premium, sum(fit@premium), 1e-12)
expect_error_like(segment(fit, lob = "auto"), "lob")

loo <- bayes_loo(fit)
stopifnot(
  identical(names(loo$estimates), c("elpd", "se", "p", "ic")),
  length(loo$pointwise) == 55, length(loo$pareto_k) == 55,
  abs(loo$estimates[["elpd"]] - 47.8) < 2  # Meyers (2019, Table 6.4)
)
invisible(capture.output(print(fit)))

# CSR: gamma sits between beta and sigma. Meyers (2019, Tables 7.1, 7.2).
csr <- meyers(tri, "paid", "premium", model = "csr", n_sims = 2000)
stopifnot(
  csr@model == "csr",
  ncol(csr@draws) == 30, colnames(csr@draws)[20] == "gamma",
  abs(csr@summary$mean[20] - 0.0446) < 0.01
)
near(mean(csr@ultimates), 37597, 0.02)

# CAY on incurred: rho sits in gamma's slot. Meyers (2019, Tables 8.1 to
# 8.3): rho 0.1709, logelr -0.3945, total ultimate 39,193, elpd_loo 68.65;
# ours at seed 0: 0.1686, -0.3947, 39,178 (10,000 simulations), 69.05.
run <- with_warnings(meyers(tri, "incurred", "premium", model = "cay", n_sims = 2000))
cay <- run$value
s <- summary(cay)
stopifnot(
  length(run$warnings) == 0,
  cay@model == "cay",
  ncol(cay@draws) == 30, colnames(cay@draws)[20] == "rho", colnames(cay@draws)[21] == "sigma[1]",
  !("gamma" %in% colnames(cay@draws)),
  all(abs(cay@draws[, "rho"]) < 1),
  abs(s$mean[s$parameter == "rho"] - 0.1709) < 0.05,
  abs(s$mean[s$parameter == "logelr"] + 0.3945) < 0.01,
  abs(bayes_loo(cay)$estimates[["elpd"]] - 68.65) < 2.5,
  nrow(as.data.frame(cay)) == 10
)
near(mean(cay@ultimates), 39193, 0.02)
invisible(capture.output(print(cay)))

# CAY needs the previous origin's cell at each age: 1992 stops at 36
# months while 1993 runs to 60. CRC takes the same triangle.
ragged <- long[!(long$origin == 1992 & long$development > 36), ]
ragged <- triangle(ragged, "origin", "development", c("paid", "premium"))
expect_error_like(meyers(ragged, "paid", "premium", model = "cay", chains = 1, tune = 50,
                         draws = 20, n_sims = 100),
                  "origin 1993 is observed at development age 48 months but origin 1992 is not")
invisible(suppressWarnings(meyers(ragged, "paid", "premium", chains = 1, tune = 50, draws = 20,
                                  n_sims = 100)))

# One seed replays the fit; another changes it.
quick <- function(seed) {
  suppressWarnings(meyers(tri, "incurred", "premium", chains = 2, tune = 200, draws = 200,
                          n_sims = 500, seed = seed))
}
a <- quick(7)
b <- quick(7)
c8 <- quick(8)
stopifnot(
  identical(a@draws, b@draws),
  identical(draw_matrix(a@ultimates), draw_matrix(b@ultimates)),
  !identical(a@draws, c8@draws),
  !identical(draw_matrix(a@ultimates), draw_matrix(c8@ultimates))
)

# Failed diagnostics warn but return the fit: 2 x 10 kept draws cannot reach
# a bulk ESS of 200 (100 per chain).
short <- with_warnings(meyers(tri, "paid", "premium", chains = 2, tune = 50, draws = 10,
                              n_sims = 100, seed = 1))
stopifnot(
  length(short$warnings) == 1,
  grepl("meyers: the MCMC diagnostics failed", short$warnings, fixed = TRUE),
  grepl("bulk ESS", short$warnings, fixed = TRUE),
  short$value@reserves@n_sims == 100
)

# Errors.
expect_error_like(meyers(tri, "paid", "premium", n_sims = 0), "n_sims must be positive")
expect_error_like(meyers(tri, "paid", "premium", model = "cit"), "should be one of")
expect_error_like(meyers(tri, "paid", "premium", seed = -1),
                  "seed must be a non-negative whole number")
expect_error_like(meyers(tri, "paid", "premium", gamma_sd = 0, n_sims = 100), "gamma_sd")
expect_error_like(meyers(tri, "incurred", "premium", model = "cay", rho_shape = 0, n_sims = 100),
                  "rho_shape")
expect_error_like(meyers(tri, "paid", "premium", rho_shape = NA), "rho_shape")
expect_error_like(meyers(tri, "paid", "missing", n_sims = 100), "missing")
expect_error_like(meyers(tri, "paid", 1), "premium must be the name of a column")
small <- data.frame(year = c(2020, 2020, 2020, 2021, 2021, 2022), age = c(12, 24, 36, 12, 24, 12),
                    paid = c(100, 150, 160, 110, 170, 130), premium = 200)
zero <- small
zero$paid[4] <- 0
expect_error_like(meyers(triangle(zero, "year", "age", c("paid", "premium")), "paid", "premium",
                         n_sims = 100), "positive cumulative values")
expect_error_like(meyers(triangle(small, "year", "age", c("paid", "premium")), "paid", "premium",
                         n_sims = 100), "more cells than parameters")

cat("prospicio R Meyers reserving tests passed\n")
