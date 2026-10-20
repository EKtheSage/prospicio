suppressMessages(library(prospicio))

# compartmental_candidate() in diagonal_backtest(), and
# compartmental_outstanding_backtest() (docs/design/reserving-compartmental.md,
# decision 8, "As built"). The seed sweeps and the targets are in
# validation/tests/reserving_compartmental_segments.rs; these tests check the
# binding on company 337 workers' compensation (the papers' triangle) with a
# small sampler: the tables' shapes, finite metrics, the fits shared between
# the two tables, replay by seed and the errors. Model 2 is left to the Rust
# validation (its sampler is too slow for a test).

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

near <- function(a, b, tol) {
  if (!all(abs(a - b) <= tol)) {
    stop(sprintf("got %s, want %s (tol %g)", format(a, digits = 17), format(b, digits = 17), tol),
         call. = FALSE)
  }
}

# Tests run from the repository root (CI and `cargo xtask r`). The papers'
# training triangle (cells at calendar years up to 1996, and accident year
# 1997 with its premium only).
all <- utils::read.csv(file.path("validation", "data", "clrd_wkcomp_337.csv"),
                       comment.char = "#")
long <- all[all$origin + all$development %/% 12 - 1 <= 1996, ]
stopifnot(nrow(long) == 45)
y1997 <- all[all$origin == 1997 & all$development == 12, ]
y1997$paid <- NA
y1997$incurred <- NA
long <- rbind(long, y1997)
tri <- triangle(long, "origin", "development", c("paid", "incurred", "premium"))
gm <- gesmann_morris_priors()

n_sims <- 400
n_diagonals <- 3
seed <- 3
small <- function() {
  compartmental_candidate(tri, "paid", "premium", gm, incurred = "incurred", chains = 2,
                          tune = 300, draws = 300)
}

cand <- small()
stopifnot(
  S7::S7_inherits(cand, compartmental_backtest_model),
  identical(cand@name, "compartmental_gm1"),
  identical(cand@paid, "paid"), identical(cand@premium, "premium")
)
out <- capture.output(print(cand))
stopifnot(grepl("compartmental_gm1", out, fixed = TRUE))
m2 <- compartmental_model_2_candidate(tri, "paid", "premium", gesmann_morris_2_priors(),
                                      incurred = "incurred")
stopifnot(identical(m2@name, "compartmental_gm2"))

# The paid table beside the ODP GLM, then the outstanding table.
t_paid <- system.time(
  paid <- suppressWarnings(diagonal_backtest(tri, "paid", list(odp = odp_glm_spec(), cand),
                                             n_diagonals, n_sims = n_sims, seed = seed))
)[["elapsed"]]
t_os <- system.time(
  os <- compartmental_outstanding_backtest(cand, n_diagonals, n_sims = n_sims, seed = seed)
)[["elapsed"]]

stopifnot(
  S7::S7_inherits(paid, backtest), S7::S7_inherits(os, backtest),
  identical(paid@models, c("odp", "compartmental_gm1")),
  identical(os@models, "compartmental_gm1_outstanding"),
  paid@n_splits == n_diagonals, os@n_splits == n_diagonals,
  identical(paid@diagonals, c("1994", "1995", "1996")),
  identical(os@diagonals, paid@diagonals),
  # 5, 6 and 7 cells over the three diagonals, as in the Rust validation.
  identical(lengths(paid@scored_rows), c(5L, 6L, 7L)),
  identical(os@scored_rows, paid@scored_rows),
  identical(paid@excluded, c(2L, 2L, 2L)), identical(os@excluded, paid@excluded)
)
for (bt in list(paid, os)) {
  finite <- bt@mean[setdiff(names(bt@mean), "model")]
  stopifnot(all(is.finite(as.matrix(finite))))
}
stopifnot(os@mean$coverage >= 0, os@mean$coverage <= 1)

# The ODP GLM is deterministic: the validation's targets to 0.001.
ae <- paid@scores[paid@scores$metric == "actual_vs_expected", ]
near(ae$value[ae$model == "odp"], c(0.89353, 0.9217, 1.0152), 0.001)
# The candidate's sampler is small, the targets are for the default one: a
# quarter of the target is the bar here.
near(ae$value[ae$model == "compartmental_gm1"] / c(1.3404, 1.0679, 1.1531), 1, 0.25)
stopifnot(os@mean$crps > 1881.9 / 2, os@mean$crps < 1881.9 * 2,
          os@mean$total_crps > 3617.3 / 2, os@mean$total_crps < 3617.3 * 2)

# One fit serves both tables: the same diagnostics, and the outstanding
# table is only forecasts, a small fraction of the sampling. The timing is
# the proof that the fits are shared: equal diagnostics would also follow
# from a refit on the same seed.
d_paid <- paid@diagnostics[paid@diagnostics$model == "compartmental_gm1", ]
stopifnot(
  all(is.na(paid@diagnostics$min_ess_bulk[paid@diagnostics$model == "odp"])),
  nrow(d_paid) == n_diagonals, all(d_paid$min_ess_bulk > 0),
  identical(os@diagnostics$min_ess_bulk, d_paid$min_ess_bulk),
  identical(os@diagnostics$max_rhat, d_paid$max_rhat),
  t_os < 0.5 * t_paid
)

# outstanding = reads the case reserves from a column; here the column is
# incurred less paid, so both tables equal those of incurred = on the same
# seed, cell for cell.
long_os <- long
long_os$outstanding <- long_os$incurred - long_os$paid
tri_os <- triangle(long_os, "origin", "development",
                   c("paid", "incurred", "premium", "outstanding"))
both <- lapply(c("outstanding", "incurred"), function(how) {
  args <- list(tri_os, "paid", "premium", gm, chains = 2, tune = 300, draws = 300)
  args[[how]] <- how
  c_how <- do.call(compartmental_candidate, args)
  list(
    paid = suppressWarnings(diagonal_backtest(tri_os, "paid", list(c_how), 1, n_sims = 200,
                                              seed = seed)),
    os = compartmental_outstanding_backtest(c_how, 1, n_sims = 200, seed = seed)
  )
})
for (table in c("paid", "os")) {
  got <- both[[1]][[table]]
  want <- both[[2]][[table]]
  stopifnot(all(is.finite(got@scores$value)), identical(got@scores$value, want@scores$value),
            identical(got@scored_rows, want@scored_rows))
}
stopifnot(length(both[[1]]$os@scored_rows[[1]]) == 7)

# A named candidate, and replay by seed.
a <- suppressWarnings(diagonal_backtest(tri, "paid", list(gm1 = small()), 1, n_sims = 200,
                                        seed = seed))
b <- suppressWarnings(diagonal_backtest(tri, "paid", list(gm1 = small()), 1, n_sims = 200,
                                        seed = seed))
stopifnot(identical(a@models, "gm1"), identical(a@scores$value, b@scores$value))

# Errors.
expect_error_like(compartmental_candidate(tri, "paid", "premium", gm), "exactly one")
expect_error_like(
  compartmental_candidate(tri, "paid", "premium", gm, outstanding = "incurred",
                          incurred = "incurred"),
  "exactly one")
expect_error_like(diagonal_backtest(tri, "incurred", list(cand), 1, n_sims = 100),
                  "another triangle or paid column")
other <- triangle(data.frame(year = c(2020, 2021, 2021), age = c(12, 12, 24),
                             paid = c(1, 2, 3)), "year", "age", "paid")
expect_error_like(diagonal_backtest(other, "paid", list(cand), 1, n_sims = 100),
                  "another triangle or paid column")
expect_error_like(compartmental_outstanding_backtest(cand, 50, n_sims = 100), "earlier period")
expect_error_like(compartmental_outstanding_backtest(odp_glm_spec(), 1), "compartmental_candidate")
