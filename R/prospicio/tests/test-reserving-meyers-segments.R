suppressMessages(library(prospicio))

# meyers_segments(), Meyers' CRC, CSR and CAY over segments
# (docs/design/reserving-bayes.md, decisions 11 and 13). The seed sweep and
# the diagonal backtest are in validation/tests/reserving_meyers_segments.rs;
# these tests check the binding.

near <- function(a, b, tol) {
  if (!all(abs(a - b) <= tol)) {
    stop(sprintf("got %s, want %s (tol %g)", format(a, digits = 17), format(b, digits = 17), tol),
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

# Tests run from the repository root (CI and `cargo xtask r`). The lines of
# clrd_lines.csv in `lines`, renamed by `names`, as the segments of `lob`,
# with paid losses and net earned premium.
clrd_all <- utils::read.csv(file.path("validation", "data", "clrd_lines.csv"), comment.char = "#")
clrd <- function(lines, names = lines) {
  parts <- Map(function(line, name) {
    d <- clrd_all[clrd_all$lob == line, ]
    d$lob <- name
    d
  }, lines, names)
  triangle(do.call(rbind, parts), "origin", "development", c("paid", "premium"), keys = "lob")
}

# Each segment's total per simulation, one column per segment.
line_totals <- function(pd) draw_matrix(aggregate(pd, keep = "lob"))

n_sims <- 4000

# Tolerances from seeds 0 to 59 at 4,000 simulations (each at least 4 sds
# and 1.25 times the largest deviation there; python/tests/
# test_reserving_meyers_segments.py): a segment against meyers() of that
# segment alone, logelr's posterior mean (sd 0.0001), the mean total reserve
# (sd 0.18%) and elpd_loo (sd 0.52, largest 1.42); twin lines' mean totals
# (sd 0.19%).
LOGELR_TOL <- 0.001
TOTAL_TOL <- 0.01
LOO_TOL <- 2.5
TWIN_TOL <- 0.01
two <- clrd(c("comauto", "wkcomp"))

# The default chains pass every diagnostic on both lines, so no warning.
run <- with_warnings(meyers_segments(two, "paid", "premium", n_sims = n_sims, seed = 5))
fits <- run$value
stopifnot(
  length(run$warnings) == 0,
  S7::S7_inherits(fits, meyers_fits),
  identical(fits@model, "crc"),
  S7::S7_inherits(fits@chain_ladder, chain_ladder_fit),
  length(fits@origins) == 20,
  identical(fits@development, seq(12L, 120L, by = 12L)),
  length(fits@premium) == 20, fits@premium[1] == 913636
)
r <- fits@reserves
u <- fits@ultimates
for (pd in list(r, u)) {
  stopifnot(
    S7::S7_inherits(pd, predictive_distribution),
    pd@n_sims == n_sims,
    identical(names(pd@keys), c("lob", "origin")),
    identical(unique(pd@keys$lob), c("comauto", "wkcomp"))
  )
}

long <- as.data.frame(fits)
stopifnot(identical(names(long), c("lob", "origin", "latest", "ultimate", "reserve", "premium",
                                   "ulr", "mean", "std_dev", "ultimate_mean", "ultimate_std_dev",
                                   "ulr_mean")),
          identical(long$ulr, long$ultimate / long$premium),
          identical(long$ulr_mean, long$ultimate_mean / long$premium),
          nrow(long) == 20)
near(sum(long$mean), mean(r), 1e-9 * mean(r))
near(sum(long$ultimate_mean), mean(u), 1e-9 * mean(u))
# Each simulation's ultimates less its reserves are the latest values.
latest <- matrix(long$latest, nrow = n_sims, ncol = 20, byrow = TRUE)
near(draw_matrix(u) - draw_matrix(r), latest, 1e-9 * max(latest))
totals <- totals_frame(fits)
stopifnot(identical(totals$lob, c("comauto", "wkcomp")),
          all(c("premium", "ultimate_mean") %in% names(totals)),
          identical(totals$ulr_mean, totals$ultimate_mean / totals$premium))
near(sum(totals$premium), sum(fits@premium), 1e-6)
stopifnot(nrow(development_frame(fits)) == 20)

d <- fits@diagnostics
stopifnot(
  is.data.frame(d),
  identical(names(d), c("lob", "max_rhat", "min_ess_bulk", "min_ess_tail", "divergences",
                        "warnings")),
  identical(d$lob, c("comauto", "wkcomp")),
  all(d$max_rhat < 1.01), all(d$warnings == ""), is.integer(d$divergences)
)
out <- capture.output(print(fits))
stopifnot(grepl("<meyers_fits> CRC, 4000 simulations, 2 segments", out[1], fixed = TRUE))

# The posterior of one segment needs a single-segment fit.
expect_error_like(fits@summary, "single-segment fit")
expect_error_like(fits@draws, "single-segment fit")
expect_error_like(bayes_loo(fits), "single-segment fit")

# One segment: its part of the joint ultimates and reserves, and a
# posterior like meyers()'s with its own seeds (the same premium, other
# draws).
one <- segment(fits, lob = "wkcomp")
stopifnot(
  S7::S7_inherits(one, meyers_fits),
  identical(names(one@reserves@keys), c("lob", "origin")),
  identical(draw_matrix(one@reserves), unname(draw_matrix(r)[, 11:20])),
  identical(draw_matrix(one@ultimates), unname(draw_matrix(u)[, 11:20])),
  identical(one@premium, fits@premium[11:20])
)
alone <- meyers(clrd("wkcomp"), "paid", "premium", n_sims = n_sims, seed = 5)
stopifnot(identical(one@premium, alone@premium),
          identical(colnames(one@draws), colnames(alone@draws)),
          nrow(one@draws) == 8000, !identical(one@draws, alone@draws))
near(one@summary$mean[1], alone@summary$mean[1], LOGELR_TOL)
near(totals_frame(one)$mean / totals_frame(alone)$mean, 1, TOTAL_TOL)
stopifnot(identical(summary(one), one@summary))
loo <- bayes_loo(one)
stopifnot(length(loo$pointwise) == 55)
near(loo$estimates[["elpd"]], bayes_loo(alone)$estimates[["elpd"]], LOO_TOL)

# Independent: the same line under two names. Shared random numbers would
# make the totals almost perfectly rank correlated; 4.5 / sqrt(n_sims) is
# the tolerance of validation/tests/reserving_meyers_segments.rs (sd 0.0083
# at 10,000 simulations over 60 seeds, about 1 / sqrt(n_sims)).
twins <- meyers_segments(clrd(c("wkcomp", "wkcomp"), c("a", "b")), "paid", "premium",
                         n_sims = n_sims, seed = 11)
t <- line_totals(twins@reserves)
stopifnot(abs(stats::cor(t[, 1], t[, 2], method = "spearman")) < 4.5 / sqrt(n_sims))
near(mean(t[, 1]) / mean(t[, 2]), 1, TWIN_TOL)

# Rank correlation: Iman-Conover reaches the target within 0.0081 at 10,000
# simulations over 60 seeds (sd at most 0.0030); 0.03 allows for 4,000. The
# ultimates move with the reserves.
ranked <- meyers_segments(two, "paid", "premium", model = "csr", n_sims = n_sims, seed = 5,
                          dependence = "rank_correlation",
                          spearman = matrix(c(1, 0.5, 0.5, 1), 2))
t <- line_totals(ranked@reserves)
near(stats::cor(t[, 1], t[, 2], method = "spearman"), 0.5, 0.03)
tu <- line_totals(ranked@ultimates)
near(diag(stats::cor(t, tu, method = "spearman")), c(1, 1), 1e-12)

# CAY, and replay by seed.
quick <- function(seed) {
  suppressWarnings(meyers_segments(two, "paid", "premium", model = "cay", chains = 2,
                                   tune = 200, draws = 200, n_sims = 500, seed = seed))
}
a <- quick(7)
stopifnot(identical(a@model, "cay"),
          "rho" %in% colnames(segment(a, lob = "comauto")@draws),
          identical(draw_matrix(a@reserves), draw_matrix(quick(7)@reserves)),
          !identical(draw_matrix(a@reserves), draw_matrix(quick(8)@reserves)))

# Failed diagnostics warn once, naming each segment, and return the fit.
run <- with_warnings(meyers_segments(two, "paid", "premium", chains = 2, tune = 50, draws = 10,
                                     n_sims = 100, seed = 1))
stopifnot(
  length(run$warnings) == 1,
  grepl("meyers_segments: the MCMC diagnostics failed", run$warnings, fixed = TRUE),
  grepl("comauto: ", run$warnings, fixed = TRUE), grepl("wkcomp: ", run$warnings, fixed = TRUE),
  all(run$value@diagnostics$warnings != ""),
  run$value@reserves@n_sims == 100
)

# Errors.
expect_error_like(meyers_segments(two, "paid", "premium", dependence = "synchronized"),
                  "no residuals")
expect_error_like(meyers_segments(two, "paid", "premium", dependence = "rank_correlation"),
                  "needs a spearman matrix")
expect_error_like(meyers_segments(two, "paid", "premium", spearman = diag(2)),
                  "only with dependence")
expect_error_like(meyers_segments(two, "paid", "premium", dependence = "copula"),
                  "should be one of")
expect_error_like(meyers_segments(two, "paid", "premium", dependence = "rank_correlation",
                                  spearman = c(1, 0.5)), "square numeric matrix")
expect_error_like(meyers_segments(two, "paid", "premium", n_sims = 0), "n_sims must be positive")
expect_error_like(meyers_segments(two, "missing", "premium"), "missing")
expect_error_like(meyers_segments(two, "paid"), "premium")
expect_error_like(meyers_segments(two, "paid", "premium", model = "ipi"), "should be one of")

cat("prospicio R Meyers segments tests passed\n")
