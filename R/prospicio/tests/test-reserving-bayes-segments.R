suppressMessages(library(prospicio))

# bayes_odp_segments(), the Bayesian ODP over segments
# (docs/design/reserving-bayes.md, decision 11). The seed sweep and the
# diagonal backtest are in validation/tests/reserving_bayes_segments.rs;
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
# clrd_lines.csv in `lines`, renamed by `names`, as the segments of `lob`.
clrd_all <- utils::read.csv(file.path("validation", "data", "clrd_lines.csv"), comment.char = "#")
clrd <- function(lines, names = lines) {
  parts <- Map(function(line, name) {
    d <- clrd_all[clrd_all$lob == line, ]
    d$lob <- name
    d
  }, lines, names)
  triangle(do.call(rbind, parts), "origin", "development", "paid", keys = "lob")
}

# Each segment's total reserve per simulation, one column per segment.
line_totals <- function(fits) draw_matrix(aggregate(fits@reserves, keep = "lob"))

n_sims <- 4000
two <- clrd(c("comauto", "wkcomp"))

# The default chains pass every diagnostic on both lines, so no warning.
run <- with_warnings(bayes_odp_segments(two, n_sims = n_sims, seed = 5))
fits <- run$value
stopifnot(
  length(run$warnings) == 0,
  S7::S7_inherits(fits, bayes_odp_fits),
  S7::S7_inherits(fits@chain_ladder, chain_ladder_fit),
  length(fits@origins) == 20,
  identical(fits@development, seq(12L, 120L, by = 12L))
)
r <- fits@reserves
stopifnot(
  S7::S7_inherits(r, predictive_distribution),
  r@n_sims == n_sims,
  identical(names(r@keys), c("lob", "origin")),
  identical(unique(r@keys$lob), c("comauto", "wkcomp"))
)

long <- as.data.frame(fits)
stopifnot(identical(names(long), c("lob", "origin", "latest", "ultimate", "reserve", "mean",
                                   "std_dev")),
          nrow(long) == 20)
near(sum(long$mean), mean(r), 1e-9 * mean(r))
totals <- totals_frame(fits)
stopifnot(identical(totals$lob, c("comauto", "wkcomp")), "scale" %in% names(totals))
# Each line's posterior mean reserve sits near its chain ladder's.
stopifnot(all(abs(totals$mean / totals$reserve - 1) < 0.03))
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
stopifnot(grepl("<bayes_odp_fits> 4000 simulations, 2 segments", out[1], fixed = TRUE))

# The posterior of one segment needs a single-segment fit.
expect_error_like(fits@scale, "use totals_frame() or segment()")
expect_error_like(fits@summary, "single-segment fit")
expect_error_like(fits@draws, "single-segment fit")
expect_error_like(bayes_loo(fits), "single-segment fit")

# One segment: its part of the joint reserves, and a posterior like
# bayes_odp()'s with its own seeds (the same scale, other draws).
one <- segment(fits, lob = "wkcomp")
stopifnot(
  S7::S7_inherits(one, bayes_odp_fits),
  identical(names(one@reserves@keys), c("lob", "origin")),
  identical(draw_matrix(one@reserves), unname(draw_matrix(r)[, 11:20]))
)
alone <- bayes_odp(clrd("wkcomp"), n_sims = n_sims, seed = 5)
near(one@scale, alone@scale, 1e-12 * alone@scale)
stopifnot(identical(colnames(one@draws), colnames(alone@draws)),
          nrow(one@draws) == 4000, !identical(one@draws, alone@draws))
near(one@summary$mean[1], alone@summary$mean[1], 0.02)
stopifnot(identical(summary(one), one@summary))
loo <- bayes_loo(one)
stopifnot(length(loo$pointwise) == 55)
near(loo$estimates[["elpd"]], bayes_loo(alone)$estimates[["elpd"]], 1)

# Independent: the same line under two names. Shared random numbers would
# make the totals almost perfectly rank correlated; 4.5 / sqrt(n_sims) is
# the tolerance of validation/tests/reserving_bayes_segments.rs, sized from
# a 40-seed sweep (sd about 1 / sqrt(n_sims)).
twins <- bayes_odp_segments(clrd(c("wkcomp", "wkcomp"), c("a", "b")), n_sims = n_sims,
                            seed = 11)
t <- line_totals(twins)
stopifnot(abs(stats::cor(t[, 1], t[, 2], method = "spearman")) < 4.5 / sqrt(n_sims))
near(mean(t[, 1]) / mean(t[, 2]), 1, 0.01)

# Rank correlation: Iman-Conover reaches the target within 0.009 at 10,000
# simulations over 40 seeds (sd at most 0.0029); 0.03 allows for 4,000.
ranked <- bayes_odp_segments(two, n_sims = n_sims, seed = 5, dependence = "rank_correlation",
                             spearman = matrix(c(1, 0.5, 0.5, 1), 2))
t <- line_totals(ranked)
near(stats::cor(t[, 1], t[, 2], method = "spearman"), 0.5, 0.03)

# Replay by seed.
quick <- function(seed) {
  suppressWarnings(bayes_odp_segments(two, chains = 2, tune = 200, draws = 200, n_sims = 500,
                                      seed = seed))
}
a <- quick(7)
stopifnot(identical(draw_matrix(a@reserves), draw_matrix(quick(7)@reserves)),
          !identical(draw_matrix(a@reserves), draw_matrix(quick(8)@reserves)))

# Failed diagnostics warn once, naming each segment, and return the fit.
run <- with_warnings(bayes_odp_segments(two, chains = 2, tune = 50, draws = 10, n_sims = 100,
                                        seed = 1))
stopifnot(
  length(run$warnings) == 1,
  grepl("bayes_odp_segments: the MCMC diagnostics failed", run$warnings, fixed = TRUE),
  grepl("comauto: ", run$warnings, fixed = TRUE), grepl("wkcomp: ", run$warnings, fixed = TRUE),
  all(run$value@diagnostics$warnings != ""),
  run$value@reserves@n_sims == 100
)

# Errors.
expect_error_like(bayes_odp_segments(two, dependence = "synchronized"), "no residuals")
expect_error_like(bayes_odp_segments(two, dependence = "rank_correlation"),
                  "needs a spearman matrix")
expect_error_like(bayes_odp_segments(two, spearman = diag(2)), "only with dependence")
expect_error_like(bayes_odp_segments(two, dependence = "copula"), "should be one of")
expect_error_like(bayes_odp_segments(two, dependence = "rank_correlation", spearman = c(1, 0.5)),
                  "square numeric matrix")
expect_error_like(bayes_odp_segments(two, n_sims = 0), "n_sims must be positive")
expect_error_like(bayes_odp_segments(two, "missing"), "missing")

cat("prospicio R Bayesian reserving segments tests passed
")
