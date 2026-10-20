suppressMessages(library(prospicio))

# compartmental_segments(), Gesmann and Morris' model 1 over segments
# (docs/design/reserving-compartmental.md, decision 8). The seed sweeps, the
# twins and the diagonal backtest are in
# validation/tests/reserving_compartmental_segments.rs; these tests check
# the binding, on company 337 workers' compensation under two keys.

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

# Tests run from the repository root (CI and `cargo xtask r`). The papers'
# training triangle (cells at calendar years up to 1996, and accident year
# 1997 with its premium only) once per name, as the segments of `lob`.
all <- utils::read.csv(file.path("validation", "data", "clrd_wkcomp_337.csv"),
                       comment.char = "#")
long <- all[all$origin + all$development %/% 12 - 1 <= 1996, ]
stopifnot(nrow(long) == 45)
y1997 <- all[all$origin == 1997 & all$development == 12, ]
y1997$paid <- NA
y1997$incurred <- NA
long <- rbind(long, y1997)
lines <- function(names) {
  parts <- lapply(names, function(n) cbind(lob = n, long))
  triangle(do.call(rbind, parts), "origin", "development", c("paid", "incurred", "premium"),
           keys = "lob")
}
two <- lines(c("a", "b"))
gm <- gesmann_morris_priors()
n_sims <- 2000

totals_by_segment <- function(pd) draw_matrix(aggregate(pd, keep = "lob"))

fits <- suppressWarnings(compartmental_segments(two, "paid", "premium", gm, incurred = "incurred",
                                                n_sims = n_sims, seed = 5))
stopifnot(
  S7::S7_inherits(fits, compartmental_fits),
  length(fits@origins) == 20, identical(fits@origins[1:2], c("1988", "1989")),
  length(fits@premium) == 20, length(fits@latest_paid) == 20,
  identical(fits@rate_swaps, c(0L, 0L))
)
r <- fits@reserves
stopifnot(
  S7::S7_inherits(r, predictive_distribution), r@n_sims == n_sims,
  identical(names(r@keys), c("lob", "origin")),
  identical(unique(r@keys$lob), c("a", "b")),
  identical(names(fits@paid@keys), c("lob", "origin", "age")),
  identical(names(fits@outstanding@keys), c("lob", "origin", "age")),
  length(fits@paid@keys$lob) == 110, length(fits@outstanding@keys$lob) == 110
)

long_frame <- as.data.frame(fits)
stopifnot(identical(names(long_frame), c("lob", "origin", "premium", "latest_paid", "mean",
                                         "std_dev", "ultimate_loss_ratio")),
          nrow(long_frame) == 20)
near(sum(long_frame$mean), mean(r), 1e-9 * mean(r))
totals <- totals_frame(fits)
stopifnot(identical(totals$lob, c("a", "b")))
near(sum(totals$mean), mean(r), 1e-9 * mean(r))
stopifnot(nrow(fits@paid_table) == 110, nrow(fits@outstanding_table) == 110,
          identical(sort(unique(fits@paid_table$lob)), c("a", "b")))
expect_error_like(development_frame(fits), "no development table")

d <- fits@diagnostics
stopifnot(
  is.data.frame(d),
  identical(names(d), c("lob", "max_rhat", "min_ess_bulk", "min_ess_tail", "divergences",
                        "warnings")),
  identical(d$lob, c("a", "b")), is.integer(d$divergences)
)
out <- capture.output(print(fits))
stopifnot(grepl("<compartmental_fits> 2000 simulations, 2 segments", out[1], fixed = TRUE))

# The posterior of one segment needs a single-segment fit.
expect_error_like(fits@summary, "single-segment fit")
expect_error_like(fits@draws, "single-segment fit")
expect_error_like(bayes_loo(fits), "single-segment fit")

# One segment: its part of the joint distributions, and a posterior like
# compartmental()'s with its own seeds (the same premium, other draws).
one <- segment(fits, lob = "b")
stopifnot(
  S7::S7_inherits(one, compartmental_fits),
  identical(draw_matrix(one@reserves), unname(draw_matrix(r)[, 11:20])),
  identical(draw_matrix(one@paid), unname(draw_matrix(fits@paid)[, 56:110])),
  identical(one@premium, fits@premium[11:20])
)
alone <- suppressWarnings(compartmental(lines("b"), "paid", "premium", gm, incurred = "incurred",
                                        n_sims = n_sims, seed = 5))
stopifnot(identical(one@premium, alone@premium),
          identical(colnames(one@draws), colnames(alone@draws)),
          nrow(one@draws) == 8000, !identical(one@draws, alone@draws),
          identical(summary(one), one@summary))
near(one@summary$mean[1] / alone@summary$mean[1], 1, 0.05)
near(totals_frame(one)$mean / totals_frame(alone)$mean, 1, 0.05)
stopifnot(length(bayes_loo(one)$pointwise) > 0)

# Independent: the same triangle under two names. Shared random numbers
# would make the totals almost perfectly rank correlated.
t <- totals_by_segment(r)
stopifnot(abs(stats::cor(t[, 1], t[, 2], method = "spearman")) < 4.5 / sqrt(n_sims))
near(mean(t[, 1]) / mean(t[, 2]), 1, 0.02)

# Rank correlation reaches the target, and a segment's future paid cells move
# with its reserves: the paid at the oldest age ranks as the reserve.
ranked <- suppressWarnings(compartmental_segments(
  two, "paid", "premium", gm, incurred = "incurred", n_sims = n_sims, seed = 5,
  dependence = "rank_correlation", spearman = matrix(c(1, 0.5, 0.5, 1), 2)))
t <- totals_by_segment(ranked@reserves)
near(stats::cor(t[, 1], t[, 2], method = "spearman"), 0.5, 0.04)
paid <- ranked@paid
last <- draw_matrix(paid)[, paid@keys$age == 120, drop = FALSE]
for (k in 1:2) {
  cols <- paid@keys$lob[paid@keys$age == 120] == c("a", "b")[k]
  near(stats::cor(rowSums(last[, cols, drop = FALSE]), t[, k], method = "spearman"), 1, 1e-9)
}

# Replay by seed.
quick <- function(seed) {
  suppressWarnings(compartmental_segments(two, "paid", "premium", gm, incurred = "incurred",
                                          chains = 2, tune = 200, draws = 200, n_sims = 300,
                                          seed = seed))
}
a <- quick(7)
stopifnot(identical(draw_matrix(a@reserves), draw_matrix(quick(7)@reserves)),
          identical(draw_matrix(a@paid), draw_matrix(quick(7)@paid)),
          !identical(draw_matrix(a@reserves), draw_matrix(quick(8)@reserves)))

# Failed diagnostics warn once, naming each segment, and return the fit.
run <- with_warnings(compartmental_segments(two, "paid", "premium", gm, incurred = "incurred",
                                            chains = 2, tune = 50, draws = 10, n_sims = 100,
                                            seed = 1))
stopifnot(
  length(run$warnings) == 1,
  grepl("compartmental_segments: the MCMC diagnostics failed", run$warnings, fixed = TRUE),
  grepl("a: ", run$warnings, fixed = TRUE), grepl("b: ", run$warnings, fixed = TRUE),
  all(run$value@diagnostics$warnings != ""),
  run$value@reserves@n_sims == 100
)

# Errors.
args <- list(two, "paid", "premium", gm, incurred = "incurred")
expect_error_like(do.call(compartmental_segments, c(args, dependence = "synchronized")),
                  "no residuals")
expect_error_like(do.call(compartmental_segments, c(args, dependence = "rank_correlation")),
                  "needs a spearman matrix")
expect_error_like(do.call(compartmental_segments, c(args, list(spearman = diag(2)))),
                  "only with dependence")
expect_error_like(do.call(compartmental_segments, c(args, dependence = "copula")),
                  "should be one of")
expect_error_like(compartmental_segments(two, "paid", "premium", gm),
                  "exactly one of outstanding")
expect_error_like(compartmental_segments(two, "paid", "premium"), "priors are required")
expect_error_like(compartmental_segments(two, "missing", "premium", gm, incurred = "incurred"),
                  "missing")
expect_error_like(compartmental(two, "paid", "premium", gm, incurred = "incurred"), "segment")

cat("prospicio R compartmental segments tests passed\n")
