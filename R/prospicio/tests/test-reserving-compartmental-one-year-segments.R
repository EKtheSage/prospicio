suppressMessages(library(prospicio))

# compartmental_one_year_segments(), the one-year view of Gesmann and Morris'
# model 1 over segments (docs/design/reserving-compartmental.md, decision 8,
# and reserving-bayes.md, decision 20). The seeds, the single-segment
# identity and the rank correlation are checked in the Rust unit tests and in
# validation/tests/reserving_compartmental_one_year.rs; these tests check the
# binding on company 337 workers' compensation under two keys, with tiny
# sampler budgets.

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

# Tests run from the repository root (CI and `cargo xtask r`).
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
gm2 <- gesmann_morris_2_priors()
n_sims <- 300

view <- function(tri, ...) {
  suppressWarnings(compartmental_one_year_segments(
    tri, "paid", "premium", gm, incurred = "incurred", chains = 2, tune = 100, draws = 100,
    n_sims = n_sims, seed = 5, refit = "reweight_only", ...))
}
totals_by_segment <- function(pd) draw_matrix(aggregate(pd, keep = "lob"))

views <- view(two)
cdr <- views@cdr
stopifnot(
  S7::S7_inherits(views, compartmental_one_year_fits),
  length(views@origins) == 20, identical(views@origins[1:2], c("1988", "1989")),
  length(views@premium) == 20, length(views@latest_paid) == 20,
  length(views@opening_ultimate) == 20, length(views@opening_reserve) == 20,
  is.null(views@approximate), views@k_threshold == 0.7,
  S7::S7_inherits(cdr, predictive_distribution), cdr@n_sims == n_sims,
  identical(names(cdr@keys), c("lob", "origin")), identical(unique(cdr@keys$lob), c("a", "b"))
)
# Every origin of the training triangle has a cell in the year (1988's 120
# months, 1997's first), and the result varies.
m <- draw_matrix(cdr)
stopifnot(all(apply(m, 2, function(v) any(v != 0))))

long_frame <- as.data.frame(views)
stopifnot(identical(names(long_frame),
                    c("lob", "origin", "premium", "latest_paid", "opening_ultimate",
                      "opening_reserve", "cdr_mean", "cdr_std_dev")),
          nrow(long_frame) == 20)
near(sum(long_frame$cdr_mean), mean(cdr), 1e-9 * (1 + abs(mean(cdr))))
totals <- totals_frame(views)
stopifnot(identical(totals$lob, c("a", "b")), "flagged_share" %in% names(totals))
near(sum(totals$opening_reserve), sum(views@opening_reserve), 1e-6)
expect_error_like(development_frame(views), "development table")
d <- views@diagnostics
stopifnot(identical(names(d), c("lob", "max_rhat", "min_ess_bulk", "min_ess_tail", "divergences",
                                "warnings")),
          identical(d$lob, c("a", "b")))
out <- capture.output(print(views))
stopifnot(grepl("<compartmental_one_year_fits> 300 simulations, 2 segments", out[1], fixed = TRUE))

# The accessors of one segment's fit need a single-segment fit.
expect_error_like(views@pareto_k, "single-segment fit")
expect_error_like(views@flagged_share, "single-segment fit")
expect_error_like(views@summary, "single-segment fit")

# One segment: its part of the joint claims development result.
one <- segment(views, lob = "b")
stopifnot(
  S7::S7_inherits(one, compartmental_one_year_fits),
  identical(draw_matrix(one@cdr), unname(m[, 11:20])),
  identical(one@premium, views@premium[11:20]),
  length(one@pareto_k) == n_sims, identical(one@refitted, integer()),
  identical(one@moment_matched, integer()), identical(one@unrefitted, integer()),
  one@flagged_share >= 0, one@flagged_share <= 1,
  nrow(one@draws) == 200, identical(summary(one), one@summary)
)
near(one@flagged_share, totals$flagged_share[2], 1e-12)
near(totals_frame(one)$cdr_std_dev, totals$cdr_std_dev[2], 1e-9)

# Independent: the same triangle under two names. Shared random numbers
# would make the totals almost perfectly rank correlated.
t <- totals_by_segment(cdr)
stopifnot(abs(stats::cor(t[, 1], t[, 2], method = "spearman")) < 4.5 / sqrt(n_sims))

# Rank correlation reaches the target, and each segment keeps its simulations.
ranked <- view(two, dependence = "rank_correlation", spearman = matrix(c(1, 0.5, 0.5, 1), 2))
tr <- totals_by_segment(ranked@cdr)
near(stats::cor(tr[, 1], tr[, 2], method = "spearman"), 0.5, 0.06)
near(sort(tr[, 1]), sort(t[, 1]), 1e-9 * (1 + max(abs(t[, 1]))))

# Replay by seed, and a smaller one_year_sims is the first simulations.
again <- view(two)
stopifnot(identical(draw_matrix(again@cdr), m))
small <- view(two, one_year_sims = 50)
stopifnot(identical(draw_matrix(small@cdr), m[1:50, ]))

# max_refits is a budget of each segment; unrefitted simulations warn.
budget <- with_warnings(compartmental_one_year_segments(
  two, "paid", "premium", gm, incurred = "incurred", chains = 2, tune = 100, draws = 100,
  n_sims = 20, seed = 5, refit = "refit_all", max_refits = 0))
stopifnot(any(grepl("max_refits", budget$warnings, fixed = TRUE)),
          identical(segment(budget$value, lob = "a")@refitted, integer()),
          length(segment(budget$value, lob = "a")@unrefitted) == 20)

# Model 1's default moment matches each segment's flagged simulations
# before it refits them (none here, with max_refits = 0).
default <- suppressWarnings(compartmental_one_year_segments(
  two, "paid", "premium", gm, incurred = "incurred", chains = 2, tune = 100, draws = 100,
  n_sims = 30, seed = 5, max_refits = 0))
stopifnot(is.null(default@approximate))
matched <- 0
for (name in c("a", "b")) {
  s <- segment(default, lob = name)
  flagged <- which(is.na(s@pareto_k) | s@pareto_k > 0.7)
  stopifnot(identical(s@refitted, integer()),
            identical(sort(c(s@moment_matched, s@unrefitted)), flagged))
  matched <- matched + length(s@moment_matched)
}
stopifnot(matched > 0)

# Since the user's decision of 2026-10-21 model 2's default refits the flagged
# simulations with the light sampler; light_refit = FALSE reweights only, with
# a warning that it is approximate (short chains may miss a diagnostic
# threshold too).
model_2_segments <- function(...) {
  with_warnings(compartmental_one_year_segments(
    two, "paid", "premium", gm2, incurred = "incurred", likelihood = "lognormal",
    scale = "loss_ratio", paid_response = "incremental", chains = 2, tune = 200, draws = 200,
    target_accept = 0.9, seed = 5, one_year_sims = 50, ...))
}
m2 <- model_2_segments(light_refit = FALSE)
stopifnot(
  grepl("approximate", m2$value@approximate, fixed = TRUE),
  any(grepl("approximate", m2$warnings, fixed = TRUE))
)
for (name in c("a", "b")) {
  s <- segment(m2$value, lob = name)
  stopifnot(length(s@pareto_k) == 50, identical(s@refitted, integer()),
            identical(s@moment_matched, integer()), identical(s@unrefitted, integer()))
}
# An explicit mode carries no warning.
chosen <- model_2_segments(refit = "reweight_only")
stopifnot(
  is.null(chosen$value@approximate),
  !any(grepl("approximate", chosen$warnings, fixed = TRUE)),
  identical(draw_matrix(chosen$value@cdr), draw_matrix(m2$value@cdr))
)

# Errors, before any sampling.
expect_error_like(compartmental_one_year_segments(two, "paid", "premium", gm,
                                                  incurred = "incurred",
                                                  dependence = "synchronized"),
                  "Synchronized")
expect_error_like(compartmental_one_year_segments(two, "paid", "premium", gm),
                  "exactly one of outstanding")
cat("ok\n")
