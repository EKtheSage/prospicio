# Aggregate lane: compound distributions and simulated events, over
# crates/prospicio-r/src/aggregate.rs. The #' comments are the package
# documentation (see distributions.R).

#' @include distributions.R
NULL

#' Compound (aggregate) loss distribution
#'
#' The distribution of `S = X_1 + ... + X_N` on the severity grid's step, by
#' Panjer's recursion or by FFT. The result's `report` property records the
#' error: `tail_mass` (probability above the last point, lumped onto it),
#' `aliasing_error` (FFT only; check it first) and `mean_error`.
#'
#' Both methods assume the claim count independent of the severities and the
#' severities independent and identically distributed. For dependence use
#' [fft_bivariate()] or [clash_aggregate()] (two lines hit by the same
#' events), one [mixed_poisson_count()] on a [mixture_distribution()]
#' (components sharing a mixing variable) or [reorder_groups()] on simulated
#' totals.
#'
#' @param frequency A [poisson_count], [negative_binomial_count] or [binomial_count].
#' @param severity A [grid_distribution], usually from [discretize()].
#' @param points Number of points in the aggregate grid.
#' @param method `"panjer"`, or `"fft"` for large claim counts where Panjer
#'   underflows.
#' @returns A [grid_distribution] whose `report` describes the calculation.
#' @export
#' @examples
#' sev <- grid_distribution(1, c(0.1, 0.3, 0.25, 0.2, 0.1, 0.05))
#' agg <- compound_distribution(poisson_count(3), sev, points = 100)
#' mean(agg)
#' agg@report$tail_mass
compound_distribution <- function(frequency, severity, points, method = c("panjer", "fft")) {
  method <- match.arg(method)
  ptr <- rust_result(compound(frequency@ptr, severity@ptr, as.double(points), method))
  grid_distribution(ptr = ptr)
}

#' FFT grid sizing
#'
#' `recommend_grid()` chooses a grid (bucket size and number of points) for
#' the compound distribution of `frequency` claims of `severity`, as
#' Mildenhall's `aggregate` sizes one when no bucket is given: the larger of
#' a lognormal or gamma fitted to the aggregate's mean and variance at `p`,
#' and one big claim on a typical bulk at `p`; then one big claim at
#' `p_star` when it fits at the same bucket. With an infinite variance the
#' single big jump alone sizes the grid. `compound_auto()` discretizes the
#' severity by rounding on that grid and runs the FFT. `round_bucket()`
#' rounds a bucket size up to `aggregate`'s ladder:
#' `{1, 2, 4, 5, 8} * 10^k` at 1 and above, a power of two below.
#'
#' @param frequency A claim count.
#' @param severity Any severity but a [sampled] distribution.
#' @param log2 At most `2^log2` points.
#' @param p Probability of the moment extent.
#' @param p_star Aggregate probability the single big jump covers.
#' @param bs Positive bucket sizes.
#' @returns `recommend_grid()`: a list with `step`, `points`, `extent`,
#'   `method` (`"moments"` or `"single_big_jump"`), `moment_extent` (`NaN`
#'   with an infinite variance), `jump_extent` and `tail_estimate` (roughly
#'   the probability beyond the grid from one claim; raise `log2` when it is
#'   not small). `compound_auto()`: a [grid_distribution] whose `report`
#'   also holds that list as `sizing`. `round_bucket()`: a numeric vector.
#' @export
#' @examples
#' g <- recommend_grid(poisson_count(10), gamma_distribution(1 / 0.49, 50 * 0.49))
#' g$step
#' agg <- compound_auto(poisson_count(5), lognormal_from_mean_cv(100, 1))
#' mean(agg)
#' agg@report$sizing$points
#' round_bucket(c(3.4, 0.3))
recommend_grid <- function(frequency, severity, log2 = 16, p = 1 - 1e-5, p_star = 1 - 1e-12) {
  rust_result(recommend_grid_rust(frequency@ptr, severity@ptr, as.double(log2),
                                  as.double(p), as.double(p_star)))
}

#' @rdname recommend_grid
#' @export
compound_auto <- function(frequency, severity, log2 = 16, p = 1 - 1e-5, p_star = 1 - 1e-12) {
  ptr <- rust_result(compound_auto_rust(frequency@ptr, severity@ptr, as.double(log2),
                                        as.double(p), as.double(p_star)))
  grid_distribution(ptr = ptr)
}

#' @rdname recommend_grid
#' @export
round_bucket <- function(bs) rust_result(round_bucket_rust(as.double(bs)))

#' Grids for several units, and the grid audit
#'
#' `recommend_portfolio_grid()` sizes one grid for the sum of independent
#' compound units, as `aggregate` sizes a `Portfolio`: [recommend_grid()]'s
#' steps on the total, the single big jump taken on whichever unit needs the
#' most room.
#'
#' `audit_grid()` audits a grid as `aggregate` 1.2.0's `sharpen` does. It
#' builds the compound distribution at the center and, unless that meets the
#' target, at doubled and halved steps on the same and half the points, until
#' the score stops improving. Each cell is scored by six moment errors (the
#' severity's and the aggregate's mean, cv and skewness against their exact
#' values, divided by `eps`, `10 eps` and `100 eps`) combined by a power mean,
#' so a score of 1 is `aggregate`'s validation boundary. The best sound cell
#' (one that loses no mass off the grid) is chosen, preferring fewer points;
#' twice the points is tried only when nothing smaller meets the target.
#' `audit_portfolio_grid()` does the same for a sum of units: the severity
#' terms score the claim-weighted mixture of the units' severities, the
#' aggregate terms the total.
#'
#' @param units A list of units, each `list(frequency, severity)`.
#' @param frequency A claim count.
#' @param severity Any severity but a [sampled] distribution.
#' @param step The center's step; by default [recommend_grid()]'s (or
#'   `recommend_portfolio_grid()`'s) at `log2`.
#' @param log2 The center's `2^log2` points (16 by default); with no `step`,
#'   the most points the recommended grid may use.
#' @param p,p_star As [recommend_grid()].
#' @param eps The validation tolerance.
#' @param power The power mean of the terms.
#' @param good_enough The target score.
#' @param max_doublings Steps doubled or halved at most this many times each way.
#' @param log2_cap `log2` never grows past it.
#' @param log2_floor The half-size row is tried only above it.
#' @returns `recommend_portfolio_grid()`: a list as [recommend_grid()]'s.
#'   `audit_grid()` and `audit_portfolio_grid()`: a list with `cells` (a
#'   data frame, the center first: `d_step` and `d_log2`, the moves from the
#'   center in doublings, `step`, `log2`, `score`, the six terms
#'   `severity_mean` to `aggregate_skew` (`NaN` when skipped), `tail_mass`,
#'   `aliasing_error`, `defective`, `selected`, `note`), `selected` (the
#'   chosen row), `step`, `log2` and `score` of the chosen cell, `probed`
#'   (`FALSE` when the center met the target), `grew` and `reason`.
#' @export
#' @examples
#' a <- audit_grid(poisson_count(50), gamma_distribution(4, 2.5), step = 0.5, log2 = 10)
#' c(a$step, a$log2)
#' a$reason
#' units <- list(list(poisson_count(10), gamma_distribution(2, 50)),
#'               list(poisson_count(2), gamma_distribution(0.5, 1000)))
#' recommend_portfolio_grid(units, log2 = 12)$points
#' audit_portfolio_grid(units, log2 = 12)$score
audit_grid <- function(frequency, severity, step = NULL, log2 = NULL, eps = 1e-4, power = 2,
                       good_enough = 0.5, max_doublings = 4, log2_cap = 20, log2_floor = 8) {
  settings <- as.double(c(eps, power, good_enough, max_doublings, log2_cap, log2_floor))
  audit_result(rust_result(audit_grid_rust(frequency@ptr, severity@ptr, null_nan(step),
                                           null_nan(log2), settings)))
}

#' @rdname audit_grid
#' @export
audit_portfolio_grid <- function(units, step = NULL, log2 = NULL, eps = 1e-4, power = 2,
                                 good_enough = 0.5, max_doublings = 4, log2_cap = 20,
                                 log2_floor = 8) {
  settings <- as.double(c(eps, power, good_enough, max_doublings, log2_cap, log2_floor))
  audit_result(rust_result(audit_portfolio_grid_rust(unit_ptrs(units, 1L), unit_ptrs(units, 2L),
                                                     null_nan(step), null_nan(log2), settings)))
}

#' @rdname audit_grid
#' @export
recommend_portfolio_grid <- function(units, log2 = 16, p = 1 - 1e-5, p_star = 1 - 1e-12) {
  rust_result(recommend_portfolio_grid_rust(unit_ptrs(units, 1L), unit_ptrs(units, 2L),
                                            as.double(log2), as.double(p), as.double(p_star)))
}

null_nan <- function(x) if (is.null(x)) NaN else as.double(x)

unit_ptrs <- function(units, k) {
  lapply(units, function(u) {
    if (length(u) != 2) stop("each unit is list(frequency, severity)", call. = FALSE)
    u[[k]]@ptr
  })
}

audit_result <- function(r) {
  r$cells <- as.data.frame(r$cells, stringsAsFactors = FALSE)
  r
}

#' Simulated years of losses
#'
#' Simulates `n_sims` years: a claim count from `frequency`, then that many
#' independent losses from `severity`. Year `i` uses only stream `i` of the
#' generator keyed by `seed`, so results do not depend on the number of
#' threads and match Python and Rust.
#'
#' @param frequency A [poisson_count], [negative_binomial_count] or [binomial_count].
#' @param severity A [lognormal], [grid_distribution] or Pareto-family severity
#'   ([pareto], [piecewise_pareto], [log_affine_pareto], [generalized_pareto]).
#' @param n_sims Number of simulated years.
#' @param seed Generator seed: a non-negative whole number below 2^53.
#' @returns An `event_set`. Use [events()] for one year's losses,
#'   [event_counts()] for the number of losses per year, [total()] for the
#'   annual totals as a [predictive_distribution], and [apply_tower()] for
#'   reinsurance.
#' @export
#' @examples
#' ev <- simulate_events(poisson_count(5), lognormal_from_mean_cv(1000, 1), 10000, seed = 42)
#' ev@n_sims
#' mean(total(ev))
simulate_events <- function(frequency, severity, n_sims, seed) {
  ptr <- rust_result(EventSet$simulate(frequency@ptr, severity@ptr, as.double(n_sims), as.double(seed)))
  event_set(ptr = ptr)
}

#' The occurrence exceedance (OEP) curve and occurrence PMLs
#'
#' `occurrence_exceedance()` is the chance that a year's largest occurrence
#' exceeds `x`, `1 - G_N(F(x))` with `G_N` the count's generating function
#' (`1 - exp(-lambda * S(x))` for a Poisson count). `occurrence_pml()`
#' inverts it: the loss whose occurrence exceedance is the annual probability
#' `p`, for any claim count (the Python package `aggregate`'s `oep` covers
#' Poisson counts). `annual_maxima()` is each simulated year's largest loss,
#' whose survival function is the simulated OEP curve.
#'
#' @param frequency A claim count, as for [simulate_events()].
#' @param severity A severity, as for [simulate_events()].
#' @param x Losses.
#' @param p Annual probabilities in `(0, 1 - P(N = 0))`.
#' @param events An `event_set`.
#' @returns `occurrence_exceedance()`: probabilities. `occurrence_pml()`: a
#'   data frame with `probability`, `loss`, `severity_survival`,
#'   `occurrence_return_period` (`1 / (E[N] S(loss))`, which can be shorter
#'   than a year) and `annual_return_period` (`1 / p`). `annual_maxima()`: a
#'   [predictive_distribution].
#' @export
#' @examples
#' sev <- lognormal_from_mean_cv(100, 2)
#' occurrence_pml(poisson_count(2), sev, c(0.1, 0.01, 0.001))
#' occurrence_exceedance(negative_binomial_count(1.5, 2), sev, c(500, 1000))
#' ev <- simulate_events(poisson_count(2), sev, 10000, seed = 1)
#' mean(annual_maxima(ev))
occurrence_pml <- function(frequency, severity, p) {
  as.data.frame(rust_result(occurrence_pml_rust(frequency@ptr, severity@ptr, as.double(p))))
}

#' @rdname occurrence_pml
#' @export
occurrence_exceedance <- function(frequency, severity, x) {
  rust_result(occurrence_exceedance_rust(frequency@ptr, severity@ptr, as.double(x)))
}

#' @rdname occurrence_pml
#' @export
annual_maxima <- function(events) {
  predictive_distribution(ptr = rust_result(events@ptr$maxima()))
}

#' Eventual ruin by the Pollaczeck–Khinchine formula
#'
#' The probability `psi(u)` that the classical surplus process, Poisson
#' claims with this severity and a premium of `1 + rho` times the expected
#' loss, ever falls below zero from initial surplus `u`, as the Python
#' package `aggregate`'s `pollaczeck_khinchine` and `eventual_ruin`. The
#' Poisson rate cancels. The maximal aggregate loss is a compound geometric
#' of the equilibrium severity `S(x) / E[X]`, by FFT; `psi(u)` is its
#' survival function on the severity's grid.
#'
#' @param severity A [grid_distribution].
#' @param rho The margin to loss, positive. Or give `loss_ratio`.
#' @param loss_ratio `1 / (1 + rho)`, in `(0, 1)`.
#' @param p Ruin probabilities at which to find the capital.
#' @param points Grid points for `psi`; the severity's by default.
#' @param cap,excess,stop_loss At most one: use `X | X <= cap`,
#'   `X | X > excess` (not shifted), or `X` with the mass above `stop_loss`
#'   moved onto it.
#' @returns A list: `psi`, a data frame of `u` and `psi` on the grid (the
#'   last row is the mass beyond it); `capital` and `capital_interpolated`
#'   at each `p` (the smallest grid point with `psi(u) < p`, or linear
#'   between grid points; `NA` where the grid does not reach it);
#'   `lundberg_exponent` (`R` with `psi(u) <= exp(-R u)`, `NA` when none on
#'   the grid), `severity_mean` and `aliasing_error`.
#' @export
#' @examples
#' sev <- discretize(gamma_from_mean_cv(100, 1), step = 1, points = 2^14, method = "rounding")
#' ruin <- eventual_ruin(sev, rho = 0.2, p = c(0.1, 0.01))
#' ruin$capital
#' # Exponential claims: exp(-1000 / 600) / 1.2 = 0.157; the step costs 1%.
#' ruin$psi$psi[ruin$psi$u == 1000]
eventual_ruin <- function(severity, rho = NULL, loss_ratio = NULL, p = numeric(), points = NULL,
                          cap = NULL, excess = NULL, stop_loss = NULL) {
  if (is.null(rho) == is.null(loss_ratio)) stop("give exactly one of rho and loss_ratio")
  if (is.null(rho)) {
    if (!(loss_ratio > 0 && loss_ratio < 1)) stop("the loss ratio must be in (0, 1)")
    rho <- 1 / loss_ratio - 1
  }
  parts <- list(cap = cap, excess = excess, stop_loss = stop_loss)
  parts <- parts[!vapply(parts, is.null, logical(1))]
  if (length(parts) > 1) stop("give at most one of cap, excess and stop_loss")
  part <- if (length(parts)) names(parts) else "all"
  value <- if (length(parts)) as.double(parts[[1]]) else 0
  r <- rust_result(eventual_ruin_rust(
    severity@ptr, as.double(rho), as.double(if (is.null(points)) 0 else points), part, value,
    as.double(p)
  ))
  list(
    psi = data.frame(u = r$u, psi = r$psi), capital = r$capital,
    capital_interpolated = r$capital_interpolated,
    lundberg_exponent = r$lundberg_exponent, severity_mean = r$severity_mean,
    aliasing_error = r$aliasing_error
  )
}

#' Ruin within a finite horizon
#'
#' The probability `psi(u, t)` that the surplus, starting at `u`, earning
#' `premium` a period and paying each period's aggregate loss at its end,
#' is negative at the end of some period up to `t`. It is found by the
#' recursion `psi_t(u) = P(S > u + c) + E[psi_{t-1}(u + c - S); S <= u + c]`
#' on the severity's grid, the convolution by FFT, so it holds for any
#' claim count a period. [eventual_ruin()] is the limit for Poisson claims
#' checked continuously; give a Poisson `rate` with `sub_periods` to check
#' `sub_periods` times a period instead (each step Poisson with
#' `rate / sub_periods`), which approaches it from below as the steps
#' shrink.
#'
#' The premium a step, `premium / sub_periods`, must be a whole number of
#' grid steps.
#'
#' @param severity A [grid_distribution].
#' @param premium The premium a period.
#' @param periods The horizon in periods.
#' @param frequency The claim count a period. Or give `rate`.
#' @param rate A Poisson rate a period.
#' @param sub_periods Steps a period, with `rate`.
#' @param points Grid points for `psi`; the severity's by default. Beyond
#'   the grid `psi` is cut off, so raise it until `psi` at the last point
#'   is small.
#' @param p Ruin probabilities at which to find the capital.
#' @returns A list: `u`, the grid; `psi`, a matrix with one row per `u` and
#'   one column per period; `capital` and `capital_interpolated`, matrices
#'   with one row per `p` and one column per period (the smallest grid
#'   point with `psi(u, t) < p`, or linear between grid points; `NA` where
#'   the grid does not reach it); `step` and `aliasing_error`.
#' @seealso [eventual_ruin()]
#' @export
#' @examples
#' # A claim of 2 with probability 0.4 a period, premium 1: a random walk.
#' walk <- finite_horizon_ruin(grid_distribution(1, c(0, 0, 1)), 1, 3,
#'   frequency = bernoulli_count(0.4), points = 16)
#' walk$psi[1, ] # 0.4, 0.4 + 0.6 * 0.4^2 by the third period
#' sev <- discretize(gamma_from_mean_cv(1, 1), step = 0.05, points = 2^12, method = "rounding")
#' ruin <- finite_horizon_ruin(sev, 1.5, 10, rate = 1, sub_periods = 10, p = 0.05)
#' ruin$capital
finite_horizon_ruin <- function(severity, premium, periods, frequency = NULL, rate = NULL,
                                sub_periods = 1, points = NULL, p = numeric()) {
  if (is.null(frequency) == is.null(rate)) stop("give exactly one of frequency and rate")
  if (!is.null(frequency) && sub_periods != 1) {
    stop("sub_periods needs a Poisson rate: give rate, not frequency")
  }
  r <- rust_result(finite_horizon_ruin_rust(
    severity@ptr, as.double(premium), as.double(periods),
    if (is.null(frequency)) NULL else frequency@ptr,
    as.double(if (is.null(rate)) 0 else rate), as.double(sub_periods),
    as.double(if (is.null(points)) 0 else points), as.double(p)
  ))
  n <- length(r$u)
  list(
    u = r$u, psi = matrix(r$psi, nrow = n, ncol = periods),
    capital = matrix(r$capital, nrow = length(p), ncol = periods),
    capital_interpolated = matrix(r$capital_interpolated, nrow = length(p), ncol = periods),
    step = r$step, aliasing_error = r$aliasing_error
  )
}

#' Eventual ruin for renewal claims by Wiener-Hopf
#'
#' The probability `psi(u)` of eventual ruin from initial surplus `u` when
#' the waits between claims follow any law (a renewal, or Sparre Andersen,
#' process), as the Python package `aggregate`'s `wiener_hopf`: claims of
#' this severity, waits drawn from `wait`, and the premium at the rate
#' `(1 + rho) E[X] / E[W]`. Ruin comes at a claim, so `psi(u)` is the
#' chance that the walk with steps `X - cW` ever rises above `u`, found by
#' the cepstral Wiener-Hopf factorization. Exponential waits give
#' [eventual_ruin()].
#'
#' @param severity A [grid_distribution].
#' @param wait A distribution: the wait between claims, with a finite mean.
#' @param rho The margin to loss, positive. Or give `loss_ratio`.
#' @param loss_ratio `1 / (1 + rho)`, in `(0, 1)`.
#' @param p Ruin probabilities at which to find the capital.
#' @param points Grid points for `psi`, at least the severity's (the
#'   default); raise it when `psi` at the last point is not small.
#' @returns A list: `psi`, a data frame of `u`, `psi` and `maximum` (the
#'   probabilities of the claim walk's all-time maximum) on the grid;
#'   `capital` and `capital_interpolated` at each `p` (`NA` where the grid
#'   does not reach it); `premium_rate`, `severity_mean` and `mean_step`
#'   (the grid's mean step of the walk, negative).
#' @export
#' @examples
#' sev <- discretize(gamma_from_mean_cv(10, 0.3), step = 1 / 32, points = 2^14, method = "rounding")
#' # Gamma waits (shape 2) are more regular than Poisson arrivals.
#' wh <- wiener_hopf(sev, gamma_distribution(2, 1), rho = 0.2, p = 0.01)
#' pk <- eventual_ruin(sev, rho = 0.2, p = 0.01)
#' c(wh$capital, pk$capital)
wiener_hopf <- function(severity, wait, rho = NULL, loss_ratio = NULL, p = numeric(), points = NULL) {
  if (is.null(rho) == is.null(loss_ratio)) stop("give exactly one of rho and loss_ratio")
  if (is.null(rho)) {
    if (!(loss_ratio > 0 && loss_ratio < 1)) stop("the loss ratio must be in (0, 1)")
    rho <- 1 / loss_ratio - 1
  }
  r <- rust_result(wiener_hopf_rust(
    severity@ptr, wait@ptr, as.double(rho), as.double(if (is.null(points)) 0 else points),
    as.double(p)
  ))
  list(
    psi = data.frame(u = r$u, psi = r$psi, maximum = r$maximum), capital = r$capital,
    capital_interpolated = r$capital_interpolated, premium_rate = r$premium_rate,
    severity_mean = r$severity_mean, mean_step = r$mean_step
  )
}

#' Two lines hit by the same events, and clash
#'
#' `fft_bivariate()` gives the joint annual loss `(A, B)` of two lines hit by
#' one count of events: each event gives a pair of losses from the joint
#' per-event `severity` (a matrix, rows for line A at `0, step_a, ...`,
#' columns for line B), and mass at 0 on a line is an event that misses it.
#' The joint law is the compound FFT with two-dimensional transforms.
#'
#' `clash_model()` is the clash model of the Python package `aggregate`
#' (`clash`): the expected events a year hitting line A only, line B only
#' and both (the clash count). Independence within an event fixes those
#' hitting neither, `only_a * only_b / both`, so there are `events` in all,
#' each hitting A with probability `trigger_a` and B with `trigger_b`. The
#' dependence between the lines comes only from the shared count: with a
#' Poisson count `Cov(A, B) = both * E[X] * E[Y]`. `clash_severity()` is its
#' per-event severity and `clash_aggregate()` its joint annual loss, with a
#' count whose mean is `events`.
#'
#' @param frequency The shared count of events: a [poisson_count],
#'   [negative_binomial_count], [binomial_count] or claim count.
#' @param severity A matrix of probabilities summing to 1.
#' @param steps The step on line A and on line B.
#' @param points Points on each line of the result.
#' @param only_a,only_b Expected events a year hitting one line only.
#' @param both Expected events a year hitting both lines, positive.
#' @param clash A `clash_model()`.
#' @param a,b [grid_distribution]s: the loss to each line given that the
#'   event hits it.
#' @returns `fft_bivariate()`, `clash_aggregate()` and `clash_severity()`: a
#'   list with `density` (the joint probabilities as a matrix; mass past the
#'   last row or column is carried on it), `steps`, `marginal_a`,
#'   `marginal_b` and `total` ([grid_distribution]s; the total is exact
#'   when the steps are equal, else on the larger step keeping the mean),
#'   `means`, `variances`, `covariance`, `correlation`, and, but for
#'   `clash_severity()`, `report`: `tail_mass` and `aliasing_mass` on each
#'   line (the mass that wrapped around the FFT buffer; request more points
#'   when it is not negligible), `expected_means` and `grid_means`.
#'   `clash_model()`: a list of class `clash_model` with the three counts,
#'   `neither`, `events`, `trigger_a` and `trigger_b`.
#' @export
#' @examples
#' clash <- clash_model(only_a = 2, only_b = 3, both = 1)
#' clash$events
#' a <- grid_distribution(1, c(0, 0.5, 0.5))
#' b <- grid_distribution(1, c(0, 1))
#' joint <- clash_aggregate(clash, poisson_count(clash$events), a, b, points = c(128, 128))
#' joint$covariance # both * E[X] * E[Y] = 1.5
#' joint$correlation
#' mean(joint$total)
fft_bivariate <- function(frequency, severity, steps = c(1, 1), points) {
  severity <- as.matrix(severity)
  points <- rep_len(as.double(points), 2)
  r <- rust_result(fft_bivariate_rust(
    frequency@ptr, as.double(t(severity)), nrow(severity), ncol(severity),
    as.double(steps[1]), as.double(steps[2]), points[1], points[2]
  ))
  joint_result(r)
}

#' @rdname fft_bivariate
#' @export
clash_model <- function(only_a, only_b, both) {
  counts <- as.double(c(only_a, only_b, both))
  r <- rust_result(clash_rust(counts))
  structure(
    list(only_a = counts[1], only_b = counts[2], both = counts[3], neither = r[1],
         events = r[2], trigger_a = r[3], trigger_b = r[4]),
    class = "clash_model"
  )
}

#' @rdname fft_bivariate
#' @export
clash_severity <- function(clash, a, b) {
  joint_result(rust_result(clash_aggregate_rust(clash_counts(clash), NULL, a@ptr, b@ptr, 0, 0)))
}

#' @rdname fft_bivariate
#' @export
clash_aggregate <- function(clash, frequency, a, b, points) {
  points <- rep_len(as.double(points), 2)
  joint_result(rust_result(clash_aggregate_rust(
    clash_counts(clash), frequency@ptr, a@ptr, b@ptr, points[1], points[2]
  )))
}

#' Per-event severity of two lines coupled by a copula
#'
#' The joint per-event severity of two losses: each line is hit with its
#' trigger probability and then loses from its grid (`a` for line A, the
#' rows; `b` for B), so mass at 0 is an event that misses the line; the two
#' per-event marginals are joined by the copula's rectangle probabilities
#' over their cdfs, as the Python package `aggregate`'s copula `bivariate`.
#' The marginals come back exactly. Give its `density` and `steps` to
#' [fft_bivariate()] for the joint annual loss.
#'
#' @param a,b [grid_distribution]s: the loss to each line given that the
#'   event hits it.
#' @param copula A two-dimensional [gaussian_copula()] or
#'   [archimedean_copula()].
#' @param trigger_a,trigger_b The probability that an event hits each line.
#' @returns A list as [fft_bivariate()] returns, without `report`.
#' @export
#' @examples
#' a <- grid_distribution(1, c(0.25, 0.5, 0.25))
#' b <- grid_distribution(1, c(0.5, 0.5))
#' s <- copula_severity(a, b, archimedean_copula("gumbel", 2), trigger_a = 0.5)
#' s$correlation
#' joint <- fft_bivariate(poisson_count(3), s$density, s$steps, points = 64)
#' joint$correlation
copula_severity <- function(a, b, copula, trigger_a = 1, trigger_b = 1) {
  joint_result(rust_result(copula_severity_rust(
    a@ptr, b@ptr, copula@ptr, as.double(trigger_a), as.double(trigger_b)
  )))
}

clash_counts <- function(clash) {
  if (!inherits(clash, "clash_model")) stop("clash must be a clash_model()")
  c(clash$only_a, clash$only_b, clash$both)
}

joint_result <- function(r) {
  out <- list(
    density = matrix(r$probs, nrow = r$rows, ncol = r$cols, byrow = TRUE), steps = r$steps,
    marginal_a = grid_distribution(ptr = r$marginal_a),
    marginal_b = grid_distribution(ptr = r$marginal_b),
    total = grid_distribution(ptr = r$total), means = r$means, variances = r$variances,
    covariance = r$covariance, correlation = r$correlation
  )
  if (!is.null(r$report)) out$report <- r$report
  out
}

#' Layer loss picks
#'
#' Reshapes a severity so that its expected loss to each layer hits a pick,
#' as the Python package `aggregate`'s `Aggregate.picks`. The attachments
#' cut the severity into layers, 0 to the first and then between
#' consecutive ones; in each the survival function is rescaled about its
#' value at the layer's top, `S*(x) = above + weight * (S(x) - S(top))`,
#' keeping its shape there, and above the top attachment the severity is
#' kept. A
#' bottom-layer pick too large for that takes `S* = 1` up to a point and a
#' straight line after, solved so the pick is met.
#'
#' @param severity A [grid_distribution].
#' @param attachments Increasing grid points.
#' @param picks The loss to each layer for `claims` expected ground-up
#'   claims.
#' @param claims The expected claim count the picks are for.
#' @returns A list: `severity`, the adjusted [grid_distribution];
#'   `straight_bottom`; and `layers`, a data frame with `bottom`, `top`,
#'   `before` and `target` (the expected loss per claim before and after),
#'   `weight` and `above` (the adjusted chance of a loss above the layer).
#' @export
#' @examples
#' sev <- discretize(gamma_from_mean_cv(100, 1.5), step = 1, points = 2^14, method = "rounding")
#' picked <- layer_loss_picks(sev, c(100, 500, 1000), c(480, 420, 60), claims = 10)
#' picked$layers
layer_loss_picks <- function(severity, attachments, picks, claims = 1) {
  r <- rust_result(layer_loss_picks_rust(
    severity@ptr, as.double(attachments), as.double(picks), as.double(claims)
  ))
  list(
    severity = grid_distribution(ptr = r$severity), straight_bottom = r$straight_bottom,
    layers = as.data.frame(r$layers)
  )
}

#' Compound moments to third order and moment approximations
#'
#' `compound_moments()` is the mean, variance, third cumulant, cv and
#' skewness of `S = X_1 + ... + X_N`, from the count's cumulants and the
#' severity's first three moments. `moment_approximation()` fits a
#' distribution to them, as the Python package `aggregate`'s `approximate`:
#' `"normal"`, `"gamma"` or `"lognormal"` (mean and cv), `"shifted_gamma"`,
#' `"shifted_lognormal"`, `"normal_power"` or `"wilson_hilferty"` (mean, cv
#' and skewness, which must be positive; the last two are defined by their
#' quantiles, Bahnemann's section 4.4).
#' It takes a count and a severity, or a `compound_moments()` result alone
#' (`moment_approximation(m, "gamma")`).
#' `approximation_cdf()` and `approximation_quantile()` evaluate a fit.
#'
#' @param frequency A claim count, as for [simulate_events()]; for
#'   `moment_approximation()` also a `compound_moments()` result, given
#'   without a severity.
#' @param severity A severity with a closed-form third moment.
#' @param kind The family.
#' @param approximation A fit from `moment_approximation()`.
#' @param x Values.
#' @param p Probabilities.
#' @returns `compound_moments()`: a named vector. `moment_approximation()`: a
#'   list with `kind`, `parameters` (named: `mean` and `sd`; `shape` and
#'   `scale`; `meanlog` and `sdlog`; plus `shift`; `mean`, `sd` and `skew`
#'   for the normal power and Wilson-Hilferty) and `moments`. The others:
#'   numeric vectors.
#' @export
#' @examples
#' m <- compound_moments(poisson_count(10), gamma_distribution(1 / 0.49, 50 * 0.49))
#' m
#' fit <- moment_approximation(poisson_count(10), gamma_distribution(1 / 0.49, 50 * 0.49),
#'                             "shifted_gamma")
#' fit$parameters
#' approximation_quantile(fit, c(0.5, 0.99))
#' moment_approximation(m, "gamma")$parameters
compound_moments <- function(frequency, severity) {
  m <- rust_result(compound_moments_rust(frequency@ptr, severity@ptr))
  stats::setNames(m, c("mean", "variance", "third_cumulant", "cv", "skew"))
}

#' @rdname compound_moments
#' @export
moment_approximation <- function(frequency, severity = NULL,
                                 kind = c("shifted_lognormal", "shifted_gamma", "gamma",
                                          "lognormal", "normal", "normal_power",
                                          "wilson_hilferty")) {
  needed <- c("mean", "variance", "third_cumulant")
  if (is.numeric(frequency)) {
    # A compound_moments() result; the kind may come second, positionally.
    if (is.character(severity) && missing(kind)) {
      kind <- severity
      severity <- NULL
    }
    if (!is.null(severity)) stop("give a compound_moments() result without a severity")
    if (!all(needed %in% names(frequency))) {
      stop("expected a claim count or a compound_moments() result")
    }
    m <- frequency[needed]
  } else {
    if (is.null(severity)) stop("give a severity with the claim count")
    m <- compound_moments(frequency, severity)[needed]
  }
  kind <- match.arg(kind, eval(formals(sys.function())$kind))
  params <- rust_result(approximation_parameters_rust(unname(m), kind))
  list(kind = kind, parameters = unlist(params), moments = m)
}

#' @rdname compound_moments
#' @export
approximation_cdf <- function(approximation, x) {
  rust_result(approximation_eval_rust(unname(approximation$moments), approximation$kind,
                                      as.double(x), FALSE))
}

#' @rdname compound_moments
#' @export
approximation_quantile <- function(approximation, p) {
  rust_result(approximation_eval_rust(unname(approximation$moments), approximation$kind,
                                      as.double(p), TRUE))
}

#' Years of losses from elsewhere
#'
#' Builds an `event_set` from your own simulation, or a catastrophe model's
#' event loss table by year, optionally with the sum insured of the risk
#' each loss hit, which a [surplus_treaty()] needs.
#'
#' @param years A list with one numeric vector of losses per year, in order.
#' @param sums_insured `NULL`, or a list of the same shape: each loss's sum
#'   insured, at least the loss.
#' @param seed Recorded in results' provenance; the samplers are not, since
#'   the losses were not drawn here.
#' @param times `NULL`, or a list of the same shape: each loss's time as the
#'   fraction of the year elapsed (in `[0, 1]`, non-decreasing within a
#'   year), which reinstatements pro rata as to time need.
#' @returns An `event_set`; [event_sums_insured()] and [event_times()] read
#'   the sums insured and times back.
#' @export
#' @examples
#' ev <- events_from_years(list(c(5, 2), numeric(), 9), list(c(10, 2), numeric(), 50))
#' event_counts(ev)
#' event_sums_insured(ev, 3)
events_from_years <- function(years, sums_insured = NULL, seed = 0, times = NULL) {
  years <- lapply(years, as.double)
  if (!is.null(sums_insured)) sums_insured <- lapply(sums_insured, as.double)
  if (!is.null(times)) times <- lapply(times, as.double)
  event_set(ptr = rust_result(EventSet$from_years(years, sums_insured, as.double(seed), times)))
}

#' Date events uniformly over the year
#'
#' The same events at times spread uniformly over the year: each year's
#' losses take sorted uniform draws, in their order, from a stream of the
#' generator keyed by the set's seed apart from the losses' own, so the
#' losses are unchanged and any year replays alone. Reinstatements pro rata
#' as to time (`xol_layer(pro_rata_time = TRUE)`) need them.
#'
#' @param x An `event_set`.
#' @returns An `event_set` whose losses carry times; [event_times()] reads
#'   them.
#' @export
#' @examples
#' ev <- with_uniform_times(simulate_events(poisson_count(3), lognormal(0, 1), 10, seed = 1))
#' event_times(ev, 1)
with_uniform_times <- function(x) event_set(ptr = x@ptr$with_uniform_times())

#' Date events by a season
#'
#' The same events at times drawn from a seasonal density: the year is cut
#' into `length(weights)` equal periods (12 for months, 52 for weeks)
#' starting at the contract's inception, and a loss falls in period `k`
#' with probability `weights[k] / sum(weights)`, uniformly within it. A
#' zero weight means no losses in that period (a hurricane season). The
#' draws are those of [with_uniform_times()], mapped through the season's
#' quantile, so equal weights give the uniform times.
#'
#' @param x An `event_set`.
#' @param weights Each period's relative weight: non-negative, not all
#'   zero.
#' @returns An `event_set` whose losses carry times; [event_times()] reads
#'   them.
#' @export
#' @examples
#' ev <- simulate_events(poisson_count(3), lognormal(0, 1), 10, seed = 1)
#' # Atlantic hurricanes, June to November, for a 1 January contract.
#' hurricanes <- with_seasonal_times(ev, c(0, 0, 0, 0, 0, 1, 2, 6, 8, 3, 1, 0))
#' event_times(hurricanes, 1)
with_seasonal_times <- function(x, weights) {
  event_set(ptr = rust_result(x@ptr$with_seasonal_times(as.double(weights))))
}

#' Times of one year's losses
#'
#' @param x An `event_set` whose losses carry times.
#' @param sim Year number, from 1 to `x@n_sims`.
#' @returns Numeric vector, one time per loss as the fraction of the year
#'   elapsed; empty when the events carry none.
#' @export
event_times <- function(x, sim) rust_result(x@ptr$times(as.double(sim)))

#' Sums insured of one year's losses
#'
#' @param x An `event_set` whose losses carry sums insured.
#' @param sim Year number, from 1 to `x@n_sims`.
#' @returns Numeric vector, one sum insured per loss; empty when the events
#'   carry none.
#' @export
event_sums_insured <- function(x, sim) rust_result(x@ptr$sums_insured(as.double(sim)))

#' Simulated years of losses (class)
#'
#' Created by [simulate_events()].
#'
#' @param ptr Internal.
#' @returns An `event_set` object with read-only properties `n_sims`, `seed`,
#'   `has_times` (whether the losses carry times of loss, see
#'   [event_times()]),
#'   `input_hash` (a hash of the years' source, see [read_ylt()], or `NULL`),
#'   `label_dims` and `label_combinations` (see [with_labels()]),
#'   `has_aggregate_part` and `aggregate_part_labels` (see
#'   [with_aggregate_part()]).
#' @export
#' @keywords internal
event_set <- S7::new_class(
  "event_set",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("EventSet"),
    n_sims = S7::new_property(S7::class_double, getter = function(self) self@ptr$n_sims()),
    seed = S7::new_property(S7::class_double, getter = function(self) self@ptr$seed()),
    has_times = S7::new_property(S7::class_logical, getter = function(self) self@ptr$has_times()),
    input_hash = S7::new_property(S7::class_any, getter = function(self) self@ptr$input_hash()),
    label_dims = S7::new_property(S7::class_character, getter = function(self) self@ptr$label_dims()),
    label_combinations = S7::new_property(S7::class_any, getter = function(self) {
      dims <- self@ptr$label_dims()
      if (!length(dims)) return(NULL)
      as.data.frame(stats::setNames(self@ptr$label_combinations(), dims), stringsAsFactors = FALSE)
    }),
    has_aggregate_part = S7::new_property(
      S7::class_logical, getter = function(self) self@ptr$has_aggregate_part()
    ),
    aggregate_part_labels = S7::new_property(S7::class_any, getter = function(self) {
      if (!self@ptr$has_aggregate_part()) return(NULL)
      dims <- self@ptr$label_dims()
      if (!length(dims)) return(NULL)
      as.data.frame(stats::setNames(self@ptr$aggregate_part_labels(), dims), stringsAsFactors = FALSE)
    })
  ),
  constructor = function(ptr) S7::new_object(S7::S7_object(), ptr = ptr)
)

#' One simulated year's losses
#'
#' @param x An `event_set` from [simulate_events()].
#' @param sim Year number, from 1 to `x@n_sims`.
#' @param ... Unused; for methods.
#' @returns Numeric vector of that year's losses, in the order drawn.
#' @export
#' @examples
#' ev <- simulate_events(poisson_count(2), lognormal(0, 1), 10, seed = 1)
#' events(ev, 1)
events <- S7::new_generic("events", "x", function(x, sim, ...) S7::S7_dispatch())

#' Number of losses in each simulated year
#'
#' @inheritParams events
#' @returns Numeric vector of length `x@n_sims`.
#' @export
#' @examples
#' ev <- simulate_events(poisson_count(2), lognormal(0, 1), 10, seed = 1)
#' event_counts(ev)
event_counts <- S7::new_generic("event_counts", "x", function(x, ...) S7::S7_dispatch())

S7::method(events, event_set) <- function(x, sim, ...) rust_result(x@ptr$events(as.double(sim)), s7_call())
S7::method(event_counts, event_set) <- function(x, ...) x@ptr$counts()
S7::method(total, event_set) <- function(dist, ...) {
  predictive_distribution(ptr = rust_result(dist@ptr$totals(), s7_call()))
}
S7::method(print, event_set) <- function(x, ...) {
  cat(sprintf("<event_set> %d simulated years, seed %s\n", as.integer(x@n_sims), format(x@seed)))
  invisible(x)
}

#' Split simulated years into a contract term
#'
#' The simulations in `n_years` consecutive blocks, for
#' [apply_tower_term()]: year `t` holds simulations `(t - 1) * n + 1` to
#' `t * n`, with `n = x@n_sims / n_years`, so path `j`'s year `t` is
#' simulation `(t - 1) * n + j`. Sums insured and times go with their losses.
#'
#' @param x An [event_set].
#' @param n_years The number of years in the term; must divide `x@n_sims`.
#' @returns A list of `n_years` [event_set]s.
#' @export
#' @examples
#' ev <- simulate_events(poisson_count(2), lognormal(0, 1), 30, seed = 1)
#' length(term_years(ev, 3))
term_years <- function(x, n_years) {
  lapply(rust_result(x@ptr$term_years(as.double(n_years))), event_set)
}

#' Underwriting years with a rank correlation
#'
#' `correlated_years()` gives a term of `n_years` years whose yearly totals
#' (each year's losses and aggregate part) have a rank correlation, set by
#' Iman-Conover, for [apply_tower_term()]. Each year's simulations move as
#' whole years, keeping their events, times, sums insured, labels and
#' aggregate part. `calibrate_year_correlation()` finds the correlation
#' between every pair of years for which the mean of a term's yearly totals
#' has `ratio` times the standard deviation of one year's total, by
#' bisection on the simulations; with the same premium each year that is
#' the ratio of the term's loss ratio's spread to one year's (independent
#' years give `1 / sqrt(n_years)`, identical ones 1).
#' `year_correlation_for_ratio()` is the closed form for Pearson's
#' correlation of the totals, `(n_years * ratio^2 - 1) / (n_years - 1)`, a
#' start.
#'
#' @param x An [event_set].
#' @param n_years The number of years, at least 2.
#' @param correlation One correlation for every pair of years, or the full
#'   matrix.
#' @param kind `"spearman"` (Spearman's rho of the totals) or `"pearson"`
#'   (the correlation of their normal scores, as [reorder_groups()]).
#' @param draws `"same"`: every year reorders all the simulations, so each
#'   has exactly the one-year distribution and the term's mean the one-year
#'   mean; `"blocks"`: year `t` is block `t` of [term_years()].
#' @param seed The same seed gives the same years.
#' @param ratio The target ratio, in `[0, 1]`.
#' @returns `correlated_years()` a list of `n_years` [event_set]s;
#'   `calibrate_year_correlation()` a list of `correlation`, the `ratio` it
#'   achieves and the `years`; `year_correlation_for_ratio()` a number.
#' @export
#' @examples
#' ev <- simulate_events(poisson_count(2), lognormal(4, 1), 4000, seed = 7)
#' years <- correlated_years(ev, 3, 0.5, seed = 11)
#' fit <- calibrate_year_correlation(ev, 3, 0.75, seed = 11)
#' c(fit$correlation, fit$ratio)
#' year_correlation_for_ratio(3, 0.75)
correlated_years <- function(x, n_years, correlation, kind = "spearman", draws = "same",
                             seed = 0) {
  correlation <- if (is.matrix(correlation)) as.double(t(correlation)) else as.double(correlation)
  lapply(rust_result(x@ptr$correlated_years(as.double(n_years), correlation, kind, draws,
                                            as.double(seed))), event_set)
}

#' @rdname correlated_years
#' @export
calibrate_year_correlation <- function(x, n_years, ratio, kind = "spearman", draws = "same",
                                       seed = 0) {
  r <- rust_result(x@ptr$calibrate_year_correlation(as.double(n_years), as.double(ratio), kind,
                                                    draws, as.double(seed)))
  r$years <- lapply(r$years, event_set)
  r
}

#' @rdname correlated_years
#' @export
year_correlation_for_ratio <- function(n_years, ratio) {
  rust_result(year_correlation_for_ratio_rust(as.double(n_years), as.double(ratio)))
}

#' Timeline of a term shared by its underwriting years
#'
#' A timeline holds the events of a whole term of accident months once:
#' simulation `j` is one path, and each event's time, a fraction of the
#' term, puts it in accident month `floor(time * months)` (the last month
#' for time 1). `timeline_from_years()` stitches one [event_set] per
#' accident year, each with times within its year (from [term_years()] or
#' [correlated_years()], say): year `t`'s event at time `tau` is at
#' `(t - 1 + tau) / T` of the term.
#'
#' `timeline_years()` splits it into contract years for
#' [apply_tower_term()]. Under `"losses_occurring"`, contract year `k` takes
#' every event in accident months `12 (k - 1) + 1` to `12 k` whole. Under
#' `"risks_attaching"`, underwriting year `u` takes `exposure[u, m]` of each
#' event in month `m`, sums insured scaled alike, at its time over the
#' months the year has a share; the same event can fall to several years.
#' [apply_tower_timeline()] does both steps under a tower's basis.
#'
#' An aggregate part of the events ([with_aggregate_part()]) has no time.
#' The timeline keeps it by accident period (each accident year of
#' `timeline_from_years()`, the whole term of `timeline()`) and spreads it
#' over the period's months by `aggregate_pattern`: a month's weight over
#' the period's weights. Each month's amount falls to the contract years
#' as an event in that month would, so a quota share or stop loss in each
#' year sees its part.
#'
#' `uniform_exposure()` writes each of `n_years` underwriting years evenly
#' over its twelve months, each policy in force for `policy_months` from
#' the middle of its month: a year's share of a month is its policies' time
#' in force there over `policy_months`, so a month in mid-term has shares
#' adding to 1 and the exposure runs `12 * n_years + policy_months` months.
#'
#' @param events An [event_set] with times. An aggregate part is the
#'   whole term's.
#' @param months The number of accident months.
#' @param aggregate_pattern Twelve non-negative weights, one for each month
#'   of an accident year, that spread the aggregate part over the months
#'   (they need not add to 1); `NULL` for uniform.
#' @param ptr Internal: the Rust timeline.
#' @param years A list of [event_set]s, one per accident year; each one's
#'   aggregate part stays its accident year's.
#' @param timeline A `timeline`.
#' @param basis `"losses_occurring"` or `"risks_attaching"`.
#' @param exposure A matrix of shares in `[0, 1]`, one row per underwriting
#'   year, one column per accident month, each column adding to at most 1.
#' @param sources Also return each year's sources: for each of its losses,
#'   in order, its event's index among all the timeline's losses, from 1.
#' @param n_years,policy_months The number of underwriting years and each
#'   policy's months in force.
#' @returns `timeline()` and `timeline_from_years()` a `timeline` with
#'   properties `events` (without the aggregate part), `months`, `n_sims`,
#'   `aggregate_pattern` and `has_aggregate_part`; `timeline_years()` a list
#'   of [event_set]s, one per contract year, or with `sources` a list of
#'   `years` and `sources`; `uniform_exposure()` a matrix.
#' @export
#' @examples
#' ev <- events_from_years(list(100), times = list(13.5 / 24))
#' shares <- matrix(0, 2, 24)
#' shares[1, 14] <- 0.25
#' shares[2, 14] <- 0.75
#' rad <- timeline_years(timeline(ev, 24), "risks_attaching", shares)
#' c(mean(total(rad[[1]])), mean(total(rad[[2]])))
#' e <- uniform_exposure(2, 12)
#' dim(e)
#' colSums(e)[13]
timeline <- S7::new_class(
  "timeline",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("Timeline"),
    events = S7::new_property(event_set, getter = function(self) event_set(self@ptr$events())),
    months = S7::new_property(S7::class_double, getter = function(self) self@ptr$months()),
    n_sims = S7::new_property(S7::class_double, getter = function(self) self@events@n_sims),
    aggregate_pattern = S7::new_property(
      S7::class_double,
      getter = function(self) self@ptr$aggregate_pattern()
    ),
    has_aggregate_part = S7::new_property(
      S7::class_logical,
      getter = function(self) self@ptr$has_aggregate_part()
    )
  ),
  constructor = function(events, months, aggregate_pattern = NULL, ptr = NULL) {
    if (is.null(ptr)) {
      ptr <- rust_result(Timeline$new(events@ptr, as.double(months), pattern_or_null(aggregate_pattern)))
    }
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

pattern_or_null <- function(pattern) if (is.null(pattern)) NULL else as.double(pattern)

#' @rdname timeline
#' @export
timeline_from_years <- function(years, aggregate_pattern = NULL) {
  ptr <- rust_result(Timeline$from_years(lapply(years, function(y) y@ptr), pattern_or_null(aggregate_pattern)))
  timeline(ptr = ptr)
}

#' @rdname timeline
#' @export
timeline_years <- function(timeline, basis, exposure, sources = FALSE) {
  exposure <- as.matrix(exposure)
  r <- rust_result(timeline@ptr$years(
    as.character(basis), as.double(t(exposure)), as.double(nrow(exposure)), isTRUE(sources)
  ))
  if (isTRUE(sources)) {
    r$years <- lapply(r$years, event_set)
    r
  } else {
    lapply(r, event_set)
  }
}

#' @rdname timeline
#' @export
uniform_exposure <- function(n_years, policy_months) {
  r <- rust_result(exposure_uniform_rust(as.double(n_years), as.double(policy_months)))
  matrix(r$shares, nrow = n_years, ncol = r$months, byrow = TRUE)
}

S7::method(print, timeline) <- function(x, ...) {
  cat(sprintf("<timeline> %d paths over %d months\n", as.integer(x@n_sims), as.integer(x@months)))
  invisible(x)
}

#' Year loss table in long form
#'
#' Years of losses from a year loss table: one row per loss with its year,
#' from 1 to `n_years`. Within a year the losses keep their order; years with
#' no row are empty.
#'
#' @param years The year of each loss, from 1.
#' @param losses The losses.
#' @param n_years The number of simulated years, including those with no
#'   loss.
#' @param seed Recorded in results' provenance.
#' @param labels `NULL`, or each row's labels as [with_labels()] takes them:
#'   a named list or data frame with a vector per label dimension, a value
#'   per row (`list(peril = ..., class = ...)`).
#' @returns An [event_set].
#' @export
#' @examples
#' ev <- event_set_from_long(c(3, 1, 3), c(5, 1, 7), 4)
#' event_counts(ev)
#' ev <- event_set_from_long(c(3, 1, 3), c(5, 1, 7), 4,
#'                           labels = list(peril = c("wind", "quake", "wind")))
#' event_labels(ev, 1)
event_set_from_long <- function(years, losses, n_years, seed = 0, labels = NULL) {
  l <- label_columns(labels)
  event_set(ptr = rust_result(EventSet$from_long(as.double(years), as.double(losses),
                                                 as.double(n_years), as.double(seed),
                                                 l$dims, l$values)))
}

#' Read a year loss table
#'
#' Years of events from a vendor's year loss table, one row per non-zero
#' event loss, checked and expanded to a run's simulations. `read_ylt()`
#' reads a CSV or Parquet file (Parquet needs the arrow package);
#' `event_set_from_ylt()` takes the columns.
#'
#' The rows are checked: losses finite and non-negative, years from 1, no two
#' rows with the same year, slot and values in the `unique_by` labels, labels
#' among the `expected` values, times within the year. They are ordered by
#' year, then time of loss when given, then slot, the file's order breaking
#' ties. Each row is one loss: rows sharing a slot, one
#' event hitting several classes, stay separate losses, so an occurrence
#' layer sees each of them. `read_ylt()` hashes the file's bytes into the
#' events' `input_hash`, which results keep in their [provenance()].
#'
#' @param path The file.
#' @param year,loss The columns of each row's year, from 1, and its loss.
#' @param slot The column of each row's event slot, or `NULL` to keep the
#'   file's order within a year and skip the check for repeats.
#' @param time The column of each row's time of loss, the fraction of its
#'   year elapsed, in `[0, 1]`; `NULL` for none. The events then carry times
#'   ([event_times()]), for [hours_clause()] and reinstatements pro rata as
#'   to time.
#' @param day,hour Instead of `time`: the columns of each row's day of the
#'   year, from 1 to 365, and hour of the day, from 0 to 24 (`NULL` for the
#'   start of the day); the time is `((day - 1) * 24 + hour) / 8760`, on the
#'   8,760-hour year of the hours clause.
#' @param labels Columns kept as labels, `c("peril", "class")` say, by which
#'   [apply_tower()] splits its results.
#' @param n_years The table's years, at least its largest year; `NULL` for
#'   the largest year.
#' @param n_sims The simulations to expand to; `NULL` for `n_years`.
#' @param expand `"tile"`: the simulations take the table's years in turn,
#'   so `n_sims` must be a multiple of `n_years`. `"resample"`: each
#'   simulation takes a year drawn uniformly with replacement from its own
#'   stream of `seed`.
#' @param seed The seed of `"resample"`.
#' @param unique_by With slots, the labels within which a year's slot may not
#'   repeat (`"peril"`: one slot may hit several classes); `NULL` for every
#'   label.
#' @param expected `NULL`, or a named list of the values allowed in each
#'   label (`list(peril = c("WS", "EQ"))`).
#' @param format `"csv"` or `"parquet"`; `NULL` for Parquet when the file
#'   ends in `.parquet` or `.pq`, CSV otherwise.
#' @param years,losses,slots Each row's year (from 1), loss and slot
#'   (`NULL` for none).
#' @param times `NULL`, or each row's time of loss, a fraction of the year.
#' @param source_hash `NULL`, or a hash of the source kept as the events'
#'   `input_hash`.
#' @returns An [event_set].
#' @export
#' @examples
#' path <- tempfile(fileext = ".csv")
#' writeLines(c("year,peril,slot,value", "2,wind,9,5", "1,quake,1,1", "2,wind,3,7"), path)
#' ev <- read_ylt(path, labels = "peril", n_sims = 4)
#' events(ev, 2)
#' event_labels(ev, 2)
#' ev@input_hash
#' writeLines(c("year,day,hour,value", "1,200,6,5", "1,199,12,1"), path)
#' timed <- read_ylt(path, slot = NULL, day = "day", hour = "hour")
#' events(timed, 1)
#' event_times(timed, 1)
read_ylt <- function(path, year = "year", loss = "value", slot = "slot", time = NULL,
                     day = NULL, hour = NULL, labels = NULL, n_years = NULL, n_sims = NULL,
                     expand = "tile", seed = 0, unique_by = NULL, expected = NULL,
                     format = NULL) {
  path <- normalizePath(path, mustWork = TRUE)
  if (is.null(format)) {
    format <- if (grepl("\\.(parquet|pq)$", tolower(path))) "parquet" else "csv"
  }
  if (!is.null(time) && (!is.null(day) || !is.null(hour))) {
    stop("give the time column, or the day and hour columns, not both", call. = FALSE)
  }
  if (!is.null(hour) && is.null(day)) stop("an hour column needs a day column", call. = FALSE)
  columns <- c(year, loss, slot, time, day, hour, labels)
  df <- switch(format,
    csv = utils::read.csv(path, colClasses = "character", check.names = FALSE,
                          fileEncoding = "UTF-8-BOM"),
    parquet = {
      if (!requireNamespace("arrow", quietly = TRUE)) {
        stop("reading a Parquet year loss table needs the arrow package", call. = FALSE)
      }
      as.data.frame(arrow::read_parquet(path))
    },
    stop('format must be "csv" or "parquet"', call. = FALSE)
  )
  missing <- setdiff(columns, names(df))
  if (length(missing)) stop(sprintf("%s: no column '%s'", path, missing[1]), call. = FALSE)
  times <- if (!is.null(time)) {
    as.numeric(df[[time]])
  } else if (!is.null(day)) {
    d <- as.numeric(df[[day]])
    h <- if (is.null(hour)) rep(0, length(d)) else as.numeric(df[[hour]])
    bad <- which(!(d == round(d) & d >= 1 & d <= 365))
    if (length(bad)) {
      stop(sprintf("row %d: the day %s is not a whole day from 1 to 365", bad[1], d[bad[1]]),
           call. = FALSE)
    }
    bad <- which(!(h >= 0 & h <= 24))
    if (length(bad)) {
      stop(sprintf("row %d: the hour %s is not from 0 to 24", bad[1], h[bad[1]]), call. = FALSE)
    }
    ((d - 1) * 24 + h) / 8760
  }
  event_set_from_ylt(
    as.numeric(df[[year]]), as.numeric(df[[loss]]),
    slots = if (is.null(slot)) NULL else as.numeric(df[[slot]]),
    times = times,
    labels = if (length(labels)) lapply(df[labels], as.character) else NULL,
    n_years = n_years, n_sims = n_sims, expand = expand, seed = seed,
    unique_by = unique_by, expected = expected,
    source_hash = rust_result(ylt_file_hash_rust(path))
  )
}

#' @rdname read_ylt
#' @export
event_set_from_ylt <- function(years, losses, slots = NULL, labels = NULL, n_years = NULL,
                               n_sims = NULL, expand = "tile", seed = 0, unique_by = NULL,
                               expected = NULL, source_hash = NULL, times = NULL) {
  l <- label_columns(labels)
  e <- label_columns(expected)
  null_or <- function(x) if (is.null(x)) NULL else as.double(x)
  event_set(ptr = rust_result(EventSet$from_ylt(
    as.double(years), as.double(losses),
    if (is.null(slots)) NULL else as.double(slots), null_or(times),
    l$dims, l$values, null_or(n_years), null_or(n_sims), expand, as.double(seed),
    if (is.null(unique_by)) NULL else as.character(unique_by),
    e$dims, e$values, source_hash
  )))
}

#' Label the events
#'
#' The same events with labels, a peril or a class say. [apply_tower()] then
#' gives gross, each layer's ceded loss and net by label, with the label
#' dimensions after `kind` and `layer`; [merge_event_sets()] keeps the labels
#' of either set. Replaces any labels the set had.
#'
#' @param x An [event_set].
#' @param labels A named list or data frame with a vector per label
#'   dimension, a value per loss, years in order as [events()] lists them.
#'   The names may not be `"kind"` or `"layer"`.
#' @returns `with_labels()`: an [event_set] (its `label_dims` property names
#'   the dimensions and `label_combinations` is a data frame of the distinct
#'   combinations, sorted). `event_labels()`: a data frame of year `sim`'s
#'   labels, a row per loss, or `NULL` when the events carry none.
#' @export
#' @examples
#' ev <- with_labels(events_from_years(list(c(5, 2), 9)), list(peril = c("wind", "flood", "wind")))
#' ev@label_combinations
#' event_labels(ev, 1)
with_labels <- function(x, labels) {
  l <- label_columns(labels)
  if (!length(l$dims)) stop("give at least one label dimension", call. = FALSE)
  event_set(ptr = rust_result(x@ptr$with_labels(l$dims, l$values)))
}

#' @rdname with_labels
#' @param sim Year number, from 1 to `x@n_sims`.
#' @export
event_labels <- function(x, sim) {
  v <- rust_result(x@ptr$labels(as.double(sim)))
  if (!length(v)) return(NULL)
  as.data.frame(stats::setNames(v, x@ptr$label_dims()), stringsAsFactors = FALSE)
}

#' An aggregate part beside the events
#'
#' The same events with an aggregate part: amounts in each year that are no
#' event and have no time, attritional losses say, one column per component
#' of `part`. Simulation `i` of `part` goes with year `i`. Its dimensions are
#' label dimensions, so the part can be given by peril or class: the set's
#' labels get the part's dimensions not among them, with `""` for every loss.
#' A part with no dimensions is one unlabelled amount a year. Replaces any
#' aggregate part the set had; label the events first ([with_labels()]).
#'
#' [apply_tower()] counts the part in gross and net, and only layers that see
#' it (the `sees_aggregate_part` property of an [xol_layer()]: quota shares
#' and stop losses) cede from it; per-occurrence layers see the events alone.
#' In each year it comes before the events, so a binding annual deductible is
#' used up on it first, unless the layer shares annual terms pro rata.
#' The events' [total()] includes it; [event_counts()] and the occurrence
#' maxima do not.
#'
#' @param x An [event_set].
#' @param part A [predictive_distribution] with `x@n_sims` simulations and
#'   non-negative draws.
#' @returns `with_aggregate_part()`: an [event_set] (its `aggregate_part_labels`
#'   property is a data frame of each column's labels, `NULL` without labels).
#'   `aggregate_part()`: year `sim`'s amounts, one per column, or `NULL`
#'   without a part.
#' @export
#' @examples
#' attritional <- predictive_distribution(matrix(c(10, 40, 30)), data.frame(class = "attritional"))
#' ev <- with_aggregate_part(events_from_years(list(numeric(0), 50, c(5, 60))), attritional)
#' tower <- reinsurance_tower(list(xol_layer("30x25", 30, 25)))
#' r <- apply_tower(tower, ev)
#' ev@aggregate_part_labels
#' aggregate_part(ev, 2)
with_aggregate_part <- function(x, part) {
  event_set(ptr = rust_result(x@ptr$with_aggregate_part(part@ptr)))
}

#' @rdname with_aggregate_part
#' @param sim Year number, from 1 to `x@n_sims`.
#' @export
aggregate_part <- function(x, sim) {
  if (!x@ptr$has_aggregate_part()) return(NULL)
  rust_result(x@ptr$aggregate_part(as.double(sim)))
}

# Label dimensions and their values as character vectors.
label_columns <- function(labels) {
  if (is.null(labels)) return(list(dims = character(), values = list()))
  labels <- as.list(labels)
  if (is.null(names(labels)) || any(names(labels) == "")) {
    stop("labels must be named, one name per label dimension", call. = FALSE)
  }
  list(dims = names(labels), values = unname(lapply(labels, as.character)))
}

#' Event loss table
#'
#' A catastrophe model's event loss table with secondary uncertainty. Each
#' row is an event with an annual rate, a mean loss, independent and
#' correlated standard deviations, and an exposure, the most it can lose.
#' Events occur as independent Poisson processes; an event's loss, given
#' that it occurs, is a beta distribution on `[0, exposure]` with the row's
#' mean and standard deviation `sd_independent + sd_correlated` (summed, as
#' catastrophe models report them for one portfolio), or the mean exactly
#' when that is 0. A row whose standard deviation no beta on its exposure can
#' have is an error naming the row.
#'
#' `elt_oep()` gives the occurrence exceedance probability
#' `1 - exp(-sum(rate * P(L > x)))` and the annual exceedance frequency;
#' `elt_occurrence_pml()` the loss at an annual probability; `simulate_elt()`
#' simulated years, each loss with the row of its event.
#'
#' @param rate,mean,sd_independent,sd_correlated,exposure The table's
#'   columns, one value per event.
#' @param elt An `event_loss_table`.
#' @param x Losses.
#' @param p Annual probabilities.
#' @param n_years The number of years to simulate.
#' @param seed Seed; year `i` draws from its own stream, so it replays alone.
#' @returns `event_loss_table()`: an `event_loss_table`, a list with the
#'   columns, `rate` (their sum, the mean number of events a year) and
#'   `expected_annual_loss`. `elt_oep()`: a data frame with `x`,
#'   `exceedance_frequency` and `oep`. `elt_occurrence_pml()`: the losses.
#'   `simulate_elt()`: a list with `events`, an [event_set], and `ids`, a list
#'   per year of each loss's event row.
#' @export
#' @examples
#' elt <- event_loss_table(rate = c(0.01, 0.05), mean = c(800, 200),
#'                         sd_independent = c(200, 0), sd_correlated = c(100, 0),
#'                         exposure = c(2000, 500))
#' elt$expected_annual_loss
#' elt_oep(elt, c(150, 1000))
#' elt_occurrence_pml(elt, 0.01)
#' sim <- simulate_elt(elt, 1000, seed = 1)
#' sim$events@n_sims
event_loss_table <- function(rate, mean, sd_independent, sd_correlated, exposure) {
  cols <- list(rate = as.double(rate), mean = as.double(mean),
               sd_independent = as.double(sd_independent),
               sd_correlated = as.double(sd_correlated), exposure = as.double(exposure))
  v <- rust_result(do.call(aggregate_elt_values, c(cols, list(x = double(), p = double()))))
  structure(list(columns = cols, rate = v$rate, expected_annual_loss = v$expected_annual_loss),
            class = "event_loss_table")
}

#' @rdname event_loss_table
#' @export
elt_oep <- function(elt, x) {
  v <- rust_result(do.call(aggregate_elt_values,
                           c(elt$columns, list(x = as.double(x), p = double()))))
  data.frame(x = as.double(x), exceedance_frequency = v$exceedance_frequency, oep = v$oep)
}

#' @rdname event_loss_table
#' @export
elt_occurrence_pml <- function(elt, p) {
  rust_result(do.call(aggregate_elt_values,
                      c(elt$columns, list(x = double(), p = as.double(p)))))$pml
}

#' @rdname event_loss_table
#' @export
simulate_elt <- function(elt, n_years, seed = 0) {
  r <- rust_result(do.call(aggregate_elt_simulate,
                           c(elt$columns, list(n_years = as.double(n_years),
                                               seed = as.double(seed)))))
  list(events = event_set(ptr = r$events), ids = r$ids)
}

#' Merge two event sets
#'
#' Both sets' losses in each year: a catastrophe event set with a
#' non-catastrophe one, say. The sets need the same number of years. With
#' times on both, each year's losses are merged in time order (`x`'s first on
#' a tie); with times on neither, `x`'s losses come first. Sums insured are
#' kept when both sets carry them. Labels ([with_labels()]) are kept when
#' either carries them: `x`'s dimensions, then `y`'s new ones, with `""` for
#' a loss without a value in a dimension.
#'
#' @param x,y [event_set] objects.
#' @returns An [event_set].
#' @export
#' @examples
#' cat <- event_set_from_long(1, 100, 2)
#' other <- event_set_from_long(c(1, 2), c(3, 5), 2)
#' events(merge_event_sets(cat, other), 1)
merge_event_sets <- function(x, y) {
  event_set(ptr = rust_result(x@ptr$merge(y@ptr)))
}

#' Event set years in another order
#'
#' Year `i` of the result is year `index[i]` of `x`, with its losses, sums
#' insured, times and labels. Pass a group's `index` from
#' [reorder_groups()] or [reorder_groups_by_copula()] (`index = TRUE`) to
#' keep a class's events with its reordered totals; any index into the years
#' works, so a resample with repeats does too.
#'
#' @param x An [event_set].
#' @param index Years of `x`, 1-based.
#' @returns An [event_set].
#' @export
#' @examples
#' e <- event_set_from_long(c(2, 3, 3), c(50, 5, 60), 3)
#' events(permute_events(e, c(3, 1, 2)), 1)
permute_events <- function(x, index) {
  event_set(ptr = rust_result(x@ptr$permute(as.double(index))))
}

#' The losses with some label values
#'
#' The losses whose label `label` takes one of `values`, every year kept (a
#' year with none of them is empty), with their sums insured, times and
#' labels, and the aggregate part's columns with one of the values: what a
#' layer scoped to those values ([with_scope()]) sees.
#'
#' @param x An [event_set] with labels.
#' @param label A label dimension.
#' @param values The values to keep.
#' @returns An [event_set] with as many years.
#' @seealso [with_scope()]
#' @export
#' @examples
#' ev <- with_labels(events_from_years(list(c(20, 15), 8)), list(peril = c("quake", "wind", "quake")))
#' w <- filter_events(ev, "peril", "wind")
#' events(w, 1)
#' events(w, 2)
filter_events <- function(x, label, values) {
  event_set(ptr = rust_result(x@ptr$filter(as.character(label), as.character(values))))
}

#' Hours clause
#'
#' Each year's losses grouped into occurrences: losses within `window_hours`
#' of the first loss of a window are one occurrence, whose loss is their sum
#' and whose time is the window's start; the next loss after the window opens
#' the next one. A year is 8,760 hours. Windows open at each first loss in
#' turn, not where they would recover most. Needs times; sums insured are not
#' carried.
#'
#' @param x An [event_set] with times.
#' @param window_hours Such as 72 for a windstorm, 168 for a flood.
#' @param labels What the occurrences carry of the losses' labels: none
#'   (`"drop"`), their losses' labels, which must agree within an occurrence
#'   (`"agree"`, an error otherwise), or the labels of the largest loss in the
#'   occurrence, the earliest of equal ones (`"largest"`). An aggregate part
#'   keeps its labels unless they are dropped.
#' @param by `NULL`, or a label dimension, `"peril"` say: the clause applies
#'   within each of its values, so losses of two perils are never one
#'   occurrence, and the occurrences keep the labels (`labels` then decides
#'   the other dimensions, `"drop"` read as `"agree"`). Occurrences are listed
#'   by the time each window opens.
#' @returns An [event_set] of occurrences.
#' @export
#' @examples
#' h <- 1 / 8760
#' ev <- events_from_years(list(c(10, 20, 5)), times = list(c(0.5, 0.5 + 30 * h, 0.5 + 40 * h)))
#' ev <- with_labels(ev, list(peril = c("wind", "flood", "wind")))
#' g <- hours_clause(ev, 72, by = "peril")
#' events(g, 1)
#' event_labels(g, 1)
hours_clause <- function(x, window_hours, labels = "drop", by = NULL) {
  event_set(ptr = rust_result(x@ptr$grouped(as.double(window_hours), labels, by)))
}
