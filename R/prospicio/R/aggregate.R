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
#' @returns An `event_set` object with read-only properties `n_sims` and `seed`.
#' @export
#' @keywords internal
event_set <- S7::new_class(
  "event_set",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("EventSet"),
    n_sims = S7::new_property(S7::class_double, getter = function(self) self@ptr$n_sims()),
    seed = S7::new_property(S7::class_double, getter = function(self) self@ptr$seed())
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
