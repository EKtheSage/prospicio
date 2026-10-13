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
#' `"normal"`, `"gamma"` or `"lognormal"` (mean and cv), `"shifted_gamma"` or
#' `"shifted_lognormal"` (mean, cv and skewness, which must be positive).
#' `approximation_cdf()` and `approximation_quantile()` evaluate a fit.
#'
#' @param frequency A claim count, as for [simulate_events()].
#' @param severity A severity with a closed-form third moment.
#' @param kind The family.
#' @param approximation A fit from `moment_approximation()`.
#' @param x Values.
#' @param p Probabilities.
#' @returns `compound_moments()`: a named vector. `moment_approximation()`: a
#'   list with `kind`, `parameters` (named: `mean` and `sd`; `shape` and
#'   `scale`; `meanlog` and `sdlog`; plus `shift`) and `moments`. The others:
#'   numeric vectors.
#' @export
#' @examples
#' m <- compound_moments(poisson_count(10), gamma_distribution(1 / 0.49, 50 * 0.49))
#' m
#' fit <- moment_approximation(poisson_count(10), gamma_distribution(1 / 0.49, 50 * 0.49),
#'                             "shifted_gamma")
#' fit$parameters
#' approximation_quantile(fit, c(0.5, 0.99))
compound_moments <- function(frequency, severity) {
  m <- rust_result(compound_moments_rust(frequency@ptr, severity@ptr))
  stats::setNames(m, c("mean", "variance", "third_cumulant", "cv", "skew"))
}

#' @rdname compound_moments
#' @export
moment_approximation <- function(frequency, severity,
                                 kind = c("shifted_lognormal", "shifted_gamma", "gamma",
                                          "lognormal", "normal")) {
  kind <- match.arg(kind)
  m <- compound_moments(frequency, severity)[c("mean", "variance", "third_cumulant")]
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
#' @returns An [event_set].
#' @export
#' @examples
#' ev <- event_set_from_long(c(3, 1, 3), c(5, 1, 7), 4)
#' event_counts(ev)
event_set_from_long <- function(years, losses, n_years, seed = 0) {
  event_set(ptr = rust_result(EventSet$from_long(as.double(years), as.double(losses),
                                                 as.double(n_years), as.double(seed))))
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
#' kept when both sets carry them.
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
#' @returns An [event_set] of occurrences.
#' @export
hours_clause <- function(x, window_hours) {
  event_set(ptr = rust_result(x@ptr$grouped(as.double(window_hours))))
}
