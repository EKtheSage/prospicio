#' Self-exciting (Hawkes) event times
#'
#' A Hawkes process with an exponential kernel, time in years: the rate
#' `baseline + sum(alpha * exp(-beta * (t - t_i)))` over past events `t_i`.
#' Each event has on average `alpha / beta` direct offspring (the branching
#' ratio, below 1), so events cluster in time, as windstorm series and
#' aftershocks do, and the stationary rate is
#' `baseline / (1 - alpha / beta)`. `alpha = 0` is the Poisson process.
#'
#' - `hawkes_simulate()`: one path's event times on `[0, horizon)` from an
#'   empty history, drawn exactly by the branching construction from stream
#'   `stream` of `seed`.
#' - `hawkes_counts()`: the number of events in each of `n_years`
#'   independent years, year `i` from stream `i - 1` of `seed`, after
#'   `burn_in` years whose events excite the year and are dropped.
#' - `hawkes_event_set()`: those years as an [event_set] with times in
#'   `[0, 1)` and a loss from `severity` each, for [apply_tower()] and
#'   [hours_clause()].
#' - `hawkes_moments()`: the mean count over `horizon` years from an empty
#'   history and the stationary variance of the count over such a window
#'   (Hawkes and Oakes' formula).
#' - `hawkes_log_likelihood()`: the log-likelihood of event times, each
#'   period a sorted vector on `[0, horizon)` from an empty history.
#'
#' @param baseline Rate of the background events, a year; positive.
#' @param alpha The rise in the rate at each event; at least 0 and below
#'   `beta`.
#' @param beta The excitation's decay rate, a year; positive.
#' @param process A `hawkes` process.
#' @param horizon Length of a path or period, in years.
#' @param seed Seed, a whole number.
#' @param stream Stream of the seed.
#' @param n_years Number of independent years.
#' @param burn_in Years simulated before each year and dropped.
#' @param severity A severity, as for [simulate_events()].
#' @param periods A list of sorted numeric vectors of event times.
#' @returns `hawkes()` an S7 `hawkes` object with properties `baseline`,
#'   `alpha`, `beta`, `branching_ratio` and `stationary_rate`;
#'   `hawkes_simulate()` a vector of times, `hawkes_counts()` a vector of
#'   counts, `hawkes_event_set()` an [event_set], `hawkes_moments()`
#'   `c(mean, variance)` and `hawkes_log_likelihood()` a number.
#' @seealso [hawkes_fit()] to fit one to event times.
#' @export
#' @examples
#' h <- hawkes(2, 12, 24)
#' h@branching_ratio
#' h@stationary_rate
#' mean(hawkes_counts(h, 2000, seed = 1))
#' hawkes_moments(h)
#' ev <- hawkes_event_set(h, lognormal_from_mean_cv(10, 1), 100, seed = 7)
#' ev@has_times
hawkes <- S7::new_class(
  "hawkes",
  package = "prospicio",
  properties = list(
    baseline = S7::class_double,
    alpha = S7::class_double,
    beta = S7::class_double,
    branching_ratio = S7::new_property(S7::class_double, getter = function(self) self@alpha / self@beta),
    stationary_rate = S7::new_property(
      S7::class_double, getter = function(self) self@baseline / (1 - self@alpha / self@beta)
    )
  ),
  constructor = function(baseline, alpha, beta) {
    rust_result(hawkes_check_rust(as.double(baseline), as.double(alpha), as.double(beta)))
    S7::new_object(S7::S7_object(), baseline = as.double(baseline), alpha = as.double(alpha),
                   beta = as.double(beta))
  }
)

#' @rdname hawkes
#' @export
hawkes_simulate <- function(process, horizon, seed, stream = 0) {
  rust_result(hawkes_simulate_rust(process@baseline, process@alpha, process@beta,
                                   as.double(horizon), as.double(seed), as.double(stream)))
}

#' @rdname hawkes
#' @export
hawkes_counts <- function(process, n_years, burn_in = 0, seed = 0) {
  rust_result(hawkes_counts_rust(process@baseline, process@alpha, process@beta,
                                 as.double(n_years), as.double(burn_in), as.double(seed)))
}

#' @rdname hawkes
#' @export
hawkes_event_set <- function(process, severity, n_years, burn_in = 0, seed = 0) {
  event_set(ptr = rust_result(EventSet$hawkes(
    process@baseline, process@alpha, process@beta, severity@ptr,
    as.double(n_years), as.double(burn_in), as.double(seed)
  )))
}

#' @rdname hawkes
#' @export
hawkes_moments <- function(process, horizon = 1) {
  m <- rust_result(hawkes_moments_rust(process@baseline, process@alpha, process@beta, as.double(horizon)))
  c(mean = m[1], variance = m[2])
}

#' @rdname hawkes
#' @export
hawkes_log_likelihood <- function(process, periods, horizon = 1) {
  rust_result(hawkes_log_likelihood_rust(process@baseline, process@alpha, process@beta,
                                         lapply(periods, as.double), as.double(horizon)))
}

#' Fit a Hawkes process to event times
#'
#' The maximum likelihood fit of a Hawkes process with an exponential
#' kernel ([hawkes()]) to event times over one or more periods of
#' `horizon` years, each sorted and on `[0, horizon)` from an empty history
#' (for years, each year's times and `horizon = 1`). The log-likelihood is
#' maximised over the log parameters by Nelder-Mead; standard errors are
#' from the observed information.
#'
#' @param periods A list of sorted numeric vectors of event times.
#' @param horizon Length of each period, in years.
#' @returns A list with `baseline`, `alpha`, `beta`, `log_likelihood`,
#'   `std_errors` (of those three), `n_events`, `converged`,
#'   `branching_ratio`, `aic` and `process`, the fitted [hawkes()] (`NULL`
#'   when the branching ratio is at or above 1).
#' @export
#' @examples
#' h <- hawkes(1.5, 6, 10)
#' years <- lapply(0:399, function(i) hawkes_simulate(h, 1, seed = 11, stream = i))
#' fit <- hawkes_fit(years)
#' c(fit$baseline, fit$alpha, fit$beta)
#' fit$std_errors
hawkes_fit <- function(periods, horizon = 1) {
  f <- rust_result(hawkes_fit_rust(lapply(periods, as.double), as.double(horizon)))
  f$process <- if (f$branching_ratio < 1) hawkes(f$baseline, f$alpha, f$beta) else NULL
  f
}
