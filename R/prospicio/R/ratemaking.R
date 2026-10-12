# Aggregate lane: ratemaking, premium at the current rate level by the
# parallelogram method and trend, over crates/prospicio-r/src/ratemaking.rs
# (docs/design/ratemaking.md). The #' comments are the package
# documentation (see distributions.R).

#' Year fractions on the half-month grid
#'
#' Werner and Modlin date trends on a half-month grid: the first of a month
#' is its start, a day near the middle is half way, and the last day is the
#' next month's start (30 June is 1 July). Ratemaking dates are year
#' fractions, so 1 July 2011 is 2011.5.
#'
#' @param year,month,day Integer vectors of the same length.
#' @returns A numeric vector.
#' @export
#' @examples
#' year_fraction(2013, 6, 30) - year_fraction(2011, 1, 1)
#' year_fraction(2011, c(1, 11), c(1, 15))
year_fraction <- function(year, month, day) {
  n <- max(length(year), length(month), length(day))
  rust_result(year_fractions(rep_len(as.integer(year), n), rep_len(as.integer(month), n),
                             rep_len(as.integer(day), n)))
}

#' Average written and earned dates of a period
#'
#' Trend lengths run between these dates (Werner and Modlin, ch. 5 and 6).
#' A `"calendar"` period is premium earned, or losses occurring, in
#' `[start, end)`; a `"policy"` period is the policies written in it, over
#' their whole term. The forecast period is the policy period of the
#' policies written while new rates are in effect. Premium is trended
#' between average written dates, losses between average earned (accident)
#' dates.
#'
#' @param start,end The period, as year fractions.
#' @param term The policy term in years.
#' @param period `"calendar"` or `"policy"`.
#' @returns `average_written_date()` and `average_earned_date()`: a number.
#' @export
#' @examples
#' # Calendar year 2011 premium to rates effective 1/1/13: 2.5 years.
#' average_written_date(2013, 2014, 1, "policy") - average_written_date(2011, 2012, 1)
#' # Accident year 2011 to rates effective 1/1/15, six-month policies: 4.25.
#' average_earned_date(2015, 2016, 0.5, "policy") - average_earned_date(2011, 2012, 0.5)
average_written_date <- function(start, end, term, period = c("calendar", "policy")) {
  period <- match.arg(period)
  rust_result(period_dates(period, as.double(start), as.double(end), as.double(term)))[[1]]
}

#' @rdname average_written_date
#' @export
average_earned_date <- function(start, end, term, period = c("calendar", "policy")) {
  period <- match.arg(period)
  rust_result(period_dates(period, as.double(start), as.double(end), as.double(term)))[[2]]
}

#' Premium at the current rate level: the parallelogram method
#'
#' The portion of a period's premium at each rate level, under uniform
#' writing and earning over a policy `term`, and the on-level factor that
#' brings it to the current level (Werner and Modlin, ch. 5). The areas are
#' exact.
#'
#' @param effective Effective dates of the rate changes, as year fractions,
#'   in order.
#' @param change Overall average changes, `0.05` for +5%.
#' @param level Instead of `change`, the cumulative level after each change,
#'   as an exhibit prints them (rounded levels carry into the factors).
#' @param start,end The period, as year fractions.
#' @param term The policy term in years.
#' @param period `"calendar"` (premium earned in the window) or `"policy"`
#'   (policies written in it).
#' @param applies For each change, `"written"` (policies written on or
#'   after the date) or `"in_force"` (all premium earned on or after it, a
#'   law change); recycled.
#' @returns A list: `portions`, a data frame with one row per cell of
#'   constant level (`written_changes` and `in_force_changes` in effect,
#'   `level`, `share`), `average_level`, `current_level` and
#'   `on_level_factor`.
#' @export
#' @examples
#' # Werner and Modlin, Table 5.19: calendar year 2011, annual policies.
#' r <- rate_level(c(2010.5, 2011, 2012.25), c(0.05, 0.10, -0.01),
#'                 start = 2011, end = 2012, term = 1)
#' r$portions
#' r$on_level_factor
rate_level <- function(effective, change = NULL, level = NULL, start, end, term,
                       period = c("calendar", "policy"), applies = "written") {
  period <- match.arg(period)
  if (is.null(change) == is.null(level)) stop("give one of `change` and `level`")
  values <- as.double(if (is.null(level)) change else level)
  r <- rust_result(rate_level_rust(as.double(effective), values,
                                   rep_len(as.character(applies), length(values)),
                                   !is.null(level), period, as.double(start), as.double(end),
                                   as.double(term)))
  list(
    portions = data.frame(written_changes = r$written_changes,
                          in_force_changes = r$in_force_changes,
                          level = r$level, share = r$share),
    average_level = r$average_level,
    current_level = r$current_level,
    on_level_factor = r$on_level_factor
  )
}

#' Trend fits and factors
#'
#' `trend_fit()` fits `log(values)` (exponential) or `values` (linear) on
#' `times` in years by least squares; for quarterly points, pass times a
#' quarter apart. `trend_factor()` is `(1 + rate)^years`.
#'
#' @param times Times in years.
#' @param values The values; positive for an exponential trend.
#' @param kind `"exponential"` or `"linear"`.
#' @param rate An annual trend, `0.02` for +2%.
#' @param years The trend length in years.
#' @returns `trend_fit()`: a named vector (`intercept`, `slope`,
#'   `r_squared`, `annual_change`), the annual change `exp(slope) - 1` for
#'   an exponential fit and the slope for a linear one. `trend_factor()`:
#'   the factors.
#' @export
#' @examples
#' t <- (0:7) / 4
#' trend_fit(t, 100 * 1.03^t)
#' trend_factor(0.02, 2.5)
trend_fit <- function(times, values, kind = c("exponential", "linear")) {
  kind <- match.arg(kind)
  f <- rust_result(trend_fit_rust(as.double(times), as.double(values), kind))
  stats::setNames(f, c("intercept", "slope", "r_squared", "annual_change"))
}

#' @rdname trend_fit
#' @export
trend_factor <- function(rate, years) {
  (1 + as.double(rate))^as.double(years)
}
