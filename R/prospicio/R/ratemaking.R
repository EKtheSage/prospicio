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
#' @param start,end The period, as year fractions.
#' @param term The policy term in years.
#' @param period `"calendar"` (premium earned in the window) or `"policy"`
#'   (policies written in it).
#' @param applies For each change, `"written"` (policies written on or
#'   after the date) or `"in_force"` (all premium earned on or after it, a
#'   law change); recycled.
#' @param level Instead of `change`, the cumulative level after each change,
#'   as an exhibit prints them (rounded levels carry into the factors).
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
rate_level <- function(effective, change = NULL, start, end, term,
                       period = c("calendar", "policy"), applies = "written", level = NULL) {
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

#' Experience for the loss ratio and pure premium methods
#'
#' Each experience year is projected to the forecast period. For the loss
#' ratio method, earned premium is brought to the current rate level and
#' trended. For both methods, losses are developed, trended and loaded for
#' LAE. The years are combined as the ratio of their totals (Werner and
#' Modlin, ch. 8). Factors are recycled to the number of years, so a single
#' number applies to every year; a net trend goes in `loss_trend`.
#'
#' @param earned_premium Earned premium by year.
#' @param exposure Earned exposure by year.
#' @param losses Reported or paid losses by year, with whatever LAE they
#'   include.
#' @param on_level_factor,premium_trend,development,loss_trend Factors by
#'   year.
#' @param lae_factor `1 +` the LAE ratio to losses not in `losses`.
#' @returns A list. `loss_ratio_experience()`: `by_year`, a data frame with
#'   each year's `projected_premium`, `projected_losses` and `loss_ratio`,
#'   and the totals `projected_premium`, `projected_losses` and
#'   `loss_ratio`. `pure_premium_experience()`: `by_year` with
#'   `projected_losses` and `pure_premium`, and the totals `exposure`,
#'   `projected_losses` and `pure_premium`.
#' @export
#' @examples
#' e <- loss_ratio_experience(c(100, 300), c(50, 250), loss_trend = 1.05)
#' e$loss_ratio
#' pure_premium_experience(c(10, 30), c(1000, 2000), lae_factor = 1.1)$pure_premium
loss_ratio_experience <- function(earned_premium, losses, on_level_factor = 1, premium_trend = 1,
                                  development = 1, loss_trend = 1, lae_factor = 1) {
  n <- length(earned_premium)
  col <- function(x) rep_len(as.double(x), n)
  r <- rust_result(loss_ratio_experience_rust(col(earned_premium), as.double(losses),
                                              col(on_level_factor), col(premium_trend),
                                              col(development), col(loss_trend),
                                              col(lae_factor)))
  list(
    by_year = data.frame(projected_premium = r$projected_premium_by_year,
                         projected_losses = r$projected_losses_by_year,
                         loss_ratio = r$loss_ratio_by_year),
    projected_premium = r$projected_premium,
    projected_losses = r$projected_losses,
    loss_ratio = r$loss_ratio
  )
}

#' @rdname loss_ratio_experience
#' @export
pure_premium_experience <- function(exposure, losses, development = 1, loss_trend = 1,
                                    lae_factor = 1) {
  n <- length(exposure)
  col <- function(x) rep_len(as.double(x), n)
  r <- rust_result(pure_premium_experience_rust(col(exposure), as.double(losses),
                                                col(development), col(loss_trend),
                                                col(lae_factor)))
  list(
    by_year = data.frame(projected_losses = r$projected_losses_by_year,
                         pure_premium = r$pure_premium_by_year),
    exposure = r$exposure,
    projected_losses = r$projected_losses,
    pure_premium = r$pure_premium
  )
}

#' The overall rate indication
#'
#' The loss ratio method's indicated change, `(LR + F) / (1 - V - Q) - 1`,
#' and the pure premium method's indicated rate,
#' `(PP + fixed) / (1 - V - Q)`, with variable expense `V`, target profit
#' `Q` (negative with investment income) and fixed expenses as a ratio `F`
#' or an amount per exposure (Werner and Modlin, ch. 7 and 8). Other costs
#' per exposure, such as net reinsurance, add to either.
#' `permissible_loss_ratio()` is `1 - F - V - Q`.
#'
#' @param loss_ratio The projected loss and LAE ratio at current rates.
#' @param pure_premium The projected pure premium, with LAE.
#' @param fixed_ratio Fixed expenses as a ratio to premium.
#' @param fixed_per_exposure Fixed expenses per exposure.
#' @param variable The variable expense provision.
#' @param profit The target underwriting profit provision.
#' @returns A number.
#' @export
#' @examples
#' # Werner and Modlin, ch. 8: +10% and $500.
#' indicated_change(0.65, 0.065, variable = 0.25, profit = 0.10)
#' indicated_rate(300, 25, variable = 0.25, profit = 0.10)
#' permissible_loss_ratio(variable = 0.25, profit = 0.10)
indicated_change <- function(loss_ratio, fixed_ratio = 0, variable, profit) {
  rust_result(indication_rust(as.double(variable), as.double(profit), as.double(fixed_ratio),
                              as.double(loss_ratio), 0, 0))[[2]]
}

#' @rdname indicated_change
#' @export
indicated_rate <- function(pure_premium, fixed_per_exposure = 0, variable, profit) {
  rust_result(indication_rust(as.double(variable), as.double(profit), 0, 0,
                              as.double(pure_premium), as.double(fixed_per_exposure)))[[3]]
}

#' @rdname indicated_change
#' @export
permissible_loss_ratio <- function(fixed_ratio = 0, variable, profit) {
  rust_result(indication_rust(as.double(variable), as.double(profit), as.double(fixed_ratio),
                              0, 0, 0))[[1]]
}

#' Expense provisions and credibility against a complement
#'
#' `premium_based_expenses()` is the premium-based projection: each expense
#' ratio split by its fixed share and summed (all shares 0 is the
#' all-variable method). `credibility_weighted()` is
#' `Z * estimate + (1 - Z) * complement`. `trended_present_rates()` is the
#' trended present rates complement as a change,
#' `(1 + prior_indicated) / (1 + prior_implemented) * trend - 1`, with
#' `trend` the factor from the last review's target effective date to the
#' new one (the loss trend over the premium trend for a loss ratio).
#'
#' @param ratio Expense ratios to premium.
#' @param fixed_share The share of each that is fixed; recycled.
#' @param estimate,complement The indication and its complement.
#' @param credibility The credibility `Z`.
#' @param prior_indicated,prior_implemented The last review's indicated
#'   and implemented changes.
#' @param trend The trend factor between the reviews' target effective
#'   dates.
#' @returns `premium_based_expenses()`: a named vector (`fixed`,
#'   `variable`); the others, numbers.
#' @export
#' @examples
#' premium_based_expenses(c(0.063, 0.085, 0.112), c(0.75, 0.75, 0))
#' credibility_weighted(200, 0.3, 300)
#' 200 * (1 + trended_present_rates(0.10, 0.06, trend_factor(0.05, 2)))
premium_based_expenses <- function(ratio, fixed_share) {
  r <- rust_result(premium_based_expenses_rust(as.double(ratio),
                                               rep_len(as.double(fixed_share), length(ratio))))
  stats::setNames(r, c("fixed", "variable"))
}

#' @rdname premium_based_expenses
#' @export
credibility_weighted <- function(estimate, credibility, complement) {
  credibility * estimate + (1 - credibility) * complement
}

#' @rdname premium_based_expenses
#' @export
trended_present_rates <- function(prior_indicated, prior_implemented, trend) {
  (1 + prior_indicated) / (1 + prior_implemented) * trend - 1
}
