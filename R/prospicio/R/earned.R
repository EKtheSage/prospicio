#' Earned exposure from policy dates
#'
#' Written, earned, unearned and in-force amounts of a table of policies,
#' by window and label. Each policy earns its amount (an exposure count, or
#' a premium) evenly over its term, from its effective date up to, and not
#' including, its expiration date.
#'
#' With `kind = "calendar"` a window holds what was written and earned in
#' it, and the unearned and in-force amounts at its end. With
#' `kind = "policy"` a window holds the policies written in it, with all
#' they earn; give `valuation` to see them as of a date.
#'
#' "At a window's end" and "as of a valuation date" mean the close of the
#' day before: a policy that expires on that date is still in force, and
#' one effective on it is not yet written.
#'
#' A mid-term change gives the policy a new full-term amount from a date;
#' it is written on that date, as the difference in full-term amounts for
#' the term still to run. A cancellation is written back on the last day in
#' force. So a calendar period's written amount can be negative, while a
#' policy period holds everything its policies wrote, net.
#'
#' `in_force_amount()` is the amount in force on each of some dates: the
#' full-term amounts, as they stand that day, of the policies effective on
#' or before it and not yet expired or cancelled.
#'
#' `calendar_windows()` gives the dates between calendar years, quarters
#' or months.
#'
#' @param effective,expiration The policies' effective and expiration
#'   dates: `Date`s, or character strings `"YYYY-MM-DD"`. The expiration is
#'   the first day the policy is not in force.
#' @param amount The full-term amount of each policy.
#' @param windows The dates the windows lie between, increasing; see
#'   `calendar_windows()`.
#' @param kind `"calendar"` or `"policy"`.
#' @param method `"daily"` (exact days), `"half_month"` (the half-month grid
#'   of the ratemaking exhibits) or `"twenty_fourths"` (the "15th of the
#'   month" rule: every policy date at mid-month).
#' @param valuation See the table as of this date; `NULL` for no cut-off.
#' @param label A label a policy; the table has a set of rows for each.
#' @param changes Mid-term changes: a data frame with columns `policy` (the
#'   position of the policy, from 1), `date` and `amount` (the full-term
#'   amount from that date); `NULL` for none.
#' @param cancellation A cancellation date a policy, the first day not in
#'   force, `NA` for a policy not cancelled; `NULL` for none.
#' @param at Dates at which to take the in-force amount.
#' @param first_year,last_year The first and last calendar year.
#' @param by `"year"`, `"quarter"` or `"month"`.
#' @returns `earned_exposure()`: a data frame with columns `label`, `from`,
#'   `to` (`Date`s), `written`, `earned`, `unearned` and `in_force`, one row
#'   a label and window. `in_force_amount()`: a numeric vector, one value a
#'   date. `calendar_windows()`: a `Date` vector, one more than there are
#'   windows.
#' @export
#' @examples
#' # Werner and Modlin's six annual policies, a quarter apart.
#' effective <- as.Date(c("2010-10-01", "2011-01-01", "2011-04-01",
#'                        "2011-07-01", "2011-10-01", "2012-01-01"))
#' expiration <- seq(effective[1], by = "quarter", length.out = 10)[5:10]
#' years <- calendar_windows(2010, 2012)
#' earned_exposure(effective, expiration, rep(1, 6), years, method = "half_month")
#'
#' # The fourth policy cancelled after nine months.
#' earned_exposure(effective, expiration, rep(1, 6), years, method = "half_month",
#'                 cancellation = c(NA, NA, NA, "2012-04-01", NA, NA))$written
#'
#' in_force_amount(effective, expiration, rep(1, 6), c("2011-01-01", "2011-06-15"))
earned_exposure <- function(effective, expiration, amount, windows, kind = "calendar",
                            method = "daily", valuation = NULL, label = NULL,
                            changes = NULL, cancellation = NULL) {
  ch <- earned_changes(changes)
  r <- rust_result(earned_exposure_rust(
    iso_dates(effective), iso_dates(expiration), as.double(amount), iso_dates(windows),
    kind, method, if (is.null(valuation)) "" else iso_dates(valuation),
    if (is.null(label)) character() else as.character(label),
    ch$policy, ch$date, ch$amount, earned_cancellation(cancellation)
  ))
  data.frame(label = r$label, from = as.Date(r$from), to = as.Date(r$to),
             written = r$written, earned = r$earned, unearned = r$unearned,
             in_force = r$in_force)
}

#' @rdname earned_exposure
#' @export
in_force_amount <- function(effective, expiration, amount, at, changes = NULL,
                            cancellation = NULL) {
  ch <- earned_changes(changes)
  rust_result(in_force_amount_rust(
    iso_dates(effective), iso_dates(expiration), as.double(amount), iso_dates(at),
    ch$policy, ch$date, ch$amount, earned_cancellation(cancellation)
  ))
}

#' @rdname earned_exposure
#' @export
calendar_windows <- function(first_year, last_year, by = "year") {
  as.Date(rust_result(calendar_windows_rust(as.double(first_year), as.double(last_year), by)))
}

# Dates as "YYYY-MM-DD": Dates are formatted, strings pass as they are, so
# that the core reports one that is not a date.
iso_dates <- function(x) {
  if (inherits(x, "Date") || inherits(x, "POSIXt")) format(as.Date(x), "%Y-%m-%d") else as.character(x)
}

# The changes' columns, empty for none.
earned_changes <- function(changes) {
  if (is.null(changes)) {
    return(list(policy = numeric(), date = character(), amount = numeric()))
  }
  list(policy = as.double(changes$policy), date = iso_dates(changes$date),
       amount = as.double(changes$amount))
}

# Cancellation dates as strings, "" for a policy not cancelled.
earned_cancellation <- function(cancellation) {
  if (is.null(cancellation)) {
    return(character())
  }
  out <- iso_dates(cancellation)
  out[is.na(out)] <- ""
  out
}
