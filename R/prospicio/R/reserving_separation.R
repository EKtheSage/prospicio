#' @include reserving.R reserving_inflation.R
NULL

#' Taylor's separation method
#'
#' Separates development from calendar-year effects (inflation) in an
#' incremental triangle (Taylor 1977; issue #979, design note
#' `docs/design/reserving-inflation.md`). Increments over a normaliser
#' `n_i` are `r_j * lambda_(i+j)`: development proportions `r_j` (summing
#' to 1) times an index `lambda_h` per calendar period. Python's
#' `Separation`.
#'
#' * `"arithmetic"` solves Taylor's equations on the column and diagonal
#'   sums (the marginal equations of a Poisson GLM on development and
#'   calendar factors).
#' * `"geometric"` is least squares on the logarithms, taken back to
#'   amounts by the geometric means; `"geometric_lognormal"` by the
#'   lognormal means (a factor `exp(sigma^2 / 2)` on every index, which
#'   needs at least 3 origins). The geometric methods need every increment
#'   positive.
#'
#' The triangle must be full: as many ages as origins, the origin grain
#' equal to the development grain, no hole. The indices after the
#' triangle's last calendar period are not fitted: give exactly one of
#' `future_rate`, `future_index` and `trend_periods`. On a triangle
#' already restated by an economic index ([restate()]) the indices measure
#' superimposed inflation.
#'
#' @param triangle A [triangle()].
#' @param column The column to fit; `NULL` for the triangle's only one.
#' @param kind `"arithmetic"` (the default), `"geometric"` or
#'   `"geometric_lognormal"`.
#' @param normaliser `n_i`, one per origin: finite and positive (claim
#'   counts, premium, exposure). `NULL` (the default) is 1 for every origin.
#' @param future_rate A constant annual rate from the last fitted index,
#'   compounded by months: above -1.
#' @param future_index A [calendar_index()]: its relative changes from the
#'   triangle's last calendar period, read at the last sub-period of each
#'   period.
#' @param trend_periods The exponential trend of the last `trend_periods`
#'   fitted indices, continued from the last: at least 2, at most the number
#'   of calendar periods.
#' @param segment The key values of the one segment of a triangle with
#'   several, as a named list or character vector (`c(lob = "auto")`);
#'   the default needs a triangle of one segment.
#' @param origins,ages,proportions,calendar,indices,future_indices,fitted,future,reserves,log_variance
#'   The fields of a `separation_fit`, as described under Value. A fit is
#'   made by `separation()`; the `separation_fit()` constructor only holds
#'   its fields.
#' @returns A `separation_fit` with the properties `origins`, `ages`
#'   (months), `normaliser` (per origin), `proportions` (per age, summing to
#'   1), `calendar` (the triangle's calendar periods), `indices` (per
#'   calendar period), `future_indices` (a data frame of the `period`s after
#'   the triangle's last, up to the last future cell's, and their `index`),
#'   `fitted` (a matrix of `n_i r_j lambda_(i+j)` on the observed cells,
#'   origins by ages, `NA` on the future ones), `future` (the future cells as
#'   [future_cells()], with no tail past the oldest age), `reserves` (the sum
#'   of each origin's future cells; the first origin's is 0), `log_variance`
#'   (the geometric methods' residual variance on the log scale; `NULL` for
#'   the arithmetic method and with no degrees of freedom left) and
#'   `total_reserve`.
#' @seealso [indexation()], [calendar_index()], [future_cells()]
#' @export
#' @examples
#' # Taylor (1977), Example 1: claims per claim incurred, cumulative.
#' rows <- list(c(50.4, 78.6, 87.6, 92.4), c(58.0, 87.2, 96.9), c(59.5, 92.7), 66.2)
#' long <- data.frame(
#'   year = rep(2001:2004, lengths(rows)),
#'   age = 12 * unlist(lapply(rows, seq_along)),
#'   s = unlist(rows)
#' )
#' fit <- separation(triangle(long, "year", "age", "s"), future_rate = 0.10)
#' round(fit@proportions, 4)
#' round(fit@indices, 1)
separation_fit <- S7::new_class(
  "separation_fit",
  package = "prospicio",
  properties = list(
    origins = S7::class_character,
    ages = S7::class_integer,
    normaliser = S7::class_double,
    proportions = S7::class_double,
    calendar = S7::class_character,
    indices = S7::class_double,
    future_indices = S7::class_data.frame,
    fitted = S7::class_any,
    future = future_cells,
    reserves = S7::class_double,
    log_variance = S7::new_union(S7::class_double, NULL),
    total_reserve = S7::new_property(S7::class_double, getter = function(self) sum(self@reserves))
  ),
  constructor = function(origins, ages, normaliser, proportions, calendar, indices, future_indices,
                         fitted, future, reserves, log_variance = NULL) {
    S7::new_object(
      S7::S7_object(),
      origins = origins, ages = ages, normaliser = normaliser, proportions = proportions,
      calendar = calendar, indices = indices, future_indices = future_indices, fitted = fitted,
      future = future, reserves = reserves, log_variance = log_variance
    )
  }
)

S7::method(print, separation_fit) <- function(x, ...) {
  cat(sprintf("separation_fit: %d origins, total_reserve = %s\n", length(x@origins),
              format(x@total_reserve, digits = 15)))
  invisible(x)
}

#' @rdname separation_fit
#' @export
separation <- function(triangle, column = NULL, kind = "arithmetic", normaliser = NULL,
                       future_rate = NULL, future_index = NULL, trend_periods = NULL,
                       segment = NULL) {
  column <- fit_column(triangle, column)
  if (!is.character(kind) || length(kind) != 1L ||
        !kind %in% c("arithmetic", "geometric", "geometric_lognormal")) {
    stop("kind must be \"arithmetic\", \"geometric\" or \"geometric_lognormal\"", call. = FALSE)
  }
  given <- c(rate = !is.null(future_rate), index = !is.null(future_index),
             trend = !is.null(trend_periods))
  if (sum(given) == 0L) {
    stop("give the future indices: one of future_rate, future_index or trend_periods", call. = FALSE)
  }
  if (sum(given) > 1L) {
    stop("give only one of future_rate, future_index and trend_periods", call. = FALSE)
  }
  if (!is.null(normaliser)) normaliser <- as.double(normaliser)
  future <- names(given)[given]
  rate <- if (future == "rate") single_number(future_rate, "future_rate") else 0
  first <- ""
  index_values <- double(0)
  index_rate <- NULL
  if (future == "index") {
    future_index <- check_calendar_index(future_index)
    first <- future_index@first
    index_values <- future_index@values
    index_rate <- future_index@future_rate
  }
  trend <- 0L
  if (future == "trend") {
    trend <- single_number(trend_periods, "trend_periods")
    if (trend != round(trend) || trend < 2) {
      stop(sprintf("trend_periods must be a whole number of at least 2, got %s", format(trend)), call. = FALSE)
    }
    trend <- as.integer(trend)
  }
  keys <- if (is.null(segment)) character() else names(segment)
  values <- if (is.null(segment)) character() else as.character(unlist(segment, use.names = FALSE))
  if (!is.null(segment) && is.null(keys)) {
    stop("segment must be named by key, as c(lob = \"auto\")", call. = FALSE)
  }
  out <- reserving_result(separation_fit_rust(
    triangle@ptr, column, keys, values, kind, normaliser, future, rate, first, index_values,
    index_rate, trend
  ))
  n <- length(out$origins)
  fitted <- matrix(out$fitted, n, length(out$ages), byrow = TRUE,
                   dimnames = list(out$origins, out$ages))
  fitted[is.nan(fitted)] <- NA
  names(out$reserves) <- out$origins
  names(out$proportions) <- out$ages
  names(out$indices) <- out$calendar
  future_cells_out <- future_cells(
    data.frame(origin = out$future$origin, age = out$future$age,
               valuation = out$future$valuation, value = out$future$value,
               stringsAsFactors = FALSE),
    origins = out$future$origins
  )
  separation_fit(
    origins = out$origins,
    ages = as.integer(out$ages),
    normaliser = out$normaliser,
    proportions = out$proportions,
    calendar = out$calendar,
    indices = out$indices,
    future_indices = data.frame(period = out$future_period, index = out$future_index,
                                stringsAsFactors = FALSE),
    fitted = fitted,
    future = future_cells_out,
    reserves = out$reserves,
    log_variance = out$log_variance
  )
}
