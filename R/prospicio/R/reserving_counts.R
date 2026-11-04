# Methods on claim counts: frequency-severity, the disposal rate method, case
# outstanding development and the Berquist-Sherman adjustments (issues #554
# and #555, docs/design/reserving-v02.md, decisions 33 and 34). Each choice
# is encoded here as a named list with a `kind`, which
# crates/prospicio-r/src/reserving_counts.rs reads.

# A selection of one value per age: NULL or "all" for the simple average of
# every origin, a number of latest origins, "latest_diagonal", a numeric
# vector named by age, or a list of n_periods, drop_high and drop_low.
selection_arg <- function(x, arg) {
  average <- function(n = NaN, high = 0, low = 0) {
    list(kind = "average", n_periods = as.double(n), drop_high = as.double(high),
         drop_low = as.double(low))
  }
  if (is.null(x) || identical(x, "all")) return(average())
  if (identical(x, "latest_diagonal")) return(list(kind = "latest_diagonal"))
  if (is.list(x) && all(names(x) %in% c("n_periods", "drop_high", "drop_low"))) {
    return(average(if (is.null(x$n_periods)) NaN else x$n_periods,
                   if (is.null(x$drop_high)) 0 else x$drop_high,
                   if (is.null(x$drop_low)) 0 else x$drop_low))
  }
  if (is.numeric(x) && !is.null(names(x))) {
    return(list(kind = "given", ages = as.double(names(x)), values = unname(as.double(x))))
  }
  if (is.numeric(x) && length(x) == 1 && !is.na(x)) return(average(x))
  stop(sprintf(paste0('%s must be NULL, "all", "latest_diagonal", a number of origins, a ',
                      "numeric vector named by age or a list of n_periods, drop_high and ",
                      "drop_low"), arg), call. = FALSE)
}

#' A chain ladder to project claim counts or severities
#'
#' The settings of a chain ladder for [frequency_severity()],
#' [disposal_method()] and [settlement_rate_adjust()], which project ultimate
#' counts or severities with it. The arguments are [chain_ladder()]'s.
#'
#' @param column The count column to develop; `NULL` for a severity
#'   development.
#' @param average,sigma_interpolation,tail,ldf,ldf_kind,ldf_weight,n_periods,drop_high,drop_low
#'   As in [chain_ladder()].
#' @returns A `development_method` list.
#' @examples
#' development_method("reported_counts", n_periods = 2)
#' @export
development_method <- function(column = NULL, average = "volume",
                               sigma_interpolation = "log-linear", tail = 1, ldf = NULL,
                               ldf_kind = NULL, ldf_weight = 1, n_periods = NULL, drop_high = 0,
                               drop_low = 0) {
  args <- development_args(average, sigma_interpolation, n_periods, drop_high, drop_low)
  structure(list(kind = "development", column = column, average = args$average, sigma = args$sigma,
                 excl = args$excl, tail = tail_ptr(tail),
                 ldf = ldf_pattern(ldf, ldf_kind, ldf_weight)),
            class = "development_method")
}

#' A value selected at a base origin and trended
#'
#' At origin `o` the value is
#' `selected * (1 + trend)^(years(o) - years(base)) / factors[o]`.
#' Friedland's approach #2 selects a frequency and a severity this way from
#' the older, mature years.
#'
#' @param selected The value at the base origin's level.
#' @param base The origin whose level `selected` is at, as a year or label.
#' @param trend The annual trend.
#' @param factors `NULL`, or one factor per origin, oldest first, that
#'   brings the origin's value to the base level for anything besides the
#'   trend (tort reform factors).
#' @returns A `trended` list for [frequency_severity()].
#' @examples
#' trended(20, 2008, 0.05)
#' @export
trended <- function(selected, base, trend, factors = NULL) {
  structure(list(kind = "trended", selected = single_number(selected, "selected"),
                 base = as.character(base), trend = single_number(trend, "trend"),
                 factors = if (is.null(factors)) numeric() else as.double(factors)),
            class = "trended")
}

# Ultimate counts: numbers, a column, a development_method() or a list of
# columns or methods for their mean.
ultimate_counts_arg <- function(x) {
  if (inherits(x, "development_method")) {
    if (is.null(x$column)) stop("ultimate_counts: development_method() needs a column",
                                call. = FALSE)
    return(unclass(x))
  }
  if (is.character(x) && length(x) == 1) return(unclass(development_method(x)))
  if (is.character(x) || is.list(x)) {
    return(list(kind = "mean", parts = unname(lapply(as.list(x), ultimate_counts_arg))))
  }
  if (is.numeric(x)) return(list(kind = "given", values = as.double(x)))
  stop("ultimate_counts must be numbers (one per origin), a column, a development_method() ",
       "or several columns or methods for their mean", call. = FALSE)
}

origin_named <- function(values, origins) stats::setNames(values, origins)

# A flattened row-major table as a matrix with NA where a cell is missing.
cell_matrix <- function(x, origins, ages) {
  m <- matrix(x, nrow = length(origins), byrow = TRUE)
  m[is.nan(m)] <- NA
  dimnames(m) <- list(origins, c(ages, "ult")[seq_len(ncol(m))])
  m
}

#' Frequency-severity
#'
#' Ultimate claims as ultimate counts times an ultimate severity, each
#' projected on its own (Friedland, *Estimating Unpaid Claims Using Basic
#' Techniques*, chapter 11, approaches #1 and #2). The triangle has one
#' segment and holds counts and amounts as columns.
#'
#' The fit has per origin `latest`, `latest_severity`, `ultimate_counts`,
#' `ultimate_severity`, `ultimate` and `reserve`, and `total_ultimate` and
#' `total_reserve`.
#'
#' @param triangle A [triangle].
#' @param amounts The claims column whose ultimate is projected.
#' @param counts The count column the severity triangle divides the amounts
#'   by.
#' @param ultimate_counts `NULL` for the chain ladder on `counts`; numbers,
#'   one per origin; a column; a [development_method()]; several columns or
#'   methods for their mean; or a [trended()] frequency with `exposure`.
#' @param severity `NULL` for the chain ladder on the severities, a
#'   [development_method()], or a [trended()] severity.
#' @param exposure The exposure column of a [trended()] frequency.
#' @param exposure_factors One factor per origin that brings the exposure to
#'   the base level.
#' @returns A `frequency_severity_fit`.
#' @examples
#' tri <- triangle(data.frame(origin = c(2020, 2020, 2021), age = c(12, 24, 12),
#'                            reported = c(100, 150, 120), counts = c(10, 12, 11)),
#'                 origin = "origin", development = "age",
#'                 columns = c("reported", "counts"))
#' frequency_severity(tri, "reported", "counts")@ultimate
#' @export
frequency_severity <- function(triangle, amounts, counts, ultimate_counts = NULL, severity = NULL,
                               exposure = NULL, exposure_factors = NULL) {
  check_triangle(triangle)
  ultimate <- if (inherits(ultimate_counts, "trended")) {
    if (is.null(exposure)) stop("a trended() frequency needs exposure, a column", call. = FALSE)
    list(kind = "frequency", exposure = exposure,
         exposure_factors = if (is.null(exposure_factors)) numeric() else
           as.double(exposure_factors),
         frequency = unclass(ultimate_counts))
  } else {
    ultimate_counts_arg(if (is.null(ultimate_counts)) counts else ultimate_counts)
  }
  sev <- if (is.null(severity)) {
    unclass(development_method())
  } else if (inherits(severity, "trended") || inherits(severity, "development_method")) {
    unclass(severity)
  } else {
    stop("severity must be NULL, a development_method() or a trended()", call. = FALSE)
  }
  out <- reserving_result(frequency_severity_rust(triangle@ptr, amounts, counts, ultimate, sev))
  o <- out$origins
  frequency_severity_fit(origins = o, latest = origin_named(out$latest, o),
                         latest_severity = origin_named(out$latest_severity, o),
                         ultimate_counts = origin_named(out$ultimate_counts, o),
                         ultimate_severity = origin_named(out$ultimate_severity, o),
                         ultimate = origin_named(out$ultimate, o),
                         reserve = origin_named(out$reserve, o))
}

#' @rdname frequency_severity
#' @param origins,latest,latest_severity,ultimate,reserve The fit's values.
#' @export
frequency_severity_fit <- S7::new_class(
  "frequency_severity_fit",
  package = "prospicio",
  properties = list(
    origins = S7::class_character,
    latest = S7::class_double,
    latest_severity = S7::class_double,
    ultimate_counts = S7::class_double,
    ultimate_severity = S7::class_double,
    ultimate = S7::class_double,
    reserve = S7::class_double,
    total_ultimate = S7::new_property(S7::class_double, getter = function(self) sum(self@ultimate)),
    total_reserve = S7::new_property(S7::class_double, getter = function(self) sum(self@reserve))
  )
)

S7::method(print, frequency_severity_fit) <- function(x, ...) {
  cat("frequency_severity_fit\n")
  print(cbind(counts = x@ultimate_counts, severity = x@ultimate_severity, ultimate = x@ultimate))
  invisible(x)
}

#' The disposal rate method
#'
#' Incremental closed counts from selected disposal rates, each priced at a
#' selected incremental severity brought to the origin's cost level
#' (Friedland, chapter 11).
#'
#' Selections (`disposal_rates`, `severities`) are `NULL` or `"all"` for the
#' simple average of every origin, a number for the latest n origins,
#' `"latest_diagonal"`, a numeric vector named by age, or a list of
#' `n_periods`, `drop_high` and `drop_low`.
#'
#' The fit has `ages`, per origin `ultimate_counts`, `latest`, `ultimate`
#' and `reserve`, by age `selected_rates` and `selected_severities` (the
#' step to ultimate last), the matrices `disposal_rates`, `severities`,
#' `trended_severities`, `incremental_counts`, `projected_severities` and
#' `incremental_paid`, and `total_ultimate` and `total_reserve`.
#'
#' @param triangle A [triangle] with one segment.
#' @param closed_counts,paid The closed count and paid claims columns.
#' @param ultimate_counts Numbers, one per origin; a column; a
#'   [development_method()]; or several for their mean.
#' @param disposal_rates,severities Selections, as in Details.
#' @param tail_severity_from The age from which the severities are pooled
#'   into one value, total trended paid over total counts, used to ultimate.
#' @param trend The annual severity trend.
#' @param base The origin of the common cost level; by default the latest.
#' @param factors One factor per origin that brings its severities to the
#'   base level besides the trend.
#' @returns A `disposal_fit`.
#' @examples
#' tri <- triangle(data.frame(origin = c(2020, 2020, 2021), age = c(12, 24, 12),
#'                            closed = c(5, 10, 6), paid = c(50, 100, 60)),
#'                 origin = "origin", development = "age", columns = c("closed", "paid"))
#' disposal_method(tri, "closed", "paid", c(10, 12))@ultimate
#' @export
disposal_method <- function(triangle, closed_counts, paid, ultimate_counts, disposal_rates = NULL,
                            severities = NULL, tail_severity_from = NULL, trend = 0, base = NULL,
                            factors = NULL) {
  check_triangle(triangle)
  out <- reserving_result(disposal_rust(
    triangle@ptr, closed_counts, paid, ultimate_counts_arg(ultimate_counts),
    selection_arg(disposal_rates, "disposal_rates"), selection_arg(severities, "severities"),
    if (is.null(tail_severity_from)) NaN else single_number(tail_severity_from, "tail_severity_from"),
    single_number(trend, "trend"), if (is.null(base)) NULL else as.character(base),
    if (is.null(factors)) numeric() else as.double(factors)))
  o <- out$origins
  a <- out$ages
  disposal_fit(origins = o, ages = a, ultimate_counts = origin_named(out$ultimate_counts, o),
               disposal_rates = cell_matrix(out$disposal_rates, o, a),
               selected_rates = stats::setNames(out$selected_rates, a),
               severities = cell_matrix(out$severities, o, a),
               trended_severities = cell_matrix(out$trended_severities, o, a),
               selected_severities = stats::setNames(out$selected_severities, c(a, "ult")),
               incremental_counts = cell_matrix(out$incremental_counts, o, a),
               projected_severities = cell_matrix(out$projected_severities, o, a),
               incremental_paid = cell_matrix(out$incremental_paid, o, a),
               latest = origin_named(out$latest, o), ultimate = origin_named(out$ultimate, o),
               reserve = origin_named(out$reserve, o))
}

#' @rdname disposal_method
#' @param origins,ages,selected_rates,selected_severities,trended_severities,incremental_counts,projected_severities,incremental_paid,latest,ultimate,reserve
#'   The fit's values.
#' @export
disposal_fit <- S7::new_class(
  "disposal_fit",
  package = "prospicio",
  properties = list(
    origins = S7::class_character,
    ages = S7::class_double,
    ultimate_counts = S7::class_double,
    disposal_rates = S7::class_any,
    selected_rates = S7::class_double,
    severities = S7::class_any,
    trended_severities = S7::class_any,
    selected_severities = S7::class_double,
    incremental_counts = S7::class_any,
    projected_severities = S7::class_any,
    incremental_paid = S7::class_any,
    latest = S7::class_double,
    ultimate = S7::class_double,
    reserve = S7::class_double,
    total_ultimate = S7::new_property(S7::class_double, getter = function(self) sum(self@ultimate)),
    total_reserve = S7::new_property(S7::class_double, getter = function(self) sum(self@reserve))
  )
)

S7::method(print, disposal_fit) <- function(x, ...) {
  cat("disposal_fit\n")
  print(cbind(counts = x@ultimate_counts, latest = x@latest, ultimate = x@ultimate))
  invisible(x)
}

#' Case outstanding development
#'
#' Each period's paid claims and closing case outstanding as ratios to the
#' case outstanding a period before, run out from the latest diagonal, with
#' paid to ultimate on the last case outstanding (Friedland, chapter 12).
#' [case_development_factor()] is the chapter's factor from case outstanding
#' to unpaid claims, `1 + (R - 1) P / (P - R)`, from the reported and paid
#' factors to ultimate.
#'
#' The fit has `ages`, per origin `latest_paid`, `latest_incurred`,
#' `ultimate`, `ibnr` and `reserve`, by age `selected_paid_on_case` and
#' `selected_case_on_case`, the matrices `paid_on_case`, `case_on_case`,
#' `case_outstanding` and `incremental_paid`, and `total_ultimate` and
#' `total_reserve`.
#'
#' @param triangle A [triangle] with one segment.
#' @param paid,incurred The paid and reported claims columns.
#' @param paid_to_ultimate Remaining paid over the case outstanding at the
#'   oldest age.
#' @param paid_on_case,case_on_case Selections, as in [disposal_method()].
#' @param reported_cdf,paid_cdf Reported and paid factors to ultimate.
#' @returns `case_outstanding()`: a `case_outstanding_fit`.
#'   `case_development_factor()`: a number.
#' @examples
#' tri <- triangle(data.frame(origin = c(2020, 2020, 2021), age = c(12, 24, 12),
#'                            paid = c(50, 90, 60), reported = c(100, 110, 120)),
#'                 origin = "origin", development = "age", columns = c("paid", "reported"))
#' case_outstanding(tri, "paid", "reported", 1)@ultimate
#' case_development_factor(1.2, 2)
#' @export
case_outstanding <- function(triangle, paid, incurred, paid_to_ultimate, paid_on_case = NULL,
                             case_on_case = NULL) {
  check_triangle(triangle)
  out <- reserving_result(case_outstanding_rust(
    triangle@ptr, paid, incurred, selection_arg(paid_on_case, "paid_on_case"),
    selection_arg(case_on_case, "case_on_case"), single_number(paid_to_ultimate, "paid_to_ultimate")))
  o <- out$origins
  a <- out$ages
  case_outstanding_fit(origins = o, ages = a, paid_on_case = cell_matrix(out$paid_on_case, o, a),
                       case_on_case = cell_matrix(out$case_on_case, o, a),
                       selected_paid_on_case = stats::setNames(out$selected_paid_on_case, a),
                       selected_case_on_case = stats::setNames(out$selected_case_on_case, a),
                       case_outstanding = cell_matrix(out$case_outstanding, o, a),
                       incremental_paid = cell_matrix(out$incremental_paid, o, a),
                       latest_paid = origin_named(out$latest_paid, o),
                       latest_incurred = origin_named(out$latest_incurred, o),
                       ultimate = origin_named(out$ultimate, o),
                       ibnr = origin_named(out$ibnr, o), reserve = origin_named(out$reserve, o))
}

#' @rdname case_outstanding
#' @param origins,ages,selected_paid_on_case,selected_case_on_case,case_outstanding,incremental_paid,latest_paid,latest_incurred,ultimate,ibnr,reserve
#'   The fit's values.
#' @export
case_outstanding_fit <- S7::new_class(
  "case_outstanding_fit",
  package = "prospicio",
  properties = list(
    origins = S7::class_character,
    ages = S7::class_double,
    paid_on_case = S7::class_any,
    case_on_case = S7::class_any,
    selected_paid_on_case = S7::class_double,
    selected_case_on_case = S7::class_double,
    case_outstanding = S7::class_any,
    incremental_paid = S7::class_any,
    latest_paid = S7::class_double,
    latest_incurred = S7::class_double,
    ultimate = S7::class_double,
    ibnr = S7::class_double,
    reserve = S7::class_double,
    total_ultimate = S7::new_property(S7::class_double, getter = function(self) sum(self@ultimate)),
    total_reserve = S7::new_property(S7::class_double, getter = function(self) sum(self@reserve))
  )
)

S7::method(print, case_outstanding_fit) <- function(x, ...) {
  cat("case_outstanding_fit\n")
  print(cbind(latest_paid = x@latest_paid, ultimate = x@ultimate, reserve = x@reserve))
  invisible(x)
}

#' @rdname case_outstanding
#' @export
case_development_factor <- function(reported_cdf, paid_cdf) {
  reserving_result(case_development_factor_rust(single_number(reported_cdf, "reported_cdf"),
                                                single_number(paid_cdf, "paid_cdf")))
}

open_counts_arg <- function(open_counts, reported, closed) {
  if (!is.null(open_counts) && is.null(reported) && is.null(closed)) {
    return(list(kind = "column", column = as.character(open_counts)))
  }
  if (is.null(open_counts) && !is.null(reported) && !is.null(closed)) {
    return(list(kind = "difference", reported = as.character(reported),
                closed = as.character(closed)))
  }
  stop("give open_counts, or reported and closed, not both", call. = FALSE)
}

case_trend_arg <- function(trend) {
  if (is.numeric(trend) && length(trend) == 1 && !is.na(trend)) {
    return(list(kind = "given", rate = as.double(trend)))
  }
  if (identical(trend, "average_case")) return(list(kind = "average_case"))
  if (is.character(trend) && length(trend) == 2) {
    return(list(kind = "fitted", amount = trend[[1]], counts = trend[[2]]))
  }
  stop('trend must be a number, "average_case" or two columns, an amount and a count',
       call. = FALSE)
}

#' The Berquist-Sherman adjustments
#'
#' Restate a triangle before it is projected (Berquist and Sherman, 1977;
#' Friedland, chapter 13), so that [chain_ladder()] or any other method runs
#' on the result unchanged. Each returns a cumulative [triangle] with the
#' adjusted columns replaced; the latest diagonal does not change.
#'
#' * `case_adequacy_adjust()` restates `incurred` at the case reserve
#'   adequacy of the latest diagonal: the average case outstanding of each
#'   age on the latest diagonal, detrended to the earlier origins, times the
#'   open counts, plus paid. `average_case()` returns the triangle with the
#'   columns `average_case` and `adjusted_average_case` added.
#' * `settlement_rate_adjust()` restates `paid` and `closed_counts` at the
#'   settlement rate of the latest diagonal: closed counts are the selected
#'   disposal rates times each origin's ultimate counts, and paid claims
#'   are read off a curve through each origin's observed (closed count,
#'   paid) pairs at those counts.
#' * `berquist_sherman_adjust()` does both, with incurred as the adjusted
#'   average case times the adjusted open counts plus the adjusted paid.
#'
#' `exponential_trend()`, `trend_by_age()` (down the origins at each age)
#' and `pooled_trend()` (one slope over every age, an intercept for each)
#' fit the trends the adjustments select from; `ratio_column()` adds a
#' column such as average paid.
#'
#' @param triangle A [triangle] with one segment.
#' @param incurred,paid The reported and paid claims columns.
#' @param trend The annual severity trend, `"average_case"` to fit it from
#'   the average case outstanding, or two column names, an amount and a count
#'   (paid claims and closed counts), to fit it from.
#' @param open_counts A column of open counts. Otherwise give `reported`
#'   and `closed`.
#' @param reported,closed Count columns whose difference is the open counts.
#' @param max_age The oldest age adjusted, where older averages are too thin
#'   to use.
#' @param closed_counts The closed count column.
#' @param ultimate_counts Numbers, one per origin; a column; a
#'   [development_method()]; or several for their mean.
#' @param disposal_rates A selection, as in [disposal_method()].
#' @param interpolation `"exponential"` or `"linear"`, the curve between two
#'   successive points.
#' @param x,y Times in years and positive values.
#' @param column A column of `triangle`.
#' @param numerator,denominator,name The columns of a ratio and its name.
#' @returns A [triangle]; the trends return `rate`, `r_squared` and `n`, a
#'   list or for `trend_by_age()` a data frame with `age`.
#' @name berquist_sherman
#' @examples
#' tri <- triangle(data.frame(origin = c(2020, 2020, 2021), age = c(12, 24, 12),
#'                            incurred = c(100, 120, 150), paid = c(50, 100, 50),
#'                            open = c(5, 2, 5)),
#'                 origin = "origin", development = "age",
#'                 columns = c("incurred", "paid", "open"))
#' case_adequacy_adjust(tri, "incurred", "paid", 0.25, open_counts = "open")
NULL

#' @rdname berquist_sherman
#' @export
case_adequacy_adjust <- function(triangle, incurred, paid, trend, open_counts = NULL,
                                 reported = NULL, closed = NULL, max_age = NULL) {
  check_triangle(triangle)
  new_triangle(triangle, reserving_result(case_adequacy_rust(
    triangle@ptr, incurred, paid, open_counts_arg(open_counts, reported, closed),
    case_trend_arg(trend), if (is.null(max_age)) NaN else single_number(max_age, "max_age"))))
}

#' @rdname berquist_sherman
#' @export
average_case <- function(triangle, incurred, paid, trend, open_counts = NULL, reported = NULL,
                         closed = NULL, max_age = NULL) {
  check_triangle(triangle)
  new_triangle(triangle, reserving_result(average_case_rust(
    triangle@ptr, incurred, paid, open_counts_arg(open_counts, reported, closed),
    case_trend_arg(trend), if (is.null(max_age)) NaN else single_number(max_age, "max_age"))))
}

#' @rdname berquist_sherman
#' @export
settlement_rate_adjust <- function(triangle, paid, closed_counts, ultimate_counts,
                                   disposal_rates = "latest_diagonal",
                                   interpolation = c("exponential", "linear")) {
  check_triangle(triangle)
  interpolation <- match.arg(interpolation)
  new_triangle(triangle, reserving_result(settlement_rate_rust(
    triangle@ptr, paid, closed_counts, ultimate_counts_arg(ultimate_counts),
    selection_arg(disposal_rates, "disposal_rates"), interpolation)))
}

#' @rdname berquist_sherman
#' @export
berquist_sherman_adjust <- function(triangle, incurred, paid, reported, closed_counts, trend,
                                    ultimate_counts, max_age = NULL,
                                    disposal_rates = "latest_diagonal",
                                    interpolation = c("exponential", "linear")) {
  check_triangle(triangle)
  interpolation <- match.arg(interpolation)
  new_triangle(triangle, reserving_result(berquist_sherman_rust(
    triangle@ptr, incurred, paid, open_counts_arg(NULL, reported, closed_counts), closed_counts,
    case_trend_arg(trend), ultimate_counts_arg(ultimate_counts),
    if (is.null(max_age)) NaN else single_number(max_age, "max_age"),
    selection_arg(disposal_rates, "disposal_rates"), interpolation)))
}

trend_list <- function(v) {
  if (is.nan(v[[1]])) return(NULL)
  list(rate = v[[1]], r_squared = v[[2]], n = v[[3]])
}

#' @rdname berquist_sherman
#' @export
exponential_trend <- function(x, y) {
  trend_list(reserving_result(exponential_trend_rust(as.double(x), as.double(y))))
}

#' @rdname berquist_sherman
#' @export
trend_by_age <- function(triangle, column) {
  check_triangle(triangle)
  out <- reserving_result(trend_by_age_rust(triangle@ptr, column))
  as.data.frame(lapply(out, function(v) replace(v, is.nan(v), NA)))
}

#' @rdname berquist_sherman
#' @export
pooled_trend <- function(triangle, column) {
  check_triangle(triangle)
  trend_list(reserving_result(pooled_trend_rust(triangle@ptr, column)))
}

#' @rdname berquist_sherman
#' @export
ratio_column <- function(triangle, numerator, denominator, name) {
  check_triangle(triangle)
  new_triangle(triangle, reserving_result(ratio_column_rust(triangle@ptr, numerator, denominator,
                                                            name)))
}
