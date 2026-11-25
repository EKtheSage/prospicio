#' @include reserving.R odp_glm.R
NULL

# A period label for the Rust side: a whole year as "2020", or the label
# itself ("2020H2", "2020Q3", "2020-07").
period_label <- function(x, arg = "first") {
  if (length(x) != 1L || is.na(x)) stop(arg, " must be one period", call. = FALSE)
  if (is.numeric(x)) {
    if (x != round(x)) stop(arg, " must be a whole year or a period label", call. = FALSE)
    return(as.character(as.integer(x)))
  }
  as.character(x)
}

# Month labels "YYYY-MM" for the Rust side, from labels or dates.
month_labels <- function(x, arg = "month") {
  if (inherits(x, c("Date", "POSIXct"))) return(format(x, "%Y-%m"))
  if (!is.character(x)) {
    stop(arg, " must be \"YYYY-MM\" labels or dates", call. = FALSE)
  }
  x
}

#' An index by calendar period
#'
#' One value per period of one grain, from a first period on, optionally
#' extended past its last period at a constant annual rate: a CPI, a wage
#' index or a claims-inflation index (issue #979, design note
#' `docs/design/reserving-inflation.md`). Python's `CalendarIndex`.
#' `calendar_index_value()` reads the index of the period that contains each
#' month.
#'
#' @param first The first period: `"2020"` (or the number 2020), `"2020H2"`,
#'   `"2020Q3"` or `"2020-07"`. Its grain is the index's grain.
#' @param values The index in `first`, then one value per period: finite and
#'   positive.
#' @param future_rate The annual rate past the last period, compounded by
#'   months from the last period's start: finite and above -1. `NULL` (the
#'   default) leaves a period past the last an error.
#' @param index A `calendar_index`.
#' @param month Months as `"YYYY-MM"` labels or dates.
#' @returns `calendar_index()`: a `calendar_index` with the properties
#'   `first`, `values`, `future_rate` and `periods` (the given periods, one
#'   per value). `calendar_index_value()`: a numeric vector, one value per
#'   month; an error for a month before the first period, or after the last
#'   when there is no future rate.
#' @seealso [indexation()], [future_cells()]
#' @export
#' @examples
#' cpi <- calendar_index("2020", c(100, 104, 110), future_rate = 0.05)
#' cpi@periods
#' calendar_index_value(cpi, c("2021-06", "2024-01"))
calendar_index <- S7::new_class(
  "calendar_index",
  package = "prospicio",
  properties = list(
    first = S7::class_character,
    values = S7::class_double,
    future_rate = S7::new_union(S7::class_double, NULL),
    periods = S7::new_property(S7::class_character, getter = function(self) {
      reserving_result(calendar_index_periods_rust(self@first, self@values, self@future_rate),
                       call = NULL)
    })
  ),
  constructor = function(first, values, future_rate = NULL) {
    first <- period_label(first)
    values <- as.double(values)
    if (!is.null(future_rate)) future_rate <- as.double(future_rate)
    reserving_result(calendar_index_periods_rust(first, values, future_rate))
    S7::new_object(S7::S7_object(), first = first, values = values, future_rate = future_rate)
  }
)

S7::method(print, calendar_index) <- function(x, ...) {
  rate <- if (is.null(x@future_rate)) "" else paste0(", future_rate = ", format(x@future_rate, digits = 15))
  cat(sprintf("calendar_index(first = \"%s\", values = c(%s)%s)\n", x@first,
              paste(format(x@values, digits = 15), collapse = ", "), rate))
  invisible(x)
}

check_calendar_index <- function(index) {
  if (!S7::S7_inherits(index, calendar_index)) stop("index must be a calendar_index()", call. = FALSE)
  index
}

#' @rdname calendar_index
#' @export
calendar_index_value <- function(index, month) {
  index <- check_calendar_index(index)
  reserving_result(calendar_index_value_rust(index@first, index@values, index@future_rate,
                                             month_labels(month)))
}

#' Indexation of a triangle and re-inflation of its projection
#'
#' An [calendar_index()] restates each increment of a triangle at the price
#' level of one base period; the restated triangle is developed by any
#' method; its projected future increments ([future_cells()]) are taken back
#' to the money of their payment periods (Taylor 1977, equation 8; design
#' note `docs/design/reserving-inflation.md`). Python's `Indexation`.
#'
#' * `restate()` divides each increment of `column` by `I(p) / I(base)`, `p`
#'   the index period that contains the cell's valuation month, in the
#'   triangle's basis (a cumulative triangle is differenced, restated and
#'   cumulated again), in every segment. Other columns are unchanged. An
#'   increment after a hole is restated at its own cell's period.
#' * `inflate()` is its inverse.
#' * `reinflate()` takes future increments at the base level to the money of
#'   their payment periods and sums them by origin. The index past its last
#'   given period needs a `future_rate`.
#' * `indexation_factor()` is `I(p) / I(base)` at each month: the factor
#'   that takes an amount at the base level to the money of `p`.
#'
#' @param index A [calendar_index()].
#' @param base The base price level: the index period that contains this
#'   month (a `"YYYY-MM"` label or a date).
#' @param x An `indexation`.
#' @param triangle A [triangle()].
#' @param column The column to restate; `NULL` for the triangle's only one.
#' @param cells [future_cells()] at the base level; for `reinflated()`, the
#'   re-inflated cells.
#' @param reserves,base_reserves For `reinflated()`: the reserves by origin in
#'   the money of the payment periods and at the base level. A `reinflated`
#'   is made by `reinflate()`; the constructor only holds its fields.
#' @param month Months as `"YYYY-MM"` labels or dates.
#' @returns `indexation()`: an `indexation` with the properties `index` and
#'   `base` (a `"YYYY-MM"` label). `restate()` and `inflate()`: a
#'   [triangle()]. `reinflate()`: a `reinflated` with the properties `cells`
#'   (the re-inflated [future_cells()]), `reserves` and `base_reserves` (one
#'   per origin), `total_reserve` and `total_base_reserve`.
#'   `indexation_factor()`: a numeric vector.
#' @seealso [calendar_index()], [future_cells()]
#' @export
#' @examples
#' long <- data.frame(
#'   year = c(2020, 2020, 2021), age = c(12, 24, 12), paid = c(100, 155, 110)
#' )
#' tri <- triangle(long, "year", "age", "paid")
#' at_2020 <- indexation(calendar_index("2020", c(1, 1.1), future_rate = 0.1), "2020-01")
#' restate(at_2020, tri)@values[1, 1, , ]
#' cells <- future_cells(data.frame(origin = "2021", age = 24, valuation = "2022-12", value = 50))
#' reinflate(at_2020, cells)@total_reserve
indexation <- S7::new_class(
  "indexation",
  package = "prospicio",
  properties = list(
    index = calendar_index,
    base = S7::class_character
  ),
  constructor = function(index, base) {
    index <- check_calendar_index(index)
    base <- month_labels(base, "base")
    if (length(base) != 1L) stop("base must be one month", call. = FALSE)
    base <- reserving_result(indexation_check_rust(index@first, index@values, index@future_rate, base))
    S7::new_object(S7::S7_object(), index = index, base = base)
  }
)

S7::method(print, indexation) <- function(x, ...) {
  cat(sprintf("indexation(base = \"%s\")\n", x@base))
  print(x@index)
  invisible(x)
}

check_indexation <- function(x) {
  if (!S7::S7_inherits(x, indexation)) stop("x must be an indexation()", call. = FALSE)
  x
}

#' @rdname indexation
#' @export
indexation_factor <- function(x, month) {
  x <- check_indexation(x)
  reserving_result(indexation_factor_rust(x@index@first, x@index@values, x@index@future_rate,
                                          x@base, month_labels(month)))
}

indexation_apply <- function(x, triangle, column, inflate) {
  x <- check_indexation(x)
  column <- fit_column(triangle, column)
  ptr <- reserving_result(indexation_apply_rust(triangle@ptr, column, x@index@first, x@index@values,
                                                x@index@future_rate, x@base, inflate))
  new_triangle(triangle, ptr)
}

#' @rdname indexation
#' @export
restate <- function(x, triangle, column = NULL) indexation_apply(x, triangle, column, FALSE)

#' @rdname indexation
#' @export
inflate <- function(x, triangle, column = NULL) indexation_apply(x, triangle, column, TRUE)

#' Projected future increments, for re-inflation
#'
#' The future increments of one segment, one per origin and age, as
#' [reinflate()] takes them (design note `docs/design/reserving-inflation.md`
#' decision 6). Python's `FutureCells`.
#'
#' * For a [chain_ladder_fit], the cells are at each age of the grid after
#'   an origin's latest one the projected cumulative value less the one
#'   before, then the rest of the ultimate (the tail) as one increment one
#'   development period after the oldest age. The cells sum to the fit's
#'   reserves. `segment` names the one segment of a fit over several, as a
#'   named list or character vector of key values (`c(lob = "auto")`); the
#'   default needs a fit of one segment.
#' * For an [odp_glm_fit], the cells are the future cells' means.
#' * For a data frame, the columns `origin`, `age` (months), `valuation`
#'   (the month the cell's development period ends in, a `"YYYY-MM"` label
#'   or a date: its index period is the cell's payment period) and `value`
#'   (at the base level). `origins` lists the origin periods (`"2020"`,
#'   `"2020Q3"`, ...), by default the origins in the order they occur.
#'
#' @param x A [chain_ladder_fit], an [odp_glm_fit] or a data frame.
#' @param segment For a [chain_ladder_fit]: the key values of the segment.
#' @param origins For a data frame: the origin periods.
#' @returns A `future_cells` with the properties `origins`, `cells` (a data
#'   frame with the columns `origin`, `age`, `valuation` and `value`) and
#'   `reserves` (the sum of the cells by origin).
#' @seealso [indexation()]
#' @export
#' @examples
#' cells <- future_cells(data.frame(
#'   origin = c("2020", "2020"), age = c(24, 36),
#'   valuation = c("2021-12", "2022-12"), value = c(50, 25)
#' ))
#' cells@reserves
future_cells <- S7::new_class(
  "future_cells",
  package = "prospicio",
  properties = list(
    origins = S7::class_character,
    cells = S7::class_data.frame,
    reserves = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(vapply(self@origins, function(o) sum(self@cells$value[self@cells$origin == o]), 0),
                      self@origins)
    })
  ),
  constructor = function(x, segment = NULL, origins = NULL) {
    if (missing(x)) {
      stop("x must be a chain_ladder_fit, an odp_glm_fit or a data frame", call. = FALSE)
    }
    if (S7::S7_inherits(x, chain_ladder_fit)) {
      keys <- if (is.null(segment)) character() else names(segment)
      values <- if (is.null(segment)) character() else as.character(unlist(segment, use.names = FALSE))
      if (!is.null(segment) && is.null(keys)) {
        stop("segment must be named by key, as c(lob = \"auto\")", call. = FALSE)
      }
      out <- reserving_result(future_cells_chain_ladder_rust(x@ptr, keys, values))
    } else if (S7::S7_inherits(x, odp_glm_fit)) {
      out <- reserving_result(future_cells_odp_glm_rust(x@ptr))
    } else if (is.data.frame(x)) {
      missing <- setdiff(c("origin", "age", "valuation", "value"), names(x))
      if (length(missing)) {
        stop("x needs the columns origin, age, valuation and value; missing ",
             paste(missing, collapse = ", "), call. = FALSE)
      }
      origin <- as.character(x$origin)
      if (is.null(origins)) origins <- unique(origin)
      age <- x$age
      bad <- if (is.numeric(age)) which(is.na(age) | age != round(age) | age < 0) else 1L
      if (length(bad)) {
        stop(sprintf("age %s is not an age in months", format(age[bad[1]])), call. = FALSE)
      }
      out <- reserving_result(future_cells_check_rust(
        as.character(origins), origin, as.integer(age),
        month_labels(x$valuation, "valuation"), as.double(x$value)
      ))
    } else {
      stop("x must be a chain_ladder_fit, an odp_glm_fit or a data frame", call. = FALSE)
    }
    S7::new_object(
      S7::S7_object(),
      origins = out$origins,
      cells = data.frame(origin = out$origin, age = out$age, valuation = out$valuation,
                         value = out$value, stringsAsFactors = FALSE)
    )
  }
)

S7::method(print, future_cells) <- function(x, ...) {
  cat(sprintf("future_cells: %d cells over %d origins\n", nrow(x@cells), length(x@origins)))
  invisible(x)
}

#' @rdname indexation
#' @export
reinflated <- S7::new_class(
  "reinflated",
  package = "prospicio",
  properties = list(
    cells = future_cells,
    reserves = S7::class_double,
    base_reserves = S7::class_double,
    total_reserve = S7::new_property(S7::class_double, getter = function(self) sum(self@reserves)),
    total_base_reserve = S7::new_property(S7::class_double, getter = function(self) {
      sum(self@base_reserves)
    })
  ),
  constructor = function(cells, reserves, base_reserves) {
    if (missing(cells) || missing(reserves) || missing(base_reserves)) {
      stop("a reinflated is made by reinflate(); give cells, reserves and base_reserves",
           call. = FALSE)
    }
    S7::new_object(S7::S7_object(), cells = cells, reserves = reserves,
                   base_reserves = base_reserves)
  }
)

S7::method(print, reinflated) <- function(x, ...) {
  cat(sprintf("reinflated: total_reserve = %s, total_base_reserve = %s\n",
              format(x@total_reserve, digits = 15), format(x@total_base_reserve, digits = 15)))
  invisible(x)
}

#' @rdname indexation
#' @export
reinflate <- function(x, cells) {
  x <- check_indexation(x)
  if (!S7::S7_inherits(cells, future_cells)) stop("cells must be future_cells()", call. = FALSE)
  tab <- cells@cells
  out <- reserving_result(reinflate_rust(
    x@index@first, x@index@values, x@index@future_rate, x@base, cells@origins,
    tab$origin, as.integer(tab$age), tab$valuation, tab$value
  ))
  names(out$reserves) <- out$origins
  names(out$base_reserves) <- out$origins
  reinflated(
    cells = future_cells(
      data.frame(origin = out$origin, age = out$age, valuation = out$valuation, value = out$value,
                 stringsAsFactors = FALSE),
      origins = out$origins
    ),
    reserves = out$reserves,
    base_reserves = out$base_reserves
  )
}
