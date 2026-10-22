# Reserving lane: the loss triangle, the chain ladder, Mack with its
# one-year view, the expected-loss methods, the ODP bootstrap and Clark's
# growth curves, over
# crates/prospicio-r/src/reserving.rs (docs/design/triangle.md,
# docs/design/reserving-v02.md). S7 classes and functions over the Rust
# objects, as in distributions.R.

#' @include distributions.R
NULL

# Year and month of each element of `x`: a Date, a POSIXct or whole-number
# years, which mean `month` of that year.
year_month <- function(x, arg, month) {
  if (inherits(x, "Date") || inherits(x, "POSIXt")) {
    lt <- as.POSIXlt(x)
    out <- list(year = lt$year + 1900L, month = lt$mon + 1L)
  } else if (is.numeric(x) && all(is.na(x) | x == round(x))) {
    out <- list(year = as.integer(x), month = rep(as.integer(month), length(x)))
  } else {
    stop(sprintf("%s must be Date, POSIXct or whole-number years", arg), call. = FALSE)
  }
  if (anyNA(out$year)) stop(sprintf("%s has missing values", arg), call. = FALSE)
  out
}

# First day of each (year, month).
month_start <- function(year, month) as.Date(sprintf("%04d-%02d-01", year, month))

#' Loss triangle
#'
#' A loss triangle with four axes, in chainladder-python's order: index
#' (segment, such as a line of business) x column (measure, such as paid
#' and incurred) x origin period x development age in months. Cells that
#' are not observed are `NA`; a zero is an observation.
#'
#' Segments are named by key columns such as `lob` and `state`. A triangle
#' without keys has one segment, labelled `"Total"`.
#'
#' `triangle()` builds one from a long table, one row per (keys, origin,
#' development). Origins span every period from the earliest to the latest
#' row and ages every development period from the youngest to the oldest.
#' Rows with the same (keys, origin, development) are summed, and `NA`
#' values are missing. Incremental input (`cumulative = FALSE`) follows
#' chainladder-python: a missing row is a period without movement.
#'
#' Read-only properties: `x@shape` (index, column, origin and development
#' lengths), `x@keys` (the key names), `x@index` (a data.frame with one
#' column per key and one row per segment; no columns without keys),
#' `x@columns`, `x@origins` (period labels such as `"1981"`,
#' `"2021Q3"`, `"2021H2"` or `"2021-08"`), `x@development` (ages in months),
#' `x@valuation` (the last day of the latest valuation month), `x@origin_grain`,
#' `x@development_grain`, `x@is_cumulative` and `x@values` (a 4-D array with
#' dimnames, whose index names join the key values with `" / "`).
#' `as.data.frame()` gives the long table back, with the key columns by
#' name.
#'
#' @param data A data.frame with one row per (keys, origin, development).
#' @param origin Name of the origin column: Date, POSIXct (any day in the
#'   origin period) or whole-number years.
#' @param development Name of the development column: ages in months (12,
#'   24, ...), or valuation dates when `development_is_valuation = TRUE`.
#' @param columns Names of the value columns (numeric).
#' @param keys Names of the key columns, such as `c("lob", "state")`, or
#'   `NULL` for a single segment labelled `"Total"`. Key values are stored
#'   as character and may not be `NA`; key names must differ from each other
#'   and from `columns`.
#' @param origin_grain,development_grain Length of an origin period and
#'   spacing of the development ages: `"M"` (month), `"Q"` (quarter), `"S"`
#'   (semester) or `"Y"` (year). The development grain must divide the
#'   origin grain.
#' @param cumulative Whether the values are cumulative (otherwise
#'   incremental).
#' @param development_is_valuation Whether `development` holds valuation dates (Date,
#'   POSIXct, or whole-number years meaning December of that year) instead
#'   of ages.
#' @param ptr A `Triangle` pointer; used internally.
#' @returns A `triangle` object.
#' @seealso [chain_ladder()], [mack()], [to_incremental()], [grain()],
#'   [subset()][subset.triangle], [aggregate()][aggregate.triangle],
#'   [as.matrix()][triangle_views], [summary()][triangle_views].
#' @export
#' @examples
#' long <- data.frame(
#'   year = c(2020, 2020, 2020, 2021, 2021, 2022),
#'   age = c(12, 24, 36, 12, 24, 12),
#'   paid = c(100, 150, 165, 110, 170, 120)
#' )
#' tri <- triangle(long, origin = "year", development = "age", columns = "paid")
#' tri
#' tri@values[1, "paid", , ]
#' tri@valuation
#'
#' # Two lines of business as a key column.
#' by_lob <- data.frame(
#'   lob = c("auto", "auto", "home"),
#'   year = c(2020, 2020, 2020),
#'   age = c(12, 24, 12),
#'   paid = c(100, 150, 40)
#' )
#' tri <- triangle(by_lob, "year", "age", "paid", keys = "lob")
#' tri@keys
#' tri@index
#' as.data.frame(tri)
triangle <- S7::new_class(
  "triangle",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("Triangle"),
    shape = S7::new_property(S7::class_integer, getter = function(self) {
      stats::setNames(self@ptr$shape(), c("index", "column", "origin", "development"))
    }),
    keys = S7::new_property(S7::class_character, getter = function(self) self@ptr$keys()),
    index = S7::new_property(S7::class_data.frame, getter = function(self) {
      parts <- self@ptr$index()
      # No keys: one segment, no columns.
      if (!length(parts)) return(data.frame(matrix(nrow = 1L, ncol = 0L)))
      as.data.frame(parts, stringsAsFactors = FALSE, optional = TRUE)
    }),
    columns = S7::new_property(S7::class_character, getter = function(self) self@ptr$columns()),
    origins = S7::new_property(S7::class_character, getter = function(self) self@ptr$origins()),
    development = S7::new_property(S7::class_integer, getter = function(self) self@ptr$development()),
    valuation = S7::new_property(S7::class_Date, getter = function(self) {
      first <- as.Date(paste0(self@ptr$valuation(), "-01"))
      seq(first, by = "month", length.out = 2)[2] - 1
    }),
    origin_grain = S7::new_property(S7::class_character, getter = function(self) self@ptr$origin_grain()),
    development_grain = S7::new_property(S7::class_character, getter = function(self) {
      self@ptr$development_grain()
    }),
    is_cumulative = S7::new_property(S7::class_logical, getter = function(self) self@ptr$is_cumulative()),
    values = S7::new_property(S7::class_double, getter = function(self) {
      triangle_array(self, self@ptr$values(), self@development)
    })
  ),
  constructor = function(data, origin, development, columns, keys = NULL,
                         origin_grain = "Y", development_grain = "Y",
                         cumulative = TRUE, development_is_valuation = FALSE,
                         ptr = NULL) {
    if (!is.null(ptr)) return(S7::new_object(S7::S7_object(), ptr = ptr))
    if (!is.data.frame(data)) stop("data must be a data.frame", call. = FALSE)
    keys <- as.character(keys)
    missing_cols <- setdiff(c(origin, development, columns, keys), names(data))
    if (length(missing_cols)) {
      stop("no columns named ", toString(missing_cols), " in data", call. = FALSE)
    }
    # as.double() would turn a factor into its level codes and a character
    # column into NA with only a warning, so insist on numbers.
    numeric_column <- function(name, what) {
      x <- data[[name]]
      if (!is.numeric(x)) {
        stop(sprintf("%s column %s must be numeric, not %s", what, name, class(x)[1]), call. = FALSE)
      }
      as.double(x)
    }
    o <- year_month(data[[origin]], "origin", 1L)
    if (isTRUE(development_is_valuation)) {
      v <- year_month(data[[development]], "development", 12L)
      ages <- double()
    } else {
      v <- list(year = integer(), month = integer())
      ages <- numeric_column(development, "development")
      if (anyNA(ages)) stop("development has missing values", call. = FALSE)
    }
    key_values <- lapply(keys, function(k) {
      v <- as.character(data[[k]])
      if (anyNA(v)) stop(sprintf("key column %s has missing values", k), call. = FALSE)
      v
    })
    values <- unlist(lapply(columns, numeric_column, what = "value"))
    ptr <- rust_result(Triangle$from_long(
      keys, as.character(unlist(key_values)), o$year, o$month,
      ages, v$year, v$month, isTRUE(development_is_valuation),
      as.character(columns), as.double(values),
      as.character(origin_grain), as.character(development_grain), isTRUE(cumulative)
    ))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

# A row-major (index, column, origin, development-like) vector as an R
# array with the triangle's dimnames; NaN (unobserved) becomes NA.
triangle_array <- function(x, flat, development) {
  dims <- c(length(x@ptr$index_names()), length(x@columns), length(x@origins),
            length(development))
  a <- aperm(array(flat, dim = rev(dims)), rev(seq_along(dims)))
  a[is.nan(a)] <- NA_real_
  dimnames(a) <- list(index = x@ptr$index_names(), column = x@columns,
                      origin = x@origins, development = as.character(development))
  a
}

new_triangle <- function(x, ptr) triangle(ptr = ptr)

# Last day of each "YYYY-MM" month; NA for "" or NA.
month_end <- function(ym) {
  out <- rep(as.Date(NA), length(ym))
  ok <- !is.na(ym) & nzchar(ym)
  n <- nchar(ym[ok])
  y <- as.integer(substr(ym[ok], 1, n - 3))
  m <- as.integer(substr(ym[ok], n - 1, n))
  out[ok] <- month_start(y + m %/% 12, m %% 12 + 1) - 1
  out
}

#' View, summarise and print a triangle
#'
#' `as.matrix()` gives one segment and measure of a [triangle] as an
#' origin x development matrix: origin labels as row names, ages in months
#' as column names, `NA` where a cell is not observed. Key arguments choose
#' the segment, one value each (compared as character); keys not named may
#' take any value, but the choice must leave one segment, so a triangle
#' with one segment needs none. `column` names the measure and may be left
#' out when there is only one. This is Python's
#' `Triangle.view(column=None, **keys)`.
#'
#' `summary()` gives one row per segment and measure: the key columns,
#' `column`, `n_origins` (origins with an observed value), `first_origin`
#' and `last_origin` of those, `valuation` (the last day of the latest
#' valuation with an observed value), `latest` (the sum over origins of the
#' latest cumulative value, so for an incremental triangle the sum of every
#' increment) and `cumulative`. Origins and valuation are `NA` for a
#' segment with no observed value of the measure.
#'
#' `print()` and `format()` show the grid of `as.matrix()` for a triangle
#' with one segment and one measure, and the `summary()` table otherwise,
#' with numbers rounded for reading; long tables show their first and last
#' rows around a `...`. Python prints the same text.
#'
#' `as.matrix()` is a method on base R's generic rather than a new `view()`
#' verb, which would mask `tibble::view()` once the tidyverse is attached.
#' A key named `x` or `column` cannot be chosen this way; use
#' [subset()][subset.triangle] first.
#'
#' Errors: an unknown key, value or column, a choice that matches several
#' segments, a missing `column` with several measures, or a key with the
#' name of a summary column.
#'
#' @param x,object A [triangle].
#' @param column Name of the measure, or `NULL` for the only one.
#' @param max_rows,max_cols Rows and development ages shown before the
#'   middle ones are left out; 0 for no limit.
#' @param ... For `as.matrix()`, key conditions as `key = value`; unused
#'   otherwise.
#' @returns `as.matrix()`: a numeric matrix. `summary()`: a data.frame.
#'   `format()`: a character vector of lines. `print()`: `x`, invisibly.
#' @name triangle_views
#' @examples
#' long <- data.frame(lob = rep(c("auto", "home"), each = 3), year = c(2020, 2020, 2021),
#'                    age = c(12, 24, 12), paid = c(100, 150, 110, 40, 60, 45))
#' tri <- triangle(long, "year", "age", "paid", keys = "lob")
#' tri
#' as.matrix(tri, lob = "auto")
#' summary(tri)
#' subset(tri, lob = "home")
NULL

S7::method(as.matrix, triangle) <- function(x, ..., column = NULL) {
  call <- sys.call()
  call[[1]] <- quote(as.matrix)
  conditions <- list(...)
  keys <- names(conditions)
  if (length(conditions) && (is.null(keys) || any(keys == ""))) {
    stop("as.matrix() conditions must be named by key, as in lob = \"auto\"", call. = FALSE)
  }
  if (any(lengths(conditions) != 1)) {
    stop("as.matrix() takes one value per key", call. = FALSE)
  }
  values <- vapply(conditions, as.character, "", USE.NAMES = FALSE)
  col <- if (is.null(column)) NULL else as.character(column)
  v <- rust_result(x@ptr$view(as.character(keys), values, col), call = call)
  m <- matrix(v$values, nrow = length(v$origins), ncol = length(v$development), byrow = TRUE,
              dimnames = list(origin = v$origins, development = as.character(v$development)))
  m[is.nan(m)] <- NA_real_
  m
}

S7::method(summary, triangle) <- function(object, ...) {
  fixed <- c("column", "n_origins", "first_origin", "last_origin", "valuation", "latest",
             "cumulative")
  clash <- intersect(object@keys, fixed)
  if (length(clash)) {
    stop(sprintf('key "%s" clashes with the "%s" column of the summary', clash[1], clash[1]),
         call. = FALSE)
  }
  s <- object@ptr$summary()
  blank_na <- function(v) replace(v, !nzchar(v), NA_character_)
  out <- s$keys
  out$column <- s$column
  out$n_origins <- s$n_origins
  out$first_origin <- blank_na(s$first_origin)
  out$last_origin <- blank_na(s$last_origin)
  out$valuation <- month_end(s$valuation)
  out$latest <- s$latest
  out$cumulative <- rep(s$cumulative, length(s$column))
  as.data.frame(out, stringsAsFactors = FALSE, optional = TRUE)
}

S7::method(format, triangle) <- function(x, max_rows = 20, max_cols = 12, ...) {
  call <- sys.call()
  call[[1]] <- quote(format)
  text <- rust_result(x@ptr$to_text(as.double(max_rows), as.double(max_cols)), call = call)
  strsplit(text, "\n", fixed = TRUE)[[1]]
}

S7::method(print, triangle) <- function(x, ...) {
  cat(format(x, ...), sep = "\n")
  invisible(x)
}

S7::method(as.data.frame, triangle) <- function(x, ...) {
  clash <- intersect(c(x@keys, x@columns), c("origin", "development"))
  if (length(clash)) {
    stop(sprintf('column "%s" clashes with the "%s" column of the long table', clash[1], clash[1]),
         call. = FALSE)
  }
  long <- x@ptr$to_long()
  out <- long$keys
  out$origin <- month_start(long$origin_year, long$origin_month)
  out$development <- long$development
  values <- lapply(long$values, function(v) replace(v, is.nan(v), NA_real_))
  out[long$names] <- values
  as.data.frame(out, stringsAsFactors = FALSE, optional = TRUE)
}

#' Select segments and columns of a triangle by name
#'
#' Keeps the segments whose key values match and the measure columns named:
#' `subset(tri, lob = "auto", state = c("CA", "NY"), columns = "paid")`.
#' Each key argument gives the values to keep (compared as character, as
#' keys are stored); segments must match every key argument and keep their
#' order. Columns are kept in the order given. This is Python's
#' `Triangle.select(columns=None, **keys)`.
#'
#' The method is on base R's [subset()] generic rather than a new verb, so
#' it does not mask `dplyr::filter()` or `dplyr::select()`. A key named `x`
#' or `columns` cannot be selected this way.
#'
#' Errors: a key, value or column that does not exist or is given twice, an
#' empty set of values, an unnamed argument, or a selection that matches no
#' segment.
#'
#' @param x A [triangle].
#' @param ... Key conditions as `key = values`.
#' @param columns Column names to keep, in order, or `NULL` for all.
#' @returns A [triangle].
#' @seealso [aggregate()][aggregate.triangle] to sum segments over keys.
#' @name subset.triangle
#' @examples
#' long <- data.frame(lob = rep(c("auto", "home"), each = 3), year = c(2020, 2020, 2021),
#'                    age = c(12, 24, 12), paid = 1:6, incurred = 2 * (1:6))
#' tri <- triangle(long, "year", "age", c("paid", "incurred"), keys = "lob")
#' subset(tri, lob = "home", columns = "incurred")@values
#' subset(tri, columns = c("incurred", "paid"))@columns
NULL

S7::method(subset, triangle) <- function(x, ..., columns = NULL) {
  # Name the generic in errors, not the S7-registered method.
  call <- sys.call()
  call[[1]] <- quote(subset)
  conditions <- list(...)
  keys <- names(conditions)
  if (length(conditions) && (is.null(keys) || any(keys == ""))) {
    stop("subset() conditions must be named by key, as in lob = \"auto\"", call. = FALSE)
  }
  values <- lapply(unname(conditions), as.character)
  cols <- if (is.null(columns)) NULL else as.character(columns)
  new_triangle(x, rust_result(x@ptr$select(as.character(keys), values, cols), call = call))
}

#' Sum a triangle over keys
#'
#' Sums the segments that share the values of the `keep` keys and drops the
#' other keys: `aggregate(tri, keep = "lob")` sums states within each line,
#' and the default `keep = character()` sums every segment into one. The
#' result has the `keep` keys in the order given and its segments sorted by
#' them. Cumulative values are summed cell by cell, and a cell is observed
#' if any segment in the group observes it; an incremental triangle is
#' summed as cumulative values and returned incremental. This is Python's
#' `Triangle.group_by(keys)`.
#'
#' The method is on [stats::aggregate()], the generic this package already
#' uses for the same operation on a [predictive_distribution] (also with
#' `keep`), rather than `group_by()`, which would mask dplyr's lazy
#' grouping verb of a different meaning.
#'
#' @param x A [triangle].
#' @param keep Names of the keys to keep; unknown or repeated keys are an
#'   error.
#' @param ... Unused.
#' @returns A [triangle].
#' @seealso [subset()][subset.triangle] to select segments by key value.
#' @name aggregate.triangle
#' @examples
#' long <- data.frame(lob = c("auto", "auto", "home"), state = c("CA", "NY", "NY"),
#'                    year = 2020, age = 12, paid = c(1, 2, 3))
#' tri <- triangle(long, "year", "age", "paid", keys = c("lob", "state"))
#' aggregate(tri, keep = "lob")@index
#' as.data.frame(aggregate(tri, keep = "lob"))
#' aggregate(tri)@values[1, "paid", , ]
NULL

S7::method(aggregate, triangle) <- function(x, keep = character(), ...) {
  call <- sys.call()
  call[[1]] <- quote(aggregate)
  new_triangle(x, rust_result(x@ptr$group_by(as.character(keep)), call = call))
}

check_triangle <- function(x) {
  if (!S7::S7_inherits(x, triangle)) stop("x must be a triangle", call. = FALSE)
}

#' Incremental and cumulative triangles
#'
#' `to_incremental()` subtracts from each observed value the previous
#' observed value of its row, so the increment after a hole covers the
#' whole gap; `to_cumulative()` takes running sums of the observed
#' increments. Each returns `x` unchanged if it is already in that form.
#'
#' @param x A [triangle].
#' @returns A [triangle].
#' @export
#' @examples
#' long <- data.frame(year = c(2020, 2020, 2021), age = c(12, 24, 12), paid = c(100, 150, 110))
#' tri <- triangle(long, "year", "age", "paid")
#' to_incremental(tri)@values[1, 1, , ]
#' to_cumulative(to_incremental(tri))@values[1, 1, , ]
to_incremental <- function(x) {
  check_triangle(x)
  new_triangle(x, x@ptr$to_incremental())
}

#' @rdname to_incremental
#' @export
to_cumulative <- function(x) {
  check_triangle(x)
  new_triangle(x, x@ptr$to_cumulative())
}

#' Latest diagonal of a triangle
#'
#' The latest observed value of every (index, column, origin).
#'
#' @param x A [triangle].
#' @returns A 3-D array (index x column x origin) with dimnames, `NA` where
#'   an origin has no observation.
#' @export
#' @examples
#' long <- data.frame(year = c(2020, 2020, 2021), age = c(12, 24, 12), paid = c(100, 150, 110))
#' latest_diagonal(triangle(long, "year", "age", "paid"))[1, "paid", ]
latest_diagonal <- function(x) {
  check_triangle(x)
  dims <- c(length(x@ptr$index_names()), length(x@columns), length(x@origins))
  a <- aperm(array(x@ptr$latest_diagonal(), dim = rev(dims)), 3:1)
  a[is.nan(a)] <- NA_real_
  dimnames(a) <- list(index = x@ptr$index_names(), column = x@columns, origin = x@origins)
  a
}

#' Link ratios of a triangle
#'
#' Age-to-age ratios of the cumulative values. Development position `k`
#' holds the ratio from age `k` to age `k + 1`, observed only where both
#' ages are observed and the earlier value is non-zero.
#'
#' @param x A [triangle].
#' @returns A [triangle] with one development position fewer, marked as not
#'   cumulative.
#' @export
#' @examples
#' long <- data.frame(year = c(2020, 2020, 2021), age = c(12, 24, 12), paid = c(100, 150, 110))
#' link_ratios(triangle(long, "year", "age", "paid"))@values[1, 1, , ]
link_ratios <- function(x) {
  check_triangle(x)
  new_triangle(x, x@ptr$link_ratios())
}

#' Change the grain of a triangle
#'
#' Coarsens the origin and development periods, as chainladder-python's
#' `grain()`. Both new grains must be multiples of the current ones, and
#' the development grain must divide the origin grain. Valuations are kept
#' every development period back from the triangle's valuation, so a
#' partial latest period keeps its exact latest diagonal.
#'
#' @param x A [triangle].
#' @param origin_grain The new origin grain: `"M"`, `"Q"`, `"S"` or `"Y"`.
#' @param development_grain The new development grain, by default the same
#'   as `origin_grain`.
#' @returns A [triangle].
#' @export
#' @examples
#' long <- data.frame(
#'   origin = as.Date(c("2020-01-01", "2020-01-01", "2020-04-01")),
#'   age = c(3, 6, 3), paid = c(10, 20, 10)
#' )
#' q <- triangle(long, "origin", "age", "paid", origin_grain = "Q", development_grain = "Q")
#' grain(q, "Y", "Q")@values[1, 1, , ]
#' grain(q, "Y")@development
grain <- function(x, origin_grain, development_grain = origin_grain) {
  check_triangle(x)
  ptr <- rust_result(x@ptr$grain(as.character(origin_grain), as.character(development_grain)))
  new_triangle(x, ptr)
}

# The single column to fit, by default the triangle's only one.
fit_column <- function(triangle, column) {
  check_triangle(triangle)
  if (!is.null(column)) return(as.character(column))
  if (length(triangle@columns) != 1) {
    stop("triangle has several columns; name the one to fit with `column`", call. = FALSE)
  }
  triangle@columns
}

# Names of per-origin values: the origin, or "segment / origin" when the
# fit has several segments. `p` is a ChainLadderFit pointer.
origin_names <- function(p) {
  if (p$n_segments() > 1) paste(p$row_segments(), p$origins(), sep = " / ") else p$origins()
}

# A long result table from Rust (keys, origin or development, values) as a
# data.frame; NaN becomes NA.
fit_table_frame <- function(t) {
  fixed <- c(if (!is.null(t$origin)) "origin", if (!is.null(t$development)) "development", t$names)
  clash <- intersect(names(t$keys), fixed)
  if (length(clash)) {
    stop(sprintf('key "%s" clashes with the "%s" column of the result', clash[1], clash[1]),
         call. = FALSE)
  }
  out <- t$keys
  if (!is.null(t$origin)) out$origin <- t$origin
  if (!is.null(t$development)) out$development <- t$development
  out[t$names] <- lapply(t$values, function(v) replace(v, is.nan(v), NA_real_))
  as.data.frame(out, stringsAsFactors = FALSE, optional = TRUE)
}

# Chain-ladder properties, read through `cl(self)`, a ChainLadderFit
# pointer. Shared by chain_ladder_fit and mack_fit.
chain_ladder_properties <- function(cl) {
  links <- function(self) {
    ages <- cl(self)$development()
    paste(ages[-length(ages)], ages[-1], sep = "-")
  }
  by_origin <- function(f) {
    S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(f(cl(self)), origin_names(cl(self)))
    })
  }
  per_age <- function(f, names) {
    S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(rust_result(f(cl(self)), call = NULL), names(self))
    })
  }
  list(
    keys = S7::new_property(S7::class_character, getter = function(self) cl(self)$keys()),
    index = S7::new_property(S7::class_data.frame, getter = function(self) {
      parts <- cl(self)$totals_table()$keys
      if (!length(parts)) return(data.frame(matrix(nrow = 1L, ncol = 0L)))
      as.data.frame(parts, stringsAsFactors = FALSE, optional = TRUE)
    }),
    origins = S7::new_property(S7::class_character, getter = function(self) cl(self)$origins()),
    development = S7::new_property(S7::class_integer, getter = function(self) cl(self)$development()),
    ldf = per_age(function(p) p$ldf(), links),
    estimated_ldf = per_age(function(p) p$estimated_ldf(), links),
    cdf =per_age(function(p) p$cdf(), function(self) paste0(cl(self)$development(), "-Ult")),
    sigma = per_age(function(p) p$sigma(), links),
    std_err = per_age(function(p) p$std_err(), links),
    estimated_std_err = per_age(function(p) p$estimated_std_err(), links),
    selected_ldf = S7::new_property(S7::class_double, getter = function(self) {
      v <- rust_result(cl(self)$selected_ldf(), call = NULL)
      stats::setNames(replace(v, is.nan(v), NA_real_), links(self))
    }),
    ldf_weight = per_age(function(p) p$ldf_weight(), links),
    blended_ldf = per_age(function(p) p$blended_ldf(), links),
    pattern_rescale = S7::new_property(NULL | S7::class_double, getter = function(self) {
      v <- rust_result(cl(self)$pattern_rescale(), call = NULL)
      if (is.nan(v)) NULL else v
    }),
    pattern_tail = S7::new_property(NULL | S7::class_double, getter = function(self) {
      v <- rust_result(cl(self)$pattern_tail(), call = NULL)
      if (is.nan(v)) NULL else v
    }),
    alpha = S7::new_property(S7::class_double, getter = function(self) cl(self)$alpha()),
    n_links = S7::new_property(S7::class_integer, getter = function(self) {
      stats::setNames(rust_result(cl(self)$n_links(), call = NULL), links(self))
    }),
    excluded = S7::new_property(S7::class_logical, getter = function(self) {
      rust_result(cl(self)$excluded(), call = NULL)
    }),
    tail = S7::new_property(S7::class_double, getter = function(self) {
      rust_result(cl(self)$tail(), call = NULL)
    }),
    tail_attachment_age = S7::new_property(S7::class_integer, getter = function(self) {
      rust_result(cl(self)$tail_attachment_age(), call = NULL)
    }),
    tail_ldf = S7::new_property(S7::class_double, getter = function(self) {
      rust_result(cl(self)$tail_ldf(), call = NULL)
    }),
    tail_sigma = S7::new_property(S7::class_double, getter = function(self) {
      rust_result(cl(self)$tail_sigma(), call = NULL)
    }),
    tail_std_err = S7::new_property(S7::class_double, getter = function(self) {
      rust_result(cl(self)$tail_std_err(), call = NULL)
    }),
    latest = by_origin(function(p) p$latest()),
    ultimate = by_origin(function(p) p$ultimate()),
    reserve = by_origin(function(p) p$reserve()),
    total_ultimate = S7::new_property(S7::class_double, getter = function(self) cl(self)$total_ultimate()),
    total_reserve = S7::new_property(S7::class_double, getter = function(self) cl(self)$total_reserve())
  )
}

# A count of link ratios to keep or drop: NULL (when allowed) or a single
# whole number from 0. Whether the combination is allowed for a method is
# the library's to say.
exclusion_count <- function(x, arg, null_ok = FALSE) {
  if (null_ok && is.null(x)) return(NULL)
  if (!is.numeric(x) || length(x) != 1 || !is.finite(x) || x < 0 || x != round(x)) {
    stop(sprintf("%s must be %sa single whole number from 0", arg,
                 if (null_ok) "NULL or " else ""), call. = FALSE)
  }
  as.double(x)
}

development_args <- function(average, sigma_interpolation, n_periods = NULL, drop_high = 0,
                             drop_low = 0, args = c("n_periods", "drop_high", "drop_low")) {
  list(average = match.arg(average, c("volume", "simple", "regression", "geometric")),
       sigma = match.arg(sigma_interpolation, c("log-linear", "mack")),
       excl = list(n_periods = exclusion_count(n_periods, args[1], null_ok = TRUE),
                   drop_high = exclusion_count(drop_high, args[2]),
                   drop_low = exclusion_count(drop_low, args[3])))
}

# A selected pattern as the binding takes it: NULL without `ldf`, or a list
# of `kind`, `ages`, `values`, `weight_ages` (NULL for one weight) and
# `weights`. `ldf` is a named numeric vector (names are ages in months) or
# a data.frame with columns `age` and `value`; `ldf_weight` a number or a
# named numeric vector. The kind is required: age-to-age factors and factors
# to ultimate cannot be told apart.
ldf_pattern <- function(ldf, ldf_kind, ldf_weight) {
  default_weight <- is.numeric(ldf_weight) && length(ldf_weight) == 1 &&
    is.null(names(ldf_weight)) && isTRUE(ldf_weight == 1)
  if (is.null(ldf)) {
    if (!is.null(ldf_kind)) stop("ldf_kind is given without ldf", call. = FALSE)
    if (!default_weight) stop("ldf_weight is given without ldf", call. = FALSE)
    return(NULL)
  }
  kinds <- c("ldf", "cdf", "percent_developed")
  if (is.null(ldf_kind)) {
    stop('ldf needs ldf_kind: "ldf" (age-to-age factors), "cdf" (factors to ultimate) or ',
         '"percent_developed"; age-to-age factors and factors to ultimate cannot be told apart',
         call. = FALSE)
  }
  if (!is.character(ldf_kind) || length(ldf_kind) != 1 || !ldf_kind %in% kinds) {
    stop(sprintf('ldf_kind must be "ldf", "cdf" or "percent_developed", got %s',
                 paste(deparse(ldf_kind), collapse = " ")), call. = FALSE)
  }
  values <- age_values(ldf, "ldf")
  if (is.numeric(ldf_weight) && length(ldf_weight) == 1 && is.null(names(ldf_weight))) {
    if (is.na(ldf_weight)) stop("ldf_weight must not be NA", call. = FALSE)
    weight <- list(weight_ages = NULL, weights = as.double(ldf_weight))
  } else {
    w <- age_values(ldf_weight, "ldf_weight")
    weight <- list(weight_ages = w$ages, weights = w$values)
  }
  c(list(kind = ldf_kind, ages = values$ages, values = values$values), weight)
}

# Ages and values of a named numeric vector (names are ages in months) or a
# data.frame with columns `age` and `value`.
age_values <- function(x, arg) {
  if (is.data.frame(x)) {
    if (!all(c("age", "value") %in% names(x))) {
      stop(sprintf("%s as a data.frame needs the columns age and value", arg), call. = FALSE)
    }
    ages <- x$age
    values <- x$value
  } else if (is.numeric(x) && !is.null(names(x))) {
    ages <- suppressWarnings(as.numeric(names(x)))
    values <- unname(x)
  } else {
    stop(sprintf(paste("%s must be a named numeric vector, the names ages in months,",
                       "or a data.frame with columns age and value"), arg), call. = FALSE)
  }
  if (!is.numeric(ages) || anyNA(ages) || any(ages != round(ages)) || any(ages < 0)) {
    stop(sprintf("%s: every age must be a whole number of months", arg), call. = FALSE)
  }
  if (!is.numeric(values)) stop(sprintf("%s: every value must be a number", arg), call. = FALSE)
  list(ages = as.double(ages), values = as.double(values))
}

single_number <- function(x, arg) {
  if (!is.numeric(x) || length(x) != 1 || is.na(x)) {
    stop(sprintf("%s must be a single number", arg), call. = FALSE)
  }
  as.double(x)
}

# NULL (not given) or a single number; a single NA is NULL.
optional_number <- function(x, arg) {
  if (is.null(x) || (length(x) == 1 && is.na(x))) return(NULL)
  single_number(x, arg)
}

# `light_refit` as Rust takes it: NULL for the model's default (model 2's
# light refit), 0 for FALSE (turned off), an empty vector for TRUE (4 chains
# of 150 warm-up and 250 kept draws), or c(chains, tune, draws).
light_refit_arg <- function(light_refit) {
  if (is.null(light_refit)) return(NULL)
  if (identical(light_refit, FALSE)) return(0)
  if (identical(light_refit, TRUE)) return(numeric(0))
  if (!is.numeric(light_refit) || length(light_refit) != 3 || anyNA(light_refit)) {
    stop("light_refit must be NULL, TRUE, FALSE or c(chains, tune, draws)", call. = FALSE)
  }
  as.numeric(light_refit)
}

# A tail property that is an age in months, or NULL when not given.
tail_age <- function(name) {
  S7::new_property(NULL | S7::class_integer, getter = function(self) self@ptr$params()[[name]])
}

#' Constant tail factor
#'
#' A given factor from the attachment age to ultimate, as
#' chainladder-python's `TailConstant`. Past the oldest age the factor is
#' spread over the following periods as `1 + x * decay^k`, the last factor
#' making up the difference; this shapes the factors past the attachment
#' ([chain_ladder_fit]'s `tail_ldf`), not the factor to ultimate. An
#' attachment before the oldest age replaces the estimated factors from
#' there. A plain number given as `tail` to [chain_ladder()] or [mack()] is
#' `tail_constant(tail)`.
#'
#' Properties: `factor`, `decay` and `attachment_age` (`NULL` for the
#' oldest age).
#'
#' @param factor Factor from the attachment age to ultimate; finite and
#'   positive (checked when the tail is fitted).
#' @param decay Share of each period's development kept in the next, from 0
#'   to 1.
#' @param attachment_age Age in months the factor attaches at (the first age
#'   at or after it), or `NULL` for the oldest age. An age at or before the
#'   youngest replaces every estimated factor (chainladder-python ignores
#'   such an attachment).
#' @returns A `tail_constant` object.
#' @seealso [tail_curve()], [tail_bondy()], [tail_log_linear()].
#' @export
#' @examples
#' long <- data.frame(year = rep(2018:2021, 4:1),
#'                    age = c(12, 24, 36, 48, 12, 24, 36, 12, 24, 12),
#'                    paid = c(100, 150, 165, 170, 110, 170, 180, 120, 175, 130))
#' tri <- triangle(long, "year", "age", "paid")
#' fit <- chain_ladder(tri, tail = tail_constant(1.05, attachment_age = 36))
#' fit@ldf
#' fit@tail
#' fit@tail_ldf
tail_constant <- S7::new_class(
  "tail_constant",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("ReservingTail"),
    factor = S7::new_property(S7::class_double, getter = function(self) self@ptr$params()$factor),
    decay = S7::new_property(S7::class_double, getter = function(self) self@ptr$params()$decay),
    attachment_age = tail_age("attachment_age")
  ),
  constructor = function(factor = 1, decay = 0.5, attachment_age = NULL) {
    ptr <- rust_result(ReservingTail$constant(
      single_number(factor, "factor"), single_number(decay, "decay"),
      optional_number(attachment_age, "attachment_age")
    ))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Curve-fitted tail
#'
#' A curve fitted to the estimated factors and extrapolated, as
#' chainladder-python's `TailCurve`. Factors above 1.00001 in the fit
#' period are regressed by least squares: `log(f - 1)` on the 1-based
#' development index `k` (`"exponential"`) or on `log(k)`
#' (`"inverse_power"`, a heavier tail). The fitted curve replaces the
#' factors from the attachment age on and runs `extrap_periods` periods past
#' the oldest age.
#'
#' Properties: `curve`, `fit_period` (two ages, `NA` for an open end),
#' `extrap_periods` and `attachment_age` (`NULL` for the oldest age).
#'
#' @param curve `"exponential"` or `"inverse_power"`.
#' @param fit_period The ages in months whose factors enter the fit, from
#'   the last age at or before the first (inclusive) to the last age at or
#'   before the second (exclusive), as chainladder-python reads them; `NA`
#'   for an open end, `NULL` fits every factor.
#' @param extrap_periods Number of periods past the oldest age the curve is
#'   extrapolated; a whole number.
#' @param attachment_age Age in months the curve attaches at (the first age
#'   at or after it), or `NULL` for the oldest age.
#' @returns A `tail_curve` object.
#' @seealso [tail_constant()], [tail_bondy()], [tail_log_linear()].
#' @export
#' @examples
#' long <- data.frame(year = rep(2018:2021, 4:1),
#'                    age = c(12, 24, 36, 48, 12, 24, 36, 12, 24, 12),
#'                    paid = c(100, 150, 165, 170, 110, 170, 180, 120, 175, 130))
#' tri <- triangle(long, "year", "age", "paid")
#' chain_ladder(tri, tail = tail_curve())@tail
#' chain_ladder(tri, tail = tail_curve("inverse_power", fit_period = c(12, NA)))@tail
tail_curve <- S7::new_class(
  "tail_curve",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("ReservingTail"),
    curve = S7::new_property(S7::class_character, getter = function(self) self@ptr$params()$curve),
    fit_period = S7::new_property(S7::class_integer, getter = function(self) {
      p <- self@ptr$params()
      c(if (is.null(p$fit_from)) NA_integer_ else p$fit_from,
        if (is.null(p$fit_to)) NA_integer_ else p$fit_to)
    }),
    extrap_periods = S7::new_property(S7::class_integer, getter = function(self) {
      as.integer(self@ptr$params()$extrap_periods)
    }),
    attachment_age = tail_age("attachment_age")
  ),
  constructor = function(curve = c("exponential", "inverse_power"), fit_period = NULL,
                         extrap_periods = 100, attachment_age = NULL) {
    curve <- match.arg(curve)
    if (is.null(fit_period)) fit_period <- c(NA, NA)
    if (length(fit_period) != 2 || !(is.numeric(fit_period) || all(is.na(fit_period)))) {
      stop("fit_period must be two ages in months, NA for an open end", call. = FALSE)
    }
    ptr <- rust_result(ReservingTail$curve(
      curve, optional_number(fit_period[1], "fit_period"), optional_number(fit_period[2], "fit_period"),
      single_number(extrap_periods, "extrap_periods"), optional_number(attachment_age, "attachment_age")
    ))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Bondy tail
#'
#' The (generalized) Bondy tail, as chainladder-python's `TailBondy`. Each
#' log factor from `earliest_age` on is taken as `b` times the one before
#' it, `b` fitted by least squares. The fitted factors are `f0^(b^j)` from
#' the factor `f0` at `earliest_age`, and those past the next one multiply
#' to the last fitted factor raised to `b / (1 - b)`. With the default
#' `earliest_age` (the age of the last factor) `b` is 1/2 and the tail
#' repeats the last factor: the classic Bondy method.
#'
#' The exponent is the exact least-squares optimum. chainladder-python's
#' optimizer stops early, so its generalized Bondy factors differ in about
#' the fifth significant digit.
#'
#' Properties: `earliest_age` and `attachment_age` (`NULL` for the age of
#' the last factor).
#'
#' @param earliest_age First age in months whose factor enters the fit (the
#'   last age at or before it, as chainladder-python reads it), or `NULL`
#'   for the age of the last factor.
#' @param attachment_age The factor from this age (the last age at or before
#'   it) to the next is kept and the fitted ones replace those after it;
#'   `NULL` is the age of the last factor. Not before `earliest_age`.
#' @returns A `tail_bondy` object.
#' @seealso [tail_constant()], [tail_curve()], [tail_log_linear()].
#' @export
#' @examples
#' long <- data.frame(year = rep(2018:2021, 4:1),
#'                    age = c(12, 24, 36, 48, 12, 24, 36, 12, 24, 12),
#'                    paid = c(100, 150, 165, 170, 110, 170, 180, 120, 175, 130))
#' tri <- triangle(long, "year", "age", "paid")
#' fit <- chain_ladder(tri, tail = tail_bondy())
#' fit@tail
#' chain_ladder(tri, tail = tail_bondy(earliest_age = 12))@tail
tail_bondy <- S7::new_class(
  "tail_bondy",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("ReservingTail"),
    earliest_age = tail_age("earliest_age"),
    attachment_age = tail_age("attachment_age")
  ),
  constructor = function(earliest_age = NULL, attachment_age = NULL) {
    ptr <- rust_result(ReservingTail$bondy(
      optional_number(earliest_age, "earliest_age"), optional_number(attachment_age, "attachment_age")
    ))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' R ChainLadder's log-linear tail
#'
#' R ChainLadder's `MackChainLadder(tail = TRUE)` rule, its `tailfactor()`
#' function, with its quirks: when the third- and second-last factors
#' multiply to more than 1.0001, `log(f - 1)` is regressed on the
#' development index over the factors above 1 and the next 100
#' extrapolated factors are multiplied; otherwise the tail is 1. A tail
#' above 2 is reset to 1. The tail is a single factor to ultimate.
#'
#' @returns A `tail_log_linear` object.
#' @seealso [tail_constant()], [tail_curve()], [tail_bondy()].
#' @export
#' @examples
#' long <- data.frame(year = rep(2018:2021, 4:1),
#'                    age = c(12, 24, 36, 48, 12, 24, 36, 12, 24, 12),
#'                    paid = c(100, 150, 165, 170, 110, 170, 180, 120, 175, 130))
#' m <- mack(triangle(long, "year", "age", "paid"), tail = tail_log_linear())
#' m@tail
#' m@tail_sigma
#' m@standard_error
tail_log_linear <- S7::new_class(
  "tail_log_linear",
  package = "prospicio",
  properties = list(ptr = S7::new_S3_class("ReservingTail")),
  constructor = function() S7::new_object(S7::S7_object(), ptr = ReservingTail$log_linear())
)

tail_classes <- function() list(tail_constant, tail_curve, tail_bondy, tail_log_linear)

# The ReservingTail pointer of a `tail` argument: a number (a constant
# factor) or a tail estimator.
tail_ptr <- function(tail) {
  for (cls in tail_classes()) if (S7::S7_inherits(tail, cls)) return(tail@ptr)
  if (!is.numeric(tail) || length(tail) != 1 || is.na(tail)) {
    stop("tail must be a single number, tail_constant(), tail_curve(), tail_bondy() or ",
         "tail_log_linear()", call. = FALSE)
  }
  tail_constant(tail)@ptr
}

tail_print <- function(x, ...) {
  props <- S7::props(x)
  props$ptr <- NULL
  args <- vapply(names(props), function(n) paste(n, "=", deparse(props[[n]], control = NULL)), "")
  cat(sprintf("%s(%s)\n", S7::S7_class(x)@name, paste(args, collapse = ", ")))
  invisible(x)
}
S7::method(print, tail_constant) <- tail_print
S7::method(print, tail_curve) <- tail_print
S7::method(print, tail_bondy) <- tail_print
S7::method(print, tail_log_linear) <- tail_print

#' Chain ladder
#'
#' Projects each origin's latest cumulative value to ultimate with
#' age-to-age factors estimated from the triangle and a tail factor. The
#' factors are Mack's weighted regressions: `average = "volume"` is the
#' volume-weighted chain-ladder factor (Mack's `alpha = 1`), `"simple"` the
#' mean of the link ratios (`alpha = 0`) and `"regression"` least squares
#' through the origin (`alpha = 2`). Results match R ChainLadder's
#' `MackChainLadder()` and chainladder-python
#' (`validation/reference/reserving_chainladder_r.csv`).
#'
#' Every segment of the triangle is fitted on its own. Per-origin
#' properties run over the origins of each segment in turn, like the rows
#' of `as.data.frame()`, and are named by origin, or by `"segment /
#' origin"` (key values joined by `" / "`) when there are several segments;
#' a single-segment fit has one value per origin.
#'
#' Development past the oldest age is a tail: a given factor, or one
#' estimated by [tail_constant()], [tail_curve()], [tail_bondy()] or
#' [tail_log_linear()] (`validation/reference/reserving_tails_python.csv`
#' and `reserving_tails_r.csv`). A tail attached before the oldest age
#' replaces the estimated factors from there.
#'
#' A selected development pattern (`ldf`, with `ldf_kind` and
#' `ldf_weight`) replaces, or blends with, the factors estimated from the
#' triangle at the ages it gives; the other ages stay estimated, so a
#' benchmark for the immature ages only is a partial pattern. Its values
#' are age-to-age factors (any ages but the oldest: development past it is
#' the tail's), or factors to ultimate or percents developed at two or more
#' ages, which select the factors between the first and the last (`cdf[a] /
#' cdf[next age]`). A development age they skip is interpolated: between
#' two given ages `b < c`, `log(cdf)` is linear in the age, so the factor
#' from `a` to the next age `a'` is `(cdf[b] / cdf[c])^((a' - a) / (c - b))`
#' (percents developed are converted to factors to ultimate first).
#' A cumulative pattern's level at its last age selects nothing. With no
#' `tail` (1) and a last age that is the oldest, the level is the tail: a
#' constant tail factor equal to the factor to ultimate there (1 over the
#' percent developed), with no parameter error, as chainladder-python's
#' `DevelopmentConstant` keeps it; `pattern_tail` reports it. Otherwise
#' the level is rescaled to the factors and tail in use, and
#' `pattern_rescale` reports the factor that multiplies the given percents
#' developed (and divides the given factors to ultimate) to reconcile the
#' level at the last age, so that a pattern given to the oldest age with a
#' tail, rescaled, times the tail is 100%.
#' At earlier ages the rescaled values are the projection's only where the
#' links are fully selected; with a blend, read `cdf` for the percents in
#' use. With a weight `w`, the factor used is
#' `w * selected + (1 - w) * estimated`. A selected factor at or after a
#' tail's attachment age is an error. In [mack()] and the bootstraps a
#' selected factor carries no parameter error and a blended one `1 - w`
#' times the estimate's, while the process error stays the triangle's: a
#' supplied pattern lowers the reported uncertainty, and a wrong benchmark
#' is not reflected in it. Results match chainladder-python's
#' `DevelopmentConstant` (`validation/reference/reserving_patterns_python.csv`).
#'
#' Properties of the fit, per origin: `latest`, `ultimate`, `reserve`; per
#' link (named `"12-24"` and so on): `ldf` (the selected factors, which the
#' projection uses: the estimated ones or a selected pattern's, replaced by
#' the tail's from its attachment age), `estimated_ldf` (the factors as
#' estimated, before a pattern or the tail replaced any), `sigma` (with
#' unestimable ones interpolated), `std_err` (of the factors used, `1 -
#' ldf_weight` times `estimated_std_err`), `estimated_std_err`,
#' `selected_ldf` (the pattern's age-to-age factors, `NA` where it selects
#' none), `ldf_weight` (0 where none is selected) and `blended_ldf` (the
#' factors before the tail); `pattern_rescale` (`NULL` for an age-to-age
#' pattern or none) and `pattern_tail` (the tail a cumulative pattern to
#' the oldest age implies with no `tail`; `NULL` otherwise); `n_links` (how many link ratios each factor averaged, after
#' the exclusions) and `excluded` (whether `n_periods`, `drop_high` or
#' `drop_low` left out a link ratio at some age); per age: `cdf` (age to ultimate, including the tail); the
#' tail: `tail` (the factor from the oldest age to ultimate),
#' `tail_attachment_age` (the age from which `ldf` holds the tail's
#' factors; the oldest age when it replaced none), `tail_ldf` (the factors
#' past the oldest age, which multiply to `tail`: one per development
#' period of the next year and one to ultimate, or a single one for
#' [tail_log_linear()]), `tail_sigma` and `tail_std_err` (the tail's
#' variance parameter and standard error, extrapolated log-linearly; 0
#' without a tail, a factor of 1; a tail below 1 is read where a tail of
#' 1.001 would be, as chainladder-python does); and `keys`, `index` (one
#' row per segment, as a triangle's), `origins`, `development`, `alpha` (`NaN`
#' for the geometric average), `total_ultimate` and
#' `total_reserve` (summed over segments). The
#' per-link, per-age and tail properties need a single-segment fit: with
#' several segments use [development_frame()] (`ldf`, `cdf`, `sigma` and
#' `std_err`; the oldest age has the tail as its `cdf`), [totals_frame()]
#' (`tail`, `tail_sigma` and `tail_std_err`) or [segment()].
#' `as.data.frame()` gives one row per segment and origin, [totals_frame()]
#' one per segment.
#'
#' @param triangle A [triangle], with any number of segments.
#' @param column Name of the column to fit; by default the only one.
#' @param average `"volume"`, `"simple"`, `"regression"` or `"geometric"`.
#'   `"geometric"` is the geometric mean of the link ratios: it gives the
#'   factors only, with no sigma or standard error, so [mack()] and the
#'   bootstraps reject it.
#' @param sigma_interpolation How a variance parameter that cannot be
#'   estimated (an age with a single link ratio) is filled in:
#'   `"log-linear"` (regress `log(sigma)` on age and extrapolate, the
#'   default of R ChainLadder) or `"mack"` (Mack 1993).
#' @param tail Development past the oldest age: a number, the factor from
#'   the oldest age to ultimate (1 means no tail), or a [tail_constant()],
#'   [tail_curve()], [tail_bondy()] or [tail_log_linear()].
#' @param ldf A selected development pattern (above): a numeric vector
#'   named by age in months, or a data.frame with columns `age` and
#'   `value`; each age a development age of the triangle. `NULL` (the
#'   default) estimates every factor.
#' @param ldf_kind How `ldf` is expressed: `"ldf"` (age-to-age factors),
#'   `"cdf"` (factors to ultimate)
#'   or `"percent_developed"`, each finite and positive and in any order.
#'   Required with `ldf`, since age-to-age factors and factors to ultimate
#'   cannot be told apart.
#' @param ldf_weight The weight `w` on each selected factor, from 0 (the
#'   estimate) to 1 (the selection, the default): a number for every
#'   selected factor, or a numeric vector named by age (the age that starts
#'   the factor's link), the others fully selected.
#' @param n_periods,drop_high,drop_low Which link ratios the factor at each
#'   age averages. `n_periods` keeps the latest `n_periods` calendar
#'   diagonals (`NULL`, the default, keeps every one; an age with fewer
#'   links uses all of them). Then the `drop_high` highest and `drop_low`
#'   lowest of the kept ratios are left out, except at an age where that
#'   would leave none. Whole numbers from 0 (`n_periods` from 1); the
#'   defaults exclude nothing. [mack()], the bootstraps and the one-year
#'   views accept them (the bootstraps and views keep the link ratios that
#'   the observed triangle keeps), and [claims_development_result()]
#'   rejects any of them, even one that removes nothing. The fit's
#'   `n_links` and `excluded` report what was used.
#' @param ptr A `ChainLadderFit` pointer; used internally.
#' @returns A `chain_ladder_fit` object.
#' @seealso [mack()] for standard errors, [segment()] for one segment.
#' @export
#' @examples
#' long <- data.frame(year = c(2020, 2020, 2020, 2021, 2021, 2022),
#'                    age = c(12, 24, 36, 12, 24, 12),
#'                    paid = c(100, 150, 165, 110, 170, 120))
#' fit <- chain_ladder(triangle(long, "year", "age", "paid"), tail = 1.05)
#' fit@ldf
#' fit@reserve
#' fit@total_reserve
#' chain_ladder(triangle(long, "year", "age", "paid"), tail = tail_bondy())@tail
#'
#' # Every line of business at once.
#' by_lob <- rbind(transform(long, lob = "auto"), transform(long, lob = "home", paid = paid / 2))
#' fits <- chain_ladder(triangle(by_lob, "year", "age", "paid", keys = "lob"))
#' fits@reserve
#' as.data.frame(fits)
#' totals_frame(fits)
#'
#' # A benchmark for the immature ages only: 50% developed at 12 months and
#' # 80% at 24 select the factor 1.6 from 12 to 24; 24 to 36 stays estimated.
#' bench <- chain_ladder(triangle(long, "year", "age", "paid"),
#'                       ldf = c("12" = 0.5, "24" = 0.8), ldf_kind = "percent_developed")
#' bench@ldf
#' bench@estimated_ldf
#' bench@pattern_rescale
#' # Half credibility on the benchmark.
#' chain_ladder(triangle(long, "year", "age", "paid"), ldf = c("12" = 0.5, "24" = 0.8),
#'              ldf_kind = "percent_developed", ldf_weight = 0.5)@ldf
chain_ladder_fit <- S7::new_class(
  "chain_ladder_fit",
  package = "prospicio",
  properties = c(
    list(ptr = S7::new_S3_class("ChainLadderFit")),
    chain_ladder_properties(function(self) self@ptr)
  ),
  constructor = function(ptr) S7::new_object(S7::S7_object(), ptr = ptr)
)

#' @rdname chain_ladder_fit
#' @export
chain_ladder <- function(triangle, column = NULL, average = "volume",
                         sigma_interpolation = "log-linear", tail = 1, ldf = NULL,
                         ldf_kind = NULL, ldf_weight = 1, n_periods = NULL, drop_high = 0,
                         drop_low = 0) {
  column <- fit_column(triangle, column)
  args <- development_args(average, sigma_interpolation, n_periods, drop_high, drop_low)
  ptr <- rust_result(triangle@ptr$chain_ladder(column, args$average, args$sigma, args$excl, tail_ptr(tail),
                                               ldf_pattern(ldf, ldf_kind, ldf_weight)))
  chain_ladder_fit(ptr = ptr)
}

#' Mack chain ladder
#'
#' Mack's distribution-free chain ladder (Mack 1993, 1999): the chain-ladder
#' projection plus the process and parameter standard errors of each
#' origin's reserve and of the total, as R ChainLadder's
#' `MackChainLadder()`. Needs at least three development ages, and every
#' sigma estimable or fillable by `sigma_interpolation`. Every segment of
#' the triangle is fitted on its own.
#'
#' A tail other than 1 is one more development step, from the oldest age to
#' ultimate, with its own sigma and standard error, as
#' `MackChainLadder(tail = ..., tail.sigma = ..., tail.se = ...)`; unless
#' given, both are extrapolated log-linearly (R's `tail_SE`). Every origin,
#' the oldest included, then carries the tail's risk. Without a tail (the
#' default) the fit is R's `MackChainLadder(tail = FALSE)`. A tail below 1
#' follows chainladder-python: it scales the ultimates and carries the risk
#' read where a tail of 1.001 would be. R's `MackChainLadder()` ignores a
#' tail below 1 altogether.
#'
#' The fit has the properties of a [chain_ladder_fit] (and the fit itself
#' as `m@chain_ladder`), plus per origin `process_risk`, `parameter_risk`
#' and `standard_error`, and `total_process_risk`, `total_parameter_risk`,
#' `total_standard_error` and `total_cv` (the total standard error over the
#' total reserve). Risks are standard errors, not variances. Its
#' `tail_sigma` and `tail_std_err` are the values used, given or
#' extrapolated. The totals' risks need a single-segment fit: with several
#' segments use [totals_frame()], which has them per segment, or
#' [segment()]. [claims_development_result()] gives Merz and Wuthrich's
#' one-year view of a single-segment fit without a tail.
#'
#' With a selected pattern (`ldf`), a selected factor is taken as known: its
#' parameter risk is `1 - w` times Mack's (none for a fully selected
#' factor), checked against R's `MackChainLadder()` on the triangle's own
#' factors (`validation/tests/reserving_patterns.rs`); the process risk
#' keeps the triangle's sigmas, around the factors used. A supplied pattern
#' therefore lowers the reported standard errors. [claims_development_result()]
#' takes the pattern too: with weights strictly between 0 and 1 its calendar
#' years need not add up to Mack's. A tail implied by a cumulative pattern
#' with no `tail` ([chain_ladder()]'s `pattern_tail`) has no parameter
#' error (`tail_std_err` 0 unless given), and its sigma is extrapolated as
#' a constant tail's unless `tail_sigma` is given.
#'
#' @inheritParams chain_ladder_fit
#' @param tail_sigma The tail's sigma (R's `tail.sigma`), or `NULL` to
#'   extrapolate it. Unused when the tail factor is 1.
#' @param tail_std_err The tail factor's standard error (R's `tail.se`), or
#'   `NULL` to extrapolate it. Unused when the tail factor is 1.
#' @returns A `mack_fit` object.
#' @export
#' @examples
#' long <- data.frame(year = rep(2018:2021, 4:1),
#'                    age = c(12, 24, 36, 48, 12, 24, 36, 12, 24, 12),
#'                    paid = c(100, 150, 165, 170, 110, 170, 180, 120, 175, 130))
#' m <- mack(triangle(long, "year", "age", "paid"))
#' m@standard_error
#' m@total_cv
#' as.data.frame(m)
#'
#' # With R ChainLadder's tail = TRUE rule.
#' mack(triangle(long, "year", "age", "paid"), tail = tail_log_linear())@total_standard_error
mack_fit <- S7::new_class(
  "mack_fit",
  package = "prospicio",
  properties = c(
    list(ptr = S7::new_S3_class("MackFit"), chain_ladder = chain_ladder_fit),
    chain_ladder_properties(function(self) self@chain_ladder@ptr),
    list(
      process_risk = S7::new_property(S7::class_double, getter = function(self) {
        stats::setNames(self@ptr$process_risk(), origin_names(self@chain_ladder@ptr))
      }),
      parameter_risk = S7::new_property(S7::class_double, getter = function(self) {
        stats::setNames(self@ptr$parameter_risk(), origin_names(self@chain_ladder@ptr))
      }),
      standard_error = S7::new_property(S7::class_double, getter = function(self) {
        stats::setNames(self@ptr$standard_error(), origin_names(self@chain_ladder@ptr))
      }),
      total_process_risk = S7::new_property(S7::class_double, getter = function(self) {
        rust_result(self@ptr$total_process_risk(), call = NULL)
      }),
      total_parameter_risk = S7::new_property(S7::class_double, getter = function(self) {
        rust_result(self@ptr$total_parameter_risk(), call = NULL)
      }),
      total_standard_error = S7::new_property(S7::class_double, getter = function(self) {
        rust_result(self@ptr$total_standard_error(), call = NULL)
      }),
      total_cv = S7::new_property(S7::class_double, getter = function(self) {
        rust_result(self@ptr$total_cv(), call = NULL)
      })
    )
  ),
  constructor = function(ptr) {
    S7::new_object(S7::S7_object(), ptr = ptr, chain_ladder = chain_ladder_fit(ptr = ptr$chain_ladder()))
  }
)

#' @rdname mack_fit
#' @export
mack <- function(triangle, column = NULL, average = "volume", sigma_interpolation = "log-linear",
                 tail = 1, tail_sigma = NULL, tail_std_err = NULL, ldf = NULL, ldf_kind = NULL,
                 ldf_weight = 1, n_periods = NULL, drop_high = 0, drop_low = 0) {
  column <- fit_column(triangle, column)
  args <- development_args(average, sigma_interpolation, n_periods, drop_high, drop_low)
  ptr <- rust_result(triangle@ptr$mack(
    column, args$average, args$sigma, args$excl, tail_ptr(tail),
    optional_number(tail_sigma, "tail_sigma"), optional_number(tail_std_err, "tail_std_err"),
    ldf_pattern(ldf, ldf_kind, ldf_weight)
  ))
  mack_fit(ptr = ptr)
}

# Checks that `value` is a single number and returns it as a double.
single_number <- function(value, arg) {
  if (!is.numeric(value) || length(value) != 1 || is.na(value)) {
    stop(sprintf("%s must be a single number", arg), call. = FALSE)
  }
  as.double(value)
}

# The loss and exposure columns of an expected-loss method.
exposure_columns <- function(triangle, column, exposure) {
  check_triangle(triangle)
  for (arg in c("column", "exposure")) {
    value <- get(arg)
    if (!is.character(value) || length(value) != 1 || is.na(value)) {
      stop(sprintf("%s must be the name of a column", arg), call. = FALSE)
    }
  }
  c(column, exposure)
}

# Properties of an expected-loss fit, read through `el(self)`, an
# ExpectedLossFit pointer, and `cl(self)`, the ChainLadderFit pointer of its
# development pattern. Shared by expected_loss_fit and cape_cod_fit.
expected_loss_properties <- function(el, cl) {
  by_origin <- function(f) {
    S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(f(el(self)), origin_names(cl(self)))
    })
  }
  c(
    chain_ladder_properties(cl)[c("keys", "index", "origins", "development", "ldf", "cdf",
                                  "latest")],
    list(
      exposure = by_origin(function(p) p$exposure()),
      apriori = by_origin(function(p) p$apriori()),
      ultimate = by_origin(function(p) p$ultimate()),
      reserve = by_origin(function(p) p$reserve()),
      total_ultimate = S7::new_property(S7::class_double, getter = function(self) {
        el(self)$total_ultimate()
      }),
      total_reserve = S7::new_property(S7::class_double, getter = function(self) {
        el(self)$total_reserve()
      })
    )
  )
}

#' Expected loss, Bornhuetter-Ferguson and Benktander
#'
#' Methods that credit each origin's latest value and an expected ultimate,
#' `apriori` times the origin's exposure, by how developed the origin is.
#' With `q = 1 / cdf` the share of the ultimate developed at the origin's
#' latest age:
#'
#' * `expected_loss()`: the ultimate is `apriori * exposure`, whatever has
#'   been observed;
#' * `bornhuetter_ferguson()`: the ultimate is
#'   `latest + (1 - q) * apriori * exposure`;
#' * `benktander()`: starting from `U(0) = apriori * exposure`,
#'   `U(k) = latest + (1 - q) * U(k - 1)` for `n_iters` steps, so
#'   `n_iters = 0` is the expected loss method, 1 is Bornhuetter-Ferguson,
#'   and many iterations approach the chain ladder. The steps are summed in
#'   closed form, so a large `n_iters` is cheap; where an origin's `cdf` is
#'   below 1/2 they diverge instead.
#'
#' The development pattern is a [chain_ladder()] fit of the loss column,
#' with the same `average`, `sigma_interpolation` and `tail`. The exposure
#' (premium, say) is another column of the same triangle: each origin's
#' latest observed cumulative value in the segment fitted, which must be
#' finite and positive. Results match chainladder-python's `ExpectedLoss`,
#' `BornhuetterFerguson` and `Benktander` with
#' `sample_weight = premium.latest_diagonal`
#' (`validation/reference/reserving_expected_loss_python.csv`).
#'
#' Every segment of the triangle is fitted on its own, with its own
#' exposure. Per-origin properties run over the origins of each segment in
#' turn and are named as a [chain_ladder_fit]'s.
#'
#' Properties of the fit, per origin: `latest`, `exposure`, `apriori` (the
#' expected loss ratio), `ultimate` and `reserve` (`ultimate - latest`; the
#' method's own, not the chain ladder's); `ldf` and `cdf` of the
#' development pattern, which need a single-segment fit; `chain_ladder`
#' (the [chain_ladder_fit] of the pattern), `keys`, `index`, `origins`,
#' `development`, `total_ultimate` and `total_reserve` (summed over
#' segments). `as.data.frame()` gives one row per segment and origin with
#' `exposure` and `apriori` after the reserve, [totals_frame()] one per
#' segment with the total `exposure`.
#'
#' Errors: an unknown column, an origin without an observed, finite,
#' positive exposure (naming the origin and, with keys, the segment), an
#' `apriori` that is not finite and positive, or a negative or fractional
#' `n_iters`.
#'
#' These are Python's `ExpectedLoss`, `BornhuetterFerguson` and
#' `Benktander`, whose `fit()` returns an `ExpectedLossFit`.
#'
#' @param triangle A [triangle], with any number of segments.
#' @param column Name of the loss column to project.
#' @param exposure Name of the exposure column; each origin's latest
#'   observed value is its exposure.
#' @param apriori Expected loss ratio: the expected ultimate per unit of
#'   exposure; positive.
#' @param n_iters Number of Bornhuetter-Ferguson steps; a non-negative whole
#'   number.
#' @param ldf,ldf_kind,ldf_weight A selected development pattern, as in
#'   [chain_ladder()]: the development pattern's factors at the ages it
#'   gives, so `cdf` and the share developed `1 / cdf` follow it.
#' @param average,sigma_interpolation,tail The development pattern, as in
#'   [chain_ladder()].
#' @param n_periods,drop_high,drop_low Which link ratios the factors average,
#'   as in [chain_ladder()].
#' @param ptr An `ExpectedLossFit` pointer; used internally.
#' @returns An `expected_loss_fit` object.
#' @seealso [cape_cod()] to estimate the apriori from the triangle,
#'   [segment()] for one segment.
#' @export
#' @examples
#' long <- data.frame(year = c(2020, 2020, 2021), age = c(12, 24, 12),
#'                    paid = c(100, 150, 200), premium = c(250, 250, 400))
#' tri <- triangle(long, "year", "age", c("paid", "premium"))
#' expected_loss(tri, "paid", "premium", apriori = 0.5)@ultimate
#'
#' # The 2021 origin is a third developed (cdf 1.5): 200 + (1/3) * 0.5 * 400.
#' bf <- bornhuetter_ferguson(tri, "paid", "premium", apriori = 0.5)
#' bf@ultimate
#' bf@cdf
#' as.data.frame(bf)
#'
#' benktander(tri, "paid", "premium", apriori = 0.5, n_iters = 2)@ultimate
expected_loss_fit <- S7::new_class(
  "expected_loss_fit",
  package = "prospicio",
  properties = c(
    list(ptr = S7::new_S3_class("ExpectedLossFit"), chain_ladder = chain_ladder_fit),
    expected_loss_properties(function(self) self@ptr, function(self) self@chain_ladder@ptr)
  ),
  constructor = function(ptr) {
    S7::new_object(S7::S7_object(), ptr = ptr, chain_ladder = chain_ladder_fit(ptr = ptr$chain_ladder()))
  }
)

#' @rdname expected_loss_fit
#' @export
expected_loss <- function(triangle, column, exposure, apriori = 1, average = "volume",
                          sigma_interpolation = "log-linear", tail = 1, ldf = NULL,
                          ldf_kind = NULL, ldf_weight = 1, n_periods = NULL, drop_high = 0,
                          drop_low = 0) {
  cols <- exposure_columns(triangle, column, exposure)
  args <- development_args(average, sigma_interpolation, n_periods, drop_high, drop_low)
  ptr <- rust_result(triangle@ptr$expected_loss(
    cols[1], cols[2], single_number(apriori, "apriori"), args$average, args$sigma, args$excl,
    tail_ptr(tail), ldf_pattern(ldf, ldf_kind, ldf_weight)
  ))
  expected_loss_fit(ptr = ptr)
}

#' @rdname expected_loss_fit
#' @export
bornhuetter_ferguson <- function(triangle, column, exposure, apriori = 1, average = "volume",
                                 sigma_interpolation = "log-linear", tail = 1, ldf = NULL,
                                 ldf_kind = NULL, ldf_weight = 1, n_periods = NULL, drop_high = 0,
                                 drop_low = 0) {
  cols <- exposure_columns(triangle, column, exposure)
  args <- development_args(average, sigma_interpolation, n_periods, drop_high, drop_low)
  ptr <- rust_result(triangle@ptr$bornhuetter_ferguson(
    cols[1], cols[2], single_number(apriori, "apriori"), args$average, args$sigma, args$excl,
    tail_ptr(tail), ldf_pattern(ldf, ldf_kind, ldf_weight)
  ))
  expected_loss_fit(ptr = ptr)
}

#' @rdname expected_loss_fit
#' @export
benktander <- function(triangle, column, exposure, apriori = 1, n_iters = 1, average = "volume",
                       sigma_interpolation = "log-linear", tail = 1, ldf = NULL,
                       ldf_kind = NULL, ldf_weight = 1, n_periods = NULL, drop_high = 0,
                       drop_low = 0) {
  cols <- exposure_columns(triangle, column, exposure)
  args <- development_args(average, sigma_interpolation, n_periods, drop_high, drop_low)
  ptr <- rust_result(triangle@ptr$benktander(
    cols[1], cols[2], single_number(apriori, "apriori"), single_number(n_iters, "n_iters"),
    args$average, args$sigma, args$excl, tail_ptr(tail), ldf_pattern(ldf, ldf_kind, ldf_weight)
  ))
  expected_loss_fit(ptr = ptr)
}

#' Cape Cod
#'
#' The Cape Cod (Stanard-Buhlmann) method: Bornhuetter-Ferguson with each
#' origin's apriori estimated from the triangle itself, as
#' chainladder-python's `CapeCod`
#' (`validation/reference/reserving_expected_loss_python.csv`).
#'
#' Origin `j`'s used-up exposure is `exposure[j] / cdf[j]`. Its latest value
#' is trended to the triangle's valuation by `(1 + trend)^(m[j] / 12)`,
#' `m[j]` the months from the end of the origin period to the valuation.
#' Origin `i`'s trended apriori is the sum over `j` of the trended latest
#' values weighted by `decay^|i - j|`, over the same weighted sum of used-up
#' exposures; dividing by `i`'s own trend factor gives the apriori its
#' Bornhuetter-Ferguson ultimate uses. With `decay = 1` every origin shares
#' one loss ratio; with `decay = 0` each origin keeps its own and the method
#' returns the chain ladder. The exposure and development pattern are as
#' for [bornhuetter_ferguson()]. Every segment of the triangle is fitted on
#' its own, with its own exposure and apriori, trended to the whole
#' triangle's valuation.
#'
#' The fit has the properties of an [expected_loss_fit], with `apriori` the
#' detrended apriori (chainladder-python's `detrended_apriori_`), and the
#' Bornhuetter-Ferguson fit itself as `expected_loss`; plus per origin
#' `trended_apriori` (chainladder-python's `apriori_`).
#' `trended_apriori` is at the valuation date's price level, so a treaty year
#' that incepts later needs more trend beyond it (one more year for the next
#' year).
#' `as.data.frame()` adds a `trended_apriori` column.
#'
#' Errors: as [bornhuetter_ferguson()], or a `trend` that is not finite and
#' above -1, or a `decay` outside 0 to 1.
#'
#' This is Python's `CapeCod`, whose `fit()` returns a `CapeCodFit`.
#'
#' @inheritParams expected_loss_fit
#' @param trend Annual trend rate of the losses, above -1.
#' @param decay Weight `decay^|i - j|` of origin `j` in origin `i`'s
#'   apriori, between 0 and 1.
#' @param ptr A `CapeCodFit` pointer; used internally.
#' @returns A `cape_cod_fit` object.
#' @seealso [segment()] for one segment.
#' @export
#' @examples
#' long <- data.frame(year = c(2020, 2020, 2021), age = c(12, 24, 12),
#'                    paid = c(100, 150, 200), premium = c(250, 250, 400))
#' tri <- triangle(long, "year", "age", c("paid", "premium"))
#' # Loss ratio (150 + 200) / (250 + 400 / 1.5) on both origins.
#' cc <- cape_cod(tri, "paid", "premium")
#' cc@apriori
#' cc@ultimate
#' cape_cod(tri, "paid", "premium", trend = 0.05, decay = 0.5)@trended_apriori
cape_cod_fit <- S7::new_class(
  "cape_cod_fit",
  package = "prospicio",
  properties = c(
    list(ptr = S7::new_S3_class("CapeCodFit"), expected_loss = expected_loss_fit,
         chain_ladder = chain_ladder_fit),
    expected_loss_properties(function(self) self@expected_loss@ptr,
                             function(self) self@chain_ladder@ptr),
    list(
      trended_apriori = S7::new_property(S7::class_double, getter = function(self) {
        stats::setNames(self@ptr$trended_apriori(), origin_names(self@chain_ladder@ptr))
      })
    )
  ),
  constructor = function(ptr) {
    el <- expected_loss_fit(ptr = ptr$expected_loss())
    S7::new_object(S7::S7_object(), ptr = ptr, expected_loss = el, chain_ladder = el@chain_ladder)
  }
)

#' @rdname cape_cod_fit
#' @export
cape_cod <- function(triangle, column, exposure, trend = 0, decay = 1, average = "volume",
                     sigma_interpolation = "log-linear", tail = 1, ldf = NULL,
                     ldf_kind = NULL, ldf_weight = 1, n_periods = NULL, drop_high = 0,
                     drop_low = 0) {
  cols <- exposure_columns(triangle, column, exposure)
  args <- development_args(average, sigma_interpolation, n_periods, drop_high, drop_low)
  ptr <- rust_result(triangle@ptr$cape_cod(
    cols[1], cols[2], single_number(trend, "trend"), single_number(decay, "decay"),
    args$average, args$sigma, args$excl, tail_ptr(tail), ldf_pattern(ldf, ldf_kind, ldf_weight)
  ))
  cape_cod_fit(ptr = ptr)
}

#' Claims development result: the one-year view of a Mack fit
#'
#' Merz and Wuthrich's (2008) one-year view of a [mack_fit]: the standard
#' error of the claims development result (CDR), the change in the
#' chain-ladder ultimate over a calendar year, for each origin and in
#' total, in the next calendar year and in every later one. The total
#' includes the covariance between origins. Results match R ChainLadder's
#' `CDR(MackChainLadder(x), dev = "all")`
#' (`validation/reference/reserving_cdr_r.csv`) and the totals of Table 4
#' in the paper.
#'
#' The formulas assume volume-weighted factors (`mack(average =
#' "volume")`, the default), no tail (`mack(tail = 1)`, the default: a
#' factor of exactly 1 that replaces no estimated factor), and the latest
#' values on one calendar diagonal with one new origin per period; anything
#' else is an error (R ChainLadder only warns about other factors). The fit must have a
#' single segment: use [segment()] to pick one.
#'
#' Properties, as standard errors (not mean squared errors):
#' `one_year_standard_error` per origin (R's `CDR(1)S.E.`) and
#' `total_one_year_standard_error`; `by_calendar_year`, an origin x
#' calendar year matrix whose column `k` is R's `CDR(k)S.E.`, zero once an
#' origin is fully developed, with one year per age-to-age factor (R adds a
#' last year past the run-off, always zero); `total_by_calendar_year`;
#' `run_off_standard_error` per origin and `total_run_off_standard_error`,
#' the square root of the yearly mean squared errors summed, which equals
#' Mack's standard error; and `origins`. `as.data.frame()` gives one row
#' per origin: `origin`, `cdr_1`, `cdr_2`, ... (the columns of
#' `by_calendar_year`) and `run_off`, as Python's
#' `MackFit.claims_development_result().to_frame()`.
#'
#' @param fit A single-segment [mack_fit] with volume-weighted factors.
#' @param ptr A `ClaimsDevelopmentResult` pointer; used internally.
#' @returns A `claims_development_result` object.
#' @seealso [mack()]; [odp_one_year()] simulates the one-year view of any
#'   averaging, tail or expected-loss method.
#' @export
#' @examples
#' long <- data.frame(year = rep(2018:2021, 4:1),
#'                    age = c(12, 24, 36, 48, 12, 24, 36, 12, 24, 12),
#'                    paid = c(100, 150, 165, 170, 110, 170, 180, 120, 175, 130))
#' m <- mack(triangle(long, "year", "age", "paid"))
#' cdr <- claims_development_result(m)
#' cdr@one_year_standard_error
#' cdr@total_one_year_standard_error
#' cdr@by_calendar_year
#' # The yearly run-off adds up to Mack's standard error.
#' all.equal(cdr@total_run_off_standard_error, m@total_standard_error)
#' as.data.frame(cdr)
claims_development_result <- S7::new_class(
  "claims_development_result",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("ClaimsDevelopmentResult"),
    origins = S7::new_property(S7::class_character, getter = function(self) self@ptr$origins()),
    one_year_standard_error = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(self@ptr$one_year_standard_error(), self@origins)
    }),
    total_one_year_standard_error = S7::new_property(S7::class_double, getter = function(self) {
      self@ptr$total_one_year_standard_error()
    }),
    by_calendar_year = S7::new_property(S7::class_double, getter = function(self) {
      years <- self@ptr$total_by_calendar_year()
      # Rust gives calendar year x origin, row-major: an origin x year matrix
      # filled by column.
      matrix(self@ptr$by_calendar_year(), nrow = length(self@origins), ncol = length(years),
             dimnames = list(origin = self@origins, calendar_year = as.character(seq_along(years))))
    }),
    total_by_calendar_year = S7::new_property(S7::class_double, getter = function(self) {
      years <- self@ptr$total_by_calendar_year()
      stats::setNames(years, seq_along(years))
    }),
    run_off_standard_error = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(self@ptr$run_off_standard_error(), self@origins)
    }),
    total_run_off_standard_error = S7::new_property(S7::class_double, getter = function(self) {
      self@ptr$total_run_off_standard_error()
    })
  ),
  constructor = function(fit, ptr = NULL) {
    if (is.null(ptr)) {
      if (!S7::S7_inherits(fit, mack_fit)) stop("fit must be a mack_fit", call. = FALSE)
      ptr <- rust_result(fit@ptr$claims_development_result())
    }
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

S7::method(as.data.frame, claims_development_result) <- function(x, ...) {
  years <- x@by_calendar_year
  out <- data.frame(origin = x@origins, stringsAsFactors = FALSE)
  for (k in seq_len(ncol(years))) out[[paste0("cdr_", k)]] <- unname(years[, k])
  out$run_off <- unname(x@run_off_standard_error)
  out
}

S7::method(print, claims_development_result) <- function(x, ...) {
  cat(sprintf("<claims_development_result> total standard error: one year %s, run-off %s\n",
              format(x@total_one_year_standard_error, digits = 10),
              format(x@total_run_off_standard_error, digits = 10)))
  print(as.data.frame(x), row.names = FALSE)
  invisible(x)
}

#' ODP bootstrap
#'
#' Over-dispersed Poisson bootstrap of the chain ladder (England and Verrall
#' 2002), as R ChainLadder's `BootChainLadder()`: adjusted Pearson residuals
#' of the volume-weighted chain ladder are resampled into pseudo triangles,
#' each is re-projected, and process error is added to every future
#' incremental value. The scale and residuals match `BootChainLadder()`
#' exactly, and the reserve distribution within Monte Carlo error
#' (`validation/reference/reserving_bootstrap_r.csv`).
#'
#' Every segment of the triangle is bootstrapped with its own residuals and
#' scale, into one joint distribution of the reserves, the segments
#' depending on each other as `dependence` says: independently by default,
#' synchronized, or rank-correlated to a Spearman matrix. The joint
#' distribution goes on to the portfolio's risk measure and its allocation
#' to the lines ([capital_allocation()]). Simulation `i` uses random stream `i` of
#' `seed` for every segment in turn, so results do not depend on the number
#' of threads.
#'
#' Properties of the fit: `chain_ladder` (the [chain_ladder_fit] the
#' bootstrap is centred on), `origins`, `development`, `fitted` (fitted
#' incremental values) and `residuals` (adjusted Pearson residuals
#' `(x - m) / sqrt(|m|) * sqrt(n / (n - p))`), both origin x development
#' matrices with `NA` where not observed, `scale` (the dispersion `phi`) and
#' `reserves`, a [predictive_distribution] of the reserve with the
#' triangle's keys and `origin` as dimensions and one component per segment
#' and origin, so `aggregate(boot@reserves, keep = "lob")` keeps the
#' dependence between segments. `mean()`, `quantile()`, [VaR()] and
#' [TVaR()] of `reserves` describe the total reserve. Columns of
#' [draw_matrix()] follow `origins`. `fitted`, `residuals` and `scale` need
#' a single-segment fit: with several segments use [segment()], or
#' [totals_frame()] for the scales.
#'
#' @param triangle A cumulative [triangle], with any number of segments,
#'   every origin observed from the first age up to its latest.
#' @param column Name of the column to fit; by default the only one.
#' @param n_sims Number of simulations; positive.
#' @param seed Seed of the simulation streams, a non-negative whole number.
#' @param process Process error on each simulated future incremental value:
#'   `"gamma"` (mean the expected value, variance `scale * |mean|`, R's
#'   `process.distr = "gamma"`) or `"none"` for parameter error only.
#' @param tail Development past the oldest age: a number (a constant
#'   factor), [tail_constant()], [tail_curve()], [tail_bondy()] or
#'   [tail_log_linear()], as in [chain_ladder()]; 1 (no tail, as R's
#'   `BootChainLadder`) by default. An estimated tail (a curve, Bondy or
#'   the log-linear rule) is refitted on each simulation's pseudo factors,
#'   which gives its parameter error; a constant one is fixed unless
#'   `tail_std_err` is given. The step from the oldest age to ultimate is
#'   one more future increment, with mean the pseudo value at the oldest
#'   age times the tail factor less 1 and the same process error.
#'   [odp_one_year()] takes it with `bootstrap_tail = TRUE`. R's
#'   rule ([tail_log_linear()]) refitted on the ODP's pseudo factors is
#'   often exactly 1 (5.7% of simulations on RAA) and on average high
#'   (RAA's oldest origin's mean reserve 78% above the plug-in); a constant
#'   tail with `tail_std_err` has neither.
#' @param tail_std_err Standard error of a constant tail factor: each
#'   simulation draws the factor from the lognormal with the factor as mean
#'   and this standard deviation. `NULL` keeps it fixed, unlike
#'   [mack_bootstrap()], which extrapolates Mack's standard error; unused
#'   when the factor is 1, and an error with an estimated tail.
#' @param dependence How the segments of a multi-segment triangle depend on
#'   each other: `"independent"`, each resampling its own residuals;
#'   `"synchronized"`, every segment resampling the residuals of the same
#'   origins and ages in each simulation (the synchronous bootstrap of
#'   Taylor and McGuire 2007 and Kirschner, Kerley and Isaacs 2008), so the
#'   lines' parameter error takes the correlation of their residuals while
#'   process error stays independent, which needs the same origins, ages and
#'   observed cells in every segment; or `"rank_correlation"`, each segment
#'   bootstrapped independently and the segments' simulations then
#'   reordered as whole rows (Iman-Conover on the segment totals) to the
#'   Spearman matrix `spearman` and put in a random order, so that row `i`
#'   is no longer simulation `i` but any subset of rows is a fair sample.
#' @param spearman With `dependence = "rank_correlation"` only: Spearman's
#'   rho between the segments' totals, a symmetric matrix with a unit
#'   diagonal and one row and column per segment in index order. It is
#'   converted to the normal scores' correlation `2 sin(pi rho / 6)`, which
#'   must be positive definite.
#' @param ldf,ldf_kind,ldf_weight A selected development pattern, as in
#'   [chain_ladder()]. The residuals and the scale come from the triangle's
#'   own volume-weighted factors; each simulation then uses
#'   `w * selected + (1 - w) * pseudo` at the selected ages, so a fully
#'   selected factor is the same in every simulation (no parameter error)
#'   and a blended one moves by `1 - w` times the pseudo factor's
#'   deviation. The origins' level still moves (each simulation projects
#'   from its pseudo latest values), so even a fully selected pattern keeps
#'   part of the spread: with every factor selected and no process error,
#'   about 73% of the unblended standard deviation on RAA. This is the
#'   ODP's parameter error in the origins' levels, which a development
#'   pattern does not fix, and it is kept on purpose; [mack_bootstrap()]
#'   projects from the observed latest values and keeps only the process
#'   error. A supplied pattern lowers the reported uncertainty. With no
#'   `tail`, a cumulative pattern to the oldest age implies a constant tail
#'   (`pattern_tail` of [chain_ladder()]), the same in every simulation
#'   unless `tail_std_err` is given.
#' @param n_periods,drop_high,drop_low Which link ratios the volume-weighted
#'   chain ladder behind [odp_bootstrap()] averages, as in [chain_ladder()]
#'   (the bootstrap has no `average`). The fitted values are back-cast with
#'   the factors under the exclusions; an incremental cell has a residual
#'   only if the link ratio it closes is kept (the first age's cells always
#'   do), and the pseudo factors average the kept links only. The drops
#'   are fixed on the observed triangle: the simulated triangles are not
#'   ranked again. The scale is the sum of the kept squared residuals over
#'   `N - p` (chainladder-python's rule), which is biased low under
#'   exclusions: on RAA with `n_periods = 3` it is about 70% of the plain
#'   scale (694 against 984), and the standard deviations are
#'   correspondingly narrow. See [odp_one_year()] for the one-year view.
#' @param ptr An `OdpBootstrapFit` pointer; used internally.
#' @returns An `odp_bootstrap_fit` object.
#' @seealso [chain_ladder()], [mack()]; [odp_one_year()] for the one-year
#'   view on the same bootstrap.
#' @export
#' @examples
#' long <- data.frame(year = rep(2018:2021, 4:1),
#'                    age = c(12, 24, 36, 48, 12, 24, 36, 12, 24, 12),
#'                    paid = c(100, 150, 165, 170, 110, 170, 180, 120, 175, 130))
#' boot <- odp_bootstrap(triangle(long, "year", "age", "paid"), n_sims = 2000, seed = 42)
#' boot@scale
#' boot@reserves@keys
#' mean(boot@reserves)
#' quantile(boot@reserves, 0.995)
odp_bootstrap_fit <- S7::new_class(
  "odp_bootstrap_fit",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("OdpBootstrapFit"),
    chain_ladder = chain_ladder_fit,
    origins = S7::new_property(S7::class_character, getter = function(self) {
      self@chain_ladder@origins
    }),
    development = S7::new_property(S7::class_integer, getter = function(self) {
      self@chain_ladder@development
    }),
    fitted = S7::new_property(S7::class_double, getter = function(self) {
      origin_matrix(self, rust_result(self@ptr$fitted(), call = NULL))
    }),
    residuals = S7::new_property(S7::class_double, getter = function(self) {
      origin_matrix(self, rust_result(self@ptr$residuals(), call = NULL))
    }),
    scale = S7::new_property(S7::class_double, getter = function(self) {
      rust_result(self@ptr$scale(), call = NULL)
    }),
    reserves = predictive_distribution
  ),
  constructor = function(ptr) {
    S7::new_object(S7::S7_object(), ptr = ptr,
                   chain_ladder = chain_ladder_fit(ptr = ptr$chain_ladder()),
                   reserves = predictive_distribution(ptr = ptr$reserves()))
  }
)

# A row-major origin x development vector as a matrix with dimnames; NaN
# (unobserved) becomes NA.
origin_matrix <- function(fit, flat) {
  m <- matrix(flat, nrow = length(fit@origins), byrow = TRUE,
              dimnames = list(origin = fit@origins, development = as.character(fit@development)))
  m[is.nan(m)] <- NA_real_
  m
}

#' @rdname odp_bootstrap_fit
#' @export
odp_bootstrap <- function(triangle, column = NULL, n_sims = 10000, seed = 0,
                          process = c("gamma", "none"), tail = 1, tail_std_err = NULL,
                          dependence = c("independent", "synchronized", "rank_correlation"),
                          spearman = NULL, ldf = NULL, ldf_kind = NULL, ldf_weight = 1,
                          n_periods = NULL, drop_high = 0, drop_low = 0) {
  column <- fit_column(triangle, column)
  process <- match.arg(process)
  dependence <- match.arg(dependence)
  excl <- development_args("volume", "log-linear", n_periods, drop_high, drop_low)$excl
  ptr <- rust_result(triangle@ptr$odp_bootstrap(column, single_number(n_sims, "n_sims"),
                                                single_number(seed, "seed"), process,
                                                tail_ptr(tail),
                                                optional_number(tail_std_err, "tail_std_err"),
                                                dependence, spearman_arg(spearman),
                                                ldf_pattern(ldf, ldf_kind, ldf_weight), excl))
  odp_bootstrap_fit(ptr = ptr)
}

# `spearman` as the binding takes it: NULL, or a square numeric matrix
# flattened row-major.
spearman_arg <- function(spearman) {
  if (is.null(spearman)) {
    return(NULL)
  }
  if (!is.numeric(spearman) || !is.matrix(spearman) || nrow(spearman) != ncol(spearman)) {
    stop("spearman must be a square numeric matrix, one row and column per segment",
         call. = FALSE)
  }
  as.double(t(spearman))
}

# The settings of bayes_odp() and bayes_odp_spec() that are numbers, checked.
bayes_odp_settings <- function(intercept_sd, origin_sd, development_sd, chains, tune, draws,
                               target_accept, max_depth) {
  list(intercept_sd = single_number(intercept_sd, "intercept_sd"),
       origin_sd = single_number(origin_sd, "origin_sd"),
       development_sd = single_number(development_sd, "development_sd"),
       chains = single_number(chains, "chains"), tune = single_number(tune, "tune"),
       draws = single_number(draws, "draws"),
       target_accept = single_number(target_accept, "target_accept"),
       max_depth = single_number(max_depth, "max_depth"))
}

# The settings of meyers() and meyers_spec() that are numbers, checked.
meyers_settings <- function(logelr_mean, logelr_sd, origin_sd, development_sd, gamma_sd,
                            rho_shape, chains, tune, draws, target_accept, max_depth) {
  list(logelr_mean = single_number(logelr_mean, "logelr_mean"),
       logelr_sd = single_number(logelr_sd, "logelr_sd"),
       origin_sd = single_number(origin_sd, "origin_sd"),
       development_sd = single_number(development_sd, "development_sd"),
       gamma_sd = single_number(gamma_sd, "gamma_sd"),
       rho_shape = single_number(rho_shape, "rho_shape"),
       chains = single_number(chains, "chains"), tune = single_number(tune, "tune"),
       draws = single_number(draws, "draws"),
       target_accept = single_number(target_accept, "target_accept"),
       max_depth = single_number(max_depth, "max_depth"))
}

#' Bayesian over-dispersed Poisson chain ladder
#'
#' The Bayesian chain ladder of England and Verrall (2002, 2006): the
#' over-dispersed Poisson model of a cumulative triangle's incremental
#' values with mean `exp(c + a[w] + b[d])` (`a[1] = b[1] = 0`), normal
#' priors on the log scale (`c` centred on the log of the mean observed
#' increment, the others on 0), and the chain ladder's Pearson scale `phi`
#' (the `scale` of [odp_bootstrap()]) plugged in. The posterior is sampled
#' with NUTS, and each draw projects the future cells up to the oldest
#' observed age (no tail) with the process error of `process`, as
#' [odp_bootstrap()] does. Every origin's and every development column's
#' observed increments must sum to more than zero; negative cells are
#' allowed. One `seed` replays a fit exactly, whatever the number of
#' threads: the chains and the simulations each take a seed derived from
#' it.
#'
#' Failed MCMC diagnostics (R-hat above 1.01, bulk or tail ESS below 100
#' per chain, any divergent transition) do not fail the fit:
#' `bayes_odp()` raises one [warning()] listing them, and they stay in the
#' fit's `diagnostics`.
#'
#' Properties of the fit: `chain_ladder` (the volume-weighted
#' [chain_ladder_fit] whose latest values and scale the model uses),
#' `origins`, `development`, `scale` (the plugged-in `phi`), `reserves` (a
#' [predictive_distribution] of the reserve with dimension `origin`, one
#' component per origin, laid out as [odp_bootstrap()]'s), `diagnostics` (a
#' list of `max_rhat`, `min_ess_bulk`, `min_ess_tail`, `divergences` and
#' `warnings`), `summary` (the posterior summary as `bayes_glm_model`'s:
#' `parameter`, `mean`, `sd`, `q05`, `q50`, `q95`, `rhat`, `ess_bulk` and
#' `ess_tail`; also `summary(fit)`) and `draws` (a matrix of the posterior
#' draws of `c`, `a[2]`, ..., `b[2]`, ..., one row per draw, chain by
#' chain). [bayes_loo()] gives the PSIS-LOO of the observed increments, with
#' the normalised over-dispersed Poisson density as the pointwise term, so
#' compare it only with models of the same incremental values; it is an
#' error on a triangle with a negative increment. `as.data.frame()`,
#' [totals_frame()], [development_frame()] and [segment()] work as for
#' [odp_bootstrap()], with the `mean` and `std_dev` of the posterior
#' predictive reserve.
#'
#' Errors: as [odp_bootstrap()]; if the triangle has several segments; if a
#' prior standard deviation is not positive and finite or a sampler setting
#' is invalid; or if an origin's or a development column's increments do
#' not sum to more than zero.
#'
#' @param triangle A cumulative [triangle] with one segment, every origin
#'   observed from the first age up to its latest.
#' @param column Name of the column to fit; by default the only one.
#' @param intercept_sd Standard deviation of the normal prior on the
#'   intercept `c`.
#' @param origin_sd Standard deviation of the normal prior on each origin
#'   effect `a[w]`, `w >= 2`.
#' @param development_sd Standard deviation of the normal prior on each
#'   development effect `b[d]`, `d >= 2`.
#' @param process Process error on each simulated future increment:
#'   `"gamma"` (mean `mu`, variance `phi * mu`) or `"none"` for parameter
#'   error only.
#' @param chains,tune,draws NUTS chains, warm-up draws per chain and kept
#'   draws per chain.
#' @param n_sims Number of predictive simulations, independent of the number
#'   of posterior draws; positive.
#' @param seed Seed of the chains and of the simulations, a non-negative
#'   whole number.
#' @param target_accept Target mean acceptance rate of the step-size
#'   adaptation; above NUTS's usual 0.8, which on triangles such as RAA
#'   usually leaves a few divergent transitions.
#' @param max_depth Largest NUTS tree depth.
#' @param ptr A `BayesOdpFit` pointer; used internally.
#' @returns A `bayes_odp_fit` object.
#' @seealso [odp_bootstrap()], [bayes_loo()].
#' @export
#' @examples
#' long <- data.frame(year = rep(2018:2021, 4:1),
#'                    age = c(12, 24, 36, 48, 12, 24, 36, 12, 24, 12),
#'                    paid = c(100, 150, 165, 170, 110, 170, 180, 120, 175, 130))
#' fit <- bayes_odp(triangle(long, "year", "age", "paid"), n_sims = 2000, seed = 1)
#' fit@diagnostics$max_rhat
#' head(fit@summary)
#' mean(fit@reserves)
#' quantile(fit@reserves, 0.995)
bayes_odp_fit <- S7::new_class(
  "bayes_odp_fit",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("BayesOdpFit"),
    chain_ladder = chain_ladder_fit,
    origins = S7::new_property(S7::class_character, getter = function(self) {
      self@chain_ladder@origins
    }),
    development = S7::new_property(S7::class_integer, getter = function(self) {
      self@chain_ladder@development
    }),
    scale = S7::new_property(S7::class_double, getter = function(self) self@ptr$scale()),
    reserves = predictive_distribution,
    diagnostics = S7::new_property(S7::class_list, getter = function(self) {
      self@ptr$diagnostics()
    }),
    summary = S7::new_property(S7::class_any, getter = function(self) {
      as.data.frame(rust_result(self@ptr$summary(), call = NULL))
    }),
    draws = S7::new_property(S7::class_any, getter = function(self) {
      nm <- self@ptr$parameter_names()
      matrix(self@ptr$draws(), ncol = length(nm), byrow = TRUE, dimnames = list(NULL, nm))
    })
  ),
  constructor = function(ptr) {
    S7::new_object(S7::S7_object(), ptr = ptr,
                   chain_ladder = chain_ladder_fit(ptr = ptr$chain_ladder()),
                   reserves = predictive_distribution(ptr = ptr$reserves()))
  }
)

#' @rdname bayes_odp_fit
#' @export
bayes_odp <- function(triangle, column = NULL, intercept_sd = 10, origin_sd = 10,
                      development_sd = 10, process = c("gamma", "none"), chains = 4,
                      tune = 1000, draws = 1000, n_sims = 10000, seed = 0,
                      target_accept = 0.95, max_depth = 10) {
  column <- fit_column(triangle, column)
  process <- match.arg(process)
  k <- bayes_odp_settings(intercept_sd, origin_sd, development_sd, chains, tune, draws,
                           target_accept, max_depth)
  ptr <- rust_result(triangle@ptr$bayes_odp(
    column, k$intercept_sd, k$origin_sd, k$development_sd, process, k$chains, k$tune, k$draws,
    single_number(n_sims, "n_sims"), single_number(seed, "seed"), k$target_accept,
    k$max_depth))
  fit <- bayes_odp_fit(ptr = ptr)
  failed <- fit@diagnostics$warnings
  if (length(failed)) {
    warning("bayes_odp: the MCMC diagnostics failed; check them before relying on the fit: ",
            paste(failed, collapse = "; "), call. = FALSE)
  }
  fit
}

#' Bayesian over-dispersed Poisson chain ladder over segments
#'
#' [bayes_odp()] fitted to every segment of a triangle column, each segment
#' with its own posterior (the same model and settings, sampled
#' independently), into one joint posterior predictive distribution of the
#' reserves laid out as [odp_bootstrap()]'s, the segments depending on each
#' other as `dependence` says. Segment `k` (in index order) takes seeds
#' derived from `seed` and `k` for its chains and its simulations, so
#' independent segments share no random numbers, and a segment's draws
#' differ from [bayes_odp()] of that segment alone. Failed MCMC diagnostics
#' raise one [warning()] listing every segment's messages, each prefixed
#' with its segment's label.
#'
#' Properties of the fit: `chain_ladder` (the volume-weighted
#' [chain_ladder_fit] of every segment), `origins`, `development`,
#' `reserves` (a [predictive_distribution] with the triangle's keys and
#' `origin` as dimensions, one component per segment and origin, so
#' `aggregate(fit@reserves, keep = "lob")` keeps the dependence between
#' segments) and `diagnostics` (a data.frame with one row per segment: the
#' key columns, `max_rhat`, `min_ess_bulk`, `min_ess_tail`, `divergences`
#' and `warnings`, the segment's messages joined by `"; "`). `scale`,
#' `summary`, `draws` and [bayes_loo()] are those of [bayes_odp_fit] and
#' need a single-segment fit: with several segments use [segment()], or
#' [totals_frame()] for the scales. `as.data.frame()`, [totals_frame()],
#' [development_frame()] and [segment()] work as for [odp_bootstrap()].
#'
#' Errors: as [bayes_odp()] for any segment (the message names the
#' segment); `dependence = "synchronized"`, before any sampling, since it
#' resamples the residuals of the same cells in every segment and a
#' posterior has no residuals to share; or a `spearman` matrix that does
#' not fit the segments.
#'
#' @inheritParams bayes_odp_fit
#' @param triangle A cumulative [triangle] with any number of segments,
#'   every origin observed from the first age up to its latest.
#' @param dependence How the segments depend on each other:
#'   `"independent"`, each segment's own posterior and simulations; or
#'   `"rank_correlation"`, the segments' simulations reordered as whole rows
#'   (Iman-Conover on the segment totals) to the Spearman matrix `spearman`
#'   and put in a random order, as [odp_bootstrap()]'s. `"synchronized"` is
#'   accepted as [odp_bootstrap()] takes it, and is an error here.
#' @param spearman With `dependence = "rank_correlation"` only: Spearman's
#'   rho between the segments' totals, as [odp_bootstrap()]'s.
#' @param ptr A `BayesOdpFits` pointer; used internally.
#' @returns A `bayes_odp_fits` object.
#' @seealso [bayes_odp()], [odp_bootstrap()].
#' @export
#' @examples
#' long <- data.frame(lob = rep(c("auto", "home"), each = 10),
#'                    year = rep(rep(2018:2021, 4:1), 2),
#'                    age = rep(c(12, 24, 36, 48, 12, 24, 36, 12, 24, 12), 2),
#'                    paid = c(100, 150, 165, 170, 110, 170, 180, 120, 175, 130,
#'                             200, 300, 330, 340, 220, 340, 360, 240, 350, 260))
#' tri <- triangle(long, "year", "age", "paid", keys = "lob")
#' # Short chains may miss a diagnostic threshold.
#' fits <- suppressWarnings(bayes_odp_segments(tri, tune = 300, draws = 300, n_sims = 1000))
#' fits@reserves@keys
#' fits@diagnostics
#' segment(fits, lob = "home")@scale
bayes_odp_fits <- S7::new_class(
  "bayes_odp_fits",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("BayesOdpFits"),
    chain_ladder = chain_ladder_fit,
    origins = S7::new_property(S7::class_character, getter = function(self) {
      self@chain_ladder@origins
    }),
    development = S7::new_property(S7::class_integer, getter = function(self) {
      self@chain_ladder@development
    }),
    scale = S7::new_property(S7::class_double, getter = function(self) {
      rust_result(self@ptr$scale(), call = NULL)
    }),
    reserves = predictive_distribution,
    diagnostics = S7::new_property(S7::class_data.frame, getter = function(self) {
      d <- self@ptr$diagnostics_table()
      out <- fit_table_frame(d$table)
      out$divergences <- as.integer(out$divergences)
      out$warnings <- d$warnings
      out
    }),
    summary = S7::new_property(S7::class_any, getter = function(self) {
      as.data.frame(rust_result(self@ptr$summary(), call = NULL))
    }),
    draws = S7::new_property(S7::class_any, getter = function(self) {
      nm <- rust_result(self@ptr$parameter_names(), call = NULL)
      matrix(rust_result(self@ptr$draws(), call = NULL), ncol = length(nm), byrow = TRUE,
             dimnames = list(NULL, nm))
    })
  ),
  constructor = function(ptr) {
    S7::new_object(S7::S7_object(), ptr = ptr,
                   chain_ladder = chain_ladder_fit(ptr = ptr$chain_ladder()),
                   reserves = predictive_distribution(ptr = ptr$reserves()))
  }
)

#' @rdname bayes_odp_fits
#' @export
bayes_odp_segments <- function(triangle, column = NULL, intercept_sd = 10, origin_sd = 10,
                               development_sd = 10, process = c("gamma", "none"), chains = 4,
                               tune = 1000, draws = 1000, n_sims = 10000, seed = 0,
                               target_accept = 0.95, max_depth = 10,
                               dependence = c("independent", "rank_correlation", "synchronized"),
                               spearman = NULL) {
  column <- fit_column(triangle, column)
  process <- match.arg(process)
  dependence <- match.arg(dependence)
  ptr <- rust_result(triangle@ptr$bayes_odp_segments(
    column, single_number(intercept_sd, "intercept_sd"), single_number(origin_sd, "origin_sd"),
    single_number(development_sd, "development_sd"), process, single_number(chains, "chains"),
    single_number(tune, "tune"), single_number(draws, "draws"), single_number(n_sims, "n_sims"),
    single_number(seed, "seed"), single_number(target_accept, "target_accept"),
    single_number(max_depth, "max_depth"), dependence, spearman_arg(spearman)))
  fit <- bayes_odp_fits(ptr = ptr)
  d <- fit@diagnostics
  failed <- d$warnings != ""
  if (any(failed)) {
    keys <- setdiff(names(d), c("max_rhat", "min_ess_bulk", "min_ess_tail", "divergences",
                                "warnings"))
    label <- if (length(keys)) {
      paste0(do.call(paste, c(unname(as.list(d[failed, keys, drop = FALSE])), sep = " / ")), ": ")
    } else {
      ""
    }
    warning("bayes_odp_segments: the MCMC diagnostics failed; check them before relying on ",
            "the fit: ", paste0(label, d$warnings[failed], collapse = "; "), call. = FALSE)
  }
  fit
}

#' One-year view of the Bayesian chain ladder
#'
#' The claims development result (CDR) over the coming year of
#' [bayes_odp()], by Pareto smoothed importance sampling of the posterior
#' (decision 16 of `docs/design/reserving-bayes.md`).
#' `bayes_odp_one_year()` fits the posterior and the lifetime reserves as
#' [bayes_odp()] does with the same settings. Each simulation picks a
#' posterior draw, draws the year's increments from it with `process`
#' error (the cells valued in the next twelve months, as in
#' [odp_one_year()]), reweights every posterior draw by the year's quasi
#' likelihood with the `scale` and the priors unchanged, and smooths the
#' weights (one Pareto k per simulation). Origin `w`'s CDR is
#' `E[R[w] | D0] - (X[w] + E[R'[w] | D0, X])`, with `R` the expected
#' run-off at the opening, `X` the year's payments and `R'` the expected
#' run-off after the year, each expectation the (weighted) mean over the
#' posterior draws; its mean is about zero. Every cell up to the end of the
#' year is observed and conditioned on, also for an origin that lags the
#' valuation.
#'
#' A simulation whose Pareto k is above `k_threshold` is flagged. With
#' `refit = "moment_match_then_refit"` (the default) its posterior draws
#' are first moved by affine maps to the moments the weights estimate
#' (moment matching, as `loo::loo_moment_match()`), and only a simulation
#' whose k stays above `k_threshold` is refitted with NUTS on the triangle
#' extended by its year, with the model's sampler. 14% to 26% of the
#' simulations are flagged on RAA and GenIns; matching rescues nearly all
#' of them, within 0.04 to 0.06 sd of their refits, at about 1 ms each
#' against about 26 ms for a refit, so a default run of 10,000 simulations
#' takes a few seconds in a release build. `"refit_flagged"` refits every
#' flagged simulation (one to two minutes there), and `"reweight_only"`
#' takes about a second. On RAA and GenIns
#' the standard deviation of the total CDR agrees with [odp_one_year()]'s
#' on average over seeds (with `"reweight_only"`, a ratio of 1.001 on RAA
#' and 1.006 on GenIns over 80 seeds); any single run of 10,000
#' simulations is within a few percent of it. A triangle with no cells in
#' the coming year (every origin at the last age) has a CDR of zero,
#' Pareto k of 0 and no refits.
#'
#' `bayes_odp_one_year_segments()` does the same in every segment of the
#' triangle, each segment's posterior and simulations as in
#' [bayes_odp_segments()] (segment `k`'s own seeds), into one joint CDR,
#' the segments depending on each other as `dependence` says:
#' `"independent"`, or `"rank_correlation"` by Iman-Conover on the
#' segments' total CDR, as the bootstraps' one-year view does.
#'
#' Failed MCMC diagnostics of the fit raise one [warning()] listing them,
#' and refits that failed theirs raise another (each prefixed with its
#' segment's label over segments).
#'
#' Properties of a `bayes_one_year_fit`: `fit` (the [bayes_odp_fit], the
#' [bayes_bf_fit] from [bayes_bf_one_year()], the [bayes_bf_normalised_fit] from
#' [bayes_bf_normalised_one_year()] or the [meyers_fit] from
#' [meyers_one_year()], with the posterior and lifetime reserves),
#' `chain_ladder`, `origins`,
#' `development`; per origin (named as a [chain_ladder_fit]'s) `latest`,
#' `opening_ultimate` and `opening_reserve` (the posterior mean reserve on
#' the observed triangle); `cdr`, a [predictive_distribution] of the CDR
#' with dimension `origin` (a negative value is an adverse development, and
#' `-quantile(fit@cdr, 0.005)` is the one-year loss at 99.5%); `pareto_k`
#' (one per simulation, `Inf` when too few weights differ to fit the tail),
#' `refitted` (the simulations refitted with NUTS, 1-based),
#' `moment_matched` (the simulations moment matching rescued, 1-based;
#' empty unless `refit = "moment_match_then_refit"`, when each flagged
#' simulation is in it, in `refitted` or in `unrefitted`), `unrefitted`
#' (the simulations due for a refit beyond `max_refits`, reweighted
#' instead, 1-based),
#' `refit_warnings` (how many refits had a diagnostic warning),
#' `k_threshold` and `flagged_share` (the share of simulations above it,
#' whether or not they were refitted). A `bayes_one_year_fits` has the
#' same, without `fit`, with `keys`, `index`, `scale` (the ODP's only),
#' `premium` (Meyers' only) and `diagnostics` (a data.frame as
#' [bayes_odp_segments()]'s), and with `cdr` keyed by the triangle's keys
#' and `origin`; `scale`, `premium`, `pareto_k`, `refitted`,
#' `moment_matched`, `unrefitted`, `refit_warnings` and `flagged_share` need a
#' single-segment fit: use
#' [segment()] or [totals_frame()]. `as.data.frame()` has one row per
#' segment and origin: `origin`, `latest`, `opening_ultimate`,
#' `opening_reserve`, and the `cdr_mean` and `cdr_std_dev` of the CDR;
#' [totals_frame()] one per segment, with the `scale` and the
#' `flagged_share` too.
#'
#' Errors: as [bayes_odp()] (or [bayes_odp_segments()]); a `k_threshold`
#' that is `NaN`.
#'
#' These are Python's `BayesOdp.one_year()` and
#' `BayesOdp.one_year_segments()`, which return a `BayesOneYearFit` and a
#' `BayesOneYearFits`.
#'
#' @inheritParams bayes_odp_fits
#' @param k_threshold The Pareto k above which a simulation's importance
#'   weights are not trusted (the threshold of Vehtari et al. 2024 for a
#'   few thousand draws).
#' @param refit What happens to a flagged simulation:
#'   `"moment_match_then_refit"` (the default) moves the posterior draws by
#'   affine maps to the moments the weights estimate (moment matching, as
#'   `loo::loo_moment_match()`), refitting only a simulation whose k stays
#'   above `k_threshold`; `"refit_flagged"` refits it with NUTS;
#'   `"reweight_only"` reweights it anyway (fast, biased where k is high);
#'   `"refit_all"` refits every simulation, to check the reweighting.
#' @param max_refits The most simulations refitted with NUTS: the first
#'   this many due for a refit, in simulation order; the others are
#'   reweighted as they are (as unreliable as their k says), listed in
#'   `unrefitted` and warned about. `NULL` (the default) refits every one,
#'   `0` none. In the `_segments` functions it applies to each segment, so
#'   k segments refit at most k times this many.
#' @param one_year_sims The number of one-year simulations, or `NULL` (the
#'   default) for the model's one-year default: `n_sims` for
#'   `bayes_odp_one_year()` and `bayes_bf_one_year()`, and `n_sims` capped
#'   at 2,000 for `meyers_one_year()`. The lifetime reserves keep `n_sims`.
#'   A smaller count gives the first simulations of a larger one.
#' @param ptr A `BayesOneYearFit` or `BayesOneYearFits` pointer; used
#'   internally.
#' @returns A `bayes_one_year_fit` or `bayes_one_year_fits` object.
#' @seealso [bayes_odp()], [odp_one_year()] for the ODP bootstrap's one-year
#'   view, [claims_development_result()] for Merz and Wuthrich's.
#' @export
#' @examples
#' long <- data.frame(year = rep(2018:2021, 4:1),
#'                    age = c(12, 24, 36, 48, 12, 24, 36, 12, 24, 12),
#'                    paid = c(100, 150, 165, 170, 110, 170, 180, 120, 175, 130))
#' tri <- triangle(long, "year", "age", "paid")
#' view <- bayes_odp_one_year(tri, n_sims = 1000, seed = 1, refit = "reweight_only")
#' view@opening_reserve
#' view@flagged_share
#' # The one-year view is narrower than the lifetime view.
#' c(one_year = sqrt(variance(view@cdr)), lifetime = sqrt(variance(view@fit@reserves)))
#' as.data.frame(view)
bayes_one_year_fit <- S7::new_class(
  "bayes_one_year_fit",
  package = "prospicio",
  properties = local({
    by_origin <- function(f) {
      S7::new_property(S7::class_double, getter = function(self) {
        stats::setNames(f(self), origin_names(self@chain_ladder@ptr))
      })
    }
    list(
      ptr = S7::new_S3_class("BayesOneYearFit"),
      fit = S7::class_any,
      chain_ladder = S7::new_property(chain_ladder_fit, getter = function(self) {
        self@fit@chain_ladder
      }),
      origins = S7::new_property(S7::class_character, getter = function(self) {
        self@chain_ladder@origins
      }),
      development = S7::new_property(S7::class_integer, getter = function(self) {
        self@chain_ladder@development
      }),
      latest = by_origin(function(self) self@chain_ladder@ptr$latest()),
      opening_ultimate = by_origin(function(self) self@ptr$opening_ultimate()),
      opening_reserve = by_origin(function(self) self@ptr$opening_reserve()),
      cdr = predictive_distribution,
      pareto_k = S7::new_property(S7::class_double, getter = function(self) self@ptr$pareto_k()),
      refitted = S7::new_property(S7::class_integer, getter = function(self) {
        as.integer(self@ptr$refitted())
      }),
      moment_matched = S7::new_property(S7::class_integer, getter = function(self) {
        as.integer(self@ptr$moment_matched())
      }),
      unrefitted = S7::new_property(S7::class_integer, getter = function(self) {
        as.integer(self@ptr$unrefitted())
      }),
      refit_warnings = S7::new_property(S7::class_integer, getter = function(self) {
        as.integer(self@ptr$refit_warnings())
      }),
      k_threshold = S7::new_property(S7::class_double, getter = function(self) {
        self@ptr$k_threshold()
      }),
      flagged_share = S7::new_property(S7::class_double, getter = function(self) {
        self@ptr$flagged_share()
      })
    )
  }),
  constructor = function(ptr) {
    fit <- switch(ptr$model(),
      bayes_bf = bayes_bf_fit(ptr = ptr$bf_fit()),
      bayes_bf_normalised = bayes_bf_normalised_fit(ptr = ptr$bf_normalised_fit()),
      meyers = meyers_fit(ptr = ptr$meyers_fit()),
      bayes_odp_fit(ptr = ptr$fit())
    )
    S7::new_object(S7::S7_object(), ptr = ptr, fit = fit,
                   cdr = predictive_distribution(ptr = ptr$cdr()))
  }
)

#' @rdname bayes_one_year_fit
#' @export
bayes_one_year_fits <- S7::new_class(
  "bayes_one_year_fits",
  package = "prospicio",
  properties = local({
    by_origin <- function(f) {
      S7::new_property(S7::class_double, getter = function(self) {
        stats::setNames(f(self), origin_names(self@chain_ladder@ptr))
      })
    }
    one <- function(class, f) {
      S7::new_property(class, getter = function(self) rust_result(f(self@ptr), call = NULL))
    }
    list(
      ptr = S7::new_S3_class("BayesOneYearFits"),
      chain_ladder = chain_ladder_fit,
      keys = S7::new_property(S7::class_character, getter = function(self) self@chain_ladder@keys),
      index = S7::new_property(S7::class_data.frame, getter = function(self) {
        self@chain_ladder@index
      }),
      origins = S7::new_property(S7::class_character, getter = function(self) {
        self@chain_ladder@origins
      }),
      development = S7::new_property(S7::class_integer, getter = function(self) {
        self@chain_ladder@development
      }),
      latest = by_origin(function(self) self@chain_ladder@ptr$latest()),
      opening_ultimate = by_origin(function(self) self@ptr$opening_ultimate()),
      opening_reserve = by_origin(function(self) self@ptr$opening_reserve()),
      cdr = predictive_distribution,
      scale = one(S7::class_double, function(p) p$scale()),
      premium = one(S7::class_double, function(p) p$premium()),
      diagnostics = S7::new_property(S7::class_data.frame, getter = function(self) {
        d <- self@ptr$diagnostics_table()
        out <- fit_table_frame(d$table)
        out$divergences <- as.integer(out$divergences)
        out$warnings <- d$warnings
        out
      }),
      pareto_k = one(S7::class_double, function(p) p$pareto_k()),
      refitted = one(S7::class_integer, function(p) as.integer(rust_result(p$refitted()))),
      moment_matched = one(S7::class_integer, function(p) {
        as.integer(rust_result(p$moment_matched()))
      }),
      unrefitted = one(S7::class_integer, function(p) {
        as.integer(rust_result(p$unrefitted()))
      }),
      refit_warnings = one(S7::class_integer, function(p) {
        as.integer(rust_result(p$refit_warnings()))
      }),
      k_threshold = S7::new_property(S7::class_double, getter = function(self) {
        self@ptr$k_threshold()
      }),
      flagged_share = one(S7::class_double, function(p) p$flagged_share())
    )
  }),
  constructor = function(ptr) {
    S7::new_object(S7::S7_object(), ptr = ptr,
                   chain_ladder = chain_ladder_fit(ptr = ptr$chain_ladder()),
                   cdr = predictive_distribution(ptr = ptr$cdr()))
  }
)

# The settings of bayes_odp() as the one-year views pass them to Rust.
bayes_odp_run <- function(intercept_sd, origin_sd, development_sd, chains, tune, draws, n_sims,
                          seed, target_accept, max_depth) {
  list(
    priors = c(single_number(intercept_sd, "intercept_sd"), single_number(origin_sd, "origin_sd"),
               single_number(development_sd, "development_sd")),
    run = c(single_number(chains, "chains"), single_number(tune, "tune"),
            single_number(draws, "draws"), single_number(n_sims, "n_sims"),
            single_number(seed, "seed"), single_number(target_accept, "target_accept"),
            single_number(max_depth, "max_depth"))
  )
}

# One warning if any NUTS refit of a one-year view failed its diagnostics,
# and another if max_refits left simulations unrefitted; `labels` prefix
# each segment's count, `refits`, `warned` and `unrefitted` count them.
warn_refits <- function(caller, labels, refits, warned, unrefitted) {
  failed <- warned > 0
  if (any(failed)) {
    warning(caller, ": the MCMC diagnostics of NUTS refits in the one-year view failed ",
            "(R-hat, ESS or a divergence): ",
            paste0(labels[failed], warned[failed], " of ", refits[failed], " refits",
                   collapse = "; "), call. = FALSE)
  }
  left <- unrefitted > 0
  if (any(left)) {
    warning(caller, ": simulations due for a refit beyond max_refits were reweighted ",
            "instead, their reweighting as unreliable as their Pareto k (listed in ",
            "unrefitted): ", paste0(labels[left], unrefitted[left], collapse = "; "),
            call. = FALSE)
  }
}

# `max_refits` as Rust takes it: NULL for no limit, else a non-negative
# whole number.
max_refits_arg <- function(max_refits) {
  if (is.null(max_refits)) NULL else single_number(max_refits, "max_refits")
}

#' @rdname bayes_one_year_fit
#' @export
bayes_odp_one_year <- function(triangle, column = NULL, intercept_sd = 10, origin_sd = 10,
                               development_sd = 10, process = c("gamma", "none"), chains = 4,
                               tune = 1000, draws = 1000, n_sims = 10000, seed = 0,
                               target_accept = 0.95, max_depth = 10, k_threshold = 0.7,
                               refit = c("moment_match_then_refit", "refit_flagged",
                                         "reweight_only", "refit_all"),
                               max_refits = NULL, one_year_sims = NULL) {
  column <- fit_column(triangle, column)
  process <- match.arg(process)
  refit <- match.arg(refit)
  s <- bayes_odp_run(intercept_sd, origin_sd, development_sd, chains, tune, draws, n_sims, seed,
                     target_accept, max_depth)
  ptr <- rust_result(triangle@ptr$bayes_odp_one_year(
    column, s$priors, process, s$run, single_number(k_threshold, "k_threshold"), refit,
    max_refits_arg(max_refits), optional_number(one_year_sims, "one_year_sims")))
  view <- bayes_one_year_fit(ptr = ptr)
  failed <- view@fit@diagnostics$warnings
  if (length(failed)) {
    warning("bayes_odp_one_year: the MCMC diagnostics failed; check them before relying on the ",
            "fit: ", paste(failed, collapse = "; "), call. = FALSE)
  }
  warn_refits("bayes_odp_one_year", "", length(view@refitted), view@refit_warnings,
              length(view@unrefitted))
  view
}

#' @rdname bayes_one_year_fit
#' @export
bayes_odp_one_year_segments <- function(triangle, column = NULL, intercept_sd = 10,
                                        origin_sd = 10, development_sd = 10,
                                        process = c("gamma", "none"), chains = 4, tune = 1000,
                                        draws = 1000, n_sims = 10000, seed = 0,
                                        target_accept = 0.95, max_depth = 10,
                                        dependence = c("independent", "rank_correlation",
                                                       "synchronized"),
                                        spearman = NULL, k_threshold = 0.7,
                                        refit = c("moment_match_then_refit", "refit_flagged",
                                                  "reweight_only", "refit_all"),
                                        max_refits = NULL, one_year_sims = NULL) {
  column <- fit_column(triangle, column)
  process <- match.arg(process)
  dependence <- match.arg(dependence)
  refit <- match.arg(refit)
  s <- bayes_odp_run(intercept_sd, origin_sd, development_sd, chains, tune, draws, n_sims, seed,
                     target_accept, max_depth)
  ptr <- rust_result(triangle@ptr$bayes_odp_one_year_segments(
    column, s$priors, process, s$run, single_number(k_threshold, "k_threshold"), refit,
    max_refits_arg(max_refits), optional_number(one_year_sims, "one_year_sims"),
    dependence, spearman_arg(spearman)))
  view <- bayes_one_year_fits(ptr = ptr)
  d <- view@diagnostics
  keys <- setdiff(names(d), c("max_rhat", "min_ess_bulk", "min_ess_tail", "divergences",
                              "warnings"))
  labels <- if (length(keys)) {
    paste0(do.call(paste, c(unname(as.list(d[, keys, drop = FALSE])), sep = " / ")), ": ")
  } else {
    rep("", nrow(d))
  }
  failed <- d$warnings != ""
  if (any(failed)) {
    warning("bayes_odp_one_year_segments: the MCMC diagnostics failed; check them before ",
            "relying on the fit: ", paste0(labels[failed], d$warnings[failed], collapse = "; "),
            call. = FALSE)
  }
  counts <- view@ptr$refit_counts()
  warn_refits("bayes_odp_one_year_segments", labels, counts$refits, counts$warned,
              counts$unrefitted)
  view
}

#' Meyers' CRC, CSR and CAY models
#'
#' Meyers' (2019, CAS Monograph 8) log-normal models of a cumulative
#' triangle `C` with premium `P`, for origins `w = 1..n` and ages
#' `d = 1..m`: `log C[w, d]` is normal with mean
#' `log P[w] + logelr + alpha[w] + beta[d] g[w]` (`alpha[1] = 0`,
#' `beta[m] = 0`) and sd `sigma[d]`, decreasing with age. `g[w]` is 1 for
#' the correlated chain ladder without its correlation between origins
#' (`"crc"`) and `(1 - gamma)^(w - 1)` for the changing settlement rate
#' (`"csr"`). The correlated accident year model (`"cay"`) keeps
#' `g[w] = 1` and adds `rho` times the previous origin's residual at the
#' same age, `log C[w - 1, d]` less its mean, to the mean of every origin
#' after the first, so neighbouring origins are correlated; Meyers fits it
#' to incurred data. The priors are those of Meyers' Stan code: `logelr`
#' normal with mean `logelr_mean` and sd `logelr_sd`, `alpha` and `beta`
#' normal around 0, `gamma` normal around 0 with sd `gamma_sd`,
#' `(1 - rho) / 2` beta with both shapes `rho_shape`, and each step of
#' `sigma[d]^2` uniform on `(1 - exp(-1e-5), 1)`.
#'
#' The ultimate is the value at the oldest age (no tail): each posterior
#' draw gives each open origin a log-normal ultimate; an origin observed at
#' the oldest age keeps its value; under `"cay"` the ultimates are drawn
#' origin by origin, each from the previous origin's residual at the oldest
#' age, observed for a closed origin and simulated otherwise.
#' The reserve is the ultimate less the latest value of the fitted column
#' (an outstanding reserve on paid data, IBNR on incurred data) and can be
#' negative. Every observed cell must be positive, and under `"cay"` every
#' observed cell after the first origin needs the previous origin's cell at
#' the same age. One `seed` replays a fit exactly, whatever the number of
#' threads: the chains and the simulations each take a seed derived from
#' it.
#'
#' Failed MCMC diagnostics (R-hat above 1.01, bulk or tail ESS below 100
#' per chain, any divergent transition) do not fail the fit: `meyers()`
#' raises one [warning()] listing them, and they stay in the fit's
#' `diagnostics`. With the defaults, all 120 fits of a seed sweep on
#' Meyers' group 353 data (paid and incurred, CRC and CSR, 30 seeds) passed
#' every threshold, and CAY on incurred passed at 199 of 200 seeds (one
#' divergent transition in 8,000 draws at the other).
#'
#' Properties of the fit: `model` (`"crc"`, `"csr"` or `"cay"`), `chain_ladder`
#' (the volume-weighted [chain_ladder_fit] the reserves are measured from),
#' `origins`, `development`, `premium` (each origin's), `ultimates` and
#' `reserves` ([predictive_distribution]s with dimension `origin`, one
#' component per origin, laid out as [odp_bootstrap()]'s reserves),
#' `diagnostics` (a list of `max_rhat`, `min_ess_bulk`, `min_ess_tail`,
#' `divergences` and `warnings`), `summary` (the posterior summary as
#' `bayes_glm_model`'s; also `summary(fit)`) and `draws` (a matrix of the
#' posterior draws of `logelr`, `alpha[2]`, ..., `beta[1]`, ..., `gamma`
#' (CSR only) or `rho` (CAY only), `sigma[1]`, ..., one row per draw, chain
#' by chain). [bayes_loo()] gives the PSIS-LOO of the observed cells with
#' Meyers' pointwise term, the normal density of `log C[w, d]` (under CAY
#' conditional on the previous origin's cell, as in Meyers' code), so
#' compare it only between CRC, CSR and CAY fits of the same column (Meyers
#' 2019, Tables 6.4, 7.3 and 8.3). `as.data.frame()`, [totals_frame()], [development_frame()] and
#' [segment()] work as for [bayes_odp()], with the `premium`, the `mean` and
#' `std_dev` of the posterior predictive reserve and the `ultimate_mean` and
#' `ultimate_std_dev` of the posterior predictive ultimate.
#'
#' Errors: if a column is missing, the triangle has several segments or an
#' origin has a gap; if a premium is missing or not positive; if a prior
#' standard deviation or `rho_shape` is not positive and finite,
#' `logelr_mean` is not finite or a sampler setting is invalid; if an
#' observed cell is not positive; if there are no more observed cells than
#' parameters; or, under `"cay"`, if a cell lacks the previous origin's
#' cell at the same age.
#'
#' @param triangle A cumulative [triangle] with one segment, every origin
#'   observed from the first age up to its latest.
#' @param column Name of the column to fit: paid, or case incurred.
#' @param premium Name of the column with each origin's premium (its latest
#'   value, as [cape_cod()] reads exposure).
#' @param model `"crc"`, the correlated chain ladder without its
#'   correlation between origins, `"csr"`, the changing settlement rate
#'   model, or `"cay"`, the correlated accident year model.
#' @param logelr_mean,logelr_sd Mean and standard deviation of the normal
#'   prior on `logelr`, the log expected loss ratio (3.162 is Meyers'
#'   rounding of `sqrt(10)`).
#' @param origin_sd Standard deviation of the normal prior on each origin
#'   effect `alpha[w]`, `w >= 2`.
#' @param development_sd Standard deviation of the normal prior on each
#'   development effect `beta[d]`, `d < m`.
#' @param gamma_sd Standard deviation of the normal prior on `gamma`
#'   (`"csr"` only).
#' @param rho_shape Both shapes of the beta prior on `(1 - rho) / 2`
#'   (`"cay"` only); Meyers' 2 makes the prior density of `rho`
#'   proportional to `1 - rho^2`.
#' @param chains,tune,draws NUTS chains, warm-up draws per chain and kept
#'   draws per chain; with 1,000 kept draws the late `sigma[d]` sometimes
#'   miss the R-hat threshold.
#' @param n_sims Number of predictive simulations, independent of the number
#'   of posterior draws; positive.
#' @param seed Seed of the chains and of the simulations, a non-negative
#'   whole number.
#' @param target_accept Target mean acceptance rate of the step-size
#'   adaptation; at NUTS's usual 0.8 the paid fits of group 353 leave
#'   divergent transitions.
#' @param max_depth Largest NUTS tree depth.
#' @param ptr A `MeyersFit` pointer; used internally.
#' @returns A `meyers_fit` object.
#' @seealso [meyers_segments()] for several segments, [bayes_odp()], [cape_cod()],
#'   [bayes_loo()].
#' @export
#' @examples
#' long <- data.frame(year = rep(1988:1993, 6:1),
#'                    age = unlist(lapply(6:1, function(k) 12 * seq_len(k))),
#'                    paid = c(952, 1529, 2813, 3647, 3724, 3832, 849, 1564, 2202, 2432,
#'                             2468, 983, 2211, 2830, 3832, 1657, 2685, 3169, 932, 1940, 1162),
#'                    premium = rep(c(5812, 4908, 5454, 5165, 5214, 5230), 6:1))
#' tri <- triangle(long, "year", "age", c("paid", "premium"))
#' # Short chains may miss a diagnostic threshold.
#' fit <- suppressWarnings(meyers(tri, "paid", "premium", model = "csr", tune = 300,
#'                                draws = 300, n_sims = 2000, seed = 1))
#' head(fit@summary)
#' mean(fit@reserves)
#' quantile(fit@ultimates, 0.995)
meyers_fit <- S7::new_class(
  "meyers_fit",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("MeyersFit"),
    model = S7::new_property(S7::class_character, getter = function(self) self@ptr$model()),
    chain_ladder = chain_ladder_fit,
    origins = S7::new_property(S7::class_character, getter = function(self) {
      self@chain_ladder@origins
    }),
    development = S7::new_property(S7::class_integer, getter = function(self) {
      self@chain_ladder@development
    }),
    premium = S7::new_property(S7::class_double, getter = function(self) self@ptr$premium()),
    ultimates = predictive_distribution,
    reserves = predictive_distribution,
    diagnostics = S7::new_property(S7::class_list, getter = function(self) {
      self@ptr$diagnostics()
    }),
    summary = S7::new_property(S7::class_any, getter = function(self) {
      as.data.frame(rust_result(self@ptr$summary(), call = NULL))
    }),
    draws = S7::new_property(S7::class_any, getter = function(self) {
      nm <- self@ptr$parameter_names()
      matrix(self@ptr$draws(), ncol = length(nm), byrow = TRUE, dimnames = list(NULL, nm))
    })
  ),
  constructor = function(ptr) {
    S7::new_object(S7::S7_object(), ptr = ptr,
                   chain_ladder = chain_ladder_fit(ptr = ptr$chain_ladder()),
                   ultimates = predictive_distribution(ptr = ptr$ultimates()),
                   reserves = predictive_distribution(ptr = ptr$reserves()))
  }
)

#' @rdname meyers_fit
#' @export
meyers <- function(triangle, column, premium, model = c("crc", "csr", "cay"),
                   logelr_mean = -0.4, logelr_sd = 3.162, origin_sd = 3.162,
                   development_sd = 3.162, gamma_sd = 0.05, rho_shape = 2,
                   chains = 4, tune = 1000, draws = 2000, n_sims = 10000, seed = 0,
                   target_accept = 0.95, max_depth = 10) {
  check_triangle(triangle)
  for (arg in c("column", "premium")) {
    value <- get(arg)
    if (!is.character(value) || length(value) != 1 || is.na(value)) {
      stop(sprintf("%s must be the name of a column", arg), call. = FALSE)
    }
  }
  model <- match.arg(model)
  k <- meyers_settings(logelr_mean, logelr_sd, origin_sd, development_sd, gamma_sd, rho_shape,
                      chains, tune, draws, target_accept, max_depth)
  ptr <- rust_result(triangle@ptr$meyers(
    column, premium, model, k$logelr_mean, k$logelr_sd, k$origin_sd, k$development_sd,
    k$gamma_sd, k$rho_shape, k$chains, k$tune, k$draws, single_number(n_sims, "n_sims"),
    single_number(seed, "seed"), k$target_accept, k$max_depth))
  fit <- meyers_fit(ptr = ptr)
  failed <- fit@diagnostics$warnings
  if (length(failed)) {
    warning("meyers: the MCMC diagnostics failed; check them before relying on the fit: ",
            paste(failed, collapse = "; "), call. = FALSE)
  }
  fit
}

#' Meyers' CRC, CSR and CAY models over segments
#'
#' [meyers()] fitted to every segment of a triangle column, each segment
#' with its own posterior (the same model and settings, sampled
#' independently) and its premium from its own `premium` column as
#' [meyers()] reads it, into joint posterior predictive distributions of
#' the ultimates and of the reserves laid out as [bayes_odp_segments()]'s,
#' the segments depending on each other as `dependence` says. Segment `k`
#' (in index order) takes seeds derived from `seed` and `k` for its chains
#' and its simulations, so independent segments share no random numbers,
#' and a segment's draws differ from [meyers()] of that segment alone.
#' Under `dependence = "rank_correlation"` each segment's ultimates move
#' with its reserves, simulation by simulation, so each row's ultimates
#' less its reserves stay the latest values. Failed MCMC diagnostics raise
#' one [warning()] listing every segment's messages, each prefixed with
#' its segment's label.
#'
#' Properties of the fit: `model`, `chain_ladder` (the volume-weighted
#' [chain_ladder_fit] of every segment), `origins`, `development`,
#' `premium` (each segment's origins in turn, in the order of `origins`),
#' `ultimates` and `reserves` ([predictive_distribution]s with the
#' triangle's keys and `origin` as dimensions, one component per segment
#' and origin, so `aggregate(fit@reserves, keep = "lob")` keeps the
#' dependence between segments) and `diagnostics` (a data.frame with one
#' row per segment: the key columns, `max_rhat`, `min_ess_bulk`,
#' `min_ess_tail`, `divergences` and `warnings`, the segment's messages
#' joined by `"; "`). `summary`, `draws` and [bayes_loo()] are those of
#' [meyers_fit] and need a single-segment fit: with several segments use
#' [segment()]. `as.data.frame()`, [totals_frame()] (with each segment's
#' total `premium` and the moments of its total ultimate),
#' [development_frame()] and [segment()] work as for [meyers()].
#'
#' Errors: as [meyers()] for any segment (the message names the segment);
#' `dependence = "synchronized"`, before any sampling, since a posterior
#' has no residuals to share; or a `spearman` matrix that does not fit the
#' segments.
#'
#' @inheritParams meyers_fit
#' @param triangle A cumulative [triangle] with any number of segments,
#'   every origin observed from the first age up to its latest.
#' @param dependence How the segments depend on each other:
#'   `"independent"`, each segment's own posterior and simulations; or
#'   `"rank_correlation"`, the segments' simulations reordered as whole rows
#'   (Iman-Conover on the segments' total reserves) to the Spearman matrix
#'   `spearman` and put in a random order, as [odp_bootstrap()]'s.
#'   `"synchronized"` is accepted as [odp_bootstrap()] takes it, and is an
#'   error here.
#' @param spearman With `dependence = "rank_correlation"` only: Spearman's
#'   rho between the segments' total reserves, as [odp_bootstrap()]'s.
#' @param ptr A `MeyersFits` pointer; used internally.
#' @returns A `meyers_fits` object.
#' @seealso [meyers()], [bayes_odp_segments()].
#' @export
#' @examples
#' paid <- c(952, 1529, 2813, 3647, 3724, 849, 1564, 2202, 2432, 983, 2211, 2830,
#'           1657, 2685, 932)
#' one <- data.frame(year = rep(1988:1992, 5:1),
#'                   age = unlist(lapply(5:1, function(k) 12 * seq_len(k))),
#'                   paid = paid,
#'                   premium = rep(c(5812, 4908, 5454, 5165, 5214), 5:1))
#' two <- one
#' two[c("paid", "premium")] <- 2 * two[c("paid", "premium")]
#' long <- rbind(cbind(lob = "auto", one), cbind(lob = "home", two))
#' tri <- triangle(long, "year", "age", c("paid", "premium"), keys = "lob")
#' # Short chains may miss a diagnostic threshold.
#' fits <- suppressWarnings(meyers_segments(tri, "paid", "premium", tune = 200, draws = 200,
#'                                          n_sims = 1000))
#' fits@reserves@keys
#' fits@diagnostics
#' segment(fits, lob = "home")@premium
meyers_fits <- S7::new_class(
  "meyers_fits",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("MeyersFits"),
    model = S7::new_property(S7::class_character, getter = function(self) self@ptr$model()),
    chain_ladder = chain_ladder_fit,
    origins = S7::new_property(S7::class_character, getter = function(self) {
      self@chain_ladder@origins
    }),
    development = S7::new_property(S7::class_integer, getter = function(self) {
      self@chain_ladder@development
    }),
    premium = S7::new_property(S7::class_double, getter = function(self) self@ptr$premium()),
    ultimates = predictive_distribution,
    reserves = predictive_distribution,
    diagnostics = S7::new_property(S7::class_data.frame, getter = function(self) {
      d <- self@ptr$diagnostics_table()
      out <- fit_table_frame(d$table)
      out$divergences <- as.integer(out$divergences)
      out$warnings <- d$warnings
      out
    }),
    summary = S7::new_property(S7::class_any, getter = function(self) {
      as.data.frame(rust_result(self@ptr$summary(), call = NULL))
    }),
    draws = S7::new_property(S7::class_any, getter = function(self) {
      nm <- rust_result(self@ptr$parameter_names(), call = NULL)
      matrix(rust_result(self@ptr$draws(), call = NULL), ncol = length(nm), byrow = TRUE,
             dimnames = list(NULL, nm))
    })
  ),
  constructor = function(ptr) {
    S7::new_object(S7::S7_object(), ptr = ptr,
                   chain_ladder = chain_ladder_fit(ptr = ptr$chain_ladder()),
                   ultimates = predictive_distribution(ptr = ptr$ultimates()),
                   reserves = predictive_distribution(ptr = ptr$reserves()))
  }
)

#' @rdname meyers_fits
#' @export
meyers_segments <- function(triangle, column, premium, model = c("crc", "csr", "cay"),
                            logelr_mean = -0.4, logelr_sd = 3.162, origin_sd = 3.162,
                            development_sd = 3.162, gamma_sd = 0.05, rho_shape = 2,
                            chains = 4, tune = 1000, draws = 2000, n_sims = 10000, seed = 0,
                            target_accept = 0.95, max_depth = 10,
                            dependence = c("independent", "rank_correlation", "synchronized"),
                            spearman = NULL) {
  check_triangle(triangle)
  for (arg in c("column", "premium")) {
    value <- get(arg)
    if (!is.character(value) || length(value) != 1 || is.na(value)) {
      stop(sprintf("%s must be the name of a column", arg), call. = FALSE)
    }
  }
  model <- match.arg(model)
  dependence <- match.arg(dependence)
  ptr <- rust_result(triangle@ptr$meyers_segments(
    column, premium, model, single_number(logelr_mean, "logelr_mean"),
    single_number(logelr_sd, "logelr_sd"), single_number(origin_sd, "origin_sd"),
    single_number(development_sd, "development_sd"), single_number(gamma_sd, "gamma_sd"),
    single_number(rho_shape, "rho_shape"), single_number(chains, "chains"),
    single_number(tune, "tune"), single_number(draws, "draws"),
    single_number(n_sims, "n_sims"), single_number(seed, "seed"),
    single_number(target_accept, "target_accept"), single_number(max_depth, "max_depth"),
    dependence, spearman_arg(spearman)))
  fit <- meyers_fits(ptr = ptr)
  d <- fit@diagnostics
  failed <- d$warnings != ""
  if (any(failed)) {
    keys <- setdiff(names(d), c("max_rhat", "min_ess_bulk", "min_ess_tail", "divergences",
                                "warnings"))
    label <- if (length(keys)) {
      paste0(do.call(paste, c(unname(as.list(d[failed, keys, drop = FALSE])), sep = " / ")), ": ")
    } else {
      ""
    }
    warning("meyers_segments: the MCMC diagnostics failed; check them before relying on ",
            "the fit: ", paste0(label, d$warnings[failed], collapse = "; "), call. = FALSE)
  }
  fit
}

#' One-year view of Meyers' CRC, CSR and CAY models
#'
#' The claims development result (CDR) over the coming year of [meyers()],
#' by Pareto smoothed importance sampling of the posterior (decision 19 of
#' `docs/design/reserving-bayes.md`). `meyers_one_year()` fits the
#' posterior and the lifetime ultimates and reserves as [meyers()] does
#' with the same settings. Each simulation picks a posterior draw, draws
#' the cumulative values of the year's cells from it (the cells valued in
#' the next twelve months, as in [odp_one_year()]), reweights every
#' posterior draw by the likelihood of those values, and smooths the
#' weights (one Pareto k per simulation). Origin `w`'s CDR is
#' `E[U[w] | D0] - E[U[w] | D0, C]`, with `U` the ultimate (the value at
#' the oldest age) and each expectation the (weighted) posterior mean of
#' the ultimate's expected value given a draw; an origin the year takes to
#' the oldest age has its value there instead of the second term. Its
#' mean is about zero. The draws are reweighted on the unconstrained scale
#' NUTS samples; the fit's posterior stays on the natural scale.
#'
#' A simulation whose Pareto k is above `k_threshold` is handled as
#' `refit` says: by default moment-matched, and refitted with NUTS only if
#' its k stays above the threshold. On Meyers' group 353 nearly every
#' simulation is flagged and matching rescues about nine in ten, so 10,000
#' simulations take 4 to 7 minutes on 16 threads, and `"refit_flagged"`
#' about 6.7 times as long. The view therefore runs at most 2,000
#' simulations by default (under a minute and a half there), whatever
#' `n_sims` (the lifetime reserves' count) is above that; `one_year_sims`
#' sets the count. Fewer simulations cost proportionally less.
#'
#' `meyers_one_year_segments()` does the same in every segment of the
#' triangle, each segment's posterior and simulations as in
#' [meyers_segments()] (segment `k`'s own seeds), into one joint CDR, the
#' segments depending on each other as `dependence` says: `"independent"`,
#' or `"rank_correlation"` by Iman-Conover on the segments' total CDR.
#'
#' Failed MCMC diagnostics of the fit raise one [warning()] listing them,
#' and refits that failed theirs raise another (each prefixed with its
#' segment's label over segments). The result is a [bayes_one_year_fit]
#' whose `fit` is the [meyers_fit] (its `as.data.frame()` adds each
#' origin's `premium`), or a [bayes_one_year_fits] with `premium`.
#'
#' Errors: as [meyers()] (or [meyers_segments()]); a `k_threshold` that is
#' `NaN`; a `one_year_sims` of 0.
#'
#' These are Python's `Meyers.one_year()` and `Meyers.one_year_segments()`.
#'
#' @inheritParams meyers_fit
#' @inheritParams meyers_fits
#' @inheritParams bayes_one_year_fit
#' @returns A `bayes_one_year_fit` or `bayes_one_year_fits` object.
#' @seealso [meyers()], [bayes_odp_one_year()], [odp_one_year()].
#' @export
#' @examples
#' long <- data.frame(
#'   year = rep(1988:1993, 6:1),
#'   age = 12 * sequence(6:1),
#'   paid = c(952, 1529, 2813, 3647, 3724, 3832, 849, 1564, 2202, 2432, 2468,
#'            983, 2211, 2830, 3832, 1657, 2685, 3169, 932, 1940, 1162),
#'   premium = rep(c(5812, 4908, 5454, 5165, 5214, 5230), 6:1))
#' tri <- triangle(long, "year", "age", c("paid", "premium"))
#' view <- suppressWarnings(meyers_one_year(tri, "paid", "premium", tune = 300, draws = 300,
#'                                          n_sims = 1000, seed = 1,
#'                                          refit = "reweight_only"))
#' view@fit@premium
#' c(one_year = sqrt(variance(view@cdr)), lifetime = sqrt(variance(view@fit@reserves)))
#' as.data.frame(view)
meyers_one_year <- function(triangle, column, premium, model = c("crc", "csr", "cay"),
                            logelr_mean = -0.4, logelr_sd = 3.162, origin_sd = 3.162,
                            development_sd = 3.162, gamma_sd = 0.05, rho_shape = 2,
                            chains = 4, tune = 1000, draws = 2000, n_sims = 10000, seed = 0,
                            target_accept = 0.95, max_depth = 10, k_threshold = 0.7,
                            refit = c("moment_match_then_refit", "refit_flagged",
                                      "reweight_only", "refit_all"),
                            max_refits = NULL, one_year_sims = NULL) {
  s <- meyers_args(triangle, column, premium, logelr_mean, logelr_sd, origin_sd,
                   development_sd, gamma_sd, rho_shape, chains, tune, draws, n_sims, seed,
                   target_accept, max_depth)
  model <- match.arg(model)
  refit <- match.arg(refit)
  ptr <- rust_result(triangle@ptr$meyers_one_year(
    column, premium, model, s$priors, s$run, single_number(k_threshold, "k_threshold"), refit,
    max_refits_arg(max_refits), optional_number(one_year_sims, "one_year_sims")))
  view <- bayes_one_year_fit(ptr = ptr)
  failed <- view@fit@diagnostics$warnings
  if (length(failed)) {
    warning("meyers_one_year: the MCMC diagnostics failed; check them before relying on the ",
            "fit: ", paste(failed, collapse = "; "), call. = FALSE)
  }
  warn_refits("meyers_one_year", "", length(view@refitted), view@refit_warnings,
              length(view@unrefitted))
  view
}

#' @rdname meyers_one_year
#' @export
meyers_one_year_segments <- function(triangle, column, premium, model = c("crc", "csr", "cay"),
                                     logelr_mean = -0.4, logelr_sd = 3.162, origin_sd = 3.162,
                                     development_sd = 3.162, gamma_sd = 0.05, rho_shape = 2,
                                     chains = 4, tune = 1000, draws = 2000, n_sims = 10000,
                                     seed = 0, target_accept = 0.95, max_depth = 10,
                                     dependence = c("independent", "rank_correlation",
                                                    "synchronized"),
                                     spearman = NULL, k_threshold = 0.7,
                                     refit = c("moment_match_then_refit", "refit_flagged",
                                               "reweight_only", "refit_all"),
                                     max_refits = NULL, one_year_sims = NULL) {
  s <- meyers_args(triangle, column, premium, logelr_mean, logelr_sd, origin_sd,
                   development_sd, gamma_sd, rho_shape, chains, tune, draws, n_sims, seed,
                   target_accept, max_depth)
  model <- match.arg(model)
  dependence <- match.arg(dependence)
  refit <- match.arg(refit)
  ptr <- rust_result(triangle@ptr$meyers_one_year_segments(
    column, premium, model, s$priors, s$run, single_number(k_threshold, "k_threshold"), refit,
    max_refits_arg(max_refits), optional_number(one_year_sims, "one_year_sims"),
    dependence, spearman_arg(spearman)))
  view <- bayes_one_year_fits(ptr = ptr)
  d <- view@diagnostics
  keys <- setdiff(names(d), c("max_rhat", "min_ess_bulk", "min_ess_tail", "divergences",
                              "warnings"))
  labels <- if (length(keys)) {
    paste0(do.call(paste, c(unname(as.list(d[, keys, drop = FALSE])), sep = " / ")), ": ")
  } else {
    rep("", nrow(d))
  }
  failed <- d$warnings != ""
  if (any(failed)) {
    warning("meyers_one_year_segments: the MCMC diagnostics failed; check them before ",
            "relying on the fit: ", paste0(labels[failed], d$warnings[failed], collapse = "; "),
            call. = FALSE)
  }
  counts <- view@ptr$refit_counts()
  warn_refits("meyers_one_year_segments", labels, counts$refits, counts$warned,
              counts$unrefitted)
  view
}

# The checked column names and the settings of meyers() as the one-year
# views pass them to Rust.
meyers_args <- function(triangle, column, premium, logelr_mean, logelr_sd, origin_sd,
                        development_sd, gamma_sd, rho_shape, chains, tune, draws, n_sims, seed,
                        target_accept, max_depth) {
  check_triangle(triangle)
  for (arg in c("column", "premium")) {
    value <- get(arg)
    if (!is.character(value) || length(value) != 1 || is.na(value)) {
      stop(sprintf("%s must be the name of a column", arg), call. = FALSE)
    }
  }
  list(
    priors = c(single_number(logelr_mean, "logelr_mean"), single_number(logelr_sd, "logelr_sd"),
               single_number(origin_sd, "origin_sd"),
               single_number(development_sd, "development_sd"),
               single_number(gamma_sd, "gamma_sd"), single_number(rho_shape, "rho_shape")),
    run = c(single_number(chains, "chains"), single_number(tune, "tune"),
            single_number(draws, "draws"), single_number(n_sims, "n_sims"),
            single_number(seed, "seed"), single_number(target_accept, "target_accept"),
            single_number(max_depth, "max_depth"))
  )
}

#' Bayesian Bornhuetter-Ferguson
#'
#' The Bayesian Bornhuetter-Ferguson of England, Verrall and Wuthrich (2012,
#' Model 3.1): the over-dispersed Poisson model of a cumulative triangle's
#' incremental values with mean `mu[w] * gamma[d]`, an informative gamma
#' prior on each origin's row parameter `mu[w]` with mean `apriori` times
#' the origin's exposure and coefficient of variation `cv`, a vague normal
#' prior on each `log gamma[d]` (the pattern is not normalised), and the
#' chain ladder's
#' Pearson scale `phi` (the `scale` of [odp_bootstrap()]) plugged in. The
#' posterior is sampled with NUTS, and each draw projects the future cells
#' up to the oldest observed age (no tail) with the process error of
#' `process`, as [bayes_odp()] does.
#'
#' The prior's strength moves the reserve between two limits. With
#' `cv = Inf` (a flat prior on `log mu[w]`) the model is the Bayesian chain
#' ladder, and the reserves agree with [bayes_odp()]'s within Monte Carlo
#' error. As `cv` falls to 0, `mu[w]` is fixed at the prior ultimate `m[w]`
#' and each origin's mean reserve tends to `m[w]` times the sum of the
#' future `S[d] / M[d]` (`S[d]` the observed increments of column `d`,
#' `M[d]` the prior ultimates of the origins observed there): EVW's
#' strong-prior limit, Mack's (2008) Bornhuetter-Ferguson with the raw
#' pattern. That equals the classical [bornhuetter_ferguson()] reserve
#' `(1 - 1 / cdf) * m[w]` only when the prior ultimates are the chain
#' ladder's.
#'
#' `mu[w]` is EVW's row parameter, not the origin's expected ultimate: that
#' is `mu[w]` times the sum of the `gamma[d]`, which under a tight prior is
#' `m[w]` times the sum of `S[d] / M[d]` over all columns. On EVW's
#' triangle that sum is 0.885, so the implied ultimate loss ratio is about
#' 11.5% below `apriori`.
#'
#' Every development column's observed increments must sum to more than
#' zero, and every origin's to more than `-phi / cv^2`; negative cells are
#' allowed. One `seed` replays a fit exactly, whatever the number of
#' threads. Failed MCMC diagnostics (R-hat above 1.01, bulk or tail ESS
#' below 100 per chain, any divergent transition) do not fail the fit:
#' `bayes_bf()` raises one [warning()] listing them, and they stay in the
#' fit's `diagnostics`.
#'
#' Properties of the fit: as [bayes_odp_fit]'s (`chain_ladder`, `origins`,
#' `development`, `scale`, `reserves`, `diagnostics`, `summary`, `draws`),
#' with `exposure` and `prior_ultimate` (`apriori * exposure`, the prior
#' mean of `mu[w]`, not of the expected ultimate) per origin;
#' the posterior draws are of `alpha[1]`, ..., `alpha[n]` (`log mu[w]`)
#' and `beta[1]`, ..., `beta[m]` (`log gamma[d]`). [bayes_loo()] gives the
#' PSIS-LOO of the observed increments, comparable with [bayes_odp()]'s on
#' the same column. `as.data.frame()`, [totals_frame()],
#' [development_frame()] and [segment()] work as for [bayes_odp()], with the
#' `exposure` and `prior_ultimate`.
#'
#' Errors: as [bayes_odp()]; if an origin has no observed, finite, positive
#' exposure; if `apriori` is not positive and finite, `cv` is not positive
#' or `development_sd` is not positive and finite; if a development
#' column's increments do not sum to more than zero; or if an origin's sum
#' to `-phi / cv^2` or less.
#'
#' @param triangle A cumulative [triangle] with one segment, every origin
#'   observed from the first age up to its latest.
#' @param column Name of the loss column to fit.
#' @param exposure Name of the exposure column; each origin's latest
#'   observed value is its exposure, as for [bornhuetter_ferguson()].
#' @param apriori Prior mean of each origin's `mu[w]` per unit of
#'   exposure. It is the expected loss ratio only when the pattern `gamma`
#'   sums to 1 (Details).
#' @param cv Coefficient of variation of the gamma prior on each `mu[w]`
#'   (shape `1 / cv^2`, EVW's `a`); `Inf` for a flat prior.
#' @param development_sd Standard deviation of the normal prior on each
#'   `log gamma[d]`, centred on 0. With `cv = Inf`, a much vaguer prior
#'   (1,000, say) leaves divergent transitions.
#' @param process Process error on each simulated future increment:
#'   `"gamma"` (mean `mu`, variance `phi * mu`) or `"none"` for parameter
#'   error only.
#' @param chains,tune,draws NUTS chains, warm-up draws per chain and kept
#'   draws per chain.
#' @param n_sims Number of predictive simulations, independent of the number
#'   of posterior draws; positive.
#' @param seed Seed of the chains and of the simulations, a non-negative
#'   whole number.
#' @param target_accept Target mean acceptance rate of the step-size
#'   adaptation.
#' @param max_depth Largest NUTS tree depth.
#' @param ptr A `BayesBfFit` pointer; used internally.
#' @returns A `bayes_bf_fit` object.
#' @seealso [bayes_odp()], [bornhuetter_ferguson()], [bayes_loo()].
#' @export
#' @examples
#' long <- data.frame(year = rep(2018:2021, 4:1),
#'                    age = c(12, 24, 36, 48, 12, 24, 36, 12, 24, 12),
#'                    paid = c(100, 150, 165, 170, 110, 170, 180, 120, 175, 130),
#'                    premium = rep(c(200, 220, 240, 260), 4:1))
#' tri <- triangle(long, "year", "age", c("paid", "premium"))
#' fit <- bayes_bf(tri, "paid", "premium", apriori = 0.85, cv = 0.1, n_sims = 2000,
#'                 seed = 1)
#' fit@prior_ultimate
#' mean(fit@reserves)
#' quantile(fit@reserves, 0.995)
bayes_bf_fit <- S7::new_class(
  "bayes_bf_fit",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("BayesBfFit"),
    chain_ladder = chain_ladder_fit,
    origins = S7::new_property(S7::class_character, getter = function(self) {
      self@chain_ladder@origins
    }),
    development = S7::new_property(S7::class_integer, getter = function(self) {
      self@chain_ladder@development
    }),
    scale = S7::new_property(S7::class_double, getter = function(self) self@ptr$scale()),
    exposure = S7::new_property(S7::class_double, getter = function(self) self@ptr$exposure()),
    prior_ultimate = S7::new_property(S7::class_double, getter = function(self) {
      self@ptr$prior_ultimate()
    }),
    reserves = predictive_distribution,
    diagnostics = S7::new_property(S7::class_list, getter = function(self) {
      self@ptr$diagnostics()
    }),
    summary = S7::new_property(S7::class_any, getter = function(self) {
      as.data.frame(rust_result(self@ptr$summary(), call = NULL))
    }),
    draws = S7::new_property(S7::class_any, getter = function(self) {
      nm <- self@ptr$parameter_names()
      matrix(self@ptr$draws(), ncol = length(nm), byrow = TRUE, dimnames = list(NULL, nm))
    })
  ),
  constructor = function(ptr) {
    S7::new_object(S7::S7_object(), ptr = ptr,
                   chain_ladder = chain_ladder_fit(ptr = ptr$chain_ladder()),
                   reserves = predictive_distribution(ptr = ptr$reserves()))
  }
)

#' @rdname bayes_bf_fit
#' @export
bayes_bf <- function(triangle, column, exposure, apriori = 1, cv = 0.1, development_sd = 10,
                     process = c("gamma", "none"), chains = 4, tune = 1000, draws = 1000,
                     n_sims = 10000, seed = 0, target_accept = 0.95, max_depth = 10) {
  cols <- exposure_columns(triangle, column, exposure)
  process <- match.arg(process)
  ptr <- rust_result(triangle@ptr$bayes_bf(
    cols[1], cols[2], single_number(apriori, "apriori"), single_number(cv, "cv"),
    single_number(development_sd, "development_sd"), process, single_number(chains, "chains"),
    single_number(tune, "tune"), single_number(draws, "draws"), single_number(n_sims, "n_sims"),
    single_number(seed, "seed"), single_number(target_accept, "target_accept"),
    single_number(max_depth, "max_depth")))
  fit <- bayes_bf_fit(ptr = ptr)
  failed <- fit@diagnostics$warnings
  if (length(failed)) {
    warning("bayes_bf: the MCMC diagnostics failed; check them before relying on the fit: ",
            paste(failed, collapse = "; "), call. = FALSE)
  }
  fit
}

#' One-year view of the Bayesian Bornhuetter-Ferguson
#'
#' The claims development result (CDR) over the coming year of
#' [bayes_bf()], by Pareto smoothed importance sampling of the posterior
#' (decisions 16 and 17 of `docs/design/reserving-bayes.md`), exactly as
#' [bayes_odp_one_year()] with this model's means `mu[w] * gamma[d]`.
#' `bayes_bf_one_year()` fits the posterior and the lifetime reserves as
#' [bayes_bf()] does with the same settings. Each simulation draws the
#' year's increments from a posterior draw with `process` error, reweights
#' every posterior draw by the year's quasi likelihood (the `scale`, the
#' priors and the prior ultimates unchanged), and takes each origin's CDR as
#' its posterior mean reserve at the opening less its payment in the year
#' and its weighted posterior mean reserve after it; a simulation whose
#' Pareto k is above `k_threshold` is moment-matched and, if that cannot
#' rescue it, refitted with NUTS, as `refit` says.
#'
#' The prior's strength moves the view between two limits. A flat prior
#' (`cv = Inf`) is the Bayesian chain ladder: on GenIns its CDR's
#' standard deviation agrees with [bayes_odp_one_year()]'s within Monte
#' Carlo error. A tight prior fixes each origin's level, so only the
#' development pattern learns from the year: on GenIns at `cv = 0.001`
#' without process error the CDR's standard deviation is the closed form of
#' England, Verrall and Wuthrich's strong-prior limit. In between it
#' narrows as the prior tightens (on GenIns with `apriori = 0.5`, the
#' total's standard deviation at `cv = 0.2` is about 70% of the flat
#' prior's, at `cv = 0.05` about 50%). There is no view over segments:
#' [bayes_bf()] fits one segment.
#'
#' Failed MCMC diagnostics of the fit raise one [warning()] listing them,
#' and refits that failed theirs raise another. The result is a
#' [bayes_one_year_fit] whose `fit` is the [bayes_bf_fit]; its
#' `as.data.frame()` adds each origin's `exposure` and `prior_ultimate`.
#'
#' Errors: as [bayes_bf()]; a `k_threshold` that is `NaN`.
#'
#' This is Python's `BayesBf.one_year()`.
#'
#' @inheritParams bayes_bf_fit
#' @inheritParams bayes_one_year_fit
#' @returns A `bayes_one_year_fit` object.
#' @seealso [bayes_bf()], [bayes_odp_one_year()].
#' @export
#' @examples
#' long <- data.frame(year = rep(2018:2021, 4:1),
#'                    age = c(12, 24, 36, 48, 12, 24, 36, 12, 24, 12),
#'                    paid = c(100, 150, 165, 170, 110, 170, 180, 120, 175, 130),
#'                    premium = rep(c(200, 220, 240, 260), 4:1))
#' tri <- triangle(long, "year", "age", c("paid", "premium"))
#' view <- bayes_bf_one_year(tri, "paid", "premium", apriori = 0.85, cv = 0.1,
#'                           n_sims = 1000, seed = 1, refit = "reweight_only")
#' view@fit@prior_ultimate
#' c(one_year = sqrt(variance(view@cdr)), lifetime = sqrt(variance(view@fit@reserves)))
#' as.data.frame(view)
bayes_bf_one_year <- function(triangle, column, exposure, apriori = 1, cv = 0.1,
                              development_sd = 10, process = c("gamma", "none"), chains = 4,
                              tune = 1000, draws = 1000, n_sims = 10000, seed = 0,
                              target_accept = 0.95, max_depth = 10, k_threshold = 0.7,
                              refit = c("moment_match_then_refit", "refit_flagged",
                                        "reweight_only", "refit_all"),
                              max_refits = NULL, one_year_sims = NULL) {
  cols <- exposure_columns(triangle, column, exposure)
  process <- match.arg(process)
  refit <- match.arg(refit)
  priors <- c(single_number(apriori, "apriori"), single_number(cv, "cv"),
              single_number(development_sd, "development_sd"))
  run <- c(single_number(chains, "chains"), single_number(tune, "tune"),
           single_number(draws, "draws"), single_number(n_sims, "n_sims"),
           single_number(seed, "seed"), single_number(target_accept, "target_accept"),
           single_number(max_depth, "max_depth"))
  ptr <- rust_result(triangle@ptr$bayes_bf_one_year(
    cols[1], cols[2], priors, process, run, single_number(k_threshold, "k_threshold"), refit,
    max_refits_arg(max_refits), optional_number(one_year_sims, "one_year_sims")))
  view <- bayes_one_year_fit(ptr = ptr)
  failed <- view@fit@diagnostics$warnings
  if (length(failed)) {
    warning("bayes_bf_one_year: the MCMC diagnostics failed; check them before relying on the ",
            "fit: ", paste(failed, collapse = "; "), call. = FALSE)
  }
  warn_refits("bayes_bf_one_year", "", length(view@refitted), view@refit_warnings,
              length(view@unrefitted))
  view
}

#' Bayesian Bornhuetter-Ferguson with a normalised pattern
#'
#' The Bayesian Bornhuetter-Ferguson whose prior sits on the ultimate: the
#' over-dispersed Poisson model of a cumulative triangle's incremental
#' values with mean `U[w] * p[d]`, the origin's expected ultimate `U[w]`
#' times the share `p[d]` of it that falls at age `d`, the shares summing
#' to 1 (`p[d] = exp(beta[d]) / sum(exp(beta))` with `beta[1] = 0`). The
#' informative gamma prior is on `U[w]` itself, with mean `apriori` times
#' the origin's exposure and coefficient of variation `cv`, so `apriori` is
#' the expected ultimate loss ratio and `cv` says how firmly it is held, as
#' in the classical [bornhuetter_ferguson()]. The shares have a vague normal
#' prior on their log ratios to the first age's, and the chain ladder's
#' Pearson scale `phi` (the `scale` of [odp_bootstrap()]) is plugged in. The
#' posterior is sampled with NUTS, and each draw projects the future cells
#' up to the oldest observed age (no tail) with the process error of
#' `process`, as [bayes_odp()] does.
#'
#' [bayes_bf()] (England, Verrall and Wuthrich's Model 3.1) instead leaves
#' the pattern unnormalised and puts its prior on `mu[w]`, which is not the
#' ultimate: there the level of `apriori` is cancelled by the pattern and
#' barely moves the reserve. Use this model when the a-priori loss ratio is
#' a judgment about the ultimate; use [bayes_bf()] to match England, Verrall
#' and Wuthrich's published results.
#'
#' The prior's strength moves the reserve between two limits. With
#' `cv = Inf` (a flat prior on `log U[w]`) the model is the Bayesian chain
#' ladder, and the reserves agree with [bayes_odp()]'s within Monte Carlo
#' error. As `cv` falls to 0, `U[w]` is fixed at the prior ultimate `m[w]`
#' and the shares are `S[d] / (M[d] + lambda)` (`S[d]` the observed
#' increments of column `d`, `M[d]` the prior ultimates of the origins
#' observed there, `lambda` such that they sum to 1). When the prior
#' ultimates are the chain ladder's, `lambda` is 0 and the reserve is
#' exactly the classical [bornhuetter_ferguson()] reserve
#' `(1 - 1 / cdf) * m[w]`; otherwise the shares move with the ultimates,
#' where the classical method keeps the chain ladder's pattern.
#'
#' Every development column's observed increments must sum to more than
#' zero, and every origin's to more than `-phi / cv^2`; negative cells are
#' allowed. One `seed` replays a fit exactly, whatever the number of
#' threads. Failed MCMC diagnostics (R-hat above 1.01, bulk or tail ESS
#' below 100 per chain, any divergent transition) do not fail the fit:
#' `bayes_bf_normalised()` raises one [warning()] listing them, and they stay in the
#' fit's `diagnostics`.
#'
#' Properties of the fit: as [bayes_odp_fit]'s (`chain_ladder`, `origins`,
#' `development`, `scale`, `reserves`, `diagnostics`, `summary`, `draws`),
#' with `exposure` and `prior_ultimate` (`apriori * exposure`, the prior
#' mean of the expected ultimate) per origin, `pattern` (the posterior
#' mean of each age's share of the ultimate) and `expected_ultimate` (the
#' posterior mean of each origin's `U[w]`);
#' the posterior draws are of `alpha[1]`, ..., `alpha[n]` (`log U[w]`)
#' and `beta[2]`, ..., `beta[m]` (the log ratios of the shares to the first
#' age's). [bayes_loo()] gives the
#' PSIS-LOO of the observed increments, comparable with [bayes_odp()]'s on
#' the same column. `as.data.frame()`, [totals_frame()],
#' [development_frame()] and [segment()] work as for [bayes_odp()], with the
#' `exposure` and `prior_ultimate`.
#'
#' Errors: as [bayes_odp()]; if an origin has no observed, finite, positive
#' exposure; if `apriori` is not positive and finite, `cv` is not positive
#' or `development_sd` is not positive and finite; if a development
#' column's increments do not sum to more than zero; or if an origin's sum
#' to `-phi / cv^2` or less.
#'
#' @param triangle A cumulative [triangle] with one segment, every origin
#'   observed from the first age up to its latest.
#' @param column Name of the loss column to fit.
#' @param exposure Name of the exposure column; each origin's latest
#'   observed value is its exposure, as for [bornhuetter_ferguson()].
#' @param apriori Prior mean of each origin's expected ultimate `U[w]` per
#'   unit of exposure: the expected ultimate loss ratio.
#' @param cv Coefficient of variation of the gamma prior on each `U[w]`
#'   (shape `1 / cv^2`); `Inf` for a flat prior.
#' @param development_sd Standard deviation of the normal prior on each
#'   `beta[d]` (the log ratio of the share at age `d` to the first age's),
#'   centred on 0.
#' @param process Process error on each simulated future increment:
#'   `"gamma"` (mean `mu`, variance `phi * mu`) or `"none"` for parameter
#'   error only.
#' @param chains,tune,draws NUTS chains, warm-up draws per chain and kept
#'   draws per chain.
#' @param n_sims Number of predictive simulations, independent of the number
#'   of posterior draws; positive.
#' @param seed Seed of the chains and of the simulations, a non-negative
#'   whole number.
#' @param target_accept Target mean acceptance rate of the step-size
#'   adaptation.
#' @param max_depth Largest NUTS tree depth.
#' @param ptr A `BayesBfNormalisedFit` pointer; used internally.
#' @returns A `bayes_bf_normalised_fit` object.
#' @seealso [bayes_bf()], [bayes_odp()], [bornhuetter_ferguson()], [bayes_loo()].
#' @export
#' @examples
#' long <- data.frame(year = rep(2018:2021, 4:1),
#'                    age = c(12, 24, 36, 48, 12, 24, 36, 12, 24, 12),
#'                    paid = c(100, 150, 165, 170, 110, 170, 180, 120, 175, 130),
#'                    premium = rep(c(200, 220, 240, 260), 4:1))
#' tri <- triangle(long, "year", "age", c("paid", "premium"))
#' fit <- bayes_bf_normalised(tri, "paid", "premium", apriori = 0.85, cv = 0.1, n_sims = 2000,
#'                 seed = 1)
#' fit@prior_ultimate
#' fit@pattern
#' mean(fit@reserves)
#' quantile(fit@reserves, 0.995)
bayes_bf_normalised_fit <- S7::new_class(
  "bayes_bf_normalised_fit",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("BayesBfNormalisedFit"),
    chain_ladder = chain_ladder_fit,
    origins = S7::new_property(S7::class_character, getter = function(self) {
      self@chain_ladder@origins
    }),
    development = S7::new_property(S7::class_integer, getter = function(self) {
      self@chain_ladder@development
    }),
    scale = S7::new_property(S7::class_double, getter = function(self) self@ptr$scale()),
    exposure = S7::new_property(S7::class_double, getter = function(self) self@ptr$exposure()),
    prior_ultimate = S7::new_property(S7::class_double, getter = function(self) {
      self@ptr$prior_ultimate()
    }),
    pattern = S7::new_property(S7::class_double, getter = function(self) self@ptr$pattern()),
    expected_ultimate = S7::new_property(S7::class_double, getter = function(self) {
      self@ptr$expected_ultimate()
    }),
    reserves = predictive_distribution,
    diagnostics = S7::new_property(S7::class_list, getter = function(self) {
      self@ptr$diagnostics()
    }),
    summary = S7::new_property(S7::class_any, getter = function(self) {
      as.data.frame(rust_result(self@ptr$summary(), call = NULL))
    }),
    draws = S7::new_property(S7::class_any, getter = function(self) {
      nm <- self@ptr$parameter_names()
      matrix(self@ptr$draws(), ncol = length(nm), byrow = TRUE, dimnames = list(NULL, nm))
    })
  ),
  constructor = function(ptr) {
    S7::new_object(S7::S7_object(), ptr = ptr,
                   chain_ladder = chain_ladder_fit(ptr = ptr$chain_ladder()),
                   reserves = predictive_distribution(ptr = ptr$reserves()))
  }
)

#' @rdname bayes_bf_normalised_fit
#' @export
bayes_bf_normalised <- function(triangle, column, exposure, apriori = 1, cv = 0.1, development_sd = 10,
                     process = c("gamma", "none"), chains = 4, tune = 1000, draws = 1000,
                     n_sims = 10000, seed = 0, target_accept = 0.95, max_depth = 10,
                     common_cv = 0) {
  cols <- exposure_columns(triangle, column, exposure)
  process <- match.arg(process)
  ptr <- rust_result(triangle@ptr$bayes_bf_normalised(
    cols[1], cols[2], single_number(apriori, "apriori"), single_number(cv, "cv"),
    single_number(development_sd, "development_sd"), process, single_number(chains, "chains"),
    single_number(tune, "tune"), single_number(draws, "draws"), single_number(n_sims, "n_sims"),
    single_number(seed, "seed"), single_number(target_accept, "target_accept"),
    single_number(max_depth, "max_depth"), single_number(common_cv, "common_cv")))
  fit <- bayes_bf_normalised_fit(ptr = ptr)
  failed <- fit@diagnostics$warnings
  if (length(failed)) {
    warning("bayes_bf_normalised: the MCMC diagnostics failed; check them before relying on the fit: ",
            paste(failed, collapse = "; "), call. = FALSE)
  }
  fit
}

#' One-year view of the Bayesian Bornhuetter-Ferguson
#'
#' The claims development result (CDR) over the coming year of
#' [bayes_bf_normalised()], by Pareto smoothed importance sampling of the posterior
#' (decisions 16 and 17 of `docs/design/reserving-bayes.md`), exactly as
#' [bayes_odp_one_year()] with this model's means `exp(alpha[w]) * p[d]`.
#' `bayes_bf_normalised_one_year()` fits the posterior and the lifetime reserves as
#' [bayes_bf_normalised()] does with the same settings. Each simulation draws the
#' year's increments from a posterior draw with `process` error, reweights
#' every posterior draw by the year's quasi likelihood (the `scale`, the
#' priors and the prior ultimates unchanged), and takes each origin's CDR as
#' its posterior mean reserve at the opening less its payment in the year
#' and its weighted posterior mean reserve after it; a simulation whose
#' Pareto k is above `k_threshold` is moment-matched and, if that cannot
#' rescue it, refitted with NUTS, as `refit` says.
#'
#' The prior's strength moves the view between two limits. A flat prior
#' (`cv = Inf`) is the Bayesian chain ladder: on GenIns its CDR's
#' standard deviation agrees with [bayes_odp_one_year()]'s within Monte
#' Carlo error. A tighter prior holds each origin's ultimate more firmly,
#' so less of the year's information goes to the ultimates and the view
#' narrows (on GenIns with `apriori = 0.5`, the total's standard deviation
#' at `cv = 0.2` is about 58% of the flat prior's, at `cv = 0.05` about
#' 32%). There is no view over segments: [bayes_bf_normalised()] fits one
#' segment.
#'
#' Failed MCMC diagnostics of the fit raise one [warning()] listing them,
#' and refits that failed theirs raise another. The result is a
#' [bayes_one_year_fit] whose `fit` is the [bayes_bf_normalised_fit]; its
#' `as.data.frame()` adds each origin's `exposure` and `prior_ultimate`.
#'
#' Errors: as [bayes_bf_normalised()]; a `k_threshold` that is `NaN`.
#'
#' This is Python's `BayesBfNormalised.one_year()`.
#'
#' @inheritParams bayes_bf_normalised_fit
#' @inheritParams bayes_one_year_fit
#' @returns A `bayes_one_year_fit` object.
#' @seealso [bayes_bf_normalised()], [bayes_odp_one_year()].
#' @export
#' @examples
#' long <- data.frame(year = rep(2018:2021, 4:1),
#'                    age = c(12, 24, 36, 48, 12, 24, 36, 12, 24, 12),
#'                    paid = c(100, 150, 165, 170, 110, 170, 180, 120, 175, 130),
#'                    premium = rep(c(200, 220, 240, 260), 4:1))
#' tri <- triangle(long, "year", "age", c("paid", "premium"))
#' view <- bayes_bf_normalised_one_year(tri, "paid", "premium", apriori = 0.85, cv = 0.1,
#'                           n_sims = 1000, seed = 1, refit = "reweight_only")
#' view@fit@prior_ultimate
#' c(one_year = sqrt(variance(view@cdr)), lifetime = sqrt(variance(view@fit@reserves)))
#' as.data.frame(view)
bayes_bf_normalised_one_year <- function(triangle, column, exposure, apriori = 1, cv = 0.1,
                              development_sd = 10, process = c("gamma", "none"), chains = 4,
                              tune = 1000, draws = 1000, n_sims = 10000, seed = 0,
                              target_accept = 0.95, max_depth = 10, k_threshold = 0.7,
                              refit = c("moment_match_then_refit", "refit_flagged",
                                        "reweight_only", "refit_all"),
                              max_refits = NULL, one_year_sims = NULL, common_cv = 0) {
  cols <- exposure_columns(triangle, column, exposure)
  process <- match.arg(process)
  refit <- match.arg(refit)
  priors <- c(single_number(apriori, "apriori"), single_number(cv, "cv"),
              single_number(development_sd, "development_sd"),
              single_number(common_cv, "common_cv"))
  run <- c(single_number(chains, "chains"), single_number(tune, "tune"),
           single_number(draws, "draws"), single_number(n_sims, "n_sims"),
           single_number(seed, "seed"), single_number(target_accept, "target_accept"),
           single_number(max_depth, "max_depth"))
  ptr <- rust_result(triangle@ptr$bayes_bf_normalised_one_year(
    cols[1], cols[2], priors, process, run, single_number(k_threshold, "k_threshold"), refit,
    max_refits_arg(max_refits), optional_number(one_year_sims, "one_year_sims")))
  view <- bayes_one_year_fit(ptr = ptr)
  failed <- view@fit@diagnostics$warnings
  if (length(failed)) {
    warning("bayes_bf_normalised_one_year: the MCMC diagnostics failed; check them before relying on the ",
            "fit: ", paste(failed, collapse = "; "), call. = FALSE)
  }
  warn_refits("bayes_bf_normalised_one_year", "", length(view@refitted), view@refit_warnings,
              length(view@unrefitted))
  view
}

#' Bootstrap of Mack's model
#'
#' The lifetime view of reserve risk under Mack's model, bootstrapped as
#' England, Verrall and Wuthrich (2019, Appendix 1) describe: the scaled
#' bias-adjusted residuals of the link ratios
#' `sqrt(n_k / (n_k - 1)) C^(alpha / 2) (F - f_k) / sigma_k` are resampled
#' into pseudo link ratios and re-averaged into pseudo factors (parameter
#' error), and every future cumulative value is drawn from the one before,
#' the observed latest value for the first, with mean `f*_k C` and Mack's
#' variance `sigma_k^2 |C|^(2 - alpha)` (process error), to the triangle's
#' last age. An origin's reserve is its last drawn value less its latest.
#' With a `tail` every origin takes one more step, from the oldest age to
#' ultimate, with mean the tail factor times the value at the oldest age and
#' variance `tail_sigma^2 |C|^(2 - alpha)` (Mack 1999); the tail factor is
#' refitted on each simulation's pseudo factors (an estimated tail) or drawn
#' from the lognormal with the factor as mean and the tail's standard error
#' (a constant one). A constant tail attached before the oldest age moves
#' each factor it replaces by the pseudo factor's deviation from the
#' estimate, which [mack()] charges with the estimated factor's standard
#' error. With a constant tail the standard deviation approximates
#' [mack()]'s with the same tail; for an estimated tail the parameter error
#' and the mean are the refit's, not Mack's extrapolated `tail.se` and the
#' chain ladder's reserve.
#'
#' The standard deviation of the reserves approximates Mack's analytic
#' standard error ([mack()]), and the mean is the chain ladder's reserve
#' (without a tail or with a constant one): the pooled residuals do not have
#' a zero mean, so by default (`centre_residuals = TRUE`) they are centred
#' before resampling.
#' Resampled as they are (`centre_residuals = FALSE`, EVW's Appendix 1 as
#' written) they bias every pseudo factor, and the mean reserve with them
#' (about 17% above the chain ladder's on RAA, 0.7% on GenIns, 0.8% below
#' on ABC); EVW's Table 4 expected reserves agree with the centred
#' bootstrap.
#'
#' Every segment is bootstrapped with its own Mack model and residuals,
#' into one joint distribution of the reserves, the segments depending on
#' each other as `dependence` says, as in [odp_bootstrap()]; synchronized,
#' every segment resamples the link-ratio residuals of the same origins and
#' ages. Simulation `i` uses random stream `i` of `seed` for every segment
#' in turn.
#'
#' Properties of the fit: `chain_ladder` (the [chain_ladder_fit] of Mack's
#' averaging), `mack` (the [mack_fit] on the observed triangle, with the
#' analytic standard errors), `origins`, `development`, `residuals` (an
#' origin x development matrix, `[o, k]` the residual of the link from age
#' `k` to `k + 1`, `NA` where there is none; never centred; single segment
#' only) and `reserves`, a [predictive_distribution] of the reserve with the
#' triangle's keys and `origin` as dimensions.
#'
#' This is Python's `MackBootstrap(...).fit()`, which returns a
#' `MackBootstrapFit`.
#'
#' @param triangle A cumulative [triangle], with any number of segments and
#'   any development grain, every origin observed from the first age up to
#'   its latest, with no negative value.
#' @param column Name of the column to fit; by default the only one.
#' @param n_sims Number of simulations; positive.
#' @param seed Seed of the simulation streams, a non-negative whole number.
#' @param process Process error on each next cumulative value: `"gamma"` or
#'   `"lognormal"` (negated for a negative mean) or `"normal"`, with Mack's
#'   mean and variance; `"residuals"` (the mean plus a resampled residual
#'   times the standard deviation); or `"none"` for parameter error only.
#' @param average,sigma_interpolation How Mack's model averages the link
#'   ratios and fills in a sigma behind a single link ratio, as in [mack()].
#' @param centre_residuals Subtract the residuals' mean before resampling
#'   them, so that the pseudo factors are unbiased and the mean reserve is
#'   the chain ladder's (without a tail or with a constant one); `FALSE`
#'   resamples them uncentred, as England,
#'   Verrall and Wuthrich's Appendix 1 is written.
#' @param tail Development past the oldest age, as in [mack()]: a number (a
#'   constant factor), [tail_constant()], [tail_curve()], [tail_bondy()] or
#'   [tail_log_linear()]; 1 (no tail) by default. [mack_one_year()] takes
#'   it with `bootstrap_tail = TRUE`.
#' @param tail_sigma The tail's sigma (R's `tail.sigma`), the process error
#'   of the step to ultimate, or `NULL` to extrapolate it. Unused when the
#'   tail factor is 1.
#' @param tail_std_err A constant tail factor's standard error (R's
#'   `tail.se`), or `NULL` to extrapolate it (unlike [odp_bootstrap()],
#'   which then keeps the factor fixed). Unused when the tail factor is 1,
#'   and an error with an estimated tail.
#' @param dependence,spearman How the segments depend on each other, as in
#'   [odp_bootstrap()].
#' @param ldf,ldf_kind,ldf_weight A selected development pattern, as in
#'   [chain_ladder()]. Mack's sigmas and residuals come from the triangle's
#'   own estimate; each simulation then uses `w * selected + (1 - w) *
#'   pseudo` at the selected ages, so a fully selected factor has no
#'   parameter error and a blended one `1 - w` times the pseudo factor's
#'   deviation, as [mack()] charges it; the process error is unchanged. A
#'   supplied pattern lowers the reported uncertainty. Each origin is
#'   projected from its observed latest value, so with every factor
#'   selected only the process error is left (unlike [odp_bootstrap()],
#'   which keeps the origins' level error). With no `tail`, a cumulative
#'   pattern to the oldest age implies a constant tail (`pattern_tail` of
#'   [chain_ladder()]), the same in every simulation unless `tail_std_err`
#'   is given.
#' @param n_periods,drop_high,drop_low Which link ratios the factors average,
#'   as in [chain_ladder()]. Only the link ratios that the observed
#'   triangle keeps have residuals and enter the pseudo factors (the drops
#'   are not ranked again in each simulation), so [mack()]'s standard error
#'   under the same settings is the reference for the standard deviation.
#'   `"geometric"` as the average is rejected.
#' @param ptr A `MackBootstrapFit` pointer; used internally.
#' @returns A `mack_bootstrap_fit` object.
#' @seealso [mack()] for the analytic standard errors, [mack_one_year()]
#'   for the one-year view on the same bootstrap, [odp_bootstrap()].
#' @export
#' @examples
#' long <- data.frame(year = rep(2018:2021, 4:1),
#'                    age = c(12, 24, 36, 48, 12, 24, 36, 12, 24, 12),
#'                    paid = c(100, 150, 165, 170, 110, 170, 180, 120, 175, 130))
#' boot <- mack_bootstrap(triangle(long, "year", "age", "paid"), n_sims = 2000, seed = 42)
#' boot@reserves@keys
#' c(mean = mean(boot@reserves), chain_ladder = boot@chain_ladder@total_reserve)
#' c(sd = sqrt(variance(boot@reserves)), mack = boot@mack@total_standard_error)
#' as.data.frame(boot)
mack_bootstrap_fit <- S7::new_class(
  "mack_bootstrap_fit",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("MackBootstrapFit"),
    chain_ladder = chain_ladder_fit,
    mack = S7::new_property(S7::class_any, getter = function(self) {
      mack_fit(ptr = self@ptr$mack())
    }),
    origins = S7::new_property(S7::class_character, getter = function(self) {
      self@chain_ladder@origins
    }),
    development = S7::new_property(S7::class_integer, getter = function(self) {
      self@chain_ladder@development
    }),
    residuals = S7::new_property(S7::class_double, getter = function(self) {
      origin_matrix(self, rust_result(self@ptr$residuals(), call = NULL))
    }),
    reserves = predictive_distribution
  ),
  constructor = function(ptr) {
    S7::new_object(S7::S7_object(), ptr = ptr,
                   chain_ladder = chain_ladder_fit(ptr = ptr$chain_ladder()),
                   reserves = predictive_distribution(ptr = ptr$reserves()))
  }
)

#' @rdname mack_bootstrap_fit
#' @export
mack_bootstrap <- function(triangle, column = NULL, n_sims = 10000, seed = 0,
                           process = c("gamma", "lognormal", "residuals", "normal", "none"),
                           average = "volume", sigma_interpolation = "log-linear",
                           centre_residuals = TRUE, tail = 1, tail_sigma = NULL,
                           tail_std_err = NULL,
                           dependence = c("independent", "synchronized", "rank_correlation"),
                           spearman = NULL, ldf = NULL, ldf_kind = NULL, ldf_weight = 1,
                           n_periods = NULL, drop_high = 0, drop_low = 0) {
  column <- fit_column(triangle, column)
  process <- match.arg(process)
  dependence <- match.arg(dependence)
  model <- development_args(average, sigma_interpolation, n_periods, drop_high, drop_low)
  ptr <- rust_result(triangle@ptr$mack_bootstrap(
    column, single_number(n_sims, "n_sims"), single_number(seed, "seed"), process,
    model$average, model$sigma, model$excl, isTRUE(centre_residuals), tail_ptr(tail),
    optional_number(tail_sigma, "tail_sigma"), optional_number(tail_std_err, "tail_std_err"),
    dependence, spearman_arg(spearman), ldf_pattern(ldf, ldf_kind, ldf_weight)
  ))
  mack_bootstrap_fit(ptr = ptr)
}

#' Simulated one-year view
#'
#' The claims development result (CDR) of the chain ladder or an
#' expected-loss method over the coming year, by re-reserving on the ODP
#' bootstrap ("actuary in the box": Ohlsson and Lauzeningks 2009; England,
#' Verrall and Wuthrich 2019). The coming year is every cell valued in the
#' twelve months after the segment's valuation: one per origin for an
#' annual development grain, four for a quarterly one (fewer for an origin
#' that reaches the last age). Each simulation resamples the residuals of
#' the volume-weighted chain ladder as [odp_bootstrap()] does, projects the
#' increments of those cells in turn from the origin's resampled latest
#' value with the bootstrap's process error, as [odp_bootstrap()] projects,
#' adds them to the observed latest value, appends the cells to the
#' triangle, refits `method` and records `CDR = opening ultimate - closing
#' ultimate`, so a negative CDR is an adverse development. An origin whose
#' remaining cells all fall in the year thus has its lifetime bootstrap
#' reserve as its one-year view. An origin at the last age gets no new
#' cell, an origin short of the latest diagonal develops from its own
#' latest cell (only the year's cells are appended), and a new origin
#' written in the coming year is not simulated. Unlike
#' [claims_development_result()] (Merz and Wuthrich), any averaging, tail
#' and development grain are allowed.
#'
#' `method` names one of [chain_ladder()], [expected_loss()],
#' [bornhuetter_ferguson()], [benktander()] and [cape_cod()], with the same
#' settings: `average`, `sigma_interpolation` and `tail` give the chain
#' ladder refitted (or the development pattern of the other methods),
#' `apriori` is read by the expected loss, Bornhuetter-Ferguson and
#' Benktander methods, `n_iters` by Benktander, `trend` and `decay` by Cape
#' Cod; giving a setting the method does not read is an error. The
#' expected-loss methods need `exposure`, the chain ladder takes none; each
#' origin's latest exposure is kept for the end of the year, and Cape Cod
#' trends to the valuation a year later.
#'
#' The bootstrap only drives the simulation: its factors are always
#' volume-weighted without a tail, whatever `method` refits. On that same
#' chain ladder the standard deviation of the CDR is not Merz and
#' Wuthrich's: the ODP's process variance is the scale times the mean,
#' Mack's `sigma^2` times the cumulative value. In total it is 0.62 times
#' Merz-Wuthrich on RAA, 1.37 on GenIns and 1.14 on ABC, and per origin
#' from 0.50 to 5.96 times
#' (`knowledge/findings/one-year-bootstrap-vs-merz-wuthrich.md`).
#'
#' `mack_one_year()` re-reserves the same way under Mack's process instead,
#' England, Verrall and Wuthrich's (2019, Appendix 1) bootstrap of Mack's
#' model: each simulation resamples the scaled bias-adjusted residuals of
#' the link ratios into pseudo factors, averaged as `mack_average`, and
#' draws each cumulative value of the coming year from the one before `C`
#' (the observed latest value for the first) with mean `f* C` and variance
#' `sigma^2 |C|^(2 - alpha)`, with the same pseudo factors all year. With the
#' volume-weighted chain ladder and no tail, its standard deviations are
#' Merz and Wuthrich's ([claims_development_result()]) within Monte Carlo
#' error, which reconciles the two, and its mean is Merz and Wuthrich's
#' zero, because by default (`centre_residuals = TRUE`) the pool of
#' residuals is centred first. Uncentred (`centre_residuals = FALSE`, EVW's
#' Appendix 1 as written), the pool's non-zero mean biases the pseudo
#' factors: the mean CDR is about -0.2 (RAA), -0.04 (GenIns) and +0.18
#' (ABC) times its standard deviation, and RAA's standard deviations up to
#' 1.3% wide. Its fit has `model = "mack"`, no `scale`, and `mack`, the
#' [mack_fit] it simulates from; it needs no negative cumulative value.
#'
#' By default the bootstrap has no tail of its own: its simulation stops at
#' the oldest age, and the development past it moves only through
#' `method`'s refitted `tail`. With `bootstrap_tail = TRUE` the bootstrap
#' takes `method`'s `tail` too, drawn in each simulation as in
#' [odp_bootstrap()] or [mack_bootstrap()] (an estimated tail refitted on
#' the pseudo factors, a constant one fixed or drawn with `tail_std_err`),
#' and its step to ultimate is valued one development period after the
#' oldest age: an origin whose step falls in the coming year (with an
#' annual grain, one already at the oldest age) takes it and is closed at
#' that simulated ultimate. The step is the whole tail: a curve or a decayed
#' constant tail is taken as the product of its factors in that one step,
#' so an origin that takes it has its lifetime tail risk in the coming
#' year, an upper bound on its one-year risk. With a constant tail attached
#' at the oldest age and Mack's process, the other origins' CDR is the tail
#' factor times its value without one in distribution (simulation by
#' simulation when the factor is fixed, `tail_std_err = 0`), and the oldest
#' origin's standard deviation is [mack()]'s with the tail.
#'
#' Every segment of the triangle is bootstrapped with its own residuals and
#' scale, into one joint distribution of the CDR, the segments depending on
#' each other as `dependence` says (as in [odp_bootstrap()]); simulation `i`
#' uses random stream `i` of `seed` for every segment in turn, so results do
#' not depend on the number of threads.
#'
#' Properties of the fit: `chain_ladder` (the bootstrap's volume-weighted
#' [chain_ladder_fit]), `keys`, `index`, `origins`, `development`; per
#' origin (named as a [chain_ladder_fit]'s) `latest`, `opening_ultimate`
#' (the method's ultimate on the observed triangle) and `opening_reserve`
#' (`opening_ultimate - latest`); `scale` (the bootstrap's `phi`, which
#' needs a single-segment fit: use [totals_frame()] or [segment()]); and
#' `cdr`, a [predictive_distribution] of the CDR with the triangle's keys
#' and `origin` as dimensions and one component per segment and origin, so
#' `aggregate(fit@cdr, keep = "lob")` keeps the dependence between
#' segments. `mean()`, `quantile()`, [VaR()] and [TVaR()] of `cdr` describe
#' the total, and `-quantile(fit@cdr, 0.005)` is the one-year loss at
#' 99.5%. Columns of [draw_matrix()] follow `origins`.
#' `as.data.frame()` has one row per segment and origin: `origin`,
#' `latest`, `opening_ultimate`, `opening_reserve`, and the `cdr_mean` and
#' `cdr_std_dev` of the simulated CDR; [totals_frame()] one per segment,
#' with the ODP bootstrap's `scale` too. `model` is `"odp"` or `"mack"`.
#'
#' Errors: as [odp_bootstrap()] (or [mack()] for `mack_one_year()`) and the
#' method's own fit; a missing `exposure` for an expected-loss method or one
#' given for the chain ladder; or a refit that fails in any simulation (a
#' zero value under a simple average, say), counted in the message with one
#' of the failures.
#'
#' These are Python's `OdpBootstrap.one_year()` and
#' `MackBootstrap.one_year()`, which return a `OneYearFit`.
#'
#' @inheritParams odp_bootstrap_fit
#' @param process Process error. For `odp_one_year()`, on each simulated
#'   incremental value: `"gamma"` (mean the expected value, variance
#'   `scale * |mean|`) or `"none"` for parameter error only. For
#'   `mack_one_year()`, on each next cumulative value: `"gamma"` or
#'   `"lognormal"` (negated for a negative mean) or `"normal"`, with Mack's
#'   mean and variance; `"residuals"` (the mean plus a resampled residual
#'   times the standard deviation, which carries the residuals' mean and
#'   variance); or `"none"`.
#' @param mack_average,mack_sigma_interpolation How Mack's model in
#'   `mack_one_year()` averages the link ratios and fills in a sigma behind
#'   a single link ratio, as in [mack()].
#' @param centre_residuals For `mack_one_year()`, subtract the residuals'
#'   mean before resampling them, so that the pseudo factors are unbiased
#'   and the mean CDR is about zero (the default); `FALSE` resamples them
#'   uncentred, as England, Verrall and Wuthrich's Appendix 1 is written.
#' @param dependence,spearman How the segments depend on each other, as in
#'   [odp_bootstrap()].
#' @param column Name of the loss column; by default the only one.
#' @param method The reserving method refitted at the start and at the end
#'   of the year: `"chain_ladder"`, `"expected_loss"`,
#'   `"bornhuetter_ferguson"`, `"benktander"` or `"cape_cod"`.
#' @param exposure Name of the exposure column for the expected-loss
#'   methods; `NULL` for the chain ladder.
#' @param apriori,n_iters As in [expected_loss_fit()].
#' @param trend,decay As in [cape_cod()].
#' @param average,sigma_interpolation,tail The chain ladder or development
#'   pattern of `method`, as in [chain_ladder()].
#' @param ldf,ldf_kind,ldf_weight A selected development pattern of
#'   `method`, as in [chain_ladder()].
#' @param bootstrap_ldf Whether the bootstrap takes `method`'s selected
#'   pattern too (the default), so that its simulations blend the pseudo
#'   factors with it as [odp_bootstrap()] and [mack_bootstrap()] do;
#'   `FALSE` simulates from the triangle's own factors and applies the
#'   pattern only when `method` is refitted. With the default, a cumulative
#'   pattern to the oldest age and no `tail` implies a tail (its level
#'   there, [chain_ladder()]): the bootstrap develops past the oldest age
#'   with it in the coming year, as with `bootstrap_tail`, and closes the
#'   origins it falls on. With a `tail`, `method` rescales the pattern's
#'   level to it, and the bootstrap reads the pattern the same way: it
#'   implies no tail, and with `bootstrap_tail = FALSE` the bootstrap has
#'   none.
#' @param n_periods,drop_high,drop_low Which link ratios the factors of
#'   `method` average, as in [chain_ladder()]. The bootstrap simulates the
#'   coming year with the same link ratios the observed triangle keeps, and
#'   `method` is fitted again on the triangle with the coming diagonal, so
#'   the window moves by one diagonal and the drops are ranked again, as
#'   next year's reserving does. The mean of the CDR is therefore not zero
#'   under exclusions. `"geometric"` as `mack_average` is rejected.
#' @param mack_n_periods,mack_drop_high,mack_drop_low Which link ratios Mack's
#'   model in `mack_one_year()` averages, as `n_periods`, `drop_high` and
#'   `drop_low` in [chain_ladder()]. They must be the same as `method`'s
#'   (`n_periods`, `drop_high`, `drop_low`) or left at their defaults, which
#'   take `method`'s; any other combination is an error naming both.
#' @param bootstrap_tail Whether the bootstrap takes `method`'s `tail` too,
#'   developing past the oldest age in the coming year (above); `FALSE` by
#'   default.
#' @param tail_sigma For `mack_one_year()` with `bootstrap_tail`, the tail's
#'   sigma (R's `tail.sigma`), or `NULL` to extrapolate it, as in
#'   [mack_bootstrap()]; an error without `bootstrap_tail`.
#' @param tail_std_err With `bootstrap_tail`, a constant tail factor's
#'   standard error: `NULL` keeps it fixed in `odp_one_year()` and
#'   extrapolates it in `mack_one_year()`, as in [odp_bootstrap()] and
#'   [mack_bootstrap()]; an error with an estimated tail or without
#'   `bootstrap_tail`.
#' @param ptr A `OneYearFit` pointer; used internally.
#' @returns A `one_year_fit` object.
#' @seealso [claims_development_result()] for Merz and Wuthrich's formulas,
#'   [odp_bootstrap()] and [mack_bootstrap()] for the lifetime view.
#' @export
#' @examples
#' long <- data.frame(year = rep(2018:2021, 4:1),
#'                    age = c(12, 24, 36, 48, 12, 24, 36, 12, 24, 12),
#'                    paid = c(100, 150, 165, 170, 110, 170, 180, 120, 175, 130),
#'                    premium = rep(c(250, 260, 270, 280), 4:1))
#' tri <- triangle(long, "year", "age", c("paid", "premium"))
#' cl <- odp_one_year(tri, "paid", n_sims = 2000, seed = 42)
#' cl@opening_reserve
#' mean(cl@cdr)
#' # The one-year view is narrower than the lifetime view.
#' c(one_year = sqrt(variance(cl@cdr)),
#'   lifetime = sqrt(variance(odp_bootstrap(tri, "paid", n_sims = 2000, seed = 42)@reserves)))
#'
#' bf <- odp_one_year(tri, "paid", "bornhuetter_ferguson", exposure = "premium", apriori = 0.7,
#'                    n_sims = 2000, seed = 42)
#' as.data.frame(bf)
#' -quantile(bf@cdr, 0.005)
#'
#' # Under Mack's process the chain ladder's one-year view is Merz and
#' # Wuthrich's, up to Monte Carlo error.
#' mk <- mack_one_year(tri, "paid", n_sims = 2000, seed = 42)
#' mk@model
#' c(simulated = sqrt(variance(mk@cdr)),
#'   merz_wuthrich = claims_development_result(mk@mack)@total_one_year_standard_error)
#'
#' # The bootstrap's own tail moves the oldest origin in the coming year.
#' tailed <- mack_one_year(tri, "paid", tail = 1.05, bootstrap_tail = TRUE, n_sims = 2000,
#'                         seed = 42)
#' as.data.frame(tailed)[1, c("origin", "cdr_std_dev")]
one_year_fit <- S7::new_class(
  "one_year_fit",
  package = "prospicio",
  properties = local({
    by_origin <- function(f) {
      S7::new_property(S7::class_double, getter = function(self) {
        stats::setNames(f(self@ptr), origin_names(self@chain_ladder@ptr))
      })
    }
    list(
      ptr = S7::new_S3_class("OneYearFit"),
      model = S7::new_property(S7::class_character, getter = function(self) self@ptr$model()),
      chain_ladder = chain_ladder_fit,
      mack = S7::new_property(S7::class_any, getter = function(self) {
        mack_fit(ptr = rust_result(self@ptr$mack(), call = NULL))
      }),
      keys = S7::new_property(S7::class_character, getter = function(self) self@chain_ladder@keys),
      index = S7::new_property(S7::class_data.frame, getter = function(self) self@chain_ladder@index),
      origins = S7::new_property(S7::class_character, getter = function(self) self@chain_ladder@origins),
      development = S7::new_property(S7::class_integer, getter = function(self) {
        self@chain_ladder@development
      }),
      latest = by_origin(function(p) p$chain_ladder()$latest()),
      opening_ultimate = by_origin(function(p) p$opening_ultimate()),
      opening_reserve = by_origin(function(p) p$opening_reserve()),
      scale = S7::new_property(S7::class_double, getter = function(self) {
        rust_result(self@ptr$scale(), call = NULL)
      }),
      cdr = predictive_distribution
    )
  }),
  constructor = function(ptr) {
    S7::new_object(S7::S7_object(), ptr = ptr,
                   chain_ladder = chain_ladder_fit(ptr = ptr$chain_ladder()),
                   cdr = predictive_distribution(ptr = ptr$cdr()))
  }
)

#' @rdname one_year_fit
#' @export
odp_one_year <- function(triangle, column = NULL,
                         method = c("chain_ladder", "expected_loss", "bornhuetter_ferguson",
                                    "benktander", "cape_cod"),
                         exposure = NULL, apriori = 1, n_iters = 1, trend = 0, decay = 1,
                         average = "volume", sigma_interpolation = "log-linear",
                         tail = 1,
                         bootstrap_tail = FALSE, tail_std_err = NULL,
                         n_sims = 10000, seed = 0, process = c("gamma", "none"),
                         dependence = c("independent", "synchronized", "rank_correlation"),
                         spearman = NULL, ldf = NULL, ldf_kind = NULL, ldf_weight = 1,
                         bootstrap_ldf = TRUE, n_periods = NULL, drop_high = 0, drop_low = 0) {
  column <- fit_column(triangle, column)
  method <- match.arg(method)
  process <- match.arg(process)
  dependence <- match.arg(dependence)
  check_one_year_settings(method, exposure, c(apriori = !missing(apriori),
                                              n_iters = !missing(n_iters),
                                              trend = !missing(trend), decay = !missing(decay)))
  check_bootstrap_tail_settings(bootstrap_tail, list(tail_std_err = tail_std_err))
  args <- development_args(average, sigma_interpolation, n_periods, drop_high, drop_low)
  ptr <- rust_result(triangle@ptr$odp_one_year(
    column, method, exposure, single_number(apriori, "apriori"), single_number(n_iters, "n_iters"),
    single_number(trend, "trend"), single_number(decay, "decay"), args$average, args$sigma, args$excl,
    tail_ptr(tail), isTRUE(bootstrap_tail),
    optional_number(tail_std_err, "tail_std_err"), single_number(n_sims, "n_sims"),
    single_number(seed, "seed"), process, dependence, spearman_arg(spearman),
    ldf_pattern(ldf, ldf_kind, ldf_weight), isTRUE(bootstrap_ldf)
  ))
  one_year_fit(ptr = ptr)
}

#' @rdname one_year_fit
#' @export
mack_one_year <- function(triangle, column = NULL,
                          method = c("chain_ladder", "expected_loss", "bornhuetter_ferguson",
                                     "benktander", "cape_cod"),
                          exposure = NULL, apriori = 1, n_iters = 1, trend = 0, decay = 1,
                          average = "volume", sigma_interpolation = "log-linear",
                          tail = 1,
                          bootstrap_tail = FALSE, tail_sigma = NULL, tail_std_err = NULL,
                          n_sims = 10000, seed = 0,
                          process = c("gamma", "lognormal", "residuals", "normal", "none"),
                          mack_average = "volume", mack_sigma_interpolation = "log-linear",
                          centre_residuals = TRUE,
                          dependence = c("independent", "synchronized", "rank_correlation"),
                          spearman = NULL, ldf = NULL, ldf_kind = NULL, ldf_weight = 1,
                          bootstrap_ldf = TRUE, n_periods = NULL, drop_high = 0, drop_low = 0,
                          mack_n_periods = NULL, mack_drop_high = 0, mack_drop_low = 0) {
  column <- fit_column(triangle, column)
  method <- match.arg(method)
  process <- match.arg(process)
  dependence <- match.arg(dependence)
  check_one_year_settings(method, exposure, c(apriori = !missing(apriori),
                                              n_iters = !missing(n_iters),
                                              trend = !missing(trend), decay = !missing(decay)))
  check_bootstrap_tail_settings(bootstrap_tail,
                                list(tail_sigma = tail_sigma, tail_std_err = tail_std_err))
  args <- development_args(average, sigma_interpolation, n_periods, drop_high, drop_low)
  model <- development_args(mack_average, mack_sigma_interpolation, mack_n_periods, mack_drop_high,
                            mack_drop_low,
                            c("mack_n_periods", "mack_drop_high", "mack_drop_low"))
  ptr <- rust_result(triangle@ptr$mack_one_year(
    column, method, exposure, single_number(apriori, "apriori"), single_number(n_iters, "n_iters"),
    single_number(trend, "trend"), single_number(decay, "decay"), args$average, args$sigma, args$excl,
    tail_ptr(tail), isTRUE(bootstrap_tail),
    optional_number(tail_sigma, "tail_sigma"), optional_number(tail_std_err, "tail_std_err"),
    single_number(n_sims, "n_sims"), single_number(seed, "seed"), process,
    model$average, model$sigma, model$excl, isTRUE(centre_residuals), dependence, spearman_arg(spearman),
    ldf_pattern(ldf, ldf_kind, ldf_weight), isTRUE(bootstrap_ldf)
  ))
  one_year_fit(ptr = ptr)
}

# The bootstrap tail's settings apply only with `bootstrap_tail = TRUE`;
# without it the bootstrap has no tail and would ignore them.
check_bootstrap_tail_settings <- function(bootstrap_tail, settings) {
  for (arg in names(settings)) {
    if (!is.null(settings[[arg]]) && !isTRUE(bootstrap_tail)) {
      stop(sprintf("%s needs bootstrap_tail = TRUE", arg), call. = FALSE)
    }
  }
}

# A one-year view's method reads only its own settings (`given` says which
# were passed), and `exposure` is NULL or one column name.
check_one_year_settings <- function(method, exposure, given) {
  reads <- list(apriori = c("expected_loss", "bornhuetter_ferguson", "benktander"),
                n_iters = "benktander", trend = "cape_cod", decay = "cape_cod")
  for (arg in names(given)[given]) {
    if (!method %in% reads[[arg]]) {
      stop(sprintf('method "%s" does not use %s', method, arg), call. = FALSE)
    }
  }
  if (!is.null(exposure) && (!is.character(exposure) || length(exposure) != 1 || is.na(exposure))) {
    stop("exposure must be the name of a column", call. = FALSE)
  }
}

#' Clark's growth-curve methods
#'
#' Clark's LDF and Cape Cod methods (Clark 2003), as R ChainLadder's
#' `ClarkLDF()` and `ClarkCapeCod()` with `adol = TRUE`: a growth curve
#' `G` and either each origin's expected ultimate (`clark_ldf()`) or one
#' expected loss ratio times each origin's exposure (`clark_cape_cod()`)
#' are fitted to the incremental losses by over-dispersed Poisson maximum
#' likelihood. Ages are measured from the average date of loss, the middle
#' of the origin period, and development stops at `max_age`. Results match
#' R ChainLadder and chainladder-python
#' (`validation/reference/reserving_clark_r.csv`), except that the Weibull
#' parameter risk uses the correct second derivative of the curve, where
#' R's has an error (`knowledge/references/r-chainladder-clark.md`).
#'
#' `clark_ldf()`'s reserve is the latest value developed by the fitted
#' curve, `latest * (G(max_age) / G(age) - 1)`; `clark_cape_cod()`'s is the
#' fitted `elr * exposure * (G(max_age) - G(age))`. Process risk is the
#' square root of `scale` times the fitted reserve, parameter risk the
#' delta method on the parameter covariance (the scale times the inverse
#' Fisher information), and `standard_error` the root of their squares'
#' sum. A singular Fisher information gives `NaN` parameter risk, as R
#' gives `NA`.
#'
#' Every segment of the triangle is fitted on its own. Properties of the
#' fit, per origin (named as a [chain_ladder_fit]'s): `latest`,
#' `expected_ultimate` (the fitted `U`, or `elr * exposure`, developed to
#' infinity), `ultimate`, `reserve`, `process_risk`, `parameter_risk`,
#' `standard_error` and, for Cape Cod, `exposure` (`NULL` for the LDF
#' method); and `chain_ladder` (the volume-weighted [chain_ladder_fit] of
#' the same column), `keys`, `index`, `origins`, `development`, `method`
#' (`"ldf"` or `"cape_cod"`), `curve`, `max_age`, `origin_width` (the
#' origin period in months), `total_ultimate` and `total_reserve` (summed
#' over segments). The fitted `omega`, `theta`, `scale` (the
#' over-dispersion `sigma^2`), `elr` (Cape Cod; `NULL` for the LDF method,
#' with any number of segments), `covariance` (of the expected ultimates or
#' the ELR, then `omega` and `theta`), `n_observations` (the incremental
#' values fitted; `scale` divides by this less the number of parameters),
#' `total_process_risk`, `total_parameter_risk` and `total_standard_error`
#' need a single-segment fit: with several segments use [totals_frame()],
#' which has them per segment (except `covariance` and `n_observations`),
#' or [segment()].
#' `growth(fit, age)` gives the share of the expected ultimate developed by
#' each development age in months (`Inf` gives 1). These are Python's
#' `ClarkLdf`, `ClarkCapeCod` and `ClarkFit`.
#'
#' @param triangle A cumulative [triangle] with at least four development
#'   ages, with any number of segments.
#' @param column Name of the loss column to fit; for `clark_ldf()` by
#'   default the only one.
#' @param exposure Name of the exposure column, such as premium; each
#'   origin's latest observed value is its exposure, which must be
#'   positive.
#' @param curve The growth curve: `"loglogistic"`,
#'   `G(x) = x^omega / (x^omega + theta^omega)`, or `"weibull"`,
#'   `G(x) = 1 - exp(-(x / theta)^omega)`.
#' @param max_age Age in months at which development stops, at least the
#'   triangle's last age; `Inf` (or `NULL`) develops to infinity.
#' @param fit A `clark_fit` with one segment.
#' @param age Development ages in months, before the shift to the average
#'   date of loss.
#' @param ptr A `ClarkFit` pointer; used internally.
#' @returns `clark_ldf()` and `clark_cape_cod()`: a `clark_fit` object.
#'   `growth()`: a numeric vector, one value per age.
#' @seealso [chain_ladder()], [mack()], [segment()].
#' @export
#' @examples
#' long <- data.frame(year = rep(2020:2024, 5:1),
#'                    age = c(12, 24, 36, 48, 60, 12, 24, 36, 48, 12, 24, 36, 12, 24, 12),
#'                    paid = c(110, 290, 370, 420, 440, 95, 300, 390, 425, 130, 320, 410,
#'                             105, 305, 120),
#'                    premium = 800)
#' tri <- triangle(long, "year", "age", c("paid", "premium"))
#' ldf <- clark_ldf(tri, "paid", curve = "weibull", max_age = 120)
#' c(omega = ldf@omega, theta = ldf@theta)
#' ldf@reserve
#' ldf@total_standard_error
#' growth(ldf, c(12, 24, 120))
#'
#' cc <- clark_cape_cod(tri, "paid", "premium")
#' cc@elr
#' as.data.frame(cc)
clark_fit <- S7::new_class(
  "clark_fit",
  package = "prospicio",
  properties = local({
    by_origin <- function(f) {
      S7::new_property(S7::class_double, getter = function(self) {
        stats::setNames(f(self@ptr), origin_names(self@chain_ladder@ptr))
      })
    }
    one <- function(f) {
      S7::new_property(S7::class_double, getter = function(self) rust_result(f(self@ptr), call = NULL))
    }
    cape_cod <- function(self) self@ptr$method() == "cape_cod"
    list(
      ptr = S7::new_S3_class("ClarkFit"),
      chain_ladder = chain_ladder_fit,
      keys = S7::new_property(S7::class_character, getter = function(self) self@chain_ladder@keys),
      index = S7::new_property(S7::class_data.frame, getter = function(self) self@chain_ladder@index),
      origins = S7::new_property(S7::class_character, getter = function(self) self@chain_ladder@origins),
      development = S7::new_property(S7::class_integer, getter = function(self) {
        self@chain_ladder@development
      }),
      method = S7::new_property(S7::class_character, getter = function(self) self@ptr$method()),
      curve = S7::new_property(S7::class_character, getter = function(self) self@ptr$curve()),
      max_age = S7::new_property(S7::class_double, getter = function(self) self@ptr$max_age()),
      omega = one(function(p) p$omega()),
      theta = one(function(p) p$theta()),
      elr = S7::new_property(getter = function(self) {
        if (!cape_cod(self)) return(NULL)
        rust_result(self@ptr$elr(), call = NULL)
      }),
      scale = one(function(p) p$scale()),
      n_observations = S7::new_property(S7::class_integer, getter = function(self) {
        rust_result(self@ptr$n_observations(), call = NULL)
      }),
      origin_width = S7::new_property(S7::class_double, getter = function(self) self@ptr$origin_width()),
      covariance = S7::new_property(S7::class_double, getter = function(self) {
        v <- rust_result(self@ptr$covariance(), call = NULL)
        n <- as.integer(round(sqrt(length(v))))
        names <- c(if (cape_cod(self)) "elr" else self@origins, "omega", "theta")
        matrix(v, n, n, byrow = TRUE, dimnames = list(names, names))
      }),
      latest = by_origin(function(p) p$chain_ladder()$latest()),
      exposure = S7::new_property(getter = function(self) {
        if (!cape_cod(self)) return(NULL)
        stats::setNames(self@ptr$exposure(), origin_names(self@chain_ladder@ptr))
      }),
      expected_ultimate = by_origin(function(p) p$expected_ultimate()),
      ultimate = by_origin(function(p) p$ultimate()),
      reserve = by_origin(function(p) p$reserve()),
      process_risk = by_origin(function(p) p$process_risk()),
      parameter_risk = by_origin(function(p) p$parameter_risk()),
      standard_error = by_origin(function(p) p$standard_error()),
      total_ultimate = S7::new_property(S7::class_double, getter = function(self) self@ptr$total_ultimate()),
      total_reserve = S7::new_property(S7::class_double, getter = function(self) self@ptr$total_reserve()),
      total_process_risk = one(function(p) p$total_process_risk()),
      total_parameter_risk = one(function(p) p$total_parameter_risk()),
      total_standard_error = one(function(p) p$total_standard_error())
    )
  }),
  constructor = function(ptr) {
    S7::new_object(S7::S7_object(), ptr = ptr, chain_ladder = chain_ladder_fit(ptr = ptr$chain_ladder()))
  }
)

# Clark's `max_age` for Rust: a single number, Inf for none.
clark_max_age <- function(max_age) {
  if (is.null(max_age)) return(Inf)
  if (!is.numeric(max_age) || length(max_age) != 1 || is.na(max_age)) {
    stop("max_age must be a single number", call. = FALSE)
  }
  as.double(max_age)
}

#' @rdname clark_fit
#' @export
clark_ldf <- function(triangle, column = NULL, curve = c("loglogistic", "weibull"), max_age = Inf) {
  column <- fit_column(triangle, column)
  curve <- match.arg(curve)
  clark_fit(ptr = rust_result(triangle@ptr$clark_ldf(column, curve, clark_max_age(max_age))))
}

#' @rdname clark_fit
#' @export
clark_cape_cod <- function(triangle, column, exposure, curve = c("loglogistic", "weibull"),
                           max_age = Inf) {
  check_triangle(triangle)
  for (arg in c("column", "exposure")) {
    value <- get(arg)
    if (!is.character(value) || length(value) != 1 || is.na(value)) {
      stop(sprintf("%s must be the name of one column", arg), call. = FALSE)
    }
  }
  curve <- match.arg(curve)
  ptr <- rust_result(triangle@ptr$clark_cape_cod(column, exposure, curve, clark_max_age(max_age)))
  clark_fit(ptr = ptr)
}

#' @rdname clark_fit
#' @export
growth <- function(fit, age) {
  if (!S7::S7_inherits(fit, clark_fit)) stop("fit must be a clark_fit", call. = FALSE)
  if (!is.numeric(age)) stop("age must be numeric", call. = FALSE)
  rust_result(fit@ptr$growth(as.double(age)))
}

check_fit <- function(fit) {
  classes <- list(chain_ladder_fit, mack_fit, expected_loss_fit, cape_cod_fit, odp_bootstrap_fit,
                  mack_bootstrap_fit, bayes_odp_fit, bayes_odp_fits, bayes_bf_fit,
                  bayes_bf_normalised_fit, meyers_fit,
                  meyers_fits, one_year_fit, bayes_one_year_fit, bayes_one_year_fits, clark_fit,
                  compartmental_fit, compartmental_fits, compartmental_one_year_fit,
                  compartmental_one_year_fits)
  if (!any(vapply(classes, function(cls) S7::S7_inherits(fit, cls), TRUE))) {
    stop("fit must be a chain_ladder_fit, mack_fit, expected_loss_fit, cape_cod_fit, ",
         "odp_bootstrap_fit, mack_bootstrap_fit, bayes_odp_fit, bayes_odp_fits, bayes_bf_fit, ",
         "bayes_bf_normalised_fit, ",
         "meyers_fit, meyers_fits, one_year_fit, bayes_one_year_fit, bayes_one_year_fits, ",
         "clark_fit, compartmental_fit, compartmental_fits, compartmental_one_year_fit or ",
         "compartmental_one_year_fits",
         call. = FALSE)
  }
}

#' Long results of a fit over every segment
#'
#' Tables of a [chain_ladder_fit], [mack_fit], [expected_loss_fit],
#' [cape_cod_fit], [odp_bootstrap_fit], [mack_bootstrap_fit],
#' [bayes_odp_fit], [bayes_odp_fits], [bayes_bf_fit], [bayes_bf_normalised_fit],
#' [meyers_fit], [meyers_fits], [one_year_fit], [bayes_one_year_fit], [bayes_one_year_fits], [clark_fit] or
#' [compartmental_fit], [compartmental_fits], [compartmental_one_year_fit] or
#' [compartmental_one_year_fits] with the triangle's key columns by name. `as.data.frame(fit)` has one row per
#' segment and origin: `origin`, `latest`, `ultimate` and `reserve` (the
#' method's own), plus for Mack `process_risk`, `parameter_risk` and
#' `standard_error`, for the expected-loss methods `exposure` and `apriori`
#' (and Cape Cod's `trended_apriori`), for the bootstraps the `mean` and
#' `std_dev` of the bootstrapped reserve (for [bayes_odp()],
#' [bayes_odp_segments()], [bayes_bf()], [bayes_bf_normalised()], [meyers()] and
#' [meyers_segments()] of the posterior predictive reserve; [bayes_bf()] and
#' [bayes_bf_normalised()] add the `exposure` and `prior_ultimate`, [meyers()] and
#' [meyers_segments()] add the `premium` and the
#' `ultimate_mean` and `ultimate_std_dev` of the predictive ultimate), and
#' for Clark (Cape Cod)
#' `exposure`, `expected_ultimate` and the three standard errors; for
#' [compartmental()] the `premium`, `latest_paid`, the `mean` and `std_dev`
#' of the posterior predictive reserve and the `ultimate_loss_ratio`. The
#' one-year view (and [bayes_odp_one_year()]'s) names them
#' `opening_ultimate` and `opening_reserve` and adds the `cdr_mean` and
#' `cdr_std_dev` of the claims development result.
#' `totals_frame()` has one row per segment with the same quantities for
#' the segment's total (for the expected-loss methods the total `exposure`,
#' for the ODP bootstrap, its one-year view, [bayes_odp()],
#' [bayes_odp_segments()], [bayes_bf()] and [bayes_bf_normalised()] also the `scale`, for
#' [bayes_odp_one_year()] the `scale` and `flagged_share`;
#' for Clark its `omega`, `theta`, `scale` and, for Cape Cod, `elr`).
#' `development_frame()` has one row per segment and age: `development`,
#' `ldf` (to the next age), `cdf` (to ultimate, with the tail), `sigma` and
#' `std_err`; the oldest age has `NA` for `ldf`, `sigma` and `std_err`. A
#' [clark_fit] has no development table of its own: use
#' `development_frame(fit@chain_ladder)` for the chain ladder's.
#'
#' `segment()` returns the fit of one segment, chosen by key values as in
#' `segment(fit, lob = "auto")` (compared as character). Keys not named may
#' take any value, so a fit with one segment needs none; a choice that
#' matches several segments is an error. For the bootstraps, the segment
#' keeps its part of the joint `reserves` (and [meyers_fits] of the joint
#' `ultimates`), and for the one-year view its
#' part of the joint `cdr`, with the same dimensions.
#'
#' These are Python's `to_frame()`, `totals_frame()`,
#' `development_frame()` and `segment(**keys)`.
#'
#' @param fit A [chain_ladder_fit], [mack_fit], [expected_loss_fit],
#'   [cape_cod_fit], [odp_bootstrap_fit], [mack_bootstrap_fit],
#'   [bayes_odp_fit], [bayes_odp_fits], [bayes_bf_fit], [bayes_bf_normalised_fit],
#'   [meyers_fit], [meyers_fits], [one_year_fit], [bayes_one_year_fit], [bayes_one_year_fits], [clark_fit],
#'   [compartmental_fit], [compartmental_fits], [compartmental_one_year_fit] or
#'   [compartmental_one_year_fits] (a [clark_fit] or
#'   a compartmental fit not for `development_frame()`).
#' @param ... Key conditions as `key = value`, one value each.
#' @returns A data.frame, or for `segment()` a fit of the same class.
#' @name fit_frames
#' @examples
#' long <- data.frame(lob = rep(c("auto", "home"), each = 6), year = c(2020, 2020, 2020, 2021, 2021, 2022),
#'                    age = c(12, 24, 36, 12, 24, 12),
#'                    paid = c(100, 150, 165, 110, 170, 120, 50, 80, 85, 60, 90, 70))
#' fits <- chain_ladder(triangle(long, "year", "age", "paid", keys = "lob"))
#' totals_frame(fits)
#' development_frame(fits)
#' segment(fits, lob = "home")@ldf
NULL

#' @rdname fit_frames
#' @export
totals_frame <- function(fit) {
  check_fit(fit)
  fit_table_frame(fit@ptr$totals_table())
}

#' @rdname fit_frames
#' @export
development_frame <- function(fit) {
  check_fit(fit)
  if (S7::S7_inherits(fit, clark_fit)) {
    stop("a clark_fit has no development table; use development_frame(fit@chain_ladder)",
         call. = FALSE)
  }
  if (S7::S7_inherits(fit, compartmental_fit) || S7::S7_inherits(fit, compartmental_fits)) {
    stop("a compartmental_fit has no development table; see its paid_table and outstanding_table",
         call. = FALSE)
  }
  if (S7::S7_inherits(fit, compartmental_one_year_fit) ||
      S7::S7_inherits(fit, compartmental_one_year_fits)) {
    stop("a compartmental_one_year_fit has no development table; see the paid_table and ",
         "outstanding_table of its fit", call. = FALSE)
  }
  fit_table_frame(fit@ptr$development_table())
}

#' @rdname fit_frames
#' @export
segment <- function(fit, ...) {
  check_fit(fit)
  call <- sys.call()
  conditions <- list(...)
  keys <- names(conditions)
  if (length(conditions) && (is.null(keys) || any(keys == ""))) {
    stop("segment() conditions must be named by key, as in lob = \"auto\"", call. = FALSE)
  }
  if (any(lengths(conditions) != 1)) {
    stop("segment() takes one value per key", call. = FALSE)
  }
  values <- vapply(conditions, as.character, "", USE.NAMES = FALSE)
  ptr <- rust_result(fit@ptr$segment(as.character(keys), values), call = call)
  S7::S7_class(fit)(ptr = ptr)
}

S7::method(as.data.frame, chain_ladder_fit) <- function(x, ...) fit_table_frame(x@ptr$long_table())
S7::method(as.data.frame, mack_fit) <- function(x, ...) fit_table_frame(x@ptr$long_table())
S7::method(as.data.frame, expected_loss_fit) <- function(x, ...) fit_table_frame(x@ptr$long_table())
S7::method(as.data.frame, cape_cod_fit) <- function(x, ...) fit_table_frame(x@ptr$long_table())
S7::method(as.data.frame, odp_bootstrap_fit) <- function(x, ...) fit_table_frame(x@ptr$long_table())
S7::method(as.data.frame, mack_bootstrap_fit) <- function(x, ...) fit_table_frame(x@ptr$long_table())
S7::method(as.data.frame, bayes_odp_fit) <- function(x, ...) fit_table_frame(x@ptr$long_table())
S7::method(summary, bayes_odp_fit) <- function(object, ...) object@summary
S7::method(as.data.frame, bayes_odp_fits) <- function(x, ...) fit_table_frame(x@ptr$long_table())
S7::method(summary, bayes_odp_fits) <- function(object, ...) object@summary
S7::method(as.data.frame, bayes_bf_fit) <- function(x, ...) fit_table_frame(x@ptr$long_table())
S7::method(summary, bayes_bf_fit) <- function(object, ...) object@summary
S7::method(as.data.frame, bayes_bf_normalised_fit) <- function(x, ...) fit_table_frame(x@ptr$long_table())
S7::method(summary, bayes_bf_normalised_fit) <- function(object, ...) object@summary
S7::method(as.data.frame, meyers_fit) <- function(x, ...) fit_table_frame(x@ptr$long_table())
S7::method(summary, meyers_fit) <- function(object, ...) object@summary
S7::method(as.data.frame, meyers_fits) <- function(x, ...) fit_table_frame(x@ptr$long_table())
S7::method(summary, meyers_fits) <- function(object, ...) object@summary
S7::method(as.data.frame, one_year_fit) <- function(x, ...) fit_table_frame(x@ptr$long_table())
S7::method(as.data.frame, bayes_one_year_fit) <- function(x, ...) {
  fit_table_frame(x@ptr$long_table())
}
S7::method(as.data.frame, bayes_one_year_fits) <- function(x, ...) {
  fit_table_frame(x@ptr$long_table())
}
S7::method(as.data.frame, clark_fit) <- function(x, ...) fit_table_frame(x@ptr$long_table())

# A segment count for print headers when there are several.
segments_note <- function(p) {
  n <- p$n_segments()
  if (n > 1) sprintf(", %d segments", n) else ""
}

S7::method(print, chain_ladder_fit) <- function(x, ...) {
  # Each segment has its own tail.
  tail <- if (x@ptr$n_segments() > 1) "" else sprintf(", tail %s", format(x@tail))
  cat(sprintf("<chain_ladder_fit> total reserve %s%s%s\n",
              format(x@total_reserve, digits = 10), tail, segments_note(x@ptr)))
  print(as.data.frame(x), row.names = FALSE)
  invisible(x)
}
S7::method(print, odp_bootstrap_fit) <- function(x, ...) {
  r <- x@reserves
  p <- x@chain_ladder@ptr
  if (p$n_segments() > 1) {
    cat(sprintf("<odp_bootstrap_fit> %d simulations%s\n", as.integer(r@n_sims), segments_note(p)))
  } else {
    cat(sprintf("<odp_bootstrap_fit> %d simulations, scale %s\n",
                as.integer(r@n_sims), format(x@scale, digits = 6)))
  }
  cat(sprintf("total reserve: chain ladder %s, bootstrap mean %s, sd %s\n",
              format(x@chain_ladder@total_reserve, digits = 10), format(mean(r), digits = 10),
              format(sqrt(variance(r)), digits = 10)))
  invisible(x)
}
S7::method(print, bayes_odp_fit) <- function(x, ...) {
  r <- x@reserves
  d <- x@diagnostics
  cat(sprintf("<bayes_odp_fit> %d draws in %d chains, %d simulations, scale %s\n",
              nrow(x@draws), as.integer(x@ptr$chains()), as.integer(r@n_sims),
              format(x@scale, digits = 6)))
  cat(sprintf("diagnostics: max R-hat %s, min bulk ESS %s, min tail ESS %s, %d divergences%s\n",
              format(d$max_rhat, digits = 4), format(d$min_ess_bulk, digits = 4),
              format(d$min_ess_tail, digits = 4), as.integer(d$divergences),
              if (length(d$warnings)) sprintf(", %d warnings", length(d$warnings)) else ""))
  cat(sprintf("total reserve: chain ladder %s, posterior mean %s, sd %s\n",
              format(x@chain_ladder@total_reserve, digits = 10), format(mean(r), digits = 10),
              format(sqrt(variance(r)), digits = 10)))
  invisible(x)
}
S7::method(print, bayes_bf_fit) <- function(x, ...) {
  r <- x@reserves
  d <- x@diagnostics
  cat(sprintf("<bayes_bf_fit> %d draws in %d chains, %d simulations, scale %s\n",
              nrow(x@draws), as.integer(x@ptr$chains()), as.integer(r@n_sims),
              format(x@scale, digits = 6)))
  cat(sprintf("diagnostics: max R-hat %s, min bulk ESS %s, min tail ESS %s, %d divergences%s\n",
              format(d$max_rhat, digits = 4), format(d$min_ess_bulk, digits = 4),
              format(d$min_ess_tail, digits = 4), as.integer(d$divergences),
              if (length(d$warnings)) sprintf(", %d warnings", length(d$warnings)) else ""))
  cat(sprintf("total prior ultimate %s\n", format(sum(x@prior_ultimate), digits = 10)))
  cat(sprintf("total reserve: chain ladder %s, posterior mean %s, sd %s
",
              format(x@chain_ladder@total_reserve, digits = 10), format(mean(r), digits = 10),
              format(sqrt(variance(r)), digits = 10)))
  invisible(x)
}
S7::method(print, bayes_bf_normalised_fit) <- function(x, ...) {
  r <- x@reserves
  d <- x@diagnostics
  cat(sprintf("<bayes_bf_normalised_fit> %d draws in %d chains, %d simulations, scale %s\n",
              nrow(x@draws), as.integer(x@ptr$chains()), as.integer(r@n_sims),
              format(x@scale, digits = 6)))
  cat(sprintf("diagnostics: max R-hat %s, min bulk ESS %s, min tail ESS %s, %d divergences%s\n",
              format(d$max_rhat, digits = 4), format(d$min_ess_bulk, digits = 4),
              format(d$min_ess_tail, digits = 4), as.integer(d$divergences),
              if (length(d$warnings)) sprintf(", %d warnings", length(d$warnings)) else ""))
  cat(sprintf("total prior ultimate %s\n", format(sum(x@prior_ultimate), digits = 10)))
  cat(sprintf("total reserve: chain ladder %s, posterior mean %s, sd %s
",
              format(x@chain_ladder@total_reserve, digits = 10), format(mean(r), digits = 10),
              format(sqrt(variance(r)), digits = 10)))
  invisible(x)
}
S7::method(print, bayes_odp_fits) <- function(x, ...) {
  r <- x@reserves
  d <- x@diagnostics
  p <- x@chain_ladder@ptr
  cat(sprintf("<bayes_odp_fits> %d simulations%s\n", as.integer(r@n_sims),
              if (p$n_segments() > 1) segments_note(p) else
                sprintf(", scale %s", format(x@scale, digits = 6))))
  cat(sprintf("diagnostics: max R-hat %s, min bulk ESS %s, min tail ESS %s, %d divergences%s\n",
              format(max(d$max_rhat), digits = 4), format(min(d$min_ess_bulk), digits = 4),
              format(min(d$min_ess_tail), digits = 4), as.integer(sum(d$divergences)),
              if (any(d$warnings != "")) {
                sprintf(", warnings in %d segments", sum(d$warnings != ""))
              } else {
                ""
              }))
  cat(sprintf("total reserve: chain ladder %s, posterior mean %s, sd %s\n",
              format(x@chain_ladder@total_reserve, digits = 10), format(mean(r), digits = 10),
              format(sqrt(variance(r)), digits = 10)))
  invisible(x)
}
S7::method(print, meyers_fits) <- function(x, ...) {
  r <- x@reserves
  d <- x@diagnostics
  p <- x@chain_ladder@ptr
  cat(sprintf("<meyers_fits> %s, %d simulations%s\n", toupper(x@model), as.integer(r@n_sims),
              segments_note(p)))
  cat(sprintf("diagnostics: max R-hat %s, min bulk ESS %s, min tail ESS %s, %d divergences%s\n",
              format(max(d$max_rhat), digits = 4), format(min(d$min_ess_bulk), digits = 4),
              format(min(d$min_ess_tail), digits = 4), as.integer(sum(d$divergences)),
              if (any(d$warnings != "")) {
                sprintf(", warnings in %d segments", sum(d$warnings != ""))
              } else {
                ""
              }))
  cat(sprintf("total ultimate: posterior mean %s, sd %s\n",
              format(mean(x@ultimates), digits = 10),
              format(sqrt(variance(x@ultimates)), digits = 10)))
  cat(sprintf("total reserve: chain ladder %s, posterior mean %s, sd %s\n",
              format(x@chain_ladder@total_reserve, digits = 10), format(mean(r), digits = 10),
              format(sqrt(variance(r)), digits = 10)))
  invisible(x)
}
S7::method(print, meyers_fit) <- function(x, ...) {
  r <- x@reserves
  d <- x@diagnostics
  cat(sprintf("<meyers_fit> %s, %d draws in %d chains, %d simulations
", toupper(x@model),
              nrow(x@draws), as.integer(x@ptr$chains()), as.integer(r@n_sims)))
  cat(sprintf("diagnostics: max R-hat %s, min bulk ESS %s, min tail ESS %s, %d divergences%s
",
              format(d$max_rhat, digits = 4), format(d$min_ess_bulk, digits = 4),
              format(d$min_ess_tail, digits = 4), as.integer(d$divergences),
              if (length(d$warnings)) sprintf(", %d warnings", length(d$warnings)) else ""))
  cat(sprintf("total ultimate: posterior mean %s, sd %s
",
              format(mean(x@ultimates), digits = 10),
              format(sqrt(variance(x@ultimates)), digits = 10)))
  cat(sprintf("total reserve: chain ladder %s, posterior mean %s, sd %s
",
              format(x@chain_ladder@total_reserve, digits = 10), format(mean(r), digits = 10),
              format(sqrt(variance(r)), digits = 10)))
  invisible(x)
}
S7::method(print, mack_bootstrap_fit) <- function(x, ...) {
  r <- x@reserves
  cat(sprintf("<mack_bootstrap_fit> %d simulations%s\n", as.integer(r@n_sims),
              segments_note(x@chain_ladder@ptr)))
  cat(sprintf("total reserve: chain ladder %s, bootstrap mean %s, sd %s\n",
              format(x@chain_ladder@total_reserve, digits = 10), format(mean(r), digits = 10),
              format(sqrt(variance(r)), digits = 10)))
  invisible(x)
}
S7::method(print, one_year_fit) <- function(x, ...) {
  cdr <- x@cdr
  p <- x@chain_ladder@ptr
  if (p$n_segments() > 1) {
    cat(sprintf("<one_year_fit> %d simulations%s\n", as.integer(cdr@n_sims), segments_note(p)))
  } else if (x@model == "mack") {
    cat(sprintf("<one_year_fit> %d simulations of Mack's bootstrap\n", as.integer(cdr@n_sims)))
  } else {
    cat(sprintf("<one_year_fit> %d simulations, scale %s\n", as.integer(cdr@n_sims),
                format(x@scale, digits = 6)))
  }
  cat(sprintf("total opening reserve %s; claims development result mean %s, sd %s\n",
              format(sum(x@opening_reserve), digits = 10), format(mean(cdr), digits = 10),
              format(sqrt(variance(cdr)), digits = 10)))
  print(if (p$n_segments() > 1) totals_frame(x) else as.data.frame(x), row.names = FALSE)
  invisible(x)
}
S7::method(print, bayes_one_year_fit) <- function(x, ...) {
  cdr <- x@cdr
  detail <- if (x@ptr$model() == "meyers") {
    sprintf("Meyers' %s", toupper(x@fit@model))
  } else {
    sprintf("scale %s", format(x@fit@scale, digits = 6))
  }
  cat(sprintf("<bayes_one_year_fit> %d simulations, %s\n", as.integer(cdr@n_sims), detail))
  cat(sprintf("Pareto k above %s: %s of the simulations, %d refitted%s\n",
              format(x@k_threshold), format(x@flagged_share, digits = 3), length(x@refitted),
              if (x@refit_warnings > 0) sprintf(" (%d with warnings)", x@refit_warnings) else ""))
  cat(sprintf("total opening reserve %s; claims development result mean %s, sd %s\n",
              format(sum(x@opening_reserve), digits = 10), format(mean(cdr), digits = 10),
              format(sqrt(variance(cdr)), digits = 10)))
  print(as.data.frame(x), row.names = FALSE)
  invisible(x)
}
S7::method(print, bayes_one_year_fits) <- function(x, ...) {
  cdr <- x@cdr
  p <- x@chain_ladder@ptr
  cat(sprintf("<bayes_one_year_fits> %d simulations%s\n", as.integer(cdr@n_sims),
              if (p$n_segments() > 1) {
                segments_note(p)
              } else if (x@ptr$model() == "meyers") {
                ""
              } else {
                sprintf(", scale %s", format(x@scale, digits = 6))
              }))
  cat(sprintf("total opening reserve %s; claims development result mean %s, sd %s\n",
              format(sum(x@opening_reserve), digits = 10), format(mean(cdr), digits = 10),
              format(sqrt(variance(cdr)), digits = 10)))
  print(totals_frame(x), row.names = FALSE)
  invisible(x)
}
S7::method(print, mack_fit) <- function(x, ...) {
  p <- x@chain_ladder@ptr
  if (p$n_segments() > 1) {
    cat(sprintf("<mack_fit> total reserve %s%s\n", format(x@total_reserve, digits = 10),
                segments_note(p)))
    print(totals_frame(x), row.names = FALSE)
  } else {
    cat(sprintf("<mack_fit> total reserve %s, standard error %s (CV %s)\n",
                format(x@total_reserve, digits = 10), format(x@total_standard_error, digits = 10),
                format(x@total_cv, digits = 4)))
    print(as.data.frame(x), row.names = FALSE)
  }
  invisible(x)
}
S7::method(print, clark_fit) <- function(x, ...) {
  p <- x@chain_ladder@ptr
  what <- sprintf("%s, %s curve", if (x@method == "ldf") "LDF" else "Cape Cod", x@curve)
  if (is.finite(x@max_age)) what <- sprintf("%s to age %s", what, format(x@max_age))
  if (p$n_segments() > 1) {
    cat(sprintf("<clark_fit> %s, total reserve %s%s\n", what, format(x@total_reserve, digits = 10),
                segments_note(p)))
    print(totals_frame(x), row.names = FALSE)
  } else {
    cat(sprintf("<clark_fit> %s, total reserve %s, standard error %s\n", what,
                format(x@total_reserve, digits = 10), format(x@total_standard_error, digits = 10)))
    print(as.data.frame(x), row.names = FALSE)
  }
  invisible(x)
}
# Prints the totals of an expected-loss fit (`name` its class) and its
# long table.
print_expected_loss <- function(x, name) {
  cat(sprintf("<%s> total ultimate %s, total reserve %s%s\n", name,
              format(x@total_ultimate, digits = 10), format(x@total_reserve, digits = 10),
              segments_note(x@chain_ladder@ptr)))
  print(as.data.frame(x), row.names = FALSE)
  invisible(x)
}
S7::method(print, expected_loss_fit) <- function(x, ...) print_expected_loss(x, "expected_loss_fit")
S7::method(print, cape_cod_fit) <- function(x, ...) print_expected_loss(x, "cape_cod_fit")

#' Priors of Gesmann and Morris' compartmental model
#'
#' The priors of [compartmental()], a required argument: the rates and
#' ratios are specific to a line and a book, so the library does not choose
#' them silently. `gesmann_morris_priors()` is the paper's case study
#' (Gesmann and Morris 2020, model 1): `k_er` (3, 0.1), `k_p` (1, 0.1),
#' `rlr` (0.7, 0.2), `rrf` (0.8, 0.1), half Student-t(10) effect sds with
#' scales 0.2 and 0.1, LKJ(1), and Student-t(1, 0, 1000) on each
#' `log sigma` (effectively flat), with no effect on the rates and no
#' development effect. On company 337 the data pull the rates 5 to 9 prior
#' sds from these medians: the posterior is a compromise with the preset,
#' not a description of it. Check any set with
#' [compartmental_prior_predictive()] before fitting.
#'
#' `gesmann_morris_2_priors()` is the paper's model 2, from its code (sec.
#' 7.2.3, where it and the text disagree): model 1's rate and ratio priors;
#' origin effects on all four parameters (`RLR` and `RRF` correlated,
#' LKJ(1)) and development-year effects on all four, every effect sd half
#' Student-t(10) on the `o` scale with one scale per parameter, shared by
#' its two groups: 0.3 for `k_er` and `k_p`, 0.7 for `rlr` and 0.5 for
#' `rrf`; and `normal(log 0.2, 0.2)` on each `log sigma` (the text gives a
#' median of 0.1). Use it with [compartmental_model_2()].
#'
#' Each positive parameter `theta` (`k_er`, `k_p`, `rlr`, `rrf`) is
#' `median * exp(sd * o)` with `o ~ N(0, 1)`: a log-normal prior with that
#' median and log-scale sd. The effects live on that `o` scale.
#'
#' The hierarchy follows from the priors. `RLR` and `RRF` always vary by
#' origin, with correlated effects. `ker_effect_sd` and `kp_effect_sd` add
#' origin effects on the rates, uncorrelated with the others, and
#' `development_effect_sd` adds development-year effects on any of the four
#' parameters; `NULL` (model 1) means no such effect. A development age
#' with no observed cell takes a new effect from the population in each
#' simulation, as an origin with no cell does.
#'
#' Errors: if a prior has the wrong form or a log-normal median or sd is
#' not finite and positive. The other settings are checked by the fit.
#'
#' @param k_er,k_p `c(median = , sd = )` of the reporting rate and the
#'   payment rate, per year.
#' @param rlr,rrf `c(median = , sd = )` of the population reported loss
#'   ratio and reserve robustness factor (paid over reported).
#' @param rlr_effect_sd,rrf_effect_sd Priors on the sds of the origin
#'   effects on the `o` scale: `list(family = "student_t", nu = , scale = )`
#'   for a half Student-t or `list(family = "normal", scale = )` for a half
#'   normal.
#' @param lkj_eta LKJ shape of the correlation of the two effects (1:
#'   uniform).
#' @param log_sigma A list of two priors, on `log sigma_os` and
#'   `log sigma_pd` in that order, each
#'   `list(family = "student_t", nu = , location = , scale = )` or
#'   `list(family = "normal", mean = , sd = )`.
#' @param ker_effect_sd,kp_effect_sd Priors on the sds of origin effects on
#'   `k_er` and `k_p`, in the form of `rlr_effect_sd`; `NULL`: no such
#'   effect.
#' @param development_effect_sd Priors on the sds of development-year
#'   effects: a named list with any of `k_er`, `k_p`, `rlr` and `rrf`, each
#'   in the form of `rlr_effect_sd` (a parameter left out has no
#'   development effect); `NULL`: no development effect.
#' @returns A `compartmental_priors` list.
#' @seealso [compartmental()], [compartmental_prior_predictive()].
#' @export
#' @examples
#' p <- gesmann_morris_priors()
#' p$k_er
#' # Wider rate priors, a half-normal effect sd on RRF.
#' q <- compartmental_priors(k_er = c(median = 2, sd = 0.3), k_p = p$k_p, rlr = p$rlr,
#'                           rrf = p$rrf, rlr_effect_sd = p$rlr_effect_sd,
#'                           rrf_effect_sd = list(family = "normal", scale = 0.2),
#'                           lkj_eta = 2, log_sigma = p$log_sigma)
#' q$rrf_effect_sd
#' # Model 2's priors: effects on the rates and by development year too.
#' p2 <- gesmann_morris_2_priors()
#' p2$kp_effect_sd
#' p2$development_effect_sd$rlr
compartmental_priors <- function(k_er, k_p, rlr, rrf, rlr_effect_sd, rrf_effect_sd, lkj_eta,
                                 log_sigma, ker_effect_sd = NULL, kp_effect_sd = NULL,
                                 development_effect_sd = NULL) {
  positive <- function(x, arg) {
    if (!is.numeric(x) || length(x) != 2 || anyNA(x) || any(!is.finite(x)) || any(x <= 0)) {
      stop(sprintf("%s must be c(median = , sd = ), both finite and positive", arg), call. = FALSE)
    }
    c(median = x[[1]], sd = x[[2]])
  }
  family_of <- function(x, arg, forms) {
    if (!is.list(x) || !is.character(x$family) || length(x$family) != 1 ||
        !x$family %in% names(forms)) {
      stop(sprintf("%s must be %s", arg, paste(vapply(forms, `[[`, "", "text"), collapse = " or ")),
           call. = FALSE)
    }
    fields <- forms[[x$family]]$fields
    values <- lapply(fields, function(f) x[[f]])
    if (any(vapply(values, function(v) !is.numeric(v) || length(v) != 1 || is.na(v), TRUE))) {
      stop(sprintf("%s must be %s", arg, forms[[x$family]]$text), call. = FALSE)
    }
    c(list(family = x$family), stats::setNames(lapply(values, as.double), fields))
  }
  half <- list(
    student_t = list(fields = c("nu", "scale"),
                     text = 'list(family = "student_t", nu = , scale = )'),
    normal = list(fields = "scale", text = 'list(family = "normal", scale = )'))
  real <- list(
    student_t = list(fields = c("nu", "location", "scale"),
                     text = 'list(family = "student_t", nu = , location = , scale = )'),
    normal = list(fields = c("mean", "sd"), text = 'list(family = "normal", mean = , sd = )'))
  if (!is.list(log_sigma) || length(log_sigma) != 2) {
    stop("log_sigma must be a list of two priors: log sigma_os, then log sigma_pd", call. = FALSE)
  }
  optional <- function(x, arg) if (is.null(x)) NULL else family_of(x, arg, half)
  dev_names <- c("k_er", "k_p", "rlr", "rrf")
  dev <- NULL
  if (!is.null(development_effect_sd)) {
    if (!is.list(development_effect_sd) || !length(development_effect_sd) ||
        is.null(names(development_effect_sd)) ||
        any(!names(development_effect_sd) %in% dev_names) ||
        anyDuplicated(names(development_effect_sd))) {
      stop(paste0("development_effect_sd must be NULL or a named list with any of k_er, k_p, ",
                  "rlr and rrf, each a half prior"), call. = FALSE)
    }
    dev <- lapply(stats::setNames(dev_names, dev_names), function(n) {
      optional(development_effect_sd[[n]], paste0("development_effect_sd$", n))
    })
    dev <- dev[!vapply(dev, is.null, TRUE)]
    if (!length(dev)) dev <- NULL
  }
  structure(list(
    k_er = positive(k_er, "k_er"), k_p = positive(k_p, "k_p"), rlr = positive(rlr, "rlr"),
    rrf = positive(rrf, "rrf"),
    rlr_effect_sd = family_of(rlr_effect_sd, "rlr_effect_sd", half),
    rrf_effect_sd = family_of(rrf_effect_sd, "rrf_effect_sd", half),
    lkj_eta = single_number(lkj_eta, "lkj_eta"),
    log_sigma = list(family_of(log_sigma[[1]], "log_sigma[[1]]", real),
                     family_of(log_sigma[[2]], "log_sigma[[2]]", real)),
    ker_effect_sd = optional(ker_effect_sd, "ker_effect_sd"),
    kp_effect_sd = optional(kp_effect_sd, "kp_effect_sd"),
    development_effect_sd = dev
  ), class = "compartmental_priors")
}

#' @rdname compartmental_priors
#' @export
gesmann_morris_priors <- function() {
  cauchy <- list(family = "student_t", nu = 1, location = 0, scale = 1000)
  compartmental_priors(
    k_er = c(median = 3, sd = 0.1), k_p = c(median = 1, sd = 0.1),
    rlr = c(median = 0.7, sd = 0.2), rrf = c(median = 0.8, sd = 0.1),
    rlr_effect_sd = list(family = "student_t", nu = 10, scale = 0.2),
    rrf_effect_sd = list(family = "student_t", nu = 10, scale = 0.1),
    lkj_eta = 1, log_sigma = list(cauchy, cauchy))
}

#' @rdname compartmental_priors
#' @export
gesmann_morris_2_priors <- function() {
  half <- function(scale) list(family = "student_t", nu = 10, scale = scale)
  log_sigma <- list(family = "normal", mean = log(0.2), sd = 0.2)
  compartmental_priors(
    k_er = c(median = 3, sd = 0.1), k_p = c(median = 1, sd = 0.1),
    rlr = c(median = 0.7, sd = 0.2), rrf = c(median = 0.8, sd = 0.1),
    rlr_effect_sd = half(0.7), rrf_effect_sd = half(0.5), lkj_eta = 1,
    log_sigma = list(log_sigma, log_sigma), ker_effect_sd = half(0.3),
    kp_effect_sd = half(0.3),
    development_effect_sd = list(k_er = half(0.3), k_p = half(0.3), rlr = half(0.7),
                                 rrf = half(0.5)))
}

# The priors as the Rust side reads them: 31 numbers and 10 family names.
compartmental_prior_values <- function(priors, call) {
  if (!inherits(priors, "compartmental_priors")) {
    stop(errorCondition(paste0(
      "priors must be a compartmental_priors list; use gesmann_morris_priors() for the paper's ",
      "case study, or compartmental_priors()"), call = call))
  }
  half <- function(h) if (h$family == "student_t") c(h$nu, h$scale) else c(h$scale, NaN)
  real <- function(r) {
    if (r$family == "student_t") c(r$nu, r$location, r$scale) else c(r$mean, r$sd, NaN)
  }
  optional <- list(priors$ker_effect_sd, priors$kp_effect_sd,
                   priors$development_effect_sd$k_er, priors$development_effect_sd$k_p,
                   priors$development_effect_sd$rlr, priors$development_effect_sd$rrf)
  list(
    values = as.double(c(priors$k_er, priors$k_p, priors$rlr, priors$rrf,
                         half(priors$rlr_effect_sd), half(priors$rrf_effect_sd), priors$lkj_eta,
                         real(priors$log_sigma[[1]]), real(priors$log_sigma[[2]]),
                         unlist(lapply(optional, function(h) {
                           if (is.null(h)) c(NaN, NaN) else half(h)
                         })))),
    families = c(priors$rlr_effect_sd$family, priors$rrf_effect_sd$family,
                 priors$log_sigma[[1]]$family, priors$log_sigma[[2]]$family,
                 vapply(optional, function(h) if (is.null(h)) "none" else h$family, "")))
}

#' Gesmann and Morris' compartmental model
#'
#' Gesmann and Morris' (2020) hierarchical compartmental reserving model
#' with one outstanding compartment, their models 1 and 2, fitted to paid
#' and outstanding jointly with NUTS. Claims flow from exposure into outstanding
#' as they are reported, at rate `k_er`, and from outstanding into paid as
#' they settle, at rate `k_p`. For origin `i` with premium `P_i` and
#' development time `t` in years, with
#' `g(t) = (exp(-k_p t) - exp(-k_er t)) / (k_er - k_p)`,
#' `OS_i(t) = P_i RLR_i k_er g(t)` and
#' `PD_i(t) = P_i RLR_i RRF_i [(1 - exp(-k_p t)) - k_p g(t)]`, so `RLR_i` is
#' the reported loss ratio, `RRF_i` the reserve robustness factor (paid over
#' reported) and `RLR_i RRF_i` the ultimate loss ratio. `RLR` and `RRF`
#' vary by origin with correlated effects (LKJ prior on the correlation) on
#' the priors' `o` scale; the priors can add origin effects on the rates
#' and development-year effects on any of the four
#' ([compartmental_priors()]), each cell then taking its own parameters.
#' NUTS samples on a pilot run's dense scale, with the gradient from dual
#' numbers (`docs/design/reserving-compartmental.md`).
#'
#' Model 1 (the defaults) has normal outstanding and cumulative paid
#' amounts about these curves, each type with its own sigma. Model 2
#' (`compartmental_model_2()`, with [gesmann_morris_2_priors()]) has
#' log-normal outstanding and incremental paid loss ratios to premium;
#' every combination of `likelihood`, `scale` and `paid_response` is
#' built. Under `likelihood = "lognormal"` every response must be
#' positive, and under `paid_response = "incremental"` each paid cell needs
#' the paid at the age before; the fit stops with an error naming the cell
#' otherwise. `fitted` other than `"paid_and_outstanding"` is accepted, but
#' the fit stops with an error saying it is not yet built.
#'
#' `compartmental_model_2()` is `compartmental()` with model 2's
#' likelihood, scale and paid response, and a longer, finer sampler: four
#' chains of 2,000 warm-up and 8,000 kept draws with target acceptance
#' 0.999. With 87 parameters, effect sds whose posteriors pile up near 0
#' and a slowly mixing `k_p`, model 1's sampler left divergent transitions
#' or an R-hat above 1.01 on most seeds; this one passes every threshold
#' on 195 of 200 seeds on company 337. A fit takes about a minute.
#'
#' An origin with a premium and no observed cell (the papers' training
#' triangle has one) draws its effects from the population in each
#' simulation, and so does a development age with no observed cell. The
#' reserve is the paid at the square's last age less the
#' latest paid; the square has as many ages as the premium column has
#' origins, and the origin and development grains must be equal. One `seed`
#' replays a fit exactly: the chains and the simulations each take a seed
#' derived from it.
#'
#' The means are exactly invariant under swapping the two rates (with `RLR`
#' and `RRF` rescaled), so only the priors tell the two modes apart. Under
#' `rate_order = "free"` (the paper's priors) the fit warns when any
#' posterior draw has population rates with `k_er <= k_p`;
#' `rate_order = "ordered"` samples `k_er = k_p + exp(delta)`, the same
#' priors restricted to `k_er > k_p`. With effects on the rates, the fit's
#' `cell_rate_swaps` is the share of draws and observed cells with
#' `k_er <= k_p` (`NA` when the rates are shared).
#'
#' Failed MCMC diagnostics (R-hat above 1.01, bulk or tail ESS below 100
#' per chain, any divergent transition) and rate swaps do not fail the fit:
#' `compartmental()` raises one [warning()] listing them, and they stay in
#' the fit's `diagnostics`. With the defaults, 194 of 200 seeds on company
#' 337 passed every threshold for model 1, and with
#' `compartmental_model_2()`'s sampler 195 of 200 for model 2; each of the
#' others just missed one.
#'
#' Properties of the fit: `origins` (those of the premium column),
#' `premium`, `latest_paid` (0 for an origin with none),
#' `population_origins` (origins drawn from the population), `reserves` (a
#' [predictive_distribution] with dimension `origin`, laid out as
#' [odp_bootstrap()]'s), `paid` and `outstanding` (the future cells of the
#' square, dimensions `origin` and `age` in months, from the same
#' simulations), `ultimate_loss_ratio` (each origin's posterior mean
#' `RLR_i RRF_i`), `rate_swaps`, `cell_rate_swaps`, `diagnostics`,
#' `summary` (also `summary(fit)`), `draws` (a matrix of the posterior
#' draws: on the natural scale `k_er`, `k_p`, `RLR`, `RRF` (population
#' values, at a zero effect), `sigma_os`, `sigma_pd`, `RLR[o]` and
#' `RRF[o]` for each origin `o` with an observed cell, with origin effects
#' on the rates `k_er[o]` and `k_p[o]` too, and with development effects
#' `dev_ker[a]`, `dev_kp[a]`, `dev_RLR[a]` and `dev_RRF[a]` for each
#' development age `a` (months) with an observed cell, on the `o` scale;
#' then `b_ker`, `b_kp`, `b_RLR`, `b_RRF`, `tau_RLR`, `tau_RRF` (and
#' `tau_ker`, `tau_kp`), `rho` (and `tau_dev_ker`, `tau_dev_kp`,
#' `tau_dev_RLR`, `tau_dev_RRF`), `log_sigma_os` and `log_sigma_pd`, a name
#' in parentheses only when the priors have that effect), and `paid_table` and `outstanding_table` (one row per
#' future cell: `origin`, `development`, `mean`, `std_dev`, `q05`, `q50`
#' and `q95`). `as.data.frame()` has one row per origin with the `premium`,
#' `latest_paid`, the `mean` and `std_dev` of the posterior predictive
#' reserve and the `ultimate_loss_ratio`; [totals_frame()] has the totals.
#' [bayes_loo()] gives the PSIS-LOO of the observed cells, outstanding
#' first, with the full log density of each response, normal or log-normal
#' (with its `-log y` term), as brms' `log_lik`, on the response's scale
#' (compare it only with fits of the same responses), plus `n_outstanding`, `elpd_outstanding` and `elpd_paid`.
#'
#' Errors: if `fitted` is not yet built; if a response is not positive
#' under the log-normal, or a paid cell has no paid at the age before
#' under incremental paid; if not exactly one of
#' `outstanding` and `incurred` is given; if a column is missing, the
#' triangle has several segments, a premium is missing or not positive, or
#' the origin and development grains differ; if a prior or sampler setting
#' is invalid; or if there are no more observed cells than parameters.
#'
#' @param triangle A cumulative [triangle] with one segment.
#' @param paid Name of the column of cumulative paid amounts.
#' @param premium Name of the column with each origin's premium (its latest
#'   value); the fit's origins are this column's.
#' @param priors A [compartmental_priors] list; required.
#' @param outstanding Name of a column of outstanding amounts.
#' @param incurred Name of a column of incurred amounts: outstanding is
#'   incurred less paid, cell by cell (the papers' outstanding on company
#'   337 is incurred with bulk and IBNR less paid). Give exactly one of
#'   `outstanding` and `incurred`.
#' @param likelihood `"normal"` (model 1) or `"lognormal"` (model 2).
#' @param scale The response scale: `"amount"` (model 1) or `"loss_ratio"`,
#'   amounts over the origin's premium (model 2).
#' @param paid_response `"cumulative"` (model 1) or `"incremental"` (model
#'   2).
#' @param fitted `"paid_and_outstanding"`; `"paid_and_incurred"`,
#'   `"paid_only"` and `"incurred_only"` are not yet built.
#' @param rate_order `"free"` or `"ordered"`, as above.
#' @param start Where the pilot run starts: `"data"`, a least-squares fit
#'   of the population curve (the prior centre if that fit fails), or
#'   `"prior_centre"`.
#' @param chains,tune,draws NUTS chains, warm-up draws per chain and kept
#'   draws per chain.
#' @param n_sims Number of predictive simulations; positive.
#' @param seed Seed of the chains and of the simulations, a non-negative
#'   whole number.
#' @param target_accept Target mean acceptance rate (the paper's
#'   `adapt_delta`, 0.99).
#' @param max_depth Largest NUTS tree depth.
#' @param ptr A `CompartmentalFit` pointer; used internally.
#' @returns A `compartmental_fit` object.
#' @seealso [compartmental_priors()], [compartmental_prior_predictive()],
#'   [meyers()], [bayes_loo()].
#' @export
#' @examples
#' long <- data.frame(year = rep(1988:1992, 5:1),
#'                    age = unlist(lapply(5:1, function(k) 12 * seq_len(k))),
#'                    paid = c(9.6, 22.8, 33.3, 40.3, 45.1, 7.9, 19.5, 29.6, 36.8,
#'                             8.7, 24.3, 35.4, 13.3, 33.0, 11.4),
#'                    incurred = c(62.7, 64.0, 65.6, 65.3, 62.3, 62.1, 56.7, 56.6, 56.8,
#'                                 64.0, 61.5, 63.2, 73.9, 75.1, 77.1),
#'                    premium = rep(c(104.4, 88.9, 86.0, 99.3, 104.9), 5:1))
#' tri <- triangle(long, "year", "age", c("paid", "incurred", "premium"))
#' # Check the priors first: the mean paid of 1988 at 60 months over its premium.
#' paths <- compartmental_prior_predictive(tri, "premium", gesmann_morris_priors(),
#'                                         n_sims = 1000)
#' quantile(marginal(paths$paid, list(origin = "1988", age = 60)), c(0.05, 0.95)) / 104.4
#' # Short chains may miss a diagnostic threshold.
#' fit <- suppressWarnings(compartmental(tri, "paid", "premium", gesmann_morris_priors(),
#'                                       incurred = "incurred", chains = 2, tune = 150,
#'                                       draws = 150, target_accept = 0.9, n_sims = 1000))
#' head(fit@summary)
#' as.data.frame(fit)
#' # Model 2, with short chains for the example (the defaults take about a
#' # minute on company 337).
#' fit2 <- suppressWarnings(compartmental_model_2(tri, "paid", "premium",
#'                                                gesmann_morris_2_priors(),
#'                                                incurred = "incurred", chains = 2,
#'                                                tune = 150, draws = 150,
#'                                                target_accept = 0.9, n_sims = 1000))
#' fit2@cell_rate_swaps
#' as.data.frame(fit2)
compartmental_fit <- S7::new_class(
  "compartmental_fit",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("CompartmentalFit"),
    origins = S7::new_property(S7::class_character, getter = function(self) self@ptr$origins()),
    premium = S7::new_property(S7::class_double, getter = function(self) self@ptr$premium()),
    latest_paid = S7::new_property(S7::class_double, getter = function(self) {
      self@ptr$latest_paid()
    }),
    population_origins = S7::new_property(S7::class_character, getter = function(self) {
      self@ptr$population_origins()
    }),
    reserves = predictive_distribution,
    paid = predictive_distribution,
    outstanding = predictive_distribution,
    ultimate_loss_ratio = S7::new_property(S7::class_double, getter = function(self) {
      self@ptr$ultimate_loss_ratio()
    }),
    rate_swaps = S7::new_property(S7::class_integer, getter = function(self) {
      as.integer(self@ptr$rate_swaps())
    }),
    cell_rate_swaps = S7::new_property(S7::class_double, getter = function(self) {
      self@ptr$cell_rate_swaps()
    }),
    diagnostics = S7::new_property(S7::class_list, getter = function(self) {
      self@ptr$diagnostics()
    }),
    summary = S7::new_property(S7::class_any, getter = function(self) {
      as.data.frame(rust_result(self@ptr$summary(), call = NULL))
    }),
    draws = S7::new_property(S7::class_any, getter = function(self) {
      nm <- self@ptr$parameter_names()
      matrix(self@ptr$draws(), ncol = length(nm), byrow = TRUE, dimnames = list(NULL, nm))
    }),
    paid_table = S7::new_property(S7::class_any, getter = function(self) {
      fit_table_frame(rust_result(self@ptr$paid_table(), call = NULL))
    }),
    outstanding_table = S7::new_property(S7::class_any, getter = function(self) {
      fit_table_frame(rust_result(self@ptr$outstanding_table(), call = NULL))
    })
  ),
  constructor = function(ptr) {
    S7::new_object(S7::S7_object(), ptr = ptr,
                   reserves = predictive_distribution(ptr = ptr$reserves()),
                   paid = predictive_distribution(ptr = ptr$paid()),
                   outstanding = predictive_distribution(ptr = ptr$outstanding()))
  }
)

# The settings and run of compartmental() and compartmental_prior_predictive().
compartmental_run <- function(chains, tune, draws, n_sims, seed, target_accept, max_depth) {
  c(single_number(chains, "chains"), single_number(tune, "tune"), single_number(draws, "draws"),
    single_number(n_sims, "n_sims"), single_number(seed, "seed"),
    single_number(target_accept, "target_accept"), single_number(max_depth, "max_depth"))
}

#' @rdname compartmental_fit
#' @export
compartmental <- function(triangle, paid, premium, priors, outstanding = NULL, incurred = NULL,
                          likelihood = c("normal", "lognormal"),
                          scale = c("amount", "loss_ratio"),
                          paid_response = c("cumulative", "incremental"),
                          fitted = c("paid_and_outstanding", "paid_and_incurred", "paid_only",
                                     "incurred_only"),
                          rate_order = c("free", "ordered"), start = c("data", "prior_centre"),
                          chains = 4, tune = 1000, draws = 2000, n_sims = 10000, seed = 0,
                          target_accept = 0.99, max_depth = 10) {
  call <- sys.call()
  inputs <- compartmental_inputs(triangle, paid, premium, priors, outstanding, incurred, call)
  settings <- c(match.arg(likelihood), match.arg(scale), match.arg(paid_response),
                match.arg(fitted), match.arg(rate_order), match.arg(start))
  ptr <- rust_result(triangle@ptr$compartmental(
    paid, inputs$os, inputs$from, premium, inputs$p$values, inputs$p$families,
    settings, compartmental_run(chains, tune, draws, n_sims, seed, target_accept, max_depth)),
    call = call)
  fit <- compartmental_fit(ptr = ptr)
  failed <- fit@diagnostics$warnings
  if (length(failed)) {
    warning("compartmental: the MCMC diagnostics failed; check them before relying on the fit: ",
            paste(failed, collapse = "; "), call. = FALSE)
  }
  fit
}

# The checked inputs shared by compartmental() and compartmental_candidate():
# the priors' wire form, the outstanding or incurred column, and which it is.
compartmental_inputs <- function(triangle, paid, premium, priors, outstanding, incurred, call) {
  check_triangle(triangle)
  for (arg in c("paid", "premium")) {
    value <- get(arg)
    if (!is.character(value) || length(value) != 1 || is.na(value)) {
      stop(sprintf("%s must be the name of a column", arg), call. = FALSE)
    }
  }
  if (missing(priors)) {
    stop(paste0("priors are required: use gesmann_morris_priors() for the paper's case study, ",
                "or compartmental_priors(), and check them with compartmental_prior_predictive()"),
         call. = FALSE)
  }
  p <- compartmental_prior_values(priors, call)
  if (is.null(outstanding) == is.null(incurred)) {
    stop(paste0("give exactly one of outstanding (a column of outstanding amounts) and ",
                "incurred (a column of incurred amounts)"), call. = FALSE)
  }
  os <- if (is.null(incurred)) outstanding else incurred
  if (!is.character(os) || length(os) != 1 || is.na(os)) {
    stop(sprintf("%s must be the name of a column",
                 if (is.null(incurred)) "outstanding" else "incurred"), call. = FALSE)
  }
  list(p = p, os = os, from = if (is.null(incurred)) "column" else "incurred")
}

#' @rdname compartmental_fit
#' @export
compartmental_model_2 <- function(triangle, paid, premium, priors, outstanding = NULL,
                                  incurred = NULL, rate_order = c("free", "ordered"),
                                  start = c("data", "prior_centre"), chains = 4, tune = 2000,
                                  draws = 8000, n_sims = 10000, seed = 0, target_accept = 0.999,
                                  max_depth = 10) {
  if (missing(priors)) {
    stop(paste0("priors are required: use gesmann_morris_2_priors() for the paper's model 2, ",
                "or compartmental_priors(), and check them with compartmental_prior_predictive()"),
         call. = FALSE)
  }
  compartmental(triangle, paid, premium, priors, outstanding = outstanding, incurred = incurred,
                likelihood = "lognormal", scale = "loss_ratio", paid_response = "incremental",
                rate_order = match.arg(rate_order), start = match.arg(start), chains = chains,
                tune = tune, draws = draws, n_sims = n_sims, seed = seed,
                target_accept = target_accept, max_depth = max_depth)
}

#' Gesmann and Morris' compartmental model over segments
#'
#' [compartmental()] fitted to every segment of a triangle, each segment
#' with its own posterior (the same model and settings, sampled
#' independently) and its premium from its own `premium` column as
#' [compartmental()] reads it, into joint posterior predictive
#' distributions of the reserves and of the future paid and outstanding
#' cells, the segments depending on each other as `dependence` says.
#' Segment `k` (in index order) takes seeds derived from `seed` and `k` for
#' its chains and its simulations, so independent segments share no random
#' numbers, and a segment's draws differ from [compartmental()] of that
#' segment alone. Under `dependence = "rank_correlation"` each segment's
#' reserves, future paid cells and future outstanding cells are reordered
#' together (one Iman-Conover reorder, ranked on the segments' total
#' reserves), so every simulation's reserve is still its paid at the
#' square's last age less the latest paid. Failed MCMC diagnostics and rate
#' swaps raise one [warning()] listing every segment's messages, each
#' prefixed with its segment's label.
#'
#' Properties of the fit: `origins`, `premium` and `latest_paid` (each
#' segment's origins in turn), `reserves`, `paid` and `outstanding`
#' ([predictive_distribution]s with the triangle's keys first among their
#' dimensions, so `aggregate(fit@reserves, keep = "lob")` keeps the
#' dependence between segments), `rate_swaps` (one count per segment) and
#' `diagnostics` (a data.frame with one row per segment: the key columns,
#' `max_rhat`, `min_ess_bulk`, `min_ess_tail`, `divergences` and
#' `warnings`, the segment's messages joined by `"; "`). `summary`, `draws`,
#' [bayes_loo()] and `paid_table` and `outstanding_table` (one row per
#' segment and future cell) are those of [compartmental_fit] and need a
#' single-segment fit for the first three: with several segments use
#' [segment()]. `as.data.frame()`, [totals_frame()] and [segment()] work as
#' for [compartmental()]. The one-year view of several segments is
#' [compartmental_one_year_segments()].
#'
#' Errors: as [compartmental()] for any segment (the message names the
#' segment); `dependence = "synchronized"`, before any sampling, since a
#' posterior has no residuals to share; or a `spearman` matrix that does not
#' fit the segments.
#'
#' @inheritParams compartmental_fit
#' @param triangle A cumulative [triangle] with any number of segments.
#' @param dependence How the segments depend on each other:
#'   `"independent"`, each segment's own posterior and simulations; or
#'   `"rank_correlation"`, the segments' simulations reordered as whole rows
#'   to the Spearman matrix `spearman`. `"synchronized"` is accepted as
#'   [odp_bootstrap()] takes it, and is an error here.
#' @param spearman With `dependence = "rank_correlation"` only: Spearman's
#'   rho between the segments' total reserves, as [odp_bootstrap()]'s.
#' @param ptr A `CompartmentalFits` pointer; used internally.
#' @returns A `compartmental_fits` object.
#' @seealso [compartmental()], [meyers_segments()].
#' @export
#' @examples
#' one <- data.frame(year = rep(1988:1992, 5:1),
#'                   age = unlist(lapply(5:1, function(k) 12 * seq_len(k))),
#'                   paid = c(9.6, 22.8, 33.3, 40.3, 45.1, 7.9, 19.5, 29.6, 36.8,
#'                            8.7, 24.3, 35.4, 13.3, 33.0, 11.4),
#'                   incurred = c(62.7, 64.0, 65.6, 65.3, 62.3, 62.1, 56.7, 56.6, 56.8,
#'                                64.0, 61.5, 63.2, 73.9, 75.1, 77.1),
#'                   premium = rep(c(104.4, 88.9, 86.0, 99.3, 104.9), 5:1))
#' two <- one
#' two[c("paid", "incurred", "premium")] <- 2 * two[c("paid", "incurred", "premium")]
#' long <- rbind(cbind(lob = "auto", one), cbind(lob = "home", two))
#' tri <- triangle(long, "year", "age", c("paid", "incurred", "premium"), keys = "lob")
#' # Short chains may miss a diagnostic threshold.
#' fits <- suppressWarnings(compartmental_segments(tri, "paid", "premium",
#'                                                 gesmann_morris_priors(),
#'                                                 incurred = "incurred", chains = 2,
#'                                                 tune = 150, draws = 150,
#'                                                 target_accept = 0.9, n_sims = 500))
#' fits@reserves@keys
#' fits@diagnostics
#' segment(fits, lob = "home")@premium
compartmental_fits <- S7::new_class(
  "compartmental_fits",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("CompartmentalFits"),
    origins = S7::new_property(S7::class_character, getter = function(self) self@ptr$origins()),
    premium = S7::new_property(S7::class_double, getter = function(self) self@ptr$premium()),
    latest_paid = S7::new_property(S7::class_double, getter = function(self) {
      self@ptr$latest_paid()
    }),
    reserves = predictive_distribution,
    paid = predictive_distribution,
    outstanding = predictive_distribution,
    rate_swaps = S7::new_property(S7::class_integer, getter = function(self) {
      as.integer(self@ptr$rate_swaps())
    }),
    diagnostics = S7::new_property(S7::class_data.frame, getter = function(self) {
      d <- self@ptr$diagnostics_table()
      out <- fit_table_frame(d$table)
      out$divergences <- as.integer(out$divergences)
      out$warnings <- d$warnings
      out
    }),
    summary = S7::new_property(S7::class_any, getter = function(self) {
      as.data.frame(rust_result(self@ptr$summary(), call = NULL))
    }),
    draws = S7::new_property(S7::class_any, getter = function(self) {
      nm <- rust_result(self@ptr$parameter_names(), call = NULL)
      matrix(rust_result(self@ptr$draws(), call = NULL), ncol = length(nm), byrow = TRUE,
             dimnames = list(NULL, nm))
    }),
    paid_table = S7::new_property(S7::class_any, getter = function(self) {
      fit_table_frame(rust_result(self@ptr$paid_table(), call = NULL))
    }),
    outstanding_table = S7::new_property(S7::class_any, getter = function(self) {
      fit_table_frame(rust_result(self@ptr$outstanding_table(), call = NULL))
    })
  ),
  constructor = function(ptr) {
    S7::new_object(S7::S7_object(), ptr = ptr,
                   reserves = predictive_distribution(ptr = ptr$reserves()),
                   paid = predictive_distribution(ptr = ptr$paid()),
                   outstanding = predictive_distribution(ptr = ptr$outstanding()))
  }
)

#' @rdname compartmental_fits
#' @export
compartmental_segments <- function(triangle, paid, premium, priors, outstanding = NULL,
                                   incurred = NULL, likelihood = c("normal", "lognormal"),
                                   scale = c("amount", "loss_ratio"),
                                   paid_response = c("cumulative", "incremental"),
                                   fitted = c("paid_and_outstanding", "paid_and_incurred",
                                              "paid_only", "incurred_only"),
                                   rate_order = c("free", "ordered"),
                                   start = c("data", "prior_centre"), chains = 4, tune = 1000,
                                   draws = 2000, n_sims = 10000, seed = 0, target_accept = 0.99,
                                   max_depth = 10,
                                   dependence = c("independent", "rank_correlation",
                                                  "synchronized"),
                                   spearman = NULL) {
  call <- sys.call()
  check_triangle(triangle)
  for (arg in c("paid", "premium")) {
    value <- get(arg)
    if (!is.character(value) || length(value) != 1 || is.na(value)) {
      stop(sprintf("%s must be the name of a column", arg), call. = FALSE)
    }
  }
  if (missing(priors)) {
    stop(paste0("priors are required: use gesmann_morris_priors() for the paper's case study, ",
                "or compartmental_priors(), and check them with compartmental_prior_predictive()"),
         call. = FALSE)
  }
  p <- compartmental_prior_values(priors, call)
  if (is.null(outstanding) == is.null(incurred)) {
    stop(paste0("give exactly one of outstanding (a column of outstanding amounts) and ",
                "incurred (a column of incurred amounts)"), call. = FALSE)
  }
  os <- if (is.null(incurred)) outstanding else incurred
  if (!is.character(os) || length(os) != 1 || is.na(os)) {
    stop(sprintf("%s must be the name of a column",
                 if (is.null(incurred)) "outstanding" else "incurred"), call. = FALSE)
  }
  settings <- c(match.arg(likelihood), match.arg(scale), match.arg(paid_response),
                match.arg(fitted), match.arg(rate_order), match.arg(start))
  dependence <- match.arg(dependence)
  ptr <- rust_result(triangle@ptr$compartmental_segments(
    paid, os, if (is.null(incurred)) "column" else "incurred", premium, p$values, p$families,
    settings, compartmental_run(chains, tune, draws, n_sims, seed, target_accept, max_depth),
    dependence, spearman_arg(spearman)), call = call)
  fit <- compartmental_fits(ptr = ptr)
  d <- fit@diagnostics
  failed <- d$warnings != ""
  if (any(failed)) {
    keys <- setdiff(names(d), c("max_rhat", "min_ess_bulk", "min_ess_tail", "divergences",
                                "warnings"))
    label <- if (length(keys)) {
      paste0(do.call(paste, c(unname(as.list(d[failed, keys, drop = FALSE])), sep = " / ")), ": ")
    } else {
      ""
    }
    warning("compartmental_segments: the MCMC diagnostics failed; check them before relying on ",
            "the fit: ", paste0(label, d$warnings[failed], collapse = "; "), call. = FALSE)
  }
  fit
}

#' Prior predictive check of the compartmental model
#'
#' Paths of the mean outstanding and cumulative paid of each origin of the
#' premium column at each age of the square, drawn from the priors alone
#' (Gesmann and Morris 2020, sec. 3.4): a check of a prior set before
#' [compartmental()] fits it. Each simulation draws the intercepts, the
#' effect sds, the correlation and each origin's effects. The paths are
#' means, without the likelihood's noise, which under the preset's
#' effectively flat prior on `log sigma` would swamp them. With
#' development effects in the priors, each simulation also draws each age's
#' effects (shared by the origins), and with effects by cell and
#' `paid_response = "incremental"` the cumulative paid path is the sum of
#' the cells' mean increments. Under `rate_order = "ordered"` the rates are
#' drawn from their priors restricted to `k_er > k_p`. The seed is derived from `seed`, as the fit's
#' simulations' is.
#'
#' Errors: if a prior setting is invalid, `n_sims` is 0, the premium column
#' is missing or not positive, or the origin and development grains differ.
#'
#' @inheritParams compartmental_fit
#' @param n_sims Number of paths; positive.
#' @param paid_response `"cumulative"` or `"incremental"`, as in the fit to
#'   be checked; it matters only with effects by cell.
#' @returns A list of two [predictive_distribution]s, `outstanding` and
#'   `paid`, each with dimensions `origin` and `age` (months).
#' @seealso [compartmental()], [gesmann_morris_priors()],
#'   [gesmann_morris_2_priors()].
#' @export
#' @examples
#' long <- data.frame(year = rep(1988:1990, 3:1), age = c(12, 24, 36, 12, 24, 12),
#'                    premium = rep(c(104.4, 88.9, 86.0), 3:1))
#' tri <- triangle(long, "year", "age", "premium")
#' paths <- compartmental_prior_predictive(tri, "premium", gesmann_morris_priors(),
#'                                         n_sims = 1000)
#' # The mean paid of 1988 at 36 months over its premium, as the preset implies.
#' quantile(marginal(paths$paid, list(origin = "1988", age = 36)), c(0.05, 0.5, 0.95)) / 104.4
compartmental_prior_predictive <- function(triangle, premium, priors,
                                           rate_order = c("free", "ordered"), n_sims = 1000,
                                           seed = 0,
                                           paid_response = c("cumulative", "incremental")) {
  call <- sys.call()
  check_triangle(triangle)
  if (!is.character(premium) || length(premium) != 1 || is.na(premium)) {
    stop("premium must be the name of a column", call. = FALSE)
  }
  if (missing(priors)) {
    stop("priors are required: use gesmann_morris_priors() for the paper's case study",
         call. = FALSE)
  }
  p <- compartmental_prior_values(priors, call)
  settings <- c("normal", "amount", match.arg(paid_response), "paid_and_outstanding",
                match.arg(rate_order), "data")
  out <- rust_result(triangle@ptr$compartmental_prior_predictive(
    premium, p$values, p$families, settings,
    compartmental_run(4, 1000, 2000, n_sims, seed, 0.99, 10)), call = call)
  list(outstanding = predictive_distribution(ptr = out$outstanding),
       paid = predictive_distribution(ptr = out$paid))
}

S7::method(as.data.frame, compartmental_fit) <- function(x, ...) {
  fit_table_frame(x@ptr$long_table())
}

#' One-year view of the compartmental model
#'
#' The claims development result over the coming year of Gesmann and
#' Morris' compartmental model ([compartmental()]), by Pareto smoothed
#' importance sampling of its posterior (decision 20 of
#' `docs/design/reserving-bayes.md`). The posterior and the lifetime
#' reserves are [compartmental()]'s for the same settings. Each simulation
#' picks a posterior draw, draws the paid and outstanding of the cells
#' valued in the next twelve months from it (every origin's next diagonal),
#' reweights every posterior draw by the likelihood of those cells, and
#' smooths the weights (one Pareto k per simulation). Origin `w`'s claims
#' development result is the expected paid to the square's last age given
#' the observed cells, less the year's paid and the same expectation given
#' the year too; its mean is about zero. An origin with no observed cell
#' (1997 on the papers' triangle) and, with development effects (model 2),
#' an age with none take their effects as part of each posterior draw, so
#' the year's cells update them.
#'
#' A simulation whose k is above `k_threshold` is flagged and handled as
#' `refit` says (as [bayes_odp_one_year()]). On company 337 about four in
#' five of model 1's simulations are flagged and all of model 2's; for
#' model 1 moment matching rescues about nineteen in twenty of them, for
#' model 2 none, and a refit takes as long as a fit (for model 2 about
#' 31 s of wall time with 8 running side by side). So by default
#' (`refit = NULL`, the user's decisions of 2026-10-16 and 2026-10-21)
#' model 1 moment-matches its flagged simulations and refits with NUTS only
#' those whose k stays above the threshold, and a model with development
#' effects (model 2) refits them with the light refit (`light_refit`): a
#' warm-started NUTS run of 4 chains, 150 warm-up and 250 kept draws,
#' within about 0.03 sd of the total claims development result of a full
#' refit, at about 20 times the run time of reweighting only (about 30
#' minutes for the default 2,000 simulations on company 337).
#' `light_refit = FALSE` turns it off: model 2 then reweights its flagged
#' simulations only, which is approximate (about 0.4 sd off a refit per
#' simulation) and warns so (`approximate`). An explicit `refit` is used as
#' it is, with the fit's own sampler unless `light_refit` is given, and
#' `max_refits` a budget. The view runs `one_year_sims` simulations, by
#' default `n_sims` capped at 2,000. `compartmental_model_2_one_year()` takes model
#' 2's settings, as [compartmental_model_2()].
#'
#' Failed MCMC diagnostics of the fit raise one [warning()] listing them,
#' refits that failed theirs raise another, simulations left unrefitted by
#' `max_refits` another, and a view that reweights only (model 2 with
#' `light_refit = FALSE`) and is approximate another.
#'
#' Properties of a `compartmental_one_year_fit`: `fit` (the
#' [compartmental_fit]), `origins` (the premium column's), per origin
#' `latest_paid`, `opening_ultimate` and `opening_reserve` (the posterior
#' mean reserve on the observed triangle); `cdr`, a
#' [predictive_distribution] of the claims development result with
#' dimension `origin` (a negative value is an adverse development);
#' `pareto_k`, `refitted`, `moment_matched` and `unrefitted` (1-based),
#' `refit_warnings`, `k_threshold` and `flagged_share`, as a
#' [bayes_one_year_fit]'s; `approximate`, why the view is approximate
#' (`NULL` unless the model's default reweighted only). `as.data.frame()` has one row per origin:
#' `origin`, `premium`, `latest_paid`, `opening_ultimate`,
#' `opening_reserve`, and the `cdr_mean` and `cdr_std_dev` of the claims
#' development result; [totals_frame()] their totals and the
#' `flagged_share`.
#'
#' `compartmental_one_year_segments()` is the view of every segment of a
#' triangle, each segment as `compartmental_one_year()` views it (its own
#' premium column and outstanding cells, the same defaults, `max_refits` a
#' budget of each segment) on the seeds of [compartmental_segments()]: a
#' segment's simulations differ from `compartmental_one_year()` of that
#' segment alone. The segments' claims development results are joined by
#' row as `dependence` says, `"independent"` as drawn or `"rank_correlation"`
#' by Iman-Conover on their totals. A `compartmental_one_year_fits` has
#' `origins`, `premium`, `latest_paid`, `opening_ultimate` and
#' `opening_reserve` (each segment's origins in turn), `cdr` (a
#' [predictive_distribution] with the triangle's keys first among its
#' dimensions), `approximate`, `k_threshold`, `diagnostics` (a data.frame
#' with one row per segment, as [compartmental_fits]'s), and for a
#' single-segment fit `pareto_k`, `refitted`, `moment_matched`,
#' `unrefitted`, `refit_warnings`, `flagged_share`, `summary` and `draws`;
#' with several segments use [segment()] or [totals_frame()], which has the
#' flagged shares. Failed diagnostics, refits and an approximate view
#' each raise one [warning()], the first two naming the segments.
#'
#' Errors: as [compartmental()]; a `k_threshold` that is `NaN`;
#' `dependence = "synchronized"`, or a `spearman` matrix that does not fit
#' the segments, before any sampling.
#'
#' This is Python's `Compartmental.one_year()`, which returns a
#' `CompartmentalOneYearFit`, and `Compartmental.one_year_segments()`, which
#' returns a `CompartmentalOneYearFits`.
#'
#' @inheritParams compartmental_fit
#' @inheritParams bayes_one_year_fit
#' @param refit What happens to a flagged simulation, as in
#'   [bayes_odp_one_year()], or `NULL` (the default) for the model's own:
#'   `"moment_match_then_refit"` without development effects (model 1),
#'   `"refit_flagged"` with the light refit with them (model 2;
#'   `light_refit = FALSE` makes it `"reweight_only"`, with a warning).
#' @param one_year_sims The number of one-year simulations, or `NULL` (the
#'   default) for `n_sims` capped at 2,000. The lifetime reserves keep
#'   `n_sims`. A smaller count gives the first simulations of a larger one.
#' @param light_refit The sampler of the refits: a shorter, warm-started NUTS
#'   run instead of the fit's own. `NULL` (the default) is the model's own:
#'   the light refit for model 2 when `refit` is `NULL`, the fit's own
#'   sampler otherwise. `TRUE` takes 4 chains of 150 warm-up and 250 kept
#'   draws, `c(chains, tune, draws)` its own. `FALSE` turns model 2's default
#'   off: with `refit = NULL` the view reweights only and is approximate,
#'   with a warning. Each refit scales its sampler by the opening posterior
#'   and starts there, with no pilot run. On company 337 it comes within
#'   about 0.03 sd of the total claims development result of a full refit of
#'   model 2. A refit warns when its closing reserve has R-hat above 1.05,
#'   fewer than 100 effective draws or a divergence.
#' @param ptr A `CompartmentalOneYearFit` or `CompartmentalOneYearFits` pointer;
#'   used internally.
#' @returns A `compartmental_one_year_fit` object, or for
#'   `compartmental_one_year_segments()` a `compartmental_one_year_fits`.
#' @seealso [compartmental()], [bayes_odp_one_year()], [meyers_one_year()].
#' @export
#' @examples
#' long <- data.frame(year = rep(1988:1992, 5:1),
#'                    age = unlist(lapply(5:1, function(k) 12 * seq_len(k))),
#'                    paid = c(9.6, 22.8, 33.3, 40.3, 45.1, 7.9, 19.5, 29.6, 36.8,
#'                             8.7, 24.3, 35.4, 13.3, 33.0, 11.4),
#'                    incurred = c(62.7, 64.0, 65.6, 65.3, 62.3, 62.1, 56.7, 56.6, 56.8,
#'                                 64.0, 61.5, 63.2, 73.9, 75.1, 77.1),
#'                    premium = rep(c(104.4, 88.9, 86.0, 99.3, 104.9), 5:1))
#' tri <- triangle(long, "year", "age", c("paid", "incurred", "premium"))
#' # Short chains, and reweighting only, for the example.
#' view <- suppressWarnings(compartmental_one_year(
#'   tri, "paid", "premium", gesmann_morris_priors(), incurred = "incurred", chains = 2,
#'   tune = 150, draws = 150, target_accept = 0.9, n_sims = 500, refit = "reweight_only"))
#' view@flagged_share
#' # The one-year view is narrower than the lifetime view.
#' c(one_year = sqrt(variance(view@cdr)), lifetime = sqrt(variance(view@fit@reserves)))
#' as.data.frame(view)
compartmental_one_year_fit <- S7::new_class(
  "compartmental_one_year_fit",
  package = "prospicio",
  properties = local({
    by_origin <- function(f) {
      S7::new_property(S7::class_double, getter = function(self) {
        stats::setNames(f(self@ptr), self@ptr$origins())
      })
    }
    list(
      ptr = S7::new_S3_class("CompartmentalOneYearFit"),
      fit = compartmental_fit,
      origins = S7::new_property(S7::class_character, getter = function(self) {
        self@ptr$origins()
      }),
      latest_paid = by_origin(function(p) p$latest_paid()),
      opening_ultimate = by_origin(function(p) p$opening_ultimate()),
      opening_reserve = by_origin(function(p) p$opening_reserve()),
      cdr = predictive_distribution,
      pareto_k = S7::new_property(S7::class_double, getter = function(self) self@ptr$pareto_k()),
      refitted = S7::new_property(S7::class_integer, getter = function(self) {
        as.integer(self@ptr$refitted())
      }),
      moment_matched = S7::new_property(S7::class_integer, getter = function(self) {
        as.integer(self@ptr$moment_matched())
      }),
      unrefitted = S7::new_property(S7::class_integer, getter = function(self) {
        as.integer(self@ptr$unrefitted())
      }),
      refit_warnings = S7::new_property(S7::class_integer, getter = function(self) {
        as.integer(self@ptr$refit_warnings())
      }),
      k_threshold = S7::new_property(S7::class_double, getter = function(self) {
        self@ptr$k_threshold()
      }),
      flagged_share = S7::new_property(S7::class_double, getter = function(self) {
        self@ptr$flagged_share()
      }),
      approximate = S7::new_property(NULL | S7::class_character, getter = function(self) {
        a <- self@ptr$approximate()
        if (length(a)) a else NULL
      })
    )
  }),
  constructor = function(ptr) {
    S7::new_object(S7::S7_object(), ptr = ptr, fit = compartmental_fit(ptr = ptr$fit()),
                   cdr = predictive_distribution(ptr = ptr$cdr()))
  }
)

#' @rdname compartmental_one_year_fit
#' @export
compartmental_one_year <- function(triangle, paid, premium, priors, outstanding = NULL,
                                   incurred = NULL, likelihood = c("normal", "lognormal"),
                                   scale = c("amount", "loss_ratio"),
                                   paid_response = c("cumulative", "incremental"),
                                   rate_order = c("free", "ordered"),
                                   start = c("data", "prior_centre"), chains = 4, tune = 1000,
                                   draws = 2000, n_sims = 10000, seed = 0,
                                   target_accept = 0.99, max_depth = 10, k_threshold = 0.7,
                                   refit = NULL, max_refits = NULL, one_year_sims = NULL,
                                   light_refit = NULL) {
  call <- sys.call()
  check_triangle(triangle)
  for (arg in c("paid", "premium")) {
    value <- get(arg)
    if (!is.character(value) || length(value) != 1 || is.na(value)) {
      stop(sprintf("%s must be the name of a column", arg), call. = FALSE)
    }
  }
  if (missing(priors)) {
    stop(paste0("priors are required: use gesmann_morris_priors() for the paper's case study, ",
                "or compartmental_priors(), and check them with compartmental_prior_predictive()"),
         call. = FALSE)
  }
  p <- compartmental_prior_values(priors, call)
  if (is.null(outstanding) == is.null(incurred)) {
    stop(paste0("give exactly one of outstanding (a column of outstanding amounts) and ",
                "incurred (a column of incurred amounts)"), call. = FALSE)
  }
  os <- if (is.null(incurred)) outstanding else incurred
  if (!is.character(os) || length(os) != 1 || is.na(os)) {
    stop(sprintf("%s must be the name of a column",
                 if (is.null(incurred)) "outstanding" else "incurred"), call. = FALSE)
  }
  settings <- c(match.arg(likelihood), match.arg(scale), match.arg(paid_response),
                "paid_and_outstanding", match.arg(rate_order), match.arg(start))
  if (!is.null(refit)) {
    refit <- match.arg(refit, c("moment_match_then_refit", "refit_flagged", "reweight_only",
                                "refit_all"))
  }
  ptr <- rust_result(triangle@ptr$compartmental_one_year(
    paid, os, if (is.null(incurred)) "column" else "incurred", premium, p$values, p$families,
    settings, compartmental_run(chains, tune, draws, n_sims, seed, target_accept, max_depth),
    single_number(k_threshold, "k_threshold"), refit, max_refits_arg(max_refits),
    optional_number(one_year_sims, "one_year_sims"), light_refit_arg(light_refit)),
    call = call)
  view <- compartmental_one_year_fit(ptr = ptr)
  failed <- view@fit@diagnostics$warnings
  if (length(failed)) {
    warning("compartmental_one_year: the MCMC diagnostics failed; check them before relying ",
            "on the fit: ", paste(failed, collapse = "; "), call. = FALSE)
  }
  warn_refits("compartmental_one_year", "", length(view@refitted), view@refit_warnings,
              length(view@unrefitted))
  if (!is.null(view@approximate)) {
    warning("compartmental_one_year: ", view@approximate, call. = FALSE)
  }
  view
}

#' @rdname compartmental_one_year_fit
#' @export
compartmental_model_2_one_year <- function(triangle, paid, premium, priors, outstanding = NULL,
                                           incurred = NULL, rate_order = c("free", "ordered"),
                                           start = c("data", "prior_centre"), chains = 4,
                                           tune = 2000, draws = 8000, n_sims = 10000, seed = 0,
                                           target_accept = 0.999, max_depth = 10,
                                           k_threshold = 0.7, refit = NULL,
                                           max_refits = NULL, one_year_sims = NULL,
                                           light_refit = NULL) {
  if (missing(priors)) {
    stop(paste0("priors are required: use gesmann_morris_2_priors() for the paper's model 2, ",
                "or compartmental_priors(), and check them with compartmental_prior_predictive()"),
         call. = FALSE)
  }
  compartmental_one_year(triangle, paid, premium, priors, outstanding = outstanding,
                         incurred = incurred, likelihood = "lognormal", scale = "loss_ratio",
                         paid_response = "incremental", rate_order = match.arg(rate_order),
                         start = match.arg(start), chains = chains, tune = tune, draws = draws,
                         n_sims = n_sims, seed = seed, target_accept = target_accept,
                         max_depth = max_depth, k_threshold = k_threshold,
                         refit = refit, max_refits = max_refits,
                         one_year_sims = one_year_sims, light_refit = light_refit)
}

#' @rdname compartmental_one_year_fit
#' @export
compartmental_one_year_fits <- S7::new_class(
  "compartmental_one_year_fits",
  package = "prospicio",
  properties = local({
    by_origin <- function(f) {
      S7::new_property(S7::class_double, getter = function(self) {
        stats::setNames(f(self@ptr), self@ptr$origins())
      })
    }
    one <- function(class, f) {
      S7::new_property(class, getter = function(self) rust_result(f(self@ptr), call = NULL))
    }
    list(
      ptr = S7::new_S3_class("CompartmentalOneYearFits"),
      origins = S7::new_property(S7::class_character, getter = function(self) {
        self@ptr$origins()
      }),
      premium = by_origin(function(p) p$premium()),
      latest_paid = by_origin(function(p) p$latest_paid()),
      opening_ultimate = by_origin(function(p) p$opening_ultimate()),
      opening_reserve = by_origin(function(p) p$opening_reserve()),
      cdr = predictive_distribution,
      approximate = S7::new_property(NULL | S7::class_character, getter = function(self) {
        a <- self@ptr$approximate()
        if (length(a)) a else NULL
      }),
      diagnostics = S7::new_property(S7::class_data.frame, getter = function(self) {
        d <- self@ptr$diagnostics_table()
        out <- fit_table_frame(d$table)
        out$divergences <- as.integer(out$divergences)
        out$warnings <- d$warnings
        out
      }),
      pareto_k = one(S7::class_double, function(p) p$pareto_k()),
      refitted = one(S7::class_integer, function(p) as.integer(rust_result(p$refitted()))),
      moment_matched = one(S7::class_integer, function(p) {
        as.integer(rust_result(p$moment_matched()))
      }),
      unrefitted = one(S7::class_integer, function(p) as.integer(rust_result(p$unrefitted()))),
      refit_warnings = one(S7::class_integer, function(p) {
        as.integer(rust_result(p$refit_warnings()))
      }),
      k_threshold = S7::new_property(S7::class_double, getter = function(self) {
        self@ptr$k_threshold()
      }),
      flagged_share = one(S7::class_double, function(p) p$flagged_share()),
      summary = S7::new_property(S7::class_any, getter = function(self) {
        as.data.frame(rust_result(self@ptr$summary(), call = NULL))
      }),
      draws = S7::new_property(S7::class_any, getter = function(self) {
        nm <- rust_result(self@ptr$parameter_names(), call = NULL)
        matrix(rust_result(self@ptr$draws(), call = NULL), ncol = length(nm), byrow = TRUE,
               dimnames = list(NULL, nm))
      })
    )
  }),
  constructor = function(ptr) {
    S7::new_object(S7::S7_object(), ptr = ptr, cdr = predictive_distribution(ptr = ptr$cdr()))
  }
)

#' @rdname compartmental_one_year_fit
#' @param dependence How the segments depend on each other:
#'   `"independent"`, each segment's own view; or `"rank_correlation"`, the
#'   segments' claims development results reordered as whole rows to the
#'   Spearman matrix `spearman` (an Iman-Conover reorder on the segments'
#'   total results). `"synchronized"` is accepted as [odp_bootstrap()] takes
#'   it, and is an error here.
#' @param spearman With `dependence = "rank_correlation"` only: Spearman's
#'   rho between the segments' total claims development results, as
#'   [odp_bootstrap()]'s.
#' @export
compartmental_one_year_segments <- function(triangle, paid, premium, priors, outstanding = NULL,
                                            incurred = NULL,
                                            likelihood = c("normal", "lognormal"),
                                            scale = c("amount", "loss_ratio"),
                                            paid_response = c("cumulative", "incremental"),
                                            rate_order = c("free", "ordered"),
                                            start = c("data", "prior_centre"), chains = 4,
                                            tune = 1000, draws = 2000, n_sims = 10000, seed = 0,
                                            target_accept = 0.99, max_depth = 10,
                                            dependence = c("independent", "rank_correlation",
                                                           "synchronized"),
                                            spearman = NULL, k_threshold = 0.7, refit = NULL,
                                            max_refits = NULL, one_year_sims = NULL,
                                            light_refit = NULL) {
  call <- sys.call()
  check_triangle(triangle)
  for (arg in c("paid", "premium")) {
    value <- get(arg)
    if (!is.character(value) || length(value) != 1 || is.na(value)) {
      stop(sprintf("%s must be the name of a column", arg), call. = FALSE)
    }
  }
  if (missing(priors)) {
    stop(paste0("priors are required: use gesmann_morris_priors() for the paper's case study, ",
                "or compartmental_priors(), and check them with compartmental_prior_predictive()"),
         call. = FALSE)
  }
  p <- compartmental_prior_values(priors, call)
  if (is.null(outstanding) == is.null(incurred)) {
    stop(paste0("give exactly one of outstanding (a column of outstanding amounts) and ",
                "incurred (a column of incurred amounts)"), call. = FALSE)
  }
  os <- if (is.null(incurred)) outstanding else incurred
  if (!is.character(os) || length(os) != 1 || is.na(os)) {
    stop(sprintf("%s must be the name of a column",
                 if (is.null(incurred)) "outstanding" else "incurred"), call. = FALSE)
  }
  settings <- c(match.arg(likelihood), match.arg(scale), match.arg(paid_response),
                "paid_and_outstanding", match.arg(rate_order), match.arg(start))
  dependence <- match.arg(dependence)
  if (!is.null(refit)) {
    refit <- match.arg(refit, c("moment_match_then_refit", "refit_flagged", "reweight_only",
                                "refit_all"))
  }
  ptr <- rust_result(triangle@ptr$compartmental_one_year_segments(
    paid, os, if (is.null(incurred)) "column" else "incurred", premium, p$values, p$families,
    settings, compartmental_run(chains, tune, draws, n_sims, seed, target_accept, max_depth),
    single_number(k_threshold, "k_threshold"), refit, max_refits_arg(max_refits),
    optional_number(one_year_sims, "one_year_sims"), light_refit_arg(light_refit), dependence,
    spearman_arg(spearman)),
    call = call)
  view <- compartmental_one_year_fits(ptr = ptr)
  d <- view@diagnostics
  keys <- setdiff(names(d), c("max_rhat", "min_ess_bulk", "min_ess_tail", "divergences",
                              "warnings"))
  labels <- if (length(keys)) {
    paste0(do.call(paste, c(unname(as.list(d[, keys, drop = FALSE])), sep = " / ")), ": ")
  } else {
    rep("", nrow(d))
  }
  failed <- d$warnings != ""
  if (any(failed)) {
    warning("compartmental_one_year_segments: the MCMC diagnostics failed; check them before ",
            "relying on the fit: ", paste0(labels[failed], d$warnings[failed], collapse = "; "),
            call. = FALSE)
  }
  counts <- view@ptr$refit_counts()
  warn_refits("compartmental_one_year_segments", labels, counts$refits, counts$warned,
              counts$unrefitted)
  if (!is.null(view@approximate)) {
    warning("compartmental_one_year_segments: ", view@approximate, call. = FALSE)
  }
  view
}

S7::method(as.data.frame, compartmental_one_year_fits) <- function(x, ...) {
  fit_table_frame(x@ptr$long_table())
}
S7::method(summary, compartmental_one_year_fits) <- function(object, ...) object@summary
S7::method(print, compartmental_one_year_fits) <- function(x, ...) {
  cdr <- x@cdr
  cat(sprintf("<compartmental_one_year_fits> %d simulations%s
", as.integer(cdr@n_sims),
              segments_note(x@ptr)))
  cat(sprintf("total opening reserve %s; claims development result mean %s, sd %s
",
              format(sum(x@opening_reserve), digits = 10), format(mean(cdr), digits = 10),
              format(sqrt(variance(cdr)), digits = 10)))
  print(totals_frame(x), row.names = FALSE)
  invisible(x)
}
S7::method(as.data.frame, compartmental_one_year_fit) <- function(x, ...) {
  fit_table_frame(x@ptr$long_table())
}
S7::method(print, compartmental_one_year_fit) <- function(x, ...) {
  cat(sprintf(paste0("<compartmental_one_year_fit> %d simulations, flagged share %s ",
                     "(%d moment-matched, %d refitted, %d unrefitted)\n"),
              length(x@pareto_k), format(x@flagged_share, digits = 3),
              length(x@moment_matched), length(x@refitted), length(x@unrefitted)))
  print(as.data.frame(x), row.names = FALSE)
  invisible(x)
}
S7::method(as.data.frame, compartmental_fits) <- function(x, ...) {
  fit_table_frame(x@ptr$long_table())
}
S7::method(summary, compartmental_fits) <- function(object, ...) object@summary
S7::method(print, compartmental_fits) <- function(x, ...) {
  r <- x@reserves
  d <- x@diagnostics
  cat(sprintf("<compartmental_fits> %d simulations%s
", as.integer(r@n_sims),
              segments_note(x@ptr)))
  cat(sprintf("diagnostics: max R-hat %s, min bulk ESS %s, min tail ESS %s, %d divergences%s
",
              format(max(d$max_rhat), digits = 4), format(min(d$min_ess_bulk), digits = 4),
              format(min(d$min_ess_tail), digits = 4), as.integer(sum(d$divergences)),
              if (any(d$warnings != "")) {
                sprintf(", warnings in %d segments", sum(d$warnings != ""))
              } else {
                ""
              }))
  cat(sprintf("total reserve: posterior mean %s, sd %s
", format(mean(r), digits = 10),
              format(sqrt(variance(r)), digits = 10)))
  print(totals_frame(x), row.names = FALSE)
  invisible(x)
}
S7::method(summary, compartmental_fit) <- function(object, ...) object@summary
S7::method(print, compartmental_fit) <- function(x, ...) {
  r <- x@reserves
  d <- x@diagnostics
  cat(sprintf("<compartmental_fit> %d draws in %d chains, %d simulations\n", nrow(x@draws),
              as.integer(x@ptr$chains()), as.integer(r@n_sims)))
  cat(sprintf("diagnostics: max R-hat %s, min bulk ESS %s, min tail ESS %s, %d divergences%s\n",
              format(d$max_rhat, digits = 4), format(d$min_ess_bulk, digits = 4),
              format(d$min_ess_tail, digits = 4), as.integer(d$divergences),
              if (length(d$warnings)) sprintf(", %d warnings", length(d$warnings)) else ""))
  cat(sprintf("total reserve: posterior mean %s, sd %s\n", format(mean(r), digits = 10),
              format(totals_frame(x)$std_dev, digits = 6)))
  print(as.data.frame(x), row.names = FALSE)
  invisible(x)
}

# The model settings of a diagonal backtest ----------------------------------

backtest_model_spec <- S7::new_class("backtest_model_spec", package = "prospicio", abstract = TRUE)

#' Models for the diagonal backtest
#'
#' Settings objects that say which models [diagonal_backtest()] fits on
#' each training triangle. They hold the settings of the model and no data:
#' the triangle, the column and the premium come from the backtest, and so
#' do the number of simulations and the seed, so these functions take the
#' settings of [bayes_odp()] and [meyers()] without `triangle`, `column`,
#' `premium`, `n_sims` and `seed`, with the same names, defaults and
#' checks.
#'
#' * `odp_glm_spec()` is the over-dispersed Poisson GLM of the incremental
#'   values with an intercept, an origin factor and a development factor
#'   (the chain ladder's model), fitted by quasi-likelihood. It has no
#'   sampler, so the backtest reports no diagnostics for it.
#' * `bayes_odp_spec()` is the Bayesian chain ladder of [bayes_odp()].
#' * `meyers_spec()` is Meyers' CRC, CSR or CAY model of [meyers()]; the
#'   backtest needs its `premium`.
#'
#' @param intercept_sd,origin_sd,development_sd Standard deviations of the
#'   normal priors of [bayes_odp()]: on the intercept, on each origin effect
#'   and on each development effect. For `meyers_spec()`, `origin_sd` and
#'   `development_sd` are those of the priors on `alpha[w]` and `beta[d]`.
#' @param process Process error of [bayes_odp()]: `"gamma"` or `"none"`.
#' @param model Meyers' model: `"crc"`, `"csr"` or `"cay"`.
#' @param logelr_mean,logelr_sd Mean and standard deviation of the normal
#'   prior on the log expected loss ratio.
#' @param gamma_sd Standard deviation of the normal prior on `gamma`
#'   (`"csr"` only).
#' @param rho_shape Both shapes of the beta prior on `(1 - rho) / 2`
#'   (`"cay"` only).
#' @param chains,tune,draws NUTS chains, warm-up draws per chain and kept
#'   draws per chain.
#' @param target_accept Target mean acceptance rate of the step-size
#'   adaptation.
#' @param max_depth Largest NUTS tree depth.
#' @returns A `bayes_odp_spec`, `meyers_spec` or `odp_glm_spec` object, to
#'   put in the `models` list of [diagonal_backtest()].
#' @seealso [diagonal_backtest()], [bayes_odp()], [meyers()].
#' @export
#' @examples
#' bayes_odp_spec(chains = 2, tune = 200, draws = 200)
#' meyers_spec(model = "csr")
#' odp_glm_spec()
odp_glm_spec <- S7::new_class(
  "odp_glm_spec",
  package = "prospicio",
  parent = backtest_model_spec,
  constructor = function() S7::new_object(S7::S7_object())
)

#' @rdname odp_glm_spec
#' @export
bayes_odp_spec <- S7::new_class(
  "bayes_odp_spec",
  package = "prospicio",
  parent = backtest_model_spec,
  properties = list(
    intercept_sd = S7::class_double, origin_sd = S7::class_double,
    development_sd = S7::class_double, process = S7::class_character,
    chains = S7::class_double, tune = S7::class_double, draws = S7::class_double,
    target_accept = S7::class_double, max_depth = S7::class_double
  ),
  constructor = function(intercept_sd = 10, origin_sd = 10, development_sd = 10,
                         process = c("gamma", "none"), chains = 4, tune = 1000, draws = 1000,
                         target_accept = 0.95, max_depth = 10) {
    process <- match.arg(process)
    k <- bayes_odp_settings(intercept_sd, origin_sd, development_sd, chains, tune, draws,
                            target_accept, max_depth)
    S7::new_object(S7::S7_object(), intercept_sd = k$intercept_sd, origin_sd = k$origin_sd,
                   development_sd = k$development_sd, process = process, chains = k$chains,
                   tune = k$tune, draws = k$draws, target_accept = k$target_accept,
                   max_depth = k$max_depth)
  }
)

#' @rdname odp_glm_spec
#' @export
meyers_spec <- S7::new_class(
  "meyers_spec",
  package = "prospicio",
  parent = backtest_model_spec,
  properties = list(
    model = S7::class_character, logelr_mean = S7::class_double, logelr_sd = S7::class_double,
    origin_sd = S7::class_double, development_sd = S7::class_double, gamma_sd = S7::class_double,
    rho_shape = S7::class_double, chains = S7::class_double, tune = S7::class_double,
    draws = S7::class_double, target_accept = S7::class_double, max_depth = S7::class_double
  ),
  constructor = function(model = c("crc", "csr", "cay"), logelr_mean = -0.4, logelr_sd = 3.162,
                         origin_sd = 3.162, development_sd = 3.162, gamma_sd = 0.05,
                         rho_shape = 2, chains = 4, tune = 1000, draws = 2000,
                         target_accept = 0.95, max_depth = 10) {
    model <- match.arg(model)
    k <- meyers_settings(logelr_mean, logelr_sd, origin_sd, development_sd, gamma_sd, rho_shape,
                         chains, tune, draws, target_accept, max_depth)
    S7::new_object(S7::S7_object(), model = model, logelr_mean = k$logelr_mean,
                   logelr_sd = k$logelr_sd, origin_sd = k$origin_sd,
                   development_sd = k$development_sd, gamma_sd = k$gamma_sd,
                   rho_shape = k$rho_shape, chains = k$chains, tune = k$tune, draws = k$draws,
                   target_accept = k$target_accept, max_depth = k$max_depth)
  }
)

sampler_line <- function(x) {
  sprintf("%s chains of %s draws after %s warm-up, target_accept %s, max_depth %s",
          format(x@chains), format(x@draws), format(x@tune), format(x@target_accept),
          format(x@max_depth))
}

S7::method(print, odp_glm_spec) <- function(x, ...) {
  cat("<odp_glm_spec> over-dispersed Poisson GLM: intercept, origin factor, development factor\n")
  invisible(x)
}

S7::method(print, bayes_odp_spec) <- function(x, ...) {
  cat(sprintf("<bayes_odp_spec> prior sds %s (intercept), %s (origin), %s (development), %s process\n",
              format(x@intercept_sd), format(x@origin_sd), format(x@development_sd), x@process))
  cat(sampler_line(x), "\n", sep = "")
  invisible(x)
}

S7::method(print, meyers_spec) <- function(x, ...) {
  cat(sprintf("<meyers_spec> %s, logelr ~ N(%s, %s), origin sd %s, development sd %s",
              x@model, format(x@logelr_mean), format(x@logelr_sd), format(x@origin_sd),
              format(x@development_sd)))
  cat(switch(x@model, csr = sprintf(", gamma sd %s", format(x@gamma_sd)),
             cay = sprintf(", rho shape %s", format(x@rho_shape)), ""), "\n", sep = "")
  cat(sampler_line(x), "\n", sep = "")
  invisible(x)
}


#' Gesmann and Morris' compartmental model in the diagonal backtest
#'
#' [compartmental()]'s model as a model of [diagonal_backtest()]. Unlike the
#' other model specs it holds the data it reads, the triangle and its
#' `paid`, outstanding (or incurred) and `premium` columns, since its
#' outstanding and premium are not in the backtest's table of paid
#' increments. Put it in the `models` of [diagonal_backtest()], on the same
#' `triangle` with `paid` as the `column`, to score the paid increments of
#' the held-out diagonals beside the other models, and call
#' [compartmental_outstanding_backtest()] for a second table, the
#' held-out outstanding of the same diagonals. Both tables use **the same
#' fits**: the candidate keeps the fit of each split (one run of the
#' sampler), whichever table is asked for first.
#'
#' The settings are [compartmental()]'s, except that the backtest's `n_sims`
#' and `seed` replace the candidate's, and the sampler runs once per split
#' (the whole `chains` by `draws` posterior), so size it for the number of
#' diagonals. The held-out diagonal's outstanding and paid never reach the
#' fit. The candidate's name in the results is `"compartmental_gm1"` for
#' model 1, `"compartmental_gm2"` for model 2 (the log-normal likelihood,
#' loss ratios and incremental paid of [compartmental_model_2()]) and
#' `"compartmental"` for other settings.
#'
#' @inheritParams compartmental_fit
#' @param name The candidate's name in the backtest.
#' @param ptr A `CompartmentalBacktestModel` pointer; used internally.
#' @returns A `compartmental_backtest_model` object for the `models` of
#'   [diagonal_backtest()] and [compartmental_outstanding_backtest()], with
#'   properties `name`, `paid` and `premium`.
#' @seealso [diagonal_backtest()], [compartmental()].
#' @export
#' @examples
#' long <- data.frame(year = rep(1988:1992, 5:1),
#'                    age = unlist(lapply(5:1, function(k) 12 * seq_len(k))),
#'                    paid = c(9.6, 22.8, 33.3, 40.3, 45.1, 7.9, 19.5, 29.6, 36.8,
#'                             8.7, 24.3, 35.4, 13.3, 33.0, 11.4),
#'                    incurred = c(62.7, 64.0, 65.6, 65.3, 62.3, 62.1, 56.7, 56.6, 56.8,
#'                                 64.0, 61.5, 63.2, 73.9, 75.1, 77.1),
#'                    premium = rep(c(104.4, 88.9, 86.0, 99.3, 104.9), 5:1))
#' tri <- triangle(long, "year", "age", c("paid", "incurred", "premium"))
#' cand <- compartmental_candidate(tri, "paid", "premium", gesmann_morris_priors(),
#'                                 incurred = "incurred", chains = 2, tune = 150,
#'                                 draws = 150, target_accept = 0.9)
#' cand
#' \donttest{
#' paid <- diagonal_backtest(tri, "paid", list(cand), n_diagonals = 1, n_sims = 200)
#' paid@mean
#' compartmental_outstanding_backtest(cand, n_diagonals = 1, n_sims = 200)@mean
#' }
compartmental_backtest_model <- S7::new_class(
  "compartmental_backtest_model",
  package = "prospicio",
  parent = backtest_model_spec,
  properties = list(
    ptr = S7::new_S3_class("CompartmentalBacktestModel"),
    name = S7::new_property(S7::class_character, getter = function(self) self@ptr$name()),
    paid = S7::new_property(S7::class_character, getter = function(self) self@ptr$paid()),
    premium = S7::new_property(S7::class_character, getter = function(self) self@ptr$premium())
  ),
  constructor = function(ptr) S7::new_object(S7::S7_object(), ptr = ptr)
)

#' @rdname compartmental_backtest_model
#' @export
compartmental_candidate <- function(triangle, paid, premium, priors, outstanding = NULL,
                                    incurred = NULL, likelihood = c("normal", "lognormal"),
                                    scale = c("amount", "loss_ratio"),
                                    paid_response = c("cumulative", "incremental"),
                                    rate_order = c("free", "ordered"),
                                    start = c("data", "prior_centre"), chains = 4, tune = 1000,
                                    draws = 2000, target_accept = 0.99, max_depth = 10) {
  call <- sys.call()
  inputs <- compartmental_inputs(triangle, paid, premium, priors, outstanding, incurred, call)
  settings <- c(match.arg(likelihood), match.arg(scale), match.arg(paid_response),
                "paid_and_outstanding", match.arg(rate_order), match.arg(start))
  ptr <- rust_result(triangle@ptr$compartmental_candidate(
    paid, inputs$os, inputs$from, premium, inputs$p$values, inputs$p$families, settings,
    compartmental_run(chains, tune, draws, 1, 0, target_accept, max_depth)), call = call)
  compartmental_backtest_model(ptr = ptr)
}

#' @rdname compartmental_backtest_model
#' @export
compartmental_model_2_candidate <- function(triangle, paid, premium, priors, outstanding = NULL,
                                            incurred = NULL, rate_order = c("free", "ordered"),
                                            start = c("data", "prior_centre"), chains = 4,
                                            tune = 2000, draws = 8000, target_accept = 0.999,
                                            max_depth = 10) {
  if (missing(priors)) {
    stop(paste0("priors are required: use gesmann_morris_2_priors() for the paper's model 2, ",
                "or compartmental_priors(), and check them with compartmental_prior_predictive()"),
         call. = FALSE)
  }
  compartmental_candidate(triangle, paid, premium, priors, outstanding = outstanding,
                          incurred = incurred, likelihood = "lognormal", scale = "loss_ratio",
                          paid_response = "incremental", rate_order = match.arg(rate_order),
                          start = match.arg(start), chains = chains, tune = tune, draws = draws,
                          target_accept = target_accept, max_depth = max_depth)
}

#' Held-out outstanding of a compartmental candidate
#'
#' The second table of [compartmental_candidate()] in the diagonal backtest:
#' for each of the latest `n_diagonals` calendar diagonals, the model is
#' fitted to the cells valued before it (the outstanding of the held-out
#' diagonal is always withheld) and forecasts the outstanding of the
#' diagonal's cells, scored by the metrics of [diagonal_backtest()]. The
#' splits, `n_sims`, `seed` and `interval` are [diagonal_backtest()]'s, so a
#' backtest with the same arguments and this candidate in its `models` has
#' fitted the model already and the fits are reused. The one model of the
#' table is named after the candidate with `_outstanding`. A held-out cell
#' without an outstanding in the triangle, or without a fitted origin or
#' development age, is counted in `excluded`.
#'
#' @param candidate A [compartmental_candidate()].
#' @inheritParams diagonal_backtest
#' @returns A [backtest] object.
#' @seealso [compartmental_candidate()], [diagonal_backtest()].
#' @export
#' @examples
#' \donttest{
#' long <- data.frame(year = rep(1988:1992, 5:1),
#'                    age = unlist(lapply(5:1, function(k) 12 * seq_len(k))),
#'                    paid = c(9.6, 22.8, 33.3, 40.3, 45.1, 7.9, 19.5, 29.6, 36.8,
#'                             8.7, 24.3, 35.4, 13.3, 33.0, 11.4),
#'                    incurred = c(62.7, 64.0, 65.6, 65.3, 62.3, 62.1, 56.7, 56.6, 56.8,
#'                                 64.0, 61.5, 63.2, 73.9, 75.1, 77.1),
#'                    premium = rep(c(104.4, 88.9, 86.0, 99.3, 104.9), 5:1))
#' tri <- triangle(long, "year", "age", c("paid", "incurred", "premium"))
#' cand <- compartmental_candidate(tri, "paid", "premium", gesmann_morris_priors(),
#'                                 incurred = "incurred", chains = 2, tune = 150,
#'                                 draws = 150, target_accept = 0.9)
#' compartmental_outstanding_backtest(cand, n_diagonals = 1, n_sims = 200)@mean
#' }
compartmental_outstanding_backtest <- function(candidate, n_diagonals, n_sims = 1000, seed = 0,
                                               interval = 0.9) {
  if (!S7::S7_inherits(candidate, compartmental_backtest_model)) {
    stop("candidate must be a compartmental_candidate()", call. = FALSE)
  }
  backtest(rust_result(candidate@ptr$outstanding_backtest(
    single_number(n_diagonals, "n_diagonals"), single_number(n_sims, "n_sims"),
    single_number(seed, "seed"), single_number(interval, "interval"))))
}

S7::method(print, compartmental_backtest_model) <- function(x, ...) {
  cat(sprintf("<compartmental_candidate> %s, paid column \"%s\", premium column \"%s\"
",
              x@name, x@paid, x@premium))
  invisible(x)
}

# The default name of a spec in a backtest, and the list the binding reads.
backtest_spec_name <- function(spec) {
  if (S7::S7_inherits(spec, odp_glm_spec)) {
    "odp"
  } else if (S7::S7_inherits(spec, bayes_odp_spec)) {
    "bayes_odp"
  } else if (S7::S7_inherits(spec, compartmental_backtest_model)) {
    spec@name
  } else {
    paste0("meyers_", spec@model)
  }
}

backtest_spec_wire <- function(spec, name) {
  run <- function(x) c(x@chains, x@tune, x@draws, 1, 0, x@target_accept, x@max_depth)
  if (S7::S7_inherits(spec, odp_glm_spec)) {
    list(name = name, kind = "odp_glm", model = "", process = "", priors = numeric(),
         run = numeric())
  } else if (S7::S7_inherits(spec, bayes_odp_spec)) {
    list(name = name, kind = "bayes_odp", model = "", process = spec@process,
         priors = c(spec@intercept_sd, spec@origin_sd, spec@development_sd), run = run(spec))
  } else if (S7::S7_inherits(spec, compartmental_backtest_model)) {
    list(name = name, kind = "compartmental", model = "", process = "", priors = numeric(),
         run = numeric(), ptr = spec@ptr)
  } else {
    list(name = name, kind = "meyers", model = spec@model, process = "",
         priors = c(spec@logelr_mean, spec@logelr_sd, spec@origin_sd, spec@development_sd,
                    spec@gamma_sd, spec@rho_shape),
         run = run(spec))
  }
}

# Diagonal backtest ------------------------------------------------------------

#' Diagonal backtest
#'
#' The retrospective validation of Meyers (2015): each model is fitted on a
#' triangle cut back to an earlier calendar diagonal and forecasts the
#' incremental values of the diagonal that came in next, and the forecast
#' is scored against what happened. This is done for the latest
#' `n_diagonals` calendar diagonals (oldest first), for every model in
#' `models`, with the same `n_sims` joint draws from `seed` for each model
#' and split, so the models share their random numbers.
#'
#' A held-out cell whose origin or development age has no training cell
#' that every model fits on cannot be forecast by a model with origin and
#' development effects (the newest origin and the oldest age on the
#' diagonal, or an age whose only training cell is a negative increment the
#' Poisson GLM drops). It is left out for every model: `excluded` counts
#' them, and `scored_rows` lists the cells that were scored.
#'
#' The metrics, per model and held-out diagonal (`scores`, and their means
#' over the diagonals in `mean`):
#'
#' * `crps`: the mean continuous ranked probability score of the cells'
#'   draws.
#' * `coverage`: the share of cells inside the central `interval` of their
#'   draws.
#' * `actual_vs_expected`: actual over expected on the diagonal's total.
#' * `total_crps`: the CRPS of the diagonal total, from the draws summed
#'   per simulation.
#' * `mean_log_density`: the mean over the scored cells of the model's
#'   held-out log density (`NA` for a model that gives none).
#' * `total_percentile`: the share of the simulated diagonal totals at or
#'   below the actual total, the probability integral transform of the
#'   diagonal: uniform over many triangles for a calibrated model.
#'
#' Meyers' own statistic, the percentile of the outcome's ultimate, is a
#' different calculation: it comes from a fit on the whole upper triangle,
#' and is the cdf of that fit's predictive total at the known outcome:
#' `cdf(total(fit@ultimates), outcome)` for [meyers()] ([cdf()], [total()]),
#' or `cdf(total(fit@reserves), outcome - latest)` for [bayes_odp()], with
#' `latest` the total of the triangle's latest diagonal.
#'
#' A Bayesian fit whose MCMC diagnostics fail (R-hat above 1.01, bulk or
#' tail ESS below 100 per chain, a divergent transition) is still scored,
#' unlike [bayes_odp()], which warns: a study over many triangles wants
#' every split and a list of the doubtful ones. `diagnostics` has one row
#' per model and split, and `warnings` lists every message. A model that
#' fails to fit is an error naming it. No R warning is raised for a failed
#' diagnostic; read `warnings`.
#'
#' A [compartmental_candidate()] is scored on the paid increments here, and
#' on the held-out outstanding by [compartmental_outstanding_backtest()],
#' which reuses the fits of this backtest.
#'
#' Errors: a `models` that is not a non-empty list of [odp_glm_spec()],
#' [bayes_odp_spec()], [meyers_spec()] and [compartmental_candidate()]
#' objects; two models with the same name (checked before any fit); a
#' [compartmental_candidate()] built on another triangle or paid column than
#' `triangle` and `column`; a triangle with several segments or with fewer
#' than `n_diagonals + 1` calendar diagonals; `premium` missing for
#' [meyers_spec()], or not positive and finite; an invalid sampler setting.
#'
#' @param triangle A cumulative [triangle] with one segment.
#' @param column Name of the column to forecast.
#' @param models A list of [odp_glm_spec()], [bayes_odp_spec()],
#'   [meyers_spec()] and [compartmental_candidate()] objects. Its names are
#'   the models' names in the results; an entry without a name is called
#'   `"odp"`, `"bayes_odp"`, `"meyers_"` and the model's name
#'   (`"meyers_crc"`) or the candidate's name (`"compartmental_gm1"`). Two
#'   models cannot share a name.
#' @param n_diagonals Number of latest calendar diagonals to hold out, one
#'   split each.
#' @param premium Premium of each origin, for [meyers_spec()]: one number
#'   per origin, or the name of a column of `triangle` whose latest value
#'   per origin is taken as [meyers()] takes it. `NULL` when no model needs
#'   it.
#' @param n_sims Number of joint draws of each forecast.
#' @param seed Seed of the sampler chains and of the draws, a non-negative
#'   whole number.
#' @param interval Width of the central interval of the `coverage` metric,
#'   in (0, 1).
#' @param raw The Rust result; used internally.
#' @returns A `backtest` object with properties:
#'   `models` (the names), `metrics`, `n_splits`, `diagonals` (the calendar
#'   period held out in each split, as a label such as `"2021"`, as the
#'   `calendar` column of the cells labels it, oldest first), `scores` (a data.frame in long form: `model`, `diagonal`,
#'   `metric`, `value`), `mean` (a data.frame: `model` and one column per
#'   metric, the mean over the diagonals), `excluded` (held-out cells left
#'   out of each split), `scored_rows` (per split, the scored cells as
#'   positions in the row-major grid of origins by development ages, from
#'   1; Python's `scored_rows` count from 0), `log_densities` (a list by model of the held-out log densities of
#'   every scored cell, the splits in order and each model's values for
#'   the same cells, `NULL` where the model gives none), `diagnostics` (a
#'   data.frame: `model`, `diagonal`, `max_rhat`, `min_ess_bulk`,
#'   `min_ess_tail`, `divergences` and `n_warnings`; one row for every model
#'   and split, `NA` for a model with no sampler) and `warnings` (a
#'   data.frame: `model`, `diagonal`, `message`).
#' @seealso [odp_glm_spec()], [bayes_odp()], [meyers()].
#' @export
#' @examples
#' long <- data.frame(year = rep(2016:2021, 6:1),
#'                    age = unlist(lapply(6:1, function(k) 12 * seq_len(k))),
#'                    paid = c(952, 1529, 2813, 3647, 3724, 3832, 849, 1564, 2202, 2432,
#'                             2468, 983, 2211, 2830, 3832, 1657, 2685, 3169, 932, 1940, 1162))
#' tri <- triangle(long, "year", "age", "paid")
#' bt <- diagonal_backtest(tri, "paid", list(odp = odp_glm_spec()), n_diagonals = 2,
#'                         n_sims = 500, seed = 1)
#' bt@mean
#' bt@diagnostics
backtest <- S7::new_class(
  "backtest",
  package = "prospicio",
  properties = list(
    models = S7::class_character, metrics = S7::class_character, n_splits = S7::class_integer,
    diagonals = S7::class_character, scores = S7::class_data.frame, mean = S7::class_data.frame,
    excluded = S7::class_integer, scored_rows = S7::class_list, log_densities = S7::class_list,
    diagnostics = S7::class_data.frame, warnings = S7::class_data.frame
  ),
  constructor = function(raw) {
    models <- raw$models
    metrics <- raw$metrics
    nm <- length(models)
    nk <- length(metrics)
    ns <- as.integer(raw$n_splits)
    diagonals <- as.character(raw$diagonals)
    # Rust lays the scores out model, then metric, then split.
    values <- array(raw$scores, dim = c(ns, nk, nm))
    rows <- expand.grid(metric = seq_len(nk), split = seq_len(ns), model = seq_len(nm))
    scores <- data.frame(model = models[rows$model], diagonal = diagonals[rows$split],
                         metric = metrics[rows$metric],
                         value = values[cbind(rows$split, rows$metric, rows$model)],
                         stringsAsFactors = FALSE)
    scores$value[is.nan(scores$value)] <- NA_real_
    means <- data.frame(model = models, stringsAsFactors = FALSE)
    for (j in seq_len(nk)) {
      means[[metrics[j]]] <- vapply(seq_len(nm), function(m) {
        v <- mean(values[, j, m])
        if (is.nan(v)) NA_real_ else v
      }, numeric(1))
    }
    each <- expand.grid(split = seq_len(ns), model = seq_len(nm))
    no_nan <- function(x) replace(x, is.nan(x), NA_real_)
    count <- function(x) as.integer(no_nan(x))
    diagnostics <- data.frame(model = models[each$model], diagonal = diagonals[each$split],
                              max_rhat = no_nan(raw$max_rhat),
                              min_ess_bulk = no_nan(raw$min_ess_bulk),
                              min_ess_tail = no_nan(raw$min_ess_tail),
                              divergences = count(raw$divergences),
                              n_warnings = count(raw$n_warnings), stringsAsFactors = FALSE)
    warnings <- data.frame(model = models[as.integer(raw$warning_model) + 1L],
                           diagonal = diagonals[as.integer(raw$warning_split) + 1L],
                           message = raw$warning_message, stringsAsFactors = FALSE)
    log_densities <- raw$log_densities
    names(log_densities) <- models
    S7::new_object(S7::S7_object(), models = models, metrics = metrics, n_splits = ns,
                   diagonals = diagonals, scores = scores, mean = means,
                   excluded = as.integer(raw$excluded),
                   scored_rows = lapply(raw$scored_rows, function(r) as.integer(r) + 1L),
                   log_densities = log_densities, diagnostics = diagnostics, warnings = warnings)
  }
)

#' @rdname backtest
#' @export
diagonal_backtest <- function(triangle, column, models, n_diagonals, premium = NULL,
                              n_sims = 1000, seed = 0, interval = 0.9) {
  check_triangle(triangle)
  if (!is.character(column) || length(column) != 1 || is.na(column)) {
    stop("column must be the name of a column", call. = FALSE)
  }
  if (!is.list(models) || length(models) == 0) {
    stop("models must be a non-empty list of model specs: odp_glm_spec(), bayes_odp_spec(), ",
         "meyers_spec() or compartmental_candidate()", call. = FALSE)
  }
  for (spec in models) {
    if (!S7::S7_inherits(spec, backtest_model_spec)) {
      stop("every entry of models must be odp_glm_spec(), bayes_odp_spec(), meyers_spec() or ",
           "compartmental_candidate()", call. = FALSE)
    }
  }
  given <- names(models)
  if (is.null(given)) given <- rep("", length(models))
  given <- ifelse(is.na(given) | given == "", vapply(models, backtest_spec_name, ""), given)
  repeated <- unique(given[duplicated(given)])
  if (length(repeated)) {
    stop(sprintf("two models are named %s: name each model once, as in list(a = ..., b = ...)",
                 paste0('"', repeated, '"', collapse = ", ")), call. = FALSE)
  }
  if (is.character(premium)) {
    if (length(premium) != 1 || is.na(premium) || !premium %in% triangle@columns) {
      stop("premium must be a number per origin or the name of a column of the triangle",
           call. = FALSE)
    }
    latest <- latest_diagonal(triangle)
    if (dim(latest)[1] != 1) {
      stop("the backtest needs a triangle with one segment", call. = FALSE)
    }
    premium <- as.numeric(latest[1, premium, ])
  } else if (!is.null(premium) && (!is.numeric(premium) || !is.null(dim(premium)))) {
    stop("premium must be NULL, a numeric vector with one value per origin, or a column name",
         call. = FALSE)
  }
  if (is.null(premium) && any(vapply(models, S7::S7_inherits, NA, meyers_spec))) {
    stop("Meyers needs premium, one value per origin", call. = FALSE)
  }
  specs <- unname(Map(backtest_spec_wire, models, given))
  raw <- rust_result(triangle@ptr$diagonal_backtest(
    column, if (is.null(premium)) NULL else as.double(premium), specs,
    single_number(n_diagonals, "n_diagonals"), single_number(n_sims, "n_sims"),
    single_number(seed, "seed"), single_number(interval, "interval")))
  backtest(raw)
}

S7::method(print, backtest) <- function(x, ...) {
  cat(sprintf("<backtest> %d model%s, %d held-out diagonal%s; mean scores:\n",
              length(x@models), if (length(x@models) == 1) "" else "s", x@n_splits,
              if (x@n_splits == 1) "" else "s"))
  print(x@mean, row.names = FALSE)
  n <- nrow(x@warnings)
  if (n) {
    cat(sprintf("%d diagnostics warning%s (see `warnings`); the splits were scored all the same\n",
                n, if (n == 1) "" else "s"))
  }
  invisible(x)
}
