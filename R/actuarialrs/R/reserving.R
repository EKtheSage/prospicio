# Reserving lane: the loss triangle, the chain ladder, Mack with its
# one-year view, the expected-loss methods and the ODP bootstrap, over
# crates/act-r/src/reserving.rs (docs/design/triangle.md,
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
  package = "actuarialrs",
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
    alpha = S7::new_property(S7::class_double, getter = function(self) cl(self)$alpha()),
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

development_args <- function(average, sigma_interpolation) {
  list(average = match.arg(average, c("volume", "simple", "regression")),
       sigma = match.arg(sigma_interpolation, c("log-linear", "mack")))
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
  package = "actuarialrs",
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
  package = "actuarialrs",
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
  package = "actuarialrs",
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
  package = "actuarialrs",
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
#' Properties of the fit, per origin: `latest`, `ultimate`, `reserve`; per
#' link (named `"12-24"` and so on): `ldf` (the selected factors, which the
#' projection uses: the estimated ones, replaced by the tail's from its
#' attachment age), `estimated_ldf` (the factors as estimated, before the
#' tail replaced any), `sigma` (with unestimable ones interpolated),
#' `std_err`; per age: `cdf` (age to ultimate, including the tail); the
#' tail: `tail` (the factor from the oldest age to ultimate),
#' `tail_attachment_age` (the age from which `ldf` holds the tail's
#' factors; the oldest age when it replaced none), `tail_ldf` (the factors
#' past the oldest age, which multiply to `tail`: one per development
#' period of the next year and one to ultimate, or a single one for
#' [tail_log_linear()]), `tail_sigma` and `tail_std_err` (the tail's
#' variance parameter and standard error, extrapolated log-linearly; 0
#' without a tail, a factor of 1; a tail below 1 is read where a tail of
#' 1.001 would be, as chainladder-python does); and `keys`, `index` (one
#' row per segment, as a triangle's), `origins`, `development`, `alpha`,
#' `total_ultimate` and `total_reserve` (summed over segments). The
#' per-link, per-age and tail properties need a single-segment fit: with
#' several segments use [development_frame()] (`ldf`, `cdf`, `sigma` and
#' `std_err`; the oldest age has the tail as its `cdf`), [totals_frame()]
#' (`tail`, `tail_sigma` and `tail_std_err`) or [segment()].
#' `as.data.frame()` gives one row per segment and origin, [totals_frame()]
#' one per segment.
#'
#' @param triangle A [triangle], with any number of segments.
#' @param column Name of the column to fit; by default the only one.
#' @param average `"volume"`, `"simple"` or `"regression"`.
#' @param sigma_interpolation How a variance parameter that cannot be
#'   estimated (an age with a single link ratio) is filled in:
#'   `"log-linear"` (regress `log(sigma)` on age and extrapolate, the
#'   default of R ChainLadder) or `"mack"` (Mack 1993).
#' @param tail Development past the oldest age: a number, the factor from
#'   the oldest age to ultimate (1 means no tail), or a [tail_constant()],
#'   [tail_curve()], [tail_bondy()] or [tail_log_linear()].
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
chain_ladder_fit <- S7::new_class(
  "chain_ladder_fit",
  package = "actuarialrs",
  properties = c(
    list(ptr = S7::new_S3_class("ChainLadderFit")),
    chain_ladder_properties(function(self) self@ptr)
  ),
  constructor = function(ptr) S7::new_object(S7::S7_object(), ptr = ptr)
)

#' @rdname chain_ladder_fit
#' @export
chain_ladder <- function(triangle, column = NULL, average = "volume",
                         sigma_interpolation = "log-linear", tail = 1) {
  column <- fit_column(triangle, column)
  args <- development_args(average, sigma_interpolation)
  ptr <- rust_result(triangle@ptr$chain_ladder(column, args$average, args$sigma, tail_ptr(tail)))
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
  package = "actuarialrs",
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
                 tail = 1, tail_sigma = NULL, tail_std_err = NULL) {
  column <- fit_column(triangle, column)
  args <- development_args(average, sigma_interpolation)
  ptr <- rust_result(triangle@ptr$mack(
    column, args$average, args$sigma, tail_ptr(tail),
    optional_number(tail_sigma, "tail_sigma"), optional_number(tail_std_err, "tail_std_err")
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
#' @param average,sigma_interpolation,tail The development pattern, as in
#'   [chain_ladder()].
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
  package = "actuarialrs",
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
                          sigma_interpolation = "log-linear", tail = 1) {
  cols <- exposure_columns(triangle, column, exposure)
  args <- development_args(average, sigma_interpolation)
  ptr <- rust_result(triangle@ptr$expected_loss(
    cols[1], cols[2], single_number(apriori, "apriori"), args$average, args$sigma,
    tail_ptr(tail)
  ))
  expected_loss_fit(ptr = ptr)
}

#' @rdname expected_loss_fit
#' @export
bornhuetter_ferguson <- function(triangle, column, exposure, apriori = 1, average = "volume",
                                 sigma_interpolation = "log-linear", tail = 1) {
  cols <- exposure_columns(triangle, column, exposure)
  args <- development_args(average, sigma_interpolation)
  ptr <- rust_result(triangle@ptr$bornhuetter_ferguson(
    cols[1], cols[2], single_number(apriori, "apriori"), args$average, args$sigma,
    tail_ptr(tail)
  ))
  expected_loss_fit(ptr = ptr)
}

#' @rdname expected_loss_fit
#' @export
benktander <- function(triangle, column, exposure, apriori = 1, n_iters = 1, average = "volume",
                       sigma_interpolation = "log-linear", tail = 1) {
  cols <- exposure_columns(triangle, column, exposure)
  args <- development_args(average, sigma_interpolation)
  ptr <- rust_result(triangle@ptr$benktander(
    cols[1], cols[2], single_number(apriori, "apriori"), single_number(n_iters, "n_iters"),
    args$average, args$sigma, tail_ptr(tail)
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
  package = "actuarialrs",
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
                     sigma_interpolation = "log-linear", tail = 1) {
  cols <- exposure_columns(triangle, column, exposure)
  args <- development_args(average, sigma_interpolation)
  ptr <- rust_result(triangle@ptr$cape_cod(
    cols[1], cols[2], single_number(trend, "trend"), single_number(decay, "decay"),
    args$average, args$sigma, tail_ptr(tail)
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
#' @seealso [mack()].
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
  package = "actuarialrs",
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
#' Every segment of the triangle is bootstrapped on its own, with its own
#' residuals and scale, into one joint distribution of the reserves.
#' Simulation `i` uses random stream `i` of `seed` for every segment in
#' turn, so results do not depend on the number of threads.
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
#' @param ptr An `OdpBootstrapFit` pointer; used internally.
#' @returns An `odp_bootstrap_fit` object.
#' @seealso [chain_ladder()], [mack()].
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
  package = "actuarialrs",
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
                          process = c("gamma", "none")) {
  column <- fit_column(triangle, column)
  process <- match.arg(process)
  for (arg in c("n_sims", "seed")) {
    value <- get(arg)
    if (!is.numeric(value) || length(value) != 1 || is.na(value)) {
      stop(sprintf("%s must be a single number", arg), call. = FALSE)
    }
  }
  ptr <- rust_result(triangle@ptr$odp_bootstrap(column, as.double(n_sims), as.double(seed), process))
  odp_bootstrap_fit(ptr = ptr)
}

check_fit <- function(fit) {
  classes <- list(chain_ladder_fit, mack_fit, expected_loss_fit, cape_cod_fit, odp_bootstrap_fit)
  if (!any(vapply(classes, function(cls) S7::S7_inherits(fit, cls), TRUE))) {
    stop("fit must be a chain_ladder_fit, mack_fit, expected_loss_fit, cape_cod_fit or ",
         "odp_bootstrap_fit", call. = FALSE)
  }
}

#' Long results of a fit over every segment
#'
#' Tables of a [chain_ladder_fit], [mack_fit], [expected_loss_fit],
#' [cape_cod_fit] or [odp_bootstrap_fit] with the triangle's key columns by
#' name. `as.data.frame(fit)` has one row per segment and origin: `origin`,
#' `latest`, `ultimate` and `reserve` (the method's own), plus for Mack
#' `process_risk`, `parameter_risk` and `standard_error`, for the
#' expected-loss methods `exposure` and `apriori` (and Cape Cod's
#' `trended_apriori`), and for the bootstrap the `mean` and `std_dev` of the
#' bootstrapped reserve. `totals_frame()` has one row per segment with the
#' same quantities for the segment's total (for the expected-loss methods
#' the total `exposure`, for the bootstrap also its `scale`).
#' `development_frame()` has one row per segment and age: `development`,
#' `ldf` (to the next age), `cdf` (to ultimate, with the tail), `sigma` and
#' `std_err`; the oldest age has `NA` for `ldf`, `sigma` and `std_err`.
#'
#' `segment()` returns the fit of one segment, chosen by key values as in
#' `segment(fit, lob = "auto")` (compared as character). Keys not named may
#' take any value, so a fit with one segment needs none; a choice that
#' matches several segments is an error. For the bootstrap, the segment
#' keeps its part of the joint `reserves`, with the same dimensions.
#'
#' These are Python's `to_frame()`, `totals_frame()`,
#' `development_frame()` and `segment(**keys)`.
#'
#' @param fit A [chain_ladder_fit], [mack_fit], [expected_loss_fit],
#'   [cape_cod_fit] or [odp_bootstrap_fit].
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
