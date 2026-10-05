# Reserving lane: the loss triangle, the chain ladder, Mack and the ODP
# bootstrap, over
# crates/act-r/src/reserving.rs (docs/design/triangle.md). S7 classes and
# functions over the Rust objects, as in distributions.R.

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

# Index labels as a list of character vectors (one per label): from a
# character vector of one-part labels, a list of parts, or a data.frame
# with one row per label.
label_list <- function(index) {
  if (is.data.frame(index)) {
    parts <- lapply(index, as.character)
    lapply(seq_len(nrow(index)), function(r) vapply(parts, `[`, "", r, USE.NAMES = FALSE))
  } else if (is.list(index)) {
    lapply(index, as.character)
  } else {
    as.list(as.character(index))
  }
}

#' Loss triangle
#'
#' A loss triangle with four axes, in chainladder-python's order: index
#' (segment, such as a line of business) x column (measure, such as paid
#' and incurred) x origin period x development age in months. Cells that
#' are not observed are `NA`; a zero is an observation.
#'
#' `triangle()` builds one from a long table, one row per (index, origin,
#' development). Origins span every period from the earliest to the latest
#' row and ages every development period from the youngest to the oldest.
#' Rows with the same (index, origin, development) are summed, and `NA`
#' values are missing. Incremental input (`cumulative = FALSE`) follows
#' chainladder-python: a missing row is a period without movement.
#'
#' Read-only properties: `x@shape` (index, column, origin and development
#' lengths), `x@index` (a data.frame of index labels, one row per index
#' position), `x@columns`, `x@origins` (period labels such as `"1981"`,
#' `"2021Q3"`, `"2021H2"` or `"2021-08"`), `x@development` (ages in months),
#' `x@valuation` (the last day of the latest valuation month), `x@origin_grain`,
#' `x@development_grain`, `x@is_cumulative` and `x@values` (a 4-D array with
#' dimnames). `as.data.frame()` gives the long table back.
#'
#' @param data A data.frame with one row per (index, origin, development).
#' @param origin Name of the origin column: Date, POSIXct (any day in the
#'   origin period) or whole-number years.
#' @param development Name of the development column: ages in months (12,
#'   24, ...), or valuation dates when `development_is_valuation = TRUE`.
#' @param columns Names of the value columns (numeric).
#' @param index Names of the index columns, or `NULL` for a single segment
#'   labelled `"Total"`. Several columns make multi-part labels.
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
#'   [subset()][subset.triangle].
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
triangle <- S7::new_class(
  "triangle",
  package = "actuarialrs",
  properties = list(
    ptr = S7::new_S3_class("Triangle"),
    index_names = S7::class_character,
    shape = S7::new_property(S7::class_integer, getter = function(self) {
      stats::setNames(self@ptr$shape(), c("index", "column", "origin", "development"))
    }),
    index = S7::new_property(S7::class_data.frame, getter = function(self) {
      parts <- self@ptr$index()
      names(parts) <- if (length(self@index_names)) self@index_names else "index"
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
  constructor = function(data, origin, development, columns, index = NULL,
                         origin_grain = "Y", development_grain = "Y",
                         cumulative = TRUE, development_is_valuation = FALSE,
                         ptr = NULL) {
    if (!is.null(ptr)) {
      return(S7::new_object(S7::S7_object(), ptr = ptr, index_names = as.character(index)))
    }
    if (!is.data.frame(data)) stop("data must be a data.frame", call. = FALSE)
    missing_cols <- setdiff(c(origin, development, columns, index), names(data))
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
    labels <- unlist(lapply(index, function(c) as.character(data[[c]])))
    if (anyNA(labels)) stop("index has missing values", call. = FALSE)
    values <- unlist(lapply(columns, numeric_column, what = "value"))
    ptr <- rust_result(Triangle$from_long(
      as.character(labels), as.double(length(index)), o$year, o$month,
      ages, v$year, v$month, isTRUE(development_is_valuation),
      as.character(columns), as.double(values),
      as.character(origin_grain), as.character(development_grain), isTRUE(cumulative)
    ))
    S7::new_object(S7::S7_object(), ptr = ptr, index_names = as.character(index))
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

new_triangle <- function(x, ptr) triangle(ptr = ptr, index = x@index_names)

S7::method(print, triangle) <- function(x, ...) {
  s <- x@shape
  cat(sprintf("<triangle> %d index x %d column x %d origin x %d development, %s, valuation %s\n",
              s[1], s[2], s[3], s[4], if (x@is_cumulative) "cumulative" else "incremental",
              format(x@valuation)))
  cat(sprintf("origin grain %s, development grain %s\n", x@origin_grain, x@development_grain))
  if (s[1] == 1 && s[2] == 1) {
    v <- x@values[1, 1, , , drop = FALSE]
    print(matrix(v, s[3], s[4], dimnames = dimnames(x@values)[3:4]))
  }
  invisible(x)
}

S7::method(as.data.frame, triangle) <- function(x, ...) {
  long <- x@ptr$to_long()
  out <- list()
  if (length(x@index_names)) out <- stats::setNames(long$index, x@index_names)
  out$origin <- month_start(long$origin_year, long$origin_month)
  out$development <- long$development
  values <- lapply(long$values, function(v) replace(v, is.nan(v), NA_real_))
  out[long$names] <- values
  as.data.frame(out, stringsAsFactors = FALSE, optional = TRUE)
}

#' Subset a triangle
#'
#' Keeps the named index positions and columns, in the order given. Naming
#' a label or column twice, or one that does not exist, is an error.
#'
#' @param x A [triangle].
#' @param index Index labels to keep, or `NULL` for all: a character vector
#'   of one-part labels, a list of character vectors (the parts of each
#'   label), or a data.frame with one row per label.
#' @param columns Column names to keep, or `NULL` for all.
#' @param ... Unused.
#' @returns A [triangle].
#' @name subset.triangle
#' @examples
#' long <- data.frame(lob = rep(c("auto", "home"), each = 3), year = c(2020, 2020, 2021),
#'                    age = c(12, 24, 12), paid = 1:6, incurred = 2 * (1:6))
#' tri <- triangle(long, "year", "age", c("paid", "incurred"), index = "lob")
#' subset(tri, index = "home", columns = "incurred")@values
NULL

S7::method(subset, triangle) <- function(x, index = NULL, columns = NULL, ...) {
  labels <- if (is.null(index)) NULL else label_list(index)
  cols <- if (is.null(columns)) NULL else as.character(columns)
  # Name the generic in errors, not the S7-registered method.
  call <- sys.call()
  call[[1]] <- quote(subset)
  new_triangle(x, rust_result(x@ptr$slice(labels, cols), call = call))
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

# Chain-ladder properties, read through `cl(self)`, a ChainLadderFit
# pointer. Shared by chain_ladder_fit and mack_fit.
chain_ladder_properties <- function(cl) {
  links <- function(self) {
    ages <- cl(self)$development()
    paste(ages[-length(ages)], ages[-1], sep = "-")
  }
  by_origin <- function(f) {
    S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(f(cl(self)), cl(self)$origins())
    })
  }
  list(
    origins = S7::new_property(S7::class_character, getter = function(self) cl(self)$origins()),
    development = S7::new_property(S7::class_integer, getter = function(self) cl(self)$development()),
    ldf = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(cl(self)$ldf(), links(self))
    }),
    cdf = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(cl(self)$cdf(), paste0(cl(self)$development(), "-Ult"))
    }),
    sigma = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(cl(self)$sigma(), links(self))
    }),
    std_err = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(cl(self)$std_err(), links(self))
    }),
    alpha = S7::new_property(S7::class_double, getter = function(self) cl(self)$alpha()),
    tail = S7::new_property(S7::class_double, getter = function(self) cl(self)$tail()),
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
#' Properties of the fit, per origin (named by origin): `latest`,
#' `ultimate`, `reserve`; per link (named `"12-24"` and so on): `ldf`,
#' `sigma` (with unestimable ones interpolated), `std_err`; per age:
#' `cdf` (age to ultimate, including the tail); and `origins`,
#' `development`, `alpha`, `tail`, `total_ultimate`, `total_reserve`.
#' `as.data.frame()` gives the per-origin table.
#'
#' @param triangle A single-segment [triangle] (use
#'   [subset()][subset.triangle] first if it has several index positions).
#' @param column Name of the column to fit; by default the only one.
#' @param average `"volume"`, `"simple"` or `"regression"`.
#' @param sigma_interpolation How a variance parameter that cannot be
#'   estimated (an age with a single link ratio) is filled in:
#'   `"log-linear"` (regress `log(sigma)` on age and extrapolate, the
#'   default of R ChainLadder) or `"mack"` (Mack 1993).
#' @param tail Factor from the oldest age to ultimate; 1 means no tail.
#' @param ptr A `ChainLadderFit` pointer; used internally.
#' @returns A `chain_ladder_fit` object.
#' @seealso [mack()] for standard errors.
#' @export
#' @examples
#' long <- data.frame(year = c(2020, 2020, 2020, 2021, 2021, 2022),
#'                    age = c(12, 24, 36, 12, 24, 12),
#'                    paid = c(100, 150, 165, 110, 170, 120))
#' fit <- chain_ladder(triangle(long, "year", "age", "paid"), tail = 1.05)
#' fit@ldf
#' fit@reserve
#' fit@total_reserve
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
  if (!is.numeric(tail) || length(tail) != 1 || is.na(tail)) {
    stop("tail must be a single number", call. = FALSE)
  }
  ptr <- rust_result(triangle@ptr$chain_ladder(column, args$average, args$sigma, as.double(tail)))
  chain_ladder_fit(ptr = ptr)
}

#' Mack chain ladder
#'
#' Mack's distribution-free chain ladder (Mack 1993, 1999): the chain-ladder
#' projection, without a tail, plus the process and parameter standard
#' errors of each origin's reserve and of the total, as R ChainLadder's
#' `MackChainLadder()`. Needs at least three development ages, and every
#' sigma estimable or fillable by `sigma_interpolation`.
#'
#' The fit has the properties of a [chain_ladder_fit] (and the fit itself
#' as `m@chain_ladder`), plus per origin `process_risk`, `parameter_risk`
#' and `standard_error`, and `total_process_risk`, `total_parameter_risk`,
#' `total_standard_error` and `total_cv` (the total standard error over the
#' total reserve). Risks are standard errors, not variances.
#'
#' @inheritParams chain_ladder_fit
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
mack_fit <- S7::new_class(
  "mack_fit",
  package = "actuarialrs",
  properties = c(
    list(ptr = S7::new_S3_class("MackFit"), chain_ladder = chain_ladder_fit),
    chain_ladder_properties(function(self) self@chain_ladder@ptr),
    list(
      process_risk = S7::new_property(S7::class_double, getter = function(self) {
        stats::setNames(self@ptr$process_risk(), self@origins)
      }),
      parameter_risk = S7::new_property(S7::class_double, getter = function(self) {
        stats::setNames(self@ptr$parameter_risk(), self@origins)
      }),
      standard_error = S7::new_property(S7::class_double, getter = function(self) {
        stats::setNames(self@ptr$standard_error(), self@origins)
      }),
      total_process_risk = S7::new_property(S7::class_double, getter = function(self) {
        self@ptr$total_process_risk()
      }),
      total_parameter_risk = S7::new_property(S7::class_double, getter = function(self) {
        self@ptr$total_parameter_risk()
      }),
      total_standard_error = S7::new_property(S7::class_double, getter = function(self) {
        self@ptr$total_standard_error()
      }),
      total_cv = S7::new_property(S7::class_double, getter = function(self) self@ptr$total_cv())
    )
  ),
  constructor = function(ptr) {
    S7::new_object(S7::S7_object(), ptr = ptr, chain_ladder = chain_ladder_fit(ptr = ptr$chain_ladder()))
  }
)

#' @rdname mack_fit
#' @export
mack <- function(triangle, column = NULL, average = "volume", sigma_interpolation = "log-linear") {
  column <- fit_column(triangle, column)
  args <- development_args(average, sigma_interpolation)
  mack_fit(ptr = rust_result(triangle@ptr$mack(column, args$average, args$sigma)))
}

fit_frame <- function(fit) {
  data.frame(origin = fit@origins, latest = unname(fit@latest),
             ultimate = unname(fit@ultimate), reserve = unname(fit@reserve))
}

#' ODP bootstrap
#'
#' Over-dispersed Poisson bootstrap of the chain ladder (England and Verrall
#' 2002), as R ChainLadder's `BootChainLadder()`: adjusted Pearson residuals
#' of the volume-weighted chain ladder are resampled into pseudo triangles,
#' each is re-projected, and process error is added to every future
#' incremental value. Simulation `i` uses random stream `i` of `seed`, so
#' results do not depend on the number of threads. The scale and residuals
#' match `BootChainLadder()` exactly, and the reserve distribution within
#' Monte Carlo error (`validation/reference/reserving_bootstrap_r.csv`).
#'
#' Properties of the fit: `chain_ladder` (the [chain_ladder_fit] the
#' bootstrap is centred on), `origins`, `development`, `fitted` (fitted
#' incremental values) and `residuals` (adjusted Pearson residuals
#' `(x - m) / sqrt(|m|) * sqrt(n / (n - p))`), both origin x development
#' matrices with `NA` where not observed, `scale` (the dispersion `phi`) and
#' `reserves`, a [predictive_distribution] of the reserve with dimension
#' `origin` and one component per origin period. `mean()`, `quantile()`,
#' [VaR()] and [TVaR()] of `reserves` describe the total reserve. Columns of
#' [draw_matrix()] follow `origins` ([marginal()] does not match origin labels
#' yet).
#'
#' @param triangle A single-segment cumulative [triangle], every origin
#'   observed from the first age up to its latest.
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
      origin_matrix(self, self@ptr$fitted())
    }),
    residuals = S7::new_property(S7::class_double, getter = function(self) {
      origin_matrix(self, self@ptr$residuals())
    }),
    scale = S7::new_property(S7::class_double, getter = function(self) self@ptr$scale()),
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

S7::method(as.data.frame, chain_ladder_fit) <- function(x, ...) fit_frame(x)
S7::method(as.data.frame, mack_fit) <- function(x, ...) {
  cbind(fit_frame(x), process_risk = unname(x@process_risk),
        parameter_risk = unname(x@parameter_risk), standard_error = unname(x@standard_error))
}
S7::method(print, chain_ladder_fit) <- function(x, ...) {
  cat(sprintf("<chain_ladder_fit> total reserve %s, tail %s\n",
              format(x@total_reserve, digits = 10), format(x@tail)))
  print(as.data.frame(x), row.names = FALSE)
  invisible(x)
}
S7::method(print, odp_bootstrap_fit) <- function(x, ...) {
  r <- x@reserves
  cat(sprintf("<odp_bootstrap_fit> %d simulations, scale %s\n",
              as.integer(r@n_sims), format(x@scale, digits = 6)))
  cat(sprintf("total reserve: chain ladder %s, bootstrap mean %s, sd %s\n",
              format(x@chain_ladder@total_reserve, digits = 10), format(mean(r), digits = 10),
              format(sqrt(variance(r)), digits = 10)))
  invisible(x)
}
S7::method(print, mack_fit) <- function(x, ...) {
  cat(sprintf("<mack_fit> total reserve %s, standard error %s (CV %s)\n",
              format(x@total_reserve, digits = 10), format(x@total_standard_error, digits = 10),
              format(x@total_cv, digits = 4)))
  print(as.data.frame(x), row.names = FALSE)
  invisible(x)
}
