#' @include odp_glm.R reserving.R reserving_inflation.R
NULL

#' The age-period-cohort reserving GLM
#'
#' Fits `log mu_ij = delta + alpha_i + beta_j + gamma_(i+j)` to the
#' increments of a full triangle: accident (cohort), development (age) and
#' calendar (period) effects, with an over-dispersed Poisson error or the
#' gamma and Tweedie errors of [odp_glm()] (issue #979, design note
#' `docs/design/reserving-inflation.md`). Python's `ApcGlm`.
#'
#' The calendar index is the sum of the accident and development indices,
#' so the three effects are identified only up to linear trends. The model
#' is fitted in the canonical parameterisation of Kuang, Nielsen and
#' Nielsen (2008a), as Python and R `apc` fit it: a `level` (the log mean of
#' the first origin's first age), an `age_slope` and an `accident_slope`
#' (the steps from age 0 to age 1 and from accident 0 to accident 1), then
#' the second differences of each effect at `l = 2, ..., k - 1` on a `k` by
#' `k` triangle. Fitted values and the deviance do not depend on the
#' identification.
#'
#' The future cells need the calendar effect past the latest diagonal.
#' `extrapolation` chooses how its second differences continue, by rules
#' that do not depend on the identification (Kuang, Nielsen and Nielsen
#' 2008b; R `apc`'s `apc.forecast.apc`):
#'
#' * `"I2"`: the future second differences are 0, so the last linear trend
#'   of the calendar effect continues.
#' * `"I1"`: the next first difference of the calendar effect is the mean
#'   of the fitted ones (the first left out, as R `apc` does), then the
#'   second differences are 0.
#' * `"I0"` (the default, R `apc`'s): the future calendar effects lie on the
#'   least-squares line through the effects of the periods from the third
#'   on. It needs at least 4 origins; the other rules need 3.
#'
#' `effects` drops one effect's second differences: `"AP"` (development and
#' calendar, Taylor's separation model as a GLM), `"AC"` (accident and
#' development, the chain ladder, with no calendar effect to extrapolate)
#' or `"PC"` (accident and calendar). The names are as Python and R `apc`
#' give them, in either case.
#'
#' The triangle must be full: as many ages as origins, the origin grain
#' equal to the development grain, no hole. The fit is an ODP GLM on the
#' canonical design, `odp_fit`: its `prediction_error` (by the delta
#' method), [predict_distribution()] and `residuals` cover the extrapolated
#' calendar effect as a function of the fitted parameters.
#'
#' @param triangle A single-segment [triangle()].
#' @param column The column to fit; `NULL` for the triangle's only one.
#' @param effects `"APC"` (the default), `"AP"`, `"AC"` or `"PC"`.
#' @param extrapolation `"I0"` (the default), `"I1"` or `"I2"`.
#' @param family The error: `"poisson"` (the default), `"gamma"` or
#'   `"tweedie"`.
#' @param power The Tweedie power, in (1, 2); only for `family = "tweedie"`.
#' @param tolerance Convergence tolerance of the IRLS fit.
#' @param max_iterations Most IRLS iterations.
#' @param origins,ages,level,age_slope,accident_slope,age_second_differences,accident_second_differences,calendar_second_differences,future_calendar_second_differences,fitted,deviance,reserves,future,odp_fit
#'   The fields of an `apc_glm_fit`, as described under Value. A fit is made
#'   by `apc_glm()`; the `apc_glm_fit()` constructor only holds its fields.
#' @returns An `apc_glm_fit` with the properties `effects` and
#'   `extrapolation` (in upper case), `origins`, `ages` (months), `level`,
#'   `age_slope`, `accident_slope`, `age_second_differences`,
#'   `accident_second_differences` and `calendar_second_differences` (at
#'   `l = 2, ..., k - 1`; empty for an effect the model drops),
#'   `future_calendar_second_differences` (the two extrapolated ones at
#'   `l = k` and `k + 1`, all later ones 0; zeros for `"I2"` and without a
#'   calendar effect), `fitted` (a matrix of the fitted means, origins by
#'   ages, `NA` on the future cells), `deviance`, `reserves` (the sum of
#'   each origin's future means; the first origin's is 0, with no tail past
#'   the oldest age), `total_reserve`, `coefficients` (the canonical
#'   parameters, named `level`, `slope_age`, `slope_accident`, then
#'   `dd_age[l]`, `dd_accident[l]` and `dd_calendar[l]`), `covariance`
#'   (their covariance matrix), `dispersion` (Pearson's), `future` (the
#'   future cells as [future_cells()]), `odp_fit` (the [odp_glm_fit] on the
#'   canonical design) and `prediction_error` (`odp_fit`'s: a data frame
#'   with a row per origin and a `Total` row).
#' @seealso [odp_glm()], [separation()], [future_cells()]
#' @export
#' @examples
#' # A 5 by 5 triangle of increments with a 5% calendar trend and noise.
#' shape <- c(0.5, 0.25, 0.12, 0.08, 0.05)
#' noise <- c(1.04, 0.97, 1.01, 0.95, 1.03)
#' long <- do.call(rbind, lapply(0:4, function(i) {
#'   j <- 0:(4 - i)
#'   data.frame(year = 2001 + i, age = 12 * (j + 1),
#'              paid = 1000 * 1.05^(i + j) * shape[j + 1] * noise[(i + 2 * j) %% 5 + 1])
#' }))
#' tri <- triangle(long, "year", "age", "paid", cumulative = FALSE)
#' fit <- apc_glm(tri, extrapolation = "I1")
#' fit@reserves
#' fit@calendar_second_differences
#' fit@prediction_error
#'
#' # The chain ladder as the accident and development model.
#' apc_glm(tri, effects = "AC")@total_reserve
apc_glm_fit <- S7::new_class(
  "apc_glm_fit",
  package = "prospicio",
  properties = list(
    effects = S7::class_character,
    extrapolation = S7::class_character,
    origins = S7::class_character,
    ages = S7::class_integer,
    level = S7::class_double,
    age_slope = S7::class_double,
    accident_slope = S7::class_double,
    age_second_differences = S7::class_double,
    accident_second_differences = S7::class_double,
    calendar_second_differences = S7::class_double,
    future_calendar_second_differences = S7::class_double,
    fitted = S7::class_any,
    deviance = S7::class_double,
    reserves = S7::class_double,
    future = future_cells,
    odp_fit = odp_glm_fit,
    total_reserve = S7::new_property(S7::class_double, getter = function(self) sum(self@reserves)),
    coefficients = S7::new_property(S7::class_double, getter = function(self) {
      self@odp_fit@coefficients
    }),
    covariance = S7::new_property(S7::class_double, getter = function(self) {
      self@odp_fit@covariance
    }),
    dispersion = S7::new_property(S7::class_double, getter = function(self) {
      self@odp_fit@dispersion
    }),
    prediction_error = S7::new_property(S7::class_data.frame, getter = function(self) {
      self@odp_fit@prediction_error
    })
  ),
  constructor = function(effects, extrapolation, origins, ages, level, age_slope, accident_slope,
                         age_second_differences, accident_second_differences,
                         calendar_second_differences, future_calendar_second_differences,
                         fitted, deviance, reserves, future, odp_fit) {
    S7::new_object(
      S7::S7_object(),
      effects = effects, extrapolation = extrapolation, origins = origins, ages = ages,
      level = level, age_slope = age_slope, accident_slope = accident_slope,
      age_second_differences = age_second_differences,
      accident_second_differences = accident_second_differences,
      calendar_second_differences = calendar_second_differences,
      future_calendar_second_differences = future_calendar_second_differences,
      fitted = fitted, deviance = deviance, reserves = reserves, future = future,
      odp_fit = odp_fit
    )
  }
)

S7::method(print, apc_glm_fit) <- function(x, ...) {
  cat(sprintf("apc_glm_fit: %s, %s, %d origins, total_reserve = %s\n", x@effects,
              x@extrapolation, length(x@origins), format(x@total_reserve, digits = 15)))
  invisible(x)
}

#' @rdname apc_glm_fit
#' @export
apc_glm <- function(triangle, column = NULL, effects = "APC", extrapolation = "I0",
                    family = "poisson", power = NULL, tolerance = 1e-10, max_iterations = 100) {
  column <- fit_column(triangle, column)
  if (!is.character(effects) || length(effects) != 1L) {
    stop("effects must be \"APC\", \"AP\", \"AC\" or \"PC\"", call. = FALSE)
  }
  if (!is.character(extrapolation) || length(extrapolation) != 1L) {
    stop("extrapolation must be \"I0\", \"I1\" or \"I2\"", call. = FALSE)
  }
  if (!is.character(family) || length(family) != 1L) {
    stop("family must be \"poisson\", \"gamma\" or \"tweedie\"", call. = FALSE)
  }
  if (identical(family, "tweedie")) {
    if (is.null(power)) stop("family = \"tweedie\" needs power", call. = FALSE)
    power <- single_number(power, "power")
  } else {
    if (!is.null(power)) stop("power is only for family = \"tweedie\"", call. = FALSE)
    power <- NaN
  }
  tolerance <- single_number(tolerance, "tolerance")
  max_iterations <- single_number(max_iterations, "max_iterations")
  out <- reserving_result(apc_glm_fit_rust(triangle@ptr, column, effects, extrapolation, family,
                                           power, tolerance, max_iterations))
  n <- length(out$origins)
  fitted <- matrix(out$fitted, n, length(out$ages), byrow = TRUE,
                   dimnames = list(out$origins, out$ages))
  fitted[is.nan(fitted)] <- NA
  names(out$reserves) <- out$origins
  future <- future_cells(
    data.frame(origin = out$future$origin, age = out$future$age,
               valuation = out$future$valuation, value = out$future$value,
               stringsAsFactors = FALSE),
    origins = out$future$origins
  )
  apc_glm_fit(
    effects = out$effects,
    extrapolation = out$extrapolation,
    origins = out$origins,
    ages = as.integer(out$ages),
    level = out$level,
    age_slope = out$age_slope,
    accident_slope = out$accident_slope,
    age_second_differences = out$age_second_differences,
    accident_second_differences = out$accident_second_differences,
    calendar_second_differences = out$calendar_second_differences,
    future_calendar_second_differences = out$future_calendar_second_differences,
    fitted = fitted,
    deviance = out$deviance,
    reserves = out$reserves,
    future = future,
    odp_fit = odp_glm_fit(ptr = out$odp)
  )
}
