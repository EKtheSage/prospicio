#' @include models.R reserving.R
NULL

#' Over-dispersed Poisson GLM on a triangle
#'
#' `odp_glm()` fits the over-dispersed Poisson model as a GLM: the
#' incremental values of `column` are quasi-Poisson with a log link, an
#' origin factor and a development factor. Its fitted future cells
#' reproduce the chain-ladder reserves, and the dispersion is Pearson's
#' chi-squared over the residual degrees of freedom. The triangle must
#' have a single segment.
#'
#' `prediction_error` gives the reserves' analytic prediction error by the
#' delta method, by origin and in total: the process variance `phi * sum(mu)`
#' and the parameter variance `g' Sigma g` with `g = sum(mu * x)` over each
#' origin's future cells. It reproduces Taylor and McGuire's (CAS Monograph
#' 3) Table 5-4 (`validation/tests/reserving.rs`).
#'
#' The fit's properties are `origins`, `reserves` (named by origin),
#' `total_reserve`, `dispersion`, `coefficients` and `std_errors` (named by
#' parameter), `covariance` (their covariance matrix), `future` (a data frame
#' of the future cells' `origin`, `development`, fitted `mean` and
#' `predictive_mean`, the mean of draws whose coefficients are drawn
#' normally) and `prediction_error` (a data frame with a row per origin and
#' a `Total` row: `reserve`, `process_se`, `parameter_se`, `rmsep` and `cv`,
#' `NA` for a zero reserve). [predict_distribution()] draws the future cells
#' jointly (no `newdata`).
#'
#' @param triangle A single-segment [triangle] of cumulative values.
#' @param column The column to fit, by default the triangle's only one.
#' @param tolerance Convergence tolerance of the IRLS fit.
#' @param max_iterations Most IRLS iterations.
#' @returns An `odp_glm_fit` object.
#' @export
#' @examples
#' long <- data.frame(year = c(2020, 2020, 2020, 2021, 2021, 2022),
#'                    age = c(12, 24, 36, 12, 24, 12),
#'                    paid = c(100, 150, 165, 110, 170, 120))
#' fit <- odp_glm(triangle(long, "year", "age", "paid"))
#' fit@total_reserve
#' fit@prediction_error
#' pd <- predict_distribution(fit, n_sims = 1000, seed = 1)
#' mean(total(pd))
odp_glm_fit <- S7::new_class(
  "odp_glm_fit",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("OdpGlmFit"),
    origins = S7::new_property(S7::class_character, getter = function(self) self@ptr$origins()),
    reserves = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(self@ptr$reserves(), self@ptr$origins())
    }),
    total_reserve = S7::new_property(S7::class_double, getter = function(self) {
      self@ptr$total_reserve()
    }),
    dispersion = S7::new_property(S7::class_double, getter = function(self) {
      self@ptr$dispersion()
    }),
    coefficients = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(self@ptr$coefficients(), self@ptr$coefficient_names())
    }),
    std_errors = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(self@ptr$std_errors(), self@ptr$coefficient_names())
    }),
    covariance = S7::new_property(S7::class_double, getter = function(self) {
      names <- self@ptr$coefficient_names()
      matrix(self@ptr$covariance(), length(names), length(names), byrow = TRUE,
             dimnames = list(names, names))
    }),
    future = S7::new_property(S7::class_data.frame, getter = function(self) {
      as.data.frame(self@ptr$future())
    }),
    prediction_error = S7::new_property(S7::class_data.frame, getter = function(self) {
      e <- as.data.frame(rust_result(self@ptr$prediction_error(), call = NULL))
      e$cv[is.nan(e$cv)] <- NA
      e
    })
  ),
  constructor = function(ptr) S7::new_object(S7::S7_object(), ptr = ptr)
)

#' @rdname odp_glm_fit
#' @export
odp_glm <- function(triangle, column = NULL, tolerance = 1e-10, max_iterations = 100) {
  column <- fit_column(triangle, column)
  ptr <- rust_result(odp_glm_fit_rust(triangle@ptr, column,
                                      single_number(tolerance, "tolerance"),
                                      single_number(max_iterations, "max_iterations")))
  odp_glm_fit(ptr = ptr)
}

S7::method(predict_distribution, odp_glm_fit) <- function(object, newdata, n_sims, seed,
                                                         parameters = c("mean_preserving",
                                                                        "normal", "fixed"),
                                                         ...) {
  if (!missing(newdata)) {
    stop("an odp_glm_fit draws its own future cells; pass no newdata", call. = FALSE)
  }
  parameters <- match.arg(parameters)
  ptr <- rust_result(object@ptr$predict_distribution(as.double(n_sims), as.double(seed),
                                                     parameters), s7_call())
  predictive_distribution(ptr = ptr)
}

S7::method(print, odp_glm_fit) <- function(x, ...) {
  cat(sprintf("<odp_glm_fit> %d origins, total reserve %s, dispersion %s\n",
              length(x@origins), format(x@total_reserve), format(x@dispersion)))
  invisible(x)
}
