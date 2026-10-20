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
#' `formula` fits another design on the triangle's cells instead, the
#' columns of [odp_glm_cells()] (`origin`, `development` and `calendar` as
#' labels, `origin_index`, `development_index` and `calendar_index` from 0,
#' `age`): fewer parameters, calendar trends, interactions (Taylor and
#' McGuire, CAS Monograph 3, Chapter 7). It is a one-sided formula run
#' through [stats::model.matrix()], so it has an intercept unless it says
#' `- 1`, and `offset()` terms are honoured. A numeric term extrapolates to
#' the future cells; a factor level that only future cells take (of
#' `calendar`, say) cannot be estimated, nor can a column that is zero on
#' every observed cell.
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
#' `residuals` is a data frame with a row per observed cell, origin-major:
#' `origin`, `development` (age in months), `calendar` (the calendar
#' period's label), `calendar_index` (0 for the earliest valuation),
#' `actual`, `fitted`, `actual_to_fitted` (the cell of the actual-to-fitted
#' heat map), `pearson`, `deviance`, `leverage` (the hat value),
#' `standardized_deviance` (as [stats::rstandard()]) and
#' `standardized_pearson`. A cell with leverage 1, the only observation of
#' its origin or age, is fitted exactly and has no standardized residual
#' (`NA`). [residual_summary()] groups the standardized deviance residuals
#' by origin, development or calendar period (Taylor and McGuire, Chapter
#' 6).
#'
#' @param triangle A single-segment [triangle] of cumulative values.
#' @param column The column to fit, by default the triangle's only one.
#' @param formula `NULL` for the cross-classified model (an origin and a
#'   development factor), or a one-sided formula on the cells.
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
#'
#' # The origins' effect a straight line in their index, an effect per age.
#' line <- odp_glm(triangle(long, "year", "age", "paid"),
#'                 formula = ~ origin_index + factor(development))
#' line@coefficients
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
    residuals = S7::new_property(S7::class_data.frame, getter = function(self) {
      r <- as.data.frame(rust_result(self@ptr$residuals(), call = NULL))
      for (col in c("standardized_deviance", "standardized_pearson")) {
        r[[col]][is.nan(r[[col]])] <- NA
      }
      r
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
odp_glm <- function(triangle, column = NULL, formula = NULL, tolerance = 1e-10,
                    max_iterations = 100) {
  column <- fit_column(triangle, column)
  x <- double(0)
  names <- character(0)
  offset <- double(0)
  if (!is.null(formula)) {
    if (!inherits(formula, "formula") || length(formula) != 2) {
      stop("formula must be a one-sided formula such as ~ origin_index + factor(development)",
           call. = FALSE)
    }
    cells <- odp_glm_cells(triangle, column)
    mf <- stats::model.frame(formula, cells, na.action = stats::na.pass)
    mm <- stats::model.matrix(formula, mf)
    x <- as.double(mm)
    names <- colnames(mm)
    offset <- stats::model.offset(mf)
    offset <- if (is.null(offset)) double(nrow(cells)) else as.double(offset)
  }
  ptr <- rust_result(odp_glm_fit_rust(triangle@ptr, column, x, names, offset,
                                      single_number(tolerance, "tolerance"),
                                      single_number(max_iterations, "max_iterations")))
  odp_glm_fit(ptr = ptr)
}

#' The cells of a triangle, for an ODP GLM's design
#'
#' A row per origin and development age of a single-segment triangle,
#' origin-major: the rows an [odp_glm()] `formula` is evaluated on.
#'
#' @inheritParams odp_glm
#' @returns A data frame with `origin`, `development` (the age in months)
#'   and `calendar` as labels; `origin_index`, `development_index` and
#'   `calendar_index` (from 0, the last from the earliest valuation), `age`
#'   in months and `exposure` (1) as numbers; and `observed` and `future`,
#'   whether the cell is fitted or predicted (a cell after a hole is
#'   neither).
#' @export
#' @examples
#' long <- data.frame(year = c(2020, 2020, 2020, 2021, 2021, 2022),
#'                    age = c(12, 24, 36, 12, 24, 12),
#'                    paid = c(100, 150, 165, 110, 170, 120))
#' odp_glm_cells(triangle(long, "year", "age", "paid"))
odp_glm_cells <- function(triangle, column = NULL) {
  column <- fit_column(triangle, column)
  as.data.frame(rust_result(odp_glm_cells_rust(triangle@ptr, column)),
                stringsAsFactors = FALSE)
}

#' Residuals of an ODP GLM by group
#'
#' The standardized deviance residuals of an [odp_glm_fit] grouped by
#' origin, development age or calendar period: a row per group with the
#' count of cells with a defined residual, their mean and their standard
#' deviation (divisor `n - 1`, `NA` below two cells). A trend in the means
#' over calendar periods points to a calendar effect the model lacks; a
#' spread that falls with age, to a variance that is not `phi * mu`
#' (Taylor and McGuire, CAS Monograph 3, Chapter 6).
#'
#' @param fit An [odp_glm_fit].
#' @param by `"development"`, `"origin"` or `"calendar"`.
#' @returns A data frame with columns `label`, `n`, `mean` and `sd`.
#' @export
#' @examples
#' long <- data.frame(year = c(2020, 2020, 2020, 2021, 2021, 2022),
#'                    age = c(12, 24, 36, 12, 24, 12),
#'                    paid = c(100, 150, 165, 110, 170, 120))
#' fit <- odp_glm(triangle(long, "year", "age", "paid"))
#' fit@residuals
#' residual_summary(fit, "calendar")
residual_summary <- function(fit, by = c("development", "origin", "calendar")) {
  if (!S7::S7_inherits(fit, odp_glm_fit)) stop("fit must be an odp_glm_fit", call. = FALSE)
  by <- match.arg(by)
  out <- as.data.frame(rust_result(fit@ptr$residual_summary(by)))
  out$mean[is.nan(out$mean)] <- NA
  out$sd[is.nan(out$sd)] <- NA
  out
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
