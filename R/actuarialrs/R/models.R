# Models lane: GLMs, GAMs, metrics, resampling and MCMC diagnostics, over
# crates/act-r/src/models.rs. R builds the design matrix with
# model.matrix(), so formulas, factors and contrasts behave as in stats::glm.

#' @include distributions.R
#' @importFrom stats coef predict
NULL

# Design matrix, response, offset and weights from a formula, kept with
# what predict() needs to rebuild the design on new data.
model_design <- function(formula, data, offset, weights) {
  mf <- stats::model.frame(formula, data, na.action = stats::na.fail)
  tt <- stats::terms(mf)
  x <- stats::model.matrix(tt, mf)
  y <- as.double(stats::model.response(mf))
  off <- stats::model.offset(mf)
  if (!is.null(offset)) off <- if (is.null(off)) offset else off + offset
  list(
    x = x, y = y,
    offset = if (is.null(off)) double() else as.double(off),
    weights = if (is.null(weights)) double() else as.double(weights),
    terms = stats::delete.response(tt),
    xlevels = stats::.getXlevels(tt, mf)
  )
}

new_design <- function(object, newdata, offset) {
  mf <- stats::model.frame(object@terms, newdata, xlev = object@xlevels, na.action = stats::na.fail)
  x <- stats::model.matrix(object@terms, mf)
  off <- stats::model.offset(mf)
  if (!is.null(offset)) off <- if (is.null(off)) offset else off + offset
  list(x = x, offset = if (is.null(off)) double() else as.double(off))
}

family_args <- function(theta, power) {
  list(theta = if (is.null(theta)) NaN else as.double(theta),
       power = if (is.null(power)) NaN else as.double(power))
}

dispersion_args <- function(dispersion) {
  if (is.null(dispersion)) list(kind = "", value = NaN)
  else if (is.numeric(dispersion)) list(kind = "fixed", value = as.double(dispersion))
  else list(kind = match.arg(dispersion, c("pearson", "deviance")), value = NaN)
}

#' Fit a generalized linear model
#'
#' Fits by IRLS on the Rust core. The design matrix is R's own
#' `model.matrix()`, so formulas, factors (treatment coding) and `offset()`
#' terms work as in [stats::glm()]. Results match statsmodels' GLM
#' (`validation/scripts/statsmodels_glm.py`).
#'
#' @param formula A model formula; `offset(...)` terms are honoured.
#' @param data A data frame.
#' @param family `"gaussian"`, `"poisson"`, `"gamma"`, `"inverse_gaussian"`,
#'   `"binomial"` (response a proportion, `weights` the trials),
#'   `"negative_binomial"` (needs `theta`) or `"tweedie"` (needs `power`).
#' @param link `"identity"`, `"log"`, `"logit"`, `"probit"`, `"cloglog"`,
#'   `"inverse"`, `"inverse_squared"` or `"power"` (needs `link_power`);
#'   `NULL` for the family's canonical link.
#' @param offset Optional offset added to any `offset()` terms.
#' @param weights Optional prior weights.
#' @param dispersion `NULL` (1 for the Poisson, binomial and negative
#'   binomial; Pearson's estimate otherwise), `"pearson"` (with the Poisson,
#'   the over-dispersed Poisson), `"deviance"` or a fixed number.
#' @param theta Negative binomial `theta` (variance `mu + mu^2 / theta`).
#' @param power Tweedie power in `(1, 2)`.
#' @param link_power Exponent of the power link.
#' @returns A `glm_model` with properties `coefficients`, `std_errors`,
#'   `p_values`, `covariance`, `dispersion`, `deviance`, `null_deviance`,
#'   `log_likelihood`, `aic`, `df_resid`, `iterations` and `fitted`. Use
#'   [stats::coef()], [stats::predict()] and [predict_distribution()].
#' @export
#' @examples
#' d <- data.frame(claims = c(1, 2, 4, 2, 1, 3), region = c("N", "N", "S", "S", "W", "W"),
#'                 exposure = c(1, 2, 1, 1, 0.5, 1.5))
#' m <- glm_fit(claims ~ region + offset(log(exposure)), d, family = "poisson")
#' coef(m)
#' predict(m, data.frame(region = "W", exposure = 2))
glm_fit <- function(formula, data, family = "poisson", link = NULL, offset = NULL, weights = NULL,
                    dispersion = NULL, theta = NULL, power = NULL, link_power = NULL) {
  des <- model_design(formula, data, offset, weights)
  fa <- family_args(theta, power)
  da <- dispersion_args(dispersion)
  ptr <- rust_result(glm_fit_design(
    as.double(des$x), colnames(des$x), des$y, des$offset, des$weights, family,
    if (is.null(link)) "" else link, da$kind, da$value, fa$theta, fa$power,
    if (is.null(link_power)) NaN else as.double(link_power)
  ))
  glm_model(ptr = ptr, terms = des$terms, xlevels = des$xlevels)
}

#' Fitted GLM (class)
#'
#' Returned by [glm_fit()].
#'
#' @param ptr,terms,xlevels Internal.
#' @returns A `glm_model` object.
#' @export
glm_model <- S7::new_class(
  "glm_model",
  package = "actuarialrs",
  properties = list(
    ptr = S7::new_S3_class("GlmModel"),
    terms = S7::class_any,
    xlevels = S7::class_any,
    coefficients = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(self@ptr$coefficients(), self@ptr$names())
    }),
    std_errors = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(self@ptr$std_errors(), self@ptr$names())
    }),
    p_values = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(self@ptr$p_values(), self@ptr$names())
    }),
    covariance = S7::new_property(S7::class_any, getter = function(self) {
      nm <- self@ptr$names()
      matrix(self@ptr$covariance(), length(nm), dimnames = list(nm, nm))
    }),
    dispersion = S7::new_property(S7::class_double, getter = function(self) self@ptr$dispersion()),
    deviance = S7::new_property(S7::class_double, getter = function(self) self@ptr$deviance()),
    null_deviance = S7::new_property(S7::class_double, getter = function(self) self@ptr$null_deviance()),
    log_likelihood = S7::new_property(S7::class_double, getter = function(self) self@ptr$log_likelihood()),
    aic = S7::new_property(S7::class_double, getter = function(self) self@ptr$aic()),
    df_resid = S7::new_property(S7::class_double, getter = function(self) self@ptr$df_resid()),
    iterations = S7::new_property(S7::class_double, getter = function(self) self@ptr$iterations()),
    fitted = S7::new_property(S7::class_double, getter = function(self) self@ptr$fitted())
  ),
  constructor = function(ptr, terms, xlevels) {
    S7::new_object(S7::S7_object(), ptr = ptr, terms = terms, xlevels = xlevels)
  }
)

#' Fit a generalized additive model
#'
#' A [glm_fit()] model in which each name in `smooths` (a numeric term of
#' the formula) becomes a cubic P-spline (mgcv's `s(x, bs = "ps")`), with
#' smoothing chosen by GCV when the dispersion is estimated and UBRE when
#' it is fixed (mgcv's `method = "GCV.Cp"`), or fixed by `smoothing`.
#'
#' @inheritParams glm_fit
#' @param smooths Names of numeric terms to smooth.
#' @param n_basis Basis functions per smooth (recycled; 10 by default).
#' @param smoothing `"auto"`, `"gcv"`, `"ubre"`, or a numeric vector of
#'   fixed smoothing parameters, one per smooth.
#' @returns A `gam_model` with properties `coefficients`, `lambdas`, `edf`,
#'   `dispersion`, `deviance`, `score` and `fitted`.
#' @export
#' @examples
#' d <- data.frame(x = seq(0, 1, length.out = 100))
#' d$y <- sin(6 * d$x) + 2
#' m <- gam_fit(y ~ x, d, smooths = "x", family = "gaussian")
#' m@edf
#' predict(m, data.frame(x = 0.5)) - (sin(3) + 2)
gam_fit <- function(formula, data, smooths, n_basis = 10, family = "poisson", link = NULL,
                    offset = NULL, weights = NULL, dispersion = NULL, theta = NULL,
                    power = NULL, smoothing = "auto") {
  des <- model_design(formula, data, offset, weights)
  fa <- family_args(theta, power)
  da <- dispersion_args(dispersion)
  fixed <- is.numeric(smoothing)
  ptr <- rust_result(gam_fit_design(
    as.double(des$x), colnames(des$x), des$y, des$offset, des$weights, family,
    if (is.null(link)) "" else link, da$kind, da$value, fa$theta, fa$power,
    as.character(smooths), rep_len(as.double(n_basis), length(smooths)),
    if (fixed) "fixed" else smoothing, if (fixed) as.double(smoothing) else double()
  ))
  gam_model(ptr = ptr, terms = des$terms, xlevels = des$xlevels)
}

#' Fitted GAM (class)
#'
#' Returned by [gam_fit()].
#'
#' @param ptr,terms,xlevels Internal.
#' @returns A `gam_model` object.
#' @export
gam_model <- S7::new_class(
  "gam_model",
  package = "actuarialrs",
  properties = list(
    ptr = S7::new_S3_class("GamModel"),
    terms = S7::class_any,
    xlevels = S7::class_any,
    coefficients = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(self@ptr$coefficients(), self@ptr$names())
    }),
    lambdas = S7::new_property(S7::class_double, getter = function(self) self@ptr$lambdas()),
    edf = S7::new_property(S7::class_double, getter = function(self) self@ptr$edf()),
    dispersion = S7::new_property(S7::class_double, getter = function(self) self@ptr$dispersion()),
    deviance = S7::new_property(S7::class_double, getter = function(self) self@ptr$deviance()),
    score = S7::new_property(S7::class_double, getter = function(self) self@ptr$score()),
    fitted = S7::new_property(S7::class_double, getter = function(self) self@ptr$fitted())
  ),
  constructor = function(ptr, terms, xlevels) {
    S7::new_object(S7::S7_object(), ptr = ptr, terms = terms, xlevels = xlevels)
  }
)

for (cls in list(glm_model, gam_model)) {
  S7::method(coef, cls) <- function(object, ...) object@coefficients
  S7::method(predict, cls) <- function(object, newdata, offset = NULL, ...) {
    nd <- new_design(object, newdata, offset)
    rust_result(object@ptr$predict(as.double(nd$x), colnames(nd$x), nd$offset), s7_call())
  }
}

#' Joint predictive distribution of a fitted model
#'
#' Draws the response for every row of `newdata` jointly: each simulation
#' draws the coefficients from their (posterior) normal approximation,
#' shared by all rows, then each row's response from the family. Rows are
#' keyed `row = 0, 1, ...`.
#'
#' @param object A [glm_model] or [gam_model].
#' @param newdata A data frame with the model's terms.
#' @param n_sims Number of simulations.
#' @param seed Generator seed.
#' @param offset Optional offset added to any `offset()` terms.
#' @param weights Optional prior weights.
#' @param ... Unused; for methods.
#' @returns A [predictive_distribution].
#' @export
#' @examples
#' d <- data.frame(y = c(2, 3, 5, 4, 6, 8), x = 1:6)
#' m <- glm_fit(y ~ x, d, family = "poisson")
#' pd <- predict_distribution(m, data.frame(x = c(7, 8)), n_sims = 1000, seed = 1)
#' mean(pd)
predict_distribution <- S7::new_generic(
  "predict_distribution", "object",
  function(object, newdata, n_sims, seed, ...) S7::S7_dispatch()
)

for (cls in list(glm_model, gam_model)) {
  S7::method(predict_distribution, cls) <- function(object, newdata, n_sims, seed,
                                                     offset = NULL, weights = NULL, ...) {
    nd <- new_design(object, newdata, offset)
    w <- if (is.null(weights)) double() else as.double(weights)
    ptr <- rust_result(object@ptr$predict_distribution(
      as.double(nd$x), colnames(nd$x), nd$offset, w, as.double(n_sims), as.double(seed)
    ), s7_call())
    predictive_distribution(ptr = ptr)
  }
}

S7::method(print, glm_model) <- function(x, ...) {
  cat(sprintf("<glm_model> %s, deviance %s on %s df\n", x@ptr$family(),
              format(x@deviance, digits = 8), format(x@df_resid)))
  print(x@coefficients)
  invisible(x)
}
S7::method(print, gam_model) <- function(x, ...) {
  cat(sprintf("<gam_model> edf %s, deviance %s\n", format(x@edf, digits = 6),
              format(x@deviance, digits = 8)))
  invisible(x)
}

#' Model metrics
#'
#' `family_deviance()` is `sum(w d(y, mu))` for a family; `gini_index()` the
#' Gini index of the ordered Lorenz curve (rows sorted by prediction,
#' exposure share against loss share); `lift_table()` cuts rows sorted by
#' predicted rate into bands of about equal exposure; `crps_draws()` the
#' continuous ranked probability score of draws for an outcome.
#'
#' @param family A family name, as in [glm_fit()].
#' @param y Outcomes.
#' @param mu,pred Predictions.
#' @param weights,exposure Optional weights or exposures.
#' @param theta,power Family parameters, as in [glm_fit()].
#' @param bands Number of lift bands.
#' @param draws Equally likely draws.
#' @returns A number, or for `lift_table()` a data frame with columns
#'   `exposure`, `expected` and `actual`.
#' @name model_metrics
#' @examples
#' family_deviance("poisson", c(1, 0, 3), c(1, 0.5, 2))
#' gini_index(c(0, 1), c(0.1, 0.9))
#' lift_table(c(0, 1, 2, 3), c(0.1, 0.9, 2.1, 2.9), bands = 2)
#' crps_draws(c(1, 2, 3), 2)
NULL

#' @rdname model_metrics
#' @export
family_deviance <- function(family, y, mu, weights = NULL, theta = NULL, power = NULL) {
  fa <- family_args(theta, power)
  rust_result(family_deviance_rust(family, fa$theta, fa$power, as.double(y), as.double(mu),
                                   if (is.null(weights)) double() else as.double(weights)))
}

#' @rdname model_metrics
#' @export
gini_index <- function(y, pred, exposure = NULL) {
  rust_result(gini_rust(as.double(y), as.double(pred),
                        if (is.null(exposure)) double() else as.double(exposure)))
}

#' @rdname model_metrics
#' @export
lift_table <- function(y, pred, exposure = NULL, bands = 10) {
  as.data.frame(rust_result(lift_rust(as.double(y), as.double(pred),
                                      if (is.null(exposure)) double() else as.double(exposure),
                                      as.double(bands))))
}

#' @rdname model_metrics
#' @export
crps_draws <- function(draws, y) rust_result(crps_rust(as.double(draws), as.double(y)))

#' Resampling splits
#'
#' `k_fold()` shuffles `n` rows with `seed` and cuts them into `k` folds;
#' `group_k_fold()` keeps each group's rows in one fold; `time_ordered()`
#' tests on each of the last `n_test` periods after training on the earlier
#' ones (with calendar diagonals as periods, the triangle backtest).
#'
#' @param n Number of rows.
#' @param k Number of folds.
#' @param seed Generator seed.
#' @param groups Group label per row.
#' @param periods Period per row (whole numbers).
#' @param n_test Number of final periods to test on.
#' @returns A list of splits, each `list(train, test)` of row numbers.
#' @name resampling
#' @examples
#' length(k_fold(10, 5, seed = 1))
#' time_ordered(c(0, 1, 2, 1, 2, 2), 1)
NULL

#' @rdname resampling
#' @export
k_fold <- function(n, k, seed) rust_result(k_fold_rust(as.double(n), as.double(k), as.double(seed)))

#' @rdname resampling
#' @export
group_k_fold <- function(groups, k, seed) {
  rust_result(group_k_fold_rust(as.character(groups), as.double(k), as.double(seed)))
}

#' @rdname resampling
#' @export
time_ordered <- function(periods, n_test) {
  rust_result(time_ordered_rust(as.double(periods), as.double(n_test)))
}

#' MCMC convergence diagnostics
#'
#' Rank-normalized split R-hat, bulk and tail effective sample sizes, the
#' effective sample size of the mean and its Monte Carlo standard error:
#' the estimators of Vehtari et al. (2021), matching the posterior package.
#'
#' @param draws A matrix with one column per chain (or a list of
#'   equal-length chains).
#' @returns A named numeric vector: `rhat`, `ess_bulk`, `ess_tail`,
#'   `ess_mean`, `mcse_mean`.
#' @export
#' @examples
#' set.seed(1)
#' mcmc_diagnostics(matrix(rnorm(4000), ncol = 4))
mcmc_diagnostics <- function(draws) {
  if (is.list(draws)) draws <- do.call(cbind, draws)
  draws <- as.matrix(draws)
  unlist(rust_result(mcmc_diagnostics_rust(as.double(draws), as.double(ncol(draws)))))
}
