# Models lane: GLMs, elastic nets, GAMs, metrics, resampling, tuning and
# MCMC diagnostics, over crates/act-r/src/models.rs. R builds the design
# matrix with model.matrix(), so formulas, factors and contrasts behave as in
# stats::glm.

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

# Penalty factors per design column, from a full or a named vector.
penalty_factors <- function(penalty_factor, x) {
  if (is.null(penalty_factor)) return(double())
  pf <- stats::setNames(rep(1, ncol(x)), colnames(x))
  if (is.null(names(penalty_factor))) {
    if (length(penalty_factor) != ncol(x)) {
      stop(sprintf("penalty_factor needs %d values, one per design column", ncol(x)))
    }
    pf[] <- penalty_factor
  } else {
    unknown <- setdiff(names(penalty_factor), names(pf))
    if (length(unknown)) stop("unknown design columns in penalty_factor: ", toString(unknown))
    pf[names(penalty_factor)] <- penalty_factor
  }
  as.double(pf)
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

#' Sandwich covariance of a GLM
#'
#' Heteroskedasticity- or cluster-robust covariance of the coefficients of
#' a [glm_fit()] model, valid when the variance function or dispersion is
#' wrong as long as the mean is right. The dispersion cancels. Results match
#' statsmodels' `cov_type = "HC0"` and `"cluster"`
#' (`validation/scripts/statsmodels_glm_robust.py`). For a non-canonical
#' link (a log-link gamma, say) the bread is the observed information, as
#' in statsmodels; `sandwich::vcovHC()` uses the expected information, so
#' the two differ slightly there and agree for canonical links.
#'
#' @param object A `glm_model`.
#' @param type `"HC0"` (White's estimator) or `"HC1"` (scaled by
#'   `n / (n - p)`). Ignored when `cluster` is given.
#' @param cluster Optional cluster labels, one per row the model was fitted
#'   on (a policy or an event, say). The scores are summed within each
#'   cluster and the result scaled by `G / (G - 1) * (n - 1) / (n - p)` for
#'   `G` clusters.
#' @returns A named covariance matrix; `sqrt(diag(.))` gives the robust
#'   standard errors.
#' @export
#' @examples
#' d <- data.frame(y = c(1, 2, 6, 1, 4, 2), x = c(0, 0, 0, 1, 1, 1),
#'                 policy = c(1, 1, 2, 2, 3, 3))
#' m <- glm_fit(y ~ x, d, family = "poisson")
#' robust_vcov(m)
#' sqrt(diag(robust_vcov(m, cluster = d$policy)))
robust_vcov <- function(object, type = c("HC0", "HC1"), cluster = NULL) {
  if (is.null(cluster)) {
    kind <- match.arg(type)
    groups <- integer()
  } else {
    kind <- "cluster"
    groups <- as.integer(factor(cluster))
  }
  nm <- object@ptr$names()
  v <- rust_result(object@ptr$robust_covariance(kind, groups))
  matrix(v, length(nm), dimnames = list(nm, nm))
}

#' Save and load a GLM
#'
#' `save_model()` writes a [glm_fit()] model to an RDS file: the Rust fit as
#' a versioned JSON artifact (spec, estimates, covariance, fit statistics,
#' fitted values, and provenance with the package version and a hash of
#' the training data), with the formula terms and factor levels that
#' [stats::predict()] needs. `load_model()` reads it back; the estimates
#' round-trip exactly. A loaded model predicts and simulates but has no
#' training data, so [robust_vcov()] needs the original fit.
#'
#' @param model A `glm_model`.
#' @param file Path of the RDS file.
#' @returns `save_model()`: `file`, invisibly. `load_model()`: a
#'   `glm_model`.
#' @export
#' @examples
#' d <- data.frame(claims = c(1, 2, 4, 2, 1, 3), region = c("N", "N", "S", "S", "W", "W"))
#' m <- glm_fit(claims ~ region, d, family = "poisson")
#' f <- tempfile(fileext = ".rds")
#' save_model(m, f)
#' m2 <- load_model(f)
#' identical(coef(m2), coef(m))
save_model <- function(model, file) {
  if (!S7::S7_inherits(model, glm_model)) stop("model must be a glm_model")
  saveRDS(list(format = "actuarialrs.glm_model", artifact = model@ptr$to_json(),
               terms = model@terms, xlevels = model@xlevels), file)
  invisible(file)
}

#' @rdname save_model
#' @export
load_model <- function(file) {
  x <- readRDS(file)
  if (!is.list(x) || !identical(x$format, "actuarialrs.glm_model")) {
    stop("not a model saved by save_model()")
  }
  glm_model(ptr = rust_result(GlmModel$from_json(x$artifact)), terms = x$terms,
            xlevels = x$xlevels)
}

#' Fit an elastic-net GLM
#'
#' The lasso (`alpha = 1`), ridge (`alpha = 0`) and everything between, for
#' every family and link of [glm_fit()], by coordinate descent inside IRLS.
#' Minimizes glmnet's objective
#' `sum(w * d) / (2 * sum(w)) + lambda * sum(pf * ((1 - alpha) / 2 * b^2 + alpha * abs(b)))`,
#' where `b` are the coefficients of the columns standardized to unit
#' standard deviation (when `standardize = TRUE`). The intercept is not
#' penalized; coefficients are reported on the design's scale. Results
#' match glmnet (`validation/scripts/r_glmnet.R`), except that for the
#' Gaussian glmnet divides the ridge part of its penalty by `sd(y)`.
#'
#' @inheritParams glm_fit
#' @param alpha Mixing between ridge (0) and the lasso (1).
#' @param lambda Penalty strengths; `NULL` for a path of `nlambda` values
#'   log-spaced from the smallest `lambda` that zeroes every coefficient
#'   down to `lambda_min_ratio` times it.
#' @param nlambda,lambda_min_ratio The default path.
#' @param standardize Penalize standardized coefficients.
#' @param penalty_factor Optional penalty factors: a vector with one per
#'   design column, or a named vector for some columns (the rest get 1); 0
#'   leaves a column unpenalized.
#' @returns An `elastic_net_model` with properties `lambda`, `coefficients`
#'   (a matrix, one column per `lambda`), `deviance`, `deviance_ratio`,
#'   `df` (non-zero coefficients), `null_deviance` and `alpha`. Use
#'   [stats::coef()], [stats::predict()] and [predict_distribution()], each
#'   with a `lambda` from the path. Tune `lambda` and `alpha` with
#'   [k_fold()] and [family_deviance()].
#' @export
#' @examples
#' set.seed(1)
#' d <- data.frame(x1 = rnorm(50), x2 = rnorm(50))
#' d$y <- 1 + 2 * d$x1 + rnorm(50)
#' m <- elastic_net_fit(y ~ x1 + x2, d, family = "gaussian", alpha = 1)
#' m@df[c(1, 20, 100)]
#' coef(m, lambda = m@lambda[20])
elastic_net_fit <- function(formula, data, family = "poisson", link = NULL, alpha = 1,
                            lambda = NULL, nlambda = 100, lambda_min_ratio = 1e-4,
                            standardize = TRUE, penalty_factor = NULL, offset = NULL,
                            weights = NULL, theta = NULL, power = NULL, link_power = NULL) {
  des <- model_design(formula, data, offset, weights)
  fa <- family_args(theta, power)
  pf <- penalty_factors(penalty_factor, des$x)
  ptr <- rust_result(elastic_net_fit_design(
    as.double(des$x), colnames(des$x), des$y, des$offset, des$weights, family,
    if (is.null(link)) "" else link, as.double(alpha),
    if (is.null(lambda)) double() else as.double(lambda), as.double(nlambda),
    as.double(lambda_min_ratio), isTRUE(standardize), pf, fa$theta, fa$power,
    if (is.null(link_power)) NaN else as.double(link_power)
  ))
  elastic_net_model(ptr = ptr, terms = des$terms, xlevels = des$xlevels)
}

#' Fitted elastic net (class)
#'
#' Returned by [elastic_net_fit()]: one fit per `lambda`.
#'
#' @param ptr,terms,xlevels Internal.
#' @returns An `elastic_net_model` object.
#' @export
elastic_net_model <- S7::new_class(
  "elastic_net_model",
  package = "actuarialrs",
  properties = list(
    ptr = S7::new_S3_class("ElasticNetPath"),
    terms = S7::class_any,
    xlevels = S7::class_any,
    lambda = S7::new_property(S7::class_double, getter = function(self) self@ptr$lambda()),
    coefficients = S7::new_property(S7::class_any, getter = function(self) {
      nm <- self@ptr$names()
      lam <- self@ptr$lambda()
      matrix(self@ptr$coefficients(), length(nm), dimnames = list(nm, format(lam, digits = 6)))
    }),
    deviance = S7::new_property(S7::class_double, getter = function(self) self@ptr$deviance()),
    deviance_ratio = S7::new_property(S7::class_double, getter = function(self) self@ptr$deviance_ratio()),
    df = S7::new_property(S7::class_double, getter = function(self) self@ptr$df()),
    null_deviance = S7::new_property(S7::class_double, getter = function(self) self@ptr$null_deviance()),
    alpha = S7::new_property(S7::class_double, getter = function(self) self@ptr$alpha())
  ),
  constructor = function(ptr, terms, xlevels) {
    S7::new_object(S7::S7_object(), ptr = ptr, terms = terms, xlevels = xlevels)
  }
)

# Position of `lambda` on the path (1-based); NULL means the only one.
lambda_index <- function(object, lambda) {
  lam <- object@lambda
  if (is.null(lambda)) {
    if (length(lam) == 1) return(1)
    stop("this path has several lambdas: pass `lambda`, one of object@lambda")
  }
  k <- which(abs(lam - lambda) <= 1e-10 * abs(lambda))
  if (length(k) != 1) stop("`lambda` is not on the path; refit with elastic_net_fit(lambda = ...)")
  k
}

S7::method(coef, elastic_net_model) <- function(object, lambda = NULL, ...) {
  if (is.null(lambda)) return(object@coefficients)
  stats::setNames(object@coefficients[, lambda_index(object, lambda)], object@ptr$names())
}

S7::method(predict, elastic_net_model) <- function(object, newdata, lambda = NULL, offset = NULL, ...) {
  nd <- new_design(object, newdata, offset)
  one <- function(k) {
    rust_result(object@ptr$predict(as.double(k), as.double(nd$x), colnames(nd$x), nd$offset))
  }
  if (is.null(lambda) && length(object@lambda) > 1) {
    return(vapply(seq_along(object@lambda), one, double(nrow(nd$x))))
  }
  one(lambda_index(object, lambda))
}

S7::method(print, elastic_net_model) <- function(x, ...) {
  lam <- x@lambda
  cat(sprintf("<elastic_net_model> %s, alpha = %s, %d lambdas from %s to %s\n",
              x@ptr$family(), format(x@alpha), length(lam), format(max(lam), digits = 6),
              format(min(lam), digits = 6)))
  invisible(x)
}

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
#' keyed `row = 0, 1, ...`. An [elastic_net_model] has no standard errors,
#' so its draws hold the coefficients fixed (process uncertainty only) and
#' take a `lambda` from its path.
#'
#' @param object A [glm_model], [elastic_net_model] or [gam_model].
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

S7::method(predict_distribution, elastic_net_model) <- function(object, newdata, n_sims, seed,
                                                               lambda = NULL, offset = NULL,
                                                               weights = NULL, ...) {
  nd <- new_design(object, newdata, offset)
  w <- if (is.null(weights)) double() else as.double(weights)
  ptr <- rust_result(object@ptr$predict_distribution(
    as.double(lambda_index(object, lambda)), as.double(nd$x), colnames(nd$x), nd$offset, w,
    as.double(n_sims), as.double(seed)
  ))
  predictive_distribution(ptr = ptr)
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
#' `log_score()` is the mean of `-log f(y)` under each row's predictive
#' distribution (the family with mean `mu`, `dispersion` and weight);
#' `pit_values()` the probability integral transform `F(y)`, randomized
#' where the distribution has atoms (counts, a Tweedie's zero) and uniform
#' when the model is calibrated; `ks_uniform()` the Kolmogorov-Smirnov
#' distance of values from the uniform, about `1.36 / sqrt(n)` or less 95%
#' of the time under uniformity.
#'
#' @param family A family name, as in [glm_fit()].
#' @param y Outcomes.
#' @param mu,pred Predictions.
#' @param weights,exposure Optional weights or exposures.
#' @param theta,power Family parameters, as in [glm_fit()].
#' @param bands Number of lift bands.
#' @param draws Equally likely draws.
#' @param dispersion The family's dispersion.
#' @param seed Seed of the PIT's randomization.
#' @param values Values to compare with the uniform.
#' @returns A number; for `pit_values()` one value per outcome; for
#'   `lift_table()` a data frame with columns `exposure`, `expected` and
#'   `actual`.
#' @name model_metrics
#' @examples
#' family_deviance("poisson", c(1, 0, 3), c(1, 0.5, 2))
#' gini_index(c(0, 1), c(0.1, 0.9))
#' lift_table(c(0, 1, 2, 3), c(0.1, 0.9, 2.1, 2.9), bands = 2)
#' crps_draws(c(1, 2, 3), 2)
#' log_score("poisson", 0, 1)
#' set.seed(1)
#' y <- rpois(500, 3)
#' ks_uniform(pit_values("poisson", y, rep(3, 500)))
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

#' @rdname model_metrics
#' @export
log_score <- function(family, y, mu, dispersion = 1, weights = NULL, theta = NULL, power = NULL) {
  fa <- family_args(theta, power)
  rust_result(log_score_rust(family, fa$theta, fa$power, as.double(y), as.double(mu),
                             as.double(dispersion),
                             if (is.null(weights)) double() else as.double(weights)))
}

#' @rdname model_metrics
#' @export
pit_values <- function(family, y, mu, dispersion = 1, weights = NULL, seed = 0, theta = NULL,
                       power = NULL) {
  fa <- family_args(theta, power)
  rust_result(pit_rust(family, fa$theta, fa$power, as.double(y), as.double(mu),
                       as.double(dispersion),
                       if (is.null(weights)) double() else as.double(weights), as.double(seed)))
}

#' @rdname model_metrics
#' @export
ks_uniform <- function(values) rust_result(ks_uniform_rust(as.double(values)))

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

#' Cross-validation and hyperparameter search
#'
#' `cross_validate()` fits a model to each split's training rows and scores
#' it on the test rows. `grid_search()` does that for each candidate and
#' picks the lowest mean score; `random_search()` draws the candidates.
#' Models are built by your own `fit` function, so any of [glm_fit()],
#' [elastic_net_fit()] or [gam_fit()] (or anything else) can be tuned.
#'
#' @param data A data frame.
#' @param splits Splits from [k_fold()], [group_k_fold()] or [time_ordered()].
#' @param fit `fit(train)` for `cross_validate()`, `fit(candidate, train)`
#'   for the searches: returns a fitted model.
#' @param score `score(model, test)`: a loss on the test rows (lower is
#'   better), such as a mean [family_deviance()].
#' @param candidates A list (or vector) of hyperparameter values.
#' @param n Number of random candidates.
#' @param draw `draw()`: one random candidate, using R's generator.
#' @param seed Seed for R's generator before drawing.
#' @returns `cross_validate()`: one score per split. The searches: a data
#'   frame with a `candidate` list column and the `score`, and attribute
#'   `best`, the row of the lowest score.
#' @name tuning
#' @examples
#' d <- data.frame(x = 1:40 / 10)
#' d$y <- 1 + 2 * d$x + sin(1:40)
#' folds <- k_fold(nrow(d), 4, seed = 1)
#' mse <- function(m, test) mean((test$y - predict(m, test))^2)
#' cross_validate(d, folds, function(train) glm_fit(y ~ x, train, family = "gaussian"), mse)
#' g <- grid_search(c(0, 0.1, 1), d, folds,
#'                  function(lam, train) elastic_net_fit(y ~ x, train, family = "gaussian",
#'                                                       lambda = lam),
#'                  mse)
#' g[attr(g, "best"), ]
NULL

#' @rdname tuning
#' @export
cross_validate <- function(data, splits, fit, score) {
  vapply(splits, function(s) {
    model <- fit(data[s$train, , drop = FALSE])
    as.double(score(model, data[s$test, , drop = FALSE]))
  }, double(1))
}

#' @rdname tuning
#' @export
grid_search <- function(candidates, data, splits, fit, score) {
  candidates <- as.list(candidates)
  if (!length(candidates)) stop("candidates must not be empty")
  scores <- vapply(candidates, function(cand) {
    mean(cross_validate(data, splits, function(train) fit(cand, train), score))
  }, double(1))
  out <- data.frame(candidate = I(candidates), score = scores)
  attr(out, "best") <- which.min(scores)
  out
}

#' @rdname tuning
#' @export
random_search <- function(n, draw, data, splits, fit, score, seed = NULL) {
  if (!is.null(seed)) set.seed(seed)
  grid_search(lapply(seq_len(n), function(i) draw()), data, splits, fit, score)
}

#' Compare models on the same splits
#'
#' Fits every model on each split's training rows and scores it on the test
#' rows with every metric: one table across engines, comparing like with
#' like. The paired `difference_std_error` (the standard error of each
#' split's score minus the best model's) is much less noisy than either
#' mean; a model within about two of them of the best is not clearly worse.
#'
#' @param models A named list of `fit(train)` functions, each returning a
#'   fitted model (from [glm_fit()], [gam_fit()], [elastic_net_fit()] or
#'   anything `score` accepts).
#' @param data A data frame.
#' @param splits Splits from [k_fold()], [group_k_fold()] or [time_ordered()].
#' @param scores A named list of `score(model, test)` losses (lower is better).
#' @returns A data frame with one row per model and metric: `model`,
#'   `metric`, `mean`, `std_error` (the split scores' standard deviation
#'   over the square root of their number) and `difference_std_error`, with
#'   attribute `split_scores`, an array indexed by model, metric and split.
#' @export
#' @examples
#' d <- data.frame(x = 1:40 / 10)
#' d$y <- 1 + 2 * d$x + sin(1:40)
#' mse <- function(m, test) mean((test$y - predict(m, test))^2)
#' compare_models(
#'   list(linear = function(train) glm_fit(y ~ x, train, family = "gaussian"),
#'        flat = function(train) glm_fit(y ~ 1, train, family = "gaussian")),
#'   d, k_fold(nrow(d), 4, seed = 1), list(mse = mse)
#' )
compare_models <- function(models, data, splits, scores) {
  if (!length(models) || is.null(names(models)) || any(names(models) == "")) {
    stop("models must be a named, non-empty list of fit functions")
  }
  if (!length(scores) || is.null(names(scores)) || any(names(scores) == "")) {
    stop("scores must be a named, non-empty list of score functions")
  }
  k <- length(splits)
  s <- array(NA_real_, c(length(models), length(scores), k),
             dimnames = list(names(models), names(scores), NULL))
  for (i in seq_len(k)) {
    train <- data[splits[[i]]$train, , drop = FALSE]
    test <- data[splits[[i]]$test, , drop = FALSE]
    for (m in names(models)) {
      fitted <- models[[m]](train)
      for (sc in names(scores)) s[m, sc, i] <- as.double(scores[[sc]](fitted, test))
    }
  }
  se <- function(v) stats::sd(v) / sqrt(length(v))
  rows <- expand.grid(metric = names(scores), model = names(models), stringsAsFactors = FALSE)
  rows <- rows[, c("model", "metric")]
  rows$mean <- mapply(function(m, sc) mean(s[m, sc, ]), rows$model, rows$metric)
  rows$std_error <- mapply(function(m, sc) se(s[m, sc, ]), rows$model, rows$metric)
  rows$difference_std_error <- mapply(function(m, sc) {
    means <- apply(s[, sc, , drop = FALSE], 1, mean)
    best <- names(models)[which.min(means)]
    se(s[m, sc, ] - s[best, sc, ])
  }, rows$model, rows$metric)
  rownames(rows) <- NULL
  attr(rows, "split_scores") <- s
  rows
}

#' Cross-validate an elastic net
#'
#' glmnet's `cv.glmnet()` on the Rust core: on each fold, fits the whole
#' `lambda` path to the other folds (warm starts) and scores the family's
#' mean deviance on the held-out fold; folds run in parallel. The score per
#' `lambda` is the folds' mean weighted by fold weight, with its standard
#' error. Matches `cv.glmnet()` on the same folds.
#'
#' @inheritParams elastic_net_fit
#' @param folds Number of folds, used when `foldid` is `NULL`.
#' @param seed Seed of the fold assignment.
#' @param foldid Optional fold number per row.
#' @returns A list with `lambda`, `mean`, `se`, `lambda_min` (lowest mean)
#'   and `lambda_1se` (the largest `lambda` within one standard error of
#'   it), and `fit`, the [elastic_net_model] on all rows over `lambda`.
#' @export
#' @examples
#' set.seed(1)
#' d <- data.frame(x1 = rnorm(80), x2 = rnorm(80))
#' d$y <- 1 + 2 * d$x1 + rnorm(80)
#' cv <- elastic_net_cv(y ~ x1 + x2, d, family = "gaussian", nlambda = 30)
#' coef(cv$fit, lambda = cv$lambda_1se)
elastic_net_cv <- function(formula, data, family = "poisson", link = NULL, alpha = 1,
                           lambda = NULL, nlambda = 100, lambda_min_ratio = 1e-4,
                           standardize = TRUE, penalty_factor = NULL, offset = NULL,
                           weights = NULL, folds = 10, seed = 1, foldid = NULL,
                           theta = NULL, power = NULL, link_power = NULL) {
  des <- model_design(formula, data, offset, weights)
  if (is.null(foldid)) {
    foldid <- integer(nrow(des$x))
    for (k in seq_along(sp <- k_fold(nrow(des$x), folds, seed))) foldid[sp[[k]]$test] <- k
  }
  fa <- family_args(theta, power)
  pf <- penalty_factors(penalty_factor, des$x)
  cv <- rust_result(elastic_net_cv_design(
    as.double(des$x), colnames(des$x), des$y, des$offset, des$weights, family,
    if (is.null(link)) "" else link, as.double(alpha),
    if (is.null(lambda)) double() else as.double(lambda), as.double(nlambda),
    as.double(lambda_min_ratio), isTRUE(standardize), pf, fa$theta, fa$power,
    if (is.null(link_power)) NaN else as.double(link_power), as.double(foldid)
  ))
  cv$fit <- elastic_net_fit(formula, data, family = family, link = link, alpha = alpha,
                            lambda = cv$lambda, standardize = standardize,
                            penalty_factor = penalty_factor, offset = offset, weights = weights,
                            theta = theta, power = power, link_power = link_power)
  cv
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

#' Expected log pointwise predictive density
#'
#' `elpd_loo()` estimates leave-one-out cross-validation from one Bayesian
#' fit by Pareto-smoothed importance sampling (PSIS-LOO); `elpd_waic()`
#' computes WAIC. Both take the pointwise log-likelihood
#' `log p(y_i | theta_s)` of each posterior draw (rows) and observation
#' (columns), and match the loo package.
#'
#' @param log_lik A matrix, one row per posterior draw and one column per
#'   observation.
#' @param r_eff Optional relative efficiency of the draws per observation
#'   (1 for independent draws).
#' @returns A list: `estimates` (named `elpd`, `se`, `p` and `ic`, the
#'   information criterion `-2 elpd`) and `pointwise`; for `elpd_loo()` also
#'   `pareto_k` per observation and `k_threshold`, above which an
#'   observation's estimate is unreliable.
#' @name elpd
#' @examples
#' set.seed(1)
#' y <- rnorm(20)
#' mu <- rnorm(400, mean(y), 1 / sqrt(20))
#' ll <- sapply(y, function(yi) dnorm(yi, mu, 1, log = TRUE))
#' elpd_loo(ll)$estimates
#' elpd_waic(ll)$estimates
NULL

elpd_result <- function(r) {
  r$estimates <- stats::setNames(r$estimates, c("elpd", "se", "p", "ic"))
  r
}

#' @rdname elpd
#' @export
elpd_loo <- function(log_lik, r_eff = NULL) {
  log_lik <- as.matrix(log_lik)
  elpd_result(rust_result(elpd_loo_rust(as.double(t(log_lik)), as.double(ncol(log_lik)),
                                        if (is.null(r_eff)) double() else as.double(r_eff))))
}

#' @rdname elpd
#' @export
elpd_waic <- function(log_lik) {
  log_lik <- as.matrix(log_lik)
  elpd_result(rust_result(elpd_waic_rust(as.double(t(log_lik)), as.double(ncol(log_lik)))))
}
