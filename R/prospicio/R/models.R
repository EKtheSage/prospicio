# Models lane: GLMs, elastic nets, GAMs, metrics, resampling, tuning and
# MCMC diagnostics, over crates/prospicio-r/src/models.rs. R builds the design
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
#'   the over-dispersed Poisson, which accepts negative responses as long
#'   as the fitted means stay positive), `"deviance"` or a fixed number.
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
  package = "prospicio",
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

#' Save and load a model
#'
#' `save_model()` writes a [glm_fit()], [gam_fit()] or [elastic_net_fit()]
#' model to an RDS file: the Rust fit as a versioned JSON artifact (spec,
#' estimates, covariance, fit statistics, fitted values, and provenance
#' with the package version and a hash of the training data; for a GAM
#' also the spline knots and constraints; for an elastic net one artifact
#' per lambda), with the formula terms and factor levels that
#' [stats::predict()] needs. `load_model()` reads it back; the estimates
#' round-trip exactly. A loaded model predicts and simulates but has no
#' training data, so [robust_vcov()] needs the original fit.
#'
#' @param model A `glm_model`, `gam_model` or `elastic_net_model`.
#' @param file Path of the RDS file.
#' @returns `save_model()`: `file`, invisibly. `load_model()`: a model of
#'   the class saved.
#' @export
#' @examples
#' d <- data.frame(claims = c(1, 2, 4, 2, 1, 3), region = c("N", "N", "S", "S", "W", "W"))
#' m <- glm_fit(claims ~ region, d, family = "poisson")
#' f <- tempfile(fileext = ".rds")
#' save_model(m, f)
#' m2 <- load_model(f)
#' identical(coef(m2), coef(m))
save_model <- function(model, file) {
  kind <- if (S7::S7_inherits(model, glm_model)) "glm_model"
    else if (S7::S7_inherits(model, gam_model)) "gam_model"
    else if (S7::S7_inherits(model, elastic_net_model)) "elastic_net_model"
    else stop("model must be a glm_model, gam_model or elastic_net_model")
  saveRDS(list(format = paste0("prospicio.", kind), artifact = model@ptr$to_json(),
               terms = model@terms, xlevels = model@xlevels), file)
  invisible(file)
}

#' @rdname save_model
#' @export
load_model <- function(file) {
  x <- readRDS(file)
  format <- if (is.list(x)) x$format else NULL
  switch(if (is.character(format)) format else "",
    prospicio.glm_model = glm_model(ptr = rust_result(GlmModel$from_json(x$artifact)),
                                      terms = x$terms, xlevels = x$xlevels),
    prospicio.gam_model = gam_model(ptr = rust_result(GamModel$from_json(x$artifact)),
                                      terms = x$terms, xlevels = x$xlevels),
    prospicio.elastic_net_model = elastic_net_model(
      ptr = rust_result(ElasticNetPath$from_json(x$artifact)),
      terms = x$terms, xlevels = x$xlevels
    ),
    stop("not a model saved by save_model()")
  )
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
  package = "prospicio",
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
  package = "prospicio",
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

#' Fit a Bayesian GLM by NUTS
#'
#' Samples the posterior of a GLM with nuts-rs, the Rust core of nutpie.
#' The design is R's `model.matrix()`, as for [glm_fit()]. Coefficients
#' have normal priors with mean 0: standard deviation `intercept_sd` for the
#' intercept and `prior_sd` for the rest (on the link scale; standardize
#' covariates). For the Gaussian, gamma and inverse Gaussian the dispersion
#' is sampled with a half-normal prior of scale `dispersion_scale`, unless
#' `dispersion` fixes it. Chains run in parallel, start near the
#' maximum-likelihood fit, and replay exactly from `seed`. Posterior means
#' and standard deviations match exact grid integration
#' (`validation/scripts/bayes_glm_grid.py`).
#'
#' @inheritParams glm_fit
#' @param prior_sd,intercept_sd Prior standard deviations of the slopes and
#'   the intercept.
#' @param dispersion A fixed dispersion, or `NULL` for the family's default.
#' @param dispersion_scale Scale of the half-normal prior on a sampled
#'   dispersion.
#' @param chains,tune,draws Chains, warm-up draws and kept draws per chain.
#' @param seed Seed, a whole number.
#' @param target_accept Target acceptance rate for step-size adaptation.
#' @param max_depth Largest tree depth.
#' @returns A `bayes_glm_model` with properties `coefficients` (posterior
#'   means), `summary` (a data frame of mean, sd, quantiles, R-hat and ESS
#'   per parameter), `draws` (a matrix, one row per draw), `dispersion_draws`
#'   and `divergences`. Use [stats::coef()], [stats::predict()],
#'   [predict_distribution()] and [bayes_loo()].
#' @export
#' @examples
#' d <- data.frame(claims = rep(c(1, 2, 3, 5), 10), x = rep(c(-1.5, -0.5, 0.5, 1.5), 10))
#' m <- bayes_glm_fit(claims ~ x, d, family = "poisson", chains = 2, tune = 300, draws = 300)
#' m@summary
bayes_glm_fit <- function(formula, data, family = "poisson", link = NULL, offset = NULL,
                          weights = NULL, prior_sd = 2.5, intercept_sd = 10, dispersion = NULL,
                          dispersion_scale = 10, chains = 4, tune = 1000, draws = 1000, seed = 0,
                          target_accept = 0.8, max_depth = 10, theta = NULL, power = NULL,
                          link_power = NULL) {
  des <- model_design(formula, data, offset, weights)
  fa <- family_args(theta, power)
  ptr <- rust_result(bayes_glm_fit_design(
    as.double(des$x), colnames(des$x), des$y, des$offset, des$weights, family,
    if (is.null(link)) "" else link, fa$theta, fa$power,
    if (is.null(link_power)) NaN else as.double(link_power), as.double(prior_sd),
    as.double(intercept_sd), if (is.null(dispersion)) NaN else as.double(dispersion),
    as.double(dispersion_scale),
    as.double(c(chains, tune, draws, seed, target_accept, max_depth))
  ))
  bayes_glm_model(ptr = ptr, terms = des$terms, xlevels = des$xlevels)
}

#' Sampled Bayesian GLM (class)
#'
#' Returned by [bayes_glm_fit()].
#'
#' @param ptr,terms,xlevels Internal.
#' @returns A `bayes_glm_model` object.
#' @export
bayes_glm_model <- S7::new_class(
  "bayes_glm_model",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("BayesGlmModel"),
    terms = S7::class_any,
    xlevels = S7::class_any,
    coefficients = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(self@ptr$posterior_mean(), self@ptr$names())
    }),
    summary = S7::new_property(S7::class_any, getter = function(self) {
      as.data.frame(self@ptr$summary())
    }),
    draws = S7::new_property(S7::class_any, getter = function(self) {
      nm <- self@ptr$names()
      matrix(self@ptr$coefficient_draws(), ncol = length(nm), byrow = TRUE,
             dimnames = list(NULL, nm))
    }),
    dispersion_draws = S7::new_property(S7::class_double, getter = function(self) {
      self@ptr$dispersion_draws()
    }),
    divergences = S7::new_property(S7::class_double, getter = function(self) self@ptr$divergences())
  ),
  constructor = function(ptr, terms, xlevels) {
    S7::new_object(S7::S7_object(), ptr = ptr, terms = terms, xlevels = xlevels)
  }
)

S7::method(print, bayes_glm_model) <- function(x, ...) {
  cat(sprintf("<bayes_glm_model> %d chains, %d divergences\n", as.integer(x@ptr$chains()),
              as.integer(x@divergences)))
  print(x@summary)
  invisible(x)
}

#' PSIS-LOO of a Bayesian GLM
#'
#' Leave-one-out expected log predictive density on the training data, by
#' Pareto-smoothed importance sampling ([elpd_loo()]), with each
#' observation's relative efficiency estimated from the chains. Its
#' `pointwise` values feed [stacking_weights()].
#'
#' @param model A [bayes_glm_model].
#' @returns As [elpd_loo()].
#' @export
#' @examples
#' d <- data.frame(claims = rep(c(1, 2, 3, 5), 10), x = rep(c(-1.5, -0.5, 0.5, 1.5), 10))
#' m <- bayes_glm_fit(claims ~ x, d, family = "poisson", chains = 2, tune = 300, draws = 300)
#' bayes_loo(m)$estimates
bayes_loo <- function(model) {
  elpd_result(rust_result(model@ptr$loo()))
}

for (cls in list(glm_model, gam_model, bayes_glm_model)) {
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
#' @param object A [glm_model], [elastic_net_model], [gam_model] or
#'   [bayes_glm_model] (posterior predictive draws).
#' @param newdata A data frame with the model's terms.
#' @param n_sims Number of simulations.
#' @param seed Generator seed.
#' @param offset Optional offset added to any `offset()` terms.
#' @param weights Optional prior weights.
#' @param parameters For a [glm_model], how the coefficients are drawn:
#'   `"normal"` (`beta ~ N(beta_hat, Sigma)`; through a log link the draws'
#'   mean is `mu_hat * exp(x' Sigma x / 2)`), `"mean_preserving"` (the same,
#'   with each row's linear predictor shifted so its draws average the
#'   fitted mean exactly; log or identity link) or `"fixed"` (process
#'   uncertainty only).
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

S7::method(predict_distribution, glm_model) <- function(object, newdata, n_sims, seed,
                                                       offset = NULL, weights = NULL,
                                                       parameters = c("normal", "mean_preserving",
                                                                      "fixed"), ...) {
  parameters <- match.arg(parameters)
  nd <- new_design(object, newdata, offset)
  w <- if (is.null(weights)) double() else as.double(weights)
  ptr <- rust_result(object@ptr$predict_distribution(
    as.double(nd$x), colnames(nd$x), nd$offset, w, as.double(n_sims), as.double(seed), parameters
  ), s7_call())
  predictive_distribution(ptr = ptr)
}

for (cls in list(gam_model, bayes_glm_model)) {
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
#' continuous ranked probability score of draws for an outcome;
#' `pinball_loss()` the weighted mean pinball (quantile) loss of predicted
#' `alpha` quantiles, `sum(w * u * (alpha - (u < 0))) / sum(w)` with
#' `u = y - pred`, lowest in expectation at the true quantile.
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
#' @param alpha Quantile level in `(0, 1)`.
#' @returns A number; for `pit_values()` one value per outcome; for
#'   `lift_table()` a data frame with columns `exposure`, `expected` and
#'   `actual`.
#' @name model_metrics
#' @examples
#' family_deviance("poisson", c(1, 0, 3), c(1, 0.5, 2))
#' gini_index(c(0, 1), c(0.1, 0.9))
#' lift_table(c(0, 1, 2, 3), c(0.1, 0.9, 2.1, 2.9), bands = 2)
#' crps_draws(c(1, 2, 3), 2)
#' pinball_loss(c(1, 0), c(0, 1), 0.9)
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
pinball_loss <- function(y, pred, alpha, weights = NULL) {
  rust_result(pinball_rust(as.double(y), as.double(pred), as.double(alpha),
                           if (is.null(weights)) double() else as.double(weights)))
}

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

#' Monitor a model: actual against expected by period
#'
#' For a stored model's predictions on new data, the actual and expected
#' totals of each period, `A = sum(w * y)` and `E = sum(w * mu)`, with the
#' z-score `(A - E) / sqrt(dispersion * sum(w * V(mu)))` under the model's
#' variance function `V`: about standard normal while the model holds. The
#' trend is the slope of `A / E - 1` per period step (periods in sorted
#' order), weighted by each period's precision, with its standard error;
#' `trend_z` beyond about 2 suggests drift.
#'
#' @param periods One period label per row (numbers or strings).
#' @param actual Observed responses.
#' @param expected The model's predicted means for the same rows, e.g.
#'   `predict(model, newdata)`.
#' @param family As in [glm_fit()].
#' @param weights Optional prior weights as fitted (exposure for a rate);
#'   leave out for counts with exposure in the offset.
#' @param dispersion The model's dispersion, e.g. `model@dispersion`.
#' @param theta,power Negative binomial `theta`, Tweedie `power`.
#' @returns A data frame with one row per period: `period`, `n`, `weight`,
#'   `actual`, `expected`, `ratio`, `std_dev` and `z`; attributes `total`
#'   (the same for all periods) and `trend` (`slope`, `std_error`, `z`).
#' @export
#' @examples
#' actual_vs_expected(c(2023, 2023, 2024, 2024), c(1, 3, 2, 6), rep(2, 4), "poisson")
actual_vs_expected <- function(periods, actual, expected, family, weights = NULL,
                               dispersion = 1, theta = NULL, power = NULL) {
  fa <- family_args(theta, power)
  r <- rust_result(actual_vs_expected_rust(
    periods, as.double(actual), as.double(expected),
    if (is.null(weights)) double() else as.double(weights), family, fa$theta, fa$power,
    as.double(dispersion)
  ))
  out <- data.frame(period = periods[r$first_row], n = r$n, weight = r$weight,
                    actual = r$actual, expected = r$expected)
  out$ratio <- out$actual / out$expected
  out$std_dev <- r$std_dev
  out$z <- (out$actual - out$expected) / out$std_dev
  tot <- stats::setNames(r$total, c("n", "weight", "actual", "expected", "std_dev"))
  attr(out, "total") <- c(tot, ratio = tot[["actual"]] / tot[["expected"]],
                          z = (tot[["actual"]] - tot[["expected"]]) / tot[["std_dev"]])
  attr(out, "trend") <- c(slope = r$trend, std_error = r$trend_std_error,
                          z = r$trend / r$trend_std_error)
  out
}

#' Bayesian and hierarchical stacking
#'
#' Model weights with a posterior, sampled by NUTS from pointwise held-out
#' log densities, the log score of the mixture as the likelihood (Yao et
#' al., 2018 and 2022). `bayes_stacking()` puts a Dirichlet prior on one
#' weight vector. `hierarchical_stacking()` lets the weights vary with
#' covariates, `w = softmax(alpha + B x)` against the last model as
#' reference, with the priors of BayesBlend's `HierarchicalBayesStacking`:
#' a model can be trusted in one part of the portfolio and not another.
#' Scale continuous covariates (BayesBlend divides by twice their standard
#' deviation) and dummy-code discrete ones first, with the dummies as the
#' first `discrete` columns. Without pooling, posterior moments match exact
#' grid integration (`validation/scripts/stacking_grid.py`).
#'
#' With `partial_pooling = TRUE`, each model's slopes on the discrete
#' columns, and separately on the continuous ones, are drawn around a
#' model-level mean, which is drawn around a global mean; `pooling_scales()`
#' sets the scales (0 removes a level: `tau_mu_global = 0` fixes the global
#' mean at 0, `tau_mu_* = 0` pools completely, `tau_sigma_* = 0` sets every
#' slope to its model's mean). BayesBlend warns that pooling needs at least
#' three covariates. `adaptive`, the rate of an exponential prior on
#' `lambda`, multiplies the prior scales by `N^lambda`, which weakens them
#' as the data grow.
#'
#' @param lpd A matrix of held-out log densities, one row per observation
#'   and one column per model (for example each model's
#'   `bayes_loo(m)$pointwise`, as columns).
#' @param covariates A matrix or data frame of numeric covariates, one row
#'   per observation.
#' @param concentration Dirichlet concentration, one per model; `NULL` is
#'   uniform.
#' @param discrete Number of leading columns of `covariates` that are dummy
#'   codes of discrete covariates.
#' @param alpha_loc,alpha_scale,beta_loc,beta_scale Normal priors on the
#'   intercepts and (without pooling) slopes.
#' @param partial_pooling Pool the slopes (BayesBlend's pooling model).
#' @param pooling Pooling scales, from `pooling_scales()`.
#' @param tau_mu_global,tau_mu_discrete,tau_mu_continuous,tau_sigma_discrete,tau_sigma_continuous
#'   Scales of the global mean, the model-level means and the slopes around
#'   them; BayesBlend's defaults are all 1.
#' @param adaptive `NULL`, or the rate of the exponential prior on `lambda`
#'   (BayesBlend's default is 4).
#' @param chains,tune,draws,seed Sampler settings.
#' @returns A `stacking_fit` with properties `weights` (posterior mean
#'   weights: one row per observation for hierarchical stacking, one row
#'   otherwise), `alpha_draws`, `beta_draws` and `divergences`. Use
#'   [stats::predict()] with new covariates for their weights, and
#'   [blend_predictive()] with the weight matrix.
#' @name bayesian_stacking
#' @examples
#' x <- seq(-0.5, 0.5, length.out = 100)
#' lpd <- cbind(a = ifelse(x < 0, -0.5, -2), b = ifelse(x < 0, -2, -0.5))
#' h <- hierarchical_stacking(lpd, data.frame(x = x), chains = 2, tune = 300, draws = 300)
#' predict(h, data.frame(x = c(-0.4, 0.4)))
#' bayes_stacking(lpd, chains = 2, tune = 300, draws = 300)@weights
NULL

#' @rdname bayesian_stacking
#' @export
bayes_stacking <- function(lpd, concentration = NULL, chains = 4, tune = 1000, draws = 1000,
                           seed = 0) {
  lpd <- as.matrix(lpd)
  ptr <- rust_result(bayes_stacking_rust(
    as.double(lpd), ncol(lpd), if (is.null(concentration)) double() else as.double(concentration),
    as.double(c(chains, tune, draws, seed))
  ))
  stacking_fit(ptr = ptr, models = model_weights_names(lpd, seq_len(ncol(lpd))), covariates = NULL)
}

#' @rdname bayesian_stacking
#' @export
hierarchical_stacking <- function(lpd, covariates, discrete = 0, alpha_loc = 0, alpha_scale = 1,
                                  beta_loc = 0, beta_scale = 1, partial_pooling = FALSE,
                                  pooling = pooling_scales(), adaptive = NULL, chains = 4,
                                  tune = 1000, draws = 1000, seed = 0) {
  lpd <- as.matrix(lpd)
  x <- as.matrix(covariates)
  ptr <- rust_result(hierarchical_stacking_rust(
    as.double(lpd), ncol(lpd), as.double(x), ncol(x),
    as.double(c(alpha_loc, alpha_scale, beta_loc, beta_scale)),
    if (partial_pooling) as.double(pooling[c("tau_mu_global", "tau_mu_discrete",
                                              "tau_mu_continuous", "tau_sigma_discrete",
                                              "tau_sigma_continuous")]) else double(),
    if (is.null(adaptive)) NA_real_ else as.double(adaptive),
    as.double(discrete),
    as.double(c(chains, tune, draws, seed))
  ))
  stacking_fit(ptr = ptr, models = model_weights_names(lpd, seq_len(ncol(lpd))), covariates = x)
}

#' @rdname bayesian_stacking
#' @export
pooling_scales <- function(tau_mu_global = 1, tau_mu_discrete = 1, tau_mu_continuous = 1,
                           tau_sigma_discrete = 1, tau_sigma_continuous = 1) {
  c(tau_mu_global = tau_mu_global, tau_mu_discrete = tau_mu_discrete,
    tau_mu_continuous = tau_mu_continuous, tau_sigma_discrete = tau_sigma_discrete,
    tau_sigma_continuous = tau_sigma_continuous)
}

#' Posterior stacking weights (class)
#'
#' Returned by [bayes_stacking()] and [hierarchical_stacking()].
#'
#' @param ptr,models,covariates Internal.
#' @returns A `stacking_fit` object.
#' @export
stacking_fit <- S7::new_class(
  "stacking_fit",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("StackingModel"),
    models = S7::class_any,
    covariates = S7::class_any,
    weights = S7::new_property(S7::class_any, getter = function(self) {
      stacking_weights_at(self, self@covariates)
    }),
    alpha_draws = S7::new_property(S7::class_double, getter = function(self) self@ptr$alpha_draws()),
    beta_draws = S7::new_property(S7::class_double, getter = function(self) self@ptr$beta_draws()),
    divergences = S7::new_property(S7::class_double, getter = function(self) self@ptr$divergences())
  ),
  constructor = function(ptr, models, covariates) {
    S7::new_object(S7::S7_object(), ptr = ptr, models = models, covariates = covariates)
  }
)

stacking_weights_at <- function(fit, x) {
  x <- if (is.null(x)) matrix(0, 0, 0) else as.matrix(x)
  w <- rust_result(fit@ptr$weights(as.double(x), ncol(x)))
  matrix(w, ncol = length(fit@models), dimnames = list(NULL, names(fit@models)))
}

S7::method(predict, stacking_fit) <- function(object, newdata = NULL, ...) {
  stacking_weights_at(object, newdata)
}

#' Model weights for blending: stacking and pseudo-BMA
#'
#' Weights from pointwise held-out log predictive densities (Yao, Vehtari,
#' Simpson and Gelman, 2018), for any model: PSIS-LOO `pointwise` values
#' from [elpd_loo()] for a Bayesian fit, or cross-validated log densities
#' for a GLM, GAM or neural network. `stacking_weights()` maximizes the log
#' score of the mixture of the models' predictive distributions; a model
#' that adds nothing gets weight exactly 0. `pseudo_bma_weights()` is
#' proportional to `exp(elpd)`; with `bootstrap = TRUE` (pseudo-BMA+) it is
#' averaged over Bayesian-bootstrap replicates, which keeps a model that is
#' only slightly better from taking all the weight. Blend the models'
#' simulations with [blend_predictive()].
#'
#' Stacking matches an SLSQP optimum polished by Newton's method to 1e-10
#' (`validation/scripts/stacking_weights.py`); `loo::stacking_weights()`
#' stops earlier and agrees to about 1e-3.
#'
#' @param lpd A matrix (or data frame) of log densities, one row per
#'   observation and one column per model.
#' @param bootstrap Use the Bayesian bootstrap (pseudo-BMA+).
#' @param n_draws Bootstrap replicates.
#' @param seed Seed; replicate `b` uses stream `b`.
#' @returns A named vector of weights summing to 1, named by the columns
#'   of `lpd`.
#' @name model_weights
#' @examples
#' lpd <- cbind(a = c(-0.1, -0.1, -3, -3), b = c(-3, -3, -0.1, -0.1))
#' stacking_weights(lpd)
#' pseudo_bma_weights(lpd, bootstrap = FALSE)
NULL

model_weights_names <- function(lpd, w) {
  nm <- colnames(lpd)
  if (is.null(nm)) nm <- paste0("model", seq_along(w))
  stats::setNames(w, nm)
}

#' @rdname model_weights
#' @export
stacking_weights <- function(lpd) {
  lpd <- as.matrix(lpd)
  model_weights_names(lpd, rust_result(stacking_weights_rust(as.double(lpd), ncol(lpd))))
}

#' @rdname model_weights
#' @export
pseudo_bma_weights <- function(lpd, bootstrap = TRUE, n_draws = 1000, seed = 0) {
  lpd <- as.matrix(lpd)
  w <- pseudo_bma_weights_rust(as.double(lpd), ncol(lpd), if (bootstrap) n_draws else 0, seed)
  model_weights_names(lpd, rust_result(w))
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
