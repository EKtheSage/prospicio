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

# Coefficient bounds, one per design column: a single value for every
# column, an unnamed vector with one per column, or a named vector for some
# (the rest unbounded). The intercept's is ignored.
coef_limits <- function(limits, x, default, what) {
  if (is.null(limits)) return(double())
  if (is.null(names(limits)) && length(limits) == 1) return(rep(as.double(limits), ncol(x)))
  out <- stats::setNames(rep(default, ncol(x)), colnames(x))
  if (is.null(names(limits))) {
    if (length(limits) != ncol(x)) {
      stop(sprintf("%s needs 1 or %d values, one per design column", what, ncol(x)))
    }
    out[] <- limits
  } else {
    unknown <- setdiff(names(limits), names(out))
    if (length(unknown)) stop("unknown design columns in ", what, ": ", toString(unknown))
    out[names(limits)] <- limits
  }
  as.double(out)
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
#' The rating variables of Python's `Terms` are formula terms here:
#' interactions `a:b`, raw powers `poly(x, 3, raw = TRUE)`, hinges
#' `I(pmax(0, x - 30))`, steps `I(x >= 30)`, bins `cut(x, c(-Inf, 30, 50,
#' Inf))`, a centred log `log(x / 40)` and natural splines
#' `splines::ns(x, knots = ...)`. For an ordered factor in cumulative coding
#' (each coefficient the step from the level below), use one step column
#' per level, `I(as.integer(f) >= 2)` and so on, on a factor with its levels
#' in order.
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
#'   `log_likelihood`, `aic`, `bic` (counting the coefficients as `aic`
#'   does; [stats::BIC()] also counts an estimated dispersion), `df_resid`,
#'   `iterations` and `fitted`. Use
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

#' GLMs with a family parameter estimated
#'
#' `glm_nb_fit()` fits a negative binomial GLM with its `theta` estimated
#' with the coefficients by maximum likelihood, as `MASS::glm.nb()`: IRLS
#' and Newton's method on `theta` alternate until both settle. `se_theta` is
#' `MASS`'s standard error and `aic` counts `theta`.
#'
#' `tweedie_glm_fit()` fits a Tweedie GLM with its power estimated with the
#' coefficients and dispersion: the profile log-likelihood over the power
#' (the exact Tweedie density, maximized over the dispersion at each power)
#' is maximized by Brent's method, with a 95% profile-likelihood interval
#' (`lower`, `upper`; an end flagged open is a search bound).
#' `tweedie_profile()` evaluates that profile on a grid of powers, as R's
#' `tweedie::tweedie.profile`, and refines the best.
#'
#' @inheritParams glm_fit
#' @param link The link, `"log"` by default.
#' @param min_power,max_power The power's search bounds, inside (1, 2).
#' @param powers Powers to profile, increasing, inside (1, 2).
#' @returns A list with `model`, a [glm_model] at the estimate (its
#'   standard errors conditional on it), and, for `glm_nb_fit()`, `theta`,
#'   `se_theta`, `log_likelihood` and `aic`; for `tweedie_glm_fit()`,
#'   `power`, `phi`, `log_likelihood`, `lower`, `upper`, `lower_open`,
#'   `upper_open` and `at_boundary`; for `tweedie_profile()`, `profile` (a
#'   data frame of `power`, `log_likelihood` and `dispersion`), `power`,
#'   `phi` and `max_log_likelihood`.
#' @export
#' @examples
#' d <- data.frame(x = rep(0:2, 20), y = (0:59 * 7) %% 11)
#' nb <- glm_nb_fit(y ~ x, d)
#' nb$theta
glm_nb_fit <- function(formula, data, link = "log", offset = NULL, weights = NULL) {
  des <- model_design(formula, data, offset, weights)
  r <- rust_result(glm_nb_fit_design(as.double(des$x), colnames(des$x), des$y, des$offset,
                                     des$weights, link))
  r$model <- glm_model(ptr = r$model, terms = des$terms, xlevels = des$xlevels)
  r
}

#' @rdname glm_nb_fit
#' @export
tweedie_glm_fit <- function(formula, data, link = "log", offset = NULL, weights = NULL,
                            min_power = 1.01, max_power = 1.99) {
  des <- model_design(formula, data, offset, weights)
  r <- rust_result(tweedie_glm_fit_design(as.double(des$x), colnames(des$x), des$y, des$offset,
                                          des$weights, link, as.double(min_power),
                                          as.double(max_power)))
  r$model <- glm_model(ptr = r$model, terms = des$terms, xlevels = des$xlevels)
  r
}

#' @rdname glm_nb_fit
#' @export
tweedie_profile <- function(formula, data, powers, link = "log", offset = NULL, weights = NULL) {
  des <- model_design(formula, data, offset, weights)
  r <- rust_result(tweedie_profile_design(as.double(des$x), colnames(des$x), des$y, des$offset,
                                          des$weights, link, as.double(powers)))
  list(profile = data.frame(power = r$powers, log_likelihood = r$log_likelihood,
                            dispersion = r$dispersion),
       power = r$power, phi = r$phi, max_log_likelihood = r$max_log_likelihood,
       model = glm_model(ptr = r$model, terms = des$terms, xlevels = des$xlevels))
}

#' Tests of nested GLMs
#'
#' `f_test()` is the F test of `small` within `big`,
#' `F = (D_small - D_big) / (q phi_big)` on `F(q, n - p_big)`, for a family
#' whose dispersion is estimated; `likelihood_ratio_test()` the scaled
#' deviance drop `(D_small - D_big) / phi_big` on `chi2(q)`, for one whose
#' dispersion is known (Poisson, binomial). They agree with
#' `anova(small, big, test = "F")` and `test = "Chisq"` (CAS Monograph 5,
#' section 6.2.1). The models are not checked to be nested: `small`'s
#' columns must lie in the span of `big`'s, on the same rows, family and
#' weights.
#'
#' @param small,big [glm_fit()] models, `small` with fewer coefficients.
#' @returns A list with `statistic`, `df` (`q`), `df_resid` (`Inf` for the
#'   likelihood ratio test), `deviance_drop` and `p_value`.
#' @export
#' @examples
#' d <- data.frame(y = c(1, 3, 2, 5, 4, 6, 8, 7), x = 1:8, z = c(0, 1, 0, 1, 1, 0, 1, 0))
#' f_test(glm_fit(y ~ x, d, "gaussian"), glm_fit(y ~ x + z, d, "gaussian"))$p_value
f_test <- function(small, big) {
  rust_result(glm_nested_test_rust(small@ptr, big@ptr, "f"))
}

#' @rdname f_test
#' @export
likelihood_ratio_test <- function(small, big) {
  rust_result(glm_nested_test_rust(small@ptr, big@ptr, "lr"))
}

#' Variance inflation factors
#'
#' `1 / (1 - R2_j)` for each column of the design of `formula`, with `R2_j`
#' that of column `j` regressed on the others and an intercept (CAS
#' Monograph 5, section 2.9; `car::vif` for single-column terms). The
#' intercept gets `NA`.
#'
#' @inheritParams glm_fit
#' @returns A named numeric vector, one per design column.
#' @export
#' @examples
#' d <- data.frame(y = 1:6, x = c(1, 2, 3, 4, 5, 7), z = c(2, 1, 4, 3, 6, 5))
#' vif(y ~ x + z, d)
vif <- function(formula, data) {
  des <- model_design(formula, data, NULL, NULL)
  v <- rust_result(vif_rust(as.double(des$x), colnames(des$x)))
  v[is.nan(v)] <- NA_real_
  stats::setNames(v, colnames(des$x))
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
    bic = S7::new_property(S7::class_double, getter = function(self) self@ptr$bic()),
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
#' `save_model()` writes a [glm_fit()], [gam_fit()], [elastic_net_fit()],
#' [cann_fit()] or [attention_cann_fit()] model to an RDS file: the Rust
#' fit as a versioned JSON artifact (spec, estimates, covariance, fit
#' statistics, fitted values, and provenance with the package version and
#' a hash of the training data; for a GAM also the spline knots and
#' constraints; for an elastic net one artifact per lambda; for a network
#' its feature scaling and every weight), with the formula terms and
#' factor levels that
#' [stats::predict()] needs. `load_model()` reads it back; the estimates
#' round-trip exactly. A loaded model predicts and simulates but has no
#' training data, so [robust_vcov()] needs the original fit.
#'
#' @param model A `glm_model`, `gam_model`, `elastic_net_model`,
#'   `cann_model` or `attention_cann_model`.
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
    else if (S7::S7_inherits(model, cann_model)) "cann_model"
    else if (S7::S7_inherits(model, attention_cann_model)) "attention_cann_model"
    else stop("model must be a glm_model, gam_model, elastic_net_model, cann_model or attention_cann_model")
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
    prospicio.cann_model = cann_model(ptr = rust_result(CannModel$from_json(x$artifact)),
                                        terms = x$terms, xlevels = x$xlevels),
    prospicio.attention_cann_model = attention_cann_model(
      ptr = rust_result(AttentionCannModel$from_json(x$artifact)),
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
#' @param lower_limits,upper_limits Optional bounds on the coefficients, on
#'   the design's scale, as glmnet's `lower.limits` and `upper.limits`: one
#'   value for every column, one per design column, or a named vector for
#'   some (the rest unbounded). Lower bounds are at most 0, upper at least
#'   0; the intercept is never bounded. `lower_limits = 0` makes every
#'   coefficient non-negative, which keeps a cumulative step basis monotone.
#' @param relax Also refit each `lambda`'s non-zero columns without the
#'   penalty (glmnet's `relax = TRUE`), within the same bounds; then
#'   [stats::coef()] and [stats::predict()] take `gamma`, the blend
#'   `gamma * penalized + (1 - gamma) * refitted`.
#' @returns An `elastic_net_model` with properties `lambda`, `coefficients`
#'   (a matrix, one column per `lambda`), `deviance`, `deviance_ratio`,
#'   `df` (non-zero coefficients), `null_deviance`, `alpha` and `relaxed`.
#'   Use [stats::coef()], [stats::predict()] and [predict_distribution()],
#'   each with a `lambda` from the path. Tune `lambda` and `alpha` with
#'   [k_fold()] and [family_deviance()], or with [elastic_net_cv()].
#' @export
#' @examples
#' set.seed(1)
#' d <- data.frame(x1 = rnorm(50), x2 = rnorm(50))
#' d$y <- 1 + 2 * d$x1 + rnorm(50)
#' m <- elastic_net_fit(y ~ x1 + x2, d, family = "gaussian", alpha = 1)
#' m@df[c(1, 20, 100)]
#' coef(m, lambda = m@lambda[20])
#' # Non-negative slopes, and the relaxed lasso.
#' r <- elastic_net_fit(y ~ x1 + x2, d, family = "gaussian", lower_limits = 0, relax = TRUE)
#' coef(r, lambda = r@lambda[20], gamma = 0)
elastic_net_fit <- function(formula, data, family = "poisson", link = NULL, alpha = 1,
                            lambda = NULL, nlambda = 100, lambda_min_ratio = 1e-4,
                            standardize = TRUE, penalty_factor = NULL, offset = NULL,
                            weights = NULL, theta = NULL, power = NULL, link_power = NULL,
                            lower_limits = NULL, upper_limits = NULL, relax = FALSE) {
  des <- model_design(formula, data, offset, weights)
  fa <- family_args(theta, power)
  pf <- penalty_factors(penalty_factor, des$x)
  ptr <- rust_result(elastic_net_fit_design(
    as.double(des$x), colnames(des$x), des$y, des$offset, des$weights, family,
    if (is.null(link)) "" else link, as.double(alpha),
    if (is.null(lambda)) double() else as.double(lambda), as.double(nlambda),
    as.double(lambda_min_ratio), isTRUE(standardize), pf,
    coef_limits(lower_limits, des$x, -Inf, "lower_limits"),
    coef_limits(upper_limits, des$x, Inf, "upper_limits"), isTRUE(relax),
    fa$theta, fa$power, if (is.null(link_power)) NaN else as.double(link_power)
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
    alpha = S7::new_property(S7::class_double, getter = function(self) self@ptr$alpha()),
    relaxed = S7::new_property(S7::class_logical, getter = function(self) self@ptr$is_relaxed())
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

S7::method(coef, elastic_net_model) <- function(object, lambda = NULL, gamma = 1, ...) {
  b <- object@coefficients
  if (gamma != 1) b[] <- rust_result(object@ptr$relaxed_coefficients(as.double(gamma)))
  if (is.null(lambda)) return(b)
  stats::setNames(b[, lambda_index(object, lambda)], object@ptr$names())
}

S7::method(predict, elastic_net_model) <- function(object, newdata, lambda = NULL, offset = NULL,
                                                    gamma = 1, ...) {
  nd <- new_design(object, newdata, offset)
  one <- function(k) {
    rust_result(object@ptr$predict(as.double(k), as.double(nd$x), colnames(nd$x), nd$offset,
                                   as.double(gamma)))
  }
  if (is.null(lambda) && length(object@lambda) > 1) {
    return(vapply(seq_along(object@lambda), one, double(nrow(nd$x))))
  }
  one(lambda_index(object, lambda))
}

S7::method(print, elastic_net_model) <- function(x, ...) {
  lam <- x@lambda
  cat(sprintf("<elastic_net_model> %s, alpha = %s, %d lambdas from %s to %s%s\n",
              x@ptr$family(), format(x@alpha), length(lam), format(max(lam), digits = 6),
              format(min(lam), digits = 6), if (x@relaxed) ", relaxed" else ""))
  invisible(x)
}

#' Fit a generalized additive model
#'
#' A [glm_fit()] model in which each name in `smooths` (a numeric term of
#' the formula) becomes a cubic P-spline (mgcv's `s(x, bs = "ps")`), with
#' smoothing chosen by GCV when the dispersion is estimated and UBRE when
#' it is fixed (mgcv's `method = "GCV.Cp"`), by REML (`smoothing = "reml"`,
#' mgcv's `method = "REML"`, which estimates an unknown dispersion with the
#' smoothing parameters as `reml_scale`), or fixed by `smoothing`.
#'
#' @inheritParams glm_fit
#' @param smooths Names of numeric terms to smooth.
#' @param n_basis Basis functions per smooth (recycled; 10 by default).
#' @param bs The basis of each smooth (recycled): `"ps"`, a P-spline;
#'   `"cp"`, a cyclic P-spline (mgcv's `bs = "cp"`), whose ends meet in value
#'   and slope over its period; or `"mpi"` / `"mpd"`, a monotone increasing
#'   or decreasing P-spline (scam's), which takes GCV, UBRE or fixed
#'   smoothing but not REML.
#' @param period A named list of `c(lo, hi)` periods for cyclic smooths
#'   (mgcv's `knots = list(x = c(lo, hi))`); a cyclic smooth left out takes
#'   the data's range.
#' @param smoothing `"auto"`, `"gcv"`, `"ubre"`, `"reml"`, or a numeric vector of
#'   fixed smoothing parameters: one per smooth, then one per tensor margin.
#' @param tensors A list of tensor product smooths (mgcv's
#'   `te(..., bs = "ps", np = FALSE)`), each a character vector of two or
#'   more numeric terms. Each margin has its own smoothing parameter.
#' @param tensor_basis Basis functions per margin of each tensor: one number
#'   (recycled) or a list with one vector per tensor (5 by default).
#' @returns A `gam_model` with properties `coefficients`, `lambdas`, `edf`,
#'   `dispersion`, `deviance`, `score`, `reml_scale` (`NA` unless REML) and
#'   `fitted`.
#' @export
#' @examples
#' d <- data.frame(x = seq(0, 1, length.out = 100))
#' d$y <- sin(6 * d$x) + 2
#' m <- gam_fit(y ~ x, d, smooths = "x", family = "gaussian")
#' m@edf
#' predict(m, data.frame(x = 0.5)) - (sin(3) + 2)
#' # A seasonal effect whose ends meet: months 0 to 12.
#' s <- data.frame(month = seq(0.25, 11.75, by = 0.5))
#' s$y <- 2 + sin(2 * pi * s$month / 12)
#' cm <- gam_fit(y ~ month, s, smooths = "month", family = "gaussian", bs = "cp",
#'               period = list(month = c(0, 12)))
#' predict(cm, data.frame(month = c(0, 12)))
gam_fit <- function(formula, data, smooths, n_basis = 10, family = "poisson", link = NULL,
                    offset = NULL, weights = NULL, dispersion = NULL, theta = NULL,
                    power = NULL, smoothing = "auto", tensors = list(), tensor_basis = 5,
                    bs = "ps", period = list()) {
  des <- model_design(formula, data, offset, weights)
  smooths <- as.character(smooths)
  unknown <- setdiff(names(period), smooths)
  if (length(unknown)) stop("period given for terms that are not smooths: ", toString(unknown))
  periods <- unlist(lapply(smooths, function(s) {
    p <- period[[s]]
    if (is.null(p)) c(NaN, NaN) else as.double(p)
  }))
  fa <- family_args(theta, power)
  da <- dispersion_args(dispersion)
  fixed <- is.numeric(smoothing)
  ptr <- rust_result(gam_fit_design(
    as.double(des$x), colnames(des$x), des$y, des$offset, des$weights, family,
    if (is.null(link)) "" else link, da$kind, da$value, fa$theta, fa$power,
    smooths, rep_len(as.double(n_basis), length(smooths)),
    rep_len(as.character(bs), length(smooths)), as.double(periods),
    if (fixed) "fixed" else smoothing, if (fixed) as.double(smoothing) else double(),
    lapply(tensors, as.character),
    if (is.list(tensor_basis)) lapply(tensor_basis, as.double) else
      lapply(tensors, function(t) rep_len(as.double(tensor_basis), length(t)))
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
    log_likelihood = S7::new_property(S7::class_double, getter = function(self) self@ptr$log_likelihood()),
    aic = S7::new_property(S7::class_double, getter = function(self) self@ptr$aic()),
    score = S7::new_property(S7::class_double, getter = function(self) self@ptr$score()),
    reml_scale = S7::new_property(S7::class_double, getter = function(self) self@ptr$reml_scale()),
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
#' @param model A [bayes_glm_model], a [bayes_odp_fit] (or a single-segment
#'   [bayes_odp_fits]), [bayes_bf_fit] or [bayes_bf_normalised_fit] for the
#'   PSIS-LOO of its observed increments, a [meyers_fit] (or a single-segment [meyers_fits]) for
#'   that of its observed cells on the log scale, or a [compartmental_fit]
#'   for that of its observed outstanding and paid amounts (with
#'   `n_outstanding`, `elpd_outstanding` and `elpd_paid`).
#' @returns As [elpd_loo()].
#' @export
#' @examples
#' d <- data.frame(claims = rep(c(1, 2, 3, 5), 10), x = rep(c(-1.5, -0.5, 0.5, 1.5), 10))
#' m <- bayes_glm_fit(claims ~ x, d, family = "poisson", chains = 2, tune = 300, draws = 300)
#' bayes_loo(m)$estimates
bayes_loo <- function(model) {
  elpd_result(rust_result(model@ptr$loo()))
}

#' Posterior predictive checks
#'
#' Does data simulated from the fitted model look like the data it was
#' fitted to? Each check compares the observed outcomes with replicated
#' data, one replicated dataset per posterior draw.
#'
#' - `bayes_ppc()` and `ppc_check()`: a test statistic of the observed data
#'   against its distribution over the replicates, with the posterior
#'   predictive p-value `P(T(y_rep) >= T(y))` (ties count half). A p-value
#'   near 0 or 1 says the model cannot reproduce that feature of the data.
#' - `bayes_loo_pit()` and `loo_pit()`: each observation's leave-one-out
#'   probability integral transform, its place among its replicates under
#'   the Pareto-smoothed importance weights for leaving it out (as R
#'   `loo`'s `E_loo` of `y_rep <= y`), randomized over ties. Uniform for a
#'   calibrated model; U-shaped when the predictive distributions are too
#'   narrow, humped when too wide.
#' - `bayes_replicate()`: the replicated training datasets.
#'
#' @param model A [bayes_glm_model]; the checks run on its training data.
#' @param statistics Test statistics: `"mean"`, `"sd"`, `"min"`, `"max"`,
#'   `"proportion_zero"`, or numbers (probabilities) for quantiles.
#' @param seed Seed of the replicates (and of the randomization over ties).
#' @param observed The observed outcomes.
#' @param replicated A matrix, one replicated dataset per row (posterior
#'   draw), one column per observation.
#' @param groups Each observation's group, from 1, to check each statistic
#'   within each group; `NULL` for none.
#' @param log_lik The pointwise log-likelihood, draws by observations.
#' @param r_eff Each observation's relative efficiency; `NULL` for 1.
#' @returns `bayes_ppc()` and `ppc_check()`: a data frame with `name`,
#'   `observed` and `p_value`, and an attribute `replicated`, a matrix of
#'   each statistic of each replicate. `bayes_loo_pit()` and `loo_pit()`: a
#'   list with `pit` and `pareto_k` per observation and `ks`, the
#'   Kolmogorov–Smirnov distance from the uniform. `bayes_replicate()`: a
#'   matrix, draws by observations.
#' @export
#' @examples
#' d <- data.frame(claims = rep(c(1, 2, 3, 5), 10), x = rep(c(-1.5, -0.5, 0.5, 1.5), 10))
#' m <- bayes_glm_fit(claims ~ x, d, family = "poisson", chains = 2, tune = 300, draws = 300)
#' bayes_ppc(m, c("mean", "sd", 0.9))
#' summary(bayes_loo_pit(m)$pit)
bayes_ppc <- function(model, statistics = c("mean", "sd", "min", "max"), seed = 0) {
  checks_frame(rust_result(model@ptr$ppc(as.character(statistics), as.double(seed))))
}

#' @rdname bayes_ppc
#' @export
bayes_loo_pit <- function(model, seed = 0) {
  rust_result(model@ptr$loo_pit(as.double(seed)))
}

#' @rdname bayes_ppc
#' @export
bayes_replicate <- function(model, seed = 0) {
  r <- rust_result(model@ptr$replicate(as.double(seed)))
  # One dispersion per draw, so one row per draw.
  matrix(r, nrow = length(model@ptr$dispersion_draws()), byrow = TRUE)
}

#' @rdname bayes_ppc
#' @export
ppc_check <- function(observed, replicated, statistics = c("mean", "sd", "min", "max"),
                      groups = NULL) {
  replicated <- as.matrix(replicated)
  checks_frame(rust_result(ppc_check_rust(as.double(observed), as.double(t(replicated)),
                                          as.character(statistics), as.double(groups))))
}

#' @rdname bayes_ppc
#' @export
loo_pit <- function(observed, replicated, log_lik, r_eff = NULL, seed = 0) {
  rust_result(loo_pit_rust(as.double(observed), as.double(t(as.matrix(replicated))),
                           as.double(t(as.matrix(log_lik))), as.double(r_eff),
                           as.double(seed)))
}

checks_frame <- function(r) {
  out <- data.frame(name = r$name, observed = r$observed, p_value = r$p_value)
  attr(out, "replicated") <- matrix(r$replicated, ncol = length(r$name), byrow = TRUE,
                                    dimnames = list(NULL, r$name))
  out
}

for (cls in list(glm_model, gam_model, bayes_glm_model)) {
  S7::method(coef, cls) <- function(object, ...) object@coefficients
  S7::method(predict, cls) <- function(object, newdata, offset = NULL, ...) {
    nd <- new_design(object, newdata, offset)
    rust_result(object@ptr$predict(as.double(nd$x), colnames(nd$x), nd$offset), s7_call())
  }
}

S7::method(residuals, glm_model) <- function(object, type = c("deviance", "pearson", "response", "working", "quantile"),
                                              seed = 0, ...) {
  type <- match.arg(type)
  if (type == "quantile") {
    return(rust_result(object@ptr$quantile_residuals(as.double(seed)), s7_call()))
  }
  rust_result(object@ptr$diagnostics(), s7_call())[[type]]
}
S7::method(hatvalues, glm_model) <- function(model, ...) {
  rust_result(model@ptr$diagnostics(), s7_call())$leverage
}
S7::method(rstandard, glm_model) <- function(model, type = c("deviance", "pearson"), ...) {
  type <- match.arg(type)
  rust_result(model@ptr$diagnostics(), s7_call())[[paste0("standardized_", type)]]
}
S7::method(cooks.distance, glm_model) <- function(model, ...) {
  rust_result(model@ptr$diagnostics(), s7_call())$cooks_distance
}

#' Diagnostics and prediction errors of a fitted GLM
#'
#' A [glm_model] fitted in this session keeps its training data, so
#' [stats::residuals()] (`type` `"deviance"`, `"pearson"`, `"response"`,
#' `"working"` or `"quantile"`), [stats::hatvalues()], [stats::rstandard()] (`type`
#' `"deviance"` or `"pearson"`) and [stats::cooks.distance()] work on it as
#' on R's `glm`. `glm_diagnostics()` returns them all, with the working
#' weights, as a data frame. `prediction_se()` is the standard error of each
#' row's predicted mean (parameter error only, as
#' `predict(type = "response", se.fit = TRUE)`); `sum_prediction_error()` the
#' prediction error of the sum of the rows' means (a reserve: an origin's
#' future cells), the parameter error by the delta method and the process
#' error from the family (Taylor and McGuire, CAS Monograph 3, (5-11) to
#' (5-13)). `glm_gcv()` is generalized cross-validation,
#' `n sum((y - mu)^2) / (n - p)^2`, and `log_likelihood_at()` the training
#' log-likelihood at a given dispersion. A model read with [load_model()]
#' has no training data: only `prediction_se()` and `sum_prediction_error()`
#' work on it.
#'
#' @param model A [glm_model].
#' @param newdata A data frame with the model's terms.
#' @param offset Optional offset added to any `offset()` terms.
#' @param weights Optional prior weights (for the process variance).
#' @param dispersion The dispersion at which to evaluate.
#' @returns `glm_diagnostics()` a data frame with one row per training row;
#'   `prediction_se()` one value per row of `newdata`;
#'   `sum_prediction_error()` a list with `total`, `parameter_variance`,
#'   `process_variance` and `rmsep`; `glm_gcv()` and `log_likelihood_at()` a
#'   number.
#' @export
#' @examples
#' d <- data.frame(claims = c(1, 2, 4, 2, 1, 3, 0, 5), region = c("N", "N", "S", "S", "W", "W", "N", "S"))
#' m <- glm_fit(claims ~ region, d, family = "poisson")
#' head(glm_diagnostics(m))
#' sum_prediction_error(m, data.frame(region = c("N", "S")))$rmsep
glm_diagnostics <- function(model) {
  as.data.frame(rust_result(model@ptr$diagnostics()))
}

#' @rdname glm_diagnostics
#' @export
prediction_se <- function(model, newdata, offset = NULL) {
  nd <- new_design(model, newdata, offset)
  rust_result(model@ptr$prediction_se(as.double(nd$x), colnames(nd$x), nd$offset))
}

#' @rdname glm_diagnostics
#' @export
sum_prediction_error <- function(model, newdata, offset = NULL, weights = NULL) {
  nd <- new_design(model, newdata, offset)
  rust_result(model@ptr$sum_prediction_error(as.double(nd$x), colnames(nd$x), nd$offset,
                                             if (is.null(weights)) double() else as.double(weights)))
}

#' @rdname glm_diagnostics
#' @export
glm_gcv <- function(model) {
  rust_result(model@ptr$gcv())
}

#' @rdname glm_diagnostics
#' @export
log_likelihood_at <- function(model, dispersion) {
  rust_result(model@ptr$log_likelihood_at(as.double(dispersion)))
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
#' @param object A [glm_model], [elastic_net_model], [gam_model],
#'   [bayes_glm_model] (posterior predictive draws) or [odp_glm_fit] (its
#'   future cells, drawn jointly; no `newdata`).
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
#' when the model is calibrated; `quantile_residuals()` its normal scores
#' `qnorm(pit_values(...))`, Dunn and Smyth's randomized quantile residuals,
#' standard normal under the model (as `statmod::qresiduals`; a fitted
#' [glm_model]'s are `residuals(m, type = "quantile", seed = 0)`);
#' `ks_uniform()` the Kolmogorov-Smirnov
#' distance of values from the uniform, about `1.36 / sqrt(n)` or less 95%
#' of the time under uniformity. `pit_from_draws()` is the PIT of each `y`
#' under the empirical distribution of `draws`, ties randomized by `u`, and
#' `pit_histogram()` counts PIT values in `bins` equal-width bins of
#' `[0, 1]`. `deviance_score()` is a score for [cross_validate()]: the
#' family's mean deviance of the test data's `response` column,
#' `sum(w d) / sum(w)`, with the weights from its `weight_column` (all 1
#' by default).
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
#' @param u The share of tied draws counted below `y`.
#' @param pit PIT values in `[0, 1]`.
#' @param bins Number of histogram bins.
#' @param response The name of the response column.
#' @param weight_column The test data's weight column, or `NULL`.
#' @returns A number; for `pit_values()`, `quantile_residuals()` and `pit_from_draws()` one value
#'   per outcome; for `pit_histogram()` the count in each bin; for
#'   `deviance_score()` a function `score(model, test)`; for
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
#' pit_from_draws(c(1, 2, 2, 3), c(2, 5))
#' pit_histogram(c(0.05, 0.15, 0.95), bins = 10)
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

#' Charts for comparing and validating models
#'
#' The charts of CAS Monograph 5, chapter 7. `double_lift_table()` sorts rows
#' by the ratio of two models' predictions `pred_a / pred_b`, cuts them into
#' `bands` of about equal exposure and gives each band's actual and predicted
#' averages over their overall averages (each model against its own): where
#' the models disagree, the better one follows the actual. `loss_ratio_table()`
#' sorts by the predicted loss ratio `pred / premium` against current premium
#' and gives each band's actual loss ratio. `roc_curve()` is the ROC curve of a
#' binary model with its area (ties counting half); `confusion_matrix()` the
#' weighted counts at a threshold (a row is positive when its score is at
#' least the threshold). `normalized_gini()` is [gini_index()] over the Gini
#' of a perfect ranking; for a 0/1 response with unit exposure
#' `AUROC = (1 + normalized Gini) / 2`.
#'
#' @param y Actual losses, or 0 and 1 for `roc_curve()` and
#'   `confusion_matrix()`.
#' @param pred,pred_a,pred_b Predicted losses (`pred_b` positive).
#' @param premium Current premium, positive.
#' @param score A binary model's scores.
#' @param threshold The score at and above which a row is positive.
#' @param exposure,weights Optional exposure or weights per row.
#' @param bands Number of bands.
#' @returns `double_lift_table()` and `loss_ratio_table()` a data frame, one
#'   row per band; `roc_curve()` a list with `curve` (a data frame of
#'   `threshold`, `false_positive_rate`, `true_positive_rate`) and `auc`;
#'   `confusion_matrix()` a named vector; `normalized_gini()` a number.
#' @name model_charts
#' @examples
#' roc_curve(c(0, 0, 1, 1), c(0.1, 0.4, 0.35, 0.8))$auc
#' y <- c(1, 0, 3, 2, 5, 1)
#' double_lift_table(y, c(1, 1, 2, 2, 4, 1), c(2, 1, 2, 1, 3, 2), bands = 2)
#' loss_ratio_table(y, c(1, 1, 2, 2, 4, 1), rep(2, 6), bands = 2)
NULL

#' @rdname model_charts
#' @export
normalized_gini <- function(y, pred, exposure = NULL) {
  rust_result(normalized_gini_rust(as.double(y), as.double(pred), opt_double(exposure)))
}

#' @rdname model_charts
#' @export
double_lift_table <- function(y, pred_a, pred_b, exposure = NULL, bands = 10) {
  as.data.frame(rust_result(double_lift_rust(as.double(y), as.double(pred_a), as.double(pred_b),
                                             opt_double(exposure), as.double(bands))))
}

#' @rdname model_charts
#' @export
loss_ratio_table <- function(y, pred, premium, exposure = NULL, bands = 10) {
  as.data.frame(rust_result(loss_ratio_chart_rust(as.double(y), as.double(pred), as.double(premium),
                                                  opt_double(exposure), as.double(bands))))
}

#' @rdname model_charts
#' @export
roc_curve <- function(y, score, weights = NULL) {
  r <- rust_result(roc_rust(as.double(y), as.double(score), opt_double(weights)))
  list(curve = data.frame(threshold = r$thresholds, false_positive_rate = r$false_positive_rate,
                          true_positive_rate = r$true_positive_rate),
       auc = r$auc)
}

#' @rdname model_charts
#' @export
confusion_matrix <- function(y, score, threshold, weights = NULL) {
  unlist(rust_result(confusion_matrix_rust(as.double(y), as.double(score), as.double(threshold),
                                           opt_double(weights))))
}

opt_double <- function(x) if (is.null(x)) double() else as.double(x)

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
quantile_residuals <- function(family, y, mu, dispersion = 1, weights = NULL, seed = 0, theta = NULL,
                               power = NULL) {
  fa <- family_args(theta, power)
  rust_result(quantile_residuals_rust(family, fa$theta, fa$power, as.double(y), as.double(mu),
                                      as.double(dispersion),
                                      if (is.null(weights)) double() else as.double(weights),
                                      as.double(seed)))
}

#' @rdname model_metrics
#' @export
ks_uniform <- function(values) rust_result(ks_uniform_rust(as.double(values)))

#' @rdname model_metrics
#' @export
pit_from_draws <- function(draws, y, u = 0.5) {
  rust_result(pit_from_draws_rust(as.double(draws), as.double(y), as.double(u)))
}

#' @rdname model_metrics
#' @export
pit_histogram <- function(pit, bins = 10) {
  rust_result(pit_histogram_rust(as.double(pit), as.double(bins)))
}

#' @rdname model_metrics
#' @export
deviance_score <- function(family, response, theta = NULL, power = NULL, weight_column = NULL) {
  force(family)
  force(response)
  function(model, test) {
    y <- test[[response]]
    w <- if (is.null(weight_column)) rep(1, nrow(test)) else test[[weight_column]]
    family_deviance(family, y, stats::predict(model, test), weights = w, theta = theta,
                    power = power) / sum(w)
  }
}

#' Resampling splits
#'
#' `k_fold()` shuffles `n` rows with `seed` and cuts them into `k` folds;
#' `group_k_fold()` keeps each group's rows in one fold; `time_ordered()`
#' tests on each of the last `n_test` periods after training on the earlier
#' ones (with calendar diagonals as periods, the triangle backtest).
#' `time_splits()` generalises it: each split tests on `horizon` consecutive
#' periods, the last ending at the latest period and each earlier one `step`
#' periods before the next, and trains on the periods before its test window
#' less the `gap` just before it, all of them or the latest `window`
#' (scikit-learn's `TimeSeriesSplit` with `horizon = step = test_size`).
#' `permutation()` shuffles `1:n` with `seed` on stream `stream`.
#'
#' @param n Number of rows.
#' @param k Number of folds.
#' @param seed Generator seed.
#' @param groups Group label per row.
#' @param periods Period per row (whole numbers).
#' @param n_test Number of final periods to test on.
#' @param stream For `permutation()`, the stream of `seed` to draw from.
#' @returns A list of splits, each `list(train, test)` of row numbers;
#'   `permutation()`, the row numbers `1:n` shuffled, the shuffle the
#'   [explanations] apply (the same as Python's `permutation`, plus one).
#' @name resampling
#' @examples
#' length(k_fold(10, 5, seed = 1))
#' time_ordered(c(0, 1, 2, 1, 2, 2), 1)
#' time_splits(0:9, 2, window = 3, horizon = 2, gap = 1, step = 2)
NULL

#' Out-of-fold target encoding with credibility
#'
#' Encodes a high-cardinality factor by each level's actual-to-expected
#' ratio, shrunk to 1 by the credibility `z = n / (n + k)` with `n` the
#' level's exposure: `z A / E + (1 - z)` (CAS Monograph 16, chapter 6). Each
#' training row is encoded from the other folds only, so its own response
#' never enters its encoding; `encode_target()` applies the encoding from all
#' the rows to new data (a level not seen in training gets 1). Its log
#' enters a GLM as a numeric column.
#'
#' @param values The factor's level per row.
#' @param y Actual responses.
#' @param expected Expected responses: the overall mean times exposure, or a
#'   first model's predictions.
#' @param splits Splits from [k_fold()] or [group_k_fold()], every row in
#'   exactly one test set.
#' @param k The credibility constant.
#' @param exposure Optional credibility weights, 1 per row by default.
#' @param encoding A `target_encoding` list from `target_encode()`.
#' @param newvalues Levels of new data.
#' @returns `target_encode()` a list with `rows` (the out-of-fold encoding of
#'   each training row) and `encoding` (a data frame of `level` and `ratio`,
#'   with `k`); `encode_target()` one value per new value.
#' @export
#' @examples
#' splits <- list(list(train = c(2, 4), test = c(1, 3)), list(train = c(1, 3), test = c(2, 4)))
#' te <- target_encode(c("a", "a", "b", "b"), c(2, 2, 0, 1), rep(1, 4), splits, k = 1)
#' te$rows
#' encode_target(te$encoding, c("a", "z"))
target_encode <- function(values, y, expected, splits, k, exposure = NULL) {
  r <- rust_result(target_encode_rust(
    as.character(values), as.double(y), as.double(expected),
    if (is.null(exposure)) double() else as.double(exposure),
    lapply(splits, function(s) as.double(s$train)), lapply(splits, function(s) as.double(s$test)),
    as.double(k)
  ))
  encoding <- data.frame(level = r$levels, ratio = r$ratios)
  attr(encoding, "k") <- k
  list(rows = r$rows, encoding = encoding)
}

#' @rdname target_encode
#' @export
encode_target <- function(encoding, newvalues) {
  out <- encoding$ratio[match(as.character(newvalues), encoding$level)]
  out[is.na(out)] <- 1
  out
}

#' Knots for hinge terms by MARS
#'
#' A forward stepwise search over hinge pairs `max(0, x - k)` and
#' `max(0, k - x)` in the manner of R earth with `degree = 1`: each step adds
#' the pair on the variable and knot that most reduce the residual sum of
#' squares (weighted when `weights` are given), with earth's minimum span
#' between knots and end span, its stopping rules and its GCV with penalty
#' `penalty`. With `prune = TRUE` a backward pass drops terms while that
#' lowers the GCV. With `family`, each step scores candidates by weighted
#' least squares on the GLM's working response and the deviance takes the
#' residual sum of squares' place. Default runs on
#' `examples/data/chalk_portfolio.csv` match earth's knots, RSS and GCV
#' (`validation/scripts/r_mars_earth.R`). The knots enter [glm_fit()] as
#' formula terms; `mars_formula()` writes them.
#'
#' @param x A data frame or named list of numeric predictors.
#' @param y The response.
#' @param weights Optional prior weights.
#' @param max_terms The most terms, the intercept included (earth's `nk`);
#'   `NULL` for earth's default `min(200, max(20, 2p)) + 1`.
#' @param threshold Stop when a step raises the R-squared by less (earth's
#'   `thresh`).
#' @param penalty The GCV's cost per knot.
#' @param min_span,end_span Observations between knots and at the ends;
#'   `NULL` for earth's defaults from the number of predictors and rows.
#' @param prune Run the backward pass by GCV.
#' @param family,link A GLM family and link as in [glm_fit()] for the
#'   deviance criterion; `NULL` for least squares.
#' @param fit A `mars_knots` list.
#' @returns `mars_knots()` a list with `terms` (a data frame of `variable`,
#'   `knot` and `direction`, one of `"up"`, `"down"` or `"linear"`),
#'   `forward_terms` (the same before pruning), `knots` and `linear` (per
#'   variable), `coefficients` (the intercept first), `rss`, `gcv`, `rsq`,
#'   `grsq`, `termination` and `steps` (a data frame of `terms`, `rss`,
#'   `rsq` and `gcv` per forward step); `mars_formula()` the right-hand
#'   side terms as a character vector, `x` for a linear term and
#'   `I(pmax(0, x - k))` for each knot.
#' @export
#' @examples
#' x <- data.frame(a = 1:40)
#' y <- pmax(0, x$a - 20) + sin(x$a)
#' fit <- mars_knots(x, y, min_span = 1, end_span = 1)
#' fit$knots$a
#' mars_formula(fit)
mars_knots <- function(x, y, weights = NULL, max_terms = NULL, threshold = 0.001, penalty = 2,
                       min_span = NULL, end_span = NULL, prune = FALSE, family = NULL, link = NULL) {
  count <- function(v) if (is.null(v)) 0 else as.double(v)
  r <- rust_result(mars_rust(
    names(x), lapply(x, as.double), as.double(y),
    if (is.null(weights)) double() else as.double(weights),
    count(max_terms), as.double(threshold), as.double(penalty), count(min_span), count(end_span),
    isTRUE(prune), if (is.null(family)) "" else family, if (is.null(link)) "" else link
  ))
  table <- function(t) data.frame(variable = t$variable, knot = t$knot, direction = t$direction)
  knots <- r$knots
  names(knots) <- names(x)
  linear <- r$linear
  names(linear) <- names(x)
  structure(list(
    terms = table(r$terms), forward_terms = table(r$forward_terms), knots = knots,
    linear = linear, coefficients = r$coefficients, rss = r$rss, gcv = r$gcv, rsq = r$rsq,
    grsq = r$grsq, termination = r$termination,
    steps = data.frame(terms = r$step_terms, rss = r$step_rss, rsq = r$step_rsq, gcv = r$step_gcv)
  ), class = "mars_knots")
}

#' @rdname mars_knots
#' @export
mars_formula <- function(fit) {
  unlist(lapply(names(fit$knots), function(v) {
    c(if (fit$linear[[v]]) v, vapply(fit$knots[[v]], function(k) sprintf("I(pmax(0, %s - %.17g))", v, k), ""))
  }))
}

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

#' @rdname resampling
#' @param window A rolling window of that many periods, or `NULL` for an
#'   expanding one.
#' @param horizon,gap,step Periods tested, left out before each test window,
#'   and moved from one split to the next.
#' @export
time_splits <- function(periods, n_test, window = NULL, horizon = 1, gap = 0, step = 1) {
  rust_result(time_splits_rust(as.double(periods), as.double(n_test),
                               if (is.null(window)) NaN else as.double(window),
                               as.double(horizon), as.double(gap), as.double(step)))
}

#' @rdname resampling
#' @export
permutation <- function(n, seed, stream = 0) {
  rust_result(permutation_rust(as.double(n), as.double(seed), as.double(stream)))
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

#' Model-agnostic explanations: permutation importance, interactions and
#' partial dependence
#'
#' Three tools that need only `predict()`, so they explain any model (CAS
#' Monograph 16, chapter 9). `permutation_importance()` measures how much a
#' score worsens when a feature's column is shuffled, which breaks its link
#' to the response and leaves every other column in place; several columns
#' named together are shuffled with one permutation (a group).
#' `interaction_screen()` shuffles two features separately and together: with
#' `L_A`, `L_B` and `L_AB` the losses, `L_A + L_B - L_AB` is about zero if
#' neither interacts with anything and positive if they interact with each
#' other or with a common third feature. A deviance is not additive on the
#' link scale, so the pairs of an additive log-link model need not be zero:
#' compare pairs with one another, on a flexible model such as a boosting
#' model. `partial_dependence()` sets a feature to each value on every row
#' and averages the predictions; for a log-link model without interactions
#' their ratios are exactly the feature's relativities.
#'
#' Shuffles come from [permutation()]: repeat `r` of split `i` (both from 0)
#' uses `permutation(n, seed, i * n_repeats + r)`, so Python's
#' `prospicio.models` functions of the same names give the same values on the
#' same rows.
#'
#' @param model A fitted model; with `splits`, a `fit(train)` function, as for
#'   [cross_validate()], fitted on each split's training rows and explained on
#'   its test rows.
#' @param data A data frame: the rows to explain (all the data with `splits`).
#' @param score `score(model, test)`: a loss (lower is better), such as
#'   [deviance_score()]; a gain such as a pseudo-R2 goes in as its negative.
#' @param features Column names, or a list whose elements are names shuffled
#'   together (labelled by the list's names, or the names joined by `+`).
#'   Never the response, exposure or weights.
#' @param pairs For `interaction_screen()`, a list of two-label vectors;
#'   every pair of `features` by default.
#' @param splits Optional splits from [k_fold()], [group_k_fold()] or
#'   [time_ordered()]; the results average over them.
#' @param n_repeats Shuffles per split, averaged.
#' @param seed Seed of the shuffles.
#' @param feature For `partial_dependence()`, one column of `data`.
#' @param values The values to set it to.
#' @returns `permutation_importance()`: a data frame with `feature`,
#'   `importance` (the mean increase in the score over the splits),
#'   `std_error` (over the splits, `NA` for one) and `relative` (percent of
#'   the largest), with attributes `by_split` (a splits by features matrix)
#'   and `base` (the unshuffled score per split). `interaction_screen()`: a
#'   data frame with `a`, `b`, `strength`, `std_error` and the mean losses
#'   `loss_a`, `loss_b`, `loss_ab`, with attribute `by_split`.
#'   `partial_dependence()`: a data frame with `value` and `mean` (the mean
#'   prediction), with attribute `by_split`.
#' @name explanations
#' @examples
#' d <- data.frame(x = 1:40 / 10, noise = ((7 * 1:40) %% 11) / 11)
#' d$y <- 1 + 2 * d$x + sin(1:40)
#' fit <- glm_fit(y ~ x, d, family = "gaussian")
#' permutation_importance(fit, d, deviance_score("gaussian", "y"), c("x", "noise"))
#' partial_dependence(fit, d, "x", c(1, 2))
NULL

explain_groups <- function(features, data) {
  groups <- as.list(features)
  if (!length(groups)) stop("features must not be empty")
  labels <- names(groups)
  if (is.null(labels)) labels <- rep("", length(groups))
  for (j in seq_along(groups)) {
    names_j <- as.character(groups[[j]])
    missing <- setdiff(names_j, names(data))
    if (length(missing)) stop("no column ", missing[1], " in the data")
    if (labels[j] == "") labels[j] <- paste(names_j, collapse = "+")
    groups[[j]] <- names_j
  }
  if (anyDuplicated(labels)) stop("feature labels must be distinct")
  names(groups) <- labels
  groups
}

# The score on `test` and, for each group of columns, the mean increase over
# the repeats; every group of a repeat is shuffled with the same permutation.
explain_losses <- function(model, test, score, groups, n_repeats, seed, split) {
  if (n_repeats < 1) stop("n_repeats must be at least 1")
  n <- nrow(test)
  base <- as.double(score(model, test))
  total <- double(length(groups))
  for (r in seq_len(n_repeats) - 1) {
    perm <- permutation(n, seed, split * n_repeats + r)
    for (g in seq_along(groups)) {
      shuffled <- test
      shuffled[groups[[g]]] <- test[perm, groups[[g]], drop = FALSE]
      total[g] <- total[g] + as.double(score(model, shuffled)) - base
    }
  }
  list(base = base, loss = total / n_repeats)
}

explain_splits <- function(model, data, splits, one) {
  if (is.null(splits)) return(list(one(model, data, 0)))
  if (!length(splits)) stop("splits must not be empty")
  if (!is.function(model)) stop("with splits, model must be a fit(train) function")
  lapply(seq_along(splits), function(i) {
    s <- splits[[i]]
    one(model(data[s$train, , drop = FALSE]), data[s$test, , drop = FALSE], i - 1)
  })
}

explain_se <- function(v) if (length(v) < 2) NA_real_ else stats::sd(v) / sqrt(length(v))

#' @rdname explanations
#' @export
permutation_importance <- function(model, data, score, features, splits = NULL, n_repeats = 1,
                                   seed = 0) {
  groups <- explain_groups(features, data)
  out <- explain_splits(model, data, splits, function(m, test, i) {
    explain_losses(m, test, score, groups, n_repeats, seed, i)
  })
  by_split <- do.call(rbind, lapply(out, function(o) o$loss))
  colnames(by_split) <- names(groups)
  importance <- colMeans(by_split)
  top <- max(importance)
  res <- data.frame(
    feature = names(groups), importance = unname(importance),
    std_error = unname(apply(by_split, 2, explain_se)),
    relative = if (top > 0) unname(100 * importance / top) else NA_real_
  )
  attr(res, "by_split") <- by_split
  attr(res, "base") <- vapply(out, function(o) o$base, double(1))
  res
}

#' @rdname explanations
#' @export
interaction_screen <- function(model, data, score, features, pairs = NULL, splits = NULL,
                               n_repeats = 1, seed = 0) {
  groups <- explain_groups(features, data)
  labels <- names(groups)
  if (is.null(pairs)) {
    if (length(labels) < 2) stop("interaction_screen needs at least two features")
    pairs <- list()
    for (j in seq_len(length(labels) - 1)) {
      for (k in (j + 1):length(labels)) pairs[[length(pairs) + 1]] <- labels[c(j, k)]
    }
  }
  pairs <- lapply(pairs, as.character)
  for (p in pairs) {
    if (length(p) != 2 || p[1] == p[2] || !all(p %in% labels)) {
      stop("(", paste(p, collapse = ", "), ") is not a pair of two different feature labels")
    }
  }
  if (!length(pairs)) stop("interaction_screen needs at least one pair")
  shuffles <- c(groups, lapply(pairs, function(p) c(groups[[p[1]]], groups[[p[2]]])))
  out <- explain_splits(model, data, splits, function(m, test, i) {
    explain_losses(m, test, score, shuffles, n_repeats, seed, i)$loss
  })
  k <- length(labels)
  single <- do.call(rbind, lapply(out, function(o) o[seq_len(k)]))
  joint <- do.call(rbind, lapply(out, function(o) o[-seq_len(k)]))
  colnames(single) <- labels
  a <- vapply(pairs, `[`, "", 1)
  b <- vapply(pairs, `[`, "", 2)
  by_split <- single[, a, drop = FALSE] + single[, b, drop = FALSE] - joint
  colnames(by_split) <- paste(a, b, sep = ":")
  res <- data.frame(
    a = a, b = b, strength = unname(colMeans(by_split)),
    std_error = unname(apply(by_split, 2, explain_se)),
    loss_a = unname(colMeans(single)[a]), loss_b = unname(colMeans(single)[b]),
    loss_ab = unname(colMeans(joint))
  )
  attr(res, "by_split") <- by_split
  res
}

#' @rdname explanations
#' @export
partial_dependence <- function(model, data, feature, values, splits = NULL) {
  if (length(feature) != 1 || !feature %in% names(data)) stop("no column ", feature, " in the data")
  if (!length(values)) stop("values must not be empty")
  out <- explain_splits(model, data, splits, function(m, test, i) {
    col <- test[[feature]]
    vapply(seq_along(values), function(v) {
      set <- test
      set[[feature]] <- if (is.factor(col)) {
        factor(rep(as.character(values[[v]]), nrow(test)), levels = levels(col))
      } else {
        rep(values[[v]], nrow(test))
      }
      mean(stats::predict(m, set))
    }, double(1))
  })
  by_split <- do.call(rbind, out)
  res <- data.frame(value = values, mean = colMeans(by_split))
  attr(res, "by_split") <- by_split
  res
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
#' @param gamma With `relax = TRUE`, the relaxations to score, as
#'   `cv.glmnet(relax = TRUE)`'s `gamma`.
#' @returns A list with `lambda`, `mean`, `se`, `lambda_min` (lowest mean)
#'   and `lambda_1se` (the largest `lambda` within one standard error of
#'   it), and `fit`, the [elastic_net_model] on all rows over `lambda`.
#'   With `relax = TRUE`, `mean` and `se` are matrices (a row per `lambda`,
#'   a column per `gamma`), and `gamma_min` and `gamma_1se` go with
#'   `lambda_min` and `lambda_1se`: the lowest mean over every pair, and
#'   the largest `lambda`, then `gamma`, within one standard error of it,
#'   as `cv.glmnet()` chooses. `fit` is then relaxed.
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
                           theta = NULL, power = NULL, link_power = NULL,
                           lower_limits = NULL, upper_limits = NULL, relax = FALSE,
                           gamma = c(0, 0.25, 0.5, 0.75, 1)) {
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
    as.double(lambda_min_ratio), isTRUE(standardize), pf,
    coef_limits(lower_limits, des$x, -Inf, "lower_limits"),
    coef_limits(upper_limits, des$x, Inf, "upper_limits"),
    if (isTRUE(relax)) as.double(gamma) else double(),
    fa$theta, fa$power, if (is.null(link_power)) NaN else as.double(link_power),
    as.double(foldid)
  ))
  if (isTRUE(relax)) {
    cv$mean <- matrix(cv$mean, length(cv$lambda))
    cv$se <- matrix(cv$se, length(cv$lambda))
  }
  cv$fit <- elastic_net_fit(formula, data, family = family, link = link, alpha = alpha,
                            lambda = cv$lambda, standardize = standardize,
                            penalty_factor = penalty_factor, offset = offset, weights = weights,
                            theta = theta, power = power, link_power = link_power,
                            lower_limits = lower_limits, upper_limits = upper_limits,
                            relax = relax)
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

#' Draws from another sampler
#'
#' Brings posterior draws from Stan, PyMC, nutpie or any other sampler into
#' prospicio's diagnostics (#395). `read_stan_csv()` reads CmdStan's CSV
#' output, one file per chain: comments skipped, saved warmup draws dropped,
#' `theta.1.2` named `theta[1,2]`, `lp__` kept, the sampler's other `__`
#' columns and each chain's divergent transitions kept as attributes.
#' `mcmc_draws()` takes draws already in R. `draws_summary()` and
#' `draws_loo()` then give the summary, R-hat and ESS as
#' `posterior::summarise_draws()` and PSIS-LOO as
#' `loo::loo(log_lik, r_eff = loo::relative_eff(exp(log_lik)))`.
#'
#' @param files CmdStan CSV files, one per chain.
#' @param x For `mcmc_draws()`: a 3-D array iteration x chain x variable
#'   with the variable names as its third dimnames (the layout of
#'   `posterior::draws_array`), any `posterior` draws object (converted with
#'   `posterior::as_draws_array()`), an rstan `stanfit` (`as.array()`) or a
#'   cmdstanr fit (`$draws()`, with its divergences). For `draws_summary()`
#'   and `draws_loo()`: draws from `read_stan_csv()` or `mcmc_draws()`, or
#'   anything `mcmc_draws()` takes.
#' @param divergences Divergent transitions in each chain, if known.
#' @param log_lik The name of the pointwise log-likelihood variable, held as
#'   `log_lik[1]`, `log_lik[2]`, ...
#' @returns `read_stan_csv()` and `mcmc_draws()`: a 3-D array iteration x
#'   chain x variable, with attributes `divergences` (per chain) and, from
#'   `read_stan_csv()`, `sampler` (the sampler's columns as another such
#'   array) and `warmup_dropped`. `draws_summary()`: a data frame with
#'   `variable`, `mean`, `sd`, `q05`, `q50`, `q95`, `rhat`, `ess_bulk` and
#'   `ess_tail`. `draws_loo()`: a list as [elpd_loo()]'s, with `r_eff` per
#'   observation.
#' @name external_draws
#' @examples
#' set.seed(1)
#' x <- array(rnorm(4 * 200 * 2), c(200, 4, 2),
#'            dimnames = list(NULL, NULL, c("mu", "sigma")))
#' draws_summary(mcmc_draws(x))
NULL

#' @rdname external_draws
#' @export
read_stan_csv <- function(files) {
  r <- rust_result(read_stan_csv_rust(as.character(files)))
  out <- array(r$values, c(r$n_iter, r$n_chain, length(r$names)),
               dimnames = list(NULL, NULL, r$names))
  attr(out, "divergences") <- r$divergences
  if (length(r$sampler_names) > 0) {
    attr(out, "sampler") <- array(r$sampler_values, c(r$n_iter, r$n_chain, length(r$sampler_names)),
                                  dimnames = list(NULL, NULL, r$sampler_names))
  }
  attr(out, "warmup_dropped") <- r$warmup_dropped
  out
}

#' @rdname external_draws
#' @export
mcmc_draws <- function(x, divergences = NULL) {
  if (inherits(x, "CmdStanMCMC")) {
    if (is.null(divergences)) divergences <- x$diagnostic_summary(quiet = TRUE)$num_divergent
    x <- x$draws()
  } else if (inherits(x, "stanfit")) {
    x <- as.array(x)
  }
  if (inherits(x, "draws")) {
    if (!requireNamespace("posterior", quietly = TRUE)) {
      stop("converting posterior draws needs the posterior package")
    }
    x <- posterior::as_draws_array(x)
  }
  if (is.null(divergences)) divergences <- attr(x, "divergences")
  x <- unclass(x)
  if (!is.numeric(x) || length(dim(x)) != 3 || is.null(dimnames(x)[[3]])) {
    stop("draws must be a 3-D array iteration x chain x variable with variable names")
  }
  out <- array(as.double(x), dim(x), dimnames = list(NULL, NULL, dimnames(x)[[3]]))
  attr(out, "divergences") <- if (is.null(divergences)) NULL else as.double(divergences)
  for (a in c("sampler", "warmup_dropped")) attr(out, a) <- attr(x, a)
  out
}

#' @rdname external_draws
#' @export
draws_summary <- function(x) {
  x <- mcmc_draws(x)
  d <- dim(x)
  r <- rust_result(draws_summary_rust(as.double(x), d[1], d[2], dimnames(x)[[3]]))
  as.data.frame(r)
}

#' @rdname external_draws
#' @export
draws_loo <- function(x, log_lik = "log_lik") {
  x <- mcmc_draws(x)
  d <- dim(x)
  elpd_result(rust_result(draws_loo_rust(as.double(x), d[1], d[2], dimnames(x)[[3]], log_lik)))
}

# The random terms `(1 + x | g)` of a formula, in lme4's syntax: the
# formula without them and each grouping factor as a list of its
# expression (`g`, `a:b`; `a/b` is `a` and `a:b`), whether it has an
# intercept and its slopes' expressions. `(1 + x || g)` is a factor for
# each term.
glmm_bars <- function(formula) {
  bars <- list()
  strip <- function(e) {
    if (is.call(e) && identical(e[[1]], as.name("(")) && is.call(e[[2]]) &&
        (identical(e[[2]][[1]], as.name("|")) || identical(e[[2]][[1]], as.name("||")))) {
      lhs <- stats::terms(stats::as.formula(call("~", e[[2]][[2]])))
      intercept <- attr(lhs, "intercept") == 1
      slopes <- attr(lhs, "term.labels")
      if (!intercept && !length(slopes)) {
        stop("a random term needs an intercept or a slope: ", deparse(e), call. = FALSE)
      }
      if (identical(e[[2]][[1]], as.name("||"))) {
        if (intercept) {
          bars[[length(bars) + 1]] <<- list(group = e[[2]][[3]], intercept = TRUE,
                                            slopes = character())
        }
        for (x in slopes) {
          bars[[length(bars) + 1]] <<- list(group = e[[2]][[3]], intercept = FALSE, slopes = x)
        }
      } else {
        bars[[length(bars) + 1]] <<- list(group = e[[2]][[3]], intercept = intercept,
                                          slopes = slopes)
      }
      return(NULL)
    }
    if (is.call(e) && identical(e[[1]], as.name("+"))) {
      parts <- Filter(Negate(is.null), lapply(as.list(e)[-1], strip))
      if (!length(parts)) return(NULL)
      return(Reduce(function(a, b) call("+", a, b), parts))
    }
    e
  }
  rhs <- strip(formula[[3]])
  fixed <- formula
  fixed[[3]] <- if (is.null(rhs)) 1 else rhs
  nest <- function(g) {
    if (!(is.call(g) && identical(g[[1]], as.name("/")))) return(list(g))
    outer <- nest(g[[2]])
    c(outer, list(call(":", outer[[length(outer)]], g[[3]])))
  }
  groups <- unlist(lapply(bars, function(b) {
    lapply(nest(b$group), function(g) list(group = g, intercept = b$intercept, slopes = b$slopes))
  }), recursive = FALSE)
  if (!length(groups)) stop("a GLMM needs at least one random term, (1 | g)", call. = FALSE)
  list(fixed = fixed, groups = groups)
}

# Each grouping factor's name, its labels, intercept and slopes on `data`,
# those whose columns `data` has (all of them, when `required`), as
# glmm_fit_design() takes them.
glmm_labels <- function(groups, data, required = TRUE) {
  out <- list(names = character(), labels = character(), intercepts = numeric(),
              slope_factor = numeric(), slope_names = character(), slope_values = numeric())
  for (g in groups) {
    vars <- unique(c(all.vars(g$group), unlist(lapply(g$slopes, function(x) {
      all.vars(str2lang(x))
    }))))
    if (!all(vars %in% names(data))) {
      if (required) {
        stop("no column ", toString(setdiff(vars, names(data))), " for ", deparse(g$group),
             call. = FALSE)
      }
      next
    }
    for (x in g$slopes) {
      v <- eval(str2lang(x), data, baseenv())
      if (!is.numeric(v)) stop("a random slope must be numeric: ", x, call. = FALSE)
      out$slope_factor <- c(out$slope_factor, length(out$names))
      out$slope_names <- c(out$slope_names, x)
      out$slope_values <- c(out$slope_values, as.double(v))
    }
    out$names <- c(out$names, deparse(g$group))
    out$intercepts <- c(out$intercepts, as.double(g$intercept))
    out$labels <- c(out$labels, do.call(paste, c(lapply(all.vars(g$group), function(v) {
      as.character(data[[v]])
    }), sep = ":")))
  }
  out
}

# A random term's name: its factor for an intercept, `g|x` for a slope.
glmm_term_names <- function(groups, terms) {
  ifelse(terms == "(Intercept)", groups, paste0(groups, "|", terms))
}

#' Generalized linear mixed models with random intercepts and slopes
#'
#' `glmm_fit()` fits a GLM with random effects,
#' `log(mu) = X beta + offset + sum(z' b)`, each grouping factor's effects
#' at a level `N(0, Sigma)` with `Sigma` unstructured (a standard
#' deviation for each term and their correlations), for the Poisson, gamma
#' or Tweedie (fixed power in (1, 2)) with the log link, by maximum
#' marginal likelihood as glmmTMB and `lme4::glmer()`: the Laplace
#' approximation by default, or adaptive Gauss-Hermite quadrature with
#' `nodes` nodes for a single factor with a single term
#' (`glmer(nAGQ = nodes)`). The dispersion is estimated by maximum
#' likelihood, as glmmTMB does. Random terms are written as in lme4:
#' `(1 | g)`, crossed factors `(1 | a) + (1 | b)`, nested ones `(1 | a/b)`
#' (that is, `(1 | a) + (1 | a:b)`), a random slope on a numeric
#' covariate correlated with the intercept `(1 + x | g)` (or `(x | g)`),
#' without the intercept `(0 + x | g)`, and independent of it
#' `(1 + x || g)`. The results match glmmTMB and `glmer`
#' (`validation/scripts/r_glmm.R`; `docs/design/models.md`, "Mixed
#' models").
#'
#' @param formula A formula with the fixed effects and one or more
#'   random terms, `(1 | g)` or `(1 + x | g)`.
#' @param data A data frame.
#' @param family `"poisson"`, `"gamma"` or `"tweedie"`.
#' @param power The Tweedie power, in (1, 2).
#' @param nodes Quadrature nodes for a single grouping factor; `NULL` for
#'   the Laplace approximation.
#' @param offset,weights Optional offset (added to any `offset()` term)
#'   and prior weights.
#' @returns A `glmm_model` with `coefficients`, `std_errors`, `covariance`
#'   (of the coefficients), `sd` and `sd_std_errors` (each random term's
#'   standard deviation, named `g` for a factor's intercept and `g|x` for
#'   its slope on `x`; the error is NaN for one at 0), `correlations` (each
#'   factor's correlation matrix of its terms), `dispersion` and
#'   `dispersion_std_error` (`phi`: the variance is `phi mu^p / w`; 1 for
#'   the Poisson), `log_likelihood`, `aic`, `bic`, `fitted`, `iterations`,
#'   `converged`, and `random_effects`: a list by factor of data frames
#'   with a row for each `level` and `term`, its conditional mode `value`,
#'   its `conditional_variance` (lme4's `condVar`) and its
#'   `prediction_variance` (with the estimates' uncertainty, glmmTMB's
#'   `condVar`). `predict()` adds each row's effects by label and
#'   covariate; a level the fit has not seen, or a factor whose columns
#'   (its labels' or its slopes') `newdata` lacks, adds 0, the population
#'   mean.
#' @export
#' @examples
#' d <- data.frame(y = (0:39 * 7) %% 5 + 0:39 %% 4, site = paste0("s", 0:39 %% 4))
#' m <- glmm_fit(y ~ 1 + (1 | site), d)
#' m@sd
#' predict(m, data.frame(site = c("s0", "new")))
glmm_fit <- function(formula, data, family = "poisson", power = NULL, nodes = NULL,
                     offset = NULL, weights = NULL) {
  bars <- glmm_bars(formula)
  des <- model_design(bars$fixed, data, offset, weights)
  g <- glmm_labels(bars$groups, data)
  ptr <- rust_result(glmm_fit_design(as.double(des$x), colnames(des$x), des$y, des$offset,
                                     des$weights, family,
                                     if (is.null(power)) NaN else as.double(power),
                                     g$names, g$labels, g$intercepts, g$slope_factor,
                                     g$slope_names, g$slope_values,
                                     if (is.null(nodes)) 0 else as.double(nodes)))
  glmm_model(ptr = ptr, terms = des$terms, xlevels = des$xlevels, groups = bars$groups)
}

#' @rdname glmm_fit
#' @export
glmm_model <- S7::new_class(
  "glmm_model",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("GlmmModel"),
    terms = S7::class_any,
    xlevels = S7::class_any,
    groups = S7::class_list,
    coefficients = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(self@ptr$coefficients(), self@ptr$names())
    }),
    std_errors = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(self@ptr$std_errors(), self@ptr$names())
    }),
    covariance = S7::new_property(S7::class_any, getter = function(self) {
      nm <- self@ptr$names()
      matrix(self@ptr$covariance(), length(nm), dimnames = list(nm, nm))
    }),
    sd = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(self@ptr$sd(), glmm_term_names(self@ptr$group_names(), self@ptr$term_names()))
    }),
    sd_std_errors = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(self@ptr$sd_std_errors(),
                      glmm_term_names(self@ptr$group_names(), self@ptr$term_names()))
    }),
    correlations = S7::new_property(S7::class_list, getter = function(self) {
      terms <- self@ptr$correlation_terms()
      out <- Map(function(r, t) matrix(r, length(t), dimnames = list(t, t)),
                 self@ptr$correlations(), terms)
      stats::setNames(out, names(self@ptr$correlations()))
    }),
    dispersion = S7::new_property(S7::class_double, getter = function(self) self@ptr$dispersion()),
    dispersion_std_error = S7::new_property(S7::class_double, getter = function(self) {
      self@ptr$dispersion_std_error()
    }),
    log_likelihood = S7::new_property(S7::class_double, getter = function(self) self@ptr$log_likelihood()),
    aic = S7::new_property(S7::class_double, getter = function(self) self@ptr$aic()),
    bic = S7::new_property(S7::class_double, getter = function(self) self@ptr$bic()),
    fitted = S7::new_property(S7::class_double, getter = function(self) self@ptr$fitted()),
    iterations = S7::new_property(S7::class_double, getter = function(self) self@ptr$iterations()),
    converged = S7::new_property(S7::class_logical, getter = function(self) self@ptr$converged()),
    random_effects = S7::new_property(S7::class_list, getter = function(self) {
      lapply(self@ptr$random_effects(), as.data.frame)
    })
  ),
  constructor = function(ptr, terms, xlevels, groups) {
    S7::new_object(S7::S7_object(), ptr = ptr, terms = terms, xlevels = xlevels, groups = groups)
  }
)

S7::method(coef, glmm_model) <- function(object, ...) object@coefficients
S7::method(predict, glmm_model) <- function(object, newdata, offset = NULL, ...) {
  nd <- new_design(object, newdata, offset)
  g <- glmm_labels(object@groups, newdata, required = FALSE)
  rust_result(object@ptr$predict(as.double(nd$x), colnames(nd$x), nd$offset, g$names, g$labels,
                                 g$intercepts, g$slope_factor, g$slope_names, g$slope_values),
              s7_call())
}

#' Bayesian GLMMs by NUTS
#'
#' `bayes_glmm_fit()` samples the posterior of the model of [glmm_fit()]
#' (random terms written as in lme4) with nuts-rs, jointly with the random
#' effects, which are sampled non-centred (`b = Lambda v` with
#' `v ~ N(0, I)`, so the chains stay out of the funnel between a standard
#' deviation and its effects). The priors: `N(0, prior_sd^2)` on each
#' coefficient (`intercept_sd` for the intercept), half-normal with scale
#' `sd_scale` on each random term's standard deviation, LKJ(`lkj`) on each
#' factor's correlations (uniform at 1), and half-normal with scale
#' `dispersion_scale` on the gamma's or Tweedie's dispersion. The chains
#' start at the Laplace fit, which is kept as `laplace`. On a single
#' factor the posterior matches exact integration, and on 800 rows with
#' random slopes it sits on the Laplace fit (`docs/design/models.md`,
#' "Mixed models").
#'
#' @inheritParams glmm_fit
#' @param prior_sd,intercept_sd Prior standard deviations of the
#'   coefficients and the intercept, on the log scale.
#' @param sd_scale Scale of the half-normal prior on each random term's
#'   standard deviation.
#' @param lkj The LKJ prior's shape on each factor's correlations.
#' @param dispersion_scale Scale of the half-normal prior on the
#'   dispersion.
#' @param chains,tune,draws Chains, warm-up draws and kept draws per chain.
#' @param seed Seed, a whole number.
#' @param target_accept Target acceptance rate for step-size adaptation.
#' @param max_depth Largest tree depth.
#' @returns A `bayes_glmm_model` with `coefficients` (posterior means),
#'   `summary` (a data frame of mean, sd, quantiles, R-hat and ESS for each
#'   coefficient, `sd[g|term]`, `cor[g|a,b]` and the `dispersion` when it
#'   is free), `draws` (a matrix of those parameters, one row per draw),
#'   `effect_draws` (a matrix of the random effects `b[g|level|term]`),
#'   `random_effects` (a list by factor of data frames with each `level`
#'   and `term`, its posterior `mean` and `sd`), `divergences` and
#'   `laplace` (the [glmm_model] the chains started from). `predict()`
#'   gives the posterior mean of each row's mean, its effects looked up as
#'   in [glmm_fit()].
#' @export
#' @examples
#' d <- data.frame(y = (0:39 * 7) %% 5 + 0:39 %% 4, site = paste0("s", 0:39 %% 4))
#' m <- bayes_glmm_fit(y ~ 1 + (1 | site), d, chains = 2, tune = 300, draws = 300)
#' m@summary
bayes_glmm_fit <- function(formula, data, family = "poisson", power = NULL, offset = NULL,
                           weights = NULL, prior_sd = 2.5, intercept_sd = 10, sd_scale = 1,
                           lkj = 1, dispersion_scale = 10, chains = 4, tune = 1000, draws = 1000,
                           seed = 0, target_accept = 0.8, max_depth = 10) {
  bars <- glmm_bars(formula)
  des <- model_design(bars$fixed, data, offset, weights)
  g <- glmm_labels(bars$groups, data)
  ptr <- rust_result(bayes_glmm_fit_design(
    as.double(des$x), colnames(des$x), des$y, des$offset, des$weights, family,
    if (is.null(power)) NaN else as.double(power), g$names, g$labels, g$intercepts,
    g$slope_factor, g$slope_names, g$slope_values,
    as.double(c(prior_sd, intercept_sd, sd_scale, lkj, dispersion_scale)),
    as.double(c(chains, tune, draws, seed, target_accept, max_depth))
  ))
  bayes_glmm_model(ptr = ptr, terms = des$terms, xlevels = des$xlevels, groups = bars$groups)
}

#' @rdname bayes_glmm_fit
#' @param ptr,terms,xlevels,groups Internal.
#' @export
bayes_glmm_model <- S7::new_class(
  "bayes_glmm_model",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("BayesGlmmModel"),
    terms = S7::class_any,
    xlevels = S7::class_any,
    groups = S7::class_list,
    coefficients = S7::new_property(S7::class_double, getter = function(self) {
      nm <- self@ptr$names()
      stats::setNames(colMeans(self@draws)[seq_along(nm)], nm)
    }),
    summary = S7::new_property(S7::class_any, getter = function(self) {
      as.data.frame(self@ptr$summary())
    }),
    draws = S7::new_property(S7::class_any, getter = function(self) {
      nm <- self@ptr$parameter_names()
      matrix(self@ptr$parameter_draws(), ncol = length(nm), byrow = TRUE,
             dimnames = list(NULL, nm))
    }),
    effect_draws = S7::new_property(S7::class_any, getter = function(self) {
      nm <- self@ptr$effect_names()
      matrix(self@ptr$effect_draws(), ncol = length(nm), byrow = TRUE,
             dimnames = list(NULL, nm))
    }),
    random_effects = S7::new_property(S7::class_list, getter = function(self) {
      lapply(self@ptr$random_effects(), as.data.frame)
    }),
    divergences = S7::new_property(S7::class_double, getter = function(self) self@ptr$divergences()),
    laplace = S7::new_property(S7::class_any, getter = function(self) {
      glmm_model(ptr = self@ptr$laplace(), terms = self@terms, xlevels = self@xlevels,
                 groups = self@groups)
    })
  ),
  constructor = function(ptr, terms, xlevels, groups) {
    S7::new_object(S7::S7_object(), ptr = ptr, terms = terms, xlevels = xlevels, groups = groups)
  }
)

S7::method(coef, bayes_glmm_model) <- function(object, ...) object@coefficients
S7::method(predict, bayes_glmm_model) <- function(object, newdata, offset = NULL, ...) {
  nd <- new_design(object, newdata, offset)
  g <- glmm_labels(object@groups, newdata, required = FALSE)
  rust_result(object@ptr$predict(as.double(nd$x), colnames(nd$x), nd$offset, g$names, g$labels,
                                 g$intercepts, g$slope_factor, g$slope_names, g$slope_values),
              s7_call())
}
S7::method(print, bayes_glmm_model) <- function(x, ...) {
  cat(sprintf("<bayes_glmm_model> %d divergences\n", as.integer(x@divergences)))
  print(x@summary)
  invisible(x)
}

#' Double GLMs: a model for the dispersion as well as the mean
#'
#' `dglm_fit()` fits a GLM for the mean, `g(mu) = X beta + offset`, with
#' variance `phi V(mu) / w`, and a gamma GLM with the log link for the
#' dispersion, `log(phi) = Z gamma + doffset`, each on its own formula, as
#' `dglm::dglm()` (Smyth's alternating scoring). The dispersion is
#' estimated by maximum likelihood by default, as `dglm()`, or by REML
#' (`method = "reml"`).
#' `dglm` 1.8.6 weights its mean step by `w V(mu) / phi` and leaves the
#' offset out of `mu.eta`, which is right only for a canonical link without
#' an offset; this fit uses the scoring weights, so it agrees with `dglm()`
#' there and with `glm()` at a constant dispersion everywhere
#' (`validation/scripts/r_dglm.R`; `docs/design/models.md`, "Double GLMs").
#'
#' @param formula The mean's formula, with the response.
#' @param dformula The dispersion's one-sided formula, `~ 1` by default.
#' @param data A data frame.
#' @param family `"gaussian"`, `"poisson"`, `"gamma"` or `"tweedie"`.
#' @param link The mean's link; the family's canonical link by default.
#' @param power The Tweedie power, in (1, 2).
#' @param method `"ml"` (the default) or `"reml"`.
#' @param offset,weights Optional offset of the mean (added to any
#'   `offset()` term) and prior weights.
#' @param doffset Optional offset of the log dispersion (added to any
#'   `offset()` term of `dformula`).
#' @param link_power The power link's exponent.
#' @returns A `dglm_model` with `coefficients`, `std_errors` and
#'   `covariance` (the mean's, `summary(fit, dispersion = 1)` in `dglm`),
#'   `dispersion_coefficients`, `dispersion_std_errors` and
#'   `dispersion_covariance` (on the log scale, `dglm`'s
#'   `summary(fit$dispersion.fit, dispersion = 2)`), `fitted`,
#'   `dispersion_fitted`, `m2loglik` (`dglm`'s, with its constants),
#'   `reml_criterion` (`m2loglik + log det(X' W X)`), `aic`, `method` and
#'   `iterations`. `predict(m, newdata, type = "dispersion")` gives the
#'   dispersion of new rows, `type = "response"` (the default) their mean.
#' @export
#' @examples
#' d <- data.frame(y = c(9, 11, 10, 10.5, 4, 18, 6, 15), urban = rep(c("no", "yes"), each = 4))
#' m <- dglm_fit(y ~ urban, ~ urban, d, family = "gamma", link = "log")
#' m@dispersion_coefficients
#' predict(m, data.frame(urban = "yes"), type = "dispersion")
dglm_fit <- function(formula, dformula = ~1, data, family = "gaussian", link = NULL, power = NULL,
                     method = "ml", offset = NULL, weights = NULL, doffset = NULL,
                     link_power = NULL) {
  des <- model_design(formula, data, offset, weights)
  ddes <- model_design(dformula, data, doffset, NULL)
  ptr <- rust_result(dglm_fit_design(
    as.double(des$x), colnames(des$x), des$y, des$offset, des$weights,
    as.double(ddes$x), colnames(ddes$x), ddes$offset, family,
    if (is.null(link)) "" else link, if (is.null(power)) NaN else as.double(power),
    if (is.null(link_power)) NaN else as.double(link_power), method
  ))
  dglm_model(ptr = ptr, terms = des$terms, xlevels = des$xlevels, dterms = ddes$terms,
             dxlevels = ddes$xlevels)
}

#' @rdname dglm_fit
#' @export
dglm_model <- S7::new_class(
  "dglm_model",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("DglmModel"),
    terms = S7::class_any,
    xlevels = S7::class_any,
    dterms = S7::class_any,
    dxlevels = S7::class_any,
    coefficients = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(self@ptr$coefficients(), self@ptr$names())
    }),
    std_errors = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(self@ptr$std_errors(), self@ptr$names())
    }),
    covariance = S7::new_property(S7::class_any, getter = function(self) {
      nm <- self@ptr$names()
      matrix(self@ptr$covariance(), length(nm), dimnames = list(nm, nm))
    }),
    dispersion_coefficients = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(self@ptr$dispersion_coefficients(), self@ptr$dispersion_names())
    }),
    dispersion_std_errors = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(self@ptr$dispersion_std_errors(), self@ptr$dispersion_names())
    }),
    dispersion_covariance = S7::new_property(S7::class_any, getter = function(self) {
      nm <- self@ptr$dispersion_names()
      matrix(self@ptr$dispersion_covariance(), length(nm), dimnames = list(nm, nm))
    }),
    fitted = S7::new_property(S7::class_double, getter = function(self) self@ptr$fitted()),
    dispersion_fitted = S7::new_property(S7::class_double, getter = function(self) {
      self@ptr$dispersion_fitted()
    }),
    m2loglik = S7::new_property(S7::class_double, getter = function(self) self@ptr$m2loglik()),
    reml_criterion = S7::new_property(S7::class_double, getter = function(self) {
      self@ptr$reml_criterion()
    }),
    aic = S7::new_property(S7::class_double, getter = function(self) self@ptr$aic()),
    method = S7::new_property(S7::class_character, getter = function(self) self@ptr$method()),
    iterations = S7::new_property(S7::class_double, getter = function(self) self@ptr$iterations())
  ),
  constructor = function(ptr, terms, xlevels, dterms, dxlevels) {
    S7::new_object(S7::S7_object(), ptr = ptr, terms = terms, xlevels = xlevels, dterms = dterms,
                   dxlevels = dxlevels)
  }
)

S7::method(coef, dglm_model) <- function(object, ...) object@coefficients
S7::method(predict, dglm_model) <- function(object, newdata, type = c("response", "dispersion"),
                                            offset = NULL, ...) {
  type <- match.arg(type)
  if (type == "response") {
    nd <- new_design(object, newdata, offset)
    return(rust_result(object@ptr$predict(as.double(nd$x), colnames(nd$x), nd$offset),
                       s7_call()))
  }
  mf <- stats::model.frame(object@dterms, newdata, xlev = object@dxlevels,
                           na.action = stats::na.fail)
  z <- stats::model.matrix(object@dterms, mf)
  off <- stats::model.offset(mf)
  if (!is.null(offset)) off <- if (is.null(off)) offset else off + offset
  rust_result(object@ptr$predict_dispersion(as.double(z), colnames(z),
                                            if (is.null(off)) double() else as.double(off)),
              s7_call())
}

#' Auto-calibration: isotonic recalibration, balance and reliability
#'
#' A model is auto-calibrated when the mean response among rows with
#' prediction `p` is `p`. `isotonic_fit()` is weighted isotonic regression by
#' pool-adjacent-violators, as scikit-learn's `IsotonicRegression` (tied `x`
#' merged into their weighted mean first, values clipped to `[y_min, y_max]`
#' when given); `predict()` reads it linearly (or as steps) between its
#' points and holds the end values beyond them. `recalibrate()` is the
#' increasing isotonic regression of the response on a model's predictions
#' on held-out data: apply it to new predictions with `predict()`. A
#' frequency model needs a floor (`y_min`), since the lowest pooled level
#' can be 0. `balance_correction()` is the multiplier `sum(w y) / sum(w
#' pred)` that makes the totals agree. `reliability()` gives mean prediction
#' against mean response in bands of about equal weight. `murphy()` splits a
#' family's mean deviance into `miscalibration - discrimination +
#' uncertainty` (Dimitriadis, Gneiting and Jordan's CORP decomposition): the
#' miscalibration is what isotonic recalibration gains.
#'
#' @param x,y Inputs and responses.
#' @param weights Weights (exposures when `y` and `pred` are rates), or
#'   `NULL`.
#' @param increasing Whether the fit is non-decreasing.
#' @param y_min,y_max Bounds on the fitted values, or `NULL`.
#' @param interpolation `"linear"` or `"step"`.
#' @param pred A model's predictions.
#' @param bins Number of bands.
#' @param family A family name, as for [glm_fit()].
#' @param theta,power For the negative binomial and Tweedie.
#' @returns `isotonic_fit()` and `recalibrate()`: an `isotonic_fit` object
#'   with properties `x` and `values`, read at new inputs by `predict()`; `balance_correction()` a number;
#'   `reliability()` a data frame with `lo`, `hi`, `mean_pred`,
#'   `mean_actual`, `weight` and `count`; `murphy()` a list with `score`,
#'   `miscalibration`, `discrimination` and `uncertainty`.
#' @export
#' @examples
#' f <- isotonic_fit(c(1, 2, 3, 4), c(1, 3, 2, 4))
#' f@values
#' predict(f, c(2.5, 0, 9))
#' m <- recalibrate(c(1, 2, 3, 4), c(0, 1, 1, 2), y_min = 0.1)
#' predict(m, c(1, 2.5, 4))
#' balance_correction(c(1, 3), c(1, 1))
#' murphy(c(0, 1, 1, 2), c(0.5, 0.5, 2, 2))
isotonic_fit <- function(x, y, weights = NULL, increasing = TRUE, y_min = NULL, y_max = NULL,
                         interpolation = c("linear", "step")) {
  interpolation <- match.arg(interpolation)
  r <- rust_result(isotonic_rust(as.double(x), as.double(y),
                                 if (is.null(weights)) double() else as.double(weights),
                                 isTRUE(increasing),
                                 if (is.null(y_min)) NaN else as.double(y_min),
                                 if (is.null(y_max)) NaN else as.double(y_max)))
  isotonic_map(x = r$x, values = r$values, increasing = isTRUE(increasing),
               interpolation = interpolation)
}

isotonic_map <- S7::new_class(
  "isotonic_fit",
  package = "prospicio",
  properties = list(
    x = S7::class_double,
    values = S7::class_double,
    increasing = S7::class_logical,
    interpolation = S7::class_character
  )
)

S7::method(predict, isotonic_map) <- function(object, newdata, ...) {
  if (length(object@x) == 1) return(rep(object@values, length(newdata)))
  stats::approx(object@x, object@values, xout = as.double(newdata), rule = 2,
                method = if (object@interpolation == "step") "constant" else "linear", f = 0,
                ties = "ordered")$y
}

S7::method(print, isotonic_map) <- function(x, ...) {
  cat("<isotonic_fit> ", length(x@x), " points, ",
      if (x@increasing) "increasing" else "decreasing", "\n", sep = "")
  invisible(x)
}

#' @rdname isotonic_fit
#' @export
recalibrate <- function(pred, y, weights = NULL, y_min = NULL, y_max = NULL) {
  isotonic_fit(pred, y, weights, TRUE, y_min, y_max)
}

#' @rdname isotonic_fit
#' @export
balance_correction <- function(y, pred, weights = NULL) {
  rust_result(balance_correction_rust(as.double(y), as.double(pred),
                                      if (is.null(weights)) double() else as.double(weights)))
}

#' @rdname isotonic_fit
#' @export
reliability <- function(y, pred, weights = NULL, bins = 10) {
  as.data.frame(rust_result(reliability_rust(as.double(y), as.double(pred),
                                             if (is.null(weights)) double() else as.double(weights),
                                             as.double(bins))))
}

#' @rdname isotonic_fit
#' @export
murphy <- function(y, pred, family = "gaussian", weights = NULL, theta = NaN, power = NaN) {
  rust_result(murphy_rust(as.double(y), as.double(pred), family,
                          if (is.null(weights)) double() else as.double(weights),
                          as.double(theta), as.double(power)))
}

#' Conformal prediction intervals
#'
#' Prediction intervals with coverage of at least `1 - alpha` (split, CQR)
#' or `1 - 2 alpha` (CV+, jackknife+) in finite samples under
#' exchangeability alone, from held-out residuals. The conformal quantile is
#' the `ceiling((1 - alpha)(n + 1))`-th smallest of `n` scores (infinite
#' when that rank exceeds `n`), as MAPIE's.
#'
#' `conformal_split()` scores a model's predictions on a calibration set:
#' `|y - pred|` (`"absolute"`, intervals `pred +/- q`), the Pearson residual
#' of a family (`"pearson"`, `mu +/- q sqrt(V(mu))`) or a residual over a
#' given spread (`"scaled"`, `pred +/- q s`); with `weights` and
#' `test_weights` the quantile is the weighted one with each test row's
#' weight at infinity. `conformal_cqr()` is conformalised quantile
#' regression on a lower and an upper quantile model's predictions.
#' `conformal_plus()` is CV+ (the jackknife+ with one fold a row) from each
#' fold's model's out-of-fold and test predictions. `conformal_intervals()`
#' does it from a model: for `"split"` a fitted model scored on `data` and
#' `y`; for `"cqr"` a list of a fitted lower and upper quantile model; for
#' `"cv+"` and `"jackknife+"` a function of a data frame and its responses
#' returning a fitted model, fitted on each split's training rows. Every
#' fitted model is read with `predict(model, newdata)`.
#'
#' @param y Responses on the calibration (or training) rows.
#' @param pred,test_pred Predictions on the calibration and test rows.
#' @param alpha The miscoverage level.
#' @param score `"absolute"`, `"pearson"` or `"scaled"`.
#' @param family A family name, for `"pearson"`.
#' @param scale,test_scale Spreads, for `"scaled"`.
#' @param weights,test_weights Weights, or `NULL`.
#' @param theta,power For the negative binomial and Tweedie.
#' @param lo,hi,test_lo,test_hi Lower and upper quantile predictions.
#' @param oof_pred Each training row's prediction by its fold's model.
#' @param folds Each training row's fold (1, 2, ...).
#' @param fold_pred A list of each fold's model's test predictions.
#' @param model A fitted model, a list of two, or a fitting function.
#' @param data,newdata Calibration (or training) and test data frames.
#' @param method `"split"`, `"cqr"`, `"cv+"` or `"jackknife+"`.
#' @param splits For `"cv+"`: a list of `list(train, test)` (1-based), each
#'   row in one test set; [k_fold()] with five folds by default.
#' @returns A data frame with `lo` and `hi`, one row a test row.
#' @export
#' @examples
#' conformal_split(0:18, rep(9, 19), 100, alpha = 0.1)
#' d <- data.frame(x = (0:39) %% 7)
#' y <- 1 + 0.5 * d$x + ((0:39 * 31) %% 7) - 3
#' fit <- function(data, y) stats::lm(y ~ x, data = cbind(data, y = y))
#' conformal_intervals(fit, d, y, data.frame(x = 3), method = "jackknife+")
conformal_split <- function(y, pred, test_pred, alpha = 0.1, score = c("absolute", "pearson", "scaled"),
                            family = "", scale = NULL, test_scale = NULL, weights = NULL,
                            test_weights = NULL, theta = NaN, power = NaN) {
  score <- match.arg(score)
  num <- function(v) if (is.null(v)) double() else as.double(v)
  as.data.frame(rust_result(conformal_split_rust(
    num(y), num(pred), num(test_pred), as.double(alpha), score, family, num(scale), num(test_scale),
    num(weights), num(test_weights), as.double(theta), as.double(power)
  )))
}

#' @rdname conformal_split
#' @export
conformal_cqr <- function(y, lo, hi, test_lo, test_hi, alpha = 0.1) {
  as.data.frame(rust_result(conformal_cqr_rust(as.double(y), as.double(lo), as.double(hi),
                                               as.double(test_lo), as.double(test_hi),
                                               as.double(alpha))))
}

#' @rdname conformal_split
#' @export
conformal_plus <- function(y, oof_pred, folds, fold_pred, alpha = 0.1) {
  as.data.frame(rust_result(conformal_plus_rust(as.double(y), as.double(oof_pred), as.double(folds),
                                                lapply(fold_pred, as.double), as.double(alpha))))
}

#' @rdname conformal_split
#' @export
conformal_intervals <- function(model, data, y, newdata, method = c("cv+", "jackknife+", "split", "cqr"),
                                alpha = 0.1, splits = NULL, score = c("absolute", "pearson"),
                                family = "") {
  method <- match.arg(method)
  score <- match.arg(score)
  pr <- function(m, d) as.double(stats::predict(m, newdata = d))
  if (method == "split") {
    return(conformal_split(y, pr(model, data), pr(model, newdata), alpha, score = score, family = family))
  }
  if (method == "cqr") {
    return(conformal_cqr(y, pr(model[[1]], data), pr(model[[2]], data),
                         pr(model[[1]], newdata), pr(model[[2]], newdata), alpha))
  }
  n <- length(y)
  if (method == "jackknife+") {
    splits <- lapply(seq_len(n), function(i) list(train = setdiff(seq_len(n), i), test = i))
  } else if (is.null(splits)) {
    splits <- k_fold(n, 5, 0)
  }
  folds <- rep(NA_integer_, n)
  for (k in seq_along(splits)) {
    if (any(!is.na(folds[splits[[k]]$test]))) stop("each row must be in exactly one split's test set")
    folds[splits[[k]]$test] <- k
  }
  if (anyNA(folds)) stop("each row must be in exactly one split's test set")
  oof <- numeric(n)
  per <- vector("list", length(splits))
  for (k in seq_along(splits)) {
    s <- splits[[k]]
    m <- model(data[s$train, , drop = FALSE], y[s$train])
    oof[s$test] <- pr(m, data[s$test, , drop = FALSE])
    per[[k]] <- pr(m, newdata)
  }
  conformal_plus(y, oof, folds, per, alpha)
}
