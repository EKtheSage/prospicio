# The Pareto family for treaty pricing, and claim counts by dispersion
# (docs/design/pareto.md). S7 classes over the Rust objects, as in
# distributions.R.

# A truncation of NULL is passed to Rust as Inf ("none").
truncation_arg <- function(truncation) {
  if (is.null(truncation)) Inf else as.double(truncation)
}

# Optional per-loss vectors are passed to Rust as empty vectors when NULL.
optional_arg <- function(x) if (is.null(x)) double() else as.double(x)

#' Survival function and layer variance
#'
#' `survival(dist, q)` is `P(X > q)`, accurate far into the tail;
#' `layer_variance(dist, limit, attachment)` is the variance of the loss to
#' the layer `limit` xs `attachment` (`Inf` for an unlimited layer).
#' Defined for the Pareto-family severities ([pareto], [piecewise_pareto],
#' [log_affine_pareto], [generalized_pareto]) and for [gamma_distribution],
#' [tweedie], [weibull_distribution], [loglogistic_distribution],
#' [inverse_gamma_distribution], [inverse_gaussian_distribution],
#' [burr_distribution], [beta_distribution], [truncated_distribution],
#' [mixture_distribution] and [custom_distribution].
#'
#' @param dist A Pareto-family severity.
#' @param q Numeric vector.
#' @param limit Layer limit, a single number.
#' @param attachment Layer attachment, a single number.
#' @param ... Unused; for methods.
#' @returns A numeric vector (`survival`) or a single number
#'   (`layer_variance`).
#' @name pareto_severity
#' @examples
#' p <- pareto(500, 2)
#' survival(p, c(1000, 2000))
#' layer_variance(p, 4000, 1000)
NULL

#' @rdname pareto_severity
#' @export
survival <- S7::new_generic("survival", "dist", function(dist, q, ...) S7::S7_dispatch())

#' @rdname pareto_severity
#' @export
layer_variance <- S7::new_generic("layer_variance", "dist", function(dist, limit, attachment, ...) {
  S7::S7_dispatch()
})

#' Single-parameter Pareto distribution
#'
#' `P(X > x) = (t / x)^alpha` for `x >= t`, optionally truncated (conditioned
#' on `X < truncation`). Parameters are read-only properties: `d@t`,
#' `d@alpha`, `d@truncation` (`Inf` if none).
#'
#' Supports [mean()], [variance()], [stats::quantile()], [cdf()],
#' [survival()], [draws()], [lev()], [stop_loss()], [layer()],
#' [layer_variance()] and [print()], and can be discretized with
#' [discretize()] or used in [simulate_events()].
#'
#' @param t Threshold; finite and positive.
#' @param alpha Pareto alpha; finite and positive.
#' @param truncation Truncation point above `t`, or `NULL` for none.
#' @returns A `pareto` object, which inherits from [distribution].
#' @seealso [pareto_fit()] to estimate `alpha` from large losses.
#' @export
#' @examples
#' p <- pareto(500, 2)
#' layer(p, 4000, 1000)
#' mean(p)
#' pareto(500, 2, truncation = 8000)
pareto <- S7::new_class(
  "pareto",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("Pareto"),
    t = S7::new_property(S7::class_double, getter = function(self) self@ptr$t()),
    alpha = S7::new_property(S7::class_double, getter = function(self) self@ptr$alpha()),
    truncation = S7::new_property(S7::class_double, getter = function(self) self@ptr$truncation())
  ),
  constructor = function(t, alpha, truncation = NULL, ptr = NULL) {
    if (is.null(ptr)) {
      ptr <- rust_result(Pareto$new(as.double(t), as.double(alpha), truncation_arg(truncation)))
    }
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Maximum likelihood fits to large losses
#'
#' Estimate the alpha of a [pareto], the alphas of a [piecewise_pareto], or
#' the initial and tail alphas of Riegel's [generalized_pareto], from large
#' losses, each conditioned on exceeding its reporting threshold (raised to
#' the lowest threshold of the fit), with losses capped by a policy limit
#' treated as censored. Untruncated Pareto estimates are closed forms;
#' truncated ones are solved numerically and clamped to `[0.001, 1000]`. A
#' piecewise Pareto truncated as a whole (`truncation_type = "wd"`) couples
#' the alphas, which are then solved together. The generalized Pareto fit
#' is untruncated; read its alphas as `t / g@beta` and `1 / g@xi`.
#'
#' @param losses Numeric vector of losses at or above `t` (`t[1]`).
#' @param t Threshold (`pareto_fit`) or thresholds (`piecewise_pareto_fit`).
#' @param reporting_thresholds Per-loss reporting thresholds, or `NULL`.
#' @param censored Logical vector, `TRUE` where a loss was capped, or `NULL`.
#' @param weights Per-loss weights, or `NULL`.
#' @param truncation Truncation point, or `NULL`.
#' @param truncation_type For `piecewise_pareto_fit()`: `"lp"` to truncate
#'   the last piece, `"wd"` the whole distribution.
#' @returns A [pareto], [piecewise_pareto] or [generalized_pareto] object.
#' @export
#' @examples
#' pareto_fit(c(1500, 2500, 4000, 10000), 1000, censored = c(FALSE, FALSE, FALSE, TRUE))
#' piecewise_pareto_fit(c(1200, 1500, 2500, 6000), c(1000, 2000))
#' g <- generalized_pareto_fit(c(1100, 1300, 1750, 2600, 4100, 9000, 25000), 1000)
#' c(alpha_ini = 1000 / g@beta, alpha_tail = 1 / g@xi)
pareto_fit <- function(losses, t, reporting_thresholds = NULL, censored = NULL,
                       weights = NULL, truncation = NULL) {
  ptr <- rust_result(Pareto$fit(
    as.double(losses), as.double(t), optional_arg(reporting_thresholds),
    optional_arg(censored), optional_arg(weights), truncation_arg(truncation)
  ))
  pareto(ptr = ptr)
}

#' @rdname pareto_fit
#' @export
piecewise_pareto_fit <- function(losses, t, reporting_thresholds = NULL, censored = NULL,
                                 weights = NULL, truncation = NULL, truncation_type = "lp") {
  ptr <- rust_result(PiecewisePareto$fit(
    as.double(losses), as.double(t), optional_arg(reporting_thresholds),
    optional_arg(censored), optional_arg(weights), truncation_arg(truncation),
    as.character(truncation_type)
  ))
  piecewise_pareto(ptr = ptr)
}

#' @rdname pareto_fit
#' @export
generalized_pareto_fit <- function(losses, t, reporting_thresholds = NULL, censored = NULL,
                                   weights = NULL) {
  ptr <- rust_result(GeneralizedPareto$fit_riegel(
    as.double(losses), as.double(t), optional_arg(reporting_thresholds),
    optional_arg(censored), optional_arg(weights)
  ))
  generalized_pareto(ptr = ptr)
}

#' Convert a local Pareto distribution to piecewise Pareto
#'
#' The local Pareto distribution with local alpha `alpha(x)` above `t`
#' (and `P(X > x) = 1` below it), approximated by a [piecewise_pareto]
#' that matches its survival function exactly at the thresholds and within
#' `rel_tolerance` between them. The conversion stops at `stop_at` or where
#' the survival function falls below `stop_survival`; the local alpha there
#' continues as the tail.
#'
#' @param t Threshold; finite and positive.
#' @param alpha A function of one amount returning the local alpha there:
#'   finite and non-negative, positive where the conversion stops.
#' @param rel_tolerance Largest relative error in the survival function.
#' @param stop_survival,stop_at Where to stop.
#' @returns A list: `severity` (a [piecewise_pareto]), `max_relative_error`
#'   and `approximated_to`.
#' @export
#' @examples
#' res <- local_pareto_to_piecewise(1000, function(x) 1.5 + 0.3 * log(x / 1000))
#' res$max_relative_error
#' res$severity
local_pareto_to_piecewise <- function(t, alpha, rel_tolerance = 1e-4, stop_survival = 1e-9,
                                      stop_at = Inf) {
  res <- rust_result(local_pareto_convert(
    as.double(t), function(x) as.double(alpha(x)), as.double(rel_tolerance),
    as.double(stop_survival), as.double(stop_at)
  ))
  list(severity = piecewise_pareto(ptr = res$severity),
       max_relative_error = res$max_relative_error,
       approximated_to = res$approximated_to)
}

#' Piecewise Pareto distribution
#'
#' Alpha `alpha[k]` above threshold `t[k]`: the general large-loss model and
#' the result of tower matching. Properties: `d@t`, `d@alpha`,
#' `d@truncation` (`Inf` if none), `d@truncation_type`.
#'
#' Supports the same operations as [pareto].
#'
#' @param t Strictly increasing positive thresholds.
#' @param alpha One alpha per threshold; interior ones may be 0, the last
#'   must be positive.
#' @param truncation Truncation point above the last threshold, or `NULL`.
#' @param truncation_type `"lp"` to truncate the last piece only, `"wd"` to
#'   truncate the whole distribution.
#' @returns A `piecewise_pareto` object, which inherits from [distribution].
#' @export
#' @examples
#' pp <- piecewise_pareto(c(1000, 2000), c(1, 2))
#' survival(pp, 4000)
#' stop_loss(pp, 2000)
piecewise_pareto <- S7::new_class(
  "piecewise_pareto",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("PiecewisePareto"),
    t = S7::new_property(S7::class_double, getter = function(self) self@ptr$t()),
    alpha = S7::new_property(S7::class_double, getter = function(self) self@ptr$alpha()),
    truncation = S7::new_property(S7::class_double, getter = function(self) self@ptr$truncation()),
    truncation_type = S7::new_property(
      S7::class_character, getter = function(self) self@ptr$truncation_type()
    )
  ),
  constructor = function(t, alpha, truncation = NULL, truncation_type = "lp", ptr = NULL) {
    if (is.null(ptr)) {
      ptr <- rust_result(PiecewisePareto$new(
        as.double(t), as.double(alpha), truncation_arg(truncation), as.character(truncation_type)
      ))
    }
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Log-affine local Pareto distribution
#'
#' The local Pareto alpha `alpha0 * (1 + gamma * log(x / t))` rises linearly
#' in the log of the amount, so with `L = log(x / t)`,
#' `P(X > x) = exp(-alpha0 L - alpha0 gamma L^2 / 2)`. Give either `gamma`
#' or `delta = alpha0 * gamma * log(2)`, the rise in alpha each time the
#' amount doubles. Properties: `d@t`, `d@alpha0`, `d@gamma`, `d@delta`.
#'
#' Supports the same operations as [pareto], and [local_alpha()].
#'
#' @param t Threshold; finite and positive.
#' @param alpha0 Local alpha at `t`; finite and positive.
#' @param gamma Non-negative; 0 gives the Pareto.
#' @param delta Alternative to `gamma`.
#' @param ptr Internal: an existing object to wrap.
#' @returns A `log_affine_pareto` object, which inherits from [distribution].
#' @export
#' @examples
#' d <- log_affine_pareto(1e6, 1.5, delta = 0.5)
#' local_alpha(d, 2e6)
#' layer(d, 4e6, 1e6)
log_affine_pareto <- S7::new_class(
  "log_affine_pareto",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("LogAffinePareto"),
    t = S7::new_property(S7::class_double, getter = function(self) self@ptr$t()),
    alpha0 = S7::new_property(S7::class_double, getter = function(self) self@ptr$alpha0()),
    gamma = S7::new_property(S7::class_double, getter = function(self) self@ptr$gamma()),
    delta = S7::new_property(S7::class_double, getter = function(self) self@ptr$delta())
  ),
  constructor = function(t, alpha0, gamma = NULL, delta = NULL, ptr = NULL) {
    if (!is.null(ptr)) return(S7::new_object(S7::S7_object(), ptr = ptr))
    if (is.null(gamma) == is.null(delta)) {
      stop("give exactly one of gamma and delta")
    }
    ptr <- if (is.null(delta)) {
      rust_result(LogAffinePareto$new(as.double(t), as.double(alpha0), as.double(gamma)))
    } else {
      rust_result(LogAffinePareto$from_delta(as.double(t), as.double(alpha0), as.double(delta)))
    }
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Local Pareto alpha
#'
#' The elasticity `-x S'(x) / S(x)` of the survival function at `x`.
#'
#' @param dist A [log_affine_pareto].
#' @param x Numeric vector.
#' @param ... Unused; for methods.
#' @returns Numeric vector the length of `x`.
#' @export
#' @examples
#' local_alpha(log_affine_pareto(1000, 2, gamma = 0.5), c(1000, 2000))
local_alpha <- S7::new_generic("local_alpha", "dist", function(dist, x, ...) S7::S7_dispatch())

S7::method(local_alpha, log_affine_pareto) <- function(dist, x, ...) {
  dist@ptr$local_alpha(as.double(x))
}

#' Generalized Pareto severity
#'
#' `P(X > x) = (1 + xi (x - location) / beta)^(-1 / xi)` above `location`.
#' `generalized_pareto_riegel(t, alpha_ini, alpha_tail)` is Riegel's
#' parameterization: local alpha `alpha_ini` at the threshold `t`, tending
#' to `alpha_tail` far out. Properties: `d@xi`, `d@beta`, `d@location`. For
#' tail estimation from draws, see [gpd_fit()].
#'
#' Supports the same operations as [pareto].
#'
#' @param xi Shape.
#' @param beta Scale; finite and positive.
#' @param location Start of the support.
#' @returns A `generalized_pareto` object, which inherits from
#'   [distribution].
#' @export
#' @examples
#' g <- generalized_pareto_riegel(1000, 2, 1.5)
#' survival(g, 2000)
#' layer(g, 4000, 1000)
generalized_pareto <- S7::new_class(
  "generalized_pareto",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("GeneralizedPareto"),
    xi = S7::new_property(S7::class_double, getter = function(self) self@ptr$xi()),
    beta = S7::new_property(S7::class_double, getter = function(self) self@ptr$beta()),
    location = S7::new_property(S7::class_double, getter = function(self) self@ptr$location())
  ),
  constructor = function(xi, beta, location = 0, ptr = NULL) {
    if (is.null(ptr)) {
      ptr <- rust_result(GeneralizedPareto$new(as.double(xi), as.double(beta), as.double(location)))
    }
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' @rdname generalized_pareto
#' @param t Threshold; finite and positive.
#' @param alpha_ini Local alpha at `t`; positive.
#' @param alpha_tail Local alpha far out; positive.
#' @export
generalized_pareto_riegel <- function(t, alpha_ini, alpha_tail) {
  ptr <- rust_result(GeneralizedPareto$riegel(as.double(t), as.double(alpha_ini), as.double(alpha_tail)))
  generalized_pareto(ptr = ptr)
}

#' Gamma distribution
#'
#' Shape `shape` and scale `scale`: mean `shape * scale`, variance
#' `shape * scale^2`, as in [stats::dgamma()]. `gamma_from_mean_cv()` takes
#' the mean and coefficient of variation; `gamma_from_mean_dispersion()` the
#' GLM parameterization, mean `mu` and dispersion `phi` (variance
#' `phi * mu^2`). Properties: `d@shape`, `d@scale`.
#'
#' Supports the same operations as [pareto], plus [log_density()].
#'
#' @param shape,scale Finite and positive.
#' @returns A `gamma_distribution` object, which inherits from [distribution].
#' @export
#' @examples
#' g <- gamma_from_mean_cv(1000, 0.5)
#' g@shape
#' layer(g, 1000, 1000)
gamma_distribution <- S7::new_class(
  "gamma_distribution",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("GammaDist"),
    shape = S7::new_property(S7::class_double, getter = function(self) self@ptr$shape()),
    scale = S7::new_property(S7::class_double, getter = function(self) self@ptr$scale())
  ),
  constructor = function(shape, scale, ptr = NULL) {
    if (is.null(ptr)) ptr <- rust_result(GammaDist$new(as.double(shape), as.double(scale)))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' @rdname gamma_distribution
#' @param mean Mean; finite and positive.
#' @param cv Coefficient of variation; finite and positive.
#' @export
gamma_from_mean_cv <- function(mean, cv) {
  gamma_distribution(ptr = rust_result(GammaDist$from_mean_cv(as.double(mean), as.double(cv))))
}

#' @rdname gamma_distribution
#' @param dispersion GLM dispersion `phi`; finite and positive.
#' @export
gamma_from_mean_dispersion <- function(mean, dispersion) {
  gamma_distribution(ptr = rust_result(GammaDist$from_mean_dispersion(as.double(mean), as.double(dispersion))))
}

#' Tweedie distribution
#'
#' Mean `mean`, dispersion `dispersion` and power `1 < power < 2`: variance
#' `dispersion * mean^power`, a point mass `exp(-lambda)` at 0 and a
#' continuous density above it. It is a Poisson number of gamma losses, the
#' GLM family for pure premium; `tweedie_from_poisson_gamma()` builds it
#' from those. Properties: `d@mean_param`, `d@dispersion`, `d@power`,
#' `d@lambda` (expected number of losses) and `d@severity` (a
#' [gamma_distribution]).
#'
#' Supports the same operations as [pareto], plus [log_density()].
#'
#' @param mean Mean; finite and positive.
#' @param dispersion Finite and positive.
#' @param power In `(1, 2)`.
#' @returns A `tweedie` object, which inherits from [distribution].
#' @export
#' @examples
#' y <- tweedie(500, 40, 1.6)
#' cdf(y, 0) - exp(-y@lambda)
#' log_density(y, c(100, 500))
tweedie <- S7::new_class(
  "tweedie",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("TweedieDist"),
    mean_param = S7::new_property(S7::class_double, getter = function(self) self@ptr$mean()),
    dispersion = S7::new_property(S7::class_double, getter = function(self) self@ptr$dispersion()),
    power = S7::new_property(S7::class_double, getter = function(self) self@ptr$power()),
    lambda = S7::new_property(S7::class_double, getter = function(self) self@ptr$lambda()),
    severity = S7::new_property(S7::class_any, getter = function(self) {
      gamma_distribution(ptr = self@ptr$severity())
    })
  ),
  constructor = function(mean, dispersion, power, ptr = NULL) {
    if (is.null(ptr)) {
      ptr <- rust_result(TweedieDist$new(as.double(mean), as.double(dispersion), as.double(power)))
    }
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' @rdname tweedie
#' @param lambda Expected number of losses; finite and positive.
#' @param shape,scale Gamma shape and scale of each loss.
#' @export
tweedie_from_poisson_gamma <- function(lambda, shape, scale) {
  tweedie(ptr = rust_result(TweedieDist$from_poisson_gamma(as.double(lambda), as.double(shape), as.double(scale))))
}

#' Weibull distribution
#'
#' `P(X > x) = exp(-(x / scale)^shape)`, as [stats::dweibull()]. Heavier
#' than exponential below shape 1. Properties: `d@shape`, `d@scale`.
#'
#' Supports the same operations as [pareto].
#'
#' @param shape,scale Finite and positive.
#' @param ptr Internal: an existing object to wrap.
#' @returns A `weibull_distribution` object, which inherits from
#'   [distribution].
#' @export
#' @examples
#' w <- weibull_distribution(0.7, 1000)
#' stop_loss(w, 5000)
weibull_distribution <- S7::new_class(
  "weibull_distribution",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("WeibullDist"),
    shape = S7::new_property(S7::class_double, getter = function(self) self@ptr$shape()),
    scale = S7::new_property(S7::class_double, getter = function(self) self@ptr$scale())
  ),
  constructor = function(shape, scale, ptr = NULL) {
    if (is.null(ptr)) ptr <- rust_result(WeibullDist$new(as.double(shape), as.double(scale)))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Loglogistic distribution
#'
#' `F(x) = (x / scale)^shape / (1 + (x / scale)^shape)`, as actuar's
#' `dllogis()` and SciPy's `fisk`; `scale` is the median. Its [cdf()] is
#' Clark's loglogistic growth curve. The mean is infinite for `shape <= 1`
#' and the variance for `shape <= 2`; limited and layer moments always
#' exist. Properties: `d@shape`, `d@scale`.
#'
#' Supports the same operations as [pareto].
#'
#' @param shape,scale Finite and positive.
#' @param ptr Internal: an existing object to wrap.
#' @returns A `loglogistic_distribution` object, which inherits from
#'   [distribution].
#' @export
#' @examples
#' g <- loglogistic_distribution(1.5, 24)
#' cdf(g, c(12, 24, 48)) # share of ultimate reported by age
#' lev(g, 120)
loglogistic_distribution <- S7::new_class(
  "loglogistic_distribution",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("LoglogisticDist"),
    shape = S7::new_property(S7::class_double, getter = function(self) self@ptr$shape()),
    scale = S7::new_property(S7::class_double, getter = function(self) self@ptr$scale())
  ),
  constructor = function(shape, scale, ptr = NULL) {
    if (is.null(ptr)) ptr <- rust_result(LoglogisticDist$new(as.double(shape), as.double(scale)))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Inverse gamma distribution
#'
#' `X = scale / G` for `G` a unit-scale gamma with shape `shape`, as
#' actuar's `dinvgamma()` and SciPy's `invgamma`. The tail is Pareto-like:
#' the mean is infinite for `shape <= 1` and the variance for `shape <= 2`;
#' limited and layer moments always exist. Properties: `d@shape`,
#' `d@scale`.
#'
#' Supports the same operations as [pareto].
#'
#' @param shape,scale Finite and positive.
#' @param ptr Internal: an existing object to wrap.
#' @returns An `inverse_gamma_distribution` object, which inherits from
#'   [distribution].
#' @export
#' @examples
#' d <- inverse_gamma_distribution(3, 2000)
#' mean(d)
#' layer(d, 4000, 1000)
inverse_gamma_distribution <- S7::new_class(
  "inverse_gamma_distribution",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("InverseGammaDist"),
    shape = S7::new_property(S7::class_double, getter = function(self) self@ptr$shape()),
    scale = S7::new_property(S7::class_double, getter = function(self) self@ptr$scale())
  ),
  constructor = function(shape, scale, ptr = NULL) {
    if (is.null(ptr)) ptr <- rust_result(InverseGammaDist$new(as.double(shape), as.double(scale)))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Inverse Gaussian distribution
#'
#' Mean `mean` and shape `shape` (lambda), variance `mean^3 / shape`, as
#' actuar's `dinvgauss()`. Properties: `d@mean_param`, `d@shape`.
#'
#' Supports the same operations as [pareto].
#'
#' @param mean,shape Finite and positive.
#' @param ptr Internal: an existing object to wrap.
#' @returns An `inverse_gaussian_distribution` object, which inherits from
#'   [distribution].
#' @export
#' @examples
#' d <- inverse_gaussian_distribution(1000, 4000)
#' sqrt(variance(d))
#' stop_loss(d, 2000)
inverse_gaussian_distribution <- S7::new_class(
  "inverse_gaussian_distribution",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("InverseGaussianDist"),
    mean_param = S7::new_property(S7::class_double, getter = function(self) self@ptr$mean_param()),
    shape = S7::new_property(S7::class_double, getter = function(self) self@ptr$shape())
  ),
  constructor = function(mean, shape, ptr = NULL) {
    if (is.null(ptr)) ptr <- rust_result(InverseGaussianDist$new(as.double(mean), as.double(shape)))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Burr distribution
#'
#' The Burr (type XII): `P(X > x) = (1 + (x / scale)^gamma)^(-alpha)`, as
#' actuar's `dburr(shape1 = alpha, shape2 = gamma, scale)` and SciPy's
#' `burr12(c = gamma, d = alpha)`. With `alpha = 1` it is the
#' [loglogistic_distribution]; with `gamma = 1`, the Lomax. Moments exist
#' below `alpha * gamma`; limited and layer moments always do. Properties:
#' `d@alpha`, `d@gamma`, `d@scale`.
#'
#' Supports the same operations as [pareto].
#'
#' @param alpha,gamma,scale Finite and positive.
#' @param ptr Internal: an existing object to wrap.
#' @returns A `burr_distribution` object, which inherits from
#'   [distribution].
#' @export
#' @examples
#' d <- burr_distribution(1.5, 2, 1500)
#' survival(d, 3000)
#' layer(d, 1e5, 1e4)
burr_distribution <- S7::new_class(
  "burr_distribution",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("BurrDist"),
    alpha = S7::new_property(S7::class_double, getter = function(self) self@ptr$alpha()),
    gamma = S7::new_property(S7::class_double, getter = function(self) self@ptr$gamma()),
    scale = S7::new_property(S7::class_double, getter = function(self) self@ptr$scale())
  ),
  constructor = function(alpha, gamma, scale, ptr = NULL) {
    if (is.null(ptr)) {
      ptr <- rust_result(BurrDist$new(as.double(alpha), as.double(gamma), as.double(scale)))
    }
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Beta distribution on `[0, scale]`
#'
#' `X / scale` is Beta(`a`, `b`): a bounded severity, such as a damage
#' ratio times a sum insured. Properties: `d@a`, `d@b`, `d@scale`.
#'
#' Supports the same operations as [pareto].
#'
#' @param a,b,scale Finite and positive.
#' @param ptr Internal: an existing object to wrap.
#' @returns A `beta_distribution` object, which inherits from
#'   [distribution].
#' @export
#' @examples
#' d <- beta_distribution(2, 5, 10000)
#' mean(d)
#' lev(d, 3000)
beta_distribution <- S7::new_class(
  "beta_distribution",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("BetaDist"),
    a = S7::new_property(S7::class_double, getter = function(self) self@ptr$a()),
    b = S7::new_property(S7::class_double, getter = function(self) self@ptr$b()),
    scale = S7::new_property(S7::class_double, getter = function(self) self@ptr$scale())
  ),
  constructor = function(a, b, scale = 1, ptr = NULL) {
    if (is.null(ptr)) ptr <- rust_result(BetaDist$new(as.double(a), as.double(b), as.double(scale)))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Truncated severity and splicing
#'
#' `truncated_distribution()` conditions a severity on
#' `lower < X <= upper`, as `aggregate`'s `sev_lb` and `sev_ub`: its
#' distribution function is `(F(x) - F(lower)) / (F(upper) - F(lower))` on
#' the window. Layer moments are exact, from the inner severity's.
#' Properties: `d@lower`, `d@upper`, `d@probability` (of the window under
#' the inner severity), `d@family` (the inner severity's).
#'
#' `splice_distribution()` is a [mixture_distribution] of such pieces:
#' component `i` conditioned on `(breaks[i], breaks[i + 1]]` with weight
#' `weights[i]`, such as a lognormal body and a Pareto tail.
#'
#' Supports the same operations as [pareto].
#'
#' @param severity Any severity but a [sampled] distribution.
#' @param lower,upper The window, `0 <= lower < upper <= Inf`.
#' @param weights Positive weights summing to 1.
#' @param components A list of severities, one per piece.
#' @param breaks One more than the components: 0, the joins, and the top
#'   (which may be `Inf`).
#' @param ptr Internal: an existing object to wrap.
#' @returns A `truncated_distribution` or a [mixture_distribution], which
#'   inherit from [distribution].
#' @export
#' @examples
#' t <- truncated_distribution(lognormal_from_mean_cv(1000, 1), upper = 5000)
#' cdf(t, 5000)
#' s <- splice_distribution(c(0.9, 0.1),
#'   list(lognormal_from_mean_cv(50, 1), pareto(100, 1.8)), c(0, 100, Inf))
#' cdf(s, 100)
truncated_distribution <- S7::new_class(
  "truncated_distribution",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("TruncatedDist"),
    lower = S7::new_property(S7::class_double, getter = function(self) self@ptr$lower()),
    upper = S7::new_property(S7::class_double, getter = function(self) self@ptr$upper()),
    probability = S7::new_property(S7::class_double, getter = function(self) self@ptr$probability()),
    family = S7::new_property(S7::class_character, getter = function(self) self@ptr$family())
  ),
  constructor = function(severity, lower = 0, upper = Inf, ptr = NULL) {
    if (is.null(ptr)) {
      ptr <- rust_result(TruncatedDist$new(severity@ptr, as.double(lower), as.double(upper)))
    }
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' @rdname truncated_distribution
#' @export
splice_distribution <- function(weights, components, breaks) {
  ptr <- rust_result(MixtureDist$splice(
    as.double(weights), lapply(components, function(c) c@ptr), as.double(breaks)
  ))
  mixture_distribution(ptr = ptr)
}

#' Mixture of severities
#'
#' A loss from component `i` with probability `weights[i]`: attritional and
#' large losses in one severity. Distribution functions, limited expected
#' values and layer moments are the weighted sums of the components'.
#' Property: `d@weights`.
#'
#' Supports the same operations as [pareto], and can be discretized or used
#' in [simulate_events()].
#'
#' @param weights Positive weights summing to 1.
#' @param components A list of severities ([lognormal], [gamma_distribution],
#'   [weibull_distribution], Pareto-family, ...).
#' @param ptr Internal: an existing object to wrap.
#' @returns A `mixture_distribution` object, which inherits from
#'   [distribution].
#' @export
#' @examples
#' m <- mixture_distribution(c(0.9, 0.1), list(lognormal_from_mean_cv(1e4, 1), pareto(1e5, 2)))
#' mean(m)
mixture_distribution <- S7::new_class(
  "mixture_distribution",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("MixtureDist"),
    weights = S7::new_property(S7::class_double, getter = function(self) self@ptr$weights())
  ),
  constructor = function(weights, components, ptr = NULL) {
    if (is.null(ptr)) {
      ptr <- rust_result(MixtureDist$new(as.double(weights), lapply(components, function(c) c@ptr)))
    }
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Custom severity from your own distribution function
#'
#' The slow path for a distribution the package does not have: give its
#' `cdf`, and its `quantile` function if you have one (sampling inverts the
#' cdf by bisection otherwise, about a hundred cdf calls per draw). The
#' mean, variance, limited expected values and layer moments are computed
#' by Gauss-Legendre quadrature of the survival function between the
#' distribution's own quantiles, ignoring the probability above the
#' `1 - 1e-12` quantile. It goes anywhere a severity does (layers,
#' [compound_distribution()], [simulate_events()], copula marginals,
#' [mixture_distribution()]); calculations that meet one run
#' single-threaded on R's main thread, since every value calls back into R.
#'
#' Properties: `d@name`, `d@has_quantile`, `d@upper` (the `1 - 1e-12`
#' quantile, where the integrals stop) and `d@last_error` (the first error a
#' function raised after construction, or `""`; that value became `NaN`).
#'
#' Supports the same operations as [pareto].
#'
#' @param cdf `function(x)` returning `P(X <= x)` for one `x >= 0`: a number
#'   in `[0, 1]`, non-decreasing in `x`. Losses are non-negative.
#' @param quantile Optional `function(p)` returning the smallest `x` with
#'   `cdf(x) >= p`.
#' @param name Shown in errors and when printed.
#' @returns A `custom_distribution` object, which inherits from
#'   [distribution]. Construction fails if a function errors, returns a
#'   value out of range, or the cdf never reaches `1 - 1e-12`.
#' @export
#' @examples
#' d <- custom_distribution(function(x) 1 - exp(-x / 100), name = "exponential")
#' mean(d)
#' lev(d, 50)
custom_distribution <- S7::new_class(
  "custom_distribution",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("CustomDist"),
    name = S7::new_property(S7::class_character, getter = function(self) self@ptr$name()),
    has_quantile = S7::new_property(S7::class_logical, getter = function(self) self@ptr$has_quantile()),
    upper = S7::new_property(S7::class_double, getter = function(self) self@ptr$upper()),
    last_error = S7::new_property(S7::class_character, getter = function(self) self@ptr$last_error())
  ),
  constructor = function(cdf, quantile = NULL, name = "custom") {
    if (!is.function(cdf)) stop("cdf must be a function")
    if (!is.null(quantile) && !is.function(quantile)) stop("quantile must be a function or NULL")
    ptr <- rust_result(CustomDist$new(cdf, quantile, as.character(name)))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Log density
#'
#' The log of the density at `x`. For a [tweedie], `x = 0` gives the log of
#' the point mass, `-lambda`.
#'
#' @param dist A [gamma_distribution] or [tweedie].
#' @param x Numeric vector.
#' @param ... Unused; for methods.
#' @returns Numeric vector the length of `x`.
#' @export
#' @examples
#' log_density(gamma_distribution(2, 3), c(1, 5))
log_density <- S7::new_generic("log_density", "dist", function(dist, x, ...) S7::S7_dispatch())

S7::method(log_density, gamma_distribution) <- function(dist, x, ...) dist@ptr$ln_pdf(as.double(x))
S7::method(log_density, tweedie) <- function(dist, x, ...) dist@ptr$ln_pdf(as.double(x))

for (cls in list(pareto, piecewise_pareto, log_affine_pareto, generalized_pareto,
                 gamma_distribution, tweedie, weibull_distribution, loglogistic_distribution,
                 inverse_gamma_distribution, inverse_gaussian_distribution, burr_distribution,
                 beta_distribution, truncated_distribution, mixture_distribution,
                 custom_distribution)) {
  S7::method(mean, cls) <- function(x, ...) x@ptr$mean()
  S7::method(variance, cls) <- function(dist, ...) dist@ptr$variance()
  S7::method(cdf, cls) <- function(dist, q, ...) dist@ptr$cdf(as.double(q))
  S7::method(survival, cls) <- function(dist, q, ...) dist@ptr$survival(as.double(q))
  S7::method(quantile, cls) <- function(x, probs, ...) {
    call <- sys.call()
    call[[1]] <- quote(quantile)
    rust_result(x@ptr$quantile(as.double(probs)), call)
  }
  S7::method(draws, cls) <- function(dist, n, seed, stream = 0, ...) {
    rust_result(dist@ptr$sample(as.double(n), as.double(seed), as.double(stream)), s7_call())
  }
  S7::method(lev, cls) <- function(dist, limit, ...) dist@ptr$lev(as.double(limit))
  S7::method(stop_loss, cls) <- function(dist, retention, ...) dist@ptr$stop_loss(as.double(retention))
  S7::method(layer, cls) <- function(dist, limit, attachment, ...) {
    dist@ptr$layer(as.double(limit), as.double(attachment))
  }
  S7::method(layer_variance, cls) <- function(dist, limit, attachment, ...) {
    dist@ptr$layer_variance(as.double(limit), as.double(attachment))
  }
}

S7::method(print, pareto) <- function(x, ...) {
  tr <- if (is.finite(x@truncation)) sprintf(", truncation = %s", format(x@truncation, digits = 15)) else ""
  cat(sprintf("<pareto> t = %s, alpha = %s%s\n",
              format(x@t, digits = 15), format(x@alpha, digits = 15), tr))
  invisible(x)
}

S7::method(print, piecewise_pareto) <- function(x, ...) {
  cat(sprintf("<piecewise_pareto> %d pieces\n", length(x@t)))
  print(data.frame(t = x@t, alpha = x@alpha))
  if (is.finite(x@truncation)) {
    cat(sprintf("truncated at %s (%s)\n", format(x@truncation, digits = 15), x@truncation_type))
  }
  invisible(x)
}

S7::method(print, log_affine_pareto) <- function(x, ...) {
  cat(sprintf("<log_affine_pareto> t = %s, alpha0 = %s, gamma = %s (delta = %s)\n",
              format(x@t, digits = 15), format(x@alpha0, digits = 15),
              format(x@gamma, digits = 15), format(x@delta, digits = 15)))
  invisible(x)
}

S7::method(print, gamma_distribution) <- function(x, ...) {
  cat(sprintf("<gamma_distribution> shape = %s, scale = %s\n",
              format(x@shape, digits = 15), format(x@scale, digits = 15)))
  invisible(x)
}
S7::method(print, tweedie) <- function(x, ...) {
  cat(sprintf("<tweedie> mean = %s, dispersion = %s, power = %s\n",
              format(x@mean_param, digits = 15), format(x@dispersion, digits = 15),
              format(x@power, digits = 15)))
  invisible(x)
}
S7::method(print, weibull_distribution) <- function(x, ...) {
  cat(sprintf("<weibull_distribution> shape = %s, scale = %s\n",
              format(x@shape, digits = 15), format(x@scale, digits = 15)))
  invisible(x)
}
S7::method(print, loglogistic_distribution) <- function(x, ...) {
  cat(sprintf("<loglogistic_distribution> shape = %s, scale = %s\n",
              format(x@shape, digits = 15), format(x@scale, digits = 15)))
  invisible(x)
}
S7::method(print, inverse_gamma_distribution) <- function(x, ...) {
  cat(sprintf("<inverse_gamma_distribution> shape = %s, scale = %s\n",
              format(x@shape, digits = 15), format(x@scale, digits = 15)))
  invisible(x)
}
S7::method(print, inverse_gaussian_distribution) <- function(x, ...) {
  cat(sprintf("<inverse_gaussian_distribution> mean = %s, shape = %s\n",
              format(x@mean_param, digits = 15), format(x@shape, digits = 15)))
  invisible(x)
}
S7::method(print, burr_distribution) <- function(x, ...) {
  cat(sprintf("<burr_distribution> alpha = %s, gamma = %s, scale = %s\n",
              format(x@alpha, digits = 15), format(x@gamma, digits = 15),
              format(x@scale, digits = 15)))
  invisible(x)
}
S7::method(print, beta_distribution) <- function(x, ...) {
  cat(sprintf("<beta_distribution> a = %s, b = %s, scale = %s\n",
              format(x@a, digits = 15), format(x@b, digits = 15), format(x@scale, digits = 15)))
  invisible(x)
}
S7::method(print, truncated_distribution) <- function(x, ...) {
  cat(sprintf("<truncated_distribution> %s on (%s, %s]\n", x@family,
              format(x@lower, digits = 15), format(x@upper, digits = 15)))
  invisible(x)
}
S7::method(print, custom_distribution) <- function(x, ...) {
  cat(sprintf("<custom_distribution> %s, mean = %s%s\n", x@name, format(mean(x), digits = 8),
              if (x@has_quantile) ", with quantile" else ""))
  invisible(x)
}

S7::method(print, mixture_distribution) <- function(x, ...) {
  cat(sprintf("<mixture_distribution> weights %s\n", paste(format(x@weights), collapse = ", ")))
  invisible(x)
}
S7::method(print, generalized_pareto) <- function(x, ...) {
  cat(sprintf("<generalized_pareto> xi = %s, beta = %s, location = %s\n",
              format(x@xi, digits = 15), format(x@beta, digits = 15),
              format(x@location, digits = 15)))
  invisible(x)
}

#' Binomial claim counts
#'
#' `n` risks, each claiming with probability `p`. Supports [pmf()],
#' [cdf()], [mean()], [variance()], `quantile()` and [draws()].
#'
#' @param n Number of trials, a non-negative whole number.
#' @param p Claim probability in `[0, 1)`.
#' @returns A `binomial_count` object, which inherits from [distribution].
#' @seealso [claim_count()] to choose a count by its dispersion.
#' @export
#' @examples
#' n <- binomial_count(10, 0.3)
#' pmf(n, 0:3)
#' mean(n)
binomial_count <- S7::new_class(
  "binomial_count",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("Binomial"),
    n = S7::new_property(S7::class_double, getter = function(self) self@ptr$n()),
    p = S7::new_property(S7::class_double, getter = function(self) self@ptr$p())
  ),
  constructor = function(n, p) {
    ptr <- rust_result(Binomial$new(as.double(n), as.double(p)))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

S7::method(pmf, binomial_count) <- function(dist, k, ...) rust_result(dist@ptr$pmf(as.double(k)), s7_call())
S7::method(cdf, binomial_count) <- function(dist, q, ...) rust_result(dist@ptr$cdf(as.double(q)), s7_call())
S7::method(mean, binomial_count) <- function(x, ...) x@ptr$mean()
S7::method(variance, binomial_count) <- function(dist, ...) dist@ptr$variance()
S7::method(quantile, binomial_count) <- function(x, probs, ...) {
  call <- sys.call()
  call[[1]] <- quote(quantile)
  rust_result(x@ptr$quantile(as.double(probs)), call)
}
S7::method(draws, binomial_count) <- function(dist, n, seed, stream = 0, ...) {
  rust_result(dist@ptr$sample(as.double(n), as.double(seed), as.double(stream)), s7_call())
}
S7::method(print, binomial_count) <- function(x, ...) {
  cat(sprintf("<binomial_count> n = %s, p = %s\n", format(x@n), format(x@p, digits = 15)))
  invisible(x)
}

#' Claim count by mean and dispersion
#'
#' The Panjer-class count with this mean and dispersion `Var[N] / E[N]`:
#' [binomial_count] below 1, [poisson_count] at 1, [negative_binomial_count]
#' above 1. A binomial needs a whole number of trials, so below 1 the trials
#' are `mean / (1 - dispersion)` rounded up: the mean is kept and the
#' dispersion moves up to the nearest attainable value.
#'
#' @param mean Mean number of claims; finite and non-negative.
#' @param dispersion Positive.
#' @returns A [binomial_count], [poisson_count] or [negative_binomial_count].
#' @export
#' @examples
#' claim_count(4, 2.5)
#' claim_count(10, 0.3)
claim_count <- function(mean, dispersion) {
  k <- rust_result(claim_count_parameters(as.double(mean), as.double(dispersion)))
  switch(k$kind,
    binomial = binomial_count(k$n, k$p),
    poisson = poisson_count(k$lambda),
    negative_binomial = negative_binomial_count(k$r, k$beta)
  )
}

#' Save and load distributions as JSON
#'
#' `dist_to_json()` writes a distribution as a versioned JSON document: its
#' family and the parameters its constructor takes, numbers bit for bit.
#' `dist_from_json()` reads it back as an equal distribution of the same
#' class. Mixtures are saved with their components. A [custom_distribution]
#' cannot be saved: it is an R function.
#'
#' @param dist Any distribution: a parametric severity, a
#'   [grid_distribution], a [sampled] distribution or a
#'   [mixture_distribution].
#' @param text A document written by `dist_to_json()`.
#' @returns `dist_to_json()`: a single string. `dist_from_json()`: the
#'   distribution.
#' @export
#' @examples
#' text <- dist_to_json(lognormal(7, 0.5))
#' d <- dist_from_json(text)
#' mean(d) == mean(lognormal(7, 0.5))
dist_to_json <- function(dist) rust_result(dist_to_json_rust(dist@ptr))

#' @rdname dist_to_json
#' @export
dist_from_json <- function(text) {
  r <- rust_result(dist_from_json_rust(as.character(text)))
  cls <- switch(r$family,
    lognormal = lognormal, pareto = pareto, piecewise_pareto = piecewise_pareto,
    log_affine_pareto = log_affine_pareto, generalized_pareto = generalized_pareto,
    gamma = gamma_distribution, tweedie = tweedie, weibull = weibull_distribution,
    loglogistic = loglogistic_distribution, inverse_gamma = inverse_gamma_distribution,
    inverse_gaussian = inverse_gaussian_distribution, burr = burr_distribution,
    beta = beta_distribution, truncated = truncated_distribution, mixture = mixture_distribution,
    grid = grid_distribution, sampled = sampled,
    stop("unknown family ", r$family)
  )
  cls(ptr = r$ptr)
}
