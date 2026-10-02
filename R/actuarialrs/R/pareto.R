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
#' [log_affine_pareto], [generalized_pareto]).
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
  package = "actuarialrs",
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
#' Estimate the alpha of a [pareto] or the alphas of a [piecewise_pareto]
#' from large losses, each conditioned on exceeding its reporting threshold
#' (raised to the lowest threshold of the fit), with losses capped by a
#' policy limit treated as censored. Untruncated estimates are closed forms;
#' a truncated (last) piece is solved numerically and clamped to
#' `[0.001, 1000]`.
#'
#' @param losses Numeric vector of losses at or above `t` (`t[1]`).
#' @param t Threshold (`pareto_fit`) or thresholds (`piecewise_pareto_fit`).
#' @param reporting_thresholds Per-loss reporting thresholds, or `NULL`.
#' @param censored Logical vector, `TRUE` where a loss was capped, or `NULL`.
#' @param weights Per-loss weights, or `NULL`.
#' @param truncation Truncation point (of the last piece), or `NULL`.
#' @returns A [pareto] or [piecewise_pareto] object.
#' @export
#' @examples
#' pareto_fit(c(1500, 2500, 4000, 10000), 1000, censored = c(FALSE, FALSE, FALSE, TRUE))
#' piecewise_pareto_fit(c(1200, 1500, 2500, 6000), c(1000, 2000))
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
                                 weights = NULL, truncation = NULL) {
  ptr <- rust_result(PiecewisePareto$fit(
    as.double(losses), as.double(t), optional_arg(reporting_thresholds),
    optional_arg(censored), optional_arg(weights), truncation_arg(truncation)
  ))
  piecewise_pareto(ptr = ptr)
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
  package = "actuarialrs",
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
#' @returns A `log_affine_pareto` object, which inherits from [distribution].
#' @export
#' @examples
#' d <- log_affine_pareto(1e6, 1.5, delta = 0.5)
#' local_alpha(d, 2e6)
#' layer(d, 4e6, 1e6)
log_affine_pareto <- S7::new_class(
  "log_affine_pareto",
  parent = distribution,
  package = "actuarialrs",
  properties = list(
    ptr = S7::new_S3_class("LogAffinePareto"),
    t = S7::new_property(S7::class_double, getter = function(self) self@ptr$t()),
    alpha0 = S7::new_property(S7::class_double, getter = function(self) self@ptr$alpha0()),
    gamma = S7::new_property(S7::class_double, getter = function(self) self@ptr$gamma()),
    delta = S7::new_property(S7::class_double, getter = function(self) self@ptr$delta())
  ),
  constructor = function(t, alpha0, gamma = NULL, delta = NULL) {
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
  package = "actuarialrs",
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

for (cls in list(pareto, piecewise_pareto, log_affine_pareto, generalized_pareto)) {
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
  package = "actuarialrs",
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
