# Idiomatic R API: S7 classes and generics over the Rust objects.
#
# The #' comments are the package documentation. `cargo xtask r` turns them
# into man/*.Rd and NAMESPACE with roxygen2 and renders the pkgdown site
# (docs/architecture.md, "Documentation"). Do not edit man/ or NAMESPACE.

#' @importFrom stats quantile aggregate
#' @rawNamespace if (getRversion() < "4.3.0") importFrom(S7, "@")
NULL

# Rust errors come back as "extendr_error" condition objects (extendr's
# result_condition feature) rather than as panics. Raise them as ordinary R
# errors, attributed to the user-facing call: by default the function that
# called rust_result(). Methods pass `call` themselves, because their own
# frame's call names the method (or, under S7 dispatch, cannot be deparsed).
rust_result <- function(x, call = sys.call(-1)) {
  if (inherits(x, "extendr_error")) {
    stop(errorCondition(as.character(x$value), call = call))
  }
  x
}

#' Abstract parent of every distribution class
#'
#' `distribution` cannot be instantiated. Test membership with
#' `S7::S7_inherits(x, distribution)`.
#'
#' @export
distribution <- S7::new_class("distribution", package = "prospicio", abstract = TRUE)

#' Lognormal distribution
#'
#' Lognormal distribution with log-scale mean `meanlog` and sd `sdlog`,
#' parameterized as in [stats::dlnorm()]. Parameters are read-only
#' properties: `d@meanlog`, `d@sdlog`.
#'
#' Supports [mean()], [variance()], [std_dev()], [stats::quantile()],
#' [cdf()], [survival()], [draws()], [lev()], [stop_loss()], [layer()],
#' [layer_variance()], [layer_second_moment()] and [print()].
#'
#' @param meanlog Mean of `log(X)`; finite.
#' @param sdlog Standard deviation of `log(X)`; finite and positive.
#' @param ptr Internal: an existing object to wrap.
#' @returns A `lognormal` object, which inherits from [distribution].
#' @seealso [lognormal_from_mean_cv()] to parameterize by mean and CV.
#' @export
#' @examples
#' d <- lognormal(7, 0.5)
#' mean(d)
#' quantile(d, c(0.5, 0.995))
#' d@sdlog
lognormal <- S7::new_class(
  "lognormal",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("Lognormal"),
    meanlog = S7::new_property(S7::class_double, getter = function(self) self@ptr$meanlog()),
    sdlog = S7::new_property(S7::class_double, getter = function(self) self@ptr$sdlog())
  ),
  constructor = function(meanlog, sdlog, ptr = NULL) {
    if (is.null(ptr)) ptr <- rust_result(Lognormal$new(as.double(meanlog), as.double(sdlog)))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Lognormal distribution from its mean and coefficient of variation
#'
#' The way severity assumptions are usually stated.
#'
#' @param mean Mean of `X`; finite and positive.
#' @param cv Coefficient of variation of `X`; finite and positive.
#' @returns A [lognormal] object.
#' @export
#' @examples
#' d <- lognormal_from_mean_cv(1000, 0.5)
#' mean(d)
#' sqrt(variance(d))
lognormal_from_mean_cv <- function(mean, cv) {
  ptr <- rust_result(Lognormal$from_mean_cv(as.double(mean), as.double(cv)))
  lognormal(ptr$meanlog(), ptr$sdlog())
}

#' Distribution function
#'
#' `P(X <= q)` for each element of `q`.
#'
#' @param dist A [distribution].
#' @param q Numeric vector of quantiles.
#' @param ... Unused; for methods.
#' @returns Numeric vector the length of `q`.
#' @export
#' @examples
#' cdf(lognormal(7, 0.5), c(500, 1500))
cdf <- S7::new_generic("cdf", "dist", function(dist, q, ...) S7::S7_dispatch())

#' Variance of a distribution
#'
#' @inheritParams cdf
#' @returns A single number.
#' @export
#' @examples
#' variance(lognormal_from_mean_cv(1000, 0.5))
variance <- S7::new_generic("variance", "dist", function(dist, ...) S7::S7_dispatch())

#' Standard deviation of a distribution
#'
#' `sqrt(variance(dist))`, for any object with a [variance()] method. Named
#' after Rust's `std_dev` (Python's `std()`), so it does not mask
#' [stats::sd()].
#'
#' @inheritParams cdf
#' @returns A single number.
#' @export
#' @examples
#' std_dev(lognormal_from_mean_cv(1000, 0.5))
std_dev <- S7::new_generic("std_dev", "dist", function(dist, ...) S7::S7_dispatch())

S7::method(std_dev, S7::class_any) <- function(dist, ...) sqrt(variance(dist))

#' Reproducible random draws
#'
#' `n` draws from stream `stream` of the generator keyed by `seed`. The same
#' `(seed, stream)` gives the same draws in R, Python and Rust.
#'
#' @inheritParams cdf
#' @param n Number of draws.
#' @param seed Generator seed: a non-negative whole number below 2^53.
#' @param stream Stream id; distinct streams are independent.
#' @returns Numeric vector of length `n`.
#' @export
#' @examples
#' draws(lognormal(0, 1), 3, seed = 42, stream = 3)
draws <- S7::new_generic("draws", "dist", function(dist, n, seed, stream = 0, ...) {
  S7::S7_dispatch()
})

S7::method(mean, lognormal) <- function(x, ...) x@ptr$mean()

S7::method(quantile, lognormal) <- function(x, probs, ...) {
  # An S3 method: its call names the method, e.g. `quantile.prospicio::lognormal`.
  call <- sys.call()
  call[[1]] <- quote(quantile)
  rust_result(x@ptr$quantile(as.double(probs)), call)
}

S7::method(cdf, lognormal) <- function(dist, q, ...) dist@ptr$cdf(as.double(q))

S7::method(variance, lognormal) <- function(dist, ...) dist@ptr$variance()

S7::method(draws, lognormal) <- function(dist, n, seed, stream = 0, ...) {
  # Under S7 dispatch the generic's frame holds the user's call.
  call <- sys.call(sys.parent())
  rust_result(dist@ptr$sample(as.double(n), as.double(seed), as.double(stream)), call)
}

S7::method(print, lognormal) <- function(x, ...) {
  cat(sprintf("<lognormal> meanlog = %s, sdlog = %s\n",
              format(x@meanlog, digits = 15), format(x@sdlog, digits = 15)))
  invisible(x)
}

# The call a user made, for error messages: under S7 dispatch the generic's
# frame holds it.
s7_call <- function() sys.call(sys.parent(2))

#' Limited expected value, stop-loss and layer
#'
#' `lev(dist, limit)` is `E[min(X, limit)]`; `stop_loss(dist, retention)` is
#' `E[max(X - retention, 0)]`, computed directly so it stays accurate far into
#' the tail; `layer(dist, limit, attachment)` is the expected loss to the layer
#' `limit` xs `attachment`. Defined for [lognormal] and [grid_distribution]
#' severities and the Pareto family ([pareto], [piecewise_pareto],
#' [log_affine_pareto], [generalized_pareto]).
#'
#' @param dist A severity.
#' @param limit,retention Numeric vector.
#' @param attachment Layer attachment, a single number.
#' @param ... Unused; for methods.
#' @returns A numeric vector (`lev`, `stop_loss`) or a single number (`layer`).
#' @name severity
#' @examples
#' d <- lognormal(7, 0.5)
#' lev(d, c(500, 1500))
#' stop_loss(d, 1500)
#' layer(d, 1000, 500)
NULL

#' @rdname severity
#' @export
lev <- S7::new_generic("lev", "dist", function(dist, limit, ...) S7::S7_dispatch())

#' @rdname severity
#' @export
stop_loss <- S7::new_generic("stop_loss", "dist", function(dist, retention, ...) S7::S7_dispatch())

#' @rdname severity
#' @export
layer <- S7::new_generic("layer", "dist", function(dist, limit, attachment, ...) S7::S7_dispatch())

S7::method(lev, lognormal) <- function(dist, limit, ...) dist@ptr$lev(as.double(limit))
S7::method(stop_loss, lognormal) <- function(dist, retention, ...) dist@ptr$stop_loss(as.double(retention))
S7::method(layer, lognormal) <- function(dist, limit, attachment, ...) {
  dist@ptr$layer(as.double(limit), as.double(attachment))
}

#' Claim-count probability mass
#'
#' `P(N = k)` for each element of `k`.
#'
#' @param dist A [poisson_count], [negative_binomial_count] or [binomial_count].
#' @param k Non-negative whole numbers.
#' @param ... Unused; for methods.
#' @returns Numeric vector the length of `k`.
#' @export
#' @examples
#' pmf(poisson_count(3), 0:3)
pmf <- S7::new_generic("pmf", "dist", function(dist, k, ...) S7::S7_dispatch())

#' Poisson claim counts
#'
#' Claim counts with mean `lambda`. Supports [pmf()], [cdf()], [mean()],
#' [variance()], `quantile()` and [draws()].
#'
#' @param lambda Mean number of claims; finite and non-negative.
#' @returns A `poisson_count` object, which inherits from [distribution].
#' @export
#' @examples
#' n <- poisson_count(3)
#' pmf(n, 0:2)
#' quantile(n, 0.9)
poisson_count <- S7::new_class(
  "poisson_count",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("Poisson"),
    lambda = S7::new_property(S7::class_double, getter = function(self) self@ptr$lambda())
  ),
  constructor = function(lambda) {
    ptr <- rust_result(Poisson$new(as.double(lambda)))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Negative binomial claim counts
#'
#' Claim counts with mean `beta * r` and variance `beta * r * (1 + beta)`
#' (Klugman, Panjer and Willmot). This is `dnbinom(size = r, prob = 1 / (1 + beta))`.
#'
#' @param r Shape; finite and positive.
#' @param beta Scale; finite and positive.
#' @returns A `negative_binomial_count` object, which inherits from [distribution].
#' @seealso [negative_binomial_count_from_mean_variance()].
#' @export
#' @examples
#' n <- negative_binomial_count(2.5, 1.5)
#' mean(n)
#' variance(n)
negative_binomial_count <- S7::new_class(
  "negative_binomial_count",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("NegativeBinomial"),
    r = S7::new_property(S7::class_double, getter = function(self) self@ptr$r()),
    beta = S7::new_property(S7::class_double, getter = function(self) self@ptr$beta())
  ),
  constructor = function(r, beta) {
    ptr <- rust_result(NegativeBinomial$new(as.double(r), as.double(beta)))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Negative binomial from its mean and variance
#'
#' @param mean Mean number of claims; finite and positive.
#' @param variance Variance; must exceed the mean.
#' @returns A [negative_binomial_count] object.
#' @export
#' @examples
#' negative_binomial_count_from_mean_variance(10, 30)
negative_binomial_count_from_mean_variance <- function(mean, variance) {
  ptr <- rust_result(NegativeBinomial$from_mean_variance(as.double(mean), as.double(variance)))
  negative_binomial_count(ptr$r(), ptr$beta())
}

for (cls in list(poisson_count, negative_binomial_count)) {
  S7::method(pmf, cls) <- function(dist, k, ...) rust_result(dist@ptr$pmf(as.double(k)), s7_call())
  S7::method(cdf, cls) <- function(dist, q, ...) rust_result(dist@ptr$cdf(as.double(q)), s7_call())
  S7::method(mean, cls) <- function(x, ...) x@ptr$mean()
  S7::method(variance, cls) <- function(dist, ...) dist@ptr$variance()
  S7::method(quantile, cls) <- function(x, probs, ...) {
    call <- sys.call()
    call[[1]] <- quote(quantile)
    rust_result(x@ptr$quantile(as.double(probs)), call)
  }
  S7::method(draws, cls) <- function(dist, n, seed, stream = 0, ...) {
    rust_result(dist@ptr$sample(as.double(n), as.double(seed), as.double(stream)), s7_call())
  }
}

S7::method(print, poisson_count) <- function(x, ...) {
  cat(sprintf("<poisson_count> lambda = %s\n", format(x@lambda, digits = 15)))
  invisible(x)
}

S7::method(print, negative_binomial_count) <- function(x, ...) {
  cat(sprintf("<negative_binomial_count> r = %s, beta = %s\n",
              format(x@r, digits = 15), format(x@beta, digits = 15)))
  invisible(x)
}

#' Distribution on an evenly spaced grid
#'
#' Named `grid_distribution` so it does not mask `graphics::grid()`.
#'
#' Probabilities `probs` at `0, step, 2 * step, ...`: the discretized
#' representation that Panjer and FFT aggregation work on. Usually made with
#' [discretize()], which also records how much error the grid introduced in
#' `g@report`.
#'
#' Supports [mean()], [variance()], `quantile()`, [cdf()], [lev()],
#' [stop_loss()] and [layer()], all exact on the grid.
#'
#' @param step Grid step; finite and positive.
#' @param probs Non-negative probabilities summing to 1.
#' @returns A `grid_distribution` object, which inherits from [distribution]. Its `report`
#'   property is `NULL` for a grid built from probabilities, or a list with
#'   `method`, `step`, `points`, `tail_mass`, `source_mean`, `grid_mean` and
#'   `mean_error`.
#' @export
#' @examples
#' g <- grid_distribution(1, c(0.5, 0.25, 0.25))
#' mean(g)
#' cdf(g, 1)
grid_distribution <- S7::new_class(
  "grid_distribution",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("Grid"),
    step = S7::new_property(S7::class_double, getter = function(self) self@ptr$step()),
    probs = S7::new_property(S7::class_double, getter = function(self) self@ptr$probs()),
    report = S7::new_property(S7::class_any, getter = function(self) self@ptr$report())
  ),
  constructor = function(step, probs, ptr = NULL) {
    if (is.null(ptr)) ptr <- rust_result(Grid$new(as.double(step), as.double(probs)))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Discretize a severity onto a grid
#'
#' `"local_moment"` matches the mean on every cell (the grid mean is the
#' limited mean up to the last point); `"rounding"` moves each loss to the
#' nearest point; `"lower"` moves each cell's mass to its left end, a
#' stochastic lower bound. Mass above the last point is lumped onto it and
#' reported in `g@report$tail_mass`.
#'
#' @param dist A [lognormal], a [grid_distribution] or a Pareto-family severity.
#' @param step Grid step.
#' @param points Number of grid points.
#' @param method One of `"local_moment"`, `"rounding"`, `"lower"`.
#' @returns A [grid_distribution] whose `report` property describes the error introduced.
#' @export
#' @examples
#' g <- discretize(lognormal(7, 0.5), step = 100, points = 200)
#' g@report$tail_mass
#' mean(g)
discretize <- function(dist, step, points, method = c("local_moment", "rounding", "lower")) {
  method <- match.arg(method)
  ptr <- rust_result(Grid$discretize(dist@ptr, as.double(step), as.double(points), method))
  grid_distribution(ptr = ptr)
}

#' Transform a grid distribution
#'
#' The distribution of `f(X)` on the same step. Each point's mass moves to
#' `f(x)`; a value between two points is split between them so its mean is
#' kept. The mean is therefore always exact, and the whole distribution is
#' exact when every value lands on a point.
#'
#' @param x A [grid_distribution].
#' @param f A function of one loss returning a finite, non-negative number;
#'   called once per point with mass.
#' @returns A list with `grid` (a [grid_distribution]) and `on_points`
#'   (whether every value landed on a grid point).
#' @export
#' @examples
#' x <- grid_distribution(1, c(0.2, 0.3, 0.3, 0.2))
#' r <- map_grid(x, function(v) min(max(v - 1, 0), 1))
#' r$grid@probs
#' r$on_points
map_grid <- function(x, f) {
  r <- rust_result(x@ptr$map(f))
  list(grid = grid_distribution(ptr = r$grid), on_points = r$on_points)
}

S7::method(mean, grid_distribution) <- function(x, ...) x@ptr$mean()
S7::method(variance, grid_distribution) <- function(dist, ...) dist@ptr$variance()
S7::method(cdf, grid_distribution) <- function(dist, q, ...) dist@ptr$cdf(as.double(q))
S7::method(quantile, grid_distribution) <- function(x, probs, ...) {
  call <- sys.call()
  call[[1]] <- quote(quantile)
  rust_result(x@ptr$quantile(as.double(probs)), call)
}
S7::method(lev, grid_distribution) <- function(dist, limit, ...) dist@ptr$lev(as.double(limit))
S7::method(stop_loss, grid_distribution) <- function(dist, retention, ...) dist@ptr$stop_loss(as.double(retention))
S7::method(layer, grid_distribution) <- function(dist, limit, attachment, ...) {
  dist@ptr$layer(as.double(limit), as.double(attachment))
}
#' Remove FFT round-off from a grid
#'
#' The grid with every probability below `eps` set to 0: the round-off an
#' FFT leaves where the true mass is zero, as `aggregate`'s `remove_fuzz`.
#' [tower_on_grid()]'s ceded and net grids already have it removed.
#'
#' @param grid A [grid_distribution].
#' @param eps The threshold, machine epsilon by default.
#' @returns A `grid_distribution`.
#' @export
#' @examples
#' remove_fuzz(grid_distribution(1, c(0.5, 0.5 - 1e-20, 1e-20)))@probs
remove_fuzz <- function(grid, eps = .Machine$double.eps) {
  p <- grid@probs
  grid_distribution(grid@step, ifelse(p < eps, 0, p))
}

S7::method(print, grid_distribution) <- function(x, ...) {
  cat(sprintf("<grid_distribution> step = %s, %d points\n", format(x@step, digits = 15), length(x@probs)))
  invisible(x)
}

#' Value at risk and tail value at risk
#'
#' `VaR(dist, p)` is the inverted empirical distribution function;
#' `TVaR(dist, p)` is the mean of the worst `1 - p`, splitting the draw at the
#' VaR so it is coherent and continuous in `p`. For a
#' [predictive_distribution] both describe the total over all components.
#' For a parametric severity or a [grid_distribution], `VaR()` is the lower
#' quantile, as `quantile()`, and a severity's `TVaR()` is
#' `VaR(p) + E[(X - VaR(p))+] / (1 - p)`, infinite where the mean is.
#' For a [sampled], `method = "linear"` picks the other convention: `VaR()`
#' is then `quantile(x, p, type = 7)` (numpy's default) and `TVaR()` the
#' mean of that quantile function over `[p, 1]`; the defaults are
#' `"inverted_cdf"` and `"exact"`.
#'
#' @param dist A [sampled], [predictive_distribution], [pot_tail],
#'   [grid_distribution], or a severity ([lognormal], [pareto] and the
#'   other parametric families).
#' @param p Probabilities in `[0, 1]` (`[0, 1)` for a severity's `TVaR()`).
#' @param ... For a [sampled], `method`: `"inverted_cdf"` or `"linear"` for
#'   `VaR()`, `"exact"` or `"linear"` for `TVaR()`. Otherwise unused.
#' @returns Numeric vector the length of `p`.
#' @name risk_measures
#' @examples
#' s <- sampled(c(1, 2, 3, 4))
#' VaR(s, 0.5)
#' TVaR(s, 0.5)
NULL

#' @rdname risk_measures
#' @export
VaR <- S7::new_generic("VaR", "dist", function(dist, p, ...) S7::S7_dispatch())

#' @rdname risk_measures
#' @export
TVaR <- S7::new_generic("TVaR", "dist", function(dist, p, ...) S7::S7_dispatch())

# A severity's TVaR from its quantile and stop loss, VaR(p) + E[(X - VaR(p))+] / (1 - p).
severity_tvar <- function(ptr, p, call) {
  p <- as.double(p)
  if (any(is.na(p) | p < 0 | p >= 1)) stop("p must be in [0, 1) for a severity's TVaR", call. = FALSE)
  if (is.infinite(ptr$mean())) return(rep(Inf, length(p)))
  q <- rust_result(ptr$quantile(p), call)
  q + vapply(q, function(v) ptr$stop_loss(v), 0) / (1 - p)
}

S7::method(VaR, lognormal) <- function(dist, p, ...) rust_result(dist@ptr$quantile(as.double(p)), s7_call())
S7::method(TVaR, lognormal) <- function(dist, p, ...) severity_tvar(dist@ptr, p, s7_call())
S7::method(VaR, grid_distribution) <- function(dist, p, ...) {
  rust_result(dist@ptr$quantile(as.double(p)), s7_call())
}

#' Third moment of a layer
#'
#' `E[min(max(X - attachment, 0), limit)^3]` for any severity, by quadrature
#' of its survival function (relative accuracy about 1e-10): `limit = Inf`
#' for the unlimited layer, `attachment = 0` for the limited third moment
#' `E[min(X, limit)^3]`. Limits and attachments are recycled.
#'
#' @param dist A severity.
#' @param limit,attachment Layer limits and attachments.
#' @returns A numeric vector.
#' @export
#' @examples
#' layer_third_moment(gamma_distribution(1, 2), c(3, Inf), 0)
layer_third_moment <- function(dist, limit, attachment = 0) {
  rust_result(layer_third_moment_rust(dist@ptr, as.double(limit), as.double(attachment)))
}

#' Level at which TVaR reaches VaR
#'
#' The level `t` with `TVaR(t) = VaR(p)`, the lower VaR, as `aggregate`'s
#' `tvar_threshold`. TVaR rises continuously from the mean at `t = 0`, so
#' `t <= p`; `p = 0` gives 0.
#'
#' @param x A [grid_distribution] or a [capital_portfolio] (its total).
#' @param p Probabilities in `[0, 1]`.
#' @param ... Unused; for methods.
#' @returns A numeric vector of levels.
#' @export
#' @examples
#' g <- grid_distribution(1, c(0.4, 0.3, 0.2, 0.1))
#' t <- tvar_threshold(g, 0.85)
#' TVaR(g, t) # quantile(g, 0.85)
tvar_threshold <- S7::new_generic("tvar_threshold", "x", function(x, p, ...) S7::S7_dispatch())

S7::method(TVaR, grid_distribution) <- function(dist, p, ...) {
  rust_result(tvar_weighted_rust(grid_points(dist), dist@probs, as.double(p)), s7_call())
}

S7::method(tvar_threshold, grid_distribution) <- function(x, p, ...) {
  rust_result(tvar_threshold_rust(grid_points(x), x@probs, as.double(p)), s7_call())
}

grid_points <- function(g) (seq_along(g@probs) - 1) * g@step

#' Distribution of equally weighted draws
#'
#' Every method describes the draws themselves: [variance()] divides by `n`
#' and `quantile()` inverts the empirical distribution function (R `type = 1`).
#'
#' @param draws Non-empty numeric vector of finite values.
#' @returns A `sampled` object, which inherits from [distribution].
#' @export
#' @examples
#' s <- sampled(c(10, 20, 30, 40))
#' quantile(s, 0.5)
#' TVaR(s, 0.5)
sampled <- S7::new_class(
  "sampled",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("Sampled"),
    draws = S7::new_property(S7::class_double, getter = function(self) self@ptr$draws())
  ),
  constructor = function(draws, ptr = NULL) {
    if (is.null(ptr)) ptr <- rust_result(Sampled$new(as.double(draws)))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

S7::method(mean, sampled) <- function(x, ...) x@ptr$mean()
S7::method(variance, sampled) <- function(dist, ...) dist@ptr$variance()
S7::method(cdf, sampled) <- function(dist, q, ...) dist@ptr$cdf(as.double(q))
S7::method(quantile, sampled) <- function(x, probs, ...) {
  call <- sys.call()
  call[[1]] <- quote(quantile)
  rust_result(x@ptr$quantile(as.double(probs)), call)
}
S7::method(VaR, sampled) <- function(dist, p, method = "inverted_cdf", ...) {
  rust_result(dist@ptr$var(as.double(p), as.character(method)), s7_call())
}
S7::method(TVaR, sampled) <- function(dist, p, method = "exact", ...) {
  rust_result(dist@ptr$tvar(as.double(p), as.character(method)), s7_call())
}
S7::method(print, sampled) <- function(x, ...) {
  cat(sprintf("<sampled> %d draws, mean = %s\n", length(x@draws), format(mean(x), digits = 6)))
  invisible(x)
}

#' Joint predictive distribution
#'
#' The result every model returns: a matrix of draws with one row per
#' simulation and one column per component, plus one key per component.
#' Components keep their dependence, so the total's quantiles come from row
#' sums. [mean()], [variance()], `quantile()`, [VaR()] and [TVaR()] describe the
#' total; use [marginal()] for one component and `aggregate()` to sum over
#' dimensions.
#'
#' @param draws Numeric matrix, `n_sims` rows by `n_components` columns.
#' @param keys Data frame with one column per dimension (character or whole
#'   numbers) and one row per component.
#' @returns A `predictive_distribution` object, which inherits from
#'   [distribution].
#' @export
#' @examples
#' pd <- predictive_distribution(
#'   matrix(c(0, 0, 0, 100, 0, 0, 100, 0), ncol = 2),
#'   data.frame(line = c("A", "B"))
#' )
#' VaR(pd, 0.75)
#' VaR(marginal(pd, list(line = "A")), 0.75)
predictive_distribution <- S7::new_class(
  "predictive_distribution",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("PredictiveDistribution"),
    dims = S7::new_property(S7::class_character, getter = function(self) self@ptr$dims()),
    keys = S7::new_property(S7::class_data.frame, getter = function(self) {
      as.data.frame(self@ptr$keys(), stringsAsFactors = FALSE)
    }),
    n_sims = S7::new_property(S7::class_double, getter = function(self) self@ptr$n_sims())
  ),
  constructor = function(draws, keys, ptr = NULL) {
    if (is.null(ptr)) {
      draws <- as.matrix(draws)
      keys <- as.data.frame(keys, stringsAsFactors = FALSE)
      ptr <- rust_result(PredictiveDistribution$new(
        names(keys), as.list(keys), as.double(draws), as.double(nrow(draws))
      ))
    }
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' One component of a predictive distribution
#'
#' @param key A named list (or vector) with one value per dimension. An origin
#'   period is named by its label (`"2021"`, `2021`, `"2021Q3"`).
#' @param ... Unused; for methods.
#' @returns A [sampled] object, or `NULL` when no component has this key.
#' @export
#' @examples
#' pd <- predictive_distribution(matrix(1:6, ncol = 2), data.frame(year = c(2023, 2024)))
#' marginal(pd, list(year = 2024))
marginal <- S7::new_generic("marginal", "dist", function(dist, key, ...) S7::S7_dispatch())

#' Total over all components
#'
#' @inheritParams marginal
#' @returns A [sampled] object with one draw per simulation.
#' @export
#' @examples
#' pd <- predictive_distribution(matrix(1:6, ncol = 2), data.frame(year = c(2023, 2024)))
#' total(pd)@draws
total <- S7::new_generic("total", "dist", function(dist, ...) S7::S7_dispatch())

#' Draw matrix of a predictive distribution
#'
#' @inheritParams marginal
#' @returns Numeric matrix with one row per simulation and one column per
#'   component.
#' @export
#' @examples
#' pd <- predictive_distribution(matrix(1:6, ncol = 2), data.frame(year = c(2023, 2024)))
#' draw_matrix(pd)
draw_matrix <- S7::new_generic("draw_matrix", "dist", function(dist, ...) S7::S7_dispatch())

#' Where a result came from
#'
#' `stream_scheme` names how simulations map to random-number streams;
#' `samplers` is a named list of sampler ids by family (for example
#' `list(gamma = "marsaglia-tsang/2026-10")`): the sampler table of the
#' build that made the draws, so every sampler they may have used. A family
#' not listed uses its first sampler (inverse transform for a distribution,
#' the documented method for a copula's frailty). A result replays only
#' with the same seed, stream scheme and samplers. `samplers` is `NULL`
#' when not recorded: a result without draws, one computed from draws made
#' elsewhere, or one made by a build from before samplers were recorded
#' (2026-10-08).
#'
#' @inheritParams marginal
#' @returns A list with `model`, `parameters`, `seed`, `stream_scheme`,
#'   `samplers`, `versions` and `input_hash`.
#' @export
#' @examples
#' pd <- predictive_distribution(matrix(1:6, ncol = 2), data.frame(year = c(2023, 2024)))
#' provenance(pd)$model
provenance <- S7::new_generic("provenance", "dist", function(dist, ...) S7::S7_dispatch())

S7::method(marginal, predictive_distribution) <- function(dist, key, ...) {
  ptr <- rust_result(dist@ptr$marginal(as.list(key)), s7_call())
  if (is.null(ptr)) NULL else sampled(ptr = ptr)
}
S7::method(total, predictive_distribution) <- function(dist, ...) sampled(ptr = dist@ptr$total())
S7::method(draw_matrix, predictive_distribution) <- function(dist, ...) {
  matrix(dist@ptr$draw_matrix(), nrow = dist@ptr$n_sims())
}
S7::method(provenance, predictive_distribution) <- function(dist, ...) dist@ptr$provenance()
S7::method(aggregate, predictive_distribution) <- function(x, keep = character(), ...) {
  call <- sys.call()
  call[[1]] <- quote(aggregate)
  predictive_distribution(ptr = rust_result(x@ptr$aggregate(as.character(keep)), call))
}
S7::method(mean, predictive_distribution) <- function(x, ...) x@ptr$mean()
S7::method(variance, predictive_distribution) <- function(dist, ...) dist@ptr$variance()
S7::method(quantile, predictive_distribution) <- function(x, probs, ...) {
  call <- sys.call()
  call[[1]] <- quote(quantile)
  rust_result(x@ptr$quantile(as.double(probs)), call)
}
S7::method(VaR, predictive_distribution) <- function(dist, p, ...) {
  rust_result(dist@ptr$var(as.double(p)), s7_call())
}
S7::method(TVaR, predictive_distribution) <- function(dist, p, ...) {
  rust_result(dist@ptr$tvar(as.double(p)), s7_call())
}
S7::method(print, predictive_distribution) <- function(x, ...) {
  cat(sprintf("<predictive_distribution> dims = %s, %d simulations x %d components\n",
              paste(x@dims, collapse = ", "), as.integer(x@n_sims),
              as.integer(x@ptr$n_components())))
  invisible(x)
}

.onLoad <- function(libname, pkgname) {
  S7::methods_register()
}

#' Join predictive distributions into a portfolio
#'
#' `join_predictive()` puts distributions of different models side by side
#' (a reserve bootstrap by origin, premium risk by line, a tower's net),
#' with a new leading key column `dim` holding each part's name, followed by
#' the union of the parts' key columns (`""` where a part lacks one).
#' Simulation `i` of the result is simulation `i` of every part.
#' `reorder_groups()` then sets the dependence between the groups of `dim`
#' by Iman-Conover on the group totals, moving each group's simulations as
#' whole rows: every group keeps its distribution and internal joint
#' structure. Use [aggregate()], [VaR()], [TVaR()] and [capital_allocation()]
#' on the result. `group_spearman()` is Spearman's rho between the group
#' totals (ties at their average rank, as `cor(method = "spearman")`), to
#' check what `reorder_groups()` achieved.
#' `reorder_groups_by_copula()` sets the dependence by a [copula] instead:
#' `n` draws of uniforms (draw `i` on stream `i - 1` of `seed`, as
#' [copula_sample()]), and each group's simulations, whole rows sorted by the
#' group's total, take the ranks of its column. The group totals then have
#' the copula's rank pattern, so a t, Gumbel or Clayton copula gives its
#' joint tails to results already simulated.
#'
#' @param parts A named list of [predictive_distribution]s with the same
#'   number of simulations.
#' @param dim Name of the new key column.
#' @param same_simulations `FALSE`: the parts were simulated separately, and
#'   two with the same seed (which would share random numbers) are refused.
#'   `TRUE`: the parts come from the same scenarios (a cover applied to a
#'   reserve) and keep their pairing.
#' @param x A [predictive_distribution].
#' @param correlation Correlation matrix, one row and column per group in
#'   order of first appearance (`unique(x@keys[[dim]])`), or a long table:
#'   a data frame whose first two columns name a pair of groups and whose
#'   third holds its value, each pair of distinct groups once.
#' @param copula A [copula] with one dimension per group, in order of first
#'   appearance.
#' @param seed Seed, a whole number.
#' @param kind `"pearson"` (the default: the normal scores' correlation, so
#'   Spearman's rho comes out near `(6 / pi) asin(r / 2)`) or `"spearman"`
#'   (a Spearman target, converted to `2 sin(pi rho / 6)` first).
#' @returns A [predictive_distribution]; `group_spearman()` a matrix with the
#'   groups as dimnames.
#' @name portfolio
#' @examples
#' a <- predictive_distribution(cbind(c(10, 12), c(20, 25)), data.frame(origin = c(2023, 2024)))
#' b <- predictive_distribution(matrix(c(50, 40), ncol = 1), data.frame(lob = "motor"))
#' p <- join_predictive(list(reserve = a, premium = b), "risk")
#' p@keys
#' n <- 4000
#' pd <- predictive_distribution(cbind(0:(n - 1), ((0:(n - 1)) * 7919) %% n),
#'                               data.frame(seg = c("a", "b")))
#' rho <- data.frame(from = "a", to = "b", rho = 0.6)
#' group_spearman(reorder_groups(pd, "seg", rho, seed = 5, kind = "spearman"), "seg")
#' # Clayton with theta = 2: Spearman's rho near 0.68, more joint low totals.
#' group_spearman(reorder_groups_by_copula(pd, "seg", archimedean_copula("clayton", 2), 5), "seg")
NULL

#' One part of a predictive distribution
#'
#' The components whose value on dimension `dim` is one of `keys`, with every
#' dimension and the draws' joint structure kept. Take one part of a result
#' this way before joining it to another: `select_components(apply_tower(...),
#' "kind", "ceded")` is a treaty's cession, ready to add to a book with
#' [join_predictive()]. Aggregating over `kind` instead adds gross, ceded and
#' net together.
#'
#' @param dist A [predictive_distribution].
#' @param dim The dimension's name.
#' @param keys Values on that dimension (character or whole numbers).
#' @returns A [predictive_distribution].
#' @export
#' @examples
#' pd <- predictive_distribution(matrix(c(10, 4, 3, 0, 7, 4), ncol = 3),
#'   data.frame(kind = c("gross", "ceded", "net"), layer = c("ground_up", "5x5", "retained")))
#' mean(select_components(pd, "kind", "ceded"))
select_components <- function(dist, dim, keys) {
  predictive_distribution(ptr = rust_result(dist@ptr$select(as.character(dim), keys)))
}

#' Rescale components to new totals
#'
#' Every simulation's components rescaled pro rata to a new total: component
#' `k` in simulation `s` becomes `c[s, k] * totals[s] / sum(c[s, ])`
#' (unchanged when the components sum to 0). This carries a total computed
#' from the components, a loss-ratio cap over several classes or a stop loss
#' on a mix of perils, back onto them in proportion to their shares; it is
#' not a capital allocation. The keys and dimensions are kept.
#'
#' @param dist A [predictive_distribution].
#' @param totals One total per simulation.
#' @returns A [predictive_distribution].
#' @export
#' @examples
#' pd <- predictive_distribution(matrix(c(6, 3, 9, 1), ncol = 2), data.frame(class = c("motor", "home")))
#' draw_matrix(scale_to_totals(pd, pmin(rowSums(draw_matrix(pd)), 10)))
scale_to_totals <- function(dist, totals) {
  predictive_distribution(ptr = rust_result(dist@ptr$scale_to_totals(as.double(totals))))
}

#' @rdname portfolio
#' @export
join_predictive <- function(parts, dim, same_simulations = FALSE) {
  if (is.null(names(parts)) || any(names(parts) == "")) stop("parts must be a named list")
  ptr <- join_rust(lapply(parts, function(p) p@ptr), names(parts), dim, isTRUE(same_simulations))
  predictive_distribution(ptr = rust_result(ptr))
}

#' @rdname portfolio
#' @export
reorder_groups <- function(x, dim, correlation, seed, kind = "pearson") {
  if (is.data.frame(correlation)) {
    if (ncol(correlation) != 3) stop("a long correlation table has three columns", call. = FALSE)
    flat <- rust_result(correlation_from_pairs_rust(
      x@ptr, dim, correlation[[1]], correlation[[2]], as.double(correlation[[3]])
    ))
  } else {
    flat <- as.double(t(as.matrix(correlation)))
  }
  predictive_distribution(ptr = rust_result(
    reorder_groups_rust(x@ptr, dim, flat, as.double(seed), as.character(kind))
  ))
}

#' @rdname portfolio
#' @export
reorder_groups_by_copula <- function(x, dim, copula, seed) {
  predictive_distribution(ptr = rust_result(
    reorder_groups_by_copula_rust(x@ptr, dim, copula@ptr, as.double(seed))
  ))
}

#' @rdname portfolio
#' @export
group_spearman <- function(x, dim) {
  groups <- as.character(unique(x@keys[[dim]]))
  m <- rust_result(group_spearman_rust(x@ptr, dim))
  matrix(m, length(groups), byrow = TRUE, dimnames = list(groups, groups))
}

#' Blend predictive distributions
#'
#' Simulation `i` of the result is simulation `i` of model `k`, with `k`
#' drawn with probability `weights[k]` from stream `i` of `seed`. Rows stay
#' whole, so sums across components remain coherent. Use weights from
#' [stacking_weights()] or [pseudo_bma_weights()].
#'
#' With a matrix of weights (one row per component, one column per model,
#' as [hierarchical_stacking()] gives them) each component draws its model
#' from the simulation's common uniform against its own weights, so
#' components with equal weights take the same model.
#'
#' @param models A list of [predictive_distribution]s with the same keys
#'   and number of simulations.
#' @param weights Non-negative weights, one per model, or a matrix with one
#'   row per component; normalized.
#' @param seed Seed, a whole number.
#' @returns A [predictive_distribution].
#' @export
#' @examples
#' a <- predictive_distribution(matrix(0, 1000, 1), data.frame(lob = "x"))
#' b <- predictive_distribution(matrix(1, 1000, 1), data.frame(lob = "x"))
#' mean(blend_predictive(list(a, b), c(0.25, 0.75), seed = 7))
blend_predictive <- function(models, weights, seed) {
  ptrs <- lapply(models, function(m) m@ptr)
  ptr <- if (is.matrix(weights)) {
    blend_by_component_rust(ptrs, as.double(weights), as.double(seed))
  } else {
    blend_rust(ptrs, as.double(weights), as.double(seed))
  }
  predictive_distribution(ptr = rust_result(ptr))
}

#' More claim counts
#'
#' Claim counts beyond the Poisson, negative binomial and binomial, as in
#' `aggregate`'s `zm`, `zt`, `logarithmic`, `mixed`, `neymana` and
#' `dfreq`:
#'
#' - `zero_modified_count()`: `base` with `P(N = 0) = p0`;
#'   `zero_truncated_count()` is `p0 = 0`.
#' - `logarithmic_count()`: `P(N = k) = -p^k / (k log(1 - p))` on `1, 2, ...`.
#' - `mixed_poisson_count()`: Poisson with mean `mean * theta`, where
#'   `theta = shift + (1 - shift) G` has mean 1 and coefficient of variation
#'   `cv`, and `G` is gamma (negative binomial, or Delaporte with a shift)
#'   or inverse Gaussian (Poisson-inverse Gaussian, or its shifted version).
#'   Variance `mean + mean^2 cv^2`. With no shift, as `aggregate`'s, `G`
#'   may also be `"beta"`, the Poisson mean `max` times a beta variable
#'   (`mixed beta cv max`), `"sichel"`, the generalized inverse Gaussian
#'   with the given `shape` (`mixed sichel cv shape`; `-0.5` is the
#'   inverse Gaussian) or, with `skewness` instead, the shape that gives
#'   it, or `"sichel_gamma"` and `"sichel_ig"`, the generalized inverse
#'   Gaussian with the skewness of the Delaporte's or the shifted inverse
#'   Gaussian's mixing with fixed part `f` (`mixed sichel.gamma cv f`,
#'   `mixed sichel.ig cv f`). `gig_parameters()` gives the generalized
#'   inverse Gaussian a Sichel mixing solved, `c(shape, omega, eta)`
#'   (`aggregate`'s `lam`, `mu / beta` and `mu`), or `NULL`.
#' - `compound_poisson_count()`: a Poisson(`rate`) number of clusters of
#'   `secondary` claims (Poisson secondaries give the Neyman type A).
#' - `empirical_count()`: `P(N = k) = probs[k + 1]`.
#' - `renewal_count()`: the number of claims by `horizon` when the waits
#'   between claims are independent draws from `wait` (a renewal, or
#'   Sparre Andersen, count; `aggregate`'s `years ... wait ...`).
#'   Exponential waits give the Poisson. The wait is a distribution, or
#'   `list(values, probs)` of waits and their probabilities, which may sum
#'   to less than 1 (the process then ends). Waits of zero arrive in
#'   clusters. The wait grid is `aggregate`'s unless `step` and `points`
#'   are given; `closed` counts waits summing to exactly `horizon` in full.
#' - `fixed_count()`: `N = n`; `bernoulli_count()`: one claim with
#'   probability `p` in `[0, 1)`; `geometric_count()`: `P(N = k) = q (1 - q)^k`
#'   on `0, 1, ...` with `q = 1 / (1 + mean)`, a negative binomial with
#'   shape 1 (`aggregate`'s `fixed`, `bernoulli` and `geometric`).
#'
#' Every aggregation function takes them. Zero-modified and logarithmic
#' counts work with Panjer's recursion; the others need `method = "fft"`.
#'
#' @param base,secondary A claim count.
#' @param p0 `P(N = 0)`, in `[0, 1)`.
#' @param p The logarithmic parameter, in `(0, 1)`, or the Bernoulli
#'   probability, in `[0, 1)`.
#' @param n The fixed number of claims.
#' @param mean,cv,shift See above.
#' @param mixing `"gamma"`, `"inverse_gaussian"`, `"beta"`, `"sichel"`,
#'   `"sichel_gamma"` or `"sichel_ig"`.
#' @param max The beta mixing's largest Poisson mean, above
#'   `mean * (1 + cv^2)`.
#' @param shape,skewness The Sichel shape, or the skewness that sets it.
#' @param f The fixed part matched by `"sichel_gamma"` and `"sichel_ig"`.
#' @param x A claim count.
#' @param rate The Poisson rate of clusters.
#' @param probs Probabilities of `0, 1, ...`, summing to 1.
#' @param wait A distribution, or `list(values, probs)`.
#' @param horizon The period, in the wait's units.
#' @param step,points The wait grid; `NULL` for `aggregate`'s.
#' @param closed Whether waits summing to `horizon` count in full.
#' @param ptr Internal.
#' @returns A `claim_count_dist`, which inherits from [distribution]: use
#'   it with [pmf()], [cdf()], [mean()], [variance()], [quantile()] and
#'   [draws()].
#' @export
#' @examples
#' zt <- zero_truncated_count(poisson_count(2))
#' pmf(zt, 0:3)
#' pig <- mixed_poisson_count(10, 0.5, mixing = "inverse_gaussian")
#' variance(pig) # 10 + 100 * 0.25
#' compound_distribution(pig, grid_distribution(1, c(0.2, 0.5, 0.3)), 200, method = "fft")
#' pmf(renewal_count(gamma_distribution(2, 1), 5), 0) # the first wait beyond 5: 6 exp(-5)
claim_count_dist <- S7::new_class(
  "claim_count_dist",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("ClaimCount"),
    kind = S7::new_property(S7::class_character, getter = function(self) self@ptr$kind())
  ),
  constructor = function(ptr) S7::new_object(S7::S7_object(), ptr = ptr)
)

S7::method(pmf, claim_count_dist) <- function(dist, k, ...) rust_result(dist@ptr$pmf(as.double(k)), s7_call())
S7::method(cdf, claim_count_dist) <- function(dist, q, ...) rust_result(dist@ptr$cdf(as.double(q)), s7_call())
S7::method(mean, claim_count_dist) <- function(x, ...) x@ptr$mean()
S7::method(variance, claim_count_dist) <- function(dist, ...) dist@ptr$variance()
S7::method(quantile, claim_count_dist) <- function(x, probs, ...) {
  call <- sys.call()
  call[[1]] <- quote(quantile)
  rust_result(x@ptr$quantile(as.double(probs)), call)
}
S7::method(draws, claim_count_dist) <- function(dist, n, seed, stream = 0, ...) {
  rust_result(dist@ptr$sample(as.double(n), as.double(seed), as.double(stream)), s7_call())
}
S7::method(print, claim_count_dist) <- function(x, ...) {
  cat(sprintf("<claim_count_dist> %s, mean %s\n", x@kind, format(mean(x))))
  invisible(x)
}

#' @rdname claim_count_dist
#' @export
zero_modified_count <- function(base, p0) {
  claim_count_dist(rust_result(ClaimCount$zero_modified(base@ptr, as.double(p0))))
}

#' @rdname claim_count_dist
#' @export
zero_truncated_count <- function(base) zero_modified_count(base, 0)

#' @rdname claim_count_dist
#' @export
logarithmic_count <- function(p) claim_count_dist(rust_result(ClaimCount$logarithmic(as.double(p))))

#' @rdname claim_count_dist
#' @export
mixed_poisson_count <- function(mean, cv,
                                mixing = c("gamma", "inverse_gaussian", "beta", "sichel",
                                           "sichel_gamma", "sichel_ig"),
                                shift = 0, max = NULL, shape = NULL, skewness = NULL, f = NULL) {
  mixing <- match.arg(mixing)
  need <- function(v, name) {
    if (is.null(v)) stop(sprintf("mixing \"%s\" needs %s", mixing, name), call. = FALSE)
    as.double(v)
  }
  param <- 0
  if (mixing == "beta") {
    param <- need(max, "max")
  } else if (mixing == "sichel") {
    if (is.null(shape) == is.null(skewness)) {
      stop("mixing \"sichel\" needs one of shape or skewness", call. = FALSE)
    }
    if (is.null(shape)) {
      mixing <- "sichel_skewness"
      param <- as.double(skewness)
    } else {
      param <- as.double(shape)
    }
  } else if (mixing %in% c("sichel_gamma", "sichel_ig")) {
    param <- need(f, "f")
  }
  claim_count_dist(rust_result(ClaimCount$mixed_poisson(
    as.double(mean), as.double(cv), mixing, as.double(shift), param
  )))
}

#' @rdname claim_count_dist
#' @export
gig_parameters <- function(x) {
  g <- x@ptr$gig()
  if (length(g) == 0) NULL else c(shape = g[[1]], omega = g[[2]], eta = g[[3]])
}

#' @rdname claim_count_dist
#' @export
compound_poisson_count <- function(rate, secondary) {
  claim_count_dist(rust_result(ClaimCount$compound_poisson(as.double(rate), secondary@ptr)))
}

#' @rdname claim_count_dist
#' @export
empirical_count <- function(probs) claim_count_dist(rust_result(ClaimCount$empirical(as.double(probs))))

#' @rdname claim_count_dist
#' @export
renewal_count <- function(wait, horizon, step = NULL, points = NULL, closed = FALSE) {
  if (is.null(step) != is.null(points)) stop("give both step and points, or neither")
  grid <- if (is.null(step)) c(0, 0) else c(as.double(step), as.double(points))
  ptr <- if (is.list(wait) && !S7::S7_inherits(wait)) {
    ClaimCount$renewal(
      NULL, as.double(wait[[1]]), as.double(wait[[2]]), as.double(horizon),
      grid[1], grid[2], isTRUE(closed)
    )
  } else {
    ClaimCount$renewal(wait@ptr, double(), double(), as.double(horizon), grid[1], grid[2], isTRUE(closed))
  }
  claim_count_dist(rust_result(ptr))
}

#' @rdname claim_count_dist
#' @export
fixed_count <- function(n) claim_count_dist(rust_result(ClaimCount$fixed(as.double(n))))

#' @rdname claim_count_dist
#' @export
bernoulli_count <- function(p) claim_count_dist(rust_result(ClaimCount$bernoulli(as.double(p))))

#' @rdname claim_count_dist
#' @export
geometric_count <- function(mean) claim_count_dist(rust_result(ClaimCount$geometric(as.double(mean))))
