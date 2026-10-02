#' @include distributions.R
NULL

#' Distortion risk measure
#'
#' A concave distortion `g` of the survival function, giving the coherent
#' risk measure `rho(X) = integral of g(S(x)) dx`. Every kind has a parameter
#' value that gives the mean.
#'
#' | `kind` | `g(s)` | `param` |
#' |---|---|---|
#' | `"tvar"` | `min(s / (1 - p), 1)` | `p` in `[0, 1]` |
#' | `"wang"` | `pnorm(qnorm(s) + lambda)` | `lambda >= 0` |
#' | `"proportional_hazard"` | `s^rho` | `rho` in `(0, 1]` |
#' | `"dual_power"` | `1 - (1 - s)^beta` | `beta >= 1` |
#'
#' @param kind One of `"tvar"`, `"wang"`, `"proportional_hazard"` and
#'   `"dual_power"`.
#' @param param The distortion's parameter.
#' @returns A `distortion` object with `kind` and `param` properties. Use it
#'   with [risk_measure()] and [allocate()].
#' @export
#' @examples
#' d <- distortion("wang", 0.5)
#' risk_measure(sampled(c(10, 20, 30, 40, 50)), d)
#' distortion_g(distortion("dual_power", 2), 0.3)
distortion <- S7::new_class(
  "distortion",
  package = "actuarialrs",
  properties = list(
    ptr = S7::new_S3_class("RiskDistortion"),
    kind = S7::new_property(S7::class_character, getter = function(self) self@ptr$kind()),
    param = S7::new_property(S7::class_double, getter = function(self) self@ptr$param())
  ),
  constructor = function(kind = c("tvar", "wang", "proportional_hazard", "dual_power"), param) {
    kind <- match.arg(kind)
    S7::new_object(S7::S7_object(), ptr = rust_result(RiskDistortion$new(kind, as.double(param))))
  }
)

S7::method(print, distortion) <- function(x, ...) {
  cat(sprintf("<distortion> %s(%s)\n", x@kind, format(x@param)))
  invisible(x)
}

#' Distortion function and rank weights
#'
#' `distortion_g()` evaluates `g` at survival probabilities;
#' `distortion_weights()` gives the weight of each of `n` equally likely
#' values sorted ascending.
#'
#' @param d A [distortion].
#' @param s Survival probabilities in `[0, 1]`.
#' @param n Number of values.
#' @returns A numeric vector; the weights are non-negative and sum to 1.
#' @export
#' @examples
#' distortion_weights(distortion("tvar", 0.5), 4)
distortion_g <- function(d, s) d@ptr$g(as.double(s))

#' @rdname distortion_g
#' @export
distortion_weights <- function(d, n) rust_result(d@ptr$weights(as.double(n)))

#' Distortion risk measure of a distribution
#'
#' @param x A [sampled], [grid_distribution] or [predictive_distribution]
#'   (measured on its total).
#' @param distortion A [distortion].
#' @param ... Unused; for methods.
#' @returns A single number. `distortion("tvar", p)` gives the same value as
#'   [TVaR()].
#' @export
#' @examples
#' x <- sampled(c(1, 2, 3, 4))
#' risk_measure(x, distortion("tvar", 0.5))
#' risk_measure(grid_distribution(1, c(0.5, 0.25, 0.25)), distortion("proportional_hazard", 0.5))
risk_measure <- S7::new_generic("risk_measure", "x", function(x, distortion, ...) S7::S7_dispatch())

measure_impl <- function(x, distortion, ...) rust_result(distortion@ptr$measure(x@ptr), s7_call())
S7::method(risk_measure, sampled) <- measure_impl
S7::method(risk_measure, grid_distribution) <- measure_impl
S7::method(risk_measure, predictive_distribution) <- measure_impl

#' Allocate a risk measure to components
#'
#' Euler allocation by co-measure: simulations are ranked by their total,
#' and each component gets the distortion-weighted sum of its own draws.
#' The contributions add up to `risk_measure(x, distortion)`; for
#' `distortion("tvar", p)` they are the CoTVaRs,
#' `E[X_j | total in its top 1 - p]`. Simulations tied on the total share
#' their weights. The components must add up to the portfolio being
#' allocated.
#'
#' @param x A [predictive_distribution].
#' @param distortion A [distortion].
#' @param ... Unused; for methods.
#' @returns The `keys` data frame of `x` with a `contribution` column.
#' @export
#' @examples
#' pd <- predictive_distribution(
#'   matrix(c(1, 4, 2, 3, 2, 1, 5, 6), ncol = 2),
#'   data.frame(lob = c("motor", "property"))
#' )
#' allocate(pd, distortion("tvar", 0.5))
allocate <- S7::new_generic("allocate", "x", function(x, distortion, ...) S7::S7_dispatch())

S7::method(allocate, predictive_distribution) <- function(x, distortion, ...) {
  out <- x@keys
  out$contribution <- rust_result(distortion@ptr$allocate(x@ptr), s7_call())
  out
}

#' Copulas
#'
#' Dependence structures that draw uniforms, one vector per simulation.
#' `gaussian_copula()` and `t_copula()` take a correlation matrix;
#' `t_copula()` adds joint extremes through its degrees of freedom.
#' `archimedean_copula()` is exchangeable: Clayton (lower-tail dependence),
#' Gumbel and Joe (upper-tail dependence) or Frank (none). Kendall's tau is
#' `(2 / pi) * asin(r)` for the Gaussian and t copulas, `theta / (theta + 2)`
#' for Clayton and `1 - 1 / theta` for Gumbel.
#'
#' @param correlation A symmetric correlation matrix, positive definite.
#' @param nu Degrees of freedom, positive.
#' @param family One of `"clayton"`, `"gumbel"`, `"frank"` and `"joe"`.
#' @param theta Positive for Clayton and Frank; at least 1 for Gumbel and Joe.
#' @param dim Number of dimensions.
#' @param ptr Internal: an existing copula to wrap.
#' @returns A `copula` object with `dimension` and `description` properties. Use
#'   it with [copula_sample()] and [copula_simulate()].
#' @export
#' @examples
#' r <- matrix(c(1, 0.5, 0.5, 1), 2)
#' copula_sample(gaussian_copula(r), 3, seed = 1)
#' t_copula(r, nu = 4)
#' archimedean_copula("clayton", 2, dim = 3)
copula <- S7::new_class(
  "copula",
  package = "actuarialrs",
  properties = list(
    ptr = S7::new_S3_class("RiskCopula"),
    dimension = S7::new_property(S7::class_double, getter = function(self) self@ptr$dim()),
    description = S7::new_property(S7::class_character, getter = function(self) self@ptr$description())
  ),
  constructor = function(ptr) S7::new_object(S7::S7_object(), ptr = ptr)
)

S7::method(print, copula) <- function(x, ...) {
  cat(sprintf("<copula> %s, dimension %d\n", x@description, as.integer(x@dimension)))
  invisible(x)
}

#' @rdname copula
#' @export
gaussian_copula <- function(correlation) {
  correlation <- as.matrix(correlation)
  copula(rust_result(RiskCopula$gaussian(as.double(correlation), as.double(nrow(correlation)))))
}

#' @rdname copula
#' @export
t_copula <- function(correlation, nu) {
  correlation <- as.matrix(correlation)
  copula(rust_result(RiskCopula$student_t(
    as.double(correlation), as.double(nrow(correlation)), as.double(nu)
  )))
}

#' @rdname copula
#' @export
archimedean_copula <- function(family = c("clayton", "gumbel", "frank", "joe"), theta, dim = 2) {
  family <- match.arg(family)
  copula(rust_result(RiskCopula$archimedean(family, as.double(theta), as.double(dim))))
}

#' Draw uniforms from a copula
#'
#' Row `i` uses stream `i - 1` of `seed`, so any row replays alone.
#'
#' @param copula A [copula].
#' @param n Number of draws.
#' @param seed Seed, a whole number.
#' @returns An `n` by `copula@dimension` matrix of uniforms in `(0, 1)`.
#' @export
#' @examples
#' copula_sample(archimedean_copula("gumbel", 2), 5, seed = 1)
copula_sample <- function(copula, n, seed) {
  u <- rust_result(copula@ptr$sample(as.double(n), as.double(seed)))
  matrix(u, nrow = n)
}

#' Simulate marginals joined by a copula
#'
#' In simulation `i`, draws uniforms from `copula` and applies each
#' marginal's quantile function.
#'
#' @param copula A [copula].
#' @param marginals A list of [lognormal] or [grid_distribution] objects,
#'   one per copula dimension.
#' @param n_sims Number of simulations.
#' @param seed Seed, a whole number.
#' @param keys A data frame with one row per marginal and one column per
#'   dimension; defaults to `data.frame(component = seq_along(marginals))`.
#' @returns A [predictive_distribution].
#' @export
#' @examples
#' pd <- copula_simulate(
#'   gaussian_copula(matrix(c(1, 0.4, 0.4, 1), 2)),
#'   list(lognormal_from_mean_cv(100, 0.2), lognormal_from_mean_cv(50, 1)),
#'   n_sims = 10000, seed = 42,
#'   keys = data.frame(lob = c("motor", "property"))
#' )
#' mean(pd)
copula_simulate <- function(copula, marginals, n_sims, seed, keys = NULL) {
  if (is.null(keys)) keys <- data.frame(component = seq_along(marginals))
  keys <- as.data.frame(keys, stringsAsFactors = FALSE)
  ptrs <- lapply(marginals, function(m) m@ptr)
  predictive_distribution(ptr = rust_result(copula@ptr$simulate(
    ptrs, as.double(n_sims), as.double(seed), names(keys), as.list(keys)
  )))
}

#' Reorder draws to a target correlation (Iman-Conover)
#'
#' Every component keeps exactly its own draws; only their pairing across
#' simulations changes. The correlation of the result's normal scores is
#' close to `correlation`, and Spearman's rho close to
#' `(6 / pi) * asin(correlation / 2)`.
#'
#' @param x A [predictive_distribution].
#' @param correlation A correlation matrix with one row and column per
#'   component.
#' @param seed Seed for shuffling the scores, a whole number.
#' @returns A [predictive_distribution] with the same keys and marginals.
#' @export
#' @examples
#' pd <- predictive_distribution(
#'   cbind(1:1000, (1:1000 * 7919) %% 1000),
#'   data.frame(lob = c("a", "b"))
#' )
#' joined <- iman_conover(pd, matrix(c(1, 0.7, 0.7, 1), 2), seed = 3)
#' cor(draw_matrix(joined), method = "spearman")
iman_conover <- function(x, correlation, seed) {
  correlation <- as.matrix(correlation)
  predictive_distribution(ptr = rust_result(
    iman_conover_reorder(x@ptr, as.double(correlation), as.double(seed))
  ))
}
