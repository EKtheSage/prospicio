#' @include distributions.R
NULL

#' Distortion risk measure
#'
#' A concave distortion `g` of the survival function, giving the coherent
#' risk measure `rho(X) = integral of g(S(x)) dx`.
#'
#' | `kind` | `g(s)` for `0 < s < 1` | `param` |
#' |---|---|---|
#' | `"tvar"` | `min(s / (1 - p), 1)` | `p` in `[0, 1]` |
#' | `"wang"` | `pnorm(qnorm(s) + lambda)` | `lambda >= 0` |
#' | `"proportional_hazard"` | `s^rho` | `rho` in `(0, 1]` |
#' | `"dual_power"` | `1 - (1 - s)^beta` | `beta >= 1` |
#' | `"exponential"` | `(1 - exp(-k s)) / (1 - exp(-k))` | `k > 0` |
#' | `"ccoc"` | `min(1, d + (1 - d) s)`, `d = r / (1 + r)` | `0 <= r` |
#' | `"bitvar"` | `(1 - w) TVaR_p0 + w TVaR_p1` | `c(p0, p1, w)` |
#' | `"weighted_tvar"` | `sum(w_i min(s / (1 - p_i), 1))` | the levels `p_i`; `weights` the `w_i` |
#' | `"capped_linear"` | `min(1, r0 + slope s)` | `slope`, with `r0` |
#' | `"capped_log_linear"` | `min(1, exp(r0) s^b)` | `b` in `(0, 1]`, with `r0` |
#' | `"lep"` | `min(1, d0 + (1 - d0) s + (d - d0) sqrt(s (1 - s)))` | `r0 <= r`, with `r0` |
#' | `"linear_yield"` | `(r0 + (1 + r) s) / (1 + r0 + r s)` | `0 <= r`, with `r0` |
#' | `"beta"` | the Beta(a, b) distribution function | `c(a, b)`, `a <= 1 <= b` |
#'
#' `"ccoc"` is the constant cost of capital: the price of a loss `X` backed
#' by assets `max X` is `(E[X] + r max X) / (1 + r)`. It and the kinds with
#' `r0 > 0` put a probability mass on the largest outcome ([distortion_mass()]).
#' See also [distortion_mixture()], [distortion_minimum()],
#' [distortion_convex()] and [calibrate_distortion()].
#'
#' @param kind The kind of distortion; see the table.
#' @param param The distortion's parameter or parameters.
#' @param weights For `"weighted_tvar"`, the weights of the levels in
#'   `param`: non-negative, summing to 1.
#' @param r0 For `"capped_linear"`, `"capped_log_linear"`, `"lep"` and
#'   `"linear_yield"`: the intercept, a minimum rate on line.
#' @param ptr Internal.
#' @returns A `distortion` object with `kind` and `param` properties, and
#'   `parameters`, the scalar parameters by name as the constructor takes
#'   them (for example `c(rho = 0.7)`, or `c(r0 = 0.1, slope = 2)`; empty for
#'   weighted TVaR, mixtures, minimums and convex envelopes). Use it with
#'   [risk_measure()] and [allocate()].
#' @export
#' @examples
#' d <- distortion("wang", 0.5)
#' d@parameters
#' risk_measure(sampled(c(10, 20, 30, 40, 50)), d)
#' distortion_g(distortion("dual_power", 2), 0.3)
#' risk_measure(sampled(c(1, 2, 3, 4)), distortion("ccoc", 0.25)) # (2.5 + 0.25 * 4) / 1.25
distortion <- S7::new_class(
  "distortion",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("RiskDistortion"),
    kind = S7::new_property(S7::class_character, getter = function(self) self@ptr$kind()),
    param = S7::new_property(S7::class_double, getter = function(self) self@ptr$param()),
    parameters = S7::new_property(
      S7::class_double,
      getter = function(self) {
        p <- self@ptr$parameters()
        if (length(p) == 0) stats::setNames(numeric(), character()) else unlist(p)
      }
    )
  ),
  constructor = function(kind = c(
                           "tvar", "wang", "proportional_hazard", "dual_power", "exponential",
                           "ccoc", "bitvar", "weighted_tvar", "capped_linear",
                           "capped_log_linear", "lep", "linear_yield", "beta"
                         ),
                         param, weights = NULL, r0 = 0, ptr = NULL) {
    if (is.null(ptr)) {
      kind <- match.arg(kind)
      ptr <- rust_result(RiskDistortion$new(
        kind, as.double(param), as.double(if (is.null(weights)) numeric() else weights),
        as.double(r0)
      ))
    }
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

S7::method(print, distortion) <- function(x, ...) {
  cat(sprintf("<distortion> %s(%s)\n", x@kind, paste(format(x@param), collapse = ", ")))
  invisible(x)
}

#' Mixtures, minimums and convex hulls of distortions
#'
#' `distortion_mixture()` is the weighted average of distortions,
#' `distortion_minimum()` their pointwise minimum, and
#' `distortion_convex()` the smallest concave distortion above the points
#' `(s, g)`, for example layers' exceedance probabilities and their prices
#' per unit of limit (a cat bond's expected loss and its spread).
#'
#' @param distortions A list of [distortion] objects.
#' @param weights Non-negative weights summing to 1, one per distortion.
#' @param s,g The points, in the unit square with `g >= s`.
#' @returns A [distortion].
#' @export
#' @examples
#' d <- distortion_mixture(list(distortion("wang", 0.3), distortion("dual_power", 2)), c(0.5, 0.5))
#' distortion_g(d, 0.1)
#' distortion_g(distortion_convex(c(0.1, 0.5), c(0.3, 0.55)), 0.05) # 0.15
distortion_mixture <- function(distortions, weights) {
  ptrs <- lapply(distortions, function(d) d@ptr)
  distortion(ptr = rust_result(RiskDistortion$mixture(ptrs, as.double(weights))))
}

#' @rdname distortion_mixture
#' @export
distortion_minimum <- function(distortions) {
  ptrs <- lapply(distortions, function(d) d@ptr)
  distortion(ptr = rust_result(RiskDistortion$minimum(ptrs)))
}

#' @rdname distortion_mixture
#' @export
distortion_convex <- function(s, g) {
  distortion(ptr = rust_result(RiskDistortion$convex(as.double(s), as.double(g))))
}

#' Calibrate a distortion to a price
#'
#' The member of a family of distortions whose risk measure of `x` equals
#' `premium`. With `assets`, `x` is capped at the assets first, so the
#' premium prices `min(X, assets)`. The premium must lie strictly between
#' the (capped) mean and maximum. CCoC is solved in closed form; the other
#' families by bisection to full precision.
#'
#' @param family One of `"ccoc"`, `"proportional_hazard"` (`"ph"`),
#'   `"wang"`, `"dual_power"` (`"dual"`), `"tvar"`, `"exponential"`,
#'   `"capped_linear"`, `"capped_log_linear"`, `"lep"` and `"linear_yield"`.
#' @param x A [sampled], [grid_distribution] or [predictive_distribution]
#'   (its total).
#' @param premium The target price.
#' @param assets Optional assets at which to cap the loss.
#' @param r0 The fixed `r0` of the families that have one.
#' @returns A [distortion].
#' @export
#' @examples
#' x <- sampled(c(22, 28, 36, 40, 40, 40, 40, 55, 65, 100))
#' calibrate_distortion("ccoc", x, (46.6 + 15) / 1.15) # r = 0.15
#' calibrate_distortion("proportional_hazard", x, (46.6 + 15) / 1.15)
calibrate_distortion <- function(family, x, premium, assets = NULL, r0 = 0) {
  a <- if (is.null(assets)) Inf else as.double(assets)
  distortion(ptr = rust_result(RiskDistortion$calibrate(
    family, x@ptr, as.double(premium), a, as.double(r0)
  )))
}

#' Distortion mass, inverse and dual
#'
#' `distortion_mass()` is the probability mass the distortion puts on the
#' largest outcome, `g(0+)`; `distortion_g_inv()` the smallest `s` with
#' `g(s) >= y`; `distortion_g_dual()` the dual `1 - g(1 - s)`, which prices
#' the bid.
#'
#' @param d A [distortion].
#' @param y,s Probabilities in `[0, 1]`.
#' @returns A number, or a numeric vector.
#' @export
#' @examples
#' distortion_mass(distortion("ccoc", 0.25)) # 0.2
distortion_mass <- function(d) d@ptr$mass()

#' @rdname distortion_mass
#' @export
distortion_g_inv <- function(d, y) d@ptr$g_inv(as.double(y))

#' @rdname distortion_mass
#' @export
distortion_g_dual <- function(d, s) d@ptr$g_dual(as.double(s))

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

#' Exponential-utility risk measures
#'
#' `entropic_risk()` is `(1 / theta) log E[exp(theta X)]`, the certainty
#' equivalent of a loss under exponential utility: it rises from the mean
#' (`theta -> 0`) to the largest value (`theta -> Inf`), and is
#' `mu + theta sigma^2 / 2` for a normal loss. `esscher_premium()` is
#' `E[X exp(h X)] / E[exp(h X)]`, the mean after tilting probability
#' towards large losses: the mean at `h = 0`, `mu + h sigma^2` for a normal
#' loss. Both treat the draws as equally likely.
#'
#' @param x A numeric vector of draws, a [sampled] or a
#'   [predictive_distribution] (measured on its total).
#' @param theta Risk aversion, positive.
#' @param h Esscher parameter.
#' @returns A single number.
#' @name exponential_utility
#' @examples
#' entropic_risk(c(0, 1), log(2))
#' esscher_premium(c(0, 1), log(3))
NULL

risk_draws <- function(x) {
  if (S7::S7_inherits(x, predictive_distribution)) return(x@ptr$total()$draws())
  if (S7::S7_inherits(x, sampled)) return(x@draws)
  as.double(x)
}

#' @rdname exponential_utility
#' @export
entropic_risk <- function(x, theta) rust_result(entropic_rust(risk_draws(x), as.double(theta)))

#' @rdname exponential_utility
#' @export
esscher_premium <- function(x, h) rust_result(esscher_rust(risk_draws(x), as.double(h)))

#' Systemic risk contributions
#'
#' `marginal_expected_shortfall()` is each component's mean over the
#' simulations where the total is in its worst `1 - p`; it equals
#' `allocate(x, distortion("tvar", p))` and adds up to the total's TVaR.
#' `esscher_allocation()` is each component's mean under the Esscher
#' transform of the total, `E[X_j exp(h S)] / E[exp(h S)]`; it adds up to
#' `esscher_premium(x, h)`. `covar()` is the total's VaR at level `q` over
#' the simulations where one component is at or above its own VaR at `p`
#' (Adrian and Brunnermeier's CoVaR, in the form of Girardi and Ergun):
#' compare it with `VaR(x, q)` to see how much that component's bad years
#' drag the portfolio.
#'
#' @param x A [predictive_distribution].
#' @param p Level in `[0, 1]`: the total's tail for
#'   `marginal_expected_shortfall()`, the component's distress for `covar()`.
#' @param h Esscher parameter.
#' @param key The component, a list with one entry per dimension.
#' @param q Level of the total's VaR.
#' @returns `marginal_expected_shortfall()` and `esscher_allocation()`: the
#'   `keys` data frame of `x` with a `contribution` column. `covar()`: a
#'   single number.
#' @name systemic_risk
#' @examples
#' pd <- predictive_distribution(
#'   matrix(c(1, 2, 3, 4, 0, 1, 5, 1), ncol = 2),
#'   data.frame(lob = c("a", "b"))
#' )
#' marginal_expected_shortfall(pd, 0.5)
#' esscher_allocation(pd, 0.1)
#' covar(pd, list(lob = "a"), 0.75, 0.5)
NULL

#' @rdname systemic_risk
#' @export
marginal_expected_shortfall <- function(x, p) {
  out <- x@keys
  out$contribution <- rust_result(mes_rust(x@ptr, as.double(p)))
  out
}

#' @rdname systemic_risk
#' @export
esscher_allocation <- function(x, h) {
  out <- x@keys
  out$contribution <- rust_result(esscher_allocation_rust(x@ptr, as.double(h)))
  out
}

#' @rdname systemic_risk
#' @export
covar <- function(x, key, p, q) {
  rust_result(covar_rust(x@ptr, as.list(key), as.double(p), as.double(q)))
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
#' nac <- nested_archimedean_copula("gumbel",
#'   nac_node(1.5, 1, list(nac_node(3, c(2, 3)), nac_node(2, c(4, 5)))))
#' copula_cdf(nac, c(1, 0.5, 1, 1, 1))
copula <- S7::new_class(
  "copula",
  package = "prospicio",
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

#' @rdname copula
#' @param tree For `nested_archimedean_copula()`, the root node, built with
#'   `nac_node(theta, leaves, children)`: `leaves` the dimensions (1, 2,
#'   ...) joined at this level and `children` a list of nested nodes. Every
#'   dimension appears once, every node has at least two members, and no
#'   child's `theta` is below its parent's. Clayton or Gumbel only.
#' @details `nested_archimedean_copula()` joins groups of dimensions by their
#'   own Archimedean copula and the groups by an outer one with weaker
#'   dependence, to any depth (McNeil 2008): Kendall's tau between two
#'   dimensions is that of the node where they first meet.
#' @export
nested_archimedean_copula <- function(family = c("clayton", "gumbel"), tree) {
  family <- match.arg(family)
  copula(rust_result(RiskCopula$nested(family, tree)))
}

#' @rdname copula
#' @param leaves,children See `tree`.
#' @export
nac_node <- function(theta, leaves = numeric(), children = list()) {
  list(as.double(theta), as.double(leaves), children)
}

#' Vine copulas
#'
#' A C- or D-vine copula built from bivariate pair copulas (Aas et al.
#' 2009). `pair_copula()` is one: a family, its parameter and a rotation by
#' 0, 90, 180 or 270 degrees (for Clayton, Gumbel and Joe; 90 and 270 give
#' negative dependence; pyvinecopulib's conventions). In a C-vine, tree `t`
#' joins the `t`-th variable in the order to every later one, given those
#' before; in a D-vine, variables `t` places apart in the order, given those
#' between. Each edge's first argument is its earlier variable. Fewer trees
#' than `d - 1` truncate the vine (the rest independence).
#' `copula_log_density()` is the vine's log-density.
#'
#' @param family One of `"independence"`, `"gaussian"`, `"clayton"`,
#'   `"gumbel"`, `"frank"` and `"joe"`.
#' @param parameter Gaussian correlation in (-1, 1); Clayton above 0; Gumbel
#'   and Joe at least 1; Frank not 0.
#' @param rotation 0, 90, 180 or 270.
#' @param kind `"c"` or `"d"`.
#' @param order The dimensions (1, 2, ...) in vine order.
#' @param pairs A list of trees, tree `t` a list of its `d - t` pair copulas.
#' @param copula A vine [copula].
#' @param u A point or a matrix with one point per row.
#' @returns `pair_copula()`: a list; `vine_copula()`: a [copula];
#'   `copula_log_density()`: one value per point.
#' @export
#' @examples
#' v <- vine_copula("d", 1:3, list(
#'   list(pair_copula("gumbel", 2), pair_copula("clayton", 1.5, 90)),
#'   list(pair_copula("frank", 3))))
#' copula_log_density(v, c(0.2, 0.7, 0.4))
#' copula_sample(v, 3, seed = 1)
vine_copula <- function(kind = c("c", "d"), order, pairs) {
  kind <- match.arg(kind)
  copula(rust_result(RiskCopula$vine(kind, as.double(order), pairs)))
}

#' @rdname vine_copula
#' @export
pair_copula <- function(family = c("independence", "gaussian", "clayton", "gumbel", "frank",
                                   "joe"),
                        parameter = 0, rotation = 0) {
  family <- match.arg(family)
  list(family, as.double(parameter), as.double(rotation))
}

#' @rdname vine_copula
#' @export
copula_log_density <- function(copula, u) {
  u <- if (is.matrix(u)) u else matrix(u, nrow = 1)
  rust_result(copula@ptr$log_density(as.double(u), as.double(nrow(u))))
}

#' Copula distribution function
#'
#' `C(u) = P(U_1 <= u_1, ..., U_d <= u_d)` at each point: the Gaussian
#' copula's in two dimensions (the bivariate normal cdf by Genz's method,
#' about `1e-15`), the Archimedean copulas' in any. The t copula has none
#' yet and is an error.
#'
#' @param copula A [copula].
#' @param u A point (a vector of `copula@dimension` values in `[0, 1]`) or a
#'   matrix with one point per row.
#' @returns The distribution function at each point.
#' @export
#' @examples
#' copula_cdf(gaussian_copula(matrix(c(1, 0.5, 0.5, 1), 2)), c(0.5, 0.5))
#' copula_cdf(archimedean_copula("gumbel", 2), rbind(c(0.2, 0.3), c(0.9, 0.9)))
copula_cdf <- function(copula, u) {
  u <- if (is.matrix(u)) u else matrix(u, nrow = 1)
  rust_result(copula@ptr$cdf(as.double(u), as.double(nrow(u))))
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

#' Generalized Pareto fit
#'
#' Maximum likelihood fit of the generalized Pareto distribution
#' `P(X > x) = (1 + xi * x / beta)^(-1 / xi)` to exceedances (values over a
#' threshold, minus the threshold), as `evd::fpot` or SciPy's
#' `genpareto.fit(floc = 0)`.
#'
#' @param exceedances At least 3 non-negative values, not all equal.
#' @returns A named numeric vector `c(xi = , beta = )`.
#' @export
#' @examples
#' y <- 2 * expm1(0.5 * -log1p(-ppoints(1000))) / 0.5 # GPD(0.5, 2) quantiles
#' gpd_fit(y)
gpd_fit <- function(exceedances) {
  stats::setNames(rust_result(gpd_mle(as.double(exceedances))), c("xi", "beta"))
}

#' Peaks-over-threshold tail
#'
#' Fits a generalized Pareto distribution to the draws above their empirical
#' `level` quantile, so that [VaR()] and [TVaR()] at levels above `level`
#' extend smoothly past the largest draw.
#'
#' @param draws A [sampled] object.
#' @param level Threshold level, for example `0.95` for the top 5%.
#' @returns A `pot_tail` object with `threshold`, `p_exceed`, `xi` and `beta`
#'   properties; [VaR()] and [TVaR()] accept levels `p >= 1 - p_exceed`.
#' @export
#' @examples
#' s <- sampled(qlnorm(ppoints(100000)))
#' tail <- pot_tail(s, 0.95)
#' VaR(tail, 0.999) / qlnorm(0.999)
#' TVaR(tail, c(0.99, 0.999))
pot_tail <- S7::new_class(
  "pot_tail",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("EvtTail"),
    threshold = S7::new_property(S7::class_double, getter = function(self) self@ptr$threshold()),
    p_exceed = S7::new_property(S7::class_double, getter = function(self) self@ptr$p_exceed()),
    xi = S7::new_property(S7::class_double, getter = function(self) self@ptr$xi()),
    beta = S7::new_property(S7::class_double, getter = function(self) self@ptr$beta())
  ),
  constructor = function(draws, level) {
    S7::new_object(S7::S7_object(), ptr = rust_result(EvtTail$fit(draws@ptr, as.double(level))))
  }
)

S7::method(VaR, pot_tail) <- function(dist, p, ...) rust_result(dist@ptr$var(as.double(p)), s7_call())
S7::method(TVaR, pot_tail) <- function(dist, p, ...) rust_result(dist@ptr$tvar(as.double(p)), s7_call())
S7::method(print, pot_tail) <- function(x, ...) {
  cat(sprintf("<pot_tail> threshold %s (top %s), GPD xi = %s, beta = %s\n",
              format(x@threshold), format(x@p_exceed), format(x@xi), format(x@beta)))
  invisible(x)
}

#' Capital allocation and diversification
#'
#' Splits the distortion risk measure of a portfolio's total, `rho(S)`, back
#' to its components, and reports each component's stand-alone measure.
#'
#' | `method` | Allocation to component `j` |
#' |---|---|
#' | `"euler"` | co-measure, as [allocate()] (CoTVaR for TVaR) |
#' | `"covariance"` | `rho(S) Cov(X_j, S) / Var(S)` |
#' | `"proportional"` | stand-alone measures scaled to `rho(S)` |
#' | `"marginal"` | `rho(S) - rho(S - X_j)` (Merton-Perold); does not add up |
#' | `"shapley"` | Shapley value of `v(T) = rho(sum of T)`; at most 12 components |
#'
#' Euler is the only method consistent with marginal changes to the
#' portfolio. The components must add up to the portfolio being allocated.
#'
#' @param x A [predictive_distribution].
#' @param distortion A [distortion].
#' @param method One of `"euler"`, `"covariance"`, `"proportional"`,
#'   `"marginal"`, `"shapley"`.
#' @returns A list with `total` (`rho(S)`), `diversification_benefit`
#'   (`sum(standalone) - total`) and `by_component`, the `keys` data frame of
#'   `x` with columns `standalone`, `allocated` and `diversification`
#'   (`standalone - allocated`).
#' @export
#' @examples
#' pd <- predictive_distribution(
#'   matrix(c(1, 4, 2, 3, 2, 1, 5, 6), ncol = 2),
#'   data.frame(lob = c("motor", "property"))
#' )
#' a <- capital_allocation(pd, distortion("tvar", 0.5), "shapley")
#' a$by_component
#' a$diversification_benefit
capital_allocation <- function(x, distortion,
                               method = c("euler", "covariance", "proportional", "marginal", "shapley")) {
  method <- match.arg(method)
  r <- rust_result(distortion@ptr$capital(x@ptr, method))
  out <- x@keys
  out$standalone <- r$standalone
  out$allocated <- r$allocated
  out$diversification <- r$standalone - r$allocated
  list(
    total = r$total,
    diversification_benefit = sum(r$standalone) - r$total,
    by_component = out
  )
}

#' Tail diagnostics
#'
#' `mean_excess()` is the empirical mean-excess function
#' `e(u) = E[X - u | X > u]` at each threshold (linear above a threshold where
#' a generalized Pareto fits); `hill_estimator()` gives Hill estimates of the
#' tail index `xi` (`1 / alpha`) from the `k` largest values. Both help
#' choose a threshold for [pot_tail()].
#'
#' @param x Numeric vector of losses or draws.
#' @param thresholds Thresholds.
#' @param k Numbers of top values to use.
#' @returns `mean_excess()`: a data frame with `threshold`, `mean_excess`
#'   (`NaN` where no value exceeds it) and `n_above`. `hill_estimator()`: a
#'   numeric vector, one estimate per `k`.
#' @name tail_diagnostics
#' @examples
#' mean_excess(c(1, 2, 3, 4), 2)
#' x <- (1 - (seq_len(2000) - 0.5) / 2000)^-0.5
#' hill_estimator(x, c(100, 200))
NULL

#' @rdname tail_diagnostics
#' @export
mean_excess <- function(x, thresholds) {
  as.data.frame(mean_excess_rust(as.double(x), as.double(thresholds)))
}

#' @rdname tail_diagnostics
#' @export
hill_estimator <- function(x, k) rust_result(hill_rust(as.double(x), as.double(k)))

#' Threshold diagnostics for peaks over threshold
#'
#' At each threshold, the mean excess and the generalized Pareto fitted to
#' the exceedances, with standard errors: the numbers behind the mean
#' residual life plot and the parameter stability plots of
#' `ismev::mrl.plot()` and `ismev::gpd.fitrange()`. Above a threshold where
#' a GPD fits, the mean excess is linear in the threshold and the shape `xi`
#' and modified scale `beta - xi * u` are constant up to their sampling
#' error: pick the lowest threshold above which they are stable, then fit
#' [pot_tail()] there. Standard errors come from the observed information
#' and hold for `xi > -0.5`.
#'
#' @param x Numeric vector of losses or draws.
#' @param thresholds Thresholds.
#' @returns A data frame with one row per threshold: `threshold`, `n_above`,
#'   `mean_excess`, `mean_excess_se` (the standard deviation of the
#'   exceedances over `sqrt(n_above)`), `xi`, `xi_se`, `beta`,
#'   `modified_scale` and `modified_scale_se`. Fields that cannot be
#'   estimated (fewer than 2 values above for the standard error of the mean
#'   excess, 3 for the GPD) are `NaN`.
#' @seealso [mean_excess()], [hill_estimator()], [pot_tail()]
#' @export
#' @examples
#' p <- (seq_len(2000) - 0.5) / 2000
#' x <- ((1 - p)^(-1 / 2) - 1)^(1 / 1.5) # a Burr: xi tends to 1/3
#' threshold_diagnostics(x, c(0.5, 1, 2))
threshold_diagnostics <- function(x, thresholds) {
  as.data.frame(threshold_diagnostics_rust(as.double(x), as.double(thresholds)))
}
