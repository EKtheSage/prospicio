# Pricing: the collective model, layer rating and reinsurance tower
# matching (docs/design/pareto.md). S7 classes and functions over the Rust
# objects, as in distributions.R.

#' Collective risk model
#'
#' A claim count and a severity, with the layer moments treaty pricing needs
#' in closed form. For the layer `limit` xs `attachment` applied to each
#' loss, with `Y` the loss to the layer from one claim, the aggregate has
#' mean `E[N] E[Y]` and variance `E[N] Var[Y] + Var[N] E[Y]^2`.
#'
#' Supports [mean()], [variance()], [layer()] (the aggregate layer mean),
#' [layer_variance()], [excess_frequency()] and [simulate_events()]-style
#' simulation with [collective_simulate()].
#'
#' @param frequency A [poisson_count], [negative_binomial_count] or
#'   [binomial_count].
#' @param severity A severity: [lognormal], [grid_distribution] or a
#'   Pareto-family distribution.
#' @returns A `collective_model` object.
#' @export
#' @examples
#' m <- collective_model(claim_count(2, 1.5), pareto(1e6, 2))
#' layer(m, 4e6, 1e6)
#' excess_frequency(m, 2e6)
#' sqrt(layer_variance(m, 4e6, 1e6))
collective_model <- S7::new_class(
  "collective_model",
  package = "prospicio",
  properties = list(ptr = S7::new_S3_class("CollectiveModel")),
  constructor = function(frequency, severity) {
    ptr <- rust_result(CollectiveModel$new(frequency@ptr, severity@ptr))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Expected excess frequency
#'
#' Expected number of losses a year above `x`.
#'
#' @param model A [collective_model] or a [tower_model].
#' @param x Numeric vector.
#' @param ... Unused; for methods.
#' @returns Numeric vector the length of `x`.
#' @export
#' @examples
#' excess_frequency(collective_model(poisson_count(3), pareto(1, 2.5)), c(2, 4))
excess_frequency <- S7::new_generic("excess_frequency", "model", function(model, x, ...) {
  S7::S7_dispatch()
})

S7::method(mean, collective_model) <- function(x, ...) x@ptr$mean()
S7::method(variance, collective_model) <- function(dist, ...) dist@ptr$variance()
S7::method(layer, collective_model) <- function(dist, limit, attachment, ...) {
  dist@ptr$layer_mean(as.double(limit), as.double(attachment))
}
S7::method(layer_variance, collective_model) <- function(dist, limit, attachment, ...) {
  dist@ptr$layer_variance(as.double(limit), as.double(attachment))
}
S7::method(excess_frequency, collective_model) <- function(model, x, ...) {
  model@ptr$excess_frequency(as.double(x))
}
S7::method(print, collective_model) <- function(x, ...) {
  cat(sprintf("<collective_model> mean = %s\n", format(mean(x), digits = 15)))
  invisible(x)
}

#' Simulate a collective model
#'
#' `n_sims` years of individual losses, as [simulate_events()].
#'
#' @param model A [collective_model].
#' @param n_sims Number of simulated years.
#' @param seed Generator seed.
#' @returns An [event_set].
#' @export
#' @examples
#' ev <- collective_simulate(collective_model(poisson_count(2), pareto(1000, 2)), 100, seed = 1)
#' ev@n_sims
collective_simulate <- function(model, n_sims, seed) {
  event_set(ptr = rust_result(model@ptr$simulate(as.double(n_sims), as.double(seed))))
}

#' Tower model
#'
#' A frequency above the lowest threshold and a [piecewise_pareto]
#' severity that reproduce a reinsurance tower ([match_tower()]), a PML
#' curve ([fit_pml_curve()]) or a set of references ([fit_references()]).
#' Properties: `m@frequency`, `m@severity`. Supports [excess_frequency()]
#' and [layer()] (expected loss a year).
#'
#' @param ptr A `TowerModel` pointer; use the fitting functions instead.
#' @returns A `tower_model` object.
#' @export
#' @examples
#' m <- match_tower(c(1000, 1500, 2000), c(100, 90, 120), frequencies = c(0.25, NA, NA))
#' layer(m, 500, 1500)
#' m@severity
tower_model <- S7::new_class(
  "tower_model",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("TowerModel"),
    frequency = S7::new_property(S7::class_double, getter = function(self) self@ptr$frequency()),
    severity = S7::new_property(S7::class_any, getter = function(self) {
      piecewise_pareto(ptr = self@ptr$severity())
    })
  ),
  constructor = function(ptr) S7::new_object(S7::S7_object(), ptr = ptr)
)

S7::method(layer, tower_model) <- function(dist, limit, attachment, ...) {
  dist@ptr$layer_loss(as.double(limit), as.double(attachment))
}
S7::method(excess_frequency, tower_model) <- function(model, x, ...) {
  model@ptr$excess_frequency(as.double(x))
}
S7::method(print, tower_model) <- function(x, ...) {
  cat(sprintf("<tower_model> frequency = %s\n", format(x@frequency, digits = 15)))
  print(x@severity)
  invisible(x)
}

#' Match a reinsurance tower
#'
#' One collective model (a frequency and a piecewise Pareto severity) that
#' reproduces the expected loss of every layer of a tower (Riegel 2018,
#' Matching Algorithm 2).
#'
#' @param attachments Increasing attachment points; layer `i` runs to the
#'   next one, the last is unlimited.
#' @param layer_losses Expected loss a year of each layer.
#' @param frequencies Expected losses a year above each attachment point,
#'   `NA` where to derive them, or `NULL` to derive all.
#' @param rule `"minimize"` (make the two alphas in each layer as close as
#'   possible) or `"midpoint"`.
#' @returns A [tower_model].
#' @export
#' @examples
#' m <- match_tower(c(1000, 1500, 2000, 2500, 3000, 5000, 10000),
#'                  c(100, 90, 50, 40, 100, 50, 50),
#'                  frequencies = c(0.25, rep(NA, 6)))
#' layer(m, 500, 1500)
match_tower <- function(attachments, layer_losses, frequencies = NULL, rule = c("minimize", "midpoint")) {
  rule <- match.arg(rule)
  freq <- if (is.null(frequencies)) double() else as.double(frequencies)
  tower_model(rust_result(TowerModel$match_tower(
    as.double(attachments), as.double(layer_losses), freq, rule
  )))
}

#' Fit a PML curve
#'
#' The model through the points of a PML curve: `amounts[j]` is exceeded
#' once in `return_periods[j]` years.
#'
#' @param return_periods Numeric vector.
#' @param amounts Numeric vector, one per return period.
#' @param tail_alpha Alpha above the largest amount.
#' @param truncation Truncation of the last piece, or `NULL`.
#' @returns A [tower_model].
#' @export
#' @examples
#' m <- fit_pml_curve(c(10, 40, 100), c(1e6, 2e6, 3e6))
#' excess_frequency(m, 2e6)
fit_pml_curve <- function(return_periods, amounts, tail_alpha = 2, truncation = NULL) {
  tower_model(rust_result(TowerModel$fit_pml_curve(
    as.double(return_periods), as.double(amounts), as.double(tail_alpha), truncation_arg(truncation)
  )))
}

#' Fit references
#'
#' A model that reproduces every reference: expected layer losses (which
#' may overlap or leave gaps) and excess frequencies. Free values between
#' references are chosen at the analytic center of what is consistent, so
#' gaps get moderate alphas.
#'
#' @param layers A data frame with columns `limit`, `attachment` and
#'   `expected_loss`, or `NULL`.
#' @param frequencies A data frame with columns `threshold` and
#'   `frequency`, or `NULL`.
#' @param default_alpha Alpha above the highest point, unless an unlimited
#'   layer sets it.
#' @param rule As in [match_tower()].
#' @param alpha_bounds `NULL`, or `c(lo, hi)`: keep the Pareto alpha between
#'   consecutive reference points within these bounds. The two pieces fitted
#'   inside a limited layer can still fall either side; check with
#'   [tower_diagnostics()].
#' @returns A [tower_model].
#' @export
#' @examples
#' m <- fit_references(
#'   layers = data.frame(limit = c(1000, 3000), attachment = c(1000, 1500),
#'                       expected_loss = c(150, 160)),
#'   frequencies = data.frame(threshold = 1000, frequency = 0.3)
#' )
#' layer(m, 3000, 1500)
fit_references <- function(layers = NULL, frequencies = NULL, default_alpha = 2,
                           rule = c("minimize", "midpoint"), alpha_bounds = NULL) {
  rule <- match.arg(rule)
  col <- function(df, name) if (is.null(df)) double() else as.double(df[[name]])
  tower_model(rust_result(TowerModel$fit_references(
    col(layers, "limit"), col(layers, "attachment"), col(layers, "expected_loss"),
    col(frequencies, "threshold"), col(frequencies, "frequency"), as.double(default_alpha), rule,
    if (is.null(alpha_bounds)) double() else as.double(alpha_bounds)
  )))
}

#' Checks on a tower model's shape
#'
#' A match reproduces its references exactly, but between and beyond them a
#' piece of the severity can be implausible: an alpha below `min_alpha` (below
#' 1 on the last, untruncated piece the mean is infinite), or neighbouring
#' alphas more than `max_ratio` apart.
#'
#' @param model A [tower_model].
#' @param min_alpha The smallest plausible alpha.
#' @param max_ratio The largest plausible ratio between neighbouring alphas.
#' @returns A list with `min_alpha`, `max_alpha`, `max_ratio`, `infinite_mean`
#'   and `warnings` (a character vector, empty when the shape looks
#'   plausible).
#' @export
#' @examples
#' m <- fit_references(
#'   layers = data.frame(limit = c(1, 5, Inf), attachment = c(1, 5, 10),
#'                       expected_loss = c(1.706, 0.675, 0.295)),
#'   frequencies = data.frame(threshold = 1, frequency = 2.8)
#' )
#' tower_diagnostics(m)$warnings
tower_diagnostics <- function(model, min_alpha = 1, max_ratio = 2) {
  model@ptr$diagnostics(as.double(min_alpha), as.double(max_ratio))
}

#' Layer rating
#'
#' Ratios and implied Pareto alphas used in both primary and reinsurance
#' pricing:
#'
#' - `ilf(severity, limit, basic_limit)`: increased limit factors
#'   `LEV(limit) / LEV(basic_limit)`;
#' - `loss_elimination_ratio(severity, deductible)`: `LEV(d) / E[X]`;
#' - `pareto_extrapolation(from, to, alpha)`: expected loss of layer `to`
#'   per unit of expected loss of layer `from`, layers as
#'   `c(limit, attachment)`;
#' - `alpha_between_layers(a, b)`: the alpha at which two layers, as
#'   `c(limit, attachment, expected_loss)`, have their losses;
#' - `alpha_between_frequency_and_layer()` and
#'   `alpha_between_frequencies()`: the alpha between a frequency above a
#'   threshold and a layer, or two frequencies.
#'
#' @param severity A severity.
#' @param limit,basic_limit,deductible Numeric.
#' @param from,to,a,b Layers, as described above.
#' @param alpha Pareto alpha.
#' @param threshold,frequency A frequency above a threshold.
#' @param attachment,expected_loss The layer for
#'   `alpha_between_frequency_and_layer()`.
#' @param threshold_1,frequency_1,threshold_2,frequency_2 Two frequencies.
#' @param truncation Pareto truncation point, or `NULL`.
#' @returns A numeric vector (`ilf`, `loss_elimination_ratio`) or a single
#'   number.
#' @name layer_rating
#' @examples
#' p <- pareto(100, 2)
#' ilf(p, c(500, 1000), 200)
#' loss_elimination_ratio(p, 150)
#' pareto_extrapolation(c(1e6, 1e6), c(2e6, 2e6), 2)
#' alpha_between_frequencies(1e6, 4, 2e6, 1)
NULL

#' @rdname layer_rating
#' @export
ilf <- function(severity, limit, basic_limit) {
  rust_result(pricing_ilf(severity@ptr, as.double(limit), as.double(basic_limit)))
}

#' Coefficient of variation of a layer's loss
#'
#' The CV of the loss to `limit` xs `attachment` per loss that reaches the
#' layer: `sd / mean` of `min(max(X - a, 0), l)` given `X > a`, the CV that
#' a limited fluctuation pure premium standard needs for a layer.
#'
#' @param severity A severity: [lognormal], [grid_distribution] or a
#'   Pareto-family distribution.
#' @param limit Positive; `Inf` for an unlimited layer.
#' @param attachment Non-negative.
#' @returns A number.
#' @export
#' @examples
#' layer_cv(pareto(100, 3), Inf, 0)
layer_cv <- function(severity, limit, attachment) {
  rust_result(pricing_layer_cv(severity@ptr, as.double(limit), as.double(attachment)))
}

#' @rdname layer_rating
#' @export
loss_elimination_ratio <- function(severity, deductible) {
  rust_result(pricing_loss_elimination_ratio(severity@ptr, as.double(deductible)))
}

#' @rdname layer_rating
#' @export
pareto_extrapolation <- function(from, to, alpha, truncation = NULL) {
  rust_result(pricing_pareto_extrapolation(as.double(from), as.double(to), as.double(alpha),
                                           truncation_arg(truncation)))
}

#' @rdname layer_rating
#' @export
alpha_between_layers <- function(a, b, truncation = NULL) {
  rust_result(pricing_alpha_between_layers(as.double(a), as.double(b), truncation_arg(truncation)))
}

#' @rdname layer_rating
#' @export
alpha_between_frequency_and_layer <- function(threshold, frequency, limit, attachment,
                                              expected_loss, truncation = NULL) {
  rust_result(pricing_alpha_between_frequency_and_layer(
    as.double(threshold), as.double(frequency), as.double(limit), as.double(attachment),
    as.double(expected_loss), truncation_arg(truncation)
  ))
}

#' @rdname layer_rating
#' @export
alpha_between_frequencies <- function(threshold_1, frequency_1, threshold_2, frequency_2,
                                      truncation = NULL) {
  rust_result(pricing_alpha_between_frequencies(
    as.double(threshold_1), as.double(frequency_1), as.double(threshold_2),
    as.double(frequency_2), truncation_arg(truncation)
  ))
}

#' Risk-loaded prices from simulated losses
#'
#' `risk_loaded_price()` prices one cover from its loss draws, or from its
#' loss distribution on a grid (a layer's exact annual loss from
#' [tower_on_grid()]);
#' `price_portfolio()` prices a portfolio and allocates the price to its
#' components. The assets backing the loss are the distortion risk measure
#' `assets` of it. The premium is either the pricing distortion
#' `distortion` of the loss, or set by a constant cost of capital `r` on
#' the capital `a - P`, which gives `P = (E[X] + r a) / (1 + r)`.
#'
#' In a portfolio, premium and assets are each allocated by co-measure
#' (the natural allocation of Mildenhall and Major, 2022): component prices
#' add up to the portfolio's, a component that diversifies the portfolio is
#' priced below its standalone price, and with a cost of capital every
#' component earns the rate on its allocated capital.
#'
#' @param x A [sampled], a [predictive_distribution] (its total) or a
#'   [grid_distribution] for
#'   `risk_loaded_price()`; a [predictive_distribution] whose components add
#'   up to the portfolio (segments or covers, not gross, ceded and net side
#'   by side) for `price_portfolio()`.
#' @param assets A [distortion] that sets the assets, for example
#'   `distortion("tvar", 0.99)`.
#' @param cost_of_capital A positive rate, or `NULL`.
#' @param distortion A pricing [distortion], or `NULL`; it must load less
#'   than `assets`. Give exactly one of `cost_of_capital` and `distortion`.
#' @returns `risk_loaded_price()`: a list with `expected_loss`, `premium`,
#'   `assets`, `margin` (`P - E[X]`), `capital` (`a - P`), `loss_ratio` and
#'   `return_on_capital`. `price_portfolio()`: a list with `total` (the
#'   portfolio's price, as above), `diversification` (the sum of standalone
#'   premiums less the portfolio premium) and `by_component`, the `keys`
#'   data frame of `x` with the allocated `expected_loss`, `premium`,
#'   `assets`, `margin`, `capital` and `return_on_capital`, and
#'   `standalone_premium`.
#' @name risk_loaded_price
#' @examples
#' risk_loaded_price(sampled(c(0, 0, 2, 6)), distortion("tvar", 0.5),
#'                   cost_of_capital = 0.25)$premium
#' # The same on a grid: 0, 2 or 6 with probabilities 1/2, 1/4, 1/4.
#' risk_loaded_price(grid_distribution(2, c(0.5, 0.25, 0, 0.25)), distortion("tvar", 0.5),
#'                   cost_of_capital = 0.25)$premium
#' pd <- predictive_distribution(
#'   matrix(c(0, 1, 4, 8, 2, 1, 0, 0), ncol = 2),
#'   data.frame(cover = c("a", "b"))
#' )
#' p <- price_portfolio(pd, distortion("tvar", 0.5), cost_of_capital = 0.1)
#' p$by_component
#' p$diversification
NULL

#' @rdname risk_loaded_price
#' @export
risk_loaded_price <- function(x, assets, cost_of_capital = NULL, distortion = NULL) {
  rust_result(pricing_price(x@ptr, assets@ptr, rate_arg(cost_of_capital), pricing_arg(distortion)))
}

#' @rdname risk_loaded_price
#' @export
price_portfolio <- function(x, assets, cost_of_capital = NULL, distortion = NULL) {
  r <- rust_result(pricing_price_portfolio(x@ptr, assets@ptr, rate_arg(cost_of_capital),
                                           pricing_arg(distortion)))
  out <- x@keys
  a <- r$allocated
  out$expected_loss <- a$expected_loss
  out$premium <- a$premium
  out$assets <- a$assets
  out$margin <- a$premium - a$expected_loss
  out$capital <- a$assets - a$premium
  out$return_on_capital <- out$margin / out$capital
  out$standalone_premium <- r$standalone$premium
  list(
    total = r$total,
    diversification = sum(r$standalone$premium) - r$total$premium,
    by_component = out
  )
}

rate_arg <- function(rate) if (is.null(rate)) NA_real_ else as.double(rate)
pricing_arg <- function(d) if (is.null(d)) NULL else d@ptr

#' MBBEFD exposure curves
#'
#' The MBBEFD class of Bernegger (1997) for property per-risk exposure
#' rating: `G(x)` is the share of a risk's expected loss below the fraction
#' `x` of its maximum possible loss (MPL), and `1/g` the probability of a
#' total loss. `swiss_re_curve(c)` gives Bernegger's one-parameter family,
#' `b = exp(3.1 - 0.15 (1 + c) c)`, `g = exp((0.78 + 0.12 c) c)`: `c = 1.5,
#' 2, 3, 4` are the Swiss Re curves and `c = 5` the Lloyd's curve.
#'
#' `exposure_curve()` evaluates `G`; `exposure_layer_share()` is the share
#' of a risk's expected loss in the layer `limit` xs `attachment`, `G(min((a
#' + l)/M, 1)) - G(min(a/M, 1))`. `severity_exposure_curve()` is the curve
#' of any severity capped at the MPL, `LEV(x M) / LEV(M)`.
#'
#' @param b,g MBBEFD parameters, `b >= 0`, `g >= 1`.
#' @param c Swiss Re curve parameter, non-negative (0 is the straight line).
#' @param curve An `mbbefd` or [tabulated_curve] object.
#' @param x Fractions of the MPL.
#' @param limit,attachment The layer.
#' @param mpl Maximum possible loss of the risk.
#' @param severity A severity: [lognormal], [grid_distribution] or a
#'   Pareto-family distribution.
#' @returns `mbbefd()` and `swiss_re_curve()`: an `mbbefd` object with
#'   properties `b`, `g`, `mean` (the mean destruction rate) and
#'   `total_loss_probability`. The others: numeric vectors.
#' @name mbbefd
#' @examples
#' c3 <- swiss_re_curve(3)
#' exposure_curve(c3, c(0.1, 0.5, 1))
#' exposure_layer_share(c3, 5e6, 5e6, 10e6)
#' severity_exposure_curve(pareto(1e5, 1.5), 1e7, c(0.5, 1))
NULL

#' @rdname mbbefd
#' @export
mbbefd <- S7::new_class(
  "mbbefd",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("Mbbefd"),
    b = S7::new_property(S7::class_double, getter = function(self) self@ptr$b()),
    g = S7::new_property(S7::class_double, getter = function(self) self@ptr$g()),
    mean = S7::new_property(S7::class_double, getter = function(self) self@ptr$mean()),
    total_loss_probability = S7::new_property(
      S7::class_double, getter = function(self) self@ptr$total_loss_probability()
    )
  ),
  constructor = function(b, g, ptr = NULL) {
    if (is.null(ptr)) ptr <- rust_result(Mbbefd$new(as.double(b), as.double(g)))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

S7::method(print, mbbefd) <- function(x, ...) {
  cat(sprintf("<mbbefd> b = %s, g = %s, total loss probability %s\n", format(x@b),
              format(x@g), format(x@total_loss_probability)))
  invisible(x)
}

#' @rdname mbbefd
#' @export
swiss_re_curve <- function(c) {
  mbbefd(ptr = rust_result(Mbbefd$swiss_re(as.double(c))))
}

#' @rdname mbbefd
#' @param rate A mean destruction rate, between the curves' at `c = 10`
#'   and `c = 0` (0.5).
#' @param share The layer's share of a risk's expected loss.
#' @details `swiss_re_for_mean_rate()` and `swiss_re_for_layer_share()`
#'   calibrate `c`: to a mean destruction rate (a cedant's loss ratio over
#'   its rate on sum insured, say), or to the share of a risk's expected
#'   loss in one layer. They return the curve, with its `c` as the
#'   attribute `"c"`.
#' @export
swiss_re_for_mean_rate <- function(rate) {
  c <- rust_result(pricing_swiss_re_c_for_mean_rate(as.double(rate)))
  structure(swiss_re_curve(c), c = c)
}

#' @rdname mbbefd
#' @export
swiss_re_for_layer_share <- function(limit, attachment, mpl, share) {
  c <- rust_result(pricing_swiss_re_c_for_layer_share(
    as.double(limit), as.double(attachment), as.double(mpl), as.double(share)
  ))
  structure(swiss_re_curve(c), c = c)
}

#' @rdname mbbefd
#' @export
exposure_curve <- function(curve, x) curve@ptr$curve(as.double(x))

#' @rdname mbbefd
#' @export
exposure_layer_share <- function(curve, limit, attachment, mpl) {
  rust_result(curve@ptr$layer_share(as.double(limit), as.double(attachment), as.double(mpl)))
}

#' @rdname mbbefd
#' @export
severity_exposure_curve <- function(severity, mpl, x) {
  rust_result(pricing_severity_exposure_curve(severity@ptr, as.double(mpl), as.double(x)))
}

#' @rdname mbbefd
#' @param u Probabilities in `(0, 1)`.
#' @export
rate_quantile <- function(curve, u) curve@ptr$rate_quantile(as.double(u))

#' Tabulated exposure curve
#'
#' An exposure curve from a published table of points `(x, G(x))` from
#' `(0, 0)` to `(1, 1)`, interpolated linearly: Salzmann's homeowners scale,
#' Ludwig's curves, ISO PSOLD tables or a reinsurer's own. The table must be
#' concave (its slopes never increase). Its destruction rate is discrete:
#' the points' `x` with probabilities from the drops in slope, and a total
#' loss with probability last slope over first. Its mean rate (`curve@mean`)
#' is the first chord's, `x1 / G(x1)`, so a table needs fine first points
#' for the expected loss to be right.
#'
#' Works with [exposure_curve()], [exposure_layer_share()] and
#' [rate_quantile()], like an [mbbefd] curve.
#'
#' `tabulated_curve_from_severity()` tabulates a severity's exposure curve
#' capped at the MPL (as [severity_exposure_curve()]) at `points` points,
#' half spaced geometrically from `1e-6` and half at equal steps of `G`,
#' so a risk profile's band can use it.
#'
#' @param x Increasing from 0 to 1.
#' @param g `G(x)`, from 0 to 1.
#' @param ptr Internal.
#' @returns A `tabulated_curve` object with properties `x`, `g` and `mean`.
#' @export
#' @examples
#' t <- tabulated_curve(c(0, 0.1, 0.5, 1), c(0, 0.4, 0.8, 1))
#' exposure_curve(t, 0.3)
#' t@mean
#' s <- tabulated_curve_from_severity(lognormal_from_mean_cv(1e5, 2), 1e6)
#' exposure_curve(s, 0.5)
tabulated_curve <- S7::new_class(
  "tabulated_curve",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("Tabulated"),
    x = S7::new_property(S7::class_double, getter = function(self) self@ptr$x()),
    g = S7::new_property(S7::class_double, getter = function(self) self@ptr$g()),
    mean = S7::new_property(S7::class_double, getter = function(self) self@ptr$mean())
  ),
  constructor = function(x, g, ptr = NULL) {
    if (is.null(ptr)) ptr <- rust_result(Tabulated$new(as.double(x), as.double(g)))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' @rdname tabulated_curve
#' @param severity A severity: [lognormal], [grid_distribution] or a
#'   Pareto-family distribution.
#' @param mpl The maximum possible loss.
#' @param points Number of points, at least 4.
#' @export
tabulated_curve_from_severity <- function(severity, mpl, points = 200) {
  tabulated_curve(ptr = rust_result(Tabulated$from_severity(
    severity@ptr, as.double(mpl), as.double(points)
  )))
}

S7::method(print, tabulated_curve) <- function(x, ...) {
  cat(sprintf("<tabulated_curve> %d points, mean rate %s\n", length(x@x), format(x@mean)))
  invisible(x)
}

#' Risk profile for property per-risk business
#'
#' Bands of sum insured, each with an expected loss (given, or premium times
#' a loss ratio) and its own exposure curve. Each band's representative risk
#' has sum insured `SI` (its total sum insured over its number of risks, say),
#' taken as its MPL; the band expects `EL / (SI * curve@mean)` losses a year.
#'
#' `profile_simulate()` draws years of losses: a Poisson number with the
#' profile's expected count, each in a band with probability proportional to
#' the band's expected count, and the band's `SI` times a destruction rate
#' from its curve. Every loss carries its `SI`, so a [surplus_treaty()] and
#' the per-risk excess of loss it inures to apply with [apply_tower()].
#' `profile_layer_loss()` and `profile_surplus_loss()` are the exposure-rated
#' expectations, which check the simulation.
#'
#' @param sums_insured One per band.
#' @param risks Number of risks per band (for reference).
#' @param curves An [mbbefd] or [tabulated_curve] for every band, or a list
#'   with one per band.
#' @param expected_loss Expected annual loss per band. Give this, or
#'   `premium` with `loss_ratio`.
#' @param premium Premium per band.
#' @param loss_ratio Expected loss ratio: one value, or one per band.
#' @param lower,upper Optional bounds of each band's sums insured, one per
#'   band (`NA` for a band without; a band may start at 0). A band with
#'   bounds spreads its risks' sums insured between them: each simulated
#'   loss draws its own `SI` between the bounds, and the exposure-rated
#'   expectations average over the band, weighted by sum insured.
#' @param spread How a band with bounds spreads its sums insured:
#'   `"uniform"` (its mean `SI` becomes `(lower + upper) / 2`, in place of
#'   `sums_insured`), `"tilted"` to keep `sums_insured` as the mean, so the
#'   spread matches both the bounds and the band's total sum insured (the
#'   density `exp(theta * s)` on the bounds, `theta` solved for that mean;
#'   `sums_insured` must lie strictly between the bounds), or `"auto"`
#'   (tilted when `sums_insured` lies strictly between the bounds, uniform
#'   otherwise). Not given, the spread is uniform with a warning for each
#'   band whose mean moves (as with `"auto"`).
#' @returns `risk_profile()`: a `risk_profile` object with properties
#'   `expected_loss` (all bands), `expected_claims` (per band: losses of
#'   every size, not only those that reach a layer) and
#'   `mean_sums_insured` (per band).
#' @export
#' @examples
#' p <- risk_profile(c(1e6, 10e6), c(800, 50), swiss_re_curve(3),
#'                   premium = c(2e6, 1e6), loss_ratio = 0.6)
#' p@expected_loss
#' ev <- profile_simulate(p, 1000, seed = 7)
#' tw <- inuring_tower(list(list(surplus_treaty("S", 1e6, 4)),
#'                          list(xol_layer("XL", 1e6, 0.5e6))))
#' mean(total(apply_tower(tw, ev)))
#' profile_surplus_loss(p, 1e6, 4)
#' profile_layer_loss(p, 1e6, 0.5e6, surplus_retention = 1e6, surplus_lines = 4)
risk_profile <- S7::new_class(
  "risk_profile",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("RiskProfile"),
    expected_loss = S7::new_property(S7::class_double, getter = function(self) self@ptr$expected_loss()),
    expected_claims = S7::new_property(S7::class_double, getter = function(self) self@ptr$expected_claims()),
    mean_sums_insured = S7::new_property(S7::class_double, getter = function(self) self@ptr$mean_sums_insured())
  ),
  constructor = function(sums_insured, risks, curves, expected_loss = NULL, premium = NULL,
                         loss_ratio = NULL, lower = NULL, upper = NULL,
                         spread = c("uniform", "tilted", "auto")) {
    warn <- missing(spread)
    spread <- match.arg(spread)
    warn <- warn || spread == "auto"
    n <- length(sums_insured)
    if (is.null(lower) != is.null(upper)) stop("give lower and upper together")
    if (!is.list(curves)) curves <- rep(list(curves), n)
    if (is.null(expected_loss) == is.null(premium)) {
      stop("give expected_loss, or premium with a loss_ratio")
    }
    if (!is.null(premium) && is.null(loss_ratio)) stop("premium needs a loss_ratio")
    ptr <- rust_result(RiskProfile$new(
      as.double(sums_insured), as.double(risks), lapply(curves, function(c) c@ptr),
      if (is.null(expected_loss)) double() else as.double(expected_loss),
      if (is.null(premium)) double() else as.double(premium),
      if (is.null(loss_ratio)) double() else as.double(loss_ratio),
      if (is.null(lower)) double() else as.double(lower),
      if (is.null(upper)) double() else as.double(upper),
      spread
    ))
    if (warn) {
      given <- ptr$sums_insured()
      mean <- ptr$mean_sums_insured()
      moved <- which(abs(mean - given) > 1e-9 * pmax(abs(given), 1))
      if (length(moved) > 0) {
        warning(sprintf(
          paste("a band spread uniformly between its bounds no longer has sums_insured as",
                "its mean (%s); expected claims and layer losses use the moved mean. Give",
                "spread = \"tilted\" or \"auto\" to keep it, or spread = \"uniform\" to",
                "silence this"),
          paste(sprintf("band %d: %s -> %s", moved, format(given[moved]), format(mean[moved])),
                collapse = ", ")
        ), call. = FALSE)
      }
    }
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

S7::method(print, risk_profile) <- function(x, ...) {
  cat(sprintf("<risk_profile> %d bands, expected loss %s\n", length(x@expected_claims),
              format(x@expected_loss)))
  invisible(x)
}

#' @rdname risk_profile
#' @param profile A `risk_profile`.
#' @param n_sims Number of simulated years.
#' @param seed Generator seed.
#' @export
profile_simulate <- function(profile, n_sims, seed) {
  event_set(ptr = rust_result(profile@ptr$simulate(as.double(n_sims), as.double(seed))))
}

#' @rdname risk_profile
#' @param limit,attachment A per-risk layer; `limit = Inf` for unlimited.
#' @param surplus_retention,surplus_lines A surplus treaty the layer inures
#'   to, or `NULL`.
#' @export
profile_layer_loss <- function(profile, limit, attachment, surplus_retention = NULL,
                               surplus_lines = NULL) {
  if (is.null(surplus_retention) != is.null(surplus_lines)) {
    stop("give both surplus_retention and surplus_lines, or neither")
  }
  rust_result(profile@ptr$expected_layer_loss(
    as.double(limit), as.double(attachment),
    if (is.null(surplus_retention)) NaN else as.double(surplus_retention),
    if (is.null(surplus_lines)) NaN else as.double(surplus_lines)
  ))
}

#' @rdname risk_profile
#' @param retention,lines A surplus treaty's retention line and lines.
#' @export
profile_surplus_loss <- function(profile, retention, lines) {
  profile@ptr$expected_surplus_loss(as.double(retention), as.double(lines))
}

#' Portfolio for natural allocation
#'
#' A portfolio held as the distribution of its total `X` and each unit's
#' conditional expectation given the total, `kappa_i(x) = E[X_i | X = x]`:
#' the representation of Mildenhall and Major (*Pricing Insurance Risk*,
#' 2022) and CAS Monograph 15 for pricing with limited liability and the
#' natural allocation. See [natural_price()].
#'
#' @param x A matrix of scenario losses, one column per unit (named), a
#'   [predictive_distribution] (each simulation an equally likely
#'   scenario), or a named list of [grid_distribution] objects with one step
#'   for independent units, convolved by FFT.
#' @param probs For a matrix, the scenarios' probabilities; equal when
#'   `NULL`.
#' @param ptr Internal.
#' @returns A `capital_portfolio` with properties `units`, `totals` (the
#'   distinct totals, ascending), `probs` and `expected` (each unit's mean).
#' @export
#' @examples
#' insco <- capital_portfolio(cbind(
#'   A = c(15, 15, 5, 7, 13, 5, 15, 26, 17, 16),
#'   B = c(7, 13, 20, 33, 20, 27, 16, 19, 8, 20),
#'   C = c(0, 0, 11, 0, 7, 8, 9, 10, 40, 64)
#' ))
#' insco@totals
#' natural_price(insco, distortion("ccoc", 0.15))
capital_portfolio <- S7::new_class(
  "capital_portfolio",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("NaturalPortfolio"),
    units = S7::new_property(S7::class_character, getter = function(self) self@ptr$units()),
    totals = S7::new_property(S7::class_double, getter = function(self) self@ptr$totals()),
    probs = S7::new_property(S7::class_double, getter = function(self) self@ptr$probs()),
    expected = S7::new_property(S7::class_double, getter = function(self) self@ptr$expected())
  ),
  constructor = function(x, probs = NULL, ptr = NULL) {
    if (is.null(ptr)) {
      ptr <- if (is.matrix(x)) {
        units <- colnames(x)
        if (is.null(units)) units <- paste0("X", seq_len(ncol(x)))
        rust_result(NaturalPortfolio$from_rows(
          units, as.double(t(x)), if (is.null(probs)) double() else as.double(probs)
        ))
      } else if (is.list(x) && !S7::S7_inherits(x)) {
        units <- names(x)
        if (is.null(units)) units <- paste0("X", seq_along(x))
        rust_result(NaturalPortfolio$from_independent(units, lapply(x, function(g) g@ptr)))
      } else {
        rust_result(NaturalPortfolio$from_predictive(x@ptr))
      }
    }
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

S7::method(print, capital_portfolio) <- function(x, ...) {
  cat(sprintf("<capital_portfolio> %d units, %d distinct totals\n", length(x@units), length(x@totals)))
  invisible(x)
}

portfolio_level <- function(port, assets, p) {
  if (!is.null(assets) && !is.null(p)) stop("give assets or p, not both")
  if (!is.null(assets)) return(as.double(assets))
  if (!is.null(p)) return(rust_result(port@ptr$assets(as.double(p))))
  port@ptr$max()
}

#' Natural allocation of a portfolio's price
#'
#' The premium of the portfolio's loss capped at the assets, `X ∧ a`, under
#' a distortion `g`, and its allocation to the units: each unit's loss
#' (equal priority in default), margin, premium, capital and assets.
#' `"linear"` gives each unit its expected share of the assets above `a`;
#' `"lifted"` the distorted share. Capital is allocated layer by layer, in
#' proportion to each unit's margin in the layer.
#'
#' @param port A [capital_portfolio].
#' @param distortion A [distortion].
#' @param assets The asset level; or give `p`. The largest total by default.
#' @param p The capital standard: assets at the total's lower `p` quantile.
#' @param allocation `"linear"` or `"lifted"`.
#' @returns A data frame, one row per unit and a `"total"` row, with
#'   `loss`, `margin`, `premium`, `capital`, `assets`, `loss_ratio`,
#'   `premium_to_capital` and `return_on_capital`.
#' @export
#' @examples
#' insco <- capital_portfolio(cbind(
#'   A = c(15, 15, 5, 7, 13, 5, 15, 26, 17, 16),
#'   B = c(7, 13, 20, 33, 20, 27, 16, 19, 8, 20),
#'   C = c(0, 0, 11, 0, 7, 8, 9, 10, 40, 64)
#' ))
#' ph <- calibrate_portfolio(insco, "proportional_hazard", return_on_capital = 0.15)
#' natural_price(insco, ph, p = 0.85, allocation = "lifted")
natural_price <- function(port, distortion, assets = NULL, p = NULL,
                          allocation = c("linear", "lifted")) {
  allocation <- match.arg(allocation)
  a <- portfolio_level(port, assets, p)
  as.data.frame(rust_result(port@ptr$price(distortion@ptr, a, allocation)))
}

#' Calibrate a distortion to a portfolio
#'
#' The member of a distortion family that prices the portfolio's loss
#' capped at the assets at a target premium, return on capital
#' (`P = (L + r a) / (1 + r)`) or loss ratio.
#'
#' @inheritParams natural_price
#' @param family As for [calibrate_distortion()].
#' @param premium,return_on_capital,loss_ratio Give exactly one.
#' @param r0 The fixed `r0` of the families that have one.
#' @returns A [distortion].
#' @export
#' @examples
#' insco <- capital_portfolio(cbind(
#'   A = c(15, 15, 5, 7, 13, 5, 15, 26, 17, 16),
#'   B = c(7, 13, 20, 33, 20, 27, 16, 19, 8, 20),
#'   C = c(0, 0, 11, 0, 7, 8, 9, 10, 40, 64)
#' ))
#' calibrate_portfolio(insco, "wang", return_on_capital = 0.15)
calibrate_portfolio <- function(port, family, assets = NULL, p = NULL, premium = NULL,
                                return_on_capital = NULL, loss_ratio = NULL, r0 = 0) {
  given <- list(premium = premium, return_on_capital = return_on_capital, loss_ratio = loss_ratio)
  given <- given[!vapply(given, is.null, logical(1))]
  if (length(given) != 1) stop("give exactly one of premium, return_on_capital and loss_ratio")
  a <- portfolio_level(port, assets, p)
  distortion(ptr = rust_result(port@ptr$calibrate(
    family, a, names(given), as.double(given[[1]]), as.double(r0)
  )))
}

#' Bodoff allocation, EPD and assets
#'
#' `bodoff_allocation()` is Bodoff's percentile layer of capital: each
#' unit's share of the assets, `integral of E[X_i / X | X > x] dx` up to the
#' assets. `epd_ratio()` is the expected policyholder deficit ratio,
#' `(E[X] - E[X ∧ a]) / E[X]`, in total and by unit under equal priority.
#' `assets_for_epd()` is the smallest assets with a total EPD ratio at most
#' `epd`.
#'
#' @inheritParams natural_price
#' @param epd A target EPD ratio in `(0, 1)`.
#' @returns `bodoff_allocation()`: a named numeric vector. `epd_ratio()`: a
#'   list with `total` and `units` (named). `assets_for_epd()`: a number.
#' @export
#' @examples
#' insco <- capital_portfolio(cbind(
#'   A = c(15, 15, 5, 7, 13, 5, 15, 26, 17, 16),
#'   B = c(7, 13, 20, 33, 20, 27, 16, 19, 8, 20),
#'   C = c(0, 0, 11, 0, 7, 8, 9, 10, 40, 64)
#' ))
#' bodoff_allocation(insco, assets = 100)
#' epd_ratio(insco, 65)
#' assets_for_epd(insco, 0.05)
bodoff_allocation <- function(port, assets = NULL, p = NULL) {
  stats::setNames(port@ptr$bodoff(portfolio_level(port, assets, p)), port@units)
}

#' @rdname bodoff_allocation
#' @export
epd_ratio <- function(port, assets) {
  r <- port@ptr$epd(as.double(assets))
  list(total = r$total, units = stats::setNames(r$units, port@units))
}

#' @rdname bodoff_allocation
#' @export
assets_for_epd <- function(port, epd) rust_result(port@ptr$assets_for_epd(as.double(epd)))

#' Priority ladder
#'
#' `priority_ladder()` is each unit's recovery from the assets `a` when it
#' ranks ahead of the other units pooled (senior, `E[X_i ∧ a]`), pro rata
#' (equal, `E[X_i min(1, a / X)]`), or behind them (junior,
#' `E[min(X_i, (a - X_{-i})^+)]`), with its expected policyholder deficit
#' ratio under each, as `aggregate`'s `priority_epd_df`. Each rule reads one
#' unit against the rest pooled, so on three or more units only the equal
#' recoveries add up to `E[X ∧ a]`. Exact under any dependence for a
#' portfolio built from scenarios.
#'
#' `priority_kappa()` says where in the total the shortfall `(X - a)^+`
#' falls when `unit` is junior: at each total `x`, the junior unit's
#' expected part `E[min((X - a)^+, X_i) | X = x]` and the senior pool's,
#' which add up to `(x - a)^+`.
#'
#' @inheritParams natural_price
#' @param unit The junior unit's name.
#' @returns `priority_ladder()`: a data frame with one row per unit and a
#'   last `total` row: `unit`, `mean`, the recoveries `senior`, `equal`,
#'   `junior`, and the EPD ratios `senior_epd`, `equal_epd`, `junior_epd`;
#'   the assets in `attr(, "assets")`. `priority_kappa()`: a data frame with
#'   `total`, `junior` and `senior`.
#' @export
#' @examples
#' port <- capital_portfolio(cbind(A = c(15, 5, 26, 17), B = c(7, 20, 19, 48)))
#' priority_ladder(port, assets = 50)
#' priority_kappa(port, "A", assets = 50)
priority_ladder <- function(port, assets = NULL, p = NULL) {
  a <- portfolio_level(port, assets, p)
  r <- rust_result(port@ptr$priority(a))
  mean <- c(r$mean, sum(r$mean))
  epd <- function(rec) ifelse(mean > 0, (mean - rec) / mean, 0)
  senior <- c(r$senior, r$total)
  equal <- c(r$equal, r$total)
  junior <- c(r$junior, r$total)
  out <- data.frame(
    unit = c(port@units, "total"), mean = mean, senior = senior, equal = equal,
    junior = junior, senior_epd = epd(senior), equal_epd = epd(equal), junior_epd = epd(junior)
  )
  attr(out, "assets") <- a
  out
}

#' @rdname priority_ladder
#' @export
priority_kappa <- function(port, unit, assets = NULL, p = NULL) {
  i <- match(unit, port@units)
  if (is.na(i)) stop("no unit ", unit)
  r <- rust_result(port@ptr$priority_kappa(as.double(i), portfolio_level(port, assets, p)))
  data.frame(total = port@ptr$totals(), junior = r$junior, senior = r$senior)
}

#' Each unit priced alone, and the comonotonic allocation
#'
#' `price_stand_alone()` prices each unit as if it were the whole book, at
#' its own lower `p` quantile (`L = E[X_i ∧ a_i]`, `P = ρ_g(X_i ∧ a_i)`),
#' against the book priced together at the VaR of its total, as
#' `aggregate`'s `Portfolio.price_stand_alone`; the `sum` row adds the
#' stand-alone loss, premium and assets, and its excess over `total` is the
#' diversification benefit.
#'
#' `comonotonic_allocation()` is Denuit et al.'s (2025) convex-order
#' improvement of the conditional mean risk sharing `E[X_i | X]`: each
#' unit's allocation as a non-decreasing function of the total, still adding
#' up to it (`aggregate`'s `make_comonotonic_allocations`).
#'
#' @inheritParams natural_price
#' @param p The VaR capital standard, in `(0, 1)`.
#' @returns `price_stand_alone()`: a data frame as [natural_price()] with
#'   rows for the units, `total` and `sum`. `comonotonic_allocation()`: a
#'   data frame with the `total` and one column per unit.
#' @export
#' @examples
#' port <- capital_portfolio(cbind(a = c(0, 1, 4, 5), b = c(0, 4, 1, 5)))
#' price_stand_alone(port, distortion("proportional_hazard", 0.5), p = 0.75)
#' comonotonic_allocation(capital_portfolio(cbind(A = c(0, 1, 6, 7), B = c(0, 3, 0, 3))))
price_stand_alone <- function(port, distortion, p) {
  as.data.frame(rust_result(port@ptr$price_stand_alone(distortion@ptr, as.double(p))))
}

#' @rdname price_stand_alone
#' @export
comonotonic_allocation <- function(port) {
  k <- stats::setNames(port@ptr$comonotonic_kappa(), port@units)
  data.frame(total = port@totals, k, check.names = FALSE)
}

S7::method(tvar_threshold, capital_portfolio) <- function(x, p, ...) {
  rust_result(tvar_threshold_rust(x@ptr$totals(), x@ptr$probs(), as.double(p)), s7_call())
}

#' Premium, margin and capital pentagon
#'
#' Loss `L`, margin `M`, premium `P`, capital `Q` and assets `a`, with
#' `P = L + M` and `a = P + Q`, from exactly three known amounts or ratios
#' (`loss_ratio = L / P`, `premium_to_capital = P / Q`,
#' `return_on_capital = M / Q`).
#'
#' @param loss,margin,premium,capital,assets,loss_ratio,premium_to_capital,return_on_capital
#'   Give exactly three.
#' @returns A named numeric vector of all eight.
#' @export
#' @examples
#' pentagon(loss = 46.6, assets = 100, return_on_capital = 0.15)
pentagon <- function(loss = NULL, margin = NULL, premium = NULL, capital = NULL, assets = NULL,
                     loss_ratio = NULL, premium_to_capital = NULL, return_on_capital = NULL) {
  given <- list(loss = loss, margin = margin, premium = premium, capital = capital,
                assets = assets, loss_ratio = loss_ratio, premium_to_capital = premium_to_capital,
                return_on_capital = return_on_capital)
  given <- given[!vapply(given, is.null, logical(1))]
  r <- rust_result(pricing_pentagon(names(given), as.double(unlist(given))))
  unlist(r[-1])
}

#' Pricing bounds
#'
#' The range of each unit's premium over every distortion that prices the
#' portfolio's loss, capped at the assets, at `premium`, under the linear
#' natural allocation (Mildenhall and Major, *Pricing Insurance Risk*,
#' chapter 11). The extremes are BiTVaR distortions, found exactly.
#'
#' @inheritParams natural_price
#' @param premium The total premium.
#' @returns A data frame with `unit`, `lower` and `upper`, and list columns
#'   `lower_distortion` and `upper_distortion` holding the [distortion]
#'   that gives each.
#' @export
#' @examples
#' insco <- capital_portfolio(cbind(
#'   A = c(15, 15, 5, 7, 13, 5, 15, 26, 17, 16),
#'   B = c(7, 13, 20, 33, 20, 27, 16, 19, 8, 20),
#'   C = c(0, 0, 11, 0, 7, 8, 9, 10, 40, 64)
#' ))
#' premium_bounds(insco, 53.565, assets = 100)[, c("unit", "lower", "upper")]
premium_bounds <- function(port, premium, assets = NULL, p = NULL) {
  r <- rust_result(port@ptr$premium_bounds(as.double(premium), portfolio_level(port, assets, p)))
  out <- data.frame(unit = port@units, lower = r$lower, upper = r$upper)
  out$lower_distortion <- lapply(r$lower_distortion, function(d) distortion(ptr = d))
  out$upper_distortion <- lapply(r$upper_distortion, function(d) distortion(ptr = d))
  out
}

#' Pricing bounds for other risks
#'
#' `pricing_bounds()` is the range of each risk's price over every
#' distortion that prices the reference risk `x` at `premium`, as the Python
#' package `aggregate`'s `PricingBounds`: the vertical slice at `premium` of
#' the convex hull of the curve `(TVaR_p(x), TVaR_p(y))`, exact for discrete
#' risks. Prices are law-invariant, so only each risk's distribution
#' matters; for a unit's share of a portfolio's price use
#' [premium_bounds()]. `distortion_envelope()` is the smallest and largest
#' `g(s)` over those distortions (`aggregate`'s `Bounds` envelopes, exact):
#' the price range of a Bernoulli risk with `P(1) = s`.
#'
#' @param x The reference risk: a [sampled], [grid_distribution] or
#'   [predictive_distribution] (its total).
#' @param risks A named list of risks to bound, of the same kinds.
#' @param premium The price of `x`, between its mean and maximum.
#' @param assets Cap every risk at the assets (`Inf` for none).
#' @param s Levels in `[0, 1]`.
#' @returns `pricing_bounds()`: a data frame with `risk`, `lower` and
#'   `upper`, and list columns `lower_distortion` and `upper_distortion`
#'   holding a [distortion] attaining each (it prices `x` at `premium`).
#'   `distortion_envelope()`: a data frame with `s`, `lower` (a concave
#'   distortion) and `upper`.
#' @export
#' @examples
#' x <- sampled(c(22, 28, 36, 40, 40, 40, 40, 55, 65, 100))
#' y <- sampled(c(0, 0, 11, 0, 7, 8, 9, 10, 40, 64))
#' pricing_bounds(x, list(X3 = y), 53.565)[, c("risk", "lower", "upper")]
#' distortion_envelope(x, 53.565, c(0.1, 0.5, 0.9))
pricing_bounds <- function(x, risks, premium, assets = Inf) {
  if (is.null(names(risks)) || any(names(risks) == "")) stop("name every risk")
  r <- rust_result(pricing_bounds_rust(
    x@ptr, lapply(risks, function(y) y@ptr), as.double(premium), as.double(assets)
  ))
  out <- data.frame(risk = names(risks), lower = r$lower, upper = r$upper)
  out$lower_distortion <- lapply(r$lower_distortion, function(d) distortion(ptr = d))
  out$upper_distortion <- lapply(r$upper_distortion, function(d) distortion(ptr = d))
  out
}

#' @rdname pricing_bounds
#' @export
distortion_envelope <- function(x, premium, s, assets = Inf) {
  r <- rust_result(distortion_envelope_rust(x@ptr, as.double(premium), as.double(s), as.double(assets)))
  data.frame(s = s, lower = r$lower, upper = r$upper)
}

#' Classical premium principles
#'
#' `classical_premium()` prices a distribution under a classical premium
#' principle with loading `t`; `calibrate_classical()` finds the loading
#' that gives a premium.
#'
#' | `principle` | Premium |
#' |---|---|
#' | `"expected_value"` | `(1 + t) mean` |
#' | `"variance"` | `mean + t var` |
#' | `"standard_deviation"` | `mean + t sd` |
#' | `"semi_variance"` | `mean + t E[(X - mean)_+^2]` |
#' | `"exponential"` | `log(E[exp(t X)]) / t` |
#' | `"esscher"` | `E[X exp(t X)] / E[exp(t X)]` |
#' | `"dutch"` | `mean + t E[(X - mean)_+]` |
#' | `"fischer"` | `mean + t E[(X - mean)_+^q]^(1/q)` |
#' | `"var"` | the lower `t` quantile |
#'
#' @param x A [sampled], [grid_distribution] or [predictive_distribution]
#'   (its total).
#' @param principle One of the principles in the table.
#' @param loading The loading `t`.
#' @param premium The target premium.
#' @param q The Fischer power.
#' @returns A number.
#' @export
#' @examples
#' classical_premium(sampled(c(0, 10)), "standard_deviation", 0.2) # 6
#' insco <- sampled(c(22, 28, 36, 40, 40, 40, 40, 55, 65, 100))
#' calibrate_classical(insco, "esscher", (46.6 + 15) / 1.15)
classical_premium <- function(x, principle, loading, q = 2) {
  rust_result(pricing_classical_premium(x@ptr, principle, as.double(loading), as.double(q)))
}

#' @rdname classical_premium
#' @export
calibrate_classical <- function(x, principle, premium, q = 2) {
  rust_result(pricing_calibrate_classical(x@ptr, principle, as.double(premium), as.double(q)))
}

#' A cover's quote
#'
#' `technical_quote()` grosses a risk-loaded price up to the premium the
#' cedant pays. The risk premium `R` (the expected loss plus the margin for
#' capital, from [risk_loaded_price()]) is grossed up for internal expenses
#' `e` and brokerage `b`, both fractions of the premium, and a fixed expense
#' `f`: `P = (R + f) / (1 - e - b)`. `implied_quote()` reads a quoted
#' premium the other way: the reinsurer keeps `P (1 - e - b) - f` as its
#' risk premium, and the margin and return on capital follow with the same
#' expected loss and assets.
#'
#' @param price A risk-loaded price, from [risk_loaded_price()].
#' @param expenses Internal expenses, a fraction of the premium.
#' @param brokerage Brokerage, a fraction of the premium.
#' @param fixed A fixed expense per contract, an amount.
#' @param limit The cover's limit at the placed share, for the rate on line;
#'   `NULL` for a cover without one, such as a quota share.
#' @param quote A quote from `technical_quote()`.
#' @param premium A quoted premium.
#' @returns A list with `expected_loss`, `margin`, `risk_premium`,
#'   `expenses`, `brokerage` (amounts), `premium`, `net_premium` (net of
#'   brokerage), `assets`, `capital`, `return_on_capital`, `loss_ratio`,
#'   `multiple`, `rate_on_line` and `payback` (`NaN` without a limit), and
#'   the inputs `implied_quote()` needs.
#' @export
#' @examples
#' p <- risk_loaded_price(sampled(c(0, 0, 2, 6)), distortion("tvar", 0.5),
#'                        cost_of_capital = 0.1)
#' q <- technical_quote(p, expenses = 0.03, brokerage = 0.10, limit = 10)
#' q$premium
#' q$rate_on_line
#' implied_quote(q, 3)$return_on_capital
technical_quote <- function(price, expenses = 0, brokerage = 0, fixed = 0, limit = NULL) {
  loads <- list(expenses = as.double(expenses), brokerage = as.double(brokerage),
                fixed = as.double(fixed), limit = if (is.null(limit)) NaN else as.double(limit))
  q <- rust_result(pricing_quote(
    price$expected_loss, price$premium, price$assets, loads$expenses, loads$brokerage,
    loads$fixed, loads$limit, NaN
  ))
  c(q, list(loads = loads))
}

#' @rdname technical_quote
#' @export
implied_quote <- function(quote, premium) {
  l <- quote$loads
  q <- rust_result(pricing_quote(
    quote$expected_loss, quote$risk_premium, quote$assets, l$expenses, l$brokerage, l$fixed,
    l$limit, as.double(premium)
  ))
  c(q, list(loads = l))
}

#' Large losses trended to the prospective period
#'
#' The first step of experience rating from individual large losses
#' (Clark, *Basics of Reinsurance Pricing*): develop each known loss to
#' ultimate (IBNER), if asked, and trend it from its accident date to the
#' prospective period, by an annual `rate` to the date `to`
#' (`(1 + rate)^(to - accident)`) or by a factor per loss. Cost the
#' trended losses with [burning_cost()].
#'
#' A record reported above a threshold in nominal terms leaves the older
#' years short of losses that trend above it: with a `threshold`, losses
#' trended below the threshold times the largest factor (`complete_above`)
#' are left out of the burning cost, so every year is complete on the same
#' basis.
#'
#' @param losses A data frame with columns `id`, `accident` (a decimal
#'   year, see [year_fraction()]) and `amount` (incurred at the latest
#'   valuation).
#' @param rate An annual severity trend, with `to`.
#' @param to The date to trend to, a decimal year: the average accident
#'   date of the prospective period.
#' @param factors A trend factor per loss, instead of `rate` and `to`.
#' @param threshold The reporting threshold of the record.
#' @param valuation The date the amounts were valued at, a decimal year;
#'   needed to develop by a [development_pattern()].
#' @param development Develops each known loss to ultimate: a
#'   [development_pattern()], read at each loss's age at `valuation`, or a
#'   factor per loss. Then give [burning_cost()] only the count IBNR
#'   factors, not an excess factor that already holds IBNER.
#' @param fx Converts the losses to the pricing currency before development
#'   and trend: one stated rate, or a rate per loss (units of the pricing
#'   currency per unit of the loss's). The threshold stays in the reported
#'   currency.
#' @returns A list with `losses` (the input columns with `year`, `fx`,
#'   `development`, `ultimate`, `factor` and `trended`), `complete_above`,
#'   and the inputs [burning_cost()] needs.
#' @export
#' @examples
#' losses <- data.frame(id = c("a", "b"), accident = c(1995.5, 1996.5),
#'                      amount = c(150, 400))
#' p <- trend_large_losses(losses, rate = 0.05, to = 1998.5)
#' p$losses$trended
trend_large_losses <- function(losses, rate = NULL, to = NULL, factors = NULL, threshold = 0,
                               valuation = NULL, development = NULL, fx = NULL) {
  if (is.null(factors) == (is.null(rate) || is.null(to))) {
    stop("pass rate and to, or factors, but not both", call. = FALSE)
  }
  args <- list(
    ids = as.character(losses$id), accident = as.double(losses$accident),
    amount = as.double(losses$amount), threshold = as.double(threshold),
    rate = if (is.null(rate)) NaN else as.double(rate),
    to = if (is.null(to)) NaN else as.double(to),
    factors = if (is.null(factors)) double() else as.double(factors),
    valuation = if (is.null(valuation)) NaN else as.double(valuation),
    dev_ages = if (inherits(development, "development_pattern")) development$ages else double(),
    dev_factors = if (inherits(development, "development_pattern")) development$factors else double(),
    dev_per_loss = if (is.null(development) || inherits(development, "development_pattern")) {
      double()
    } else {
      as.double(development)
    },
    fx = if (is.null(fx)) double() else as.double(fx)
  )
  r <- rust_result(do.call(pricing_trend_large_losses, args))
  list(
    losses = data.frame(id = r$id, year = r$year, accident = r$accident, amount = r$amount,
                        fx = r$fx, development = r$development, ultimate = r$ultimate,
                        factor = r$factor, trended = r$trended),
    complete_above = r$complete_above,
    args = args
  )
}

#' Burning cost of large losses
#'
#' Apply each year's treaty to the year's trended losses (a loss belongs to
#' the year of its accident date), with any annual terms, develop each
#' year's layer total with an excess development factor, and divide by the
#' year's on-level premium. The overall loss cost is the developed losses
#' over the premium, summed over the years (Clark's convention);
#' `mean_loss_cost` weights the years equally.
#'
#' @param prepared Trended losses, from [trend_large_losses()].
#' @param towers The [reinsurance_tower] in force each year, as a list, or
#'   one tower for every year (to price the prospective terms). Every tower
#'   needs the same number of layers.
#' @param years The accident years to cost.
#' @param premium Each year's on-level subject premium.
#' @param development Each year's development factor for the layer losses:
#'   Clark's excess factor, or, when the losses were developed one by one in
#'   [trend_large_losses()], the count IBNR factor only. A count
#'   [development_pattern()] is read at each year's average age at the
#'   record's valuation date.
#' @param index An [index_clause()] on the treaty: each loss meets the
#'   retention and limit moved by the clause at its own ratio in `ratios`.
#'   Towers with annual terms (an aggregate deductible or limit, a corridor)
#'   or a surplus treaty are an error.
#' @param ratios With `index`, each loss's index ratio, in the record's
#'   order: the index at its payment over the index at the treaty's base
#'   date.
#' @returns A list with `years`, a data frame (`year`, `premium`, `losses`
#'   (the count through the tower), `development`), the year-by-layer
#'   matrices `layer_loss`, `developed` and `year_loss_cost`, and per layer
#'   `loss_cost` and `mean_loss_cost`.
#' @export
#' @examples
#' losses <- data.frame(id = c("a", "b"), accident = c(1995.5, 1996.5),
#'                      amount = c(150, 400))
#' p <- trend_large_losses(losses, factors = c(1, 1))
#' tw <- reinsurance_tower(list(xol_layer("200x100", 200, 100)))
#' b <- burning_cost(p, tw, c(1995, 1996), premium = c(1000, 1000),
#'                   development = c(1, 1.5))
#' b$developed
#' b$loss_cost
burning_cost <- function(prepared, towers, years, premium, development, index = NULL,
                         ratios = NULL) {
  if (is.null(index) != is.null(ratios)) {
    stop("pass index and ratios together, or neither", call. = FALSE)
  }
  if (S7::S7_inherits(towers, reinsurance_tower)) towers <- rep(list(towers), length(years))
  if (inherits(development, "development_pattern")) {
    development <- rust_result(pricing_count_development(
      development$ages, development$factors, as.integer(years), prepared$args$valuation
    ))
  }
  args <- c(prepared$args, list(
    towers = lapply(towers, function(t) t@ptr), years = as.integer(years),
    premium = as.double(premium), development = as.double(development),
    index_franchise = if (is.null(index)) NaN else index$franchise,
    index_kind = if (is.null(index)) "full" else index$kind,
    ratios = if (is.null(ratios)) double() else as.double(ratios)
  ))
  r <- rust_result(do.call(pricing_burning_cost, args))
  n <- length(r$year)
  layers <- length(r$loss_cost)
  m <- function(x) matrix(x, nrow = n, ncol = layers, dimnames = list(r$year, NULL))
  list(
    years = data.frame(year = r$year, premium = r$premium, losses = r$losses,
                       development = r$development),
    layer_loss = m(r$layer_loss),
    developed = m(r$developed),
    year_loss_cost = m(r$year_loss_cost),
    loss_cost = r$loss_cost,
    mean_loss_cost = r$mean_loss_cost
  )
}

#' Development pattern
#'
#' Development factors to ultimate by age, for developing each known large
#' loss to ultimate (IBNER) in [trend_large_losses()] or as a count pattern
#' for the claims not yet reported in [burning_cost()]. The factor at an
#' age interpolates linearly between the given ages, takes the first factor
#' at younger ages and the last at older ones ([pattern_factor()]).
#'
#' `development_pattern_from_history()` estimates one from each claim's
#' valuations: the age-to-age factor from `ages[k]` to `ages[k + 1]` is the
#' sum of the amounts at the later age over the sum at the earlier, over the
#' claims valued at both with a positive earlier amount (a claim first seen
#' later is IBNR, not IBNER). The factor at the last age is 1. With every
#' claim valued at every age up to its latest, this is the volume-weighted
#' chain ladder on the summed triangle.
#'
#' @param ages Strictly increasing ages, in years since the accident date.
#' @param factors The factor to ultimate at each age.
#' @param valuations A data frame with columns `id`, `age` and `amount`, one
#'   row per valuation of a claim; ages not in `ages` are ignored.
#' @returns A `development_pattern`: a list with `ages` and `factors`.
#' @export
#' @examples
#' p <- development_pattern_from_history(
#'   data.frame(id = c("a", "a", "b", "b"), age = c(1, 2, 1, 2),
#'              amount = c(100, 150, 50, 50)),
#'   ages = c(1, 2)
#' )
#' p$factors
#' pattern_factor(p, 1.5)
development_pattern <- function(ages, factors) {
  ages <- as.double(ages)
  factors <- as.double(factors)
  rust_result(pricing_pattern_at(ages, factors, double()))
  structure(list(ages = ages, factors = factors), class = "development_pattern")
}

#' @rdname development_pattern
#' @export
development_pattern_from_history <- function(valuations, ages) {
  r <- rust_result(pricing_development_pattern(
    as.character(valuations$id), as.double(valuations$age), as.double(valuations$amount),
    as.double(ages)
  ))
  structure(list(ages = r$ages, factors = r$factors), class = "development_pattern")
}

#' @rdname development_pattern
#' @param pattern A `development_pattern`.
#' @param age Ages at which to read the factor to ultimate.
#' @export
pattern_factor <- function(pattern, age) {
  rust_result(pricing_pattern_at(pattern$ages, pattern$factors, as.double(age)))
}

#' Index clause
#'
#' An index (stability) clause: the treaty's retention and limit move with
#' an index, by the ratio `r` of the index at a loss's payment to the index
#' at the treaty's base date, within a franchise. With `kind = "full"`,
#' nothing moves while `|r - 1|` is within the franchise, and beyond it the
#' terms move by all of `r`, up or down. With `kind = "severe"` (a severe
#' inflation clause), the terms move only by the rise beyond the franchise,
#' the ratio over `1 + franchise`, and never down. Use it in [burning_cost()].
#'
#' @param franchise A fraction, such as 0.1 for 10%.
#' @param kind `"full"` or `"severe"`.
#' @returns An `index_clause`: a list with `franchise` and `kind`.
#' @export
#' @examples
#' losses <- data.frame(id = "a", accident = 2020.5, amount = 700)
#' p <- trend_large_losses(losses, factors = 1)
#' tw <- reinsurance_tower(list(xol_layer("400 xs 100", 400, 100)))
#' # Index up 25%: 500 xs 125, so the loss cedes 500.
#' burning_cost(p, tw, 2020, premium = 1000, development = 1,
#'              index = index_clause(0.1), ratios = 1.25)$layer_loss
index_clause <- function(franchise, kind = c("full", "severe")) {
  kind <- match.arg(kind)
  franchise <- as.double(franchise)
  if (!(is.finite(franchise) && franchise >= 0)) {
    stop("franchise must be finite and non-negative", call. = FALSE)
  }
  structure(list(franchise = franchise, kind = kind), class = "index_clause")
}

#' As-if history of large losses
#'
#' Each year's losses under the program in force that year, next to the
#' year's trended losses under the proposed tower. The historical side takes
#' every loss of the year as reported (in its own currency, untrended and
#' undeveloped), so it can be checked against the cedant's recoveries; the
#' proposed side is the as-if layer loss of [burning_cost()] before
#' development by year. The historical towers may differ in layers from year
#' to year.
#'
#' @param prepared Trended losses, from [trend_large_losses()].
#' @param historical A list of [reinsurance_tower] objects, the program in
#'   force in each of `years`.
#' @param years The accident years.
#' @param proposed The proposed [reinsurance_tower].
#' @returns A list with `years`, a data frame (`year`, `reported_losses`,
#'   `trended_losses`), `historical`, a list per year of each layer's
#'   cession named by layer, and `proposed`, a year-by-layer matrix.
#' @export
#' @examples
#' p <- trend_large_losses(data.frame(id = "a", accident = 2019.5, amount = 300),
#'                         factors = 1.2)
#' then <- reinsurance_tower(list(xol_layer("150 xs 150", 150, 150)))
#' now <- reinsurance_tower(list(xol_layer("200 xs 200", 200, 200)))
#' a <- as_if_history(p, list(then), 2019, now)
#' a$historical
#' a$proposed
as_if_history <- function(prepared, historical, years, proposed) {
  args <- c(prepared$args, list(
    historical = lapply(historical, function(t) t@ptr), years = as.integer(years),
    proposed = proposed@ptr
  ))
  r <- rust_result(do.call(pricing_as_if, args))
  list(
    years = data.frame(year = r$year, reported_losses = r$reported_losses,
                       trended_losses = r$trended_losses),
    historical = stats::setNames(
      Map(function(v, n) stats::setNames(v, n), r$historical, r$historical_layers),
      r$year
    ),
    proposed = matrix(r$proposed, nrow = length(r$year), ncol = length(r$proposed_layers),
                      dimnames = list(r$year, r$proposed_layers))
  )
}

#' Payment pattern
#'
#' The share of the ultimate loss paid at each time, in years from the
#' treaty's inception. `payment_pattern_from_development()` takes
#' development factors to ultimate by age, such as a paid chain ladder's
#' `cdf`: the share paid by each age is one over its factor, and the last
#' factor must be 1 (add any tail as a last age). Ages are read as times from
#' inception; add a shift to `times` when losses start developing later.
#'
#' `pattern_values()` gives the discount factor at an annual effective
#' `rate` (the present value of a unit of ultimate loss), the discounted
#' capital years (capital held in each year in proportion to the share
#' outstanding at its start, charged at the start of the year) and the share
#' outstanding at times `t`. With a pattern that does not depend on the size
#' of the loss, every simulated year's present value is its loss times the
#' discount factor.
#'
#' @param times Non-negative, strictly increasing times, in years.
#' @param shares The share paid at each time; they sum to 1.
#' @param ages Development ages.
#' @param factors Development factors to ultimate at each age.
#' @param pattern A `payment_pattern`.
#' @param rate Annual effective discount rate.
#' @param t Times at which to read the share outstanding.
#' @returns `payment_pattern()` and `payment_pattern_from_development()`: a
#'   `payment_pattern`, a list with `times`, `shares` and `duration` (the
#'   mean time to payment). `pattern_values()`: a list with
#'   `discount_factor`, `capital_years` and `outstanding`.
#' @export
#' @examples
#' p <- payment_pattern_from_development(c(1, 2, 3), c(4, 1.25, 1))
#' p$shares
#' pattern_values(p, rate = 0.05, t = 1)
payment_pattern <- function(times, shares) {
  r <- rust_result(pricing_payment_pattern(as.double(times), as.double(shares), FALSE))
  structure(r, class = "payment_pattern")
}

#' @rdname payment_pattern
#' @export
payment_pattern_from_development <- function(ages, factors) {
  r <- rust_result(pricing_payment_pattern(as.double(ages), as.double(factors), TRUE))
  structure(r, class = "payment_pattern")
}

#' @rdname payment_pattern
#' @export
pattern_values <- function(pattern, rate = 0, t = double()) {
  rust_result(pricing_pattern_values(pattern$times, pattern$shares, as.double(rate),
                                     as.double(t)))
}

#' Price over a payout
#'
#' A one-period price restated over a payout. With a cost of capital `r`,
#' the premium solves `P = E[X] d + r (a - P) A`: the loss discounted by the
#' pattern's discount factor `d`, plus the cost of the initial capital held
#' while losses are outstanding over the discounted capital years `A`. With a
#' pricing distortion, the premium is discounted with the losses. Give the
#' rule the price was made with. A pattern that pays everything within the
#' first year, at a zero rate, gives back the one-period price.
#'
#' @param price A one-period price, from [risk_loaded_price()].
#' @param pattern A [payment_pattern()].
#' @param rate Annual effective discount rate.
#' @param cost_of_capital The price's cost of capital; `NULL` for a price
#'   made with a pricing distortion.
#' @returns A list with `expected_loss`, `discounted_loss`, `premium`,
#'   `assets`, `capital`, `margin` (over the discounted loss),
#'   `discount_factor` and `capital_years`.
#' @export
#' @examples
#' p <- risk_loaded_price(sampled(c(0, 0, 2, 6)), distortion("tvar", 0.5),
#'                        cost_of_capital = 0.25)
#' slow <- payment_pattern(c(1, 2, 3), c(0.2, 0.3, 0.5))
#' price_over_payout(p, slow, cost_of_capital = 0.25)$premium
price_over_payout <- function(price, pattern, rate = 0, cost_of_capital = NULL) {
  rust_result(pricing_price_over_payout(
    price$expected_loss, price$premium, price$assets, pattern$times, pattern$shares,
    as.double(rate), if (is.null(cost_of_capital)) NaN else as.double(cost_of_capital)
  ))
}
