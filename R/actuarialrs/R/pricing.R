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
  package = "actuarialrs",
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
  package = "actuarialrs",
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
                           rule = c("minimize", "midpoint")) {
  rule <- match.arg(rule)
  col <- function(df, name) if (is.null(df)) double() else as.double(df[[name]])
  tower_model(rust_result(TowerModel$fit_references(
    col(layers, "limit"), col(layers, "attachment"), col(layers, "expected_loss"),
    col(frequencies, "threshold"), col(frequencies, "frequency"), as.double(default_alpha), rule
  )))
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
#' `risk_loaded_price()` prices one cover from its loss draws;
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
#' @param x A [sampled] or a [predictive_distribution] (its total) for
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
