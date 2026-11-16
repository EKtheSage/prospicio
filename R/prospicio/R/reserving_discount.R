#' @include distributions.R reserving.R
NULL

#' A discount curve
#'
#' Annual effective spot rates by maturity in years, annually compounded:
#' linear in the rate between maturities and flat before the first and
#' after the last (issue #553; design note `docs/design/reserving-ifrs17.md`).
#' The discount factor of a payment at `t` years is `(1 + r(t))^-t`.
#' `discount_curve_factor()` and `discount_curve_rate()` read the curve at
#' `t`.
#'
#' @param maturities Maturities in years: finite, positive and strictly
#'   increasing. A flat curve is one maturity (the rate holds at every
#'   maturity).
#' @param rates The spot rate at each maturity: finite and above -1.
#' @param curve A `discount_curve`.
#' @param t Times in years.
#' @returns `discount_curve()`: a `discount_curve` with the properties
#'   `maturities` and `rates`. `discount_curve_factor()` and
#'   `discount_curve_rate()`: a numeric vector, one value per `t`.
#' @export
#' @examples
#' curve <- discount_curve(c(1, 5), c(0.02, 0.04))
#' discount_curve_rate(curve, 3)
#' discount_curve_factor(curve, c(1, 2))
discount_curve <- S7::new_class(
  "discount_curve",
  package = "prospicio",
  properties = list(
    maturities = S7::class_double,
    rates = S7::class_double
  ),
  constructor = function(maturities, rates) {
    maturities <- as.double(maturities)
    rates <- as.double(rates)
    reserving_result(discount_factor_rust(maturities, rates, numeric(0)))
    S7::new_object(S7::S7_object(), maturities = maturities, rates = rates)
  }
)

S7::method(print, discount_curve) <- function(x, ...) {
  cat(sprintf("discount_curve(maturities = c(%s), rates = c(%s))\n",
              paste(format(x@maturities, digits = 15), collapse = ", "),
              paste(format(x@rates, digits = 15), collapse = ", ")))
  invisible(x)
}

curve_arg <- function(curve) {
  if (!S7::S7_inherits(curve, discount_curve)) {
    stop("curve must be a discount_curve()", call. = FALSE)
  }
  curve
}

#' @rdname discount_curve
#' @export
discount_curve_factor <- function(curve, t) {
  curve <- curve_arg(curve)
  reserving_result(discount_factor_rust(curve@maturities, curve@rates, as.double(t)))
}

#' @rdname discount_curve
#' @export
discount_curve_rate <- function(curve, t) {
  curve <- curve_arg(curve)
  reserving_result(discount_rate_rust(curve@maturities, curve@rates, as.double(t)))
}

#' Discounted reserves and the IFRS 17 risk adjustment
#'
#' A stochastic fit's unpaid claims discounted simulation by simulation, by
#' origin and in total, and a risk adjustment for non-financial risk read off
#' the discounted total with the confidence level it corresponds to (IFRS 17
#' para. 119; issue #553, design note `docs/design/reserving-ifrs17.md`).
#'
#' * `discounted_reserves()` spreads each simulation's unpaid over that
#'   simulation's development pattern (the rises of its running maximum after
#'   each origin's latest age; with a tail, the tail's share one period past
#'   the oldest age) and discounts every payment on the curve. A
#'   [compartmental()] fit is paid by its simulated paid cells instead.
#'   Quantiles of the discounted total come from the discounted draws, never
#'   from scaling undiscounted quantiles.
#' * `basis` has no default (decision 4b): `"paid"` for a fit on paid claims,
#'   paid along its own patterns; `"incurred"` for a fit on incurred claims,
#'   where each reserve (the IBNR) plus the origin's case reserve (the fit's
#'   latest incurred less `latest_paid`) is paid along `paid_pattern`.
#' * `risk_adjustment()` is `"confidence"`, `Q_p(X) - BE` with the
#'   inverted-cdf quantile; `"tvar"`, `TVaR_p(X) - BE`; or
#'   `"cost_of_capital"`, `rate` times the capital held in each future year,
#'   discounted, the capital running off with the best estimate
#'   (`K_t = K_0 runoff[t] / runoff[1]`), charged at the end of each year
#'   (Solvency II's risk margin, the default) or at its start.
#'
#' Supported fits: [odp_bootstrap()], [mack_bootstrap()], [bayes_odp()],
#' [bayes_odp_segments()] fits, [bayes_bf()], [bayes_bf_normalised()],
#' [meyers()] and its fits over segments, [compartmental()] and
#' [compartmental_segments()]. Analytic [mack()] (no simulations) and
#' [bayes_hierarchical_bf()] (no pattern per simulation) raise a
#' `prospicio_not_supported` error saying why.
#'
#' A `discounted_reserves` result has the properties `unpaid` and
#' `discounted` (predictive distributions on the fit's components and
#' simulations), `best_estimate` (the mean discounted total), `runoff` (the
#' best estimate outstanding at `t = 0, 1, ...` whole years, ending with 0),
#' `discount_factors` (each component's mean present value over its mean
#' unpaid; 1 where that is 0), `curve`, `timing` and `segment_labels` (one
#' character vector per segment of a fit over segments).
#' `segment_runoff(d, i)` is segment `i`'s run-off, as `runoff`, with `i`
#' counted from 1 in the order of `segment_labels` (a fit over two or more
#' segments only).
#'
#' @param fit A stochastic reserving fit (see Details).
#' @param basis `"paid"` or `"incurred"`: what the fit's column is.
#' @param rate A flat annual effective rate. Give `rate` or `curve`.
#' @param curve A [discount_curve()].
#' @param timing `"mid"` (the default) or `"end"`: payments at the middle or
#'   the end of each development period.
#' @param paid_pattern For `basis = "incurred"`: the cumulative paid share at
#'   each of the fit's pattern ages (with the tail's age when the fit has a
#'   tail), ending at 1.
#' @param latest_paid For `basis = "incurred"`: each origin's latest paid, one
#'   per component of the fit's reserves in their order (segment-major for a
#'   fit over segments).
#' @param d A `discounted_reserves` result.
#' @param method `"confidence"`, `"tvar"` or `"cost_of_capital"`.
#' @param p The level of `"confidence"` and `"tvar"`, in `[0, 1]`.
#' @param capital `K_0` of `"cost_of_capital"`, with no default: `c(var = q)`
#'   (`Q_q(X) - BE`), `c(tvar = q)`, or an amount (a number or
#'   `c(amount = k)`), such as the one-year view's capital, the mean claims
#'   development result less its 0.5% quantile (the 99.5% loss less the
#'   expected loss). An amount is the total's, so no segment results are
#'   reported for it.
#' @param i A segment, counted from 1 in the order of `segment_labels`.
#' @param charge `"end"` (the default) or `"start"` of each year.
#' @returns `discounted_reserves()`: a `discounted_reserves` result.
#'   `segment_runoff()`: a numeric vector.
#'   `risk_adjustment()`: a list with `amount`, `best_estimate`,
#'   `implied_confidence` (the share of discounted draws at or below
#'   `best_estimate + amount`), `capital` (`K_0`, or `NULL`) and `segments`, a
#'   data frame of each segment's stand-alone result (no rows for a fit of one
#'   segment).
#' @name discounted_reserves
#' @export
#' @examples
#' long <- data.frame(year = rep(2019:2022, 4:1),
#'                    age = c(12, 24, 36, 48, 12, 24, 36, 12, 24, 12),
#'                    paid = c(100, 150, 170, 175, 110, 160, 185, 120, 185, 130))
#' fit <- mack_bootstrap(triangle(long, "year", "age", "paid"), n_sims = 1000, seed = 1)
#' d <- discounted_reserves(fit, basis = "paid", rate = 0.03)
#' d@best_estimate
#' d@discount_factors
#' risk_adjustment(d, "confidence", p = 0.75)$amount
#' risk_adjustment(d, "cost_of_capital", rate = 0.06, capital = c(var = 0.995))$amount
discounted_reserves <- function(fit, basis, rate = NULL, curve = NULL, timing = c("mid", "end"),
                                paid_pattern = NULL, latest_paid = NULL) {
  if (missing(basis)) {
    stop("basis is required: \"paid\" for a fit on paid claims, \"incurred\" (with ",
         "paid_pattern and latest_paid) for a fit on incurred claims", call. = FALSE)
  }
  basis <- match.arg(basis, c("paid", "incurred"))
  timing <- match.arg(timing)
  if (is.null(rate) == is.null(curve)) {
    stop("give exactly one of rate (a flat annual rate) and curve (a discount_curve())",
         call. = FALSE)
  }
  if (is.null(curve)) {
    curve <- discount_curve(1, single_number(rate, "rate"))
  }
  curve <- curve_arg(curve)
  ptr <- if (S7::S7_inherits(fit) && "ptr" %in% S7::prop_names(fit)) fit@ptr else fit
  num <- function(x) if (is.null(x)) NULL else as.double(x)
  res <- reserving_result(discounted_reserves_rust(ptr, basis, curve@maturities, curve@rates,
                                                   timing, num(paid_pattern),
                                                   num(latest_paid)))
  discounted_reserves_class(ptr = res)
}

component_names <- function(pd) {
  keys <- pd@keys
  if (ncol(keys) == 0) {
    return(NULL)
  }
  apply(keys, 1, function(k) paste(trimws(k), collapse = "/"))
}

#' @rdname discounted_reserves
#' @export
discounted_reserves_class <- S7::new_class(
  "discounted_reserves",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("DiscountedReserves"),
    unpaid = predictive_distribution,
    discounted = predictive_distribution,
    best_estimate = S7::new_property(S7::class_double, getter = function(self) {
      self@ptr$best_estimate()
    }),
    runoff = S7::new_property(S7::class_double, getter = function(self) self@ptr$runoff()),
    discount_factors = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(self@ptr$discount_factors(), component_names(self@unpaid))
    }),
    curve = S7::new_property(S7::class_any, getter = function(self) {
      discount_curve(self@ptr$maturities(), self@ptr$rates())
    }),
    timing = S7::new_property(S7::class_character, getter = function(self) self@ptr$timing()),
    segment_labels = S7::new_property(S7::class_list, getter = function(self) {
      self@ptr$segment_labels()
    })
  ),
  constructor = function(ptr) {
    S7::new_object(S7::S7_object(), ptr = ptr,
                   unpaid = predictive_distribution(ptr = ptr$unpaid()),
                   discounted = predictive_distribution(ptr = ptr$discounted()))
  }
)

#' @rdname discounted_reserves
#' @export
segment_runoff <- function(d, i) {
  if (!S7::S7_inherits(d, discounted_reserves_class)) {
    stop("d must be the result of discounted_reserves()", call. = FALSE)
  }
  reserving_result(d@ptr$segment_runoff(single_number(i, "i")))
}

S7::method(print, discounted_reserves_class) <- function(x, ...) {
  cat(sprintf("discounted_reserves: best estimate %s (undiscounted %s), timing %s\n",
              format(x@best_estimate, digits = 7), format(mean(x@unpaid), digits = 7),
              x@timing))
  invisible(x)
}

#' @rdname discounted_reserves
#' @export
risk_adjustment <- function(d, method = c("confidence", "tvar", "cost_of_capital"), p = NULL,
                            rate = NULL, capital = NULL, charge = c("end", "start")) {
  if (!S7::S7_inherits(d, discounted_reserves_class)) {
    stop("d must be the result of discounted_reserves()", call. = FALSE)
  }
  method <- match.arg(method)
  charge <- match.arg(charge)
  kind <- NULL
  if (method == "cost_of_capital") {
    if (!is.null(p)) {
      stop("method = \"cost_of_capital\" takes capital, not p: capital = c(var = q)",
           call. = FALSE)
    }
    if (!is.null(capital)) {
      if (!is.numeric(capital) || length(capital) != 1) {
        stop("capital must be c(var = q), c(tvar = q), c(amount = k) or a number",
             call. = FALSE)
      }
      if (!is.null(names(capital)) && nzchar(names(capital))) kind <- names(capital)
      capital <- unname(as.double(capital))
    }
    if (!is.null(rate)) rate <- single_number(rate, "rate")
  } else if (!is.null(rate) || !is.null(capital)) {
    stop("rate and capital are for method = \"cost_of_capital\"", call. = FALSE)
  }
  if (!is.null(p)) p <- single_number(p, "p")
  r <- reserving_result(d@ptr$risk_adjustment(method, p, rate, kind, capital, charge))
  seg <- r$segments
  r$segments <- data.frame(
    segment = vapply(seg, function(s) paste(s$label, collapse = "/"), ""),
    amount = vapply(seg, function(s) s$amount, 0),
    best_estimate = vapply(seg, function(s) s$best_estimate, 0),
    implied_confidence = vapply(seg, function(s) s$implied_confidence, 0),
    capital = vapply(seg, function(s) if (is.null(s$capital)) NA_real_ else s$capital, 0),
    stringsAsFactors = FALSE
  )
  r
}
