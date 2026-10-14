#' @include distributions.R pareto.R
NULL

#' Distribution on finitely many values, of either sign
#'
#' Probabilities `probs` on `values`, any sign and any spacing: a row of a
#' P&L (premium less loss less expenses) or any other signed outcome, as the
#' Python package `aggregate`'s `GridDistribution` on an irregular support.
#' Values may repeat and come in any order: equal values (to `1e-12`) are
#' merged, zero probabilities dropped, and the probabilities rescaled to sum
#' to 1.
#'
#' Supports [mean()], [variance()], [cdf()], [survival()], [pmf()],
#' `quantile()` (the lower quantile), [VaR()] and [TVaR()], following
#' `aggregate` exactly. `TVaR()` is the upper tail even when the values are a
#' payoff, whose adverse tail is the low one. `discrete_skewness()` is `NA`
#' when the variance is 0; `quantile_upper()` is the smallest value with
#' `F > p` (`NA` at `p = 1`); `return_periods()` is `aggregate`'s
#' return-period table, each period read from both tails (a payoff off the
#' low rows, a loss off the high ones).
#'
#' @param values Numeric values.
#' @param probs Non-negative probabilities with a positive total.
#' @param ptr Internal.
#' @param dist A `discrete_distribution`.
#' @param p Probabilities in `[0, 1]`.
#' @param periods Return periods, at least 1; `NULL` for `aggregate`'s 2, 5,
#'   10, 25, 50, 100, 200, 250, 500 and 1000.
#' @returns `discrete_distribution()` a `discrete_distribution`, which
#'   inherits from [distribution]; `return_periods()` a data frame with `p`,
#'   `period`, `var`, `tvar`, `excess` (VaR less the mean) and `var_to_mean`.
#' @export
#' @examples
#' # A margin: 9 less the loss of two claims of 1, 2 or 4.
#' m <- discrete_distribution(c(7, 6, 5, 4, 3, 1), c(0.25, 0.3, 0.09, 0.2, 0.12, 0.04))
#' mean(m)
#' quantile(m, 0.01) # the 1-in-100 bad year
#' TVaR(m, 0.01)
#' head(return_periods(m), 3)
discrete_distribution <- S7::new_class(
  "discrete_distribution",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("DiscreteDist"),
    values = S7::new_property(S7::class_double, getter = function(self) self@ptr$values()),
    probs = S7::new_property(S7::class_double, getter = function(self) self@ptr$probs())
  ),
  constructor = function(values, probs, ptr = NULL) {
    if (is.null(ptr)) ptr <- rust_result(DiscreteDist$new(as.double(values), as.double(probs)))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' @rdname discrete_distribution
#' @export
discrete_skewness <- function(dist) dist@ptr$skewness()

#' @rdname discrete_distribution
#' @export
quantile_upper <- function(dist, p) rust_result(dist@ptr$quantile_upper(as.double(p)))

#' @rdname discrete_distribution
#' @export
return_periods <- function(dist, periods = NULL) {
  r <- rust_result(dist@ptr$tail(as.double(if (is.null(periods)) numeric() else periods)))
  data.frame(
    p = r$p, period = r$period, var = r$var, tvar = r$tvar, excess = r$excess,
    var_to_mean = r$var_to_mean
  )
}

S7::method(mean, discrete_distribution) <- function(x, ...) x@ptr$mean()
S7::method(variance, discrete_distribution) <- function(dist, ...) dist@ptr$variance()
S7::method(cdf, discrete_distribution) <- function(dist, q, ...) dist@ptr$cdf(as.double(q))
S7::method(survival, discrete_distribution) <- function(dist, q, ...) dist@ptr$survival(as.double(q))
S7::method(pmf, discrete_distribution) <- function(dist, k, ...) dist@ptr$pmf(as.double(k))
S7::method(quantile, discrete_distribution) <- function(x, probs, ...) {
  call <- sys.call()
  call[[1]] <- quote(quantile)
  rust_result(x@ptr$quantile(as.double(probs)), call)
}
S7::method(VaR, discrete_distribution) <- function(dist, p, ...) {
  rust_result(dist@ptr$quantile(as.double(p)), s7_call())
}
S7::method(TVaR, discrete_distribution) <- function(dist, p, ...) {
  rust_result(dist@ptr$tvar(as.double(p)), s7_call())
}
S7::method(print, discrete_distribution) <- function(x, ...) {
  v <- x@values
  cat(sprintf("<discrete_distribution> %d values from %s to %s, mean %s\n",
              length(v), format(v[1]), format(v[length(v)]), format(mean(x))))
  invisible(x)
}
