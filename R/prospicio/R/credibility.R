# Aggregate lane: classical credibility, over
# crates/prospicio-r/src/credibility.rs. The #' comments are the package
# documentation (see distributions.R).

#' Limited fluctuation credibility
#'
#' The expected number of claims for an observed value to lie within `k` of
#' its mean with probability `p`, under a Poisson count and the normal
#' approximation. With `z = qnorm((1 + p) / 2)` and `n_F = (z / k)^2`, the
#' full standards in claims are `n_F` for the frequency, `n_F * cv^2` for
#' the severity and `n_F * (1 + cv^2)` for the pure premium.
#' `limited_fluctuation_credibility()` is the square root rule
#' `min(1, sqrt(n / standard))`.
#'
#' @param p Probability in `(0, 1)`.
#' @param k Tolerance, positive.
#' @param cv The severity's coefficient of variation.
#' @param n Number of claims.
#' @param standard A full standard, in claims.
#' @returns `credibility_standard()`: a named vector (`frequency`,
#'   `severity`, `pure_premium`). `limited_fluctuation_credibility()`: the
#'   credibility factors.
#' @export
#' @examples
#' s <- credibility_standard(0.90, 0.05, cv = 2)
#' s
#' limited_fluctuation_credibility(c(100, 1082, 5000), s[["frequency"]])
credibility_standard <- function(p = 0.90, k = 0.05, cv = 0) {
  s <- rust_result(credibility_standards(as.double(p), as.double(k), as.double(cv)))
  stats::setNames(s, c("frequency", "severity", "pure_premium"))
}

#' @rdname credibility_standard
#' @export
limited_fluctuation_credibility <- function(n, standard) {
  pmin(1, sqrt(pmax(as.double(n), 0) / as.double(standard)))
}

#' Bühlmann–Straub credibility
#'
#' The Bühlmann–Straub model, as `actuar::cm(~group)`: group `i` in period
#' `j` has ratio `X_ij` (loss per unit of exposure) with weight `w_ij`.
#' Without weights it is Bühlmann's model. The between-group variance is
#' Bühlmann and Straub's unbiased estimator (`actuar`'s
#' `"Buhlmann-Gisler"` and `"Ohlsson"`) or Bichsel and Straub's iterative
#' one.
#'
#' @param ratios A matrix, one row per group and one column per period;
#'   `NA` for a missing cell.
#' @param weights A matrix of the same shape with `NA` in the same places,
#'   or `NULL` for equal weights.
#' @param estimator `"unbiased"` or `"iterative"`.
#' @param tol,max_iter Tolerance and iteration limit of the iterative
#'   estimator.
#' @returns A list: `collective_mean`, `between_variance`,
#'   `within_variance`, `k` (`s2 / a`), and per group `group_means`,
#'   `group_weights`, `credibility` and `premiums`, with `iterations`.
#' @export
#' @examples
#' x <- rbind(c(1, 2, 3), c(4, 5, 9))
#' fit <- buhlmann_straub(x)
#' fit$premiums
buhlmann_straub <- function(ratios, weights = NULL, estimator = c("unbiased", "iterative"),
                            tol = sqrt(.Machine$double.eps), max_iter = 100) {
  estimator <- match.arg(estimator)
  ratios <- as.matrix(ratios)
  w <- if (is.null(weights)) double() else as.double(as.matrix(weights))
  rust_result(buhlmann_straub_rust(as.double(ratios), w, as.double(nrow(ratios)),
                                   as.double(ncol(ratios)), estimator, as.double(tol),
                                   as.double(max_iter)))
}
