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

#' Hachemeister's regression credibility
#'
#' Hachemeister's regression model, as `actuar::cm(~risk, regformula =
#' formula, regdata = data)`: risk `i` in period `t` has ratio `X_it` with
#' weight `w_it` and a mean on a design shared by the risks (a trend line
#' `~ time`, say). Each risk's own weighted least squares fit is shrunk
#' towards the collective by a credibility matrix
#' `Z_i = A (A + s2 W_i)^-1`.
#'
#' At the origin the matrices are full and the between covariance `A` is
#' Hachemeister's iterative estimator whatever `estimator` says, as in
#' `actuar` (on nearly collinear coefficients its iteration may wander in
#' the last digits). With `adjust_intercept = TRUE` (`actuar`'s
#' `adj.intercept`) the design is orthogonalised under the collective
#' weights, the matrices are diagonal and each diagonal is estimated as
#' Bühlmann–Straub's by `estimator`; the formula must then have an
#' intercept as its first column.
#'
#' @param ratios A matrix, one row per risk and one column per period; `NA`
#'   for a missing cell.
#' @param weights A matrix of the same shape with `NA` in the same places,
#'   or `NULL` for equal weights.
#' @param formula A one-sided formula for the design, evaluated in `data`.
#' @param data A data frame with one row per period; by default
#'   `time = 1, 2, ...`.
#' @param estimator `"unbiased"` or `"iterative"`.
#' @param adjust_intercept Put the intercept at the barycentre of the
#'   periods.
#' @param tol,max_iter Tolerance and iteration limit of the iterations.
#' @returns A `hachemeister_fit` with properties `collective` (named by the
#'   design's columns), `coefficients` and `individual` (one row per risk),
#'   `credibility` (a `p x p x risks` array in the basis fitted), `between`,
#'   `within_variance`, `risk_weights`, `transition` (`R` with the design
#'   `Y = Q R` when adjusted, else `NULL`) and `iterations`; `predict(fit,
#'   newdata)` gives each risk's premium at each row of `newdata`.
#' @export
#' @examples
#' x <- rbind(c(10, 12, 13, 15, 16), c(8, 8.5, 9.5, 9, 10), c(14, 13, 16, 17, 19))
#' fit <- hachemeister(x, adjust_intercept = TRUE)
#' fit@collective
#' predict(fit, data.frame(time = 6))
hachemeister <- function(ratios, weights = NULL, formula = ~time,
                         data = data.frame(time = seq_len(ncol(ratios))),
                         estimator = c("unbiased", "iterative"), adjust_intercept = FALSE,
                         tol = sqrt(.Machine$double.eps), max_iter = 100) {
  estimator <- match.arg(estimator)
  ratios <- as.matrix(ratios)
  w <- if (is.null(weights)) double() else as.double(as.matrix(weights))
  terms <- stats::delete.response(stats::terms(formula, data = data))
  y <- stats::model.matrix(terms, stats::model.frame(terms, data))
  if (nrow(y) != ncol(ratios)) {
    stop("data needs one row per period (column of ratios)", call. = FALSE)
  }
  r <- rust_result(hachemeister_rust(as.double(ratios), w, as.double(nrow(ratios)),
                                     as.double(ncol(ratios)), as.double(y), as.double(ncol(y)),
                                     estimator, isTRUE(adjust_intercept), as.double(tol),
                                     as.double(max_iter)))
  p <- ncol(y)
  k <- nrow(ratios)
  names <- colnames(y)
  coef_matrix <- function(v) matrix(v, k, p, dimnames = list(rownames(ratios), names))
  hachemeister_fit(
    collective = stats::setNames(r$collective, names),
    coefficients = coef_matrix(r$coefficients),
    individual = coef_matrix(r$individual),
    credibility = array(r$credibility, c(p, p, k)),
    between = matrix(r$between, p, p),
    within_variance = r$within_variance,
    risk_weights = r$risk_weights,
    transition = if (length(r$transition)) matrix(r$transition, p, p) else NULL,
    iterations = r$iterations,
    terms = terms
  )
}

#' @rdname hachemeister
#' @export
hachemeister_fit <- S7::new_class(
  "hachemeister_fit",
  package = "prospicio",
  properties = list(
    collective = S7::class_double,
    coefficients = S7::class_any,
    individual = S7::class_any,
    credibility = S7::class_any,
    between = S7::class_any,
    within_variance = S7::class_double,
    risk_weights = S7::class_double,
    transition = S7::class_any,
    iterations = S7::class_double,
    terms = S7::class_any
  )
)

S7::method(predict, hachemeister_fit) <- function(object, newdata, ...) {
  x <- stats::model.matrix(object@terms, stats::model.frame(object@terms, newdata))
  p <- x %*% t(object@coefficients)
  dimnames(p) <- list(NULL, rownames(object@coefficients))
  p
}

S7::method(print, hachemeister_fit) <- function(x, ...) {
  cat("<hachemeister_fit> ", nrow(x@coefficients), " risks, collective:\n", sep = "")
  print(x@collective)
  invisible(x)
}

# Models lane: Bayesian hierarchical credibility, over prospicio_bayes's
# credibility module.

#' Bayesian hierarchical credibility
#'
#' Bühlmann–Straub's model, or Jewell's hierarchy of it, as a normal
#' random-effects model sampled with NUTS. Group `i` in period `j` has ratio
#' `X_ij ~ N(theta_i, sigma^2 / w_ij)`; each group's risk premium `theta_i`
#' is normal about its node at the level above, each node about its own
#' parent, and the top about the collective mean `mu`. The means are
#' integrated out exactly, so NUTS samples only the standard deviations;
#' given them, each premium is the credibility formula `Z * M + (1 - Z) *
#' E[parent]`, and the Bayesian premium averages it over their posterior.
#'
#' `credibility_premiums()` gives that formula at fixed structure
#' parameters: Bühlmann–Straub's premiums with one level and
#' Bühlmann–Gisler's hierarchical ones (as `actuar::cm(~ cohort +
#' cohort:group)`) with more. `credibility_predictive()` draws each group's
#' risk premium, or with `weights` its ratio next period.
#'
#' @param ratios A matrix, one row per group and one column per period; `NA`
#'   for a missing cell.
#' @param weights A matrix of the same shape with `NA` in the same places.
#' @param levels A list of vectors, one per level above the groups,
#'   outermost first: each group's node at that level, from 1. `NULL` for
#'   Bühlmann–Straub's model.
#' @param collective_mean `c(mean, sd)` of a normal prior on `mu`, or `NULL`
#'   for a flat one.
#' @param within_prior `list("half_normal", scale)` or `list("half_cauchy",
#'   scale)` on `sigma`; `NULL` for half-normal at five times the pooled
#'   within standard deviation.
#' @param between_priors A list of such priors, one per level, outermost
#'   first and the groups' last; `NULL` for half-normal at five times each
#'   level's spread.
#' @param chains,tune,draws,seed NUTS settings.
#' @param within_variance `s^2`.
#' @param between_variances One per level, outermost first and the groups'
#'   last (`a` with one level; `b` then `a` with two).
#' @param fit A `hierarchical_credibility` fit.
#' @param n_sims Number of simulations.
#' @returns `hierarchical_credibility()`: a list of class
#'   `hierarchical_credibility` with `premiums`, `credibility`,
#'   `level_premiums`, `collective_mean`, `group_means`, `group_weights`,
#'   `divergences` and `summary` (a data frame of the structure parameters'
#'   posterior: `within_variance` and `between_variance[l]`).
#'   `credibility_premiums()`: a list with `premiums`, `credibility`,
#'   `level_premiums`, `level_credibility` and `collective_mean`.
#'   `credibility_predictive()`: a [predictive_distribution] with dimension
#'   `group`.
#' @export
#' @examples
#' x <- rbind(c(1, 2, 3, 2), c(4, 5, 6, 5), c(2, 3, 4, 3))
#' w <- matrix(1, 3, 4)
#' fit <- hierarchical_credibility(x, w, chains = 2, tune = 300, draws = 300)
#' fit$premiums
#' credibility_premiums(x, w, within_variance = 0.7, between_variances = 2)$premiums
hierarchical_credibility <- function(ratios, weights, levels = NULL, collective_mean = NULL,
                                     within_prior = NULL, between_priors = NULL, chains = 4,
                                     tune = 1000, draws = 1000, seed = 0) {
  ratios <- as.matrix(ratios)
  weights <- as.matrix(weights)
  wk <- if (is.null(within_prior)) "" else as.character(within_prior[[1]])
  ws <- if (is.null(within_prior)) NaN else as.double(within_prior[[2]])
  bk <- vapply(between_priors, function(p) as.character(p[[1]]), "")
  bs <- vapply(between_priors, function(p) as.double(p[[2]]), 0)
  ptr <- rust_result(hierarchical_credibility_rust(
    as.double(ratios), as.double(weights), as.double(nrow(ratios)), as.double(ncol(ratios)),
    lapply(levels, as.double), as.double(collective_mean), wk, ws, bk, bs,
    as.double(chains), as.double(tune), as.double(draws), as.double(seed)
  ))
  out <- ptr$values()
  out$summary <- as.data.frame(rust_result(ptr$summary()))
  out$ptr <- ptr
  structure(out, class = "hierarchical_credibility")
}

#' @rdname hierarchical_credibility
#' @export
credibility_premiums <- function(ratios, weights, within_variance, between_variances,
                                 levels = NULL, collective_mean = NULL) {
  ratios <- as.matrix(ratios)
  weights <- as.matrix(weights)
  rust_result(credibility_premiums_rust(
    as.double(ratios), as.double(weights), as.double(nrow(ratios)), as.double(ncol(ratios)),
    lapply(levels, as.double), as.double(within_variance), as.double(between_variances),
    as.double(collective_mean)
  ))
}

#' @rdname hierarchical_credibility
#' @export
credibility_predictive <- function(fit, weights = NULL, n_sims = 10000, seed = 0) {
  predictive_distribution(ptr = rust_result(fit$ptr$predictive(
    as.double(weights), as.double(n_sims), as.double(seed)
  )))
}

#' @export
print.hierarchical_credibility <- function(x, ...) {
  cat(sprintf("<hierarchical_credibility> %d groups, collective mean %s\n",
              length(x$premiums), format(x$collective_mean)))
  print(x$summary[, c("name", "mean", "sd", "rhat")], row.names = FALSE)
  invisible(x)
}
