#' @include distributions.R normal.R
NULL

#' Multivariate normal and lognormal joints
#'
#' Many capital-model inputs arrive as a mean vector and a covariance matrix,
#' not as draws: a reserve by origin with Mack's covariance across origins,
#' lines joined by a correlation matrix. `multivariate_normal()` and
#' `multivariate_lognormal()` hold such a joint with its exact marginals and
#' moments; [predictive_from_gaussian()] draws it by seed.
#'
#' `multivariate_lognormal()` takes the mean and covariance on the original
#' scale, matched exactly (`log_cov[i, j] = log(1 + cov[i, j] / (m[i] m[j]))`,
#' `mu[i] = log(m[i]) - log_cov[i, i] / 2`, the conversion behind Mack's
#' lognormal reserve), or `mu` and `log_cov` on the log scale.
#'
#' The covariance must be symmetric and positive definite: a zero variance
#' or a perfect correlation is singular and refused. A covariance at or below
#' `-m[i] m[j]` has no lognormal, and the covariance on the log scale can fail
#' to be positive definite even when `cov` is; both are refused too.
#'
#' `marginal(joint, j)` is component `j`'s marginal, exact (a
#' [normal_distribution] or a [lognormal]). `total()` of a normal joint is
#' normal with mean `sum(mean)` and variance `sum(cov)`; a sum of lognormals
#' is not lognormal, so a lognormal joint gives its total's moments as
#' `total_mean` and `total_variance` and errors in `total()`.
#'
#' @param mean The mean vector (positive for a lognormal).
#' @param cov The covariance matrix, one row and column per component.
#' @param mu,log_cov For `multivariate_lognormal()`, the mean and covariance on
#'   the log scale instead of `mean` and `cov`.
#' @param ptr Internal.
#' @returns A `multivariate_gaussian` with properties `kind` (`"normal"` or
#'   `"lognormal"`), `dimension`, `mean`, `cov` and `correlation` (on the original
#'   scale), `mu` and `log_cov` (a lognormal's log scale, empty for a normal),
#'   `total_mean` and `total_variance`.
#' @seealso [predictive_from_gaussian()]
#' @export
#' @examples
#' cov <- matrix(c(400, 120, 120, 100), 2)
#' mvn <- multivariate_normal(c(100, 50), cov)
#' total(mvn)@sigma^2 # 400 + 2 * 120 + 100
#' marginal(mvn, 2)@sigma
#' mvl <- multivariate_lognormal(c(100, 50), cov)
#' mvl@cov # the covariance given
#' mvl@total_variance
multivariate_normal <- function(mean, cov) {
  multivariate_gaussian(ptr = rust_result(GaussianJointR$normal(as.double(mean), as.double(as.matrix(cov)))))
}

#' @rdname multivariate_normal
#' @export
multivariate_lognormal <- function(mean = NULL, cov = NULL, mu = NULL, log_cov = NULL) {
  moments <- !is.null(mean) || !is.null(cov)
  log_scale <- !is.null(mu) || !is.null(log_cov)
  if (moments == log_scale) {
    stop("give mean and cov, or mu and log_cov", call. = FALSE)
  }
  ptr <- if (moments) {
    rust_result(GaussianJointR$lognormal_moments(as.double(mean), as.double(as.matrix(cov))))
  } else {
    rust_result(GaussianJointR$lognormal_log_scale(as.double(mu), as.double(as.matrix(log_cov))))
  }
  multivariate_gaussian(ptr = ptr)
}

square <- function(x, d) if (length(x) == 0) matrix(numeric(), 0, 0) else matrix(x, d, d)

#' @rdname multivariate_normal
#' @export
multivariate_gaussian <- S7::new_class(
  "multivariate_gaussian",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("GaussianJointR"),
    kind = S7::new_property(S7::class_character, getter = function(self) self@ptr$kind()),
    dimension = S7::new_property(S7::class_double, getter = function(self) self@ptr$dim()),
    mean = S7::new_property(S7::class_double, getter = function(self) self@ptr$mean()),
    cov = S7::new_property(S7::class_any, getter = function(self) square(self@ptr$cov(), self@dimension)),
    correlation = S7::new_property(S7::class_any, getter = function(self) {
      square(self@ptr$correlation(), self@dimension)
    }),
    mu = S7::new_property(S7::class_double, getter = function(self) self@ptr$mu()),
    log_cov = S7::new_property(S7::class_any, getter = function(self) square(self@ptr$log_cov(), self@dimension)),
    total_mean = S7::new_property(S7::class_double, getter = function(self) self@ptr$total_mean()),
    total_variance = S7::new_property(S7::class_double, getter = function(self) self@ptr$total_variance())
  ),
  constructor = function(ptr) S7::new_object(S7::S7_object(), ptr = ptr)
)

S7::method(marginal, multivariate_gaussian) <- function(dist, key, ...) {
  if (dist@kind == "normal") {
    normal_distribution(ptr = rust_result(dist@ptr$marginal_normal(as.double(key))))
  } else {
    lognormal(ptr = rust_result(dist@ptr$marginal_lognormal(as.double(key))))
  }
}

S7::method(total, multivariate_gaussian) <- function(dist, ...) {
  normal_distribution(ptr = rust_result(dist@ptr$total_normal()))
}

S7::method(print, multivariate_gaussian) <- function(x, ...) {
  cat("<multivariate_gaussian> ", x@kind, ", dimension ", x@dimension, "\n", sep = "")
  invisible(x)
}

#' Draw a multivariate normal or lognormal joint
#'
#' `n_sims` draws of a [multivariate_normal()] or [multivariate_lognormal()],
#' one component per key. Simulation `i` draws only from stream `i` of
#' `seed`, so the result reproduces by seed on any number of threads; its
#' provenance records the model, its dimension, the seed and a hash of the
#' mean and covariance.
#'
#' @param joint A `multivariate_gaussian`.
#' @param n_sims Number of simulations.
#' @param seed Seed.
#' @param keys A data frame with one row per component and one column per
#'   dimension; by default `component = 1, 2, ...`.
#' @returns A [predictive_distribution].
#' @export
#' @examples
#' mvn <- multivariate_normal(c(100, 50), matrix(c(400, 120, 120, 100), 2))
#' pd <- predictive_from_gaussian(mvn, 20000, 1, keys = data.frame(line = c("auto", "property")))
#' mean(pd)
predictive_from_gaussian <- function(joint, n_sims, seed, keys = NULL) {
  if (is.null(keys)) keys <- data.frame(component = seq_len(joint@dimension))
  keys <- as.data.frame(keys, stringsAsFactors = FALSE)
  predictive_distribution(ptr = rust_result(joint@ptr$simulate(
    as.double(n_sims), as.double(seed), names(keys), as.list(keys)
  )))
}
