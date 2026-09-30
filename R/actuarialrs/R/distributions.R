# Idiomatic R API: S7 classes and generics over the Rust objects.
#
# The #' comments are the package documentation. `cargo xtask r` turns them
# into man/*.Rd and NAMESPACE with roxygen2 and renders the pkgdown site
# (docs/architecture.md, "Documentation"). Do not edit man/ or NAMESPACE.

#' @importFrom stats quantile
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
distribution <- S7::new_class("distribution", package = "actuarialrs", abstract = TRUE)

#' Lognormal distribution
#'
#' Lognormal distribution with log-scale mean `meanlog` and sd `sdlog`,
#' parameterized as in [stats::dlnorm()]. Parameters are read-only
#' properties: `d@meanlog`, `d@sdlog`.
#'
#' Supports [mean()], [variance()], [stats::quantile()], [cdf()], [draws()]
#' and [print()].
#'
#' @param meanlog Mean of `log(X)`; finite.
#' @param sdlog Standard deviation of `log(X)`; finite and positive.
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
  package = "actuarialrs",
  properties = list(
    ptr = S7::new_S3_class("Lognormal"),
    meanlog = S7::new_property(S7::class_double, getter = function(self) self@ptr$meanlog()),
    sdlog = S7::new_property(S7::class_double, getter = function(self) self@ptr$sdlog())
  ),
  constructor = function(meanlog, sdlog) {
    ptr <- rust_result(Lognormal$new(as.double(meanlog), as.double(sdlog)))
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
  # An S3 method: its call names the method, e.g. `quantile.actuarialrs::lognormal`.
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

.onLoad <- function(libname, pkgname) {
  S7::methods_register()
}
