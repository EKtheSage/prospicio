# Idiomatic R API: S7 classes and generics over the Rust objects.

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

#' Abstract parent of every distribution class.
distribution <- S7::new_class("distribution", package = "actuarialrs", abstract = TRUE)

#' Lognormal distribution with log-scale mean `meanlog` and sd `sdlog`,
#' parameterized as in [stats::dlnorm()]. Parameters are read-only
#' properties: `d@meanlog`, `d@sdlog`.
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
    S7::new_object(distribution(), ptr = ptr)
  }
)

#' Lognormal distribution with the given mean and coefficient of variation.
lognormal_from_mean_cv <- function(mean, cv) {
  ptr <- rust_result(Lognormal$from_mean_cv(as.double(mean), as.double(cv)))
  lognormal(ptr$meanlog(), ptr$sdlog())
}

#' Distribution function `P(X <= q)`.
cdf <- S7::new_generic("cdf", "dist", function(dist, q, ...) S7::S7_dispatch())

#' Variance of a distribution.
variance <- S7::new_generic("variance", "dist", function(dist, ...) S7::S7_dispatch())

#' `n` reproducible draws from stream `stream` of the generator keyed by
#' `seed`. The same `(seed, stream)` gives the same draws in R, Python and
#' Rust.
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
