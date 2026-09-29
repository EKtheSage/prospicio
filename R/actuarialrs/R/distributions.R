# Idiomatic R API: constructor functions and S3 methods over the Rust objects.

# Rust errors come back as "extendr_error" condition objects (extendr's
# result_condition feature) rather than as panics. Raise them as ordinary R
# errors, attributed to the user-facing function that was called.
rust_result <- function(x) {
  if (inherits(x, "extendr_error")) {
    stop(errorCondition(as.character(x$value), call = sys.call(-1)))
  }
  x
}

new_act_lognormal <- function(ptr) {
  structure(list(ptr = ptr), class = c("act_lognormal", "act_distribution"))
}

#' Lognormal distribution with log-scale mean `meanlog` and sd `sdlog`,
#' parameterized as in [stats::dlnorm()].
lognormal <- function(meanlog, sdlog) {
  ptr <- rust_result(Lognormal$new(as.double(meanlog), as.double(sdlog)))
  new_act_lognormal(ptr)
}

#' Lognormal distribution with the given mean and coefficient of variation.
lognormal_from_mean_cv <- function(mean, cv) {
  ptr <- rust_result(Lognormal$from_mean_cv(as.double(mean), as.double(cv)))
  new_act_lognormal(ptr)
}

#' Distribution function `P(X <= q)`.
cdf <- function(dist, q, ...) UseMethod("cdf")

#' Variance of a distribution.
variance <- function(dist, ...) UseMethod("variance")

#' `n` reproducible draws from stream `stream` of the generator keyed by
#' `seed`. The same `(seed, stream)` gives the same draws in R, Python and
#' Rust.
draws <- function(dist, n, seed, stream = 0, ...) UseMethod("draws")

mean.act_lognormal <- function(x, ...) x$ptr$mean()

quantile.act_lognormal <- function(x, probs, ...) {
  rust_result(x$ptr$quantile(as.double(probs)))
}

cdf.act_lognormal <- function(dist, q, ...) dist$ptr$cdf(as.double(q))

variance.act_lognormal <- function(dist, ...) dist$ptr$variance()

draws.act_lognormal <- function(dist, n, seed, stream = 0, ...) {
  rust_result(dist$ptr$sample(as.double(n), as.double(seed), as.double(stream)))
}

print.act_lognormal <- function(x, ...) {
  cat(sprintf("<lognormal> meanlog = %s, sdlog = %s\n",
              format(x$ptr$meanlog(), digits = 15), format(x$ptr$sdlog(), digits = 15)))
  invisible(x)
}
