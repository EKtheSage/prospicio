# Low-level bindings to crates/act-r, in the form rextendr generates.
# Internal: use the functions in distributions.R.

#' @useDynLib actuarialrs, .registration = TRUE
NULL

Lognormal <- new.env(parent = emptyenv())

Lognormal$new <- function(meanlog, sdlog) .Call(wrap__Lognormal__new, meanlog, sdlog)

Lognormal$from_mean_cv <- function(mean, cv) .Call(wrap__Lognormal__from_mean_cv, mean, cv)

Lognormal$meanlog <- function() .Call(wrap__Lognormal__meanlog, self)

Lognormal$sdlog <- function() .Call(wrap__Lognormal__sdlog, self)

Lognormal$mean <- function() .Call(wrap__Lognormal__mean, self)

Lognormal$variance <- function() .Call(wrap__Lognormal__variance, self)

Lognormal$cdf <- function(x) .Call(wrap__Lognormal__cdf, self, x)

Lognormal$quantile <- function(p) .Call(wrap__Lognormal__quantile, self, p)

Lognormal$sample <- function(n, seed, stream) .Call(wrap__Lognormal__sample, self, n, seed, stream)

`$.Lognormal` <- function(self, name) {
  func <- Lognormal[[name]]
  environment(func) <- environment()
  func
}

`[[.Lognormal` <- `$.Lognormal`
