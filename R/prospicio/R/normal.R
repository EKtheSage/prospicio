#' @include distributions.R pareto.R
NULL

#' Normal distribution
#'
#' The normal distribution with mean `mu` and standard deviation `sigma`, on
#' the whole real line: asset returns, log returns and other outcomes of
#' either sign. Its TVaR is `mu + sigma * dnorm(z) / (1 - p)` with
#' `z = qnorm(p)` (*Pricing Insurance Risk*, section 4.3.3): VaR plus a
#' multiple of `sigma` that depends only on `p`.
#'
#' Supports [mean()], [variance()], [cdf()], [survival()], `quantile()`,
#' [VaR()], [TVaR()], [log_density()] and [draws()].
#'
#' @param mu The mean.
#' @param sigma The standard deviation, positive.
#' @param ptr Internal.
#' @returns A `normal_distribution`, which inherits from [distribution].
#' @export
#' @examples
#' x <- normal_distribution(0, 1)
#' quantile(x, 0.99)
#' TVaR(x, 0.99) # 2.665 sigma
normal_distribution <- S7::new_class(
  "normal_distribution",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("NormalDist"),
    mu = S7::new_property(S7::class_double, getter = function(self) self@ptr$mu()),
    sigma = S7::new_property(S7::class_double, getter = function(self) self@ptr$sigma())
  ),
  constructor = function(mu, sigma, ptr = NULL) {
    if (is.null(ptr)) ptr <- rust_result(NormalDist$new(as.double(mu), as.double(sigma)))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Ogive from grouped data
#'
#' The piecewise linear distribution function through the points
#' `(x, cdf)`, the usual reading of grouped loss data (losses uniform within
#' each band), as in *Pricing Insurance Risk*, Exercise 50.
#' `ogive_from_counts()` builds one from `counts[k]` losses between
#' `bounds[k]` and `bounds[k + 1]`. Its limited moments are exact, band by
#' band.
#'
#' Supports [mean()], [variance()], [cdf()], [survival()], `quantile()` (the
#' lower quantile), [VaR()], [TVaR()], [lev()], [stop_loss()], [layer()],
#' [layer_variance()], [layer_second_moment()], [log_density()] and
#' [draws()].
#'
#' @param x At least two band boundaries, finite, non-negative and strictly
#'   increasing.
#' @param cdf `F` at each boundary, non-decreasing from 0 to 1.
#' @param bounds Band boundaries, one more than `counts`.
#' @param counts Non-negative counts (or weights) per band, with a positive
#'   total.
#' @param ptr Internal.
#' @returns An `ogive`, which inherits from [distribution].
#' @export
#' @examples
#' o <- ogive_from_counts(c(0, 1000, 5000, 10000), c(40, 50, 10))
#' cdf(o, 3000)
#' VaR(o, 0.95)
#' TVaR(o, 0.95)
ogive <- S7::new_class(
  "ogive",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("OgiveDist"),
    x = S7::new_property(S7::class_double, getter = function(self) self@ptr$x()),
    cdf_values = S7::new_property(S7::class_double, getter = function(self) self@ptr$cdf_values())
  ),
  constructor = function(x, cdf, ptr = NULL) {
    if (is.null(ptr)) ptr <- rust_result(OgiveDist$new(as.double(x), as.double(cdf)))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' @rdname ogive
#' @export
ogive_from_counts <- function(bounds, counts) {
  ogive(ptr = rust_result(OgiveDist$from_counts(as.double(bounds), as.double(counts))))
}

for (cls in list(normal_distribution, ogive)) {
  S7::method(mean, cls) <- function(x, ...) x@ptr$mean()
  S7::method(variance, cls) <- function(dist, ...) dist@ptr$variance()
  S7::method(cdf, cls) <- function(dist, q, ...) dist@ptr$cdf(as.double(q))
  S7::method(survival, cls) <- function(dist, q, ...) dist@ptr$survival(as.double(q))
  S7::method(log_density, cls) <- function(dist, x, ...) dist@ptr$ln_pdf(as.double(x))
  S7::method(quantile, cls) <- function(x, probs, ...) {
    call <- sys.call()
    call[[1]] <- quote(quantile)
    rust_result(x@ptr$quantile(as.double(probs)), call)
  }
  S7::method(VaR, cls) <- function(dist, p, ...) rust_result(dist@ptr$quantile(as.double(p)), s7_call())
  S7::method(draws, cls) <- function(dist, n, seed, stream = 0, ...) {
    call <- sys.call(sys.parent())
    rust_result(dist@ptr$sample(as.double(n), as.double(seed), as.double(stream)), call)
  }
}

S7::method(TVaR, normal_distribution) <- function(dist, p, ...) {
  rust_result(dist@ptr$tvar(as.double(p)), s7_call())
}
S7::method(print, normal_distribution) <- function(x, ...) {
  cat(sprintf("<normal_distribution> mu = %s, sigma = %s\n",
              format(x@mu, digits = 15), format(x@sigma, digits = 15)))
  invisible(x)
}

S7::method(TVaR, ogive) <- function(dist, p, ...) severity_tvar(dist@ptr, p, s7_call())
S7::method(lev, ogive) <- function(dist, limit, ...) dist@ptr$lev(as.double(limit))
S7::method(stop_loss, ogive) <- function(dist, retention, ...) dist@ptr$stop_loss(as.double(retention))
S7::method(layer, ogive) <- function(dist, limit, attachment, ...) {
  dist@ptr$layer(as.double(limit), as.double(attachment))
}
S7::method(layer_variance, ogive) <- function(dist, limit, attachment, ...) {
  dist@ptr$layer_variance(as.double(limit), as.double(attachment))
}
S7::method(layer_second_moment, ogive) <- function(dist, limit, attachment, ...) {
  dist@ptr$layer_second_moment(as.double(limit), as.double(attachment))
}
S7::method(print, ogive) <- function(x, ...) {
  b <- x@x
  cat(sprintf("<ogive> %d points from %s to %s, mean %s\n",
              length(b), format(b[1]), format(b[length(b)]), format(mean(x))))
  invisible(x)
}
