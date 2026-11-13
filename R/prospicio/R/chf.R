#' @include distributions.R pareto.R normal.R
NULL

#' A distribution from its characteristic function
#'
#' A law's bucket masses on a grid, by inverting its characteristic
#' function: `aggregate`'s `AggregateCHF` (`chfagg`). The characteristic
#' function is sampled on the FFT's frequency grid, times
#' `sinc(t * step / 2)`, and inverted once, so bucket `k` holds the
#' probability within half a step of `x_min + k * step`. With `step` left
#' out, the window is sized from the law's tails (Chernoff bounds) to leave
#' `1 - bucket_sizing_p` of the mass outside, half on each side, and the
#' step is rounded up to 1, 2, 4, 5 or 8 times a power of ten (a power of
#' two below 1); when the dropped frequencies are above 1e-6 the step moves
#' down that ladder while that helps. A Tweedie's mass at 0 is inverted
#' apart, so it does not ring.
#'
#' A stable law with one power tail is inverted along the line
#' `Im s = theta`, where the integrand decays like `exp(-theta x) f(x)`:
#' what wraps is the damped tail, and the mass beyond the window is left
#' out and reported in `report$outside_above` (or, for a left tail, added
#' to the first bucket and reported in `report$outside_below`). The
#' automatic tilt balances the damped wrap against the round-off the tilt
#' amplifies, and caps the step.
#'
#' The masses are as inverted, not renormalized. Supports [mean()],
#' [variance()], [cdf()] (the masses at grid points at or below `q`),
#' `quantile()` and [VaR()] (the first grid point whose cumulative mass
#' reaches `p`); [chf_to_grid()] gives a [grid_distribution] from 0.
#'
#' @param law A [normal_distribution()], [gamma_distribution()],
#'   [exponential_distribution()], [inverse_gaussian_distribution()],
#'   [tweedie()], an [affine_distribution()] of one of them, a
#'   [stable_distribution()],
#'   or your own characteristic function: an R function of a numeric vector
#'   of frequencies `t` returning `E exp(i t X)` at each, as a complex
#'   vector. Your own function needs `step` and `x_min`.
#' @param log2 The grid has `2^log2` buckets, 1 to 28.
#' @param step The bucket; sized from the tails when `NULL`.
#' @param x_min The grid's first point, with `step`; placed from the tails
#'   when `NULL`.
#' @param bucket_sizing_p The mass the window covers when it is sized from
#'   the tails.
#' @param tilt `TRUE`: a law with one power tail (a [stable_distribution()]
#'   with `beta` 1 or -1) at the automatic tilt, every other law untilted.
#'   `FALSE`: never tilt (a heavy tail wraps onto the window). A number
#'   `tau >= 0`: exactly `tau` e-folds of damping across the window, for a
#'   law with one power tail only.
#' @param ptr Internal.
#' @returns A `chf_distribution`, which inherits from [distribution], with
#'   properties `xs`, `masses`, `step`, `x_min` and `report` (a list:
#'   `log2`, `step_given`, `truncation_error` (`|phi(pi / step)|` of the
#'   continuous part), `is_truncated` (above 1e-6), `outside_below` and
#'   `outside_above` (the Chernoff bounds on the mass outside the window,
#'   which wraps in, or under a tilt the tail mass beyond the window),
#'   `tilt_left` and `tilt_right` (the e-folds of damping on each side),
#'   `edge_error` (the modelled worst relative error at the heavy end of a
#'   law with one power tail, `NaN` otherwise), `total_mass`, `law_mean`
#'   and `law_variance`).
#' @export
#' @examples
#' d <- invert_chf(gamma_distribution(4, 25), log2 = 12)
#' c(d@step, d@x_min, mean(d))
#' d@report$truncation_error
#' own <- invert_chf(function(t) (1 - 25i * t)^-4, log2 = 12, step = 0.25, x_min = 0)
#' max(abs(own@masses - d@masses))
invert_chf <- function(law, log2 = 16, step = NULL, x_min = NULL,
                       bucket_sizing_p = 0.99999, tilt = TRUE) {
  if (isTRUE(tilt)) {
    mode <- "auto"
    tau <- 0
  } else if (isFALSE(tilt)) {
    mode <- "off"
    tau <- 0
  } else if (is.numeric(tilt) && length(tilt) == 1) {
    mode <- "fixed"
    tau <- as.double(tilt)
  } else {
    stop("tilt must be TRUE, FALSE or a number", call. = FALSE)
  }
  target <- if (is.function(law)) {
    f <- law
    # An error comes back as its message, which Rust raises.
    function(t) {
      tryCatch({
        z <- as.complex(f(t))
        c(Re(z), Im(z))
      }, error = function(e) conditionMessage(e))
    }
  } else {
    law@ptr
  }
  na <- function(x) if (is.null(x)) NaN else as.double(x)
  ptr <- rust_result(ChfDist$new(target, as.double(log2), na(step), na(x_min),
                                 as.double(bucket_sizing_p), mode, tau))
  chf_distribution(ptr = ptr)
}

#' @rdname invert_chf
#' @export
chf_distribution <- S7::new_class(
  "chf_distribution",
  parent = distribution,
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("ChfDist"),
    xs = S7::new_property(S7::class_double, getter = function(self) self@ptr$xs()),
    masses = S7::new_property(S7::class_double, getter = function(self) self@ptr$masses()),
    step = S7::new_property(S7::class_double, getter = function(self) self@ptr$step()),
    x_min = S7::new_property(S7::class_double, getter = function(self) self@ptr$x_min()),
    report = S7::new_property(S7::class_list, getter = function(self) self@ptr$report())
  ),
  constructor = function(ptr) S7::new_object(S7::S7_object(), ptr = ptr)
)

#' The masses of a characteristic-function inversion as a grid
#'
#' The masses on `0, step, ...`, negative round-off set to 0 and the sum
#' rescaled to 1, for [compound_distribution()] and the risk measures on a grid.
#'
#' @param x A [chf_distribution].
#' @returns A [grid_distribution].
#' @export
#' @examples
#' g <- chf_to_grid(invert_chf(gamma_distribution(4, 25), log2 = 12))
#' TVaR(g, 0.99)
chf_to_grid <- function(x) {
  grid_distribution(ptr = rust_result(x@ptr$to_grid()))
}

S7::method(mean, chf_distribution) <- function(x, ...) x@ptr$mean()
S7::method(variance, chf_distribution) <- function(dist, ...) dist@ptr$variance()
S7::method(cdf, chf_distribution) <- function(dist, q, ...) dist@ptr$cdf(as.double(q))
S7::method(quantile, chf_distribution) <- function(x, probs, ...) {
  call <- sys.call()
  call[[1]] <- quote(quantile)
  rust_result(x@ptr$quantile(as.double(probs)), call)
}
S7::method(VaR, chf_distribution) <- function(dist, p, ...) {
  rust_result(dist@ptr$quantile(as.double(p)), s7_call())
}
S7::method(print, chf_distribution) <- function(x, ...) {
  cat(sprintf("<chf_distribution> x_min = %s, step = %s, %d buckets\n",
              format(x@x_min, digits = 15), format(x@step, digits = 15), length(x@masses)))
  invisible(x)
}

#' An alpha-stable law
#'
#' The parameters of `S(alpha, beta, scale, loc)` for [invert_chf()]: with
#' `beta` 1 or -1 its one power tail is inverted along a tilted line, with
#' `abs(beta) < 1` with `tilt = FALSE` only. At `alpha = 2` it is the normal
#' with variance `2 scale^2`. For `alpha < 2`,
#' `P(X > x) ~ tail_constant * x^-alpha` and
#' `P(X < -x) ~ left_tail_constant * x^-alpha`.
#'
#' @param alpha The index of stability, in `(0, 2]`.
#' @param beta The skewness, in `[-1, 1]`.
#' @param scale Positive.
#' @param loc The location, read in `parameterization`.
#' @param parameterization `"S1"` (Nolan's S1, scipy's default) or `"S0"`.
#' @returns A `stable_distribution` with properties `alpha`, `beta`,
#'   `scale`, `loc` (S1), `loc0` (S0), `tail_constant`,
#'   `left_tail_constant` and `mean`.
#' @export
#' @examples
#' s <- stable_distribution(1.5, 1, 2)
#' s@tail_constant
#' d <- invert_chf(s, log2 = 12)
#' d@report$tilt_right
stable_distribution <- S7::new_class(
  "stable_distribution",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("StableLaw"),
    alpha = S7::new_property(S7::class_double, getter = function(self) self@ptr$alpha()),
    beta = S7::new_property(S7::class_double, getter = function(self) self@ptr$beta()),
    scale = S7::new_property(S7::class_double, getter = function(self) self@ptr$scale()),
    loc = S7::new_property(S7::class_double, getter = function(self) self@ptr$loc()),
    loc0 = S7::new_property(S7::class_double, getter = function(self) self@ptr$loc0()),
    tail_constant = S7::new_property(S7::class_double,
                                     getter = function(self) self@ptr$tail_constant()),
    left_tail_constant = S7::new_property(S7::class_double,
                                          getter = function(self) self@ptr$left_tail_constant()),
    mean = S7::new_property(S7::class_double, getter = function(self) self@ptr$mean())
  ),
  constructor = function(alpha, beta, scale = 1, loc = 0, parameterization = "S1") {
    ptr <- rust_result(StableLaw$new(as.double(alpha), as.double(beta), as.double(scale),
                                     as.double(loc), as.character(parameterization)))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

S7::method(print, stable_distribution) <- function(x, ...) {
  cat(sprintf("<stable_distribution> alpha = %s, beta = %s, scale = %s, loc = %s (S1)\n",
              format(x@alpha, digits = 15), format(x@beta, digits = 15),
              format(x@scale, digits = 15), format(x@loc, digits = 15)))
  invisible(x)
}
