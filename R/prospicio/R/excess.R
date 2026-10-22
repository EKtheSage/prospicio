#' @include distributions.R reserving.R
NULL

#' Excess-layer development from a ground-up triangle
#'
#' A layer's development pattern derived from the ground-up incurred
#' triangle (issue #192; design note `docs/design/reserving-excess.md`).
#' The layer's age-to-age factors follow from the ground-up factors and a
#' severity by maturity: the layer's expected loss at each age, per claim,
#' relates to the ultimate severity's. Bloom and Patrik (CAS E-Forum Fall
#' 2013), Sahasrabuddhe (E-Forum Fall 2010) and Pinto and Gogol (PCAS 74,
#' 1987) print the examples the package reproduces.
#'
#' * `excess_layer()` is the layer `limit` xs `attachment`, the same for
#'   every origin; `limit = Inf` is the unlimited layer above `attachment`.
#' * `scale_shift()` is the default severity by maturity: the severity at each
#'   age is the `ultimate` severity divided by a shift solved from the
#'   ground-up triangle. `ground_up_cap` is the per-claim limit of a capped
#'   ground-up column.
#' * `pareto_shape()` is Pinto and Gogol's shape drift above `threshold`:
#'   a Pareto shape at every age of the column (`alphas`, named by age in
#'   months) and at ultimate, and the base development the column holds:
#'   `base = "counts"` for the claims above the threshold, or an
#'   `excess_layer()` for the losses in a base layer attaching at or above
#'   it.
#' * `excess_pattern()` derives the pattern. `column` is the ground-up
#'   incurred (reported) column, or under `pareto_shape()` the base
#'   development. `counts` names a reported claim count column on the same
#'   ages (not with `pareto_shape()`). `tail` is the ground-up tail, as for
#'   [chain_ladder()]; it fixes one more shift and the layer's tail. Only an
#'   incurred basis is derived.
#' * `excess_selected()` gives the derived pattern as the `ldf`, `ldf_kind`
#'   and `ldf_weight` arguments of [chain_ladder()], [mack()] and the
#'   expected-loss methods, which is how a pattern is used:
#'   `do.call(chain_ladder, c(list(layer_triangle, tail = p@tail),
#'   excess_selected(p, weight)))`.
#' * `excess_credibility_weights()` gives the weight on the derived pattern
#'   at each age, `1 - z` with `z = n / (n + k)`, `n` the layer triangle's
#'   claim count (its origins with a positive earlier value without counts) at
#'   the earlier age, capped so that removing any one origin moves the
#'   blended factor by at most `max_influence` of it. Where the layer's own
#'   factor cannot be estimated the weight is 1. `k` has no default (design
#'   note, decision 19): the simulation study favours about 20.
#' * `layer_share()` is the layer's share of a severity's mean,
#'   `(LEV(b) - LEV(a)) / mean`. `layer_at_shift()` is the expected loss to
#'   the layer of one claim at the maturity whose claims are the ultimate
#'   ones divided by `shift`, `(LEV(c b) - LEV(c a)) / c`.
#' * `layer_exposure()` adds to a triangle the column `name`, the `exposure`
#'   column times each origin's layer share, for a Bornhuetter-Ferguson or
#'   Cape Cod prior. The share is a severity's `layer_share()` or an
#'   exposure curve's [exposure_layer_share()] (design note, decision 2: the
#'   reserving core takes shares, not curves).
#'
#' An `excess_pattern` has the properties `ages`, `layer_ldf` (age to age,
#' one fewer than the ages), `layer_cdf` (to ultimate, with the tail),
#' `ground_up_ldf`, `ground_up_tail`, `shifts` (empty under
#' `pareto_shape()`), `tail` (the layer's factor from the oldest age to
#' ultimate) and `ldf` (the age-to-age factors named by age).
#'
#' @param attachment Where the layer starts: finite and not negative.
#' @param limit The layer's width: positive, or `Inf`.
#' @param ultimate The ultimate severity at the price level of the layer's
#'   origins: any severity distribution with exact limited moments.
#' @param ground_up_cap The per-claim limit of a capped ground-up column, or
#'   `NULL`.
#' @param threshold The threshold above which claims are Pareto.
#' @param alphas The Pareto shape at every age of the column, a named numeric
#'   vector (names are ages in months).
#' @param ultimate_alpha The shape of the fully developed claims.
#' @param base `"counts"` or an `excess_layer()`.
#' @param triangle A single-segment [triangle] holding the ground-up column.
#' @param column The ground-up incurred column, by default the triangle's
#'   only one.
#' @param layer An `excess_layer()`.
#' @param model A `scale_shift()` or a `pareto_shape()`.
#' @param counts The name of a claim count column, or `NULL`.
#' @param tail The ground-up tail: a number or a tail estimator.
#' @param pattern An `excess_pattern`.
#' @param weight The weight on the derived factors, a number or a named
#'   numeric vector (names are ages in months).
#' @param layer_triangle The layer's own [triangle], on the pattern's ages.
#' @param k The credibility constant, positive.
#' @param max_influence The share of the blended factor one origin may move.
#' @param severity A severity distribution.
#' @param shift The scale shift `c` (`X_ultimate = c X_j`), positive.
#' @param exposure The exposure column of the triangle.
#' @param shares One share for every origin, or one per origin, each between
#'   0 and 1.
#' @param name The new column's name.
#' @returns `excess_pattern()`: an `excess_pattern`. `excess_selected()`: a
#'   list of `ldf`, `ldf_kind` and `ldf_weight`. `excess_credibility_weights()`:
#'   a numeric vector named by age. `layer_share()` and `layer_at_shift()`:
#'   a number. `layer_exposure()`: a [triangle].
#' @name excess_pattern
#' @examples
#' # Bloom and Patrik's four claims (100, 375, 250 and 500 at 12 months)
#' # developing 3.60 and 1.25 to ultimate, in the layer excess of 350.
#' long <- data.frame(year = 2020, age = c(12, 24, 36),
#'                    incurred = c(1225, 4410, 5512.5))
#' tri <- triangle(long, "year", "age", "incurred")
#' probs <- numeric(21)
#' probs[c(5, 16, 11, 21)] <- 0.25
#' p <- excess_pattern(tri, "incurred", excess_layer(350),
#'                     scale_shift(grid_distribution(112.5, probs)))
#' p@layer_ldf
#' p@layer_cdf
#' excess_selected(p)
#' layer_share(pareto(1, 2), excess_layer(4))
NULL

# The attachment and limit of a layer argument.
layer_args <- function(layer, arg = "layer") {
  if (!S7::S7_inherits(layer, excess_layer)) {
    stop(sprintf("%s must be an excess_layer()", arg), call. = FALSE)
  }
  list(attachment = layer@attachment, limit = layer@limit)
}

severity_arg <- function(x, arg) {
  if (!S7::S7_inherits(x, distribution)) {
    stop(sprintf("%s must be a severity distribution", arg), call. = FALSE)
  }
  x@ptr
}

#' @rdname excess_pattern
#' @export
excess_layer <- S7::new_class(
  "excess_layer",
  package = "prospicio",
  properties = list(
    attachment = S7::class_double,
    limit = S7::class_double,
    exhaustion = S7::new_property(S7::class_double, getter = function(self) {
      self@attachment + self@limit
    })
  ),
  constructor = function(attachment, limit = Inf) {
    attachment <- single_number(attachment, "attachment")
    limit <- single_number(limit, "limit")
    if (!is.finite(attachment) || attachment < 0) {
      stop("attachment must be finite and not negative", call. = FALSE)
    }
    if (limit <= 0) stop("limit must be positive, or Inf for an unlimited layer", call. = FALSE)
    S7::new_object(S7::S7_object(), attachment = attachment, limit = limit)
  }
)

S7::method(print, excess_layer) <- function(x, ...) {
  cat(sprintf("excess_layer(attachment = %s, limit = %s)\n",
              format(x@attachment, digits = 15), format(x@limit, digits = 15)))
  invisible(x)
}

#' @rdname excess_pattern
#' @export
scale_shift <- S7::new_class(
  "scale_shift",
  package = "prospicio",
  properties = list(
    ultimate = S7::class_any,
    ground_up_cap = S7::new_union(NULL, S7::class_double)
  ),
  constructor = function(ultimate, ground_up_cap = NULL) {
    severity_arg(ultimate, "ultimate")
    if (!is.null(ground_up_cap)) ground_up_cap <- single_number(ground_up_cap, "ground_up_cap")
    S7::new_object(S7::S7_object(), ultimate = ultimate, ground_up_cap = ground_up_cap)
  }
)

#' @rdname excess_pattern
#' @export
pareto_shape <- S7::new_class(
  "pareto_shape",
  package = "prospicio",
  properties = list(
    threshold = S7::class_double,
    alphas = S7::class_double,
    ultimate_alpha = S7::class_double,
    base = S7::class_any
  ),
  constructor = function(threshold, alphas, ultimate_alpha, base = "counts") {
    threshold <- single_number(threshold, "threshold")
    ultimate_alpha <- single_number(ultimate_alpha, "ultimate_alpha")
    a <- age_values(alphas, "alphas")
    if (!(identical(base, "counts") || S7::S7_inherits(base, excess_layer))) {
      stop("base must be \"counts\" or an excess_layer()", call. = FALSE)
    }
    S7::new_object(S7::S7_object(), threshold = threshold,
                   alphas = stats::setNames(a$values, format(a$ages, trim = TRUE, scientific = FALSE)),
                   ultimate_alpha = ultimate_alpha, base = base)
  }
)

#' @rdname excess_pattern
#' @export
excess_pattern <- S7::new_class(
  "excess_pattern",
  package = "prospicio",
  properties = list(
    ages = S7::class_double,
    layer_ldf = S7::class_double,
    layer_cdf = S7::class_double,
    ground_up_ldf = S7::class_double,
    ground_up_tail = S7::class_double,
    shifts = S7::class_double,
    tail = S7::class_double,
    ldf = S7::new_property(S7::class_double, getter = function(self) {
      stats::setNames(self@layer_ldf, format(self@ages[seq_along(self@layer_ldf)],
                                              trim = TRUE, scientific = FALSE))
    })
  ),
  constructor = function(triangle, column = NULL, layer, model, counts = NULL, tail = 1) {
    column <- fit_column(triangle, column)
    l <- layer_args(layer)
    if (!is.null(counts) && (!is.character(counts) || length(counts) != 1)) {
      stop("counts must be NULL or the name of a column", call. = FALSE)
    }
    nan <- NaN
    args <- if (S7::S7_inherits(model, scale_shift)) {
      list(kind = "scale_shift", ultimate = severity_arg(model@ultimate, "ultimate"),
           ground_up_cap = if (is.null(model@ground_up_cap)) nan else model@ground_up_cap,
           threshold = nan, alpha_ages = numeric(), alphas = numeric(), ultimate_alpha = nan,
           base_attachment = nan, base_limit = nan)
    } else if (S7::S7_inherits(model, pareto_shape)) {
      base <- if (S7::S7_inherits(model@base, excess_layer)) layer_args(model@base, "base") else
        list(attachment = nan, limit = nan)
      list(kind = "pareto_shape", ultimate = NULL, ground_up_cap = nan,
           threshold = model@threshold, alpha_ages = as.double(names(model@alphas)),
           alphas = unname(model@alphas), ultimate_alpha = model@ultimate_alpha,
           base_attachment = base$attachment, base_limit = base$limit)
    } else {
      stop("model must be a scale_shift() or a pareto_shape()", call. = FALSE)
    }
    out <- rust_result(excess_pattern_rust(
      triangle@ptr, column, l$attachment, l$limit, args$kind, args$ultimate, args$ground_up_cap,
      args$threshold, args$alpha_ages, args$alphas, args$ultimate_alpha, args$base_attachment,
      args$base_limit, counts, tail_ptr(tail)))
    S7::new_object(S7::S7_object(), ages = out$ages, layer_ldf = out$layer_ldf,
                   layer_cdf = out$layer_cdf, ground_up_ldf = out$ground_up_ldf,
                   ground_up_tail = out$ground_up_tail, shifts = out$shifts,
                   tail = out$layer_tail)
  }
)

S7::method(print, excess_pattern) <- function(x, ...) {
  cat("excess_pattern\n")
  print(rbind(layer_ldf = x@ldf), digits = 7)
  cat(sprintf("tail = %s\n", format(x@tail, digits = 15)))
  invisible(x)
}

#' @rdname excess_pattern
#' @export
excess_selected <- function(pattern, weight = 1) {
  if (!S7::S7_inherits(pattern, excess_pattern)) {
    stop("pattern must be an excess_pattern", call. = FALSE)
  }
  ok <- is.numeric(weight) && (length(weight) == 1 && is.null(names(weight)) ||
                                 !is.null(names(weight)))
  if (!ok || anyNA(weight)) {
    stop("weight must be a number or a named numeric vector, the names ages in months",
         call. = FALSE)
  }
  list(ldf = pattern@ldf, ldf_kind = "ldf", ldf_weight = weight)
}

#' @rdname excess_pattern
#' @export
excess_credibility_weights <- function(pattern, layer_triangle, column = NULL, k,
                                       max_influence, counts = NULL) {
  if (!S7::S7_inherits(pattern, excess_pattern)) {
    stop("pattern must be an excess_pattern", call. = FALSE)
  }
  column <- fit_column(layer_triangle, column)
  if (!is.null(counts) && (!is.character(counts) || length(counts) != 1)) {
    stop("counts must be NULL or the name of a column", call. = FALSE)
  }
  out <- rust_result(excess_credibility_weights_rust(
    pattern@ages, pattern@layer_ldf, pattern@layer_cdf, pattern@ground_up_ldf,
    pattern@ground_up_tail, pattern@shifts, pattern@tail, layer_triangle@ptr, column, counts,
    single_number(k, "k"), single_number(max_influence, "max_influence")))
  stats::setNames(out$weights, format(out$ages, trim = TRUE, scientific = FALSE))
}

#' @rdname excess_pattern
#' @export
layer_share <- function(severity, layer) {
  l <- layer_args(layer)
  rust_result(excess_layer_share_rust(severity_arg(severity, "severity"), l$attachment, l$limit))
}

#' @rdname excess_pattern
#' @export
layer_at_shift <- function(severity, layer, shift) {
  l <- layer_args(layer)
  rust_result(excess_layer_at_shift_rust(severity_arg(severity, "severity"), l$attachment, l$limit,
                                         single_number(shift, "shift")))
}

#' @rdname excess_pattern
#' @export
layer_exposure <- function(triangle, exposure, shares, name) {
  check_triangle(triangle)
  if (!is.numeric(shares) || anyNA(shares) || !length(shares)) {
    stop("shares must be a number or a numeric vector", call. = FALSE)
  }
  ptr <- rust_result(excess_layer_exposure_rust(triangle@ptr, exposure, as.double(shares), name))
  triangle(ptr = ptr)
}
