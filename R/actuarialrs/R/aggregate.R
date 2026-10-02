# Aggregate lane: compound distributions, simulated events and reinsurance,
# over crates/act-r/src/aggregate.rs. The #' comments are the package
# documentation (see distributions.R).

#' @include distributions.R
NULL

#' Compound (aggregate) loss distribution
#'
#' The distribution of `S = X_1 + ... + X_N` on the severity grid's step, by
#' Panjer's recursion or by FFT. The result's `report` property records the
#' error: `tail_mass` (probability above the last point, lumped onto it),
#' `aliasing_error` (FFT only; check it first) and `mean_error`.
#'
#' @param frequency A [poisson_count] or [negative_binomial_count].
#' @param severity A [grid_distribution], usually from [discretize()].
#' @param points Number of points in the aggregate grid.
#' @param method `"panjer"`, or `"fft"` for large claim counts where Panjer
#'   underflows.
#' @returns A [grid_distribution] whose `report` describes the calculation.
#' @export
#' @examples
#' sev <- grid_distribution(1, c(0.1, 0.3, 0.25, 0.2, 0.1, 0.05))
#' agg <- compound_distribution(poisson_count(3), sev, points = 100)
#' mean(agg)
#' agg@report$tail_mass
compound_distribution <- function(frequency, severity, points, method = c("panjer", "fft")) {
  method <- match.arg(method)
  ptr <- rust_result(compound(frequency@ptr, severity@ptr, as.double(points), method))
  grid_distribution(ptr = ptr)
}

#' Simulated years of losses
#'
#' Simulates `n_sims` years: a claim count from `frequency`, then that many
#' independent losses from `severity`. Year `i` uses only stream `i` of the
#' generator keyed by `seed`, so results do not depend on the number of
#' threads and match Python and Rust.
#'
#' @param frequency A [poisson_count] or [negative_binomial_count].
#' @param severity A [lognormal] or [grid_distribution].
#' @param n_sims Number of simulated years.
#' @param seed Generator seed: a non-negative whole number below 2^53.
#' @returns An `event_set`. Use [events()] for one year's losses,
#'   [event_counts()] for the number of losses per year, [total()] for the
#'   annual totals as a [predictive_distribution], and [apply_tower()] for
#'   reinsurance.
#' @export
#' @examples
#' ev <- simulate_events(poisson_count(5), lognormal_from_mean_cv(1000, 1), 10000, seed = 42)
#' ev@n_sims
#' mean(total(ev))
simulate_events <- function(frequency, severity, n_sims, seed) {
  ptr <- rust_result(EventSet$simulate(frequency@ptr, severity@ptr, as.double(n_sims), as.double(seed)))
  event_set(ptr = ptr)
}

#' Simulated years of losses (class)
#'
#' Created by [simulate_events()].
#'
#' @param ptr Internal.
#' @returns An `event_set` object with read-only properties `n_sims` and `seed`.
#' @export
#' @keywords internal
event_set <- S7::new_class(
  "event_set",
  package = "actuarialrs",
  properties = list(
    ptr = S7::new_S3_class("EventSet"),
    n_sims = S7::new_property(S7::class_double, getter = function(self) self@ptr$n_sims()),
    seed = S7::new_property(S7::class_double, getter = function(self) self@ptr$seed())
  ),
  constructor = function(ptr) S7::new_object(S7::S7_object(), ptr = ptr)
)

#' One simulated year's losses
#'
#' @param x An `event_set` from [simulate_events()].
#' @param sim Year number, from 1 to `x@n_sims`.
#' @param ... Unused; for methods.
#' @returns Numeric vector of that year's losses, in the order drawn.
#' @export
#' @examples
#' ev <- simulate_events(poisson_count(2), lognormal(0, 1), 10, seed = 1)
#' events(ev, 1)
events <- S7::new_generic("events", "x", function(x, sim, ...) S7::S7_dispatch())

#' Number of losses in each simulated year
#'
#' @inheritParams events
#' @returns Numeric vector of length `x@n_sims`.
#' @export
#' @examples
#' ev <- simulate_events(poisson_count(2), lognormal(0, 1), 10, seed = 1)
#' event_counts(ev)
event_counts <- S7::new_generic("event_counts", "x", function(x, ...) S7::S7_dispatch())

S7::method(events, event_set) <- function(x, sim, ...) rust_result(x@ptr$events(as.double(sim)), s7_call())
S7::method(event_counts, event_set) <- function(x, ...) x@ptr$counts()
S7::method(total, event_set) <- function(dist, ...) {
  predictive_distribution(ptr = rust_result(dist@ptr$totals(), s7_call()))
}
S7::method(print, event_set) <- function(x, ...) {
  cat(sprintf("<event_set> %d simulated years, seed %s\n", as.integer(x@n_sims), format(x@seed)))
  invisible(x)
}

#' Excess-of-loss reinsurance layer
#'
#' `limit` xs `attachment` on each loss, then annual terms. For one year,
#' `ceded = share * min(max(sum of per-loss recoveries - aggregate_deductible,
#' 0), aggregate_limit)`.
#'
#' @param name Layer name, unique within a tower.
#' @param limit Per-occurrence limit; may be `Inf`.
#' @param attachment Per-occurrence attachment.
#' @param share Placed share, in `(0, 1]`.
#' @param aggregate_deductible Annual aggregate deductible.
#' @param aggregate_limit Annual aggregate limit; `Inf` for none.
#' @param reinstatements Number of reinstatements, which sets the annual limit
#'   to `limit * (reinstatements + 1)`; cannot be combined with a finite
#'   `aggregate_limit`.
#' @returns An `xol_layer` object with read-only properties for each term.
#' @export
#' @examples
#' l <- xol_layer("5x5", 5e6, 5e6, reinstatements = 1)
#' ceded(l, 7e6)
#' ceded(l, c(12e6, 20e6, 30e6))
xol_layer <- S7::new_class(
  "xol_layer",
  package = "actuarialrs",
  properties = list(
    ptr = S7::new_S3_class("XolLayer"),
    name = S7::new_property(S7::class_character, getter = function(self) self@ptr$name()),
    limit = S7::new_property(S7::class_double, getter = function(self) self@ptr$limit()),
    attachment = S7::new_property(S7::class_double, getter = function(self) self@ptr$attachment()),
    share = S7::new_property(S7::class_double, getter = function(self) self@ptr$share())
  ),
  constructor = function(name, limit, attachment, share = 1, aggregate_deductible = 0,
                         aggregate_limit = Inf, reinstatements = NULL) {
    reinst <- if (is.null(reinstatements)) -1 else as.double(reinstatements)
    ptr <- rust_result(XolLayer$new(
      as.character(name), as.double(limit), as.double(attachment), as.double(share),
      as.double(aggregate_deductible), as.double(aggregate_limit), reinst
    ))
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Ceded loss for one year
#'
#' @param layer An [xol_layer].
#' @param losses Numeric vector of one year's losses.
#' @param ... Unused; for methods.
#' @returns A single number.
#' @export
#' @examples
#' ceded(xol_layer("L", 10, 5), c(8, 20, 12))
ceded <- S7::new_generic("ceded", "layer", function(layer, losses, ...) S7::S7_dispatch())

S7::method(ceded, xol_layer) <- function(layer, losses, ...) layer@ptr$ceded(as.double(losses))
S7::method(print, xol_layer) <- function(x, ...) {
  cat(sprintf("<xol_layer> %s: %s xs %s, share %s\n", x@name,
              format(x@limit), format(x@attachment), format(x@share)))
  invisible(x)
}

#' Reinsurance tower
#'
#' Layers applied to the same ground-up losses (no inuring order yet).
#'
#' @param layers A list of [xol_layer] objects with unique names.
#' @returns A `reinsurance_tower` object; apply it with [apply_tower()].
#' @export
#' @examples
#' tw <- reinsurance_tower(list(xol_layer("5x5", 5e6, 5e6), xol_layer("15x10", 15e6, 10e6)))
#' tw@layer_names
reinsurance_tower <- S7::new_class(
  "reinsurance_tower",
  package = "actuarialrs",
  properties = list(
    ptr = S7::new_S3_class("ReinsuranceTower"),
    layer_names = S7::new_property(S7::class_character, getter = function(self) self@ptr$layer_names())
  ),
  constructor = function(layers) {
    ptrs <- lapply(layers, function(l) l@ptr)
    S7::new_object(S7::S7_object(), ptr = rust_result(ReinsuranceTower$new(ptrs)))
  }
)

#' Apply a reinsurance tower to simulated years
#'
#' @param tower A [reinsurance_tower].
#' @param events An `event_set` from [simulate_events()].
#' @param ... Unused; for methods.
#' @returns A [predictive_distribution] with dimensions `kind` and `layer`:
#'   `("gross", "ground_up")`, `("ceded", <layer name>)` per layer and
#'   `("net", "retained")`. `aggregate(result, keep = "kind")` gives gross,
#'   total ceded and net per year.
#' @export
#' @examples
#' ev <- simulate_events(poisson_count(2), lognormal_from_mean_cv(3e6, 1.5), 1000, seed = 7)
#' tw <- reinsurance_tower(list(xol_layer("5x5", 5e6, 5e6)))
#' res <- apply_tower(tw, ev)
#' aggregate(res, keep = "kind")@keys
apply_tower <- S7::new_generic("apply_tower", "tower", function(tower, events, ...) S7::S7_dispatch())

S7::method(apply_tower, reinsurance_tower) <- function(tower, events, ...) {
  predictive_distribution(ptr = rust_result(tower@ptr$apply(events@ptr), s7_call()))
}
S7::method(print, reinsurance_tower) <- function(x, ...) {
  cat(sprintf("<reinsurance_tower> %s\n", paste(x@layer_names, collapse = ", ")))
  invisible(x)
}
