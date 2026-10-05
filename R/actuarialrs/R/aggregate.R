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
#' @param frequency A [poisson_count], [negative_binomial_count] or [binomial_count].
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
#' @param frequency A [poisson_count], [negative_binomial_count] or [binomial_count].
#' @param severity A [lognormal], [grid_distribution] or Pareto-family severity
#'   ([pareto], [piecewise_pareto], [log_affine_pareto], [generalized_pareto]).
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
#' @param reinstatements Number of free reinstatements, which sets the annual
#'   limit to `limit * (reinstatements + 1)`; cannot be combined with a finite
#'   `aggregate_limit`.
#' @param premium Upfront premium for the placed share; used only by
#'   `reinstatement_rates`.
#' @param reinstatement_rates Paid reinstatements: one rate per reinstatement,
#'   as a fraction of `premium` (1 is 100%), pro rata as to amount. Sets the
#'   annual limit to `limit * (length(reinstatement_rates) + 1)`; cannot be
#'   combined with `reinstatements` or a finite `aggregate_limit`.
#' @param ptr Internal: an existing layer to wrap.
#' @returns An `xol_layer` object with read-only properties for each term.
#' @seealso [quota_share()] and [aggregate_stop_loss()] for the other contract
#'   types, which are layers too.
#' @export
#' @examples
#' l <- xol_layer("5x5", 5e6, 5e6, reinstatements = 1)
#' ceded(l, 7e6)
#' ceded(l, c(12e6, 20e6, 30e6))
#' paid <- xol_layer("10x10", 10, 10, premium = 2, reinstatement_rates = c(1, 0.5))
#' reinstatement_premium(paid, c(22, 12))
xol_layer <- S7::new_class(
  "xol_layer",
  package = "actuarialrs",
  properties = list(
    ptr = S7::new_S3_class("XolLayer"),
    name = S7::new_property(S7::class_character, getter = function(self) self@ptr$name()),
    limit = S7::new_property(S7::class_double, getter = function(self) self@ptr$limit()),
    attachment = S7::new_property(S7::class_double, getter = function(self) self@ptr$attachment()),
    share = S7::new_property(S7::class_double, getter = function(self) self@ptr$share()),
    aggregate_deductible = S7::new_property(
      S7::class_double, getter = function(self) self@ptr$aggregate_deductible()
    ),
    aggregate_limit = S7::new_property(S7::class_double, getter = function(self) self@ptr$aggregate_limit()),
    premium = S7::new_property(S7::class_double, getter = function(self) self@ptr$premium()),
    reinstatement_rates = S7::new_property(
      S7::class_double, getter = function(self) self@ptr$reinstatement_rates()
    )
  ),
  constructor = function(name, limit, attachment, share = 1, aggregate_deductible = 0,
                         aggregate_limit = Inf, reinstatements = NULL, premium = 0,
                         reinstatement_rates = NULL, ptr = NULL) {
    if (is.null(ptr)) {
      reinst <- if (is.null(reinstatements)) -1 else as.double(reinstatements)
      ptr <- rust_result(XolLayer$new(
        as.character(name), as.double(limit), as.double(attachment), as.double(share),
        as.double(aggregate_deductible), as.double(aggregate_limit), reinst,
        as.double(premium), as.double(reinstatement_rates), !is.null(reinstatement_rates)
      ))
    }
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Quota share
#'
#' Cedes `cession` of every loss: an [xol_layer] with unlimited cover from 0
#' and `share = cession`.
#'
#' @param name Layer name, unique within a tower.
#' @param cession Ceded fraction, in `(0, 1]`.
#' @returns An [xol_layer].
#' @export
#' @examples
#' ceded(quota_share("QS", 0.4), c(10, 5))
quota_share <- function(name, cession) {
  xol_layer(ptr = rust_result(XolLayer$quota_share(as.character(name), as.double(cession))))
}

#' Aggregate stop-loss
#'
#' `limit` xs `retention` on the year's total loss: an [xol_layer] with
#' unlimited cover from 0, annual deductible `retention` and annual limit
#' `limit`. It covers the total of the losses it sees: gross, or net of
#' earlier stages in an [inuring_tower()].
#'
#' @param name Layer name, unique within a tower.
#' @param limit Annual limit; may be `Inf`.
#' @param retention Annual retention.
#' @returns An [xol_layer].
#' @export
#' @examples
#' ceded(aggregate_stop_loss("SL", 50, 100), c(60, 70))
aggregate_stop_loss <- function(name, limit, retention) {
  xol_layer(ptr = rust_result(XolLayer$stop_loss(
    as.character(name), as.double(limit), as.double(retention)
  )))
}

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

#' Ceded loss per event
#'
#' Takes one year's losses as chronological: the annual deductible absorbs
#' the first recoveries and the annual limit stops the last ones. The entries
#' sum to [ceded()].
#'
#' @param layer An [xol_layer].
#' @param losses Numeric vector of one year's losses, in time order.
#' @param ... Unused; for methods.
#' @returns A numeric vector, one entry per loss.
#' @export
#' @examples
#' l <- xol_layer("L", 10, 5, aggregate_deductible = 4, aggregate_limit = 15)
#' ceded_by_event(l, c(8, 20, 12))
ceded_by_event <- S7::new_generic("ceded_by_event", "layer", function(layer, losses, ...) S7::S7_dispatch())

S7::method(ceded_by_event, xol_layer) <- function(layer, losses, ...) {
  layer@ptr$ceded_by_event(as.double(losses))
}

#' Reinstatement premium for one year
#'
#' With layer loss `L` at 100% after annual terms,
#' `premium * sum(rate_k * min(max(L - k * limit, 0), limit) / limit)` over
#' `k = 0, 1, ...`; zero when reinstatements are free.
#'
#' @param layer An [xol_layer].
#' @param losses Numeric vector of one year's losses.
#' @param ... Unused; for methods.
#' @returns A single number.
#' @export
#' @examples
#' l <- xol_layer("10x10", 10, 10, premium = 2, reinstatement_rates = c(1, 0.5))
#' reinstatement_premium(l, c(15, 25))
reinstatement_premium <- S7::new_generic(
  "reinstatement_premium", "layer", function(layer, losses, ...) S7::S7_dispatch()
)

S7::method(reinstatement_premium, xol_layer) <- function(layer, losses, ...) {
  layer@ptr$reinstatement_premium(as.double(losses))
}
S7::method(print, xol_layer) <- function(x, ...) {
  cat(sprintf("<xol_layer> %s: %s xs %s, share %s\n", x@name,
              format(x@limit), format(x@attachment), format(x@share)))
  invisible(x)
}

#' Reinsurance tower
#'
#' Layers applied to the same ground-up losses, in one stage. For inuring
#' order, use [inuring_tower()].
#'
#' @param layers A list of [xol_layer] objects with unique names.
#' @param ptr Internal: an existing tower to wrap.
#' @returns A `reinsurance_tower` object; apply it with [apply_tower()]. Its
#'   `stages` property gives each layer's stage, from 1.
#' @export
#' @examples
#' tw <- reinsurance_tower(list(xol_layer("5x5", 5e6, 5e6), xol_layer("15x10", 15e6, 10e6)))
#' tw@layer_names
reinsurance_tower <- S7::new_class(
  "reinsurance_tower",
  package = "actuarialrs",
  properties = list(
    ptr = S7::new_S3_class("ReinsuranceTower"),
    layer_names = S7::new_property(S7::class_character, getter = function(self) self@ptr$layer_names()),
    stages = S7::new_property(S7::class_double, getter = function(self) self@ptr$stages())
  ),
  constructor = function(layers, ptr = NULL) {
    if (is.null(ptr)) {
      ptr <- rust_result(ReinsuranceTower$new(lapply(layers, function(l) l@ptr)))
    }
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Reinsurance tower with inuring order
#'
#' Stages apply in order: each stage's layers see the losses net of all
#' earlier stages, event by event, with annual terms used up in event order
#' (see [ceded_by_event()]).
#'
#' @param stages A list of stages, each a list of [xol_layer] objects. Layer
#'   names must be unique across stages.
#' @returns A [reinsurance_tower].
#' @export
#' @examples
#' # A 50% quota share inures to the benefit of a 5 xs 5 cover.
#' tw <- inuring_tower(list(list(quota_share("QS", 0.5)), list(xol_layer("5x5", 5, 5))))
#' tw@stages
#' tower_ceded(tw, 30)
inuring_tower <- function(stages) {
  ptrs <- lapply(stages, function(stage) lapply(stage, function(l) l@ptr))
  reinsurance_tower(ptr = rust_result(ReinsuranceTower$inuring(ptrs)))
}

#' Ceded loss of each layer in a tower for one year
#'
#' @param tower A [reinsurance_tower].
#' @param losses Numeric vector of one year's losses, in time order.
#' @returns A named numeric vector, one entry per layer.
#' @export
#' @examples
#' tw <- reinsurance_tower(list(xol_layer("5x5", 5, 5), xol_layer("10x10", 10, 10)))
#' tower_ceded(tw, c(7, 30))
tower_ceded <- function(tower, losses) {
  stats::setNames(tower@ptr$ceded(as.double(losses)), tower@layer_names)
}

#' Apply a reinsurance tower to simulated years
#'
#' With an `event_set` each year's individual losses go through the tower.
#' With a [predictive_distribution] (a reserve bootstrap, modelled premium
#' risk) each simulation's total is one aggregate loss: an adverse
#' development cover or loss portfolio transfer on reserves, a stop-loss or
#' quota share on premium risk. An occurrence layer then sees the total as
#' one occurrence, so it acts as an aggregate excess of loss.
#'
#' @param tower A [reinsurance_tower].
#' @param events An `event_set` from [simulate_events()], or a
#'   [predictive_distribution].
#' @param ... Unused; for methods.
#' @returns A [predictive_distribution] with dimensions `kind` and `layer`:
#'   `("gross", "ground_up")`, `("ceded", <layer name>)` per layer,
#'   `("net", "retained")`, then `("reinstatement_premium", <layer name>)` per
#'   layer with paid reinstatements. `aggregate(result, keep = "kind")` gives
#'   gross, total ceded and net per year; net is a loss, before premiums.
#' @export
#' @examples
#' ev <- simulate_events(poisson_count(2), lognormal_from_mean_cv(3e6, 1.5), 1000, seed = 7)
#' tw <- reinsurance_tower(list(xol_layer("5x5", 5e6, 5e6)))
#' res <- apply_tower(tw, ev)
#' aggregate(res, keep = "kind")@keys
apply_tower <- S7::new_generic("apply_tower", "tower", function(tower, events, ...) S7::S7_dispatch())

S7::method(apply_tower, reinsurance_tower) <- function(tower, events, ...) {
  ptr <- if (S7::S7_inherits(events, predictive_distribution)) {
    tower@ptr$apply_aggregate(events@ptr)
  } else {
    tower@ptr$apply(events@ptr)
  }
  predictive_distribution(ptr = rust_result(ptr, s7_call()))
}
S7::method(print, reinsurance_tower) <- function(x, ...) {
  cat(sprintf("<reinsurance_tower> %s\n", paste(x@layer_names, collapse = ", ")))
  invisible(x)
}

#' Reinsurance tower on the aggregate grid
#'
#' Exact annual distributions of gross, each layer's ceded loss and net, by
#' FFT, with no sampling error. Each layer's per-occurrence recoveries form a
#' severity grid, which is compounded with the same claim count; annual terms
#' and the share then apply to the total. With attachments, limits and annual
#' terms on multiples of the step, the grids are exact for the discretized
#' problem. Otherwise losses between points are split between their
#' neighbours: means stay exact and `on_points` is `FALSE`.
#'
#' The grids are marginal: use [apply_tower()] on simulated events for joint
#' results. `net` is given when no layer has annual terms, or when the last
#' stage is a single aggregate cover such as an [aggregate_stop_loss()]; otherwise it is
#' `NULL`. A tower whose annual terms inure to a later stage is rejected,
#' because those terms depend on event order.
#'
#' @param tower A [reinsurance_tower].
#' @param frequency A [poisson_count], [negative_binomial_count] or [binomial_count].
#' @param severity A [grid_distribution].
#' @param points Number of points in every aggregate grid.
#' @returns A list with `gross` (a [grid_distribution] whose `report`
#'   describes the compound calculation), `ceded` (a named list of
#'   [grid_distribution]s at the placed share; a share `c` gives step
#'   `c * step`), `net` (a [grid_distribution] or `NULL`),
#'   `expected_reinstatement_premium` (named, 0 without paid reinstatements)
#'   and `on_points`.
#' @export
#' @examples
#' sev <- grid_distribution(1, c(0, 0.4, 0.3, 0.2, 0.1))
#' tw <- reinsurance_tower(list(xol_layer("2x2", 2, 2)))
#' r <- tower_on_grid(tw, poisson_count(3), sev, points = 200)
#' mean(r$ceded[["2x2"]])
#' r$on_points
tower_on_grid <- function(tower, frequency, severity, points) {
  r <- rust_result(tower@ptr$on_grid(frequency@ptr, severity@ptr, as.double(points)))
  names <- tower@layer_names
  list(
    gross = grid_distribution(ptr = r$gross),
    ceded = stats::setNames(lapply(r$ceded, function(p) grid_distribution(ptr = p)), names),
    net = if (is.null(r$net)) NULL else grid_distribution(ptr = r$net),
    expected_reinstatement_premium = stats::setNames(r$expected_reinstatement_premium, names),
    on_points = r$on_points
  )
}
