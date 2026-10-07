# Aggregate lane: reinsurance layers and towers, over
# crates/prospicio-r/src/reinsurance.rs. The #' comments are the package
# documentation (see distributions.R).

#' @include aggregate.R
NULL

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
#' @param pro_rata_time Paid reinstatements also pro rata as to time: the
#'   limit a loss at time `t` (the fraction of the year elapsed) uses up is
#'   charged at `1 - t`. Needs `reinstatement_rates`, and events with times
#'   ([with_uniform_times()], or `times` in [events_from_years()]).
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
#' timed <- xol_layer("10x10", 10, 10, premium = 2, reinstatement_rates = c(1, 0.5),
#'                    pro_rata_time = TRUE)
#' reinstatement_premium(timed, c(22, 12), times = c(0.25, 0.5))
xol_layer <- S7::new_class(
  "xol_layer",
  package = "prospicio",
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
    ),
    pro_rata_time = S7::new_property(S7::class_logical, getter = function(self) self@ptr$pro_rata_time())
  ),
  constructor = function(name, limit, attachment, share = 1, aggregate_deductible = 0,
                         aggregate_limit = Inf, reinstatements = NULL, premium = 0,
                         reinstatement_rates = NULL, pro_rata_time = FALSE, ptr = NULL) {
    if (is.null(ptr)) {
      reinst <- if (is.null(reinstatements)) -1 else as.double(reinstatements)
      ptr <- rust_result(XolLayer$new(
        as.character(name), as.double(limit), as.double(attachment), as.double(share),
        as.double(aggregate_deductible), as.double(aggregate_limit), reinst,
        as.double(premium), as.double(reinstatement_rates), !is.null(reinstatement_rates),
        isTRUE(pro_rata_time)
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

#' Surplus treaty
#'
#' Each risk cedes the part of its sum insured above the retention line
#' `retention`, up to `lines` lines, and the same share of every loss on it:
#' `min(max(SI - retention, 0), lines * retention) / SI`. With a retention of
#' 1m and 9 lines (a capacity of 9m), a 5m risk cedes 80% and a 20m risk
#' 45%. The events must carry sums insured ([events_from_years()] with
#' `sums_insured`); it can inure to a per-risk excess of loss in a later
#' stage of an [inuring_tower()]. Applying it to an aggregate loss or on the
#' grid is refused: it works risk by risk.
#'
#' @param name Layer name, unique within a tower.
#' @param retention The retention line; positive.
#' @param lines Number of lines of capacity; positive.
#' @returns An [xol_layer].
#' @export
#' @examples
#' s <- surplus_treaty("surplus", 1e6, 9)
#' ceded(s, c(2e6, 2e6), sums_insured = c(5e6, 20e6))
surplus_treaty <- function(name, retention, lines) {
  xol_layer(ptr = rust_result(XolLayer$surplus(as.character(name), as.double(retention),
                                                as.double(lines))))
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
#' @param sums_insured For a [surplus_treaty()], the sum insured of the
#'   risk each loss hit, one per loss.
#' @param ... Unused; for methods.
#' @returns A single number.
#' @export
#' @examples
#' ceded(xol_layer("L", 10, 5), c(8, 20, 12))
#' ceded(surplus_treaty("S", 1e6, 9), c(2e6, 2e6), sums_insured = c(5e6, 20e6))
ceded <- S7::new_generic("ceded", "layer", function(layer, losses, ...) S7::S7_dispatch())

S7::method(ceded, xol_layer) <- function(layer, losses, sums_insured = NULL, ...) {
  if (is.null(sums_insured)) return(layer@ptr$ceded(as.double(losses)))
  rust_result(layer@ptr$ceded_with_sums_insured(as.double(losses), as.double(sums_insured)))
}

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
#' `k = 0, 1, ...`; zero when reinstatements are free. Pro rata as to time,
#' the limit each loss uses up is charged at `1 - t`, its time's share of the
#' year left.
#'
#' @param layer An [xol_layer].
#' @param losses Numeric vector of one year's losses, in time order.
#' @param times For a layer pro rata as to time, each loss's time as the
#'   fraction of the year elapsed; without them its premium is `NaN`.
#' @param ... Unused; for methods.
#' @returns A single number.
#' @export
#' @examples
#' l <- xol_layer("10x10", 10, 10, premium = 2, reinstatement_rates = c(1, 0.5))
#' reinstatement_premium(l, c(15, 25))
reinstatement_premium <- S7::new_generic(
  "reinstatement_premium", "layer", function(layer, losses, ...) S7::S7_dispatch()
)

S7::method(reinstatement_premium, xol_layer) <- function(layer, losses, times = NULL, ...) {
  rust_result(layer@ptr$reinstatement_premium(as.double(losses), as.double(times)))
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
  package = "prospicio",
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

#' Save and load reinsurance towers as JSON
#'
#' `tower_to_json()` writes a programme as a versioned JSON document: every
#' stage and layer with all its terms, numbers bit for bit.
#' `tower_from_json()` reads it back to an equal tower, rebuilding each
#' layer through the same checks as [xol_layer()], so a document with an
#' impossible term is refused.
#'
#' @param tower A [reinsurance_tower].
#' @param text A document written by `tower_to_json()`.
#' @returns `tower_to_json()`: a single string. `tower_from_json()`: a
#'   [reinsurance_tower].
#' @export
#' @examples
#' tw <- inuring_tower(list(list(surplus_treaty("S", 1e6, 4)),
#'                          list(xol_layer("XL", 2e6, 1e6))))
#' text <- tower_to_json(tw)
#' identical(tower_to_json(tower_from_json(text)), text)
tower_to_json <- function(tower) tower@ptr$to_json()

#' @rdname tower_to_json
#' @export
tower_from_json <- function(text) {
  reinsurance_tower(ptr = rust_result(ReinsuranceTower$from_json(as.character(text))))
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
