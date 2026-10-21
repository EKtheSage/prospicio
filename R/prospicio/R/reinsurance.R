# Aggregate lane: reinsurance layers and towers, over
# crates/prospicio-r/src/reinsurance.rs. The #' comments are the package
# documentation (see distributions.R).

#' @include aggregate.R
NULL

#' Excess-of-loss reinsurance layer
#'
#' `limit` xs `attachment` on each loss, then annual terms. For one year,
#' `ceded = share * min(max(sum of per-loss recoveries - aggregate_deductible,
#' 0), aggregate_limit)`, less any loss corridor ([with_loss_corridor()])
#' before the annual limit.
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
#' @param premium Upfront premium for the placed share: the base of paid
#'   reinstatements, ceding and profit commissions and a sliding scale's loss
#'   ratio. [with_rate_on_line()] and its siblings set it from a quote.
#' @param reinstatement_rates Paid reinstatements: one rate per reinstatement,
#'   as a fraction of `premium` (1 is 100%), pro rata as to amount. Sets the
#'   annual limit to `limit * (length(reinstatement_rates) + 1)`; cannot be
#'   combined with `reinstatements` or a finite `aggregate_limit`.
#' @param pro_rata_time Paid reinstatements also pro rata as to time: the
#'   limit a loss at time `t` (the fraction of the year elapsed) uses up is
#'   charged at `1 - t`. Needs `reinstatement_rates`, and events with times
#'   ([with_uniform_times()], or `times` in [events_from_years()]).
#' @param allocation How the annual terms (aggregate deductible and limit,
#'   reinstatements) split the year's recovery among its events:
#'   `"event_order"` uses the aggregate limit up in the order the events
#'   occur, so a later event gets what is left; `"pro_rata"` scales every
#'   event's recovery by the same factor, the annual recovery after the terms
#'   over the one before them. The annual total is the same; the split by
#'   event, and so what an inuring stage sees, differs. Cannot be combined
#'   with `pro_rata_time`.
#' @param ptr Internal: an existing layer to wrap.
#' @returns An `xol_layer` object with read-only properties for each term
#'   (`loss_corridor` is `c(lower, upper, retained)`, `ceding_commission` the
#'   flat rate, `sliding_scale` a data frame of `commission` and `loss_ratio`
#'   (or `NULL`), `profit_commission` `c(share, allowance)` and
#'   `swing_rating` `c(basic, lcm, minimum, maximum)`; `profit_commission_carry`
#'   `list(years)` and `sliding_scale_carry` `list(deficit, credit, years)`
#'   (`NULL` when nothing is carried, `years` `NULL` without limit); each empty when there
#'   is none), and `sees_aggregate_part`, whether the layer cedes from an
#'   event set's aggregate part ([with_aggregate_part()]: on the loss with
#'   attachment 0 and no limit, a quota share or a stop loss).
#' @seealso [quota_share()] and [aggregate_stop_loss()] for the other contract
#'   types, which are layers too; [with_loss_corridor()] for a corridor;
#'   [with_rate_on_line()], [with_ceding_commission()] and
#'   [with_swing_rating()] for premiums and commissions.
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
#' ceded_by_event(xol_layer("10x5", 10, 5, reinstatements = 1), c(15, 20, 10))
#' ceded_by_event(xol_layer("10x5", 10, 5, reinstatements = 1, allocation = "pro_rata"),
#'                c(15, 20, 10))
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
    pro_rata_time = S7::new_property(S7::class_logical, getter = function(self) self@ptr$pro_rata_time()),
    sees_aggregate_part = S7::new_property(
      S7::class_logical, getter = function(self) self@ptr$sees_aggregate_part()
    ),
    allocation = S7::new_property(S7::class_character, getter = function(self) self@ptr$allocation()),
    loss_corridor = S7::new_property(S7::class_double, getter = function(self) self@ptr$loss_corridor()),
    ceding_commission = S7::new_property(
      S7::class_double, getter = function(self) self@ptr$ceding_commission()
    ),
    sliding_scale = S7::new_property(S7::class_any, getter = function(self) {
      a <- self@ptr$sliding_scale()
      if (!length(a)) return(NULL)
      n <- length(a) / 2
      data.frame(commission = a[seq_len(n)], loss_ratio = a[n + seq_len(n)])
    }),
    profit_commission = S7::new_property(S7::class_double, getter = function(self) {
      p <- self@ptr$profit_commission()
      if (length(p)) stats::setNames(p, c("share", "allowance")) else p
    }),
    swing_rating = S7::new_property(S7::class_double, getter = function(self) {
      p <- self@ptr$swing_rating()
      if (length(p)) stats::setNames(p, c("basic", "lcm", "minimum", "maximum")) else p
    }),
    profit_commission_carry = S7::new_property(S7::class_any, getter = function(self) {
      p <- self@ptr$profit_commission_carry()
      if (!length(p)) return(NULL)
      list(years = if (is.nan(p[[1]])) NULL else p[[1]])
    }),
    sliding_scale_carry = S7::new_property(S7::class_any, getter = function(self) {
      p <- self@ptr$sliding_scale_carry()
      if (!length(p)) return(NULL)
      list(deficit = p[[1]] == 1, credit = p[[2]] == 1,
           years = if (is.nan(p[[3]])) NULL else p[[3]])
    })
  ),
  constructor = function(name, limit, attachment, share = 1, aggregate_deductible = 0,
                         aggregate_limit = Inf, reinstatements = NULL, premium = 0,
                         reinstatement_rates = NULL, pro_rata_time = FALSE,
                         allocation = "event_order", ptr = NULL) {
    if (is.null(ptr)) {
      reinst <- if (is.null(reinstatements)) -1 else as.double(reinstatements)
      ptr <- rust_result(XolLayer$new(
        as.character(name), as.double(limit), as.double(attachment), as.double(share),
        as.double(aggregate_deductible), as.double(aggregate_limit), reinst,
        as.double(premium), as.double(reinstatement_rates), !is.null(reinstatement_rates),
        isTRUE(pro_rata_time), as.character(allocation)
      ))
    }
    S7::new_object(S7::S7_object(), ptr = ptr)
  }
)

#' Loss corridor
#'
#' The same layer with a loss corridor: of the annual layer loss at 100%
#' after the annual deductible, the cedant keeps `retained` of the part
#' between `lower` and `upper`; the annual limit then caps what is left, so
#' the reinsurer still pays up to the full annual limit. Reinstatement
#' premiums follow the loss after the corridor. A corridor quoted as loss
#' ratios `lr` on the reinsurer's premium `P` for a placed share `s` is
#' `lr * P / s` (for a quota share, `P / s` is the subject premium).
#'
#' @param layer An [xol_layer] (any contract type).
#' @param lower Lower bound; non-negative.
#' @param upper Upper bound; finite, above `lower`.
#' @param retained Share of the band the cedant keeps, in `(0, 1]`.
#' @returns An [xol_layer].
#' @export
#' @examples
#' # A 30% quota share; the cedant keeps the loss ratio between 70% and 90%
#' # of a subject premium of 100.
#' qs <- with_loss_corridor(quota_share("QS", 0.3), 70, 90)
#' ceded(qs, c(50, 30))
#' ceded(qs, 120)
with_loss_corridor <- function(layer, lower, upper, retained = 1) {
  xol_layer(ptr = rust_result(layer@ptr$with_loss_corridor(
    as.double(lower), as.double(upper), as.double(retained)
  )))
}

#' Layer premium from a quote
#'
#' The same layer with its premium for the placed share set from a quote at
#' 100% of the layer: a rate on line (`share * rol * limit`; needs a finite
#' limit), a rate on the subject premium (`share * rate * subject_premium`;
#' a quota share's ceded premium is a rate of 1), or a deposit amount
#' (`share * amount`). Set the share first. The premium is the base of
#' paid reinstatements, commissions and a sliding scale's loss ratio.
#'
#' @param layer An [xol_layer] (any contract type).
#' @param rol Rate on line.
#' @param rate Rate on the subject premium.
#' @param subject_premium The subject (underlying) premium.
#' @param amount Deposit premium at 100% of the layer.
#' @returns An [xol_layer].
#' @export
#' @examples
#' with_rate_on_line(xol_layer("10x10", 10e6, 10e6), 0.125)@premium
#' with_premium_rate(quota_share("QS", 0.3), 1, 1000)@premium
#' with_deposit_premium(xol_layer("5x5", 5e6, 5e6, share = 0.4), 1e6)@premium
with_rate_on_line <- function(layer, rol) {
  xol_layer(ptr = rust_result(layer@ptr$with_rate_on_line(as.double(rol))))
}

#' @rdname with_rate_on_line
#' @export
with_premium_rate <- function(layer, rate, subject_premium) {
  xol_layer(ptr = rust_result(layer@ptr$with_premium_rate(as.double(rate), as.double(subject_premium))))
}

#' @rdname with_rate_on_line
#' @export
with_deposit_premium <- function(layer, amount) {
  xol_layer(ptr = rust_result(layer@ptr$with_deposit_premium(as.double(amount))))
}

#' Ceding and profit commissions
#'
#' The same layer with a commission paid back to the cedant, figured on the
#' year's premium and ceded loss:
#'
#' - `with_ceding_commission()`: a flat `rate` of the year's premium (the
#'   swing-rated premium, if any; reinstatement premiums carry none).
#' - `with_sliding_scale()`: the commission rate at the year's ceded loss
#'   ratio (ceded loss over premium), linear between the anchors
#'   `(commission, loss_ratio)` and flat beyond the first and last, as
#'   `aggregate`'s `slide`. It replaces a flat commission.
#' - `with_profit_commission()`: `share * max(premium * (1 - allowance) -
#'   ceded, 0)`, as `aggregate`'s `pc <share> after <allowance>`; the
#'   allowance includes any ceding commission deducted before profit.
#'
#' A sliding scale and a profit commission need the premium set first, and
#' cannot be combined with swing rating or paid reinstatements. The tower's
#' results report each as its own component ([apply_tower()]).
#'
#' @param layer An [xol_layer].
#' @param rate Flat commission rate, in `[0, 1]`.
#' @param commission Commission rates at the anchors; must not rise with the
#'   loss ratio.
#' @param loss_ratio Loss ratios of the anchors, one per commission.
#' @param share Share of the profit, in `[0, 1]`.
#' @param allowance The reinsurer's expense allowance, as a fraction of
#'   premium.
#' @returns An [xol_layer].
#' @export
#' @examples
#' qs <- with_premium_rate(quota_share("QS", 0.5), 1, 200)
#' slid <- with_sliding_scale(qs, c(0.45, 0.25, 0.19), c(0.60, 0.70, 0.80))
#' ceding_commission_for(slid, c(55, 65, 90))
#' pc <- with_profit_commission(qs, 0.25, 0.1)
#' profit_commission_for(pc, c(60, 95))
with_ceding_commission <- function(layer, rate) {
  xol_layer(ptr = rust_result(layer@ptr$with_ceding_commission(as.double(rate))))
}

#' @rdname with_ceding_commission
#' @export
with_sliding_scale <- function(layer, commission, loss_ratio) {
  xol_layer(ptr = rust_result(layer@ptr$with_sliding_scale(as.double(commission), as.double(loss_ratio))))
}

#' @rdname with_ceding_commission
#' @export
with_profit_commission <- function(layer, share, allowance = 0) {
  xol_layer(ptr = rust_result(layer@ptr$with_profit_commission(as.double(share), as.double(allowance))))
}

#' Deficits and credits carried over a contract term
#'
#' The same layer, carrying amounts between the years of a term
#' ([apply_tower_term()]):
#'
#' - `with_profit_commission_carry_forward()`: a year whose ceded loss
#'   exceeds `premium * (1 - allowance)` pays no profit commission and
#'   carries the excess, which later years' profits cover before any
#'   commission is paid.
#' - `with_sliding_scale_carry_forward()`: deficits (ceded loss above the
#'   last anchor's loss ratio), credits (below the first anchor's) or both.
#'   Each year adds what was carried to its own ceded loss before the
#'   commission is read off the scale, and carries what is then still beyond
#'   the scale's ends.
#'
#' Amounts are used up oldest first. Set the profit commission or sliding
#' scale first. A single year ([apply_tower()]) starts with nothing carried.
#' `profit_commissions_over()` and `ceding_commissions_over()` give the
#' commissions over consecutive years with the given ceded losses.
#'
#' @param layer An [xol_layer].
#' @param years How many later years an amount may be carried into; `NULL`
#'   (the default) until it is used up.
#' @param deficit,credit Whether to carry deficits and credits.
#' @param ceded Numeric vector of the layer's ceded loss in consecutive
#'   years, at the placed share.
#' @returns An [xol_layer]; for the `*_over()` functions, a numeric vector,
#'   one commission per year.
#' @export
#' @examples
#' qs <- with_premium_rate(quota_share("QS", 0.5), 1, 200)
#' pc <- with_profit_commission_carry_forward(with_profit_commission(qs, 0.25, 0.1))
#' profit_commissions_over(pc, c(120, 40))
#' slid <- with_sliding_scale(qs, c(0.45, 0.25), c(0.60, 0.70))
#' ceding_commissions_over(with_sliding_scale_carry_forward(slid), c(80, 55))
with_profit_commission_carry_forward <- function(layer, years = NULL) {
  xol_layer(ptr = rust_result(layer@ptr$with_profit_commission_carry_forward(nan_if_null(years))))
}

#' @rdname with_profit_commission_carry_forward
#' @export
with_sliding_scale_carry_forward <- function(layer, deficit = TRUE, credit = FALSE, years = NULL) {
  xol_layer(ptr = rust_result(layer@ptr$with_sliding_scale_carry_forward(
    isTRUE(deficit), isTRUE(credit), nan_if_null(years)
  )))
}

#' @rdname with_profit_commission_carry_forward
#' @export
profit_commissions_over <- function(layer, ceded) {
  layer@ptr$profit_commissions_over(as.double(ceded))
}

#' @rdname with_profit_commission_carry_forward
#' @export
ceding_commissions_over <- function(layer, ceded) {
  layer@ptr$ceding_commissions_over(as.double(ceded))
}

#' Swing-rated premium
#'
#' The same layer, swing rated: the year's premium is `pmin(pmax(share *
#' basic + lcm * ceded, share * minimum), share * maximum)`, with the terms
#' quoted for 100% of the layer, as in `aggregate`, and the ceded loss at the
#' placed share. It replaces the fixed premium in the tower's results
#' (component `swing_premium`). It cannot be combined with paid
#' reinstatements, a sliding scale or a profit commission.
#'
#' @param layer An [xol_layer].
#' @param basic Basic premium.
#' @param lcm Loss conversion factor, such as `100 / 80`.
#' @param minimum Minimum premium; `NULL` for `basic`.
#' @param maximum Maximum premium; `NULL` for no cap.
#' @returns An [xol_layer].
#' @seealso [retro_premium()] for the same formula on an account's loss.
#' @export
#' @examples
#' l <- with_swing_rating(xol_layer("L", 1000, 0), 0, 1.25, 100, 300)
#' premium_for(l, c(40, 200, 400))
with_swing_rating <- function(layer, basic, lcm, minimum = NULL, maximum = NULL) {
  xol_layer(ptr = rust_result(layer@ptr$with_swing_rating(
    as.double(basic), as.double(lcm), nan_if_null(minimum), nan_if_null(maximum)
  )))
}

#' Premium and commissions for a year's ceded loss
#'
#' The year's premium (swing rated, or else the layer's fixed premium),
#' ceding commission (flat or sliding) and profit commission, for each
#' value of the layer's ceded loss at the placed share.
#'
#' @param layer An [xol_layer].
#' @param ceded Numeric vector of annual ceded losses.
#' @returns A numeric vector, one entry per ceded loss.
#' @export
#' @examples
#' qs <- with_ceding_commission(with_premium_rate(quota_share("QS", 0.3), 1, 1000), 0.3)
#' premium_for(qs, 150)
#' ceding_commission_for(qs, 150)
premium_for <- function(layer, ceded) {
  layer@ptr$premium_for(as.double(ceded))
}

#' @rdname premium_for
#' @export
ceding_commission_for <- function(layer, ceded) {
  layer@ptr$ceding_commission_for(as.double(ceded))
}

#' @rdname premium_for
#' @export
profit_commission_for <- function(layer, ceded) {
  layer@ptr$profit_commission_for(as.double(ceded))
}

#' Retrospectively rated premium
#'
#' `pmin(pmax(basic + lcm * losses, minimum), maximum)`: the premium of a
#' retrospectively rated policy for each value of the account's loss (net of
#' any reinsurance that inures to it), as `aggregate`'s `RetroTerms`. For a
#' retro plan, `basic` and `lcm` include the tax multiplier.
#'
#' @param losses Numeric vector of annual losses.
#' @param basic Basic premium.
#' @param lcm Loss conversion factor.
#' @param minimum Minimum premium; `NULL` for `basic`.
#' @param maximum Maximum premium; `NULL` for no cap.
#' @returns A numeric vector, one premium per loss.
#' @seealso [with_swing_rating()] for a swing-rated reinsurance layer.
#' @export
#' @examples
#' retro_premium(c(0, 500, 1500), basic = 1000, lcm = 1.1, maximum = 2500)
retro_premium <- function(losses, basic, lcm, minimum = NULL, maximum = NULL) {
  rust_result(retro_premium_rust(
    as.double(losses), as.double(basic), as.double(lcm), nan_if_null(minimum), nan_if_null(maximum)
  ))
}

# extendr refuses NA for a double, so a missing term travels as NaN.
nan_if_null <- function(x) if (is.null(x)) NaN else as.double(x)

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
#' one occurrence, so it acts as an aggregate excess of loss. A numeric
#' vector holds plain annual losses, one per simulation, each taken the same
#' way, with no `event_set` or [predictive_distribution] to build.
#'
#' A contract stated on the loss ratio is a layer on money amounts: a
#' loss-ratio cap `c` on subject premium `P` is
#' `aggregate_stop_loss(name, c * P, 0)` with `with_premium_rate(layer, 1, P)`,
#' so the ceded loss ratio is the capped one, and a sliding scale keyed on it is [with_sliding_scale()].
#' Several scales on the same loss-ratio grid add pointwise into one;
#' [ceding_commission_for()] evaluates a scale at any ceded loss.
#'
#' @param tower A [reinsurance_tower].
#' @param events An `event_set` from [simulate_events()], a
#'   [predictive_distribution], or a numeric vector of annual losses.
#' @param ... Unused; for methods.
#' @returns A [predictive_distribution] with dimensions `kind` and `layer`:
#'   `("gross", "ground_up")`, `("ceded", <layer name>)` per layer,
#'   `("net", "retained")`, then `("reinstatement_premium", <layer name>)` per
#'   layer with paid reinstatements, `("swing_premium", <layer name>)` per
#'   swing-rated layer, `("ceding_commission", <layer name>)` per layer with a
#'   flat or sliding commission and `("profit_commission", <layer name>)` per
#'   layer with a profit commission. `aggregate(result, keep = "kind")` gives
#'   gross, total ceded and net per year; net is a loss, before premiums and
#'   commissions. Take one kind with [select_components()] before joining it
#'   to a book: summing over `kind` adds gross, ceded and net together.
#'   With labels on the events ([with_labels()]) the label dimensions follow
#'   `kind` and `layer`: gross, each layer's ceded loss and net come once per
#'   row of the events' `label_combinations`, each the sum over that
#'   combination's events (a layer's split by event under its `allocation`),
#'   and the other components get `""` labels; `aggregate(result, keep =
#'   c("kind", "layer"))` gives back the unlabelled result.
#' @export
#' @examples
#' ev <- simulate_events(poisson_count(2), lognormal_from_mean_cv(3e6, 1.5), 1000, seed = 7)
#' tw <- reinsurance_tower(list(xol_layer("5x5", 5e6, 5e6)))
#' res <- apply_tower(tw, ev)
#' aggregate(res, keep = "kind")@keys
#' cover <- with_sliding_scale(
#'   with_premium_rate(aggregate_stop_loss("cover", 120, 0), 1, 100),
#'   c(0.3, 0.1), c(0.5, 0.9)
#' )
#' annual <- apply_tower(reinsurance_tower(list(cover)), c(40, 70, 150))
#' draw_matrix(annual)
apply_tower <- S7::new_generic("apply_tower", "tower", function(tower, events, ...) S7::S7_dispatch())

S7::method(apply_tower, reinsurance_tower) <- function(tower, events, ...) {
  ptr <- if (S7::S7_inherits(events, predictive_distribution)) {
    tower@ptr$apply_aggregate(events@ptr)
  } else if (is.numeric(events)) {
    tower@ptr$apply_annual(as.double(events))
  } else {
    tower@ptr$apply(events@ptr)
  }
  predictive_distribution(ptr = rust_result(ptr, s7_call()))
}
#' Reinsurance tower over a contract term
#'
#' Applies the tower to consecutive years of the same simulated paths:
#' `years[[t]]` holds year `t`'s losses, and simulation `j` of every year
#' belongs to path `j`. Each year's ceded and net loss and premiums are what
#' [apply_tower()] gives for that year alone; a profit commission or sliding
#' scale that carries deficits or credits forward
#' ([with_profit_commission_carry_forward()]) reads what the path's earlier
#' years left. [term_years()] splits one simulation into the years of a term.
#'
#' @param tower A [reinsurance_tower].
#' @param years A list of [event_set]s, or of [predictive_distribution]s
#'   (each simulation's total taken as one aggregate loss), one per year,
#'   each with the same number of simulations.
#' @returns A [predictive_distribution] with dimensions `kind`, `layer` and
#'   `year`: each of [apply_tower()]'s components once per year, `year` from
#'   1.
#' @export
#' @examples
#' qs <- with_premium_rate(quota_share("QS", 1), 1, 100)
#' qs <- with_profit_commission_carry_forward(with_profit_commission(qs, 0.25, 0.1))
#' ev <- simulate_events(poisson_count(2), lognormal_from_mean_cv(30, 1), 3000, seed = 7)
#' term <- apply_tower_term(reinsurance_tower(list(qs)), term_years(ev, 3))
#' term@dims
apply_tower_term <- function(tower, years) {
  if (!is.list(years) || length(years) == 0) {
    stop("years must be a non-empty list", call. = FALSE)
  }
  ptrs <- lapply(years, function(y) y@ptr)
  ptr <- if (S7::S7_inherits(years[[1]], predictive_distribution)) {
    tower@ptr$apply_aggregate_term(ptrs)
  } else {
    tower@ptr$apply_term(ptrs)
  }
  predictive_distribution(ptr = rust_result(ptr))
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
#'   `c * step`), `stage_ceded` (one [grid_distribution] per stage, its
#'   layers' total cession, or `NULL` for a stage of several layers one of
#'   which has annual terms) with `stage_sizes` (the layers in each), `net`
#'   (a [grid_distribution] or `NULL`),
#'   `expected_reinstatement_premium` (named, 0 without paid reinstatements),
#'   `expected_premium` (the swing-rated premium's mean, or else the fixed
#'   premium), `expected_ceding_commission`, `expected_profit_commission`
#'   (each named, 0 without the term) and `on_points`.
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
    stage_ceded = lapply(r$stage_ceded, function(p) if (is.null(p)) NULL else grid_distribution(ptr = p)),
    stage_sizes = r$stage_sizes,
    net = if (is.null(r$net)) NULL else grid_distribution(ptr = r$net),
    expected_reinstatement_premium = stats::setNames(r$expected_reinstatement_premium, names),
    expected_premium = stats::setNames(r$expected_premium, names),
    expected_ceding_commission = stats::setNames(r$expected_ceding_commission, names),
    expected_profit_commission = stats::setNames(r$expected_profit_commission, names),
    on_points = r$on_points
  )
}

#' A tower on an annual loss distribution
#'
#' Applies a tower to an annual loss on a grid, each year's total treated as
#' one loss (as [apply_tower()] does for a predictive distribution's
#' total): a quota share or a stop-loss priced on a loss-ratio or
#' annual-loss distribution, with its expected sliding-scale or profit
#' commission. Every layer's terms are a function of the annual total, so
#' inuring stages and annual terms are all allowed, and the net always
#' exists. The expectations are exact for the grid; the ceded and net grids
#' are exact when their losses fall on grid points (`on_points`).
#'
#' @param tower A [reinsurance_tower] or [inuring_tower].
#' @param annual A [grid_distribution] of the annual loss.
#' @returns A list as [tower_on_grid()]'s, without `stage_ceded`: `gross`,
#'   `ceded` (named), `net`, `expected_reinstatement_premium`,
#'   `expected_premium`, `expected_ceding_commission`,
#'   `expected_profit_commission` (each named) and `on_points`.
#' @export
#' @examples
#' probs <- numeric(81)
#' probs[c(51, 61, 71, 81)] <- 0.25 # loss ratios 50% to 80% on 100
#' qs <- with_ceding_commission(with_deposit_premium(quota_share("QS", 0.4), 100), 0.3)
#' r <- tower_on_annual_grid(reinsurance_tower(list(qs)), grid_distribution(1, probs))
#' mean(r$ceded[["QS"]])
#' r$expected_ceding_commission
tower_on_annual_grid <- function(tower, annual) {
  r <- rust_result(tower@ptr$on_annual_grid(annual@ptr))
  names <- tower@layer_names
  list(
    gross = grid_distribution(ptr = r$gross),
    ceded = stats::setNames(lapply(r$ceded, function(p) grid_distribution(ptr = p)), names),
    net = grid_distribution(ptr = r$net),
    expected_reinstatement_premium = stats::setNames(r$expected_reinstatement_premium, names),
    expected_premium = stats::setNames(r$expected_premium, names),
    expected_ceding_commission = stats::setNames(r$expected_ceding_commission, names),
    expected_profit_commission = stats::setNames(r$expected_profit_commission, names),
    on_points = r$on_points
  )
}

#' Price each view of a tower on the grid
#'
#' Prices the gross, every layer's cession and the net of [tower_on_grid()]
#' each on its own by a distortion, as the Python package `aggregate`'s
#' `reins_price_df`: the expected loss, the premium (the distorted
#' expectation, [risk_measure()]), the margin `premium - loss` and the loss
#' ratio `loss / premium`. A concave distortion's prices are subadditive, so
#' then the ceded and net premiums together are at least the gross premium.
#'
#' @param grids The result of [tower_on_grid()].
#' @param distortion A [distortion].
#' @returns A data frame with columns `view` (`"gross"`, each layer's name,
#'   `"stage_<k>"` for the total cession of each stage with several layers
#'   that has one, `"net"` when the tower has a net grid), `loss`,
#'   `premium`, `margin` and `loss_ratio`.
#' @export
#' @examples
#' sev <- grid_distribution(1, c(0, 0.4, 0.3, 0.2, 0.1))
#' tw <- reinsurance_tower(list(xol_layer("2x2", 2, 2)))
#' r <- tower_on_grid(tw, poisson_count(3), sev, points = 200)
#' tower_price(r, distortion("proportional_hazard", 0.7))
tower_price <- function(grids, distortion) {
  # A one-layer stage's total is its layer, already a view.
  stages <- grids$stage_ceded
  keep <- which(grids$stage_sizes > 1 & !vapply(stages, is.null, logical(1)))
  totals <- if (length(keep)) stats::setNames(stages[keep], paste0("stage_", keep)) else list()
  views <- c(list(gross = grids$gross), grids$ceded, totals,
             if (!is.null(grids$net)) list(net = grids$net))
  loss <- vapply(views, mean, numeric(1))
  premium <- vapply(views, function(g) risk_measure(g, distortion), numeric(1))
  data.frame(view = names(views), loss = unname(loss), premium = unname(premium),
             margin = unname(premium - loss), loss_ratio = unname(loss / premium))
}

#' Statistics of a per-occurrence layer
#'
#' The numbers on a reinsurance submission for `share` of `limit` xs
#' `attachment` per occurrence on a severity grid, as the Python package
#' `aggregate`'s `reins_stats_df`: the chances that a claim reaches the
#' layer, `P(X > attachment)` (exclusive), and exhausts it,
#' `P(X >= attachment + limit)` (inclusive); the chance of a loss to the
#' layer in the year; the loss on line, the annual ceded mean over
#' `share * limit` (0 for an unlimited layer); and the moments of the excess
#' count (the count thinned to the claims that reach the layer), the layer
#' severity given a loss to it and the annual ceded loss. The share scales
#' the moments exactly.
#'
#' @param frequency A [poisson_count], [negative_binomial_count] or [binomial_count].
#' @param severity A [grid_distribution].
#' @param limit Per-occurrence limit; may be `Inf`.
#' @param attachment Attachment.
#' @param share Placed share, in `(0, 1]`.
#' @returns A list with `pr_attach`, `pr_detach`, `pr_loss`, `loss_on_line`,
#'   and `count`, `severity` and `aggregate`, each a named vector as
#'   [compound_moments()].
#' @export
#' @examples
#' sev <- grid_distribution(1, c(0, 0.4, 0.3, 0.2, 0.1))
#' s <- occurrence_layer_statistics(poisson_count(3), sev, limit = 2, attachment = 2)
#' s$pr_attach
#' s$aggregate
occurrence_layer_statistics <- function(frequency, severity, limit, attachment, share = 1) {
  r <- rust_result(occurrence_layer_statistics_rust(
    frequency@ptr, severity@ptr, as.double(share), as.double(limit), as.double(attachment)
  ))
  names <- c("mean", "variance", "third_cumulant", "cv", "skew")
  for (k in c("count", "severity", "aggregate")) r[[k]] <- stats::setNames(r[[k]], names)
  r
}

#' Statistics of a total cession and of an aggregate cover
#'
#' `occurrence_cession_statistics()` is the total cession of several
#' per-occurrence layers side by side, as the Python package `aggregate`'s
#' `Ceded` column of `reins_stats_df`: a claim cedes the sum of
#' `share * min(max(x - attachment, 0), limit)` over the layers. It gives the
#' chance a claim reaches the lowest layer and of a loss in the year, the
#' loss on line over the placed limits, and the moments of the ceded loss
#' per ground-up claim and in the year, exact for the grid.
#' `aggregate_layer_statistics()` is a layer of an annual total, an
#' aggregate cover such as a stop loss: the chances of reaching
#' (`P(S > attachment)`, also the chance of a loss to it) and exhausting it
#' (`P(S >= attachment + limit)`), the loss on line and the moments of the
#' ceded loss.
#'
#' @inheritParams occurrence_layer_statistics
#' @param limits,attachments,shares One per layer; `shares` is recycled.
#' @param annual A [grid_distribution] of the annual total the cover sees,
#'   such as `tower_on_grid()`'s `gross` or `net`.
#' @param limit,attachment,share The aggregate cover.
#' @returns `occurrence_cession_statistics()`: a list with `pr_attach`,
#'   `pr_loss`, `loss_on_line`, `severity` and `aggregate`.
#'   `aggregate_layer_statistics()`: a list with `pr_attach`, `pr_detach`,
#'   `loss_on_line` and `aggregate`. Moments are named vectors as
#'   [compound_moments()].
#' @export
#' @examples
#' sev <- grid_distribution(1, c(0, 0.4, 0.3, 0.2, 0.1))
#' occurrence_cession_statistics(poisson_count(3), sev, limits = c(1, 2), attachments = c(1, 2))
#' annual <- tower_on_grid(reinsurance_tower(list(xol_layer("2x2", 2, 2))),
#'                         poisson_count(3), sev, points = 200)$gross
#' aggregate_layer_statistics(annual, limit = 5, attachment = 6)
occurrence_cession_statistics <- function(frequency, severity, limits, attachments, shares = 1) {
  shares <- rep_len(as.double(shares), length(limits))
  r <- rust_result(occurrence_cession_statistics_rust(
    frequency@ptr, severity@ptr, shares, as.double(limits), as.double(attachments)
  ))
  names <- c("mean", "variance", "third_cumulant", "cv", "skew")
  for (k in c("severity", "aggregate")) r[[k]] <- stats::setNames(r[[k]], names)
  r
}

#' @rdname occurrence_cession_statistics
#' @export
aggregate_layer_statistics <- function(annual, limit, attachment, share = 1) {
  r <- rust_result(aggregate_layer_statistics_rust(
    annual@ptr, as.double(share), as.double(limit), as.double(attachment)
  ))
  r$aggregate <- stats::setNames(r$aggregate, c("mean", "variance", "third_cumulant", "cv", "skew"))
  r
}
