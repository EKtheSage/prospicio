#' @include distributions.R discrete.R
NULL

#' P&L ledger
#'
#' A treaty's profit and loss as distributions, as the Python package
#' `aggregate`'s `PnL`: a source (a [grid_distribution], or `list(values,
#' probs)`) and its groups, each a book that sells cover (`"sell"`: books
#' `+consideration - obligation`) or buys it (`"buy"`: pays premium and
#' receives recoveries and commission). Every row (each leg, each side's
#' total, each group's result, running nets, tier and net-of blocks, and
#' with several groups the grand rows and the total impact) is a sum of
#' signed legs at each atom, with its exact distribution.
#'
#' `pnl_leg()` is one cash flow, a number or a function of the source's
#' values (called once with all of them and returning one value each),
#' written as a positive magnitude. `ledger_sheet()` is `aggregate`'s
#' `economic_df`: each row's mean, standard deviation, coefficient of
#' variation, skewness and scenario columns `k01` to `k99`, its conditional
#' mean given that the grand result equals its quantile at that level, so a
#' column is one state of the world and foots down the sheet (`k01` is the
#' bad year). `ledger_marginal()` gives each row's own quantiles,
#' `ledger_card()` the card (`summary_df`), `ledger_legs()` the declared
#' legs, `ledger_result()` and `ledger_row()` a row's distribution as a
#' [discrete_distribution]. The source's probabilities are rescaled to 1.
#'
#' A leg with a `step` has its own distribution put on a grid of that step
#' (`aggregate`'s `bs`), each value scattered linearly to its two
#' neighbours, which keeps its mean; only its marginal quantiles change, and
#' `ledger_validation()` compares the exact and grid means.
#' `ledger_ratios()` is `aggregate`'s `economic_ratios_df`: for each group,
#' span, net-of block and `All`, premium `P`, loss `L` net of recoveries,
#' expense `E` net of commission and result `M`, the result's `SD`, ratios
#' of means `LR`, `ER`, `CR`, means of ratios `E_LR`, `E_ER`, `E_CR`, and
#' `P_share` and `M_share` of the first block; a leg without a kind counts
#' as premium on the consideration side and loss on the other.
#' `ledger_stack()` is `stack_marginal_pnls`: each ledger's result
#' statistics, then each impact `c(label, target, base)` as the difference
#' of two stacked rows, statistic by statistic.
#'
#' @param label A leg's or group's label; leg labels are unique across the
#'   ledger.
#' @param value A number, or a function of the source's values.
#' @param kind `NULL` or one of `"premium"`, `"loss"`, `"expense"`,
#'   `"recovery"`, `"commission"`.
#' @param step `NULL`, or the step of a grid for the leg's own distribution.
#' @param role `"sell"` or `"buy"`.
#' @param consideration,obligation Lists of `pnl_leg()`s.
#' @param margin_label A seller's own margin label on a ledger that also
#'   buys cover; `"Gross"` by default.
#' @param source A [grid_distribution] or `list(values, probs)`.
#' @param groups A list of `pnl_group()`s.
#' @param name The ledger's name.
#' @param result_name The result row's name.
#' @param spans `NULL` or a list of `list(label, lo, hi)`: groups `lo` to
#'   `hi - 1` (counting from 0, as `aggregate`) with a subtotal block, and a
#'   net-of block when a later group follows.
#' @param force_tower Present one group as a tower.
#' @param ledger A `pnl_ledger()`.
#' @param row A row's label.
#' @param ledgers A named list of `pnl_ledger()`s, in row order.
#' @param impacts `NULL` or a list of `c(label, target, base)`.
#' @returns `pnl_leg()` and `pnl_group()` lists; `pnl_ledger()` a ledger;
#'   `ledger_sheet()`, `ledger_marginal()`, `ledger_card()` and
#'   `ledger_legs()`, `ledger_validation()`, `ledger_ratios()` and
#'   `ledger_stack()` data frames; `ledger_result()` and `ledger_row()` a
#'   [discrete_distribution].
#' @export
#' @examples
#' book <- pnl_group("Book", "sell", list(pnl_leg("Premium", 10)), list(pnl_leg("Loss", identity)))
#' pnl <- pnl_ledger(list(c(2, 8), c(0.5, 0.5)), list(book), name = "Book")
#' mean(ledger_result(pnl))
#' ledger_sheet(pnl)[, c("label", "EX", "k01", "k99")]
pnl_ledger <- function(source, groups, name = "ledger", result_name = "result", spans = NULL,
                       force_tower = FALSE) {
  src <- pnl_source(source)
  values <- src$values
  probs <- src$probs
  leg_group <- integer()
  leg_side <- character()
  leg_label <- character()
  leg_kind <- character()
  leg_values <- double()
  leg_step <- double()
  for (gi in seq_along(groups)) {
    g <- groups[[gi]]
    for (side in c("consideration", "obligation")) {
      for (leg in g[[side]]) {
        v <- if (is.function(leg$value)) leg$value(values) else leg$value
        v <- as.double(v)
        if (length(v) == 1) v <- rep(v, length(values))
        if (length(v) != length(values)) {
          stop(sprintf("leg %s: the function must return one value per source value", leg$label))
        }
        leg_group <- c(leg_group, gi)
        leg_side <- c(leg_side, side)
        leg_label <- c(leg_label, leg$label)
        leg_kind <- c(leg_kind, if (is.null(leg$kind)) "" else leg$kind)
        leg_values <- c(leg_values, v)
        leg_step <- c(leg_step, if (is.null(leg$step)) NaN else as.double(leg$step))
      }
    }
  }
  span_label <- vapply(spans, function(s) as.character(s[[1]]), character(1))
  span_lo <- vapply(spans, function(s) as.double(s[[2]]), double(1))
  span_hi <- vapply(spans, function(s) as.double(s[[3]]), double(1))
  ptr <- rust_result(PnlLedger$new(
    values, probs,
    vapply(groups, function(g) g$label, character(1)),
    vapply(groups, function(g) g$role, character(1)),
    vapply(groups, function(g) if (is.null(g$margin_label)) "" else g$margin_label, character(1)),
    as.double(leg_group), leg_side, leg_label, leg_kind, leg_values, leg_step,
    as.character(name), as.character(result_name), span_label, span_lo, span_hi, isTRUE(force_tower)
  ))
  structure(list(ptr = ptr), class = "pnl_ledger")
}

#' @rdname pnl_ledger
#' @export
pnl_leg <- function(label, value, kind = NULL, step = NULL) {
  kinds <- c("premium", "loss", "expense", "recovery", "commission")
  if (!is.null(kind) && !kind %in% kinds) stop(sprintf("unknown leg kind %s", kind))
  list(label = as.character(label), value = value, kind = kind, step = step)
}

#' @rdname pnl_ledger
#' @export
pnl_group <- function(label, role = c("sell", "buy"), consideration = list(), obligation = list(),
                      margin_label = NULL) {
  role <- match.arg(role)
  list(label = as.character(label), role = role, consideration = consideration,
       obligation = obligation, margin_label = margin_label)
}

#' @rdname pnl_ledger
#' @export
ledger_sheet <- function(ledger) {
  df <- as.data.frame(rust_result(ledger$ptr$sheet()), stringsAsFactors = FALSE)
  if (!ledger$ptr$is_tower()) df$step <- NULL
  df
}

#' @rdname pnl_ledger
#' @export
ledger_marginal <- function(ledger) {
  df <- as.data.frame(rust_result(ledger$ptr$marginal()), stringsAsFactors = FALSE)
  if (!ledger$ptr$is_tower()) df$step <- NULL
  df
}

#' @rdname pnl_ledger
#' @export
ledger_card <- function(ledger) {
  df <- as.data.frame(rust_result(ledger$ptr$card()), stringsAsFactors = FALSE)
  if (!ledger$ptr$is_tower()) df$step <- NULL
  df
}

#' @rdname pnl_ledger
#' @export
ledger_legs <- function(ledger) as.data.frame(ledger$ptr$legs(), stringsAsFactors = FALSE)

#' @rdname pnl_ledger
#' @export
ledger_validation <- function(ledger) as.data.frame(ledger$ptr$validation(), stringsAsFactors = FALSE)

#' @rdname pnl_ledger
#' @export
ledger_ratios <- function(ledger) as.data.frame(ledger$ptr$ratios(), stringsAsFactors = FALSE)

#' @rdname pnl_ledger
#' @export
ledger_stack <- function(ledgers, impacts = NULL) {
  cols <- c("EX", "SD", "CV", "Skew", sprintf("P%02d", c(1, 5, 10, 25, 50, 75, 90, 95, 99)))
  stats <- lapply(ledgers, function(l) rust_result(l$ptr$result_stats()))
  for (imp in impacts) {
    if (!all(imp[2:3] %in% names(stats))) {
      stop(sprintf("impact %s names a ledger that is not stacked", imp[[1]]))
    }
    stats[[imp[[1]]]] <- stats[[imp[[2]]]] - stats[[imp[[3]]]]
  }
  m <- do.call(rbind, stats)
  colnames(m) <- cols
  data.frame(label = names(stats), m, row.names = NULL, stringsAsFactors = FALSE)
}

#' @rdname pnl_ledger
#' @export
ledger_result <- function(ledger) discrete_distribution(ptr = ledger$ptr$result())

#' @rdname pnl_ledger
#' @export
ledger_row <- function(ledger, row) discrete_distribution(ptr = rust_result(ledger$ptr$row(as.character(row))))

#' @export
print.pnl_ledger <- function(x, ...) {
  r <- ledger_result(x)
  cat(sprintf("<pnl_ledger> %d rows, result mean %s\n", length(x$ptr$rows()), format(mean(r))))
  invisible(x)
}

pnl_source <- function(source) {
  if (S7::S7_inherits(source, grid_distribution)) {
    p <- source@probs
    keep <- p > 0
    list(values = (seq_along(p) - 1)[keep] * source@step, probs = p[keep])
  } else {
    list(values = as.double(source[[1]]), probs = as.double(source[[2]]))
  }
}

pnl_flat_expenses <- function(expenses) {
  idx <- integer()
  basis <- character()
  value <- double()
  label <- character()
  for (i in seq_along(expenses)) {
    e <- expenses[[i]]
    for (t in e$terms) {
      idx <- c(idx, i)
      basis <- c(basis, t[[1]])
      value <- c(value, as.double(t[[2]]))
      label <- c(label, if (is.null(e$label)) "" else e$label)
    }
  }
  list(index = as.double(idx), basis = basis, value = value, label = label)
}

#' Book of a seller and its annual covers
#'
#' The ledger of a book, as the Python package `aggregate`'s `pnl` and
#' `xpnl` builders: a gross premium less the annual loss less expenses,
#' over a source of annual gross losses, with the annual covers of a
#' [reinsurance_tower]. `book_ledger()` lays it out as one of three faces:
#'
#' - `"consolidated"` (`pnl`): one group, the net position. With a tower it
#'   books net premium (gross less ceded premiums) less the net loss less
#'   the commission received less expenses, over the net loss; a
#'   loss-basis expense books as its rate times the expected gross loss.
#' - `"walk"` (`xpnl`): the gross book, then one buy group per stage of the
#'   tower, with the running net and the covers' total impact.
#' - `"peel-top-down"` and `"peel-bottom-up"`: one buy group per layer, with
#'   an `All aggregate` subtotal.
#'
#' Every layer's terms are taken at the annual total. The builders book a
#' fixed premium and a flat ceding commission; a layer with loss-sensitive
#' terms is refused (write its legs with [pnl_leg()]).
#'
#' `pnl_expense()` is one expense leg: an amount (`"fixed"`), a fraction of
#' the gross premium (`"premium"`) or of the gross loss (`"loss"`), or
#' several terms. Unlabelled, a book's only expense is the leg `expense`;
#' with several, a single-basis one is `<basis> expense` and a mixed one
#' `expense`. `derive_premium()` is the gross premium whose expenses leave
#' a technical premium, `(T + F) / (1 - r)` (`aggregate`'s `derive
#' premium`).
#'
#' @param name The ledger's name.
#' @param premium The gross premium.
#' @param expenses A list of `pnl_expense()`s.
#' @param premium_label,loss_label,step_label The premium leg's label
#'   (`"Premium"`), the loss leg's (`"Loss"`, also the gross book's margin on
#'   a walk) and the gross book's step on a walk (`"Gross"`).
#' @param retro `NULL`, or `c(basic, lcm, minimum, maximum)` (`NA` for an
#'   open end) to rate the book retrospectively, `aggregate`'s `retro
#'   <collar> premium`: the premium is `clip(basic + lcm * x, minimum,
#'   maximum)` at the year's gross loss `x`, and `premium` (the collar's
#'   basic premium in `aggregate`) is what premium-basis expenses are taken
#'   on. A retro book buys no cover here.
#' @param basis `"fixed"`, `"premium"` or `"loss"`, or a list of
#'   `list(basis, value)` terms.
#' @param value The amount or rate, with a single `basis`.
#' @param label The leg's label.
#' @param book A `pnl_book()`.
#' @param source A [grid_distribution] or `list(values, probs)` of annual
#'   gross losses.
#' @param tower `NULL` or a [reinsurance_tower].
#' @param face `"consolidated"`, `"walk"`, `"peel-top-down"` or
#'   `"peel-bottom-up"`.
#' @param technical The technical premium.
#' @returns `pnl_book()` and `pnl_expense()` lists, `book_ledger()` a
#'   [pnl_ledger()], `derive_premium()` a number.
#' @export
#' @examples
#' cover <- with_ceding_commission(with_deposit_premium(aggregate_stop_loss("agg 3 xs 5", 3, 5), 1), 0.1)
#' book <- pnl_book("Ex3", 10, list(pnl_expense("premium", 0.1)))
#' gross <- list(c(2, 3, 4, 5, 6, 8), c(0.25, 0.3, 0.09, 0.2, 0.12, 0.04))
#' walk <- book_ledger(book, gross, reinsurance_tower(list(cover)), "walk")
#' mean(ledger_result(walk))
pnl_book <- function(name, premium, expenses = list(), premium_label = NULL, loss_label = NULL,
                     step_label = NULL, retro = NULL) {
  if (!is.null(retro)) {
    retro <- as.double(retro)
    if (length(retro) == 2) retro <- c(retro, NA, NA)
    if (length(retro) != 4) stop("retro is c(basic, lcm, minimum, maximum)")
    retro[is.na(retro)] <- NaN
  }
  structure(list(name = as.character(name), premium = as.double(premium), expenses = expenses,
                 premium_label = premium_label, loss_label = loss_label, step_label = step_label,
                 retro = retro),
            class = "pnl_book")
}

#' @rdname pnl_book
#' @export
pnl_expense <- function(basis, value = NULL, label = NULL) {
  terms <- if (is.character(basis)) {
    if (is.null(value)) stop("an expense with one basis needs a value")
    list(list(basis, value))
  } else {
    basis
  }
  if (length(terms) == 0) stop("an expense needs at least one term")
  list(terms = terms, label = label)
}

#' @rdname pnl_book
#' @export
book_ledger <- function(book, source, tower = NULL, face = "consolidated") {
  src <- pnl_source(source)
  e <- pnl_flat_expenses(book$expenses)
  lbl <- function(x) if (is.null(x)) "" else as.character(x)
  ptr <- rust_result(PnlLedger$book(
    src$values, src$probs, book$name, book$premium, e$index, e$basis, e$value, e$label,
    lbl(book$premium_label), lbl(book$loss_label), lbl(book$step_label),
    if (is.null(tower)) NULL else tower@ptr, as.character(face),
    if (is.null(book$retro)) double() else book$retro
  ))
  structure(list(ptr = ptr), class = "pnl_ledger")
}

#' @rdname pnl_book
#' @export
derive_premium <- function(technical, expenses) {
  e <- pnl_flat_expenses(expenses)
  rust_result(pnl_derive_premium(as.double(technical), e$index, e$basis, e$value, e$label))
}

#' Occurrence covers on the joint gross and ceded loss
#'
#' The ledger of a [pnl_book()] with per-occurrence covers, as the Python
#' package `aggregate`'s `xpnl` and `pnl` over its `occ_bivariate`. The
#' ceded aggregate is not a function of the gross aggregate (the claim count
#' decouples them), so the source is their joint law: `occurrence_source()`
#' sends each claim on the severity's grid to `(x, c(x))`, `c` the tower's
#' cession of one event (split linearly between grid points), compounds the
#' pairs by a two-dimensional FFT on `points` points per axis, and drops
#' cells below `1e-15` of the total mass as round-off.
#'
#' `book_occurrence_ledger()` lays the book out over that joint: on a walk
#' the occurrence covers are one group named after the first layer, then
#' each aggregate cover of `aggregate` recovers its cession of the net of
#' the occurrence covers; a peel needs exactly one occurrence layer; the
#' consolidated face is over the net loss at each atom. A single occurrence
#' layer may carry annual terms and paid reinstatements, on the joint's
#' unlimited annual cession `R`: its recovery `A(R)` and premium `D + h(R)`
#' then vary with the year. On a tier of several layers they are refused.
#'
#' @param frequency A claim count, as [poisson_count()].
#' @param severity A [grid_distribution] of one claim.
#' @param occurrence A [reinsurance_tower] of per-occurrence covers.
#' @param points The number of points on each axis, `c(gross, ceded)`.
#' @param book A [pnl_book()].
#' @param joint `list(gross, ceded, probs)`, from `occurrence_source()`.
#' @param aggregate `NULL` or a [reinsurance_tower] of aggregate covers.
#' @param face `"consolidated"`, `"walk"`, `"peel-top-down"` or
#'   `"peel-bottom-up"`.
#' @returns `occurrence_source()` a `list(gross, ceded, probs)`,
#'   `book_occurrence_ledger()` a [pnl_ledger()].
#' @export
#' @examples
#' sev <- grid_distribution(1, c(0, 0.5, 0.3, 0, 0.2))
#' occ <- reinsurance_tower(list(with_ceding_commission(
#'   with_deposit_premium(xol_layer("occ 2 xs 2", 2, 2), 2), 0.25)))
#' joint <- occurrence_source(poisson_count(2), sev, occ, c(64, 32))
#' book <- pnl_book("Ex4", 20, list(pnl_expense("premium", 0.1)))
#' mean(ledger_result(book_occurrence_ledger(book, joint, occ, face = "walk")))
occurrence_source <- function(frequency, severity, occurrence, points) {
  points <- rep_len(as.double(points), 2)
  rust_result(pnl_occurrence_source(
    frequency@ptr, severity@step, severity@probs, occurrence@ptr, points[1], points[2]
  ))
}

#' @rdname occurrence_source
#' @export
book_occurrence_ledger <- function(book, joint, occurrence, aggregate = NULL, face = "consolidated") {
  e <- pnl_flat_expenses(book$expenses)
  lbl <- function(x) if (is.null(x)) "" else as.character(x)
  ptr <- rust_result(PnlLedger$book_occurrence(
    as.double(joint[[1]]), as.double(joint[[2]]), as.double(joint[[3]]),
    book$name, book$premium, e$index, e$basis, e$value, e$label,
    lbl(book$premium_label), lbl(book$loss_label), lbl(book$step_label),
    occurrence@ptr, if (is.null(aggregate)) NULL else aggregate@ptr, as.character(face)
  ))
  structure(list(ptr = ptr), class = "pnl_ledger")
}

#' Premium ladder from target combined ratios
#'
#' The premium ladder of the Python package `aggregate`'s `pnl_program`,
#' as numbers: the net technical premium `net_loss / net_combined_ratio`,
#' each layer's deposit at 100% placement `round(ceded_loss /
#' combined_ratio / share)` (or the premium it states), and the booked
#' premium `round((T_net + sum share * deposit) / (1 - expense_ratio))`.
#' The book's own margin is `(1 - net_combined_ratio) T_net` whatever the
#' cover costs. `round_premium()` rounds as `aggregate` writes a premium:
#' to whole units above 100, to cents at or below it, ties to even;
#' `size_premium()` sizes one from a target loss ratio.
#'
#' @param net_loss Expected net loss.
#' @param net_combined_ratio Target net combined ratio.
#' @param expense_ratio Gross expenses as a fraction of premium.
#' @param layers `NULL` or a data frame with `ceded_loss` (expected, at the
#'   placed share), `share`, `combined_ratio` and optionally `stated` (a
#'   premium the contract states at the placed share, kept; `NA` to price).
#' @param premium A premium.
#' @param expected_loss Expected loss.
#' @param loss_ratio Target loss ratio.
#' @returns `pnl_program()` a list of `premium`, `technical`, `deposits` and
#'   `ceded_premium`; the others a number.
#' @export
#' @examples
#' pnl_program(406, 0.9, 0.25, data.frame(ceded_loss = 86, share = 1, combined_ratio = 0.6))
#' round_premium(c(714.12, 16.285))
pnl_program <- function(net_loss, net_combined_ratio, expense_ratio, layers = NULL) {
  n <- if (is.null(layers)) 0 else nrow(layers)
  col <- function(name, default) {
    if (n == 0) return(double())
    v <- if (name %in% names(layers)) as.double(layers[[name]]) else rep(default, n)
    v[is.na(v)] <- default
    v
  }
  stated <- col("stated", NaN)
  rust_result(pnl_program_rust(
    as.double(net_loss), as.double(net_combined_ratio), as.double(expense_ratio),
    col("ceded_loss", 0), col("share", 1), col("combined_ratio", 1), stated
  ))
}

#' @rdname pnl_program
#' @export
round_premium <- function(premium) vapply(as.double(premium), round_premium_rust, double(1))

#' @rdname pnl_program
#' @export
size_premium <- function(expected_loss, loss_ratio) {
  rust_result(size_premium_rust(as.double(expected_loss), as.double(loss_ratio)))
}
