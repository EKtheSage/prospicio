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
#' @param label A leg's or group's label; leg labels are unique across the
#'   ledger.
#' @param value A number, or a function of the source's values.
#' @param kind `NULL` or one of `"premium"`, `"loss"`, `"expense"`,
#'   `"recovery"`, `"commission"`.
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
#' @returns `pnl_leg()` and `pnl_group()` lists; `pnl_ledger()` a ledger;
#'   `ledger_sheet()`, `ledger_marginal()`, `ledger_card()` and
#'   `ledger_legs()` data frames; `ledger_result()` and `ledger_row()` a
#'   [discrete_distribution].
#' @export
#' @examples
#' book <- pnl_group("Book", "sell", list(pnl_leg("Premium", 10)), list(pnl_leg("Loss", identity)))
#' pnl <- pnl_ledger(list(c(2, 8), c(0.5, 0.5)), list(book), name = "Book")
#' mean(ledger_result(pnl))
#' ledger_sheet(pnl)[, c("label", "EX", "k01", "k99")]
pnl_ledger <- function(source, groups, name = "ledger", result_name = "result", spans = NULL,
                       force_tower = FALSE) {
  if (S7::S7_inherits(source, grid_distribution)) {
    p <- source@probs
    keep <- p > 0
    values <- (seq_along(p) - 1)[keep] * source@step
    probs <- p[keep]
  } else {
    values <- as.double(source[[1]])
    probs <- as.double(source[[2]])
  }
  leg_group <- integer()
  leg_side <- character()
  leg_label <- character()
  leg_kind <- character()
  leg_values <- double()
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
    as.double(leg_group), leg_side, leg_label, leg_kind, leg_values,
    as.character(name), as.character(result_name), span_label, span_lo, span_hi, isTRUE(force_tower)
  ))
  structure(list(ptr = ptr), class = "pnl_ledger")
}

#' @rdname pnl_ledger
#' @export
pnl_leg <- function(label, value, kind = NULL) {
  kinds <- c("premium", "loss", "expense", "recovery", "commission")
  if (!is.null(kind) && !kind %in% kinds) stop(sprintf("unknown leg kind %s", kind))
  list(label = as.character(label), value = value, kind = kind)
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
