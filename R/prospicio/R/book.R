# A book of deals on one shared simulation (Aggregate lane), over
# `prospicio_pricing::book` (`docs/design/time-value.md`, decisions 21 to 27).

#' A book of deals on one shared simulation
#'
#' `deal_run()` holds one deal's premium, loss and deductions in each
#' simulation of a shared simulation, and its inception in months from 1
#' January of the book's first year; `deal_run_from_tower()` takes them
#' from a result of a tower, the amounts behind [deal_ratios()], with each
#' year's reinstatement premium added to the deposit (or swing-rated)
#' premium unless `reinstatement_premium = FALSE`.
#' `deal_book()` holds several deals: simulation `j` of every deal is the
#' same year, so the book adds them without reordering.
#'
#' `book_amounts()` gives every item of every deal and of the book in each
#' simulation; `book_summary()` their means and ratios to the mean premium,
#' the book's sums over sums. `book_distribution()` gives the book's
#' statistics of an item with each deal's contribution at the book's ranks
#' and its marginal effect (the book less the book without it).
#' `book_cashflows()` moves each deal's investor flows to its inception and
#' sums them, with the book's IRR, NPV and MOIC on the summed flows, and
#' `book_calendar()` places premium, losses and flows in calendar years.
#'
#' @param name The deal's name.
#' @param premium,loss The deal's premium and loss, one a simulation.
#' @param deductions A named list of deductions (commission, brokerage, a
#'   fee), each one amount a simulation.
#' @param outside_nmbf Names of `deductions` that NMBF leaves out (a
#'   manager's fee).
#' @param inception The month the deal incepts, from 1 January of the
#'   book's first year. It moves the deal's flows and patterns, not its
#'   losses.
#' @param tower A [reinsurance_tower].
#' @param result A [predictive_distribution] from `tower`.
#' @param layer The layer's or deal's name in `tower`.
#' @param reinstatement_premium Whether each year's premium includes the
#'   reinstatement premium the tower computes for the year, pro rata as to
#'   amount and, where the layer says so, as to time (`TRUE`, the default),
#'   or is the deposit alone (`FALSE`, as [deal_ratios()] reads it).
#' @param cashflows `NULL`, or a [cashflow_run()] of `terms`, one
#'   simulation for each of the deal's; the deal takes its flows and the
#'   terms' patterns.
#' @param terms The [cashflow_terms()] of `cashflows`.
#' @param flows `NULL`, or the investor's net flows from any model: a matrix
#'   with a row a simulation and a column a month from inception.
#' @param funded,opening With `flows`, the capital paid in during the term
#'   and at inception, one a simulation.
#' @param patterns `NULL`, or `list(written, earned, paid)`, monthly
#'   cumulative shares from inception, for `book_calendar()`.
#' @param deals A list of `deal_run()`s.
#' @param book A `deal_book()`.
#' @param names Deal names.
#' @param item An item of `book_amounts()`: `"premium"`, `"loss"`, a
#'   deduction, `"total_deductions"`, `"nmbf"` or `"margin"`.
#' @param percentiles,tvar_levels,sd_multiples,quantile,tvar The grid of
#'   statistics, as [statistics()].
#' @param tail `"upper"`, or `"lower"` to read low values as the risk, as
#'   for a margin.
#' @param rate The annual rate the NPV discounts at, to month 0.
#' @param lower,upper,points,bisections,newton_steps,extend The IRR's
#'   search, as [irr()]: from -95% to 100% a year by default, a simulation
#'   whose return is above the range flagged by `root_above` and found with
#'   `extend = TRUE` (or a larger `upper`).
#' @param start_year The calendar year of book month 0.
#' @returns `deal_run()` and `deal_run_from_tower()` a `deal_run`;
#'   `deal_book()`, `book_select()` and `book_shift()` a `deal_book`;
#'   `book_names()` the deals' names in order, `book_n_sims()` the number of
#'   simulations and `book_items()` the items of `book_amounts()`
#'   (`premium`, `loss`, each deduction, `total_deductions`, `nmbf` and
#'   `margin`);
#'   `book_amounts()` a [predictive_distribution] with dimensions `item` and
#'   `deal` (the deals, then `"total"`); `book_summary()` a data frame of
#'   `deal`, `item`, `mean` and `ratio`; `book_distribution()` a list of
#'   `total` (named statistics) and `contribution` and `marginal` (matrices
#'   with a row a statistic and a column a deal); `book_cashflows()` a list
#'   of `months`, `flows` (a matrix, a row a simulation), `mean_by_deal` (a
#'   matrix, a column a deal), `irr` (a data frame as [irr()]), `npv`,
#'   `moic` and `irr_by_deal` (a list of data frames); `book_calendar()` a
#'   [predictive_distribution] with dimensions `item`, `deal` and `year`.
#' @seealso [deal_ratios()], [cashflow_terms()], [statistics()].
#' @export
#' @examples
#' a <- deal_run("a", rep(100, 4), c(40, 80, 20, 160),
#'               deductions = list(brokerage = rep(2.5, 4)))
#' b <- deal_run("b", rep(300, 4), c(150, 330, 90, 210),
#'               deductions = list(brokerage = rep(7.5, 4)))
#' book <- deal_book(list(a, b))
#' book_summary(book)
#' book_distribution(book, "loss", 90, tvar_levels = 50)$contribution
#' a <- deal_run("a", rep(100, 4), c(40, 80, 20, 160),
#'               flows = matrix(c(-10, 0, 10.1), 4, 3, byrow = TRUE),
#'               funded = rep(0, 4), opening = rep(10, 4))
#' book_cashflows(deal_book(list(a)))$irr
deal_run <- function(name, premium, loss, deductions = list(), outside_nmbf = character(),
                     inception = 0, cashflows = NULL, terms = NULL, flows = NULL,
                     funded = NULL, opening = NULL, patterns = NULL) {
  premium <- as.double(premium)
  dnames <- names(deductions)
  if (length(deductions) > 0 && (is.null(dnames) || any(dnames == ""))) {
    stop("deductions must be a named list", call. = FALSE)
  }
  if (length(deductions) == 0) dnames <- character()
  values <- unlist(lapply(deductions, function(d) rep_len(as.double(d), length(premium))),
                   use.names = FALSE)
  ptr <- rust_result(BookDeal$new(as.character(name), premium, as.double(loss),
                                  as.character(dnames), as.double(values),
                                  as.integer(dnames %in% outside_nmbf), as.double(inception)))
  deal_extras(structure(list(ptr = ptr), class = "deal_run"), cashflows, terms, flows,
              funded, opening, patterns)
}

#' @rdname deal_run
#' @export
deal_run_from_tower <- function(name, tower, result, layer, inception = 0, cashflows = NULL,
                                terms = NULL, flows = NULL, funded = NULL, opening = NULL,
                                patterns = NULL, reinstatement_premium = TRUE) {
  ptr <- rust_result(BookDeal$from_tower(as.character(name), tower@ptr, result@ptr,
                                         as.character(layer), as.double(inception),
                                         isTRUE(reinstatement_premium)))
  deal_extras(structure(list(ptr = ptr), class = "deal_run"), cashflows, terms, flows,
              funded, opening, patterns)
}

deal_extras <- function(deal, cashflows, terms, flows, funded, opening, patterns) {
  if (!is.null(patterns)) {
    deal$ptr <- rust_result(deal$ptr$with_patterns(as.double(patterns[[1]]),
                                                   as.double(patterns[[2]]),
                                                   as.double(patterns[[3]])))
  }
  if (!is.null(cashflows)) {
    if (!inherits(terms, "cashflow_terms")) {
      stop("cashflows needs its terms, a cashflow_terms()", call. = FALSE)
    }
    deal$ptr <- rust_result(deal$ptr$with_flows(ncol(cashflows$flows),
                                                as.double(t(cashflows$flows)),
                                                as.double(cashflows$funded),
                                                as.double(cashflows$opening), terms$ptr))
  } else if (!is.null(flows)) {
    flows <- as.matrix(flows)
    deal$ptr <- rust_result(deal$ptr$with_flows(ncol(flows), as.double(t(flows)),
                                                as.double(funded), as.double(opening), NULL))
  }
  deal
}

#' @export
print.deal_run <- function(x, ...) {
  cat(sprintf("<deal_run> %s: %d simulations, incepting in month %d\n", x$ptr$name(),
              x$ptr$n_sims(), x$ptr$inception()))
  invisible(x)
}

#' @rdname deal_run
#' @export
deal_book <- function(deals) {
  if (inherits(deals, "deal_run")) deals <- list(deals)
  if (!all(vapply(deals, inherits, logical(1), "deal_run"))) {
    stop("every deal must be a deal_run()", call. = FALSE)
  }
  ptr <- rust_result(Book$new(lapply(deals, function(d) d$ptr)))
  structure(list(ptr = ptr), class = "deal_book")
}

#' @export
print.deal_book <- function(x, ...) {
  cat(sprintf("<deal_book> %s: %d simulations\n", paste(x$ptr$names(), collapse = ", "),
              x$ptr$n_sims()))
  invisible(x)
}

check_book <- function(book) {
  if (!inherits(book, "deal_book")) stop("book must be a deal_book()", call. = FALSE)
}

#' @rdname deal_run
#' @export
book_names <- function(book) {
  check_book(book)
  book$ptr$names()
}

#' @rdname deal_run
#' @export
book_n_sims <- function(book) {
  check_book(book)
  book$ptr$n_sims()
}

#' @rdname deal_run
#' @export
book_items <- function(book) {
  check_book(book)
  book$ptr$items()
}

#' @rdname deal_run
#' @export
book_select <- function(book, names) {
  check_book(book)
  structure(list(ptr = rust_result(book$ptr$select(as.character(names)))), class = "deal_book")
}

#' @rdname deal_run
#' @export
book_shift <- function(book, name, inception) {
  check_book(book)
  structure(list(ptr = rust_result(book$ptr$shifted(as.character(name), as.double(inception)))),
            class = "deal_book")
}

#' @rdname deal_run
#' @export
book_amounts <- function(book) {
  check_book(book)
  predictive_distribution(ptr = rust_result(book$ptr$amounts()))
}

#' @rdname deal_run
#' @export
book_summary <- function(book) {
  check_book(book)
  r <- rust_result(book$ptr$summary())
  data.frame(deal = r$deal, item = r$item, mean = r$mean, ratio = r$ratio)
}

#' @rdname deal_run
#' @export
book_distribution <- function(book, item, percentiles, tvar_levels = double(),
                              sd_multiples = double(), quantile = "inverted_cdf",
                              tvar = "exact", tail = "upper") {
  check_book(book)
  r <- rust_result(book$ptr$distribution(as.character(item), as.double(percentiles),
                                         as.double(tvar_levels), as.double(sd_multiples),
                                         quantile, tvar, tail))
  shape <- function(v) matrix(v, nrow = length(r$co_names), dimnames = list(r$co_names, r$deals))
  list(total = stats::setNames(r$total, r$names), contribution = shape(r$contribution),
       marginal = shape(r$marginal))
}

#' The marginal price of a deal on a book
#'
#' The price of each deal written last: the book's loss priced with the
#' deal and without it, on the assets of the capital standard `assets` and
#' by exactly one of `cost_of_capital` and `distortion`, as
#' [price_portfolio()] prices a portfolio, and the differences. Under a
#' cost of capital `r` the premium is `(EL + r dA) / (1 + r)`, with `EL` the
#' deal's expected loss and `dA` the assets it adds. Under a subadditive
#' standard (every distortion) it is at most the deal's allocated premium
#' from [price_portfolio()]; the only deal in a book has its stand-alone
#' price.
#'
#' @param book A [deal_book()].
#' @param assets The capital standard, a [distortion].
#' @param cost_of_capital,distortion Exactly one: the cost of capital or a
#'   pricing [distortion].
#' @param deals The deals to price, all by default.
#' @returns A data frame with a row a deal: `deal`, `expected_loss`,
#'   `assets` (`dA`), `premium`, `margin`, and the book's `book_premium`,
#'   `book_assets`, `premium_without` and `assets_without`.
#' @seealso [price_portfolio()], [book_distribution()].
#' @export
#' @examples
#' a <- deal_run("a", rep(10, 4), c(1, 2, 3, 9))
#' b <- deal_run("b", rep(10, 4), c(4, 1, 2, 3))
#' book_marginal_price(deal_book(list(a, b)), distortion("tvar", 0.5), cost_of_capital = 0.1)
book_marginal_price <- function(book, assets, cost_of_capital = NULL, distortion = NULL,
                                deals = NULL) {
  check_book(book)
  if (is.null(deals)) deals <- book$ptr$names()
  r <- rust_result(book$ptr$marginal_price(as.character(deals), assets@ptr,
                                           rate_arg(cost_of_capital), pricing_arg(distortion)))
  as.data.frame(r)
}

#' Price adequacy of a book's deals
#'
#' For each deal and the book: the premium charged and kept (less every
#' deduction) beside the expected loss and margin, the allocated (Euler)
#' and marginal assets and capital and the return on each, the technical
#' and marginal premiums, the target premium that returns the target on
#' the allocated capital, and the adequacy, kept over target. The book's
#' losses are priced on the assets of the capital standard `assets` by
#' exactly one of `cost_of_capital` and `distortion`, as [price_portfolio()]
#' prices a portfolio; the target return is the book's return at its
#' technical price (the cost of capital, under one). On a book of one deal
#' the allocated and marginal bases agree.
#'
#' @param book A [deal_book()].
#' @param assets The capital standard, a [distortion].
#' @param cost_of_capital,distortion Exactly one: the cost of capital or a
#'   pricing [distortion].
#' @returns A data frame with a row a deal then `"total"`: `deal`,
#'   `premium_charged`, `premium_kept`, `expected_loss`, `margin`,
#'   `allocated_assets`, `allocated_capital`, `return_on_allocated`,
#'   `marginal_assets`, `marginal_capital`, `return_on_marginal`,
#'   `technical_premium`, `marginal_premium`, `target_premium`, `adequacy`
#'   and `target_return`.
#' @seealso [book_marginal_price()], [price_portfolio()].
#' @export
#' @examples
#' a <- deal_run("a", rep(10, 4), c(1, 2, 3, 9))
#' b <- deal_run("b", rep(3, 4), c(4, 1, 2, 3))
#' book_adequacy(deal_book(list(a, b)), distortion("tvar", 0.5), cost_of_capital = 0.1)
book_adequacy <- function(book, assets, cost_of_capital = NULL, distortion = NULL) {
  check_book(book)
  r <- rust_result(book$ptr$adequacy(assets@ptr, rate_arg(cost_of_capital), pricing_arg(distortion)))
  as.data.frame(r)
}

#' @rdname deal_run
#' @export
book_cashflows <- function(book, rate = 0, lower = -0.95, upper = 1, points = 400L,
                           bisections = 8L, newton_steps = 3L, extend = FALSE) {
  check_book(book)
  r <- rust_result(book$ptr$cashflows(as.double(rate), as.double(lower), as.double(upper),
                                      as.double(points), as.double(bisections),
                                      as.double(newton_steps), isTRUE(extend)))
  irr_frame <- function(x) {
    data.frame(rate = x$rate, found = x$found, npv = x$npv, sign_changes = x$sign_changes,
               root_above = x$root_above, root_below = x$root_below)
  }
  list(months = seq_len(r$months) - 1,
       flows = matrix(r$flows, ncol = r$months, byrow = TRUE),
       mean_by_deal = matrix(r$mean_by_deal, nrow = r$months, dimnames = list(NULL, r$deals)),
       irr = irr_frame(r$irr), npv = r$npv, moic = r$moic,
       irr_by_deal = stats::setNames(lapply(r$irr_by_deal, irr_frame), r$deals))
}

#' @rdname deal_run
#' @export
book_calendar <- function(book, start_year) {
  check_book(book)
  predictive_distribution(ptr = rust_result(book$ptr$calendar(as.integer(start_year))))
}
