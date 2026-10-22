# A collateralised deal's monthly cash flows (Aggregate lane), over
# prospicio_pricing::cashflow (docs/design/time-value.md, decisions 13 to 20).

#' A collateral requirement
#'
#' What a collateral account must hold at the end of each month: a sum of
#' loads on bases, floored, with an optional switch to a limit. With `W`,
#' `E` and `C` the premium written, the premium earned and the loss paid to
#' date, `u` the ultimate loss ratio and `P` the premium, the bases are
#' `"unearned-premium"` (`W - E`), `"reserve"` (`u E - C`), `"outstanding"`
#' (`u P - C`), `"written"` (`W`), `"earned"` (`E`) and `"fixed"` (1, so the
#' load is an amount).
#'
#' @param loads A named numeric vector, the load on each base.
#' @param floor The least the requirement can be.
#' @param limit `NULL`, or `c(trigger, cap)`: when `trigger * u >= cap` the
#'   requirement is `cap * W - C` in place of the loads.
#' @returns A `cashflow_requirement` for [cashflow_terms()].
#' @seealso [cashflow_terms()].
#' @export
#' @examples
#' cashflow_requirement(c("unearned-premium" = 0.3, reserve = 1.2), limit = c(1.2, 1.5))
cashflow_requirement <- function(loads, floor = 0, limit = NULL) {
  if (is.null(names(loads)) || any(names(loads) == "")) stop("name every load by its base")
  if (!is.null(limit) && length(limit) != 2) stop("limit is c(trigger, cap)")
  structure(list(bases = names(loads), loads = unname(as.double(loads)),
                 floor = as.double(floor), limit = as.double(limit)),
            class = "cashflow_requirement")
}

#' @export
print.cashflow_requirement <- function(x, ...) {
  cat("<cashflow_requirement>", paste(sprintf("%s %s", format(x$loads), x$bases),
                                      collapse = " + "))
  if (x$floor != 0) cat(sprintf(", floor %s", format(x$floor)))
  if (length(x$limit) == 2) {
    cat(sprintf(", limit %s W - C when %s u >= %s", format(x$limit[2]), format(x$limit[1]),
                format(x$limit[2])))
  }
  cat("\n")
  invisible(x)
}

#' A premium adjustment
#'
#' Paid into the trust at the end of a month: `written_rate` less `u` (the
#' ultimate loss ratio, or with `base = "to-date"` the loss paid to date
#' over the premium written to date), less the commission rate when
#' `less_commission`, times the premium written to date.
#'
#' @param month The month, before commutation (0 is the first).
#' @param written_rate The rate on written premium the loss ratio is set
#'   against.
#' @param less_commission Whether the commission rate comes off too.
#' @param base `"ultimate"` or `"to-date"`.
#' @returns A `cashflow_adjustment` for [cashflow_terms()]'s `adjustments`.
#' @seealso [cashflow_terms()].
#' @export
#' @examples
#' cashflow_adjustment(18, 0.3, less_commission = TRUE)
cashflow_adjustment <- function(month, written_rate, less_commission = FALSE,
                                base = c("ultimate", "to-date")) {
  structure(list(month = as.double(month), written_rate = as.double(written_rate),
                 less_commission = isTRUE(less_commission), base = match.arg(base)),
            class = "cashflow_adjustment")
}

# A set of months as the kind and values `Cashflow$new` reads: "monthly",
# "quarterly" or "at-commutation" (when allowed); one number, an interval
# from month 0; several numbers, or list(months = ...), the months listed;
# list(every = n, first = m).
cashflow_dates <- function(x, name, commutation = FALSE) {
  if (is.character(x)) {
    kinds <- c("monthly", "quarterly", if (commutation) "at-commutation")
    if (length(x) != 1 || !x %in% kinds) {
      stop(sprintf("%s: use %s, a number of months, several months, or list(every =, first =)",
                   name, paste(sprintf("\"%s\"", kinds), collapse = ", ")), call. = FALSE)
    }
    return(list(kind = x, values = double()))
  }
  if (is.list(x)) {
    if (!is.null(x$months)) return(list(kind = "months", values = as.double(x$months)))
    if (!is.null(x$every)) {
      first <- if (is.null(x$first)) 0 else x$first
      return(list(kind = "every", values = as.double(c(x$every, first))))
    }
    stop(sprintf("%s: a list holds months, or every and first", name), call. = FALSE)
  }
  if (is.numeric(x) && length(x) == 1) return(list(kind = "every", values = c(as.double(x), 0)))
  if (is.numeric(x)) return(list(kind = "months", values = as.double(x)))
  stop(sprintf("%s: unknown dates", name), call. = FALSE)
}

#' A collateralised deal's monthly cash flows
#'
#' `cashflow_terms()` holds a deal's premium, monthly patterns, collateral
#' requirement and one rule for each account. Month `t` runs from 0 to the
#' commutation month `T - 1`. Each month the trust takes the capital called,
#' the premium collected and any adjustment, pays the month's claims, earns
#' interest and releases what it holds above the requirement to the
#' operating account, which pays the expenses and releases to the investor
#' what it holds above its buffer. At commutation the trust pays the
#' commutation requirement to the cedant, the operating account takes the
#' rest (or makes good a shortfall), pays the fee and distributes what is
#' left.
#'
#' `cashflow_run()` simulates the investor's flows for each ultimate loss
#' ratio `u` and commission rate: `flows` has a row a simulation and a
#' column a month (`months`, 0 to `T - 1`), positive when paid to the
#' investor, so `npv(run$flows, run$months, rate, unit = "months")` and
#' [irr()] read them. `cashflow_trace()` gives every account line of one
#' simulation.
#'
#' Dates (`funding_dates`, `collateral_release`, `equity_release`) are
#' `"monthly"`, `"quarterly"`, a number of months (an interval from month
#' 0), several months, `list(months = ...)` or `list(every = n, first = m)`;
#' the two releases also take `"at-commutation"`.
#'
#' @param premium The premium `P`.
#' @param commutation_month `T`: the deal commutes at the end of month
#'   `T - 1`.
#' @param written,earned,paid Cumulative monthly shares of the premium
#'   written and earned and of the ultimate loss paid, all of one length, at
#'   least `T` plus `buffer_months`.
#' @param requirement A [cashflow_requirement()].
#' @param commutation_requirement What the trust pays at commutation.
#' @param collected_rate The share of written premium paid into the trust.
#' @param lag The months between writing and collection.
#' @param adjustments A list of [cashflow_adjustment()]s.
#' @param funding_dates The months capital may be called.
#' @param funding_rule `"shortfall"`, what the trust lacks, or
#'   `"obligation-increase"`, the rise in `S_t - premium_offset * W_{t-lag}`
#'   over `reference`, the obligation at the previous funding date or the
#'   highest at any earlier one.
#' @param premium_offset,reference See `funding_rule`.
#' @param funding_source `"investor"`, or `"operating-first"`: the
#'   operating account pays what it holds first.
#' @param collateral_release When the trust releases its excess.
#' @param annual_yield The trust's yield, compounded monthly.
#' @param interest_basis `"average"`, `"opening"` or `"closing"` balance.
#' @param opening_fixed,opening_share The operating account's opening
#'   capital, `opening_fixed + opening_share * premium`.
#' @param expense_rate The annual expense rate on the requirement, a
#'   twelfth a month.
#' @param buffer_months,buffer_paid,buffer_expenses The operating account
#'   keeps the requirement's rise over the next `buffer_months` months, with
#'   the claims paid and the expenses over them when asked.
#' @param equity_release When the operating account releases its excess to
#'   the investor.
#' @param fee `NULL`, or a performance fee's `list(breakpoints, slopes)`, as
#'   [tiered_fee()].
#' @param fee_basis What the fee's tiers read: `"profit"`,
#'   `"profit-over-premium"` or `"profit-over-capital"`.
#' @returns `cashflow_terms()` a `cashflow_terms`; `cashflow_run()` a list
#'   of `flows` (a matrix), `months` and, a value a simulation, `funded`,
#'   `opening`, `released`, `distributed`, `fee` and `commutation`;
#'   `cashflow_trace()` a list of `accounts` (a data frame, a row a month),
#'   `commutation_paid`, `fee` and `conservation_error` (the cash created or
#'   lost over the deal, 0 up to rounding).
#' @seealso [cashflow_requirement()], [cashflow_adjustment()], [npv()],
#'   [irr()].
#' @export
#' @examples
#' # A premium of 100 written at once, earned and paid evenly over three
#' # months; collateral is the reserve on earned premium and the deal
#' # commutes at the outstanding loss.
#' terms <- cashflow_terms(100, 3, rep(1, 3), (1:3) / 3, (1:3) / 3,
#'                         cashflow_requirement(c(reserve = 1)),
#'                         commutation_requirement = cashflow_requirement(c(outstanding = 1)))
#' run <- cashflow_run(terms, c(0.6, 1.5))
#' round(run$flows, 9)
#' run$funded
#' npv(run$flows, run$months, 0.05, unit = "months")
#' cashflow_trace(terms, 1.5)$accounts
cashflow_terms <- function(premium, commutation_month, written, earned, paid, requirement,
                           commutation_requirement = requirement, collected_rate = 1, lag = 0,
                           adjustments = list(), funding_dates = "monthly",
                           funding_rule = c("shortfall", "obligation-increase"),
                           premium_offset = 1, reference = c("previous-date", "running-max"),
                           funding_source = c("investor", "operating-first"),
                           collateral_release = "monthly", annual_yield = 0,
                           interest_basis = c("average", "opening", "closing"),
                           opening_fixed = 0, opening_share = 0, expense_rate = 0,
                           buffer_months = 0, buffer_paid = FALSE, buffer_expenses = FALSE,
                           equity_release = "monthly", fee = NULL,
                           fee_basis = c("profit", "profit-over-premium",
                                         "profit-over-capital")) {
  if (!inherits(requirement, "cashflow_requirement") ||
      !inherits(commutation_requirement, "cashflow_requirement")) {
    stop("give the requirements as cashflow_requirement()s")
  }
  if (inherits(adjustments, "cashflow_adjustment")) adjustments <- list(adjustments)
  if (!all(vapply(adjustments, inherits, logical(1), "cashflow_adjustment"))) {
    stop("give the adjustments as a list of cashflow_adjustment()s")
  }
  adj <- function(field, type) vapply(adjustments, function(a) a[[field]], type)
  funding <- cashflow_dates(funding_dates, "funding_dates")
  release <- cashflow_dates(collateral_release, "collateral_release", commutation = TRUE)
  equity <- cashflow_dates(equity_release, "equity_release", commutation = TRUE)
  if (!is.null(fee) && (!is.list(fee) || length(fee) != 2)) {
    stop("fee is list(breakpoints, slopes)")
  }
  r <- requirement
  cr <- commutation_requirement
  ptr <- rust_result(Cashflow$new(
    as.double(premium), as.double(commutation_month), as.double(written), as.double(earned),
    as.double(paid), r$bases, r$loads, r$floor, r$limit, cr$bases, cr$loads, cr$floor, cr$limit,
    as.double(collected_rate), as.double(lag), adj("month", double(1)),
    adj("written_rate", double(1)), as.double(adj("less_commission", logical(1))),
    adj("base", character(1)), funding$kind, funding$values, match.arg(funding_rule),
    as.double(premium_offset), match.arg(reference), match.arg(funding_source), release$kind,
    release$values, as.double(annual_yield), match.arg(interest_basis), as.double(opening_fixed),
    as.double(opening_share), as.double(expense_rate), as.double(buffer_months),
    isTRUE(buffer_paid), isTRUE(buffer_expenses), equity$kind, equity$values,
    if (is.null(fee)) double() else as.double(fee[[1]]),
    if (is.null(fee)) double() else as.double(fee[[2]]), match.arg(fee_basis)
  ))
  structure(list(ptr = ptr), class = "cashflow_terms")
}

#' @export
print.cashflow_terms <- function(x, ...) {
  cat(sprintf("<cashflow_terms> premium %s, commutation after %d months\n",
              format(x$ptr$premium()), x$ptr$commutation_month()))
  invisible(x)
}

#' @rdname cashflow_terms
#' @param terms A `cashflow_terms`.
#' @param u Ultimate loss ratios, one a simulation (`cashflow_trace()`: one).
#' @param commission Commission rates, recycled to `u`.
#' @param months_requirement `NULL`, or each simulation's requirement month
#'   by month in place of `requirement`: a matrix with a row a simulation
#'   and `commutation_month + buffer_months` columns. The commutation
#'   requirement is still read from its terms.
#' @export
cashflow_run <- function(terms, u, commission = 0, months_requirement = NULL) {
  if (!inherits(terms, "cashflow_terms")) stop("terms must be a cashflow_terms()")
  u <- as.double(u)
  commission <- rep_len(as.double(commission), length(u))
  req <- double()
  if (!is.null(months_requirement)) {
    if (is.null(dim(months_requirement))) {
      months_requirement <- matrix(months_requirement, nrow = 1)
    }
    width <- terms$ptr$commutation_month() + terms$ptr$buffer_months()
    if (nrow(months_requirement) != length(u) || ncol(months_requirement) != width) {
      stop(sprintf("months_requirement needs %d rows and %d columns", length(u), width),
           call. = FALSE)
    }
    req <- as.double(t(months_requirement))
  }
  r <- rust_result(terms$ptr$run(u, commission, req))
  r$flows <- matrix(r$flows, nrow = length(u), ncol = r$months, byrow = TRUE)
  r$months <- seq_len(r$months) - 1
  r
}

#' @rdname cashflow_terms
#' @export
cashflow_trace <- function(terms, u, commission = 0) {
  if (!inherits(terms, "cashflow_terms")) stop("terms must be a cashflow_terms()")
  r <- rust_result(terms$ptr$trace(as.double(u), as.double(commission)))
  scalars <- c("commutation_paid", "fee", "conservation_error")
  accounts <- as.data.frame(r[setdiff(names(r), scalars)])
  list(accounts = cbind(month = seq_len(nrow(accounts)) - 1, accounts),
       commutation_paid = r$commutation_paid, fee = r$fee,
       conservation_error = r$conservation_error)
}
