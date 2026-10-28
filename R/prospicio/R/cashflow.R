# A collateralised deal's monthly cash flows (Aggregate lane), over
# prospicio_pricing::cashflow (docs/design/time-value.md, decisions 13 to 20),
# and the deal valued after binding (decision 34).

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
#' The reserve on earned premium is small early in the term: a requirement
#' on it alone releases the premium collected ahead of earning to the
#' investor at once, who then pays the claims later, so the flows start
#' positive ([cashflow_run()]'s `paid_first`), can have several internal
#' rates of return ([irr()]'s `sign_changes`) and the one found is usually
#' meaningless. A collateralised fund usually wants `"outstanding"`, with
#' `"unearned-premium"` beside it.
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
#' @param deduct_commission Take each simulation's commission rate
#'   (`cashflow_run()`'s `commission`) off `collected_rate`. Without it the
#'   commission does not reduce the premium paid in and is read only by
#'   adjustments with `less_commission` and by the fee: a flat commission
#'   `c` is then `collected_rate = 1 - c`, and a sliding scale needs this
#'   option.
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
#'   `opening`, `released`, `distributed`, `fee`, `commutation` and
#'   `paid_first` (whether the first nonzero investor flow is positive: the
#'   investor is paid before putting money in, so its flows can have
#'   several internal rates of return and the one [irr()] finds is usually
#'   meaningless; a requirement on the reserve alone does this);
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
                           commutation_requirement = requirement, collected_rate = 1,
                           deduct_commission = FALSE, lag = 0,
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
    as.double(collected_rate), isTRUE(deduct_commission), as.double(lag), adj("month", double(1)),
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
#' @param paid_patterns `NULL` for the terms' paid pattern in every
#'   simulation, or paid patterns in its place (#600): a matrix with a row a
#'   pattern and one column per month of the terms' paid pattern, each row
#'   non-decreasing and ending at the same share as the terms'. Without
#'   `pattern_index` there is one row a simulation, in order. Not with
#'   `months_requirement`.
#' @param pattern_index `NULL`, or each simulation's row of `paid_patterns`
#'   (from 1).
#' @export
cashflow_run <- function(terms, u, commission = 0, months_requirement = NULL,
                         paid_patterns = NULL, pattern_index = NULL) {
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
  paid <- double()
  n_patterns <- 0
  if (!is.null(paid_patterns)) {
    if (is.null(dim(paid_patterns))) paid_patterns <- matrix(paid_patterns, nrow = 1)
    paid <- as.double(t(paid_patterns))
    n_patterns <- nrow(paid_patterns)
  } else if (!is.null(pattern_index)) {
    stop("pattern_index needs paid_patterns", call. = FALSE)
  }
  index <- if (is.null(pattern_index)) double() else as.double(pattern_index) - 1
  r <- rust_result(terms$ptr$run(u, commission, req, paid, n_patterns, index))
  r$flows <- matrix(r$flows, nrow = length(u), ncol = r$months, byrow = TRUE)
  r$months <- seq_len(r$months) - 1
  r
}

#' @rdname cashflow_terms
#' @param paid_pattern `NULL` for the terms' paid pattern, or one paid
#'   pattern in its place, as a row of `paid_patterns`.
#' @export
cashflow_trace <- function(terms, u, commission = 0, paid_pattern = NULL) {
  if (!inherits(terms, "cashflow_terms")) stop("terms must be a cashflow_terms()")
  paid <- if (is.null(paid_pattern)) double() else as.double(paid_pattern)
  r <- rust_result(terms$ptr$trace(as.double(u), as.double(commission), paid))
  scalars <- c("opening_balance", "commutation_paid", "fee", "conservation_error")
  accounts <- as.data.frame(r[setdiff(names(r), scalars)])
  list(accounts = cbind(month = seq_len(nrow(accounts)) - 1, accounts),
       commutation_paid = r$commutation_paid, fee = r$fee,
       conservation_error = r$conservation_error)
}

#' A bound deal valued from its actual accounts
#'
#' After binding, `cashflow_valuation()` takes a deal's actual accounts at
#' the end of month `month - 1` and `cashflow_forward()` simulates the
#' months from `month` to commutation from them (#603): the trust and the
#' operating account open at the actual balances, the premium already
#' collected is not collected again, and each simulation's ultimate loss
#' ratio is conditioned on the actual paid `A` ([condition_on_paid()]), its
#' loss still to pay following the rest of the paid pattern. Valuation is
#' at a month end, the engine's step.
#'
#' The forward return takes the net asset value (`nav`, the trust's and
#' operating account's balances) as the investment at the valuation month:
#' `forward_rows` are the forward flows with `nav` paid in at their first
#' month. The return since inception puts the investor's actual flows to
#' date (`flows`) before each simulation's forward flows:
#' `since_inception_rows`. Read either with [npv()] and [irr()] on
#' `forward_offsets` or `since_inception_offsets` with `unit = "months"`.
#'
#' A valuation after a premium adjustment month is refused. Under
#' `funding_rule = "obligation-increase"` a valuation after month 0 needs
#' `funding_reference`, the obligation the next call is measured from (the
#' trace's `funding_reference` column). At month 0 the state must be empty,
#' and the forward flows are [cashflow_run()]'s.
#'
#' `cashflow_actual_state()` gives the state of one simulated path, to
#' value it as if it had happened.
#'
#' @param terms A [cashflow_terms()].
#' @param month The valuation month `v`: the state is at the end of month
#'   `v - 1` and the forward run starts at month `v`.
#' @param premium_collected The premium paid into the trust to date. What
#'   the deal's schedule would have collected by then and the trust has not
#'   received (or received beyond it) is collected (or paid back) in month
#'   `v`.
#' @param paid The losses paid from the trust to date.
#' @param collateral,operating The trust's and the operating account's
#'   balances.
#' @param funded The capital the investor has paid into the trust, without
#'   the opening capital.
#' @param released The equity released to the investor.
#' @param flows `NULL`, or the investor's actual flows in months `0` to
#'   `v - 1`, for the return since inception.
#' @param funding_reference `NULL`, or the obligation reference under the
#'   obligation-increase funding rule.
#' @returns `cashflow_valuation()` a `cashflow_valuation` with `share_paid`
#'   (the paid pattern reached before month `v`) and `nav`;
#'   `cashflow_actual_state()` a list of the arguments above, for
#'   `do.call(cashflow_valuation, c(list(terms, month), state))`;
#'   `cashflow_forward()` a list of `kept` (the simulations kept, from 1),
#'   `ultimate` (their conditioned loss ratios), `nav`, `flows` (a matrix,
#'   a row a kept simulation, months `months`), `forward_rows` and
#'   `forward_offsets`, `since_inception_rows` and
#'   `since_inception_offsets` (`NULL` without `flows`), `forward_moic`
#'   (`1 + net / (nav + funded after the valuation)`),
#'   `since_inception_moic`, and the totals since inception `funded`,
#'   `released`, `distributed`, `fee` and `commutation`;
#'   `cashflow_valuation_trace()` the accounts as [cashflow_trace()], with
#'   `opening_balance` the actual balances.
#' @seealso [cashflow_terms()], [condition_on_paid()], [deal_ave()].
#' @export
#' @examples
#' ramp <- function(m) pmin((1:48) / m, 1)
#' terms <- cashflow_terms(100, 36, ramp(12), ramp(24), ramp(36),
#'                         cashflow_requirement(c(outstanding = 1)))
#' # A path at a loss ratio of 0.8, valued after 12 months.
#' state <- cashflow_actual_state(terms, 0.8, 12)
#' val <- do.call(cashflow_valuation, c(list(terms, 12), state))
#' f <- cashflow_forward(val, c(0.5, 0.8, 1.2))
#' all.equal(f$since_inception_rows[2, ], cashflow_run(terms, 0.8)$flows[1, ])
#' npv(f$forward_rows, f$forward_offsets, 0.05, unit = "months")
cashflow_valuation <- function(terms, month, premium_collected = 0, paid = 0, collateral = 0,
                               funded = 0, released = 0, operating = 0, flows = NULL,
                               funding_reference = NULL) {
  if (!inherits(terms, "cashflow_terms")) stop("terms must be a cashflow_terms()")
  state <- list(month = as.double(month),
                amounts = as.double(c(premium_collected, paid, collateral, funded, released,
                                      operating)),
                flows = if (is.null(flows)) double() else as.double(flows),
                has_flows = !is.null(flows),
                reference = if (is.null(funding_reference)) double() else
                  as.double(funding_reference))
  r <- rust_result(terms$ptr$valuation(state$month, state$amounts, state$flows,
                                       state$has_flows, state$reference))
  structure(list(terms = terms, state = state, share_paid = r$share_paid, nav = r$nav),
            class = "cashflow_valuation")
}

#' @export
print.cashflow_valuation <- function(x, ...) {
  cat(sprintf("<cashflow_valuation> month %d, share paid %s, nav %s\n",
              as.integer(x$state$month), format(x$share_paid), format(x$nav)))
  invisible(x)
}

#' @rdname cashflow_valuation
#' @param u Ultimate loss ratios, one a simulation (`cashflow_actual_state()`,
#'   `cashflow_valuation_trace()`: one).
#' @param commission Commission rates, recycled to `u`.
#' @param paid_pattern `NULL` for the terms' paid pattern, or the path's own,
#'   as [cashflow_trace()]'s.
#' @export
cashflow_actual_state <- function(terms, u, month, commission = 0, paid_pattern = NULL) {
  if (!inherits(terms, "cashflow_terms")) stop("terms must be a cashflow_terms()")
  paid <- if (is.null(paid_pattern)) double() else as.double(paid_pattern)
  r <- rust_result(terms$ptr$actual_state(as.double(u), as.double(month),
                                          as.double(commission), paid))
  if (month == 0) r$funding_reference <- NULL
  r
}

#' @rdname cashflow_valuation
#' @param valuation A `cashflow_valuation`.
#' @param conditioning How each simulation's ultimate is conditioned on the
#'   paid to date: `"bornhuetter-ferguson"` or `"truncation"`, as
#'   [condition_on_paid()].
#' @export
cashflow_forward <- function(valuation, u, commission = 0,
                             conditioning = c("bornhuetter-ferguson", "truncation")) {
  if (!inherits(valuation, "cashflow_valuation")) {
    stop("valuation must be a cashflow_valuation()")
  }
  s <- valuation$state
  u <- as.double(u)
  commission <- rep_len(as.double(commission), length(u))
  r <- rust_result(valuation$terms$ptr$forward(s$month, s$amounts, s$flows, s$has_flows,
                                               s$reference, u, commission,
                                               match.arg(conditioning)))
  n <- length(r$kept)
  m <- r$months
  v <- as.integer(s$month)
  r$kept <- r$kept + 1
  r$flows <- matrix(r$flows, nrow = n, ncol = m, byrow = TRUE)
  r$months <- v + seq_len(m) - 1
  r$forward_rows <- matrix(r$forward_rows, nrow = n, ncol = m, byrow = TRUE)
  r$forward_offsets <- seq_len(m) - 1
  if (!is.null(r$since_inception_rows)) {
    r$since_inception_rows <- matrix(r$since_inception_rows, nrow = n, ncol = v + m,
                                     byrow = TRUE)
    r$since_inception_offsets <- seq_len(v + m) - 1
  }
  r
}

#' @rdname cashflow_valuation
#' @export
cashflow_valuation_trace <- function(valuation, u, commission = 0,
                                     conditioning = c("bornhuetter-ferguson", "truncation")) {
  if (!inherits(valuation, "cashflow_valuation")) {
    stop("valuation must be a cashflow_valuation()")
  }
  s <- valuation$state
  r <- rust_result(valuation$terms$ptr$valuation_trace(
    s$month, s$amounts, s$flows, s$has_flows, s$reference, as.double(u),
    as.double(commission), match.arg(conditioning)))
  scalars <- c("opening_balance", "commutation_paid", "fee", "conservation_error")
  accounts <- as.data.frame(r[setdiff(names(r), scalars)])
  list(accounts = cbind(month = as.integer(s$month) + seq_len(nrow(accounts)) - 1, accounts),
       opening_balance = r$opening_balance, commutation_paid = r$commutation_paid, fee = r$fee,
       conservation_error = r$conservation_error)
}

#' Simulated loss ratios conditioned on the loss paid to date
#'
#' Each simulation's ultimate loss ratio `u` conditioned on the loss `paid`
#' to date when the paid pattern has reached `share_paid` (#603).
#' `"bornhuetter-ferguson"` keeps every simulation, its ultimate the actual
#' paid plus its own unpaid, `paid / premium + u (1 - share_paid)`: future
#' payments do not depend on past ones, and the forward distribution
#' narrows as the deal runs off. `"truncation"` keeps the simulations with
#' `u * premium >= paid`, unchanged; late in a deal it can drop most of
#' them.
#'
#' @param u Ultimate loss ratios, one a simulation.
#' @param premium The premium.
#' @param paid The loss paid to date.
#' @param share_paid The share of the paid pattern reached.
#' @param conditioning `"bornhuetter-ferguson"` or `"truncation"`.
#' @returns A list of `kept` (the simulations kept, from 1) and `ultimate`
#'   (their loss ratios given the paid).
#' @seealso [cashflow_valuation()].
#' @export
#' @examples
#' condition_on_paid(c(0.5, 0.8), 100, 30, 0.4)
#' condition_on_paid(c(0.2, 0.8), 100, 30, 0.4, "truncation")
condition_on_paid <- function(u, premium, paid, share_paid,
                              conditioning = c("bornhuetter-ferguson", "truncation")) {
  r <- rust_result(cashflow_condition_on_paid(as.double(u), as.double(premium), as.double(paid),
                                              as.double(share_paid), match.arg(conditioning)))
  r$kept <- r$kept + 1
  r
}

#' Actual against expected paid loss of a bound deal
#'
#' The actual cumulative paid `actual[t]` at the end of each month
#' `t = 0, 1, ...` against what the priced simulations expected (#603):
#' each simulation's `u P S(t)`, with `S` the terms' paid pattern or the
#' simulation's own from `paid_patterns`. The expected paid comes from the
#' priced simulations, not conditioned ones. A reported loss works the same
#' way with the reported pattern as `paid_patterns`.
#'
#' @param actual The actual cumulative paid by the end of each month from
#'   month 0.
#' @param terms A [cashflow_terms()].
#' @param u The priced ultimate loss ratios, one a simulation.
#' @param paid_patterns,pattern_index `NULL`, or paid patterns as
#'   [cashflow_run()]'s.
#' @param levels The percentile levels of the `q` columns.
#' @returns A data frame, a row a month: `month`, `actual`, `expected` (the
#'   mean over the simulations), `ratio` (`actual / expected`),
#'   `percentile` (the actual's percentile among the simulations' expected
#'   paid, ties counted half) and a column `q<100 level>` for each level
#'   (numpy's linear quantile).
#' @seealso [cashflow_valuation()].
#' @export
#' @examples
#' terms <- cashflow_terms(100, 4, rep(1, 4), rep(1, 4), c(0.25, 0.5, 0.75, 1),
#'                         cashflow_requirement(c(outstanding = 1)))
#' deal_ave(c(15, 45), terms, c(0.4, 0.6, 0.8))
deal_ave <- function(actual, terms, u, paid_patterns = NULL, pattern_index = NULL,
                     levels = c(0.05, 0.5, 0.95)) {
  if (!inherits(terms, "cashflow_terms")) stop("terms must be a cashflow_terms()")
  paid <- double()
  n_patterns <- 0
  if (!is.null(paid_patterns)) {
    if (is.null(dim(paid_patterns))) paid_patterns <- matrix(paid_patterns, nrow = 1)
    paid <- as.double(t(paid_patterns))
    n_patterns <- nrow(paid_patterns)
  } else if (!is.null(pattern_index)) {
    stop("pattern_index needs paid_patterns", call. = FALSE)
  }
  index <- if (is.null(pattern_index)) double() else as.double(pattern_index) - 1
  levels <- as.double(levels)
  r <- rust_result(terms$ptr$deal_ave(as.double(actual), as.double(u), paid, n_patterns, index,
                                      levels))
  out <- data.frame(month = as.integer(r$month), actual = r$actual, expected = r$expected,
                    ratio = r$ratio, percentile = r$percentile)
  q <- matrix(r$quantiles, nrow = nrow(out), ncol = length(levels), byrow = TRUE)
  for (k in seq_along(levels)) out[[sprintf("q%g", 100 * levels[k])]] <- q[, k]
  out
}
