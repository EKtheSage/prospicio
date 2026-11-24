#' Insurance charge and savings (Table M, Table L)
#'
#' For aggregate losses `A` with mean `E` and an entry ratio `r` (losses as
#' a multiple of their expectation), the insurance charge is
#' `phi(r) = E[max(A / E - r, 0)]` and the insurance savings
#' `psi(r) = E[max(r - A / E, 0)]`, with `phi(r) - psi(r) = 1 - r`. These
#' are the two columns of a Table M.
#'
#' With a per-occurrence limit, pass the limited aggregate. Left alone it
#' gives the limited Table M, with entry ratios on the limited expected
#' loss. With `unlimited_expected` it gives a Table L: entry ratios on the
#' unlimited expected loss, and the share of losses the limit removes,
#' `k = (E - E*) / E`, added to the charge, so that `phi - psi = 1 - r`
#' still.
#'
#' @param aggregate The aggregate losses: a [grid_distribution] (from
#'   [compound_distribution()] or [compound_auto()]), or a numeric vector
#'   of simulated or observed totals, each equally likely.
#' @param entry_ratios Entry ratios at which to evaluate.
#' @param unlimited_expected For a Table L, the expected aggregate loss
#'   without the per-occurrence limit; `NULL` otherwise.
#' @returns A data frame with columns `entry_ratio`, `charge` and
#'   `savings`, and attributes `expected` (what the entry ratios multiply)
#'   and `loss_elimination` (`k`; 0 unless a Table L).
#' @export
#' @examples
#' # Four equally likely years.
#' insurance_charge(c(0, 50, 100, 250), c(0, 0.5, 1, 2))
#'
#' # A compound Poisson on a grid.
#' sev <- discretize(gamma_from_mean_cv(10, 1), step = 0.25, points = 2^10)
#' agg <- compound_distribution(poisson_count(4), sev, 2^12, method = "fft")
#' insurance_charge(agg, c(0.5, 1, 1.5))
#'
#' # Table L: the same years with each occurrence limited, unlimited mean 132.5.
#' tl <- insurance_charge(c(110, 100, 0, 200), c(0, 1), unlimited_expected = 132.5)
#' attr(tl, "loss_elimination")
insurance_charge <- function(aggregate, entry_ratios, unlimited_expected = NULL) {
  r <- rust_result(insurance_charge_rust(retro_aggregate(aggregate), as.double(entry_ratios),
                                         null_nan(unlimited_expected)))
  out <- data.frame(entry_ratio = r$entry_ratio, charge = r$charge, savings = r$savings)
  attr(out, "expected") <- r$expected
  attr(out, "loss_elimination") <- r$loss_elimination
  out
}

# A grid's pointer, or the totals as doubles.
retro_aggregate <- function(aggregate) {
  if (is.numeric(aggregate)) as.double(aggregate) else aggregate@ptr
}

#' Retrospective rating plan
#'
#' A retrospective premium is `T * (b + c * L)` for a year's losses `L`,
#' no less than a minimum `H` and no more than a maximum `G`, with `T` the
#' tax multiplier, `c` the loss conversion factor and `b` the basic
#' premium. `retro_balance()` finds the basic premium at which the plan
#' collects the guaranteed-cost premium `T * (e + E)` on average, from the
#' insurance charge at the entry ratios where the minimum and the maximum
#' bind. `retro_premium()` is the premium for given losses.
#'
#' A plan with a loss limit takes the losses with each occurrence limited:
#' pass the limited aggregate and `unlimited_expected`, so that the plan
#' balances against a Table L. Its basic premium then carries the cost of
#' the limit.
#'
#' @param aggregate The plan's aggregate losses, as in
#'   [insurance_charge()].
#' @param expected_loss The expected losses `E`, without any loss limit.
#' @param expense Expenses and profit `e`, excluding the premium tax.
#' @param loss_conversion The loss conversion factor `c`.
#' @param tax_multiplier The tax multiplier `T`.
#' @param minimum,maximum The minimum and maximum premium; `Inf` for no
#'   maximum.
#' @param unlimited_expected With a loss limit, the expected aggregate loss
#'   without it (normally `expected_loss`); `NULL` otherwise.
#' @param basic The basic premium `b`.
#' @param losses A year's losses, or a vector of them.
#' @returns `retro_balance()`: a list with `basic`, `entry_ratio_minimum`
#'   and `entry_ratio_maximum` (where the minimum and the maximum bind; the
#'   first is negative when the minimum never binds),
#'   `net_insurance_charge` and `expected_premium`. `retro_premium()`: the
#'   premiums.
#' @export
#' @examples
#' losses <- c(0, 10, 30, 50, 70, 90, 120, 150, 200, 280)
#' b <- retro_balance(losses, expected_loss = 100, expense = 30, loss_conversion = 1.1,
#'                    tax_multiplier = 1.03, minimum = 80, maximum = 220)
#' b$basic
#' # The plan collects the guaranteed-cost premium on average.
#' mean(retro_premium(b$basic, losses, 1.1, 1.03, 80, 220))
#' b$expected_premium
retro_balance <- function(aggregate, expected_loss, expense, loss_conversion = 1,
                          tax_multiplier = 1, minimum = 0, maximum = Inf,
                          unlimited_expected = NULL) {
  rust_result(retro_balance_rust(retro_aggregate(aggregate), null_nan(unlimited_expected),
                                 as.double(expected_loss), as.double(expense),
                                 as.double(loss_conversion), as.double(tax_multiplier),
                                 as.double(minimum), as.double(maximum)))
}

#' @rdname retro_balance
#' @export
retro_premium <- function(basic, losses, loss_conversion = 1, tax_multiplier = 1,
                          minimum = 0, maximum = Inf) {
  rust_result(retro_premium_rust(as.double(basic), as.double(losses),
                                 as.double(loss_conversion), as.double(tax_multiplier),
                                 as.double(minimum), as.double(maximum)))
}
