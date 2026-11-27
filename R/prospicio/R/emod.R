#' Experience modification: the workers compensation split plan
#'
#' The modification of NCCI's Experience Rating Plan, with every sum of
#' its worksheet:
#' `mod = (A_p + W A_e + (1 - W) E_e + B) / (E + B)`, with expected losses
#' `E` (payroll in hundreds times the expected loss rate, summed over the
#' class rows) and their primary part `E_p` (times the D-ratio),
#' `E_e = E - E_p`, actual primary and excess losses `A_p` and `A_e`, the
#' weighting value `W` and the ballast `B`.
#'
#' Each claim is held to the per-claim limit, the claims of one accident
#' together to the per-accident limit (the cut taken from their excess
#' first), split at the split point, and a medical-only claim enters at
#' `medical_only_share` of both parts. The bureau's tables (expected loss
#' rates, D-ratios, weighting values, ballasts and split points) are
#' licensed, so they are inputs.
#'
#' @param exposures A data frame with one row per class in each year or
#'   policy of the experience period: columns `class`, `payroll`,
#'   `expected_loss_rate` (per 100 of payroll) and `d_ratio`.
#' @param claims A data frame with column `amount` (incurred) and,
#'   optionally, `medical_only` (logical) and `accident` (the accident each
#'   claim arose from, `NA` for an accident of its own).
#' @param weight The weighting value `W`, from 0 to 1.
#' @param ballast The ballast `B`.
#' @param split_point The part of each claim that is primary.
#' @param per_claim_limit,per_accident_limit The most of one claim, and of
#'   all the claims of one accident, that enters; `Inf` for none.
#' @param medical_only_share The share of a medical-only claim that enters.
#' @returns A list: `modification` (unrounded; the bureau shows two
#'   decimals), `expected`, `expected_primary`, `expected_excess`,
#'   `actual_primary`, `actual_excess`, `weight`, `ballast`, `stabilizing`
#'   (`(1 - W) E_e + B`), `adjusted_actual` and `adjusted_expected`, and the
#'   rows: `classes` (`exposures` with `expected` and `expected_primary`)
#'   and `claims` (`claims` with `limited`, `primary` and `excess`, what
#'   enters after the limits and the medical-only share).
#' @export
#' @examples
#' exposures <- data.frame(class = "8810", payroll = 3e6,
#'                         expected_loss_rate = 0.11, d_ratio = 0.35)
#' claims <- data.frame(amount = c(25000, 2000), medical_only = c(FALSE, TRUE))
#' x <- split_plan_modification(exposures, claims, weight = 0.14,
#'                              ballast = 5000, split_point = 18000)
#' x$modification
#' x$claims
split_plan_modification <- function(exposures, claims, weight, ballast, split_point,
                                    per_claim_limit = Inf, per_accident_limit = Inf,
                                    medical_only_share = 0.3) {
  for (col in c("class", "payroll", "expected_loss_rate", "d_ratio")) {
    if (is.null(exposures[[col]])) stop("exposures needs a column ", col)
  }
  if (is.null(claims$amount)) stop("claims needs a column amount")
  m <- length(claims$amount)
  medical_only <- if (is.null(claims$medical_only)) rep(FALSE, m) else as.logical(claims$medical_only)
  accident <- if (is.null(claims$accident)) rep("", m) else as.character(claims$accident)
  accident[is.na(accident)] <- ""
  r <- rust_result(split_plan_modification_rust(
    as.character(exposures$class), as.double(exposures$payroll),
    as.double(exposures$expected_loss_rate), as.double(exposures$d_ratio),
    as.double(claims$amount), as.double(medical_only), accident,
    as.double(weight), as.double(ballast), as.double(split_point),
    as.double(per_claim_limit), as.double(per_accident_limit), as.double(medical_only_share)
  ))
  classes <- as.data.frame(exposures)
  classes$expected <- r$class_expected
  classes$expected_primary <- r$class_expected_primary
  rows <- as.data.frame(claims)
  rows$limited <- r$claim_limited
  rows$primary <- r$claim_primary
  rows$excess <- r$claim_excess
  c(r[c("modification", "expected", "expected_primary", "expected_excess",
        "actual_primary", "actual_excess", "weight", "ballast", "stabilizing",
        "adjusted_actual", "adjusted_expected")],
    list(classes = classes, claims = rows))
}
