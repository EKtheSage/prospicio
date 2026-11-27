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

#' Experience modification: the general liability experience ratio plan
#'
#' The modification of ISO's Commercial General Liability Experience
#' Rating Plan, with every sum of its worksheet:
#' `AER = (reported + expected unreported) / subject loss cost` and
#' `mod = 1 + Z (AER - EER) / EER`.
#'
#' Each occurrence enters at its basic-limits loss plus ALAE, held together
#' to the maximum single limit; a year's expected unreported losses are its
#' subject loss cost times its detrend factor, `EER` and its unreported
#' share. The bureau's tables (subject loss costs, `EER`, the MSL,
#' credibility and unreported shares) are licensed, so they are inputs.
#'
#' @param years A data frame with one row per policy year: columns
#'   `subject_loss_cost` (the company subject basic-limits loss cost),
#'   `unreported_share` (the share of its losses expected still unreported)
#'   and, optionally, `detrend_factor` (1 by default).
#' @param occurrences A data frame with one row per occurrence: columns
#'   `year` (the row of `years` it falls in, from 1), `loss` (basic limits)
#'   and `alae`.
#' @param maximum_single_limit The MSL; `Inf` for none.
#' @param expected_ratio `EER`.
#' @param credibility `Z`, from 0 to 1.
#' @returns A list: `modification` (unrounded), `credit_debit`,
#'   `actual_ratio`, `expected_ratio`, `credibility`, `subject_loss_cost`,
#'   `reported` and `expected_unreported`, and `years` (`years` with
#'   `reported` and `expected_unreported`).
#' @export
#' @examples
#' years <- data.frame(subject_loss_cost = 1e5, unreported_share = 0.25)
#' occurrences <- data.frame(year = 1, loss = 70000, alae = 5000)
#' x <- experience_ratio_modification(years, occurrences, maximum_single_limit = 50000,
#'                                    expected_ratio = 0.8, credibility = 0.3)
#' x$modification
experience_ratio_modification <- function(years, occurrences, maximum_single_limit,
                                          expected_ratio, credibility) {
  for (col in c("subject_loss_cost", "unreported_share")) {
    if (is.null(years[[col]])) stop("years needs a column ", col)
  }
  for (col in c("year", "loss", "alae")) {
    if (is.null(occurrences[[col]])) stop("occurrences needs a column ", col)
  }
  n <- length(years$subject_loss_cost)
  detrend <- if (is.null(years$detrend_factor)) rep(1, n) else as.double(years$detrend_factor)
  r <- rust_result(experience_ratio_modification_rust(
    as.double(years$subject_loss_cost), as.double(years$unreported_share), detrend,
    as.double(occurrences$year), as.double(occurrences$loss), as.double(occurrences$alae),
    as.double(maximum_single_limit), as.double(expected_ratio), as.double(credibility)
  ))
  rows <- as.data.frame(years)
  rows$reported <- r$year_reported
  rows$expected_unreported <- r$year_expected_unreported
  c(r[c("modification", "credit_debit", "actual_ratio", "expected_ratio", "credibility",
        "subject_loss_cost", "reported", "expected_unreported")],
    list(years = rows))
}
