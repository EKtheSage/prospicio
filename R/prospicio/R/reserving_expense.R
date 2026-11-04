# Recoveries, ALAE and ULAE (issue #558, docs/design/reserving-v02.md, decision
# 35): the ratio of two columns developed to ultimate, and unpaid ULAE. The
# work is in crates/prospicio-r/src/reserving_expense.rs.

ratio_development_fit <- S7::new_class(
  "ratio_development_fit",
  package = "prospicio",
  properties = list(
    origins = S7::class_character,
    ages = S7::class_double,
    ratios = S7::class_any,
    averages = S7::class_double,
    factors = S7::class_double,
    cdf = S7::class_double,
    latest_ratio = S7::class_double,
    projected_ratio = S7::class_double,
    selected_ratio = S7::class_double,
    ultimate_claims = S7::class_double,
    latest = S7::class_double,
    ultimate = S7::class_double,
    unpaid = S7::class_double,
    total_ultimate = S7::new_property(S7::class_double, getter = function(self) sum(self@ultimate)),
    total_unpaid = S7::new_property(S7::class_double, getter = function(self) sum(self@unpaid))
  )
)

#' Development of a ratio of two columns
#'
#' The ratio of two columns of one triangle, developed to ultimate and applied
#' to ultimate claims: Friedland's ratio approach to salvage and subrogation
#' (*Estimating Unpaid Claims Using Basic Techniques*, chapter 14) and to ALAE
#' (chapter 16). Each cell's ratio comes from the two cumulative columns; a
#' cell whose denominator is zero or missing has none. The latest ratio is
#' developed to ultimate, `selected` replaces it where the actuary judges
#' otherwise, and the numerator's ultimate is the ratio times the ultimate
#' claims from any method.
#'
#' @param triangle A single-segment triangle holding both columns.
#' @param numerator The cumulative column to project: received recoveries,
#'   paid ALAE.
#' @param denominator The cumulative column it is a ratio to: paid claims.
#' @param ultimate_claims The ultimate claims, one value per origin, oldest
#'   first, from any method.
#' @param average,sigma_interpolation,tail,ldf,ldf_kind,ldf_weight,n_periods,drop_high,drop_low
#'   As in [chain_ladder()], but the average is `"simple"` by default, as
#'   Friedland averages ratios. Additive factors need the simple average and
#'   no `tail` or `ldf`.
#' @param selected `NULL`, or ultimate ratios named by origin, which replace
#'   the projected ones: `c("2008" = 0.345)`.
#' @param additive Average the differences between ages and add them to
#'   ultimate (Friedland's chapter 16, Exhibit I, Sheets 9 and 10) instead of
#'   multiplying age-to-age factors.
#' @param additive_selected `NULL`, or selected differences named by the age
#'   they start from, for `additive`.
#' @param additive_tail The addition past the oldest age, for `additive`.
#' @returns A `ratio_development_fit` with `origins`, `ages`, the `ratios`
#'   matrix, the `averages` of the factors or differences, the `factors` in
#'   use, `cdf`, `latest_ratio`, `projected_ratio`, `selected_ratio`,
#'   `ultimate_claims`, the `latest` numerator, its `ultimate`, `unpaid` (for
#'   recoveries, the recoverable), `total_ultimate` and `total_unpaid`.
#' @examples
#' rows <- data.frame(origin = c(2020, 2020, 2021), age = c(12, 24, 12),
#'                    alae = c(10, 30, 20), paid = c(100, 150, 200))
#' tri <- triangle(rows, "origin", "age", c("alae", "paid"))
#' fit <- ratio_development(tri, "alae", "paid", c(160, 320))
#' fit@ultimate
#' @export
ratio_development <- function(triangle, numerator, denominator, ultimate_claims, average = "simple",
                              sigma_interpolation = "log-linear", tail = 1, ldf = NULL,
                              ldf_kind = NULL, ldf_weight = 1, n_periods = NULL, drop_high = 0,
                              drop_low = 0, selected = NULL, additive = FALSE,
                              additive_selected = NULL, additive_tail = 0) {
  check_triangle(triangle)
  args <- development_args(average, sigma_interpolation, n_periods, drop_high, drop_low)
  if (!is.null(selected) && (!is.numeric(selected) || is.null(names(selected)))) {
    stop('selected must be numbers named by origin, such as c("2008" = 0.345)', call. = FALSE)
  }
  if (!isTRUE(additive) && !isFALSE(additive)) {
    stop("additive must be TRUE or FALSE", call. = FALSE)
  }
  if (!additive && (!is.null(additive_selected) || !identical(as.double(additive_tail), 0))) {
    stop("additive_selected and additive_tail need additive = TRUE", call. = FALSE)
  }
  if (!is.null(additive_selected) &&
      (!is.numeric(additive_selected) || is.null(names(additive_selected)))) {
    stop("additive_selected must be numbers named by age", call. = FALSE)
  }
  out <- reserving_result(ratio_development_rust(
    triangle@ptr, numerator, denominator, as.double(ultimate_claims), args$average, args$sigma,
    args$excl, tail_ptr(tail), ldf_pattern(ldf, ldf_kind, ldf_weight),
    if (is.null(selected)) numeric() else selected * 1.0, additive,
    if (is.null(additive_selected)) numeric() else as.double(names(additive_selected)),
    if (is.null(additive_selected)) numeric() else unname(as.double(additive_selected)),
    single_number(additive_tail, "additive_tail")))
  o <- out$origins
  a <- out$ages
  ratios <- matrix(out$ratios, nrow = length(o), byrow = TRUE, dimnames = list(o, a))
  ratios[is.nan(ratios)] <- NA
  named <- function(x) stats::setNames(x, o)
  ratio_development_fit(
    origins = o, ages = a, ratios = ratios, averages = out$averages, factors = out$factors,
    cdf = stats::setNames(out$cdf, a), latest_ratio = named(out$latest_ratio),
    projected_ratio = named(out$projected_ratio), selected_ratio = named(out$selected_ratio),
    ultimate_claims = named(out$ultimate_claims), latest = named(out$latest),
    ultimate = named(out$ultimate), unpaid = named(out$unpaid))
}

ulae_fit <- S7::new_class(
  "ulae_fit",
  package = "prospicio",
  properties = list(
    basis = S7::class_double,
    ratios = S7::class_double,
    total_ratio = S7::class_double,
    ratio = S7::class_double,
    unpaid = S7::class_double,
    total = S7::class_double,
    expected_claims = S7::class_double,
    bornhuetter_ferguson = S7::class_double,
    development = S7::class_double
  )
)

#' Unpaid unallocated loss adjustment expense
#'
#' Unpaid ULAE from calendar-year paid ULAE and a claims basis (Friedland,
#' *Estimating Unpaid Claims Using Basic Techniques*, chapter 17). The
#' classical technique divides paid ULAE by `paid`; Kittel's refinement by the
#' average of `paid` and `reported` (calendar incurred claims); Conger and
#' Nolibos' generalized approach by the basis `B = U1 R + U2 P + U3 C`. The
#' classical technique and Kittel apply the ratio to half of case outstanding
#' and all of the IBNR, or with `pure_ibnr` to half of the known claims and all
#' of the pure IBNR. Conger and Nolibos apply it as
#' `W* (U1 (L - R) + U2 (L - P) + U3 (L - C))`. Mango and Allen's refinement is
#' the classical technique with expected paid claims as `paid`.
#'
#' @param paid_ulae Paid ULAE by calendar year, `M`.
#' @param paid Paid claims by calendar year, `P`. Claims include ALAE.
#' @param case_outstanding,ibnr Case outstanding and total IBNR at the
#'   valuation date, one value for the book or one per accident year.
#' @param method `"classical"`, `"kittel"` or `"conger_nolibos"`.
#' @param reported By calendar year: for `"kittel"` incurred claims (paid
#'   claims plus the change in case outstanding and IBNR), for
#'   `"conger_nolibos"` the ultimate cost of claims reported, `R`, or in their
#'   simplification the accident year's ultimate claims.
#' @param ratio The selected ULAE ratio; `NULL` uses the ratio of the totals.
#' @param opening For `"classical"`, the share of a claim's ULAE sustained when
#'   it opens. An insurer spending 25% opening and 75% closing applies 75% of
#'   the ratio to case outstanding and 100% to IBNR (Friedland's erratum to
#'   page 388).
#' @param pure_ibnr `NULL`, or the part of `ibnr` on claims not yet reported.
#'   `"conger_nolibos"` needs it unless `U1` is 0: it is `L - R(t)`.
#' @param weights For `"conger_nolibos"`, `c(U1, U2)` or `c(U1, U2, U3)`, the
#'   shares spent opening, maintaining and closing claims, summing to 1.
#' @param closed,unclosed For `"conger_nolibos"` unless `U3` is 0: the ultimate
#'   cost of claims closed by calendar year, `C`, and of claims not yet closed
#'   by row of `case_outstanding`, `L - C(t)`.
#' @param ultimate For `"conger_nolibos"`, `NULL` or ultimate claims by
#'   accident year, `L`, over the same claims as the calendar history, for the
#'   three totals in the result.
#' @returns A `ulae_fit` with `basis` and `ratios` by calendar year,
#'   `total_ratio`, the `ratio` applied, `unpaid` by row of `case_outstanding`
#'   and its `total`. With `ultimate`, Conger and Nolibos' `expected_claims`
#'   (`W* L - M`), `bornhuetter_ferguson` (`W* (L - B)`) and `development`
#'   (`M (L / B - 1)`); `NA` otherwise. Under their simplification the
#'   `bornhuetter_ferguson` form loses the pure IBNR and `unpaid` is the
#'   estimate.
#' @examples
#' fit <- ulae_estimate(c(14.352, 15.321), c(333, 358), 603, 316, ratio = 0.045)
#' fit@total
#' @export
ulae_estimate <- function(paid_ulae, paid, case_outstanding, ibnr,
                          method = c("classical", "kittel", "conger_nolibos"), reported = NULL,
                          ratio = NULL, opening = 0.5, pure_ibnr = NULL, weights = NULL,
                          closed = NULL, unclosed = NULL, ultimate = NULL) {
  method <- match.arg(method)
  if (method != "classical" && !identical(as.double(opening), 0.5)) {
    stop('opening is a setting of method = "classical"', call. = FALSE)
  }
  conger_nolibos <- method == "conger_nolibos"
  if (conger_nolibos) {
    if (!is.numeric(weights) || !length(weights) %in% 2:3) {
      stop('method = "conger_nolibos" needs weights: c(U1, U2) or c(U1, U2, U3)', call. = FALSE)
    }
    weights <- as.double(c(weights, 0)[1:3])
  } else if (!is.null(weights) || !is.null(closed) || !is.null(unclosed) || !is.null(ultimate)) {
    stop('weights, closed, unclosed and ultimate are settings of method = "conger_nolibos"',
         call. = FALSE)
  }
  numbers <- function(x) if (is.null(x)) numeric() else as.double(x)
  out <- reserving_result(ulae_rust(
    method, as.double(paid_ulae), as.double(paid), numbers(reported), numbers(closed),
    as.double(case_outstanding), as.double(ibnr),
    if (is.null(pure_ibnr)) NULL else as.double(pure_ibnr), numbers(unclosed), numbers(ultimate),
    if (is.null(ratio)) NaN else single_number(ratio, "ratio"),
    single_number(opening, "opening"), numbers(weights)))
  na <- function(x) if (is.nan(x)) NA_real_ else x
  ulae_fit(basis = out$basis, ratios = out$ratios, total_ratio = out$total_ratio, ratio = out$ratio,
           unpaid = out$unpaid, total = out$total, expected_claims = na(out$expected_claims),
           bornhuetter_ferguson = na(out$bornhuetter_ferguson),
           development = na(out$development))
}
