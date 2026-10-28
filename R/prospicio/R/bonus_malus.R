# Bonus-malus scales as Markov chains (Aggregate lane), over
# `prospicio_pricing::bonus_malus` (`docs/design/credibility.md`,
# "Bonus-malus scales"; issue #563).

#' Bonus-malus and no-claim-discount scales
#'
#' A policyholder sits on one of the levels `0, ..., levels - 1` of a scale,
#' 0 the best, and moves each year by the number of claims reported. Under
#' Poisson claims with mean `lambda` the levels are a Markov chain; over a
#' mixed Poisson portfolio the scale has a stationary distribution of levels
#' and Norberg's optimal relativities ([bonus_malus_portfolio()]).
#'
#' `bonus_malus()` builds the scale from a rule (down `down` levels after a
#' claim-free year, up `up_per_claim` levels for each claim, capped at both
#' ends; the -1/+top scale is `down = 1, up_per_claim = levels - 1`) or from
#' a `table`: an integer matrix with one row per level and one column per
#' claim count from 0, entry `[l + 1, k + 1]` the level after `k` claims from
#' level `l`, the last column for that many claims or more.
#'
#' Levels are numbered from 0 as in the literature; a vector over the levels
#' has level `l` in element `l + 1`.
#'
#' @param levels The number of levels, at least 2 (with a rule).
#' @param start The level a newcomer starts on.
#' @param down Levels down after a claim-free year.
#' @param up_per_claim Levels up for each claim, at least 1.
#' @param table A matrix of next levels, instead of a rule.
#' @param ptr Internal.
#' @returns A `bonus_malus` with properties `levels`, `start` and
#'   `transitions` (the table).
#' @seealso [bonus_malus_portfolio()], [bonus_malus_stationary()]
#' @export
#' @examples
#' scale <- bonus_malus(6, start = 5, up_per_claim = 3)
#' scale@transitions
#' bonus_malus_transition(scale, 0.1)
bonus_malus <- function(levels = NULL, start, down = 1, up_per_claim = NULL, table = NULL) {
  ptr <- if (is.null(table)) {
    if (is.null(levels) || is.null(up_per_claim)) {
      stop("give levels and up_per_claim, or a table", call. = FALSE)
    }
    BonusMalusR$rule(as.double(levels), as.double(start), as.double(down), as.double(up_per_claim))
  } else {
    table <- as.matrix(table)
    BonusMalusR$table(as.double(start), as.double(t(table)), as.double(ncol(table)))
  }
  bonus_malus_scale(ptr = rust_result(ptr))
}

#' @rdname bonus_malus
#' @export
bonus_malus_scale <- S7::new_class(
  "bonus_malus",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("BonusMalusR"),
    levels = S7::new_property(S7::class_double, getter = function(self) self@ptr$levels()),
    start = S7::new_property(S7::class_double, getter = function(self) self@ptr$start()),
    transitions = S7::new_property(S7::class_any, getter = function(self) {
      matrix(self@ptr$transitions(), nrow = self@levels, byrow = TRUE)
    })
  ),
  constructor = function(ptr) S7::new_object(S7::S7_object(), ptr = ptr)
)

S7::method(print, bonus_malus_scale) <- function(x, ...) {
  cat("<bonus_malus> ", x@levels, " levels, start ", x@start, "\n", sep = "")
  invisible(x)
}

#' A bonus-malus scale at one claim frequency
#'
#' For a policyholder with Poisson claims of mean `lambda`:
#' `bonus_malus_transition()` is the one-year transition matrix (row `l + 1`,
#' column `j + 1` the probability of moving from level `l` to `j`);
#' `bonus_malus_stationary()` the stationary distribution of the levels;
#' `bonus_malus_after()` the distribution after `years` years from the start
#' level; `bonus_malus_years()` the years from the start level until that
#' distribution is within `tolerance` of the stationary one in total
#' variation (`NA` past 100,000 years); `bonus_malus_efficiency()`
#' Loimaranta's efficiency `d ln r(lambda) / d ln lambda` of the scale with
#' `relativities`, exact, and `bonus_malus_mean_relativity()` that
#' `r(lambda)`, the stationary mean relativity. The last two take a vector
#' of frequencies.
#'
#' @param scale A [bonus_malus()] scale.
#' @param lambda The expected claim frequency.
#' @param years Years from the start level.
#' @param tolerance The total variation distance to reach, in `(0, 1)`.
#' @param relativities One relativity per level.
#' @returns A matrix, a vector over the levels, a number of years or a
#'   vector over `lambda`.
#' @export
#' @examples
#' scale <- bonus_malus(10, start = 6, up_per_claim = 2)
#' bonus_malus_stationary(scale, 0.1)
#' bonus_malus_years(scale, 0.1, 0.01)
#' r <- bonus_malus_portfolio(scale, 0.1, cv = 1)$relativity
#' bonus_malus_efficiency(scale, c(0.05, 0.1, 0.2), r)
bonus_malus_transition <- function(scale, lambda) {
  n <- scale@levels
  matrix(rust_result(scale@ptr$transition_matrix(as.double(lambda))), n, n, byrow = TRUE)
}

#' @rdname bonus_malus_transition
#' @export
bonus_malus_stationary <- function(scale, lambda) {
  rust_result(scale@ptr$stationary_at(as.double(lambda)))
}

#' @rdname bonus_malus_transition
#' @export
bonus_malus_after <- function(scale, lambda, years) {
  rust_result(scale@ptr$distribution_after(as.double(lambda), as.double(years)))
}

#' @rdname bonus_malus_transition
#' @export
bonus_malus_years <- function(scale, lambda, tolerance = 0.01) {
  n <- rust_result(scale@ptr$years_to_stationarity(as.double(lambda), as.double(tolerance)))
  if (is.nan(n)) NA_real_ else n
}

#' @rdname bonus_malus_transition
#' @export
bonus_malus_efficiency <- function(scale, lambda, relativities) {
  vapply(as.double(lambda), function(l) {
    rust_result(scale@ptr$efficiency(l, as.double(relativities)))
  }, numeric(1))
}

#' @rdname bonus_malus_transition
#' @export
bonus_malus_mean_relativity <- function(scale, lambda, relativities) {
  vapply(as.double(lambda), function(l) {
    rust_result(scale@ptr$mean_relativity(l, as.double(relativities)))
  }, numeric(1))
}

#' A bonus-malus scale over a mixed Poisson portfolio
#'
#' The stationary distribution of the levels and Norberg's optimal
#' relativities under quadratic loss over a portfolio of a priori classes
#' with frequencies `lambdas` and weights `weights`, and a residual
#' heterogeneity `Theta` of mean 1 and coefficient of variation `cv`, gamma
#' (a negative binomial portfolio) or inverse Gaussian. A policyholder of
#' class `k` has Poisson claims with mean `lambdas[k] * Theta`; the
#' relativity of level `l` is `E[Theta | level l]` over the stationary
#' portfolio, the a priori premium charged as a factor, so the relativities
#' balance to 1: `sum(stationary * relativity) == 1`. A level the portfolio
#' never reaches gets `NaN`.
#'
#' @param scale A [bonus_malus()] scale.
#' @param lambdas The classes' expected claim frequencies.
#' @param cv The coefficient of variation of `Theta`.
#' @param mixing `"gamma"` or `"inverse_gaussian"`.
#' @param weights The classes' weights; equal by default.
#' @returns A data frame with `level` (from 0), `stationary` and
#'   `relativity`.
#' @export
#' @examples
#' # -1/+top on six levels, a negative binomial portfolio.
#' scale <- bonus_malus(6, start = 5, up_per_claim = 5)
#' bm <- bonus_malus_portfolio(scale, 0.1, cv = 1)
#' bm
#' sum(bm$stationary * bm$relativity)
bonus_malus_portfolio <- function(scale, lambdas, cv, mixing = "gamma", weights = NULL) {
  lambdas <- as.double(lambdas)
  if (is.null(weights)) weights <- rep(1, length(lambdas))
  r <- rust_result(scale@ptr$portfolio(lambdas, as.double(weights), as.double(cv), mixing))
  data.frame(level = seq_len(scale@levels) - 1, stationary = r$stationary, relativity = r$relativity)
}
