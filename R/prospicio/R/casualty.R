# Casualty exposure rating (Aggregate lane): a table of policies rated
# for a per-policy treaty, over prospicio_pricing::casualty
# (docs/design/casualty-exposure.md).

#' ILF and ELF tables for casualty exposure rating
#'
#' A table read as a limited expected value curve: linear between its
#' limits and from the origin to the first, checked concave, and refused
#' beyond the last limit. `ilf_table()` takes increased limit factors,
#' `elf_table()` excess loss factors `ELF(x) = E[(X - x)+] / E[X]`.
#'
#' @param limits Positive, increasing limits.
#' @param factors Increased limit factors at `limits`.
#' @param elfs Excess loss factors at `limits`, decreasing in (0, 1).
#' @returns A table for [casualty_profile()]'s `curves`.
#' @export
#' @examples
#' ilf_table(c(100e3, 1e6, 2e6), c(1, 1.6, 1.8))
ilf_table <- function(limits, factors) {
  structure(list(kind = "ilf", limits = as.double(limits), values = as.double(factors)),
            class = "prospicio_casualty_table")
}

#' @rdname ilf_table
#' @export
elf_table <- function(limits, elfs) {
  structure(list(kind = "elf", limits = as.double(limits), values = as.double(elfs)),
            class = "prospicio_casualty_table")
}

#' @export
print.prospicio_casualty_table <- function(x, ...) {
  cat(sprintf("<%s table> %d limits\n", toupper(x$kind), length(x$limits)))
  print(stats::setNames(x$values, format(x$limits, big.mark = ",", scientific = FALSE)))
  invisible(x)
}

#' A book of casualty policies for exposure rating
#'
#' Policy `i` pays `share * min((X - attachment)+, limit)` for a ground-up
#' loss `X` per occurrence. A per-policy treaty `limit` xs `attachment`
#' cedes the part above its retention; the share of the policy's expected
#' loss it takes is its exposure factor,
#' `[LEV(a + min(l, (D + R)/s)) - LEV(a + min(l, D/s))] / [LEV(a + l) - LEV(a)]`.
#' Each policy is rated on its own curve.
#'
#' @param curves A list of severities, [ilf_table()]s and [elf_table()]s.
#' @param limit,premium,elr One value per policy: the limit above the
#'   attachment (`Inf` for none, as in workers compensation), the on-level
#'   premium and the expected loss ratio.
#' @param attachment,share Underlying attachment (0) and share of the layer
#'   written (1), recycled.
#' @param curve Each policy's curve, an index into `curves` (1).
#' @param tag A segment or line per policy, for reporting and for leaving a
#'   segment out (`NA` for none).
#' @returns A `casualty_profile`.
#' @seealso [casualty_rate()], [casualty_sweep()], [casualty_count_test()].
#' @export
#' @examples
#' # Clark's workers compensation example, 750,000 xs 250,000.
#' book <- casualty_profile(list(elf_table(c(250e3, 1e6), c(0.030, 0.006))),
#'                          limit = Inf, premium = 100e3, elr = 0.70)
#' casualty_rate(book, 750e3, 250e3, alae = "in_severity")$ceded
casualty_profile <- S7::new_class(
  "casualty_profile",
  package = "prospicio",
  properties = list(
    ptr = S7::new_S3_class("CasualtyBook"),
    tag = S7::class_character,
    premium = S7::class_double
  ),
  constructor = function(curves, limit, premium, elr, attachment = 0, share = 1, curve = 1,
                         tag = NA) {
    if (!is.list(curves) || inherits(curves, "prospicio_casualty_table")) curves <- list(curves)
    kinds <- vapply(curves, function(c) {
      if (inherits(c, "prospicio_casualty_table")) c$kind else "severity"
    }, "")
    sev <- lapply(curves, function(c) if (inherits(c, "prospicio_casualty_table")) NULL else c@ptr)
    lim <- lapply(curves, function(c) if (inherits(c, "prospicio_casualty_table")) c$limits else NULL)
    val <- lapply(curves, function(c) if (inherits(c, "prospicio_casualty_table")) c$values else NULL)
    n <- max(length(limit), length(premium), length(elr))
    col <- function(v, name) {
      if (length(v) == 1) v <- rep(v, n)
      if (length(v) != n) stop(sprintf("%s has %d values for %d policies", name, length(v), n))
      v
    }
    premium <- col(as.double(premium), "premium")
    tag <- col(as.character(tag), "tag")
    ptr <- rust_result(casualty_book_rust(
      kinds, sev, lim, val, col(as.double(limit), "limit"), premium, col(as.double(elr), "elr"),
      col(as.double(attachment), "attachment"), col(as.double(share), "share"),
      col(as.double(curve), "curve") - 1, ifelse(is.na(tag), "", tag)
    ))
    S7::new_object(S7::S7_object(), ptr = ptr, tag = tag, premium = premium)
  }
)

S7::method(print, casualty_profile) <- function(x, ...) {
  cat(sprintf("<casualty_profile> %d policies, premium %s\n", length(x@premium),
              format(sum(x@premium), big.mark = ",")))
  invisible(x)
}

casualty_loading <- function(alae, loading) {
  alae <- match.arg(alae, c("pro_rata", "on_top", "in_severity"))
  list(alae = alae, loading = if (is.null(loading)) NaN else as.double(loading))
}

casualty_rows <- function(profile, r) {
  data.frame(policy = seq_along(r$ceded), tag = profile@tag, factor = r$factor,
             expected_loss = r$expected_loss, ceded = r$ceded, premium = r$premium)
}

#' Rate a casualty treaty over a book of policies
#'
#' `casualty_rate()` is the ILF form: each policy's `premium * elr` times
#' its exposure factor on its own curve. `casualty_rate_by_frequency()` is
#' the frequency form: each policy's expected ground-up occurrences above
#' `threshold` (`counts`, from [casualty_occurrences()] say) times the
#' expected ceded amount given `X > threshold` under `excess`, a severity of
#' the losses above it. Both keep the result policy by policy, so a segment
#' is left out by subsetting on `tag`.
#'
#' ALAE has no default. `"pro_rata"`: ALAE in proportion to loss, the
#' ceded loss times `1 + loading` on a loss-only ELR. `"on_top"`: the
#' policy limit caps loss only and ALAE is `loading` times it; the treaty's
#' retention and limit apply to loss and ALAE (Clark's second formula).
#' `"in_severity"`: the curves are of loss and ALAE, and `loading` is not
#' given.
#'
#' @param profile A [casualty_profile()].
#' @param limit,attachment The per-policy treaty, `limit` xs `attachment`.
#' @param alae `"pro_rata"`, `"on_top"` or `"in_severity"`.
#' @param loading The ALAE loading `e`, required with `"pro_rata"` and
#'   `"on_top"`.
#' @param threshold The size the counts start at.
#' @param counts Each policy's expected occurrences above `threshold` a year.
#' @param excess A severity of the losses above `threshold`.
#' @returns A data frame, one row per policy: `policy`, `tag`, `factor` and
#'   `expected_loss` (`NaN` in the frequency form), `ceded`, `premium`.
#'   The treaty's loss cost is `sum(ceded) / sum(premium)`.
#' @export
#' @examples
#' book <- casualty_profile(list(pareto(100e3, 1.3)), limit = c(1e6, 2e6),
#'                          premium = c(300, 200), elr = 0.6, tag = c("gl", "wf"))
#' r <- casualty_rate(book, 500e3, 500e3, alae = "pro_rata", loading = 0.1)
#' sum(r$ceded[r$tag != "wf"])
casualty_rate <- function(profile, limit, attachment, alae, loading = NULL) {
  a <- casualty_loading(alae, loading)
  casualty_rows(profile, rust_result(profile@ptr$rate(as.double(limit), as.double(attachment),
                                                        a$alae, a$loading)))
}

#' @rdname casualty_rate
#' @export
casualty_rate_by_frequency <- function(profile, limit, attachment, alae, threshold, counts,
                                       excess, loading = NULL) {
  a <- casualty_loading(alae, loading)
  casualty_rows(profile, rust_result(profile@ptr$rate_by_frequency(
    as.double(limit), as.double(attachment), a$alae, a$loading, as.double(threshold),
    as.double(counts), excess@ptr
  )))
}

#' Counts, the Pareto sweep and the count test for casualty exposure
#'
#' `casualty_occurrences()` gives each policy's expected ground-up
#' occurrences above `t` a year from its own curve,
#' `premium * elr / (share * [LEV(a + l) - LEV(a)]) * S(t)`, a table's `S`
#' its slope just above `t`. `casualty_sweep()` runs the frequency form at
#' each alpha of a Pareto from `threshold`, the counts held fixed; the
#' central alpha and its source are yours to state, since prospicio ships
#' no benchmark. `casualty_expected_claims()` is the number of claims a year
#' recorded at or above the ground-up size `t` (a policy records an
#' occurrence once it passes its attachment and caps it at `a + l`), and
#' `casualty_count_test()` sets the expected count over the experience
#' period beside the observed one at each size, with the Poisson
#' probability of a count at least as extreme in the direction observed
#' ([poisson_tail()]).
#'
#' @inheritParams casualty_rate
#' @param t A ground-up size.
#' @param alphas Pareto alphas.
#' @param sizes Ground-up sizes, at or above `threshold`.
#' @param exposure_years The experience period in the profile's years: each
#'   year's on-level premium over the profile's, summed.
#' @param observed The trended ground-up losses, a claim closed at its limit
#'   at the limit.
#' @returns `casualty_occurrences()`: one count per policy.
#'   `casualty_sweep()`: a data frame of `alpha`, `ceded`, `loss_cost`.
#'   `casualty_expected_claims()`: a number. `casualty_count_test()`: a
#'   data frame of `size`, `expected`, `observed`, `p_value`.
#' @export
#' @examples
#' book <- casualty_profile(list(pareto(1e6, 1.5)), limit = 10e6, premium = 1e6, elr = 0.6)
#' n <- casualty_occurrences(book, 1e6)
#' casualty_sweep(book, 5e6, 5e6, alae = "in_severity", threshold = 1e6, counts = n,
#'                alphas = c(1, 1.5, 2))
#' casualty_count_test(book, c(1e6, 2e6), threshold = 1e6, counts = n,
#'                     excess = pareto(1e6, 1.5), exposure_years = 5,
#'                     observed = c(1.2e6, 2.5e6))
casualty_occurrences <- function(profile, t) {
  rust_result(profile@ptr$occurrences_above(as.double(t)))
}

#' @rdname casualty_occurrences
#' @export
casualty_sweep <- function(profile, limit, attachment, alae, threshold, counts, alphas,
                           loading = NULL) {
  a <- casualty_loading(alae, loading)
  as.data.frame(rust_result(profile@ptr$sweep(
    as.double(limit), as.double(attachment), a$alae, a$loading, as.double(threshold),
    as.double(counts), as.double(alphas)
  )))
}

#' @rdname casualty_occurrences
#' @export
casualty_expected_claims <- function(profile, t, threshold, counts, excess) {
  rust_result(profile@ptr$expected_claims(as.double(t), as.double(threshold), as.double(counts),
                                          excess@ptr))
}

#' @rdname casualty_occurrences
#' @export
casualty_count_test <- function(profile, sizes, threshold, counts, excess, exposure_years,
                                observed) {
  as.data.frame(rust_result(profile@ptr$count_test(
    as.double(sizes), as.double(threshold), as.double(counts), excess@ptr,
    as.double(exposure_years), as.double(observed)
  )))
}

#' The Pareto anchor of a casualty exposure rating
#'
#' The Pareto above `threshold` fitted to the trended losses at or above
#' it, with claims closed at their policy limit censored:
#' `alpha = uncensored count / sum(log(y / threshold))`, its standard error
#' `alpha / sqrt(n)`, and the number of exceedances it rests on.
#'
#' @param threshold The Pareto's threshold.
#' @param losses Trended ground-up losses; those below `threshold` are
#'   ignored.
#' @param censored Where a loss was closed at its policy limit, recycled.
#' @returns A list: `threshold`, `alpha`, `standard_error`, `exceedances`,
#'   `censored`.
#' @export
#' @examples
#' pareto_anchor(1e6, c(1.5e6, 2.5e6, 4e6, 2e6), censored = c(FALSE, FALSE, FALSE, TRUE))
pareto_anchor <- function(threshold, losses, censored = FALSE) {
  losses <- as.double(losses)
  censored <- as.double(rep_len(as.logical(censored), length(losses)))
  rust_result(pareto_anchor_rust(as.double(threshold), losses, censored))
}

#' One-sided Poisson tail toward an observed count
#'
#' The probability, under a Poisson with mean `expected`, of a count at
#' least as far from it as `observed` on the side observed: `P(N >= n)`
#' when `n >= expected`, `P(N <= n)` below it.
#'
#' @param expected The Poisson mean.
#' @param observed A whole number.
#' @returns A probability.
#' @export
#' @examples
#' poisson_tail(3, 0) # exp(-3)
poisson_tail <- function(expected, observed) {
  rust_result(poisson_tail_rust(as.double(expected), as.double(observed)))
}
