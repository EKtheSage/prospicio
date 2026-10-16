# Design note: casualty exposure rating

Status: **Proposed** (method approved by the user 2026-10-16; no code yet) ·
Depends on: `prospicio_prob::Pareto` and its censored fit,
`prospicio_pricing::layer`, `prospicio_pricing::experience` · Lane: Aggregate

## Goal

A casualty excess-of-loss treaty over a book of excess (umbrella) policies
often has almost no experience in its layer. The case behind this note was a
5M xs 5M per-policy layer over an excess book with 49 claims, of which 3 were
material and none reached 5M. Two of those three were in a segment the cedant
now excludes, and the primary book could not reach the layer either: its limits
were 1M and 2M, and its largest loss was 2.8M. Classical credibility for 3
claims is about 5% (`√(3/1082)`).

Exposure rating prices such a layer policy by policy: each policy's expected
loss is shared out along a severity curve, and the share that falls in the
treaty layer is the treaty's expected loss. prospicio has the property side
(`RiskProfile` with MBBEFD and tabulated exposure curves) and the casualty
pieces (`Pareto` with a censored fit, `ilf`, `excess_frequency`, the alphas
implied by two frequencies or a frequency and a layer, and the large-loss
burning cost). It has no rater that takes a list of casualty policies, each
with its own attachment, limit and premium. This note adds that rater, the
range of results across tail weights, and two checks against the book's own
experience.

The reference is Clark's *Basics of Reinsurance Pricing*, the casualty
exposure rating section with increased limits factors (ILFs) and allocated
loss adjustment expense (ALAE) treatment. Whether its worked example can be
reproduced exactly is to be confirmed at the start of PR 1, as was done for
its experience rating example (`knowledge/references/clark-reinsurance-pricing.md`).

## Method

For policy `i`, take:

- an underlying attachment `a_i` (0 for a primary policy),
- a limit `l_i`,
- the share `s_i` of the layer it writes,
- a subject premium `P_i` and an expected loss ratio `ELR_i`,
- a ground-up severity `X` per occurrence.

The policy pays `Y_i = s_i · min((X − a_i)+, l_i)`. A per-policy treaty
`R xs D` cedes `min((Y_i − D)+, R)`. The part ceded lies on the ground-up
interval `[a_i + D/s_i, a_i + min((D + R)/s_i, l_i)]`, which is empty when
the policy cannot reach the treaty.

Two forms give the treaty's expected loss, and the rater supports both:

1. **The ILF form.** The policy's expected loss `P_i · ELR_i` is shared out
   by limited expected values (LEVs):
   `ceded_i = P_i · ELR_i · [LEV(top_i) − LEV(bottom_i)] / [LEV(a_i + l_i) − LEV(a_i)] · s_i`,
   with `bottom_i` and `top_i` the ends of the ceded interval (the share
   rescales back to the policy's own amounts). This form needs a curve from
   the ground up: a severity, or an ILF table, which is a LEV curve up to a
   constant.
2. **The frequency form.** The expected number `n_i` of occurrences that
   reach a threshold `t` under policy `i`, times the expected treaty loss
   given that `X > t`. A Pareto from `t` is enough here; the body of the
   curve below `t` never enters. In the motivating case, this is the form
   the primary book anchors.

## Decisions (proposed)

1. **Both forms, one policy table.** A `CasualtyProfile` holds rows of
   `(attachment, limit, share, premium, elr)`, with an optional
   `expected_count_above(t)` for the frequency form and an optional tag
   (segment, line) for reporting. A treaty is a per-policy `XsLayer`. The
   result keeps every policy's ceded expected loss, so that a segment can be
   shown or left out without refitting.
2. **The severity is any `Severity`, or an ILF table.** An `IlfTable`
   (limits and factors at a basic limit) is read as a LEV curve. Between
   the table's limits it interpolates in log of the limit, and the result
   must stay concave, as with the log-log `TabulatedCurve`. Beyond the last
   limit it is an error rather than an extrapolation. For a Pareto layer
   above the table, use the frequency form or a spliced severity.
3. **ALAE is an explicit choice:**
   - `Included`: the severity is loss plus ALAE, and limits apply to the sum.
   - `ProRata(e)`: the limits apply to loss only, and every ceded amount
     carries a factor `1 + e`.
   - `Excluded`: no ALAE.

   There is no default, because the treatment moves an excess layer
   materially. In the motivating case, a largest primary loss of 2.8M on
   1M and 2M limits points at ALAE outside the limits, or at stacked limits.
4. **The anchor is fitted, not read off one count.** The Pareto above a
   threshold `t` is `Pareto::fit` on the primary claims above `t`, after
   trend, with claims closed at their policy limit marked `censored`. The
   censored maximum likelihood estimate is
   `α = uncensored count / Σ ln(y/t)`, and its standard error comes from
   `fit_with_standard_error`. `alpha_between_frequencies` stays available
   when only counts at two thresholds are known. The fit uses the primary
   book because excess claims are too few. The note's PR records how many
   exceedances the anchor rests on.
5. **The result is a sweep, not one number.** `CasualtyProfile::sweep(&
   treaty, alphas)` gives the treaty's expected loss and loss cost at each
   alpha (the motivating case used 1.0 to 2.0). The report puts it beside a
   central alpha that the caller names, with its source: the fitted alpha,
   a benchmark from the market's casualty Pareto alphas or ILF curves, or a
   credibility blend of the two. At α = 1, from a 1M threshold, 20% of the
   claims above 1M pass 5M; at α = 2, 4% do. The expected cost of a 5M xs
   5M layer differs by a factor of about 7 between the two, so a range with
   no stated central value does not set a price.
6. **Losses are trended and developed before any check or anchor.** Both
   checks and the anchor take `experience::Prepared` losses (trend, IBNER,
   `complete_above`). "No loss reached the layer" is a statement about the
   untrended record. A 4.95M loss from earlier years, trended to the rating
   period, is above 5M.
7. **Check 1: counts against the model (a Poisson test).** At each of a few
   thresholds (say 1M, 2M, 3M, and the attachment), the expected number of
   claims over the experience period comes from the profile, on-levelled to
   each year's exposure. It is set beside the observed count of trended
   claims, with the Poisson probability of a count at least as extreme. This
   tests the alpha directly. Where the layer has never been reached, it is
   more informative than the burning cost.
8. **Check 2: the burning cost, with and without a segment.** The burning
   cost is the existing `Prepared::burning_cost` on the trended losses. It
   is run on all losses and with the tagged segment left out, and both are
   shown beside the exposure result. Whether an excluded segment is left
   out of the rate is the cedant's wording to settle, not the library's.
   The library makes leaving it out a visible choice, never a silent one.
   The severity anchor stays on the full record unless the user says
   otherwise.
9. **Credibility weighting is reported, not imposed.** The classical
   weight (`prospicio_pricing::credibility`) of the experience against
   the exposure estimate is given. In the motivating case it is about 5%,
   so the exposure estimate is the price.
10. **Clash is out of scope for now.** A per-policy treaty rated policy by
    policy cannot see one occurrence hitting several policies. Wildfire and
    construction defect are both exposed to that. The result says so, and a
    clash load is the user's input until an event-linked model of policies
    is designed.

## Plan

- **PR 1:** `CasualtyProfile`, `IlfTable` and the two forms with ALAE, in
  `prospicio_pricing::casualty`. Validation:
  - closed forms on a Pareto (the ILF form and the frequency form agree
    for an excess policy above the threshold);
  - the ILF form against `ilf` and `layer` for one policy;
  - Clark's casualty exposure example, if it reproduces.
- **PR 2:** the sweep, the censored anchor with its standard error, the
  count test and the burning-cost comparison with and without a segment.
  Validation: the Poisson tail against `prospicio-prob`, and the anchor
  against `Pareto::fit` on the same data.
- **PR 3:** Python (`prospicio.pricing.CasualtyProfile`) and R
  (`casualty_profile()`, `casualty_sweep()`), with a guide example handed
  to the Guides lane.
- Excel, if wanted, is the Excel lane's to wrap afterwards.

## Open questions for the user

- Should the rater take on-level premium and ELR per policy, or per segment
  with a policy's premium as weight? The note assumes per policy.
- Should the market alpha benchmarks be shipped as data in the library, or
  always be the user's input? The note assumes the user's input, with the
  source named in the report.
