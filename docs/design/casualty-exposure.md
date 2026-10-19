# Design note: casualty exposure rating

Status: **In progress** (method approved by the user 2026-10-16, open
questions answered 2026-10-17; PRs 1 and 2 done: `prospicio_pricing::casualty`) ·
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

The reference is Clark's *Basics of Reinsurance Pricing*, section 3A(b)
(casualty exposure rating) and 3B(a) (umbrella policies), checked
2026-10-17 against the 2014 revision. What it prints:

- **One numeric example, and it is workers compensation.** Four state and
  hazard group rows, each with a standard premium, an ELR and two excess
  loss factors (ELFs), for a 750,000 xs 250,000 treaty:
  `exposure factor = ELF(250,000) − ELF(1,000,000)`, treaty losses =
  premium × ELR × factor (1,680, 2,240, 4,250 and 5,525), 13,695 in
  total and a 3.42% loss cost on 400,000. The arithmetic reproduces
  exactly from the printed inputs, so it is PR 1's parity target.
- **The liability formulas, without numbers.** The exposure factor
  `[E[X; min(PL, AP + Lim)] − E[X; min(PL, AP)]] / E[X; PL]` for a
  primary policy; with ALAE included in the treaty's loss,
  `[E[X; min(PL, (AP + Lim)/(1 + e))] − E[X; min(PL, AP/(1 + e))]] /
  E[X; PL]`; and for an umbrella `UL` xs, limit `PL`,
  `[E[X; min(UL + PL, UL + AP + Lim)] − E[X; min(UL + PL, UL + AP)]] /
  [E[X; UL + PL] − E[X; UL]]`. Its two illustrations are symbolic, not
  numeric: an umbrella 1M xs 1M under a 500K xs 500K treaty gives
  `[E[X;2M] − E[X;1.5M]] / [E[X;2M] − E[X;1M]]`, and a 1M policy limit
  with an ALAE loading of 1.5 leaves a treaty attaching at 1.5M
  unexposed. Its truncated Pareto ("BQPST") and mixed exponential
  severities have no printed parameters, so no liability number can be
  reproduced. These formulas are checked instead against closed forms on
  a Pareto, and the two illustrations as identities.
- **The drop-down of an umbrella** uses an aggregate excess factor `ϕ` on
  the underlying policy, defined only in the note's later section on
  aggregate distributions; it is left out of PR 1 (decision 11).

## Method

For policy `i`, take:

- an underlying attachment `a_i` (0 for a primary policy),
- a limit `l_i`,
- the share `s_i` of the layer it writes,
- a subject premium `P_i` and an expected loss ratio `ELR_i`,
- a ground-up severity `X` per occurrence.

The policy pays `Y_i = s_i · min((X − a_i)+, l_i)`. A per-policy treaty
`R xs D` cedes `min((Y_i − D)+, R)`. The part ceded lies on the ground-up
interval `[a_i + min(D/s_i, l_i), a_i + min((D + R)/s_i, l_i)]`, which is
empty when the policy cannot reach the treaty.

Two forms give the treaty's expected loss, and the rater supports both:

1. **The ILF form.** The policy's expected loss `P_i · ELR_i` is shared out
   by limited expected values (LEVs):
   `ceded_i = P_i · ELR_i · [LEV(top_i) − LEV(bottom_i)] / [LEV(a_i + l_i) − LEV(a_i)]`,
   with `bottom_i` and `top_i` the ends of the ceded interval. The share
   cancels: it scales the ceded amount and the policy's loss alike, and
   enters only through the interval. This form needs a curve from the
   ground up: a severity, or an ILF or ELF table, each a LEV curve up to a
   constant.
2. **The frequency form.** The expected number `n_i` of occurrences that
   reach a threshold `t` under policy `i`, times the expected treaty loss
   given that `X > t`. A Pareto from `t` is enough here; the body of the
   curve below `t` never enters. In the motivating case, this is the form
   the primary book anchors.

## Decisions

1. **Both forms, one policy table.** A `CasualtyProfile` holds rows of
   `(attachment, limit, share, premium, elr, curve)` with an optional tag
   (segment, line) for reporting; the frequency form
   (`rate_by_frequency`) takes each policy's expected count above the
   threshold as an argument, with a severity of the losses above it. A treaty is a per-policy `XsLayer`. The
   result keeps every policy's ceded expected loss, so that a segment can be
   shown or left out without refitting.
2. **The severity is any `Severity`, or an ILF table.** An `IlfTable`
   (limits and factors at a basic limit) is read as a LEV curve. It is
   linear between the table's limits and from the origin to the first
   (`LEV(0) = 0`), so it is exact at every printed limit, and it must be
   concave (slopes that never increase), which the constructor checks.
   Beyond the last limit it is an error rather than an extrapolation. For a Pareto layer
   above the table, use the frequency form or a spliced severity.
3. **ALAE is an explicit choice**, in Clark's two treatments and a
   third for a severity fitted to loss and ALAE together:
   - `ProRata(e)`: the treaty shares ALAE in proportion to loss. The
     exposure factor is the loss-only one, and the ceded amount carries
     `1 + e` on a loss-only ELR. With an ELR that already holds ALAE,
     `ProRata(0)` is Clark's first case.
   - `OnTop(e)`: the policy limit applies to loss only, ALAE is `e` times
     the capped loss, and the treaty's attachment and limit apply to loss
     plus ALAE: Clark's second formula, with `AP` and `AP + Lim` divided
     by `1 + e` before they are capped at the policy limit. The ELR is
     loss only, and the ceded amount carries `1 + e`.
   - `InSeverity`: the severity is loss plus ALAE, and the limits apply to
     the sum.

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
11. **Premium and ELR per policy** (user, 2026-10-17). Each row carries its
    own on-level premium and expected loss ratio. A segment's single ELR is
    given by repeating it on the segment's rows; a per-segment form, with a
    policy's premium as its weight, can come later if the per-policy table
    proves too granular in use.
12. **No market data ships with the library** (user, 2026-10-17). Benchmark
    alphas, ILF tables and excess loss factors are always the caller's
    input, and the report names the source the caller gives.
13. **A curve per row.** A row names its severity curve: a `Severity`, an
    `IlfTable`, or an `ElfTable` of excess loss factors (`ELF(x) = E[(X −
    x)+] / E[X]`, read between its limits as the ILF table is). Clark's
    workers compensation example has one ELF curve per state and hazard
    group, and the treaty rates each row on its own curve; a policy
    limit is optional, since compensation has none, and with no limit the
    factor is `ELF(AP) − ELF(AP + Lim)`.
14. **The umbrella's drop-down is left out of PR 1.** It needs an
    aggregate excess factor on the underlying policy, which Clark defines
    with the aggregate distributions; without it the umbrella's factor is
    the formula above, which ignores exhaustion of the underlying
    aggregate. The result says so.

## Plan

- **PR 1 (done):** `CasualtyProfile`, `IlfTable`, `ElfTable` and the two
  forms with ALAE, in `prospicio_pricing::casualty`. Validation:
  - Clark's workers compensation example exactly (each row's factor and
    treaty loss, 13,695 and 3.42%);
  - Clark's liability formulas against closed forms on a Pareto: the
    primary factor, the ALAE `OnTop` factor (and its zero when
    `AP/(1 + e)` reaches the policy limit, Clark's 1.5 loading example),
    the umbrella factor (his 1M xs 1M under 500K xs 500K as the identity
    he prints);
  - the ILF form and the frequency form agree for an excess policy above
    the threshold;
  - the ILF form against `ilf` and `layer` for one policy.
- **PR 2 (done):** the sweep, the censored anchor with its standard error, the
  count test and the burning-cost comparison with and without a segment.
  - `CasualtyProfile::sweep(&treaty, alae, threshold, counts, alphas)`: the
    frequency form with `Pareto(threshold, α)` at each alpha, the counts
    held fixed.
  - `CasualtyProfile::occurrences_above(t)`: each policy's ground-up
    occurrences above `t` from its own curve, `premium × ELR / (s ·
    [LEV(a + l) − LEV(a)]) × S(t)`, a table's `S` its slope (the curve's
    constant cancels). These are the counts the frequency form and the
    sweep take: the body of the curves sets the frequency, the Pareto the
    tail.
  - `ParetoAnchor::fit(t, losses, censored)`: `Pareto::fit_with_standard_error`
    on the losses at or above `t`, with the number of exceedances and of
    censored ones, and `interval(z)` for the sweep's range.
  - `CasualtyProfile::expected_claims` and `count_test` (check 1): a policy
    records an occurrence once it passes the attachment and caps it at
    `a + l`, so it adds `counts · S(max(t, a))` at sizes `t ≤ a + l` and
    nothing above; the observed count is of the trended losses at or above
    `t` (`Prepared::trended`), and `poisson_tail` gives the one-sided
    probability toward the observation. `exposure_years` is the experience
    period in units of the profile's year.
  - Check 2 is `Prepared::burning_cost` run twice, on the record and on
    `Prepared::without(ids)`, with the segment's premium left out of the
    years' bases by the caller.
  - Decision 9's classical weight is `credibility::LimitedFluctuation`
    on the claims in or near the layer, shown in the validation test;
    no new function.
  Validation (`validation/tests/pricing.rs`, `clark_casualty`): the
  anchor against `Pareto::fit` and its closed form on the same data, the
  sweep against a Pareto's layer at α = 1 and 2, the count test's tail
  against `prospicio-prob`'s Poisson pmf, the burning cost of 1M xs 1M
  with and without a segment, and the weight `√(1/1082)`.
- **PR 3:** Python (`prospicio.pricing.CasualtyProfile`) and R
  (`casualty_profile()`, `casualty_sweep()`), with a guide example handed
  to the Guides lane.
- Excel, if wanted, is the Excel lane's to wrap afterwards.

## Answered questions

- Premium and ELR per policy or per segment: per policy, to try first
  (decision 11).
- Market alpha benchmarks shipped or the user's input: the user's input;
  prospicio ships no data (decision 12).
- Whether Clark's example reproduces: its one numeric example (workers
  compensation, by ELF) does, exactly; its liability formulas print no
  numbers and are checked against closed forms instead (Goal, above).
