# Design note: IFRS 17 measurement (GMM and PAA, the CSM, the loss component, OCI, reinsurance held)

Status: **Accepted** (2026-11-27; issue #982, in the 0.4.0 milestone at the
user's request of 2026-11-27). The user decided open decisions 1 to 11 on
2026-11-27 as recommended. PR 1 (the general model's `Group`) is built
(branch `claude/project-thread-n5mttf-ifrs17-gmm`), and PR 2 (the current
curve and the OCI option, branch `claude/project-thread-n5mttf-ifrs17-oci`)
but for Example 15's parity, and PR 3 (the PAA, branch
`claude/project-thread-n5mttf-ifrs17-paa`) but for Example 10's parity, and
PR 4 (reinsurance held, branch
`claude/project-thread-n5mttf-ifrs17-reinsurance`) but for Examples 11 and
12's parity, and PR 5's Python half (the bindings and the JSON state,
branch `claude/project-thread-n5mttf-ifrs17-bindings`); the R half of PR 5
and PR 6 are to come. Lane: Reserving
(`crates/prospicio-reserving/`, the lane's binding modules) · Depends on:
`reserving-ifrs17.md` (the `discount` module: `DiscountCurve`,
`DiscountedReserves`, `RiskAdjustment`), `reinsurer-default.md` (Aggregate
lane, read only: a reinsurer's probability of default and loss given
default, for decision 9)

## Goal

`reserving-ifrs17.md` built the liability for incurred claims (LIC): each
simulation's unpaid claims discounted, and a risk adjustment (RA) with its
confidence level. A full IFRS 17 measurement needs the rest of the
insurance contract liability and the movements that report it:

1. **The liability for remaining coverage (LRC)** under the general
   measurement model (GMM, also called the building block approach) and
   under the premium allocation approach (PAA).
2. **The contractual service margin (CSM)**: set at initial recognition,
   accreted at the locked-in rate, adjusted for changes that relate to
   future service, released by coverage units.
3. **Onerous groups**: the loss at initial recognition or later, the loss
   component, its systematic allocation and its reversal.
4. **Insurance finance income or expense (IFIE)** split between profit or
   loss, at the locked-in rate, and other comprehensive income (OCI), the
   rest, when the entity takes the OCI option.
5. **Reinsurance contracts held**: the reinsurance CSM (a net cost or a net
   gain), the loss-recovery component on onerous underlying groups, and the
   reinsurer's non-performance risk.

The output is what an actuary hands to finance each reporting period: the
measurement components (present value of future cash flows, RA, CSM) and
the remaining coverage and incurred claims columns, each as an opening to
closing reconciliation, and the lines of the statement of profit or loss
(insurance revenue, insurance service expenses, IFIE in P&L and in OCI).

## What exists

* `prospicio_reserving::discount` (`reserving-ifrs17.md`): `DiscountCurve`
  (flat or spot rates by maturity, annually compounded),
  `discounted_reserves` from any stochastic fit with patterns, giving
  `DiscountedReserves` (the unpaid and its present value by simulation,
  the best estimate, its run-off by year), and `RiskAdjustment` (confidence
  level, TVaR, cost of capital) with the implied confidence level. This is
  the LIC for a group's incurred claims at a valuation date.
* `prospicio_pricing::PaymentPattern` and `price_over_payout`
  (`time-value.md`) discount a pattern of a premium's losses. The two
  crates do not depend on each other (`time-value.md` decision 37(a)).
* `prospicio_pricing::earned` (`primary-rating.md`, #981): written,
  earned, unearned and in-force exposure and premium from policy dates,
  with endorsements and cancellations. The PAA's revenue is the premium
  earned on the passage of time, which is this.
* `docs/design/reinsurer-default.md` (#980): a panel of reinsurers with
  probabilities of default and loss given default, and the expected credit
  loss on recoveries.
* No type for a group of contracts, an expected cash flow by period, a
  coverage unit, or a locked-in rate exists anywhere in the workspace.

## The method

The library measures one **group of contracts** (IFRS 17 paragraphs 14 to
24: a portfolio, an annual cohort, a profitability bucket; how a book is
split into groups is the user's). It does not project cash flows: the user
brings, for each reporting period, the expected future cash flows as at
the period's end, as they would from their own models or from the
library's fits (decision 2). The library does the accounting: the
roll-forward, the CSM and the loss component, and the split of the
finance effect. Every amount is deterministic, a probability-weighted
expected value (paragraph 33); only the RA comes from a distribution.

Notation, for one group at the end of period `t` (`t = 0` is initial
recognition):

* `F_t`: the present value of future cash flows (outflows less inflows),
  at the current curve; `F_t^L` the same at the locked-in curve.
* `R_t`: the RA for non-financial risk (an input, or from
  `RiskAdjustment` on a simulated distribution).
* `CSM_t`, `LC_t`: the CSM and the loss component; at most one of them is
  positive.
* `cu_t`: the coverage units provided in period `t`; `CU_t^+`: those
  expected after `t`.

### 1. Initial recognition (GMM; paragraphs 38 and 47)

```
FCF_0 = F_0 + R_0                (fulfilment cash flows; negative is a profit)
CSM_0 = max(0, -FCF_0 - pre-recognition cash flows derecognised)
LC_0  = max(0,  FCF_0 + ...)     (a loss in P&L at once, paragraph 47)
```

IFRS 17 Example 1 is the target: premium 900 received at once, outflows of
200 a year for three years at 5% (present value 545), RA 120: `FCF_0 =
-235`, `CSM_0 = 235`. With outflows of 400 a year (present value 1,089):
`FCF_0 = 309`, no CSM, and a loss of 309 (Example 1B).

### 2. Subsequent measurement (GMM; paragraphs 40 to 46, B96 to B119)

Each period, in this order (decision 4):

1. **Accretion** of the CSM at the locked-in rate, the rate at initial
   recognition (paragraph 44(b), B72(b)): `CSM += CSM · i^L_t`.
2. **Changes that relate to future service** (B96), measured at the
   locked-in curve: experience adjustments of premiums for future service,
   changes in the estimates of the present value of future cash flows and
   in the RA for future service. They adjust the CSM; a fall below zero
   is a loss and starts (or adds to) the loss component (paragraph 48);
   a favourable change first reverses the loss component, then rebuilds
   the CSM (paragraph 50(b)).
3. **Release** for the service of the period by coverage units (B119):
   `release_t = CSM · cu_t / (cu_t + CU_t^+)`, after steps 1 and 2.
4. The difference between the changes at the current and at the locked-in
   curve is a finance effect, not a CSM adjustment (B97(a)); it goes to
   IFIE (section 4).

IFRS 17 Example 2 is the target: the CSM of 235 accretes 12 at 5% and
releases 82 (247 / 3) in year 1, closing at 165; in year 2 a favourable
change of 68 for future service (Example 2A) gives 165 + 8 + 68 - 121 =
120, and an unfavourable one (2B) takes the CSM to nil and a loss of 113.

### 3. The loss component (paragraphs 49 to 52)

The loss component is a part of the LRC that tracks the loss recognised.
Changes in fulfilment cash flows of the LRC (releases of expected claims
and expenses, RA release, finance effects) are allocated between the loss
component and the rest of the LRC on a systematic basis (paragraph 50(a));
the share in the loss component is not revenue. The common basis, and the
one in IFRS 17 Example 8, is the ratio of the loss component to the
present value of future outflows plus the RA at the start of the period:
529 / 1,329 = 39.8% there (decision 5). Example 8's year 1 is the target:
a loss of 529, IFIE of 54 (33 on the LRC, 21 on the loss component),
insurance revenue of 289, a loss component of 359 at the end of the year
and a liability of 903.

### 4. Insurance finance income or expense and OCI (paragraphs 87 to 92, B128 to B136)

IFIE is the effect of the time value of money and its changes on the
group: the unwind of `F` and `R` at the current curve, the CSM's accretion
at the locked-in curve (section 2), and the effect of a change in the
curve. Without the OCI option all of it is in P&L. With it (paragraph 88(b),
for groups without direct participation features), P&L takes a systematic
allocation, the unwind at the locked-in curve (B131), and OCI takes the
rest, so that

```
IFIE_P&L = unwind of F^L and R at the locked-in curve + CSM accretion
IFIE_OCI = IFIE_total - IFIE_P&L          (accumulated OCI = F^L - F at the end)
```

For the LIC under the PAA the locked-in rate is the rate at the date the
claims were incurred (B133). IFRS 17 Example 15 is the parity target, read
before the PR that builds this (decision 11).

### 5. The premium allocation approach (paragraphs 53 to 59)

Eligible when the coverage period is a year or less, or the result is not
materially different from the GMM (paragraph 53; the user's judgment, not
tested). The LRC is

```
LRC_t = LRC_{t-1} + premiums received - acquisition cash flows paid
        + amortisation of acquisition cash flows - insurance revenue
        (+ accretion if there is a significant financing component, paragraph 56)
```

with insurance revenue the expected premium receipts allocated to the
period, by the passage of time (`prospicio_pricing::earned`, through the
bindings, decision 7) or by the expected pattern of incurred claims when it
differs significantly (B126). The entity may expense acquisition cash
flows when the coverage period is a year or less (paragraph 59(a)). The
onerous test (paragraph 57): when facts and circumstances indicate, a loss
component equals the GMM fulfilment cash flows of the remaining coverage
less the PAA LRC. The LIC need not be discounted when the claims are paid
within a year (paragraph 59(b)); otherwise it is `discounted_reserves` and
its RA. IFRS 17 Example 10 is the target.

### 6. Reinsurance contracts held (paragraphs 60 to 70A, B119C to B119F)

Measured like a GMM group with the signs reversed, with three
differences:

* **No loss at initial recognition**: the net cost or net gain of buying
  the cover is the reinsurance CSM, positive or negative (paragraph 65),
  except a net cost for past events, which is an expense at once
  (paragraph 65A).
* **The loss-recovery component** (paragraphs 66A, 66B, B119C to B119F):
  when an underlying group is onerous at recognition, or onerous contracts
  are added to it, the reinsurance CSM is adjusted and income recognised
  by `loss on the underlying × the share of its claims the reinsurance is
  expected to recover`. The component then tracks the underlying loss
  component and cannot exceed the part of it that the cover will recover.
* **Non-performance risk** (paragraph 63): the present value of the
  recoveries is reduced for the reinsurer's risk of default, including
  disputes; changes in it are in P&L, not the CSM. The expected credit
  loss comes from `reinsurer-default.md`'s panel (decision 9).
* **The RA** is the risk transferred to the reinsurer (paragraph 64).

IFRS 17 Examples 11 and 12A to 12C are the targets (decision 11).

### 7. What a period reports (paragraphs 100 to 105)

`GroupPeriod` holds, for one group and one period, the two
reconciliations the standard asks for, as rows of one table so that they
add up:

* **By remaining coverage and incurred claims** (paragraph 100): the LRC
  excluding the loss component, the loss component, and the LIC (present
  value and RA).
* **By measurement component** (paragraph 101, GMM only): the present
  value of future cash flows, the RA, and the CSM.

Rows: opening, new contracts, CSM accretion, changes for future service,
changes for current service (CSM release, RA release, experience
adjustments), changes for past service (the LIC), IFIE (P&L and OCI),
cash flows (premiums, claims and expenses, acquisition), closing. Beside
them the P&L lines: insurance revenue (B121 to B124: the expected claims
and expenses released, the RA released, the CSM released, the recovery of
acquisition cash flows; less the investment components), insurance service
expenses, losses on onerous groups and their reversals, and IFIE.

## API (sketch, for the decisions)

Rust (`crates/prospicio-reserving/src/ifrs17/`, a new module beside
`discount`):

```rust
pub enum Approach { General, PremiumAllocation { financing: bool } }

pub struct ExpectedCashFlows {   // as at one valuation date, by future period
    pub premiums: Vec<f64>, pub claims: Vec<f64>, pub expenses: Vec<f64>,
    pub acquisition: Vec<f64>, pub investment_component: Vec<f64>,
    pub timing: PaymentTiming,   // from discount.rs
}

pub struct PeriodInputs {        // what happened in period t, and the view at its end
    pub actual: ActualCashFlows,  // premiums received, claims and expenses paid, acquisition paid
    pub expected: ExpectedCashFlows,  // future, as at the end of t
    pub risk_adjustment: RiskAdjustmentInput,  // an amount, or a RiskAdjustment on draws
    pub coverage_units: f64,      // cu_t
    pub future_coverage_units: Vec<f64>,
    pub current_curve: DiscountCurve,
    pub incurred: Option<IncurredClaims>,  // the LIC: DiscountedReserves and its RA, or amounts
}

pub struct Group { /* approach, locked-in curve, OCI option, state after the last period */ }
impl Group {
    pub fn recognise(approach: Approach, at: ExpectedCashFlows, ra: f64,
                     curve: DiscountCurve, options: GroupOptions) -> Result<Self>;
    pub fn roll_forward(&mut self, period: &PeriodInputs) -> Result<GroupPeriod>;
    pub fn add_contracts(&mut self, ...) -> Result<()>;   // a cohort's later contracts, B72(b)
}

pub struct ReinsuranceHeld { /* a Group with the signs reversed, linked to underlying groups */ }
impl ReinsuranceHeld {
    pub fn recognise(..., underlying: &[(&Group, f64 /* share of claims recovered */)],
                     default: Option<NonPerformance>) -> Result<Self>;
    pub fn roll_forward(&mut self, period: &PeriodInputs, underlying: &[&GroupPeriod]) -> Result<GroupPeriod>;
}

pub struct GroupPeriod {
    pub rows: Vec<ReconciliationRow>,  // label, LRC excl. LC, LC, LIC PV, LIC RA; PV, RA, CSM
    pub insurance_revenue: f64, pub insurance_service_expenses: f64,
    pub onerous_losses: f64, pub ifie_pl: f64, pub ifie_oci: f64,
    pub closing: Balances,
}
```

Python and R: a `Group` class (Python) and an S7 object (R) over the same,
taking cash flows as columns by future period and returning each period's
reconciliation as a dict of columns or a data frame, and its state as the
versioned JSON of the other fits (`risk_rs.ifrs17_group`), so that a group
rolled forward this quarter can be saved and rolled forward next quarter.

## Open decisions

1. **Scope of the first build.** (a) The GMM (sections 1 to 3 and 7),
   then OCI, the PAA and reinsurance held, each in its own PR. (b) The PAA
   first, since most non-life groups use it. (c) The PAA only, leaving the
   GMM for long-term business. Recommended: (a): the PAA's onerous test
   needs the GMM's fulfilment cash flows, so the GMM is built first in any
   case, and Examples 1, 2 and 8 make a clean parity start.
2. **Where the expected cash flows come from.** (a) The user brings them
   by period; the library does the accounting only, and the guide shows
   how to build them from a fit's `DiscountedReserves` (the LIC) and an
   expected loss ratio with a pattern (the LRC). (b) The library builds the
   LRC cash flows from a pricing model (`prospicio-pricing`'s `Price` and
   `PaymentPattern`), which needs the two crates joined (decision 8).
   Recommended: (a): cash flow projection is where companies differ, and
   the accounting is where the standard is exact.
3. **Deterministic or by simulation.** (a) One expected set of cash flows
   per period, the RA an input or computed from draws; the measurement is
   of expected values, as the standard is. (b) A roll-forward per
   simulation, so that the CSM and the P&L lines have distributions (for
   ORSA or a capital model, #983). Recommended: (a) now; (b) is the ESG's
   business (#983) and can call (a) per scenario.
4. **The order of the CSM steps within a period.** (a) Accrete, adjust for
   future service, then release (section 2), the order of paragraph 44 and
   the IFRS 17 examples. (b) The entity's choice of order as a setting.
   Recommended: (a), with the release basis (next decision) a setting.
5. **The coverage units and the loss component's allocation.** Coverage
   units: (a) undiscounted by default, discounted as a setting (B119 allows
   either; the IASB's 2018 TRG discussion noted both are used); (b)
   discounted only. The loss component: (a) the ratio of section 3 (the
   loss component over outflows plus RA at the start of the period), as
   Example 8; (b) the user's ratio as an input. Recommended: (a) and (a),
   with (b) of the second as a later setting if a user asks.
6. **Granularity and cohorts.** (a) One `Group` per annual cohort and
   profitability bucket, with contracts added within the cohort's year at a
   weighted-average locked-in rate (B72(b)); grouping a book is the user's.
   (b) A `Portfolio` type that groups contracts by rules. Recommended: (a).
7. **The PAA's revenue pattern.** (a) The user gives the expected premium
   receipts allocated to each period (from `prospicio_pricing::earned`
   through the bindings, or their own); (b) the library computes it from
   policy dates, which needs `prospicio-pricing`. Recommended: (a): the
   bindings can chain the two crates, the Rust crates stay apart.
8. **The crate.** (a) A module of `prospicio-reserving`, beside `discount`,
   so the LIC, the curve and the RA are in reach and the crate graph is
   unchanged. (b) A new crate `prospicio-ifrs17` that depends on
   `prospicio-reserving` and `prospicio-pricing`, so it can take a pricing
   model's cash flows and the earned premium. Recommended: (a) now; (b)
   only if decisions 2 and 7 go to (b).
9. **Non-performance risk.** (a) An expected credit loss the user gives
   per period (an amount, or a probability of default and a loss given
   default applied to the recoveries); (b) `reinsurer-default.md`'s panel,
   which lives in `prospicio-aggregate`. Recommended: (a), with the guide
   showing (b) through the bindings.
10. **The variable fee approach, transition and presentation.** The VFA for
    direct participating contracts (paragraphs 45, B101 to B118), the
    modified retrospective and fair value transition approaches (C5 to
    C24), the presentation of the statement of financial position, and
    tax are out of this note. Recommended: out, each an issue if a user asks.
11. **The parity targets.** IFRS 17's Illustrative Examples (IFRS
    Foundation, the 2025 HTML edition) are the reference: Examples 1, 2
    (2A, 2B), 3 (3A, 3B), 6, 7 and 8's first year were read on 2026-11-27
    and their figures are in `knowledge/references/ifrs17-illustrative-examples.md`;
    Examples 8's later years, 10 (PAA), 11 and 12A to 12C (reinsurance
    held) and 15 (IFIE allocation) are to be read before the PR that
    builds each. The examples round to whole currency units at each step,
    so the tests match them to the unit, as with Werner and Modlin.
    Recommended: as stated; a published actuarial worked example on a
    non-life group (a CAS or IFoA paper) is added if one is found.

## Pull requests

1. `ifrs17::Group` under the GMM: initial recognition, the roll-forward,
   the CSM, the loss component, the reconciliation rows and the P&L lines
   (Examples 1, 2, 3, 6, 7 and 8). **Done** (branch
   `claude/project-thread-n5mttf-ifrs17-gmm`): `validation/tests/reserving_ifrs17_gmm.rs`
   matches Examples 1A, 1B, 2A, 2B, 3A, 7 and 8 to the unit (Example 8's
   years 2 and 3 within one and a half units, since the example chains on
   figures rounded twice: its closing present value of 94 is
   743 + 37 - 286 - 400, against 100 / 1.05 = 95.24). What it settled:
   * the loss component's share of the finance expense is its share of the
     unwind of its own base (the claims and expenses, investment components
     excluded, plus the RA), so that it runs off with the base by the end of
     the coverage; it equals Example 8's 21 and 15 (a share of the whole
     finance expense would leave a residue when premiums or investment
     components remain to be paid);
   * a loss component never exceeds the fulfilment cash flows; any excess
     at a period end is released with the period's share;
   * the RA's change is all insurance service result, the curve is the
     locked-in curve throughout, claims are paid in the period they are
     incurred (the LIC is nil, as in the examples), and contracts are not
     added during the cohort's year: the current curve and OCI come with
     PR 2, the LIC with PR 3, `add_contracts` with a later PR;
   * Example 6 (variable returns, a current rate beside the locked-in one,
     discretion) needs the current curve, so it moves to PR 2; its CSM
     release (213 / 394 × 198 = 107) is the coverage units' ratio PR 1
     already uses.
2. The current curve and the OCI option (Examples 6 and 15). **Done**
   for Example 6 (branch `claude/project-thread-n5mttf-ifrs17-oci`):
   `validation/tests/reserving_ifrs17_oci.rs` matches IE65 and IE71 to the
   unit. What it settled:
   * `PeriodInputs::with_current_curve` gives the curve at the period end
     (maturities from it); `Balances` keeps the present value at both the
     current and the locked-in curve, and the finance row is what the
     current measurement leaves after the changes for future service,
     which are measured at the locked-in curve (B97(a));
   * a change in financial assumptions is a revision of the opening view
     (`with_financial_revision`), its effect finance, not CSM; the
     period's expected cash flows are read from the revised view, so a
     claim paid as revised is no experience adjustment (Example 6's
     death benefit of 32);
   * `GroupOptions::accretion_curve` accretes the CSM at the rates for
     nominal cash flows when they differ from the locked-in curve that
     measures the CSM's adjustments (B72(b) and (c), Example 6's 4% and
     10%);
   * `GroupOptions::oci` puts in profit or loss the finance expense at the
     locked-in curve on the cash flows as revised (B131, B132(a)(i)) and
     the rest in OCI, so profit is the same as under a locked-in
     measurement and the accumulated OCI, the locked-in present value less
     the current one, recycles to nil when the coverage ends; the effective
     yield and projected crediting rate allocations (B132(a)(ii)) are not
     built;
   * the loss component's share of the finance expense stays at the
     locked-in curve;
   * Example 15 could not be read from a cloud session (the fetched text
     stops at IE70, and `www.ifrs.org` is refused to the shell), so its
     parity is a Next item for a session that can read it.
3. The PAA: the LRC, the onerous test, the LIC from `DiscountedReserves`
   (Example 10). **Done** but for Example 10's parity (branch
   `claude/project-thread-n5mttf-ifrs17-paa`,
   `validation/tests/reserving_ifrs17_paa.rs`, worked by hand). What it
   settled:
   * `ifrs17::paa::PaaGroup` is recognised from the expected premium
     receipts and acquisition cash flows of the whole coverage and a
     revenue pattern (equal weights for the passage of time, or the
     expected claims' pattern, B126; decision 7(a), the user's weights);
     each period's revenue and amortisation is what is not yet recognised
     times the period's weight over the remaining weights, so a revision
     of the expected premiums is spread forward;
   * acquisition cash flows are amortised on the same pattern or expensed
     when paid (`expense_acquisition`, paragraph 59(a));
   * the LIC is an input at each period end (`Lic`, or
     `Lic::from_reserves` from `DiscountedReserves` and its RA); claims
     incurred enter it at their fulfilment cash flows, its opening present
     value unwinds at `lic_curve` (none when undiscounted, paragraph
     59(b)), and the residual is the change for past service, a new
     `CoverageLine::PastService` row;
   * the onerous test takes the GMM fulfilment cash flows of the remaining
     coverage when the user has them (paragraph 57); without them the loss
     component runs off on the revenue pattern as a reversal;
   * the LRC does not accrete: no significant financing component
     (paragraph 56) is assumed;
   * IFRS 17 Example 10 was out of reach of a cloud session, as Example 15
     was, so its parity is a Next item.
4. Reinsurance held: the reinsurance CSM, the loss-recovery component,
   non-performance risk (Examples 11 and 12A to 12C). **Done** but for
   the examples' parity (branch
   `claude/project-thread-n5mttf-ifrs17-reinsurance`,
   `validation/tests/reserving_ifrs17_reinsurance.rs`, worked by hand, the
   user's choice of 2026-11-28 to build first and add the parity later).
   What it settled:
   * `ifrs17::reinsurance::ReinsuranceHeld` is its own type, measured as an
     asset (recoveries less premiums, less the non-performance reduction,
     plus the RA), not a `Group` with the signs reversed: it has no loss
     component, and its CSM may be negative (a net gain); its own
     `CededCashFlows` hold the premiums paid and the recoveries;
   * the CSM is released by coverage units, an expense in the allocation
     of premiums paid when a net cost and income when a net gain; with
     `retroactive` a net cost at recognition is expensed (65A);
   * the loss-recovery component is the share recovered times each
     underlying `Group`'s loss component, read from its `GroupPeriod`
     rows: at recognition it is income and an adjustment of the CSM (66A),
     only for an underlying group recognised at the same time (not yet
     rolled forward); later losses and reversals of the underlying group
     are income or expense in the same way, the CSM adjusted for the
     reinsurance's change in fulfilment cash flows less that matched amount
     (66(c)(i)); its release follows the underlying release and is left out
     of both the allocation of premiums paid and the amounts recovered
     (B119F); an underlying PAA group is not linked yet;
   * non-performance (decision 9) is an input of its own,
     `NonPerformance { default_probability, loss_given_default }`, a
     recovery due in `τ` years reduced by `(1 − (1 − p)^τ) × lgd`, rather
     than a dependency on `prospicio-aggregate`'s panel; every change in
     the reduction, its unwind and its release on recoveries received
     included, is in the amounts recovered (paragraph 67);
   * the asset for incurred claims is nil at each period end (recoveries
     received when incurred), and there is no OCI option for the
     reinsurance group, both as in PR 1;
   * Examples 11 and 12A to 12C were out of reach of a cloud session, so
     their parity is a Next item.
5. Python and R bindings and the JSON state. Built in Python
   (branch `claude/project-thread-n5mttf-ifrs17-bindings`):
   * `Ifrs17Group`, `PaaGroup` and `ReinsuranceHeld` in
     `prospicio.reserving`, built from dicts of cash-flow columns by future
     period (unknown keys refused with the list of allowed ones), rolled
     forward in place, each period returned as a dict whose reconciliations
     are dicts of columns (`line` names in snake case, as `csm_recognised`);
   * the state as versioned JSON (`risk_rs.ifrs17_group`,
     `risk_rs.ifrs17_paa_group`, `risk_rs.ifrs17_reinsurance_held`, format
     1, numbers written exactly as the other fits do), `to_json`,
     `from_json` and pickling, so a reloaded group rolls forward to the
     same numbers;
   * reinsurance held takes its underlying groups as `Ifrs17Group`
     objects: at recognition with their shares, and each period the groups
     already rolled forward to the same period, from which the matched
     onerous losses are read;
   * the R half follows in a PR of its own: CRAN was refused to cloud
     sessions on 2026-11-28, so S7 and testthat could not be installed.
6. A guide section, `docs/guides/reserving-methods.md` (the Guides lane's
   file, with the user's go-ahead or as an issue for that lane).

Each PR appends its Done bullet to `docs/lanes/done/reserving.md`, adds a
knowledge fragment, and a changelog fragment under "Reserving".

## Sources

- IFRS Foundation, *IFRS 17 Insurance Contracts* (2017, amended 2020),
  paragraphs and Appendix B as cited. The paragraph numbers in this note
  are from the standard as the author knows it and were not checked
  against the text, except where an example is quoted.
- IFRS Foundation, *IFRS 17 Insurance Contracts, Illustrative Examples*,
  2025 HTML edition,
  https://www.ifrs.org/content/dam/ifrs/publications/html-standards/english/2025/issued/ifrs17-ie.html.
