---
type: Finding
title: The IFRS 17 risk adjustment and the Solvency II risk margin as prospicio builds them
description: discounted_reserves (#553 PR 1) reads the risk adjustment off each simulation's discounted total; by a confidence level (inverted-cdf quantile), TVaR (exact on the draws) or a cost of capital with EIOPA's method 2 run-off and the capital charged at the end (Solvency II's risk margin, the default) or the start of each year; every result reports the confidence level it corresponds to. Regulatory paragraph references are the design note's and have not been checked against the standards' text.
tags: [ifrs17, solvency-ii, risk-adjustment, risk-margin, discounting, cost-of-capital, reserving]
status: stable
generated: { by: claude-code/local-session, at: 2026-11-14T12:00:00+00:00 }
sources:
  - id: note
    resource: ../docs/design/reserving-ifrs17.md
    title: docs/design/reserving-ifrs17.md (the method, decisions 1 to 19)
  - id: code
    resource: ../crates/prospicio-reserving/src/discount.rs
    title: prospicio_reserving::discount (DiscountCurve, discounted_reserves, RiskAdjustment, its unit tests)
  - id: timevalue
    resource: ../docs/design/time-value.md
    title: docs/design/time-value.md (decision 4, capital charged at the start of each year)
---

# Finding

* **What the risk adjustment is read from.** Every convention works on the
  discounted total `X`, one value per simulation: each simulation's unpaid
  by origin (its reserve; on the incurred basis its reserve plus the
  origin's case reserve) is spread over that simulation's paid pattern (or, for
  `Compartmental`, its paid cells), each payment discounted from the
  valuation date by `(1 + r(t))^-t` on an annual effective spot curve
  (linear between maturities, flat outside), and the origins (and segments)
  summed. The best estimate is `BE = mean(X)`. Quantiles of `X` are never
  formed by scaling undiscounted quantiles by a discount factor, which is
  exact for one origin only and changes the ranks of a total.[^code][^note]
* **Spreading by a bootstrap's pattern.** A bootstrap's pattern can pass 1
  and come back after an origin's latest age (pseudo factors below 1).
  Spreading the reserve by the raw increments over `1 - c(a_w)` then
  amplifies it into large payments of opposite signs when `c(a_w)` is just
  below 1: on RAA with 20,000 ODP bootstrap simulations at 3% mid-period,
  the unit present values by origin ran from -33 to 68 and the discounted
  total from 0.05 to 1.92 times the undiscounted, which corrupts the tail
  the risk adjustment is read from. The shares are therefore the rises of
  the pattern's running maximum over their sum: each in `[0, 1]`, so each
  draw's present value is its reserve times a weighted average of its
  payment times' factors (decision 17; a unit test checks the bound on
  2,000 ODP and Mack bootstrap simulations of RAA).[^note][^code]
* **An incurred fit discounts the whole unpaid.** A fit on incurred claims
  has reserves of ultimate less latest incurred, the IBNR alone; discounting
  only those leaves the case reserves (latest incurred less latest paid)
  out of the liability for incurred claims and its risk adjustment. Under
  the incurred basis the caller therefore brings each origin's latest paid
  with the paid pattern, and each simulation's unpaid, reserve plus case
  reserve, is spread along the paid pattern from the origin's latest age
  by the same running-maximum rule; at a zero rate the discounted total's
  mean is the mean reserve plus the sum of the case reserves (a unit test
  on RAA), and with every latest paid equal to the latest incurred the
  result is the IBNR-only one. The case reserves are the same in every
  simulation, so they move the mean, not the spread, of the undiscounted
  total (decision 18, the user's of 2026-11-15).[^note][^code]
* **Cells against spreading for `Compartmental`.** On company 337 at 5%
  mid-period (quick sampler, Windows) the discounted total from the paid
  cells has mean 125,357 and sd 5,791, and spread by the fit's own
  (population) pattern 125,450 and 5,780: the paired difference, -92 with a
  standard error of 16, is systematic but 0.07% of the mean, and the
  process timing of the cells barely changes the spread (decision 19; the
  test asserts 0.5% and 2%, not the digits).[^note]
* **Confidence level.** `RA = Q_p(X) - BE`, with the inverted-cdf quantile
  (`prospicio_prob::risk::var_sorted`, numpy's `inverted_cdf`, R's
  `type = 1`): the smallest draw with `F(x) >= p`. Since `F(Q(p)) >= p`,
  the implied confidence of `ConfidenceLevel(p)` is at least `p` and below
  `p + 1/n`; a unit test checks it on 997 draws at five levels.[^code]
* **TVaR.** `RA = TVaR_p(X) - BE`, the exact TVaR of the empirical
  distribution (`tvar_sorted`: the draw at the VaR counted with the share
  of its mass above `p`), not the mean of the whole draws above the
  quantile.[^code]
* **Cost of capital.** `K_0` is a capital the caller names: `Q_q(X) - BE`,
  `TVaR_q(X) - BE` or an amount (no default, since the lifetime 99.5% VaR is
  larger than Solvency II's one-year capital and neither is standard under
  IFRS 17). Each later year's capital follows the best estimate
  outstanding, `K_t = K_0 BE(t) / BE(0)`, where `BE(t)` is the mean
  discounted outstanding after `t` valued at `t`: the proportional run-off
  that EIOPA's guidelines list as method 2 of their simplifications for the
  risk margin (as the design note describes it; not checked against the
  guidelines' text).[^note]
  * End of year (the default): `RA = rate Σ_t K_t v(t + 1)`, the shape of
    the Solvency II risk margin (the note cites Delegated Regulation
    2015/35 art. 37; not checked against its text).
  * Start of year: `RA = rate Σ_t K_t v(t)`, `price_over_payout`'s
    convention in `prospicio-pricing` (year 0 charged in full).[^timevalue]
  * On a flat curve the two differ by exactly one year's discount, `v(1)`;
    the unit test checks both against hand-built sums from the run-off.[^code]
* **Implied confidence level.** Every result reports
  `#{X_s <= BE + RA} / n`, the confidence level the risk adjustment
  corresponds to, which IFRS 17 asks an entity to disclose when it uses
  another technique (the note cites para. 119; not checked against the
  standard's text). For the confidence level and TVaR the draws are counted
  against the quantile or TVaR itself, so rounding in `BE + RA` cannot drop
  a draw tied with it.[^code]
* **Segments.** For a fit over segments the risk adjustment of the joint
  total keeps the fit's dependence between segments; each segment's
  stand-alone result is reported beside it and the total is not allocated.
  A capital given as an amount is the total's, so no stand-alone results
  are reported for it.[^note][^code]
* **Not built.** The Solvency II risk margin's lambda and floor proposed
  for 2027 (the note mentions Delegated Regulation 2026/269; its formula
  has not been verified and is not implemented), Meyers' chapter 11 risk
  margin, ULAE and expense risk, reinsurance held and an illiquidity
  premium other than through the curve.[^note]

[^note]: docs/design/reserving-ifrs17.md
[^code]: crates/prospicio-reserving/src/discount.rs
[^timevalue]: docs/design/time-value.md
