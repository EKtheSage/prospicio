### Reserving

- Discounted reserves and an IFRS 17 risk adjustment (`prospicio_reserving::discount`, #553, Rust):
  `discounted_reserves` discounts each simulation of a bootstrap or Bayesian fit's reserves (and of
  their fits over segments) on a `DiscountCurve` (flat or spot rates), spreading each reserve over
  that simulation's pattern (`Compartmental`: its paid cells) at mid-period or end-of-period timing,
  on a stated `PaymentBasis` (under `Incurred { paid_pattern, latest_paid }`, the whole unpaid, each
  reserve plus its origin's case reserve, spread along the paid pattern, as `DiscountedReserves::unpaid`);
  `DiscountedReserves::risk_adjustment` by a confidence level, TVaR or a
  cost of capital (charged at the end or the start of each year), with the implied confidence level
  and each segment's stand-alone result.
