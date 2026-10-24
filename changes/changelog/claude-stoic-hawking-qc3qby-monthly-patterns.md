### Pricing

- Monthly patterns for the cash-flow engine from a payment pattern by
  development age (#601): `PaymentPattern::paid_by` and `to_monthly` read
  it at each month end, as a step, linearly or log-linearly between its
  times, and `MonthlyPatterns::from_annual` builds the written, earned and
  paid patterns from a written pattern and a policy term, losses paid from
  the accident month. Python `PaymentPattern.paid_by`, `to_monthly` and
  `monthly_patterns`; R `payment_pattern_monthly()` and
  `cashflow_patterns_from_annual()`.
