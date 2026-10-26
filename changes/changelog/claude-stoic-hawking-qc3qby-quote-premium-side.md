### Breaking changes

- Rust: `Loads` gains the public fields `premium_tax` and `income_tax`, and
  `Quote` gains `terms` and `other_commission`; code that builds either in
  full adds them (`0.0`, `PremiumTerms::none()`).
- `Quote`'s `return_on_capital` is after income tax; it is unchanged
  without one.

### Pricing

- The quote's premium side and taxes: an expected reinstatement premium
  per unit of deposit (read off a tower grid with
  `PremiumTerms::from_layer`), brokerage on it or not, and a flat ceding
  commission, solved for in `Quote::technical_with`; sliding-scale and
  profit commissions and a swing's expected premium read back at a quoted
  deposit with `Quote::at_with_commission`. A premium tax is grossed up as
  brokerage is; an income tax on the margin keeps the price's return on
  capital after tax (`pre_tax_return_on_capital` beside it). New amounts
  `reinstatement_premium`, `premium_in_all`, `premium_tax`, `commission`
  and `income_tax`. Python `Quote.technical(..., premium_tax=,
  income_tax=, reinstatement_rate=, commission=,
  brokerage_on_reinstatements=)` and `Quote.at(premium, commission=)`; R
  `technical_quote()` with the same arguments and `implied_quote(quote,
  premium, commission)`.
