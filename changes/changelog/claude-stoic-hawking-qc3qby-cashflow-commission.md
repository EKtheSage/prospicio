### Breaking changes

- Rust: `PremiumFlow` gains the public field `deduct_commission`; code
  that builds one in full adds it (`false` keeps the old behaviour).

### Pricing

- The cash-flow engine takes each simulation's commission rate off the
  premium paid into the trust under `deduct_commission` (off by default),
  so a sliding-scale deal runs in one call (#609). Python
  `CashflowTerms(..., deduct_commission=True)`, R
  `cashflow_terms(..., deduct_commission = TRUE)`.
- `Cashflows::paid_first` flags simulations whose investor is paid before
  putting money in, as a requirement on the reserve alone does; their IRR
  is usually meaningless (#608). Python `Cashflows.paid_first`, R
  `cashflow_run()$paid_first`; the reserve base's docs say a
  collateralised fund usually wants the outstanding loss.
