### Pricing

- `monthly_patterns(..., accident_period="year")` (R
  `cashflow_patterns_from_annual(..., accident_period = "year")`) reads an
  accident-year, quarter or month payment pattern from the start of its
  period and converts it to the deal's underwriting year with the deal's
  own writing and policy term, with `reading=` as on `UnderwritingBasis`;
  `MonthlyPatterns::from_paid` takes such a converted paid pattern in Rust
  (#722).
