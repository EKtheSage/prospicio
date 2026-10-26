### Reserving

- An accident-period development pattern converts to an underwriting-year
  basis from a writing pattern and a policy term:
  `SelectedPattern::to_underwriting_basis` and `UnderwritingBasis` in Rust,
  `UnderwritingBasis` and `ldf_underwriting=` beside `ldf_origin` on every
  method that takes `ldf` in Python, and `underwriting_basis()` with
  `ldf_underwriting` in R. The underwriting year's share developed is the
  accident pattern convolved with its exposure over accident months; the
  default is even writing over 12 months and 12-month policies, the
  parallelogram. In Python and R the writing must be no longer than the
  triangle's origin period (`UnderwritingBasis::check_within` in Rust),
  so underwriting quarters need three months of writing (#633, #639).
