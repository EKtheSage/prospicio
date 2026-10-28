### Reserving

- **Breaking:** an annual accident-year pattern converted to an underwriting basis
  (`SelectedPattern::to_underwriting_basis`, Python and R `ldf_underwriting`) is now read
  between its ages as the Weibull fitted through the yearly averages and corrected to the
  nearest monotone accident-month pattern whose averages are the given values exactly
  (`reserving-v02.md` decision 25). On a simulated parallelogram book the converted share is
  within 0.2% of the truth at every age, where the previous reading was 31% too far developed at
  12 months and 8% short at 24. Converted patterns from accident periods longer than a month
  given at ages that are not every month from 1 change, and now start at 1 month; accident-month
  patterns and patterns given at every month are unchanged. The previous reading stays
  available: Rust `UnderwritingBasis::default().with_reading(AnnualReading::LnLinear)`, Python
  `UnderwritingBasis(reading="ln-linear")`, R `underwriting_basis(reading = "ln-linear")`, with
  `"loglogistic"` as a third choice. Accident-year patterns given at quarterly or semiannual
  ages read the same way, including patterns still developing at their last age. A single
  given value is now an error under the curve readings (#646, #650).
- **Breaking:** `UnderwritingBasis` has a private field for the reading, so struct literals
  `UnderwritingBasis { writing, policy_months }` no longer compile: use `UnderwritingBasis::new`
  or `UnderwritingBasis::even` (#650).
