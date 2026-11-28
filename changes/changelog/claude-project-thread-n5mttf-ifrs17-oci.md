### Reserving

- `ifrs17::Group` measures the present value at a current discount curve
  given each period (`PeriodInputs::with_current_curve`), with the changes
  that adjust the CSM at the locked-in curve and the difference in
  insurance finance income or expense; a change in financial assumptions is
  a revision of the opening view (`with_financial_revision`), a finance
  effect. `GroupOptions::oci` disaggregates the finance expense between
  profit or loss (at the locked-in curve) and OCI, and
  `GroupOptions::accretion_curve` accretes the CSM at the rates for nominal
  cash flows. IFRS 17 Example 6 is matched to the unit (#982). Breaking for
  struct literals of `Balances` and `GroupOptions` (new fields; use
  `..Default::default()`); unreleased since 0.3.0.
