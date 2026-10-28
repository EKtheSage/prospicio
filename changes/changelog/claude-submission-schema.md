### Guides

- `docs/guides/submission-schema.md`, the submission's files column by column (type, unit, meaning,
  which calls read them), a `check_submission` function with the checks the examples rely on, and the
  tidy tables a deal analysis writes for an investor report (`deal.csv`, `structures.csv`,
  `metrics.csv`, `actual_vs_expected.csv`), made from the sample submission for a quota share in two
  structures. A test fails when a file's header and the guide disagree (#605, #669).

### Examples

- `examples/templates/deal_report.qmd` (R) and `deal_report_py.qmd` (Python), a Quarto report
  template for an investor or committee: the structures, the loss ratio, ceded and net, the
  investor's IRR and MOIC with the probability of loss, the structures side by side and actual
  against expected. Parameters in the header and no data inside; CI renders both on
  `examples/data/deal_report/` (#605, #669).
