### Pricing

- `monthly_patterns(..., accident_period="year")` (R
  `cashflow_patterns_from_annual(..., accident_period = "year")`) reads an
  accident-year, quarter or month payment pattern from the start of its
  period and converts it to the deal's underwriting year with the deal's
  own writing and policy term, with `reading=` as on `UnderwritingBasis`;
  `MonthlyPatterns::from_paid` takes such a converted paid pattern in Rust
  (#722, #745).

### Models

- Tabular networks with learned embeddings of rating factors and several
  claim heads on a shared body: `prospicio_nn::TabularNet` with `Head`,
  `Target` and `TabularNetFit` (`embeddings`, `embedding_design`, the
  `risk_rs.tabular_net_fit` artifact); one head without embeddings is
  `Cann` bit for bit. Python `TabularNet`, `Head`, `TabularNetFit`, R
  `tabular_net_fit()`, `net_head()`, `network_embeddings()`,
  `embedding_columns()` (#577, #749).
