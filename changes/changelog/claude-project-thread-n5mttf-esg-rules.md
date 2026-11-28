### Aggregate and reinsurance

- The multi-year projection (`prospicio_aggregate::dfa`) takes an AR(2)
  underwriting cycle on the premium's rate adequacy (`Cycle`, with its
  period and moments), tax at a flat rate with losses carried forward or
  not (`Tax`), and dividends above or capital injections up to a ratio of
  premium or reserves (`CapitalRule`); ruin and breach are read on the
  surplus before dividends and injections. Rust only for now (#983).
