### Aggregate and reinsurance

- The parity references against `aggregate` are regenerated with
  `aggregate` 2.1.0 and every script pins it; no prospicio result changes.
  The values that moved are fixes in `aggregate` toward prospicio's: the
  skewness it quotes for a layer, `PricingBounds` at `p -> 1` and below the
  top atom, and the lower envelope of `Bounds`. The tolerances on layer
  skewness are now 1e-8. The P&L ledger's annual aggregate deductible and
  limit on an occurrence cover are checked against `aggregate` 2.1.0 to
  1e-12 (#768).
