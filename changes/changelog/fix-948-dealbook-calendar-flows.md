### Pricing

- `DealBook::calendar` no longer panics on a book where some deals have
  investor flows and others do not; the table then has no `investor_flow`
  item, as `cashflows` already requires every deal's flows (#948, #1013).
