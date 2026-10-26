### Breaking changes

- `DealRun::from_tower` (Python `DealRun.from_tower`, R
  `deal_run_from_tower()`) now adds each year's reinstatement premium to the
  deposit premium; pass `reinstatement_premium=False` (Rust
  `from_tower_with(..., false)`) for the old premium. Rust: `DealColumns`
  gains the public field `reinstatement_premium`, which `from_columns` adds
  to the premium; code that builds one in full adds it (#595).

### Pricing

- A deal book counts the reinstatement premium: `DealRun::from_tower` adds
  each year's reinstatement premium from the tower, pro rata as to amount
  and, where the layer says so, as to time, to the deposit or swing-rated
  premium, so the book's premium, margin and returns include it. Python
  `DealRun.premium` and `DealRun.loss` read a run back (#595).
