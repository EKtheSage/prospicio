### Pricing

- `earned`: mid-term changes (`Policy::endorsed`, a new full-term amount
  from a date) and cancellations (`Policy::cancelled`), each written when
  it happens, and two more earning methods, `EarningMethod::HalfMonth` (the
  ratemaking exhibits' grid) and `TwentyFourths` (the "15th of the month"
  rule). `Policy::amount_at` and `in_force` give the full-term amount in
  effect. Rust only for now (#981).
