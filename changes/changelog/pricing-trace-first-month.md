### Pricing

- **Breaking**: `Trace` is `#[non_exhaustive]`, so Rust code outside the crate can no longer build one with a struct literal, and a later field is not a break. `Trace` has a new public field, `first_month` (0 from inception, the valuation month from a valuation), and `ActualState::from_trace` refuses every trace from a valuation by it, including one whose trust and operating balances were both exactly 0 (#960, #1018, #1043).
