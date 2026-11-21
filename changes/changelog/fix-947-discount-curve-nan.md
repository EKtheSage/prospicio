### Reserving

- `DiscountCurve.rate` and `factor` (Python, R and Rust) return NaN for a NaN time instead of panicking; a negative time still accumulates at the short rate, now documented (#947, #1006).
