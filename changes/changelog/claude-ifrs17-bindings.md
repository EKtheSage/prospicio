### Reserving

- Discounted reserves and the IFRS 17 risk adjustment in Python and R
  (#553): `fit.discounted(basis=..., rate=... | curve=DiscountCurve(...))`
  on the stochastic reserving fits and `DiscountedReserves.risk_adjustment`
  by a confidence level, TVaR or a cost of capital, each with its implied
  confidence level; R `discounted_reserves()`, `risk_adjustment()` and
  `discount_curve()`. The basis (`"paid"` or `"incurred"`) is required.
- A fit over one segment (a triangle without keys) no longer reports that
  segment beside the total in `discounted_reserves`' segment labels and
  risk adjustments (#553, decision 25).

### Guides

- `reserving-methods.md` section 11: the discounted reserve at a flat rate
  and on a spot curve, the IFRS 17 risk adjustment and its implied
  confidence level, and the one-year view's capital (its 99.5% loss less the
  expected loss) as the capital of a cost-of-capital risk margin (#553).
