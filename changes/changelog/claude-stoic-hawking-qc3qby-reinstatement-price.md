### Pricing

- The upfront premium of a cover whose reinstatement premium counts in
  its own price (#587): `risk_load::price_with_reinstatements` solves
  `P = price(X − P g)`, with `g` each simulation's reinstatement premium
  per unit of upfront premium, under a cost of capital or a distortion.
  It returns the upfront premium with the expected reinstatement premium,
  the net outcome's assets and the bisection steps taken. Python
  `price_with_reinstatements` (`ReinstatementPrice`), R
  `price_with_reinstatements()`.
