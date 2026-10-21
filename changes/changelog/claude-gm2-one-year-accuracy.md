### Reserving

- `BayesOneYear::light_refit` (`LightRefit`; Python `light_refit=` and R `light_refit =` on the `Compartmental` one-year functions): refit the one-year view's flagged simulations with a shorter NUTS run scaled by the opening posterior. On company 337 model 2's totals come within about 0.03 sd of full refits at a thirtieth of their cost (2,000 simulations in about 31 minutes). Off by default; model 2 still reweights only unless the setting is given (#394).
