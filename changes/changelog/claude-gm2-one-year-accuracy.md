### Reserving

- `BayesOneYear::light_refit` (`LightRefit`; Python `light_refit=` and R `light_refit =` on the `Compartmental` one-year functions): refit the one-year view's flagged simulations with a shorter NUTS run scaled by the opening posterior. On company 337 model 2's totals come within about 0.03 sd of full refits at about a thirtieth of their cost (2,000 simulations in about 31 minutes of wall time on 16 threads); model 1's refits with it were not measured. Off by default; model 2 still reweights only unless the setting is given (#394).
