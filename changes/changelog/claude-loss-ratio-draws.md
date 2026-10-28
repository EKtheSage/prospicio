### Reserving

- Ultimates and loss ratios per simulation from every fit with reserve draws, for towers, deal books and the cash-flow engine (`UltimateDraws`, `reserving-v02.md` decision 26): `ultimates()` (the reserves with each origin's latest value added, the same components and rows), `ultimate_draws(origin)`, `loss_ratio_draws(origin, premium)`, `total_ultimate_draws()` and `total_loss_ratio_draws(premium)` on `OdpBootstrapFit(s)`, `MackBootstrapFit(s)`, `BayesOdpFit(s)`, `BayesBfFit`, `BayesBfNormalisedFit`, `MeyersFit(s)` and `CompartmentalFit(s)`. Index `i` is the same simulation for every origin, every call and every fit with the same seed. Meyers' ultimates are its own simulated draws; a compartmental model's are on the paid basis. An origin is named by its component key (the segment's key values first over segments). Python `fit.ultimates`, `fit.ultimate_draws(origin=None)` and `fit.loss_ratio_draws(origin, premium)`; R `ultimate_draws(fit, origin = NULL)`, `loss_ratio_draws(fit, origin = NULL, premium)` and `ultimate_distribution(fit)`; `None` or `NULL` gives the total (#602).

### Distributions and probability

- The docs of `PredictiveDistribution`, `Tower::apply_annual`, `DealBook` and `CashflowTerms::run` state the simulation order they keep, so vectors per simulation from one fit stay paired through a tower, a deal book and the cash flows (#602).
