# Findings

* [Normal draws bias the log-link mean](odp-normal-draws-bias.md) - Drawing beta from N(beta_hat, Sigma) and exponentiating raises each mean by exp(x'Sigma x / 2); MeanPreserving draws remove it exactly.
* [Negative increments in the ODP](odp-negative-increments.md) - A quasi-likelihood ODP needs only V(mu) = mu and mu > 0; negative responses are fitted with a quasi-deviance that is not a distance.
* [ODP log density](odp-continuous-density.md) - The ODP's lattice probability is -inf off the lattice; its continuation through Gamma, normalized by C(lambda), is a proper density in y.
* [Stacking pooling funnel](stacking-pooling-funnel.md) - With one slope per pooled group, the group scale and the slope trade off; NUTS diverges. Three or more covariates mix well.
* [R callbacks must stay on R's main thread](r-callbacks-main-thread.md) - An R function held inside a Rust object may be called only from R's thread; parallel code that meets one runs in order on the calling thread.
* [Reinstatements pro rata as to time](reinstatement-pro-rata-time.md) - Each event's used limit is charged at the share of the year left after it; sorted uniform times in drawn order date i.i.d. losses exactly; one exhausting loss a year costs half the amount-only premium.
* [Bondy least squares](bondy-least-squares-stop.md) - TailBondy fits b with scipy least_squares at default tolerances, which stop on a 1e-8 relative cost change; b is off by up to 4e-6, the tail's results by up to 3e-5.
* [The ODP one-year bootstrap against Merz-Wuthrich](one-year-bootstrap-vs-merz-wuthrich.md) - Re-reserving on the ODP bootstrap, projecting from the pseudo latest value, gives 0.50 to 5.98 times Merz-Wuthrich's one-year SD per origin; the ODP's process variance is the gap, and with Mack's bootstrap the same re-reserving matches within 0.6% (a CI unit test).
