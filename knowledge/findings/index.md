# Findings

* [Normal draws bias the log-link mean](odp-normal-draws-bias.md) - Drawing beta from N(beta_hat, Sigma) and exponentiating raises each mean by exp(x'Sigma x / 2); MeanPreserving draws remove it exactly.
* [Negative increments in the ODP](odp-negative-increments.md) - A quasi-likelihood ODP needs only V(mu) = mu and mu > 0; negative responses are fitted with a quasi-deviance that is not a distance.
* [ODP log density](odp-continuous-density.md) - The ODP's lattice probability is -inf off the lattice; its continuation through Gamma, normalized by C(lambda), is a proper density in y.
* [Stacking pooling funnel](stacking-pooling-funnel.md) - With one slope per pooled group, the group scale and the slope trade off; NUTS diverges. Three or more covariates mix well.
