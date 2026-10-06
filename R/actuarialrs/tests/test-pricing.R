suppressMessages(library(actuarialrs))

# Reports the values on failure, so a CI log shows what diverged.
near <- function(a, b, rel = 1e-12) {
  if (!all(abs(a - b) <= rel * pmax(abs(b), 1e-300))) {
    stop(sprintf("got %s, want %s (rel %g)", format(a, digits = 17), format(b, digits = 17), rel),
         call. = FALSE)
  }
}

# Collective model.
sev <- pareto(1e6, 2)
m <- collective_model(claim_count(2, 1.5), sev)
near(layer(m, 4e6, 1e6), 1.6e6)
near(excess_frequency(m, 2e6), 0.5)
near(layer_variance(m, 4e6, 1e6), 2 * (layer_variance(sev, 4e6, 1e6) + layer(sev, 4e6, 1e6)^2) +
       (3 - 2) * layer(sev, 4e6, 1e6)^2)
stopifnot(is.infinite(variance(m)))
stopifnot(collective_simulate(m, 50, seed = 3)@n_sims == 50)

# Layer rating.
p <- pareto(100, 2)
near(ilf(p, 1000, 200), lev(p, 1000) / lev(p, 200))
near(loss_elimination_ratio(p, 150), lev(p, 150) / mean(p))
r <- pareto_extrapolation(c(1000, 1000), c(3000, 2000), 1.7)
near(alpha_between_layers(c(1000, 1000, 1), c(3000, 2000, r)), 1.7, 1e-11)
near(alpha_between_frequency_and_layer(1e6, 2, 1e6, 1e6, 1e6), 2, 1e-11)
near(alpha_between_frequencies(1e6, 4, 2e6, 1), 2)
stopifnot(inherits(try(alpha_between_layers(c(1000, 1000, 1), c(1000, 2000, 1)), silent = TRUE),
                   "try-error"))

# Tower matching: Riegel (2018), Table 3 and the first pieces of Table 4.
a <- c(1000, 1500, 2000, 2500, 3000, 5000, 10000)
e <- c(100, 90, 50, 40, 100, 50, 50)
tm <- match_tower(a, e, frequencies = c(0.25, rep(NA, 6)))
for (i in seq_along(a)) {
  limit <- if (i < length(a)) a[i + 1] - a[i] else Inf
  near(layer(tm, limit, a[i]), e[i], 1e-10)
}
stopifnot(all(abs(tm@severity@alpha[1:3] - c(2.374, 0.199, 0.175)) < 5e-4))
near(layer(match_tower(a, e, rule = "midpoint"), 500, 1500), 90, 1e-10)

# PML curve and references.
pml <- fit_pml_curve(c(5, 20, 50), c(2e6, 5e6, 9e6), tail_alpha = 1.8)
near(excess_frequency(pml, 5e6), 1 / 20)
fr <- fit_references(
  layers = data.frame(limit = c(1000, 1500), attachment = c(1000, 1500), expected_loss = c(120, 110)),
  frequencies = data.frame(threshold = 2500, frequency = 0.05)
)
near(layer(fr, 1500, 1500), 110, 1e-11)
near(excess_frequency(fr, 2500), 0.05, 1e-11)


# Risk-loaded prices from simulated losses.
pd <- predictive_distribution(
  matrix(c(0, 1, 4, 8, 2, 1, 0, 0), ncol = 2),
  data.frame(cover = c("a", "b"))
)
assets <- distortion("tvar", 0.5)
p <- price_portfolio(pd, assets, cost_of_capital = 0.1)
stopifnot(
  isTRUE(all.equal(sum(p$by_component$premium), p$total$premium)),
  isTRUE(all.equal(p$by_component$return_on_capital, c(0.1, 0.1))),
  isTRUE(all.equal(p$by_component$premium, c(3.5, 0.75 / 1.1))),
  p$diversification > 0
)
one <- risk_loaded_price(sampled(c(0, 0, 2, 6)), assets, cost_of_capital = 0.25)
stopifnot(isTRUE(all.equal(one$premium, 2.4)), isTRUE(all.equal(one$return_on_capital, 0.25)))
w <- risk_loaded_price(pd, assets, distortion = distortion("wang", 0.2))
stopifnot(w$expected_loss < w$premium, w$premium < w$assets)
stopifnot(inherits(try(risk_loaded_price(pd, assets), silent = TRUE), "try-error"))

# MBBEFD exposure curves.
c3 <- swiss_re_curve(3)
g3 <- exposure_curve(c3, c(0, 0.5, 1))
stopifnot(abs(g3[1]) < 1e-15, abs(g3[3] - 1) < 1e-12, g3[2] > 0.5)
stopifnot(abs(exposure_layer_share(c3, 5e6, 5e6, 1e7) + exposure_layer_share(c3, 5e6, 0, 1e7) - 1) < 1e-12)
m <- mbbefd(0.5, 4)
stopifnot(m@total_loss_probability == 0.25, m@mean > 0, m@mean < 1)
stopifnot(inherits(try(mbbefd(-1, 2), silent = TRUE), "try-error"))
sg <- severity_exposure_curve(pareto(1e5, 1.5), 1e7, c(0, 1))
stopifnot(sg[1] == 0, abs(sg[2] - 1) < 1e-12)

cat("actuarialrs R pricing tests passed\n")

# Tabulated exposure curves and destruction-rate quantiles.
tc <- tabulated_curve(c(0, 0.1, 0.5, 1), c(0, 0.4, 0.8, 1))
stopifnot(abs(exposure_curve(tc, 0.3) - 0.6) < 1e-15, abs(tc@mean - 0.25) < 1e-15)
stopifnot(identical(rate_quantile(tc, c(0.75, 0.76, 0.91)), c(0.1, 0.5, 1)))
stopifnot(abs(exposure_layer_share(tc, 5e6, 5e6, 10e6) - 0.2) < 1e-12)
stopifnot(inherits(try(tabulated_curve(c(0, 0.5, 1), c(0, 0.3, 1)), silent = TRUE), "try-error"))
c3 <- swiss_re_curve(3)
u <- (seq_len(100000) - 0.5) / 100000
stopifnot(abs(mean(rate_quantile(c3, u)) / c3@mean - 1) < 1e-3)
