suppressMessages(library(prospicio))

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

cat("prospicio R pricing tests passed\n")

# Tabulated exposure curves and destruction-rate quantiles.
tc <- tabulated_curve(c(0, 0.1, 0.5, 1), c(0, 0.4, 0.8, 1))
stopifnot(abs(exposure_curve(tc, 0.3) - 0.6) < 1e-15, abs(tc@mean - 0.25) < 1e-15)
stopifnot(identical(rate_quantile(tc, c(0.75, 0.76, 0.91)), c(0.1, 0.5, 1)))
stopifnot(abs(exposure_layer_share(tc, 5e6, 5e6, 10e6) - 0.2) < 1e-12)
stopifnot(inherits(try(tabulated_curve(c(0, 0.5, 1), c(0, 0.3, 1)), silent = TRUE), "try-error"))
c3 <- swiss_re_curve(3)
u <- (seq_len(100000) - 0.5) / 100000
stopifnot(abs(mean(rate_quantile(c3, u)) / c3@mean - 1) < 1e-3)

# Risk profile: simulation against exposure rating, surplus inuring to a per-risk XL.
rp <- risk_profile(c(0.5e6, 3e6, 20e6), c(2000, 300, 20),
                   list(swiss_re_curve(2), swiss_re_curve(3),
                        tabulated_curve(c(0, 0.02, 0.2, 1), c(0, 0.3, 0.8, 1))),
                   expected_loss = c(0.6e6, 0.5e6, 0.4e6))
stopifnot(abs(rp@expected_loss - 1.5e6) < 1e-6)
ev <- profile_simulate(rp, 50000, seed = 11)
tw <- inuring_tower(list(list(surplus_treaty("surplus", 1e6, 5)), list(xol_layer("xl", 1e6, 0.5e6))))
dm <- draw_matrix(apply_tower(tw, ev))
for (k in 2:3) {
  want <- if (k == 2) profile_surplus_loss(rp, 1e6, 5) else profile_layer_loss(rp, 1e6, 0.5e6, 1e6, 5)
  stopifnot(abs(mean(dm[, k]) - want) < 4 * sd(dm[, k]) / sqrt(50000))
}
rq <- risk_profile(c(1e6, 10e6), c(800, 50), swiss_re_curve(3), premium = c(2e6, 1e6), loss_ratio = 0.6)
stopifnot(abs(rq@expected_loss - 1.8e6) < 1e-6, abs(profile_layer_loss(rq, Inf, 0) - 1.8e6) < 1e-6)
stopifnot(inherits(try(risk_profile(1e6, 1, swiss_re_curve(3), premium = 1), silent = TRUE), "try-error"))

# Sums insured spread between bounds: a 2m retention cedes 3/8 of a band
# running from 1m to 5m, against 1/3 for its 3m mean risk.
rb <- risk_profile(c(0.5e6, 3e6), c(2000, 400), swiss_re_curve(3), expected_loss = c(0.6e6, 1.2e6),
                   lower = c(NA, 1e6), upper = c(NA, 5e6))
stopifnot(abs(profile_surplus_loss(rb, 2e6, 4) / 0.45e6 - 1) < 1e-9)
sb <- draw_matrix(apply_tower(inuring_tower(list(list(surplus_treaty("S", 2e6, 4)))),
                              profile_simulate(rb, 50000, seed = 3)))[, 2]
stopifnot(abs(mean(sb) - 0.45e6) < 4 * sd(sb) / sqrt(50000))
stopifnot(inherits(try(risk_profile(3e6, 1, swiss_re_curve(3), expected_loss = 1, lower = 1e6),
                       silent = TRUE), "try-error"))
stopifnot(inherits(try(risk_profile(3e6, 1, swiss_re_curve(3), expected_loss = 1, lower = 5e6,
                                    upper = 1e6), silent = TRUE), "try-error"))

# A tilted spread keeps the band's mean sum insured (2m, not the midpoint).
rt <- risk_profile(2e6, 400, swiss_re_curve(3), expected_loss = 1.2e6, lower = 1e6, upper = 5e6,
                   spread = "tilted")
rp2 <- risk_profile(2e6, 400, swiss_re_curve(3), expected_loss = 1.2e6)
stopifnot(abs(rt@expected_claims / rp2@expected_claims - 1) < 1e-12)
ct <- profile_surplus_loss(rt, 2e6, 4)
stopifnot(ct < 0.45e6)
st <- draw_matrix(apply_tower(reinsurance_tower(list(surplus_treaty("S", 2e6, 4))),
                              profile_simulate(rt, 50000, seed = 3)))[, 2]
stopifnot(abs(mean(st) - ct) < 4 * sd(st) / sqrt(50000))
stopifnot(inherits(try(risk_profile(5e6, 1, swiss_re_curve(3), expected_loss = 1, lower = 1e6,
                                    upper = 5e6, spread = "tilted"), silent = TRUE), "try-error"))

# Natural allocation on Monograph 15's InsCo, against aggregate 1.0.1.
insco <- capital_portfolio(cbind(
  A = c(15, 15, 5, 7, 13, 5, 15, 26, 17, 16),
  B = c(7, 13, 20, 33, 20, 27, 16, 19, 8, 20),
  C = c(0, 0, 11, 0, 7, 8, 9, 10, 40, 64)
))
stopifnot(identical(insco@totals, c(22, 28, 36, 40, 55, 65, 100)))
np <- natural_price(insco, distortion("ccoc", 0.15))
stopifnot(abs(np$premium[4] - 53.565217391304344) < 1e-12, max(abs(np$assets[1:3] - c(16, 20, 64))) < 1e-12)
ph <- calibrate_portfolio(insco, "proportional_hazard", return_on_capital = 0.15)
lifted <- natural_price(insco, ph, p = 0.85, allocation = "lifted")
stopifnot(max(abs(lifted$capital[1:3] - c(1.947189207380, -0.070344428313, 16.219694433834))) < 1e-6)
stopifnot(abs(sum(lifted$premium[1:3]) - lifted$premium[4]) < 1e-10)
stopifnot(abs(sum(bodoff_allocation(insco, assets = 100)) - 100) < 1e-10)
stopifnot(abs(epd_ratio(insco, 65)$total - 3.5 / 46.6) < 1e-15, abs(assets_for_epd(insco, 3.5 / 46.6) - 65) < 1e-9)
pent <- pentagon(loss = 46.6, assets = 100, return_on_capital = 0.15)
stopifnot(abs(pent[["premium"]] - 53.565217391304344) < 1e-12)
stopifnot(inherits(try(pentagon(loss = 1, premium = 2), silent = TRUE), "try-error"))
# Independent units by FFT: PIR's Discrete case.
g1 <- grid_distribution(1, replace(numeric(11), c(1, 9, 11), c(0.5, 0.25, 0.25)))
g2 <- grid_distribution(1, replace(numeric(91), c(1, 2, 91), c(0.5, 0.25, 0.25)))
disc <- capital_portfolio(list(X1 = g1, X2 = g2))
stopifnot(identical(disc@units, c("X1", "X2")), abs(sum(disc@expected) - 27.25) < 1e-12)
# Pricing bounds and classical principles on InsCo, against aggregate 1.0.1.
pb <- premium_bounds(insco, 53.565217391304344, assets = 100)
stopifnot(max(abs(pb$lower - c(13.09782608695652, 17.465726050623715, 19.25473801560758))) < 1e-9)
stopifnot(max(abs(pb$upper - c(15.032744226390545, 20.411684782608695, 22.098038028339595))) < 1e-9)
stopifnot(S7::S7_inherits(pb$lower_distortion[[1]], distortion))
xs <- sampled(c(22, 28, 36, 40, 40, 40, 40, 55, 65, 100))
stopifnot(abs(calibrate_classical(xs, "esscher", 53.565217391304344) - 0.012851355964986997) < 1e-9)
stopifnot(abs(classical_premium(sampled(c(0, 10)), "standard_deviation", 0.2) - 6) < 1e-15)
