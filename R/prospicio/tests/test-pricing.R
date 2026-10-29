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
# Reinstatement premium in its own price (#587).
x <- c(0, 1, 0, 3, 8, 0.5, 12, 0)
plain <- risk_loaded_price(sampled(x), assets, cost_of_capital = 0.15)
cst <- price_with_reinstatements(x, rep(0.4, 8), assets, cost_of_capital = 0.15)
stopifnot(isTRUE(all.equal(cst$upfront, plain$premium / 1.4, tolerance = 1e-10)))
g <- pmin(x / 10, 1)
id <- price_with_reinstatements(x, g, assets, distortion = distortion("proportional_hazard", 1))
stopifnot(isTRUE(all.equal(id$upfront, mean(x) / (1 + mean(g)), tolerance = 1e-10)))
r <- price_with_reinstatements(x, g, assets, cost_of_capital = 0.15)
stopifnot(
  r$upfront < plain$premium,
  isTRUE(all.equal(r$return_on_capital, 0.15, tolerance = 1e-9)),
  isTRUE(all.equal(r$expected_premium, r$upfront * (1 + mean(g)))),
  r$iterations > 0
)

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


# Tabulated exposure curves and destruction-rate quantiles.
tc <- tabulated_curve(c(0, 0.1, 0.5, 1), c(0, 0.4, 0.8, 1))
stopifnot(abs(exposure_curve(tc, 0.3) - 0.6) < 1e-15, abs(tc@mean - 0.25) < 1e-15)
stopifnot(identical(rate_quantile(tc, c(0.75, 0.76, 0.91)), c(0.1, 0.5, 1)))
stopifnot(abs(exposure_layer_share(tc, 5e6, 5e6, 10e6) - 0.2) < 1e-12)
stopifnot(inherits(try(tabulated_curve(c(0, 0.5, 1), c(0, 0.3, 1)), silent = TRUE), "try-error"))
ll <- tabulated_curve(c(0, 0.1, 0.5, 1), c(0, 0.4, 0.8, 1), interpolation = "log-log")
stopifnot(abs(exposure_curve(ll, 0.3) - 0.4 * 3^(log(2) / log(5))) < 1e-15)
stopifnot(ll@interpolation == "log-log", tc@interpolation == "linear")
xs <- c(0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1)
stopifnot(inherits(try(tabulated_curve(xs, exposure_curve(swiss_re_curve(3), xs), "log-log"),
                       silent = TRUE), "try-error"))
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
# The book's conventions: Pricing Insurance Risk's Example 235 with the outcome
# at the assets lumped with the defaults, and InsCo with no capital in the
# bottom layer (Monograph 15's equation 8.5 as printed).
e235 <- capital_portfolio(cbind(X1 = c(1, 2, 3, 4, 5, 6, 7, 8, 12, 10),
                                X2 = c(1, 2, 3, 4, 5, 9, 8, 12, 13, 20)))
near(natural_price(e235, distortion("tvar", 0.8), assets = 10, atom = "lumped")$premium[1:2], c(4.3, 5.7))
near(natural_price(e235, distortion("tvar", 0.8), assets = 10)$premium[1], 4.16)
# The natural allocation set (Exercise 260) and Bodoff's layers priced by the
# CCoC (Table 15.24), as in Python.
rows <- as.matrix(expand.grid(X2 = c(0, 1, 90), X1 = c(0, 9, 10))[, c("X1", "X2")])
adjusted <- capital_portfolio(rows, probs = as.vector(outer(c(0.5, 0.25, 0.25), c(0.5, 0.25, 0.25))))
aset <- allocation_set(adjusted, distortion("tvar", 0.65), assets = 100)
stopifnot(identical(aset$unit, c("X1", "X2")),
          max(abs(c(aset$lower, aset$upper) - c(6.1429, 64.4643, 6.25, 64.5714))) < 5e-5)
bodoff <- capital_portfolio(rbind(c(Hu = 0, EQ = 0), c(99, 0), c(0, 100), c(99, 100)),
                            probs = c(0.76, 0.19, 0.04, 0.01))
plc <- bodoff_price(bodoff, distortion("ccoc", 0.1), assets = 100)
stopifnot(max(abs(plc$premium[1:2] - c(24.8729, 5.8635))) < 5e-5,
          max(abs(plc$assets[1:2] - bodoff_allocation(bodoff, assets = 100))) < 1e-12)
ccoc <- calibrate_portfolio(insco, "ccoc", return_on_capital = 0.15)
none <- natural_price(insco, ccoc, assets = 100, allocation = "lifted", bottom_layer = "none")
stopifnot(max(abs(none$capital[1:3] - c(5.419, 6.082, 34.934))) < 5e-4)
near(none$premium, natural_price(insco, ccoc, assets = 100, allocation = "lifted")$premium)
# Kappa and assets at p, against Python's Portfolio.kappa(unit) and assets(p).
near(sapply(c(0.8, 0.85, 0.9, 1), function(p) portfolio_assets(insco, p)), c(55, 65, 65, 100))
near(portfolio_kappa(insco, "A"), c(15, 15, 5, 10, 26, 17, 16))
near(portfolio_kappa(insco, "B"), c(7, 13, 20, 24, 19, 8, 20))
near(portfolio_kappa(insco, "C"), c(0, 0, 11, 6, 10, 40, 64))
near(portfolio_kappa(insco, 2), portfolio_kappa(insco, "B"))
near(portfolio_kappa(insco, "A") + portfolio_kappa(insco, "B") + portfolio_kappa(insco, "C"), insco@totals)
stopifnot(inherits(try(portfolio_kappa(insco, "D"), silent = TRUE), "try-error"))
stopifnot(inherits(try(portfolio_kappa(insco, 4), silent = TRUE), "try-error"))
stopifnot(inherits(try(portfolio_assets(insco, 1.5), silent = TRUE), "try-error"))
stopifnot(abs(epd_ratio(insco, 65)$total - 3.5 / 46.6) < 1e-15, abs(assets_for_epd(insco, 3.5 / 46.6) - 65) < 1e-9)
pent <- pentagon(loss = 46.6, assets = 100, return_on_capital = 0.15)
stopifnot(abs(pent[["premium"]] - 53.565217391304344) < 1e-12)
d <- pent[["discount"]]
stopifnot(abs(d - pent[["margin"]] / (100 - 46.6)) < 1e-15,
          abs((1 - d) * 46.6 + d * 100 - pent[["premium"]]) < 1e-12)
stopifnot(inherits(try(pentagon(loss = 1, premium = 2), silent = TRUE), "try-error"))
# Independent units by FFT: PIR's Discrete case.
g1 <- grid_distribution(1, replace(numeric(11), c(1, 9, 11), c(0.5, 0.25, 0.25)))
g2 <- grid_distribution(1, replace(numeric(91), c(1, 2, 91), c(0.5, 0.25, 0.25)))
disc <- capital_portfolio(list(X1 = g1, X2 = g2))
stopifnot(identical(disc@units, c("X1", "X2")), abs(sum(disc@expected) - 27.25) < 1e-12)
# The priority ladder at assets 90, against aggregate 1.0.1's priority_epd_df.
pl <- priority_ladder(disc, assets = 90)
stopifnot(identical(pl$unit, c("X1", "X2", "total")), attr(pl, "assets") == 90)
stopifnot(max(abs(pl$equal[1:2] - c(4.396684, 21.728316))) < 1e-6)
stopifnot(max(abs(pl$junior_epd[1:2] - c(0.25, 0.049451))) < 1e-6, abs(pl$senior[3] - 26.125) < 1e-12)
pk <- priority_kappa(disc, "X1", assets = 90)
stopifnot(max(abs(unlist(pk[pk$total == 98, c("junior", "senior")]) - c(8, 0))) < 1e-9)
stopifnot(inherits(try(priority_kappa(disc, "X3", assets = 90), silent = TRUE), "try-error"))
# Pricing bounds and classical principles on InsCo, against aggregate 1.0.1.
pb <- premium_bounds(insco, 53.565217391304344, assets = 100)
stopifnot(max(abs(pb$lower - c(13.09782608695652, 17.465726050623715, 19.25473801560758))) < 1e-9)
stopifnot(max(abs(pb$upper - c(15.032744226390545, 20.411684782608695, 22.098038028339595))) < 1e-9)
stopifnot(S7::S7_inherits(pb$lower_distortion[[1]], distortion))
xs <- sampled(c(22, 28, 36, 40, 40, 40, 40, 55, 65, 100))
stopifnot(abs(calibrate_classical(xs, "esscher", 53.565217391304344) - 0.012851355964986997) < 1e-9)
stopifnot(abs(classical_premium(sampled(c(0, 10)), "standard_deviation", 0.2) - 6) < 1e-15)
# Pricing bounds for another risk, against aggregate 1.0.1's PricingBounds.
xb <- sampled(c(22, 28, 36, 40, 40, 40, 40, 55, 65, 100))
yb <- sampled(c(0, 0, 11, 0, 7, 8, 9, 10, 40, 64))
pr <- pricing_bounds(xb, list(X3 = yb), 53.565217391304344)
stopifnot(identical(pr$risk, "X3"), abs(pr$upper - 22.098038028339587) < 1e-9, pr$lower < pr$upper)
stopifnot(S7::S7_inherits(pr$lower_distortion[[1]], distortion))
# At P = max X (capped at 65) TVaR_1 gives max Y; aggregate says 52.
stopifnot(abs(pricing_bounds(xb, list(X3 = yb), 65, assets = 65)$upper - 64) < 1e-12)
env <- distortion_envelope(xb, 53.565217391304344, c(0.1, 0.5))
stopifnot(max(abs(env$lower - c(0.1372282608695652, 0.5652173913043478))) < 1e-12)
stopifnot(inherits(try(pricing_bounds(xb, list(yb), 50), silent = TRUE), "try-error"))

# tvar_threshold on PIR's Discrete case, aggregate 1.0.1.
stopifnot(abs(tvar_threshold(disc, 0.9) - 0.859375) < 1e-11)
# Stand-alone prices and the comonotonic allocation, aggregate 1.0.1.
sa <- price_stand_alone(disc, distortion("proportional_hazard", 0.7), p = 0.9)
stopifnot(identical(sa$unit, c("X1", "X2", "total", "sum")),
          all(abs(sa$premium[c(1, 3, 4)] - c(5.682435936634864, 39.27576430346102,
                                              40.02270174816368)) < 1e-10))
co <- comonotonic_allocation(disc)
stopifnot(abs(co$X1[co$total == 8] - 6.607142857142858) < 1e-10,
          max(abs(co$X1 + co$X2 - co$total)) < 1e-12)

# A loss distribution on a grid prices as its equally likely draws (#204).
gp <- risk_loaded_price(grid_distribution(2, c(0.5, 0.25, 0, 0.25)), distortion("tvar", 0.6),
                        cost_of_capital = 0.1)
sp <- risk_loaded_price(sampled(c(0, 0, 2, 6)), distortion("tvar", 0.6), cost_of_capital = 0.1)
stopifnot(abs(gp$premium - sp$premium) < 1e-12, abs(gp$assets - sp$assets) < 1e-12)

# A quote grosses the risk premium up, and a quoted premium reads back (#197).
p <- risk_loaded_price(sampled(c(0, 0, 2, 6)), distortion("tvar", 0.5), cost_of_capital = 0.1)
q <- technical_quote(p, expenses = 0.03, brokerage = 0.10, limit = 10)
stopifnot(abs(q$premium * 0.87 - p$premium) < 1e-12, abs(q$return_on_capital - 0.1) < 1e-12,
          abs(q$rate_on_line - q$premium / 10) < 1e-15,
          abs(implied_quote(q, q$premium)$risk_premium - p$premium) < 1e-12,
          implied_quote(q, 3)$return_on_capital > 0.1,
          is.nan(technical_quote(p)$rate_on_line))

# The premium side and taxes: the parts add up to all the premium, and the
# return after income tax is the price's.
t <- technical_quote(p, expenses = 0.03, brokerage = 0.10, limit = 10, premium_tax = 0.02,
                     income_tax = 0.21, reinstatement_rate = 0.4, commission = 0.05)
stopifnot(abs(t$expected_loss + t$margin + t$expenses + t$brokerage + t$premium_tax +
                t$commission - t$premium_in_all) < 1e-12,
          abs(t$return_on_capital - 0.1) < 1e-12, t$pre_tax_return_on_capital > 0.1,
          abs(implied_quote(t, t$premium)$risk_premium - t$risk_premium) < 1e-12,
          abs(implied_quote(t, t$premium, commission = 0.01)$risk_premium -
                (t$risk_premium - 0.01)) < 1e-12,
          identical(technical_quote(p, expenses = 0.03, limit = 10)$premium,
                    technical_quote(p, expenses = 0.03, limit = 10, reinstatement_rate = 0)$premium))

# Experience rating from large losses reproduces Clark's study note (#190).
clark <- data.frame(
  id = as.character(1:14),
  date = as.Date(c("1988-09-20", "1988-10-11", "1989-03-15", "1990-06-21", "1990-10-24",
                   "1991-01-10", "1992-02-23", "1992-04-30", "1992-09-22", "1993-01-01",
                   "1993-05-18", "1993-08-01", "1994-08-15", "1995-07-12")),
  amount = c(240946, 821499, 158129, 114051, 78043, 162533, 324298, 100549, 75476, 171885,
             94218, 170297, 87133, 771249),
  factor = c(1.411, 1.408, 1.385, 1.317, 1.300, 1.289, 1.234, 1.225, 1.206, 1.193, 1.175,
             1.166, 1.119, 1.080)
)
clark$accident <- 1997.5 - as.numeric(as.Date("1997-07-01") - clark$date) / 365.25
tr <- trend_large_losses(clark, rate = 0.04, to = 1997.5)
stopifnot(all(abs(tr$losses$factor - clark$factor) <= 0.0005 + 1e-12))
pr <- trend_large_losses(clark, factors = clark$factor)
tw <- reinsurance_tower(list(xol_layer("400 xs 100", 4e5, 1e5)))
bc <- burning_cost(pr, tw, 1988:1995,
                   premium = c(1422554, 1823103, 2054034, 2147147, 2151541, 2159198, 2167158,
                               2187654),
                   development = c(1, 1, 1, 1, 1.01, 1.05, 1.15, 1.3))
stopifnot(all(abs(bc$year_loss_cost[, 1] -
                  c(0.450, 0.065, 0.025, 0.051, 0.152, 0.104, 0, 0.238)) <= 0.0005 + 1e-12),
          abs(bc$loss_cost - 0.124) <= 0.0005,
          identical(bc$years$year, 1988:1995))

# Losses developed one by one (IBNER) and a count pattern for the rest (#190).
dp <- development_pattern_from_history(
  data.frame(id = c("a", "a", "a", "b", "b", "c", "c"), age = c(1, 2, 3, 1, 2, 2, 3),
             amount = c(100, 180, 200, 100, 120, 500, 550)),
  ages = c(1, 2, 3)
)
stopifnot(abs(dp$factors[1] - 300 / 200 * 750 / 680) < 1e-14, dp$factors[3] == 1,
          abs(pattern_factor(dp, 2.5) - (dp$factors[2] + 1) / 2) < 1e-14)
ll <- data.frame(id = c("a", "b"), accident = c(2020.5, 2022.5), amount = c(90, 150))
pd <- trend_large_losses(ll, factors = c(1, 1), threshold = 100, valuation = 2024.5,
                         development = development_pattern(c(2, 4), c(2, 1.5)))
stopifnot(identical(pd$losses$ultimate, c(135, 300)))
bd <- burning_cost(pd, reinsurance_tower(list(xol_layer("x", 1000, 100))), c(2020, 2022),
                   premium = c(1000, 1000), development = development_pattern(c(1, 3), c(1.4, 1)))
stopifnot(identical(bd$years$losses, c(0L, 1L)), abs(bd$developed[2, 1] - 240) < 1e-12)
stopifnot(abs(trend_large_losses(ll, factors = c(1, 1), development = c(1, 1.1))$losses$ultimate[2]
              - 165) < 1e-12)

# An index clause moves each loss's terms; losses in another currency (#190).
ip <- trend_large_losses(data.frame(id = c("a", "b"), accident = c(2020.2, 2020.6),
                                    amount = c(700, 300)), factors = c(1, 1))
itw <- reinsurance_tower(list(xol_layer("400 xs 100", 400, 100)))
ib <- burning_cost(ip, itw, 2020, premium = 1000, development = 1,
                   index = index_clause(0.1), ratios = c(1.25, 1.05))
stopifnot(abs(ib$layer_loss[1, 1] - 700) < 1e-12,
          identical(burning_cost(ip, itw, 2020, premium = 1000, development = 1,
                                 index = index_clause(0.1, "severe"), ratios = c(1, 1))$loss_cost,
                    burning_cost(ip, itw, 2020, premium = 1000, development = 1)$loss_cost))
stopifnot(inherits(try(burning_cost(ip, itw, 2020, premium = 1000, development = 1,
                                    index = index_clause(0.1)), silent = TRUE), "try-error"))
annual <- reinsurance_tower(list(xol_layer("x", 400, 100, aggregate_limit = 800)))
stopifnot(inherits(try(burning_cost(ip, annual, 2020, premium = 1000, development = 1,
                                    index = index_clause(0.1), ratios = c(1, 1)), silent = TRUE),
                   "try-error"))
fxp <- trend_large_losses(data.frame(id = c("a", "b"), accident = c(2020.5, 2020.5),
                                     amount = c(90, 120)), factors = c(1, 1), threshold = 100,
                          fx = c(2, 0.5))
stopifnot(identical(fxp$losses$trended, c(180, 60)), identical(fxp$losses$fx, c(2, 0.5)))

# As-if history beside the proposed tower (#191).
ap <- trend_large_losses(data.frame(id = c("a", "b", "c"), accident = c(2019.3, 2019.7, 2020.5),
                                    amount = c(300, 80, 500)), factors = c(1.2, 1.2, 1.1),
                         threshold = 100)
then20 <- inuring_tower(list(list(quota_share("QS", 0.5)), list(xol_layer("200 xs 100", 200, 100))))
now <- reinsurance_tower(list(xol_layer("200 xs 200", 200, 200), xol_layer("600 xs 400", 600, 400)))
ai <- as_if_history(ap, list(reinsurance_tower(list(xol_layer("150 xs 50", 150, 50))), then20),
                    c(2019, 2020), now)
stopifnot(identical(ai$years$reported_losses, c(2L, 1L)),
          identical(unname(ai$historical[["2019"]]), 180),
          identical(ai$historical[["2020"]], c(QS = 250, `200 xs 100` = 150)),
          max(abs(ai$proposed["2020", ] - c(200, 150))) < 1e-12)

# Payment patterns and a price over the payout (#195).
pp <- payment_pattern(c(1, 2, 3, 4), rep(0.25, 4))
v <- 1 / 1.05
pv <- pattern_values(pp, rate = 0.05, t = 1)
stopifnot(abs(pv$discount_factor - (1 - v^4) / 0.05 / 4) < 1e-14, pv$outstanding == 0.75,
          abs(pv$capital_years - (1 + 0.75 * v + 0.5 * v^2 + 0.25 * v^3)) < 1e-14,
          pp$duration == 2.5)
p1 <- risk_loaded_price(sampled(c(0, 0, 2, 6)), distortion("tvar", 0.5), cost_of_capital = 0.25)
same <- price_over_payout(p1, payment_pattern(0.5, 1), cost_of_capital = 0.25)
slow <- price_over_payout(p1, pp, rate = 0.05, cost_of_capital = 0.25)
stopifnot(abs(same$premium - p1$premium) < 1e-14,
          abs(slow$premium - (slow$discounted_loss + 0.25 * slow$capital * slow$capital_years))
          < 1e-12,
          inherits(try(payment_pattern_from_development(c(1, 2), c(2, 1.1)), silent = TRUE),
                   "try-error"))

# Issue #211: tower diagnostics and alpha bounds.
local({
  layers <- data.frame(limit = c(1, 5, Inf), attachment = c(1, 5, 10),
                       expected_loss = c(1.706, 0.675, 0.295))
  freq <- data.frame(threshold = 1, frequency = 2.8)
  free <- fit_references(layers, freq)
  d <- tower_diagnostics(free)
  stopifnot(d$min_alpha < 1, !d$infinite_mean, length(d$warnings) > 0)
  b <- fit_references(layers, freq, alpha_bounds = c(1.2, 3))
  stopifnot(abs(layer(b, 5, 5) / 0.675 - 1) < 1e-10, layer(b, 2, 2) < layer(free, 2, 2))
})

# Issue #212: exposure helpers, the spread default's warning, a band from 0.
c3 <- swiss_re_curve(3)
w <- tryCatch(risk_profile(2, 400, c3, expected_loss = 1.2, lower = 1, upper = 5),
              warning = function(w) conditionMessage(w))
stopifnot(is.character(w), grepl("no longer has sums_insured", w))
ra <- risk_profile(2, 400, c3, expected_loss = 1.2, lower = 1, upper = 5, spread = "auto")
stopifnot(abs(ra@mean_sums_insured - 2) < 1e-9)
r0 <- risk_profile(1, 10, c3, premium = 1, loss_ratio = 0.6, lower = 0, upper = 2, spread = "auto")
stopifnot(is.finite(profile_layer_loss(r0, 0.5, 0.5)))
sev <- lognormal_from_mean_cv(1e5, 2)
tc <- tabulated_curve_from_severity(sev, 1e6)
xs <- c(0.01, 0.1, 0.3, 0.7)
stopifnot(max(abs(exposure_curve(tc, xs) - severity_exposure_curve(sev, 1e6, xs))) < 1e-3)
stopifnot(abs(attr(swiss_re_for_mean_rate(c3@mean), "c") - 3) < 1e-9)
share <- exposure_layer_share(c3, 4, 2, 10)
stopifnot(abs(attr(swiss_re_for_layer_share(4, 2, 10, share), "c") - 3) < 1e-9)
# A Pareto with alpha 3: CV 1/sqrt(3).
stopifnot(abs(layer_cv(pareto(100, 3), Inf, 0) - 1 / sqrt(3)) < 1e-9)

# Return metrics on cash flows (#480): one root at 10%, two at 10% and 20%
# (the one nearest 0 reported), none.
flows <- rbind(c(-100, 0, 121), c(-100, 230, -132), c(10, 5, 1))
r <- irr(flows, c(0, 1, 2))
stopifnot(nrow(r) == 3, identical(r$found, c(TRUE, TRUE, FALSE)))
stopifnot(identical(r$sign_changes, c(1L, 2L, 0L)), is.na(r$rate[3]))
stopifnot(all(abs(r$rate[1:2] - 0.1) < 1e-12), abs(r$npv[1]) < 1e-9)
stopifnot(abs(irr(flows[2, ], c(0, 1, 2), lower = 0.15)$rate - 0.2) < 1e-12)
by_month <- irr(c(-100, 0, 0, 103), 0:3, unit = "months")$rate
by_day <- irr(c(-100, 0, 0, 103), c(0, 31, 59, 90), unit = "days")$rate
stopifnot(abs(by_month - (1.03^4 - 1)) < 1e-12, abs(by_day - (1.03^(365 / 90) - 1)) < 1e-12)
stopifnot(abs(npv(c(-50, 60), c(0, 1), 0.05) - (-50 + 60 / 1.05)) < 1e-12)
stopifnot(abs(npv(flows, c(0, 1, 2), 0.1)[1]) < 1e-12)
stopifnot(abs(moic(30, 40, 60) - 1.3) < 1e-15, abs(leverage(0.5, 300, 50, 100) - 1) < 1e-15)
stopifnot(all(abs(moic(c(30, 0), 40, 60) - c(1.3, 1)) < 1e-15))
stopifnot(all(abs(tiered_fee(c(-5, 10, 20, 50), c(0, 10, 30), c(0.05, 0.1, 0.25)) -
                  c(0, 0.5, 1.5, 7.5)) < 1e-14))
stopifnot(inherits(try(irr(flows, c(0, 1)), silent = TRUE), "try-error"))
# A root outside the searched range is flagged and found on request (#670).
out <- irr(rbind(c(-100, 400), c(10, 5), c(-100, 1.5)), c(0, 1))
stopifnot(identical(out$found, c(FALSE, FALSE, FALSE)),
          identical(out$root_above, c(TRUE, FALSE, FALSE)),
          identical(out$root_below, c(FALSE, FALSE, TRUE)))
out <- irr(rbind(c(-100, 400), c(10, 5), c(-100, 1.5)), c(0, 1), extend = TRUE)
stopifnot(identical(out$found, c(TRUE, FALSE, TRUE)), abs(out$rate[1] - 3) < 1e-12,
          abs(out$rate[3] + 0.985) < 1e-12)
stopifnot(inherits(try(npv(1, 0, 0.1, unit = "weeks"), silent = TRUE), "try-error"))
stopifnot(inherits(try(moic(1, 0, 0), silent = TRUE), "try-error"))
stopifnot(inherits(try(tiered_fee(1, c(1, 1), c(0.1, 0.2)), silent = TRUE), "try-error"))

# A collateralised deal's monthly cash flows (#479).
thirds <- (1:3) / 3
terms <- cashflow_terms(100, 3, rep(1, 3), thirds, thirds, cashflow_requirement(c(reserve = 1)),
                        commutation_requirement = cashflow_requirement(c(outstanding = 1)))
run <- cashflow_run(terms, c(0.6, 1.5))
stopifnot(all(abs(run$flows - rbind(c(80, -20, -20), c(50, -50, -50))) < 1e-9))
stopifnot(all(abs(run$funded - c(40, 100)) < 1e-9), identical(run$months, c(0, 1, 2)))
tr <- cashflow_trace(terms, 1.5)
stopifnot(all(abs(tr$accounts$investor - run$flows[2, ]) < 1e-9), abs(tr$conservation_error) < 1e-9)
stopifnot(identical(tr$accounts$month, c(0, 1, 2)))
ramp <- function(m) pmin((1:60) / m, 1)
full <- function(...) {
  args <- list(100, 30, ramp(12), ramp(36), ramp(48),
               cashflow_requirement(c("unearned-premium" = 0.3, reserve = 1.2), limit = c(1.2, 1.5)),
               commutation_requirement = cashflow_requirement(c("unearned-premium" = 0.3,
                                                                outstanding = 1)),
               collected_rate = 0.7, lag = 2,
               adjustments = list(cashflow_adjustment(18, 0.1, less_commission = TRUE)),
               funding_dates = "quarterly", funding_rule = "obligation-increase",
               premium_offset = 0.7, reference = "running-max", annual_yield = 0.04,
               opening_fixed = 5, opening_share = 0.1, expense_rate = 0.02, buffer_months = 12,
               buffer_paid = TRUE, buffer_expenses = TRUE, fee = list(c(0, 0.1), c(0.1, 0.2)),
               fee_basis = "profit-over-capital")
  changes <- list(...)
  args[names(changes)] <- changes
  do.call(cashflow_terms, args)
}
# The Rust and Python tests' trough under the obligation rule.
stopifnot(abs(min(cashflow_trace(full(), 0.9, 0.2)$accounts$collateral_balance) +
                8.885288354521) < 1e-9)
stopifnot(min(cashflow_trace(full(funding_rule = "shortfall"), 0.9, 0.2)$accounts$
                collateral_balance) >= -1e-9)
variants <- list(
  full(),
  full(funding_rule = "shortfall", funding_source = "operating-first"),
  full(funding_dates = c(0, 7, 20), premium_offset = 0.5, reference = "previous-date",
       collateral_release = "at-commutation", interest_basis = "opening"),
  full(collateral_release = "quarterly", equity_release = list(every = 6, first = 5),
       interest_basis = "closing", fee = list(10, 0.25), fee_basis = "profit",
       adjustments = list(cashflow_adjustment(18, 0.1, TRUE, base = "to-date"))),
  full(equity_release = "at-commutation", buffer_months = 0)
)
u <- c(0.4, 0.9, 1.3, 2)
for (v in variants) {
  out <- cashflow_run(v, u, 0.2)
  for (j in seq_along(u)) {
    tj <- cashflow_trace(v, u[j], 0.2)
    stopifnot(abs(tj$conservation_error) < 1e-9, all(abs(tj$accounts$investor - out$flows[j, ]) < 1e-9),
              abs(tj$fee - out$fee[j]) < 1e-12)
  }
}
short <- full(funding_rule = "shortfall")
value <- npv(cashflow_run(short, u, 0.2)$flows, 0:29, 0.05, unit = "months")
stopifnot(all(diff(value) < 0))
# A requirement given month by month, as the terms compute it.
short0 <- full(funding_rule = "shortfall", buffer_months = 0)
rows <- t(sapply(c(0.7, 1.4), function(x) cashflow_trace(short0, x, 0.2)$accounts$requirement))
given <- cashflow_run(short0, c(0.7, 1.4), 0.2, months_requirement = rows)
stopifnot(all(abs(given$flows - cashflow_run(short0, c(0.7, 1.4), 0.2)$flows) < 1e-9))
stopifnot(inherits(try(cashflow_run(short0, c(0.7, 1.4), 0.2, months_requirement = rows[, 1:3]),
                       silent = TRUE), "try-error"))
# Each simulation paid by its own pattern (#600).
fast <- ramp(24)
slow <- ramp(56)
cc <- rep(0.2, 4)
same <- cashflow_run(short, u, cc, paid_patterns = ramp(48), pattern_index = rep(1, 4))
stopifnot(identical(same$flows, cashflow_run(short, u, cc)$flows))
index <- c(1, 2, 2, 1)
mixed <- cashflow_run(short, u, cc, paid_patterns = rbind(fast, slow), pattern_index = index)
for (k in 1:2) {
  sims <- which(index == k)
  one <- cashflow_terms(100, 30, ramp(12), ramp(36), list(fast, slow)[[k]],
                        cashflow_requirement(c("unearned-premium" = 0.3, reserve = 1.2),
                                             limit = c(1.2, 1.5)),
                        commutation_requirement = cashflow_requirement(c("unearned-premium" = 0.3,
                                                                         outstanding = 1)),
                        collected_rate = 0.7, lag = 2,
                        adjustments = list(cashflow_adjustment(18, 0.1, less_commission = TRUE)),
                        funding_dates = "quarterly", funding_rule = "shortfall",
                        premium_offset = 0.7, reference = "running-max", annual_yield = 0.04,
                        opening_fixed = 5, opening_share = 0.1, expense_rate = 0.02,
                        buffer_months = 12, buffer_paid = TRUE, buffer_expenses = TRUE,
                        fee = list(c(0, 0.1), c(0.1, 0.2)), fee_basis = "profit-over-capital")
  stopifnot(identical(mixed$flows[sims, , drop = FALSE], cashflow_run(one, u[sims], 0.2)$flows))
}
per <- cashflow_run(short, u, cc, paid_patterns = rbind(fast, slow, slow, fast))
stopifnot(identical(per$flows, mixed$flows))
tr <- cashflow_trace(short, 1.3, 0.2, paid_pattern = slow)
stopifnot(abs(tr$conservation_error) < 1e-9, identical(tr$accounts$investor, mixed$flows[3, ]))
stopifnot(inherits(try(cashflow_run(short, 0.5, paid_patterns = rbind(fast, slow)), silent = TRUE),
                   "try-error"),
          inherits(try(cashflow_run(short, 0.5, pattern_index = 1), silent = TRUE), "try-error"))
# A bound deal valued from its actual accounts (#603).
refused <- function(expr) inherits(try(expr, silent = TRUE), "try-error")
f <- full()
uu <- c(0, 0.4, 0.9, 1.3)
zero <- cashflow_valuation(f, 0, flows = double())
for (how in c("bornhuetter-ferguson", "truncation")) {
  fw <- cashflow_forward(zero, uu, c(0, 0.25, 0.1, 0.3), how)
  stopifnot(identical(fw$forward_rows, cashflow_run(f, uu, c(0, 0.25, 0.1, 0.3))$flows),
            identical(fw$since_inception_rows, fw$forward_rows))
}
for (x in c(0.4, 0.9, 1.3)) {
  path <- cashflow_trace(f, x, 0.2)$accounts$investor
  for (v in c(1, 12, 18)) {
    val <- do.call(cashflow_valuation, c(list(f, v), cashflow_actual_state(f, x, v, 0.2)))
    stopifnot(abs(val$share_paid - v / 48) < 1e-15)
    fw <- cashflow_forward(val, x, 0.2)
    stopifnot(fw$kept == 1, all(fw$months == v:29),
              all(abs(fw$since_inception_rows[1, ] - path) < 1e-9))
    vt <- cashflow_valuation_trace(val, x, 0.2)
    stopifnot(vt$accounts$month[1] == v, abs(vt$conservation_error) < 1e-9,
              abs(vt$opening_balance - val$nav) < 1e-12)
  }
}
state <- cashflow_actual_state(f, 0.9, 12, 0.2)
val <- do.call(cashflow_valuation, c(list(f, 12), state))
fw <- cashflow_forward(val, c(0.5, 0.9, 1.4), 0.2)
stopifnot(fw$nav == state$collateral + state$operating,
          fw$forward_rows[1, 1] == fw$flows[1, 1] - fw$nav, all(fw$forward_offsets == 0:17),
          all(diff(npv(fw$forward_rows, fw$forward_offsets, 0.05, unit = "months")) < 0),
          all(diff(npv(fw$since_inception_rows, fw$since_inception_offsets, 0.05,
                       unit = "months")) < 0),
          length(fw$forward_moic) == 3, length(fw$since_inception_moic) == 3)
cut <- cashflow_forward(val, c(0.05, 0.9), conditioning = "truncation")
stopifnot(identical(cut$kept, 2), 0.05 * 100 < state$paid)
bf <- condition_on_paid(c(0.05, 0.9), 100, state$paid, val$share_paid)
stopifnot(identical(bf$kept, c(1, 2)),
          abs(bf$ultimate[1] - (state$paid / 100 + 0.05 * (1 - 12 / 48))) < 1e-15)
stopifnot(identical(condition_on_paid(c(0.2, 0.8), 100, 30, 0.4, "truncation"),
                    list(kept = 2, ultimate = 0.8)))
stopifnot(refused(do.call(cashflow_valuation, c(list(f, 19), cashflow_actual_state(f, 0.9, 19, 0.2)))),
          refused(cashflow_valuation(f, 30, paid = 1)),
          refused(do.call(cashflow_valuation, c(list(f, 12), modifyList(state,
                                                                      list(funding_reference = NULL))))),
          refused(cashflow_valuation(f, 0, paid = 1)),
          refused(cashflow_forward(val, 0.5, conditioning = "chain-ladder")))
blind <- do.call(cashflow_valuation, c(list(f, 12), modifyList(state, list(flows = NULL))))
stopifnot(is.null(cashflow_forward(blind, 0.5)$since_inception_rows))
# Actual against expected paid by month (#603).
quarters <- cashflow_terms(100, 4, rep(1, 4), rep(1, 4), c(0.25, 0.5, 0.75, 1),
                           cashflow_requirement(c(outstanding = 1)))
ave <- deal_ave(c(15, 45), quarters, c(0.4, 0.6, 0.8), levels = c(0, 0.5, 1))
stopifnot(identical(ave$month, 0:1), identical(ave$expected, c(15, 30)),
          identical(ave$percentile, c(0.5, 1)), ave$ratio[2] == 1.5,
          identical(ave$q50, c(15, 30)), identical(ave$q100, c(20, 40)))
ave <- deal_ave(20, quarters, c(0.4, 0.4),
                paid_patterns = rbind(c(0.5, 1, 1, 1), c(0.25, 0.5, 0.75, 1)))
stopifnot(ave$expected == 15, ave$percentile == 0.75)
stopifnot(refused(deal_ave(1, quarters, 0.5, levels = 1.5)))
req <- cashflow_requirement(c(reserve = 1))
fails <- function(...) inherits(try(cashflow_terms(100, 3, rep(1, 3), thirds, thirds, req, ...),
                                    silent = TRUE), "try-error")
stopifnot(inherits(try(cashflow_terms(100, 3, rep(1, 3), thirds, thirds,
                                      cashflow_requirement(c(ibnr = 1))), silent = TRUE),
                   "try-error"))
stopifnot(inherits(try(cashflow_requirement(1), silent = TRUE), "try-error"))
stopifnot(fails(adjustments = list(cashflow_adjustment(3, 0.1))), fails(funding_dates = 0),
          fails(funding_dates = "weekly"), fails(collateral_release = list(first = 2)))
stopifnot(inherits(try(cashflow_terms(100, 4, rep(1, 3), thirds, thirds, req), silent = TRUE),
                   "try-error"))

# #214: the unpaid part of a pattern, a grid normalized, a distortion's
# single parameter.
p <- payment_pattern_from_development(c(1, 2, 3, 4), c(4, 2, 1.25, 1))
u <- payment_pattern_unpaid(p, 1.5)
stopifnot(isTRUE(all.equal(u$times, c(0.5, 1.5, 2.5))),
          isTRUE(all.equal(u$shares, c(0.25, 0.3, 0.2) / 0.75, tolerance = 1e-14)))
stopifnot(inherits(try(payment_pattern_unpaid(p, 4), silent = TRUE), "try-error"))
stopifnot(inherits(try(grid_distribution(1, c(0.4999999, 0.25, 0.25)), silent = TRUE), "try-error"))
g <- grid_distribution(1, c(0.4999999, 0.25, 0.25), normalize = TRUE)
stopifnot(abs(sum(g@probs) - 1) < 1e-15)
stopifnot(inherits(try(grid_distribution(1, c(0, 0), normalize = TRUE), silent = TRUE), "try-error"))
stopifnot(identical(distortion("proportional_hazard", 0.7)@parameter, 0.7),
          is.na(distortion("lep", 0.1, r0 = 0.05)@parameter))

# #490: a book of deals on one shared simulation.
a <- deal_run("a", rep(100, 4), c(40, 80, 20, 160),
              deductions = list(brokerage = rep(2.5, 4), fee = rep(5, 4)), outside_nmbf = "fee")
b <- deal_run("b", rep(300, 4), c(150, 330, 90, 210),
              deductions = list(brokerage = rep(7.5, 4), fee = rep(15, 4)), outside_nmbf = "fee")
book <- deal_book(list(a, b))
stopifnot(identical(book$ptr$names(), c("a", "b")),
          identical(book$ptr$items(), c("premium", "loss", "brokerage", "fee", "total_deductions",
                                        "nmbf", "margin")))
# #596: the accessors Python has, the same values as its test_deal_book_sums_and_summary.
stopifnot(identical(book_names(book), c("a", "b")), book_n_sims(book) == 4,
          identical(book_items(book), c("premium", "loss", "brokerage", "fee", "total_deductions",
                                        "nmbf", "margin")),
          identical(book_names(book_select(book, "b")), "b"),
          inherits(try(book_names(list()), silent = TRUE), "try-error"))
amounts <- book_amounts(book)
k <- amounts@keys
near(draw_matrix(amounts)[, k$item == "loss" & k$deal == "total"], c(190, 410, 110, 370))
s <- book_summary(book)
near(s$ratio[s$deal == "total" & s$item == "loss"], 270 / 400)
near(s$mean[s$deal == "a" & s$item == "nmbf"], 100 - 75 - 2.5)
d <- book_distribution(book, "loss", 90, tvar_levels = 50)
near(rowSums(d$contribution), d$total[rownames(d$contribution)])
near(d$total[["TVaR 50"]], (410 + 370) / 2)
stopifnot(identical(colnames(book_distribution(book, "margin", 10, tail = "lower")$marginal),
                    c("a", "b")))
stopifnot(inherits(try(deal_book(list(a, a)), silent = TRUE), "try-error"),
          inherits(try(book_distribution(book, "loss", 90, tail = "middle"), silent = TRUE),
                   "try-error"),
          identical(book_select(book, "b")$ptr$names(), "b"))
# #588: the marginal price of each deal written last.
tv <- distortion("tvar", 0.5)
m <- book_marginal_price(book, tv, cost_of_capital = 0.1)
near(m$assets, c(390 - 270, 390 - 120))
near(m$premium, (m$expected_loss + 0.1 * m$assets) / 1.1)
pp <- price_portfolio(select_components(select_components(amounts, "item", "loss"), "deal",
                                        c("a", "b")), tv, cost_of_capital = 0.1)
stopifnot(all(m$premium <= pp$by_component$premium + 1e-9))
one <- book_marginal_price(book_select(book, "b"), tv, distortion = distortion("tvar", 0.25))
near(one$premium, risk_loaded_price(sampled(c(150, 330, 90, 210)), tv,
                                    distortion = distortion("tvar", 0.25))$premium)
stopifnot(identical(book_marginal_price(book, tv, cost_of_capital = 0.1, deals = "b")$deal, "b"),
          inherits(try(book_marginal_price(book, tv, cost_of_capital = 0.1, deals = "z"),
                       silent = TRUE), "try-error"))
# #569: an efficient frontier of retentions and limits.
fr <- reinsurance_frontier(grid_distribution(1, rep(0.25, 4)), c(1, 2), c(1, Inf),
                           loading = 0.5, risk = "var", level = 0.9)
near(c(fr$expected_ceded[2], fr$premium[2], fr$net_risk[2]), c(0.75, 1.125, 1))
h <- 0.01
pe <- exp(-(0:2999) * h) * (1 - exp(-h))
pe[3000] <- pe[3000] + 1 - sum(pe)
rs <- (0:59) * h
fe <- reinsurance_frontier(grid_distribution(h, pe), rs, Inf, loading = 0.25, risk = "var", level = 0.99)
stopifnot(abs(rs[which.min(fe$net_risk + fe$premium)] - log(1.25)) <= 2 * h)
stopifnot(inherits(try(reinsurance_frontier(c(1, 2, 3), 1, 1), silent = TRUE), "try-error"))
fa <- reinsurance_frontier(c(0, 1, 2, 3), c(1, 2), c(1, Inf), loading = 0.5, risk = "var", level = 0.9)
near(fa$premium, fr$premium)
# A run from a tower takes each year's reinstatement premium (#595).
tw <- reinsurance_tower(list(xol_layer("cat", 10, 10, premium = 2, reinstatement_rates = c(1, 0.5))))
res <- apply_tower(tw, event_set_from_long(c(1, 2, 3, 3, 4, 4, 4), c(5, 15, 22, 12, 40, 40, 40), 4))
run <- deal_run_from_tower("cat", tw, res, "cat")
near(run$ptr$premium(), c(2, 3, 4.2, 5))
near(run$ptr$premium() - 2, draw_matrix(res)[, 4])
deposit <- deal_run_from_tower("cat", tw, res, "cat", reinstatement_premium = FALSE)
stopifnot(identical(deposit$ptr$premium(), rep(2, 4)),
          identical(deposit$ptr$loss(), run$ptr$loss()))
flows_a <- matrix(c(-10, 0, 10.1), 4, 3, byrow = TRUE)
a <- deal_run("a", rep(100, 4), c(40, 80, 20, 160), flows = flows_a, funded = rep(0, 4),
              opening = rep(10, 4))
b <- deal_run("b", rep(300, 4), c(150, 330, 90, 210), flows = 2 * flows_a, funded = rep(0, 4),
              opening = rep(20, 4), inception = 1)
one <- book_cashflows(deal_book(list(a)))
stopifnot(all(one$irr$found))
near(one$irr$rate, rep(1.01^6 - 1, 4), rel = 1e-9)
out <- book_cashflows(deal_book(list(a, b)), 0.05)
near(out$flows[1, ], c(-10, -20, 10.1, 20.2))
near(out$moic, rep(1 + 0.3 / 30, 4))
near(out$mean_by_deal[, "b"], c(0, -20, 0, 20.2))
b0 <- deal_run("b", rep(300, 4), c(150, 330, 90, 210), flows = 2 * flows_a, funded = rep(0, 4),
               opening = rep(20, 4))
near(book_cashflows(book_shift(deal_book(list(a, b0)), "b", 1), 0.05)$npv, out$npv)
stopifnot(identical(names(out$irr_by_deal), c("a", "b")))
terms <- cashflow_terms(100, 3, rep(1, 3), thirds, thirds, cashflow_requirement(c(reserve = 1)))
run <- cashflow_run(terms, c(0.6, 1.5, 0.2, 0.9))
cal <- book_calendar(deal_book(list(deal_run("c", rep(100, 4), c(60, 150, 20, 90),
                                             inception = 11, cashflows = run, terms = terms))),
                     2027)
k <- cal@keys
w <- k$item == "earned_premium" & k$deal == "total"
stopifnot(identical(as.integer(k$year[w]), c(2027L, 2028L)))
# earned evenly over three months from December 2027: a third in 2027
near(draw_matrix(cal)[1, w], c(100 / 3, 200 / 3), rel = 1e-12)

cat("prospicio R pricing tests passed\n")

# A tower quoted layer by layer and as one: two layers without annual terms.
local({
  sev <- grid_distribution(1, c(0, 0.3, 0.3, 0.2, 0.1, 0.1))
  l2 <- with_ceding_commission(with_deposit_premium(xol_layer("2x2", 2, 2), 1), 0.1)
  tw <- reinsurance_tower(list(xol_layer("1x1", 1, 1), l2))
  g <- tower_on_grid(tw, poisson_count(2), sev, points = 256)
  stopifnot(!is.null(g$total_ceded))
  q <- tower_quote(tw, g, distortion("tvar", 0.99), cost_of_capital = 0.12, brokerage = 0.1)
  stopifnot(identical(names(q$layers), c("1x1", "2x2")))
  el <- q$layers[["1x1"]]$expected_loss + q$layers[["2x2"]]$expected_loss
  stopifnot(abs(q$total$expected_loss - el) < 1e-9)
  stopifnot(abs(q$diversification -
    (q$layers[["1x1"]]$premium + q$layers[["2x2"]]$premium - q$total$premium)) < 1e-12)
  stopifnot(q$diversification > 0)
  for (x in c(q$layers, list(q$total))) stopifnot(abs(x$return_on_capital - 0.12) < 1e-9)
  # The commission of the second layer, weighted by the technical deposits.
  w <- q$layers[["2x2"]]$premium / (q$layers[["1x1"]]$premium + q$layers[["2x2"]]$premium)
  stopifnot(abs(q$total$commission_rate - 0.1 * w) < 1e-12)
  # A layer's quote reads back at its own deposit.
  back <- implied_quote(q$layers[["2x2"]], q$layers[["2x2"]]$premium)
  stopifnot(abs(back$return_on_capital - 0.12) < 1e-9)
  # Paid reinstatements on both layers: no compound total.
  tr <- reinsurance_tower(list(
    xol_layer("1x1", 1, 1, premium = 1, reinstatement_rates = 1),
    xol_layer("2x2", 2, 2, premium = 1, reinstatement_rates = 1)))
  gr <- tower_on_grid(tr, poisson_count(2), sev, points = 256)
  stopifnot(is.null(gr$total_ceded))
  err <- tryCatch(tower_quote(tr, gr, distortion("tvar", 0.99), cost_of_capital = 0.12),
                  error = function(e) conditionMessage(e))
  stopifnot(grepl("not one compound total", err))
})

# A deposit solved under a sliding scale and a profit commission (item 11).
local({
  probs <- c(0.05, 0.15, 0.3, 0.3, 0.15, 0.05)
  ceded <- grid_distribution(20, probs)
  x <- 20 * (seq_along(probs) - 1)
  p <- risk_loaded_price(ceded, distortion("tvar", 0.99), cost_of_capital = 0.1)
  qs <- with_premium_rate(quota_share("QS", 0.5), 1, 200)
  scale <- function(lr) pmin(0.35, pmax(0.20, 0.35 - 0.75 * (lr - 0.5)))
  q <- layer_quote(p, with_sliding_scale(qs, c(0.35, 0.20), c(0.5, 0.7)), ceded,
                   expenses = 0.02, income_tax = 0.21)
  stopifnot(abs(q$commission - sum(probs * q$premium * scale(x / q$premium))) < 1e-9 * q$premium)
  stopifnot(abs(q$return_on_capital - 0.1) < 1e-9)
  stopifnot(q$premium > technical_quote(p, expenses = 0.02, income_tax = 0.21)$premium)
  # The quote reads back at its own deposit with its commission.
  back <- implied_quote(q, q$premium, q$commission)
  stopifnot(abs(back$return_on_capital - 0.1) < 1e-9)
  q <- layer_quote(p, with_profit_commission(qs, 0.2, 0.1), ceded, expenses = 0.02)
  pc <- sum(probs * 0.2 * pmax(0.9 * q$premium - x, 0))
  stopifnot(abs(q$commission - pc) < 1e-9 * q$premium)
  flat <- layer_quote(p, with_ceding_commission(qs, 0.25), ceded, expenses = 0.02)
  stopifnot(abs(flat$premium - technical_quote(p, expenses = 0.02, commission = 0.25)$premium) <
              1e-12 * flat$premium)
  sw <- with_swing_rating(xol_layer("10x0", 10, 0), 5, 1.1)
  err <- tryCatch(layer_quote(p, sw, ceded), error = function(e) conditionMessage(e))
  stopifnot(grepl("swing", err))
})

# A commission deducted from the premium paid in (#609) and runs that pay
# the investor first (#608).
local({
  ramp <- function(m) pmin((1:40) / m, 1)
  req <- cashflow_requirement(c("unearned-premium" = 0.3, outstanding = 1.25))
  u <- c(0.5, 0.8)
  c_ <- c(0.2, 0.3)
  on <- cashflow_terms(100, 30, ramp(12), ramp(24), ramp(36), req, deduct_commission = TRUE)
  r <- cashflow_run(on, u, c_)
  for (j in 1:2) {
    flat <- cashflow_terms(100, 30, ramp(12), ramp(24), ramp(36), req, collected_rate = 1 - c_[j])
    stopifnot(max(abs(r$flows[j, ] - cashflow_run(flat, u[j], c_[j])$flows[1, ])) < 1e-9)
  }
  stopifnot(!any(r$paid_first))
  off <- cashflow_terms(100, 30, ramp(12), ramp(24), ramp(36), req)
  stopifnot(max(abs(cashflow_run(off, u, c_)$flows - r$flows)) > 1)
  reserve <- cashflow_terms(100, 3, rep(1, 3), (1:3) / 3, (1:3) / 3,
                            cashflow_requirement(c(reserve = 1)),
                            commutation_requirement = cashflow_requirement(c(outstanding = 1)))
  stopifnot(identical(cashflow_run(reserve, c(0.6, 1.5))$paid_first, c(TRUE, TRUE)))
})

# Monthly patterns from a payment pattern by development age (#601).
local({
  p <- payment_pattern_from_development(c(1, 2, 3, 5), c(5, 2, 1.25, 1))
  for (how in c("step", "linear", "log-linear")) {
    m <- payment_pattern_monthly(p, 72, how)
    stopifnot(max(abs(m[12 * c(1, 2, 3, 5)] - c(0.2, 0.5, 0.8, 1))) < 1e-12)
    stopifnot(all(diff(m) >= -1e-15), m[72] == 1)
  }
  pats <- cashflow_patterns_from_annual(pmin((1:72) / 12, 1), p)
  stopifnot(abs(pats$earned[1] - 1 / 144) < 1e-15, abs(pats$earned[23] - 1) < 1e-15)
  stopifnot(all(diff(pats$paid) >= -1e-15))
  r <- cashflow_run(cashflow_terms(100, 60, pats$written, pats$earned, pats$paid,
                                   cashflow_requirement(c(outstanding = 1))), 0.7)
  stopifnot(length(r$funded) == 1)
})
