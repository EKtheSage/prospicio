library(prospicio)

close <- function(got, want, rel) {
  stopifnot(all(abs(got - want) <= rel * abs(want)))
}

sev <- grid_distribution(1, c(0.1, 0.3, 0.25, 0.2, 0.1, 0.05))

# Panjer and FFT agree and match the compound mean and variance.
for (n in list(poisson_count(3), negative_binomial_count(2.5, 1.5))) {
  a <- compound_distribution(n, sev, 300)
  b <- compound_distribution(n, sev, 300, method = "fft")
  stopifnot(identical(a@report$method, "panjer"), identical(b@report$method, "fft"))
  stopifnot(max(abs(a@probs - b@probs)) < 1e-13, b@report$aliasing_error < 1e-12)
  close(mean(a), mean(n) * mean(sev), 1e-12)
  close(variance(a), mean(n) * variance(sev) + variance(n) * mean(sev)^2, 1e-10)
}

# Panjer underflow points to FFT; errors name the user's call.
unit <- grid_distribution(1, c(0, 1))
e <- tryCatch(compound_distribution(poisson_count(800), unit, 10), error = identity)
stopifnot(grepl("FFT", conditionMessage(e)), identical(conditionCall(e)[[1]], quote(compound_distribution)))

# Simulated events: reproducible, 1-based years, totals as a predictive distribution.
lsev <- lognormal_from_mean_cv(3e6, 1.5)
ev <- simulate_events(poisson_count(2), lsev, 20000, seed = 11)
ev2 <- simulate_events(poisson_count(2), lsev, 20000, seed = 11)
stopifnot(identical(events(ev, 123), events(ev2, 123)), identical(ev@n_sims, 20000))
stopifnot(sum(event_counts(ev)) == sum(sapply(1:20000, function(i) length(events(ev, i)))))
e <- tryCatch(events(ev, 0), error = identity)
stopifnot(grepl("out of range", conditionMessage(e)), identical(conditionCall(e), quote(events(ev, 0))))
tot <- total(ev)
stopifnot(S7::S7_inherits(tot, predictive_distribution), identical(provenance(tot)$seed, 11))
stopifnot(identical(provenance(tot)$stream_scheme, "chacha20/sim-index/v1"),
          identical(provenance(tot)$samplers, list(gamma = "marsaglia-tsang/2026-10")))

# Layers and towers.
l <- xol_layer("L", 10, 5, share = 0.5, aggregate_deductible = 4, aggregate_limit = 15)
stopifnot(identical(ceded(l, c(8, 20, 12)), 7.5))
stopifnot(identical(ceded(xol_layer("5x5", 5e6, 5e6, reinstatements = 1), c(12e6, 20e6, 30e6)), 10e6))
stopifnot(inherits(tryCatch(xol_layer("L", 1, 0, aggregate_limit = 2, reinstatements = 1), error = identity), "error"))
stopifnot(inherits(tryCatch(reinsurance_tower(list(xol_layer("L", 1, 0), xol_layer("L", 2, 1))), error = identity), "error"))

tw <- reinsurance_tower(list(xol_layer("5x5", 5e6, 5e6, reinstatements = 1), xol_layer("15x10", 15e6, 10e6, share = 0.6)))
res <- apply_tower(tw, ev)
stopifnot(identical(res@dims, c("kind", "layer")))
m <- draw_matrix(res)
stopifnot(max(abs(m[, 1] - (m[, 2] + m[, 3] + m[, 4]))) <= 1e-6 * max(m[, 1]))
stopifnot(identical(aggregate(res, keep = "kind")@keys$kind, c("gross", "ceded", "net")))

# Simulated mean ceded loss matches the exact layer value (within 4 standard errors).
ev <- simulate_events(poisson_count(2), lsev, 100000, seed = 3)
ce <- marginal(apply_tower(reinsurance_tower(list(xol_layer("5x5", 5e6, 5e6))), ev), list(kind = "ceded", layer = "5x5"))
se <- sqrt(variance(ce) / 100000)
stopifnot(abs(mean(ce) - 2 * layer(lsev, 5e6, 5e6)) < 4 * se)

# Quota share, stop-loss and inuring stages.
stopifnot(ceded(quota_share("QS", 0.25), c(8, 4)) == 3)
sl <- aggregate_stop_loss("SL", 50, 100)
stopifnot(ceded(sl, c(60, 70)) == 30, ceded(sl, 200) == 50)
stopifnot(inherits(try(quota_share("QS", 0), silent = TRUE), "try-error"))
tw <- inuring_tower(list(
  list(xol_layer("A", 10, 5, aggregate_limit = 15), xol_layer("B", 100, 18)),
  list(aggregate_stop_loss("SL", 20, 20))
))
stopifnot(identical(tw@stages, c(1, 1, 2)))
stopifnot(identical(tower_ceded(tw, c(20, 20)), c(A = 15, B = 4, SL = 1)))
stopifnot(inherits(try(inuring_tower(list(list(), list(xol_layer("A", 1, 0)))), silent = TRUE), "try-error"))

# Per-event allocation and reinstatement premiums.
l <- xol_layer("L", 10, 5, share = 0.5, aggregate_deductible = 4, aggregate_limit = 15)
stopifnot(identical(ceded_by_event(l, c(8, 20, 12)), c(0, 4.5, 3)))
paid <- xol_layer("10x10", 10, 10, premium = 2, reinstatement_rates = c(1, 0.5))
stopifnot(paid@aggregate_limit == 30, identical(paid@reinstatement_rates, c(1, 0.5)))
stopifnot(reinstatement_premium(paid, c(15, 25)) == 2.5)
stopifnot(inherits(try(xol_layer("L", 1, 0, reinstatements = 1, reinstatement_rates = 1), silent = TRUE), "try-error"))
ev <- simulate_events(poisson_count(2), lsev, 2000, seed = 3)
res <- apply_tower(reinsurance_tower(list(xol_layer("5x5", 5e6, 5e6, premium = 1e6, reinstatement_rates = 1))), ev)
stopifnot(identical(res@keys$kind, c("gross", "ceded", "net", "reinstatement_premium")))
m <- draw_matrix(res)
stopifnot(max(abs(m[, 4] - 1e6 * pmin(m[, 2], 5e6) / 5e6)) <= 1e-6)

# Towers on the grid.
gsev <- grid_distribution(1, c(0, 0.4, 0.3, 0.2, 0.1))
r <- tower_on_grid(reinsurance_tower(list(xol_layer("2x2", 2, 2))), poisson_count(3), gsev, 200)
stopifnot(r$on_points, abs(mean(r$ceded[["2x2"]]) - 1.2) < 1e-12)
stopifnot(abs(mean(r$gross) - mean(r$ceded[["2x2"]]) - mean(r$net)) < 1e-10)
stopifnot(r$gross@report$tail_mass < 1e-12, r$expected_reinstatement_premium[["2x2"]] == 0)
r <- tower_on_grid(reinsurance_tower(list(xol_layer("2x2", 2, 2, reinstatements = 1))), poisson_count(3), gsev, 200)
stopifnot(is.null(r$net))
bad <- inuring_tower(list(list(xol_layer("A", 2, 2, reinstatements = 1)), list(xol_layer("B", 4, 4))))
stopifnot(inherits(try(tower_on_grid(bad, poisson_count(3), gsev, 100), silent = TRUE), "try-error"))
m <- map_grid(gsev, function(v) min(max(v - 1.5, 0), 1))
stopifnot(!m$on_points, abs(mean(m$grid) - sum(gsev@probs * pmin(pmax(0:4 - 1.5, 0), 1))) < 1e-15)
stopifnot(inherits(try(map_grid(gsev, function(v) -1), silent = TRUE), "try-error"))
stopifnot(inherits(try(map_grid(gsev, function(v) stop("boom")), silent = TRUE), "try-error"))

# Portfolio: aggregate cover on any predictive distribution, join, reorder.
n <- 3000
res <- predictive_distribution(cbind(((0:(n - 1)) * 7919) %% n, ((0:(n - 1)) * 31) %% n + 1),
                               data.frame(origin = c(2023, 2024)))
prem <- predictive_distribution(matrix(((0:(n - 1)) * 104729) %% n, ncol = 1),
                                data.frame(lob = "motor"))
adc <- apply_tower(reinsurance_tower(list(aggregate_stop_loss("adc", 500, 3000))), res)
k <- draw_matrix(aggregate(adc, keep = "kind"))
stopifnot(all(abs(k[, 1] - k[, 2] - k[, 3]) < 1e-9), max(k[, 2]) == 500)
pf <- join_predictive(list(reserve = res, premium = prem), "risk")
stopifnot(identical(names(pf@keys), c("risk", "origin", "lob")), nrow(pf@keys) == 3)
ro <- reorder_groups(pf, "risk", matrix(c(1, 0.7, 0.7, 1), 2), seed = 5)
stopifnot(identical(sort(draw_matrix(ro)[, 3]), sort(draw_matrix(prem)[, 1])))
tot <- draw_matrix(aggregate(ro, keep = "risk"))
stopifnot(cor(tot[, 1], tot[, 2], method = "spearman") > 0.5)
stopifnot(inherits(try(join_predictive(list(res, prem), "risk"), silent = TRUE), "try-error"))

# FFT grid sizing (aggregate 1.0.1: round_bucket rungs; bs 1/16 on this book).
stopifnot(identical(round_bucket(c(1.1, 2.5, 5.5, 2412, 0.3)), c(2, 4, 8, 4000, 0.5)))
g <- recommend_grid(poisson_count(10), gamma_distribution(1 / 0.49, 50 * 0.49))
stopifnot(g$step == 0.0625, g$points == 65536, g$method == "moments")
h <- recommend_grid(poisson_count(20), pareto(100, 1.5), log2 = 14)
stopifnot(h$method == "single_big_jump", is.nan(h$moment_extent))
agg <- compound_auto(poisson_count(10), gamma_distribution(1 / 0.49, 50 * 0.49))
stopifnot(agg@report$aliasing_error < 1e-12, agg@report$sizing$points == 65536)
stopifnot(abs(mean(agg) / 500 - 1) < 1e-6)

cat("prospicio R aggregate tests passed\n")

# Surplus treaty on events that carry sums insured.
s <- surplus_treaty("surplus", 1e6, 4)
stopifnot(abs(ceded(s, c(0.5e6, 2e6, 1e6, 10e6), sums_insured = c(0.5e6, 2e6, 5e6, 10e6)) - 5.8e6) < 1e-6)
ev <- events_from_years(list(c(0.5e6, 2e6), c(1e6, 10e6), numeric()),
                        list(c(0.5e6, 2e6), c(5e6, 10e6), numeric()))
stopifnot(identical(event_sums_insured(ev, 2), c(5e6, 10e6)), identical(event_counts(ev), c(2, 2, 0)))
tw <- inuring_tower(list(list(s), list(xol_layer("1x1", 1e6, 1e6))))
res <- apply_tower(tw, ev)
dm <- draw_matrix(res)
stopifnot(abs(mean(dm[, 2]) - 5.8e6 / 3) < 1e-6, abs(mean(dm[, 3]) - 1e6 / 3) < 1e-6)
stopifnot(inherits(try(apply_tower(tw, events_from_years(list(1))), silent = TRUE), "try-error"))
stopifnot(inherits(try(events_from_years(list(5), list(4)), silent = TRUE), "try-error"))

# Reinstatements pro rata as to time.
timed <- xol_layer("10x10", 10, 10, premium = 2, reinstatement_rates = c(1, 0.5), pro_rata_time = TRUE)
stopifnot(timed@pro_rata_time, is.nan(reinstatement_premium(timed, 22)))
close(reinstatement_premium(timed, c(22, 12), times = c(0.25, 0.5)),
      2 * (10 * 0.75 + 0.5 * 2 * 0.5) / 10, 1e-12)
stopifnot(inherits(try(xol_layer("f", 10, 10, reinstatements = 1, pro_rata_time = TRUE),
                       silent = TRUE), "try-error"))
n <- 20000
ev <- events_from_years(rep(list(20), n), seed = 9)
one <- xol_layer("10x10", 10, 10, premium = 2, reinstatement_rates = 1, pro_rata_time = TRUE)
stopifnot(inherits(try(apply_tower(reinsurance_tower(list(one)), ev), silent = TRUE), "try-error"))
dated <- with_uniform_times(ev)
rp <- draw_matrix(apply_tower(reinsurance_tower(list(one)), dated))[, 4]
stopifnot(abs(mean(rp) - 1) < 4 * sd(rp) / sqrt(n), length(event_times(ev, 1)) == 0)
# Losses only in the second half of the year: 2 E[1 - t] = 0.5.
late <- with_seasonal_times(ev, c(0, 1))
stopifnot(all(sapply(1:100, function(i) event_times(late, i)) >= 0.5))
rp <- draw_matrix(apply_tower(reinsurance_tower(list(one)), late))[, 4]
stopifnot(abs(mean(rp) - 0.5) < 4 * sd(rp) / sqrt(n))
close(event_times(with_seasonal_times(ev, rep(1, 12)), 7), event_times(dated, 7), 1e-15)
stopifnot(inherits(try(with_seasonal_times(ev, c(0, 0)), silent = TRUE), "try-error"))
et <- events_from_years(list(c(5, 2), 9), times = list(c(0.1, 0.6), 0.3))
stopifnot(event_times(et, 2) == 0.3)
stopifnot(inherits(try(events_from_years(list(c(5, 2)), times = list(c(0.6, 0.1))), silent = TRUE),
                   "try-error"))

# Loss corridors.
qs <- with_loss_corridor(quota_share("QS", 0.3), 70, 90)
stopifnot(identical(qs@loss_corridor, c(70, 90, 1)), length(quota_share("Q", 0.3)@loss_corridor) == 0)
close(ceded(qs, c(50, 30)), 21, 1e-12)
close(ceded(qs, 120), 30, 1e-12)
lc <- with_loss_corridor(xol_layer("L", 10, 5, aggregate_deductible = 4, aggregate_limit = 12), 5, 9, 0.5)
stopifnot(ceded(lc, c(8, 20, 12)) == 12)
stopifnot(inherits(try(with_loss_corridor(qs, 2, 1), silent = TRUE), "try-error"))
back <- tower_from_json(tower_to_json(reinsurance_tower(list(qs))))
stopifnot(identical(back@layer_names, "QS"))

# Contract terms: premiums from quotes, commissions, swing and retro.
close(with_rate_on_line(xol_layer("10x10", 10, 10, share = 0.4), 0.2)@premium, 0.8, 1e-12)
close(with_deposit_premium(xol_layer("10x10", 10, 10, share = 0.4), 3)@premium, 1.2, 1e-12)
stopifnot(xol_layer("L", 10, 10, premium = 2.5)@premium == 2.5)
qs <- with_premium_rate(quota_share("QS", 0.5), 1, 200)
stopifnot(qs@premium == 100, length(qs@ceding_commission) == 0, is.null(qs@sliding_scale))
slid <- with_sliding_scale(qs, c(0.25, 0.45, 0.19), c(0.70, 0.60, 0.80))
stopifnot(identical(slid@sliding_scale$loss_ratio, c(0.60, 0.70, 0.80)))
close(ceding_commission_for(slid, c(55, 65, 75, 90)), c(45, 35, 22, 19), 1e-10)
pc <- with_profit_commission(qs, 0.25, 0.1)
stopifnot(identical(pc@profit_commission, c(share = 0.25, allowance = 0.1)))
close(profit_commission_for(pc, c(60, 95)), c(7.5, 0), 1e-12)
close(ceding_commission_for(with_ceding_commission(qs, 0.3), 500), 30, 1e-12)
stopifnot(inherits(try(with_sliding_scale(quota_share("Q", 0.5), 0.3, 0.6), silent = TRUE), "try-error"))
stopifnot(inherits(try(with_sliding_scale(qs, c(0.2, 0.3), c(0.6, 0.7)), silent = TRUE), "try-error"))
sw <- with_swing_rating(xol_layer("L", 1000, 0, share = 0.4), 20, 1.25, 50, 300)
stopifnot(identical(sw@swing_rating, c(basic = 20, lcm = 1.25, minimum = 50, maximum = 300)))
close(premium_for(sw, c(40, 1000)), c(58, 120), 1e-12)
stopifnot(inherits(try(with_swing_rating(slid, 0, 1), silent = TRUE), "try-error"))
close(retro_premium(c(0, 500, 1500), 1000, 1.1, maximum = 2500), c(1000, 1550, 2500), 1e-9)
stopifnot(retro_premium(1e6, 1000, 1.1) > 1e6)
# In a tower: one component per term, joint with the ceded losses, and the
# grid's expectations.
xl <- with_ceding_commission(with_swing_rating(xol_layer("3x3", 3, 3, share = 0.5), 1, 1.2, 2, 6), 0.1)
book <- with_profit_commission(with_sliding_scale(with_premium_rate(quota_share("QS", 0.5), 1, 30),
                                                  c(0.4, 0.2), c(0.5, 0.9)), 0.3, 0.25)
tw <- reinsurance_tower(list(book, xl))
sev <- grid_distribution(1, c(0.1, 0.3, 0.25, 0.2, 0.1, 0.05))
res <- apply_tower(tw, simulate_events(poisson_count(3), sev, 5000, seed = 9))
m <- draw_matrix(res)
stopifnot(ncol(m) == 8)
stopifnot(isTRUE(all.equal(m[, 5], premium_for(xl, m[, 3]))))
stopifnot(isTRUE(all.equal(m[, 6], ceding_commission_for(book, m[, 2]))))
g <- tower_on_grid(tw, poisson_count(3), sev, points = 400)
close(g$expected_premium[["QS"]], 15, 1e-12)
stopifnot(abs(g$expected_ceding_commission[["QS"]] - mean(m[, 6])) < 4 * sd(m[, 6]) / sqrt(nrow(m)))
stopifnot(identical(tower_to_json(tower_from_json(tower_to_json(tw))), tower_to_json(tw)))

# Towers save and load as JSON.
tw <- inuring_tower(list(list(quota_share("QS", 0.3), surplus_treaty("S", 1e6, 4)),
                         list(xol_layer("xl", 2e6, 1e6, premium = 3e5, reinstatement_rates = c(1, 0.5),
                                        pro_rata_time = TRUE))))
txt <- tower_to_json(tw)
back <- tower_from_json(txt)
stopifnot(identical(tower_to_json(back), txt), identical(back@layer_names, tw@layer_names))
stopifnot(inherits(try(tower_from_json("{}"), silent = TRUE), "try-error"))

# The occurrence PML against aggregate 1.0.1's oep on
# "agg O 2 claims sev lognorm 100 cv 2 poisson".
sev <- lognormal_from_mean_cv(100, 2)
pml <- occurrence_pml(poisson_count(2), sev, c(0.1, 0.001))
stopifnot(all(abs(pml$loss / c(348.934754485, 2906.643201) - 1) < 1e-9))
stopifnot(all(abs(pml$annual_return_period - c(10, 1000)) < 1e-9))
nb <- negative_binomial_count(1.5, 2)
stopifnot(abs(occurrence_exceedance(nb, sev, occurrence_pml(nb, sev, 0.05)$loss) - 0.05) < 1e-10)
stopifnot(inherits(try(occurrence_pml(poisson_count(0.1), sev, 0.2), silent = TRUE), "try-error"))
stopifnot(mean(annual_maxima(simulate_events(poisson_count(2), sev, 2000, seed = 1))) > 0)

# Compound moments and approximations against aggregate 1.0.1 on
# "agg B 10 claims sev gamma 50 cv 0.7 poisson".
g <- gamma_distribution(1 / 0.49, 50 * 0.49)
m <- compound_moments(poisson_count(10), g)
stopifnot(abs(m[["mean"]] / 500 - 1) < 1e-12, abs(m[["skew"]] / 0.5129464825493272 - 1) < 1e-12)
fit <- moment_approximation(poisson_count(10), g, "shifted_gamma")
stopifnot(abs(fit$parameters[["shift"]] / -252.52525252525868 - 1) < 1e-10)
q <- approximation_quantile(fit, c(0.5, 0.99))
stopifnot(all(abs(approximation_cdf(fit, q) - c(0.5, 0.99)) < 1e-10))
stopifnot(inherits(try(compound_moments(poisson_count(2), pareto(100, 2.5)), silent = TRUE),
                   "try-error"))

# Layer statistics and prices by view against aggregate 1.0.1 on
# "agg C 2 claims 1000 xs 0 sev lognorm 100 cv 2 occurrence net of 500 xs 100 poisson".
sev <- discretize(lognormal_from_mean_cv(100, 2), 1, 1001, "rounding")
s <- occurrence_layer_statistics(poisson_count(2), sev, limit = 500, attachment = 100)
stopifnot(abs(s$pr_attach / 0.261655626646 - 1) < 1e-10,
          abs(s$loss_on_line / 0.151169062665 - 1) < 1e-9,
          abs(s$count[["mean"]] - 2 * s$pr_attach) < 1e-14)
half <- occurrence_layer_statistics(poisson_count(2), sev, 500, 100, share = 0.5)
stopifnot(abs(half$aggregate[["skew"]] / s$aggregate[["skew"]] - 1) < 1e-12)
r <- tower_on_grid(reinsurance_tower(list(xol_layer("occ", 500, 100))), poisson_count(2), sev,
                   points = 2^14)
p <- tower_price(r, distortion("proportional_hazard", 0.7))
stopifnot(identical(p$view, c("gross", "occ", "net")),
          abs(p$premium / c(297.443189953534, 139.358753189474, 164.315772212950) - 1) < 2e-8)
stopifnot(inherits(try(occurrence_layer_statistics(poisson_count(2), sev, 500, 100, share = 0),
                       silent = TRUE), "try-error"))
# Eventual ruin: claims of 1, 2 or 3 at rho 0.1, against aggregate 1.0.1.
sev3 <- grid_distribution(1, c(0, 0.5, 0.3, 0.2, numeric(1020)))
ruin <- eventual_ruin(sev3, rho = 0.1, p = c(0.1, 1e-300))
stopifnot(abs(ruin$psi$psi[ruin$psi$u == 10] - 0.15362022867039649) < 1e-12)
stopifnot(abs(ruin$severity_mean - 1.7) < 1e-12, ruin$lundberg_exponent > 0, ruin$aliasing_error < 1e-12)
stopifnot(!is.na(ruin$capital[1]), is.na(ruin$capital[2]), ruin$capital_interpolated[1] <= ruin$capital[1])
same <- eventual_ruin(sev3, loss_ratio = 1 / 1.1)
stopifnot(max(abs(same$psi$psi - ruin$psi$psi)) < 1e-14)
stopifnot(abs(eventual_ruin(sev3, rho = 0.1, cap = 2)$severity_mean - 1.1 / 0.8) < 1e-12)
stopifnot(inherits(try(eventual_ruin(sev3, rho = 0.1, cap = 2, excess = 1), silent = TRUE), "try-error"))
stopifnot(inherits(try(eventual_ruin(sev3, rho = 0), silent = TRUE), "try-error"))
# Wiener-Hopf: the same claims after gamma waits (shape 2), against
# aggregate 1.0.1's wiener_hopf at rho 0.1.
wh <- wiener_hopf(sev3, gamma_distribution(2, 1), rho = 0.1, p = 0.01)
stopifnot(abs(wh$psi$psi[wh$psi$u == 10] - 0.1973674584712206) < 1e-10, wh$capital == 31)
stopifnot(abs(wh$premium_rate - 1.1 * 1.7 / 2) < 1e-12, wh$mean_step < 0)
stopifnot(inherits(try(wiener_hopf(sev3, gamma_distribution(2, 1), rho = 0), silent = TRUE), "try-error"))
# Clash against aggregate 1.0.1: clash C 2 3 1 claims dsev [1 2] [.5 .5]
# dsev [1] poisson, bs 1 and log2 (7, 7).
clash <- clash_model(2, 3, 1)
stopifnot(clash$neither == 6, clash$events == 12, clash$trigger_a == 0.25)
ca <- grid_distribution(1, c(0, 0.5, 0.5))
cb <- grid_distribution(1, c(0, 1))
joint <- clash_aggregate(clash, poisson_count(12), ca, cb, points = c(128, 128))
stopifnot(identical(dim(joint$density), c(128L, 128L)))
stopifnot(abs(joint$density[4, 3] - 0.02323830165624712) < 1e-12)
stopifnot(abs(joint$correlation / 0.2738612787524785 - 1) < 1e-9)
stopifnot(abs(joint$total@probs[6] - 0.08374051103504515) < 1e-12)
stopifnot(max(joint$report$aliasing_mass) < 1e-14, abs(joint$covariance - 1.5) < 1e-9)
sev2 <- clash_severity(clash, ca, cb)$density
same <- fft_bivariate(poisson_count(12), sev2, steps = c(1, 1), points = 128)
stopifnot(abs(same$covariance - joint$covariance) < 1e-12)
stopifnot(inherits(try(clash_model(1, 1, 0), silent = TRUE), "try-error"))
stopifnot(inherits(try(clash_aggregate(clash, poisson_count(2), ca, cb, 8), silent = TRUE), "try-error"))
# Layer loss picks against aggregate 1.0.1's Aggregate.picks.
gsev <- discretize(gamma_from_mean_cv(100, 1.5), step = 1, points = 2^14, method = "rounding")
picked <- layer_loss_picks(gsev, c(100, 500, 1000), c(480, 420, 60), claims = 10)
stopifnot(!picked$straight_bottom, abs(picked$severity@probs[201] - 0.0008386073686922624) < 1e-12)
stopifnot(max(abs(10 * picked$layers$target - c(480, 420, 60))) < 1e-12)
stopifnot(layer_loss_picks(gsev, c(100, 500, 1000), c(700, 250, 50), claims = 10)$straight_bottom)
stopifnot(inherits(try(layer_loss_picks(gsev, 100.5, 10), silent = TRUE), "try-error"))
# Total cession of a stage and an aggregate cover, against aggregate 1.0.1
# (book T of validation/scripts/aggregate_reinsurance.py).
gt <- discretize(gamma_from_mean_cv(50, 1.5), step = 1, points = 1001, method = "rounding")
ct <- occurrence_cession_statistics(poisson_count(5), gt, limits = c(100, 200), attachments = c(100, 200))
stopifnot(abs(ct$aggregate[["mean"]] / 67.33279562141576 - 1) < 1e-9)
two <- tower_on_grid(reinsurance_tower(list(xol_layer("a", 100, 100), xol_layer("b", 200, 200))),
                     poisson_count(5), gt, points = 2^14)
stopifnot(identical(two$stage_sizes, 2), abs(mean(two$stage_ceded[[1]]) / 67.33279562141576 - 1) < 1e-9)
tp <- tower_price(two, distortion("proportional_hazard", 0.7))
stopifnot(identical(tp$view, c("gross", "a", "b", "stage_1", "net")))
stopifnot(abs(tp$premium[tp$view == "stage_1"] / 111.0007062267948 - 1) < 2e-8)
cover <- aggregate_layer_statistics(two$net, limit = 300, attachment = 200)
stopifnot(abs(cover$pr_attach - 0.3912338371214888) < 1e-12, abs(cover$aggregate[["mean"]] / 38.7761195844111 - 1) < 1e-9)

# A quota share on an annual loss-ratio grid (#206).
probs <- numeric(81)
probs[c(51, 61, 71, 81)] <- 0.25
qs <- with_ceding_commission(with_deposit_premium(quota_share("QS", 0.4), 100), 0.3)
r <- tower_on_annual_grid(reinsurance_tower(list(qs)), grid_distribution(1, probs))
stopifnot(abs(mean(r$ceded[["QS"]]) - 26) < 1e-12, abs(r$expected_ceding_commission[["QS"]] - 12) < 1e-12,
          abs(mean(r$net) - 39) < 1e-12)

# An event loss table against its analytic OEP (#196).
elt <- event_loss_table(rate = c(0.01, 0.04, 0.2), mean = c(10000, 3000, 800),
                        sd_independent = c(0, 0, 0), sd_correlated = c(0, 0, 0),
                        exposure = c(10000, 3000, 800))
stopifnot(abs(elt_oep(elt, 5000)$oep - (1 - exp(-0.01))) < 1e-14,
          abs(elt_oep(elt, 500)$exceedance_frequency - 0.25) < 1e-14)
sim <- simulate_elt(elt, 20000, seed = 3)
hit <- mean(vapply(seq_len(20000), function(y) max(c(0, events(sim$events, y))) > 2000, NA))
p2 <- elt_oep(elt, 2000)$oep
stopifnot(abs(hit - p2) < 4 * sqrt(p2 * (1 - p2) / 20000),
          identical(lengths(sim$ids), as.integer(event_counts(sim$events))))
beta <- event_loss_table(0.05, 500, 200, 50, 2000)
stopifnot(abs(elt_oep(beta, elt_occurrence_pml(beta, 0.01))$oep - 0.01) < 1e-9)
stopifnot(inherits(try(event_loss_table(0.1, 50, 60, 0, 100), silent = TRUE), "try-error"))
stopifnot(identical(as.numeric(event_counts(event_set_from_long(c(2, 2), c(3, 4), 3))), c(0, 2, 0)))

# Merging event sets and the hours clause (#196).
ma <- event_set_from_long(1, 100, 2)
mb <- event_set_from_long(c(1, 2), c(3, 5), 2)
stopifnot(identical(events(merge_event_sets(ma, mb), 1), c(100, 3)),
          identical(events(merge_event_sets(ma, mb), 2), 5))
h <- 1 / 8760
grouped <- hours_clause(event_set(ptr = prospicio:::EventSet$from_years(
  list(c(1, 2, 4)), NULL, 0, list(c(0, 72 * h, 72.5 * h)))), 72)
stopifnot(identical(events(grouped, 1), c(3, 4)),
          inherits(try(hours_clause(ma, 72), silent = TRUE), "try-error"))

# Two lines coupled by a copula.
ca <- grid_distribution(1, c(0.1, 0.3, 0.4, 0.2))
cb <- grid_distribution(2, c(0.2, 0.5, 0.3))
cs <- copula_severity(ca, cb, archimedean_copula("clayton", 3), trigger_a = 0.5)
stopifnot(abs(cs$density[1, 1] + sum(cs$density[1, -1]) - (0.5 + 0.05)) < 1e-15,
          max(abs(colSums(cs$density) - c(0.2, 0.5, 0.3))) < 1e-15, cs$correlation > 0)
cz <- copula_severity(ca, cb, gaussian_copula(diag(2)))
stopifnot(abs(cz$density[3, 2] - 0.4 * 0.5) < 1e-15)
cj <- fft_bivariate(poisson_count(3), cs$density, cs$steps, points = 64)
stopifnot(cj$correlation > 0,
          inherits(try(copula_severity(ca, cb, t_copula(matrix(c(1, 0.5, 0.5, 1), 2), 4)),
                       silent = TRUE), "try-error"))
# The P&L ledger: the design note's example 2 against aggregate 1.0.1's PnL.
gross <- pnl_group("Gross", "sell",
  list(pnl_leg("Premium", function(x) pmin(pmax(5 + 1.1 * x, 12), 40), kind = "premium")),
  list(pnl_leg("Loss", identity, kind = "loss"), pnl_leg("LAE", function(x) 0.05 * x)))
xl <- pnl_group("XL", "buy", list(pnl_leg("XL premium", 4)),
  list(pnl_leg("XL recovery", function(x) pmin(pmax(x - 15, 0), 25)), pnl_leg("XL commission", 0.4)))
pnl <- pnl_ledger(list(c(0, 10, 20, 50), c(0.4, 0.3, 0.2, 0.1)), list(gross, xl), name = "Ex2",
                  result_name = "margin")
s <- ledger_sheet(pnl)
m <- s[s$row == "margin", ]
stopifnot(abs(m$EX - 6.3) < 1e-12, abs(m$SD - 2.913760457) < 1e-9)
stopifnot(max(abs(c(m$k01, m$k50, m$k99) - c(1.9, 7.4, 8.9))) < 1e-12)
stopifnot(abs(s$k50[s$row == "Premium"] - 27) < 1e-12, s$label[nrow(s)] == "Impact")
card <- ledger_card(pnl)
stopifnot(card$side[nrow(card)] == "Impact", abs(card$EX[nrow(card)] + 0.1) < 1e-12)
stopifnot(abs(TVaR(ledger_result(pnl), 0.5) - 8.5) < 1e-12, abs(mean(ledger_row(pnl, "Loss")) + 12) < 1e-12)
stopifnot(identical(ledger_legs(pnl)$kind[1:2], c("premium", "loss")))
one <- pnl_ledger(grid_distribution(1, c(0.5, 0, 0.5)),
                  list(pnl_group("Book", "sell", list(pnl_leg("P", 3)), list(pnl_leg("L", identity)))))
stopifnot(identical(ledger_sheet(one)$label, c("P", "L", "Total")), is.null(ledger_sheet(one)$step))
stopifnot(inherits(try(pnl_leg("L", 1, kind = "bogus"), silent = TRUE), "try-error"))

# Rebucketed legs, the ratio exhibit and the stack (aggregate 1.0.1).
local({
  src <- list(c(0, 10, 20, 50), c(0.4, 0.3, 0.2, 0.1))
  gross <- pnl_group("Gross", "sell",
    list(pnl_leg("Premium", function(x) pmin(pmax(5 + 1.1 * x, 12), 40), kind = "premium")),
    list(pnl_leg("Loss", identity, kind = "loss"),
         pnl_leg("LAE", function(x) 0.05 * x, kind = "expense")))
  xl <- pnl_group("XL", "buy", list(pnl_leg("XL premium", 4, kind = "premium")),
    list(pnl_leg("XL recovery", function(x) pmin(pmax(x - 15, 0), 25), kind = "recovery"),
         pnl_leg("XL commission", 0.4, kind = "commission")))
  r <- ledger_ratios(pnl_ledger(src, list(gross, xl), name = "Ex2", result_name = "margin"))
  stopifnot(identical(r$step, c("Gross", "XL", "All")))
  stopifnot(abs(r$E_LR[1] - 0.46064814814814825) < 1e-12)
  stopifnot(abs(r$M_share[3] - 0.984375) < 1e-12)
  b <- pnl_ledger(src, list(pnl_group("Ex2b", "sell", list(pnl_leg("consideration", 20)),
    list(pnl_leg("Loss", function(x) 1.03 * x, step = 4)))))
  v <- ledger_validation(b)
  stopifnot(v$label == "Loss", abs(v$EX + 12.36) < 1e-12, v$abs_err < 1e-12)
  m <- ledger_marginal(b)
  stopifnot(identical(unname(unlist(m[m$row == "Loss", c("P01", "P10", "P50")])), c(-52, -48, -8)))
  c2 <- pnl_ledger(src, list(pnl_group("Ex2c", "sell",
    list(pnl_leg("Prem", function(x) ifelse(x == 20, 30, 10))), list(pnl_leg("Loss", identity)))))
  s <- ledger_stack(list(ex2b = b, ex2c = c2), list(c("c less b", "ex2c", "ex2b")))
  stopifnot(identical(s$label, c("ex2b", "ex2c", "c less b")))
  stopifnot(abs(s$EX[3] + 5.64) < 1e-12, abs(s$P01[3] + 8.5) < 1e-12)
  stopifnot(inherits(try(ledger_stack(list(a = b), list(c("x", "a", "nope"))), silent = TRUE), "try-error"))
})

# The book builders over annual covers (aggregate 1.0.1's pnl and xpnl).
local({
  gross <- list(c(2, 3, 4, 5, 6, 8), c(0.25, 0.3, 0.09, 0.2, 0.12, 0.04))
  high <- with_ceding_commission(with_deposit_premium(aggregate_stop_loss("High", 3, 5), 1), 0.1)
  low <- with_deposit_premium(aggregate_stop_loss("Low", 2, 3), 0.5)
  book <- pnl_book("Ex6", 10, list(pnl_expense("premium", 0.1)))
  walk <- book_ledger(book, gross, reinsurance_tower(list(high)), "walk")
  s <- ledger_sheet(walk)
  stopifnot(abs(s$EX[s$row == "High recovery"] - 0.24) < 1e-12)
  stopifnot(abs(s$k01[s$row == "High recovery"] - 2 / 3) < 1e-12)
  net <- book_ledger(book, gross, reinsurance_tower(list(high)))
  stopifnot(abs(mean(ledger_result(net)) - 4.54) < 1e-12, "commission" %in% ledger_sheet(net)$row)
  peel <- book_ledger(book, gross, reinsurance_tower(list(low, high)), "peel-top-down")
  st <- ledger_sheet(peel)$step
  stopifnot(match("High", st) < match("Low", st), "All aggregate" %in% st)
  stopifnot(abs(mean(ledger_result(peel)) - 4.85) < 1e-12)
  plain <- pnl_book("P", 10, list(pnl_expense("premium", 0.1, label = "Acq"),
                                  pnl_expense(list(list("loss", 0.05), list("fixed", 1)))))
  r <- ledger_sheet(book_ledger(plain, gross))
  stopifnot(identical(r$row[3:4], c("Acq", "expense")))
  stopifnot(abs(derive_premium(5, list(pnl_expense("premium", 0.1), pnl_expense("fixed", 1))) - 6 / 0.9) < 1e-12)
  stopifnot(inherits(try(derive_premium(5, list(pnl_expense("loss", 0.1))), silent = TRUE), "try-error"))
  stopifnot(inherits(try(book_ledger(book, gross, NULL, "peel-top-down"), silent = TRUE), "try-error"))
})

# Occurrence covers on the joint gross and ceded loss (aggregate 1.0.1's
# example 4): net 13.5, the cover's recovery 0.8.
local({
  sev <- grid_distribution(1, c(0, 0.5, 0.3, 0, 0.2))
  occ <- reinsurance_tower(list(with_ceding_commission(
    with_deposit_premium(xol_layer("occ 2 xs 2", 2, 2), 2), 0.25)))
  joint <- occurrence_source(poisson_count(2), sev, occ, c(64, 32))
  stopifnot(length(joint$gross) == 280)
  book <- pnl_book("Ex4", 20, list(pnl_expense("premium", 0.1)))
  s <- ledger_sheet(book_occurrence_ledger(book, joint, occ, face = "walk"))
  stopifnot(abs(s$EX[s$row == "net through occ 2 xs 2"] - 13.5) < 1e-10)
  stopifnot(abs(s$k01[s$row == "net through occ 2 xs 2"] - 7.5) < 1e-10)
  stopifnot(abs(s$EX[s$row == "occ 2 xs 2 recovery"] - 0.8) < 1e-10)
  agg <- reinsurance_tower(list(with_deposit_premium(aggregate_stop_loss("agg 3 xs 5", 3, 5), 0.5)))
  w <- book_occurrence_ledger(book, joint, occ, agg, "walk")
  c7 <- book_occurrence_ledger(book, joint, occ, agg)
  stopifnot(abs(mean(ledger_result(w)) - mean(ledger_result(c7))) < 1e-12)
  stopifnot("Net of occurrence" %in% ledger_sheet(w)$step)
})

# Variable terms as legs, retro and pnl_program's ladder (aggregate 1.0.1).
local({
  gross <- list(c(2, 3, 4, 5, 6, 8), c(0.25, 0.3, 0.09, 0.2, 0.12, 0.04))
  book <- pnl_book("V", 10, list(pnl_expense("premium", 0.1)))
  base <- with_deposit_premium(aggregate_stop_loss("ceded agg", 3, 5), 1)
  slide <- reinsurance_tower(list(with_sliding_scale(base, c(0.3, 0.1), c(0.2, 0.6))))
  s <- ledger_sheet(book_ledger(book, gross, slide, "walk"))
  stopifnot(abs(s$EX[s$row == "sliding commission"] - 0.268) < 1e-12)
  pc <- reinsurance_tower(list(with_profit_commission(base, 0.25, 0.1)))
  stopifnot(abs(mean(ledger_result(book_ledger(book, gross, pc))) - 4.629) < 1e-12)
  retro <- pnl_book("V", 2, list(pnl_expense("premium", 0.1)), retro = c(2, 1.1, 3, 9))
  stopifnot(abs(mean(ledger_row(book_ledger(retro, gross), "Premium")) - 6.108) < 1e-12)
  p <- pnl_program(405.97794085441933, 0.9, 0.25, data.frame(
    ceded_loss = c(86.78504710067914, 8.144345556686755), share = 1, combined_ratio = c(0.6, 0.5)))
  stopifnot(p$premium == 817, identical(p$deposits, c(145, 16.29)))
  stopifnot(identical(round_premium(c(146.5, 16.285)), c(146, 16.29)), size_premium(499.885, 0.7) == 714)
})

# Reinstatement premiums on the joint (aggregate 1.0.1): D + h(R) varies.
local({
  sev <- grid_distribution(1, c(0, 0.5, 0.3, 0, 0.2))
  occ <- reinsurance_tower(list(xol_layer("occ 2 xs 2", 2, 2, premium = 2, reinstatement_rates = 1)))
  joint <- occurrence_source(poisson_count(2), sev, occ, c(64, 32))
  book <- pnl_book("R", 20, list(pnl_expense("premium", 0.1)))
  s <- ledger_sheet(book_occurrence_ledger(book, joint, occ, face = "walk"))
  stopifnot(abs(s$EX[s$row == "occ 2 xs 2 premium"] + 2.65936) < 1e-5)
  stopifnot(abs(mean(ledger_result(book_occurrence_ledger(book, joint, occ))) - 12.323104) < 1e-6)
})
