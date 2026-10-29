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
# Attachment and exhaustion on simulated years (#581).
tl <- xol_layer("L", 10, 10, share = 0.5, aggregate_deductible = 3, aggregate_limit = 15)
ts <- layer_trial_statistics(tl, events_from_years(list(numeric(), 12, 20, c(20, 15, 30), c(9, 10))))
stopifnot(ts$n_years == 5, ts$p_attach == 0.6, ts$p_exhaust_occurrence == 0.4,
          ts$p_exhaust_annual == 0.2, ts$occurrences_mean == 1,
          abs(ts$ceded_given_attach - 0.5 * 22 / 3) < 1e-12,
          abs(ts$occurrences_se - sqrt(1.5 / 5)) < 1e-12)
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
# A Spearman target from a long table, checked against R's own Spearman.
sp_n <- 20000
sp <- predictive_distribution(sapply(c(1, 7919, 104729), function(m) ((0:(sp_n - 1)) * m) %% sp_n),
                              data.frame(class = c("A", "B", "C")))
sp_long <- data.frame(from = c("B", "A", "C"), to = c("A", "C", "B"), rho = c(0.5, 0.3, 0.7))
sp_r <- reorder_groups(sp, "class", sp_long, seed = 9, kind = "spearman")
sp_got <- group_spearman(sp_r, "class")
sp_want <- matrix(c(1, 0.5, 0.3, 0.5, 1, 0.7, 0.3, 0.7, 1), 3, dimnames = list(c("A", "B", "C"), c("A", "B", "C")))
stopifnot(all(abs(sp_got - sp_want) < 0.01))
stopifnot(abs(sp_got["B", "C"] - cor(draw_matrix(sp_r), method = "spearman")[2, 3]) < 1e-12)
stopifnot(group_spearman(reorder_groups(sp, "class", sp_want, seed = 9), "class")["B", "C"] < 0.695)
stopifnot(inherits(try(reorder_groups(sp, "class", sp_long[1:2, ], seed = 9), silent = TRUE), "try-error"))
stopifnot(inherits(try(join_predictive(list(res, prem), "risk"), silent = TRUE), "try-error"))
# A copula reorder: whole rows move, and the totals take the copula draws' ranks.
cp_n <- 3000
cp <- predictive_distribution(
  cbind(((0:(cp_n - 1)) * 7919) %% cp_n, 3 * (((0:(cp_n - 1)) * 7919) %% cp_n), ((0:(cp_n - 1)) * 104729) %% cp_n),
  data.frame(seg = c("a", "a", "b"), item = c(0, 1, 0))
)
gum <- archimedean_copula("gumbel", 2)
cp_r <- reorder_groups_by_copula(cp, "seg", gum, seed = 9)
cp_m <- draw_matrix(cp_r)
stopifnot(all(cp_m[, 2] == 3 * cp_m[, 1]), identical(sort(cp_m[, 3]), sort(draw_matrix(cp)[, 3])))
cp_u <- copula_sample(gum, cp_n, seed = 9)
stopifnot(identical(rank(cp_m[, 1] + cp_m[, 2]), rank(cp_u[, 1])), identical(rank(cp_m[, 3]), rank(cp_u[, 2])))
stopifnot(abs(group_spearman(cp_r, "seg")["a", "b"] - 0.68) < 0.03)
stopifnot(inherits(try(reorder_groups_by_copula(cp, "seg", archimedean_copula("clayton", 1, dim = 3), 1), silent = TRUE), "try-error"))

# FFT grid sizing (aggregate 1.0.1: round_bucket rungs; bs 1/16 on this book).
stopifnot(identical(round_bucket(c(1.1, 2.5, 5.5, 2412, 0.3)), c(2, 4, 8, 4000, 0.5)))
g <- recommend_grid(poisson_count(10), gamma_distribution(1 / 0.49, 50 * 0.49))
stopifnot(g$step == 0.0625, g$points == 65536, g$method == "moments")
h <- recommend_grid(poisson_count(20), pareto(100, 1.5), log2 = 14)
stopifnot(h$method == "single_big_jump", is.nan(h$moment_extent))
agg <- compound_auto(poisson_count(10), gamma_distribution(1 / 0.49, 50 * 0.49))
stopifnot(agg@report$aliasing_error < 1e-12, agg@report$sizing$points == 65536)
stopifnot(abs(mean(agg) / 500 - 1) < 1e-6)

# The grid audit against aggregate 1.2.0's sharpen: from bs 1, log2 13 the
# capped lognormal moves to bs 4; from 0.5, 2^10 the gamma book grows to 11.
capped <- policy_limits_distribution(lognormal_from_mean_cv(100, 1.5), 1000, 1)
a <- audit_grid(poisson_count(100), capped, step = 1, log2 = 13)
stopifnot(a$step == 4, a$log2 == 13, !a$grew, a$probed, a$cells$selected[a$selected])
stopifnot(abs(a$score / 0.035326 - 1) < 1e-4, a$cells$note[1] == "center")
stopifnot(abs(a$cells$aggregate_cv[a$selected] / 0.020144 - 1) < 1e-4)
gb <- audit_grid(poisson_count(50), gamma_distribution(4, 2.5), step = 0.5, log2 = 10)
stopifnot(gb$step == 1, gb$log2 == 11, gb$grew, abs(gb$score / 0.722254 - 1) < 1e-5)
one <- audit_portfolio_grid(list(list(poisson_count(50), gamma_distribution(4, 2.5))),
                            step = 0.5, log2 = 10)
stopifnot(isTRUE(all.equal(one$cells$score, gb$cells$score)))
units <- list(list(poisson_count(10), gamma_distribution(2, 50)),
              list(poisson_count(2), gamma_distribution(0.5, 1000)))
stopifnot(recommend_portfolio_grid(units, log2 = 12)$points <= 2^12)
stopifnot(audit_portfolio_grid(units, log2 = 12)$score < 0.5)
stopifnot(inherits(try(audit_grid(poisson_count(5), capped, step = -1, log2 = 10),
                       silent = TRUE), "try-error"))

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
# #596: has_times, as Python's EventSet.has_times.
stopifnot(dated@has_times, !ev@has_times)
rp <- draw_matrix(apply_tower(reinsurance_tower(list(one)), dated))[, 4]
stopifnot(abs(mean(rp) - 1) < 4 * sd(rp) / sqrt(n), length(event_times(ev, 1)) == 0)

# The annual limit shared pro rata across the year's events.
order <- xol_layer("10x5", 10, 5, reinstatements = 1)
pro <- xol_layer("10x5", 10, 5, reinstatements = 1, allocation = "pro_rata")
stopifnot(order@allocation == "event_order", pro@allocation == "pro_rata")
close(ceded_by_event(order, c(15, 20, 10)), c(10, 10, 0), 1e-12)
close(ceded_by_event(pro, c(15, 20, 10)), c(8, 8, 4), 1e-12)
stopifnot(inherits(try(xol_layer("f", 10, 5, allocation = "by_size"), silent = TRUE), "try-error"))
stopifnot(inherits(try(xol_layer("f", 10, 10, premium = 2, reinstatement_rates = 1,
                                 pro_rata_time = TRUE, allocation = "pro_rata"),
                       silent = TRUE), "try-error"))
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

# Deficits and credits carried over a contract term.
pcc <- with_profit_commission_carry_forward(with_profit_commission(
  with_premium_rate(quota_share("QS", 1), 1, 100), 0.5, 0.2))
stopifnot(is.null(pcc@profit_commission_carry$years), is.null(pc@profit_commission_carry))
close(profit_commissions_over(pcc, c(120, 70, 30, 85, 50)), c(0, 0, 10, 0, 12.5), 1e-12)
pc1 <- with_profit_commission_carry_forward(with_profit_commission(
  with_premium_rate(quota_share("QS", 1), 1, 100), 0.5, 0.2), years = 1)
stopifnot(pc1@profit_commission_carry$years == 1)
close(profit_commissions_over(pc1, c(120, 70, 30, 85, 50)), c(0, 0, 25, 0, 12.5), 1e-12)
ss <- with_sliding_scale(with_premium_rate(quota_share("QS", 1), 1, 100), c(0.4, 0.2), c(0.5, 0.7))
ssc <- with_sliding_scale_carry_forward(ss, deficit = FALSE, credit = TRUE)
stopifnot(identical(ssc@sliding_scale_carry, list(deficit = FALSE, credit = TRUE, years = NULL)))
close(ceding_commissions_over(ssc, c(90, 40, 30, 60)), c(20, 40, 40, 40), 1e-12)
stopifnot(inherits(try(with_profit_commission_carry_forward(qs), silent = TRUE), "try-error"))
stopifnot(inherits(try(with_sliding_scale_carry_forward(ss, FALSE, FALSE), silent = TRUE), "try-error"))
ev <- simulate_events(poisson_count(3), lognormal_from_mean_cv(10, 1.5), 3 * 1000, seed = 11)
yrs <- term_years(ev, 3)
stopifnot(length(yrs) == 3, yrs[[2]]@n_sims == 1000)
stopifnot(identical(events(yrs[[2]], 5), events(ev, 1005)))
ptc <- with_profit_commission_carry_forward(with_profit_commission(
  with_premium_rate(quota_share("QS", 1), 1, 30), 0.3, 0.25))
term <- apply_tower_term(reinsurance_tower(list(ptc)), yrs)
stopifnot(identical(term@dims, c("kind", "layer", "year")))
tm <- draw_matrix(term)
stopifnot(ncol(tm) == 3 * 4)
# Each path's commissions are the layer's over its own ceded losses; year 1's
# match apply_tower alone.
for (j in c(1, 17, 500)) {
  close(tm[j, c(4, 8, 12)], profit_commissions_over(ptc, tm[j, c(2, 6, 10)]), 1e-12)
}
# Rank-correlated underwriting years keep each year's distribution.
ev2 <- simulate_events(poisson_count(2), lognormal_from_mean_cv(10, 1), 2000, seed = 3)
one <- sort(draw_matrix(total(ev2))[, 1])
cy <- correlated_years(ev2, 3, 0.5, seed = 5)
ct <- sapply(cy, function(y) draw_matrix(total(y))[, 1])
for (t in 1:3) stopifnot(identical(sort(ct[, t]), one))
stopifnot(abs(cor(ct[, 1], ct[, 2], method = "spearman") - 0.5) < 0.05)
cm <- correlated_years(ev2, 3, matrix(c(1, .5, .5, .5, 1, .5, .5, .5, 1), 3), seed = 5)
stopifnot(identical(events(cm[[3]], 9), events(cy[[3]], 9)))
stopifnot(length(correlated_years(ev2, 2, 0.3, kind = "pearson", draws = "blocks")) == 2)
cal <- calibrate_year_correlation(ev2, 3, 0.7, seed = 5)
stopifnot(abs(cal$ratio - 0.7) < 1e-3, length(cal$years) == 3)
cmean <- rowMeans(sapply(cal$years, function(y) draw_matrix(total(y))[, 1]))
close(sd(cmean) / sd(one), cal$ratio, 1e-9)
stopifnot(abs(year_correlation_for_ratio(3, 1 / sqrt(3))) < 1e-12)
stopifnot(inherits(try(correlated_years(ev2, 3, 0.5, kind = "kendall"), silent = TRUE), "try-error"))
stopifnot(inherits(try(year_correlation_for_ratio(1, 0.5), silent = TRUE), "try-error"))
alone <- draw_matrix(apply_tower(reinsurance_tower(list(ptc)), yrs[[1]]))
stopifnot(isTRUE(all.equal(unname(alone), unname(tm[, 1:4]))))
agg <- apply_tower_term(reinsurance_tower(list(pcc)), lapply(yrs[1:2], total))
stopifnot(ncol(draw_matrix(agg)) == 8)
stopifnot(identical(tower_to_json(tower_from_json(tower_to_json(reinsurance_tower(list(ssc))))),
                    tower_to_json(reinsurance_tower(list(ssc)))))

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
# The same fit from the compound_moments() result, by position or by name.
stopifnot(identical(moment_approximation(m, "shifted_gamma"), fit),
          identical(moment_approximation(m, kind = "shifted_gamma"), fit),
          identical(moment_approximation(m)$kind, "shifted_lognormal"))
stopifnot(inherits(try(moment_approximation(m, g), silent = TRUE), "try-error"),
          inherits(try(moment_approximation(poisson_count(10)), silent = TRUE), "try-error"),
          inherits(try(moment_approximation(c(1, 2, 3)), silent = TRUE), "try-error"))
q <- approximation_quantile(fit, c(0.5, 0.99))
stopifnot(all(abs(approximation_cdf(fit, q) - c(0.5, 0.99)) < 1e-10))
stopifnot(inherits(try(compound_moments(poisson_count(2), pareto(100, 2.5)), silent = TRUE),
                   "try-error"))
# The normal power and Wilson-Hilferty approximations (#329), and the third
# moment of a layer.
z99 <- qnorm(0.99)
sd_s <- m[["mean"]] * m[["cv"]]
np_fit <- moment_approximation(m, "normal_power")
stopifnot(abs(approximation_quantile(np_fit, 0.99) /
  (m[["mean"]] + sd_s * (z99 + m[["skew"]] * (z99^2 - 1) / 6)) - 1) < 1e-12)
wh_fit <- moment_approximation(m, "wilson_hilferty")
stopifnot(abs(approximation_cdf(wh_fit, approximation_quantile(wh_fit, 0.9)) - 0.9) < 1e-12)
lt <- layer_third_moment(gamma_distribution(1, 2), c(3, Inf), 0)
stopifnot(abs(lt[1] / (48 * (1 - exp(-1.5) * (1 + 1.5 + 1.5^2 / 2))) - 1) < 1e-10, abs(lt[2] / 48 - 1) < 1e-10)

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
# Finite-horizon ruin: a random walk (+1 with 0.6, -1 with 0.4) is ruined
# within 2000 periods with the eventual (2 / 3)^(u + 1).
walk <- finite_horizon_ruin(grid_distribution(1, c(0, 0, 1)), 1, 2000,
  frequency = bernoulli_count(0.4), points = 64, p = 0.1)
stopifnot(identical(dim(walk$psi), c(64L, 2000L)))
stopifnot(max(abs(walk$psi[1:5, 2000] - (2 / 3)^(1:5))) < 1e-12, abs(walk$psi[1, 3] - 0.496) < 1e-12)
stopifnot(walk$capital[1, 2000] == 5, all(diff(walk$psi[1, ]) >= -1e-15))
# Poisson in sub-periods approaches exp(-u / 3) / 1.5 from below.
gexp <- discretize(gamma_from_mean_cv(1, 1), step = 0.05, points = 2^12, method = "rounding")
fin <- finite_horizon_ruin(gexp, 1.5, 60, rate = 1, sub_periods = 30)
stopifnot(fin$psi[21, 1] < fin$psi[21, 20], fin$psi[21, 20] < fin$psi[21, 60])
stopifnot(fin$psi[21, 60] < exp(-1 / 3) / 1.5, fin$psi[21, 60] > exp(-1 / 3) / 1.5 - 0.03)
stopifnot(inherits(try(finite_horizon_ruin(gexp, 1.5, 3), silent = TRUE), "try-error"))
stopifnot(inherits(try(finite_horizon_ruin(gexp, 1.51, 3, rate = 1), silent = TRUE), "try-error"))
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

# pnl_program's rates and an engine's stated premium (aggregate 1.0.1).
local({
  p <- pnl_program(405.97794085441933, 0.9, 0.25, data.frame(
    ceded_loss = c(86.78504710067914, 8.144345556686755), share = 1, combined_ratio = c(0.6, 0.5),
    aggregate = c(FALSE, TRUE)))
  stopifnot(identical(p$rates, c(0.17747858, 0.024241071)))
  stopifnot(abs(pnl_program_stated(1000, 0.25)$premium - 1000 / 0.75) < 1e-9)
  k <- pnl_program(406, 0.9, 0.25, data.frame(share = 0.5, stated = 20))
  stopifnot(is.na(k$rates))
})

# A loss-basis expense on the consolidated face, at each year's gross loss.
local({
  gross <- list(c(2, 3, 4, 5, 6, 8), c(0.25, 0.3, 0.09, 0.2, 0.12, 0.04))
  cover <- reinsurance_tower(list(with_deposit_premium(aggregate_stop_loss("SL", 3, 5), 1)))
  book <- pnl_book("B", 10, list(pnl_expense("loss", 0.05)))
  c1 <- book_ledger(book, gross, cover)
  w <- book_ledger(book, gross, cover, "walk")
  sdv <- function(d) sqrt(d@ptr$variance())
  stopifnot(abs(sdv(ledger_result(c1)) - sdv(ledger_result(w))) < 1e-12)
  m <- book_ledger(pnl_book("B", 10, list(pnl_expense("loss", 0.05)), expense_at_expected_loss = TRUE),
                   gross, cover)
  stopifnot(abs(mean(ledger_result(m)) - mean(ledger_result(c1))) < 1e-12)
})

# The stitched peel of several occurrence layers (aggregate 1.0.1).
local({
  sev <- grid_distribution(1, c(0, 0.4, 0.3, 0, 0.2, 0, 0.1))
  occ <- reinsurance_tower(list(
    with_deposit_premium(xol_layer("occ 1 xs 1", 1, 1), 1),
    with_ceding_commission(with_deposit_premium(xol_layer("occ 2 xs 2", 2, 2), 1), 0.2)))
  book <- pnl_book("S", 20, list(pnl_expense("premium", 0.1)))
  peel <- book_occurrence_peel(book, poisson_count(2), sev, occ, points = 64)
  stopifnot(ledger_is_stitched(peel))
  s <- ledger_sheet(peel)
  stopifnot(!("k01" %in% names(s)))
  stopifnot(abs(s$SD[s$row == "net through occ 1 xs 1"] - 1.897366596) < 1e-8)
  stopifnot(abs(s$SD[s$step == "All" & s$label == "Impact"] + 2.201414) < 1e-6)
})

# Issue #193: select the ceded part of a tower's result before joining.
local({
  ev <- simulate_events(poisson_count(2), lognormal_from_mean_cv(3e6, 1.5), 2000, seed = 7)
  res <- apply_tower(reinsurance_tower(list(xol_layer("5x5", 5e6, 5e6))), ev)
  ceded <- select_components(res, "kind", "ceded")
  by_kind <- aggregate(res, keep = "kind")
  stopifnot(abs(mean(ceded) - mean(marginal(by_kind, list("ceded")))) < 1e-6 * mean(ceded))
  stopifnot(inherits(try(select_components(res, "kind", "none"), silent = TRUE), "try-error"))
})

# Results by event label.
lab <- event_set_from_long(c(1, 1, 2, 2), c(15, 20, 10, 4), 2,
                           labels = list(peril = c("wind", "quake", "quake", "wind")))
stopifnot(identical(lab@label_dims, "peril"), identical(event_labels(lab, 2)$peril, c("quake", "wind")))
stopifnot(identical(lab@label_combinations$peril, c("quake", "wind")))
res <- apply_tower(reinsurance_tower(list(xol_layer("10x5", 10, 5, aggregate_limit = 15))), lab)
stopifnot(identical(res@dims, c("kind", "layer", "peril")))
plain <- apply_tower(reinsurance_tower(list(xol_layer("10x5", 10, 5, aggregate_limit = 15))),
                     events_from_years(list(c(15, 20), c(10, 4))))
close(draw_matrix(aggregate(res, keep = "kind")), draw_matrix(aggregate(plain, keep = "kind")), 1e-12)
stopifnot(is.null(event_labels(events_from_years(list(1)), 1)))
mixed <- merge_event_sets(lab, events_from_years(list(1, 2)))
stopifnot(identical(mixed@label_combinations$peril, c("", "quake", "wind")))
stopifnot(inherits(try(with_labels(lab, list(kind = rep("a", 4))), silent = TRUE), "try-error"))

# A capped total carried back onto its components.
st_pd <- predictive_distribution(matrix(c(6, 3, 9, 1), ncol = 2), data.frame(class = c("motor", "home")))
st_m <- draw_matrix(scale_to_totals(st_pd, pmin(rowSums(draw_matrix(st_pd)), 10)))
stopifnot(all(st_m == matrix(c(4, 3, 6, 1), ncol = 2)))
stopifnot(inherits(try(scale_to_totals(st_pd, 1), silent = TRUE), "try-error"))

# Plain annual losses through apply_tower (#477): a loss-ratio cap of 120%
# on a premium of 100 with a commission sliding from 30% at 50% to 10% at 90%.
cover <- with_sliding_scale(
  with_premium_rate(aggregate_stop_loss("cover", 120, 0), 1, 100),
  c(0.3, 0.1), c(0.5, 0.9)
)
annual <- apply_tower(reinsurance_tower(list(cover)), c(40, 70, 150))
am <- draw_matrix(annual)
stopifnot(identical(as.vector(am[, 2]), c(40, 70, 120)))
close(as.vector(am[, 4]), c(30, 20, 10), 1e-12)
close(as.vector(am[, 4]), ceding_commission_for(cover, am[, 2]), 1e-12)
stopifnot(inherits(try(apply_tower(reinsurance_tower(list(cover)), c(1, -1)), silent = TRUE), "try-error"))

# A year loss table read and tiled (#484): year 1 by slot, the file's order
# breaking the tie in slot 5, where one event hits two classes.
ylt_path <- tempfile(fileext = ".csv")
writeLines(c("year,peril,slot,class,value,item", "1,wind,5,home,10,a", "1,wind,5,auto,4,b",
             "2,quake,1,home,30,c", "1,wind,2,home,2,d"), ylt_path)
yev <- read_ylt(ylt_path, labels = c("peril", "class"), n_years = 3, n_sims = 6)
stopifnot(identical(yev@n_sims, 6), identical(events(yev, 1), c(2, 10, 4)),
          identical(events(yev, 2), 30), length(events(yev, 3)) == 0,
          identical(events(yev, 4), events(yev, 1)))
stopifnot(identical(event_labels(yev, 1)$class, c("home", "home", "auto")))
stopifnot(startsWith(yev@input_hash, "blake3:"),
          identical(provenance(total(yev))$input_hash, yev@input_hash))
# A file of negative losses and its sign-flipped copy give one event set (#593).
neg_path <- tempfile(fileext = ".csv")
writeLines(c("year,peril,slot,class,value,item", "1,wind,5,home,-10,a", "1,wind,5,auto,-4,b",
             "2,quake,1,home,-30,c", "1,wind,2,home,-2,d"), neg_path)
nev <- read_ylt(neg_path, labels = c("peril", "class"), n_years = 3, losses_negative = TRUE)
stopifnot(identical(events(nev, 1), c(2, 10, 4)), identical(events(nev, 2), 30))
stopifnot(inherits(try(read_ylt(neg_path, labels = c("peril", "class")), silent = TRUE), "try-error"),
          inherits(try(read_ylt(ylt_path, labels = c("peril", "class"), losses_negative = TRUE),
                       silent = TRUE), "try-error"))
neg_layer <- xol_layer("10x5", 10, 5)
stopifnot(ceded(neg_layer, 12) == 7,
          inherits(try(ceded(neg_layer, c(12, -3)), silent = TRUE), "try-error"),
          inherits(try(ceded_by_event(neg_layer, -1), silent = TRUE), "try-error"))
e <- tryCatch(read_ylt(ylt_path, labels = c("peril", "class"), unique_by = "peril"), error = identity)
stopifnot(grepl("rows 1 and 2", conditionMessage(e)))
e <- tryCatch(read_ylt(ylt_path, unique_by = character()), error = identity)
stopifnot(grepl("repeat year 1, slot 5", conditionMessage(e)))
e <- tryCatch(read_ylt(ylt_path, labels = c("peril", "class"), expected = list(peril = "wind")),
              error = identity)
stopifnot(grepl("not one of", conditionMessage(e)))
e <- tryCatch(read_ylt(ylt_path, labels = c("peril", "class"), n_sims = 5), error = identity)
stopifnot(grepl("whole multiple", conditionMessage(e)))
rev <- read_ylt(ylt_path, labels = c("peril", "class"), n_sims = 7, expand = "resample", seed = 3)
rev2 <- event_set_from_ylt(c(1, 1, 2, 1), c(10, 4, 30, 2), slots = c(5, 5, 1, 2),
                           labels = list(class = c("home", "auto", "home", "home")), n_sims = 7,
                           expand = "resample", seed = 3)
stopifnot(identical(lapply(1:7, function(i) events(rev, i)), lapply(1:7, function(i) events(rev2, i))))
stopifnot(is.null(rev2@input_hash))

# #594: a time column orders the rows and feeds an hours clause by peril.
h <- 1 / 8760
timed_path <- tempfile(fileext = ".csv")
writeLines(c("year,slot,peril,time,value",
             sprintf("1,1,wind,%.17g,20", 0.5 + 30 * h),
             sprintf("1,2,flood,%.17g,7", 0.5 + 10 * h),
             "1,3,wind,0.5,10", "2,1,wind,0.25,3"), timed_path)
tev <- read_ylt(timed_path, time = "time", labels = "peril")
stopifnot(identical(events(tev, 1), c(10, 7, 20)),
          isTRUE(all.equal(event_times(tev, 1), c(0.5, 0.5 + 10 * h, 0.5 + 30 * h), tolerance = 0)))
g <- hours_clause(tev, 72, by = "peril")
stopifnot(identical(events(g, 1), c(30, 7)), identical(event_labels(g, 1)$peril, c("wind", "flood")))
stopifnot(identical(events(hours_clause(tev, 72), 1), 37), is.null(event_labels(hours_clause(tev, 72), 1)))
e <- tryCatch(hours_clause(tev, 72, labels = "agree"), error = identity)
stopifnot(grepl("labelled", conditionMessage(e)))
stopifnot(identical(event_labels(hours_clause(tev, 72, labels = "largest"), 1)$peril, "wind"))
days_path <- tempfile(fileext = ".csv")
writeLines(c("year,day,hour,value", "1,200,6,5", "1,1,0,1"), days_path)
dev <- read_ylt(days_path, slot = NULL, day = "day", hour = "hour")
stopifnot(identical(events(dev, 1), c(1, 5)),
          identical(event_times(dev, 1), c(0, (199 * 24 + 6) / 8760)))
e <- tryCatch(read_ylt(days_path, slot = NULL, time = "day", day = "day"), error = identity)
stopifnot(grepl("not both", conditionMessage(e)))

# #483: a reorder's index keeps a class's events with its totals, ties and all.
n <- 600
yrs <- lapply(0:(n - 1), function(i) if (i %% 3 == 0) numeric(0) else c(10 * (i %% 4), i %% 2))
large <- event_set_from_long(rep(seq_len(n), lengths(yrs)), unlist(yrs), n)
tot <- vapply(yrs, sum, 0)
pd <- predictive_distribution(cbind(tot, ((0:(n - 1)) * 7919) %% 7), data.frame(class = c("large", "other")))
for (r in list(reorder_groups(pd, "class", matrix(c(1, 0.7, 0.7, 1), 2), 3, index = TRUE),
               reorder_groups_by_copula(pd, "class", gaussian_copula(matrix(c(1, 0.7, 0.7, 1), 2)), 3,
                                        index = TRUE))) {
  stopifnot(identical(names(r$index), c("large", "other")))
  ev <- permute_events(large, r$index$large)
  m <- draw_matrix(r$result)
  stopifnot(all(vapply(seq_len(n), function(i) sum(events(ev, i)), 0) == m[, 1]))
  stopifnot(all(m[, 2] == draw_matrix(pd)[r$index$other, 2]))
}
stopifnot(S7::S7_inherits(reorder_groups(pd, "class", matrix(c(1, 0.7, 0.7, 1), 2), 3), predictive_distribution))
stopifnot(inherits(try(permute_events(large, n + 1), silent = TRUE), "try-error"))
stopifnot(inherits(try(permute_events(large, 0), silent = TRUE), "try-error"))

# #483: an aggregate part, seen by quota shares and stop losses, skipped by per risk layers.
part <- predictive_distribution(matrix(c(1, 2, 3, 9, 38, 27), 3), data.frame(peril = c("wind", "fire")))
ap <- with_aggregate_part(with_labels(events_from_years(list(numeric(0), 50, c(5, 60))),
                                      list(peril = c("wind", "flood", "wind"))), part)
stopifnot(ap@has_aggregate_part, identical(ap@aggregate_part_labels$peril, c("wind", "fire")))
stopifnot(identical(ap@label_combinations$peril, c("fire", "flood", "wind")))
stopifnot(identical(aggregate_part(ap, 2), c(2, 38)))
xl <- xol_layer("30x25", 30, 25)
qs <- quota_share("QS", 0.5)
stopifnot(qs@sees_aggregate_part, !xl@sees_aggregate_part)
apr <- apply_tower(reinsurance_tower(list(xl, qs)), ap)
apm <- function(kind, layer, peril) {
  k <- apr@keys
  draw_matrix(apr)[, which(k$kind == kind & k$layer == layer & k$peril == peril)]
}
stopifnot(all(apm("ceded", "30x25", "wind") == c(0, 25, 30)))
stopifnot(all(apm("ceded", "30x25", "fire") == 0))
stopifnot(all(apm("ceded", "QS", "fire") == c(4.5, 19, 13.5)))
stopifnot(all(draw_matrix(total(ap))[, 1] == c(10, 90, 95)))
stopifnot(is.null(aggregate_part(events_from_years(list(1)), 1)))
stopifnot(inherits(try(with_aggregate_part(events_from_years(list(1)), part), silent = TRUE), "try-error"))

# A deal's deductions (#485): premium legs shift the total deductions
# without spreading them, and only the manager's fee sits outside NMBF.
deal <- with_premium_rate(aggregate_stop_loss("deal", 120, 0), 1, 100)
deal <- with_brokerage(with_sliding_scale(deal, c(0.35, 0.15), c(0.5, 0.9)), 0.025)
for (leg in list(c("tpa", 0.01), c("tax", 0.03), c("ri_cost", 0.02), c("fronting", 0.005))) {
  deal <- with_deduction(deal, leg[[1]], as.double(leg[[2]]))
}
deal <- with_deduction(deal, "ms_fee", 0.05, inside_nmbf = FALSE)
stopifnot(deal@brokerage == 0.025, is.null(deal@brokerage_scale))
stopifnot(identical(deal@deductions$inside_nmbf, c(TRUE, TRUE, TRUE, TRUE, FALSE)))
dtw <- reinsurance_tower(list(deal))
dres <- apply_tower(dtw, c(20, 45, 60, 75, 95, 150))
dr <- deal_ratios(dtw, dres, "deal")
dm <- draw_matrix(dr)
rcol <- function(k) dm[, which(dr@keys$ratio == k)]
stopifnot(all(abs(rcol("total_deductions") - rcol("ceding_commission") - rcol("brokerage") - 0.115) < 1e-12))
stopifnot(all(abs(rcol("nmbf") - rcol("margin") - 0.05) < 1e-12))
stopifnot(abs(sd(rcol("total_deductions")) - sd(rcol("ceding_commission"))) < 1e-12)
stopifnot(identical(tower_to_json(tower_from_json(tower_to_json(dtw))), tower_to_json(dtw)))
stopifnot(inherits(try(with_deduction(deal, "tax", 0.01), silent = TRUE), "try-error"))
stopifnot(inherits(try(with_deduction(deal, "x", 0.01, basis = "lots"), silent = TRUE), "try-error"))
stopifnot(inherits(try(deal_ratios(dtw, dres, "nope"), silent = TRUE), "try-error"))

# The scales read the loss ratio before an annual cap when asked (#477).
capped <- with_premium_rate(xol_layer("deal", 1000, 0, aggregate_limit = 120), 1, 100)
capped <- with_sliding_scale(capped, c(0.3, 0.1), c(0.5, 1.5))
before <- with_scales_before_annual_terms(capped)
stopifnot(before@scales_before_annual_terms, !capped@scales_before_annual_terms)
cres <- apply_tower(reinsurance_tower(list(before)), c(40, 140))
cm <- draw_matrix(cres)
ccol <- function(k) cm[, which(cres@keys$kind == k)]
stopifnot(all(ccol("ceded") == c(40, 120)), all(ccol("loss_before_terms") == c(40, 140)))
stopifnot(abs(ccol("ceding_commission")[2] - 100 * (0.3 - 0.2 * 0.9)) < 1e-12)
held <- with_brokerage_within_commission_range(
  with_brokerage_scale(with_sliding_scale(with_premium_rate(quota_share("QS", 0.5), 1, 200),
                                          c(0.3, 0.1), c(0.5, 0.7)), c(0.03, 0.01), c(0.5, 0.9))
)
stopifnot(abs(brokerage_for(held, 90) - 2) < 1e-12, held@brokerage_within_commission_range)

# A deal's scale reads its sections' loss ratio together ([deal-scale]).
sa <- with_sliding_scale(with_premium_rate(aggregate_stop_loss("A", 50, 0), 1, 100), c(0.35, 0.15), c(0.5, 0.9))
sb <- with_sliding_scale(with_premium_rate(aggregate_stop_loss("B", Inf, 50), 1, 100), c(0.35, 0.15), c(0.5, 0.9))
dtw2 <- with_deal_scale(reinsurance_tower(list(sa, sb)), "deal", c("A", "B"),
  commission = c(0.2, 0.1), loss_ratio = c(0.5, 0.9), brokerage = 0.01, carry_deficit = TRUE)
stopifnot(identical(dtw2@deal_names, "deal"))
dres2 <- apply_tower(dtw2, 160)
dm2 <- draw_matrix(dres2)
dcol <- function(k, l) dm2[, which(dres2@keys$kind == k & dres2@keys$layer == l)]
stopifnot(abs(dcol("ceding_commission", "deal") - 25) < 1e-12, abs(dcol("brokerage", "deal") - 2) < 1e-12)
stopifnot(abs(dcol("ceding_commission", "A") - 35) < 1e-12)
dr2 <- deal_ratios(dtw2, dres2, "deal")
dm3 <- draw_matrix(dr2)
stopifnot(abs(dm3[, which(dr2@keys$ratio == "ceding_commission")] - 75 / 200) < 1e-12)
stopifnot(identical(tower_to_json(tower_from_json(tower_to_json(dtw2))), tower_to_json(dtw2)))
# The reinsurer's ledger of the deal ([deal-ledger]): 200 - 160 - 75 - 2 = -37.
dl <- deal_ledger(dtw2, list(160, 1), "deal")
stopifnot(abs(mean(ledger_result(dl)) + 37) < 1e-12)
stopifnot(abs(mean(ledger_row(dl, "Brokerage")) + 2) < 1e-12)
stopifnot(inherits(try(deal_ledger(dtw2, list(160, 1), "C"), silent = TRUE), "try-error"))
# On an annual loss grid ([deal-grid]): 40 or 160, equally likely.
dprobs <- numeric(17)
dprobs[c(5, 17)] <- 0.5
dg <- tower_on_annual_grid(dtw2, grid_distribution(10, dprobs))
dboth <- draw_matrix(apply_tower(dtw2, c(40, 160)))
dsim <- mean(dboth[, which(dres2@keys$kind == "ceding_commission" & dres2@keys$layer == "deal")])
stopifnot(abs(dg$expected_deal_ceding_commission[["deal"]] - dsim) < 1e-12)
stopifnot(abs(dg$expected_deal_brokerage[["deal"]] - 2) < 1e-12)
stopifnot(inherits(try(tower_on_grid(dtw2, poisson_count(2), grid_distribution(10, c(0, 0.5, 0.5)), 64), silent = TRUE), "try-error"))
stopifnot(inherits(try(with_deal_scale(reinsurance_tower(list(sa, sb)), "deal", "A", brokerage = 0.01), silent = TRUE), "try-error"))
stopifnot(inherits(try(with_deal_scale(reinsurance_tower(list(sa, sb)), "deal", c("A", "B")), silent = TRUE), "try-error"))
stopifnot(inherits(try(with_deal_scale(reinsurance_tower(list(sa, sb)), "d", c("A", "B"), brokerage = 0.01, carry_years = 1.5), silent = TRUE), "try-error"))

# A timeline shared by its underwriting years: an event straddling two years
# under risks attaching splits by exposure share; under losses occurring it
# falls whole to its accident year.
ev <- events_from_years(list(100), times = list(13.5 / 24))
tl <- timeline(ev, 24)
stopifnot(tl@months == 24, tl@n_sims == 1)
shares <- matrix(0, 2, 24)
shares[1, 14] <- 0.25
shares[2, 14] <- 0.75
rad <- timeline_years(tl, "risks_attaching", shares, sources = TRUE)
stopifnot(identical(events(rad$years[[1]], 1), 25), identical(events(rad$years[[2]], 1), 75))
stopifnot(identical(rad$sources, list(1, 1)))
lod <- timeline_years(tl, "losses_occurring", shares)
stopifnot(length(events(lod[[1]], 1)) == 0, identical(events(lod[[2]], 1), 100))
tw <- reinsurance_tower(list(xol_layer("XL", 50, 20)))
stopifnot(is.null(tw@cover_basis))
stopifnot(inherits(try(apply_tower_timeline(tw, tl, shares), silent = TRUE), "try-error"))
tw <- with_cover_basis(tw, "risks_attaching")
stopifnot(identical(tw@cover_basis, "risks_attaching"))
ceded <- draw_matrix(select_components(apply_tower_timeline(tw, tl, shares), "kind", "ceded"))
stopifnot(all(abs(ceded[1, ] - c(5, 50)) < 1e-12))
stopifnot(inherits(try(with_cover_basis(tw, "claims_made"), silent = TRUE), "try-error"))
stopifnot(inherits(try(timeline(ev, 0), silent = TRUE), "try-error"))
# Uniform exposure: mid-term months add to 1; stitched years equal LOD years.
ue <- uniform_exposure(2, 12)
stopifnot(identical(dim(ue), c(2L, 36L)), abs(sum(ue[, 13]) - 1) < 1e-12)
stopifnot(all(abs(rowSums(ue) - 12) < 1e-12))
ev3 <- with_uniform_times(simulate_events(poisson_count(2), lognormal(3, 1), 3 * 200, seed = 5))
yrs <- term_years(ev3, 3)
st <- timeline_from_years(yrs)
stopifnot(st@months == 36, st@n_sims == 200)
back <- timeline_years(st, "losses_occurring", uniform_exposure(3, 12))
for (t in 1:3) {
  stopifnot(all(abs(draw_matrix(total(back[[t]])) - draw_matrix(total(yrs[[t]]))) < 1e-9))
}
# An aggregate part on a timeline: each accident year keeps its own under
# losses occurring, follows the month's shares under risks attaching, and is
# spread by the pattern.
ap_year <- function(amount) {
  with_aggregate_part(events_from_years(list(numeric(0), numeric(0)), times = list(numeric(0), numeric(0))),
                      predictive_distribution(matrix(c(amount, 2 * amount), 2), data.frame(peril = "fire")))
}
ap_years <- list(ap_year(120), ap_year(60))
apt <- timeline_from_years(ap_years)
stopifnot(apt@has_aggregate_part, identical(apt@aggregate_pattern, rep(1, 12)), !apt@events@has_aggregate_part)
lo <- timeline_years(apt, "losses_occurring", uniform_exposure(2, 12))
stopifnot(identical(aggregate_part(lo[[1]], 2), 240), identical(aggregate_part(lo[[2]], 1), 60))
ra_shares <- rbind(c(rep(1, 12), rep(0.25, 12)), c(rep(0, 12), rep(0.75, 12)))
ra <- timeline_years(apt, "risks_attaching", ra_shares)
stopifnot(abs(aggregate_part(ra[[1]], 1) - 135) < 1e-9, abs(aggregate_part(ra[[2]], 2) - 90) < 1e-9)
dec <- timeline_from_years(ap_years, aggregate_pattern = c(rep(0, 11), 1))
late <- rbind(c(rep(1, 11), 0, rep(0, 12)), c(rep(0, 11), 1, rep(1, 12)))
stopifnot(abs(aggregate_part(timeline_years(dec, "risks_attaching", late)[[1]], 1)) < 1e-12)
stopifnot(inherits(try(timeline_from_years(ap_years, aggregate_pattern = rep(1, 11)), silent = TRUE), "try-error"))
term <- timeline(ap_year(240), 24, aggregate_pattern = rep(1, 12))
stopifnot(abs(aggregate_part(timeline_years(term, "losses_occurring", uniform_exposure(2, 12))[[2]], 1) - 120) < 1e-9)
qs_tl <- with_cover_basis(reinsurance_tower(list(quota_share("QS", 0.5))), "losses_occurring")
ceded <- draw_matrix(select_components(apply_tower_timeline(qs_tl, apt, uniform_exposure(2, 12)), "kind", "ceded"))
stopifnot(all(abs(ceded[1, ] - c(60, 30)) < 1e-9))
# Interlocking cuts each year's limit by its share of the event: 50 xs 20 on
# 25 and 75 of an event of 100 cedes 5 and 37.5, not 5 and 50.
tw <- with_cover_basis(reinsurance_tower(list(xol_layer("XL", 50, 20))), "risks_attaching")
stopifnot(identical(tw@interlocking, FALSE))
on <- with_interlocking(tw)
stopifnot(identical(on@interlocking, TRUE), identical(with_interlocking(on, FALSE)@interlocking, FALSE))
ceded <- draw_matrix(select_components(apply_tower_timeline(on, tl, shares), "kind", "ceded"))
stopifnot(all(abs(ceded[1, ] - c(5, 37.5)) < 1e-12))

# A layer scoped to one peril (#584): its annual limit is its own.
ev <- with_labels(
  events_from_years(list(c(20, 15), c(8, 30, 12))),
  list(peril = c("quake", "wind", "wind", "flood", "wind"))
)
xl <- xol_layer("10x5", 10, 5, aggregate_limit = 15)
stopifnot(is.null(xl@scope))
ws <- with_scope(xl, "peril", "wind")
stopifnot(identical(ws@scope, list(label = "peril", values = "wind")))
key <- list("ceded", "10x5", "wind")
# Wind has the whole limit (10, then 3 and 7); unscoped, quake and flood
# use it first and wind gets 5 a year.
stopifnot(mean(marginal(apply_tower(reinsurance_tower(list(ws)), ev), key)) == 10)
stopifnot(mean(marginal(apply_tower(reinsurance_tower(list(xl)), ev), key)) == 5)
w <- filter_events(ev, "peril", "wind")
stopifnot(w@n_sims == 2, identical(events(w, 1), 15), identical(events(w, 2), c(8, 12)))
stopifnot(mean(marginal(apply_tower(reinsurance_tower(list(xl)), w), key)) == 10)
tw <- reinsurance_tower(list(ws))
stopifnot(identical(tower_to_json(tower_from_json(tower_to_json(tw))), tower_to_json(tw)))
stopifnot(grepl("\"scope\"", tower_to_json(tw)))
stopifnot(inherits(try(apply_tower(tw, events_from_years(list(15))), silent = TRUE), "try-error"))
stopifnot(inherits(try(filter_events(ev, "class", "a"), silent = TRUE), "try-error"))

# A tower's ceded losses by event, for an occurrence retro (#590).
ev <- events_from_years(list(c(20, 8, 15), 4))
tw <- reinsurance_tower(list(xol_layer("10x5", 10, 5, aggregate_limit = 15)))
ce <- ceded_events(tw, ev, "10x5")
stopifnot(identical(events(ce, 1), c(10, 3, 2)), length(events(ce, 2)) == 0)
stopifnot(all.equal(sum(events(ce, 1)), mean(marginal(apply_tower(tw, ev), list("ceded", "10x5"))) * 2))
a <- ceded_events(reinsurance_tower(list(xol_layer("A", 20, 10))), events_from_years(list(30), times = list(0.2)), "A")
b <- ceded_events(reinsurance_tower(list(xol_layer("B", 10, 10))), events_from_years(list(c(25, 12)), times = list(c(0.2, 0.6))), "B")
book <- merge_event_sets(a, b)
stopifnot(identical(events(book, 1), c(20, 10, 2)))
stopifnot(identical(events(hours_clause(book, 72), 1), c(30, 2)))
stopifnot(inherits(try(ceded_events(tw, ev, "nope"), silent = TRUE), "try-error"))
