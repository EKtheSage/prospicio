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

# Towers save and load as JSON.
tw <- inuring_tower(list(list(quota_share("QS", 0.3), surplus_treaty("S", 1e6, 4)),
                         list(xol_layer("xl", 2e6, 1e6, premium = 3e5, reinstatement_rates = c(1, 0.5),
                                        pro_rata_time = TRUE))))
txt <- tower_to_json(tw)
back <- tower_from_json(txt)
stopifnot(identical(tower_to_json(back), txt), identical(back@layer_names, tw@layer_names))
stopifnot(inherits(try(tower_from_json("{}"), silent = TRUE), "try-error"))
