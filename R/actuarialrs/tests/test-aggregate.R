library(actuarialrs)

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

cat("actuarialrs R aggregate tests passed\n")
