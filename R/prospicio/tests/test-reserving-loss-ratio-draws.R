suppressMessages(library(prospicio))

# Ultimates and loss ratios per simulation (#602, reserving-v02.md decision
# 26): element i is simulation i across origins, calls and fits with the
# same seed, and a tower on the draws cedes what it cedes on the fit's own
# distribution.

near <- function(a, b, rel = 1e-12) {
  if (!all(abs(a - b) <= rel * pmax(abs(b), 1))) {
    stop(sprintf("got %s, want %s (rel %g)", format(a, digits = 17), format(b, digits = 17), rel),
         call. = FALSE)
  }
}

validation <- function(...) file.path("validation", ...)
raa_long <- utils::read.csv(validation("data", "raa.csv"), comment.char = "#")
raa <- triangle(raa_long, "origin", "development", "value")

boot <- odp_bootstrap(raa, n_sims = 2000, seed = 602)
latest <- boot@chain_ladder@latest
reserves <- draw_matrix(boot@reserves)
for (j in seq_along(boot@origins)) {
  u <- ultimate_draws(boot, boot@origins[[j]])
  stopifnot(length(u) == 2000)
  near(u, latest[[j]] + reserves[, j])
}
# By label or integer; the loss ratio in the same order.
stopifnot(identical(ultimate_draws(boot, 1990), ultimate_draws(boot, "1990")))
near(loss_ratio_draws(boot, 1990, premium = 25000), ultimate_draws(boot, 1990) / 25000)
# The total, summed simulation by simulation.
total <- ultimate_draws(boot)
near(total, sum(latest) + rowSums(reserves))
near(loss_ratio_draws(boot, premium = 4e5), total / 4e5)
# The joint ultimates keep the reserves' layout.
ultimates <- ultimate_distribution(boot)
stopifnot(identical(dim(draw_matrix(ultimates)), dim(reserves)))
near(draw_matrix(ultimates)[, 10], ultimate_draws(boot, 1990))

# The same seed, the same order.
again <- odp_bootstrap(raa, n_sims = 2000, seed = 602)
stopifnot(identical(ultimate_draws(again, 1985), ultimate_draws(boot, 1985)))

# A tower on the total draws cedes what it cedes on the fit's ultimates.
adc <- reinsurance_tower(list(aggregate_stop_loss("adc", 30000, 220000)))
on_draws <- draw_matrix(apply_tower(adc, total))[, 2]
on_fit <- draw_matrix(apply_tower(adc, ultimates))[, 2]
near(on_draws, on_fit, 1e-12)
stopifnot(mean(on_draws) > 0, mean(on_draws) < 30000)

# Over segments the key holds the segment.
two <- raa_long[rep(seq_len(nrow(raa_long)), 2), ]
two$lob <- rep(c("A", "B"), each = nrow(raa_long))
two$value <- two$value * rep(c(1, 2), each = nrow(raa_long))
fits <- mack_bootstrap(triangle(two, "origin", "development", "value", keys = "lob"),
                       n_sims = 500, seed = 4)
l2 <- fits@chain_ladder@latest
r2 <- draw_matrix(fits@reserves)
near(ultimate_draws(fits, list("B", 1990)), l2[[20]] + r2[, 20])
stopifnot(inherits(try(ultimate_draws(fits, 1990), silent = TRUE), "try-error"))
stopifnot(inherits(try(loss_ratio_draws(fits, list("A", 1990), premium = 0), silent = TRUE),
                   "try-error"))
stopifnot(inherits(try(ultimate_draws(chain_ladder(raa)), silent = TRUE), "try-error"))
