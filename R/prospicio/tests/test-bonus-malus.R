library(prospicio)

# -1/+1 on two levels under a gamma Theta: pi_0 = (a / (a + lambda))^a and
# r_0 = a / (a + lambda), a = 1 / cv^2.
two <- bonus_malus(2, start = 1, up_per_claim = 1)
a <- 1 / 0.8^2
bm <- bonus_malus_portfolio(two, 0.12, cv = 0.8)
stopifnot(abs(bm$stationary[1] - (a / (a + 0.12))^a) < 1e-12)
stopifnot(abs(bm$relativity[1] - a / (a + 0.12)) < 1e-12, identical(bm$level, c(0, 1)))

# The SciPy reference: -1/+top on six levels at lambda 0.1, gamma cv 1.
top <- bonus_malus(6, start = 5, up_per_claim = 5)
stopifnot(identical(top@transitions[1, ], c(0, 5)), top@levels == 6, top@start == 5)
bm <- bonus_malus_portfolio(top, 0.1, cv = 1)
stopifnot(abs(bm$relativity[1] - 2 / 3) < 1e-12, abs(bm$relativity[6] - 1.9090909090908708) < 1e-10)
stopifnot(abs(sum(bm$stationary * bm$relativity) - 1) < 1e-12)

# Three a priori classes on -1/+3: balanced, increasing relativities.
scale <- bonus_malus(12, start = 7, up_per_claim = 3)
cl <- bonus_malus_portfolio(scale, c(0.06, 0.1, 0.18), cv = 1.3, weights = c(0.5, 0.3, 0.2))
stopifnot(abs(sum(cl$stationary) - 1) < 1e-12, abs(sum(cl$stationary * cl$relativity) - 1) < 1e-12)
stopifnot(all(diff(cl$relativity) > 0))
m <- bonus_malus_transition(scale, 0.1)
stopifnot(identical(dim(m), c(12L, 12L)), max(abs(rowSums(m) - 1)) < 1e-15)
pi <- bonus_malus_stationary(scale, 0.1)
stopifnot(max(abs(bonus_malus_after(scale, 0.1, 3000) - pi)) < 1e-12)
stopifnot(max(abs(pi %*% m - pi)) < 1e-14)
n <- bonus_malus_years(scale, 0.1, 0.01)
stopifnot(n > 0, sum(abs(bonus_malus_after(scale, 0.1, n) - pi)) / 2 < 0.01)
eta <- bonus_malus_efficiency(scale, c(0.05, 0.1), cl$relativity)
stopifnot(length(eta) == 2, all(eta > 0 & eta < 1))
h <- 1e-5
fd <- diff(log(bonus_malus_mean_relativity(scale, 0.1 * c(1 - h, 1 + h), cl$relativity))) /
  diff(log(0.1 * c(1 - h, 1 + h)))
stopifnot(abs(eta[2] - fd) < 1e-8)
ig <- bonus_malus_portfolio(scale, 0.1, cv = 0.8, mixing = "inverse_gaussian")
stopifnot(nrow(ig) == 12)

# A table: the level after k claims from each level, 0-based.
custom <- bonus_malus(start = 1, table = rbind(c(0, 1), c(0, 2), c(1, 2)))
stopifnot(custom@levels == 3, max(abs(bonus_malus_stationary(custom, 0.2) - bonus_malus_after(custom, 0.2, 500))) < 1e-12)
stopifnot(inherits(try(bonus_malus(4, start = 4, up_per_claim = 1), silent = TRUE), "try-error"))
stopifnot(inherits(try(bonus_malus_portfolio(scale, 0.1, cv = 1, mixing = "beta"), silent = TRUE), "try-error"))
stopifnot(inherits(try(bonus_malus_efficiency(scale, 0.1, 1:5), silent = TRUE), "try-error"))

cat("prospicio R bonus-malus tests passed\n")
