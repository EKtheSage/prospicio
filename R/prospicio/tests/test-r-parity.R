library(prospicio)
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# The R bindings that #977 moved onto the Rust core: each matches the
# closed form that R computed before, recycling as R does.

# Ratemaking helpers, vectorised and recycled.
near(trend_factor(c(0.02, 0.05), 2.5), (1 + c(0.02, 0.05))^2.5)
near(trend_factor(0.03, c(0, 1, 2)), 1.03^(0:2))
stopifnot(length(trend_factor(double(), 1)) == 0)
near(credibility_weighted(c(200, 100), c(0.3, 1), 300), c(270, 100))
near(trended_present_rates(c(0.10, 0), 0.06, trend_factor(0.05, 2)),
     (1 + c(0.10, 0)) / 1.06 * 1.05^2 - 1)

# The log pointwise predictive density, against its definition.
set.seed(1)
y <- rnorm(20)
mu <- rnorm(400, mean(y), 1 / sqrt(20))
ll <- sapply(y, function(yi) dnorm(yi, mu, 1, log = TRUE))
near(elpd_lppd(ll), sum(log(colMeans(exp(ll)))), rel = 1e-10)
stopifnot(elpd_lppd(ll) > elpd_loo(ll)$estimates[["elpd"]])

# The P&L stack: an impact row is its target less its base, and a row the
# stack lacks is an error from Rust.
src <- list(c(2, 8), c(0.5, 0.5))
a <- pnl_ledger(src, list(pnl_group("A", "sell", list(pnl_leg("P", 10)), list(pnl_leg("L", identity)))))
b <- pnl_ledger(src, list(pnl_group("B", "sell", list(pnl_leg("P", 12)), list(pnl_leg("L", identity)))))
s <- ledger_stack(list(a = a, b = b), list(c("b less a", "b", "a")))
stopifnot(identical(s$label, c("a", "b", "b less a")))
near(s$EX, c(5, 7, 2))
near(s$P50[3], s$P50[2] - s$P50[1])
expect_error_like(ledger_stack(list(a = a), list(c("x", "a", "nope"))), "nope")
stopifnot(nrow(ledger_stack(list(a = a))) == 1)
