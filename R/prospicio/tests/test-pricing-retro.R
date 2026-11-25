library(prospicio)
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# A sample and the grid of its values: losses of 0, 50, 100 or 250.
years <- c(250, 0, 100, 50)
t <- insurance_charge(years, c(0, 0.5, 1, 2.5))
stopifnot(identical(names(t), c("entry_ratio", "charge", "savings")), attr(t, "expected") == 100)
stopifnot(near_abs(t$charge, c(1, 0.625, 0.375, 0), 1e-15))
stopifnot(near_abs(t$savings, c(0, 0.125, 0.375, 1.5), 1e-15), attr(t, "loss_elimination") == 0)
g <- insurance_charge(grid_distribution(50, c(0.25, 0.25, 0.25, 0, 0, 0.25)), c(0, 0.5, 1, 2.5))
stopifnot(near_abs(g$charge, t$charge, 1e-15), near_abs(g$savings, t$savings, 1e-15))

# The identity on an FFT aggregate, and the exponential's exp(-r).
sev <- discretize(gamma_from_mean_cv(10, 1), step = 0.25, points = 2^10)
agg <- compound_distribution(poisson_count(4), sev, 2^12, method = "fft")
r <- seq(0, 4, by = 0.25)
a <- insurance_charge(agg, r)
stopifnot(near_abs(a$charge - a$savings, 1 - r, 1e-12), all(diff(a$charge) < 0), all(diff(a$savings) > 0))

# Table L: four years with each occurrence limited to 100.
tl <- insurance_charge(c(110, 100, 0, 200), c(0, 1, 10), unlimited_expected = 132.5)
k <- 30 / 132.5
stopifnot(near_abs(attr(tl, "loss_elimination"), k, 1e-15), attr(tl, "expected") == 132.5)
stopifnot(near_abs(tl$charge, c(1, (67.5 / 4 + 30) / 132.5, k), 1e-15))
stopifnot(near_abs(tl$charge - tl$savings, 1 - c(0, 1, 10), 1e-14))

# A balanced plan collects the guaranteed-cost premium on average.
prem <- function(b, x, minimum, maximum) pmin(pmax(1.03 * (b$basic + 1.1 * x), minimum), maximum)
losses <- c(0, 10, 30, 50, 70, 90, 120, 150, 200, 280)
b <- retro_balance(losses, 100, 30, loss_conversion = 1.1, tax_multiplier = 1.03, minimum = 80, maximum = 220)
# The premium is the reinsurance module's retro_premium(), with the tax
# multiplier in its basic and lcm.
stopifnot(near_abs(retro_premium(losses, basic = 1.03 * b$basic, lcm = 1.03 * 1.1, minimum = 80, maximum = 220),
                   prem(b, losses, 80, 220), 1e-12))
stopifnot(near_abs(mean(prem(b, losses, 80, 220)), 1.03 * 130, 1e-9))
stopifnot(near_abs(b$expected_premium, 1.03 * 130, 1e-12))
stopifnot(near_abs(b$basic, 30 - 0.1 * 100 + 1.1 * 100 * b$net_insurance_charge, 1e-9))
# No maximum, and a plan that never binds.
b <- retro_balance(losses, 100, 30, 1.1, 1.03, minimum = 60)
stopifnot(is.infinite(b$entry_ratio_maximum), b$basic >= 0)
stopifnot(near_abs(mean(prem(b, losses, 60, Inf)), 1.03 * 130, 1e-9))
# A minimum that would need a negative basic premium is refused.
expect_error_like(retro_balance(losses, 100, 30, 1.1, 1.03, minimum = 120), "negative basic premium")
b <- retro_balance(losses, 100, 30, 1.1, 1.03, minimum = 0, maximum = 1000)
stopifnot(near_abs(b$basic, 20, 1e-9), b$entry_ratio_minimum < 0)
# A loss limit balances against Table L.
limited <- pmin(losses, 150)
b <- retro_balance(limited, 100, 30, 1.1, 1.03, 80, 200, unlimited_expected = 100)
stopifnot(near_abs(mean(prem(b, limited, 80, 200)), 1.03 * 130, 1e-9))

# Refusals.
expect_error_like(insurance_charge(numeric(), 1), "empty")
expect_error_like(insurance_charge(c(1, -2), 1), "non-negative")
expect_error_like(insurance_charge(c(10, 30), 1, unlimited_expected = 19), "limited")
expect_error_like(retro_balance(limited, 100, 30, 1.1, 1.03, 80, 200), "Table L")
expect_error_like(retro_balance(losses, 100, 30, 1.1, 1.03, 134, 300), "minimum")
expect_error_like(retro_balance(losses, 100, 30, 1.1, 1.03, 50, 133), "maximum")
