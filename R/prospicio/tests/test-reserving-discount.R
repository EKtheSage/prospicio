suppressMessages(library(prospicio))
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# Discounted reserves and the IFRS 17 risk adjustment (#553, PR 2;
# docs/design/reserving-ifrs17.md): the R bindings of
# prospicio_reserving::discount, with the core checks of
# python/tests/test_reserving_discount.py.

long <- utils::read.csv(validation("data", "raa.csv"), comment.char = "#")
raa <- triangle(long, "origin", "development", "value")
n <- 10
latest_age <- n - seq_len(n)  # 0-based: origin w is at age index n - w

# A zero rate leaves every draw as it was.
boot <- mack_bootstrap(raa, n_sims = 500, seed = 3)
for (timing in c("mid", "end")) {
  d <- discounted_reserves(boot, basis = "paid", rate = 0, timing = timing)
  stopifnot(identical(d@timing, timing),
            identical(draw_matrix(d@unpaid), draw_matrix(boot@reserves)))
  near(draw_matrix(d@discounted), draw_matrix(boot@reserves), rel = 1e-9, abs = 1e-9)
  near(d@runoff[1], d@best_estimate, rel = 1e-9)
  stopifnot(d@runoff[length(d@runoff)] == 0)
  near(unname(d@discount_factors), rep(1, n), rel = 1e-9)
}
zero <- discounted_reserves(boot, basis = "paid", curve = discount_curve(c(1, 10), c(0, 0)))
near(zero@best_estimate, mean(boot@reserves), rel = 1e-9)

# A flat rate, every simulation on the chain ladder's pattern: the
# hand-built cash flow, mid-period.
cl <- chain_ladder(raa)
selected <- mack_bootstrap(raa, n_sims = 300, seed = 5, ldf = stats::setNames(unname(cl@ldf), raa@development[-n]),
                           ldf_kind = "ldf")
pattern <- c(vapply(seq_len(n - 1), function(k) 1 / prod(cl@ldf[k:(n - 1)]), 0), 1)
rate <- 0.03
unit <- vapply(latest_age, function(a) {
  rest <- 1 - pattern[a + 1]
  if (rest <= 1e-12) return((1 + rate)^-0.5)
  k <- (a + 1):(n - 1)
  sum((pattern[k + 1] - pattern[k]) / rest * (1 + rate)^-((k - a) - 0.5))
}, 0)
d <- discounted_reserves(selected, basis = "paid", rate = rate)
stopifnot(identical(d@timing, "mid"))
r <- draw_matrix(selected@reserves)
near(draw_matrix(d@discounted), sweep(r, 2, unit, `*`), rel = 1e-9, abs = 1e-9)
near(unname(d@discount_factors[-1]), unit[-1], rel = 1e-9)
stopifnot(unname(d@discount_factors[1]) == 1)
same <- discounted_reserves(selected, basis = "paid", curve = discount_curve(1, rate))
near(same@best_estimate, d@best_estimate, rel = 1e-12)

# Validation target 3: at the end of each period, each origin's factor is
# the pricing PaymentPattern's from its latest age (the same pattern in
# years, each share paid at its age).
end <- discounted_reserves(selected, basis = "paid", rate = 0.04, timing = "end")
years <- as.numeric(raa@development) / 12
for (w in 2:n) {
  a <- latest_age[w] + 1
  k <- (a + 1):n
  want <- sum((pattern[k] - pattern[k - 1]) / (1 - pattern[a]) * 1.04^-(years[k] - years[a]))
  near(unname(end@discount_factors[w]), want, rel = 1e-12)
}

# The incurred basis adds the case reserves, here 20% of the latest.
latest_incurred <- unname(cl@latest)
latest_paid <- 0.8 * latest_incurred
inc <- discounted_reserves(selected, basis = "incurred", rate = 0, paid_pattern = pattern,
                           latest_paid = latest_paid)
near(draw_matrix(inc@unpaid), sweep(r, 2, 0.2 * latest_incurred, `+`), rel = 1e-12, abs = 1e-9)
near(inc@best_estimate, mean(selected@reserves) + sum(0.2 * latest_incurred), rel = 1e-9)

# The risk adjustment and its implied confidence.
d <- discounted_reserves(boot, basis = "paid", rate = 0.03)
total <- rowSums(draw_matrix(d@discounted))
ra <- risk_adjustment(d, "confidence", p = 0.75)
near(ra$amount, quantile(total(d@discounted), 0.75) - d@best_estimate, rel = 1e-9)
stopifnot(ra$implied_confidence >= 0.75, ra$implied_confidence < 0.75 + 1 / 500,
          is.null(ra$capital), nrow(ra$segments) == 0)
tv <- risk_adjustment(d, "tvar", p = 0.75)
stopifnot(tv$amount > ra$amount)

# The cost of capital under both charges, by hand from the run-off.
run <- d@runoff
t <- seq_along(run) - 1
k0 <- 1000
coc <- 0.06
want_end <- coc * sum(k0 * run / run[1] * 1.03^-(t + 1))
want_start <- coc * sum(k0 * run / run[1] * 1.03^-t)
near(risk_adjustment(d, "cost_of_capital", rate = coc, capital = k0)$amount, want_end, rel = 1e-9)
near(risk_adjustment(d, "cost_of_capital", rate = coc, capital = c(amount = k0),
                     charge = "start")$amount, want_start, rel = 1e-9)
var <- risk_adjustment(d, "cost_of_capital", rate = coc, capital = c(var = 0.995))
near(var$capital, quantile(total(d@discounted), 0.995) - d@best_estimate, rel = 1e-9)
stopifnot(var$implied_confidence == mean(total <= d@best_estimate + var$amount))

# Segments: each stand-alone beside the total.
two <- triangle(rbind(cbind(long, lob = "A"), transform(cbind(long, lob = "B"), value = value / 2)),
                "origin", "development", "value", keys = "lob")
seg <- discounted_reserves(odp_bootstrap(two, n_sims = 400, seed = 2), basis = "paid", rate = 0.03)
stopifnot(identical(seg@segment_labels, list("A", "B")))
ra <- risk_adjustment(seg, "confidence", p = 0.9)
stopifnot(identical(ra$segments$segment, c("A", "B")),
          ra$segments$amount[1] > ra$segments$amount[2])

# The errors.
expect_error_like(discounted_reserves(boot, rate = 0.03), "basis is required")
expect_error_like(discounted_reserves(boot, basis = "paid"), "exactly one of rate")
expect_error_like(discounted_reserves(boot, basis = "incurred", rate = 0.03), "needs paid_pattern")
expect_error_like(discounted_reserves(boot, basis = "incurred", rate = 0.03,
                                      paid_pattern = pattern, latest_paid = latest_paid[-1]),
                  "latest paid")
expect_error_like(discount_curve(c(2, 1), c(0.03, 0.04)), "maturity")
msg <- tryCatch(discounted_reserves(mack(raa), basis = "paid", rate = 0.03),
                prospicio_not_supported = function(e) conditionMessage(e))
stopifnot(grepl("no simulations", msg, fixed = TRUE))
expect_error_like(risk_adjustment(d, "confidence"), "needs p")
expect_error_like(risk_adjustment(d, "cost_of_capital", rate = 0.06), "needs capital")
expect_error_like(risk_adjustment(d, "cost_of_capital", rate = 0.06, capital = c(lots = 1)),
                  "capital must be named")

# The curve.
curve <- discount_curve(c(1, 5), c(0.02, 0.04))
near(discount_curve_rate(curve, c(0.5, 3, 9)), c(0.02, 0.03, 0.04), rel = 1e-12)
near(discount_curve_factor(curve, 2), 1.025^-2, rel = 1e-12)
