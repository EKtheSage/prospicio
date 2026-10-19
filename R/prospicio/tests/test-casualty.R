library(prospicio)

near <- function(a, b, tol = 1e-12) {
  if (!all(abs(a - b) <= tol * pmax(abs(b), 1))) {
    stop(sprintf("got %s, want %s", paste(format(a, digits = 17), collapse = " "),
                 paste(format(b, digits = 17), collapse = " ")), call. = FALSE)
  }
}
fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")

# Clark's workers compensation example: 750,000 xs 250,000 on four ELF
# curves, every printed factor and treaty loss, 13,695 and 3.42%.
clark <- data.frame(state = c("AL", "AL", "NJ", "NJ"), premium = 100e3,
                    elr = c(0.70, 0.70, 0.85, 0.85),
                    elf250 = c(0.030, 0.040, 0.070, 0.100),
                    elf1m = c(0.006, 0.008, 0.020, 0.035),
                    factor = c(0.024, 0.032, 0.050, 0.065),
                    loss = c(1680, 2240, 4250, 5525))
curves <- Map(function(a, b) elf_table(c(250e3, 1e6), c(a, b)), clark$elf250, clark$elf1m)
book <- casualty_profile(curves, limit = Inf, premium = clark$premium, elr = clark$elr,
                         curve = 1:4, tag = clark$state)
r <- casualty_rate(book, 750e3, 250e3, alae = "in_severity")
near(r$factor, clark$factor, 1e-14)
near(r$ceded, clark$loss, 1e-12)
near(sum(r$ceded), 13695, 1e-12)
stopifnot(round(100 * sum(r$ceded) / sum(r$premium), 2) == 3.42)
near(sum(r$ceded[r$tag == "NJ"]), 9775, 1e-12)

# ALAE has no default; on top is Clark's second formula.
sev <- pareto(100e3, 1.3)
one <- casualty_profile(list(sev), limit = 1e6, premium = 1, elr = 1)
stopifnot(fails(casualty_rate(one, 500e3, 250e3)))
stopifnot(fails(casualty_rate(one, 500e3, 250e3, alae = "pro_rata")))
stopifnot(fails(casualty_rate(one, 500e3, 250e3, alae = "in_severity", loading = 0.1)))
f <- casualty_rate(one, 500e3, 250e3, alae = "on_top", loading = 0.5)$ceded / 1.5
near(f, (lev(sev, 750e3 / 1.5) - lev(sev, 250e3 / 1.5)) / lev(sev, 1e6), 1e-12)

# The frequency form on the curve's own counts is the ILF form, and the
# sweep runs it over alphas.
p <- pareto(1e6, 1.4)
book <- casualty_profile(list(p, ilf_table(c(1e6, 5e6, 10e6), c(1, 1.5, 1.7))),
                         limit = c(10e6, 5e6), premium = c(2e6, 1e6), elr = c(0.65, 0.6),
                         curve = c(1, 2), tag = c("gl", "wf"))
n <- casualty_occurrences(book, 1e6)
near(n[2], 0.6e6 / 1.5 * 0.5 / 4e6, 1e-12)
gl <- casualty_profile(list(p), limit = 10e6, premium = 2e6, elr = 0.65)
near(casualty_rate_by_frequency(gl, 5e6, 5e6, alae = "in_severity", threshold = 1e6,
                                counts = casualty_occurrences(gl, 1e6), excess = p)$ceded,
     casualty_rate(gl, 5e6, 5e6, alae = "in_severity")$ceded, 1e-12)
s <- casualty_sweep(book, 5e6, 5e6, alae = "in_severity", threshold = 1e6, counts = n,
                    alphas = c(1, 1.4, 2))
stopifnot(identical(names(s), c("alpha", "ceded", "loss_cost")), nrow(s) == 3,
          all(diff(s$ceded) < 0))
near(s$loss_cost, s$ceded / 3e6, 1e-15)

# The anchor is the censored fit; the count test's tail is poisson_tail().
a <- pareto_anchor(1e6, c(1.1e6, 1.7e6, 2.4e6, 2.0e6, 0.5e6),
                   censored = c(FALSE, FALSE, FALSE, TRUE, FALSE))
stopifnot(a$exceedances == 4, a$censored == 1)
near(a$alpha, 3 / sum(log(c(1.1, 1.7, 2.4, 2.0))), 1e-14)
near(a$standard_error, a$alpha / sqrt(3), 1e-14)
t <- casualty_count_test(book, c(1e6, 3e6), threshold = 1e6, counts = n, excess = p,
                         exposure_years = 5, observed = c(1.2e6, 3.5e6, 0.9e6))
stopifnot(identical(t$observed, c(2, 1)))
lam <- 5 * casualty_expected_claims(book, 1e6, 1e6, n, p)
near(t$expected[1], lam, 1e-14)
near(t$p_value[1], poisson_tail(lam, 2), 1e-15)
near(poisson_tail(3, 0), exp(-3), 1e-15)
near(poisson_tail(2, 4), 1 - ppois(3, 2), 1e-14)

cat("test-casualty: ok\n")
