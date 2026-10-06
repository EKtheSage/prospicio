suppressMessages(library(actuarialrs))

# Reports the values on failure, so a CI log shows what diverged.
near <- function(a, b, rel = 1e-12) {
  if (!all(abs(a - b) <= rel * pmax(abs(b), 1e-300))) {
    stop(sprintf("got %s, want %s (rel %g)", format(a, digits = 17), format(b, digits = 17), rel),
         call. = FALSE)
  }
}

# Pareto: closed forms and truncation.
p <- pareto(500, 2)
near(layer(p, 4000, 1000), 200)
near(mean(p), 1000)
near(survival(p, 1000), 0.25)
stopifnot(is.infinite(mean(pareto(1, 0.9))))
tp <- pareto(500, 2, truncation = 8000)
near(tp@truncation, 8000)
near(quantile(tp, 1), 8000)
near(layer_variance(tp, 1e4, 0), variance(tp), 1e-10)
stopifnot(inherits(try(pareto(1, 0), silent = TRUE), "try-error"))

# Fits.
f <- pareto_fit(c(1500, 2500, 4000, 10000), 1000, censored = c(FALSE, FALSE, FALSE, TRUE))
near(f@alpha, 3 / log(150))
pf <- piecewise_pareto_fit(c(1200, 1500, 2500, 6000), c(1000, 2000))
near(pf@alpha[1], 2 / (log(1.2) + log(1.5) + 2 * log(2)))

# Piecewise Pareto.
pp <- piecewise_pareto(c(1000, 2000), c(1, 2))
near(survival(pp, 4000), 0.125)
near(stop_loss(pp, 2000), 1000)
lp <- piecewise_pareto(c(1000, 2000), c(1, 2), truncation = 5000)
wd <- piecewise_pareto(c(1000, 2000), c(1, 2), truncation = 5000, truncation_type = "wd")
stopifnot(lp@truncation_type == "lp", wd@truncation_type == "wd")
stopifnot(survival(lp, 1500) != survival(wd, 1500))

# Log-affine and generalized Pareto.
d <- log_affine_pareto(1e6, 1.5, delta = 0.5)
near(local_alpha(d, 2e6), 2)
near(layer(log_affine_pareto(1000, 2.5, gamma = 0), 4000, 1000), layer(pareto(1000, 2.5), 4000, 1000))
stopifnot(inherits(try(log_affine_pareto(1, 1), silent = TRUE), "try-error"))
g <- generalized_pareto_riegel(1000, 2, 1.5)
near(survival(g, 2000), (7 / 3)^-1.5)
near(lev(g, 5000) + stop_loss(g, 5000), mean(g))

# Claim counts by dispersion.
stopifnot(S7::S7_inherits(claim_count(4, 1), poisson_count))
stopifnot(S7::S7_inherits(claim_count(4, 2.5), negative_binomial_count))
b <- claim_count(10, 0.3)
stopifnot(S7::S7_inherits(b, binomial_count), b@n == 15)
near(mean(b), 10)
near(sum(pmf(binomial_count(10, 0.3), 0:10)), 1)

# The Pareto family feeds discretization and aggregation.
sev <- piecewise_pareto(c(1000, 3000), c(1.2, 2), truncation = 50000)
gd <- discretize(sev, step = 100, points = 501)
near(mean(gd), mean(sev), 1e-12)
agg <- compound_distribution(binomial_count(20, 0.1), gd, points = 10001)
near(mean(agg), 2 * mean(gd), 1e-9)
ev <- simulate_events(claim_count(2, 1.5), pareto(1000, 2), n_sims = 10, seed = 1)
stopifnot(S7::S7_inherits(ev, event_set))
# Generalized Pareto fit, whole-distribution truncated fit, local Pareto.
g <- generalized_pareto_riegel(1000, 3, 1.5)
fit <- generalized_pareto_fit(draws(g, 50000, seed = 4), 1000)
stopifnot(abs(1000 / fit@beta / 3 - 1) < 0.05, abs(1 / fit@xi / 1.5 - 1) < 0.05)
truth <- piecewise_pareto(c(1000, 2500), c(1.2, 0.8), truncation = 60000, truncation_type = "wd")
wd <- piecewise_pareto_fit(draws(truth, 50000, seed = 5), c(1000, 2500), truncation = 60000,
                           truncation_type = "wd")
stopifnot(wd@truncation_type == "wd", all(abs(wd@alpha / c(1.2, 0.8) - 1) < 0.05))
exact <- log_affine_pareto(1000, 1.5, gamma = 0.4)
conv <- local_pareto_to_piecewise(1000, function(x) local_alpha(exact, x), rel_tolerance = 1e-5)
stopifnot(conv$max_relative_error <= 1e-5)
near(survival(conv$severity, 7777), survival(exact, 7777), 1.01e-5)
stopifnot(inherits(try(local_pareto_to_piecewise(1, function(x) stop("boom")), silent = TRUE),
                   "try-error"))

# Gamma and Tweedie.
g <- gamma_from_mean_cv(1000, 0.5)
near(g@shape, 4, 1e-12)
near(lev(g, 1500) + stop_loss(g, 1500), 1000, 1e-12)
near(survival(gamma_distribution(1, 3), 6), exp(-2), 1e-15)
near(log_density(gamma_distribution(1, 2), 3), -1.5 - log(2), 1e-15)
near(gamma_from_mean_dispersion(200, 0.25)@shape, 4, 1e-12)
y <- tweedie(500, 40, 1.6)
near(cdf(y, 0), exp(-y@lambda), 1e-15)
near(variance(y), 40 * 500^1.6, 1e-12)
near(lev(y, 800) + stop_loss(y, 800), 500, 1e-12)
z <- tweedie_from_poisson_gamma(y@lambda, y@severity@shape, y@severity@scale)
near(z@mean_param, 500, 1e-12)
near(log_density(y, 0), -y@lambda, 1e-15)
stopifnot(inherits(try(tweedie(1, 1, 2), silent = TRUE), "try-error"))
sg <- discretize(g, 50, 400)
near(mean(sg), lev(g, 399 * 50), 1e-9)

# Weibull and mixtures.
w <- weibull_distribution(1, 2)
near(stop_loss(w, 3), 2 * exp(-1.5), 1e-14)
near(quantile(w, cdf(w, 1.7)), 1.7, 1e-12)
mx <- mixture_distribution(c(0.9, 0.1), list(lognormal_from_mean_cv(1e4, 1), pareto(1e5, 2)))
near(mean(mx), 29000, 1e-12)
near(mx@weights, c(0.9, 0.1), 1e-15)
stopifnot(inherits(try(mixture_distribution(0.5, list(w)), silent = TRUE), "try-error"))
# A sampled distribution has no exact layer moments, so it is no component.
e <- try(mixture_distribution(1, list(sampled(c(1, 2, 3)))), silent = TRUE)
stopifnot(inherits(e, "try-error"), grepl("no exact layer moments", e))
gm <- discretize(mx, 1000, 2000)
near(mean(gm), lev(mx, 1999 * 1000), 1e-9)

# Loglogistic: shape 1 has F(x) = x / (x + scale), LEV = scale log(1 + u / scale).
ll <- loglogistic_distribution(1, 2)
near(cdf(ll, 3), 0.6, 1e-15)
near(lev(ll, 3), 2 * log(2.5), 1e-13)
stopifnot(is.infinite(mean(ll)))
gc <- loglogistic_distribution(1.5, 24)
near(cdf(gc, 24), 0.5, 1e-15)
near(lev(gc, 1e4) + stop_loss(gc, 1e4), mean(gc), 1e-12)
stopifnot(inherits(try(loglogistic_distribution(0, 1), silent = TRUE), "try-error"))

cat("actuarialrs R pareto tests passed\n")
