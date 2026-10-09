library(prospicio)

close <- function(got, want, rel) {
  stopifnot(all(abs(got - want) <= rel * abs(want)))
}

d <- lognormal(7, 0.5)

# Parity with base R's lognormal functions.
close(mean(d), exp(7 + 0.5^2 / 2), 1e-13)
close(variance(d), (exp(0.5^2) - 1) * exp(2 * 7 + 0.5^2), 1e-12)
p <- c(0.01, 0.5, 0.995)
close(quantile(d, p), qlnorm(p, 7, 0.5), 1e-12)
close(cdf(d, c(500, 1500)), plnorm(c(500, 1500), 7, 0.5), 1e-12)

# S7 classes: read-only parameter properties, a shared abstract parent.
stopifnot(S7::S7_inherits(d, lognormal), S7::S7_inherits(d, distribution))
stopifnot(identical(d@meanlog, 7), identical(d@sdlog, 0.5))
stopifnot(inherits(tryCatch(d@sdlog <- 1, error = identity), "error"))
stopifnot(inherits(tryCatch(d@ptr <- 1, error = identity), "error"))
stopifnot(identical(capture.output(print(d)), "<lognormal> meanlog = 7, sdlog = 0.5"))

m <- lognormal_from_mean_cv(1000, 0.5)
close(mean(m), 1000, 1e-12)
close(sqrt(variance(m)), 500, 1e-12)
stopifnot(S7::S7_inherits(m, lognormal))

# Errors from Rust surface as ordinary R errors naming the R call.
e <- tryCatch(lognormal(0, -1), error = identity)
stopifnot(grepl("sdlog", conditionMessage(e)), identical(conditionCall(e)[[1]], quote(lognormal)))
e <- tryCatch(quantile(d, 2), error = identity)
stopifnot(grepl("probability 2", conditionMessage(e)), identical(conditionCall(e), quote(quantile(d, 2))))
e <- tryCatch(draws(d, 3, seed = -1), error = identity)
stopifnot(grepl("seed", conditionMessage(e)), identical(conditionCall(e), quote(draws(d, 3, seed = -1))))

# Same kernel as Rust and Python: these draws are pinned in all three.
x <- draws(lognormal(0, 1), 3, seed = 42, stream = 3)
stopifnot(identical(x, c(1.0007760893701914, 1.6293872534754683, 1.0763869265482304)))

# Severity: limited expected value, stop-loss, layer.
close(lev(d, 1500) + stop_loss(d, 1500), mean(d), 1e-12)
close(layer(d, 1000, 500), lev(d, 1500) - lev(d, 500), 1e-12)

# Claim counts against base R.
n <- poisson_count(3)
close(pmf(n, 0:9), dpois(0:9, 3), 1e-12)
close(cdf(n, 0:9), ppois(0:9, 3), 1e-12)
stopifnot(identical(quantile(n, 0.9), qpois(0.9, 3)))
nb <- negative_binomial_count(2.5, 1.5)
close(pmf(nb, 0:9), dnbinom(0:9, size = 2.5, prob = 1 / 2.5), 1e-12)
close(variance(negative_binomial_count_from_mean_variance(10, 30)), 30, 1e-12)
stopifnot(identical(draws(nb, 5, seed = 1), draws(nb, 5, seed = 1)))
e <- tryCatch(pmf(n, -1), error = identity)
stopifnot(grepl("non-negative whole", conditionMessage(e)), identical(conditionCall(e), quote(pmf(n, -1))))

# Grids: local moment matching keeps the limited mean; the report is attached.
g <- discretize(d, step = 100, points = 200)
stopifnot(S7::S7_inherits(g, grid_distribution), identical(g@report$method, "local_moment"))
close(sum(g@probs), 1, 1e-12)
close(mean(g), lev(d, 199 * 100), 1e-9)
close(g@report$mean_error, -stop_loss(d, 199 * 100), 1e-6)
stopifnot(mean(discretize(d, 100, 200, "lower")) <= mean(d), is.null(grid_distribution(1, c(0.5, 0.5))@report))
stopifnot(inherits(tryCatch(grid_distribution(1, c(0.5, 0.4)), error = identity), "error"))

# Sampled and the joint predictive distribution.
s <- sampled(c(1, 2, 3, 4))
stopifnot(identical(VaR(s, 0.5), 2), identical(TVaR(s, 0.5), 3.5))
pd <- predictive_distribution(
  matrix(c(0, 0, 0, 100, 0, 0, 100, 0), ncol = 2),
  data.frame(line = c("A", "B"))
)
stopifnot(identical(VaR(pd, 0.75), 100), identical(VaR(marginal(pd, list(line = "A")), 0.75), 0))
stopifnot(is.null(marginal(pd, list(line = "C"))))
lob <- predictive_distribution(
  rbind(c(1, 2, 10, 20), c(3, 4, 30, 40), c(5, 6, 50, 60)),
  data.frame(lob = c("Auto", "Auto", "Home", "Home"), origin = c(2023, 2024, 2023, 2024))
)
by_lob <- aggregate(lob, keep = "lob")
stopifnot(identical(by_lob@keys$lob, c("Auto", "Home")))
stopifnot(identical(draw_matrix(by_lob), rbind(c(3, 30), c(7, 70), c(11, 110))))
stopifnot(identical(total(lob)@draws, c(33, 77, 121)), identical(lob@keys$origin, c(2023, 2024, 2023, 2024)))
stopifnot(identical(provenance(lob)$model, "r"))
stopifnot("samplers" %in% names(provenance(lob)), is.null(provenance(lob)$samplers))
e <- tryCatch(aggregate(lob, keep = "state"), error = identity)
stopifnot(grepl("dimension", conditionMessage(e)), identical(conditionCall(e), quote(aggregate(lob, keep = "state"))))

cat("prospicio R tests passed\n")

# Save and load as JSON.
for (d in list(lognormal(7, 0.5), pareto(1e5, 1.5), pareto(1e5, 1.5, truncation = 1e7),
               piecewise_pareto(c(1, 10, 100), c(1.2, 1.8, 2.5)), log_affine_pareto(100, 1.5, gamma = 0.3),
               generalized_pareto(0.25, 3), gamma_distribution(2, 500), tweedie(1000, 2, 1.5),
               weibull_distribution(1.5, 1000), loglogistic_distribution(4, 900),
               mixture_distribution(c(0.7, 0.3), list(lognormal(7, 0.5), pareto(1e5, 2))),
               grid_distribution(0.5, c(0.1, 0.4, 0.3, 0.2)), sampled(c(3, 1, 2)))) {
  text <- dist_to_json(d)
  back <- dist_from_json(text)
  stopifnot(identical(class(back), class(d)), identical(dist_to_json(back), text),
            identical(mean(back), mean(d)))
}
stopifnot(inherits(try(dist_to_json(custom_distribution(function(x) 1 - exp(-x))), silent = TRUE), "try-error"))

# More claim counts, against aggregate 1.0.1 where it has them.
zt <- zero_truncated_count(poisson_count(2))
stopifnot(pmf(zt, 0) == 0, abs(mean(zt) - 2 / (1 - exp(-2))) < 1e-12)
zm <- zero_modified_count(negative_binomial_count(2, 1.5), 0.3)
stopifnot(abs(pmf(zm, 0) - 0.3) < 1e-15)
pig <- mixed_poisson_count(10, 0.5, mixing = "inverse_gaussian")
stopifnot(abs(pmf(pig, 0) - 0.0030337404) < 1e-10, abs(variance(pig) - 35) < 1e-9)
ney <- compound_poisson_count(2, poisson_count(3))
stopifnot(abs(mean(ney) - 6) < 1e-12)
stopifnot(abs(pmf(logarithmic_count(0.5), 1) - 0.5 / log(2)) < 1e-15)
stopifnot(mean(empirical_count(c(0.5, 0.25, 0.25))) == 0.75)
sev <- grid_distribution(1, c(0.1, 0.3, 0.25, 0.2, 0.1, 0.05))
a <- compound_distribution(zm, sev, 100, method = "panjer")
b <- compound_distribution(zm, sev, 100, method = "fft")
stopifnot(max(abs(a@probs - b@probs)) < 1e-12)
stopifnot(inherits(try(compound_distribution(pig, sev, 100, method = "panjer"), silent = TRUE), "try-error"))
