library(actuarialrs)

near <- function(a, b, rel = 1e-10) {
  if (!all(abs(a - b) <= rel * pmax(abs(b), 1e-300))) {
    stop(sprintf("got %s, want %s", paste(format(a, digits = 17), collapse = " "),
                 paste(format(b, digits = 17), collapse = " ")), call. = FALSE)
  }
}

# GLMs agree with stats::glm on the same formula. R's covariance uses the
# working weights of its last iteration, so it is refitted to a tight
# tolerance for a like-for-like comparison.
tight <- stats::glm.control(epsilon = 1e-14, maxit = 100)
d <- data.frame(claims = c(1, 2, 4, 2, 1, 3, 0, 5), region = c("N", "N", "S", "S", "W", "W", "N", "S"),
                age = c(30, 45, 50, 22, 61, 38, 27, 55), exposure = c(1, 2, 1, 1, 0.5, 1.5, 0.8, 1.2))
m <- glm_fit(claims ~ region + age + offset(log(exposure)), d, family = "poisson")
r <- stats::glm(claims ~ region + age + offset(log(exposure)), d, family = poisson(), control = tight)
near(coef(m), coef(r), 1e-8)
near(m@std_errors, sqrt(diag(vcov(r))), 1e-7)
near(m@deviance, deviance(r), 1e-9)
near(m@null_deviance, r$null.deviance, 1e-9)
near(m@log_likelihood, as.numeric(logLik(r)), 1e-9)
nd <- data.frame(region = c("W", "S"), age = c(40, 50), exposure = c(2, 1))
near(predict(m, nd), predict(r, nd, type = "response"), 1e-8)
stopifnot(inherits(try(predict(m, data.frame(region = "E", age = 1, exposure = 1)), silent = TRUE),
                   "try-error"))
pd <- predict_distribution(m, nd, n_sims = 2000, seed = 1)
stopifnot(nrow(draw_matrix(pd)) == 2000)

g <- glm_fit(age ~ region, d, family = "gamma", link = "log")
rg <- stats::glm(age ~ region, d, family = Gamma(link = "log"), control = tight)
near(coef(g), coef(rg), 1e-8)
near(g@dispersion, summary(rg)$dispersion, 1e-8)

# Elastic net: a path from all-zero to the GLM; glmnet when installed.
set.seed(2)
e <- data.frame(x1 = rnorm(60), x2 = rnorm(60), x3 = rnorm(60))
e$y <- 1 + 2 * e$x1 - e$x2 + rnorm(60)
en <- elastic_net_fit(y ~ x1 + x2 + x3, e, family = "gaussian", alpha = 1, nlambda = 20,
                      lambda_min_ratio = 1e-3)
stopifnot(length(en@lambda) == 20, en@df[1] == 0, en@df[20] == 3)
stopifnot(all(abs(coef(en)[-1, 1]) < 1e-10), nrow(predict(en, e)) == 60)
zero <- elastic_net_fit(y ~ x1 + x2 + x3, e, family = "gaussian", lambda = 0)
near(coef(zero, lambda = 0), coef(stats::lm(y ~ x1 + x2 + x3, e)), 1e-8)
lam <- en@lambda[10]
stopifnot(length(predict(en, e, lambda = lam)) == 60)
stopifnot(inherits(try(coef(en, lambda = 0.123456), silent = TRUE), "try-error"))
pf <- elastic_net_fit(y ~ x1 + x2 + x3, e, family = "gaussian", lambda = en@lambda[2],
                      penalty_factor = c(x3 = 0))
stopifnot(coef(pf, lambda = en@lambda[2])[["x3"]] != 0)
pd <- predict_distribution(zero, e[1:3, ], n_sims = 100, seed = 1)
stopifnot(S7::S7_inherits(pd, predictive_distribution), is.finite(mean(pd)))
if (requireNamespace("glmnet", quietly = TRUE)) {
  x <- as.matrix(e[, c("x1", "x2", "x3")])
  gp <- glmnet::glmnet(x, e$claims <- rpois(60, exp(0.2 * e$x1)), family = "poisson",
                       alpha = 1, lambda = c(0.2, 0.05), thresh = 1e-20)
  ep <- elastic_net_fit(claims ~ x1 + x2 + x3, e, family = "poisson", alpha = 1,
                        lambda = c(0.2, 0.05))
  near(unname(coef(ep)), unname(as.matrix(stats::coef(gp))), 1e-7)
}

# GAM: a smooth curve.
s <- data.frame(x = seq(0, 1, length.out = 100))
s$y <- sin(6 * s$x) + 2
gm <- gam_fit(y ~ x, s, smooths = "x", n_basis = 12, family = "gaussian")
stopifnot(max(abs(predict(gm, s) - s$y)) < 0.02)
near(gam_fit(y ~ x, s, smooths = "x", family = "gaussian", smoothing = 1e8)@edf, 2, 1e-4)

# Metrics, splits, MCMC.
near(family_deviance("gaussian", c(1, 3), c(2, 2)), 2)
near(gini_index(c(0, 1), c(0.1, 0.9)), 0.5)
stopifnot(identical(round(lift_table(c(0, 1, 2, 3), c(0.1, 0.9, 2.1, 2.9), bands = 2)$actual), c(1, 5)))
near(crps_draws(3, 1), 2)
f <- k_fold(10, 5, seed = 1)
stopifnot(identical(sort(unlist(lapply(f, `[[`, "test"))), as.double(1:10)))
stopifnot(identical(time_ordered(c(0, 1, 2, 1, 2, 2), 1)[[1]]$test, c(3, 5, 6)))
set.seed(1)
dg <- mcmc_diagnostics(matrix(rnorm(4000), ncol = 4))
stopifnot(dg[["rhat"]] < 1.01, dg[["ess_bulk"]] > 2000)
if (requireNamespace("posterior", quietly = TRUE)) {
  x <- matrix(rnorm(2000), ncol = 4)
  near(mcmc_diagnostics(x)[["ess_bulk"]], posterior::ess_bulk(x), 1e-10)
}

cat("actuarialrs R models tests passed\n")
