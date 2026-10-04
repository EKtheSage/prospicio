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

# Sandwich covariance: for the canonical link, (X'WX)^-1 (U'U) (X'WX)^-1
# with scores U = X (y - mu).
X <- model.matrix(r)
U <- X * (d$claims - fitted(r))
B <- vcov(r)
near(robust_vcov(m), B %*% crossprod(U) %*% B, 1e-6)
near(robust_vcov(m, "HC1"), robust_vcov(m) * 8 / 4, 1e-12)
cl <- c(1, 1, 2, 2, 3, 3, 4, 4)
Uc <- rowsum(U, cl)
near(robust_vcov(m, cluster = cl), B %*% crossprod(Uc) %*% B * 4 / 3 * 7 / 4, 1e-6)
stopifnot(inherits(try(robust_vcov(m, cluster = cl[-1]), silent = TRUE), "try-error"))

# Save and load: exact round trip; predictions agree.
mf <- tempfile(fileext = ".rds")
save_model(m, mf)
m2 <- load_model(mf)
stopifnot(identical(coef(m2), coef(m)), identical(m2@covariance, m@covariance))
stopifnot(identical(predict(m2, nd), predict(m, nd)))
stopifnot(identical(m2@ptr$input_hash(), m@ptr$input_hash()))
stopifnot(inherits(try(robust_vcov(m2), silent = TRUE), "try-error"))
saveRDS(list(1), mf)
stopifnot(inherits(try(load_model(mf), silent = TRUE), "try-error"))

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

# Cross-validation and search.
folds <- k_fold(nrow(e), 5, seed = 1)
mse <- function(m, test) mean((test$y - predict(m, test))^2)
cvs <- cross_validate(e, folds, function(train) glm_fit(y ~ x1 + x2 + x3, train, family = "gaussian"), mse)
stopifnot(length(cvs) == 5, all(cvs > 0))
g <- grid_search(c(0, 0.05, 5), e, folds, function(lam, train) {
  elastic_net_fit(y ~ x1 + x2 + x3, train, family = "gaussian", lambda = lam)
}, mse)
stopifnot(nrow(g) == 3, g$score[3] > g$score[1], attr(g, "best") %in% 1:2)
r <- random_search(4, function() 10^stats::runif(1, -3, 0), e, folds, function(lam, train) {
  elastic_net_fit(y ~ x1 + x2 + x3, train, family = "gaussian", lambda = lam)
}, mse, seed = 1)
stopifnot(nrow(r) == 4)
fid <- (seq_len(nrow(e)) - 1) %% 5 + 1
ecv <- elastic_net_cv(y ~ x1 + x2 + x3, e, family = "gaussian", nlambda = 15,
                      lambda_min_ratio = 1e-3, foldid = fid)
stopifnot(ecv$lambda_1se >= ecv$lambda_min, length(ecv$mean) == 15,
          S7::S7_inherits(ecv$fit, elastic_net_model))
if (requireNamespace("glmnet", quietly = TRUE)) {
  gcv <- glmnet::cv.glmnet(as.matrix(e[, c("x1", "x2", "x3")]), e$y, lambda = ecv$lambda,
                           foldid = fid, thresh = 1e-20, type.measure = "deviance")
  near(ecv$mean, gcv$cvm, 1e-7)
  near(ecv$se, gcv$cvsd, 1e-6)
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

# Log score and PIT.
set.seed(3)
yp <- rpois(2000, 3)
stopifnot(ks_uniform(pit_values("poisson", yp, rep(3, 2000), seed = 1)) < 1.63 / sqrt(2000))
stopifnot(ks_uniform(pit_values("poisson", yp, rep(4.5, 2000))) > 3 / sqrt(2000))
near(log_score("poisson", yp, rep(3, 2000)), -mean(dpois(yp, 3, log = TRUE)), 1e-12)
yg <- rgamma(500, shape = 2, scale = 1.5)
near(log_score("gamma", yg, rep(3, 500), dispersion = 0.5),
     -mean(dgamma(yg, shape = 2, scale = 1.5, log = TRUE)), 1e-12)

# ELPD: PSIS-LOO and WAIC, against loo when installed.
set.seed(5)
ye <- c(rnorm(20), 5)
mue <- rnorm(400, mean(ye), 1 / sqrt(21))
lle <- sapply(ye, function(yi) dnorm(yi, mue, 1, log = TRUE))
lo <- elpd_loo(lle)
wa <- elpd_waic(lle)
stopifnot(which.max(lo$pareto_k) == 21, length(lo$pointwise) == 21)
near(lo$estimates[["ic"]], -2 * lo$estimates[["elpd"]], 1e-12)
if (requireNamespace("loo", quietly = TRUE)) {
  ref <- loo::loo(lle, r_eff = rep(1, 21))
  near(lo$estimates[["elpd"]], ref$estimates["elpd_loo", "Estimate"], 1e-9)
  near(lo$pareto_k, ref$diagnostics$pareto_k, 1e-6)
  rw <- suppressWarnings(loo::waic(lle))
  near(wa$estimates[["elpd"]], rw$estimates["elpd_waic", "Estimate"], 1e-12)
}

# compare_models: one table, consistent with cross_validate.
cd <- data.frame(x = 1:40 / 10)
cd$y <- 1 + 2 * cd$x + sin(1:40)
cf <- k_fold(nrow(cd), 4, seed = 1)
mse <- function(m, test) mean((test$y - predict(m, test))^2)
lin <- function(train) glm_fit(y ~ x, train, family = "gaussian")
cmp <- compare_models(list(linear = lin, flat = function(train) glm_fit(y ~ 1, train, family = "gaussian")),
                      cd, cf, list(mse = mse, mae = function(m, test) mean(abs(test$y - predict(m, test)))))
stopifnot(nrow(cmp) == 4, identical(cmp$model, c("linear", "linear", "flat", "flat")))
near(attr(cmp, "split_scores")["linear", "mse", ], cross_validate(cd, cf, lin, mse), 1e-12)
stopifnot(cmp$difference_std_error[1] == 0, cmp$mean[3] > cmp$mean[1])
stopifnot(inherits(try(compare_models(list(lin), cd, cf, list(mse = mse)), silent = TRUE), "try-error"))

cat("actuarialrs R models tests passed\n")
