library(prospicio)

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
stopifnot(is.na(gm@reml_scale))
s$noisy <- s$y + 0.1 * sin(97 * (seq_len(100) - 1))
gr <- gam_fit(noisy ~ x, s, smooths = "x", n_basis = 12, family = "gaussian", smoothing = "reml")
stopifnot(gr@reml_scale > 0.002, gr@reml_scale < 0.02, gr@edf > 4, gr@edf < 12)
# A cyclic smooth: its ends meet over the period, and the period wraps.
cs <- data.frame(month = (seq_len(240) - 0.5) / 20)
cs$y <- 2 + sin(2 * pi * cs$month / 12)
gc <- gam_fit(y ~ month, cs, smooths = "month", family = "gaussian", bs = "cp",
              period = list(month = c(0, 12)))
pe <- predict(gc, data.frame(month = c(0, 12, 1, 13)))
stopifnot(abs(pe[1] - pe[2]) < 1e-12, abs(pe[3] - pe[4]) < 1e-12, abs(pe[1] - 2) < 0.01)
stopifnot(inherits(try(gam_fit(y ~ month, cs, smooths = "month", family = "gaussian",
                               period = list(month = c(0, 12))), silent = TRUE), "try-error"))
cf <- tempfile(fileext = ".rds")
save_model(gc, cf)
stopifnot(identical(predict(load_model(cf), data.frame(month = 13)), pe[[4]]))
# Monotone smooths never turn back.
ms <- data.frame(x = (0:199) / 199)
ms$y <- ms$x^2 + 0.05 * sin(1.7 * (0:199))
gi <- gam_fit(y ~ x, ms, smooths = "x", family = "gaussian", bs = "mpi")
stopifnot(all(diff(gi@fitted) >= -1e-12))
ms$z <- -ms$y
gd <- gam_fit(z ~ x, ms, smooths = "x", family = "gaussian", bs = "mpd")
stopifnot(all(diff(gd@fitted) <= 1e-12))
stopifnot(inherits(try(gam_fit(y ~ x, ms, smooths = "x", family = "gaussian", bs = "mpi",
                               smoothing = "reml"), silent = TRUE), "try-error"))
# A tensor product smooth.
tz <- data.frame(x = (((0:199) * 37) %% 101) / 100, z = (((0:199) * 53) %% 97) / 96)
tz$y <- sin(3 * tz$x) * cos(2 * tz$z) + 0.05 * sin(91 * (0:199))
gt <- gam_fit(y ~ x + z, tz, smooths = character(), family = "gaussian", smoothing = "reml",
              tensors = list(c("x", "z")))
stopifnot(length(gt@lambdas) == 2, max(abs(predict(gt, tz) - tz$y)) < 0.2)

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

# Stacking and pseudo-BMA weights; blending.
lpd <- cbind(a = c(-0.1, -0.1, -3, -3), b = c(-3, -3, -0.1, -0.1))
near(stacking_weights(lpd), c(a = 0.5, b = 0.5), 1e-9)
dom <- cbind(a = lpd[, 1], worse = lpd[, 1] - 1)
stopifnot(identical(stacking_weights(dom), c(a = 1, worse = 0)))
near(pseudo_bma_weights(dom, bootstrap = FALSE)[["a"]], 1 / (1 + exp(-4)), 1e-12)
pb <- pseudo_bma_weights(dom, n_draws = 500, seed = 3)
stopifnot(abs(sum(pb) - 1) < 1e-12, pb[["a"]] > 0.5)
stopifnot(inherits(try(stacking_weights(lpd[, 1, drop = FALSE]), silent = TRUE), "try-error"))
p1 <- predictive_distribution(cbind(rep(1, 200), rep(-1, 200)), data.frame(lob = c("x", "y")))
p2 <- predictive_distribution(cbind(rep(2, 200), rep(-2, 200)), data.frame(lob = c("x", "y")))
mix <- blend_predictive(list(p1, p2), c(1, 1), seed = 5)
stopifnot(all(rowSums(draw_matrix(mix)) == 0))
stopifnot(inherits(try(blend_predictive(list(p1, p2), 1, seed = 5), silent = TRUE), "try-error"))

# Save and load GAMs and elastic nets.
gf <- tempfile(fileext = ".rds")
save_model(gm, gf)
gm2 <- load_model(gf)
stopifnot(identical(coef(gm2), coef(gm)), identical(predict(gm2, s), predict(gm, s)))
save_model(en, gf)
en2 <- load_model(gf)
stopifnot(identical(en2@lambda, en@lambda), identical(en2@coefficients, en@coefficients))
stopifnot(identical(predict(en2, e, lambda = en@lambda[5]), predict(en, e, lambda = en@lambda[5])))
stopifnot(inherits(try(save_model(list(), gf), silent = TRUE), "try-error"))

# Monitoring: actual against expected by period.
ae <- actual_vs_expected(rep(2021:2024, each = 2), c(4, 6, 5, 6, 6, 6, 6, 7), rep(5, 8), "poisson")
stopifnot(identical(ae$period, 2021:2024), identical(ae$actual, c(10, 11, 12, 13)))
near(attr(ae, "trend")[["slope"]], 0.1, 1e-12)
near(attr(ae, "trend")[["std_error"]], sqrt(0.1 / 5), 1e-12)
near(attr(ae, "total")[["z"]], 6 / sqrt(40), 1e-12)
aw <- actual_vs_expected(c("b", "a", "b"), c(1, 2, 3), c(2, 2, 2), "gamma", weights = c(1, 2, 3),
                         dispersion = 0.5)
stopifnot(identical(aw$period, c("a", "b")), identical(aw$expected, c(4, 8)))
near(aw$std_dev[2], sqrt(0.5 * 4 * 4), 1e-12)

# Bayesian GLM by NUTS.
bd <- data.frame(claims = rep(c(1, 2, 3, 5), 20), x = rep(c(-1.5, -0.5, 0.5, 1.5), 20))
bm <- bayes_glm_fit(claims ~ x, bd, family = "poisson", chains = 2, tune = 300, draws = 300, seed = 4)
bm2 <- bayes_glm_fit(claims ~ x, bd, family = "poisson", chains = 2, tune = 300, draws = 300, seed = 4)
stopifnot(identical(bm@draws, bm2@draws), nrow(bm@draws) == 600, bm@divergences == 0)
stopifnot(all(bm@summary$rhat < 1.05), identical(bm@summary$parameter, c("(Intercept)", "x")))
gm_b <- glm_fit(claims ~ x, bd, family = "poisson")
stopifnot(abs(coef(bm)[["x"]] - coef(gm_b)[["x"]]) < 0.5 * gm_b@std_errors[["x"]])
bl <- bayes_loo(bm)
stopifnot(length(bl$pointwise) == 80, is.finite(bl$estimates[["elpd"]]))
stopifnot(length(predict(bm, data.frame(x = c(0, 1)))) == 2)
bpd <- predict_distribution(bm, data.frame(x = c(0, 1)), n_sims = 100, seed = 1)
stopifnot(nrow(draw_matrix(bpd)) == 100)
bg <- bayes_glm_fit(claims ~ x, bd, family = "gaussian", chains = 2, tune = 300, draws = 300)
stopifnot(tail(bg@summary$parameter, 1) == "dispersion")

# Posterior predictive checks.
brep <- bayes_replicate(bm, seed = 2)
stopifnot(identical(dim(brep), c(600L, 80L)), identical(brep, bayes_replicate(bm, seed = 2)))
bc <- bayes_ppc(bm, c("mean", "sd", 0.9, "proportion_zero"), seed = 2)
stopifnot(identical(bc$name, c("mean", "sd", "q0.9", "proportion_zero")))
stopifnot(bc$p_value[1] > 0.05, bc$p_value[1] < 0.95, bc$p_value[2] > 0.95)
stopifnot(identical(dim(attr(bc, "replicated")), c(600L, 4L)))
# The same from the replicates directly.
bc2 <- ppc_check(bd$claims, brep, c("mean", "sd", 0.9, "proportion_zero"))
stopifnot(isTRUE(all.equal(bc2$p_value, bc$p_value)))
bgc <- ppc_check(bd$claims, brep, "mean", groups = rep(1:4, 20))
stopifnot(identical(bgc$name, paste0("mean[", 0:3, "]")))
bp <- bayes_loo_pit(bm, seed = 2)
stopifnot(length(bp$pit) == 80, all(bp$pit >= 0 & bp$pit <= 1), length(bp$pareto_k) == 80)
ll_m <- matrix(bm@ptr$log_likelihood(), nrow = 600, byrow = TRUE)
bp2 <- loo_pit(bd$claims, brep, ll_m, seed = 2)
stopifnot(length(bp2$pit) == 80, bp2$ks >= 0)
stopifnot(inherits(try(ppc_check(bd$claims, brep, "median"), silent = TRUE), "try-error"))

# Bayesian and hierarchical stacking.
hx <- seq(-0.5, 0.5, length.out = 100)
hl <- cbind(a = ifelse(hx < 0, -0.5, -2), b = ifelse(hx < 0, -2, -0.5))
hs <- hierarchical_stacking(hl, data.frame(x = hx), chains = 2, tune = 300, draws = 300, seed = 2)
hw <- predict(hs, data.frame(x = c(-0.4, 0, 0.4)))
stopifnot(identical(colnames(hw), c("a", "b")), all(abs(rowSums(hw) - 1) < 1e-12))
stopifnot(hw[1, "a"] > 0.7, hw[3, "a"] < 0.3, nrow(hs@weights) == 100, hs@divergences == 0)
bs <- bayes_stacking(hl, chains = 2, tune = 300, draws = 300)
stopifnot(nrow(bs@weights) == 1, abs(bs@weights[1, "a"] - 0.5) < 0.1)
q1 <- predictive_distribution(cbind(rep(1, 50), rep(1, 50)), data.frame(lob = c("x", "y")))
q2 <- predictive_distribution(cbind(rep(2, 50), rep(2, 50)), data.frame(lob = c("x", "y")))
qm <- blend_predictive(list(q1, q2), rbind(c(1, 0), c(0, 1)), seed = 3)
stopifnot(all(draw_matrix(qm)[, 1] == 1), all(draw_matrix(qm)[, 2] == 2))

# Partial pooling with discrete dummies first, and adaptive priors.
region <- (seq_len(120) - 1) %% 4
px <- data.frame(r1 = as.double(region == 1), r2 = as.double(region == 2),
                 r3 = as.double(region == 3), x = ((seq_len(120) - 1) * 37) %% 101 / 100 - 0.5)
pl <- cbind(a = ifelse(region < 2, -0.5, -2), b = ifelse(region < 2, -2, -0.5))
ps <- hierarchical_stacking(pl, px, discrete = 3, partial_pooling = TRUE, adaptive = 4,
                            chains = 2, tune = 300, draws = 300, seed = 9)
pw <- predict(ps, data.frame(r1 = 0, r2 = 0, r3 = c(0, 1), x = 0))
stopifnot(pw[1, "a"] > 0.7, pw[2, "a"] < 0.3)
stopifnot(inherits(try(hierarchical_stacking(pl, px, discrete = 5), silent = TRUE), "try-error"))

# The over-dispersed Poisson accepts a negative response (R's quasipoisson
# refuses it); with a group dummy the fitted means are the group means.
qd <- data.frame(y = c(5, -1, 4, 2, 3, 4), g = rep(c("a", "b"), each = 3))
qm <- glm_fit(y ~ g, qd, family = "poisson", dispersion = "pearson")
near(unname(coef(qm)), c(log(8 / 3), log(3 / (8 / 3))), 1e-10)
stopifnot(inherits(try(glm_fit(y ~ g, qd, family = "poisson"), silent = TRUE), "try-error"))

# Mean-preserving parameter draws centre on the fitted means.
pm <- glm_fit(y ~ g, data.frame(y = c(3, 9, 1, 8, 20, 5), g = rep(c("a", "b"), each = 3)),
              family = "poisson", dispersion = "pearson")
pnew <- data.frame(g = c("a", "b"))
pc <- predict_distribution(pm, pnew, n_sims = 20000, seed = 1, parameters = "mean_preserving")
stopifnot(abs(mean(rowSums(draw_matrix(pc))) / sum(predict(pm, pnew)) - 1) < 0.02)
pn <- predict_distribution(pm, pnew, n_sims = 20000, seed = 1)
stopifnot(mean(rowSums(draw_matrix(pn))) > mean(rowSums(draw_matrix(pc))))

# pit_from_draws(), pit_histogram() and deviance_score() as Python's (#382,
# #360); a GAM's log-likelihood and AIC.
stopifnot(identical(pit_from_draws(c(1, 2, 3, 4), c(2.5, 2)), c(0.5, 0.375)))
stopifnot(identical(pit_histogram(c(0, 0.05, 0.15, 1), bins = 10), c(2, 1, 0, 0, 0, 0, 0, 0, 0, 1)))
cv_data <- data.frame(y = c(1, 0, 2, 3, 1, 4, 2, 0), x = 1:8)
sc <- deviance_score("poisson", response = "y")
cv_fit <- glm_fit(y ~ x, cv_data[1:6, ], family = "poisson")
near(sc(cv_fit, cv_data[7:8, ]),
     family_deviance("poisson", c(2, 0), predict(cv_fit, cv_data[7:8, ])) / 2, 1e-14)
stopifnot(length(cross_validate(cv_data, k_fold(8, 2, seed = 1),
                                function(d) glm_fit(y ~ x, d, family = "poisson"), sc)) == 2)
near(gm@aic, -2 * gm@log_likelihood + 2 * gm@edf, 1e-14)
stopifnot(is.finite(gm@log_likelihood))


# Nested tests, BIC and VIF agree with R (#351).
quine <- MASS::quine
quine$y <- log(quine$Days + 1)
qs <- glm_fit(y ~ Eth + Sex, quine, family = "gaussian")
qb <- glm_fit(y ~ Eth + Sex + Age + Lrn, quine, family = "gaussian")
rs <- stats::glm(y ~ Eth + Sex, data = quine)
rb <- stats::glm(y ~ Eth + Sex + Age + Lrn, data = quine)
ft <- f_test(qs, qb)
ra <- anova(rs, rb, test = "F")
near(ft$statistic, ra$F[2], 1e-9)
near(ft$p_value, ra$`Pr(>F)`[2], 1e-8)
near(likelihood_ratio_test(qs, qb)$p_value, anova(rs, rb, test = "Chisq")$`Pr(>Chi)`[2], 1e-8)
qp <- glm_fit(Days ~ Eth + Sex + Age + Lrn, quine, family = "poisson")
near(qp@bic, BIC(stats::glm(Days ~ Eth + Sex + Age + Lrn, family = poisson, data = quine)), 1e-9)
v <- vif(y ~ Eth + Sex + Age + Lrn, quine)
stopifnot(is.na(v[["(Intercept)"]]))
xs <- as.data.frame(stats::model.matrix(~ Eth + Sex + Age + Lrn, quine))[, -1]
near(v[["SexM"]], 1 / (1 - summary(stats::lm(SexM ~ ., data = xs))$r.squared), 1e-9)


# Charts (#351): AUROC against the Wilcoxon statistic and the normalized Gini.
set.seed(3)
yb <- rbinom(200, 1, 0.3)
sc <- runif(200) + 0.4 * yb
rc <- roc_curve(yb, sc)
w <- stats::wilcox.test(sc[yb == 1], sc[yb == 0])$statistic
near(rc$auc, unname(w) / (sum(yb) * sum(1 - yb)), 1e-12)
near(rc$auc, 0.5 * (1 + normalized_gini(yb, sc)), 1e-12)
cm <- confusion_matrix(yb, sc, 0.7)
stopifnot(cm[["true_positive"]] == sum(yb == 1 & sc >= 0.7))
dl <- double_lift_table(c(1, 0, 3, 2, 5, 1), c(1, 1, 2, 2, 4, 1), c(2, 1, 2, 1, 3, 2), bands = 2)
near(sum(dl$exposure * dl$actual_relativity), 6, 1e-12)
lr <- loss_ratio_table(c(1, 0, 3, 2, 5, 1), c(1, 1, 2, 2, 4, 1), rep(2, 6), bands = 2)
near(lr$loss_ratio, lr$actual / lr$premium, 1e-15)


# Residuals, hat values, Cook's distances and prediction errors agree with
# R's glm (#334).
qg <- glm_fit(Days ~ Eth + Sex + Age + Lrn, quine, family = "poisson")
rg <- stats::glm(Days ~ Eth + Sex + Age + Lrn, family = poisson, data = quine, control = tight)
for (type in c("deviance", "pearson", "response", "working")) {
  near(residuals(qg, type = type), unname(residuals(rg, type = type)), 1e-6)
}
near(hatvalues(qg), unname(hatvalues(rg)), 1e-6)
# R 4.3's rstandard() and cooks.distance() for a Poisson glm, by their
# formulas: newer R releases compute them differently, so compare with
# the definitions rather than with whichever stats is installed.
h_r <- unname(hatvalues(rg))
near(rstandard(qg), unname(residuals(rg, "deviance")) / sqrt(1 - h_r), 1e-6)
near(cooks.distance(qg), unname(residuals(rg, "pearson"))^2 * h_r / ((1 - h_r)^2 * 7), 1e-6)
near(prediction_se(qg, quine), unname(predict(rg, type = "response", se.fit = TRUE)$se.fit), 1e-6)
stopifnot(nrow(glm_diagnostics(qg)) == nrow(quine))
spe <- sum_prediction_error(qg, quine[1, ])
near(spe$parameter_variance, prediction_se(qg, quine[1, ])^2, 1e-10)
n_q <- nrow(quine)
near(glm_gcv(qg), n_q * sum(residuals(qg, "response")^2) / (n_q - 7)^2, 1e-12)
near(log_likelihood_at(qg, 1), qg@log_likelihood, 1e-12)
# Randomized quantile residuals (#464): a count's lies between qnorm of
# F(y - 1) and F(y), reproducibly by seed. The bounds are taken from the
# upper tail, where `ppois` would round F(y - 1) to 1 on two rows.
qr <- residuals(qg, type = "quantile", seed = 3)
mu_r <- unname(fitted(rg))
upper <- function(k) -qnorm(ppois(k, mu_r, lower.tail = FALSE))
stopifnot(all(qr >= upper(quine$Days - 1) - 1e-6),
          all(qr <= upper(quine$Days) + 1e-6),
          identical(qr, residuals(qg, type = "quantile", seed = 3)))
near(qr, quantile_residuals("poisson", quine$Days, qg@fitted, seed = 3), 1e-12)
# A gamma's is exact: qnorm of its cdf at the fit's dispersion.
quine$Days1 <- quine$Days + 1
qgg <- glm_fit(Days1 ~ Eth + Sex + Age + Lrn, quine, family = "gamma", link = "log")
rgg <- stats::glm(Days1 ~ Eth + Sex + Age + Lrn, family = Gamma(link = "log"), data = quine,
                  control = tight)
phi_g <- summary(rgg)$dispersion
near(residuals(qgg, type = "quantile"),
     qnorm(pgamma(quine$Days1, shape = 1 / phi_g, scale = unname(fitted(rgg)) * phi_g)), 1e-5)
# Out-of-fold target encoding (#357): a row's own response never enters it.
vals <- paste0("l", (0:59) %% 7)
ys <- as.double(((0:59) * 13) %% 5)
sp <- k_fold(60, 5, seed = 3)
te <- target_encode(vals, ys, rep(2, 60), sp, k = 4)
ys2 <- ys
ys2[11] <- ys2[11] + 100
stopifnot(target_encode(vals, ys2, rep(2, 60), sp, k = 4)$rows[11] == te$rows[11])
a0 <- sum(ys[vals == "l0"])
n0 <- sum(vals == "l0")
near(encode_target(te$encoding, c("l0", "zz")), c(n0 / (n0 + 4) * a0 / (2 * n0) + 4 / (n0 + 4), 1), 1e-12)


# The negative binomial's theta agrees with MASS::glm.nb, and the Tweedie
# power is the maximum of its profile.
quine <- MASS::quine
nb <- glm_nb_fit(Days ~ Eth + Sex + Age + Lrn, quine)
rnb <- MASS::glm.nb(Days ~ Eth + Sex + Age + Lrn, quine)
near(nb$theta, rnb$theta, 1e-6)
near(nb$se_theta, rnb$SE.theta, 1e-5)
near(nb$log_likelihood, as.numeric(logLik(rnb)), 1e-9)
near(nb$aic, AIC(rnb), 1e-9)
near(coef(nb$model), coef(rnb), 1e-6)
tw <- tweedie_glm_fit(Days ~ Eth + Sex + Age + Lrn, quine)
stopifnot(tw$lower < tw$power, tw$power < tw$upper)
pr <- tweedie_profile(Days ~ Eth + Sex + Age + Lrn, quine, powers = c(1.1, 1.3, 1.5, 1.7, 1.9))
near(pr$power, tw$power, 1e-2)
near(pr$max_log_likelihood, tw$log_likelihood, 1e-6)

# GLMMs agree with glmmTMB 1.1.15.2 and lme4 (validation/scripts/r_glmm.R,
# its stored reference): nested factors on lme4's grouseticks, crossed ones
# on the simulated book and random slopes on the slopes book.
root <- if (dir.exists("validation")) "validation" else file.path("..", "..", "..", "validation")
lines <- readLines(file.path(root, "reference", "glmm_r.csv"))
ref <- utils::read.csv(text = lines[!startsWith(lines, "#")], stringsAsFactors = FALSE)
want <- function(dataset, model, quantity, arg = "") {
  arg_col <- ifelse(is.na(ref$arg), "", ref$arg)
  ref$expected[ref$dataset == dataset & ref$model == model & ref$quantity == quantity &
                 arg_col == arg]
}
gt <- utils::read.csv(file.path(root, "data", "glmm_grouseticks.csv"),
                      colClasses = c(year = "character", location = "character",
                                     brood = "character"))
g <- glmm_fit(ticks ~ year + height + (1 | location/brood), gt)
stopifnot(g@converged, identical(names(g@sd), c("location", "location:brood")))
near(coef(g), sapply(names(coef(g)), function(n) want("grouseticks", "poisson", "coef", n)), 2e-5)
near(g@sd, c(want("grouseticks", "poisson", "sd", "location"),
             want("grouseticks", "poisson", "sd", "location:brood")), 2e-5)
near(g@log_likelihood, want("grouseticks", "poisson", "loglik"), 1e-8)
re <- g@random_effects$location
near(re$value[re$level == "32"], want("grouseticks", "poisson", "ranef", "location|32"), 1e-4)
# The same fit with the nested factor written out.
g2 <- glmm_fit(ticks ~ year + height + (1 | location) + (1 | location:brood), gt)
near(g2@log_likelihood, g@log_likelihood, 1e-12)
# Without the grouping columns, or at an unseen level, predict() gives the
# population mean; with them, each row's intercepts are added.
nd <- gt[1:2, c("year", "height")]
nd$year <- factor(nd$year, levels = c("95", "96", "97"))
population <- exp(as.vector(stats::model.matrix(~ year + height, nd) %*% coef(g)))
nd$year <- as.character(nd$year)
near(predict(g, nd), population, 1e-12)
near(predict(g, cbind(nd, location = "nowhere", brood = "none")), population, 1e-12)
near(predict(g, gt[1:2, ]), g@fitted[1:2], 1e-9)

sim <- utils::read.csv(file.path(root, "data", "glmm_simulated.csv"))
s <- glmm_fit(severity ~ x + urban + (1 | territory) + (1 | class), sim, family = "gamma")
near(s@dispersion, want("simulated", "gamma", "dispersion"), 2e-5)
near(s@log_likelihood, want("simulated", "gamma", "loglik"), 1e-8)
tw <- glmm_fit(pure ~ x + urban + (1 | territory) + (1 | class), sim, family = "tweedie",
               power = 1.5)
near(tw@dispersion, want("simulated", "tweedie", "dispersion"), 2e-5)
near(tw@log_likelihood, want("simulated", "tweedie", "loglik"), 1e-8)
stopifnot(inherits(try(glmm_fit(ticks ~ year, gt), silent = TRUE), "try-error"),
          inherits(try(glmm_fit(ticks ~ year + (0 | location), gt), silent = TRUE),
                   "try-error"))

sl <- utils::read.csv(file.path(root, "data", "glmm_slopes.csv"))
p <- glmm_fit(count ~ x + w + (1 + x | territory) + offset(log(exposure)), sl)
stopifnot(p@converged, identical(names(p@sd), c("territory", "territory|x")))
near(p@sd, c(want("slopes", "poisson", "sd", "territory"),
             want("slopes", "poisson", "sd", "territory|x")), 2e-5)
near(p@correlations$territory["x", "(Intercept)"],
     want("slopes", "poisson", "cor", "territory|x|(Intercept)"), 2e-4)
near(p@log_likelihood, want("slopes", "poisson", "loglik"), 1e-8)
re <- p@random_effects$territory
near(re$value[re$level == "t07" & re$term == "x"],
     want("slopes", "poisson", "ranef", "territory|t07|x"), 1e-4)
near(predict(p, sl[1:3, ]), p@fitted[1:3], 1e-9)
# `(x | g)` has the intercept too; `||` makes the terms independent.
near(glmm_fit(count ~ x + w + (x | territory) + offset(log(exposure)), sl)@log_likelihood,
     p@log_likelihood, 1e-10)
dg <- glmm_fit(count ~ x + w + (1 + x || territory) + offset(log(exposure)), sl)
near(dg@log_likelihood, want("slopes", "poisson_diag", "loglik"), 1e-8)
stopifnot(identical(unique(dg@random_effects$territory$term), c("(Intercept)", "x")))
near(predict(dg, sl[1:3, ]), dg@fitted[1:3], 1e-9)
stopifnot(inherits(try(glmm_fit(count ~ x + (1 + x | territory), sl, nodes = 5),
                       silent = TRUE), "try-error"))

# The NUTS fit of the same model sits on the Laplace fit.
bp <- bayes_glmm_fit(count ~ x + w + (1 + x | territory) + offset(log(exposure)), sl,
                     chains = 4, tune = 500, draws = 500, seed = 2026)
bs <- bp@summary
stopifnot(bp@divergences == 0, all(bs$rhat < 1.05),
          identical(bs$parameter[4:6], c("sd[territory|(Intercept)]", "sd[territory|x]",
                                         "cor[territory|x,(Intercept)]")),
          all(abs(bs$mean[1:3] - coef(bp@laplace)) < 0.2 * bs$sd[1:3]),
          identical(dim(bp@draws), c(2000L, 6L)), ncol(bp@effect_draws) == 60)
bre <- bp@random_effects$territory
lre <- bp@laplace@random_effects$territory
stopifnot(identical(bre$level, lre$level), identical(bre$term, lre$term),
          all(abs(bre$mean - lre$value) < 0.4 * bre$sd))
stopifnot(all(predict(bp, sl[1:3, ]) > 0))

# Double GLMs against dglm 1.8.6 corrected (validation/scripts/r_dglm.R);
# where dglm is right (the Poisson without an offset), against dglm live.
dref <- utils::read.csv(file.path(root, "reference", "dglm_r.csv"), comment.char = "#")
dwant <- function(case, method, quantity, arg = "") {
  r <- dref[dref$case == case & dref$method == method & dref$quantity == quantity &
              dref$arg == arg, ]
  stopifnot(nrow(r) == 1)
  r$expected
}
dnear <- function(a, b, tol) stopifnot(all(abs(a - b) <= tol * pmax(abs(b), 1e-300)))
dg <- dglm_fit(severity ~ x + urban, ~ x + urban, sim, family = "gamma", link = "log",
               method = "reml")
stopifnot(dg@method == "reml")
for (nm in names(dg@coefficients)) {
  dnear(dg@coefficients[[nm]], dwant("gamma", "reml", "coef", nm), 1e-8)
  dnear(dg@std_errors[[nm]], dwant("gamma", "reml", "se", nm), 1e-8)
}
for (nm in names(dg@dispersion_coefficients)) {
  dnear(dg@dispersion_coefficients[[nm]], dwant("gamma", "reml", "dcoef", nm), 1e-8)
  dnear(dg@dispersion_std_errors[[nm]], dwant("gamma", "reml", "dse", nm), 1e-8)
}
dnear(dg@m2loglik, dwant("gamma", "reml", "m2loglik"), 1e-10)
dnear(dg@reml_criterion, dwant("gamma", "reml", "reml_criterion"), 1e-10)
dnear(predict(dg, sim), dg@fitted, 1e-12)
dnear(predict(dg, sim, type = "dispersion"), dg@dispersion_fitted, 1e-12)
dp <- dglm_fit(count ~ x + urban, ~ urban, sim, family = "poisson", method = "ml",
               offset = log(sim$exposure))
for (nm in names(dp@coefficients)) {
  dnear(dp@coefficients[[nm]], dwant("poisson_offset", "ml", "coef", nm), 1e-8)
}
dd <- dglm_fit(severity ~ x + urban, ~ urban, sim, family = "gamma", link = "log",
               method = "ml", doffset = log(sim$exposure))
for (nm in names(dd@dispersion_coefficients)) {
  dnear(dd@dispersion_coefficients[[nm]], dwant("gamma_doffset", "ml", "dcoef", nm), 1e-8)
}
if (requireNamespace("dglm", quietly = TRUE)) {
  live <- dglm::dglm(count ~ x + urban, ~ urban, family = stats::poisson, data = sim,
                     method = "ml", control = dglm::dglm.control(epsilon = 1e-14, maxit = 1000))
  ours <- dglm_fit(count ~ x + urban, ~ urban, sim, family = "poisson", method = "ml")
  dnear(ours@coefficients, stats::coef(live), 1e-6)
  dnear(ours@m2loglik, live$m2loglik, 1e-10)
}
stopifnot(inherits(try(dglm_fit(severity ~ x, ~ 1, sim, method = "pql"), silent = TRUE),
                   "try-error"))

# Elastic-net bounds and relaxed fits against glmnet 5.1
# (validation/scripts/r_glmnet_relax.R; its gamma = 0 is gamma = 1e-5).
pol <- utils::read.csv(file.path(root, "data", "glm_policies.csv"))
pol$age2 <- pol$age^2 / 100
nref <- utils::read.csv(file.path(root, "reference", "elastic_net_relax_glmnet.csv"))
ncv <- utils::read.csv(file.path(root, "reference", "elastic_net_relax_cv_glmnet.csv"))
pf <- claims ~ age + age2 + region + offset(log(exposure))
gf <- gauss ~ age + age2 + region
net_case <- function(case, lambda = NULL, relax = FALSE) {
  switch(case,
    poisson_nonneg = elastic_net_fit(pf, pol, lambda = lambda, lower_limits = 0),
    poisson_nonneg_relaxed = elastic_net_fit(pf, pol, lambda = lambda, lower_limits = 0,
                                             relax = TRUE),
    poisson_relaxed = elastic_net_fit(pf, pol, lambda = lambda, relax = TRUE),
    gaussian_relaxed = elastic_net_fit(gf, pol, family = "gaussian", lambda = lambda, relax = TRUE),
    gaussian_bounded = elastic_net_fit(gf, pol, family = "gaussian", lambda = lambda,
                                       lower_limits = c(regionC = -2),
                                       upper_limits = c(age2 = 0.1, regionD = 5)))
}
terms_order <- c("(Intercept)", "age", "age2", "region[B]", "region[C]", "region[D]")
for (case in unique(nref$case)) {
  rows <- nref[nref$case == case & nref$quantity == "coef", ]
  lams <- unique(rows$arg)
  fit <- net_case(case, lams)
  for (g in unique(ifelse(is.na(rows$gamma), 1, rows$gamma))) {
    b <- coef(fit, gamma = g)
    sub <- rows[(if (g == 1) is.na(rows$gamma) else rows$gamma == g), ]
    got <- b[cbind(match(sub$term, terms_order), match(sub$arg, lams))]
    stopifnot(all(abs(got - sub$expected) <= 1e-9 + 1e-7 * abs(sub$expected)))
  }
}
relaxed <- net_case("poisson_relaxed", nref$arg[nref$case == "poisson_relaxed"][2])
mm <- stats::model.matrix(~ age + age2 + region, pol)[1:3, ]
stopifnot(relaxed@relaxed)
near(predict(relaxed, pol[1:3, ], gamma = 0.4),
     exp(drop(mm %*% coef(relaxed, lambda = relaxed@lambda, gamma = 0.4)) +
         log(pol$exposure[1:3])), 1e-12)
f <- tempfile(fileext = ".rds")
save_model(relaxed, f)
back <- load_model(f)
stopifnot(back@relaxed, identical(coef(back, gamma = 0.3), coef(relaxed, gamma = 0.3)))
foldid <- (seq_len(nrow(pol)) - 1) %% 5 + 1
for (case in c("poisson_relaxed", "gaussian_relaxed")) {
  rows <- ncv[ncv$case == case, ]
  lams <- rows$arg[rows$quantity == "mean" & rows$gamma == 1]
  gammas <- sort(unique(rows$gamma[!is.na(rows$gamma)]))
  cv <- elastic_net_cv(if (case == "poisson_relaxed") pf else gf, pol,
                       family = if (case == "poisson_relaxed") "poisson" else "gaussian",
                       lambda = lams, foldid = foldid, relax = TRUE, gamma = gammas)
  m <- rows[rows$quantity == "mean", ]
  got <- cv$mean[cbind(match(m$arg, lams), match(m$gamma, gammas))]
  stopifnot(all(abs(got - m$expected) <= 1e-12 + 1e-7 * abs(m$expected)))
  for (q in c("lambda_min", "gamma_min", "lambda_1se", "gamma_1se")) {
    near(cv[[q]], rows$expected[rows$quantity == q], 1e-12)
  }
  stopifnot(cv$fit@relaxed)
}

# Model-agnostic explanations (#355): the portfolio of Python's
# test_models.py, the same rows, shuffles and pinned values.
i <- 0:299
ex <- data.frame(age = 18 + (37 * i) %% 60,
                 power = factor(c("low", "mid", "high")[(i %/% 7) %% 3 + 1], c("low", "mid", "high")),
                 noise = ((13 * i) %% 17) / 17, exposure = 0.5 + ((11 * i) %% 10) / 20)
young <- ex$age < 25
high <- ex$power == "high"
ex$y <- floor(ex$exposure * exp(0.5 + 0.8 * young + 0.3 * high + 0.5 * (young & high)) *
                (1 + 0.6 * sin(1.7 * i)) + 0.5)
additive <- y ~ I(as.numeric(age < 25)) + power + offset(log(exposure))
planted <- y ~ I(as.numeric(age < 25)) + power + I(as.numeric(age < 25) * as.numeric(power == "high")) +
  offset(log(exposure))
dev <- deviance_score("poisson", "y")
fit <- glm_fit(additive, ex, family = "poisson")
imp <- permutation_importance(fit, ex, dev, list("age", "power", "noise", c("age", "power")),
                              n_repeats = 2, seed = 5)
stopifnot(identical(imp$feature, c("age", "power", "noise", "age+power")), imp$importance[3] == 0,
          max(imp$relative) == 100, is.na(imp$std_error[1]))
near(imp$importance[c(1, 2, 4)], c(0.48670981297841526, 0.13778344497063177, 0.6330018332232121),
     1e-9)
# Over folds: fold i shuffles on stream i, so fold 0 is a single fit's run.
splits <- k_fold(nrow(ex), 4, seed = 2)
cv_imp <- permutation_importance(function(train) glm_fit(additive, train, family = "poisson"), ex, dev,
                                 c("age", "power"), splits = splits, seed = 1)
stopifnot(nrow(attr(cv_imp, "by_split")) == 4)
near(cv_imp$importance[1], 0.4464341484022761, 1e-9)
fold0 <- glm_fit(additive, ex[splits[[1]]$train, ], family = "poisson")
one <- permutation_importance(fold0, ex[splits[[1]]$test, ], dev, c("age", "power"), seed = 1)
near(attr(one, "by_split")[1, ], attr(cv_imp, "by_split")[1, ], 1e-12)
# Partial dependence of an additive log-link GLM: its relativities.
pd <- partial_dependence(fit, ex, "power", c("low", "mid", "high"))
near(pd$mean[2:3] / pd$mean[1], exp(unname(coef(fit)[3:4])), 1e-12)
pd_cv <- partial_dependence(function(train) glm_fit(additive, train, family = "poisson"), ex, "age",
                            c(20, 40), splits = k_fold(nrow(ex), 3, seed = 1))
stopifnot(nrow(attr(pd_cv, "by_split")) == 3, pd_cv$mean[1] > pd_cv$mean[2])
# The interaction screen: the planted pair above the additive model's.
screen <- function(f) {
  interaction_screen(glm_fit(f, ex, family = "poisson"), ex, dev, c("age", "power", "noise"),
                     n_repeats = 20, seed = 0)
}
with_term <- screen(planted)
stopifnot(identical(with_term$a, c("age", "age", "power")), identical(with_term$b, c("power", "noise", "noise")),
          with_term$strength[1] > 0, screen(additive)$strength[1] < 0, all(with_term$strength[2:3] == 0))
near(with_term$strength[1], 0.020418701731271027, 1e-9)
near(with_term$loss_a + with_term$loss_b - with_term$loss_ab, with_term$strength, 1e-12)
# CAS Monograph 16, Tables 9.1 to 9.4: seed 3 shuffles three rows as the
# monograph does, and age's importance is 0.133, 100 on its scale.
stopifnot(identical(permutation(3, 3), c(3, 1, 2)))
toy <- structure(list(), class = "toy_age_model")
predict.toy_age_model <- function(object, newdata, ...) newdata$age / 100
registerS3method("predict", "toy_age_model", predict.toy_age_model)
toy_data <- data.frame(age = c(20, 30, 40), height = c(1.4, 1.5, 1.6), y = c(0.2, 0.3, 0.4))
mae <- function(m, test) mean(abs(test$y - stats::predict(m, test)))
toy_imp <- permutation_importance(toy, toy_data, mae, c("age", "height"), seed = 3)
stopifnot(round(toy_imp$importance[1], 3) == 0.133, toy_imp$importance[2] == 0,
          identical(toy_imp$relative, c(100, 0)))

# MARS knots (#465) against R earth 5.3, degree = 1, on the Chalk portfolio
# (validation/scripts/r_mars_earth.R).
chalk <- utils::read.csv("examples/data/chalk_portfolio.csv", comment.char = "#")
chalk_x <- chalk[c("age", "veh_age", "exper", "noise")]
chalk_freq <- chalk$claims / chalk$exposure
mk <- mars_knots(chalk_x, chalk_freq)
stopifnot(identical(mk$knots$age, 28), identical(mk$knots$veh_age, 5), length(mk$knots$noise) == 0,
          identical(unname(mk$linear), c(TRUE, TRUE, FALSE, FALSE)))
near(mk$rss, 7229.20770861623, 1e-9)
near(mk$gcv, 0.36178591949671, 1e-9)
mkw <- mars_knots(chalk_x, chalk_freq, weights = chalk$exposure, max_terms = 15, threshold = 0)
stopifnot(identical(mkw$knots$exper, c(2, 7, 26)), mkw$steps$terms[4] == 2)
near(mkw$rss, 5166.57719014687, 1e-9)
mkb <- mars_knots(chalk_x, chalk_freq, max_terms = 15, threshold = 0, prune = TRUE)
near(mkb$gcv, 0.361653658764754, 1e-9)
stopifnot(nrow(mkb$terms) == 6, nrow(mkb$forward_terms) > nrow(mkb$terms))
rhs <- paste(mars_formula(mkw), collapse = " + ")
refit <- glm_fit(stats::as.formula(paste("freq ~", rhs)), cbind(chalk_x, freq = chalk_freq),
                 family = "gaussian", weights = chalk$exposure)
near(refit@deviance, mkw$rss, 1e-8)

# Draws from another sampler (#395): real CmdStan output against posterior
# and loo on the same files (validation/scripts/r_stan_csv.R).
root <- if (dir.exists("validation")) "validation" else file.path("..", "..", "..", "validation")
sd8 <- read_stan_csv(file.path(root, "data", "stan_eight_schools", sprintf("chain-%d.csv", 1:4)))
stopifnot(identical(dim(sd8), c(250L, 4L, 19L)), attr(sd8, "warmup_dropped") == 150,
          "divergent__" %in% dimnames(attr(sd8, "sampler"))[[3]])
sref <- utils::read.csv(file.path(root, "reference", "stan_csv_posterior.csv"), stringsAsFactors = FALSE)
ssum <- draws_summary(sd8)
sloo <- draws_loo(sd8)
est <- c(elpd_loo = "elpd", p_loo = "p", looic = "ic", se_elpd_loo = "se")
for (i in seq_len(nrow(sref))) {
  v <- sref$variable[i]; q <- sref$quantity[i]; want8 <- sref$expected[i]
  tol <- max(sref$abs_tol[i], sref$rel_tol[i] * abs(want8))
  got <- if (startsWith(v, "chain ")) {
    attr(sd8, "divergences")[as.integer(sub("chain ", "", v))]
  } else if (v == "log_lik") {
    sloo$estimates[[est[[q]]]]
  } else if (startsWith(v, "log_lik[") && q %in% c("r_eff", "elpd_loo", "pareto_k")) {
    j <- as.integer(sub("log_lik\\[(\\d+)\\]", "\\1", v))
    switch(q, r_eff = sloo$r_eff[j], elpd_loo = sloo$pointwise[j], pareto_k = sloo$pareto_k[j])
  } else {
    ssum[[q]][ssum$variable == v]
  }
  if (!(abs(got - want8) <= tol)) stop(sprintf("%s %s: got %.17g, want %.17g", v, q, got, want8))
}
# The existing chain fixture through mcmc_draws, against posterior.
mc <- utils::read.csv(file.path(root, "data", "mcmc_chains.csv"))
vars <- setdiff(names(mc), c("chain", "iteration"))
arr <- array(NA_real_, c(1001, 4, length(vars)), dimnames = list(NULL, NULL, vars))
for (c in 0:3) arr[, c + 1, ] <- as.matrix(mc[mc$chain == c, vars])
mref <- utils::read.csv(file.path(root, "reference", "mcmc_posterior.csv"))
msum <- draws_summary(mcmc_draws(arr))
for (i in which(mref$quantity %in% c("rhat", "ess_bulk", "ess_tail"))) {
  near(msum[[mref$quantity[i]]][msum$variable == mref$variable[i]], mref$expected[i], 1e-10)
}
if (requireNamespace("posterior", quietly = TRUE)) {
  pd <- posterior::as_draws_array(arr)
  near(draws_summary(pd)$ess_bulk, posterior::summarise_draws(pd, "ess_bulk")$ess_bulk, 1e-10)
  near(draws_summary(posterior::as_draws_df(pd))$rhat,
       posterior::summarise_draws(pd, "rhat")$rhat, 1e-10)
}
stopifnot(inherits(try(mcmc_draws(matrix(1:4, 2)), silent = TRUE), "try-error"))

cat("prospicio R models tests passed\n")

# Auto-calibration (#545): isotonic regression and the balance correction
# against scikit-learn 1.5.2, recalibration, reliability and the CORP terms.
iref <- utils::read.csv(file.path(root, "reference", "isotonic_sklearn.csv"), stringsAsFactors = FALSE,
                        colClasses = "character")
inum <- function(s) if (is.na(s) || s == "") NULL else as.double(strsplit(s, ";")[[1]])
for (i in seq_len(nrow(iref))) {
  r <- iref[i, ]
  if (r$case == "isotonic") {
    f <- isotonic_fit(inum(r$x), inum(r$y), inum(r$w), increasing = r$increasing == "1")
    want <- inum(r$expected)
    stopifnot(all(abs(predict(f, inum(r$query)) - want) <= 1e-12 * pmax(1, abs(want))))
  } else {
    want <- as.double(r$expected)
    stopifnot(abs(balance_correction(inum(r$y), inum(r$query), inum(r$w)) - want) <= 1e-14 * want)
  }
}
m <- recalibrate(c(1, 2, 3, 4), c(0, 1, 1, 2), y_min = 0.1)
stopifnot(all(abs(predict(m, c(1, 2.5, 4)) - c(0.1, 1, 2)) < 1e-12))
stopifnot(identical(predict(isotonic_fit(c(1, 3), c(1, 3), interpolation = "step"), 2), 1))
yy <- ((0:199 * 37) %% 11)
pp <- 1 + yy * 0.7 + (0:199 %% 3)
for (fam in c("gaussian", "poisson")) {
  d <- murphy(yy, pp, family = fam)
  stopifnot(abs(d$score - (d$miscalibration - d$discrimination + d$uncertainty)) < 1e-12)
  stopifnot(abs(murphy(yy, predict(isotonic_fit(pp, yy), pp), family = fam)$miscalibration) < 1e-12)
}
rb <- reliability(yy, pp, bins = 4)
stopifnot(sum(rb$count) == 200, abs(sum(rb$weight * rb$mean_actual) - sum(yy)) < 1e-9)
cat("calibration tests passed\n")
