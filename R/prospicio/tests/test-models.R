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

cat("prospicio R models tests passed\n")

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
