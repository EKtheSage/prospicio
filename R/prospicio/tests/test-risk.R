library(prospicio)

x <- sampled(c(10, 20, 30, 40, 50))

# Distortions: TVaR agrees with TVaR(); identity parameters give the mean.
stopifnot(abs(risk_measure(x, distortion("tvar", 0.6)) - TVaR(x, 0.6)) < 1e-12)
for (d in list(distortion("wang", 0), distortion("proportional_hazard", 1), distortion("dual_power", 1))) {
  stopifnot(abs(risk_measure(x, d) - 30) < 1e-12)
}
stopifnot(abs(distortion_g(distortion("dual_power", 2), 0.3) - 0.51) < 1e-15)
stopifnot(identical(distortion_weights(distortion("tvar", 0.5), 4), c(0, 0, 0.5, 0.5)))
stopifnot(risk_measure(grid_distribution(1, c(0.5, 0.25, 0.25)), distortion("tvar", 0.5)) == 1.5)
stopifnot(inherits(try(distortion("proportional_hazard", 1.5), silent = TRUE), "try-error"))
stopifnot(inherits(try(distortion("variance", 1), silent = TRUE), "try-error"))

# More families, on Monograph 15's InsCo totals (assets 100, 15% cost of
# capital); calibration matches aggregate 1.0.1's parameters.
insco <- sampled(c(22, 28, 36, 40, 40, 40, 40, 55, 65, 100))
insco_p <- (46.6 + 15) / 1.15
ccoc <- distortion("ccoc", 0.15)
stopifnot(abs(risk_measure(insco, ccoc) - insco_p) < 1e-12)
stopifnot(abs(distortion_mass(ccoc) - 0.15 / 1.15) < 1e-15)
stopifnot(abs(risk_measure(insco, distortion("bitvar", c(0, 1, 0.15 / 1.15))) - insco_p) < 1e-12)
for (d in list(
  distortion("weighted_tvar", c(0.1, 0.5, 0.9), weights = c(0.2, 0.5, 0.3)),
  distortion("capped_linear", 1.4, r0 = 0.05), distortion("capped_log_linear", 0.8, r0 = 0.1),
  distortion("lep", 0.2, r0 = 0.02), distortion("linear_yield", 0.5, r0 = 0.03),
  distortion("beta", c(0.6, 1.5)),
  distortion_mixture(list(distortion("wang", 0.3), ccoc), c(0.4, 0.6)),
  distortion_minimum(list(distortion("proportional_hazard", 0.5), distortion("tvar", 0.6))),
  distortion_convex(c(0.01, 0.1), c(0.05, 0.25))
)) {
  m <- risk_measure(insco, d)
  stopifnot(m >= 46.6, m <= 100, distortion_g_dual(d, 0.3) <= 0.3 + 1e-12)
  stopifnot(distortion_g(d, distortion_g_inv(d, 0.4)) >= 0.4 - 1e-12)
}
want <- c(proportional_hazard = 0.7204792831878889, wang = 0.3427309477594301,
          dual_power = 1.5951514670652984)
for (f in names(want)) {
  d <- calibrate_distortion(f, insco, insco_p)
  stopifnot(abs(risk_measure(insco, d) - insco_p) < 1e-9, abs(d@param - want[[f]]) < 1e-6)
}
stopifnot(abs(calibrate_distortion("ccoc", insco, insco_p)@param - 0.15) < 1e-12)
capped <- calibrate_distortion("tvar", insco, 50, assets = 65)
stopifnot(abs(risk_measure(sampled(pmin(c(22, 28, 36, 40, 40, 40, 40, 55, 65, 100), 65)), capped) - 50) < 1e-9)
stopifnot(inherits(try(calibrate_distortion("tvar", insco, 70, assets = 65), silent = TRUE), "try-error"))

# Copulas: uniforms in (0, 1), rows replay, Kendall's tau near its closed form.
r <- matrix(c(1, 0.6, 0.6, 1), 2)
for (cop in list(gaussian_copula(r), t_copula(r, 4), archimedean_copula("gumbel", 2.5))) {
  u <- copula_sample(cop, 2000, seed = 5)
  stopifnot(nrow(u) == 2000, ncol(u) == 2, all(u > 0 & u < 1))
  stopifnot(identical(copula_sample(cop, 10, seed = 5), u[1:10, ]))
}
tau <- cor(copula_sample(gaussian_copula(r), 2000, seed = 1), method = "kendall")[1, 2]
stopifnot(abs(tau - 2 / pi * asin(0.6)) < 0.04)
tau <- cor(copula_sample(archimedean_copula("clayton", 2), 2000, seed = 1), method = "kendall")[1, 2]
stopifnot(abs(tau - 0.5) < 0.04)
stopifnot(gaussian_copula(r)@dimension == 2, archimedean_copula("joe", 2, dim = 3)@dimension == 3)
stopifnot(inherits(try(gaussian_copula(matrix(c(1, 2, 2, 1), 2)), silent = TRUE), "try-error"))
stopifnot(inherits(try(archimedean_copula("gumbel", 0.5), silent = TRUE), "try-error"))

# Simulation with a copula, then allocation that adds up.
pd <- copula_simulate(
  gaussian_copula(r),
  list(lognormal_from_mean_cv(100, 0.3), lognormal_from_mean_cv(50, 1.5)),
  n_sims = 5000, seed = 3,
  keys = data.frame(lob = c("motor", "property"))
)
stopifnot(identical(pd@keys$lob, c("motor", "property")))
for (d in list(distortion("tvar", 0.99), distortion("wang", 0.5))) {
  a <- allocate(pd, d)
  stopifnot(identical(a$lob, c("motor", "property")))
  stopifnot(abs(sum(a$contribution) - risk_measure(pd, d)) < 1e-9 * risk_measure(pd, d))
}
stopifnot(identical(copula_simulate(gaussian_copula(r), list(lognormal(0, 1), lognormal(0, 1)), 10, 1)@keys$component, c(1, 2)))
stopifnot(inherits(try(copula_simulate(gaussian_copula(r), list(lognormal(0, 1)), 10, 1), silent = TRUE), "try-error"))

# Iman-Conover keeps marginals and reaches the target.
indep <- copula_simulate(gaussian_copula(diag(2)), list(lognormal(0, 0.5), lognormal(1, 1)), 4000, seed = 9)
joined <- iman_conover(indep, matrix(c(1, 0.8, 0.8, 1), 2), seed = 2)
m0 <- draw_matrix(indep)
m1 <- draw_matrix(joined)
stopifnot(identical(sort(m0[, 1]), sort(m1[, 1])), identical(sort(m0[, 2]), sort(m1[, 2])))
stopifnot(abs(cor(m1, method = "spearman")[1, 2] - 6 / pi * asin(0.4)) < 0.02)

# EVT: GPD fit on exact quantiles, and a peaks-over-threshold tail.
y <- 2 * expm1(0.25 * -log1p(-ppoints(2000))) / 0.25
f <- gpd_fit(y)
stopifnot(identical(names(f), c("xi", "beta")), abs(f[["xi"]] - 0.25) < 0.02)
stopifnot(inherits(try(gpd_fit(c(1, 1, 1)), silent = TRUE), "try-error"))
s <- sampled(qlnorm(ppoints(20000)))
tl <- pot_tail(s, 0.9)
stopifnot(abs(tl@p_exceed - 0.1) < 1e-3, length(VaR(tl, c(0.99, 0.999))) == 2)
stopifnot(TVaR(tl, 0.99) > VaR(tl, 0.99))
stopifnot(inherits(try(VaR(tl, 0.5), silent = TRUE), "try-error"))

# Capital allocation.
cpd <- predictive_distribution(
  t(sapply(0:299, function(i) c((i * 37) %% 101, (i * 53) %% 97, (i * i) %% 89))),
  data.frame(lob = c("a", "b", "c"))
)
d <- distortion("tvar", 0.9)
eu <- capital_allocation(cpd, d)
stopifnot(isTRUE(all.equal(eu$by_component$allocated, allocate(cpd, d)$contribution)))
for (m in c("euler", "covariance", "proportional", "shapley")) {
  a <- capital_allocation(cpd, d, m)
  stopifnot(abs(sum(a$by_component$allocated) - a$total) < 1e-9)
  stopifnot(abs(sum(a$by_component$diversification) - a$diversification_benefit) < 1e-9)
}
stopifnot(sum(capital_allocation(cpd, d, "marginal")$by_component$allocated) < eu$total)

# Tail diagnostics.
me <- mean_excess(c(1, 2, 3, 4), 2)
stopifnot(me$mean_excess == 1.5, me$n_above == 2)
hx <- (1 - (seq_len(2000) - 0.5) / 2000)^-0.5
stopifnot(abs(hill_estimator(hx, 200) - 0.5) < 0.02)
p <- (seq_len(1000) - 0.5) / 1000
bx <- ((1 - p)^(-1 / 2) - 1)^(1 / 1.5)
td <- threshold_diagnostics(bx, c(1, 2, 1e9))
stopifnot(nrow(td) == 3, td$n_above[1] == sum(bx > 1), td$n_above[3] == 0)
# Against ismev::gpd.fit on the same data at u = 1 (validation/reference/evt_thresholds_r.csv).
stopifnot(abs(td$xi[1] - 0.26834735586661257) < 1e-9)
stopifnot(abs(td$xi_se[1] - 0.079570667971910328) < 1e-8)
stopifnot(is.nan(td$xi[3]))

# Exponential spectral, entropic, Esscher, MES, CoVaR.
k <- 3
u <- (seq_len(100000) - 0.5) / 100000
stopifnot(abs(risk_measure(sampled(u), distortion("exponential", k)) -
  (exp(k) * (k - 1) + 1) / (k * expm1(k))) < 1e-6)
stopifnot(abs(entropic_risk(c(0, 1), log(2)) - log2(1.5)) < 1e-12)
stopifnot(abs(esscher_premium(sampled(c(0, 1)), log(3)) - 0.75) < 1e-12)
stopifnot(inherits(try(entropic_risk(1, 0), silent = TRUE), "try-error"))
stopifnot(isTRUE(all.equal(marginal_expected_shortfall(cpd, 0.9)$contribution,
                           allocate(cpd, d)$contribution)))
ea <- esscher_allocation(cpd, 0.02)
stopifnot(abs(sum(ea$contribution) - esscher_premium(cpd, 0.02)) < 1e-9)
spd <- predictive_distribution(matrix(c(1, 2, 3, 4, 0, 1, 5, 1), ncol = 2),
                               data.frame(lob = c("a", "b")))
stopifnot(covar(spd, list(lob = "a"), 0.75, 0.5) == 5)
stopifnot(inherits(try(covar(spd, list(lob = "z"), 0.75, 0.5), silent = TRUE), "try-error"))


# Distortion parameters by name (#205).
stopifnot(identical(distortion("proportional_hazard", 0.7)@parameters, c(rho = 0.7)),
          identical(distortion("capped_linear", 2, r0 = 0.1)@parameters, c(r0 = 0.1, slope = 2)))

# Copula distribution functions.
gc <- gaussian_copula(matrix(c(1, 0.6, 0.6, 1), 2))
stopifnot(abs(copula_cdf(gc, c(0.5, 0.5)) - (0.25 + asin(0.6) / (2 * pi))) < 1e-15)
ac <- archimedean_copula("clayton", 2)
stopifnot(max(abs(copula_cdf(ac, rbind(c(0.3, 0.6), c(1, 0.4))) -
                  c((0.3^-2 + 0.6^-2 - 1)^-0.5, 0.4))) < 1e-14,
          inherits(try(copula_cdf(t_copula(matrix(c(1, 0.5, 0.5, 1), 2), 4), c(0.5, 0.5)),
                       silent = TRUE), "try-error"))

# Nested Archimedean copulas.
nac <- nested_archimedean_copula("gumbel",
  nac_node(1.5, 1, list(nac_node(3, c(2, 3)), nac_node(2, c(4, 5)))))
stopifnot(nac@dimension == 5, abs(copula_cdf(nac, c(1, 0.5, 1, 1, 1)) - 0.5) < 1e-15)
one <- nested_archimedean_copula("clayton", nac_node(2, 1:3))
flat <- archimedean_copula("clayton", 2, dim = 3)
stopifnot(identical(copula_sample(one, 20, seed = 4), copula_sample(flat, 20, seed = 4)))
stopifnot(inherits(try(nested_archimedean_copula("gumbel",
  nac_node(3, 1, list(nac_node(2, 2:3)))), silent = TRUE), "try-error"))

# Vine copulas.
v <- vine_copula("d", 1:3, list(
  list(pair_copula("gumbel", 2), pair_copula("clayton", 1.5, 90)),
  list(pair_copula("frank", 3))))
stopifnot(v@dimension == 3, is.finite(copula_log_density(v, c(0.2, 0.7, 0.4))))
stopifnot(length(copula_log_density(v, rbind(c(0.2, 0.7, 0.4), c(0.5, 0.5, 0.5)))) == 2)
stopifnot(all(dim(copula_sample(v, 4, seed = 1)) == c(4, 3)))
stopifnot(inherits(try(pair_copula("gaussian", 0.5, 90), silent = TRUE), "list"))
stopifnot(inherits(try(vine_copula("d", 1:3, list(list(pair_copula("gaussian", 0.5, 90),
  pair_copula("frank", 1)))), silent = TRUE), "try-error"))
stopifnot(inherits(try(copula_log_density(archimedean_copula("gumbel", 2), c(0.5, 0.5)),
  silent = TRUE), "try-error"))

# Iman-Conover with t scores and shuffled rows (#378).
set.seed(4)
ic_pd <- predictive_distribution(cbind(rlnorm(5000, 0, 0.5), rlnorm(5000, 0, 1)),
                                 data.frame(lob = c("a", "b")))
ic_r <- matrix(c(1, 0.3, 0.3, 1), 2)
joint <- function(p) {
  m <- draw_matrix(p)
  q <- apply(m, 2, function(v) sort(v)[5000 - 50 + 1])
  sum(m[, 1] >= q[1] & m[, 2] >= q[2])
}
ic_normal <- iman_conover(ic_pd, ic_r, seed = 5)
stopifnot(joint(iman_conover(ic_pd, ic_r, seed = 5, dof = 2)) > 1.5 * joint(ic_normal))
stopifnot(!is.unsorted(draw_matrix(ic_normal)[, 1]))
ic_shuffled <- draw_matrix(iman_conover(ic_pd, ic_r, seed = 5, shuffle_rows = TRUE))[, 1]
stopifnot(is.unsorted(ic_shuffled), identical(sort(ic_shuffled), draw_matrix(ic_normal)[, 1]))
stopifnot(inherits(try(iman_conover(ic_pd, ic_r, seed = 5, dof = 2, scores = rep(0, 5000)),
                       silent = TRUE), "try-error"))

# Sobol points and the multivariate normal cdf (#565).
stopifnot(identical(sobol_points(4, 2, scramble = FALSE), rbind(c(0, 0), c(0.5, 0.5), c(0.75, 0.25), c(0.25, 0.75))))
s <- sobol_points(256, 3, seed = 5)
stopifnot(identical(s, sobol_points(256, 3, seed = 5)), all(sort(floor(s[, 2] * 256)) == 0:255))
stopifnot(inherits(try(sobol_points(4, 1112), silent = TRUE), "try-error"))
for (d in c(3, 5, 8)) {
  r <- matrix(0.5, d, d)
  diag(r) <- 1
  p <- mvn_cdf(rep(0, d), r, abs_tol = 1e-6)
  stopifnot(attr(p, "converged"), abs(p - 1 / (d + 1)) < 3e-6)
}
p <- mvn_cdf(c(0.1, -0.5, 1.2), diag(3))
stopifnot(abs(p - prod(pnorm(c(0.1, -0.5, 1.2)))) < 1e-5)
q <- mvn_cdf(rep(1, 3), diag(3), lower = rep(-1, 3))
stopifnot(abs(q - (pnorm(1) - pnorm(-1))^3) < 1e-5)
stopifnot(attr(mvn_cdf(c(0, 0), matrix(c(1, 0.5, 0.5, 1), 2)), "error") == 0)
stopifnot(inherits(try(mvn_cdf(rep(0, 3), matrix(c(1, 1, 0, 1, 1, 0, 0, 0, 1), 3)), silent = TRUE), "try-error"))
r <- rbind(c(1, 0.5, 0.2), c(0.5, 1, 0.4), c(0.2, 0.4, 1))
g <- gaussian_copula(r)
e <- copula_cdf_estimate(g, c(0.6, 0.4, 0.7), abs_tol = 1e-7)
stopifnot(e$converged, abs(copula_cdf(g, c(0.6, 0.4, 0.7)) - e$value) < 2e-5)
u <- copula_sample_qmc(g, 2^14, seed = 11)
stopifnot(abs(mean(u[, 1] <= 0.6 & u[, 2] <= 0.4 & u[, 3] <= 0.7) - e$value) < 2e-3)
stopifnot(inherits(try(copula_sample_qmc(archimedean_copula("gumbel", 2), 8), silent = TRUE), "try-error"))

# The Student t pair and fitting pair copulas (#568).
tv <- vine_copula("d", 1:2, list(list(pair_copula("student", 0.6, nu = 4))))
stopifnot(abs(copula_log_density(tv, c(0.3, 0.6)) - (-0.02730765499122425)) < 1e-10)
stopifnot(inherits(try(vine_copula("d", 1:2, list(list(pair_copula("student", 0.6)))), silent = TRUE), "try-error"))
stopifnot(identical(pseudo_obs(c(3, 1, 2, 2)), c(0.8, 0.2, 0.5, 0.5)))
stopifnot(abs(kendall_tau(1:4, c(1, 3, 2, 4)) - 4 / 6) < 1e-15)
set.seed(568)
x <- rnorm(1500)
y <- 0.7 * x + sqrt(1 - 0.49) * rnorm(1500)
uu <- pseudo_obs(x)
vv <- pseudo_obs(y)
g <- fit_pair_copula(uu, vv, "gaussian")
stopifnot(abs(g$pair[[2]] - 0.7) < 0.05, g$n_parameters == 1)
sel <- select_pair_copula(uu, vv)
stopifnot(sel$aic <= g$aic + 1e-9)
it <- select_pair_copula(runif(300), runif(300), independence_test = 0.05)
stopifnot(it$pair[[1]] %in% c("independence", "gaussian", "student", "clayton", "gumbel", "frank", "joe"))
cat("prospicio R risk tests passed\n")
# The covariance allocation of the margin keeps each unit's mean, and the
# Shapley value of any cost agrees with the distortion's Shapley.
rows <- expand.grid(a = c(0, 0, 8, 10), b = c(0, 0, 1, 90))
sd_pd <- predictive_distribution(as.matrix(rows), data.frame(unit = c("X1", "X2")))
tv <- distortion("tvar", 0.75)
cv <- capital_allocation(sd_pd, tv, "covariance")
cm <- capital_allocation(sd_pd, tv, "covariance_margin")
stopifnot(abs(sum(cm$by_component$allocated) - cm$total) < 1e-9)
stopifnot(all(abs(cm$by_component$allocated -
  (c(4.5, 22.75) + (cv$total - 27.25) * cv$by_component$allocated / cv$total)) < 1e-9))
sh <- shapley_allocation(sd_pd, function(x) risk_measure(sampled(x), tv))
stopifnot(all(abs(sh$by_component$allocated -
  capital_allocation(sd_pd, tv, "shapley")$by_component$allocated) < 1e-9))
sdp <- shapley_allocation(sd_pd, function(x) mean(x) + 0.5 * sqrt(mean((x - mean(x))^2)))
stopifnot(abs(sdp$by_component$allocated[1] -
  0.5 * (sdp$by_component$standalone[1] + sdp$total - sdp$by_component$standalone[2])) < 1e-9)
stopifnot(inherits(try(shapley_allocation(sd_pd, function(x) stop("boom")), silent = TRUE), "try-error"))

# Haircut and equal risk by stand-alone TVaR: Pricing Insurance Risk,
# Exercises 205 and 206.
hc <- haircut_allocation(sd_pd, 80, "tvar", 0.75)
stopifnot(all(abs(hc$by_component$allocated - c(8, 72)) < 1e-12))
er <- equal_risk_allocation(sd_pd, 80, "tvar")
stopifnot(abs(er$level - 0.6796) < 5e-5, all(abs(er$by_component$allocated - c(9.56, 70.44)) < 5e-3))
stopifnot(abs(sum(equal_risk_allocation(sd_pd, 80, "epd")$by_component$allocated) - 80) < 1e-9)
# Worst VaR by the Rearrangement Algorithm: Pricing Insurance Risk,
# Exercise 59, from the guide's rotated starts.
ra_m <- lapply(c(1, 2, 3), function(cv) lognormal_from_mean_cv(10, cv))
for (k in list(c(7, 353.8), c(13, 353.6), c(19, 352.9))) {
  stopifnot(abs(worst_var(ra_m, 0.99, n = 40, rotate = k[1])$value - k[2]) < 0.05)
}
stopifnot(rearrange(cbind(1:4, 1:4))$value == 5)


# A results table's statistics: R's quantile(type = 7) and the integral of
# that quantile function for the linear TVaR; VaR() and TVaR() keep their
# defaults and take the same methods.
st_x <- c(7, 0, 100, 3, 1, 0, 45, 7, 12, 20)
st_p <- c(0, 10, 25, 50, 90, 99, 100)
st <- statistics(st_x, st_p, tvar_levels = c(0, 50, 90, 99), quantile = "linear", tvar = "linear")
stopifnot(all(abs(st[paste("Percentile", st_p)] - quantile(st_x, st_p / 100, type = 7, names = FALSE)) < 1e-12))
st_tvar <- function(p) integrate(function(u) quantile(st_x, u, type = 7, names = FALSE), p, 1,
                                 subdivisions = 1000L, rel.tol = 1e-10)$value / (1 - p)
for (q in c(0, 50, 90, 99)) stopifnot(abs(st[[paste("TVaR", q)]] / st_tvar(q / 100) - 1) < 1e-6)
stopifnot(st[["Mean"]] == 19.5, abs(st[["Mean + 2 SD"]] - 19.5 - 2 * sqrt(mean((st_x - 19.5)^2))) < 1e-12)
st_s <- sampled(st_x)
stopifnot(VaR(st_s, 0.25, method = "linear") == st[["Percentile 25"]])
stopifnot(TVaR(st_s, 0.9, method = "linear") == st[["TVaR 90"]])
stopifnot(VaR(st_s, 0.25) == quantile(st_x, 0.25, type = 1, names = FALSE))
stopifnot(inherits(try(VaR(st_s, 0.5, method = "nearest"), silent = TRUE), "try-error"))
# The k-th largest year and the mean of the k worst whole years (#582).
ky_x <- as.numeric((0:1999 * 7919) %% 1013)
ky_d <- sort(ky_x, decreasing = TRUE)
ky_s <- sampled(ky_x)
stopifnot(VaR(ky_s, 0.99, method = "kth_largest") == ky_d[20], VaR(ky_s, 0.99) == ky_d[21])
stopifnot(abs(TVaR(ky_s, 0.99, method = "whole_years") - mean(ky_d[1:20])) < 1e-9)
stopifnot(abs(TVaR(ky_s, 0.99, method = "whole_years") - TVaR(ky_s, 0.99)) < 1e-9)
stopifnot(abs(TVaR(ky_s, 0.987, method = "whole_years") - mean(ky_d[1:26])) < 1e-9)
ky_st <- statistics(ky_x, 99, tvar_levels = 98.7, quantile = "kth_largest", tvar = "whole_years")
stopifnot(ky_st[["Percentile 99"]] == ky_d[20], abs(ky_st[["TVaR 98.7"]] - mean(ky_d[1:26])) < 1e-9)
stopifnot(inherits(try(VaR(ky_s, 0.5, method = "kth"), silent = TRUE), "try-error"))
# The mean between two levels (#583), the difference of two tails.
wt_s <- sampled(1:2000)
wt_area <- function(p, method = "exact") if (p == 1) 0 else (1 - p) * TVaR(wt_s, p, method = method)
for (ab in list(c(0.98, 0.99), c(0.5, 0.9), c(0.123, 0.987), c(0.9, 1))) {
  for (m in c("exact", "linear")) {
    want <- (wt_area(ab[1], m) - wt_area(ab[2], m)) / (ab[2] - ab[1])
    stopifnot(abs(window_tvar(wt_s, ab[1], ab[2], method = m) - want) < 1e-9 * want)
  }
}
stopifnot(window_tvar(wt_s, 0.98, 0.99, method = "whole_years") == mean(1961:1980),
          abs(window_tvar(grid_distribution(1, c(0.5, 0.3, 0.2)), 0.6, 0.9) - 0.4 / 0.3) < 1e-12,
          inherits(try(window_tvar(wt_s, 0.9, 0.9), silent = TRUE), "try-error"))
st_pd <- predictive_distribution(cbind(1:10, 2 * (1:10)), data.frame(lob = c("a", "b")))
st_tab <- statistics(st_pd, 50, tvar_levels = 90, sd_multiples = 1)
stopifnot(identical(st_tab$lob, c("a", "b")), all(st_tab$Mean == c(5.5, 11)))
stopifnot(all(st_tab[["Percentile 50"]] == c(5, 10)), isTRUE(all.equal(st_tab[["TVaR 90"]], c(10, 20), tolerance = 1e-12)))

# Items at the driver's percentiles, against the reference's selection in R:
# o = order() (stable), round() halves to even, 0-based h.
set.seed(7)
co_n <- 2001
co_ulr <- round(rlnorm(co_n, -0.4, 0.3), 2)
co_cc <- 0.3 - 0.2 * pmin(pmax(co_ulr - 0.5, 0), 0.5)
co <- co_statistics(list(cc = co_cc, margin = 1 - co_ulr - co_cc), co_ulr, c(50, 90, 99.5), tvar_levels = c(90, 99))
co_o <- order(co_ulr)
for (q in c(50, 90, 99.5)) {
  stopifnot(co[1, paste("Percentile", q)] == co_cc[co_o[round(q / 100 * (co_n - 1)) + 1]])
}
for (q in c(90, 99)) {
  tail <- co_o[(ceiling(q / 100 * (co_n - 1)) + 1):co_n]
  stopifnot(isTRUE(all.equal(co[1, paste("TVaR", q)], mean(co_cc[tail]), tolerance = 1e-12)))
}
stopifnot(co$Mean[1] == co_cc[co_o[which(co_ulr[co_o] >= mean(co_ulr))[1]]], identical(co$item, c("cc", "margin")))
co_small <- c(0.6, 0.9, 0.5, 0.7, 0.8)
co_pick <- sapply(c("round_half_even", "floor", "ceil"), function(r)
  co_statistics(list(x = co_small), co_small, 62.5, tvar_levels = 50, rule = r)[["Percentile 62.5"]])
stopifnot(all(co_pick == c(0.7, 0.7, 0.8)))
co_pd <- predictive_distribution(cbind(co_ulr, co_cc), data.frame(kind = c("ulr", "cc")))
co_tab <- co_statistics(co_pd, list(kind = "ulr"), 90, tvar_levels = 90)
stopifnot(identical(co_tab$kind, c("ulr", "cc")), co_tab[2, "Percentile 90"] == co[1, "Percentile 90"])
stopifnot(nrow(co_statistics(co_pd, NULL, 50)) == 2)
stopifnot(inherits(try(co_statistics(list(x = 1), c(1, 2), 50), silent = TRUE), "try-error"))

# Risk ratios under the textbook and the reference's conventions.
rr_x <- c(-3, -1, 0, 2, 4, 4)
rr <- risk_ratios(rr_x, tvar_levels = 50)
stopifnot(identical(names(rr), c("Sharpe", "Sortino", "Omega", "TVaR ratio 50")))
stopifnot(abs(rr[["Sharpe"]] - 1 / sqrt(mean(rr_x^2) - 1)) < 1e-12, abs(rr[["Sortino"]] - 1 / sqrt(10 / 6)) < 1e-12)
stopifnot(abs(rr[["Omega"]] - 2.5) < 1e-12, abs(rr[["TVaR ratio 50"]] - 0.75) < 1e-12)
rr_ref <- risk_ratios(rr_x, sortino = "below_threshold_sd", omega = "count")
stopifnot(abs(rr_ref[["Sortino"]] - 1) < 1e-12, rr_ref[["Omega"]] == 1.5)
rr_pd <- risk_ratios(predictive_distribution(cbind(rr_x, 2 * rr_x), data.frame(lob = c("a", "b"))), threshold = 1)
stopifnot(identical(rr_pd$lob, c("a", "b")), rr_pd$Sharpe[1] == 0)
stopifnot(inherits(try(risk_ratios(rr_x, omega = "ratio"), silent = TRUE), "try-error"))

# Bootstrap standard errors (#488): one draw shared by every column, a part
# held fixed.
bs_x <- as.double((seq_len(1000) * 37) %% 1000)
bs <- bootstrap_se(bs_x, c(50, 99), replicates = 200, seed = 7)
stopifnot(bs["Mean"] > 7, bs["Mean"] < 11, identical(bs, bootstrap_se(bs_x, c(50, 99), replicates = 200, seed = 7)))
bs_by <- bootstrap_se(list(a = bs_x, b = 2 * bs_x), 50, replicates = 50, seed = 1)
stopifnot(identical(bs_by$item, c("a", "b")),
          abs(bs_by[["Percentile 50"]][2] - 2 * bs_by[["Percentile 50"]][1]) < 1e-9)
bs_held <- bootstrap_se(bs_x, 50, replicates = 50, seed = 1, hold_fixed = bs_x)
stopifnot(all(abs(bs_held) < 1e-9))
bs_none <- bootstrap_se(bs_x, 50, replicates = 50, seed = 1, hold_fixed = 0 * bs_x)
bs_plain <- bootstrap_se(bs_x, 50, replicates = 50, seed = 1)
stopifnot(abs(bs_none["Percentile 50"] - bs_plain["Percentile 50"]) < 1e-9)
bs_pd <- bootstrap_se(predictive_distribution(cbind(bs_x[1:200], bs_x[1:200] + 1), data.frame(lob = c("a", "b"))),
                      90, replicates = 30, seed = 2)
stopifnot(identical(bs_pd$lob, c("a", "b")), abs(bs_pd$SD[1] - bs_pd$SD[2]) < 1e-9)
stopifnot(inherits(try(bootstrap_se(bs_x, 50, replicates = 1), silent = TRUE), "try-error"))


# Multivariate normal and lognormal joints from a mean and covariance (#564).
mv_sd <- c(20, 10, 30)
mv_r <- matrix(c(1, 0.6, -0.2, 0.6, 1, 0.3, -0.2, 0.3, 1), 3)
mv_cov <- mv_r * outer(mv_sd, mv_sd)
mv_mean <- c(100, 50, 200)
mvn <- multivariate_normal(mv_mean, mv_cov)
stopifnot(mvn@kind == "normal", mvn@dimension == 3, all(mvn@mean == mv_mean))
stopifnot(max(abs(mvn@cov - mv_cov)) < 1e-12, max(abs(mvn@correlation - mv_r)) < 1e-12)
stopifnot(total(mvn)@mu == 350, abs(total(mvn)@sigma^2 - sum(mv_cov)) < 1e-9)
stopifnot(marginal(mvn, 3)@sigma == 30)
stopifnot(inherits(try(marginal(mvn, 4), silent = TRUE), "try-error"))
mv_pd <- predictive_from_gaussian(mvn, 50000, 7, keys = data.frame(line = c("a", "b", "c")))
mv_x <- draw_matrix(mv_pd)
stopifnot(max(abs(colMeans(mv_x) - mv_mean)) < 4 * 30 / sqrt(50000))
stopifnot(max(abs(cov(mv_x) - mv_cov) / outer(mv_sd, mv_sd)) < 0.03)
stopifnot(ks.test(rowSums(mv_x), "pnorm", 350, total(mvn)@sigma)$p.value > 0.01)
stopifnot(identical(draw_matrix(predictive_from_gaussian(mvn, 50000, 7, keys = data.frame(line = c("a", "b", "c")))), mv_x))
stopifnot(inherits(try(multivariate_normal(c(0, 0), matrix(1, 2, 2)), silent = TRUE), "try-error"))
mvl <- multivariate_lognormal(mv_mean, mv_cov)
stopifnot(mvl@kind == "lognormal", max(abs(mvl@mean / mv_mean - 1)) < 1e-12)
stopifnot(max(abs(mvl@cov - mv_cov)) < 1e-8, abs(mvl@total_mean - 350) < 1e-9)
mvl_back <- multivariate_lognormal(mu = mvl@mu, log_cov = mvl@log_cov)
stopifnot(max(abs(mvl_back@cov - mv_cov)) < 1e-8)
mvl_m <- marginal(mvl, 2)
stopifnot(abs(exp(mvl_m@meanlog + mvl_m@sdlog^2 / 2) - 50) < 1e-10)
mvl_x <- draw_matrix(predictive_from_gaussian(mvl, 50000, 3))
stopifnot(all(mvl_x > 0), ks.test(mvl_x[, 2], "plnorm", mvl_m@meanlog, mvl_m@sdlog)$p.value > 0.01)
stopifnot(inherits(try(total(mvl), silent = TRUE), "try-error"))
stopifnot(inherits(try(multivariate_lognormal(c(1, 1), matrix(c(1, -0.9, -0.9, 1), 2)), silent = TRUE), "try-error"))
stopifnot(inherits(try(multivariate_lognormal(c(1, 1)), silent = TRUE), "try-error"))
cat("multivariate joint tests passed\n")
