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

cat("prospicio R risk tests passed\n")
