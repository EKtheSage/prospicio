library(prospicio)

near <- function(a, b, tol) stopifnot(all(abs(a - b) <= tol * pmax(1, abs(b))))

# Limited fluctuation: 1,082 claims at 90% within 5%.
s <- credibility_standard(0.90, 0.05, cv = 2)
near(s[["frequency"]], (qnorm(0.95) / 0.05)^2, 1e-13)
near(s[["pure_premium"]], 5 * s[["frequency"]], 1e-13)
near(limited_fluctuation_credibility(s[["frequency"]] / 4, s[["frequency"]]), 0.5, 1e-15)

# Bühlmann on a hand case: s2 = 4, a = 8 - 4/3.
fit <- buhlmann_straub(rbind(c(1, 2, 3), c(4, 5, 9)))
near(fit$within_variance, 4, 1e-14)
near(fit$between_variance, 8 - 4 / 3, 1e-14)

# Bühlmann–Straub with weights and a missing cell, both estimators.
x <- rbind(c(1, 2, NA), c(4, 5, 9), c(2, 2.5, 3))
w <- rbind(c(10, 20, NA), c(5, 5, 8), c(30, 40, 50))
fu <- buhlmann_straub(x, w)
stopifnot(identical(fu$group_weights, c(30, 18, 120)))
fi <- buhlmann_straub(x, w, estimator = "iterative")
stopifnot(fi$iterations > 0)
stopifnot(inherits(try(buhlmann_straub(x), silent = TRUE), "try-error"))

# Against actuar when it is installed (its parity also runs in Rust).
if (requireNamespace("actuar", quietly = TRUE)) {
  data(hachemeister, package = "actuar")
  r <- as.matrix(hachemeister[, 2:13]); ww <- as.matrix(hachemeister[, 14:25])
  ref <- actuar::cm(~state, hachemeister, ratios = ratio.1:ratio.12, weights = weight.1:weight.12)
  ours <- buhlmann_straub(r, ww)
  near(ours$premiums, as.numeric(predict(ref)), 1e-12)
}

# Hachemeister's regression model on a hand case and against actuar's
# premiums at time 13 (validation/reference/credibility_regression_actuar.csv).
hx <- rbind(c(10, 12, 13, 15, 16), c(8, 8.5, 9.5, 9, 10), c(14, 13, 16, 17, 19))
hf <- hachemeister(hx, adjust_intercept = TRUE)
stopifnot(identical(names(hf@collective), c("(Intercept)", "time")))
stopifnot(identical(dim(hf@credibility), c(2L, 2L, 3L)), hf@credibility[1, 2, 1] == 0)
hp <- predict(hf, data.frame(time = 6:7))
stopifnot(identical(dim(hp), c(2L, 3L)), hp[1, 2] < hp[1, 1], hp[1, 1] < hp[1, 3])
near(hp[1, ], as.numeric(hf@coefficients %*% c(1, 6)), 1e-13)
h1 <- hachemeister(hx, formula = ~1, adjust_intercept = TRUE)
near(h1@coefficients[, 1], buhlmann_straub(hx)$premiums, 1e-10)
stopifnot(inherits(try(hachemeister(hx, formula = ~ I(time * 0 + 2) + time, adjust_intercept = TRUE),
                       silent = TRUE), "try-error"))
if (requireNamespace("actuar", quietly = TRUE)) {
  data(hachemeister, package = "actuar")
  r <- as.matrix(hachemeister[, 2:13]); ww <- as.matrix(hachemeister[, 14:25])
  for (adj in c(FALSE, TRUE)) {
    ref <- actuar::cm(~state, hachemeister, regformula = ~time, regdata = data.frame(time = 1:12),
                      ratios = ratio.1:ratio.12, weights = weight.1:weight.12, adj.intercept = adj)
    ours <- hachemeister(r, ww, adjust_intercept = adj)
    tol <- if (adj) 1e-10 else 1e-6
    want <- as.numeric(predict(ref, newdata = data.frame(time = 13)))
    stopifnot(all(abs(predict(ours, data.frame(time = 13))[1, ] / want - 1) < tol))
  }
}

cat("prospicio R credibility tests passed\n")

# Bayesian hierarchical credibility: the credibility formula at actuar's
# structure parameters (validation/reference/credibility_hierarchical_actuar.csv)
# and a short NUTS fit.
# Hachemeister's data (actuar's hachemeister).
hr <- rbind(
  c(1738, 1642, 1794, 2051, 2079, 2234, 2032, 2035, 2115, 2262, 2267, 2517),
  c(1364, 1408, 1597, 1444, 1342, 1675, 1470, 1448, 1464, 1831, 1612, 1471),
  c(1759, 1685, 1479, 1763, 1674, 2103, 1502, 1622, 1828, 2155, 2233, 2059),
  c(1223, 1146, 1010, 1257, 1426, 1532, 1953, 1123, 1343, 1243, 1762, 1306),
  c(1456, 1499, 1609, 1741, 1482, 1572, 1606, 1735, 1607, 1573, 1613, 1690)
)
hw <- rbind(
  c(7861, 9251, 8706, 8575, 7917, 8263, 9456, 8003, 7365, 7832, 7849, 9077),
  c(1622, 1742, 1523, 1515, 1622, 1602, 1964, 1515, 1527, 1748, 1654, 1861),
  c(1147, 1357, 1329, 1204, 998, 1077, 1277, 1218, 896, 1003, 1108, 1121),
  c(407, 396, 348, 341, 315, 328, 352, 331, 287, 384, 321, 342),
  c(2902, 3172, 3046, 3068, 2693, 2910, 3275, 2697, 2663, 3017, 3242, 3425)
)
p <- credibility_premiums(hr, hw, within_variance = 139120025.92528549,
                          between_variances = c(87263.695756774905, 13414.843135533525),
                          levels = list(c(1, 2, 1, 2, 2)))
stopifnot(abs(p$collective_mean - 1742.22) < 0.01)
stopifnot(all(abs(p$premiums - c(2049.733, 1522.032, 1864.280, 1488.504, 1587.097)) < 1e-3))
hc <- hierarchical_credibility(hr, hw, chains = 2, tune = 300, draws = 300, seed = 1)
stopifnot(length(hc$premiums) == 5, all(hc$summary$rhat < 1.1), hc$divergences >= 0)
pd <- credibility_predictive(hc, n_sims = 2000, seed = 3)
stopifnot(identical(pd@dims, "group"))
stopifnot(inherits(try(credibility_premiums(hr, hw, 1, c(1, 1)), silent = TRUE), "try-error"))
