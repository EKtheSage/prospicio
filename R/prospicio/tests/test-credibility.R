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

cat("prospicio R credibility tests passed\n")
