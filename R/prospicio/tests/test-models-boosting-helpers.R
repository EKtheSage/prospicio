library(prospicio)
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# The arithmetic R's boosters share with Python's, from the Rust core (#977).

y <- c(5, 1, 3, 2)
w <- rep(1, 4)
stopifnot(prospicio:::rust_result(prospicio:::weighted_quantile_rust(y, w, 0.25)) == 1)
stopifnot(prospicio:::rust_result(prospicio:::weighted_quantile_rust(y, w, 0.26)) == 2)

b <- prospicio:::rust_result(prospicio:::booster_base_score_rust("log", c(1, 2, 6), c(1, 2, 1),
                                                                  c(0, 0, log(2))))
near(b, log(11 / 5), 1e-14)
expect_error_like(prospicio:::rust_result(prospicio:::booster_base_score_rust("log", 0, 1, 0)),
                  "positive")

phi <- prospicio:::rust_result(prospicio:::pearson_dispersion_rust("gamma", NaN, NaN, c(1, 3),
                                                                    c(2, 2), c(1, 2)))
near(phi, (0.25 + 0.5) / 2, 1e-14)

bad <- prospicio:::rust_result(prospicio:::monotone_violations_rust(c(1, 2, 1.5, 3), 1L, 1e-9))
stopifnot(identical(as.integer(bad), 3L))

y <- c(1, 3, 2)
mu <- c(2, 2, 2.5)
wt <- c(1, 2, 3)
test <- data.frame(y = y, w = wt)
model <- structure(list(), class = "fixed_means")
predict.fixed_means <- function(object, newdata, ...) mu
registerS3method("predict", "fixed_means", predict.fixed_means)
score <- prospicio:::deviance_score("poisson", "y", weight_column = "w")
near(score(model, test), family_deviance("poisson", y, mu, weights = wt) / sum(wt), 1e-14)
