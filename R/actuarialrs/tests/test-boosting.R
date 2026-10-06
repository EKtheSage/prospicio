library(actuarialrs)

near <- function(a, b, rel = 1e-10) {
  if (!all(abs(a - b) <= rel * pmax(abs(b), 1e-300))) {
    stop(sprintf("got %s, want %s", paste(format(a, digits = 17), collapse = " "),
                 paste(format(b, digits = 17), collapse = " ")), call. = FALSE)
  }
}

# simulate_from_means() needs no engine.
pd <- simulate_from_means("gamma", c(100, 200), n_sims = 20000, seed = 2, dispersion = 0.25)
stopifnot(abs(mean(pd) / 300 - 1) < 0.02)
two <- simulate_from_means("gaussian", cbind(2, 6), n_sims = 40000, seed = 5)
stopifnot(abs(mean(two) - 4) < 0.05, abs(variance(two) - 5) < 0.15)
stopifnot(inherits(try(simulate_from_means("poisson", -1, 10, 1), silent = TRUE), "try-error"))
stopifnot(inherits(try(booster_fit(y ~ x, data.frame(x = 1, y = 1), family = "binomial"),
                       silent = TRUE), "try-error"))
stopifnot(inherits(try(booster_fit(y ~ x, data.frame(x = 1, y = 1), family = "tweedie"),
                       silent = TRUE), "try-error"))

# Claim counts with exposure: rate 0.1 for x < 0.5, 0.3 above.
set.seed(11)
n <- 2000
d <- data.frame(x = runif(n), exposure = 0.2 + 0.8 * runif(n))
d$claims <- rpois(n, ifelse(d$x < 0.5, 0.1, 0.3) * d$exposure)
f <- claims ~ x + offset(log(exposure))

for (engine in c("lightgbm", "xgboost")) {
  if (!requireNamespace(engine, quietly = TRUE)) next
  params <- if (engine == "lightgbm") list(num_leaves = 3) else list(max_depth = 2)
  before <- .Random.seed
  m <- booster_fit(f, d, engine = engine, n_rounds = 150, learning_rate = 0.1, params = params)
  # R's generator is left as it was.
  stopifnot(identical(before, .Random.seed))
  mu <- predict(m, d)
  rate <- mu / d$exposure
  stopifnot(abs(mean(rate[d$x < 0.45]) - 0.1) < 0.03, abs(mean(rate[d$x > 0.55]) - 0.3) < 0.06)
  # Doubling the exposure doubles every mean; xgboost's margins are single
  # precision.
  d2 <- transform(d, exposure = 2 * exposure)
  near(predict(m, d2), 2 * mu, if (engine == "lightgbm") 1e-9 else 1e-5)
  stopifnot(m@engine == engine, m@dispersion == 1)
}

if (requireNamespace("lightgbm", quietly = TRUE)) {
  # The adapter is the engine: the same call by hand gives the same means.
  small <- d[1:500, ]
  m <- booster_fit(f, small, n_rounds = 30, params = list(num_leaves = 4))
  off <- log(small$exposure)
  base <- log(sum(small$claims) / sum(exp(off)))
  near(m@base, base, 1e-12)
  x <- matrix(small$x, dimnames = list(NULL, "x"))
  data <- lightgbm::lgb.Dataset(x, label = small$claims, weight = rep(1, 500),
                                init_score = off + base)
  raw <- lightgbm::lgb.train(list(objective = "poisson", learning_rate = 0.05, seed = 0,
                                  deterministic = TRUE, verbosity = -1, num_leaves = 4),
                             data, nrounds = 30, verbose = -1)
  near(predict(m, small), exp(predict(raw, x, type = "raw") + off + base), 1e-12)

  # Gamma severities: shape 2, so the dispersion is near 1 / 2.
  set.seed(3)
  s <- data.frame(x = runif(800))
  s$y <- rgamma(800, shape = 2, scale = (100 + 100 * s$x) / 2)
  g <- booster_fit(y ~ x, s, family = "gamma", n_rounds = 100, params = list(num_leaves = 4))
  stopifnot(abs(sum(predict(g, s)) / sum(s$y) - 1) < 0.05, abs(g@dispersion - 0.5) < 0.1)
  tw <- booster_fit(y ~ x, s, family = "tweedie", power = 1.5, n_rounds = 50)
  stopifnot(tw@dispersion > 0)

  # Bootstrap refits add parameter uncertainty to the predictive total.
  mid <- d[1:1000, ]
  plain <- booster_fit(f, mid, n_rounds = 50)
  boot <- booster_fit(f, mid, n_rounds = 50, n_boot = 8, seed = 4)
  p <- predict_distribution(plain, mid, n_sims = 4000, seed = 1)
  b <- predict_distribution(boot, mid, n_sims = 4000, seed = 1)
  stopifnot(abs(mean(p) / sum(predict(plain, mid)) - 1) < 0.02, variance(b) > variance(p))

  # compare_models() takes a booster like any other fit.
  dev <- function(m, test) family_deviance("poisson", test$claims, predict(m, test)) / nrow(test)
  cmp <- compare_models(list(glm = function(tr) glm_fit(f, tr), gbm = function(tr) booster_fit(f, tr, n_rounds = 50)),
                        d[1:600, ], k_fold(600, 3, seed = 1), list(deviance = dev))
  stopifnot(setequal(cmp$model, c("glm", "gbm")))
}

# Quantile objective: a fan of quantiles whose spread grows with x.
stopifnot(abs(pinball_loss(c(1, 0), c(0, 1), 0.9) - 0.5) < 1e-15)
stopifnot(inherits(try(booster_fit(y ~ x, data.frame(x = 1, y = 1), family = "quantile"),
                       silent = TRUE), "try-error"))
for (engine in c("lightgbm", "xgboost")) {
  if (!requireNamespace(engine, quietly = TRUE)) next
  set.seed(7)
  qd <- data.frame(x = runif(3000))
  qd$y <- 100 + 200 * qd$x * rnorm(3000)
  params <- if (engine == "lightgbm") list(num_leaves = 4) else list(max_depth = 2)
  fits <- lapply(c(0.9, 0.1, 0.5), function(a) {
    booster_fit(y ~ x, qd, family = "quantile", alpha = a, engine = engine, n_rounds = 200,
                learning_rate = 0.1, params = params)
  })
  q <- predict_quantiles(fits, qd)
  stopifnot(identical(colnames(q), c("0.1", "0.5", "0.9")), all(q[, 1] <= q[, 2]), all(q[, 2] <= q[, 3]))
  inside <- mean(qd$y >= q[, 1] & qd$y <= q[, 3])
  stopifnot(abs(inside - 0.8) < 0.04)
  flat <- rep(fits[[1]]@base, nrow(qd))
  stopifnot(pinball_loss(qd$y, predict(fits[[1]], qd), 0.9) < 0.9 * pinball_loss(qd$y, flat, 0.9))
  stopifnot(inherits(try(predict_distribution(fits[[1]], qd, 10, 1), silent = TRUE), "try-error"))
  off <- try(booster_fit(y ~ x, qd, family = "quantile", alpha = 0.5, offset = rep(1, 3000),
                         engine = engine, n_rounds = 5), silent = TRUE)
  stopifnot(inherits(off, "try-error"))
}
