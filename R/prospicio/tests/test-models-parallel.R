library(prospicio)
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# cross_validate() and grid_search() through a process-based map give the
# serial scores exactly, in the splits' and candidates' order (#761).
set.seed(5)
pd <- data.frame(x1 = stats::rnorm(60), x2 = stats::rnorm(60))
pd$y <- 1 + pd$x1 - 0.5 * pd$x2 + stats::rnorm(60)
folds <- k_fold(nrow(pd), 3, seed = 2)
mse <- function(m, test) mean((test$y - predict(m, test))^2)
# A fit that draws random numbers: a random subsample of its training rows.
fit <- function(lam, train) {
  keep <- sort(sample.int(nrow(train), floor(0.8 * nrow(train))))
  elastic_net_fit(y ~ x1 + x2, train[keep, , drop = FALSE], family = "gaussian", lambda = lam)
}

set.seed(11)
state <- .Random.seed
serial <- grid_search(c(0, 0.05, 0.5), pd, folds, fit, mse, seed = 3)
stopifnot(identical(.Random.seed, state))
# A closure made at the top level does not carry the globals it calls to a
# worker process (R sends the global environment by reference), so the
# one-argument fit keeps `fit` in an environment of its own.
cv_fit <- local({
  f <- fit
  function(train) f(0.05, train)
})
serial_cv <- cross_validate(pd, folds, cv_fit, mse, seed = 4)

cl <- parallel::makeCluster(2)
invisible(parallel::clusterEvalQ(cl, {
  Sys.setenv(RAYON_NUM_THREADS = 1)
  library(prospicio)
}))
psock <- function(x, f) parallel::parLapply(cl, x, f)
parallel_grid <- grid_search(c(0, 0.05, 0.5), pd, folds, fit, mse, seed = 3, map = psock)
parallel_cv <- cross_validate(pd, folds, cv_fit, mse, seed = 4, map = psock)
parallel::stopCluster(cl)

stopifnot(identical(serial$score, parallel_grid$score))
stopifnot(identical(attr(serial, "best"), attr(parallel_grid, "best")))
stopifnot(identical(serial_cv, parallel_cv))

# Without map or seed the scores are the serial loop's, as before.
plain <- function(lam, train) elastic_net_fit(y ~ x1 + x2, train, family = "gaussian", lambda = lam)
by_hand <- vapply(c(0, 0.5), function(lam) {
  mean(vapply(folds, function(s) mse(plain(lam, pd[s$train, ]), pd[s$test, ]), double(1)))
}, double(1))
stopifnot(identical(grid_search(c(0, 0.5), pd, folds, plain, mse)$score, by_hand))

# A map that drops tasks is an error.
expect_error_like(grid_search(c(0, 0.5), pd, folds, plain, mse, map = function(x, f) lapply(x[1], f)),
                  "map returned 1 results for 6 tasks")
expect_error_like(cross_validate(pd, folds, function(train) plain(0, train), mse, map = "lapply"),
                  "map must be a function")

cat("parallel tuning tests passed\n")
