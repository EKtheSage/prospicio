library(actuarialrs)

close <- function(got, want, rel) {
  stopifnot(all(abs(got - want) <= rel * abs(want)))
}

d <- lognormal(7, 0.5)

# Parity with base R's lognormal functions.
close(mean(d), exp(7 + 0.5^2 / 2), 1e-13)
close(variance(d), (exp(0.5^2) - 1) * exp(2 * 7 + 0.5^2), 1e-12)
p <- c(0.01, 0.5, 0.995)
close(quantile(d, p), qlnorm(p, 7, 0.5), 1e-12)
close(cdf(d, c(500, 1500)), plnorm(c(500, 1500), 7, 0.5), 1e-12)

m <- lognormal_from_mean_cv(1000, 0.5)
close(mean(m), 1000, 1e-12)
close(sqrt(variance(m)), 500, 1e-12)

# Errors from Rust surface as ordinary R errors naming the R call.
e <- tryCatch(lognormal(0, -1), error = identity)
stopifnot(grepl("sdlog", conditionMessage(e)), identical(conditionCall(e)[[1]], quote(lognormal)))
e <- tryCatch(quantile(d, 2), error = identity)
stopifnot(grepl("probability 2", conditionMessage(e)))
e <- tryCatch(draws(d, 3, seed = -1), error = identity)
stopifnot(grepl("seed", conditionMessage(e)))

# Same kernel as Rust and Python: these draws are pinned in all three.
x <- draws(lognormal(0, 1), 3, seed = 42, stream = 3)
stopifnot(identical(x, c(1.0007760893701914, 1.6293872534754683, 1.0763869265482304)))

cat("actuarialrs R tests passed\n")
