library(actuarialrs)

near <- function(a, b, rel) {
  if (!all(abs(a - b) <= rel * abs(b))) {
    stop(sprintf("got %s, want %s", paste(format(a, digits = 17), collapse = " "),
                 paste(format(b, digits = 17), collapse = " ")), call. = FALSE)
  }
}

ln <- lognormal_from_mean_cv(1000, 1)
for (q in list(NULL, function(p) quantile(ln, p))) {
  d <- custom_distribution(function(x) cdf(ln, x), q, name = "lognormal")
  near(mean(d), mean(ln), 1e-7)
  near(lev(d, c(100, 1000, 5000)), lev(ln, c(100, 1000, 5000)), 1e-9)
  near(layer(d, 2000, 1000), layer(ln, 2000, 1000), 1e-8)
  near(quantile(d, 0.99), quantile(ln, 0.99), 1e-12)
  stopifnot(d@has_quantile == !is.null(q), d@name == "lognormal")
}

# It goes where a severity goes; R is called from the main thread only.
d <- custom_distribution(function(x) cdf(ln, x), function(p) quantile(ln, p))
a <- simulate_events(poisson_count(3), d, 2000, 5)
b <- simulate_events(poisson_count(3), ln, 2000, 5)
near(mean(total(a)), mean(total(b)), 1e-12)
m <- mixture_distribution(c(0.5, 0.5), list(d, ln))
near(mean(m), mean(ln), 1e-6)

# Errors.
stopifnot(inherits(try(custom_distribution(function(x) stop("boom")), silent = TRUE), "try-error"))
msg <- tryCatch(custom_distribution(function(x) 0.5 * (1 - exp(-x))), error = conditionMessage)
stopifnot(grepl("does not reach", msg))
stopifnot(inherits(try(custom_distribution(3), silent = TRUE), "try-error"))
late <- custom_distribution(function(x) if (x == 12345) stop("late") else 1 - exp(-x))
stopifnot(late@last_error == "", is.nan(cdf(late, 12345)), grepl("late", late@last_error))
