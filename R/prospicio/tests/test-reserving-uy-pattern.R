suppressMessages(library(prospicio))

# An accident-period development pattern on an underwriting basis
# (docs/design/reserving-v02.md, decision 20, issue #633):
# underwriting_basis() and ldf_underwriting on the methods that take ldf.
# The Rust core is checked in validation/tests/reserving_uy_pattern.rs;
# these tests reproduce its hand-computed parallelogram through the
# binding, as the Python tests do.

near <- function(a, b, rel = 1e-12) {
  a <- unname(a)
  b <- unname(b)
  if (length(a) != length(b) || !all(a == b | abs(a - b) <= rel * abs(b))) {
    stop(sprintf("got %s, want %s (rel %g)", paste(format(a, digits = 17), collapse = " "),
                 paste(format(b, digits = 17), collapse = " "), rel), call. = FALSE)
  }
}

expect_error_like <- function(expr, pattern) {
  msg <- tryCatch({
    expr
    NULL
  }, error = conditionMessage)
  if (is.null(msg)) stop(sprintf("expected an error mentioning %s", shQuote(pattern)), call. = FALSE)
  if (!grepl(pattern, msg, fixed = TRUE)) {
    stop(sprintf("error %s does not mention %s", shQuote(msg), shQuote(pattern)), call. = FALSE)
  }
}

# The parallelogram's share developed of a linear pattern t/60, by hand.
parallelogram <- function(a) {
  vapply(a, function(a) {
    if (a <= 12) a * (a + 1) * (2 * a + 1) / (1728 * 60)
    else if (a >= 23 && a <= 60) (a - 11.5) / 60
    else if (a >= 83) 1
    else stop("no closed form")
  }, numeric(1))
}

# 1. The basis.
b <- underwriting_basis()
stopifnot(inherits(b, "underwriting_basis"), b$policy_months == 12, length(b$exposure) == 24)
t <- 0:23
near(b$exposure, ifelse(t <= 11, 2 * t + 1, 47 - 2 * t) / 288, 1e-14)
expect_error_like(underwriting_basis(policy_months = 0), "at least 1")
expect_error_like(underwriting_basis(c(1, -1)), "not negative")
expect_error_like(underwriting_basis(c(0, 0)), "positive sum")

# 2. An accident-month pattern on underwriting years.
linear <- stats::setNames((1:60) / 60, 1:60)
book <- latest_valuation(data.frame(year = 2017:2019, age = c(83, 36, 12),
                                    paid = c(100, 100, 100)), "year", "paid", age = "age")
fit <- chain_ladder(book, ldf = linear, ldf_kind = "percent_developed", ldf_origin = "M",
                    ldf_underwriting = underwriting_basis())
near(fit@ultimate, 100 / parallelogram(c(83, 36, 12)), 1e-13)

# 3. An accident-year pattern at every month, with no ldf_origin.
year <- function(a) sum(pmin(pmax(a - 0:11, 0), 60)) / (12 * 60)
ay <- stats::setNames(vapply(1:71, year, numeric(1)), 1:71)
fit <- chain_ladder(book, ldf = ay, ldf_kind = "percent_developed",
                    ldf_underwriting = underwriting_basis())
near(fit@ultimate, 100 / parallelogram(c(83, 36, 12)), 1e-13)

# 4. Errors.
one <- latest_valuation(data.frame(year = 2019, age = 12, paid = 100), "year", "paid",
                        age = "age")
expect_error_like(chain_ladder(one, ldf = c("12" = 0.5, "24" = 1), ldf_kind = "percent_developed",
                               ldf_underwriting = underwriting_basis(rep(c(1, 2), 6), 6)),
                  "give an accident-month pattern")
small <- triangle(data.frame(year = c(2023, 2023, 2024), age = c(12, 24, 12),
                             paid = c(13, 37, 13)), "year", "age", "paid")
expect_error_like(chain_ladder(small, ldf_underwriting = underwriting_basis()),
                  "ldf_underwriting is given without ldf")
expect_error_like(mack(small, ldf_underwriting = underwriting_basis()),
                  "ldf_underwriting is given without ldf")
expect_error_like(chain_ladder(small, ldf = c("12" = 1.5), ldf_kind = "ldf",
                               ldf_underwriting = list(writing = 1, policy_months = 12)),
                  "underwriting_basis()")

# 5. The writing fits in the triangle's origin period: on underwriting
# quarters the default's year of writing is an error, three months convert.
quarters <- triangle(data.frame(origin = as.Date(c("2023-01-01", "2023-01-01", "2023-04-01")),
                                age = c(3, 6, 3), paid = c(10, 30, 12)),
                     "origin", "age", "paid", origin_grain = "Q", development_grain = "Q")
expect_error_like(chain_ladder(quarters, ldf = linear, ldf_kind = "percent_developed",
                               ldf_origin = "M", ldf_underwriting = underwriting_basis()),
                  "longer than the triangle's origin period")
qb <- underwriting_basis(c(1, 1, 1))
stopifnot(length(qb$exposure) == 15)
fit <- chain_ladder(quarters, ldf = linear, ldf_kind = "percent_developed", ldf_origin = "M",
                    ldf_underwriting = qb)
share <- function(a) sum(qb$exposure * pmin(pmax(a - (seq_along(qb$exposure) - 1), 0), 60) / 60)
near(fit@selected_ldf[1], share(6) / share(3), 1e-12)
