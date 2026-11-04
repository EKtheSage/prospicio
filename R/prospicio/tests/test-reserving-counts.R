suppressMessages(library(prospicio))

# Methods on claim counts (docs/design/reserving-v02.md, decisions 33 and 34):
# frequency-severity, disposal rates, case outstanding development and the
# Berquist-Sherman adjustments. The Rust core is checked against Friedland's
# exhibits and chainladder-python in validation/tests/reserving_counts.rs;
# these tests reproduce printed totals on XYZ Insurer through the bindings,
# the same numbers as python/tests/test_reserving_counts.py, and check the
# binding's arguments and errors.

near <- function(a, b, tol = 1e-9) {
  a <- unname(a)
  b <- unname(b)
  if (length(a) != length(b) || !all(abs(a - b) <= tol)) {
    stop(sprintf("got %s, want %s (tol %g)", paste(format(a, digits = 17), collapse = " "),
                 paste(format(b, digits = 17), collapse = " "), tol), call. = FALSE)
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

find_data <- function() {
  dir <- getwd()
  for (i in 1:6) {
    path <- file.path(dir, "validation", "data", "friedland_xyz.csv")
    if (file.exists(path)) return(path)
    dir <- dirname(dir)
  }
  stop("validation/data/friedland_xyz.csv not found")
}

columns <- c("reported", "paid", "reported_counts", "closed_counts", "premium")
all_rows <- read.csv(find_data(), comment.char = "#")

xyz <- function(first = 1998, last = 2008, max_age = 132) {
  rows <- all_rows[all_rows$origin >= first & all_rows$origin <= last &
                     all_rows$development <= max_age, ]
  triangle(rows, origin = "origin", development = "development", columns = columns)
}

small <- triangle(data.frame(origin = c(2020, 2020, 2021), age = c(12, 24, 12),
                             reported = c(100, 150, 120), paid = c(50, 100, 60),
                             reported_counts = c(10, 12, 11), closed_counts = c(5, 10, 6)),
                  origin = "origin", development = "age",
                  columns = c("reported", "paid", "reported_counts", "closed_counts"))

# Frequency-severity equals the chain ladder when the patterns agree.
base <- c(10, 16, 20)
sev <- c(5, 6, 7)
d <- do.call(rbind, lapply(1:3, function(k) {
  n <- 4 - k
  data.frame(origin = 2018 + k, age = 12 * seq_len(n), counts = base[seq_len(n)] * k,
             amounts = base[seq_len(n)] * sev[seq_len(n)] * k)
}))
tri <- triangle(d, origin = "origin", development = "age", columns = c("amounts", "counts"))
fs <- frequency_severity(tri, "amounts", "counts")
cl <- chain_ladder(tri, "amounts")
near(fs@ultimate, cl@ultimate)
near(fs@total_reserve, sum(fs@reserve))

# A trended frequency and severity.
fit <- frequency_severity(small, "reported", "reported_counts",
                          ultimate_counts = trended(0.5, 2021, 0),
                          severity = trended(20, 2021, 0.1), exposure = "paid")
near(fit@ultimate, c(50 * 20 / 1.1, 30 * 20))
stopifnot(identical(fit@origins, c("2020", "2021")))

# Friedland's XYZ Insurer, chapter 12: 513,944 and the 233 left out of 2000.
fit <- case_outstanding(xyz(), "paid", "reported", 1.10, paid_on_case = 2, case_on_case = 2)
near(fit@total_ultimate, 513944 + 0.501 * 465, 8)
near(fit@ibnr, fit@ultimate - fit@latest_incurred)
near(fit@reserve, fit@ultimate - fit@latest_paid)

# Chapter 11, the disposal rate method.
reported <- c(1455, 1554, 1629, 2258, 2390, 1657, 1264, 1036)
cdf <- c(1.000, 1.000, 1.001, 1.002, 1.005, 1.013, 1.035, 1.131)
rates <- stats::setNames(c(0.244, 0.572, 0.704, 0.814, 0.910, 0.952, 0.982, 0.994), 12 * 1:8)
fit <- disposal_method(xyz(2001, 2008, 96), "closed_counts", "paid", reported * cdf,
                       disposal_rates = rates, severities = 2, tail_severity_from = 84,
                       trend = 0.05, base = 2008, factors = c(rep(0.670, 5), 0.750, 1, 1))
near(fit@total_ultimate, 385676, 6)
near(fit@ages, 12 * 1:8)
stopifnot(is.na(fit@disposal_rates[8, 2]), length(fit@selected_severities) == 9)

# Chapter 13, the adjustments keep the latest diagonal.
t <- xyz()
counts <- development_method("reported_counts", n_periods = 2)
latest <- function(x, column) latest_diagonal(x, column)
case <- case_adequacy_adjust(t, "reported", "paid", 0.05, reported = "reported_counts",
                             closed = "closed_counts", max_age = 84)
stopifnot(identical(case@columns, t@columns))
settled <- settlement_rate_adjust(t, "paid", "closed_counts", counts)
both <- berquist_sherman_adjust(t, "reported", "paid", "reported_counts", "closed_counts", 0.05,
                                counts, max_age = 84)
near(chain_ladder(case, "reported")@latest, chain_ladder(t, "reported")@latest)
near(chain_ladder(settled, "paid")@latest, chain_ladder(t, "paid")@latest)
near(chain_ladder(both, "reported")@latest, chain_ladder(t, "reported")@latest)
extra <- average_case(t, "reported", "paid", 0.05, reported = "reported_counts",
                      closed = "closed_counts")
stopifnot(all(c("average_case", "adjusted_average_case") %in% extra@columns))
invisible(settlement_rate_adjust(t, "paid", "closed_counts",
                                 list(counts, development_method("closed_counts", n_periods = 2))))

# A trend fitted on the average case is the same rate given by hand.
averages <- average_case(t, "reported", "paid", 0, reported = "reported_counts",
                         closed = "closed_counts")
rate <- pooled_trend(averages, "average_case")$rate
stopifnot(rate > 0)
fitted <- case_adequacy_adjust(t, "reported", "paid", "average_case", reported = "reported_counts",
                               closed = "closed_counts")
given <- case_adequacy_adjust(t, "reported", "paid", rate, reported = "reported_counts",
                              closed = "closed_counts")
near(chain_ladder(fitted, "reported")@ultimate, chain_ladder(given, "reported")@ultimate)
expect_error_like(case_adequacy_adjust(t, "reported", "paid", "average",
                                       reported = "reported_counts", closed = "closed_counts"),
                  "average_case")

# Trends.
near(exponential_trend(0:2, c(100, 110, 121))$rate, 0.1)
stopifnot(is.null(exponential_trend(0, 100)))
avg <- ratio_column(t, "paid", "closed_counts", "average_paid")
by_age <- trend_by_age(avg, "average_paid")
near(round(by_age$rate[1] * 100, 1), 8.1)
stopifnot(is.finite(pooled_trend(avg, "average_paid")$rate))
near(case_development_factor(1.2, 2), 1.5)

# Arguments.
expect_error_like(disposal_method(small, "closed_counts", "paid", c(10, 12),
                                  disposal_rates = "latest"), "disposal_rates must be")
expect_error_like(disposal_method(small, "closed_counts", "paid", c(10, 12), base = 1999),
                  "not an origin")
expect_error_like(frequency_severity(small, "paid", "closed_counts",
                                     ultimate_counts = trended(1, 2021, 0)), "needs exposure")
expect_error_like(case_adequacy_adjust(small, "reported", "paid", 0.1, open_counts = "x",
                                       reported = "y"), "not both")
expect_error_like(frequency_severity(small, "paid", "closed_counts", severity = 1),
                  "severity must be")
