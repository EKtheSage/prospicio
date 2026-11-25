library(prospicio)
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# Werner and Modlin, chapter 4: six annual policies a quarter apart.
effective <- as.Date(c("2010-10-01", "2011-01-01", "2011-04-01", "2011-07-01", "2011-10-01", "2012-01-01"))
expiration <- as.Date(c("2011-10-01", "2012-01-01", "2012-04-01", "2012-07-01", "2012-10-01", "2013-01-01"))
years <- calendar_windows(2010, 2012)
stopifnot(inherits(years, "Date"), identical(format(years), c("2010-01-01", "2011-01-01", "2012-01-01", "2013-01-01")))
stopifnot(length(calendar_windows(2024, 2024, by = "quarter")) == 5, length(calendar_windows(2024, 2024, "month")) == 13)

# Tables 4.7 and 4.11 on the half-month grid, exactly.
t <- earned_exposure(effective, expiration, rep(1, 6), years, method = "half_month")
stopifnot(identical(names(t), c("label", "from", "to", "written", "earned", "unearned", "in_force")))
stopifnot(inherits(t$from, "Date"), identical(t$written, c(1, 4, 1)), near_abs(t$earned, c(0.25, 3.25, 2.5), 1e-12))
stopifnot(identical(t$in_force, c(1, 4, 1)))
# By exact days: within two days in the year, and dates as strings.
td <- earned_exposure(format(effective), format(expiration), rep(1, 6), format(years))
stopifnot(near_abs(td$earned[1], 92 / 365, 1e-15), near_abs(td$earned, c(0.25, 3.25, 2.5), 6e-3))
# Tables 4.9 and 4.13: policy years, as of 12/31/12.
tp <- earned_exposure(effective, expiration, rep(1, 6), years, kind = "policy", valuation = "2013-01-01")
stopifnot(identical(tp$written, c(1, 4, 1)), near_abs(tp$earned, c(1, 4, 1), 1e-12))
# Table 4.15: in force as of three dates.
stopifnot(identical(in_force_amount(effective, expiration, rep(1, 6), c("2011-01-01", "2011-06-15", "2012-01-01")), c(2, 3, 4)))

# Policy D cancelled on March 31, 2012: 400 written in 2011 and -100 in 2012.
cancel <- c(NA, NA, NA, "2012-04-01", NA, NA)
tc <- earned_exposure(effective[4], expiration[4], 400, years, method = "half_month", cancellation = "2012-04-01")
stopifnot(near_abs(tc$written, c(0, 400, -100), 1e-12), near_abs(tc$earned, c(0, 200, 100), 1e-12))
tc <- earned_exposure(effective, expiration, rep(1, 6), years, method = "half_month", cancellation = cancel)
stopifnot(near_abs(tc$written, c(1, 4, 0.75), 1e-12))
# By policy year it is all in 2011.
stopifnot(near_abs(earned_exposure(effective, expiration, rep(1, 6), years, kind = "policy", method = "half_month",
                                   cancellation = cancel)$written, c(1, 3.75, 1), 1e-12))

# Table 3.1's Policy C: 1,000, then 1,200 full term from January 1, 2011.
ch <- data.frame(policy = 1, date = "2011-01-01", amount = 1200)
tm <- earned_exposure("2010-07-01", "2011-07-01", 1000, calendar_windows(2010, 2011), method = "half_month", changes = ch)
stopifnot(near_abs(tm$written, c(1000, 100), 1e-12), near_abs(tm$earned, c(500, 600), 1e-12))
stopifnot(identical(in_force_amount("2010-07-01", "2011-07-01", 1000, c("2010-12-31", "2011-01-01"), changes = ch), c(1000, 1200)))

# Twenty-fourths, Table 4.24: 240 written in January and in December 2010.
tw <- earned_exposure(c("2010-01-09", "2010-12-28"), c("2011-01-09", "2011-12-28"), c(240, 240),
                      calendar_windows(2010, 2011), method = "twenty_fourths", label = c("jan", "dec"))
stopifnot(identical(tw$label, c("dec", "dec", "jan", "jan")), near_abs(tw$earned, c(10, 230, 230, 10), 1e-10))

# Refusals.
expect_error_like(earned_exposure("2023-02-29", "2024-01-01", 1, years), "YYYY-MM-DD")
expect_error_like(earned_exposure(effective, expiration, rep(1, 5), years), "amount has 5 entries")
expect_error_like(earned_exposure(effective, expiration, rep(1, 6), years, method = "weekly"), "method")
expect_error_like(earned_exposure(effective, expiration, rep(1, 6), years, kind = "fiscal"), "kind")
expect_error_like(earned_exposure(effective, expiration, rep(1, 6), years,
                                  changes = data.frame(policy = 7, date = "2011-06-01", amount = 2)), "policy 7")
expect_error_like(calendar_windows(2024, 2024, by = "week"), "window length")
