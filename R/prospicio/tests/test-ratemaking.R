library(prospicio)

near <- function(a, b, tol) stopifnot(all(abs(a - b) <= tol * pmax(1, abs(b))))

# The half-month grid and trend lengths (Werner and Modlin, ch. 5 and 6).
near(year_fraction(2013, 6, 30) - year_fraction(2011, 1, 1), 2.5, 1e-15)
near(year_fraction(2011, c(11, 12), c(15, 31)), c(2011.875, 2012), 1e-15)
stopifnot(inherits(try(year_fraction(2011, 2, 29), silent = TRUE), "try-error"))
near(average_written_date(2013, 2014, 1, "policy") - average_written_date(2011, 2012, 1), 2.5, 1e-15)
near(average_earned_date(2015, 2016, 0.5, "policy") - average_earned_date(2011, 2012, 0.5), 4.25, 1e-15)

# Table 5.19, calendar year 2011: annual and six-month policies.
eff <- c(2010.5, 2011, 2012.25)
chg <- c(0.05, 0.10, -0.01)
r <- rate_level(eff, chg, start = 2011, end = 2012, term = 1)
near(r$portions$share, c(0.125, 0.375, 0.5), 1e-14)
near(rate_level(eff, chg, 2011, 2012, 1)$on_level_factor, r$on_level_factor, 0)
near(r$average_level, 1.09625, 1e-14)
near(r$on_level_factor, 1.14345 / 1.09625, 1e-14)
near(rate_level(eff, chg, start = 2011, end = 2012, term = 0.5)$portions$share, c(0.25, 0.75), 1e-14)
# Policy year 2012 with the printed levels: 0.9975.
py <- rate_level(eff, level = c(1.05, 1.155, 1.1435), start = 2012, end = 2013, term = 1,
                 period = "policy")
stopifnot(round(py$on_level_factor, 4) == 0.9975)
# Table 5.23: a law change on 7/1/11.
law <- rate_level(c(2010.5, 2011, 2011.5, 2012.25), c(0.05, 0.10, -0.05, -0.01),
                  start = 2011, end = 2012, term = 1,
                  applies = c("written", "written", "in_force", "written"))
near(law$portions$share, c(0.125, 0.25, 0.125, 0.125, 0.375), 1e-14)
near(law$on_level_factor, 1.017084, 1e-6)
stopifnot(inherits(try(rate_level(eff, start = 2011, end = 2012, term = 1), silent = TRUE),
                   "try-error"))

# Trend fits and factors.
t <- (0:7) / 4
f <- trend_fit(t, 100 * 1.03^t)
near(f[["annual_change"]], 0.03, 1e-12)
near(trend_fit(t, 5 + 2 * t, "linear")[["annual_change"]], 2, 1e-12)
near(trend_factor(0.02, 2.5), 1.050752, 1e-6)
