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

# The indication: Werner and Modlin, ch. 8 and Appendices A and B.
near(indicated_change(0.65, variable = 0.25, profit = 0.10, fixed_ratio = 0.065), 0.10, 1e-14)
near(indicated_rate(300, 0.25, 0.10, 25), 500, 1e-14)
near(permissible_loss_ratio(0.25, 0.10, 0.065), 0.585, 1e-14)
stopifnot(inherits(try(indicated_rate(1, variable = 0.9, profit = 0.2), silent = TRUE), "try-error"))
a <- loss_ratio_experience(
  c(1122372, 1154508, 1280545, 1369976, 1397750), c(856495, 867184, 835120, 821509, 797866),
  on_level_factor = c(1.2161, 1.2176, 1.1311, 1.0892, 1.0991),
  premium_trend = c(1.1342, 1.1116, 1.0879, 1.0663, 1.0452),
  development = c(1.0000, 0.9799, 1.0003, 1.0282, 1.0966),
  loss_trend = c(0.9912, 0.9962, 1.0012, 1.0062, 1.0113), lae_factor = 1.143)
stopifnot(identical(round(a$by_year$projected_losses), c(970359, 967578, 955974, 971450, 1011357)))
stopifnot(round(a$loss_ratio, 3) == 0.619)
stopifnot(round(indicated_change(0.619, 0.17, 0.05, fixed_ratio = 0.113), 3) == -0.062)
b <- pure_premium_experience(
  c(12760, 12766, 12805, 12834, 13411), c(5161624, 4820968, 4112172, 5052052, 6559224),
  development = c(1.0000, 1.0012, 1.0055, 1.0186, 1.0555),
  loss_trend = c(1.1939, 1.1705, 1.1476, 1.1251, 1.1030), lae_factor = 1.012)
stopifnot(round(b$pure_premium, 2) == 469.88)
stopifnot(round(indicated_rate(493.63 + 29.11 + 74.57 + 15.68, 0.138, 0.05,
                               fixed_per_exposure = 77.83), 2) == 850.76)
near(premium_based_expenses(c(0.06, 0.10), c(0.5, 0)), c(fixed = 0.03, variable = 0.13), 1e-15)
near(credibility_weighted(200, 0.3, 300), 270, 1e-14)
near(200 * (1 + trended_present_rates(0.10, 0.06, trend_factor(0.05, 2))), 228.8207547, 1e-7)
