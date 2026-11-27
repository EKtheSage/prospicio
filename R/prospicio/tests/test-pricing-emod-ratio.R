library(prospicio)
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# The general liability experience ratio plan (#981), by hand: ISO
# publishes no worked exhibit, so the worksheet's sums are taken one by one.
years <- data.frame(subject_loss_cost = c(200000, 220000, 240000),
                    unreported_share = c(0.10, 0.30, 0.60),
                    detrend_factor = c(0.95, 1, 1))
occurrences <- data.frame(year = c(1, 1, 2), loss = c(90000, 10000, 30000),
                          alae = c(20000, 1000, 0))
x <- experience_ratio_modification(years, occurrences, maximum_single_limit = 100000,
                                   expected_ratio = 0.75, credibility = 0.4)
stopifnot(identical(x$years$reported, c(111000, 30000, 0)))
unreported <- c(200000 * 0.95 * 0.75 * 0.10, 220000 * 0.75 * 0.30, 240000 * 0.75 * 0.60)
stopifnot(near(x$years$expected_unreported, unreported, 1e-12))
aer <- (141000 + sum(unreported)) / 660000
stopifnot(near(x$actual_ratio, aer, 1e-14))
stopifnot(near(x$modification, 1 + 0.4 * (aer - 0.75) / 0.75, 1e-14))
stopifnot(identical(x$subject_loss_cost, 660000))

# Without a detrend column every factor is 1; without credibility the mod is 1.
x <- experience_ratio_modification(data.frame(subject_loss_cost = 1e5, unreported_share = 0.4),
                                   data.frame(year = 1, loss = 5e5, alae = 0),
                                   maximum_single_limit = Inf, expected_ratio = 0.5,
                                   credibility = 0)
stopifnot(identical(x$modification, 1))

expect_error_like(
  experience_ratio_modification(data.frame(subject_loss_cost = 1e5, unreported_share = 0.4),
                                data.frame(year = 2, loss = 1, alae = 0),
                                maximum_single_limit = 1e5, expected_ratio = 0.5,
                                credibility = 0.5),
  "counted from 1"
)
