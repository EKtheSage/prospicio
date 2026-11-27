library(prospicio)
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# The workers compensation split plan (#981) against NCCI's "ABCs of
# Experience Rating": the worksheet's summary (Exhibit H) and the split and
# medical-only examples (Exhibits A and B).
exposures <- data.frame(class = "all", payroll = 100, expected_loss_rate = 176190,
                        d_ratio = 56172 / 176190)
claims <- data.frame(amount = c(49985, 17359, 17000, 16225))
x <- split_plan_modification(exposures, claims, weight = 0.14, ballast = 47400,
                             split_point = 18000, per_claim_limit = 250000)
stopifnot(near(x$actual_primary, 68584, 1e-12))
stopifnot(near(x$actual_excess, 31985, 1e-12))
stopifnot(round(x$adjusted_actual) == 223677, round(x$adjusted_expected) == 223590)
stopifnot(round(x$modification, 2) == 1.00)

claims <- data.frame(amount = c(500000, 100000, 5000, 20000),
                     medical_only = c(FALSE, FALSE, FALSE, TRUE))
x <- split_plan_modification(data.frame(class = "8810", payroll = 1e6, expected_loss_rate = 1,
                                        d_ratio = 0.3),
                             claims, weight = 0.2, ballast = 1e4, split_point = 18000,
                             per_claim_limit = 250000)
stopifnot(identical(x$claims$limited, c(250000, 100000, 5000, 20000)))
stopifnot(near(x$claims$primary, c(18000, 18000, 5000, 5400), 1e-12))
stopifnot(near_abs(x$claims$excess, c(232000, 82000, 0, 600), 1e-9))
stopifnot(identical(x$classes$expected, 10000))

# One accident's claims are limited together; NA is an accident of its own.
x <- split_plan_modification(data.frame(class = "8810", payroll = 1e6, expected_loss_rate = 1,
                                        d_ratio = 0.3),
                             data.frame(amount = c(200000, 150000, 400000),
                                        accident = c("a", "a", NA)),
                             weight = 0.2, ballast = 0, split_point = 18000,
                             per_accident_limit = 300000)
stopifnot(near(sum(x$claims$limited[1:2]), 300000, 1e-12))
stopifnot(identical(x$claims$limited[3], 300000))

expect_error_like(split_plan_modification(data.frame(class = "a", payroll = 1,
                                                     expected_loss_rate = 1, d_ratio = 0.3),
                                          data.frame(amount = 1), weight = 1.5, ballast = 0,
                                          split_point = 18000),
                  "weight")
cat("split plan tests passed\n")
