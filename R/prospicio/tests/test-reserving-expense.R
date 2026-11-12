suppressMessages(library(prospicio))
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# Recoveries, ALAE and ULAE (docs/design/reserving-v02.md, decision 37). The
# Rust core is checked exhibit by exhibit in validation/tests/reserving_expense.rs;
# these tests reproduce Friedland's headline numbers through the R bindings
# (Estimating Unpaid Claims Using Basic Techniques, chapters 14, 16 and 17), the
# same numbers as python/tests/test_reserving_expense.py, and check the
# binding's arguments and errors.



# The data sit in the repository, next to the Rust validation.
data_dir <- function() {
  args <- commandArgs(FALSE)
  file <- sub("^--file=", "", args[grepl("^--file=", args)])
  starts <- c(if (length(file)) dirname(normalizePath(file[1])), getwd())
  for (start in starts) {
    dir <- start
    repeat {
      candidate <- file.path(dir, "validation", "data")
      if (dir.exists(candidate)) return(candidate)
      parent <- dirname(dir)
      if (parent == dir) break
      dir <- parent
    }
  }
  stop("validation/data not found", call. = FALSE)
}

book <- function(name) {
  rows <- utils::read.csv(file.path(data_dir(), name))
  triangle(rows, "origin", "age", setdiff(names(rows), c("origin", "age")))
}

ages <- 12 * 1:11
pattern <- function(cdf) stats::setNames(cdf, ages)

selected_claims <- function(tri, reported_cdf, paid_cdf) {
  reported <- chain_ladder(tri, "reported", ldf = pattern(reported_cdf), ldf_kind = "cdf")
  paid <- chain_ladder(tri, "paid", ldf = pattern(paid_cdf), ldf_kind = "cdf")
  unname((reported@ultimate + paid@ultimate) / 2)
}

# Chapter 14, Exhibit I.
physical <- book("friedland_auto_physical_damage.csv")
ss_claims <- selected_claims(physical, c(1.115, 1.001, rep(1, 9)),
                             c(1.279, 1.005, 1.001, rep(1, 8)))
ss_cdf <- c(1.499, 1.009, rep(1, 9))
ss <- ratio_development(physical, "received_ss", "paid", ss_claims, ldf = pattern(ss_cdf),
                        ldf_kind = "cdf", selected = c("2008" = 0.345))
stopifnot(ss@selected_ratio[["2008"]] == 0.345, is.na(ss@ratios["2008", "24"]))
near_abs(ss@ultimate[["2007"]], 6036, 1)
near_abs(ss@total_ultimate, 44934, 2)
near_abs(ss@total_unpaid, 3054, 2)
near_abs(tail(ss@unpaid, 2), c(79, 2966), 1)

medial <- ratio_development(physical, "received_ss", "paid", ss_claims, n_periods = 5,
                            drop_high = 1, drop_low = 1)
near_abs(medial@averages[1:2], c(1.486, 1.009), 0.0005)
near_abs(medial@cdf, ss_cdf, 0.0015)

# Chapter 16, Exhibit I.
property <- book("friedland_auto_property_damage.csv")
alae_claims <- selected_claims(property,
                               c(1.101, 0.990, 0.989, 0.991, 0.993, 0.998, 0.999, 0.999, 1, 1, 1),
                               c(1.584, 1.029, 1.007, 1.004, 1.002, 1.001, 1.001, 1.001, 1, 1, 1))
alae_cdf <- c(1.332, 1.201, 1.140, 1.086, 1.050, 1.021, 1.007, 1.003, 1.002, 1, 1)
alae <- ratio_development(property, "paid_alae", "paid", alae_claims, ldf = pattern(alae_cdf),
                          ldf_kind = "cdf", selected = c("2007" = 0.0077, "2008" = 0.0077))
near_abs(alae@total_ultimate, 12485, 2)
near_abs(alae@unpaid[["2007"]], -94, 1)
near_abs(alae@ultimate[["2006"]], 1302, 1)
near_abs(alae@total_unpaid, 800, 2)

additive <- ratio_development(property, "paid_alae", "paid", alae_claims, n_periods = 3,
                              additive = TRUE)
near_abs(additive@total_ultimate, 13215, 2)
near_abs(additive@total_unpaid, 1530, 2)
near_abs(round(additive@cdf[[1]], 4), 0.0019, 0)
moved <- ratio_development(property, "paid_alae", "paid", alae_claims, n_periods = 3,
                           additive = TRUE, additive_selected = c("12" = 0.001),
                           additive_tail = 0.0001)
near_abs(moved@factors[1], 0.001, 0)
near_abs(moved@cdf[[11]], 0.0001, 0)

expect_error_like(ratio_development(property, "paid_alae", "paid", alae_claims, additive_tail = 0.1),
                  "additive = TRUE")
expect_error_like(ratio_development(property, "paid_alae", "paid", alae_claims, selected = 0.1),
                  "named by origin")
expect_error_like(ratio_development(property, "paid_alae", "paid", alae_claims[-1]), "has 10 rows")

# Chapter 17, Exhibits I and II.
xyz_ulae <- c(14.352e6, 15.321e6, 16.870e6, 17.112e6, 17.331e6)
xyz_paid <- c(333e6, 358e6, 334e6, 347e6, 391e6)
xyz_incurred <- c(535.213e6, 492.265e6, 435.985e6, 432.966e6, 475.300e6)

classical <- ulae_estimate(xyz_ulae, xyz_paid, 603e6, 316e6, ratio = 0.045)
near_abs(classical@total, 27787500, 1e-6)
near_abs(round(classical@total_ratio, 3), 0.046, 0)
stopifnot(is.na(classical@expected_claims))
near_abs(ulae_estimate(xyz_ulae, xyz_paid, 603e6, 316e6, ratio = 0.045, pure_ibnr = 19e6)@total,
     21105000, 1e-6)
near_abs(ulae_estimate(xyz_ulae, xyz_paid, 100, 10, ratio = 0.1, opening = 0.25)@total,
     0.1 * (0.75 * 100 + 10), 1e-9)
near_abs(ulae_estimate(xyz_ulae, xyz_paid, 603e6, 316e6)@ratio, sum(xyz_ulae) / sum(xyz_paid), 1e-15)

kittel <- ulae_estimate(xyz_ulae, xyz_paid, 603e6, 316e6, method = "kittel",
                        reported = xyz_incurred, ratio = 0.040)
near_abs(kittel@basis[1], 434106500, 1e-6)
near_abs(kittel@total, 24700000, 1e-6)
near_abs(ulae_estimate(xyz_ulae, xyz_paid, 603e6, 316e6, method = "kittel", reported = xyz_incurred,
                   ratio = 0.040, pure_ibnr = 19e6)@total, 18760000, 1e-6)

# Chapter 17, Exhibit IV.
pqr_ulae <- c(1978, 4820, 8558, 12039, 13143, 15286)
pqr_paid <- c(4590, 14600, 38390, 58297, 86074, 105466)
pqr_ult_reported <- c(27200, 76700, 106900, 154300, 163100, 176400)
pqr_ultimate <- c(28600, 79200, 108400, 156700, 163400, 177100)
generalized <- function(weights) {
  ulae_estimate(pqr_ulae, pqr_paid, 292130, 113853, method = "conger_nolibos",
                reported = pqr_ult_reported, ratio = 0.100, pure_ibnr = 8800, weights = weights,
                ultimate = pqr_ultimate)
}
sixty <- generalized(c(0.6, 0.4))
near_abs(sixty@expected_claims, 15516, 0.5)
near_abs(sixty@bornhuetter_ferguson, 16767, 0.5)
near_abs(sixty@development, 17152, 0.5)
near_abs(sixty@total, 16767, 0.5)
seventy <- generalized(c(0.7, 0.3, 0))
near_abs(seventy@bornhuetter_ferguson, 12795, 0.5)
near_abs(seventy@development, 12201, 0.5)
near_abs(ulae_estimate(pqr_ulae, pqr_paid, 292130, 113853, method = "conger_nolibos",
                   reported = pqr_ultimate, ratio = 0.100, pure_ibnr = 0.04 * 177100,
                   weights = c(0.6, 0.4))@total, 16664, 0.5)

closing <- ulae_estimate(10, 100, 30, 20, method = "conger_nolibos", reported = 200,
                         weights = c(0.5, 0.25, 0.25), pure_ibnr = 8, closed = 80, unclosed = 60)
near_abs(closing@basis, 0.5 * 200 + 0.25 * 100 + 0.25 * 80, 1e-12)
near_abs(closing@total, closing@ratio * (0.5 * 8 + 0.25 * 50 + 0.25 * 60), 1e-12)

expect_error_like(ulae_estimate(1, 1, 1, 1, method = "conger_nolibos", reported = 1), "needs weights")
expect_error_like(ulae_estimate(1, 1, 1, 1, weights = c(0.5, 0.5)), "conger_nolibos")
expect_error_like(ulae_estimate(1, 1, 1, 1, reported = 1), "reported")
expect_error_like(ulae_estimate(1, 1, 1, 1, method = "kittel", reported = 1, opening = 0.25),
                  "opening")
stopifnot(inherits(tryCatch(ulae_estimate(1:2, 1, 1, 1), error = identity), "error"))

cat("test-reserving-expense.R: OK\n")
