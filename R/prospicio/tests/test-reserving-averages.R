suppressMessages(library(prospicio))

# Averages over the latest periods, excluding the highest and lowest link
# ratios, and the geometric average (`n_periods`, `drop_high`, `drop_low`
# and `average = "geometric"`; docs/design/reserving-v02.md, decision 13).
# The Rust core is checked against chainladder-python in
# validation/tests/reserving_averages.rs; these tests check the binding's
# arguments, results and errors against the same reference file.

near <- function(a, b, rel = 1e-9) {
  a <- unname(a)
  b <- unname(b)
  if (length(a) != length(b) || !all(abs(a - b) <= rel * pmax(abs(b), 1e-300))) {
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

# Tests run from the repository root (CI and `cargo xtask r`).
validation <- function(...) file.path("validation", ...)
raa_long <- utils::read.csv(validation("data", "raa.csv"), comment.char = "#")
raa <- triangle(raa_long, "origin", "development", "value")

lines <- grep("^#", readLines(validation("reference", "reserving_averages_python.csv")),
              value = TRUE, invert = TRUE)
ref <- utils::read.csv(text = lines, stringsAsFactors = FALSE)
ref <- ref[ref$dataset == "raa", ]
want <- function(method, quantity) {
  rows <- ref[ref$method == method & ref$quantity == quantity, ]
  rows[order(rows$arg), "expected"]
}

# One case each against chainladder-python 0.10.1.
volume3 <- chain_ladder(raa, average = "volume", n_periods = 3)
stopifnot(volume3@excluded)
near(volume3@ldf, want("average=volume;n_periods=3", "ldf"))
near(volume3@n_links, want("average=volume;n_periods=3", "n_links"))
stopifnot(is.integer(volume3@n_links),
          identical(names(volume3@n_links), names(volume3@ldf)))
mack3 <- mack(raa, average = "volume", n_periods = 3)
near(mack3@sigma, want("average=volume;n_periods=3", "sigma"))
# The reference has no standard error for the oldest link, which has one ratio.
near(utils::head(mack3@std_err, 8), want("average=volume;n_periods=3", "std_err"))

medial <- chain_ladder(raa, average = "simple", drop_high = 1, drop_low = 1)
near(medial@ldf, want("average=simple;drop_high=1;drop_low=1", "ldf"))
near(medial@n_links, want("average=simple;drop_high=1;drop_low=1", "n_links"))
stopifnot(medial@excluded)

# No exclusions is the default fit, and reports every link.
base <- chain_ladder(raa)
stopifnot(!base@excluded, identical(unname(base@n_links), 9:1))
same <- chain_ladder(raa, n_periods = NULL, drop_high = 0, drop_low = 0)
near(same@ldf, base@ldf, 1e-15)

# The other methods take the same settings.
long_p <- merge(raa_long, data.frame(origin = 1981:1990, premium = 100000 * (1:10)), by = "origin")
raa_p <- triangle(long_p, "origin", "development", c("value", "premium"))
for (fit in list(
  expected_loss(raa_p, "value", "premium", apriori = 0.6, n_periods = 3),
  bornhuetter_ferguson(raa_p, "value", "premium", apriori = 0.6, drop_high = 1, drop_low = 1),
  benktander(raa_p, "value", "premium", apriori = 0.6, n_periods = 5, drop_high = 1),
  cape_cod(raa_p, "value", "premium", n_periods = 4)
)) {
  stopifnot(is.finite(fit@total_reserve))
}
bf3 <- bornhuetter_ferguson(raa_p, "value", "premium", apriori = 0.6, n_periods = 3)
bf_all <- bornhuetter_ferguson(raa_p, "value", "premium", apriori = 0.6)
stopifnot(bf3@total_reserve != bf_all@total_reserve)

# Geometric: exp(mean(log(link ratios))) at each age.
geo <- chain_ladder(raa, average = "geometric")
wide <- tapply(raa_long$value, list(raa_long$origin, raa_long$development), sum)
geo_want <- vapply(1:9, function(k) {
  r <- wide[, k + 1] / wide[, k]
  exp(mean(log(r[is.finite(r)])))
}, 0)
near(geo@ldf, geo_want)
near(geo@n_links, 9:1)
stopifnot(is.nan(geo@alpha), !geo@excluded)
stopifnot(chain_ladder(raa, average = "geometric", n_periods = 3)@excluded)

# Rejected combinations surface the library's errors.
expect_error_like(mack(raa, average = "geometric"), "geometric")
expect_error_like(mack_bootstrap(raa, n_sims = 10, average = "geometric"), "geometric")
expect_error_like(chain_ladder(raa, n_periods = 0), "n_periods")
expect_error_like(odp_bootstrap(raa, n_sims = 10, n_periods = 0), "n_periods")
expect_error_like(odp_one_year(raa, n_sims = 10, average = "geometric"), "geometric")

# The bootstraps and the one-year views follow the exclusions
# (decision 14): settings that leave out no link ratio change nothing, seed
# for seed; real ones change the draws; the scale is biased low.
for (boot in list(odp_bootstrap, mack_bootstrap)) {
  plain <- boot(raa, n_sims = 200, seed = 7)
  stopifnot(identical(mean(boot(raa, n_sims = 200, seed = 7, n_periods = 50)@reserves),
                      mean(plain@reserves)))
  stopifnot(mean(boot(raa, n_sims = 200, seed = 7, n_periods = 3)@reserves) !=
              mean(plain@reserves))
  stopifnot(mean(boot(raa, n_sims = 200, seed = 7, drop_high = 1, drop_low = 1)@reserves) !=
              mean(plain@reserves))
}
near(odp_bootstrap(raa, n_sims = 10, seed = 1)@scale, 983.6, 1e-4)
near(odp_bootstrap(raa, n_sims = 10, seed = 1, n_periods = 3)@scale, 694.0, 1e-3)

# One-year views take the method's exclusions; Mack's bootstrap may name
# them too, but they must be the method's.
for (view in list(odp_one_year, mack_one_year)) {
  inherited <- view(raa, n_sims = 200, seed = 3, n_periods = 3)
  plain <- view(raa, n_sims = 200, seed = 3)
  stopifnot(mean(inherited@cdr) != mean(plain@cdr))
}
named <- mack_one_year(raa, n_sims = 200, seed = 3, n_periods = 3, mack_n_periods = 3)
stopifnot(identical(mean(named@cdr), mean(mack_one_year(raa, n_sims = 200, seed = 3,
                                                        n_periods = 3)@cdr)))
expect_error_like(mack_one_year(raa, n_sims = 10, mack_n_periods = 3), "exclusions")
expect_error_like(mack_one_year(raa, n_sims = 10, n_periods = 4, mack_n_periods = 3), "exclusions")
expect_error_like(claims_development_result(mack(raa, n_periods = 3)), "MackBootstrap")

# Argument types are checked in R.
expect_error_like(chain_ladder(raa, n_periods = 2.5), "n_periods must be NULL or a single whole")
expect_error_like(chain_ladder(raa, n_periods = "3"), "n_periods must be NULL or a single whole")
expect_error_like(chain_ladder(raa, drop_high = -1), "drop_high must be a single whole")
expect_error_like(chain_ladder(raa, drop_low = c(1, 2)), "drop_low must be a single whole")
expect_error_like(chain_ladder(raa, drop_low = NA), "drop_low must be a single whole")
expect_error_like(mack_one_year(raa, mack_drop_high = 1.5), "mack_drop_high must be a single whole")

# A cumulative pattern with a factor to ultimate below 1, in any order, as
# Friedland's XYZ reported pattern has (0.992 at 96 and 108 months).
long <- data.frame(
  year = rep(2019:2023, 5:1),
  age = c(12, 24, 36, 48, 60, 12, 24, 36, 48, 12, 24, 36, 12, 24, 12),
  paid = c(100, 150, 165, 170, 172, 110, 170, 180, 185, 120, 175, 192, 130, 200, 140)
)
tri <- triangle(long, "year", "age", "paid")
cdf <- c("12" = 2.5, "24" = 1.2, "36" = 0.99, "48" = 0.995, "60" = 1)
fit <- chain_ladder(tri, "paid", ldf = cdf, ldf_kind = "cdf")
near(fit@ldf, cdf[-5] / cdf[-1], 1e-12)
pct <- chain_ladder(tri, "paid", ldf = 1 / cdf, ldf_kind = "percent_developed")
near(pct@ldf, cdf[-5] / cdf[-1], 1e-12)
# The pattern reaches the oldest age, so its last value is the tail level.
near(fit@pattern_tail, cdf[5])
near(pct@pattern_tail, cdf[5])
near(fit@tail, cdf[5])

# n_links and excluded are per-segment properties: with several segments they
# raise, as the other single-segment properties do.
two <- triangle(rbind(transform(raa_long, lob = "a"),
                      transform(raa_long, lob = "b", value = value * 1.1)),
                "origin", "development", "value", keys = "lob")
seg_fit <- chain_ladder(two, n_periods = 3)
expect_error_like(seg_fit@n_links, "n_links needs a single-segment fit, and this one has 2 segments")
expect_error_like(seg_fit@excluded, "excluded needs a single-segment fit, and this one has 2 segments")
stopifnot(segment(seg_fit, lob = "a")@excluded,
          identical(unname(segment(seg_fit, lob = "b")@n_links), c(3L, 3L, 3L, 3L, 3L, 3L, 3L, 2L, 1L)))

cat("reserving averages: ok\n")
