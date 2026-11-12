library(prospicio)
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# One vocabulary for option names across Python, R and Excel (#834, step 1):
# each spelling an option accepts in any front end is accepted here too and
# gives the same result as the canonical name.


rows <- list(c(110, 290, 370, 420, 440, 452), c(95, 300, 390, 425, 447),
             c(130, 320, 410, 470), c(105, 305, 395), c(120, 310), 125)
long <- do.call(rbind, lapply(seq_along(rows), function(i) {
  data.frame(origin = 2019 + i, age = 12 * seq_along(rows[[i]]), paid = rows[[i]],
             incurred = 1.3 * rows[[i]])
}))
tri <- function(grain = "Y") {
  triangle(long, "origin", "age", c("paid", "incurred"), origin_grain = grain,
           development_grain = grain)
}

# The count distribution of a zero-inflated model.
root <- if (dir.exists("validation")) "validation" else file.path("..", "..", "..", "validation")
bio <- utils::read.csv(file.path(root, "data", "biochemists.csv"))
stopifnot(identical(coef(zeroinfl_fit(art ~ ment, bio, dist = "negbin")),
                    coef(zeroinfl_fit(art ~ ment, bio, dist = "negative_binomial"))))

# Log-linear in three spellings: Mack's sigma, the Munich chain ladder's
# sigmas and the paid-incurred chain's variance extrapolation.
want <- mack(tri(), "paid")@total_reserve
for (s in c("loglinear", "log_linear")) {
  stopifnot(identical(mack(tri(), "paid", sigma_interpolation = s)@total_reserve, want))
  stopifnot(identical(
    munich_chain_ladder(tri(), sigma_interpolation_paid = s, sigma_interpolation_incurred = s)@ultimate_paid,
    munich_chain_ladder(tri())@ultimate_paid))
}
want <- paid_incurred_chain(tri(), "paid", "incurred")@total_reserve
for (s in c("log-linear", "log_linear")) {
  stopifnot(identical(paid_incurred_chain(tri(), "paid", "incurred", extrapolation = s)@total_reserve,
                      want))
}

# Grains in any case, trimmed, by letter or by word.
want <- chain_ladder(tri("Y"), "paid")@ultimate
for (g in c("y", " Y ", "year", "Year")) {
  stopifnot(identical(chain_ladder(tri(g), "paid")@ultimate, want))
}

# The mixing alias "ig".
stopifnot(identical(pmf(mixed_poisson_count(10, 0.5, mixing = "ig"), 0:5),
                    pmf(mixed_poisson_count(10, 0.5, mixing = "inverse_gaussian"), 0:5)))

# A selected pattern's kind in Excel's spellings.
pattern <- list(
  ldf = c(`12` = 2.5, `24` = 1.3, `36` = 1.1, `48` = 1.05, `60` = 1.02),
  cdf = c(`12` = 4.0, `24` = 1.6, `36` = 1.25, `48` = 1.12, `60` = 1.04),
  percent_developed = c(`12` = 0.25, `24` = 0.625, `36` = 0.8, `48` = 0.89, `60` = 0.96)
)
for (p in list(c("ldf", "age-to-age"), c("cdf", "age-to-ultimate"),
               c("percent_developed", "percent"), c("percent_developed", "percent developed"))) {
  v <- pattern[[p[1]]]
  stopifnot(identical(chain_ladder(tri(), "paid", ldf = v, ldf_kind = p[2])@ultimate,
                      chain_ladder(tri(), "paid", ldf = v, ldf_kind = p[1])@ultimate))
}
bench <- c(`12` = 0.18, `24` = 0.47, `36` = 0.66, `48` = 0.78, `60` = 0.855, `72` = 0.90)
for (k in c("percent", "percent developed")) {
  stopifnot(identical(benchmark_curve(bench, kind = k)@omega, benchmark_curve(bench)@omega))
}

# Distortion families by short or long name, in any case, with spaces or
# hyphens.
x <- sampled(c(22, 28, 36, 40, 40, 40, 40, 55, 65, 100))
premium <- (46.6 + 0.15 * 100) / 1.15
for (f in c("ph", "proportional_hazard", "Proportional Hazard", "proportional-hazard")) {
  stopifnot(abs(risk_measure(x, calibrate_distortion(f, x, premium)) - premium) < 1e-9)
}
expect_error_like(calibrate_distortion("bitvar", x, premium), "unknown distortion family")

cat("option name tests passed\n")
