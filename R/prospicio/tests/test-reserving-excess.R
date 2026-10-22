suppressMessages(library(prospicio))

# Excess-layer development from a ground-up triangle
# (docs/design/reserving-excess.md, PR 3). The Rust core is checked against
# the literature in validation/tests/reserving_excess.rs; these tests
# reproduce the same printed targets through the R bindings (Bloom and
# Patrik, CAS E-Forum Fall 2013; Pinto and Gogol, PCAS 74; Bear, PCAS 79),
# the same numbers as python/tests/test_reserving_excess.py, and check the
# binding's arguments and errors.

near <- function(a, b, tol = 1e-11) {
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

# The ultimate claims 450, 1,687.5, 1,125 and 2,250 times the scale, each
# with probability 1/4, as a grid of step 112.5 times the scale.
ultimate <- function(scale) {
  probs <- numeric(21)
  probs[c(5, 16, 11, 21)] <- 0.25
  grid_distribution(112.5 * scale, probs)
}
ground_up <- function(scale) {
  total <- 925 * scale
  long <- data.frame(year = 2020, age = c(12, 24, 36),
                     incurred = c(total, total * 3.6, total * 4.5))
  triangle(long, "year", "age", "incurred")
}
pattern <- function(scale, layer) {
  excess_pattern(ground_up(scale), "incurred", layer, scale_shift(ultimate(scale)))
}

# Bloom and Patrik, Year One.
p <- pattern(1, excess_layer(350))
near(p@ages, c(12, 24, 36), 0)
near(p@layer_ldf[1], 17.2)
near(p@layer_ldf[2], 4112.5 / 3010)
near(p@layer_cdf[1], 23.5)
near(p@shifts, c(4.5, 1.25, 1), 1e-12)
near(p@ground_up_ldf, c(3.6, 1.25), 1e-12)
near(p@tail, 1, 0)
stopifnot(identical(names(p@ldf), c("12", "24")))

# Years Two and Three.
two <- pattern(1.05, excess_layer(350))
near(two@layer_ldf, c(14.768, 1.3583), 5e-4)
near(two@layer_cdf[1], 20.06, 5e-3)
three <- pattern(1.1025, excess_layer(350))
near(three@layer_ldf, c(13.0797, 1.3511), 5e-5)
near(three@layer_cdf[1], 17.6719, 5e-5)

# The layer at each shift: 175, 3,010 and 4,112.5 for the four claims.
for (i in 1:3) {
  near(4 * layer_at_shift(ultimate(1), excess_layer(350), c(4.5, 1.25, 1)[i]),
       c(175, 3010, 4112.5)[i], 1e-9)
}

# A sparse layer triangle (empty at 12 months) runs on the derived pattern.
sparse <- triangle(
  data.frame(year = c(2020, 2020, 2020, 2021, 2021, 2022), age = c(12, 24, 36, 12, 24, 12),
             layer = c(0, 3010, 4112.5, 0, 3231, 0)),
  "year", "age", "layer")
expect_error_like(chain_ladder(sparse, "layer"), "")
sel <- excess_selected(p)
stopifnot(identical(sel$ldf_kind, "ldf"), sel$ldf_weight == 1)
fit <- do.call(chain_ladder, c(list(sparse, "layer", tail = p@tail), sel))
near(fit@ldf, c(17.2, 4112.5 / 3010))
near(fit@ultimate[2], 3231 * 4112.5 / 3010, 1e-9)
# Mack and the expected-loss methods take the same arguments.
full <- triangle(
  data.frame(year = c(2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2022, 2023, 2023, 2024),
             age = c(12, 24, 36, 12, 24, 36, 12, 24, 36, 12, 24, 12),
             layer = c(175, 3010, 4112.5, 175, 3010, 4112.5, 175, 3010, 4112.5, 175, 3010, 175),
             premium = 10000),
  "year", "age", c("layer", "premium"))
m <- do.call(mack, c(list(full, "layer", tail = p@tail), sel))
near(m@ldf[1], 17.2)
bf <- do.call(bornhuetter_ferguson,
              c(list(full, "layer", "premium", apriori = 0.2, tail = p@tail), sel))
near(bf@ldf[1], 17.2)

# A weight blends with the triangle's own factors.
stopifnot(excess_selected(p, 0.5)$ldf_weight == 0.5)
w2 <- excess_selected(p, c("12" = 1, "24" = 0.25))$ldf_weight
stopifnot(identical(names(w2), c("12", "24")))
expect_error_like(excess_selected(p, "half"), "weight must be")

# Pinto and Gogol p. 247: 2.5 (4)^0.5 = 5.0, from the losses excess of
# 25,000 and from the counts above it.
two_ages <- function(d) triangle(data.frame(year = 2020, age = c(12, 24), base = c(1, d)),
                                 "year", "age", "base")
shape <- pareto_shape(25000, c("12" = 1.75, "24" = 1.25), 1.25, base = excess_layer(25000))
ps <- excess_pattern(two_ages(2.5), "base", excess_layer(100000), shape)
near(ps@layer_ldf, 5, 1e-12)
near(ps@tail, 1, 0)
stopifnot(length(ps@shifts) == 0)
counts_shape <- pareto_shape(25000, c("12" = 1.75, "24" = 1.25), 1.25)
pc <- excess_pattern(two_ages(2.5 * 0.25 / 0.75), "base", excess_layer(100000), counts_shape)
near(pc@layer_ldf, 5, 1e-12)

# Bear p. 138: 3.5 (2.0)^0.15 = 3.8835, printed 3.88.
bear <- pareto_shape(1e5, c("12" = 1.25, "24" = 1.10), 1.10, base = excess_layer(1e5, 2e5))
pb <- excess_pattern(two_ages(3.5), "base", excess_layer(2e5, 4e5), bear)
near(pb@layer_ldf, 3.5 * 2^0.15, 1e-12)
near(pb@layer_ldf, 3.8835, 5e-5)
near(pb@layer_ldf, 3.88, 5e-3)

# Credibility weights: the layer triangle's ratios equal the pattern's
# (172 to 235 is 4112.5 / 3010), so the cap is idle and the weights are
# k / (n + k) in the weight on the derived pattern, 1 - n / (n + k).
rows <- data.frame(
  year = c(2020, 2020, 2020, 2021, 2021, 2021, 2022, 2022, 2023),
  age = c(12, 24, 36, 12, 24, 36, 12, 24, 12),
  layer = c(10, 172, 235, 10, 172, 235, 10, 172, 10),
  count = c(2, 2, 2, 3, 3, 3, 5, 5, 1))
ltri <- triangle(rows, "year", "age", c("layer", "count"))
w <- excess_credibility_weights(p, ltri, "layer", k = 3, max_influence = 0.1)
stopifnot(identical(names(w), c("12", "24")))
near(w, c(0.5, 0.6))
wc <- excess_credibility_weights(p, ltri, "layer", k = 3, max_influence = 0.1, counts = "count")
near(wc, c(3 / 13, 3 / 8))
cl <- do.call(chain_ladder, c(list(ltri, "layer", tail = p@tail), excess_selected(p, w)))
near(cl@ldf_weight, unname(w))
expect_error_like(excess_credibility_weights(p, ltri, "layer", k = 0, max_influence = 0.1), "k")
expect_error_like(excess_credibility_weights(p, ltri, "layer", k = 1, max_influence = 0), "max_influence")

# Layer shares and exposure.
near(layer_share(pareto(1, 2), excess_layer(4)), 0.125, 1e-12)
ln <- lognormal(8, 1)
lay <- excess_layer(5000, 20000)
near(layer_at_shift(ln, lay, 1) / mean(ln), layer_share(ln, lay), 1e-12)
etri <- triangle(
  data.frame(year = c(2020, 2020, 2021), age = c(12, 24, 12), paid = c(10, 15, 11),
             premium = c(100, 100, 120)),
  "year", "age", c("paid", "premium"))
out <- layer_exposure(etri, "premium", c(0.1, 0.2), "layer_premium")
stopifnot("layer_premium" %in% out@columns, identical(etri@columns, c("paid", "premium")))
expect_error_like(layer_exposure(etri, "premium", c(0.1, 0.2, 0.3), "x"), "")
expect_error_like(layer_exposure(etri, "premium", 1.5, "x"), "share")
# The exposure-curve route (decision 2): the curve's own layer share.
share <- exposure_layer_share(swiss_re_curve(3), 5e6, 5e6, 10e6)
stopifnot(share > 0, share < 1)
one <- layer_exposure(etri, "premium", share, "layer_premium")
stopifnot(identical(one@columns, c("paid", "premium", "layer_premium")))

# A tail and counts.
ctri <- triangle(
  data.frame(year = 2020, age = c(12, 24, 36), incurred = c(1225, 4410, 5512.5),
             n = c(10, 12, 12)),
  "year", "age", c("incurred", "n"))
model <- scale_shift(ultimate(1))
plain <- excess_pattern(ctri, "incurred", excess_layer(350), model)
counted <- excess_pattern(ctri, "incurred", excess_layer(350), model, counts = "n")
stopifnot(!isTRUE(all.equal(counted@layer_ldf, plain@layer_ldf)))
tailed <- excess_pattern(ctri, "incurred", excess_layer(350), model, tail = 1.1)
near(tailed@ground_up_tail, 1.1, 1e-12)
stopifnot(tailed@tail != 1)
via_object <- excess_pattern(ctri, "incurred", excess_layer(350), model, tail = tail_constant(1.1))
near(via_object@layer_cdf, tailed@layer_cdf, 1e-12)

# Errors and objects.
expect_error_like(excess_layer(-1), "attachment")
expect_error_like(excess_layer(0, 0), "limit")
expect_error_like(excess_pattern(ctri, "incurred", 350, model), "excess_layer")
expect_error_like(excess_pattern(ctri, "incurred", excess_layer(350), "scale"), "model")
expect_error_like(pareto_shape(1, c("12" = 1), 1, base = "losses"), "base")
expect_error_like(
  excess_pattern(ctri, "incurred", excess_layer(350),
                 pareto_shape(100, c("12" = 1.5, "24" = 1.2, "36" = 1.1), 1.1), counts = "n"),
  "count")
expect_error_like(layer_at_shift(ultimate(1), excess_layer(350), 0), "shift")
expect_error_like(scale_shift(1), "severity")
stopifnot(excess_layer(350, 650)@exhaustion == 1000, excess_layer(350)@limit == Inf)

cat("ok\n")
