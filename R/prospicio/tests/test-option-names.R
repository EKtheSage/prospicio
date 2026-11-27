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

# A distortion by aggregate's short name (DistortionKind::NAMES, #834).
for (p in list(c("ph", "proportional_hazard", 0.7), c("dual", "dual_power", 2),
               c("exp", "exponential", 0.5))) {
  stopifnot(identical(distortion_g(distortion(p[1], as.double(p[3])), c(0.1, 0.4, 0.9)),
                      distortion_g(distortion(p[2], as.double(p[3])), c(0.1, 0.4, 0.9))))
}
stopifnot(identical(distortion_g(distortion("ly", 0.2, r0 = 0.05), 0.3),
                    distortion_g(distortion("linear_yield", 0.2, r0 = 0.05), 0.3)))
expect_error_like(prospicio:::rust_result(prospicio:::RiskDistortion$new("roe", 0.1, numeric(), 0)),
                  'kind must be "ccoc", "ph", "proportional_hazard", "wang"')

# The GLM family and link names from one table (#834, step 2).
gd <- data.frame(x = rep(0:4, 6), y = 1 + 0.3 * rep(0:4, 6) + rep(c(0, 0.1, 0.2), each = 10))
stopifnot(identical(coef(glm_fit(y ~ x, gd, family = "normal")),
                    coef(glm_fit(y ~ x, gd, family = "gaussian"))))
expect_error_like(glm_fit(y ~ x, gd, family = "weibull"),
                  'family must be "gaussian", "normal", "poisson", "gamma", "inverse_gaussian", "binomial", "negative_binomial", "negbin" or "tweedie", got "weibull"')
expect_error_like(glm_fit(y ~ x, gd, family = "poisson", link = "sqrt"),
                  'link must be "identity", "log", "logit", "probit", "cloglog", "inverse", "inverse_squared" or "power", got "sqrt"')

# The mixing alias "ig".
stopifnot(identical(pmf(mixed_poisson_count(10, 0.5, mixing = "ig"), 0:5),
                    pmf(mixed_poisson_count(10, 0.5, mixing = "inverse_gaussian"), 0:5)))

# The mixing's names from one table (#834, step 2): the same message as
# Python for an unknown name, and "ig" for a bonus-malus portfolio.
expect_error_like(prospicio:::rust_result(prospicio:::ClaimCount$mixed_poisson(10, 0.5, "lognormal", 0, 0)),
                  'mixing must be "gamma", "inverse_gaussian", "ig", "beta", "sichel", "sichel_skewness", "sichel_gamma" or "sichel_ig", got "lognormal"')
scale6 <- bonus_malus(6, start = 5, down = 1, up_per_claim = 2)
stopifnot(identical(bonus_malus_portfolio(scale6, 0.1, cv = 0.8, mixing = "ig")$relativity,
                    bonus_malus_portfolio(scale6, 0.1, cv = 0.8, mixing = "inverse_gaussian")$relativity))

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

# The allocation method in Excel's spelling (AllocationMethod::NAMES, #956).
pd <- predictive_distribution(matrix(c(1, 4, 2, 3, 2, 1, 5, 6), ncol = 2),
                              data.frame(lob = c("motor", "property")))
want <- capital_allocation(pd, distortion("tvar", 0.5), "euler")$by_component$allocated
stopifnot(identical(capital_allocation(pd, distortion("tvar", 0.5), "natural")$by_component$allocated,
                    want))
expect_error_like(capital_allocation(pd, distortion("tvar", 0.5), "nope"), "should be one of")

# A cash-flow option in any case, with "_", "-" or a space between the words
# (prospicio_pricing::cashflow::option_key, #956).
ramp <- function(m) pmin((1:36) / m, 1)
flows_of <- function(base, adjustment, rule, reference, source, interest, fee, release) {
  req <- cashflow_requirement(setNames(c(1, 0.2), c(base, "reserve")))
  terms <- cashflow_terms(100, 30, ramp(12), ramp(24), ramp(36), req,
                          adjustments = list(cashflow_adjustment(18, 0.1, base = adjustment)),
                          funding_rule = rule, reference = reference, funding_source = source,
                          annual_yield = 0.04, interest_basis = interest,
                          fee = list(c(0, 100), c(0.1, 0.2)), fee_basis = fee,
                          equity_release = release)
  cashflow_run(terms, c(0.6, 1.5))$flows
}
want <- flows_of("unearned-premium", "to-date", "obligation-increase", "running-max",
                 "operating-first", "opening", "profit-over-premium", "at-commutation")
stopifnot(identical(
  flows_of("Unearned_Premium", "TO DATE", "Obligation Increase", "running_max", " Operating-First ",
           "OPENING", "profit_over_premium", "At Commutation"),
  want))
expect_error_like(cashflow_terms(100, 30, ramp(12), ramp(24), ramp(36),
                                 cashflow_requirement(c(nope = 1))), "base must be")

# The piecewise Pareto's truncation kind in Excel's spellings (Truncation::NAMES, #956).
for (p in list(c("lp", "lp"), c("last_piece", "lp"), c("wd", "wd"), c("whole_distribution", "wd"))) {
  got <- piecewise_pareto(c(1000, 2000), c(1, 2), truncation = 5000, truncation_type = p[1])
  want <- piecewise_pareto(c(1000, 2000), c(1, 2), truncation = 5000, truncation_type = p[2])
  stopifnot(got@truncation_type == p[2], survival(got, 1500) == survival(want, 1500))
}
expect_error_like(piecewise_pareto(1, 1, truncation = 2, truncation_type = "nope"),
                  "truncation_type must be")

# The P&L face in Excel's short names (Face::NAMES, #956).
local({
  gross <- list(c(2, 3, 4, 5, 6, 8), c(0.25, 0.3, 0.09, 0.2, 0.12, 0.04))
  high <- with_ceding_commission(with_deposit_premium(aggregate_stop_loss("High", 3, 5), 1), 0.1)
  low <- with_deposit_premium(aggregate_stop_loss("Low", 2, 3), 0.5)
  book <- pnl_book("Ex6", 10, list(pnl_expense("premium", 0.1)))
  tower <- reinsurance_tower(list(low, high))
  for (p in list(c("top-down", "peel-top-down"), c("Top_Down", "peel-top-down"),
                 c("bottom-up", "peel-bottom-up"), c("peel bottom up", "peel-bottom-up"))) {
    want <- book_ledger(book, gross, tower, p[2])
    got <- book_ledger(book, gross, tower, p[1])
    stopifnot(identical(mean(ledger_result(got)), mean(ledger_result(want))),
              identical(ledger_sheet(got)$step, ledger_sheet(want)$step))
  }
  expect_error_like(book_ledger(book, gross, NULL, "sideways"), "face must be")
})

# The rate-change rule and the period kind in Excel's spellings (Applies::NAMES and
# PeriodKind::NAMES, #956).
eff <- c(2010.5, 2011, 2011.5, 2012.25)
chg <- c(0.05, 0.10, -0.05, -0.01)
law <- function(a, period = "calendar") {
  rate_level(eff, chg, start = 2011, end = 2012, term = 1, period = period,
             applies = c("written", "written", a, "written"))$on_level_factor
}
for (a in c("inforce", "In-Force", "in force")) stopifnot(identical(law(a), law("in_force")))
expect_error_like(law("sometimes"), "applies must be")
stopifnot(identical(law("in_force", "accident"), law("in_force", "calendar")))
stopifnot(law("in_force", "policy") != law("in_force", "calendar"))
expect_error_like(law("in_force", "fiscal"), "should be one of")

# The rule past the pattern's end: Excel's "none" for "no_development" (#956).
for (s in c("no_development", "none")) {
  stopifnot(identical(as_latest_valuation(tri(), "paid", past_pattern_end = s)@past_pattern_end,
                      "no_development"))
}
expect_error_like(as_latest_valuation(tri(), "paid", past_pattern_end = "hold"),
                  "past_pattern_end must be")

# The loglogistic growth curve in Excel's spellings (#956).
bench <- stats::setNames(c(0.18, 0.47, 0.66, 0.78, 0.855, 0.90, 0.93, 0.951, 0.965, 0.975),
                         seq(12, 120, by = 12))
want <- benchmark_curve(bench)
for (s in c("loglogistic", "log-logistic", "log_logistic")) {
  got <- benchmark_curve(bench, curve = s)
  stopifnot(identical(got@omega, want@omega), identical(got@curve, "loglogistic"))
}
expect_error_like(benchmark_curve(bench, curve = "gompertz"), "should be one of")

# Meyers' model name in any case, as Python takes it (#956).
for (s in c("CSR", "Csr")) {
  stopifnot(identical(meyers_spec(model = s)@model, "csr"))
}
expect_error_like(meyers_spec(model = "ipi"), "should be one of")

# The event loss table's means-alone bucketing in Python's and Excel's
# spellings (#956).
elt <- event_loss_table(rate = c(0.5, 0.25, 0.05), mean = c(130, 270.5, 1234.5),
                        sd_independent = rep(0, 3), sd_correlated = rep(0, 3),
                        exposure = c(260, 541, 2469))
want <- elt_severity_grid(elt, 10, 256, "means_linear")$severity@probs
for (s in c("means_linear", "linear")) {
  got <- elt_severity_grid(elt, 10, 256, s)
  stopifnot(identical(got$severity@probs, want), identical(got$report$bucketing, "means_linear"))
  stopifnot(identical(elt_aggregate_grid(elt, 10, 256, s)$annual@probs,
                      elt_aggregate_grid(elt, 10, 256, "means_linear")$annual@probs))
}
expect_error_like(elt_severity_grid(elt, 10, 256, "nope"), "should be one of")

# The paid-incurred chain's R ChainLadder convention in every spelling (#956).
want <- paid_incurred_chain(tri(), "paid", "incurred", convention = "r_chainladder")@total_reserve
for (s in c("r-chainladder", "chainladder", "r")) {
  stopifnot(identical(
    paid_incurred_chain(tri(), "paid", "incurred", convention = s)@total_reserve, want))
}
expect_error_like(paid_incurred_chain(tri(), "paid", "incurred", convention = "nope"),
                  "should be one of")

# Cover basis: "lod" and "rad" are the same as the full names (#1035).
cover_ev <- events_from_years(list(100), times = list(13.5 / 24))
cover_shares <- matrix(0, 2, 24)
cover_shares[1, 14] <- 0.25
cover_shares[2, 14] <- 0.75
for (pair in list(c("risks_attaching", "rad"), c("losses_occurring", "lod"))) {
  a <- timeline_years(timeline(cover_ev, 24), pair[1], cover_shares)
  b <- timeline_years(timeline(cover_ev, 24), pair[2], cover_shares)
  stopifnot(identical(sapply(a, function(y) mean(total(y))), sapply(b, function(y) mean(total(y)))))
}
expect_error_like(timeline_years(timeline(cover_ev, 24), "claims_made", cover_shares),
                  'basis must be "losses_occurring", "lod", "risks_attaching" or "rad", got "claims_made"')

cat("option name tests passed\n")

