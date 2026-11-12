suppressMessages(library(prospicio))
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# industry_prior(), pattern_prior() and bayes_bf_industry(): the industry
# prior for the Bayesian BF from other companies' triangles
# (docs/design/reserving-bayes.md, decision 29, #604). The recovery of
# simulated companies' parameters is checked in
# validation/tests/reserving_industry_prior.rs; these tests check the binding
# on four companies with exact cells, whose prior has a closed form, the same
# numbers Python checks (python/tests/test_reserving_industry_prior.py).



check <- function(ok, what) if (!isTRUE(ok)) stop(what, call. = FALSE)

shares <- c(0.4, 0.3, 0.2, 0.1)
levels <- c(0.6, 0.8, 0.7, 0.9)
wobble <- c(1.1, 0.9, 1.0, 1.0)
premium <- 100

# Four origins at four ages, exactly premium * level * wobble * share
# cumulated: each company's chain ladder returns its pattern and loss ratios.
rows <- function(level, noise = NULL) {
  out <- NULL
  k <- 0
  for (w in 1:4) {
    cum <- 0
    for (d in seq_len(5 - w)) {
      k <- k + 1
      cell <- if (is.null(noise)) premium * level * wobble[w] * shares[d] else
        premium * level * shares[d] * noise[k]
      cum <- cum + cell
      out <- rbind(out, data.frame(year = 2009 + w, age = 12 * d, paid = cum, premium = premium))
    }
  }
  out
}

company <- function(level) triangle(rows(level), "year", "age", "paid")
companies <- lapply(levels, company)
premiums <- rep(list(rep(premium, 4)), 4)

# The closed form (the Rust docs of IndustryPrior): r = 0.02 / 3, n = 4.
apriori <- mean(levels)
r <- 0.02 / 3
within <- sqrt(r / (1 - r / 4))
between <- sqrt((1 + var(levels) / apriori^2) / (1 + within^2 / 4) - 1)

prior <- industry_prior(companies, "paid", premiums = premiums, min_developed = 0)
near_abs(prior@apriori, apriori, 1e-12)
near_abs(prior@within_cv, within, 1e-12)
near_abs(prior@cv, within, 1e-12)
near_abs(prior@between_cv, between, 1e-12)
near_abs(prior@common_cv, prior@between_cv, 0)
near_abs(prior@process_cv, 0, 1e-6)
near_abs(prior@company_loss_ratios, levels, 1e-12)
near_abs(prior@apriori_se, sqrt(var(levels) / 4), 1e-12)
near_abs(prior@pattern@ages, c(12, 24, 36, 48), 0)
near_abs(prior@pattern@mean, log(shares[-1] / 0.4), 1e-12)
near_abs(prior@pattern@sd, rep(0.01, 3), 0)
near_abs(prior@percent_developed, c(0.4, 0.7, 0.9, 1.0), 1e-12)
report <- prior@report
near_abs(c(report$n_companies, report$n_companies_used, report$n_origins_used), c(4, 4, 16), 0)
near_abs(report$origins_used, rep(4, 4), 0)
near_abs(report$companies_by_age, rep(4, 3), 0)
check(any(grepl("only 4 companies", report$warnings, fixed = TRUE)), "no small-panel warning")
check(grepl("^industry_prior\\(apriori = 0.7500", capture.output(print(prior))), "print")

# min_developed 0.5 leaves out the newest origin (0.4 developed).
near_abs(industry_prior(companies, "paid", premiums = premiums)@report$origins_used, rep(3, 4), 0)
warnings <- industry_prior(companies, "paid", premiums = premiums, min_companies = 3)@report$warnings
check(!any(grepl("only", warnings, fixed = TRUE)), "a small-panel warning at min_companies 3")
check(sum(grepl("set to 0.01", warnings, fixed = TRUE)) == 3, "the floor warnings")

# One triangle with a segment per company, premium from a column.
stacked <- do.call(rbind, lapply(seq_along(levels), function(k) {
  cbind(rows(levels[k]), company = sprintf("c%d", k - 1))
}))
by_segment <- industry_prior(triangle(stacked, "year", "age", c("paid", "premium"),
                                      keys = "company"),
                             "paid", exposure = "premium", min_developed = 0)
near_abs(by_segment@apriori, prior@apriori, 1e-12)
near_abs(by_segment@between_cv, prior@between_cv, 1e-12)

expect_error_like(industry_prior(companies[1], "paid", premiums = premiums[1]),
                  "at least two companies")
expect_error_like(industry_prior(companies[1:2], "paid", premiums = premiums[1]), "premiums")
expect_error_like(industry_prior(companies[1:2], "paid", premiums = premiums[1:2],
                                 min_developed = 2), "min_developed")
expect_error_like(industry_prior(companies, "paid"), "premiums")

# The pattern prior and the cut.
p <- pattern_prior(c(12, 24, 36), shares = c(0.5, 0.3, 0.2), sd = c(0.1, 0.2))
near_abs(p@shares, c(0.5, 0.3, 0.2), 1e-12)
near_abs(p@percent_developed, c(0.5, 0.8, 1.0), 1e-12)
near_abs(p@sd, c(0.1, 0.2), 0)
near_abs(pattern_prior(c(12, 24), mean = 0, sd = 1)@shares, c(0.5, 0.5), 1e-12)
expect_error_like(pattern_prior(c(12, 24), mean = 0, sd = 0), "positive")
expect_error_like(pattern_prior(c(12, 24), sd = 1), "either mean or shares")

cut <- industry_prior_at_age(prior, 36)
near_abs(cut@apriori, prior@apriori * 0.9, 1e-12)
near_abs(cut@tail_factor, 1 / 0.9, 1e-12)
near_abs(cut@pattern@ages, c(12, 24, 36), 0)
check(any(grepl("36 months", cut@report$warnings, fixed = TRUE)), "the cut's warning")

# A short fit on a program whose cells wobble about the industry pattern (so
# its ODP scale is positive): the pattern prior (sd 0.01) holds the posterior
# pattern at the industry shares.
noise <- c(1.04, 0.97, 1.02, 0.99, 0.96, 1.03, 1.01, 0.98, 1.02, 0.97)
program <- triangle(rows(0.7, noise), "year", "age", c("paid", "premium"))
fit <- bayes_bf_industry(program, "paid", "premium", prior, chains = 2, tune = 500, draws = 500,
                         n_sims = 200, seed = 1)
check(S7::S7_inherits(fit, bayes_bf_normalised_fit), "a bayes_bf_normalised_fit")
near_abs(fit@pattern, shares, 0.02 * max(shares))
again <- bayes_bf_normalised(program, "paid", "premium", apriori = prior@apriori, cv = prior@cv,
                             common_cv = prior@common_cv, pattern_prior = prior@pattern,
                             chains = 2, tune = 500, draws = 500, n_sims = 200, seed = 1)
near_abs(again@pattern, fit@pattern, 0)
expect_error_like(bayes_bf_industry(program, "paid", "premium", cut, chains = 1, tune = 50,
                                    draws = 50, n_sims = 10), "truncated")
view <- bayes_bf_industry_one_year(program, "paid", "premium", prior, chains = 2, tune = 500,
                                   draws = 500, n_sims = 200, seed = 1, one_year_sims = 50,
                                   refit = "reweight_only")
near_abs(view@fit@pattern, fit@pattern, 0)

cat("test-reserving-industry-prior.R: ok\n")
