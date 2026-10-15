suppressMessages(library(prospicio))

# Selected development patterns (`ldf`, `ldf_kind`, `ldf_weight`) on the
# chain ladder, the expected-loss methods, Mack and the bootstraps
# (docs/design/reserving-v02.md, decision 12). The Rust core is checked
# against chainladder-python and R in validation/tests/reserving_patterns.rs;
# these tests check the binding's arguments, conversions and errors.

near <- function(a, b, rel = 1e-12) {
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

long <- data.frame(
  year = rep(2019:2023, 5:1),
  age = c(12, 24, 36, 48, 60, 12, 24, 36, 48, 12, 24, 36, 12, 24, 12),
  paid = c(100, 150, 165, 170, 172, 110, 170, 180, 185, 120, 175, 192, 130, 200, 140),
  premium = rep(c(250, 260, 270, 280, 290), 5:1)
)
tri <- triangle(long, "year", "age", c("paid", "premium"))
base <- chain_ladder(tri, "paid")

# Each kind selects the same factors.
sel <- c("12" = 1.6, "24" = 1.1, "36" = 1.03, "48" = 1.01)
cdf <- c(rev(cumprod(rev(sel))), 1)
names(cdf) <- c("12", "24", "36", "48", "60")
by_ldf <- chain_ladder(tri, "paid", ldf = sel, ldf_kind = "ldf")
by_cdf <- chain_ladder(tri, "paid", ldf = cdf, ldf_kind = "cdf")
by_pct <- chain_ladder(tri, "paid", ldf = 1 / cdf, ldf_kind = "percent_developed")
by_frame <- chain_ladder(tri, "paid", ldf = data.frame(age = c(12, 24, 36, 48), value = unname(sel)),
                         ldf_kind = "ldf")
for (fit in list(by_ldf, by_cdf, by_pct, by_frame)) {
  near(fit@ldf, sel)
  near(fit@selected_ldf, sel)
  near(fit@ldf_weight, rep(1, 4))
  stopifnot(all(fit@std_err == 0), all(fit@estimated_std_err[1:3] > 0))
  near(fit@estimated_ldf, base@ldf)
  stopifnot(identical(names(fit@selected_ldf), c("12-24", "24-36", "36-48", "48-60")))
}
stopifnot(is.null(by_ldf@pattern_rescale), is.null(base@pattern_rescale))
near(by_cdf@pattern_rescale, 1)
near(by_pct@pattern_rescale, 1)

# A partial pattern for the immature ages; the rest stays estimated.
part <- chain_ladder(tri, "paid", ldf = c("12" = 0.5, "24" = 0.8), ldf_kind = "percent_developed")
stopifnot(identical(unname(is.na(part@selected_ldf)), c(FALSE, TRUE, TRUE, TRUE)))
near(part@ldf, c(1.6, base@ldf[-1]))
near(part@ldf_weight, c(1, 0, 0, 0))
near(part@pattern_rescale, 1 / (0.8 * base@cdf[2]))
near(part@ultimate[5], 140 * 1.6 * base@cdf[2])
near(part@std_err[-1], base@std_err[-1])

# Blends: one weight, or by age.
blend <- chain_ladder(tri, "paid", ldf = sel[1:2], ldf_kind = "ldf", ldf_weight = 0.4)
near(blend@ldf[1:2], 0.4 * sel[1:2] + 0.6 * base@ldf[1:2])
near(blend@blended_ldf, blend@ldf)
near(blend@std_err[1:2], 0.6 * base@std_err[1:2])
by_age <- chain_ladder(tri, "paid", ldf = sel[1:2], ldf_kind = "ldf", ldf_weight = c("24" = 0))
near(by_age@ldf, c(1.6, base@ldf[-1]))
near(chain_ladder(tri, "paid", ldf = sel, ldf_kind = "ldf", ldf_weight = 0)@ultimate,
     base@ultimate)

# Tails: a cumulative pattern is rescaled so that, at the oldest age, its
# percent developed times the tail is 100%; an age-to-age one stops at the
# last age and the tail multiplies on top.
pct <- c("12" = 0.5, "24" = 0.8, "36" = 0.9, "48" = 0.93, "60" = 0.95)
tailed <- chain_ladder(tri, "paid", tail = 1.02, ldf = pct, ldf_kind = "percent_developed")
near(tailed@pattern_rescale, 1 / (0.95 * 1.02))
near(0.95 * tailed@pattern_rescale * tailed@tail, 1)
near(tailed@cdf[1], 2 / tailed@pattern_rescale)
near(totals_frame(tailed)$pattern_rescale, tailed@pattern_rescale)
stopifnot(is.null(chain_ladder(tri, "paid", tail = 1.02, ldf = sel, ldf_kind = "ldf")@pattern_rescale))
# A cumulative pattern may end at a constant tail's attachment age.
attached <- chain_ladder(tri, "paid", tail = tail_constant(1.1, attachment_age = 36),
                         ldf = c("12" = 2, "24" = 1.25, "36" = 1.1), ldf_kind = "cdf")
near(attached@ldf[1:2], c(1.6, 1.25 / 1.1))

# The expected-loss methods develop with the pattern.
pct <- c("12" = 0.5, "24" = 0.8, "36" = 0.9, "48" = 0.95, "60" = 1)
latest <- c(172, 185, 192, 200, 140)
exposure <- c(250, 260, 270, 280, 290)
bf <- bornhuetter_ferguson(tri, "paid", "premium", apriori = 0.7, ldf = pct,
                           ldf_kind = "percent_developed")
near(bf@cdf, 1 / pct)
near(bf@ultimate, latest + (1 - rev(pct)) * 0.7 * exposure)
near(expected_loss(tri, "paid", "premium", apriori = 0.7, ldf = pct,
                   ldf_kind = "percent_developed")@cdf, 1 / pct)
near(benktander(tri, "paid", "premium", apriori = 0.7, ldf = pct,
                ldf_kind = "percent_developed")@ultimate, bf@ultimate)
cc <- cape_cod(tri, "paid", "premium", ldf = pct, ldf_kind = "percent_developed")
near(cc@apriori[1], sum(latest) / sum(exposure * rev(pct)))
stopifnot(all(c("estimated_ldf", "selected_ldf", "ldf_weight", "blended_ldf",
                "estimated_std_err") %in% names(development_frame(bf))))

# Mack: a selected factor has no parameter error, a blended one 1 - w
# times the estimate's; the process risk is the triangle's.
m <- mack(tri, "paid")
own <- stats::setNames(unname(m@ldf), m@development[-5])
full <- mack(tri, "paid", ldf = own, ldf_kind = "ldf")
near(full@ultimate, m@ultimate)
near(full@process_risk, m@process_risk)
stopifnot(all(full@parameter_risk == 0), full@total_parameter_risk == 0)
blended <- mack(tri, "paid", ldf = own, ldf_kind = "ldf", ldf_weight = 0.4)
near(blended@parameter_risk, 0.6 * m@parameter_risk, 1e-9)
near(blended@total_parameter_risk, 0.6 * m@total_parameter_risk, 1e-9)
near(blended@process_risk, m@process_risk)
stopifnot(claims_development_result(full)@total_one_year_standard_error <
            claims_development_result(m)@total_one_year_standard_error)

# Bootstraps: Mack's with every factor selected and no process error is
# the deterministic reserve in every simulation.
fixed <- mack_bootstrap(tri, "paid", n_sims = 200, seed = 1, process = "none", ldf = own,
                        ldf_kind = "ldf")
near(mean(fixed@reserves), base@total_reserve, 1e-9)
stopifnot(variance(fixed@reserves) < 1e-9)
odp <- odp_bootstrap(tri, "paid", n_sims = 500, seed = 1, ldf = own, ldf_kind = "ldf")
stopifnot(variance(odp@reserves) < variance(odp_bootstrap(tri, "paid", n_sims = 500, seed = 1)@reserves))
# One-year views: the bootstrap takes the method's pattern by default.
one <- odp_one_year(tri, "paid", n_sims = 200, seed = 1, ldf = own, ldf_kind = "ldf")
stopifnot(is.finite(variance(one@cdr)))
mk <- mack_one_year(tri, "paid", n_sims = 200, seed = 1, process = "none", ldf = own,
                    ldf_kind = "ldf")
stopifnot(variance(mk@cdr) < 1e-9)
mk_free <- mack_one_year(tri, "paid", n_sims = 200, seed = 1, process = "none", ldf = own,
                         ldf_kind = "ldf", bootstrap_ldf = FALSE)
stopifnot(variance(mk_free@cdr) > 0)

# Errors: the kind is required and checked; the pattern is checked.
expect_error_like(chain_ladder(tri, "paid", ldf = sel), "ldf needs ldf_kind")
expect_error_like(chain_ladder(tri, "paid", ldf = sel, ldf_kind = "LDF"), "ldf_kind must be")
expect_error_like(chain_ladder(tri, "paid", ldf = sel, ldf_kind = c("ldf", "cdf")),
                  "ldf_kind must be")
expect_error_like(chain_ladder(tri, "paid", ldf_kind = "ldf"), "ldf_kind is given without ldf")
expect_error_like(chain_ladder(tri, "paid", ldf_weight = 0.5), "ldf_weight is given without ldf")
expect_error_like(chain_ladder(tri, "paid", ldf = c(1.6, 1.1), ldf_kind = "ldf"),
                  "named numeric vector")
expect_error_like(chain_ladder(tri, "paid", ldf = c(a = 1.6), ldf_kind = "ldf"),
                  "whole number of months")
expect_error_like(chain_ladder(tri, "paid", ldf = data.frame(age = 12, factor = 1.6),
                               ldf_kind = "ldf"), "columns age and value")
expect_error_like(chain_ladder(tri, "paid", ldf = c("18" = 1.6), ldf_kind = "ldf"),
                  "not a development age")
expect_error_like(chain_ladder(tri, "paid", ldf = c("60" = 1.01), ldf_kind = "ldf"),
                  "use a tail")
expect_error_like(chain_ladder(tri, "paid", ldf = c("12" = 0.5, "24" = 1.2),
                               ldf_kind = "percent_developed"), "in (0, 1]")
expect_error_like(chain_ladder(tri, "paid", ldf = c("12" = 0.8, "24" = 0.5),
                               ldf_kind = "percent_developed"), "not decrease")
expect_error_like(chain_ladder(tri, "paid", ldf = c("12" = 1.2, "24" = 1.5), ldf_kind = "cdf"),
                  "not increase")
expect_error_like(chain_ladder(tri, "paid", ldf = c("12" = 1.5), ldf_kind = "cdf"),
                  "two or more ages")
expect_error_like(chain_ladder(tri, "paid", ldf = sel, ldf_kind = "ldf", ldf_weight = 1.5),
                  "between 0 and 1")
expect_error_like(chain_ladder(tri, "paid", ldf = sel[1], ldf_kind = "ldf",
                               ldf_weight = c("24" = 0.5)), "no selected factor")
expect_error_like(chain_ladder(tri, "paid", tail = tail_constant(1.1, attachment_age = 36),
                               ldf = c("36" = 1.05), ldf_kind = "ldf"), "tail")

# Segments: the frames carry the pattern's columns, one rescale per segment.
both <- rbind(transform(long, lob = "auto"), transform(long, lob = "home", paid = paid / 2))
fits <- chain_ladder(triangle(both, "year", "age", "paid", keys = "lob"),
                     ldf = c("12" = 0.5, "24" = 0.8), ldf_kind = "percent_developed")
stopifnot(identical(tail(names(development_frame(fits)), 5),
                    c("estimated_ldf", "selected_ldf", "ldf_weight", "blended_ldf",
                      "estimated_std_err")))
near(totals_frame(fits)$pattern_rescale,
     c(segment(fits, lob = "auto")@pattern_rescale, segment(fits, lob = "home")@pattern_rescale))
stopifnot(!"selected_ldf" %in% names(development_frame(base)))

cat("test-reserving-patterns.R: ok\n")
stopifnot(is.null(totals_frame(chain_ladder(tri, "paid"))$pattern_tail))

# A cumulative pattern to the oldest age with no tail keeps its level as
# the tail; with a tail it is rescaled, and a partial one implies none.
pct95 <- c("12" = 0.5, "24" = 0.8, "36" = 0.9, "48" = 0.93, "60" = 0.95)
implied <- chain_ladder(tri, "paid", ldf = pct95, ldf_kind = "percent_developed")
near(implied@pattern_tail, 1 / 0.95)
near(implied@tail, 1 / 0.95)
near(implied@pattern_rescale, 1)
stopifnot(implied@tail_std_err == 0)
near(implied@ultimate[5], 140 / 0.5)
stopifnot(is.null(chain_ladder(tri, "paid", tail = 1.02, ldf = pct95,
                               ldf_kind = "percent_developed")@pattern_tail))
stopifnot(is.null(part@pattern_tail), is.null(by_ldf@pattern_tail))
near(by_cdf@pattern_tail, 1)
m95 <- mack(tri, "paid", tail_sigma = 1, ldf = pct95, ldf_kind = "percent_developed")
stopifnot(m95@chain_ladder@tail_std_err == 0)
one95 <- odp_one_year(tri, "paid", n_sims = 100, seed = 1, ldf = pct95,
                      ldf_kind = "percent_developed")
stopifnot(is.finite(variance(one95@cdr)))
expect_error_like(odp_one_year(tri, "paid", n_sims = 20, seed = 1, tail = 1.02, ldf = pct95,
                               ldf_kind = "percent_developed"),
                  "the tail the bootstrap's pattern implies")

# A skipped age is interpolated: log cdf linear in age.
gap <- chain_ladder(tri, "paid", ldf = c("12" = 0.5, "60" = 0.98), ldf_kind = "percent_developed")
near(gap@selected_ldf, rep((0.98 / 0.5)^0.25, 4))

cat("test-reserving-patterns.R: ok\n")
