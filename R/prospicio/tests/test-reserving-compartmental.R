suppressMessages(library(prospicio))

# compartmental() and compartmental_model_2(), Gesmann and Morris'
# compartmental models 1 and 2 (docs/design/reserving-compartmental.md). The
# parity with the paper's brms code in Stan and its printed posterior is
# validation/tests/reserving_compartmental.rs (model 1) and
# reserving_compartmental_gm2.rs (model 2); these tests check the binding on
# company 337 workers' compensation as the papers fit it.

near <- function(a, b, rel) {
  if (!all(abs(a - b) <= rel * pmax(abs(b), 1e-300))) {
    stop(sprintf("got %s, want %s (rel %g)", format(a, digits = 17), format(b, digits = 17), rel),
         call. = FALSE)
  }
}

expect_error_like <- function(expr, pattern) {
  msg <- tryCatch({
    expr
    NULL
  }, error = conditionMessage)
  if (is.null(msg)) stop("expected an error", call. = FALSE)
  if (!grepl(pattern, msg, fixed = TRUE)) {
    stop(sprintf("error %s does not mention %s", shQuote(msg), shQuote(pattern)), call. = FALSE)
  }
}

# The warnings `expr` raises, and its value.
with_warnings <- function(expr) {
  found <- character()
  value <- withCallingHandlers(expr, warning = function(w) {
    found <<- c(found, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = found)
}

# Tests run from the repository root (CI and `cargo xtask r`). The papers'
# training triangle: the cells at calendar years up to 1996, and accident
# year 1997 with its (direct) premium only.
all <- utils::read.csv(file.path("validation", "data", "clrd_wkcomp_337.csv"),
                       comment.char = "#")
long <- all[all$origin + all$development %/% 12 - 1 <= 1996, ]
stopifnot(nrow(long) == 45)
y1997 <- all[all$origin == 1997 & all$development == 12, ]
y1997$paid <- NA
y1997$incurred <- NA
long <- rbind(long, y1997)
tri <- triangle(long, "origin", "development", c("paid", "incurred", "premium"))
gm <- gesmann_morris_priors()

# The priors and their preset.
stopifnot(
  inherits(gm, "compartmental_priors"),
  identical(gm$k_er, c(median = 3, sd = 0.1)),
  identical(gm$rrf_effect_sd, list(family = "student_t", nu = 10, scale = 0.1)),
  identical(gm$log_sigma[[2]], list(family = "student_t", nu = 1, location = 0, scale = 1000)),
  gm$lkj_eta == 1
)
q <- compartmental_priors(k_er = c(2, 0.3), k_p = gm$k_p, rlr = gm$rlr, rrf = gm$rrf,
                          rlr_effect_sd = gm$rlr_effect_sd,
                          rrf_effect_sd = list(family = "normal", scale = 0.2), lkj_eta = 2,
                          log_sigma = list(list(family = "normal", mean = 8, sd = 2),
                                           gm$log_sigma[[2]]))
stopifnot(identical(q$k_er, c(median = 2, sd = 0.3)), q$rrf_effect_sd$family == "normal")
expect_error_like(compartmental_priors(k_er = c(-1, 0.1), k_p = gm$k_p, rlr = gm$rlr,
                                       rrf = gm$rrf, rlr_effect_sd = gm$rlr_effect_sd,
                                       rrf_effect_sd = gm$rrf_effect_sd, lkj_eta = 1,
                                       log_sigma = gm$log_sigma),
                  "k_er must be c(median = , sd = )")
expect_error_like(compartmental_priors(k_er = gm$k_er, k_p = gm$k_p, rlr = gm$rlr,
                                       rrf = gm$rrf, rlr_effect_sd = list(family = "cauchy"),
                                       rrf_effect_sd = gm$rrf_effect_sd, lkj_eta = 1,
                                       log_sigma = gm$log_sigma),
                  "rlr_effect_sd must be")
expect_error_like(compartmental_priors(k_er = gm$k_er, k_p = gm$k_p, rlr = gm$rlr,
                                       rrf = gm$rrf, rlr_effect_sd = gm$rlr_effect_sd,
                                       rrf_effect_sd = gm$rrf_effect_sd, lkj_eta = 1,
                                       log_sigma = gm$log_sigma[1]),
                  "log_sigma must be a list of two priors")

# Priors are required.
expect_error_like(compartmental(tri, "paid", "premium", incurred = "incurred"),
                  "priors are required")
expect_error_like(compartmental(tri, "paid", "premium", list(), incurred = "incurred"),
                  "priors must be a compartmental_priors list")

# The other settings are named but not yet built.
for (setting in list(list(fitted = "paid_and_incurred"),
                     list(fitted = "paid_only"), list(fitted = "incurred_only"))) {
  expect_error_like(do.call(compartmental, c(list(tri, "paid", "premium", gm,
                                                  incurred = "incurred"), setting)),
                    "not yet built")
}
expect_error_like(compartmental(tri, "paid", "premium", gm), "give exactly one of outstanding")
expect_error_like(compartmental(tri, "paid", "premium", gm, outstanding = "incurred",
                                incurred = "incurred"),
                  "give exactly one of outstanding")
expect_error_like(compartmental(tri, "paid", "premium", gm, incurred = "incurred",
                                rate_order = "sorted"),
                  "'arg' should be one of")
bad_eta <- gm
bad_eta$lkj_eta <- 0
expect_error_like(compartmental(tri, "paid", "premium", bad_eta, incurred = "incurred"), "lkj_eta")

# The prior predictive check: mean paths, replayed by seed.
paths <- compartmental_prior_predictive(tri, "premium", gm, n_sims = 500, seed = 4)
again <- compartmental_prior_predictive(tri, "premium", gm, n_sims = 500, seed = 4)
stopifnot(
  identical(names(paths), c("outstanding", "paid")),
  S7::S7_inherits(paths$paid, predictive_distribution),
  identical(paths$paid@dims, c("origin", "age")),
  paths$paid@n_sims == 500, nrow(paths$outstanding@keys) == 100,
  identical(draw_matrix(paths$paid), draw_matrix(again$paid))
)
paid_120 <- mean(marginal(paths$paid, list(origin = "1988", age = 120)))
stopifnot(paid_120 / 104437 > 0.3, paid_120 / 104437 < 0.9)

# The default chains (4 x 2,000 draws) pass every diagnostic at seed 0.
run <- with_warnings(compartmental(tri, "paid", "premium", gm, incurred = "incurred",
                                   n_sims = 2000))
fit <- run$value
stopifnot(
  length(run$warnings) == 0,
  S7::S7_inherits(fit, compartmental_fit),
  identical(fit@origins, as.character(1988:1997)),
  identical(fit@population_origins, "1997"),
  fit@premium[1] == 104437, fit@latest_paid[10] == 0,
  fit@rate_swaps == 0L,
  length(fit@ultimate_loss_ratio) == 10,
  fit@diagnostics$divergences == 0, length(fit@diagnostics$warnings) == 0
)
r <- fit@reserves
stopifnot(
  S7::S7_inherits(r, predictive_distribution), r@n_sims == 2000,
  identical(r@dims, "origin"),
  identical(fit@paid@dims, c("origin", "age")),
  nrow(fit@paid@keys) == 55, nrow(fit@outstanding@keys) == 55,
  provenance(r)$model == "compartmental_gm1"
)
# Seed 0 at the defaults: 173,390 (Stan 173,341).
near(mean(r), 173390, 0.02)

d <- fit@draws
stopifnot(
  nrow(d) == 8000,
  identical(colnames(d)[1:6], c("k_er", "k_p", "RLR", "RRF", "sigma_os", "sigma_pd")),
  "b_ker" %in% colnames(d), colnames(d)[ncol(d)] == "log_sigma_pd"
)
s <- summary(fit)
stopifnot(identical(names(s), c("parameter", "mean", "sd", "q05", "q50", "q95", "rhat",
                                "ess_bulk", "ess_tail")))
stopifnot(abs(s$mean[s$parameter == "b_kp"] - (-8.488)) < 0.1)  # Stan -8.488 (0.358)

frame <- as.data.frame(fit)
stopifnot(
  identical(names(frame), c("origin", "premium", "latest_paid", "mean", "std_dev",
                            "ultimate_loss_ratio")),
  nrow(frame) == 10, all(frame$mean > 0)
)
near(sum(frame$mean), mean(r), 1e-9)
near(totals_frame(fit)$mean, mean(r), 1e-9)
for (cells in list(fit@paid_table, fit@outstanding_table)) {
  stopifnot(
    identical(names(cells), c("origin", "development", "mean", "std_dev", "q05", "q50", "q95")),
    nrow(cells) == 55, all(cells$q05 <= cells$q50), all(cells$q50 <= cells$q95)
  )
}
expect_error_like(development_frame(fit), "has no development table")
stopifnot(identical(as.data.frame(segment(fit)), frame))

loo <- bayes_loo(fit)
stopifnot(
  identical(names(loo$estimates), c("elpd", "se", "p", "ic")),
  length(loo$pointwise) == 90, loo$n_outstanding == 45,
  abs(loo$estimates[["elpd"]] - (-811.0)) < 2  # Stan elpd_loo -811.0
)
near(loo$elpd_outstanding + loo$elpd_paid, loo$estimates[["elpd"]], 1e-12)
invisible(utils::capture.output(print(fit)))

# One seed replays a fit exactly; another differs.
small <- function(seed) {
  suppressWarnings(compartmental(tri, "paid", "premium", gm, incurred = "incurred", chains = 2,
                                 tune = 100, draws = 100, n_sims = 200, seed = seed))
}
a <- small(7)
b <- small(7)
c8 <- small(8)
stopifnot(
  identical(a@draws, b@draws),
  identical(draw_matrix(a@reserves), draw_matrix(b@reserves)),
  !identical(a@draws, c8@draws)
)

# Failed diagnostics raise one warning and stay in the fit.
run <- with_warnings(compartmental(tri, "paid", "premium", gm, incurred = "incurred", chains = 2,
                                   tune = 20, draws = 20, n_sims = 100))
stopifnot(
  length(run$warnings) == 1,
  startsWith(run$warnings, "compartmental: the MCMC diagnostics failed"),
  length(run$value@diagnostics$warnings) > 0
)

# Model 2 (PR C): the log-normal likelihood on loss ratios with incremental
# paid, and the hierarchy of gesmann_morris_2_priors().
stopifnot(is.na(fit@cell_rate_swaps))  # model 1: the rates are shared
gm2 <- gesmann_morris_2_priors()
half <- function(scale) list(family = "student_t", nu = 10, scale = scale)
stopifnot(
  inherits(gm2, "compartmental_priors"),
  identical(gm2$k_er, gm$k_er), identical(gm2$rrf, gm$rrf),
  identical(gm2$rlr_effect_sd, half(0.7)), identical(gm2$rrf_effect_sd, half(0.5)),
  identical(gm2$ker_effect_sd, half(0.3)), identical(gm2$kp_effect_sd, half(0.3)),
  identical(gm2$development_effect_sd,
            list(k_er = half(0.3), k_p = half(0.3), rlr = half(0.7), rrf = half(0.5))),
  identical(gm2$log_sigma[[1]], list(family = "normal", mean = log(0.2), sd = 0.2)),
  is.null(gm$ker_effect_sd), is.null(gm$kp_effect_sd), is.null(gm$development_effect_sd)
)
# A development effect on some parameters only; the others have none.
some <- compartmental_priors(k_er = gm$k_er, k_p = gm$k_p, rlr = gm$rlr, rrf = gm$rrf,
                             rlr_effect_sd = gm$rlr_effect_sd,
                             rrf_effect_sd = gm$rrf_effect_sd, lkj_eta = 1,
                             log_sigma = gm$log_sigma,
                             development_effect_sd = list(rlr = list(family = "normal",
                                                                     scale = 0.5)))
stopifnot(identical(names(some$development_effect_sd), "rlr"), is.null(some$kp_effect_sd))
expect_error_like(compartmental_priors(k_er = gm$k_er, k_p = gm$k_p, rlr = gm$rlr,
                                       rrf = gm$rrf, rlr_effect_sd = gm$rlr_effect_sd,
                                       rrf_effect_sd = gm$rrf_effect_sd, lkj_eta = 1,
                                       log_sigma = gm$log_sigma, ker_effect_sd = 0.3),
                  "ker_effect_sd must be")
expect_error_like(compartmental_priors(k_er = gm$k_er, k_p = gm$k_p, rlr = gm$rlr,
                                       rrf = gm$rrf, rlr_effect_sd = gm$rlr_effect_sd,
                                       rrf_effect_sd = gm$rrf_effect_sd, lkj_eta = 1,
                                       log_sigma = gm$log_sigma,
                                       development_effect_sd = half(0.3)),
                  "development_effect_sd must be NULL or a named list")
expect_error_like(compartmental_priors(k_er = gm$k_er, k_p = gm$k_p, rlr = gm$rlr,
                                       rrf = gm$rrf, rlr_effect_sd = gm$rlr_effect_sd,
                                       rrf_effect_sd = gm$rrf_effect_sd, lkj_eta = 1,
                                       log_sigma = gm$log_sigma,
                                       development_effect_sd = list(k_p = list(family = "t"))),
                  "development_effect_sd$k_p must be")
expect_error_like(compartmental_model_2(tri, "paid", "premium", incurred = "incurred"),
                  "gesmann_morris_2_priors()")

# Shorter chains than compartmental_model_2()'s (4 x 2,000 warm-up and
# 8,000 draws, about a minute, warning-free at seed 0; the Rust validation
# runs them): about 20 seconds. They may miss an ESS threshold.
fit2 <- suppressWarnings(compartmental_model_2(tri, "paid", "premium", gm2,
                                               incurred = "incurred", tune = 1000,
                                               draws = 2000, n_sims = 2000))
nm <- colnames(fit2@draws)
stopifnot(
  S7::S7_inherits(fit2, compartmental_fit),
  identical(fit2@population_origins, "1997"),
  provenance(fit2@reserves)$model == "compartmental_gm2",
  all(paste0("k_p[", 1988:1996, "]") %in% nm),
  all(paste0("dev_RLR[", seq(12, 108, 12), "]") %in% nm),
  !"dev_kp[120]" %in% nm,
  all(c("tau_ker", "tau_kp", "tau_dev_ker", "tau_dev_kp", "tau_dev_RLR", "tau_dev_RRF") %in% nm),
  is.double(fit2@cell_rate_swaps), fit2@cell_rate_swaps >= 0, fit2@cell_rate_swaps <= 1
)
# Stan (4 x 25,000 draws): total reserve 152,242, b_kp -5.064 (sd 1.722),
# elpd_loo 204.29 on the loss ratios.
near(mean(fit2@reserves), 152242, 0.03)
s2 <- summary(fit2)
stopifnot(abs(s2$mean[s2$parameter == "b_kp"] - (-5.064)) < 0.5)
loo2 <- bayes_loo(fit2)
stopifnot(abs(loo2$estimates[["elpd"]] - 204.29) < 3, length(loo2$pointwise) == 90)
stopifnot(nrow(as.data.frame(fit2)) == 10, nrow(fit2@paid_table) == 55)

# Every combination of likelihood, scale and paid response is built.
for (setting in list(list(scale = "loss_ratio"), list(paid_response = "incremental"),
                     list(likelihood = "lognormal", scale = "loss_ratio"))) {
  priors <- if (is.null(setting$likelihood)) gm else {
    p <- gm
    p$log_sigma <- gm2$log_sigma
    p
  }
  mixed <- suppressWarnings(do.call(compartmental, c(
    list(tri, "paid", "premium", priors, incurred = "incurred", chains = 2, tune = 100,
         draws = 100, n_sims = 200), setting)))
  stopifnot(all(is.finite(draw_matrix(mixed@reserves)[1, ])), is.na(mixed@cell_rate_swaps))
}

# The log-normal needs positive responses; incremental paid the paid before.
small_long <- data.frame(year = rep(1988:1990, 3:1), age = c(12, 24, 36, 12, 24, 12),
                         paid = c(9.6, 22.8, 33.3, 7.9, 19.5, 8.7),
                         incurred = c(9.6, 64.0, 65.6, 62.1, 56.7, 64.0),
                         premium = rep(c(104.4, 88.9, 86.0), 3:1))
zero <- triangle(small_long, "year", "age", c("paid", "incurred", "premium"))
expect_error_like(compartmental_model_2(zero, "paid", "premium", gm2, incurred = "incurred",
                                        chains = 1, tune = 10, draws = 10, n_sims = 10),
                  "origin 1988 at development age 12 months has outstanding 0")
gap_long <- small_long
gap_long$incurred[1] <- 62.7
gap_long$paid[1] <- NA
gap <- triangle(gap_long, "year", "age", c("paid", "incurred", "premium"))
expect_error_like(compartmental_model_2(gap, "paid", "premium", gm2, outstanding = "incurred",
                                        chains = 1, tune = 10, draws = 10, n_sims = 10),
                  "incremental paid needs the paid of the age before")

# The prior predictive check with development effects, replayed by seed.
p2 <- compartmental_prior_predictive(tri, "premium", gm2, n_sims = 300, seed = 4,
                                     paid_response = "incremental")
again2 <- compartmental_prior_predictive(tri, "premium", gm2, n_sims = 300, seed = 4,
                                         paid_response = "incremental")
stopifnot(
  p2$paid@n_sims == 300, nrow(p2$outstanding@keys) == 100,
  identical(draw_matrix(p2$paid), draw_matrix(again2$paid))
)
paid2_120 <- mean(marginal(p2$paid, list(origin = "1988", age = 120)))
stopifnot(paid2_120 / 104437 > 0.2, paid2_120 / 104437 < 1.5)

# The one-year view (docs/design/reserving-bayes.md, decision 20): the
# lifetime fit is compartmental()'s, the CDR by premium origin; with
# reweighting only, nothing is matched or refitted. The checks against
# refits are in validation/tests/reserving_compartmental_one_year.rs.
view <- suppressWarnings(compartmental_one_year(tri, "paid", "premium", gm,
                                                incurred = "incurred", n_sims = 2000,
                                                refit = "reweight_only"))
lifetime <- suppressWarnings(compartmental(tri, "paid", "premium", gm, incurred = "incurred",
                                           n_sims = 2000))
stopifnot(
  S7::S7_inherits(view, compartmental_one_year_fit),
  S7::S7_inherits(view@fit, compartmental_fit),
  identical(draw_matrix(view@fit@reserves), draw_matrix(lifetime@reserves)),
  identical(view@origins, as.character(1988:1997)),
  identical(names(view@opening_reserve), view@origins),
  isTRUE(all.equal(unname(view@latest_paid + view@opening_reserve),
                   unname(view@opening_ultimate))),
  identical(view@cdr@dims, "origin"),
  length(view@pareto_k) == 2000,
  identical(view@refitted, integer()),
  identical(view@moment_matched, integer()),
  identical(view@unrefitted, integer()),
  view@flagged_share > 0 && view@flagged_share < 1,
  variance(view@cdr) < variance(view@fit@reserves),
  identical(names(as.data.frame(view)),
            c("origin", "premium", "latest_paid", "opening_ultimate", "opening_reserve",
              "cdr_mean", "cdr_std_dev")),
  "flagged_share" %in% names(totals_frame(view))
)
out <- capture.output(print(view))
stopifnot(grepl("<compartmental_one_year_fit> 2000 simulations", out[1], fixed = TRUE))
# Moment matching without refits: the flagged simulations are matched or
# left unrefitted, with a warning for the latter.
budget <- with_warnings(compartmental_one_year(tri, "paid", "premium", gm,
                                               incurred = "incurred", n_sims = 40,
                                               max_refits = 0))
flagged <- which(is.nan(budget$value@pareto_k) | budget$value@pareto_k > 0.7)
stopifnot(
  identical(budget$value@refitted, integer()),
  identical(sort(c(budget$value@moment_matched, budget$value@unrefitted)), flagged),
  length(budget$value@unrefitted) == 0 ||
    any(grepl("beyond max_refits", budget$warnings, fixed = TRUE))
)
expect_error_like(development_frame(view), "development table")
expect_error_like(compartmental_one_year(tri, "paid", "premium", gm, n_sims = 10),
                  "exactly one")

# The user's decisions of 2026-10-16: at most 2,000 one-year simulations
# by default (one_year_sims sets the count; a smaller count gives the first
# simulations of a larger one), and model 2's default view reweights only,
# with a warning that it is approximate.
capped <- suppressWarnings(compartmental_one_year(tri, "paid", "premium", gm,
                                                  incurred = "incurred",
                                                  refit = "reweight_only"))
first <- suppressWarnings(compartmental_one_year(tri, "paid", "premium", gm,
                                                 incurred = "incurred",
                                                 refit = "reweight_only",
                                                 one_year_sims = 300))
stopifnot(
  length(capped@pareto_k) == 2000,
  identical(first@pareto_k, capped@pareto_k[1:300]),
  is.null(capped@approximate)
)
expect_error_like(compartmental_one_year(tri, "paid", "premium", gm, incurred = "incurred",
                                         one_year_sims = 0), "n_sims")
# Model 2 with short chains (they may miss a diagnostic threshold).
two <- with_warnings(compartmental_model_2_one_year(tri, "paid", "premium", gm2,
                                                    incurred = "incurred", chains = 2,
                                                    tune = 200, draws = 200,
                                                    target_accept = 0.9,
                                                    one_year_sims = 50))
stopifnot(
  length(two$value@pareto_k) == 50,
  identical(two$value@refitted, integer()),
  identical(two$value@moment_matched, integer()),
  grepl("approximate", two$value@approximate, fixed = TRUE),
  any(grepl("approximate", two$warnings, fixed = TRUE))
)
# Refits stay a choice: an explicit mode carries no warning.
chosen <- with_warnings(compartmental_model_2_one_year(tri, "paid", "premium", gm2,
                                                       incurred = "incurred", chains = 2,
                                                       tune = 200, draws = 200,
                                                       target_accept = 0.9,
                                                       one_year_sims = 50,
                                                       refit = "reweight_only"))
stopifnot(
  is.null(chosen$value@approximate),
  !any(grepl("approximate", chosen$warnings, fixed = TRUE)),
  identical(chosen$value@pareto_k, two$value@pareto_k)
)
expect_error_like(compartmental_one_year(tri, "paid", "premium", gm, incurred = "incurred",
                                         refit = "sometimes"), "should be one of")

# A warm-started, shorter refit (decision 24) implies a refit: model 2
# refits its flagged simulations and is not approximate.
light <- with_warnings(compartmental_model_2_one_year(tri, "paid", "premium", gm2,
                                                      incurred = "incurred", chains = 2,
                                                      tune = 200, draws = 200,
                                                      target_accept = 0.9, one_year_sims = 4,
                                                      light_refit = c(2, 60, 60)))
stopifnot(
  is.null(light$value@approximate),
  !any(grepl("approximate", light$warnings, fixed = TRUE)),
  length(light$value@refitted) == 4
)
params <- provenance(light$value@cdr)$parameters
stopifnot(identical(params[["refit"]], "RefitFlagged"),
          identical(params[["light_refit"]], "2x60+60"))
default_light <- suppressWarnings(compartmental_model_2_one_year(
  tri, "paid", "premium", gm2, incurred = "incurred", chains = 2, tune = 200, draws = 200,
  target_accept = 0.9, one_year_sims = 2, light_refit = TRUE))
stopifnot(identical(provenance(default_light@cdr)$parameters[["light_refit"]], "4x150+250"))
expect_error_like(compartmental_one_year(tri, "paid", "premium", gm, incurred = "incurred",
                                         light_refit = c(2, 60)), "light_refit")
