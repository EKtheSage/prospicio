suppressMessages(library(prospicio))

# compartmental(), Gesmann and Morris' compartmental model 1
# (docs/design/reserving-compartmental.md). The parity with the paper's brms
# code in Stan and its printed posterior is
# validation/tests/reserving_compartmental.rs; these tests check the binding
# on company 337 workers' compensation as the papers fit it.

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
for (setting in list(list(likelihood = "lognormal"), list(scale = "loss_ratio"),
                     list(paid_response = "incremental"), list(fitted = "paid_and_incurred"),
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
