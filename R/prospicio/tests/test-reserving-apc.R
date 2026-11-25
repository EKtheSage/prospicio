suppressMessages(library(prospicio))
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# The age-period-cohort reserving GLM and the separation method as a GLM
# (#979, docs/design/reserving-inflation.md, PR 7): the R bindings of
# prospicio_reserving::apc and Separation::glm, against
# validation/reference/reserving_apc_python.csv as
# validation/tests/reserving_apc.rs reads it.

ref <- "reserving_apc_python.csv"
want <- function(method, quantity, arg = "") {
  reference(ref, dataset = "genins", method = method, quantity = quantity, arg = arg)
}
genins <- triangle(utils::read.csv(validation("data", "genins.csv"), comment.char = "#"),
                   "origin", "development", "value")

# The prediction error rows are variances: by origin (arg the year) and in
# total (no arg). The first origin, with no future cell, has no row.
check_error <- function(error, method) {
  for (k in seq_len(nrow(error))[-1]) {
    arg <- if (error$origin[[k]] == "Total") "" else error$origin[[k]]
    near_unit(error$process_se[[k]]^2, want(method, "process_variance", arg), 1e-6)
    near_unit(error$parameter_se[[k]]^2, want(method, "parameter_variance", arg), 1e-6)
  }
}

# The canonical parameters, deviance and fitted values of each model (the
# rows without a rule, fitted with I0), then the reserves, future cells and
# prediction error under each rule.
codes <- c(apc = "APC", ac = "AC", ap = "AP", pc = "PC")
for (code in names(codes)) {
  method <- paste0("poisson_", code)
  fit <- apc_glm(genins, effects = tolower(codes[[code]]))
  stopifnot(S7::S7_inherits(fit, apc_glm_fit), fit@effects == codes[[code]],
            fit@extrapolation == "I0", S7::S7_inherits(fit@odp_fit, odp_glm_fit))
  near(fit@deviance, want(method, "deviance"), 1e-6)
  near(fit@level, want(method, "level"), 1e-6)
  near(fit@age_slope, want(method, "slope_age"), 1e-6)
  near(fit@accident_slope, want(method, "slope_cohort"), 1e-6)
  dds <- list(dd_age = fit@age_second_differences, dd_cohort = fit@accident_second_differences,
              dd_period = fit@calendar_second_differences)
  has <- c(dd_age = code != "pc", dd_cohort = code != "ap", dd_period = code != "ac")
  for (q in names(dds)) {
    stopifnot(length(dds[[q]]) == if (has[[q]]) 8L else 0L)
    for (l in seq_along(dds[[q]])) near(dds[[q]][[l]], want(method, q, l + 1L), 1e-6)
  }
  for (i in seq_along(fit@origins)) {
    for (j in seq_along(fit@ages)) {
      cell <- fit@fitted[i, j]
      if (i + j <= 11L) {
        arg <- sprintf("origin=%s;age=%s", fit@origins[[i]], fit@ages[[j]])
        near(cell, want(method, "fitted", arg), 1e-6)
      } else {
        stopifnot(is.na(cell))
      }
    }
  }
  # The canonical coefficients are the fields above, by name.
  coefs <- fit@coefficients
  near(coefs[c("level", "slope_age", "slope_accident")],
       c(fit@level, fit@age_slope, fit@accident_slope), 1e-12)
  stopifnot(identical(dim(fit@covariance), c(length(coefs), length(coefs))),
            identical(rownames(fit@covariance), names(coefs)), fit@dispersion > 0)
  if (has[["dd_period"]]) near(coefs[["dd_calendar[2]"]], fit@calendar_second_differences[[1]], 1e-12)

  rules <- if (code == "ac") "" else c("i2", "i1", "i0")
  for (rule in rules) {
    method <- if (rule == "") "poisson_ac" else sprintf("poisson_%s_%s", code, rule)
    fit <- apc_glm(genins, effects = codes[[code]], extrapolation = if (rule == "") "I0" else rule)
    stopifnot(fit@extrapolation == if (rule == "") "I0" else toupper(rule))
    for (k in seq_along(fit@origins)) {
      near_unit(fit@reserves[[k]], want(method, "reserve", fit@origins[[k]]), 1e-6)
    }
    near(fit@total_reserve, want(method, "total_reserve"), 1e-6)
    cells <- fit@future@cells
    stopifnot(nrow(cells) == 45L, S7::S7_inherits(fit@future, future_cells))
    for (k in seq_len(nrow(cells))) {
      arg <- sprintf("origin=%s;age=%s", cells$origin[[k]], cells$age[[k]])
      near(cells$value[[k]], want(method, "future", arg), 1e-6)
    }
    near(unname(fit@future@reserves), unname(fit@reserves), 1e-12)
    check_error(fit@prediction_error, method)
    near(fit@prediction_error$reserve, c(fit@reserves, fit@total_reserve), 1e-12)
    if (rule %in% c("i2", "")) {
      stopifnot(identical(fit@future_calendar_second_differences, c(0, 0)))
    } else {
      stopifnot(length(fit@future_calendar_second_differences) == 2L,
                any(fit@future_calendar_second_differences != 0))
    }
  }
}

# R apc's documented I0 total on Taylor-Ashe, 19,061,428.
i0 <- apc_glm(genins)
near_abs(i0@total_reserve, want("poisson_apc_i0", "documented_total"), 1.0)
near_abs(i0@total_reserve, 19061428, 1.0)
stopifnot(i0@reserves[[1]] == 0, i0@effects == "APC", i0@extrapolation == "I0")
# The ODP fit underneath carries the prediction error and the draws.
near(i0@odp_fit@total_reserve, i0@total_reserve, 1e-12)
draws <- predict_distribution(i0@odp_fit, n_sims = 200, seed = 1)
stopifnot(length(total(draws)@draws) == 200L)

# The accident and development model is the chain ladder.
near(apc_glm(genins, effects = "AC")@reserves, odp_glm(genins)@reserves, 1e-8)

# The separation method as a GLM: the arithmetic method's reserves, and
# statsmodels' prediction error with a 5% rate and a 4 period trend.
for (case in list(list("separation_rate", list(future_rate = 0.05)),
                  list("separation_trend", list(trend_periods = 4)))) {
  glm <- do.call(separation_glm, c(list(genins), case[[2]]))
  plain <- do.call(separation, c(list(genins), case[[2]]))
  stopifnot(S7::S7_inherits(glm, odp_glm_fit))
  near_unit(glm@reserves, plain@reserves, 1e-8)
  near(glm@total_reserve, want(case[[1]], "total_reserve"), 1e-6)
  check_error(glm@prediction_error, case[[1]])
}
cpi <- calendar_index("2001", 1.05^(0:9), future_rate = 0.05)
near(separation_glm(genins, future_index = cpi)@total_reserve,
     separation(genins, future_index = cpi)@total_reserve, 1e-8)
expect_error_like(separation_glm(genins, normaliser = rep(2, 10), future_rate = 0.05), "normaliser")
expect_error_like(separation_glm(genins), "future")
expect_error_like(separation_glm(genins, future_rate = 0.05, trend_periods = 3), "only one")

# The settings are checked.
expect_error_like(apc_glm(genins, effects = "APCX"), "effects must be")
expect_error_like(apc_glm(genins, effects = c("AP", "AC")), "effects must be")
expect_error_like(apc_glm(genins, extrapolation = "I3"), "extrapolation must be")
expect_error_like(apc_glm(genins, family = "binomial"), "family must be")
expect_error_like(apc_glm(genins, family = "tweedie"), "needs power")
expect_error_like(apc_glm(genins, power = 1.5), "only for family")
expect_error_like(apc_glm(genins, tolerance = -1), "tolerance")
stopifnot(apc_glm(genins, extrapolation = "i1")@extrapolation == "I1")
gamma <- apc_glm(genins, family = "gamma")
stopifnot(is.finite(gamma@total_reserve), abs(gamma@total_reserve - i0@total_reserve) > 1)

# I0 needs 4 origins; 3 are enough for I1 in a sub-model (the full model
# would have no residual degrees of freedom). Not a full triangle fails.
three <- triangle(
  data.frame(year = c(2020, 2020, 2020, 2021, 2021, 2022), age = c(12, 24, 36, 12, 24, 12),
             paid = c(100, 60, 20, 110, 70, 120)),
  "year", "age", "paid", cumulative = FALSE
)
expect_error_like(apc_glm(three), "at least 4")
expect_error_like(apc_glm(three, effects = "AP"), "at least 4")
expect_error_like(apc_glm(three, extrapolation = "I1"), "degrees of freedom")
stopifnot(length(apc_glm(three, effects = "AP", extrapolation = "I1")@reserves) == 3L)
not_full <- triangle(
  data.frame(year = c(2020, 2020, 2020, 2021), age = c(12, 24, 36, 12),
             paid = c(100, 150, 170, 110)),
  "year", "age", "paid"
)
expect_error_like(apc_glm(not_full), "full triangle")
