suppressMessages(library(prospicio))

# odp_glm(): Taylor and McGuire, CAS Monograph 3, Table 5-4 (issue #333).

check <- function(ok, what) if (!isTRUE(ok)) stop(what, call. = FALSE)
within <- function(a, b, tol, what) {
  check(all(abs(a - b) <= tol), sprintf("%s: got %s, want %s", what,
                                         paste(format(a, digits = 10), collapse = " "),
                                         paste(b, collapse = " ")))
}

increments <- list(  # Table 1-1, accident years 1988 to 1997
  c(41821, 34729, 20147, 15965, 11285, 5924, 4775, 3742, 3435, 2958),
  c(48167, 39495, 24444, 18178, 10840, 7379, 5683, 4758, 3959),
  c(52058, 47459, 27359, 17916, 11448, 8846, 5869, 5391),
  c(57251, 49510, 27036, 20871, 14304, 10552, 7742),
  c(59213, 54129, 29566, 22484, 14114, 10000),
  c(59475, 52076, 26836, 22332, 14756),
  c(65607, 44648, 27062, 22655),
  c(56748, 39315, 26748),
  c(52212, 40030),
  c(43962)
)
long <- data.frame(
  year = rep(1988:1997, lengths(increments)),
  age = unlist(lapply(increments, function(r) 12 * seq_along(r))),
  paid = unlist(lapply(increments, cumsum))
)
tri <- triangle(long, "year", "age", "paid")
fit <- odp_glm(tri)

# Forecast and RMSEP for 1989 to 1997, then the total, as printed.
e <- fit@prediction_error
check(identical(e$origin[11], "Total"), "the last row is the total")
check(e$rmsep[1] == 0 && is.na(e$cv[1]), "the oldest year has no reserve")
within(e$reserve[-1], c(3398, 8155, 14579, 22645, 31865, 45753, 60093, 80983, 105874, 373346),
       0.5, "forecasts")
within(e$rmsep[-1], c(924, 1363, 1775, 2169, 2523, 3036, 3577, 4538, 6786, 14076), 0.5,
       "RMSEPs")
within(e$process_se[11], 6539, 0.5, "total process sd")
within(e$parameter_se[11], 12465, 0.5, "total parameter sd")
within(sqrt(sum(e$rmsep[1:10]^2)), 10275, 0.5, "years combined as if independent")

# The fit reproduces the chain ladder.
cl <- chain_ladder(tri)
within(fit@total_reserve, cl@total_reserve, 1e-6, "total reserve")
within(unname(fit@reserves), unname(cl@reserve), 1e-6, "reserves")
check(identical(names(fit@reserves), as.character(1988:1997)), "reserves are named by origin")
f <- fit@future
check(nrow(f) == 45 && identical(names(f), c("origin", "development", "mean", "predictive_mean")),
      "future cells")
within(sum(f$mean), fit@total_reserve, 1e-6, "future means add up to the reserve")
check(all(f$predictive_mean > f$mean), "normal draws lift each cell's mean")
k <- length(fit@coefficients)
check(k == 19 && identical(dim(fit@covariance), c(k, k)), "19 coefficients")
within(sqrt(diag(fit@covariance)), fit@std_errors, 1e-9, "standard errors")
check(fit@dispersion > 0, "dispersion")

# Joint draws of the future cells: mean-preserving by default.
pd <- predict_distribution(fit, n_sims = 4000, seed = 7)
totals <- total(pd)@draws
within(mean(totals), fit@total_reserve, 0.02 * fit@total_reserve, "mean of the draws")
within(stats::sd(totals), e$rmsep[11], 0.1 * e$rmsep[11], "sd of the draws")
fixed <- total(predict_distribution(fit, n_sims = 4000, seed = 7, parameters = "fixed"))@draws
check(stats::sd(fixed) < stats::sd(totals), "fixed parameters draw process noise only")

err <- tryCatch(predict_distribution(fit, data.frame(x = 1), n_sims = 10, seed = 1),
                error = conditionMessage)
check(grepl("no newdata", err), "newdata is rejected")
err <- tryCatch(predict_distribution(fit, n_sims = 10, seed = 1, parameters = "bogus"),
                error = conditionMessage)
check(grepl("should be one of", err), "an unknown parameters is rejected")
check(any(grepl("odp_glm_fit", capture.output(print(fit)))), "print")

# Residual diagnostics: Taylor and McGuire, Chapter 6.
r <- fit@residuals
check(nrow(r) == 55 && r$origin[1] == "1988" && r$development[1] == 12, "a row per observed cell")
check(r$calendar[1] == "1988" && r$calendar_index[55] == 9, "calendar periods")
check(identical(which(is.na(r$standardized_deviance)), c(10L, 55L)),
      "cells fitted exactly have no standardized residual")
within(sum(r$leverage), 19, 1e-6, "leverages add up to the parameters")
heat_map <- c(98, 100, 100, 104, 113, 87, 96, 92, 100, 100,
              99, 99, 106, 103, 95, 95, 99, 102, 100,
              96, 108, 107, 91, 90, 102, 92, 104,
              97, 103, 96, 97, 103, 111, 111,
              95, 107, 100, 100, 97, 100,
              98, 105, 93, 101, 104,
              109, 91, 95, 104,
              106, 90, 105,
              103, 97,
              100)
check(identical(round(100 * r$actual_to_fitted), heat_map), "Figure 6-22")
by_age <- residual_summary(fit)
spread <- by_age$sd[1:8] * sqrt((by_age$n[1:8] - 1) / by_age$n[1:8])
within(spread, c(1.34, 1.59, 0.88, 0.62, 0.83, 0.77, 0.65, 0.40), 0.005,
       "spread by development year")
check(by_age$label[1] == "12" && by_age$n[10] == 0 && is.na(by_age$mean[10]), "by age")
within(residual_summary(fit, "calendar")$mean,
       c(-0.38, -0.13, -0.40, 0.56, 0.42, -0.43, 0.23, -0.27, -0.26, 0.32), 0.005,
       "mean by calendar year")
check(residual_summary(fit, "origin")$label[1] == "1988", "by origin")
err <- tryCatch(residual_summary(fit, "diagonal"), error = conditionMessage)
check(grepl("should be one of", err), "an unknown grouping is rejected")
