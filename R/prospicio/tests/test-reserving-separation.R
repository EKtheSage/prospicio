suppressMessages(library(prospicio))
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# Taylor's separation method (#979, docs/design/reserving-inflation.md,
# PR 5): the R bindings of prospicio_reserving::separation, with the cases
# of python/tests/test_reserving_separation.py and
# validation/tests/reserving_separation.rs.

ref <- "reserving_separation_statsmodels.csv"
want <- function(name, method, quantity, arg = "") {
  reference(ref, dataset = name, method = method, quantity = quantity, arg = arg)
}
load_dataset <- function(name) {
  long <- utils::read.csv(validation("data", paste0(name, ".csv")), comment.char = "#")
  triangle(long, "origin", "development", "value")
}

# The fit against statsmodels: proportions, indices, reserves, totals and the
# geometric methods' residual variance, on GenIns and ABC.
for (name in c("genins", "abc")) {
  tri <- load_dataset(name)
  for (method in c("arithmetic", "geometric", "geometric_lognormal")) {
    fit <- separation(tri, kind = method, future_rate = 0.05)
    stopifnot(S7::S7_inherits(fit, separation_fit))
    for (k in seq_along(fit@ages)) {
      near(fit@proportions[[k]], want(name, method, "proportion", fit@ages[[k]]), 1e-9)
    }
    for (k in seq_along(fit@calendar)) {
      near(fit@indices[[k]], want(name, method, "index", fit@calendar[[k]]), 1e-9)
    }
    for (k in seq_along(fit@origins)) {
      near_unit(fit@reserves[[k]], want(name, method, "reserve", fit@origins[[k]]), 1e-9)
    }
    near(fit@total_reserve, want(name, method, "total_reserve"), 1e-9)
    near(sum(fit@proportions), 1, 1e-12)
    if (method == "arithmetic") {
      stopifnot(is.null(fit@log_variance))
    } else {
      near(fit@log_variance, want(name, method, "log_variance"), 1e-9)
    }
  }
}

# The fitted and future cells.
genins <- load_dataset("genins")
for (method in c("arithmetic", "geometric", "geometric_lognormal")) {
  fit <- separation(genins, kind = method, future_rate = 0.05)
  n <- length(fit@origins)
  stopifnot(identical(dim(fit@fitted), c(n, n)))
  for (i in seq_len(n)) {
    for (j in seq_len(n)) {
      cell <- fit@fitted[i, j]
      if (i + j <= n + 1L) {
        arg <- sprintf("origin=%s.0;age=%s.0", fit@origins[[i]], fit@ages[[j]])
        near(cell, want("genins", method, "fitted", arg), 1e-9)
      } else {
        stopifnot(is.na(cell))
      }
    }
  }
  cells <- fit@future@cells
  stopifnot(nrow(cells) == n * (n - 1) / 2, S7::S7_inherits(fit@future, future_cells))
  for (k in seq_len(nrow(cells))) {
    arg <- sprintf("origin=%s;age=%s", cells$origin[[k]], cells$age[[k]])
    near(cells$value[[k]], want("genins", method, "future", arg), 1e-9)
  }
  near(unname(fit@future@reserves), unname(fit@reserves), 1e-12)
  stopifnot(fit@reserves[[1]] == 0)
  near(sum(fit@reserves), fit@total_reserve, 1e-12)
}

# The future indices continue at the rate; the other fields.
fit <- separation(genins, future_rate = 0.05)
stopifnot(fit@future_indices$period[[1]] == "2011", fit@calendar[[1]] == "2001",
          length(fit@calendar) == 10L, fit@origins[[1]] == "2001",
          identical(fit@ages[1:2], c(12L, 24L)), identical(fit@normaliser, rep(1, 10)))
near(fit@future_indices$index, fit@indices[[10]] * 1.05^seq_along(fit@future_indices$index), 1e-12)

# Taylor (1977), Example 1, as printed (4 decimals for r_j, 1 for lambda_h).
rows <- list(c(50.4, 78.6, 87.6, 92.4), c(58.0, 87.2, 96.9), c(59.5, 92.7), 66.2)
taylor_long <- data.frame(
  year = rep(2001:2004, lengths(rows)),
  age = 12 * unlist(lapply(rows, seq_along)),
  s = unlist(rows)
)
taylor <- separation(triangle(taylor_long, "year", "age", "s"), future_rate = 0.10)
near_abs(taylor@proportions, c(0.5835, 0.2878, 0.0866, 0.0421), 5e-5)
near_abs(taylor@indices, c(86.4, 98.9, 102.0, 113.9), 0.05)
printed <- rbind(c(50.4, 28.5, 8.8, 4.8), c(57.7, 29.4, 9.9, NA), c(59.5, 32.8, NA, NA),
                 c(66.5, NA, NA, NA))
stopifnot(identical(is.na(taylor@fitted), is.na(printed)))
near_abs(taylor@fitted[!is.na(printed)], printed[!is.na(printed)], 0.06)
near_abs(taylor@future@cells$value, c(5.3, 10.8, 5.8, 36.1, 11.9, 6.4), 0.06)

# A normaliser scales the indices and leaves the proportions.
plain <- separation(genins, future_rate = 0.05)
scaled <- separation(genins, normaliser = rep(2, 10), future_rate = 0.05)
stopifnot(identical(scaled@normaliser, rep(2, 10)))
near(scaled@proportions, plain@proportions, 1e-12)
near(plain@indices, 2 * scaled@indices, 1e-12)
near(plain@total_reserve, scaled@total_reserve, 1e-12)
expect_error_like(separation(genins, normaliser = c(1, 2), future_rate = 0.05), "normaliser")

# The future from a rate, an index or a trend.
cpi <- calendar_index("2001", 1.05^(0:9), future_rate = 0.05)
by_index <- separation(genins, future_index = cpi)
near(by_index@future_indices$index, plain@future_indices$index, 1e-9)
near(by_index@proportions, plain@proportions, 1e-14)
trended <- separation(genins, trend_periods = 3)
slope <- (log(trended@indices[[10]]) - log(trended@indices[[8]])) / 2
near(trended@future_indices$index,
     trended@indices[[10]] * exp(slope * seq_along(trended@future_indices$index)), 1e-9)
expect_error_like(separation(genins, trend_periods = 1), "trend")
expect_error_like(separation(genins, future_rate = -1), "rate")

# The settings are checked.
expect_error_like(separation(genins), "future")
expect_error_like(separation(genins, future_rate = 0.05, trend_periods = 3), "only one")
expect_error_like(separation(genins, kind = "harmonic", future_rate = 0.05), "kind")
expect_error_like(separation(genins, future_index = 3), "calendar_index")

# A triangle that is not full, a negative increment, too few origins.
not_full <- triangle(
  data.frame(year = c(2020, 2020, 2020, 2021), age = c(12, 24, 36, 12),
             paid = c(100, 150, 170, 110)),
  "year", "age", "paid"
)
expect_error_like(separation(not_full, future_rate = 0.05), "full triangle")
two_by_two <- function(second) {
  triangle(data.frame(year = c(2020, 2020, 2021), age = c(12, 24, 12), paid = c(100, second, 110)),
           "year", "age", "paid")
}
expect_error_like(separation(two_by_two(90), kind = "geometric", future_rate = 0.05), "positive")
two <- separation(two_by_two(150), kind = "geometric", future_rate = 0.05)
stopifnot(is.nan(two@log_variance))
expect_error_like(
  separation(two_by_two(150), kind = "geometric_lognormal", future_rate = 0.05), "3 origins"
)

# A segment is chosen by its key values.
seg_long <- data.frame(
  lob = rep(c("a", "b"), each = 3), year = rep(c(2020, 2020, 2021), 2),
  age = rep(c(12, 24, 12), 2), paid = c(100, 150, 110, 10, 25, 12)
)
by_lob <- triangle(seg_long, "year", "age", "paid", keys = "lob")
expect_error_like(separation(by_lob, future_rate = 0.05), "segment")
a <- separation(by_lob, future_rate = 0.05, segment = c(lob = "a"))
near(a@total_reserve, separation(two_by_two(150), future_rate = 0.05)@total_reserve, 1e-14)
b <- separation(by_lob, future_rate = 0.05, segment = list(lob = "b"))
stopifnot(abs(a@total_reserve - b@total_reserve) > 1e-6, identical(b@origins, c("2020", "2021")))
expect_error_like(separation(by_lob, future_rate = 0.05, segment = c(lob = "c")), "lob")
