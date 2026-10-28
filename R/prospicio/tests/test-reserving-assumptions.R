suppressMessages(library(prospicio))

# Mack's calendar-year effect and development-factor correlation tests
# (Mack 1994, appendices H and G; docs/design/reserving-v02.md, decision
# 27). The Rust core is checked against R ChainLadder and chainladder-python
# in validation/tests/reserving_assumptions.rs; these tests check the
# binding's arguments, results, print, summary and errors against the same
# R ChainLadder reference file.

near <- function(a, b, rel = 1e-12, abs = 1e-15) {
  a <- unname(a)
  b <- unname(b)
  if (length(a) != length(b) || !all(abs(a - b) <= pmax(rel * abs(b), abs))) {
    stop(sprintf("got %s, want %s", paste(format(a, digits = 17), collapse = " "),
                 paste(format(b, digits = 17), collapse = " ")), call. = FALSE)
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

# Tests run from the repository root (CI and `cargo xtask r`).
validation <- function(...) file.path("validation", ...)
load_triangle <- function(name) {
  long <- utils::read.csv(validation("data", paste0(name, ".csv")), comment.char = "#")
  triangle(long, "origin", "development", "value")
}
lines <- grep("^#", readLines(validation("reference", "reserving_assumptions_r.csv")),
              value = TRUE, invert = TRUE)
# The last column, `source`, holds commas: keep the first seven fields.
fields <- lapply(strsplit(lines[-1], ",", fixed = TRUE), function(f) c(f, "")[1:7])
ref <- stats::setNames(as.data.frame(do.call(rbind, fields), stringsAsFactors = FALSE),
                       strsplit(lines[1], ",", fixed = TRUE)[[1]][1:7])
ref$expected <- as.numeric(ref$expected)
want <- function(dataset, test, quantity, arg = "") {
  rows <- ref[ref$dataset == dataset & ref$test == test & ref$quantity == quantity &
                ref$arg == arg, ]
  if (nrow(rows) != 1) stop(sprintf("no reference row %s %s %s %s", dataset, test, quantity, arg))
  rows$expected
}

# Against R ChainLadder's cyEffTest and dfCorTest on RAA, GenIns and ABC.
for (name in c("raa", "genins", "abc")) {
  tri <- load_triangle(name)
  for (ci in c(0.95, 0.9)) {
    cy <- if (ci == 0.95) calendar_year_effect(tri) else calendar_year_effect(tri, ci = ci)
    test <- sprintf("calendar_year_effect;ci=%s", ci)
    stopifnot(S7::S7_inherits(cy, calendar_year_test), cy@ci == ci)
    near(cy@z, want(name, test, "total_z"))
    near(cy@expected, want(name, test, "total_expected"))
    near(cy@variance, want(name, test, "total_variance"))
    near(cy@range, c(want(name, test, "lower"), want(name, test, "upper")))
    stopifnot(cy@passes == as.logical(want(name, test, "passes")))
    tab <- cy@test_table
    stopifnot(nrow(tab) == want(name, test, "n_diagonals"))
    for (r in seq_len(nrow(tab))) {
      year <- substr(tab$valuation[r], 1, 4)
      columns <- c(S_j = "small", L_j = "large", Z_j = "z", n = "n", m = "m",
                   E_Zj = "expected", Var_Zj = "variance")
      for (col in names(columns)) near(tab[[col]][r], want(name, test, columns[[col]], year))
    }
  }
  for (ci in c(0.5, 0.75)) {
    dc <- development_correlation(tri, ci = ci)
    test <- sprintf("development_correlation;ci=%s", ci)
    stopifnot(S7::S7_inherits(dc, development_correlation_test), dc@ci == ci)
    near(dc@t, want(name, test, "t"))
    near(dc@variance, want(name, test, "variance"))
    near(dc@range, c(want(name, test, "lower"), want(name, test, "upper")))
    stopifnot(dc@passes == as.logical(want(name, test, "passes")))
    pairs <- dc@pairs
    stopifnot(nrow(pairs) == want(name, test, "n_pairs"))
    for (i in seq_len(nrow(pairs))) {
      near(pairs$T_k[i], want(name, test, "t_k", as.character(i - 1)))
      near(pairs$weight[i], want(name, test, "weight", as.character(i - 1)))
    }
    stopifnot(all(pairs$n == pairs$weight + 1), all(pairs$age == 12 * seq_len(nrow(pairs))))
  }
}

# Mack's printed values, on RAA: T = .070, Z = 14, E(Z) = 12.875,
# Var(Z) = 3.9785; and a Mack fit tests the same link ratios.
raa <- load_triangle("raa")
fit <- mack(raa)
dc <- development_correlation(fit)
stopifnot(abs(dc@t - 0.070) < 5e-4, abs(dc@variance - 1 / 28) < 1e-15, dc@ci == 0.5)
cy <- calendar_year_effect(fit)
stopifnot(cy@z == 14, cy@expected == 12.875, abs(cy@variance - 3.9785) < 5e-5, cy@ci == 0.95)
stopifnot(identical(cy@test_table, calendar_year_effect(raa)@test_table))

# Exclusions: the fit's tests read only its kept link ratios.
kept <- calendar_year_effect(mack(raa, n_periods = 4))
stopifnot(identical(kept@test_table$valuation, c("1987-12", "1988-12", "1989-12", "1990-12")))
dropped <- mack(raa, drop_high = 1, drop_low = 1)
stopifnot(sum(calendar_year_effect(dropped)@test_table$n) < sum(cy@test_table$n))
stopifnot(all(development_correlation(dropped)@pairs$n >= 2))

# print and summary, shaped as R ChainLadder's.
out <- capture.output(print(cy))
stopifnot(any(grepl("Z = 14, E[Z] = 12.875", out, fixed = TRUE)),
          any(grepl("95%-range", out, fixed = TRUE)),
          any(grepl("Calendar year effect: not detected", out, fixed = TRUE)))
s <- summary(cy)
stopifnot(identical(names(s), c("Table", "Totals", "Range")),
          identical(rownames(s$Totals), c("Z", "E[Z]", "Var[Z]")),
          s$Totals["Z", "Totals"] == 14,
          identical(rownames(s$Range), c("Lower", "Upper")))
out <- capture.output(print(dc))
stopifnot(any(grepl("50%-range", out, fixed = TRUE)),
          any(grepl("Development factor correlation: not detected", out, fixed = TRUE)))
s <- summary(dc)
stopifnot(identical(names(s), c("Results", "Range", "Table")),
          identical(rownames(s$Results), c("T", "E[T]", "Var[T]")),
          s$Results["E[T]", "Value"] == 0)
abc <- calendar_year_effect(load_triangle("abc"))
stopifnot(!abc@passes,
          any(grepl("Calendar year effect: detected", capture.output(print(abc)), fixed = TRUE)))

# Errors.
expect_error_like(calendar_year_effect(raa, ci = 1), "ci")
expect_error_like(development_correlation(raa, ci = 0), "ci")
expect_error_like(calendar_year_effect(raa, ci = c(0.5, 0.9)), "ci must be a single number")
expect_error_like(calendar_year_effect(raa, column = "missing"), "missing")
expect_error_like(calendar_year_effect(42), "x must be a triangle or a mack_fit")
raa_long <- utils::read.csv(validation("data", "raa.csv"), comment.char = "#")
keyed <- triangle(rbind(cbind(raa_long, lob = "a"), cbind(raa_long, lob = "b")),
                  "origin", "development", "value", keys = "lob")
expect_error_like(calendar_year_effect(mack(keyed)), "segment")
expect_error_like(development_correlation(mack(keyed)), "segment")
