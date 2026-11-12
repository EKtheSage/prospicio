suppressMessages(library(prospicio))
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# One condition class for a setting a reserving method does not support
# (#580, docs/design/reserving-v02.md decision 29): `prospicio_not_supported`,
# an error, signalled where the setting is chosen when it can be; a failure
# on the data is a plain error.

# The condition `expr` signals, which must be an error.
condition_of <- function(expr) {
  cond <- tryCatch({
    expr
    NULL
  }, error = function(e) e)
  if (is.null(cond)) stop("expected an error", call. = FALSE)
  cond
}

expect_not_supported <- function(expr, pattern) {
  cond <- condition_of(expr)
  msg <- conditionMessage(cond)
  if (!inherits(cond, "prospicio_not_supported") || !inherits(cond, "error")) {
    stop(sprintf("error %s has classes %s, not prospicio_not_supported", shQuote(msg),
                 toString(class(cond))), call. = FALSE)
  }
  if (!grepl(pattern, msg, fixed = TRUE)) {
    stop(sprintf("error %s does not mention %s", shQuote(msg), shQuote(pattern)), call. = FALSE)
  }
  if (grepl("[prospicio_not_supported]", msg, fixed = TRUE)) {
    stop(sprintf("error %s keeps the marker", shQuote(msg)), call. = FALSE)
  }
}

# Tests run from the repository root (CI and `cargo xtask r`).
long <- utils::read.csv(file.path("validation", "data", "raa.csv"), comment.char = "#")
raa <- triangle(long, "origin", "development", "value")

# The geometric average in Mack's model and in the one-year bootstrap.
expect_not_supported(mack(raa, average = "geometric"),
                     "the geometric average is not supported by Mack's model")
expect_not_supported(odp_one_year(raa, average = "geometric", n_sims = 10),
                     "the geometric average is not supported")

# Merz and Wuthrich's closed form: another alpha, a tail.
expect_not_supported(claims_development_result(mack(raa, average = "simple")),
                     "needs volume-weighted factors")
expect_not_supported(claims_development_result(mack(raa, tail = 1.05)),
                     "a tail factor is not supported by Merz and W")

# Synchronized Bayesian segments, refused before any sampling.
two <- triangle(rbind(cbind(long, line = "a"), cbind(long, line = "b")), "origin",
                "development", "value", keys = "line")
expect_not_supported(bayes_odp_segments(two, dependence = "synchronized"),
                     "Synchronized dependence is not supported")

# Compartmental: one response under free rates is refused from the settings,
# before the triangle's columns are read (raa has no premium or outstanding).
gm <- gesmann_morris_priors()
for (fitted in c("paid_only", "incurred_only")) {
  expect_not_supported(compartmental(raa, "value", "value", gm, outstanding = "value",
                                     fitted = fitted),
                       "needs rate_order Ordered")
}

# A failure on the data stays a plain error.
cond <- condition_of(odp_bootstrap(raa, n_sims = 0))
stopifnot(inherits(cond, "error"), !inherits(cond, "prospicio_not_supported"))
