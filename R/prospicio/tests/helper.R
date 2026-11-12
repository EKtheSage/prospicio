# Helpers shared by the R test files (#845). Each test file sources this
# file first; `validation/scripts/run_r_tests.py` runs the files.
#
# with_package(pkg): TRUE when an optional package is installed. When it
# is not, it says so on a "SKIP" line and, under the runner, records the
# skip so the runner's summary lists it, so that a parity block guarded by
# an optional package never skips silently.
with_package <- function(pkg) {
  if (requireNamespace(pkg, quietly = TRUE)) {
    return(TRUE)
  }
  file <- sub("^--file=", "", grep("^--file=", commandArgs(FALSE), value = TRUE)[1])
  line <- sprintf("SKIP %s: package %s is not installed", basename(file), pkg)
  cat(line, "\n", sep = "")
  log <- Sys.getenv("PROSPICIO_R_SKIP_LOG")
  if (nzchar(log)) cat(line, "\n", sep = "", file = log, append = TRUE)
  FALSE
}

# near(a, b, rel, abs): every element within rel of b relatively or abs
# absolutely (the larger), or equal. Names are dropped; a and b have one
# length, or one of them has length 1. The one tolerance check of the R
# tests (#845); near_abs() and near_unit() name its two other common forms.
near <- function(a, b, rel = 1e-12, abs = 0) {
  a <- unname(a)
  b <- unname(b)
  same_length <- length(a) == length(b) || length(a) == 1L || length(b) == 1L
  ok <- same_length && isTRUE(all(a == b | base::abs(a - b) <= pmax(rel * base::abs(b), abs)))
  if (!ok) {
    stop(sprintf("got %s, want %s (rel %g, abs %g)",
                 paste(format(a, digits = 17), collapse = " "),
                 paste(format(b, digits = 17), collapse = " "), rel, abs), call. = FALSE)
  }
  invisible(TRUE)
}

# Within tol absolutely.
near_abs <- function(a, b, tol) near(a, b, rel = 0, abs = tol)

# Within tol relatively for |b| above 1 and absolutely below it:
# tol * max(1, |b|).
near_unit <- function(a, b, tol) near(a, b, rel = tol, abs = tol)

# The error message of expr, which must fail, contains pattern (fixed text).
expect_error_like <- function(expr, pattern) {
  msg <- tryCatch({
    expr
    NULL
  }, error = conditionMessage)
  if (is.null(msg)) stop(sprintf("expected an error mentioning %s", shQuote(pattern)), call. = FALSE)
  if (!grepl(pattern, msg, fixed = TRUE)) {
    stop(sprintf("error %s does not mention %s", shQuote(msg), shQuote(pattern)), call. = FALSE)
  }
  invisible(msg)
}

# expr's value and the messages of the warnings it raised, muffled.
with_warnings <- function(expr) {
  found <- character()
  value <- withCallingHandlers(expr, warning = function(w) {
    found <<- c(found, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = found)
}

# A path under validation/, from the repository root.
validation <- function(...) file.path("validation", ...)
