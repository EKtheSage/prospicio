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

# near(): each file sets its own `near` from one of the two factories below,
# so that every file keeps the tolerance rule it was written with (they
# differ: relative or absolute, a floor under |b|, an absolute floor, exact
# equality allowed, lengths checked). Without a default tolerance a call
# that gives none is an error, as before.
#
#   near_rel(rel, abs = 0, floor = 1e-300, exact = FALSE, same_length = FALSE, unname = FALSE)
#     a function(a, b, rel, abs) passing when, elementwise,
#     |a - b| <= max(rel * max(|b|, floor), abs) (or a == b with exact = TRUE)
#   near_abs(tol, same_length = FALSE, unname = FALSE)
#     a function(a, b, tol) passing when |a - b| <= tol elementwise
check_near <- function(a, b, ok_of, what, same_length, unname) {
  if (unname) {
    a <- base::unname(a)
    b <- base::unname(b)
  }
  if ((same_length && length(a) != length(b)) || !all(ok_of(a, b))) {
    stop(sprintf("got %s, want %s (%s)", paste(format(a, digits = 17), collapse = " "),
                 paste(format(b, digits = 17), collapse = " "), what), call. = FALSE)
  }
  invisible(TRUE)
}

near_rel <- function(rel = NULL, abs = 0, floor = 1e-300, exact = FALSE,
                     same_length = FALSE, unname = FALSE) {
  rel0 <- rel
  abs0 <- abs
  function(a, b, rel = rel0, abs = abs0) {
    if (is.null(rel)) stop("near(): no tolerance given", call. = FALSE)
    ok_of <- function(a, b) {
      ok <- base::abs(a - b) <= pmax(rel * pmax(base::abs(b), floor), abs)
      if (exact) a == b | ok else ok
    }
    check_near(a, b, ok_of, sprintf("rel %g, abs %g", rel, abs), same_length, unname)
  }
}

near_abs <- function(tol = NULL, same_length = FALSE, unname = FALSE) {
  tol0 <- tol
  function(a, b, tol = tol0) {
    if (is.null(tol)) stop("near(): no tolerance given", call. = FALSE)
    check_near(a, b, function(a, b) base::abs(a - b) <= tol, sprintf("tol %g", tol), same_length, unname)
  }
}

# expect_error_like(expr, pattern): expr fails with a message containing
# pattern (fixed text).
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

# with_warnings(expr): list(value, warnings), the warnings muffled and kept.
with_warnings <- function(expr) {
  found <- character()
  value <- withCallingHandlers(expr, warning = function(w) {
    found <<- c(found, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = found)
}

# validation(...): a path under validation/ (the tests run from the
# repository root).
validation <- function(...) file.path("validation", ...)
