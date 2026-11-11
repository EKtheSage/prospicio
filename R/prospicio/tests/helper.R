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
