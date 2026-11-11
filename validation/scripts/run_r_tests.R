#!/usr/bin/env Rscript
# Run the R package's test files and report every one (#845).
#
#   Rscript validation/scripts/run_r_tests.R              # every tests/test-*.R
#   Rscript validation/scripts/run_r_tests.R reserving    # files whose name contains "reserving"
#
# Each file runs in its own Rscript process from the repository root, as CI
# ran them before, so one file's state cannot leak into another. A failing
# file does not stop the run: every file runs, and the summary at the end
# lists each failure, the optional packages whose blocks were skipped (from
# `with_package()` in tests/helper.R) and the slowest files. The exit status
# is 1 when any file failed.

args <- commandArgs(trailingOnly = TRUE)
tests <- file.path("R", "prospicio", "tests")
if (!dir.exists(tests)) stop("run from the repository root (no ", tests, ")")
files <- sort(list.files(tests, pattern = "^test-.*\\.R$", full.names = TRUE))
if (length(args)) {
  keep <- Reduce(`|`, lapply(args, function(a) grepl(a, basename(files), fixed = TRUE)))
  files <- files[keep]
  if (!length(files)) stop("no test file matches ", paste(args, collapse = ", "))
}

rscript <- file.path(R.home("bin"), if (.Platform$OS.type == "windows") "Rscript.exe" else "Rscript")
skip_log <- tempfile("prospicio-skips-", fileext = ".txt")
file.create(skip_log)
old <- Sys.getenv("PROSPICIO_R_SKIP_LOG", unset = NA)
Sys.setenv(PROSPICIO_R_SKIP_LOG = skip_log)

results <- data.frame(file = basename(files), seconds = NA_real_, status = NA_integer_)
for (k in seq_along(files)) {
  cat(sprintf("\n== %s\n", files[k]))
  start <- proc.time()[["elapsed"]]
  status <- system2(rscript, shQuote(files[k]))
  results$seconds[k] <- proc.time()[["elapsed"]] - start
  results$status[k] <- status
  cat(sprintf("== %s: %s in %.0f s\n", files[k], if (status == 0) "passed" else "FAILED", results$seconds[k]))
}

if (is.na(old)) Sys.unsetenv("PROSPICIO_R_SKIP_LOG") else Sys.setenv(PROSPICIO_R_SKIP_LOG = old)
skips <- unique(readLines(skip_log))
failed <- results[results$status != 0, ]

cat("\n==== R tests:", nrow(results) - nrow(failed), "passed,", nrow(failed), "failed\n")
if (nrow(failed)) cat(sprintf("FAILED %s\n", failed$file), sep = "")
cat(sprintf("Skipped blocks (%d):\n", length(skips)))
if (length(skips)) cat(sprintf("  %s\n", skips), sep = "") else cat("  none\n")
slow <- head(results[order(-results$seconds), ], 10)
cat("The slowest R test files:\n")
cat(sprintf("  %5.0f s  %s\n", slow$seconds, slow$file), sep = "")
quit(status = if (nrow(failed)) 1 else 0)
