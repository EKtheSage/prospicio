# Regenerate validation/reference/reserving_tails_r.csv from R ChainLadder:
# MackChainLadder with a tail (tail = TRUE, tail = <number>, tail.se and
# tail.sigma given).
#
# Run from the repository root:
#
#     Rscript validation/scripts/reserving_tails_r.R
#
# Triangles are read from validation/data/*.csv as in reserving_r.R. `arg` is
# the 0-based development index k for per-age quantities (k = n - 1 is the
# oldest age, whose cdf is the tail factor), the origin year for per-origin
# quantities, and empty for totals and tail quantities.

suppressPackageStartupMessages(library(ChainLadder))

OUT <- "validation/reference/reserving_tails_r.csv"
DATASETS <- c("raa", "genins", "abc")
VER <- as.character(packageVersion("ChainLadder"))
REL <- 1e-9
ZERO_ABS <- 1e-9

read_triangle <- function(name) {
  d <- read.csv(file.path("validation/data", paste0(name, ".csv")), comment.char = "#")
  origins <- sort(unique(d$origin))
  devs <- sort(unique(d$development))
  tri <- matrix(NA_real_, length(origins), length(devs),
                dimnames = list(origin = origins, dev = devs))
  tri[cbind(match(d$origin, origins), match(d$development, devs))] <- d$value
  as.triangle(tri)
}

rows <- character(0)
notes <- character(0)

emit <- function(dataset, method, quantity, arg, value, source) {
  if (!is.finite(value)) stop(sprintf("%s %s %s %s is not finite", dataset, method, quantity, arg))
  abs_tol <- if (value == 0) ZERO_ABS else 0
  rows <<- c(rows, sprintf("%s,%s,%s,%s,%s,%s,%s,%s", dataset, method, quantity,
                           arg, sprintf("%.15g", value), format(abs_tol),
                           format(REL), source))
}

# Runs MackChainLadder, recording any warning or printed message.
mack <- function(name, method, tri, ...) {
  out <- NULL
  printed <- capture.output(
    out <- withCallingHandlers(
      MackChainLadder(tri, ...),
      warning = function(w) {
        msg <- gsub("\\s+", " ", conditionMessage(w))
        notes <<- c(notes, sprintf("# %s %s: R warning: %s", name, method, msg))
        invokeRestart("muffleWarning")
      }))
  for (p in printed) notes <<- c(notes, sprintf("# %s %s: R printed: %s", name, method, p))
  out
}

# method -> MackChainLadder arguments. `tail.se` and `tail.sigma` of
# mack_tail_given are set per dataset below, from the fitted pattern.
METHODS <- list(
  mack_tail_loglinear            = list(tail = TRUE),
  mack_tail_loglinear_sigma_mack = list(tail = TRUE, est.sigma = "Mack"),
  mack_tail_loglinear_alpha2     = list(tail = TRUE, alpha = 2),
  mack_tail_constant             = list(tail = 1.05),
  mack_tail_constant_sigma_mack  = list(tail = 1.05, est.sigma = "Mack"),
  mack_tail_given                = list(tail = 1.05)
)

for (name in DATASETS) {
  tri <- read_triangle(name)
  n <- ncol(tri)
  origins <- rownames(tri)
  latest <- getLatestCumulative(tri)

  for (method in names(METHODS)) {
    args <- METHODS[[method]]
    if (method == "mack_tail_given") {
      # A tail standard error and sigma well away from the extrapolated ones:
      # half the last factor's standard error and twice its sigma.
      base <- MackChainLadder(tri)
      args$tail.se <- base$f.se[n - 1] / 2
      args$tail.sigma <- base$sigma[n - 1] * 2
      emit(name, method, "given_tail_std_err", "", args$tail.se,
           sprintf("R ChainLadder %s: MackChainLadder(tri)$f.se[n-1] / 2 (an input)", VER))
      emit(name, method, "given_tail_sigma", "", args$tail.sigma,
           sprintf("R ChainLadder %s: MackChainLadder(tri)$sigma[n-1] * 2 (an input)", VER))
    }
    m <- do.call(mack, c(list(name, method, tri), args))
    shown <- args
    if (!is.null(shown$tail.se)) shown$tail.se <- "f.se[n-1]/2"
    if (!is.null(shown$tail.sigma)) shown$tail.sigma <- "sigma[n-1]*2"
    call <- sprintf("R ChainLadder %s: MackChainLadder(tri, %s)", VER,
                    paste(names(shown), sapply(shown, function(v)
                      if (is.character(v) && !grepl("\\[", v)) sprintf("'%s'", v) else as.character(v)),
                      sep = "=", collapse = ", "))
    tail_factor <- m$f[n]
    emit(name, method, "tail_factor", "", tail_factor, paste0(call, "$f[n]"))
    if (tail_factor > 1) {
      emit(name, method, "tail_std_err", "", m$f.se[n], paste0(call, "$f.se[n]"))
      emit(name, method, "tail_sigma", "", m$sigma[n], paste0(call, "$sigma[n]"))
    }
    f <- m$f[1:n]
    cdf <- rev(cumprod(rev(f)))
    for (k in seq_len(n)) emit(name, method, "cdf", k - 1, cdf[k], paste0(call, "$f cumulative product"))
    ult <- m$FullTriangle[, ncol(m$FullTriangle)]
    last <- ncol(m$Mack.S.E)
    for (i in seq_along(origins)) {
      o <- origins[i]
      emit(name, method, "ultimate", o, ult[i], paste0(call, "$FullTriangle[,last]"))
      emit(name, method, "reserve", o, ult[i] - latest[i], paste0(call, " summary ByOrigin IBNR"))
      emit(name, method, "se", o, m$Mack.S.E[i, last], paste0(call, "$Mack.S.E[,last]"))
      emit(name, method, "process_risk", o, m$Mack.ProcessRisk[i, last],
           paste0(call, "$Mack.ProcessRisk[,last]"))
      emit(name, method, "parameter_risk", o, m$Mack.ParameterRisk[i, last],
           paste0(call, "$Mack.ParameterRisk[,last]"))
    }
    emit(name, method, "total_ultimate", "", sum(ult), paste0(call, " summary Totals Ultimate"))
    emit(name, method, "total_reserve", "", sum(ult - latest), paste0(call, " summary Totals IBNR"))
    emit(name, method, "total_standard_error", "", m$Total.Mack.S.E, paste0(call, "$Total.Mack.S.E"))
    emit(name, method, "total_process_risk", "", m$Total.ProcessRisk[last],
         paste0(call, "$Total.ProcessRisk[last]"))
    emit(name, method, "total_parameter_risk", "", m$Total.ParameterRisk[last],
         paste0(call, "$Total.ParameterRisk[last]"))
  }
}

header <- c(
  sprintf("# Generated by validation/scripts/reserving_tails_r.R with R %s, ChainLadder %s.",
          paste(R.version$major, R.version$minor, sep = "."), VER),
  "# Triangles from validation/data/{raa,genins,abc}.csv. Mack risks are standard errors at",
  "# ultimate, including the tail. tail = TRUE is R's log-linear rule (tailfactor); the tail's",
  "# sigma and standard error are extrapolated log-linearly (tail_SE) unless given.",
  unique(notes),
  "dataset,method,quantity,arg,expected,abs_tol,rel_tol,source"
)
con <- file(OUT, "wb")
writeLines(c(header, rows), con)
close(con)
cat(sprintf("wrote %d cases to %s\n", length(rows), OUT))
