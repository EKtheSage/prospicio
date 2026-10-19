# Regenerate validation/reference/reserving_averages_r.csv from R ChainLadder.
#
# Run from the repository root:
#
#     Rscript validation/scripts/reserving_averages_r.R
#
# Averages over the latest periods and excluding high and low link ratios
# (decision 13 of docs/design/reserving-v02.md) against R ChainLadder's
# MackChainLadder(weights = W), where W is a 0/1 matrix of the triangle's
# dimensions: W[i, j] = 1 keeps the link ratio from development j to j + 1 of
# origin i. The script builds W itself from the rule of the decision, apart
# from the library:
#
# * n_periods = n keeps the link ratios that start on the latest n diagonals
#   of link ratios (starting valuation i + j at least the latest less n);
# * drop_high and drop_low then leave out the highest and lowest of those at
#   each age, ties broken by the starting value, unless that would leave
#   none, when all of them stay.
#
# Cases for which R falls back to Mack's sigma rule (a warning) are left out and
# listed in the header. Triangles are read from validation/data/*.csv. Rows give Mack's factors,
# sigmas, standard errors by origin and in total for each setting and alpha.
# `arg` is the 0-based development index k (k links age k to k+1) for the
# per-age quantities, the origin year for the per-origin ones, empty for
# totals.

suppressPackageStartupMessages(library(ChainLadder))

OUT <- "validation/reference/reserving_averages_r.csv"
DATASETS <- c("raa", "genins")
VER <- as.character(packageVersion("ChainLadder"))
REL <- 1e-9
ZERO_ABS <- 1e-9

SETTINGS <- list(
  list(n_periods = 3, drop_high = 0, drop_low = 0),
  list(n_periods = NA, drop_high = 1, drop_low = 1),
  list(n_periods = 5, drop_high = 1, drop_low = 0),
  list(n_periods = 4, drop_high = 1, drop_low = 1)
)
ALPHAS <- list(volume = 1, simple = 0, regression = 2)

read_triangle <- function(name) {
  d <- read.csv(file.path("validation/data", paste0(name, ".csv")), comment.char = "#")
  origins <- sort(unique(d$origin))
  devs <- sort(unique(d$development))
  tri <- matrix(NA_real_, length(origins), length(devs),
                dimnames = list(origin = origins, dev = devs))
  tri[cbind(match(d$origin, origins), match(d$development, devs))] <- d$value
  as.triangle(tri)
}

# The 0/1 weight matrix of a setting.
weights_for <- function(tri, s) {
  n <- nrow(tri)
  w <- matrix(0, nrow(tri), ncol(tri))
  for (j in seq_len(ncol(tri) - 1)) {
    pairs <- which(!is.na(tri[, j]) & !is.na(tri[, j + 1]))
    if (!is.na(s$n_periods)) {
      # Starting valuation (i - 1) + (j - 1) at least n - 1 - n_periods.
      pairs <- pairs[(pairs - 1) + (j - 1) >= (n - 1) - s$n_periods]
    }
    drops <- s$drop_high + s$drop_low
    if (drops > 0 && length(pairs) > drops) {
      ratio <- tri[pairs, j + 1] / tri[pairs, j]
      ord <- order(ratio, tri[pairs, j])
      pairs <- pairs[ord][(s$drop_low + 1):(length(pairs) - s$drop_high)]
    }
    w[pairs, j] <- 1
  }
  w
}

label <- function(s) {
  parts <- character(0)
  if (!is.na(s$n_periods)) parts <- c(parts, sprintf("n_periods=%d", s$n_periods))
  if (s$drop_high > 0) parts <- c(parts, sprintf("drop_high=%d", s$drop_high))
  if (s$drop_low > 0) parts <- c(parts, sprintf("drop_low=%d", s$drop_low))
  paste(parts, collapse = ";")
}

rows <- character(0)
notes <- character(0)

emit <- function(dataset, method, quantity, arg, value, source) {
  abs_tol <- if (value == 0) ZERO_ABS else 0
  rows <<- c(rows, sprintf("%s,%s,%s,%s,%s,%s,%s,%s", dataset, method, quantity,
                           arg, sprintf("%.15g", value), format(abs_tol),
                           format(REL), source))
}

for (name in DATASETS) {
  tri <- read_triangle(name)
  n <- ncol(tri)
  origins <- rownames(tri)
  for (avg in names(ALPHAS)) {
    for (s in SETTINGS) {
      method <- sprintf("average=%s;%s", avg, label(s))
      w <- weights_for(tri, s)
      warned <- FALSE
      m <- withCallingHandlers(
        MackChainLadder(tri, weights = w, alpha = ALPHAS[[avg]], est.sigma = "log-linear"),
        warning = function(cond) {
          warned <<- TRUE
          notes <<- c(notes, sprintf("# %s %s: R warning: %s", name, method,
                                     gsub("\\s+", " ", conditionMessage(cond))))
          invokeRestart("muffleWarning")
        })
      # R switches to Mack's sigma rule when its log-linear fit is not
      # appropriate; the case then says nothing about the log-linear rule.
      if (warned) next
      call <- sprintf(
        "R ChainLadder %s: MackChainLadder(tri, weights = W, alpha = %s, est.sigma = 'log-linear')",
        VER, ALPHAS[[avg]])
      for (k in seq_len(n - 1)) {
        emit(name, method, "ldf", k - 1, m$f[k], paste0(call, "$f"))
        emit(name, method, "sigma", k - 1, m$sigma[k], paste0(call, "$sigma"))
      }
      for (i in seq_along(origins))
        emit(name, method, "se", origins[i], m$Mack.S.E[i, n], paste0(call, "$Mack.S.E[,n]"))
      emit(name, method, "total_standard_error", "", m$Total.Mack.S.E,
           paste0(call, "$Total.Mack.S.E"))
    }
  }
}

header <- c(
  sprintf("# Generated by validation/scripts/reserving_averages_r.R with R %s, ChainLadder %s.",
          paste(R.version$major, R.version$minor, sep = "."), VER),
  "# Mack's model under link-ratio exclusions, as MackChainLadder(weights = W). Mack risks are",
  "# standard errors (not variances) at ultimate.",
  unique(notes),
  "dataset,method,quantity,arg,expected,abs_tol,rel_tol,source"
)
con <- file(OUT, "wb")
writeLines(c(header, rows), con)
close(con)
cat(sprintf("wrote %d cases to %s\n", length(rows), OUT))
