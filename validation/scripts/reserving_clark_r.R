# Regenerate validation/reference/reserving_clark_r.csv from R ChainLadder's
# ClarkLDF and ClarkCapeCod.
#
# Run from the repository root:
#
#     Rscript validation/scripts/reserving_clark_r.R
#
# Triangles are read from validation/data/*.csv with ages in months (12, 24,
# ..., 120), so theta is in months and R's default origin width is 12 with
# the average date of loss at 6 months. ClarkLDF runs on genins and raa,
# ClarkCapeCod on genins_premium (premium 10e6 + 0.4e6 * (0:9)), each with
# both growth curves and maxage Inf and 240 months.
#
# `method` is `clark_ldf` or `clark_cape_cod` followed by `curve=`,
# `max_age=` (months, `inf` for none) and `optim=`:
#
# * `optim=default` is R as shipped. Its L-BFGS-B stops when the
#   log-likelihood changes by less than factr * eps = 1.5e-8 relative
#   (factr = eps^-0.5), short of the maximum: its parameters, reserves and
#   standard errors are up to a few 1e-3 from a converged fit (the header
#   reports the largest gap). These rows carry rel_tol 1e-2.
# * `optim=tight` is the same R code with factr = 1, so L-BFGS-B runs until
#   it cannot improve; the parameters then agree with a converged fit to
#   about 1e-7 and every row carries rel_tol 1e-5.
#
# R's Weibull second derivative d2G/domega2 is 2 v log(x/theta) (1 - u)
# where the derivative of its own first derivative v log(x/theta) is
# v log(x/theta)^2 (1 - u) (u = (x/theta)^omega, v = u exp(-u)); see
# knowledge/references/r-chainladder-clark.md. The Weibull parameter and total
# standard errors (per origin, total, and omega_se/theta_se/elr_se) are
# therefore taken from the same R code with that one entry corrected
# (`optim=tight` only); the header reports how far R's shipped values are
# from the corrected ones.
#
# `arg` is the origin year for per-origin quantities and empty otherwise.

suppressPackageStartupMessages(library(ChainLadder))

OUT <- "validation/reference/reserving_clark_r.csv"
VER <- as.character(packageVersion("ChainLadder"))
NS <- asNamespace("ChainLadder")
REL <- c(default = 1e-2, tight = 1e-5)
MAXAGES <- c(Inf, 240)
CURVES <- c("loglogistic", "weibull")

read_triangle <- function(name, column) {
  d <- read.csv(file.path("validation/data", paste0(name, ".csv")), comment.char = "#")
  origins <- sort(unique(d$origin))
  devs <- sort(unique(d$development))
  tri <- matrix(NA_real_, length(origins), length(devs),
                dimnames = list(origin = origins, dev = devs))
  tri[cbind(match(d$origin, origins), match(d$development, devs))] <- d[[column]]
  tri
}

# The correct d2G/dtheta2 of the Weibull growth curve: R's, with the
# omega-omega entry v log(x/theta)^2 (1 - u).
weibull_fixed <- NS$weibull
weibull_fixed@d2Gdt2 <- function(x, theta) {
  d2 <- NS$weibull@d2Gdt2(x, theta)
  om <- theta[1L]
  th <- theta[2L]
  u <- (x / th)^om
  v <- exp(-u) * u
  a <- v * log(x / th)^2 * (1 - u)
  a[x <= 0 | is.infinite(x) | is.na(a)] <- 0
  if (length(x) > 1L) d2[, 1] <- a else d2[1, 1] <- a
  d2
}

# `f` (ClarkLDF or ClarkCapeCod) with L-BFGS-B's factr set to `factr` and,
# when `fix_weibull`, the corrected Weibull second derivative.
patched <- function(f, factr, fix_weibull) {
  txt <- deparse(f)
  txt <- gsub(".Machine$double.eps^-0.5", factr, txt, fixed = TRUE)
  g <- eval(parse(text = txt))
  env <- new.env(parent = NS)
  if (fix_weibull) assign("weibull", weibull_fixed, envir = env)
  environment(g) <- env
  g
}

rows <- character(0)
notes <- character(0)
gaps <- list()
weibull_gap <- 0

emit <- function(dataset, method, quantity, arg, value, rel, source) {
  if (!is.finite(value)) stop(sprintf("%s %s %s %s is not finite", dataset, method, quantity, arg))
  rows <<- c(rows, sprintf("%s,%s,%s,%s,%s,0,%s,%s", dataset, method, quantity, arg,
                           sprintf("%.15g", value), format(rel), source))
}

# Runs `expr`, recording R's warnings as header notes.
quiet <- function(label, expr) {
  withCallingHandlers(expr, warning = function(w) {
    msg <- gsub("\\s+", " ", conditionMessage(w))
    notes <<- c(notes, sprintf("# %s: R warning: %s", label, msg))
    invokeRestart("muffleWarning")
  })
}

# The reference quantities of a Clark fit, as a named list of
# (quantity, arg) -> value.
quantities <- function(x, origins) {
  q <- list()
  put <- function(name, arg, value) q[[paste(name, arg, sep = "|")]] <<- value
  put("omega", "", x$THETAG[1])
  put("theta", "", x$THETAG[2])
  put("sigma2", "", x$SIGMA2)
  v <- vcov(x)
  put("omega_se", "", sqrt(v["omega", "omega"]))
  put("theta_se", "", sqrt(v["theta", "theta"]))
  if (x$method == "CapeCod") {
    put("elr", "", x$ELR)
    put("elr_se", "", sqrt(v["ELR", "ELR"]))
  } else {
    for (i in seq_along(origins)) put("expected_ultimate", origins[i], x$THETAU[i])
  }
  for (i in seq_along(origins)) {
    put("ultimate", origins[i], x$UltimateValue[i])
    put("reserve", origins[i], x$FutureValue[i])
    put("process_se", origins[i], x$ProcessSE[i])
    put("parameter_se", origins[i], x$ParameterSE[i])
    put("standard_error", origins[i], x$StdError[i])
  }
  put("total_ultimate", "", x$Total$UltimateValue)
  put("total_reserve", "", x$Total$FutureValue)
  put("total_process_se", "", x$Total$ProcessSE)
  put("total_parameter_se", "", x$Total$ParameterSE)
  put("total_standard_error", "", x$Total$StdError)
  q
}

# Quantities that come from the Hessian, so from R's Weibull d2G/domega2.
HESSIAN <- c("omega_se", "theta_se", "elr_se", "parameter_se", "standard_error",
             "total_parameter_se", "total_standard_error")

CASES <- list(
  list(dataset = "genins", column = "value", method = "clark_ldf", fn = ClarkLDF),
  list(dataset = "raa", column = "value", method = "clark_ldf", fn = ClarkLDF),
  list(dataset = "genins_premium", column = "paid", method = "clark_cape_cod", fn = ClarkCapeCod)
)

for (case in CASES) {
  tri <- read_triangle(case$dataset, case$column)
  origins <- rownames(tri)
  premium <- if (case$method == "clark_cape_cod") read_triangle(case$dataset, "premium")[, 1]
  rname <- if (case$method == "clark_ldf") "ClarkLDF" else "ClarkCapeCod"
  run <- function(f, G, maxage) {
    if (is.null(premium)) f(tri, G = G, maxage = maxage)
    else f(tri, Premium = premium, G = G, maxage = maxage)
  }
  for (G in CURVES) for (maxage in MAXAGES) {
    ma <- if (is.infinite(maxage)) "inf" else format(maxage)
    base <- sprintf("%s;curve=%s;max_age=%s", case$method, G, ma)
    call <- sprintf("%s(tri%s, G='%s', maxage=%s)", rname,
                    if (is.null(premium)) "" else ", Premium=premium", G,
                    if (is.infinite(maxage)) "Inf" else format(maxage))
    weib <- G == "weibull"
    shipped <- quiet(paste(case$dataset, base, "optim=default"), run(case$fn, G, maxage))
    tight <- quiet(paste(case$dataset, base, "optim=tight"),
                   run(patched(case$fn, "1", fix_weibull = weib), G, maxage))
    qd <- quantities(shipped, origins)
    qt <- quantities(tight, origins)
    if (weib) {
      as_shipped <- run(patched(case$fn, "1", fix_weibull = FALSE), G, maxage)
      qb <- quantities(as_shipped, origins)
      for (key in names(qt)) {
        if (strsplit(key, "|", fixed = TRUE)[[1]][1] %in% HESSIAN)
          weibull_gap <- max(weibull_gap, abs(qb[[key]] - qt[[key]]) / abs(qt[[key]]))
      }
    }
    for (key in names(qt)) {
      parts <- strsplit(key, "|", fixed = TRUE)[[1]]
      quantity <- parts[1]
      arg <- if (length(parts) > 1) parts[2] else ""
      from_hessian <- quantity %in% HESSIAN
      src_d <- sprintf("R ChainLadder %s: %s", VER, call)
      src_t <- sprintf("R ChainLadder %s: %s with optim factr = 1%s", VER, call,
                       if (weib) " and the corrected Weibull d2G/domega2" else "")
      emit(case$dataset, paste0(base, ";optim=tight"), quantity, arg, qt[[key]], REL["tight"], src_t)
      if (!(weib && from_hessian)) {
        emit(case$dataset, paste0(base, ";optim=default"), quantity, arg, qd[[key]], REL["default"], src_d)
        gap <- abs(qd[[key]] - qt[[key]]) / abs(qt[[key]])
        gaps[[quantity]] <- max(c(gaps[[quantity]], gap))
      }
    }
  }
}

gap_lines <- vapply(names(gaps), function(q) sprintf("#   %s: %.2g", q, gaps[[q]]), "")
if (max(unlist(gaps)) > REL["default"] / 2) stop("optim=default gap exceeds half its tolerance")
header <- c(
  sprintf("# Generated by validation/scripts/reserving_clark_r.R with R %s, ChainLadder %s.",
          paste(R.version$major, R.version$minor, sep = "."), VER),
  "# Ages in months; adol = TRUE (origin width 12, average date of loss 6 months). reserve is",
  "# R's FutureValue (ClarkLDF: the truncated LDF applied to the latest value); *_se are standard",
  "# errors. optim=default is R as shipped; optim=tight runs L-BFGS-B with factr = 1 (see script).",
  "# Largest relative gap between optim=default and optim=tight, by quantity:",
  gap_lines,
  sprintf("# Weibull standard errors (optim=tight) differ from R's shipped d2G/domega2 by up to %.2g relative.",
          weibull_gap),
  unique(notes),
  "dataset,method,quantity,arg,expected,abs_tol,rel_tol,source"
)
con <- file(OUT, "wb")
writeLines(c(header, rows), con)
close(con)
cat(sprintf("wrote %d cases to %s\n", length(rows), OUT))
