# Regenerate validation/reference/credibility_actuar.csv and
# validation/reference/hachemeister.csv from the R package actuar.
#
#   install.packages("actuar")
#   Rscript validation/scripts/r_credibility.R
#
# actuar (GPL) is used only to produce reference values. Rows:
#
# - The Bühlmann–Straub model (prospicio_pricing::credibility) fitted by
#   actuar::cm on Hachemeister's data (five states, twelve quarters of
#   average claim amounts and claim counts), with the unbiased and the
#   iterative (Bichsel–Straub) estimator; Bühlmann's model on the same
#   ratios with equal weights; and the weighted fit with four cells
#   missing. Each fit records the collective mean, the between and within
#   variances, and each state's credibility factor and premium.
# - Limited fluctuation standards (n_F = (qnorm((1 + p)/2) / k)^2) and the
#   square root rule.
#
# `hachemeister.csv` holds the data the fits read: state, quarter, ratio,
# weight.

library(actuar)
data(hachemeister)

out <- "validation/reference/credibility_actuar.csv"
source <- paste0("R actuar ", as.character(packageVersion("actuar")))

ratios <- as.matrix(hachemeister[, 2:13])
weights <- as.matrix(hachemeister[, 14:25])
data_rows <- do.call(rbind, lapply(1:5, function(i) data.frame(
  state = i, quarter = 1:12, ratio = ratios[i, ], weight = weights[i, ])))
write.csv(data_rows, "validation/reference/hachemeister.csv", row.names = FALSE, quote = FALSE)

rows <- list()
add <- function(case, quantity, arg, value, rel) {
  rows[[length(rows) + 1]] <<- data.frame(
    distribution = "buhlmann_straub", params = case, quantity = quantity, arg = arg,
    expected = sprintf("%.17g", value), abs_tol = 0, rel_tol = rel, source = source)
}

cm_fit <- function(r, w, method) {
  d <- data.frame(state = 1:5, r, w)
  names(d) <- c("state", paste0("r", 1:12), paste0("w", 1:12))
  cm(~state, d, ratios = r1:r12, weights = w1:w12, method = method)
}

record <- function(case, f, method, weighted) {
  est <- if (method == "iterative" && weighted) f$iterative else f$unbiased
  add(case, "collective_mean", "", f$means[[1]], 1e-12)
  add(case, "between_variance", "", est[1], 1e-10)
  add(case, "within_variance", "", est[2], 1e-12)
  p <- predict(f)
  for (i in 1:5) {
    add(case, "credibility", i, f$cred[i], 1e-11)
    add(case, "premium", i, p[i], 1e-12)
  }
}

record("weights=yes;estimator=unbiased",
       cm(~state, hachemeister, ratios = ratio.1:ratio.12, weights = weight.1:weight.12),
       "unbiased", TRUE)
record("weights=yes;estimator=iterative",
       cm(~state, hachemeister, ratios = ratio.1:ratio.12, weights = weight.1:weight.12,
          method = "iterative"),
       "iterative", TRUE)
record("weights=no;estimator=unbiased",
       cm(~state, hachemeister, ratios = ratio.1:ratio.12), "unbiased", FALSE)
# Four cells missing (state 1 quarter 3, state 2 quarters 11-12, state 4 quarter 1).
rm <- ratios; wm <- weights
miss <- rbind(c(1, 3), c(2, 11), c(2, 12), c(4, 1))
rm[miss] <- NA; wm[miss] <- NA
record("weights=yes;estimator=unbiased;missing=1.3|2.11|2.12|4.1",
       cm_fit(rm, wm, "Buhlmann-Gisler"), "unbiased", TRUE)

for (p in c(0.90, 0.95, 0.99)) {
  for (k in c(0.025, 0.05, 0.10)) {
    rows[[length(rows) + 1]] <- data.frame(
      distribution = "limited_fluctuation", params = sprintf("p=%s;k=%s", p, k),
      quantity = "frequency_standard", arg = "",
      expected = sprintf("%.17g", (qnorm((1 + p) / 2) / k)^2), abs_tol = 0, rel_tol = 1e-13,
      source = sprintf("R %s qnorm", getRversion()))
  }
}

write.csv(do.call(rbind, rows), out, row.names = FALSE, quote = FALSE)
cat("wrote", out, "\n")
