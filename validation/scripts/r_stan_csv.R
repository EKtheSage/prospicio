# Regenerate validation/reference/stan_csv_posterior.csv from the CmdStan
# output in validation/data/stan_eight_schools/ (written by
# validation/scripts/stan_csv_fixture.sh).
#
#   Rscript validation/scripts/r_stan_csv.R
#
# Reads the four chains with read.csv, independently of the Rust reader:
# comments skipped, the 150 saved warmup draws dropped, `theta.1` named
# `theta[1]`. Records, for every variable (lp__ included), posterior's
# summarise_draws values (mean, sd, the 5%, 50% and 95% quantiles, rhat,
# ess_bulk, ess_tail); the divergent transitions of each chain after
# warmup; and loo's PSIS-LOO with r_eff = relative_eff(exp(log_lik),
# chain_id): elpd_loo, p_loo, looic, se_elpd_loo, and per observation
# r_eff, elpd_loo and pareto_k.
#
# loo is GPL-3: this script only produces reference values.

suppressMessages({ library(posterior); library(loo) })
dir <- "validation/data/stan_eight_schools"
warmup <- 150
chains <- lapply(1:4, function(c) {
  x <- read.csv(file.path(dir, sprintf("chain-%d.csv", c)), comment.char = "#",
                check.names = FALSE)
  x[-seq_len(warmup), ]
})
divergent <- vapply(chains, function(x) sum(x$divergent__ != 0), numeric(1))
model_cols <- setdiff(names(chains[[1]]), grep("__$", names(chains[[1]]), value = TRUE))
model_cols <- c("lp__", model_cols)
rename <- function(n) vapply(n, function(v) {
  if (!grepl(".", v, fixed = TRUE)) return(v)
  parts <- strsplit(v, ".", fixed = TRUE)[[1]]
  sprintf("%s[%s]", parts[1], paste(parts[-1], collapse = ","))
}, character(1), USE.NAMES = FALSE)
arr <- array(NA_real_, dim = c(nrow(chains[[1]]), 4, length(model_cols)),
             dimnames = list(NULL, NULL, rename(model_cols)))
for (c in 1:4) arr[, c, ] <- as.matrix(chains[[c]][, model_cols])
draws <- as_draws_array(arr)

fmt <- function(x) sprintf("%.17g", x)
source <- paste0("posterior ", packageVersion("posterior"), " and loo ", packageVersion("loo"))
rows <- list()
add <- function(variable, quantity, value, rel, abs = 1e-12) {
  rows[[length(rows) + 1]] <<- data.frame(
    variable = variable, quantity = quantity, expected = fmt(value),
    abs_tol = abs, rel_tol = rel, source = source
  )
}
s <- summarise_draws(draws, mean, sd, ~quantile(.x, c(0.05, 0.5, 0.95)),
                     rhat, ess_bulk, ess_tail)
for (i in seq_len(nrow(s))) {
  v <- s$variable[i]
  add(v, "mean", s$mean[i], 1e-12)
  add(v, "sd", s$sd[i], 1e-12)
  add(v, "q05", s[["5%"]][i], 1e-12)
  add(v, "q50", s[["50%"]][i], 1e-12)
  add(v, "q95", s[["95%"]][i], 1e-12)
  add(v, "rhat", s$rhat[i], 1e-10)
  add(v, "ess_bulk", s$ess_bulk[i], 1e-10)
  add(v, "ess_tail", s$ess_tail[i], 1e-10)
}
for (c in 1:4) add(sprintf("chain %d", c), "divergences", divergent[c], 0, 0)

ll_cols <- grep("^log_lik\\[", dimnames(arr)[[3]], value = TRUE)
ll <- arr[, , ll_cols, drop = FALSE]
r_eff <- relative_eff(exp(ll))
l <- loo(ll, r_eff = r_eff)
est <- l$estimates
add("log_lik", "elpd_loo", est["elpd_loo", "Estimate"], 1e-9)
add("log_lik", "p_loo", est["p_loo", "Estimate"], 1e-8)
add("log_lik", "looic", est["looic", "Estimate"], 1e-9)
add("log_lik", "se_elpd_loo", est["elpd_loo", "SE"], 1e-8)
for (i in seq_along(ll_cols)) {
  add(ll_cols[i], "r_eff", r_eff[i], 1e-10)
  add(ll_cols[i], "elpd_loo", l$pointwise[i, "elpd_loo"], 1e-9)
  add(ll_cols[i], "pareto_k", l$diagnostics$pareto_k[i], 1e-6, 1e-8)
}
write.csv(do.call(rbind, rows), "validation/reference/stan_csv_posterior.csv",
          row.names = FALSE, quote = FALSE)
cat("wrote validation/reference/stan_csv_posterior.csv\n")
