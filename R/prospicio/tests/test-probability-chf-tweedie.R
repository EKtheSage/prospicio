library(prospicio)
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# invert_chf on the Tweedie law at any power (#780), against aggregate
# 2.1.0's AggregateCHF: the rows of
# validation/reference/chf_tweedie_aggregate.csv the Rust parity test reads.
ref <- read.csv(validation("reference", "chf_tweedie_aggregate.csv"), stringsAsFactors = FALSE)
params <- function(text) {
  kv <- strsplit(strsplit(text, ";", fixed = TRUE)[[1]], "=", fixed = TRUE)
  stats::setNames(as.numeric(vapply(kv, `[`, "", 2)), vapply(kv, `[`, "", 1))
}
fits <- list()
for (i in seq_len(nrow(ref))) {
  r <- ref[i, ]
  if (is.null(fits[[r$params]])) {
    v <- params(r$params)
    law <- tweedie_power_distribution(v[["power"]], v[["mean"]], v[["dispersion"]])
    fits[[r$params]] <- invert_chf(law, log2 = v[["log2"]])
  }
  d <- fits[[r$params]]
  got <- switch(r$quantity,
    step = d@step, x_min = d@x_min,
    truncation_error = d@report$truncation_error,
    outside_below = d@report$outside_below, outside_above = d@report$outside_above,
    sum = d@report$total_mass, mean = mean(d), sd = sqrt(variance(d)),
    quantile = quantile(d, as.numeric(r$arg)),
    mass = d@masses[as.integer(r$arg) + 1])
  near(got, r$expected, rel = r$rel_tol, abs = r$abs_tol)
}

t <- tweedie_power_distribution(3, 10, 0.5)
stopifnot(t@variance == 500, t@power == 3)
near(sqrt(tweedie_power_distribution(2.5, 10, cv = 0.3)@variance) / 10, 0.3)
expect_error_like(tweedie_power_distribution(0.5, 1, 1), "power")
expect_error_like(tweedie_power_distribution(2.5, 1), "one of dispersion and cv")
