library(prospicio)
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# An event loss table as a compound Poisson severity on a grid (#781),
# against aggregate 2.1.0: the rows of
# validation/reference/elt_grid_aggregate.csv the Rust parity test reads.
step <- 10
points <- 1024
means <- event_loss_table(rate = c(0.5, 0.25, 0.05, 0.01, 0.002),
                          mean = c(130, 270.5, 1234.5, 2999.9, 7777.7),
                          sd_independent = rep(0, 5), sd_correlated = rep(0, 5),
                          exposure = 2 * c(130, 270.5, 1234.5, 2999.9, 7777.7))
secondary <- event_loss_table(rate = c(0.01, 0.05, 0.002, 0.2, 0.0005),
                              mean = c(800, 250, 1500, 40, 6000),
                              sd_independent = c(200, 0, 300, 30, 3000),
                              sd_correlated = c(100, 0, 300, 10, 1000),
                              exposure = c(2000, 500, 5000, 300, 20000))
probs <- function(g) g@probs
# aggregate leaves the annual mass above the grid out; the FFT lumps it on
# the last node.
annual <- function(elt, rule) {
  a <- elt_aggregate_grid(elt, step, points, rule)$annual
  p <- probs(a)
  p[points] <- p[points] - a@report$tail_mass
  p
}
m_sev <- elt_severity_grid(means, step, points, "means_linear")$severity
s_sev <- probs(elt_severity_grid(secondary, step, points)$severity)
tower <- reinsurance_tower(list(xol_layer("1000 xs 500", 1000, 500)))
ceded <- probs(tower_on_grid(tower, poisson_count(means$rate), m_sev, points)$ceded[[1]])
grids <- list(
  "table=means|sev" = probs(m_sev),
  "table=means|agg" = annual(means, "means_linear"),
  "table=means_ceded|agg" = ceded,
  "table=secondary|sev" = s_sev,
  "table=secondary|agg" = annual(secondary, "nearest"),
  "table=secondary_exact|sev" = s_sev
)
ref <- read.csv(validation("reference", "elt_grid_aggregate.csv"), stringsAsFactors = FALSE)
got <- vapply(seq_len(nrow(ref)), function(i) {
  grids[[paste(ref$params[i], ref$quantity[i], sep = "|")]][ref$arg[i] + 1]
}, 0)
bad <- which(abs(got - ref$expected) > ref$abs_tol)
if (length(bad)) stop("elt grid parity fails at rows ", paste(head(bad, 10), collapse = ", "))

# The report: the 20000-exposure row reaches past a short grid.
narrow <- elt_severity_grid(secondary, step, 64)$report
stopifnot(narrow$bucketing == "nearest", narrow$rate_above > 0,
          narrow$grid_mean < narrow$severity_mean)
expect_error_like(elt_severity_grid(secondary, step, 64, "linear"), "should be one of")
