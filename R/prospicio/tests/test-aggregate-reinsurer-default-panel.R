library(prospicio)
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# Reinsurer default on a tower (reinsurer(), reinsurer_panel(), #980): the
# result's components and identities, exact values on a panel whose
# defaults are certain, determinism by seed, the errors, and the cases of
# validation/reference/reinsurer_default.csv on stratified lognormal years
# at a few standard errors.
col <- function(pd, ...) marginal(pd, list(...))@draws

r <- reinsurer("Re A", 0.002, 0.4, collateral = 1e6, correlation = 0.3)
stopifnot(r@name == "Re A", r@probability == 0.002, r@recovery == 0.4,
          r@collateral == 1e6, r@correlation == 0.3)
plain <- reinsurer("Re B", 0.01, 0.5)
stopifnot(plain@collateral == 0, plain@correlation == 0)

# The Rust doctest's panel: Re A always defaults, Re B never does.
years <- apply_tower(reinsurance_tower(list(xol_layer("10x10", 10, 10))), c(5, 15, 30))
certain <- panel_line(reinsurer_panel(list(reinsurer("Re A", 1, 0.4), reinsurer("Re B", 0, 0.4))),
                      c("Re A", "Re B"), "10x10", 0.5)
stopifnot(identical(certain@reinsurer_names, c("Re A", "Re B")))
after <- apply_panel(certain, years, seed = 7)
stopifnot(identical(col(after, "credit_loss", "10x10"), c(0, 1.5, 3)))
stopifnot(identical(col(after, "recovered", "10x10"), c(0, 3.5, 7)))
stopifnot(identical(col(after, "net_after_default", "retained"), c(5, 11.5, 23)))
by <- panel_by_reinsurer(certain, years, seed = 7)
stopifnot(identical(col(by, "default", "Re A"), c(1, 1, 1)))
stopifnot(identical(col(by, "default", "Re B"), c(0, 0, 0)))
stopifnot(identical(col(by, "exposure", "Re A"), c(0, 2.5, 5)))

# The identities on random defaults; C is on no line.
gross <- seq(0, 78, by = 3)
years <- apply_tower(reinsurance_tower(list(xol_layer("A", 20, 15), xol_layer("B", 40, 35),
                                            xol_layer("C", 10, 75))), gross)
panel <- reinsurer_panel(list(reinsurer("R1", 0.4, 0.3, collateral = 5), reinsurer("R2", 0.5, 0.6)))
panel <- panel_line(panel, c("R1", "R2", "R1", "R2"), c("A", "A", "B", "B"), c(0.6, 0.4, 0.3, 0.7))
after <- apply_panel(panel, years, seed = 11)
total <- 0
for (l in c("A", "B", "C")) {
  credit <- col(after, "credit_loss", l)
  near(credit + col(after, "recovered", l), col(years, "ceded", l), abs = 1e-12)
  total <- total + credit
}
stopifnot(all(col(after, "credit_loss", "C") == 0), any(total > 0))
near(col(after, "net_after_default", "retained"), col(years, "net", "retained") + total, abs = 1e-12)
by <- panel_by_reinsurer(panel, years, seed = 11)
near(col(by, "credit_loss", "R1") + col(by, "credit_loss", "R2"), total, abs = 1e-12)

# The draws are fixed by the seed, 0 by default; a driver ranks the years.
years <- apply_tower(reinsurance_tower(list(xol_layer("A", 20, 15))), seq(10, 59))
one_panel <- panel_line(reinsurer_panel(list(reinsurer("R1", 0.5, 0.2, correlation = 0.4))), "R1", "A", 1)
d3 <- col(panel_by_reinsurer(one_panel, years, seed = 3), "default", "R1")
stopifnot(identical(col(panel_by_reinsurer(one_panel, years, seed = 3), "default", "R1"), d3))
stopifnot(!identical(col(panel_by_reinsurer(one_panel, years, seed = 4), "default", "R1"), d3))
stopifnot(identical(col(panel_by_reinsurer(one_panel, years), "default", "R1"),
                    col(panel_by_reinsurer(one_panel, years, seed = 0), "default", "R1")))
driven <- apply_panel(panel_driver(one_panel, -seq_len(50)), years, seed = 3)
stopifnot(length(col(driven, "credit_loss", "A")) == 50)

# The errors.
years <- apply_tower(reinsurance_tower(list(xol_layer("A", 20, 15))), c(10, 40))
expect_error_like(reinsurer("R1", 1.5, 0.4), "probability")
expect_error_like(reinsurer_panel(list()), "at least one reinsurer")
expect_error_like(reinsurer_panel(list(reinsurer("R1", 0.1, 0.4), reinsurer("R1", 0.2, 0.4))), "unique")
two <- reinsurer_panel(list(reinsurer("R1", 0.1, 0.4), reinsurer("R2", 0.1, 0.4)))
expect_error_like(panel_line(two, "R9", "A", 0.5), "no reinsurer")
expect_error_like(panel_line(two, c("R1", "R2"), "A", c(0.5, 0.5, 0.5)), "equal lengths")
expect_error_like(panel_line(two, c("R1", "R1"), "A", 0.5), "already has a line")
expect_error_like(apply_panel(panel_line(two, "R1", "A", 0.5), years), "sum")
expect_error_like(apply_panel(panel_line(two, "R1", "Z", 1), years), "Z")
expect_error_like(apply_panel(panel_line(two, "R1", "A", 1), years, seed = -1), "seed")

# The reference's cases on stratified years: the gross loss at the
# lognormal's quantiles (i - 1/2) / n, so the only sampling error is the
# default draws'.
ref <- read.csv(validation("reference", "reinsurer_default.csv"), colClasses = "character")
params <- function(s) {
  pairs <- strsplit(strsplit(s, ";", fixed = TRUE)[[1]], "=", fixed = TRUE)
  stats::setNames(as.double(vapply(pairs, `[`, "", 2)), vapply(pairs, `[`, "", 1))
}
p <- params(ref$params[1])
n <- 100000
gross <- exp(p[["meanlog"]] + p[["sdlog"]] * qnorm((seq_len(n) - 0.5) / n))
tower <- reinsurance_tower(lapply(c("A", "B"), function(l) {
  xol_layer(l, p[[paste0("limit_", l)]], p[[paste0("attach_", l)]])
}))
strat <- apply_tower(tower, gross)
panel_of <- function(s) {
  p <- params(s)
  names <- paste0("R", seq_len(p[["reinsurers"]]))
  panel <- reinsurer_panel(lapply(names, function(r) {
    reinsurer(r, p[[paste0("p_", r)]], p[[paste0("rec_", r)]],
              collateral = p[[paste0("col_", r)]], correlation = p[[paste0("rho_", r)]])
  }))
  for (r in names) for (l in c("A", "B")) {
    s <- p[[paste0("line_", r, "_", l)]]
    if (s > 0) panel <- panel_line(panel, r, l, s)
  }
  panel
}
cases <- ref[ref$distribution != "gross" &
               ref$quantity %in% c("credit_loss_mean", "credit_loss_mean_by_layer", "default_frequency"), ]
stopifnot(nrow(cases) > 10)
for (k in seq_len(nrow(cases))) {
  row <- cases[k, ]
  panel <- panel_of(row$params)
  x <- if (row$quantity == "credit_loss_mean_by_layer") {
    col(apply_panel(panel, strat, seed = 5), "credit_loss", row$arg)
  } else {
    by <- panel_by_reinsurer(panel, strat, seed = 5)
    kind <- if (row$quantity == "default_frequency") "default" else "credit_loss"
    who <- if (row$arg == "total") panel@reinsurer_names else row$arg
    Reduce(`+`, lapply(who, function(r) col(by, kind, r)))
  }
  near_abs(mean(x), as.double(row$expected), 5 * sd(x) / sqrt(length(x)))
}
