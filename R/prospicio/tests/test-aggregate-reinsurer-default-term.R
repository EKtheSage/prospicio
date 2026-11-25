library(prospicio)
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# Reinsurer default over a term (apply_panel_term(), panel_by_reinsurer_term(),
# #980): collateral used up year by year on a certain default, a term of one
# year drawing as apply_panel() does, a default final for the rest of the
# path at the rate 1 - (1 - p)^t, the identities and the errors.
col <- function(pd, ...) marginal(pd, list(...))@draws

# One path of two years, ceding 5 and then 10; Re A certainly defaults,
# recovers nothing and has collateral of 8.
term <- apply_tower_term(reinsurance_tower(list(xol_layer("10x10", 10, 10))),
                         list(events_from_years(list(15)), events_from_years(list(25))))
panel <- panel_line(reinsurer_panel(list(reinsurer("Re A", 1, 0, collateral = 8))), "Re A", "10x10", 1)
after <- apply_panel_term(panel, term, n_years = 2)
stopifnot(identical(col(after, "credit_loss", "10x10", 1L), 0))
stopifnot(identical(col(after, "credit_loss", "10x10", 2L), 7))
stopifnot(identical(col(after, "recovered", "10x10", 2L), 3))
stopifnot(identical(col(after, "net_after_default", "retained", 2L), 22))
m <- ncol(draw_matrix(term))
stopifnot(identical(unname(draw_matrix(after)[, seq_len(m), drop = FALSE]),
                    unname(draw_matrix(term))))
by <- panel_by_reinsurer_term(panel, term, n_years = 2, seed = 3)
stopifnot(identical(by@dims, c("kind", "reinsurer", "year")))
stopifnot(identical(col(by, "exposure", "Re A", 2L), 7))
stopifnot(identical(col(by, "default", "Re A", 1L), 1))

# A term of one year draws as apply_panel() does on that year.
ev <- events_from_years(as.list(seq(0, 78, by = 2)))
tower <- reinsurance_tower(list(xol_layer("A", 20, 15)))
one_panel <- panel_line(reinsurer_panel(list(reinsurer("R1", 0.4, 0.3, collateral = 2, correlation = 0.5))),
                        "R1", "A", 1)
one <- panel_by_reinsurer(one_panel, apply_tower(tower, ev), seed = 9)
as_term <- panel_by_reinsurer_term(one_panel, apply_tower_term(tower, list(ev)), n_years = 1, seed = 9)
for (kind in c("credit_loss", "exposure", "default")) {
  stopifnot(identical(col(as_term, kind, "R1", 1L), col(one, kind, "R1")))
}

# Final default at the rate 1 - (1 - p)^t, and the identities.
n_paths <- 4000
n_years <- 4
p <- 0.2
years <- lapply(seq_len(n_years) - 1, function(t) {
  events_from_years(as.list(as.double((7 * (seq_len(n_paths) - 1) + 13 * t * t + t) %% 60)))
})
term <- apply_tower_term(reinsurance_tower(list(xol_layer("A", 20, 15), xol_layer("B", 20, 35))), years)
panel <- reinsurer_panel(list(reinsurer("R1", p, 0.3, collateral = 4),
                              reinsurer("R2", 0.1, 0.5, correlation = 0.4)))
panel <- panel_line(panel, c("R1", "R2", "R1"), c("A", "A", "B"), c(0.5, 0.5, 1))
by <- panel_by_reinsurer_term(panel, term, n_years, seed = 5)
after <- apply_panel_term(panel, term, n_years, seed = 5)
last <- rep(0, n_paths)
for (t in seq_len(n_years)) {
  d <- col(by, "default", "R1", t)
  stopifnot(all(d >= last))
  last <- d
  q <- 1 - (1 - p)^t
  near_abs(mean(d), q, 5 * sqrt(q * (1 - q) / n_paths))
  credit <- 0
  for (l in c("A", "B")) {
    c_l <- col(after, "credit_loss", l, t)
    stopifnot(identical(col(after, "recovered", l, t), col(term, "ceded", l, t) - c_l))
    credit <- credit + c_l
  }
  by_total <- col(by, "credit_loss", "R1", t) + col(by, "credit_loss", "R2", t)
  near(credit, by_total, abs = 1e-9)
  near(col(after, "net_after_default", "retained", t), col(term, "net", "retained", t) + by_total,
       abs = 1e-9)
}

# The errors.
term <- apply_tower_term(reinsurance_tower(list(xol_layer("A", 20, 15))),
                         list(events_from_years(list(10, 40)), events_from_years(list(30, 5))))
years <- apply_tower(reinsurance_tower(list(xol_layer("A", 20, 15))), c(10, 40))
small <- panel_line(reinsurer_panel(list(reinsurer("R1", 0.1, 0.4))), "R1", "A", 1)
expect_error_like(apply_panel(small, term), "apply_term")
expect_error_like(apply_panel_term(small, years, 1), "term")
expect_error_like(apply_panel_term(small, term, 3), "year")
expect_error_like(apply_panel_term(small, term, 0), "at least one year")
expect_error_like(apply_panel_term(small, term, 2^40), "needs components in every year")
expect_error_like(panel_by_reinsurer_term(small, term, 2^53 - 1), "needs components in every year")
expect_error_like(apply_panel_term(small, term, 1.5), "n_years")
expect_error_like(apply_panel_term(small, term, 2, seed = -1), "seed")
expect_error_like(apply_panel_term(panel_driver(small, c(1, 2)), term, 2), "path-year")
expect_error_like(panel_by_reinsurer_term("small", term, 2), "panel must be a reinsurer_panel")
expect_error_like(apply_panel_term(small, 1:3, 2), "result must be a predictive_distribution")
