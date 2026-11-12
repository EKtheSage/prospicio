suppressMessages(library(prospicio))
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# bayes_hierarchical_bf(), the hierarchical Bayesian Bornhuetter-Ferguson over
# programs (docs/design/reserving-bayes.md, decision 30). The model's limits
# are checked in the Rust crate and in validation/tests; these tests check
# the binding on a small panel of three programs, one of them thin (two
# origins): the fit's fields, that the programs' reserves add up to the joint
# total, replay by seed, the pattern prior, the diagnostics warning and the
# errors.



# The warnings `expr` raises, and its value.

# Programs a (four origins), b (three) and c (two, a thin one), ages 12 to 48.
programs <- list(
  a = list(`2020` = list(c(60, 100, 125, 135), 200), `2021` = list(c(66, 108, 133), 210),
           `2022` = list(c(70, 115), 215), `2023` = list(72, 220)),
  b = list(`2021` = list(c(35, 60, 75), 120), `2022` = list(c(38, 64), 125),
           `2023` = list(40, 130)),
  c = list(`2022` = list(c(22, 38), 80), `2023` = list(25, 85))
)
panel <- function(programs) {
  rows <- do.call(rbind, lapply(names(programs), function(p) {
    do.call(rbind, lapply(names(programs[[p]]), function(y) {
      cells <- programs[[p]][[y]][[1]]
      data.frame(program = p, year = as.integer(y), age = 12 * seq_along(cells), paid = cells,
                 premium = programs[[p]][[y]][[2]])
    }))
  }))
  triangle(rows, "year", "age", c("paid", "premium"), keys = "program")
}
tri <- panel(programs)

quick <- list(chains = 2, tune = 300, draws = 300, n_sims = 600)
run <- with_warnings(do.call(bayes_hierarchical_bf,
  c(list(tri, "paid", "premium", apriori = 0.7, level_cv = 0.3, seed = 3), quick)))
fit <- run$value

stopifnot(
  S7::S7_inherits(fit, hierarchical_bf_fit),
  identical(fit@ages, c(12L, 24L, 36L, 48L)),
  fit@scale_per_premium > 0,
  identical(names(fit@programs), c("a", "b", "c")),
  all(vapply(fit@programs, function(p) S7::S7_inherits(p, hierarchical_bf_program), TRUE)),
  identical(vapply(fit@programs, function(p) length(p@origins), 0L, USE.NAMES = FALSE),
            c(4L, 3L, 2L))
)
for (p in fit@programs) {
  n <- length(p@origins)
  stopifnot(
    length(p@exposure) == n, length(p@latest) == n, length(p@prior_ultimate) == n,
    length(p@expected_ultimate) == n, length(p@pattern) == 4,
    all(p@pattern > 0), p@scale > 0, p@loss_ratio > 0, p@own_loss_ratio > 0,
    is.finite(p@shrinkage),
    S7::S7_inherits(p@reserves, predictive_distribution), p@reserves@n_sims == 600
  )
  near(sum(p@pattern), 1, 1e-9)
}
thin <- fit@programs$c
stopifnot(identical(thin@origins, c("2022", "2023")),
          identical(thin@exposure, c(80, 85)), identical(thin@latest, c(38, 25)),
          mean(thin@reserves) > 0)
near(thin@prior_ultimate, 0.7 * c(80, 85), 1e-12)
near(thin@scale, fit@scale_per_premium * (80 + 85) / 2, 1e-9)

# The programs' reserves add up to the joint total.
r <- fit@reserves
stopifnot(S7::S7_inherits(r, predictive_distribution), r@n_sims == 600,
          identical(r@dims, c("program", "origin")))
near(sum(vapply(fit@programs, function(p) mean(p@reserves), 0)), mean(r), 1e-9)
stopifnot(mean(r) > 0)
# Paired by posterior draw: each program's draws are the joint draws' columns
# for that program, simulation by simulation.
joint <- draw_matrix(r)
stopifnot(nrow(joint) == 600, ncol(joint) == nrow(r@keys))
for (label in names(fit@programs)) {
  own <- draw_matrix(fit@programs[[label]]@reserves)
  columns <- which(r@keys[[1]] == label)
  stopifnot(length(columns) == length(fit@programs[[label]]@origins),
            identical(unname(joint[, columns]), unname(own)))
}

# The posterior and its frames.
nm <- fit@ptr$parameter_names()
stopifnot(identical(nm[1:2], c("log_level", "log_tau")), "log_omega" %in% nm,
          identical(dim(fit@draws), c(600L, length(nm))),
          identical(colnames(fit@draws), nm))
s <- fit@summary
stopifnot(identical(names(s), c("parameter", "mean", "sd", "q05", "q50", "q95", "rhat",
                                "ess_bulk", "ess_tail")),
          identical(s$parameter, nm))
h <- fit@hyperparameters
stopifnot(identical(h$parameter, c("level", "tau", "omega")),
          all(h$mean > 0), all(h$sd > 0), all(h$q05 > 0))
f <- as.data.frame(fit)
stopifnot(identical(f$program, c("a", "b", "c")), identical(f$origins, c(4L, 3L, 2L)),
          identical(names(f), c("program", "origins", "exposure", "latest", "prior_ultimate",
                                "expected_ultimate", "scale", "loss_ratio", "own_loss_ratio",
                                "shrinkage", "mean", "std_dev")),
          all(f$std_dev > 0))
near(f$exposure[3], 165, 1e-12)
near(sum(f$mean), mean(r), 1e-9)
stopifnot(identical(summary(fit), s), is.list(fit@diagnostics))
invisible(utils::capture.output(print(fit), print(fit@programs$a)))

# Replay by seed.
quiet <- function(...) with_warnings(do.call(bayes_hierarchical_bf, list(...)))$value
a <- quiet(tri, "paid", "premium", seed = 5, chains = 2, tune = 300, draws = 300, n_sims = 600)
b <- quiet(tri, "paid", "premium", seed = 5, chains = 2, tune = 300, draws = 300, n_sims = 600)
d <- quiet(tri, "paid", "premium", seed = 6, chains = 2, tune = 300, draws = 300, n_sims = 600)
stopifnot(identical(a@draws, b@draws), identical(mean(a@reserves), mean(b@reserves)),
          !identical(a@draws, d@draws))

# One segment is one program.
one <- quiet(panel(programs["a"]), "paid", "premium", seed = 1, chains = 2, tune = 300,
             draws = 300, n_sims = 600)
stopifnot(identical(names(one@programs), "a"), one@reserves@n_sims == 600)

# A pattern prior on the triangle's ages.
prior <- pattern_prior(c(12, 24, 36, 48), c(0.4, 0.6, 0.7), c(0.3, 0.3, 0.3))
withp <- quiet(tri, "paid", "premium", seed = 2, pattern_prior = prior, chains = 2, tune = 300,
               draws = 300, n_sims = 600)
stopifnot(length(withp@programs) == 3)
wrong <- pattern_prior(c(12, 24), 0.4, 0.3)
expect_error_like(bayes_hierarchical_bf(tri, "paid", "premium", pattern_prior = wrong, chains = 2,
                                        tune = 100, draws = 100, n_sims = 100), "age")

# A run too short to converge: one warning, the fit still returned.
short <- with_warnings(bayes_hierarchical_bf(tri, "paid", "premium", chains = 2, tune = 20,
                                             draws = 20, n_sims = 100, seed = 1))
stopifnot(length(short$warnings) == 1,
          grepl("bayes_hierarchical_bf: the MCMC diagnostics failed", short$warnings),
          length(short$value@diagnostics$warnings) > 0)

# Errors.
expect_error_like(bayes_hierarchical_bf(tri, "paid", "premium", n_sims = 0), "n_sims")
expect_error_like(bayes_hierarchical_bf(tri, "paid", "premium", process = "lognormal"),
                  "gamma")
expect_error_like(bayes_hierarchical_bf(tri, "paid", "missing", chains = 2, tune = 100,
                                        draws = 100, n_sims = 100), "missing")
expect_error_like(bayes_hierarchical_bf(tri, "paid", "premium", level_cv = "a"), "level_cv")
for (arg in c("apriori", "cv", "tau_scale", "omega_scale")) {
  args <- list(tri, "paid", "premium", chains = 2, tune = 100, draws = 100, n_sims = 100)
  args[[arg]] <- 0
  msg <- tryCatch({
    do.call(bayes_hierarchical_bf, args)
    NULL
  }, error = conditionMessage)
  if (is.null(msg)) stop(sprintf("%s = 0 should fail", arg), call. = FALSE)
}

# A program whose column falls by many of its scales has no posterior: the error
# names the program and omega_scale.
falling <- list(
  a = programs$a,
  b = list(`2021` = list(c(100, 20), 100), `2022` = list(100, 100))
)
expect_error_like(bayes_hierarchical_bf(panel(falling), "paid", "premium", chains = 2,
                                        tune = 100, draws = 100, n_sims = 100),
                  "b at 24 months")
expect_error_like(bayes_hierarchical_bf(panel(falling), "paid", "premium", chains = 2,
                                        tune = 100, draws = 100, n_sims = 100),
                  "omega_scale")

# The default acceptance rate is the Rust model's.
stopifnot(formals(bayes_hierarchical_bf)$target_accept == 0.99)

# A named list of triangles, one program each, equals the panel with a program
# key, draw for draw, whatever the list's order.
by_name <- setNames(lapply(names(programs), function(n) panel(programs[n])), names(programs))
settings <- list(seed = 4, chains = 2, tune = 300, draws = 300, n_sims = 600)
whole <- do.call(quiet, c(list(tri, "paid", "premium"), settings))
stacked <- do.call(quiet, c(list(by_name, "paid", "premium"), settings))
flipped <- do.call(quiet, c(list(rev(by_name), "paid", "premium"), settings))
stopifnot(identical(names(stacked@programs), c("a", "b", "c")),
          identical(stacked@draws, whole@draws),
          identical(draw_matrix(stacked@reserves), draw_matrix(whole@reserves)),
          identical(names(flipped@programs), c("a", "b", "c")),
          identical(flipped@draws, whole@draws))

# Errors of the list form.
go <- function(x, column = "paid") {
  bayes_hierarchical_bf(x, column, "premium", chains = 2, tune = 100, draws = 100, n_sims = 100)
}
expect_error_like(go(list()), "no programs")
expect_error_like(go(unname(by_name)), "needs a name")
expect_error_like(go(setNames(by_name[1:2], c("a", ""))), "needs a name")
expect_error_like(go(setNames(by_name[1:2], c("a", "a"))), "used twice")
expect_error_like(go(list(a = by_name$a, b = 3)), "program b is not a triangle")
expect_error_like(go(list(x = tri, b = by_name$b)), 'program "x" has 3 segments')
expect_error_like(go(by_name, "missing"), 'program "a" has no column "missing"')
quarterly <- triangle(data.frame(year = 2022, age = c(3, 6), paid = c(1, 2), premium = 9),
                      "year", "age", c("paid", "premium"), origin_grain = "Q",
                      development_grain = "Q")
expect_error_like(go(list(a = by_name$a, q = quarterly)), 'program "q" differs')
incremental <- triangle(data.frame(year = 2022, age = c(12, 24), paid = c(1, 2), premium = 9),
                        "year", "age", c("paid", "premium"), cumulative = FALSE)
expect_error_like(go(list(a = by_name$a, i = incremental)), 'program "i" differs')
# A column named program clashes with the stacked key, in R and Python (#838).
with_program <- triangle(data.frame(year = 2022, age = c(12, 24), paid = c(1, 2), premium = 9,
                                    program = 1), "year", "age", c("paid", "premium", "program"))
expect_error_like(go(list(a = with_program, b = by_name$b), "program"), "clashes with the key")

# The one-year view, bayes_hierarchical_bf_one_year(): one view over the
# stacked programs. A few simulations with the reweighting only (the refits of
# a whole panel are the Rust suite's).
yq <- list(chains = 2, tune = 200, draws = 200, n_sims = 400, one_year_sims = 40,
           refit = "reweight_only")
vrun <- with_warnings(do.call(bayes_hierarchical_bf_one_year,
  c(list(tri, "paid", "premium", apriori = 0.7, level_cv = 0.3, seed = 5), yq)))
view <- vrun$value
n_cells <- 4 + 3 + 2
stopifnot(
  S7::S7_inherits(view, hierarchical_bf_one_year_fit),
  S7::S7_inherits(view@fit, hierarchical_bf_fit),
  identical(view@program, c(rep("a", 4), rep("b", 3), rep("c", 2))),
  identical(view@origins, c("2020", "2021", "2022", "2023", "2021", "2022", "2023", "2022", "2023")),
  length(view@latest) == n_cells, length(view@opening_ultimate) == n_cells,
  length(view@opening_reserve) == n_cells,
  S7::S7_inherits(view@cdr, predictive_distribution), view@cdr@n_sims == 40,
  identical(view@cdr@dims, c("program", "origin")),
  length(view@pareto_k) == 40, length(view@refitted) == 0, length(view@moment_matched) == 0,
  view@k_threshold == 0.7, view@flagged_share >= 0, view@flagged_share <= 1
)
near(view@opening_ultimate, view@latest + view@opening_reserve, 1e-9)
# By program, the CDR aggregates to the program's rows.
by_program <- aggregate(view@cdr, "program")
stopifnot(S7::S7_inherits(by_program, predictive_distribution),
          identical(by_program@dims, "program"))

frame <- as.data.frame(view)
stopifnot(nrow(frame) == n_cells,
          identical(names(frame), c("program", "origin", "latest", "opening_ultimate",
                                    "opening_reserve", "cdr_mean", "cdr_std_dev")),
          identical(frame$program, view@program), all(frame$cdr_std_dev >= 0), sum(frame$cdr_std_dev > 0) == n_cells - 1)

# A seed replays; the list form of named triangles is the stacked panel.
again <- suppressWarnings(do.call(bayes_hierarchical_bf_one_year,
  c(list(tri, "paid", "premium", apriori = 0.7, level_cv = 0.3, seed = 5), yq)))
stopifnot(identical(draw_matrix(again@cdr), draw_matrix(view@cdr)),
          identical(again@pareto_k, view@pareto_k))
listed <- suppressWarnings(do.call(bayes_hierarchical_bf_one_year,
  c(list(by_name, "paid", "premium", apriori = 0.7, level_cv = 0.3, seed = 5), yq)))
stopifnot(identical(draw_matrix(listed@cdr), draw_matrix(view@cdr)))

# Fewer simulations than the fit's; the view prints.
out <- capture.output(print(view))
stopifnot(any(grepl("hierarchical_bf_one_year_fit", out, fixed = TRUE)))

# Errors.
expect_error_like(bayes_hierarchical_bf_one_year(tri, "paid", "premium", refit = "sometimes"),
                  "should be one of")
expect_error_like(do.call(bayes_hierarchical_bf_one_year,
  c(list(tri, "paid", "premium", k_threshold = NaN), yq)), "k_threshold")

# Programs b and c stop at 36 and 24 months, short of a's 48: their next
# diagonals fall at ages they never observed, which the view warns of (#762).
stopifnot(length(view@warnings) == 1,
          grepl("b (observed to age 36, its next diagonal to 48)", view@warnings, fixed = TRUE),
          grepl("c (observed to age 24, its next diagonal to 36)", view@warnings, fixed = TRUE),
          !grepl("a (", view@warnings, fixed = TRUE),
          # Reweighting only (yq), the view is approximate, not expensive.
          grepl("approximate", view@warnings, fixed = TRUE),
          !grepl("expensive", view@warnings, fixed = TRUE),
          any(vrun$warnings == paste0("bayes_hierarchical_bf_one_year: ", view@warnings)))
# Programs that share their ages do not warn.
shared <- panel(list(
  a = list(`2021` = list(c(60, 100, 125), 200), `2022` = list(c(66, 108), 210),
           `2023` = list(70, 215)),
  b = list(`2021` = list(c(35, 60, 75), 120), `2022` = list(c(38, 64), 125),
           `2023` = list(40, 130))
))
even <- with_warnings(do.call(bayes_hierarchical_bf_one_year,
  c(list(shared, "paid", "premium", apriori = 0.7, level_cv = 0.3, seed = 5), yq)))
stopifnot(length(even$value@warnings) == 0,
          !any(grepl("never observed", even$warnings, fixed = TRUE)))

# A light refit reaches the view: every simulation refitted with a shorter,
# warm-started sampler, recorded in the provenance; a bad one is an error.
lq <- list(chains = 2, tune = 150, draws = 150, n_sims = 100, one_year_sims = 3,
           refit = "refit_all")
light <- suppressWarnings(do.call(bayes_hierarchical_bf_one_year,
  c(list(tri, "paid", "premium", apriori = 0.7, level_cv = 0.3, seed = 5,
         light_refit = c(2, 60, 60)), lq)))
stopifnot(identical(light@refitted, 1:3),
          grepl("expensive", light@warnings, fixed = TRUE),
          identical(provenance(light@cdr)$parameters[["light_refit"]], "2x60+60"))
plain <- suppressWarnings(do.call(bayes_hierarchical_bf_one_year,
  c(list(tri, "paid", "premium", apriori = 0.7, level_cv = 0.3, seed = 5,
         light_refit = TRUE), yq)))
# The default refits with the light refit (#762); a refit mode chosen keeps
# the fit's own sampler unless a light refit is given.
dflt <- suppressWarnings(do.call(bayes_hierarchical_bf_one_year,
  c(list(tri, "paid", "premium", apriori = 0.7, level_cv = 0.3, seed = 5),
    lq[names(lq) != "refit"])))
chosen <- suppressWarnings(do.call(bayes_hierarchical_bf_one_year,
  c(list(tri, "paid", "premium", apriori = 0.7, level_cv = 0.3, seed = 5), lq)))
stopifnot(identical(provenance(dflt@cdr)$parameters[["light_refit"]], "4x150+250"),
          is.null(provenance(chosen@cdr)$parameters[["light_refit"]]))
# A view that reweights only records no sampler.
stopifnot(is.null(provenance(plain@cdr)$parameters[["light_refit"]]))
expect_error_like(do.call(bayes_hierarchical_bf_one_year,
  c(list(tri, "paid", "premium", light_refit = c(0, 60, 60)), lq)), "light_refit")
expect_error_like(do.call(bayes_hierarchical_bf_one_year,
  c(list(tri, "paid", "premium", light_refit = "fast"), lq)), "light_refit")
