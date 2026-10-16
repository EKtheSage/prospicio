suppressMessages(library(prospicio))

# bayes_odp_one_year() and bayes_odp_one_year_segments(), the one-year view
# of the Bayesian ODP chain ladder by PSIS reweighting
# (docs/design/reserving-bayes.md, decision 16). The comparison with the ODP
# bootstrap over seeds and the reweighting against refits are in
# validation/tests/reserving_bayes_one_year.rs; these tests check the
# binding.

near <- function(a, b, tol) {
  if (!all(abs(a - b) <= tol)) {
    stop(sprintf("got %s, want %s (tol %g)", format(a, digits = 17), format(b, digits = 17), tol),
         call. = FALSE)
  }
}

expect_error_like <- function(expr, pattern) {
  msg <- tryCatch({
    expr
    NULL
  }, error = conditionMessage)
  if (is.null(msg)) stop("expected an error", call. = FALSE)
  if (!grepl(pattern, msg, fixed = TRUE)) {
    stop(sprintf("error %s does not mention %s", shQuote(msg), shQuote(pattern)), call. = FALSE)
  }
}

# The warnings `expr` raises, and its value.
with_warnings <- function(expr) {
  found <- character()
  value <- withCallingHandlers(expr, warning = function(w) {
    found <<- c(found, conditionMessage(w))
    invokeRestart("muffleWarning")
  })
  list(value = value, warnings = found)
}

# Tests run from the repository root (CI and `cargo xtask r`).
long <- utils::read.csv(file.path("validation", "data", "genins.csv"), comment.char = "#")
genins <- triangle(long, "origin", "development", "value")
tiny_long <- data.frame(year = rep(2020:2023, 4:1),
                        age = c(12, 24, 36, 48, 12, 24, 36, 12, 24, 12),
                        paid = c(100, 150, 165, 170, 110, 170, 180, 120, 175, 130))
tiny <- triangle(tiny_long, "year", "age", "paid")
two_long <- rbind(cbind(lob = "auto", tiny_long), cbind(lob = "home", transform(tiny_long,
                                                                                 paid = 2 * paid)))
two <- triangle(two_long, "year", "age", "paid", keys = "lob")

n_sims <- 2000

# GenIns passes every diagnostic with the default chains: no warning.
run <- with_warnings(bayes_odp_one_year(genins, n_sims = n_sims, seed = 3,
                                        refit = "reweight_only"))
view <- run$value
cdr_rows <- draw_matrix(view@cdr)
total <- rowSums(cdr_rows)
stopifnot(
  length(run$warnings) == 0,
  S7::S7_inherits(view, bayes_one_year_fit),
  S7::S7_inherits(view@fit, bayes_odp_fit),
  S7::S7_inherits(view@chain_ladder, chain_ladder_fit),
  identical(view@origins, as.character(2001:2010)),
  identical(view@development, as.integer(12 * 1:10)),
  identical(names(view@opening_reserve), view@origins),
  identical(unname(view@latest), unname(view@chain_ladder@latest)),
  S7::S7_inherits(view@cdr, predictive_distribution),
  identical(view@cdr@n_sims, n_sims),
  identical(dim(cdr_rows), c(as.integer(n_sims), 10L)),
  # 2001 is fully developed: no cell in the year, no change.
  all(cdr_rows[, 1] == 0),
  length(view@pareto_k) == n_sims,
  identical(view@refitted, integer()),
  identical(view@moment_matched, integer()),
  identical(view@refit_warnings, 0L),
  identical(view@k_threshold, 0.7),
  isTRUE(all.equal(view@flagged_share, mean(is.nan(view@pareto_k) | view@pareto_k > 0.7))),
  view@flagged_share > 0.1, view@flagged_share < 0.45,
  sqrt(variance(view@cdr)) < sqrt(variance(view@fit@reserves))
)
near(unname(view@opening_ultimate), unname(view@latest + view@opening_reserve),
     1e-9 * max(view@opening_ultimate))
# The mean is zero up to Monte Carlo error.
stopifnot(abs(mean(total)) < 5 * sd(total) / sqrt(n_sims))
near(sum(view@opening_reserve), mean(view@fit@reserves), 0.02 * mean(view@fit@reserves))

# The lifetime fit is bayes_odp()'s with the same settings.
fit <- bayes_odp(genins, n_sims = n_sims, seed = 3)
stopifnot(identical(view@fit@draws, fit@draws),
          identical(draw_matrix(view@fit@reserves), draw_matrix(fit@reserves)))

# Loosely the ODP bootstrap's one-year view (the validation test pins it).
boot <- odp_one_year(genins, n_sims = n_sims, seed = 3)
near(sd(total), sqrt(variance(boot@cdr)), 0.12 * sqrt(variance(boot@cdr)))

# Frames.
df <- as.data.frame(view)
stopifnot(
  identical(names(df), c("origin", "latest", "opening_ultimate", "opening_reserve", "cdr_mean",
                         "cdr_std_dev")),
  nrow(df) == 10
)
tot <- totals_frame(view)
stopifnot(nrow(tot) == 1, all(c("scale", "flagged_share", "cdr_std_dev") %in% names(tot)),
          isTRUE(all.equal(tot$flagged_share, view@flagged_share)))
near(tot$cdr_std_dev, sqrt(variance(view@cdr)), 1e-9 * sd(total))
stopifnot(nrow(development_frame(view)) == 10,
          S7::S7_inherits(segment(view), bayes_one_year_fit))
out <- capture.output(print(view))
stopifnot(grepl("<bayes_one_year_fit> 2000 simulations", out[1], fixed = TRUE),
          grepl("Pareto k above 0.7", out[2], fixed = TRUE))

# Replay by seed.
a <- bayes_odp_one_year(tiny, n_sims = 300, seed = 11, refit = "reweight_only")
b <- bayes_odp_one_year(tiny, n_sims = 300, seed = 11, refit = "reweight_only")
stopifnot(identical(draw_matrix(a@cdr), draw_matrix(b@cdr)), identical(a@pareto_k, b@pareto_k))

# Refits: the flagged simulations (1-based), or every one.
short <- function(...) {
  suppressWarnings(bayes_odp_one_year(tiny, tune = 200, draws = 200, n_sims = 12, seed = 2, ...))
}
only <- short(refit = "reweight_only")
flagged <- short(k_threshold = 0.3, refit = "refit_flagged")
every <- short(refit = "refit_all")
expected <- which(is.nan(flagged@pareto_k) | flagged@pareto_k > 0.3)
stopifnot(
  identical(flagged@refitted, expected),
  identical(flagged@k_threshold, 0.3),
  identical(every@refitted, 1:12),
  identical(only@pareto_k, flagged@pareto_k),
  identical(only@pareto_k, every@pareto_k),
  # A refitted row is refit_all's, the others reweight_only's.
  identical(draw_matrix(only@cdr)[-expected, , drop = FALSE],
            draw_matrix(flagged@cdr)[-expected, , drop = FALSE]),
  identical(draw_matrix(every@cdr)[expected, , drop = FALSE],
            draw_matrix(flagged@cdr)[expected, , drop = FALSE]),
  identical(flagged@moment_matched, integer())
)

# Moment matching: the same years, each flagged simulation moment-matched
# or refitted (never both), a refitted row refit_all's and an unflagged one
# reweight_only's.
matched <- short(k_threshold = 0.3, refit = "moment_match_then_refit")
# Moment matching is the default (decision 18).
default <- short(k_threshold = 0.3)
unflagged <- setdiff(1:12, expected)
stopifnot(
  identical(draw_matrix(default@cdr), draw_matrix(matched@cdr)),
  identical(default@moment_matched, matched@moment_matched),
  identical(matched@pareto_k, only@pareto_k),
  length(intersect(matched@moment_matched, matched@refitted)) == 0,
  identical(sort(c(matched@moment_matched, matched@refitted)), expected),
  identical(draw_matrix(matched@cdr)[matched@refitted, , drop = FALSE],
            draw_matrix(every@cdr)[matched@refitted, , drop = FALSE]),
  identical(draw_matrix(matched@cdr)[unflagged, , drop = FALSE],
            draw_matrix(only@cdr)[unflagged, , drop = FALSE])
)

# Twenty draws per chain cannot reach the ESS threshold: the fit and its
# refits warn.
run <- with_warnings(bayes_odp_one_year(tiny, chains = 2, tune = 20, draws = 20, n_sims = 3,
                                        seed = 1, refit = "refit_all"))
refit_msgs <- grep("refits in the one-year view", run$warnings, fixed = TRUE, value = TRUE)
stopifnot(
  any(grepl("the MCMC diagnostics failed", run$warnings, fixed = TRUE)),
  length(refit_msgs) == 1,
  run$value@refit_warnings > 0,
  grepl(sprintf("%d of 3 refits", run$value@refit_warnings), refit_msgs, fixed = TRUE)
)

# Segments.
views <- suppressWarnings(bayes_odp_one_year_segments(two, tune = 300, draws = 300,
                                                      n_sims = 1000, seed = 4,
                                                      refit = "reweight_only"))
stopifnot(
  S7::S7_inherits(views, bayes_one_year_fits),
  identical(views@keys, "lob"),
  identical(views@origins, rep(as.character(2020:2023), 2)),
  identical(names(views@cdr@keys), c("lob", "origin")),
  length(views@opening_reserve) == 8,
  identical(unname(views@latest), c(170, 180, 175, 130, 340, 360, 350, 260)),
  identical(views@k_threshold, 0.7),
  nrow(views@diagnostics) == 2,
  nrow(as.data.frame(views)) == 8,
  identical(totals_frame(views)$lob, c("auto", "home")),
  "flagged_share" %in% names(totals_frame(views))
)
for (field in c("scale", "pareto_k", "refitted", "moment_matched", "refit_warnings",
                "flagged_share")) {
  expect_error_like(S7::prop(views, field), "single-segment")
}
home <- segment(views, lob = "home")
stopifnot(
  S7::S7_inherits(home, bayes_one_year_fits),
  length(home@pareto_k) == 1000,
  identical(home@refitted, integer()),
  identical(home@moment_matched, integer()),
  home@scale > 0,
  isTRUE(all.equal(home@flagged_share, totals_frame(views)$flagged_share[2]))
)
out <- capture.output(print(views))
stopifnot(grepl("<bayes_one_year_fits> 1000 simulations, 2 segments", out[1], fixed = TRUE))

# Rank correlation between the segments' total CDR.
ranked <- suppressWarnings(bayes_odp_one_year_segments(
  two, tune = 300, draws = 300, n_sims = 2000, seed = 6, refit = "reweight_only",
  dependence = "rank_correlation", spearman = matrix(c(1, 0.8, 0.8, 1), 2)))
lines <- draw_matrix(aggregate(ranked@cdr, keep = "lob"))
near(stats::cor(lines[, 1], lines[, 2], method = "spearman"), 0.8, 0.05)

# Errors.
expect_error_like(bayes_odp_one_year(genins, n_sims = 10, refit = "sometimes"), "should be one of")
expect_error_like(bayes_odp_one_year(genins, n_sims = 10, k_threshold = NaN), "k_threshold")
expect_error_like(bayes_odp_one_year_segments(two, n_sims = 10, dependence = "synchronized"),
                  "ynchroni")

# bayes_bf_one_year() (decision 17): the same class, with the bayes_bf_fit.
# Its comparison with bayes_odp and the strong-prior limit is in
# validation/tests/reserving_bayes_bf_one_year.rs.
prem_long <- transform(tiny_long, premium = rep(c(200, 220, 240, 260), 4:1))
prem <- triangle(prem_long, "year", "age", c("paid", "premium"))
bf_view <- suppressWarnings(bayes_bf_one_year(
  prem, "paid", "premium", apriori = 0.85, cv = 0.1, tune = 300, draws = 300, n_sims = 1000,
  seed = 3, refit = "reweight_only"))
bf_fit <- suppressWarnings(bayes_bf(prem, "paid", "premium", apriori = 0.85, cv = 0.1,
                                    tune = 300, draws = 300, n_sims = 1000, seed = 3))
stopifnot(
  S7::S7_inherits(bf_view, bayes_one_year_fit),
  S7::S7_inherits(bf_view@fit, bayes_bf_fit),
  identical(bf_view@fit@draws, bf_fit@draws),
  identical(draw_matrix(bf_view@fit@reserves), draw_matrix(bf_fit@reserves)),
  length(bf_view@pareto_k) == 1000,
  identical(bf_view@refitted, integer()),
  all(draw_matrix(bf_view@cdr)[, 1] == 0),
  variance(bf_view@cdr) < variance(bf_fit@reserves),
  isTRUE(all.equal(as.data.frame(bf_view)$prior_ultimate, bf_fit@prior_ultimate)),
  "cdr_std_dev" %in% names(as.data.frame(bf_view)),
  isTRUE(all.equal(totals_frame(bf_view)$flagged_share, bf_view@flagged_share))
)
out <- capture.output(print(bf_view))
stopifnot(grepl("<bayes_one_year_fit> 1000 simulations", out[1], fixed = TRUE))
expect_error_like(bayes_bf_one_year(prem, "paid", "premium", n_sims = 10, k_threshold = NaN),
                  "k_threshold")

# meyers_one_year() and meyers_one_year_segments() (decision 19): the same
# classes, with the meyers_fit. Their checks against refits are in
# validation/tests/reserving_meyers_one_year.rs.
m_long <- data.frame(
  year = rep(1988:1993, 6:1),
  age = 12 * sequence(6:1),
  paid = c(952, 1529, 2813, 3647, 3724, 3832, 849, 1564, 2202, 2432, 2468,
           983, 2211, 2830, 3832, 1657, 2685, 3169, 932, 1940, 1162),
  premium = rep(c(5812, 4908, 5454, 5165, 5214, 5230), 6:1))
m_tri <- triangle(m_long, "year", "age", c("paid", "premium"))
m_view <- suppressWarnings(meyers_one_year(m_tri, "paid", "premium", model = "cay", tune = 300,
                                           draws = 300, n_sims = 600, seed = 5,
                                           refit = "reweight_only"))
m_fit <- suppressWarnings(meyers(m_tri, "paid", "premium", model = "cay", tune = 300,
                                 draws = 300, n_sims = 600, seed = 5))
m_default <- suppressWarnings(meyers_one_year(m_tri, "paid", "premium", model = "cay",
                                              tune = 300, draws = 300, n_sims = 40, seed = 5))
m_flagged <- which(is.nan(m_default@pareto_k) | m_default@pareto_k > 0.7)
stopifnot(
  S7::S7_inherits(m_view, bayes_one_year_fit),
  S7::S7_inherits(m_view@fit, meyers_fit),
  identical(m_view@fit@draws, m_fit@draws),
  identical(draw_matrix(m_view@fit@reserves), draw_matrix(m_fit@reserves)),
  identical(m_view@origins, as.character(1988:1993)),
  length(m_view@pareto_k) == 600,
  identical(m_view@refitted, integer()),
  all(draw_matrix(m_view@cdr)[, 1] == 0),
  variance(m_view@cdr) < variance(m_fit@reserves),
  isTRUE(all.equal(as.data.frame(m_view)$premium, c(5812, 4908, 5454, 5165, 5214, 5230))),
  identical(m_default@pareto_k, m_view@pareto_k[1:40]),
  identical(sort(c(m_default@moment_matched, m_default@refitted)), m_flagged)
)
# The opening reserve is the lifetime reserve's posterior mean, within Monte
# Carlo error (the total is heavy-tailed on six origins).
m_total <- rowSums(draw_matrix(m_fit@reserves))
near(sum(m_view@opening_reserve), mean(m_total), 5 * sd(m_total) / sqrt(length(m_total)))

m_two_long <- rbind(cbind(lob = "auto", m_long),
                    cbind(lob = "home", transform(m_long, paid = 2 * paid, premium = 2 * premium)))
m_two <- triangle(m_two_long, "year", "age", c("paid", "premium"), keys = "lob")
m_views <- suppressWarnings(meyers_one_year_segments(m_two, "paid", "premium", tune = 200,
                                                     draws = 200, n_sims = 500, seed = 2,
                                                     refit = "reweight_only"))
m_home <- segment(m_views, lob = "home")
stopifnot(
  S7::S7_inherits(m_views, bayes_one_year_fits),
  identical(m_views@cdr@dims, c("lob", "origin")),
  nrow(m_views@diagnostics) == 2,
  isTRUE(all.equal(m_home@premium, 2 * c(5812, 4908, 5454, 5165, 5214, 5230))),
  length(m_home@pareto_k) == 500,
  "premium" %in% names(totals_frame(m_views))
)
expect_error_like(m_home@scale, "scale")
out <- capture.output(print(m_view))
stopifnot(grepl("<bayes_one_year_fit> 600 simulations, Meyers' CAY", out[1], fixed = TRUE))
out <- capture.output(print(m_home))
stopifnot(grepl("<bayes_one_year_fits> 500 simulations", out[1], fixed = TRUE))
invisible(capture.output(print(m_views)))
expect_error_like(meyers_one_year(m_tri, "paid", "premium", n_sims = 10, k_threshold = NaN),
                  "k_threshold")
expect_error_like(meyers_one_year_segments(m_two, "paid", "premium", n_sims = 10,
                                           dependence = "synchronized"), "ynchroni")
