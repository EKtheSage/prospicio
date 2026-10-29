suppressMessages(library(prospicio))

# A development pattern per simulation (reserving-v02.md decision 28): row i
# is simulation i of the fit, the pattern ends at 1, and the monthly form
# feeds the cash-flow engine.

validation <- function(...) file.path("validation", ...)
raa_long <- utils::read.csv(validation("data", "raa.csv"), comment.char = "#")
raa <- triangle(raa_long, "origin", "development", "value")
cl <- chain_ladder(raa)
cdf <- unname(cl@cdf)
n_age <- length(cdf)

fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")

check_fit <- function(fit, n_sims, tol) {
  p <- pattern_draws(fit)
  stopifnot(nrow(p) == n_sims, ncol(p) %in% c(n_age, n_age + 1), !anyNA(p))
  stopifnot(identical(colnames(p)[1:3], c("12", "24", "36")))
  # Every row ends at exactly 1 (with no tail, at the oldest age).
  stopifnot(all(p[, ncol(p)] == 1))
  # The mean pattern is near the chain ladder's (the pseudo factors are
  # centred on it).
  stopifnot(max(abs(colMeans(p)[seq_len(n_age)] - 1 / cdf)) < tol)

  m <- monthly_pattern_draws(fit, 120, monotone = "running_max")
  stopifnot(nrow(m) == n_sims, ncol(m) == 120, all(m[, 120] == 1))
  stopifnot(all(m >= 0), all(m <= 1), all(m[, -1] - m[, -120] >= 0))
  # The default (Weibull) reading and the log-linear one differ inside the
  # ages and agree at the ends.
  ln <- monthly_pattern_draws(fit, 120, basis = "origin_ln_linear", monotone = "running_max")
  stopifnot(identical(dim(ln), dim(m)), all(ln[, 120] == 1), max(abs(ln - m)) > 1e-6)
  # Too few months is an error, not a truncation; more are padded with 1.
  stopifnot(fails(monthly_pattern_draws(fit, 12, monotone = "running_max")))
  longer <- monthly_pattern_draws(fit, 150, basis = "origin_ln_linear", monotone = "running_max")
  stopifnot(ncol(longer) == 150, identical(longer[, 1:120], ln), all(longer[, 121:150] == 1))
  # An underwriting year of policies written evenly over 12 months lags the
  # accident year.
  uw <- monthly_pattern_draws(fit, 144, basis = underwriting_basis(reading = "ln-linear"), monotone = "running_max")
  stopifnot(nrow(uw) == n_sims, all(uw[, 144] == 1), mean(uw[, 12]) < mean(ln[, 12]))
  # A one-segment fit answers to any label. Bad arguments:
  stopifnot(identical(pattern_draws(fit, "Auto"), pattern_draws(fit)))
  stopifnot(fails(monthly_pattern_draws(fit, 120, monotone = "none")),
            fails(monthly_pattern_draws(fit, 120, basis = "weibull")),
            fails(monthly_pattern_draws(fit, 120, basis = 3)),
            fails(monthly_pattern_draws(fit, 0)))
  # Same seed, same patterns; the draws are the fit's own simulations.
  invisible(p)
}

odp <- odp_bootstrap(raa, n_sims = 200, seed = 600)
p_odp <- check_fit(odp, 200, 0.05)
# (Some seeds give a Mack pseudo factor that is negative, so a non-positive
# share, which the monthly conversion refuses; this one does not.)
mack <- mack_bootstrap(raa, n_sims = 200, seed = 601)
check_fit(mack, 200, 0.06)

# The reserves are what they were: the same seed gives the same patterns.
stopifnot(identical(pattern_draws(odp_bootstrap(raa, n_sims = 200, seed = 600)), p_odp))

# A fit refuses a pattern of the wrong kind of object.
stopifnot(fails(pattern_draws(cl)), fails(pattern_draws(1)))

# Two segments by name.
two <- raa_long[rep(seq_len(nrow(raa_long)), 2), ]
two$lob <- rep(c("A", "B"), each = nrow(raa_long))
two$value <- two$value * rep(c(1, 2), each = nrow(raa_long))
tri2 <- triangle(two, "origin", "development", "value", keys = "lob")
fits <- odp_bootstrap(tri2, n_sims = 300, seed = 5)
stopifnot(fails(pattern_draws(fits)), fails(pattern_draws(fits, "C")))
pa <- pattern_draws(fits, "A")
pb <- pattern_draws(fits, "B")
stopifnot(nrow(pa) == 300, identical(dim(pa), dim(pb)), all(pa[, ncol(pa)] == 1))
# The segments are independent fits on their own seeds.
stopifnot(!identical(pa, pb))
ma <- monthly_pattern_draws(fits, 120, segment = "A", monotone = "running_max")
mb <- monthly_pattern_draws(fits, 120, segment = c("B"), monotone = "running_max")
stopifnot(identical(dim(ma), c(300L, 120L)), identical(dim(mb), c(300L, 120L)))
macks <- mack_bootstrap(tri2, n_sims = 200, seed = 5)
stopifnot(nrow(pattern_draws(macks, "B")) == 200)

# End to end: each simulation paid by its own pattern.
ramp <- function(m) pmin((1:120) / m, 1)
terms <- cashflow_terms(100, 60, ramp(12), ramp(36), ramp(48),
                        cashflow_requirement(c(reserve = 1)))
latest <- sum(cl@latest)
u <- loss_ratio_draws(odp, premium = latest / 0.7)
stopifnot(length(u) == 200)
patterns <- monthly_pattern_draws(odp, 120, monotone = "running_max")
run <- cashflow_run(terms, u, paid_patterns = patterns)
stopifnot(nrow(run$flows) == 200, !anyNA(run$flows))
# Identical patterns equal to the terms' own reproduce the terms' run.
same <- matrix(ramp(48), nrow = 200, ncol = 120, byrow = TRUE)
stopifnot(identical(cashflow_run(terms, u, paid_patterns = same)$flows, cashflow_run(terms, u)$flows))
# A different pattern per simulation moves the flows.
stopifnot(!identical(run$flows, cashflow_run(terms, u)$flows))
# The default refuses a falling pattern, naming it; running_max is the
# remedy, so when no simulation falls the two agree.
refuse <- try(monthly_pattern_draws(odp, 120), silent = TRUE)
if (!inherits(refuse, "try-error")) stopifnot(identical(refuse, patterns))

# Meyers' CRC, CSR and CAY (#678): each simulation's pattern is exp(beta_d g)
# of its posterior draw, ending at exactly 1 at the oldest age (no tail).
comauto <- utils::read.csv(validation("data", "clrd_comauto_353.csv"), comment.char = "#")
comauto <- comauto[comauto$origin - 1988 + comauto$development %/% 12 <= 10, ]
tri353 <- triangle(comauto, "origin", "development", c("paid", "premium"))
for (model in c("crc", "csr", "cay")) {
  fit <- suppressWarnings(meyers(tri353, "paid", "premium", model = model, tune = 300,
                                 draws = 300, n_sims = 1000, seed = 678))
  p <- pattern_draws(fit)
  stopifnot(nrow(p) == 1000, ncol(p) == 10, identical(colnames(p), as.character(12 * 1:10)))
  stopifnot(all(p[, 10] == 1), all(p > 0), nrow(unique(p)) > 100)
  # A late share falls in nearly every simulation on group 353: the default
  # refuses, running_max holds each share at the largest before it.
  stopifnot(fails(monthly_pattern_draws(fit, 120)))
  m <- monthly_pattern_draws(fit, 120, monotone = "running_max")
  stopifnot(identical(dim(m), c(1000L, 120L)), all(m[, 120] == 1),
            all(m[, -1] - m[, -120] >= 0))
}

# Over segments, each segment has its own patterns, kept by segment().
two353 <- rbind(transform(comauto, lob = "A"),
                transform(comauto, lob = "B", paid = 2 * paid, premium = 2 * premium))
tri2 <- triangle(two353, "origin", "development", c("paid", "premium"), keys = "lob")
fits <- suppressWarnings(meyers_segments(tri2, "paid", "premium", tune = 300, draws = 300,
                                         n_sims = 1000, seed = 679))
pa <- pattern_draws(fits, "A")
pb <- pattern_draws(fits, "B")
stopifnot(nrow(pa) == 1000, all(pa[, 10] == 1), all(pb[, 10] == 1), !identical(pa, pb))
stopifnot(fails(pattern_draws(fits)))
stopifnot(identical(pattern_draws(segment(fits, lob = "B")), pb))

# A pattern held flat until its last step (decision 28 item 4): ten origins 97%
# developed at 12 months, flat to within 0.04% until 1 at 120 months. The
# running maximum holds each draw at its first share; no Weibull runs
# through that shape, and the default reading used to refuse it as not
# increasing. It now reads the held shares exactly.
noise <- c(0, 4e-4, -3e-4, 2e-4, -4e-4, 3e-4, -2e-4, 1e-4, 0)
flat <- do.call(rbind, lapply(0:9, function(i) {
  k <- 0:(9 - i)
  u <- 1000 * (1 + 0.1 * i)
  e <- ifelse(k == 0, 0, noise[(i * 7 + k * 3) %% 9 + 1])
  data.frame(origin = 2011 + i, development = 12 * (k + 1),
             paid = ifelse(k == 9, u, 0.97 * u * (1 + e)))
}))
flat_tri <- triangle(flat, "origin", "development", "paid")
for (fit in list(odp_bootstrap(flat_tri, n_sims = 200, seed = 1),
                 mack_bootstrap(flat_tri, n_sims = 200, seed = 1))) {
  p <- pattern_draws(fit)
  m <- monthly_pattern_draws(fit, 132, monotone = "running_max")
  stopifnot(identical(dim(m), c(200L, 132L)), all(m[, 132] == 1),
            all(m[, -1] - m[, -132] >= 0))
  held <- t(apply(p, 1, function(x) pmin(cummax(x), 1)))
  stopifnot(max(abs(m[, 12 * (1:10)] - held)) < 1e-9)
  uw <- monthly_pattern_draws(fit, 144, basis = underwriting_basis(), monotone = "running_max")
  stopifnot(all(uw[, 144] == 1))
}
# The compartmental model (#679): each simulation's pattern is the ODE's
# expected cumulative paid over that at the square's last age (120 months
# on company 337, one age past the paid triangle's), which never falls.
wk <- utils::read.csv(validation("data", "clrd_wkcomp_337.csv"), comment.char = "#")
wk337 <- wk[wk$origin + wk$development %/% 12 - 1 <= 1996, ]
y1997 <- wk[wk$origin == 1997 & wk$development == 12, ]
y1997$paid <- NA
y1997$incurred <- NA
wk337 <- rbind(wk337, y1997)
tri337 <- triangle(wk337, "origin", "development", c("paid", "incurred", "premium"))
cfit <- suppressWarnings(compartmental(tri337, "paid", "premium", gesmann_morris_priors(),
                                       incurred = "incurred", n_sims = 1000, seed = 679))
p <- pattern_draws(cfit)
stopifnot(nrow(p) == 1000, ncol(p) == 10, identical(colnames(p), as.character(12 * 1:10)))
stopifnot(all(p[, 10] == 1), all(p[, 1] > 0), nrow(unique(p)) > 100)
stopifnot(all(p[, -1] - p[, -10] >= 0))
# The default takes them, as running_max would leave them.
m <- monthly_pattern_draws(cfit, 120)
stopifnot(identical(dim(m), c(1000L, 120L)), all(m[, 120] == 1),
          identical(m, monthly_pattern_draws(cfit, 120, monotone = "running_max")))

# Over segments, each segment has its own patterns, kept by segment().
two337 <- rbind(transform(wk337, lob = "A"),
                transform(wk337, lob = "B", paid = 2 * paid, incurred = 2 * incurred,
                          premium = 2 * premium))
tri2 <- triangle(two337, "origin", "development", c("paid", "incurred", "premium"),
                 keys = "lob")
cfits <- suppressWarnings(compartmental_segments(tri2, "paid", "premium",
                                                 gesmann_morris_priors(),
                                                 incurred = "incurred", n_sims = 1000,
                                                 seed = 680))
pa <- pattern_draws(cfits, "A")
pb <- pattern_draws(cfits, "B")
stopifnot(nrow(pa) == 1000, all(pa[, 10] == 1), all(pb[, 10] == 1), !identical(pa, pb))
stopifnot(fails(pattern_draws(cfits)))
stopifnot(identical(pattern_draws(segment(cfits, lob = "B")), pb))
