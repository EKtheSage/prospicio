suppressMessages(library(prospicio))

# A development pattern per posterior draw of Meyers' CRC, CSR and CAY
# (docs/design/reserving-bayes.md decision 27, #678): row i is simulation i of
# the fit's reserves, the pattern ends at 1, and the monthly form feeds the
# cash-flow engine. The formula and the alignment with the reserves are
# checked in Rust (meyers.rs); this checks the binding on group 353.

fails <- function(expr) inherits(try(expr, silent = TRUE), "try-error")

long <- utils::read.csv(file.path("validation", "data", "clrd_comauto_353.csv"),
                        comment.char = "#")
long <- long[long$origin - 1988 + long$development %/% 12 <= 10, ]
tri <- triangle(long, "origin", "development", c("paid", "incurred", "premium"))
quick <- function(...) suppressWarnings(meyers(tri, "paid", "premium", tune = 300, draws = 300,
                                               n_sims = 400, seed = 678, ...))

for (model in c("crc", "csr", "cay")) {
  fit <- quick(model = model)
  p <- pattern_draws(fit)
  stopifnot(nrow(p) == 400, ncol(p) == 10, !anyNA(p), all(p > 0))
  stopifnot(identical(colnames(p), as.character(12 * 1:10)))
  # The oldest age is the reference: no tail.
  stopifnot(all(p[, 10] == 1))
  # Group 353 develops: the mean pattern rises from well below 1.
  m <- colMeans(p)
  stopifnot(m[1] < 0.7, m[9] > 0.9)

  # The monthly form, for the cash-flow engine.
  mo <- monthly_pattern_draws(fit, 120, monotone = "running_max")
  stopifnot(nrow(mo) == 400, ncol(mo) == 120, all(abs(mo[, 120] - 1) < 1e-12))
  stopifnot(all(mo[, -1] - mo[, -120] >= -1e-12))
  # The default refuses a falling pattern; when none falls the two agree.
  refuse <- try(monthly_pattern_draws(fit, 120), silent = TRUE)
  if (!inherits(refuse, "try-error")) stopifnot(identical(refuse, mo))
  # Too few months is an error, not a truncation.
  stopifnot(fails(monthly_pattern_draws(fit, 24, monotone = "running_max")))

  # A determinism check only: the same seed gives the same fit. That the
  # patterns draw no random number is checked by the Rust replay test.
  again <- quick(model = model)
  stopifnot(identical(pattern_draws(again), p))
  stopifnot(identical(ultimate_draws(again), ultimate_draws(fit)))
}

# Two segments by name, alone and with a rank correlation: each segment keeps
# its own simulations as whole rows, patterns included.
two <- long[rep(seq_len(nrow(long)), 2), ]
two$lob <- rep(c("A", "B"), each = nrow(long))
two$paid <- two$paid * rep(c(1, 2), each = nrow(long))
two$premium <- two$premium * rep(c(1, 2), each = nrow(long))
tri2 <- triangle(two, "origin", "development", c("paid", "premium"), keys = "lob")
segs <- function(...) suppressWarnings(meyers_segments(tri2, "paid", "premium", tune = 300,
                                                       draws = 300, n_sims = 400, seed = 678, ...))
free <- segs()
pa <- pattern_draws(free, "A")
pb <- pattern_draws(free, "B")
stopifnot(identical(dim(pa), c(400L, 10L)), identical(dim(pb), c(400L, 10L)))
stopifnot(!identical(pa, pb), all(pa[, 10] == 1), all(pb[, 10] == 1))
stopifnot(fails(pattern_draws(free)), fails(pattern_draws(free, "C")))
ma <- monthly_pattern_draws(free, 120, segment = "A", monotone = "running_max")
stopifnot(identical(dim(ma), c(400L, 120L)))

ranked <- segs(dependence = "rank_correlation", spearman = matrix(c(1, 0.8, 0.8, 1), 2))
# Each ranked row's reserves are some free row j's, and its pattern is that
# same row j's pattern.
moved <- 0
for (s in c("A", "B")) {
  key <- function(fits) apply(draw_matrix(segment(fits, lob = s)@reserves), 1, function(r) {
    paste(sprintf("%a", r), collapse = " ")
  })
  from <- key(free)
  to <- key(ranked)
  stopifnot(!anyDuplicated(from))
  j <- match(to, from)
  stopifnot(!anyNA(j))
  stopifnot(identical(unname(pattern_draws(ranked, s)), unname(pattern_draws(free, s)[j, ])))
  moved <- moved + sum(j != seq_along(j))
}
stopifnot(moved > 100)

# A fit of the wrong kind is refused.
stopifnot(fails(pattern_draws(1)))
