suppressMessages(library(prospicio))

# The latest diagonal on its own (a latest valuation) projected with a
# pattern from elsewhere (docs/design/reserving-v02.md, decisions 16 to 19,
# issue #544): latest_valuation(), as_latest_valuation(), the five methods
# on a latest valuation, to_chain_ladder(), ldf_origin and the rule past a
# pattern's end. The Rust core is checked in
# validation/tests/reserving_latest_diagonal.rs, reserving_fit_apply.rs and
# reserving_diagonal_segments.rs; these tests reproduce their numbers
# through the binding, as the Python tests do.

near <- function(a, b, rel = 1e-12) {
  a <- unname(a)
  b <- unname(b)
  if (length(a) != length(b) || !all(a == b | abs(a - b) <= rel * abs(b))) {
    stop(sprintf("got %s, want %s (rel %g)", paste(format(a, digits = 17), collapse = " "),
                 paste(format(b, digits = 17), collapse = " "), rel), call. = FALSE)
  }
}

expect_error_like <- function(expr, pattern) {
  msg <- tryCatch({
    expr
    NULL
  }, error = conditionMessage)
  if (is.null(msg)) stop(sprintf("expected an error mentioning %s", shQuote(pattern)), call. = FALSE)
  if (!grepl(pattern, msg, fixed = TRUE)) {
    stop(sprintf("error %s does not mention %s", shQuote(msg), shQuote(pattern)), call. = FALSE)
  }
}

root <- if (dir.exists("validation")) "validation" else file.path("..", "..", "..", "validation")
read_rows <- function(...) {
  lines <- readLines(file.path(root, ...))
  utils::read.csv(text = lines[!startsWith(lines, "#")], stringsAsFactors = FALSE,
                  colClasses = c(arg = "character"))
}
genins_long <- utils::read.csv(file.path(root, "data", "genins.csv"), comment.char = "#")
wkcomp_long <- utils::read.csv(file.path(root, "data", "clrd_wkcomp.csv"), comment.char = "#")
raa_long <- utils::read.csv(file.path(root, "data", "raa.csv"), comment.char = "#")
cdf_of <- function(ages, values) stats::setNames(values, ages)

# 1. Three accident years at December 2024.
dec_2024 <- as.Date("2024-12-31")
three <- latest_valuation(data.frame(year = 2022:2024, paid = c(900, 600, 150), premium = 1000),
                          "year", "paid", valuation = dec_2024, exposure = "premium")
stopifnot(identical(three@ages, c(36L, 24L, 12L)))
annual <- c("12" = 2, "24" = 1.25, "36" = 1)
stopifnot(identical(unname(chain_ladder(three, ldf = annual, ldf_kind = "cdf")@ultimate),
                    c(900, 750, 300)))
near(bornhuetter_ferguson(three, apriori = 0.7, ldf = annual, ldf_kind = "cdf")@ultimate[3],
     500, 1e-12)
# An apriori per origin (#634), named in any order or in origin order (oldest
# first, whatever the order of the rows).
shuffled <- latest_valuation(data.frame(year = c(2024, 2022, 2023), paid = c(150, 600 * 1.5, 600),
                                        premium = 1000),
                             "year", "paid", valuation = dec_2024, exposure = "premium")
for (apriori in list(c("2024" = 0.8, "2022" = 0.6, "2023" = 0.7), c(0.6, 0.7, 0.8))) {
  bf <- bornhuetter_ferguson(shuffled, apriori = apriori, ldf = annual, ldf_kind = "cdf")
  stopifnot(identical(unname(bf@apriori), c(0.6, 0.7, 0.8)))
  near(unname(bf@ultimate), c(900, 600 + 0.2 * 0.7 * 1000, 150 + 0.5 * 0.8 * 1000), 1e-12)
}
expect_error_like(bornhuetter_ferguson(shuffled, apriori = c("2021" = 0.5, "2022" = 0.6,
                                                             "2023" = 0.7, "2024" = 0.8),
                                       ldf = annual, ldf_kind = "cdf"),
                  "2021 is not among the origins fitted")

# 2. The six-origin hand diagonal.
hand_latest <- c(980, 860, 700, 520, 400, 150)
hand_cdf <- c(1, 1.02, 1.07, 1.15, 1.3, 12 / 7)
hand_ages <- c(72, 60, 48, 36, 24, 12)
hand <- latest_valuation(data.frame(year = 2019:2024, age = hand_ages, paid = hand_latest,
                                    premium = 200), "year", "paid", age = "age", exposure = "premium")
pattern <- cdf_of(hand_ages, hand_cdf)
reported <- 1 / hand_cdf
cl <- chain_ladder(hand, ldf = pattern, ldf_kind = "cdf")
near(cl@ultimate, hand_latest * hand_cdf)
near(cl@ultimate[6], 257.142857142857, 1e-12)
bf <- bornhuetter_ferguson(hand, apriori = 0.7, ldf = pattern, ldf_kind = "cdf")
near(bf@ultimate, hand_latest + 0.7 * 200 * (1 - reported))
near(bf@ultimate[6], 208.333333333333, 1e-12)
bk <- benktander(hand, apriori = 0.7, n_iters = 2, ldf = pattern, ldf_kind = "cdf")
near(bk@ultimate, hand_latest + (hand_latest + 0.7 * 200 * (1 - reported)) * (1 - reported))
cc <- cape_cod(hand, ldf = pattern, ldf_kind = "cdf")
cc_elr <- sum(hand_latest) / sum(200 * reported)
near(cc@ultimate, hand_latest + cc_elr * 200 * (1 - reported))
near(chain_ladder(hand, ldf = 1 / pattern, ldf_kind = "percent_developed")@ultimate, cl@ultimate)
near(expected_loss(hand, apriori = 0.7, ldf = 1 / pattern,
                   ldf_kind = "percent_developed")@ultimate, rep(140, 6))

# 3. Half-year origins with the annual pattern.
half <- latest_valuation(data.frame(origin = as.Date(c("2023-01-01", "2023-07-01", "2024-01-01")),
                                    paid = c(500, 300, 100), premium = 800),
                         "origin", "paid", valuation = dec_2024, exposure = "premium",
                         origin_grain = "S")
stopifnot(identical(half@ages, c(24L, 18L, 12L)), half@origin_grain == "S")
cdf18 <- sqrt(2 * 1.25)
cl <- chain_ladder(half, ldf = annual, ldf_kind = "cdf")
stopifnot(identical(cl@development, c(12L, 18L, 24L, 30L, 36L)))
near(cl@ultimate, c(625, 300 * cdf18, 200))
near(bornhuetter_ferguson(half, apriori = 0.5, ldf = annual, ldf_kind = "cdf")@ultimate,
     c(500 + 400 * (1 - 1 / 1.25), 300 + 400 * (1 - 1 / cdf18), 100 + 400 * 0.5))
ldf <- c("12" = 1.6, "24" = 1.25)
near(chain_ladder(half, ldf = ldf, ldf_kind = "ldf")@ultimate, cl@ultimate)
near(chain_ladder(half, ldf = ldf, ldf_kind = "ldf", tail = 1.05)@ultimate, cl@ultimate * 1.05)

# 4. Past the pattern's end.
past <- function(rule) {
  latest_valuation(data.frame(year = 2021:2024, age = c(48, 36, 24, 12), paid = 100), "year",
                   "paid", age = "age", past_pattern_end = rule)
}
short <- c("12" = 1.5, "24" = 1.2)
expect_error_like(chain_ladder(past("reject"), ldf = short, ldf_kind = "ldf"),
                  "outside the pattern's ages from 12 to 36 months")
stopifnot(chain_ladder(past("no_development"), ldf = short, ldf_kind = "ldf")@ultimate[1] == 100)
near(chain_ladder(past("tail"), ldf = short, ldf_kind = "ldf", tail = 1.05)@ultimate[1:2],
     c(102.53798721500054, 105))
ten <- stats::setNames(rep(1.1, 9), seq(12, 108, by = 12))
d <- latest_valuation(data.frame(year = 2000:2004, age = c(144, 132, 120, 108, 12), paid = 100),
                      "year", "paid", age = "age", past_pattern_end = "tail")
fit <- chain_ladder(d, ldf = ten, ldf_kind = "ldf", tail = 1.05)
near(fit@ultimate[1:4] / 100, c(1.0132158418990078, 1.0253798721500054, 1.05, 1.155), 1e-14)
level <- c("12" = 0.4, "60" = 0.8, "120" = 0.95)
d <- latest_valuation(data.frame(year = 2000:2002, age = c(120, 108, 12), paid = 100), "year",
                      "paid", age = "age", past_pattern_end = "no_development")
fit <- chain_ladder(d, ldf = level, ldf_kind = "percent_developed")
near(fit@pattern_rescale, 1.0526315789473684, 1e-15)
near(fit@ultimate[3], 237.5, 1e-14)
fit <- chain_ladder(with_past_pattern_end(d, "reject"), ldf = level,
                    ldf_kind = "percent_developed")
near(fit@ultimate[3], 250, 1e-14)
near(fit@pattern_tail, 1 / 0.95, 1e-14)
near(fit@pattern_rescale, 1, 1e-15)
expect_error_like(chain_ladder(past("no_development"), ldf = short, ldf_kind = "ldf", tail = 1.05),
                  "is a factor of 1 past the pattern's end")
expect_error_like(chain_ladder(past("tail"), ldf = short, ldf_kind = "ldf"), "tail")
expect_error_like(chain_ladder(past("tail"), ldf = short, ldf_kind = "ldf",
                               tail = tail_constant(1.05, attachment_age = 36)), "attachment")
expect_error_like(with_past_pattern_end(past("reject"), "hold"), "past_pattern_end must be")

# 5. ldf_origin.
monthly <- stats::setNames((1:24) / 24, 1:24)
book <- latest_valuation(data.frame(year = 2023:2024, age = c(24, 12), paid = c(37, 13)), "year",
                         "paid", age = "age")
near(chain_ladder(book, ldf = monthly, ldf_kind = "percent_developed",
                  ldf_origin = "M")@ultimate, c(48, 48), 1e-13)
small <- triangle(data.frame(year = c(2023, 2023, 2024), age = c(12, 24, 12),
                             paid = c(13, 37, 13)), "year", "age", "paid")
fit <- chain_ladder(small, ldf = monthly, ldf_kind = "percent_developed", ldf_origin = "M")
near(fit@selected_ldf[1], 37 / 13, 1e-14)
same <- chain_ladder(small, ldf = c("12" = 1.5), ldf_kind = "ldf", ldf_origin = "Y")
plain <- chain_ladder(small, ldf = c("12" = 1.5), ldf_kind = "ldf")
stopifnot(identical(same@ultimate, plain@ultimate))
expect_error_like(chain_ladder(small, ldf = monthly, ldf_kind = "percent_developed",
                               ldf_weight = c("12" = 0.5), ldf_origin = "M"), "weight")
quarters <- latest_valuation(data.frame(origin = as.Date(c("2024-01-01", "2024-04-01")),
                                        age = c(6, 3), paid = c(10, 5)), "origin", "paid",
                             age = "age", origin_grain = "Q")
expect_error_like(chain_ladder(quarters, ldf = c("12" = 2, "24" = 1), ldf_kind = "cdf",
                               ldf_origin = "Y"), "whole multiple")
expect_error_like(chain_ladder(small, ldf_origin = "M"), "ldf_origin is given without ldf")
expect_error_like(mack(small, ldf_origin = "M"), "ldf_origin is given without ldf")

# ldf_origin in an expected-loss method, a bootstrap and a one-year view:
# each equals the same call with the averaged pattern given directly
# (13/48 and 37/48 developed at 12 and 24 months, where it ends).
four <- triangle(data.frame(year = c(rep(2021, 4), rep(2022, 3), rep(2023, 2), 2024),
                            age = c(12, 24, 36, 48, 12, 24, 36, 12, 24, 12),
                            paid = c(100, 280, 380, 400, 110, 300, 400, 120, 330, 130),
                            premium = c(rep(500, 4), rep(520, 3), rep(540, 2), 560)),
                 "year", "age", c("paid", "premium"))
direct <- c("12" = 48 / 13, "24" = 48 / 37)
near_draws <- function(a, b) {
  stopifnot(identical(dim(a), dim(b)), all(abs(a - b) <= 1e-9 + 1e-10 * abs(b)))
}
bf_on <- function(x, ...) bornhuetter_ferguson(x, apriori = 0.8, ...)
near(bf_on(four, column = "paid", exposure = "premium", ldf = monthly,
           ldf_kind = "percent_developed", ldf_origin = "M")@ultimate,
     bf_on(four, column = "paid", exposure = "premium", ldf = direct, ldf_kind = "cdf")@ultimate)
young <- latest_valuation(data.frame(year = 2023:2024, age = c(24, 12), paid = c(330, 130),
                                     premium = c(540, 560)), "year", "paid", age = "age",
                          exposure = "premium")
near(bf_on(young, ldf = monthly, ldf_kind = "percent_developed", ldf_origin = "M")@ultimate,
     bf_on(young, ldf = direct, ldf_kind = "cdf")@ultimate)
near_draws(draw_matrix(odp_bootstrap(four, "paid", n_sims = 200, seed = 11, ldf = monthly,
                                     ldf_kind = "percent_developed",
                                     ldf_origin = "M")@reserves),
           draw_matrix(odp_bootstrap(four, "paid", n_sims = 200, seed = 11, ldf = direct,
                                     ldf_kind = "cdf")@reserves))
near_draws(draw_matrix(odp_one_year(four, "paid", n_sims = 200, seed = 11, ldf = monthly,
                                    ldf_kind = "percent_developed", ldf_origin = "M")@cdr),
           draw_matrix(odp_one_year(four, "paid", n_sims = 200, seed = 11, ldf = direct,
                                    ldf_kind = "cdf")@cdr))
unaveraged <- odp_bootstrap(four, "paid", n_sims = 200, seed = 11, ldf = monthly,
                            ldf_kind = "percent_developed")
stopifnot(abs(mean(unaveraged@reserves) -
                mean(odp_bootstrap(four, "paid", n_sims = 200, seed = 11, ldf = direct,
                                   ldf_kind = "cdf")@reserves)) > 1)

# 6. Off-grid reading on RAA.
raa <- triangle(raa_long, "origin", "development", "value")
sel <- chain_ladder(raa, ldf = c("12" = 1.5, "18" = 1.1, "36" = 1.2), ldf_kind = "ldf")@selected_ldf
near(sel[1:3], c(1.5484201731845508, 1.0656022367666107, 1.129243234657234), 1e-15)
stopifnot(all(is.na(sel[-(1:3)])))
sparse <- chain_ladder(raa, ldf = c("12" = 1.5, "36" = 1.2), ldf_kind = "ldf")@selected_ldf
stopifnot(identical(unname(sparse[1:3]), c(1.5, NA, 1.2)), all(is.na(sparse[-(1:3)])))

# Books rebuilt as target() in validation/tests/reserving_fit_apply.rs builds them.
genins <- triangle(genins_long, "origin", "development", "value")
wkcomp <- triangle(wkcomp_long, "origin", "development", c("paid", "premium"))
remake <- function(long, columns, keep = function(year, age) TRUE,
                   value = function(c, year, v) v, shift = 0) {
  rows <- long[keep(long$origin, long$development), , drop = FALSE]
  for (k in seq_along(columns)) {
    rows[[columns[k]]] <- value(k, rows$origin, rows[[columns[k]]])
  }
  rows$origin <- rows$origin + shift
  triangle(rows, "origin", "development", columns)
}
target <- function(dataset, name) {
  if (dataset == "genins") {
    g <- function(...) remake(genins_long, "value", ...)
    return(switch(name,
      diagonal = g(),
      perturbed = g(value = function(c, year, v) v * (1 + 0.01 * (year - 2000))),
      double = g(value = function(c, year, v) v * 2),
      from_2006 = g(keep = function(year, age) year >= 2006),
      to_2004 = g(keep = function(year, age) year <= 2004),
      shifted = g(shift = 10),
      early = g(keep = function(year, age) year + age %/% 12 - 1 < 2008)
    ))
  }
  w <- function(...) remake(wkcomp_long, c("paid", "premium"), ...)
  switch(name,
    diagonal = w(), diagonal_raw = w(),
    perturbed = w(value = function(c, year, v) {
      v <- v * (1 + 0.01 * (year - 1987))
      if (c == 2) v * 1.1 else v
    }),
    shifted = w(shift = 10),
    from_1993 = w(keep = function(year, age) year >= 1993)
  )
}
params <- function(method) {
  parts <- strsplit(strsplit(method, ";", fixed = TRUE)[[1]][-1], "=", fixed = TRUE)
  stats::setNames(lapply(parts, `[`, 2), vapply(parts, `[`, "", 1))
}
check_row <- function(row, found) {
  e <- row$expected
  if (is.null(found) || is.na(found) || abs(found - e) > max(row$abs_tol, row$rel_tol * abs(e))) {
    stop(sprintf("%s %s %s %s: got %.17g, want %.17g", row$dataset, row$method, row$quantity,
                 row$arg, found, e), call. = FALSE)
  }
}
patterns_of <- function(rows) {
  out <- list()
  for (r in split(rows[rows$quantity == "pattern", ], seq_len(sum(rows$quantity == "pattern")))) {
    key <- paste(r$dataset, r$method)
    out[[key]] <- c(out[[key]], stats::setNames(r$expected, r$arg))
  }
  out
}
expected_method <- function(base, d, pattern_args, extra = list()) {
  name <- strsplit(base, ";", fixed = TRUE)[[1]][1]
  p <- params(base)
  args <- c(list(d), pattern_args, extra)
  switch(name,
    chain_ladder = do.call(chain_ladder, args),
    bornhuetter_ferguson = do.call(bornhuetter_ferguson,
                                   c(args, list(apriori = as.numeric(p$apriori)))),
    benktander = do.call(benktander, c(args, list(apriori = as.numeric(p$apriori),
                                                  n_iters = as.numeric(p$n_iters)))),
    cape_cod = do.call(cape_cod, c(args, list(trend = as.numeric(p$trend),
                                              decay = as.numeric(p$decay))))
  )
}
read_fit <- function(row, fit) {
  switch(row$quantity,
    total_ultimate = fit@total_ultimate,
    ultimate = unname(fit@ultimate[row$arg]),
    apriori = unname(fit@trended_apriori[row$arg])
  )
}

# 7. chainladder-python's DevelopmentConstant on the latest diagonal.
rows <- read_rows("reference", "reserving_latest_diagonal_python.csv")
patterns <- patterns_of(rows)
fits <- list()
checked <- 0
for (k in which(rows$quantity != "pattern")) {
  r <- rows[k, ]
  key <- paste(r$dataset, r$method)
  if (is.null(fits[[key]])) {
    kind <- if (grepl("style=ldf", r$method, fixed = TRUE)) "ldf" else "cdf"
    d <- if (r$dataset == "genins") as_latest_valuation(genins) else
      as_latest_valuation(wkcomp, "paid", exposure = "premium")
    fits[[key]] <- expected_method(r$method, d, list(ldf = patterns[[key]], ldf_kind = kind))
  }
  check_row(r, read_fit(r, fits[[key]]))
  checked <- checked + 1
}
stopifnot(checked == 55)

# 8. A fit applied to another book against chainladder-python, both arms.
genins_tail <- function(method) {
  p <- params(method)
  switch(p$tail, none = 1, constant = as.numeric(p$factor),
         tail_curve(if (is.null(p$curve)) "exponential" else p$curve))
}
rows <- read_rows("reference", "reserving_fit_apply_python.csv")
patterns <- patterns_of(rows)
for (arm in c("own", "reference")) {
  sources <- list()
  applied <- list()
  checked <- 0
  arm_rows <- if (arm == "own") rows else rows[grepl(";target=", rows$method, fixed = TRUE), ]
  for (k in seq_len(nrow(arm_rows))) {
    r <- arm_rows[k, ]
    split <- strsplit(r$method, ";target=", fixed = TRUE)[[1]]
    base <- split[1]
    target_name <- if (length(split) > 1) split[2] else NULL
    key <- paste(r$dataset, base)
    if (is.null(sources[[key]])) {
      sources[[key]] <- if (r$dataset == "genins") {
        chain_ladder(genins, tail = genins_tail(base))
      } else {
        chain_ladder(wkcomp, "paid")
      }
    }
    source <- sources[[key]]
    if (is.null(target_name)) {
      found <- switch(r$quantity,
        pattern = unname(source@cdf[match(as.integer(r$arg), source@development)]),
        tail = source@tail,
        refitted_tail_perturbed = {
          refit <- chain_ladder(target("genins", "perturbed"), tail = genins_tail(base))@tail
          stopifnot(abs(refit - source@tail) > r$rel_tol * source@tail)
          refit
        }
      )
      check_row(r, found)
      checked <- checked + 1
      next
    }
    akey <- paste(r$dataset, r$method)
    if (is.null(applied[[akey]])) {
      book <- target(r$dataset, target_name)
      d <- if (r$dataset == "genins") as_latest_valuation(book) else
        as_latest_valuation(book, "paid", exposure = "premium")
      if (target_name == "diagonal_raw") {
        frame <- as.data.frame(d)
        frame$age <- source@development[1]
        frame$year <- as.integer(frame$origin)
        d <- latest_valuation(frame, "year", "latest", age = "age", exposure = "exposure")
      }
      if (arm == "own") {
        fit <- expected_method(base, d, to_chain_ladder(source))
        by_development <- expected_method(base, d, list(), list(development = source))
        stopifnot(identical(fit@ultimate, by_development@ultimate))
      } else {
        fit <- expected_method(base, d, list(ldf = patterns[[key]], ldf_kind = "cdf"))
      }
      applied[[akey]] <- fit
    }
    check_row(r, read_fit(r, applied[[akey]]))
    checked <- checked + 1
  }
  stopifnot(checked > 100)
}

# to_chain_ladder() round-trips through ldf, ldf_kind and tail.
source <- chain_ladder(genins, tail = tail_curve())
frozen <- to_chain_ladder(source)
stopifnot(frozen$ldf_kind == "ldf", length(frozen$tail) == 1,
          identical(as.integer(names(frozen$ldf)), utils::head(source@development, -1)))
d <- as_latest_valuation(genins)
a <- do.call(chain_ladder, c(list(d), frozen))
stopifnot(identical(unname(a@ldf), unname(source@ldf)))
near(a@ultimate, source@ultimate, 1e-14)

# 9. Segments.
perturbed <- transform(genins_long, value = value * (1 + 0.01 * (origin - 2000)))
keyed <- triangle(rbind(transform(genins_long, lob = "a"), transform(perturbed, lob = "b")),
                  "origin", "development", "value", keys = "lob")
fits <- chain_ladder(keyed, tail = tail_curve())
d <- as_latest_valuation(keyed)
stopifnot(identical(d@keys, "lob"), identical(d@index$lob, c("a", "b")))
applied <- chain_ladder(d, development = fits)
for (lob in c("a", "b")) {
  s <- segment(fits, lob = lob)
  f <- segment(applied, lob = lob)
  stopifnot(identical(unname(f@ldf), unname(s@ldf)))
  near(f@ultimate, s@ultimate, 1e-14)
}
mapped <- chain_ladder(d, development = fits, segment_map = c(a = "a", b = "a"))
by_hand <- do.call(chain_ladder, c(list(segment(as_latest_valuation(keyed), lob = "b")),
                                   to_chain_ladder(segment(fits, lob = "a"))))
stopifnot(identical(unname(segment(mapped, lob = "b")@ultimate), unname(by_hand@ultimate)))
one <- chain_ladder(genins, tail = tail_curve())
broadcast <- chain_ladder(d, development = one)
for (lob in c("a", "b")) {
  alone <- do.call(chain_ladder, c(list(segment(as_latest_valuation(keyed), lob = lob)),
                                   to_chain_ladder(one)))
  stopifnot(identical(unname(segment(broadcast, lob = lob)@ultimate), unname(alone@ultimate)))
}
bf <- bornhuetter_ferguson(as_latest_valuation(wkcomp, "paid", exposure = "premium"), apriori = 0.6,
                           development = chain_ladder(wkcomp, "paid"))
bf_fit <- bornhuetter_ferguson(wkcomp, "paid", "premium", apriori = 0.6)
near(bf@ultimate, bf_fit@ultimate, 1e-12)
# Another method's fit passes its @chain_ladder, segment by segment.
keyed_wk <- triangle(rbind(transform(wkcomp_long, lob = "a"),
                           transform(wkcomp_long, lob = "b", paid = paid * 1.1)),
                     "origin", "development", c("paid", "premium"), keys = "lob")
d_wk <- as_latest_valuation(keyed_wk, "paid", exposure = "premium")
bf_keyed <- bornhuetter_ferguson(keyed_wk, "paid", "premium", apriori = 0.6, tail = tail_curve())
cl_keyed <- chain_ladder(keyed_wk, "paid", tail = tail_curve())
for (via in list(bf_keyed, bf_keyed@chain_ladder)) {
  for (method in list(chain_ladder, function(x, ...) bornhuetter_ferguson(x, apriori = 0.6, ...))) {
    found <- method(d_wk, development = via)
    want <- method(d_wk, development = cl_keyed)
    for (lob in c("a", "b")) {
      stopifnot(identical(segment(found, lob = lob)@ultimate, segment(want, lob = lob)@ultimate))
    }
  }
}
plain <- chain_ladder(keyed)
expect_error_like(chain_ladder(d, development = plain, segment_map = c(a = "a")), "no source")
two <- triangle(data.frame(year = rep(c(2001, 2001, 2002), 2), age = rep(c(12, 24, 12), 2),
                           value = rep(c(1, 2, 1), 2), line = rep(c("x", "y"), each = 3)),
                "year", "age", "value", keys = "line")
# The match leaves out the message's hint, whose wording is decision 19
# item 11's to settle.
expect_error_like(chain_ladder(d, development = chain_ladder(two)), "the source has keys")
expect_error_like(chain_ladder(d, ldf = c("12" = 1.5), ldf_kind = "ldf", development = plain),
                  "the development comes from the source fit")
expect_error_like(chain_ladder(d, tail = 1.05, development = plain),
                  "the development comes from the source fit")
# Link ratio exclusions are refused too, rather than ignored.
expect_error_like(chain_ladder(d, n_periods = 3, development = plain), "no link ratio exclusions")
expect_error_like(chain_ladder(d, drop_high = 1, development = plain), "no link ratio exclusions")
expect_error_like(to_chain_ladder(plain), "segment()")
expect_error_like(chain_ladder(d, segment_map = c(a = "a")), "segment_map is given without")
expect_error_like(chain_ladder(keyed, development = plain), "as_latest_valuation()")
expect_error_like(chain_ladder(d, "value"), "give column and exposure only for a triangle")
expect_error_like(mack(d), "a latest valuation has no link ratios")
expect_error_like(odp_bootstrap(d), "a latest valuation has no link ratios")
expect_error_like(chain_ladder(d, development = "fit"), "development must be a chain_ladder_fit")

# Cape Cod trends every segment to one valuation: b a year older.
both <- rbind(transform(wkcomp_long, lob = "a"), transform(wkcomp_long, lob = "b",
                                                            origin = origin - 1))
both <- triangle(both, "origin", "development", c("paid", "premium"), keys = "lob")
d <- as_latest_valuation(both, "paid", exposure = "premium")
own <- chain_ladder(wkcomp, "paid")
cdf <- stats::setNames(own@cdf, own@development)
together <- cape_cod(d, trend = 0.05, ldf = cdf, ldf_kind = "cdf")
alone <- cape_cod(segment(as_latest_valuation(both, "paid", exposure = "premium"), lob = "b"),
                  trend = 0.05, ldf = cdf, ldf_kind = "cdf")
near(segment(together, lob = "b")@trended_apriori, alone@trended_apriori * 1.05)
near(segment(together, lob = "b")@apriori, alone@apriori)
near(segment(together, lob = "b")@ultimate, alone@ultimate)

# 10. Constructors.
d <- as_latest_valuation(keyed)
frame <- as.data.frame(d)
stopifnot(identical(names(frame), c("lob", "origin", "age", "latest")))
frame$valuation <- as.Date(sprintf("%d-12-01", as.integer(frame$origin) + frame$age %/% 12 - 1))
frame$year <- as.integer(frame$origin)
again <- latest_valuation(frame, "year", "latest", valuation = "valuation", keys = "lob")
stopifnot(identical(again@ages, d@ages), identical(again@latest, d@latest),
          identical(again@origins, d@origins), identical(again@index, d@index))
by_age <- latest_valuation(frame, "year", "latest", age = "age", keys = "lob")
stopifnot(identical(by_age@ages, d@ages))
exposure <- as.double(seq_len(nrow(frame)))
with_e <- with_exposure(d, exposure)
stopifnot(identical(with_e@exposure, exposure), identical(as.data.frame(with_e)$exposure, exposure))
n_a <- sum(frame$lob == "a")
stopifnot(identical(segment(as_latest_valuation(keyed), lob = "b")@latest,
                    frame$latest[-seq_len(n_a)]))
expect_error_like(latest_valuation(frame, "year", "latest", keys = "lob"), "exactly one of age")
expect_error_like(latest_valuation(frame, "year", "latest", valuation = 2024), "as a Date")
expect_error_like(latest_valuation(frame, "year", "latest", age = "age", past_pattern_end = "hold"),
                  "past_pattern_end must be")
plain_d <- latest_valuation(data.frame(year = 2023:2024, age = c(24, 12), paid = c(1, 2)), "year",
                            "paid", age = "age", past_pattern_end = "tail")
stopifnot(plain_d@past_pattern_end == "tail", is.null(plain_d@exposure),
          length(plain_d@keys) == 0)
printed <- utils::capture.output(print(plain_d))
stopifnot(startsWith(printed[1], "<latest_valuation> 2 origins"),
          S7::S7_inherits(plain_d, latest_valuation),
          !exists("bordereau", envir = asNamespace("prospicio"), inherits = FALSE))
expect_error_like(with_exposure(genins, 1), "x must be a latest valuation")

cat("test-reserving-latest-diagonal.R: ok\n")
