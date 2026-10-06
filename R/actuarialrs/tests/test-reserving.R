suppressMessages(library(actuarialrs))

# Reports the values on failure, so a CI log shows what diverged.
near <- function(a, b, rel = 1e-12) {
  if (!all(abs(a - b) <= rel * pmax(abs(b), 1e-300))) {
    stop(sprintf("got %s, want %s (rel %g)", format(a, digits = 17), format(b, digits = 17), rel),
         call. = FALSE)
  }
}

# The error message of `expr`, which must fail.
error_of <- function(expr) {
  msg <- tryCatch({
    expr
    NULL
  }, error = conditionMessage)
  if (is.null(msg)) stop("expected an error", call. = FALSE)
  msg
}

expect_error_like <- function(expr, pattern) {
  msg <- error_of(expr)
  if (!grepl(pattern, msg, fixed = TRUE)) {
    stop(sprintf("error %s does not mention %s", shQuote(msg), shQuote(pattern)), call. = FALSE)
  }
}

# Tests run from the repository root (CI and `cargo xtask r`).
validation <- function(...) file.path("validation", ...)
read_long <- function(name) utils::read.csv(validation("data", paste0(name, ".csv")), comment.char = "#")
load_triangle <- function(name) triangle(read_long(name), "origin", "development", "value")

tris <- list(raa = load_triangle("raa"), genins = load_triangle("genins"), abc = load_triangle("abc"))

# Shapes and accessors.
raa <- tris$raa
stopifnot(
  identical(unname(raa@shape), c(1L, 1L, 10L, 10L)),
  identical(raa@origins, as.character(1981:1990)),
  identical(raa@development, seq(12L, 120L, by = 12L)),
  identical(raa@columns, "value"),
  identical(raa@keys, character()),
  identical(dim(raa@index), c(1L, 0L)),
  identical(dimnames(raa@values)$index, "Total"),
  identical(raa@valuation, as.Date("1990-12-31")),
  raa@is_cumulative,
  raa@origin_grain == "Y", raa@development_grain == "Y",
  identical(dim(raa@values), c(1L, 1L, 10L, 10L)),
  is.na(raa@values[1, 1, "1990", "24"])
)
near(raa@values[1, "value", "1981", "12"], 5012)
near(sum(latest_diagonal(raa)), 160987)
near(sum(latest_diagonal(tris$genins)), 34358090)
near(sum(latest_diagonal(tris$abc)), 10221194)
near(link_ratios(raa)@values[1, 1, "1981", "12"], 8269 / 5012)
stopifnot(!link_ratios(raa)@is_cumulative, link_ratios(raa)@shape[["development"]] == 9L)

# Chain ladder and Mack against R ChainLadder: every reference row, with
# its own tolerance (a case passes if it meets either).
# The free-text `source` column holds unquoted commas, so keep the first
# seven fields of each line.
lines <- grep("^#", readLines(validation("reference", "reserving_chainladder_r.csv")),
              value = TRUE, invert = TRUE)
fields <- lapply(strsplit(lines, ",", fixed = TRUE), `[`, 1:7)
ref <- utils::read.csv(text = vapply(fields, paste, "", collapse = ","))
settings <- list(
  chain_ladder = c("volume", "log-linear"), mack = c("volume", "log-linear"),
  chain_ladder_simple = c("simple", "log-linear"), mack_alpha0 = c("simple", "log-linear"),
  mack_alpha2 = c("regression", "log-linear"), mack_sigma_mack = c("volume", "mack")
)
fits <- list()
fit_for <- function(dataset, method) {
  key <- paste(dataset, method)
  if (is.null(fits[[key]])) {
    s <- settings[[method]]
    if (is.null(s)) stop("no settings for reference method ", method, call. = FALSE)
    # Chain-ladder rows go through chain_ladder(), the rest through mack().
    fit <- if (startsWith(method, "chain_ladder")) chain_ladder else mack
    fits[[key]] <<- fit(tris[[dataset]], average = s[1], sigma_interpolation = s[2])
  }
  fits[[key]]
}
value_of <- function(fit, quantity, arg) {
  k <- as.integer(arg) + 1L
  o <- as.character(arg)
  switch(quantity,
    ata_factor = fit@ldf[[k]], cdf = fit@cdf[[k]], sigma = fit@sigma[[k]], f_se = fit@std_err[[k]],
    ultimate = fit@ultimate[[o]], reserve = fit@reserve[[o]], se = fit@standard_error[[o]],
    process_risk = fit@process_risk[[o]], parameter_risk = fit@parameter_risk[[o]],
    total_ultimate = fit@total_ultimate, total_reserve = fit@total_reserve,
    total_standard_error = fit@total_standard_error, total_process_risk = fit@total_process_risk,
    total_parameter_risk = fit@total_parameter_risk,
    stop("unknown quantity ", quantity)
  )
}
for (r in seq_len(nrow(ref))) {
  row <- ref[r, ]
  got <- value_of(fit_for(row$dataset, row$method), row$quantity, row$arg)
  err <- abs(got - row$expected)
  if (!(got == row$expected || err <= row$abs_tol || err <= row$rel_tol * abs(row$expected))) {
    stop(sprintf("%s %s %s[%s]: got %.17g, want %.17g", row$dataset, row$method, row$quantity,
                 row$arg, got, row$expected), call. = FALSE)
  }
}
stopifnot(nrow(ref) > 700)

# The deterministic chain ladder gives the projection Mack builds on.
for (name in names(tris)) {
  cl <- chain_ladder(tris[[name]])
  m <- mack(tris[[name]])
  stopifnot(identical(cl@ultimate, m@ultimate), identical(cl@ldf, m@chain_ladder@ldf))
  near(m@total_cv, m@total_standard_error / m@total_reserve)
  near(sum(cl@reserve), cl@total_reserve, 1e-12)
}
cl <- chain_ladder(raa, "value", tail = 1.05)
near(cl@cdf[["120-Ult"]], 1.05)
near(cl@ultimate[["1981"]], 18834 * 1.05)
stopifnot(identical(names(cl@ldf)[1], "12-24"), cl@alpha == 1, cl@tail == 1.05)
df <- as.data.frame(mack(raa))
stopifnot(identical(names(df), c("origin", "latest", "ultimate", "reserve", "process_risk",
                                 "parameter_risk", "standard_error")), nrow(df) == 10)
invisible(utils::capture.output(print(raa), print(cl), print(mack(raa))))

# Tails against R ChainLadder's MackChainLadder(tail = ...) and
# chainladder-python's TailConstant, TailCurve and TailBondy with
# MackChainladder (validation/reference/reserving_tails_{r,python}.csv):
# every row, as validation/tests/reserving_tails.rs. ldf and cdf run past
# the oldest age into the tail's factors. mack_tail_given gives tail_sigma
# and tail_std_err: twice the last sigma, half the last standard error.
tail_methods <- list(
  mack_tail_loglinear = list(tail = tail_log_linear()),
  mack_tail_loglinear_sigma_mack = list(tail = tail_log_linear(), sigma_interpolation = "mack"),
  mack_tail_loglinear_alpha2 = list(tail = tail_log_linear(), average = "regression"),
  mack_tail_constant = list(tail = 1.05),
  mack_tail_constant_sigma_mack = list(tail = 1.05, sigma_interpolation = "mack"),
  mack_tail_given = list(tail = 1.05),
  tail_constant = list(tail = tail_constant(1.05)),
  tail_constant_decay = list(tail = tail_constant(1.1, decay = 0.75)),
  tail_constant_attach = list(tail = tail_constant(1.05, attachment_age = 72)),
  tail_constant_below_one = list(tail = tail_constant(0.98)),
  tail_curve_exponential = list(tail = tail_curve()),
  tail_curve_inverse_power = list(tail = tail_curve("inverse_power")),
  tail_curve_fit_period = list(tail = tail_curve(fit_period = c(36, 108), extrap_periods = 50)),
  tail_curve_off_grid = list(tail = tail_curve(fit_period = c(30, 102))),
  tail_curve_attach = list(tail = tail_curve(attachment_age = 60)),
  tail_bondy = list(tail = tail_bondy()),
  tail_bondy_generalized = list(tail = tail_bondy(earliest_age = 36)),
  tail_bondy_off_grid = list(tail = tail_bondy(earliest_age = 30)),
  tail_bondy_attach = list(tail = tail_bondy(earliest_age = 36, attachment_age = 72))
)
given_tail <- function(tri) {
  cl <- chain_ladder(tri)
  c(sigma = 2 * cl@sigma[[length(cl@sigma)]], std_err = cl@std_err[[length(cl@std_err)]] / 2)
}
tail_fits <- list()
tail_fit_for <- function(dataset, method) {
  key <- paste(dataset, method)
  if (is.null(tail_fits[[key]])) {
    args <- tail_methods[[method]]
    if (is.null(args)) stop("no settings for reference method ", method, call. = FALSE)
    if (method == "mack_tail_given") {
      given <- given_tail(tris[[dataset]])
      args$tail_sigma <- given[["sigma"]]
      args$tail_std_err <- given[["std_err"]]
    }
    tail_fits[[key]] <<- do.call(mack, c(list(tris[[dataset]]), args))
  }
  tail_fits[[key]]
}
tail_value_of <- function(dataset, method, quantity, arg) {
  if (startsWith(quantity, "given_tail_")) {
    return(given_tail(tris[[dataset]])[[sub("given_tail_", "", quantity)]])
  }
  fit <- tail_fit_for(dataset, method)
  ldf <- c(fit@ldf, fit@tail_ldf)
  k <- as.integer(arg) + 1L
  switch(quantity,
    ldf = ldf[[k]],
    cdf = if (k <= length(fit@cdf)) fit@cdf[[k]] else prod(ldf[k:length(ldf)]),
    tail_factor = fit@tail, tail_sigma = fit@tail_sigma, tail_std_err = fit@tail_std_err,
    value_of(fit, quantity, arg)
  )
}
for (reference in c("reserving_tails_r.csv", "reserving_tails_python.csv")) {
  tail_lines <- grep("^#", readLines(validation("reference", reference)), value = TRUE, invert = TRUE)
  tail_fields <- lapply(strsplit(tail_lines, ",", fixed = TRUE), `[`, 1:7)
  tail_ref <- utils::read.csv(text = vapply(tail_fields, paste, "", collapse = ","),
                              colClasses = c(arg = "character"))
  stopifnot(nrow(tail_ref) > 1000)
  for (r in seq_len(nrow(tail_ref))) {
    row <- tail_ref[r, ]
    got <- tail_value_of(row$dataset, row$method, row$quantity, row$arg)
    err <- abs(got - row$expected)
    if (!(got == row$expected || err <= row$abs_tol || err <= row$rel_tol * abs(row$expected))) {
      stop(sprintf("%s %s %s %s[%s]: got %.17g, want %.17g", reference, row$dataset, row$method,
                   row$quantity, row$arg, got, row$expected), call. = FALSE)
    }
  }
}

# Tail estimators: properties, printing and the selected factors.
curve <- tail_curve("inverse_power", fit_period = c(36, NA), attachment_age = 60)
stopifnot(identical(curve@curve, "inverse_power"), identical(curve@fit_period, c(36L, NA)),
          identical(curve@extrap_periods, 100L), identical(curve@attachment_age, 60L),
          identical(tail_curve()@fit_period, c(NA_integer_, NA_integer_)),
          identical(tail_constant(1.05)@factor, 1.05), identical(tail_constant()@decay, 0.5),
          is.null(tail_constant()@attachment_age), is.null(tail_bondy()@earliest_age),
          identical(tail_bondy(36)@earliest_age, 36L),
          identical(tail_constant(1.05, attachment_age = NA)@attachment_age, NULL))
stopifnot(identical(utils::capture.output(print(curve)),
                    'tail_curve(curve = "inverse_power", fit_period = c(36, NA), extrap_periods = 100, attachment_age = 60)'),
          identical(utils::capture.output(print(tail_log_linear())), "tail_log_linear()"),
          identical(utils::capture.output(print(tail_constant(1.05))),
                    "tail_constant(factor = 1.05, decay = 0.5, attachment_age = NULL)"))
fit <- chain_ladder(raa, tail = curve)
base <- chain_ladder(raa)
stopifnot(identical(fit@ldf[1:4], base@ldf[1:4]), fit@ldf[[5]] != base@ldf[[5]],
          identical(names(fit@ldf), names(base@ldf)),
          identical(fit@estimated_ldf, base@ldf), identical(fit@tail_attachment_age, 60L),
          identical(base@tail_attachment_age, 120L),
          identical(mack(raa, tail = curve)@estimated_ldf, base@ldf))
near(prod(fit@tail_ldf), fit@tail, 1e-12)
near(fit@cdf[["120-Ult"]], fit@tail, 1e-12)
# A number is a constant tail; without a tail above 1 there is no tail risk.
stopifnot(identical(chain_ladder(raa, tail = 1.05)@ultimate,
                    chain_ladder(raa, tail = tail_constant(1.05))@ultimate))
m <- mack(raa)
stopifnot(m@tail == 1, m@tail_sigma == 0, m@tail_std_err == 0, m@standard_error[["1981"]] == 0,
          identical(m@tail_ldf, chain_ladder(raa)@tail_ldf))
given <- mack(raa, tail = 1.05, tail_sigma = 1.5, tail_std_err = 0.003)
stopifnot(given@tail_sigma == 1.5, given@tail_std_err == 0.003, given@standard_error[["1981"]] > 0)
invisible(utils::capture.output(print(given), print(tail_bondy(36))))
expect_error_like(chain_ladder(raa, tail = "1.05"), "tail must be a single number, tail_constant()")
expect_error_like(tail_curve("weibull"), "should be one of")
expect_error_like(tail_curve(fit_period = 36), "fit_period must be two ages")
expect_error_like(tail_constant(1.05, attachment_age = 6.5), "attachment_age must be a non-negative whole")
expect_error_like(tail_constant(c(1, 2)), "factor must be a single number")
expect_error_like(chain_ladder(raa, tail = tail_curve(fit_period = c(108, NA))), "tail")
expect_error_like(mack(raa, tail = 1.05, tail_sigma = -1), "tail")
expect_error_like(mack(raa, tail_sigma = "a"), "tail_sigma must be a single number")
# With several segments each has its own tail.
raa_long <- read_long("raa")
lc_tail <- triangle(rbind(transform(raa_long, lob = "a"),
                          transform(raa_long, lob = "b", value = value * (1 + development / 240))),
                    "origin", "development", "value", keys = "lob")
seg_fit <- chain_ladder(lc_tail, tail = tail_bondy())
stopifnot(segment(seg_fit, lob = "a")@tail == chain_ladder(raa, tail = tail_bondy())@tail)
expect_error_like(seg_fit@tail, "2 segments; use totals_frame() or segment()")
expect_error_like(seg_fit@tail_ldf, "2 segments; use segment()")
expect_error_like(seg_fit@estimated_ldf, "2 segments; use segment()")
seg_totals <- totals_frame(seg_fit)
stopifnot(identical(names(seg_totals), c("lob", "latest", "ultimate", "reserve", "tail",
                                         "tail_sigma", "tail_std_err")),
          identical(seg_totals$tail[1], segment(seg_fit, lob = "a")@tail),
          identical(seg_totals$tail_sigma[2], segment(seg_fit, lob = "b")@tail_sigma))
invisible(utils::capture.output(print(seg_fit)))

# Long-table round trip.
long <- as.data.frame(raa)
stopifnot(identical(names(long), c("origin", "development", "value")),
          inherits(long$origin, "Date"), nrow(long) == 55)
back <- triangle(long, "origin", "development", "value")
stopifnot(identical(back@values, raa@values), identical(back@origins, raa@origins))
stopifnot(identical(subset(raa)@values, raa@values), identical(aggregate(raa)@values, raa@values))

# Incremental and cumulative.
inc <- to_incremental(raa)
stopifnot(!inc@is_cumulative)
near(inc@values[1, 1, "1981", "24"], 8269 - 5012)
stopifnot(identical(to_cumulative(inc)@values, raa@values))
inc_long <- as.data.frame(inc)
from_inc <- triangle(inc_long, "origin", "development", "value", cumulative = FALSE)
stopifnot(identical(to_cumulative(from_inc)@values, raa@values))

# Development as valuation dates (and as valuation years).
raw <- read_long("raa")
by_val <- raw
by_val$development <- as.Date(sprintf("%d-12-31", raw$origin + raw$development / 12 - 1))
stopifnot(identical(triangle(by_val, "origin", "development", "value",
                             development_is_valuation = TRUE)@values,
                    raa@values))
by_year <- transform(raw, development = origin + development / 12 - 1)
stopifnot(identical(triangle(by_year, "origin", "development", "value",
                             development_is_valuation = TRUE)@values,
                    raa@values))

# Several key columns and value columns; subset and aggregate by name.
two <- rbind(transform(raw, lob = "auto", state = "CA", paid = value, incurred = 1.5 * value),
             transform(raw, lob = "home", state = "NY", paid = 2 * value, incurred = 3 * value))
multi <- triangle(two, "origin", "development", c("paid", "incurred"), keys = c("lob", "state"))
stopifnot(
  identical(multi@keys, c("lob", "state")),
  identical(unname(multi@shape), c(2L, 2L, 10L, 10L)),
  identical(multi@index, data.frame(lob = c("auto", "home"), state = c("CA", "NY"))),
  identical(dimnames(multi@values)$index, c("auto / CA", "home / NY"))
)
home <- subset(multi, lob = "home", state = "NY", columns = "incurred")
near(home@values[!is.na(home@values)], 3 * raa@values[!is.na(raa@values)])
both <- subset(multi, state = c("NY", "CA"))
stopifnot(identical(both@index$lob, c("auto", "home")), identical(both@keys, c("lob", "state")))
multi_long <- as.data.frame(multi)
stopifnot(identical(names(multi_long), c("lob", "state", "origin", "development", "paid", "incurred")))
again <- triangle(multi_long, "origin", "development", c("paid", "incurred"), keys = multi@keys)
stopifnot(identical(again@values, multi@values), identical(as.data.frame(again), multi_long))
# Key order follows the argument; factor and numeric keys become character.
flipped <- triangle(multi_long, "origin", "development", "paid", keys = c("state", "lob"))
stopifnot(identical(flipped@index, data.frame(state = c("CA", "NY"), lob = c("auto", "home"))),
          identical(names(as.data.frame(flipped))[1:2], c("state", "lob")))
coded <- transform(raw, company = factor(ifelse(origin < 1985, 7, 8)))
by_company <- triangle(coded, "origin", "development", "value", keys = "company")
stopifnot(identical(by_company@index$company, c("7", "8")),
          identical(as.data.frame(by_company)$company[1], "7"))
one <- triangle(two, "origin", "development", "paid", keys = "lob")
near(chain_ladder(subset(one, lob = "home"))@total_reserve, 2 * chain_ladder(raa)@total_reserve,
     1e-12)

# Two keys (lob x coverage) with paid and incurred, as in the Python and
# Rust tests: Auto from RAA and Home from GenIns (scaled), origins re-based
# to 2011, each coverage a share of the line with its own tilt by origin.
lob_coverage <- do.call(rbind, lapply(list(list("Auto", "raa", 1), list("Home", "genins", 1e-3)), function(l) {
  rows <- read_long(l[[2]])
  offset <- rows$origin - min(rows$origin)
  do.call(rbind, lapply(list(list("BI", 0.7), list("PD", 0.3)), function(cv) {
    paid <- rows$value * l[[3]] * cv[[2]] * (1 + cv[[2]] * offset / 20)
    data.frame(lob = l[[1]], coverage = cv[[1]], year = 2011 + offset, age = rows$development,
               paid = paid, incurred = paid * 1.2 + 10 * ((seq_along(paid) - 1) %% 3))
  }))
}))
measures <- c("paid", "incurred")
lc <- triangle(lob_coverage, "year", "age", measures, keys = c("lob", "coverage"))
stopifnot(identical(unname(lc@shape), c(4L, 2L, 10L, 10L)))
# Selection by name: one value or several per key, ANDed across keys.
stopifnot(
  identical(subset(lc, lob = "Auto", coverage = "BI")@index, data.frame(lob = "Auto", coverage = "BI")),
  identical(subset(lc, lob = "Auto", coverage = c("PD", "BI"))@index$coverage, c("BI", "PD")),
  identical(subset(lc, coverage = "BI", columns = c("incurred", "paid"))@columns, c("incurred", "paid")),
  identical(subset(lc, coverage = "BI")@index$lob, c("Auto", "Home")),
  identical(unname(subset(lc, columns = "paid")@shape), c(4L, 1L, 10L, 10L))
)
# Grouping equals building with fewer keys; totals are sums of segments.
by_lob <- aggregate(lc, keep = "lob")
stopifnot(identical(by_lob@keys, "lob"), identical(by_lob@index, data.frame(lob = c("Auto", "Home"))))
stopifnot(identical(by_lob@values, triangle(lob_coverage, "year", "age", measures, keys = "lob")@values))
total <- aggregate(lc)
stopifnot(identical(total@keys, character()),
          identical(total@values, triangle(lob_coverage, "year", "age", measures)@values))
for (m in measures) {
  segments <- sum(latest_diagonal(lc)[, m, ])
  near(sum(latest_diagonal(by_lob)[, m, ]), segments, 1e-12)
  near(sum(latest_diagonal(total)[, m, ]), segments, 1e-12)
}
stopifnot(identical(aggregate(lc, keep = c("coverage", "lob"))@index[1, ],
                    data.frame(coverage = "BI", lob = "Auto")))
# Chain ladder on a group equals chain ladder on the summed triangle.
summed <- triangle(lob_coverage[lob_coverage$lob == "Auto", ], "year", "age", measures)
for (m in measures) {
  grouped <- chain_ladder(subset(by_lob, lob = "Auto"), m)
  direct <- chain_ladder(summed, m)
  near(grouped@ultimate, direct@ultimate)
  near(grouped@ldf, direct@ldf)
}
# Incremental triangles are summed as cumulative values.
stopifnot(identical(aggregate(to_incremental(lc), keep = "lob")@values, to_incremental(by_lob)@values))
expect_error_like(subset(lc, line = "Auto"), "no key named line")
expect_error_like(subset(lc, coverage = "GL"), "no segment has coverage = \"GL\"")
expect_error_like(subset(lc, lob = c("Auto", "Auto")), "supplied twice")
expect_error_like(subset(lc, lob = character()), "at least one value")
expect_error_like(subset(lc, lob = "Home", columns = "reported"), "no column named reported")
expect_error_like(subset(lc, "Auto"), "must be named by key")
expect_error_like(subset(lc, lob = "Auto", lob = "Home"), "key lob is supplied twice")
expect_error_like(aggregate(lc, keep = "line"), "no key named line")
expect_error_like(aggregate(lc, keep = c("lob", "lob")), "key lob is supplied twice")

# Views: as.matrix() is one segment x measure of @values; summary() one row
# per segment x measure; print() the grid or the summary, as in Python.
for (k in seq_len(nrow(lc@index))) {
  for (col in measures) {
    v <- as.matrix(lc, lob = lc@index$lob[k], coverage = lc@index$coverage[k], column = col)
    stopifnot(is.matrix(v), identical(dimnames(v), list(origin = lc@origins,
                                                        development = as.character(lc@development))),
              identical(unname(v), unname(lc@values[k, col, , ])))
  }
}
auto_bi <- as.matrix(lc, lob = "Auto", coverage = "BI", column = "paid")
long_lc <- as.data.frame(lc)
row <- long_lc[long_lc$lob == "Auto" & long_lc$coverage == "BI", ][5, ]
near(auto_bi[format(row$origin, "%Y"), as.character(row$development)], row$paid)
stopifnot(identical(unname(as.matrix(raa)), unname(raa@values[1, 1, , ])),
          identical(as.matrix(raa), as.matrix(raa, column = "value")))
expect_error_like(as.matrix(lc, lob = "Auto", column = "paid"), "2 segments match")
expect_error_like(as.matrix(lc, lob = "Auto", coverage = "BI"), "2 columns")
expect_error_like(as.matrix(lc, lob = "Auto", coverage = "BI", column = "x"), "no column named x")
expect_error_like(as.matrix(lc, line = "Auto", column = "paid"), "no key named line")
expect_error_like(as.matrix(lc, lob = c("Auto", "Home"), column = "paid"), "one value per key")
expect_error_like(as.matrix(lc, "Auto"), "must be named by key")

s <- summary(lc)
stopifnot(
  is.data.frame(s), nrow(s) == 8,
  identical(names(s), c("lob", "coverage", "column", "n_origins", "first_origin", "last_origin",
                        "valuation", "latest", "cumulative")),
  identical(s$lob, rep(c("Auto", "Home"), each = 4)),
  identical(s$column, rep(measures, 4)),
  identical(s$n_origins, rep(10L, 8)),
  identical(s$first_origin, rep("2011", 8)), identical(s$last_origin, rep("2020", 8)),
  identical(s$valuation, rep(as.Date("2020-12-31"), 8)),
  all(s$cumulative)
)
# The latest totals are the latest diagonal's sums.
near(s$latest, as.vector(t(apply(latest_diagonal(lc), c(1, 2), sum, na.rm = TRUE))))
near(summary(to_incremental(lc))$latest, s$latest, 1e-9)
stopifnot(!any(summary(to_incremental(lc))$cumulative))

words <- function(lines) strsplit(trimws(lines), "[[:space:]]+")
out <- format(lc)
stopifnot(startsWith(out[1], "Triangle: 4 segments x 2 columns, keys lob, coverage"),
          identical(words(out[2])[[1]], c("lob", "coverage", "column", "n_origins", "first_origin",
                                          "last_origin", "valuation", "latest")),
          identical(words(out[3])[[1]][1:5], c("Auto", "BI", "paid", "10", "2011")),
          length(out) == 2 + 8,
          identical(utils::capture.output(print(lc)), out))
raa_out <- format(raa)
stopifnot(identical(raa_out[1], "Triangle: value (cumulative, valuation 1990-12)"),
          identical(words(raa_out[3])[[1]][1:3], c("1981", "5,012", "8,269")),
          identical(words(raa_out[12])[[1]], c("1990", "2,063")),
          identical(words(format(link_ratios(raa))[3])[[1]][2], "1.650"))
# Empty and holey segments, and a large triangle, print.
holey <- triangle(data.frame(lob = c("auto", "auto", "home"), year = 2020, age = c(12, 36, 12),
                             paid = c(1, 3, NA)), "year", "age", "paid", keys = "lob")
stopifnot(any(grepl("home", format(holey))),
          identical(is.na(summary(holey)$first_origin), c(FALSE, TRUE)),
          is.na(summary(holey)$valuation[2]), summary(holey)$latest[2] == 0,
          identical(words(format(subset(holey, lob = "home"))[3])[[1]], "2020"),
          identical(words(format(subset(holey, lob = "auto"))[3])[[1]], c("2020", "1", "3")),
          all(is.na(as.matrix(holey, lob = "home"))))
big_rows <- do.call(rbind, lapply(0:39, function(k) data.frame(year = 2000 + k, age = 12 * seq_len(40 - k))))
big <- triangle(transform(big_rows, paid = 1), "year", "age", "paid")
big_out <- format(big)
stopifnot(length(big_out) == 1 + 1 + 20 + 1 + 1,
          identical(big_out[length(big_out)], "[40 origins x 40 ages]"),
          length(words(big_out[2])[[1]]) == 12 + 1,
          length(format(big, max_rows = 0, max_cols = 0)) == 42,
          identical(dim(as.matrix(big)), c(40L, 40L)))
invisible(utils::capture.output(print(big), print(holey)))
bad_rows <- tryCatch(format(big, max_rows = -1), error = function(e) e)
stopifnot(grepl("max_rows", conditionMessage(bad_rows)),
          identical(conditionCall(bad_rows)[[1]], quote(format)))
expect_error_like(summary(triangle(data.frame(column = "a", year = 2020, age = 12, paid = 1),
                                   "year", "age", "paid", keys = "column")), "clashes")

# Every segment at once: each segment's fit equals fitting it alone.
cl <- chain_ladder(lc, "paid")
m <- mack(lc, "paid")
stopifnot(identical(cl@keys, c("lob", "coverage")), identical(cl@index, lc@index),
          length(cl@reserve) == 40, identical(names(cl@reserve)[11], "Auto / PD / 2011"))
for (k in seq_len(nrow(lc@index))) {
  lob <- lc@index$lob[k]
  coverage <- lc@index$coverage[k]
  rows <- (10 * (k - 1) + 1):(10 * k)
  alone <- subset(lc, lob = lob, coverage = coverage)
  one_cl <- chain_ladder(alone, "paid")
  one_m <- mack(alone, "paid")
  stopifnot(identical(unname(cl@reserve[rows]), unname(one_cl@reserve)),
            identical(cl@origins[rows], one_cl@origins),
            identical(unname(m@standard_error[rows]), unname(one_m@standard_error)))
  seg <- segment(m, lob = lob, coverage = coverage)
  stopifnot(S7::S7_inherits(seg, mack_fit), identical(seg@ldf, one_m@ldf),
            identical(seg@total_standard_error, one_m@total_standard_error),
            identical(seg@reserve, one_m@reserve))
}
near(cl@total_reserve, sum(cl@reserve))
# Long results.
df <- as.data.frame(m)
stopifnot(identical(names(df), c("lob", "coverage", "origin", "latest", "ultimate", "reserve",
                                 "process_risk", "parameter_risk", "standard_error")),
          nrow(df) == 40, identical(df$origin[11], "2011"),
          identical(df$reserve, unname(m@reserve)))
totals <- totals_frame(m)
home_pd <- segment(m, lob = "Home", coverage = "PD")
stopifnot(identical(totals$lob, c("Auto", "Auto", "Home", "Home")),
          identical(totals$standard_error[4], home_pd@total_standard_error),
          identical(totals$reserve[4], home_pd@total_reserve))
dev <- development_frame(cl)
stopifnot(identical(names(dev), c("lob", "coverage", "development", "ldf", "cdf", "sigma", "std_err")),
          nrow(dev) == 40, is.na(dev$ldf[40]),
          identical(dev$ldf[31:39], unname(segment(cl, lob = "Home", coverage = "PD")@ldf)))
# Per-age properties and the totals' standard errors need one segment.
expect_error_like(cl@ldf, "4 segments; use development_frame()")
expect_error_like(m@total_standard_error, "use totals_frame()")
expect_error_like(segment(cl, lob = "Auto"), "2 segments match")
expect_error_like(segment(cl, line = "Auto"), "no key named line")
expect_error_like(segment(cl, "Auto"), "must be named by key")
expect_error_like(segment(cl, lob = c("Auto", "Home")), "one value per key")
invisible(utils::capture.output(print(cl), print(m)))
# The bootstrap: one joint distribution over lob x coverage x origin.
boot_lc <- odp_bootstrap(lc, "paid", n_sims = 2000, seed = 3)
r <- boot_lc@reserves
stopifnot(identical(r@dims, c("lob", "coverage", "origin")), nrow(r@keys) == 40,
          identical(r@keys$coverage[11], "PD"))
home_bi <- segment(boot_lc, lob = "Home", coverage = "BI")
again <- segment(odp_bootstrap(lc, "paid", n_sims = 2000, seed = 3), lob = "Home", coverage = "BI")
stopifnot(identical(home_bi@reserves@dims, r@dims), nrow(home_bi@reserves@keys) == 10,
          identical(draw_matrix(home_bi@reserves), draw_matrix(again@reserves)))
by_lob_reserves <- aggregate(r, keep = "lob")
stopifnot(identical(by_lob_reserves@keys$lob, c("Auto", "Home")))
near(mean(by_lob_reserves), mean(r))
boot_totals <- totals_frame(boot_lc)
stopifnot(identical(names(boot_totals), c("lob", "coverage", "latest", "ultimate", "reserve",
                                          "scale", "mean", "std_dev")))
alone <- odp_bootstrap(subset(lc, lob = "Home", coverage = "BI"), "paid", n_sims = 10)
stopifnot(identical(boot_totals$scale[3], alone@scale), identical(home_bi@scale, alone@scale),
          nrow(as.data.frame(boot_lc)) == 40)
near(as.data.frame(boot_lc)$mean[11], mean(draw_matrix(r)[, 11]), 1e-12)
near(boot_totals$mean[3], mean(home_bi@reserves), 1e-12)
expect_error_like(boot_lc@scale, "use totals_frame()")
expect_error_like(boot_lc@residuals, "use segment()")
invisible(utils::capture.output(print(boot_lc)))

# Grain: quarterly origins and ages to annual origins.
q <- data.frame(
  origin = as.Date(rep(c("2020-01-01", "2020-04-01", "2020-07-01", "2020-10-01"), 4:1)),
  age = c(3, 6, 9, 12, 3, 6, 9, 3, 6, 3),
  paid = 10
)
qt <- triangle(q, "origin", "age", "paid", origin_grain = "Q", development_grain = "Q")
stopifnot(identical(qt@origins, c("2020Q1", "2020Q2", "2020Q3", "2020Q4")),
          identical(qt@development, c(3L, 6L, 9L, 12L)))
yt <- grain(qt, "Y", "Q")
stopifnot(identical(yt@origins, "2020"), yt@origin_grain == "Y", yt@development_grain == "Q")
near(yt@values[1, 1, "2020", ], c(10, 20, 30, 40))
near(grain(qt, "Y", "Y")@values[1, 1, "2020", "12"], 40)
# development_grain defaults to origin_grain.
ya <- grain(qt, "Y")
stopifnot(ya@development_grain == "Y", identical(ya@values, grain(qt, "Y", "Y")@values),
          identical(ya@valuation, as.Date("2020-12-31")))
stopifnot(identical(grain(raa, "Y", "Y")@values, raa@values))

# Errors are ordinary R errors carrying the Rust message.
near(chain_ladder(multi, "paid")@total_reserve, 3 * chain_ladder(raa)@total_reserve, 1e-12)
expect_error_like(chain_ladder(multi), "several columns")
expect_error_like(chain_ladder(raa, "paid"), "no column named paid")
expect_error_like(chain_ladder(raa, tail = 0), "tail factor 0")
expect_error_like(chain_ladder(raa, average = "median"), "should be one of")
expect_error_like(chain_ladder(raa, tail = NA), "tail must be a single number")
expect_error_like(triangle(transform(raw, value = factor(value)), "origin", "development", "value"),
                  "value column value must be numeric")
expect_error_like(triangle(transform(raw, value = as.character(value)), "origin", "development", "value"),
                  "value column value must be numeric")
expect_error_like(triangle(transform(raw, development = factor(development)), "origin", "development",
                           "value"), "development column development must be numeric")
expect_error_like(grain(qt, "X"), "origin_grain must be")
expect_error_like(grain(qt), "origin_grain")
expect_error_like(grain(raa, "Q", "Q"), "invalid grain change")
expect_error_like(subset(multi, columns = c("paid", "paid")), "supplied twice")
expect_error_like(subset(multi, lob = "nope"), "no segment has lob = \"nope\"")
expect_error_like(mack(raa, sigma_interpolation = "x"), "should be one of")
short <- triangle(data.frame(y = c(2020, 2020, 2021), d = c(12, 24, 12), v = c(1, 2, 3)), "y", "d", "v")
expect_error_like(mack(short), "at least 3 development ages")
expect_error_like(triangle(transform(raw, value = Inf), "origin", "development", "value"), "infinite")
expect_error_like(triangle(raw, "origin", "development", "value", origin_grain = "Y",
                           development_grain = "Y", cumulative = TRUE, keys = "lob"), "no columns named lob")
expect_error_like(triangle(two, "origin", "development", "paid", keys = c("lob", "lob")),
                  "key lob is supplied twice")
expect_error_like(triangle(two, "origin", "development", "paid", keys = "paid"),
                  "paid is both a key and a value column")
expect_error_like(triangle(transform(two, lob = replace(lob, 3, NA)), "origin", "development", "paid",
                           keys = "lob"), "key column lob has missing values")
clash <- triangle(transform(raw, year = origin, origin = "x"), "year", "development", "value",
                  keys = "origin")
expect_error_like(as.data.frame(clash), "clashes with the")
expect_error_like(triangle(transform(raw, development = replace(development, 2, 25)), "origin", "development",
                           "value", development_grain = "Y"), "not on the development grid")
expect_error_like(triangle(transform(raw, origin = origin + 0.5), "origin", "development", "value"),
                  "whole-number years")

# ODP bootstrap against R ChainLadder's BootChainLadder
# (validation/reference/reserving_bootstrap_r.csv). The tolerances for
# simulated quantities assume 20000 simulations with this seed, as in
# validation/tests/reserving.rs.
boot_lines <- grep("^#", readLines(validation("reference", "reserving_bootstrap_r.csv")),
                   value = TRUE, invert = TRUE)
boot_fields <- lapply(strsplit(boot_lines, ",", fixed = TRUE), `[`, 1:7)
boot_ref <- utils::read.csv(text = vapply(boot_fields, paste, "", collapse = ","),
                            colClasses = c(arg = "character"))
boot_ref <- boot_ref[boot_ref$dataset == "raa", ]
boot <- odp_bootstrap(raa, n_sims = 20000, seed = 20261004)
stopifnot(S7::S7_inherits(boot, odp_bootstrap_fit),
          S7::S7_inherits(boot@reserves, predictive_distribution),
          S7::S7_inherits(boot@chain_ladder, chain_ladder_fit))
near(boot@scale, boot_ref$expected[boot_ref$quantity == "scale"], 1e-9)
res_ref <- boot_ref[boot_ref$quantity == "residual", ]
stopifnot(nrow(res_ref) == 55)
for (r in seq_len(nrow(res_ref))) {
  at <- strsplit(res_ref$arg[r], ":", fixed = TRUE)[[1]]
  got <- boot@residuals[at[1], as.integer(at[2]) + 1L]
  want <- res_ref$expected[r]
  if (!(abs(got - want) <= 1e-9 * max(abs(want), 1))) {
    stop(sprintf("residual %s: got %.17g, want %.17g", res_ref$arg[r], got, want), call. = FALSE)
  }
}
# Origin x development matrices, NA below the latest diagonal.
stopifnot(identical(dim(boot@residuals), c(10L, 10L)),
          identical(dimnames(boot@fitted)$origin, raa@origins),
          is.na(boot@residuals["1990", "24"]), is.na(boot@fitted["1990", "24"]))
near(rowSums(boot@fitted, na.rm = TRUE), unname(boot@chain_ladder@latest), 1e-12)
near(boot@chain_ladder@total_reserve, chain_ladder(raa)@total_reserve)
for (quantity in c("mean_total", "sd_total")) {
  row <- boot_ref[boot_ref$method == "odp_bootstrap_gamma" & boot_ref$quantity == quantity, ]
  got <- if (quantity == "mean_total") mean(boot@reserves) else sqrt(variance(boot@reserves))
  if (!(abs(got - row$expected) <= row$abs_tol)) {
    stop(sprintf("%s: got %.17g, want %.17g +- %g", quantity, got, row$expected, row$abs_tol),
         call. = FALSE)
  }
}
# One component per origin period; the oldest origin is fully developed.
stopifnot(identical(boot@reserves@dims, "origin"),
          identical(boot@reserves@keys$origin, raa@origins),
          identical(boot@origins, raa@origins),
          identical(boot@development, raa@development),
          boot@reserves@n_sims == 20000,
          identical(dim(draw_matrix(boot@reserves)), c(20000L, 10L)),
          all(draw_matrix(boot@reserves)[, 1] == 0),
          provenance(boot@reserves)$model == "odp_bootstrap")
# Same seed, same draws; process error adds variance.
a <- odp_bootstrap(raa, n_sims = 500, seed = 7)
stopifnot(identical(draw_matrix(a@reserves), draw_matrix(odp_bootstrap(raa, n_sims = 500, seed = 7)@reserves)),
          !identical(draw_matrix(a@reserves), draw_matrix(odp_bootstrap(raa, n_sims = 500, seed = 8)@reserves)))
gamma <- odp_bootstrap(raa, "value", n_sims = 5000, seed = 1, process = "gamma")
param <- odp_bootstrap(raa, "value", n_sims = 5000, seed = 1, process = "none")
stopifnot(variance(param@reserves) < variance(gamma@reserves), param@scale == gamma@scale)
invisible(utils::capture.output(print(boot)))
expect_error_like(odp_bootstrap(raa, process = "poisson"), "should be one of")
expect_error_like(odp_bootstrap(raa, n_sims = 0), "n_sims must be positive")
expect_error_like(odp_bootstrap(raa, n_sims = 1.5), "n_sims must be a non-negative whole number")
expect_error_like(odp_bootstrap(raa, seed = NA), "seed must be a single number")
stopifnot(identical(odp_bootstrap(multi, "paid", n_sims = 10)@reserves@dims, c("lob", "state", "origin")))
expect_error_like(odp_bootstrap(raa, "paid", n_sims = 10), "no column named paid")
tiny <- triangle(data.frame(year = c(2020, 2020, 2021), age = c(12, 24, 12), paid = c(1, 2, 1)),
                 "year", "age", "paid")
expect_error_like(odp_bootstrap(tiny, n_sims = 10), "degrees of freedom")

# Merz-Wuthrich one-year view against R ChainLadder's
# CDR(MackChainLadder(tri), dev = "all"): every reference row
# (validation/reference/reserving_cdr_r.csv), as in
# validation/tests/reserving_cdr.rs. R reports one calendar year per age,
# so years past the run-off are zero. The source column holds commas too.
cdr_lines <- grep("^#", readLines(validation("reference", "reserving_cdr_r.csv")),
                  value = TRUE, invert = TRUE)
cdr_fields <- lapply(strsplit(cdr_lines, ",", fixed = TRUE), `[`, 1:7)
cdr_ref <- utils::read.csv(text = vapply(cdr_fields, paste, "", collapse = ","),
                           colClasses = c(arg = "character"))
cdr_sigma <- c(cdr = "log-linear", cdr_sigma_mack = "mack")
cdr_fits <- list()
for (i in seq_len(nrow(cdr_ref))) {
  row <- cdr_ref[i, ]
  key <- paste(row$dataset, row$method)
  if (is.null(cdr_fits[[key]])) {
    fit <- mack(load_triangle(row$dataset), sigma_interpolation = cdr_sigma[[row$method]])
    cdr_fits[[key]] <- list(mack = fit, cdr = claims_development_result(fit))
  }
  fit <- cdr_fits[[key]]$mack
  cdr <- cdr_fits[[key]]$cdr
  years <- cdr@by_calendar_year
  got <- switch(row$quantity,
    cdr_se = {
      parts <- strsplit(row$arg, ":", fixed = TRUE)[[1]]
      k <- as.integer(parts[2])
      if (k <= ncol(years)) years[parts[1], k] else 0
    },
    total_cdr_se = {
      k <- as.integer(row$arg)
      if (k <= ncol(years)) cdr@total_by_calendar_year[[k]] else 0
    },
    reserve = fit@reserve[[row$arg]],
    one_year_se = cdr@one_year_standard_error[[row$arg]],
    mack_se = fit@standard_error[[row$arg]],
    total_reserve = fit@total_reserve,
    total_one_year_se = cdr@total_one_year_standard_error,
    total_mack_se = fit@total_standard_error,
    stop("unknown reference quantity ", row$quantity, call. = FALSE)
  )
  # R's Mack.S.E. is also the summed yearly run-off.
  run_off <- switch(row$quantity,
    mack_se = cdr@run_off_standard_error[[row$arg]],
    total_mack_se = cdr@total_run_off_standard_error,
    got
  )
  for (value in c(got, run_off)) {
    err <- abs(value - row$expected)
    if (!(value == row$expected || err <= row$abs_tol || err <= row$rel_tol * abs(row$expected))) {
      stop(sprintf("%s %s %s[%s]: got %.17g, want %.17g", row$dataset, row$method, row$quantity,
                   row$arg, value, row$expected), call. = FALSE)
    }
  }
}
stopifnot(nrow(cdr_ref) > 1800, length(cdr_fits) == 10)

# Merz and Wuthrich (2008), Table 4, printed to the unit: the totals and
# the youngest origin.
mw <- mack(load_triangle("mw2008"), sigma_interpolation = "mack")
cdr <- claims_development_result(mw)
stopifnot(
  S7::S7_inherits(cdr, claims_development_result),
  abs(mw@total_reserve - 2237826) <= 1,
  abs(cdr@total_one_year_standard_error - 81080) <= 1,
  abs(cdr@total_run_off_standard_error - 108401) <= 1,
  abs(cdr@one_year_standard_error[["2009"]] - 53320) <= 1,
  abs(mw@standard_error[["2009"]] - 69552) <= 1,
  identical(cdr@origins, as.character(2001:2009)),
  identical(dim(cdr@by_calendar_year), c(9L, 8L)),
  identical(names(dimnames(cdr@by_calendar_year)), c("origin", "calendar_year")),
  identical(cdr@by_calendar_year[, "1"], cdr@one_year_standard_error),
  cdr@one_year_standard_error[["2001"]] == 0,
  # Fully developed, and one factor from it: the run-off is over.
  all(cdr@by_calendar_year["2001", ] == 0),
  cdr@by_calendar_year["2002", 1] > 0, all(cdr@by_calendar_year["2002", -1] == 0),
  identical(names(cdr@total_by_calendar_year), as.character(1:8)),
  identical(cdr@total_by_calendar_year[["1"]], cdr@total_one_year_standard_error)
)
near(unname(cdr@run_off_standard_error), unname(mw@standard_error), 1e-9)
df <- as.data.frame(claims_development_result(mack(raa)))
stopifnot(identical(names(df), c("origin", paste0("cdr_", 1:9), "run_off")),
          identical(df$origin, raa@origins),
          identical(df$cdr_1, unname(claims_development_result(mack(raa))@one_year_standard_error)))
invisible(utils::capture.output(print(cdr)))
expect_error_like(claims_development_result(mack(raa, average = "simple")),
                  "claims development result: needs volume-weighted")
expect_error_like(claims_development_result(mack(lc, "paid")), "4 segments; use segment()")
# Merz and Wuthrich assume no tail: a factor other than 1, or a factor of 1
# that replaces estimated factors, is an error.
for (tail in list(1.05, tail_constant(1.05), tail_log_linear(), tail_constant(1, attachment_age = 84))) {
  expect_error_like(claims_development_result(mack(raa, tail = tail)),
                    "claims development result: needs no tail factor")
}
stopifnot(identical(claims_development_result(mack(raa, tail = 1))@by_calendar_year,
                    claims_development_result(mack(raa))@by_calendar_year))
expect_error_like(claims_development_result(chain_ladder(raa)), "fit must be a mack_fit")
home_pd <- segment(mack(lc, "paid"), lob = "Home", coverage = "PD")
near(claims_development_result(home_pd)@total_run_off_standard_error,
     home_pd@total_standard_error, 1e-9)
# Clark's growth curves against R ChainLadder 0.2.21 and chainladder-python
# 0.10.1 (validation/reference/reserving_clark_*.csv), with each row's own
# tolerance, as validation/tests/reserving_clark.rs.
premium_long <- read_long("genins_premium")
clark_tris <- list(raa = raa, genins = tris$genins,
                   genins_premium = triangle(premium_long, "origin", "development",
                                             c("paid", "premium")))
clark_fits <- list()
clark_fit_for <- function(dataset, method) {
  # The optimizer setting of the R reference does not change our fit.
  parts <- strsplit(method, ";", fixed = TRUE)[[1]]
  parts <- parts[!startsWith(parts, "optim=")]
  key <- paste(dataset, paste(parts, collapse = ";"))
  if (is.null(clark_fits[[key]])) {
    kv <- strsplit(parts[-1], "=", fixed = TRUE)
    s <- stats::setNames(vapply(kv, `[`, "", 2), vapply(kv, `[`, "", 1))
    max_age <- if (s[["max_age"]] == "inf") Inf else as.numeric(s[["max_age"]])
    tri <- clark_tris[[dataset]]
    clark_fits[[key]] <<- if (parts[1] == "clark_cape_cod") {
      clark_cape_cod(tri, "paid", "premium", curve = s[["curve"]], max_age = max_age)
    } else {
      clark_ldf(tri, if (dataset == "genins_premium") "paid" else "value", curve = s[["curve"]],
                max_age = max_age)
    }
  }
  clark_fits[[key]]
}
clark_value <- function(fit, quantity, arg) {
  cov <- fit@covariance
  p <- nrow(cov)
  switch(quantity,
    omega = fit@omega, theta = fit@theta, elr = fit@elr, sigma2 = fit@scale,
    omega_se = sqrt(cov[p - 1, p - 1]), theta_se = sqrt(cov[p, p]), elr_se = sqrt(cov[1, 1]),
    total_ultimate = fit@total_ultimate, total_reserve = fit@total_reserve,
    total_process_se = fit@total_process_risk, total_parameter_se = fit@total_parameter_risk,
    total_standard_error = fit@total_standard_error,
    ldf = growth(fit, as.numeric(arg) + 12) / growth(fit, as.numeric(arg)),
    expected_ultimate = fit@expected_ultimate[[arg]], ultimate = fit@ultimate[[arg]],
    reserve = fit@reserve[[arg]], process_se = fit@process_risk[[arg]],
    parameter_se = fit@parameter_risk[[arg]], standard_error = fit@standard_error[[arg]],
    stop("unknown quantity ", quantity)
  )
}
for (reference in c("reserving_clark_r.csv", "reserving_clark_python.csv")) {
  clark_lines <- grep("^#", readLines(validation("reference", reference)), value = TRUE, invert = TRUE)
  clark_fields <- lapply(strsplit(clark_lines, ",", fixed = TRUE), `[`, 1:7)
  clark_ref <- utils::read.csv(text = vapply(clark_fields, paste, "", collapse = ","),
                               colClasses = c(arg = "character"))
  stopifnot(nrow(clark_ref) > 90)
  for (r in seq_len(nrow(clark_ref))) {
    row <- clark_ref[r, ]
    got <- clark_value(clark_fit_for(row$dataset, row$method), row$quantity, row$arg)
    err <- abs(got - row$expected)
    abs_tol <- if (is.na(row$abs_tol)) 0 else row$abs_tol
    if (!(got == row$expected || err <= abs_tol || err <= row$rel_tol * abs(row$expected))) {
      stop(sprintf("%s %s %s[%s]: got %.17g, want %.17g", row$dataset, row$method, row$quantity,
                   row$arg, got, row$expected), call. = FALSE)
    }
  }
}

# The fit's fields, as in python/tests/test_reserving.py.
clark <- clark_ldf(raa)
stopifnot(S7::S7_inherits(clark, clark_fit), S7::S7_inherits(clark@chain_ladder, chain_ladder_fit),
          clark@method == "ldf", clark@curve == "loglogistic", clark@max_age == Inf,
          is.null(clark@elr), is.null(clark@exposure),
          identical(clark@latest, chain_ladder(raa)@latest),
          identical(clark@origins, raa@origins), identical(clark@development, raa@development),
          identical(names(clark@reserve), raa@origins),
          identical(dim(clark@covariance), c(12L, 12L)),
          identical(rownames(clark@covariance), c(raa@origins, "omega", "theta")),
          clark@scale > 0, growth(clark, Inf) == 1)
# RAA has 55 observed incremental values (act_reserving's unit test).
stopifnot(identical(clark@n_observations, 55L), clark@origin_width == 12)
# Every origin starts at age 0, so U = latest / G(latest age); the 1990
# origin is at 12 months, 6 from the average date of loss.
near(clark@expected_ultimate[["1990"]], clark@latest[["1990"]] / growth(clark, 12), 1e-12)
near(growth(clark, c(12, 24)), c(growth(clark, 12), growth(clark, 24)))
near(clark@reserve, clark@ultimate - clark@latest, 1e-9)
near(clark@total_reserve, sum(clark@reserve), 1e-12)
near(clark@standard_error, sqrt(clark@process_risk^2 + clark@parameter_risk^2), 1e-12)
df <- as.data.frame(clark)
stopifnot(identical(names(df), c("origin", "latest", "ultimate", "reserve", "expected_ultimate",
                                 "process_risk", "parameter_risk", "standard_error")),
          nrow(df) == 10, identical(df$standard_error, unname(clark@standard_error)))
cc <- clark_cape_cod(clark_tris$genins_premium, "paid", "premium", curve = "weibull")
stopifnot(cc@method == "cape_cod", cc@curve == "weibull", cc@exposure[[1]] == 1e7,
          identical(names(cc@exposure), cc@origins),
          identical(dimnames(cc@covariance)[[1]], c("elr", "omega", "theta")),
          cc@elr > 0)
near(cc@expected_ultimate, cc@elr * cc@exposure)
stopifnot(identical(names(as.data.frame(cc))[5:6], c("exposure", "expected_ultimate")),
          "elr" %in% names(totals_frame(cc)))
weibull <- clark_ldf(raa, curve = "weibull", max_age = 240)
stopifnot(weibull@max_age == 240, identical(clark_ldf(raa, max_age = NULL)@reserve, clark@reserve))
invisible(utils::capture.output(print(clark), print(cc), print(weibull)))

# Every segment at once: ten times the losses give the same curve and ten
# times the amounts and standard errors.
two_lob <- rbind(transform(premium_long, lob = "a"), transform(premium_long, lob = "b", paid = 10 * paid))
both <- triangle(two_lob, "origin", "development", c("paid", "premium"), keys = "lob")
for (fit_one in list(function(t) clark_ldf(t, "paid", max_age = 240),
                     function(t) clark_cape_cod(t, "paid", "premium", curve = "weibull"))) {
  fit <- fit_one(both)
  alone <- fit_one(clark_tris$genins_premium)
  a <- segment(fit, lob = "a")
  b <- segment(fit, lob = "b")
  stopifnot(S7::S7_inherits(a, clark_fit), identical(fit@keys, "lob"), length(fit@reserve) == 20,
            identical(names(fit@reserve)[11], "b / 2001"))
  near(a@ultimate, alone@ultimate, 1e-9)
  near(a@omega, alone@omega, 1e-9)
  near(b@theta, a@theta, 1e-7)
  near(b@total_standard_error, 10 * a@total_standard_error, 1e-6)
  totals <- totals_frame(fit)
  stopifnot(identical(totals$lob, c("a", "b")), identical(totals$omega, c(a@omega, b@omega)),
            identical(as.data.frame(fit)$standard_error, unname(fit@standard_error)))
  near(totals$reserve[1], a@total_reserve, 1e-12)
  expect_error_like(fit@omega, "2 segments; use totals_frame()")
  expect_error_like(fit@covariance, "use segment()")
  expect_error_like(growth(fit, 12), "use segment()")
  expect_error_like(fit@n_observations, "use segment()")
  stopifnot(fit@origin_width == 12)
  if (fit@method == "ldf") stopifnot(is.null(fit@elr)) else expect_error_like(fit@elr, "2 segments")
  invisible(utils::capture.output(print(fit)))
}
expect_error_like(clark_ldf(raa, curve = "gompertz"), "should be one of")
expect_error_like(clark_ldf(raa, max_age = 119), "max_age = 119 is invalid")
expect_error_like(clark_ldf(raa, max_age = NA), "max_age must be a single number")
expect_error_like(clark_ldf(short), "at least 4 development ages")
expect_error_like(clark_ldf(multi), "several columns")
expect_error_like(clark_cape_cod(raa, "value", "premium"), "no column named premium")
expect_error_like(clark_cape_cod(raa, "value"), "argument \"exposure\" is missing")
expect_error_like(clark_cape_cod(raa, "value", c("a", "b")), "exposure must be the name of one column")
expect_error_like(development_frame(clark), "development_frame(fit@chain_ladder)")
expect_error_like(growth(chain_ladder(raa), 12), "fit must be a clark_fit")

# Expected-loss methods against chainladder-python 0.10.1
# (validation/reference/reserving_expected_loss_python.csv): every row.
# Paid losses with the latest premium as exposure.
premium_triangle <- function(name) triangle(read_long(name), "origin", "development", c("paid", "premium"))
premium_sets <- c("clrd_wkcomp", "genins_premium")
premium_tris <- stats::setNames(lapply(premium_sets, premium_triangle), premium_sets)
el_lines <- grep("^#", readLines(validation("reference", "reserving_expected_loss_python.csv")),
                 value = TRUE, invert = TRUE)
el_ref <- utils::read.csv(text = el_lines, colClasses = c(arg = "character"))
stopifnot(nrow(el_ref) == 816)
# The fit a reference `method` (name;key=value;...) describes.
el_fit_for <- function(dataset, method) {
  parts <- strsplit(method, ";", fixed = TRUE)[[1]]
  kv <- strsplit(parts[-1], "=", fixed = TRUE)
  settings <- stats::setNames(lapply(kv, `[`, 2), vapply(kv, `[`, "", 1))
  average <- settings$average
  num <- lapply(settings[names(settings) != "average"], as.numeric)
  fit <- switch(parts[1], expected_loss = expected_loss, bornhuetter_ferguson = bornhuetter_ferguson,
                benktander = benktander, cape_cod = cape_cod,
                stop("no method ", parts[1], call. = FALSE))
  do.call(fit, c(list(premium_tris[[dataset]], "paid", "premium", average = average), num))
}
el_fits <- list()
for (i in seq_len(nrow(el_ref))) {
  case <- el_ref[i, ]
  key <- paste(case$dataset, case$method)
  if (is.null(el_fits[[key]])) el_fits[[key]] <- el_fit_for(case$dataset, case$method)
  fit <- el_fits[[key]]
  got <- switch(case$quantity,
                total_ultimate = fit@total_ultimate, total_reserve = fit@total_reserve,
                ultimate = fit@ultimate[[case$arg]], reserve = fit@reserve[[case$arg]],
                apriori = fit@trended_apriori[[case$arg]],
                detrended_apriori = fit@apriori[[case$arg]],
                stop("no quantity ", case$quantity, call. = FALSE))
  err <- abs(got - case$expected)
  if (!(err <= case$abs_tol || err <= case$rel_tol * abs(case$expected))) {
    stop(sprintf("%s %s %s %s: got %.17g, want %.17g", case$dataset, case$method, case$quantity,
                 case$arg, got, case$expected), call. = FALSE)
  }
}
stopifnot(length(el_fits) == 2 * 14)

# The family's identities, as in the Rust and Python tests.
wk <- premium_tris$clrd_wkcomp
el <- expected_loss(wk, "paid", "premium", apriori = 0.7)
bf <- bornhuetter_ferguson(wk, "paid", "premium", apriori = 0.7)
wk_cl <- chain_ladder(wk, "paid")
stopifnot(S7::S7_inherits(el, expected_loss_fit), S7::S7_inherits(bf, expected_loss_fit),
          identical(benktander(wk, "paid", "premium", apriori = 0.7, n_iters = 0)@ultimate, el@ultimate),
          identical(benktander(wk, "paid", "premium", apriori = 0.7)@ultimate, bf@ultimate),
          identical(unname(el@apriori), rep(0.7, 10)),
          identical(el@exposure[["1988"]], 1691130),
          identical(names(bf@ultimate), as.character(1988:1997)),
          identical(bf@chain_ladder@ultimate, wk_cl@ultimate),
          identical(bf@ldf, wk_cl@ldf), identical(bf@cdf, wk_cl@cdf),
          identical(bf@latest, wk_cl@latest), identical(bf@development, wk_cl@development),
          identical(bf@keys, character()), identical(dim(bf@index), c(1L, 0L)),
          identical(unname(bf@reserve), unname(bf@ultimate - bf@latest)))
near(unname(el@ultimate), 0.7 * unname(el@exposure), 1e-15)
near(benktander(wk, "paid", "premium", apriori = 0.7, n_iters = 10000)@ultimate, wk_cl@ultimate)
near(bf@total_reserve, sum(bf@reserve))
# Cape Cod without decay keeps each origin's own loss ratio: the chain ladder.
cc <- cape_cod(wk, "paid", "premium", trend = 0.05, decay = 0)
stopifnot(S7::S7_inherits(cc, cape_cod_fit), S7::S7_inherits(cc@expected_loss, expected_loss_fit),
          identical(cc@expected_loss@apriori, cc@apriori),
          identical(cc@trended_apriori[["1997"]], cc@apriori[["1997"]]),
          identical(cc@chain_ladder@ultimate, wk_cl@ultimate))
near(cc@ultimate, wk_cl@ultimate)
near(cc@trended_apriori[["1988"]] / cc@apriori[["1988"]], 1.05^9)
# Settings reach Rust: the development pattern and Cape Cod's decay.
simple <- bornhuetter_ferguson(wk, "paid", "premium", apriori = 0.7, average = "simple", tail = 1.01)
stopifnot(identical(simple@ldf, chain_ladder(wk, "paid", average = "simple")@ldf),
          identical(simple@chain_ladder@tail, 1.01))
# The tail is chain_ladder()'s: a number or a tail estimator.
bondy <- bornhuetter_ferguson(wk, "paid", "premium", apriori = 0.7, tail = tail_bondy())
stopifnot(identical(bondy@cdf, chain_ladder(wk, "paid", tail = tail_bondy())@cdf),
          !identical(bondy@cdf, bf@cdf),
          identical(cape_cod(wk, "paid", "premium", tail = tail_constant(1.01))@chain_ladder@tail, 1.01))
expect_error_like(expected_loss(wk, "paid", "premium", tail = "a"), "tail must be a single number")
stopifnot(!identical(cape_cod(wk, "paid", "premium", decay = 0.5)@ultimate,
                     cape_cod(wk, "paid", "premium")@ultimate))
el_df <- as.data.frame(bf)
stopifnot(identical(names(el_df), c("origin", "latest", "ultimate", "reserve", "exposure", "apriori")),
          identical(el_df$ultimate, unname(bf@ultimate)),
          identical(names(as.data.frame(cc)), c("origin", "latest", "ultimate", "reserve", "exposure",
                                               "apriori", "trended_apriori")),
          identical(totals_frame(bf)$exposure, sum(bf@exposure)),
          identical(development_frame(bf), development_frame(wk_cl)))
invisible(utils::capture.output(print(bf), print(cc)))

# Every segment at once, each with its own exposure.
both_long <- do.call(rbind, lapply(premium_sets, function(name) transform(read_long(name), lob = name)))
both <- triangle(both_long, "origin", "development", c("paid", "premium"), keys = "lob")
for (method in list(function(t) bornhuetter_ferguson(t, "paid", "premium", apriori = 0.6),
                    function(t) cape_cod(t, "paid", "premium", trend = 0.02, decay = 0.8))) {
  fit <- method(both)
  alone <- method(wk)
  seg <- segment(fit, lob = "clrd_wkcomp")
  stopifnot(identical(fit@keys, "lob"), length(fit@ultimate) == 20,
            identical(names(fit@ultimate)[1], "clrd_wkcomp / 1988"),
            identical(S7::S7_class(seg), S7::S7_class(fit)),
            identical(seg@exposure, alone@exposure),
            identical(unname(fit@ultimate[1:10]), unname(seg@ultimate)),
            identical(totals_frame(fit)$exposure[1], sum(alone@exposure)),
            nrow(as.data.frame(fit)) == 20, nrow(development_frame(fit)) == 20)
  # Cape Cod trends to the triangle's valuation (2010 here, 1997 alone) and
  # back, so the two agree to rounding.
  near(seg@ultimate, alone@ultimate, 1e-14)
  near(totals_frame(fit)$reserve[1], alone@total_reserve, 1e-12)
  expect_error_like(fit@ldf, "2 segments; use development_frame()")
  invisible(utils::capture.output(print(fit)))
}

# Errors.
holey_premium <- triangle(data.frame(year = c(2020, 2020, 2021), age = c(12, 24, 12),
                                     paid = c(100, 150, 200), premium = c(250, 250, NA)),
                          "year", "age", c("paid", "premium"))
expect_error_like(bornhuetter_ferguson(holey_premium, "paid", "premium"),
                  "origin 2021 has no observed, finite, positive exposure in column premium")
expect_error_like(expected_loss(wk, "paid", "exposure"), "no column named exposure")
expect_error_like(expected_loss(wk, "paid"), "exposure")
expect_error_like(expected_loss(wk, "paid", 2), "exposure must be the name of a column")
expect_error_like(bornhuetter_ferguson(wk, "paid", "premium", apriori = 0), "apriori = 0 is invalid")
expect_error_like(bornhuetter_ferguson(wk, "paid", "premium", apriori = NA), "apriori must be a single number")
expect_error_like(cape_cod(wk, "paid", "premium", decay = 2), "decay = 2 is invalid")
expect_error_like(cape_cod(wk, "paid", "premium", trend = -1), "trend = -1 is invalid")
expect_error_like(benktander(wk, "paid", "premium", n_iters = -1), "n_iters must be a non-negative whole number")
expect_error_like(benktander(wk, "paid", "premium", n_iters = 1.5), "n_iters must be a non-negative whole number")
expect_error_like(expected_loss(wk, "paid", "premium", average = "median"), "should be one of")
expect_error_like(totals_frame(1), "fit must be a chain_ladder_fit")

cat("actuarialrs R reserving tests passed\n")
