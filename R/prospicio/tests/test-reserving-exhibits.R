# Friedland's exhibits and the weighted selection of ultimates
# (docs/design/reserving-v02.md, decision 30; issue #557): Friedland (2010),
# chapter 7, Exhibit I (U.S. Industry Auto) and chapter 15, Exhibit I (XYZ
# Insurer), to the printed digits, and chainladder-python's
# VotingChainladder on GenIns from validation/reference.
suppressMessages(library(prospicio))

near <- function(a, b, tol) {
  if (!all(abs(a - b) <= tol)) {
    stop(sprintf("got %s, want %s (tol %g)", paste(format(a, digits = 15), collapse = " "),
                 paste(format(b, digits = 15), collapse = " "), tol), call. = FALSE)
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

# Tests run from the repository root (CI and `cargo xtask r`).
read_data <- function(name) {
  utils::read.csv(file.path("validation", "data", paste0(name, ".csv")), comment.char = "#")
}
us <- triangle(read_data("friedland_us_auto"), "origin", "development",
               c("reported", "paid", "premium"))
ages <- seq(12, 120, by = 12)
cdf <- list(
  reported = c(1.292, 1.110, 1.051, 1.023, 1.011, 1.006, 1.003, 1.001, 1.000, 1.000),
  paid = c(2.390, 1.404, 1.184, 1.085, 1.040, 1.020, 1.011, 1.006, 1.004, 1.002)
)
by_age <- function(values) stats::setNames(values, ages[seq_along(values)])

# Chapter 7, Exhibit I, sheet 1, parts 3 and 4.
selected <- c(1.164, 1.056, 1.027, 1.012, 1.005, 1.003, 1.002, 1.001, 1.000)
ex <- development_exhibit(us, "reported", ldf = by_age(selected), ldf_kind = "ldf")
stopifnot(
  identical(rownames(ex@averages),
            c("simple latest 5", "simple latest 3", "medial latest 5x1", "volume latest 5",
              "volume latest 3", "geometric latest 4")),
  identical(colnames(ex@link_ratios)[1], "12-24"),
  identical(names(ex@cdf)[10], "120-Ult"),
  is.na(ex@link_ratios["2007", "12-24"]),
  identical(dim(ex@values), c(10L, 10L))
)
near(round(ex@averages["medial latest 5x1", ], 3),
     c(1.165, 1.057, 1.027, 1.010, 1.004, 1.003, 1.002, 1.001, 1.000), 1e-12)
near(round(ex@averages["geometric latest 4", ], 3),
     c(1.164, 1.057, 1.027, 1.011, 1.004, 1.003, 1.002, 1.001, 1.000), 1e-12)
near(unname(ex@selected), selected, 1e-12)
near(round(unname(ex@cdf), 3), cdf$reported, 1e-12)
near(round(ex@link_ratios["1998", "12-24"], 3), 1.166, 1e-12)

# The percents developed use the book's rounded factors to ultimate.
ex <- development_exhibit(us, "paid", ldf = by_age(cdf$paid), ldf_kind = "cdf", averages = list())
near(round(100 * unname(ex@percent_developed), 1),
     c(41.8, 71.2, 84.5, 92.2, 96.2, 98.0, 98.9, 99.4, 99.6, 99.8), 1e-9)
near(ex@tail, 1.002, 1e-12)
stopifnot(nrow(ex@averages) == 0L)
frame <- as.data.frame(development_exhibit(us, "paid"))
stopifnot(identical(utils::tail(rownames(frame), 3), c("selected", "cdf", "percent_developed")),
          is.na(frame["simple latest 5", "120-ult"]))
own <- development_exhibit(us, "paid", averages = list(
  all = list(average = "simple"), latest = list(average = "volume", n_periods = 3)))
near(unname(own@averages["latest", ]),
     unname(development_exhibit(us, "paid")@averages["volume latest 3", ]), 1e-12)
expect_error_like(development_exhibit(us, "paid", averages = list(list(average = "simple"))),
                  "named list")

# Chapter 7, Exhibit I, sheet 4: every method's totals.
apriori <- stats::setNames(rep(c(0.75, 0.65), each = 5), 1998:2007)
methods <- list(
  "development reported" = chain_ladder(us, "reported", ldf = by_age(cdf$reported),
                                        ldf_kind = "cdf"),
  "development paid" = chain_ladder(us, "paid", ldf = by_age(cdf$paid), ldf_kind = "cdf"),
  "expected claims" = expected_loss(us, "reported", "premium", apriori = apriori,
                                    ldf = by_age(cdf$reported), ldf_kind = "cdf"),
  "bf paid" = bornhuetter_ferguson(us, "paid", "premium", apriori = apriori,
                                   ldf = by_age(cdf$paid), ldf_kind = "cdf"),
  "cape cod" = cape_cod(us, "reported", "premium", ldf = by_age(cdf$reported), ldf_kind = "cdf")
)
ux <- ultimate_exhibit(us, reported = "reported", paid = "paid", methods = methods)
stopifnot(identical(ux@ages, as.integer(rev(ages))), identical(ux@methods, names(methods)))
near(sum(ux@reported - ux@paid), 45431219, 0.5)
book <- list(
  "development reported" = c(569172456, 25690869, 71122088),
  "development paid" = c(572041099, 28559512, 73990731),
  "expected claims" = c(569281839, 25800252, 71231471),
  "bf paid" = c(570568198, 27086611, 72517830),
  "cape cod" = c(570800677, 27319090, 72750309)
)
for (m in names(book)) {
  near(c(sum(ux@ultimate[[m]]), sum(ux@ibnr[[m]]), sum(ux@unpaid[[m]])), book[[m]], 0.5)
}
near(round(ux@ibnr[["development paid"]][1:3]), c(-2829, 18769, 22495), 1e-9)

# Chapter 15, Exhibit I, sheets 2 and 6: XYZ Insurer's selection.
xyz_methods <- list(
  "development both" = c(15822, 25107, 37246, 38798, 48169, 44373, 70780, 71362, 53392, 42680,
                         41531),
  "bf reported" = c(15822, 25107, 37246, 38798, 48169, 44373, 70792, 71554, 53906, 40300, 36842),
  "bf paid" = c(15975, 25128, 37294, 39274, 47313, 45070, 71688, 77898, 56031, 34988, 33988),
  "frequency-severity 2" = c(rep(NA, 9), 30512, 30140)
)
xyz <- ultimate_exhibit(
  origin = 1998:2008, age = 132 - 12 * (0:10),
  reported = c(15822, 25107, 37246, 38798, 48169, 44373, 70288, 70655, 48804, 31732, 18632),
  paid = c(15822, 24817, 36782, 38519, 44437, 39320, 52811, 40026, 22819, 11865, 3409),
  methods = xyz_methods
)
book_selected <- c(15822, 25107, 37246, 38798, 48169, 44373, 70780, 74726, 54968, 40300, 33491)
by_origin <- list(
  "development both" = c(rep(1, 7), rep(0, 4)),
  "bf reported" = c(rep(0, 7), 0.5, 0.5, 1, 0.5),
  "bf paid" = c(rep(0, 7), 0.5, 0.5, 0, 0),
  "frequency-severity 2" = c(rep(0, 10), 0.5)
)
age_rows <- data.frame(age = c(12, 24, 36, 60),
                       "development both" = c(0, 0, 0, 1), "bf reported" = c(0.5, 1, 0.5, 0),
                       "bf paid" = c(0, 0, 0.5, 0), "frequency-severity 2" = c(0.5, 0, 0, 0),
                       check.names = FALSE)
for (table in list(select_ultimates(xyz, by_origin), select_ultimates(xyz, by_age = age_rows))) {
  stopifnot(identical(table$origin[12], "total"), table[["weight_frequency-severity 2"]][11] == 0.5)
  near(table$ultimate[1:11], book_selected, 0.5)
  near(table$ultimate[12], 483781, 1)
  near(table$ibnr[12], 34155, 1)
}
expect_error_like(select_ultimates(xyz, list("development both" = 0.5, "bf reported" = 0.25)),
                  "sum to 0.75")
expect_error_like(select_ultimates(xyz, "frequency-severity 2"), "no ultimate there")
expect_error_like(select_ultimates(xyz, "chain ladder"), "no method")
expect_error_like(select_ultimates(xyz, "bf paid", by_age = age_rows), "not both")
near(select_ultimates(xyz, "bf paid", total = FALSE)$ultimate, xyz_methods[["bf paid"]], 0)

# chainladder-python's VotingChainladder on GenIns.
genins <- triangle(read_data("genins_premium"), "origin", "development", c("paid", "premium"))
fits <- list(cl = chain_ladder(genins, "paid"),
             bf = bornhuetter_ferguson(genins, "paid", "premium", apriori = 0.75),
             cc = cape_cod(genins, "paid", "premium"))
gx <- ultimate_exhibit(genins, paid = "paid", methods = fits)
phased <- rbind(matrix(c(1, 0, 0), 4, 3, byrow = TRUE), matrix(c(0.5, 0.5, 0), 3, 3, byrow = TRUE),
                matrix(c(0, 0.5, 0.5), 2, 3, byrow = TRUE), c(0, 0, 1))
cases <- list(
  phased = list(cl = phased[, 1], bf = phased[, 2], cc = phased[, 3]),
  blend = list(cl = 0.5, bf = 0.25, cc = 0.25),
  single_bf = "bf"
)
tables <- lapply(cases, function(w) select_ultimates(gx, w))
ref <- utils::read.csv(file.path("validation", "reference", "reserving_exhibits_python.csv"),
                       comment.char = "#", colClasses = c(arg = "character"))
for (k in seq_len(nrow(ref))) {
  r <- ref[k, ]
  t <- tables[[r$method]]
  i <- match(r$arg, gx@origins)
  got <- switch(
    sub(":.*", "", r$quantity),
    ultimate = if (grepl(":", r$quantity)) gx@ultimate[[sub(".*:", "", r$quantity)]][i] else
      t$ultimate[i],
    reserve = t$unpaid[i],
    total_ultimate = t$ultimate[nrow(t)],
    total_reserve = t$unpaid[nrow(t)]
  )
  near(got, r$expected, 1e-9 * abs(r$expected))
}
stopifnot(nrow(ref) == 156L)
near(select_ultimates(fits, cases$blend, triangle = genins, paid = "paid", total = FALSE)$ultimate,
     tables$blend$ultimate[1:10], 0)

# A fit of other origins is refused.
expect_error_like(ultimate_exhibit(us, paid = "paid", methods = list(cl = fits$cl)), "origins")
expect_error_like(select_ultimates(list(cl = fits$cl), "cl"), "triangle")
