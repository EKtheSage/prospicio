suppressMessages(library(prospicio))

# Triangles read from files (docs/design/triangle.md, decision 6): CSV and
# Excel through the Rust readers, and Schedule P from a file in the CAS loss
# reserve database layout. The Rust core is checked in
# crates/prospicio-reserving/src/io.rs and validation/tests/reserving_io.rs;
# python/tests/test_reserving_io.py checks the same files.

validation <- function(...) file.path("validation", ...)

expect_error_like <- function(expr, pattern) {
  msg <- tryCatch({
    expr
    NULL
  }, error = function(e) conditionMessage(e))
  if (is.null(msg)) stop("expected an error matching ", pattern, call. = FALSE)
  if (!grepl(pattern, msg, fixed = TRUE)) {
    stop(sprintf("error %s does not contain %s", shQuote(msg), shQuote(pattern)), call. = FALSE)
  }
}

same_cells <- function(a, b) {
  stopifnot(identical(a@origins, b@origins), identical(a@development, b@development),
            identical(a@valuation, b@valuation))
  va <- unname(a@values)
  vb <- unname(b@values)
  stopifnot(identical(is.na(va), is.na(vb)), all(va == vb, na.rm = TRUE))
}

raa_long <- utils::read.csv(validation("data", "raa.csv"), comment.char = "#")
raa <- triangle(raa_long, "origin", "development", "value")

# CSV, long, with the comment line of the repository's data files.
tri <- read_triangle(validation("data", "raa.csv"), "origin", "development", comment = "#")
stopifnot(identical(tri@columns, "value"))
same_cells(tri, raa)

# CSV, wide, written from the triangle.
m <- as.matrix(raa)
wide <- data.frame(origin = rownames(m), m, check.names = FALSE)
path <- tempfile(fileext = ".csv")
utils::write.csv(wide, path, row.names = FALSE, na = "")
tri <- read_triangle(path, "origin", columns = "paid")
stopifnot(identical(tri@columns, "paid"))
same_cells(tri, raa)
# The ages given in place of the header.
names(wide) <- c("origin", paste0("age", seq_len(ncol(m))))
utils::write.csv(wide, path, row.names = FALSE, na = "")
same_cells(read_triangle(path, "origin", ages = 12 * seq_len(ncol(m))), raa)
expect_error_like(read_triangle(path, "origin"), 'column "age1" of a wide layout')

# Excel: the first sheet (wide) and the long sheet from B3, dates as origins.
xlsx <- validation("data", "raa.xlsx")
same_cells(read_triangle(xlsx, "origin", columns = "paid"), raa)
same_cells(read_triangle(xlsx, "origin", "development", "paid", sheet = "long"), raa)
expect_error_like(read_triangle(xlsx, "origin", "development", "paid", sheet = "bad"),
                  'cell C3 (column "paid") of sheet "bad": "n/a" is not a number')
expect_error_like(read_triangle(xlsx, "origin", sheet = "none"), 'has no sheet "none"')

# Errors name the line and column.
writeLines(c("year,age,paid", "2020,12,100", "2020,24,1O0"), path)
expect_error_like(read_triangle(path, "year", "age"), 'line 3, column "paid": "1O0" is not a number')
expect_error_like(read_triangle(path, "year", "dev"), 'no column named "dev"')
expect_error_like(read_triangle(sub("csv$", "dat", path), "year"), "cannot tell the format")
expect_error_like(read_triangle(path, "year", "age", ages = c(12, 24)), "wide layout")

# Schedule P from a file in the CAS layout: group 353 equals the committed
# square of validation/data/clrd_comauto_353.csv (paid, case incurred, net
# premium), as calibration_panel.py checks it.
sample <- validation("data", "schedule_p_comauto_sample.csv")
sp <- schedule_p_triangle(sample, company = 353, columns = c("paid", "case_incurred", "premium"),
                          lower = TRUE)
stopifnot(identical(sp@keys, "company"), identical(sp@index$company, "353"))
comauto <- utils::read.csv(validation("data", "clrd_comauto_353.csv"), comment.char = "#")
expected <- triangle(comauto, "origin", "development", c("paid", "incurred", "premium"))
same_cells(sp, expected)

all_companies <- schedule_p_triangle(sample)
stopifnot(identical(all_companies@index$company, c("353", "8672")),
          identical(all_companies@columns,
                    c("paid", "incurred", "case_incurred", "premium", "premium_direct")),
          sum(!is.na(all_companies@values[1, "paid", , ])) == 55)
expect_error_like(schedule_p_triangle(sample, company = 1), "no rows for company 1")
expect_error_like(schedule_p_triangle(sample, columns = "reported"), 'unknown Schedule P column "reported"')

cat("test-reserving-io.R: all checks passed\n")
