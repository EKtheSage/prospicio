library(prospicio)
source(file.path(local({ f <- grep("^--file=", commandArgs(FALSE), value = TRUE); if (length(f)) dirname(sub("^--file=", "", f[1])) else file.path("R", "prospicio", "tests") }), "helper.R"))

# A Rust error is raised in R, whether or not the R code calling a native
# function passes its result through rust_result() (#957).

# ale(): the grid's errors are the core's, attributed to ale().
lin_model <- structure(list(), class = "r_errors_lin")
predict.r_errors_lin <- function(object, newdata, ...) 2 * newdata$x
frame <- data.frame(x = c(0.5, 1, 2, 3, 5, 8))
expect_error_like(ale(lin_model, frame, "x", bins = 0), "bins")
e <- tryCatch(ale(lin_model, frame, "x", bins = 1.5), error = function(e) e)
stopifnot(inherits(e, "error"), !inherits(e, "extendr_error"),
          grepl("bins", conditionMessage(e)), identical(conditionCall(e)[[1]], quote(ale)))
stopifnot(nrow(ale(lin_model, frame, "x", bins = 2)) == 3)

# A fallible native function called directly raises an R error rather than
# returning an `extendr_error` object.
e <- tryCatch(prospicio:::ale_grid_rust(c(1, 2, 3), 0), error = function(e) e)
stopifnot(inherits(e, "error"), !inherits(e, "extendr_error"), grepl("bins", conditionMessage(e)))
e <- tryCatch(prospicio:::shap_exact_rust(c(1, 2, 3), 1, 2), error = function(e) e)
stopifnot(inherits(e, "error"), !inherits(e, "extendr_error"))

# A caller's own rust_result() call keeps its attribution: the wrapper
# leaves the error to it.
f <- function() prospicio:::rust_result(prospicio:::ale_grid_rust(c(1, 2, 3), 0), quote(mine()))
e <- tryCatch(f(), error = function(e) e)
stopifnot(identical(conditionCall(e), quote(mine())))
# Without one, the error is attributed to the wrapper's caller.
g <- function() { z <- prospicio:::ale_grid_rust(c(1, 2, 3), 0); z }
e <- tryCatch(g(), error = function(e) e)
stopifnot(identical(conditionCall(e), quote(g())))

# A setting a method does not support keeps its condition class through
# rust_result() as through reserving_result().
ns <- structure(list(value = "[prospicio_not_supported] no tail here"), class = "extendr_error")
e <- tryCatch(prospicio:::rust_result(ns), error = function(e) e)
stopifnot(inherits(e, "prospicio_not_supported"), conditionMessage(e) == "no tail here")
