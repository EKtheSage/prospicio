#' @include reserving_inflation.R distributions.R
NULL

#' Index paths, one per simulation
#'
#' One path of index levels per simulation for the periods after a
#' [calendar_index()]'s last given period, on the index's scale (levels, not
#' rates), as [reinflate_draws()] takes them (issue #979, design note
#' `docs/design/reserving-inflation.md`, decision 11). Python's
#' `IndexPaths`. The package does not generate paths: they are an input,
#' from an economic scenario generator or a fitted time series.
#'
#' @param first The first period of every path: `"2025"` (or the number
#'   2025), `"2025H1"`, `"2025Q1"` or `"2025-01"`. Its grain is the paths'
#'   grain.
#' @param values A numeric matrix with one row per simulation and one column
#'   per period from `first` on: finite and positive.
#' @returns An `index_paths` with the properties `first`, `values`, and the
#'   read-only `n_sims` (the rows), `n_periods` (the columns) and `periods`
#'   (the period labels, one per column).
#' @seealso [reinflate_draws()], [calendar_index()]
#' @export
#' @examples
#' # Two simulations of 2023 and 2024.
#' paths <- index_paths("2023", rbind(c(1.10, 1.20), c(1.05, 1.08)))
#' paths@n_sims
#' paths@periods
index_paths <- S7::new_class(
  "index_paths",
  package = "prospicio",
  properties = list(
    first = S7::class_character,
    values = S7::class_double,
    n_sims = S7::new_property(S7::class_integer, getter = function(self) nrow(self@values)),
    n_periods = S7::new_property(S7::class_integer, getter = function(self) ncol(self@values)),
    periods = S7::new_property(S7::class_character, getter = function(self) {
      reserving_result(index_paths_periods_rust(self@first, nrow(self@values),
                                                as.double(t(self@values))), call = NULL)
    })
  ),
  validator = function(self) {
    if (!is.matrix(self@values)) "values must be a numeric matrix with one row per simulation"
  },
  constructor = function(first, values) {
    if (missing(first) || missing(values)) {
      stop("index_paths() needs first and values", call. = FALSE)
    }
    first <- period_label(first)
    if (!is.matrix(values) || !is.numeric(values)) {
      stop("values must be a numeric matrix with one row per simulation", call. = FALSE)
    }
    storage.mode(values) <- "double"
    dimnames(values) <- NULL
    reserving_result(index_paths_periods_rust(first, nrow(values), as.double(t(values))))
    S7::new_object(S7::S7_object(), first = first, values = values)
  }
)

S7::method(print, index_paths) <- function(x, ...) {
  cat(sprintf("index_paths(first = \"%s\"): %d simulations of %d periods\n", x@first,
              x@n_sims, x@n_periods))
  invisible(x)
}

#' Re-inflate simulated future cells
#'
#' Simulated future increments at the base level of an [indexation()],
#' taken to the money of their payment periods simulation by simulation:
#' [reinflate()] applied to every simulation of a
#' [predictive_distribution] of future cells (design note
#' `docs/design/reserving-inflation.md`, decision 11). Python's
#' `Indexation.reinflate_draws`.
#'
#' `draws` holds one component per future cell. Its dimensions must include
#' `origin` (each key an origin period: a label such as `"2021"` or
#' `"2021Q3"`, or a whole year) and exactly one of `development` or `age`
#' (each key the age in months at the end of the cell's development
#' period). The cell is paid in the month `age - 1` months after the
#' origin's first month. Other dimensions (a line of business) pass through
#' untouched.
#'
#' * With `paths = NULL`, each cell of every simulation is multiplied by the
#'   [indexation_factor()] of its payment month: the index's future rate
#'   where it has one, an error where a payment period needs it and the
#'   index has none.
#' * With `paths`, simulation `i` of `draws` takes row `i` of the paths. The
#'   paths replace the future rate: a payment period inside the given index
#'   takes its given value, one from the paths' first period on takes that
#'   simulation's level, and one past the paths' last period is an error.
#'   The paths must start in the period right after the index's last given
#'   one, at the index's grain, and have one row per simulation of `draws`.
#'   The base must be one of the given periods, so that the base level is
#'   the same in every simulation.
#'
#' Which draws can be passed: [predict_distribution()] of an [odp_glm_fit]
#' gives future increments with the dimensions `origin` and `development`.
#' The bootstraps' reserve draws are by origin only, without cells, and are
#' refused for their missing age. A [compartmental_fit]'s `paid` draws
#' (dimensions `origin` and `age`) are cumulative paid, so they pass the
#' checks but would be re-inflated wrongly: difference them by origin first
#' (the first future cell less the origin's latest paid).
#'
#' @param x An [indexation()].
#' @param draws A [predictive_distribution] of future increments at the base
#'   level, one component per cell.
#' @param paths `NULL` (the default) for the index's future rate, or an
#'   [index_paths()] with one row per simulation of `draws`.
#' @returns A [predictive_distribution] with the same dimensions, keys (in
#'   the same order) and simulations, the cells re-inflated, and the
#'   provenance of `draws` with a `reinflated` parameter.
#'   `aggregate(out, "origin")` gives the re-inflated reserves by origin.
#' @seealso [index_paths()], [indexation()], [reinflate()]
#' @export
#' @examples
#' # Origin 2022's cells at ages 24 and 36, paid in 2023 and 2024, at the
#' # 2022 level, in two simulations.
#' draws <- predictive_distribution(
#'   rbind(c(100, 50), c(120, 40)),
#'   data.frame(origin = "2022", development = c(24, 36))
#' )
#' at_2022 <- indexation(calendar_index("2021", c(1, 1.1)), "2022-01")
#' paths <- index_paths("2023", rbind(c(1.21, 1.331), c(1.155, 1.21)))
#' out <- reinflate_draws(at_2022, draws, paths)
#' draw_matrix(out)
#' mean(aggregate(out, "origin"))
reinflate_draws <- function(x, draws, paths = NULL) {
  x <- check_indexation(x)
  if (!S7::S7_inherits(draws, predictive_distribution)) {
    stop("draws must be a predictive_distribution", call. = FALSE)
  }
  if (is.null(paths)) {
    first <- NULL
    n_sims <- 0L
    values <- double()
  } else {
    if (!S7::S7_inherits(paths, index_paths)) {
      stop("paths must be NULL or index_paths()", call. = FALSE)
    }
    first <- paths@first
    n_sims <- nrow(paths@values)
    values <- as.double(t(paths@values))
  }
  ptr <- reserving_result(reinflate_draws_rust(
    draws@ptr, x@index@first, x@index@values, x@index@future_rate, x@base, first, n_sims, values
  ))
  predictive_distribution(ptr = ptr)
}
