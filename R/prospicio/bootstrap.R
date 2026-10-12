# Copies the Rust sources the package builds from into src/rust/, so that a
# source tarball built from this directory compiles on its own.
#
# pkgbuild runs this script before `R CMD build` (`Config/build/bootstrap:
# TRUE` in DESCRIPTION), with the repository checkout around it. That is how
# remotes::install_github(subdir = "R/prospicio") and r-universe build the
# package: they build a tarball of this directory alone, which cannot reach
# the workspace's crates/ the way an `R CMD INSTALL R/prospicio` in a checkout
# does (src/Makevars builds from src/rust/ when it exists, from the workspace
# otherwise).
#
# src/rust/ holds a workspace of its own: the root Cargo.toml with its
# members cut down to prospicio-r, the root Cargo.lock, and crates/ with
# prospicio-r and the workspace crates it depends on. Third-party crates still
# come from crates.io at build time; vendoring them for offline builds is
# CRAN's step, not this one.

root <- normalizePath(file.path("..", ".."), mustWork = FALSE)
manifest <- file.path(root, "Cargo.toml")
if (!file.exists(file.path(root, "crates", "prospicio-r", "Cargo.toml"))) {
  stop("bootstrap.R must run in R/prospicio of a prospicio checkout")
}

out <- file.path("src", "rust")
unlink(out, recursive = TRUE)
dir.create(file.path(out, "crates"), recursive = TRUE)

# The workspace crates prospicio-r needs: its [dependencies] that are
# `prospicio-*.workspace = true`, and theirs, transitively.
internal_deps <- function(crate) {
  lines <- readLines(file.path(root, "crates", crate, "Cargo.toml"))
  section <- ""
  deps <- character()
  for (line in lines) {
    if (grepl("^\\[", line)) section <- trimws(line)
    if (section == "[dependencies]") {
      m <- regmatches(line, regexpr("^prospicio[a-z-]*(?=\\.workspace)", line, perl = TRUE))
      deps <- c(deps, m)
    }
  }
  deps
}
needed <- "prospicio-r"
queue <- needed
while (length(queue) > 0) {
  new <- setdiff(internal_deps(queue[1]), needed)
  needed <- c(needed, new)
  queue <- c(queue[-1], new)
}

for (crate in needed) {
  src <- file.path(root, "crates", crate)
  files <- list.files(src, recursive = TRUE, all.files = TRUE)
  files <- files[!grepl("^(target|tests|benches)/", files)]
  for (f in files) {
    dest <- file.path(out, "crates", crate, f)
    dir.create(dirname(dest), recursive = TRUE, showWarnings = FALSE)
    file.copy(file.path(src, f), dest)
  }
}

# The root manifest keeps [workspace.package] and [workspace.dependencies],
# which the crates inherit; only its members change.
toml <- readLines(manifest)
start <- grep("^\\[workspace\\]$", toml)
end <- grep("^resolver", toml)
stopifnot(length(start) == 1, length(end) == 1, end > start)
toml <- c(
  toml[seq_len(start)],
  'members = ["crates/prospicio-r"]',
  toml[end:length(toml)]
)
writeLines(toml, file.path(out, "Cargo.toml"))
file.copy(file.path(root, "Cargo.lock"), file.path(out, "Cargo.lock"))

message("bootstrap.R: copied ", paste(sort(needed), collapse = ", "), " into src/rust")
