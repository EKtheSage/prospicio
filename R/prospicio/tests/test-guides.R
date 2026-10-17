# The R guides in docs/guides/*.qmd run, and every number they print is checked.
#
# Each guide's ```{r} chunks run in order in one environment. A chunk followed
# by a ```text block (only blank lines between) must print exactly that text,
# trailing spaces ignored, so a number in a guide cannot go stale when the
# bindings change. The R twin of python/tests/test_guides.py. CI runs the R
# tests from the repository root; run from elsewhere (an installed package's
# check) without the repository above, it skips.
suppressMessages(library(prospicio))

find_guides <- function(start = getwd()) {
  dir <- normalizePath(start, winslash = "/", mustWork = FALSE)
  repeat {
    candidate <- file.path(dir, "docs", "guides")
    if (dir.exists(candidate)) return(candidate)
    parent <- dirname(dir)
    if (identical(parent, dir)) return(NULL)
    dir <- parent
  }
}

# The fenced blocks of a text: info string, code, first and last line.
fenced_blocks <- function(lines) {
  blocks <- list()
  i <- 1
  while (i <= length(lines)) {
    m <- regmatches(lines[i], regexec("^```(.*)$", lines[i]))[[1]]
    if (length(m) == 0) {
      i <- i + 1
      next
    }
    start <- i
    i <- i + 1
    body <- character()
    while (i <= length(lines) && !startsWith(lines[i], "```")) {
      body <- c(body, lines[i])
      i <- i + 1
    }
    blocks[[length(blocks) + 1]] <- list(info = trimws(m[2]), code = body, first = start, last = i)
    i <- i + 1
  }
  blocks
}

normalized <- function(x) {
  x <- sub("[[:space:]]+$", "", x)
  while (length(x) > 0 && x[length(x)] == "") x <- x[-length(x)]
  x
}

is_r_chunk <- function(info) grepl("^\\{r[ ,}]", info)

run_guide <- function(path) {
  lines <- readLines(path, encoding = "UTF-8", warn = FALSE)
  stopifnot(!any(grepl("\u2014", lines))) # no em dash
  blocks <- fenced_blocks(lines)
  env <- new.env(parent = globalenv())
  ran <- 0
  checked <- 0
  for (k in seq_along(blocks)) {
    b <- blocks[[k]]
    if (!is_r_chunk(b$info)) next
    ran <- ran + 1
    where <- sprintf("%s, R chunk %d (line %d)", basename(path), ran, b$first)
    out <- tryCatch(
      utils::capture.output(source(exprs = parse(text = b$code), local = env,
                                   echo = FALSE, print.eval = TRUE)),
      error = function(e) stop(where, " raised: ", conditionMessage(e), call. = FALSE)
    )
    if (k < length(blocks) && blocks[[k + 1]]$info == "text") {
      nxt <- blocks[[k + 1]]
      between <- if (nxt$first - 1 >= b$last + 1) lines[(b$last + 1):(nxt$first - 1)] else character()
      if (all(trimws(between) == "")) {
        checked <- checked + 1
        got <- normalized(out)
        want <- normalized(nxt$code)
        if (!identical(got, want)) {
          stop(where, " prints something other than the text under it\n--- guide\n",
               paste(want, collapse = "\n"), "\n--- printed\n", paste(got, collapse = "\n"),
               call. = FALSE)
        }
      }
    }
  }
  if (ran == 0) stop(basename(path), " has no R chunk", call. = FALSE)
  if (checked == 0) stop(basename(path), " has no R chunk followed by the text it prints", call. = FALSE)
  c(ran = ran, checked = checked)
}

# The harness itself compares printed output.
local({
  tmp <- tempfile(fileext = ".qmd")
  writeLines(c("```{r}", "x <- 2", "cat(x * 3, '\\n')", "```", "", "```text", "6", "```",
               "", "```{r}", "cat(x, '\\n')", "```", "", "```text", "3", "```"), tmp)
  err <- tryCatch({ run_guide(tmp); NULL }, error = function(e) conditionMessage(e))
  unlink(tmp)
  stopifnot(!is.null(err), grepl("R chunk 2 (line 10)", err, fixed = TRUE))
})

guides <- find_guides()
if (is.null(guides)) {
  message("docs/guides not found above ", getwd(), "; skipping the R guides")
} else {
  files <- list.files(guides, pattern = "\\.qmd$", full.names = TRUE)
  stopifnot(length(files) > 0)
  for (f in files) {
    n <- run_guide(f)
    message(sprintf("%s: %d R chunks ran, %d checked", basename(f), n[["ran"]], n[["checked"]]))
  }
}
