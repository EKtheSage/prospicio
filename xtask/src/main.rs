//! Build tasks that span languages. Run `cargo xtask help`.
//!
//! Building a binding also regenerates its documentation, so the docs can
//! never lag the code they describe (docs/architecture.md, "Documentation"):
//!
//! - `python`: build the extension and its type stubs, run the tests, render
//!   the great-docs site, whose user guide holds the guides in docs/guides and
//!   the notebooks in examples/, executed (the notebooks are committed without
//!   outputs, so the site is where their results are published).
//! - `r`: regenerate `man/` and `NAMESPACE` with roxygen2, install the
//!   package, run the tests, render the pkgdown site, whose articles are the
//!   R twins (`.qmd`) of the guides and of the notebooks.
//! - `rust`: rustdoc for the pure-Rust crates, warnings denied.
//! - `docs`: all three, collected into one site under `target/docs-site`.
//! - `generated`: only what writes the committed generated files (the
//!   Python stub; R's `man/`, `NAMESPACE` and `R/extendr-wrappers.R`), the
//!   R package's install with its `undoc`/`codoc` check and pkgdown's index
//!   check, without rendering either site (#823). With `--check`, what a
//!   pull request runs.
//! - `xll`: build the 64-bit Excel add-in as `target/xll/prospicio.xll`
//!   and, when a MinGW compiler (and off Windows, Wine) is found, run the
//!   fake-Excel smoke test on it (docs/design/xll.md).
//!
//! `--check` fails if a build changed a generated file that is committed
//! (the Python stub, R's `man/`, `NAMESPACE` and `R/extendr-wrappers.R`):
//! someone changed a doc comment or a binding without rebuilding. `--no-tests` skips the bindings' test suites
//! (CI's `python` and `r` jobs run them; the docs build need not run them
//! again).

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const USAGE: &str = "\
usage: cargo xtask <task> [--check] [--no-tests]

tasks:
  python   build the Python extension and stubs, test, render its docs
           (with the guides and the executed notebooks)
  r        install the R package, regenerate man/ and NAMESPACE, test, render its docs
  rust     build rustdoc for the pure-Rust crates
  docs     all of the above, collected into target/docs-site
  generated  only regenerate the committed generated files (the stub, man/,
           NAMESPACE, extendr-wrappers.R) and check the R package and
           pkgdown's index, without rendering the sites
  xll      build the 64-bit Excel add-in as target/xll/prospicio.xll and
           smoke-test it in a fake Excel when MinGW (and Wine) are found

  --check     fail if the build changed a committed generated file
  --no-tests  skip the Python and R test suites
";

/// Generated files that are committed, per binding: the build must leave
/// them unchanged under `--check`.
const PYTHON_GENERATED: &[&str] = &["python/prospicio/prospicio_native.pyi"];
const R_GENERATED: &[&str] = &[
    "R/prospicio/man",
    "R/prospicio/NAMESPACE",
    "R/prospicio/R/extendr-wrappers.R",
];

type Result<T = ()> = std::result::Result<T, String>;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let opts = Options {
        check: args.iter().any(|a| a == "--check"),
        tests: !args.iter().any(|a| a == "--no-tests"),
    };
    let tasks: Vec<&str> = args
        .iter()
        .map(String::as_str)
        .filter(|a| *a != "--check" && *a != "--no-tests")
        .collect();
    let result = match tasks.as_slice() {
        ["python"] => python(opts),
        ["r"] => r(opts),
        ["rust"] => rust(),
        ["docs"] => docs(opts),
        ["generated"] => generated(opts),
        ["xll"] => xll(),
        [] | ["help"] | ["-h"] | ["--help"] => {
            print!("{USAGE}");
            return ExitCode::SUCCESS;
        }
        _ => Err(format!("unknown arguments {args:?}\n\n{USAGE}")),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("xtask: {e}");
            ExitCode::FAILURE
        }
    }
}

/// The flags shared by the binding tasks.
#[derive(Clone, Copy)]
struct Options {
    /// `--check`: fail if a committed generated file changed.
    check: bool,
    /// Run the test suites (off with `--no-tests`).
    tests: bool,
}

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives one level below the workspace root")
        .to_path_buf()
}

fn python(opts: Options) -> Result {
    let dir = root().join("python");
    // great-docs needs Python 3.11+, although the package supports 3.9+.
    // uv keeps an existing environment that satisfies this.
    let uv = || {
        let mut cmd = Command::new("uv");
        cmd.env("UV_PYTHON", ">=3.11").current_dir(&dir);
        cmd
    };
    step("python: sync the environment");
    // `examples`: Jupyter, to execute the notebooks for the user guide.
    run(uv().args(["sync", "--group", "docs", "--group", "examples"]))?;
    step("python: build the extension and regenerate its type stubs");
    run(uv().args([
        "run",
        "maturin",
        "develop",
        "--uv",
        "--release",
        "--generate-stubs",
    ]))?;
    if opts.tests {
        step("python: test");
        run(uv().args(["run", "pytest", "tests"]))?;
    }
    step("python: stage the user guide (the guides, and the notebooks executed)");
    stage_user_guide(&uv)?;
    step("python: render the docs site");
    // UTF-8 mode: great-docs writes _quarto.yml in the locale's encoding,
    // which on Windows cannot hold the reference titles' non-ASCII letters.
    run(uv()
        .args(["run", "great-docs", "build"])
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1"))?;
    if opts.check {
        unchanged(PYTHON_GENERATED)?;
    }
    Ok(())
}

/// Where the Python site's user guide is staged; `user_guide` in
/// python/great-docs.yml points here.
const USER_GUIDE: &str = "target/user-guide";

/// One page of the table of contents (docs/guides/toc.yml): a guide
/// (`docs/guides/<name>.md`) or a notebook (`examples/<name>.ipynb`).
#[derive(Debug, PartialEq)]
struct TocPage {
    name: String,
    notebook: bool,
}

/// One section of the table of contents: a title, a line for the R site's
/// article index and the pages in reading order.
#[derive(Debug, PartialEq)]
struct TocSection {
    title: String,
    description: String,
    pages: Vec<TocPage>,
}

/// Parses docs/guides/toc.yml, a small subset of YAML (see the file's header):
/// `- section:` starts a section, `description:` sets its line, and
/// `- guide:` or `- notebook:` adds a page.
fn parse_toc(text: &str) -> Result<Vec<TocSection>> {
    let value = |v: &str| v.trim().trim_matches(|c| c == '"' || c == '\'').to_owned();
    let mut sections: Vec<TocSection> = Vec::new();
    for (i, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') || line == "pages:" {
            continue;
        }
        let at = format!("toc.yml line {}", i + 1);
        if let Some(title) = line.strip_prefix("- section:") {
            sections.push(TocSection {
                title: value(title),
                description: String::new(),
                pages: Vec::new(),
            });
        } else if let Some(d) = line.strip_prefix("description:") {
            let section = sections
                .last_mut()
                .ok_or(format!("{at}: before any section"))?;
            section.description = value(d);
        } else if let Some((notebook, name)) = line
            .strip_prefix("- guide:")
            .map(|n| (false, n))
            .or_else(|| line.strip_prefix("- notebook:").map(|n| (true, n)))
        {
            let section = sections
                .last_mut()
                .ok_or(format!("{at}: before any section"))?;
            section.pages.push(TocPage {
                name: value(name),
                notebook,
            });
        } else {
            return Err(format!("{at}: not understood: {raw}"));
        }
    }
    Ok(sections)
}

/// Checks the table of contents against the files: every guide (but the
/// README) and every notebook is listed once, and every entry names a file.
fn check_toc(sections: &[TocSection], guides: &[String], notebooks: &[String]) -> Result {
    let mut problems = Vec::new();
    let mut seen: Vec<(&str, bool)> = Vec::new();
    for section in sections {
        if section.pages.is_empty() {
            problems.push(format!("section \"{}\" has no pages", section.title));
        }
        for page in &section.pages {
            let (kind, known) = if page.notebook {
                ("notebook examples/", notebooks)
            } else {
                ("guide docs/guides/", guides)
            };
            if !known.contains(&page.name) {
                problems.push(format!("{kind}{} is listed but does not exist", page.name));
            }
            if seen.contains(&(page.name.as_str(), page.notebook)) {
                problems.push(format!("{kind}{} is listed twice", page.name));
            }
            seen.push((page.name.as_str(), page.notebook));
        }
    }
    for (kind, names, notebook) in [
        ("guide docs/guides/", guides, false),
        ("notebook examples/", notebooks, true),
    ] {
        for name in names {
            if !seen.contains(&(name.as_str(), notebook)) {
                problems.push(format!("{kind}{name} is not listed"));
            }
        }
    }
    if problems.is_empty() {
        Ok(())
    } else {
        Err(format!(
            "docs/guides/toc.yml does not match the files:\n  {}",
            problems.join("\n  ")
        ))
    }
}

/// The guides in docs/guides (file stems, without the README) and the
/// notebooks in examples/.
fn guide_and_notebook_names(root: &Path) -> Result<(Vec<String>, Vec<String>)> {
    let guides = files(&root.join("docs/guides"), "md")?
        .iter()
        .map(|p| file_stem(p))
        .filter(|s| s != "README")
        .collect();
    let notebooks = files(&root.join("examples"), "ipynb")?
        .iter()
        .map(|p| file_stem(p))
        .collect();
    Ok((guides, notebooks))
}

/// Reads docs/guides/toc.yml and checks it against the files. Both sites'
/// sections and page order come from it.
fn load_toc(root: &Path) -> Result<Vec<TocSection>> {
    let sections = parse_toc(&read(&root.join("docs/guides/toc.yml"))?)?;
    let (guides, notebooks) = guide_and_notebook_names(root)?;
    check_toc(&sections, &guides, &notebooks)?;
    Ok(sections)
}

/// The staged file name of the `n`th page (from 1) of the Python site's user
/// guide. great-docs sorts the pages by file name and strips the number from
/// the page's URL, and lists the sidebar sections in the order of their first
/// page, so a zero-padded global number gives both orders.
fn staged_name(n: usize, name: &str) -> String {
    format!("{n:03}-{name}.md")
}

/// Stages the Python site's user guide in the order of docs/guides/toc.yml:
/// each page is a guide in docs/guides or a notebook in examples/, executed
/// and exported to Markdown with its outputs and charts. The notebooks are
/// committed without outputs (.gitattributes), so this is where their results
/// are published. Each page's `guide-section` front matter is its section of
/// the table of contents, and a numeric prefix orders the pages, which
/// great-docs strips from their URLs: `user-guide/<name>.html`.
fn stage_user_guide(uv: &dyn Fn() -> Command) -> Result {
    let root = root();
    let toc = load_toc(&root)?;
    let stage = root.join(USER_GUIDE);
    if stage.exists() {
        fs::remove_dir_all(&stage).map_err(|e| format!("clearing {}: {e}", stage.display()))?;
    }
    fs::create_dir_all(&stage).map_err(|e| format!("creating {}: {e}", stage.display()))?;
    let (_, notebooks) = guide_and_notebook_names(&root)?;
    let mut n = 0;
    for section in &toc {
        for entry in &section.pages {
            n += 1;
            let name = &entry.name;
            let target = stage.join(staged_name(n, name));
            if !entry.notebook {
                // A guide links to a notebook on GitHub; on the site, to its page.
                let mut text = read(&root.join(format!("docs/guides/{name}.md")))?;
                for s in &notebooks {
                    text =
                        text.replace(&format!("](../../examples/{s}.ipynb"), &format!("]({s}.md"));
                }
                if text.lines().any(|l| l.trim() == EXECUTE_MARK) {
                    let text = executed(&text, &root.join("docs/guides"));
                    write(&target.with_extension("qmd"), &page(&text, &section.title))?;
                } else {
                    write(&target, &page(&text, &section.title))?;
                }
                continue;
            }
            // nbconvert runs the kernel in the notebook's directory, as Jupyter does.
            // Its default limit of 30 s per cell could stop a heavy cell on a slow
            // runner; 600 s still stops a hung kernel before the job's timeout.
            run(uv()
                .args([
                    "run",
                    "jupyter",
                    "nbconvert",
                    "--to",
                    "markdown",
                    "--execute",
                    "--ExecutePreprocessor.timeout=600",
                ])
                .arg("--output-dir")
                .arg(&stage)
                .args(["--output", name])
                .arg(root.join(format!("examples/{name}.ipynb"))))?;
            let exported = stage.join(format!("{name}.md"));
            let text = read(&exported)?.replace("](../docs/guides/", "](");
            write(&target, &page(&text, &section.title))?;
            fs::remove_file(&exported)
                .map_err(|e| format!("removing {}: {e}", exported.display()))?;
        }
    }
    Ok(())
}

/// A user guide page: the first `# ` heading outside a code block becomes the
/// title in front matter, and `section` groups the page in the sidebar.
fn page(text: &str, section: &str) -> String {
    let mut title = None;
    let mut fenced = false;
    let mut body = String::with_capacity(text.len());
    for line in text.lines() {
        if line.starts_with("```") {
            fenced = !fenced;
        }
        if title.is_none()
            && !fenced
            && let Some(heading) = line.strip_prefix("# ")
        {
            title = Some(heading.trim().to_owned());
            continue;
        }
        body.push_str(&fence_info(line));
        body.push('\n');
    }
    let quote = |s: &str| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""));
    let title = title.map_or(String::new(), |t| format!("title: {}\n", quote(&t)));
    format!(
        "---\n{title}guide-section: {}\n---\n\n{}",
        quote(section),
        body.trim_start()
    )
}

/// A guide that holds this line on its own runs when the Python site is
/// built, so that its figures and tables are drawn: it is staged as a Quarto
/// document whose `python` blocks are cells (see `executed`).
const EXECUTE_MARK: &str = "<!-- site: execute -->";

/// A guide as a Quarto document that runs on the site: each `python` block
/// becomes a cell, a `python no-run` block a cell that is shown and not run,
/// and the `text` block after a cell, which `python/tests/test_guides.py`
/// checks against the block's output, is left out because the cell's own
/// output is shown. A hidden first cell runs the guide from docs/guides, as
/// the test does, and loads `docs/guides/site/figures.py`, which draws the
/// figures in the site's colours for both modes.
fn executed(text: &str, guides: &Path) -> String {
    let dir = guides.to_string_lossy().replace('\\', "/");
    let mut out = format!(
        "```{{python}}\n#| include: false\nimport os\nos.chdir(r\"{dir}\")\n\
         exec(open(\"site/figures.py\", encoding=\"utf-8\").read())\n```\n\n"
    );
    let mut lines = text.lines().peekable();
    let mut after_cell = false;
    while let Some(line) = lines.next() {
        if line.trim() == EXECUTE_MARK {
            continue;
        }
        let Some(info) = line.strip_prefix("```") else {
            if !line.trim().is_empty() {
                after_cell = false;
            }
            out.push_str(line);
            out.push('\n');
            continue;
        };
        let info = info.trim();
        if info == "text" && after_cell {
            // Skip the checked output and the blank line after it.
            for l in lines.by_ref() {
                if l.starts_with("```") {
                    break;
                }
            }
            if lines.peek().is_some_and(|l| l.trim().is_empty()) {
                lines.next();
            }
            after_cell = false;
            continue;
        }
        let head = match info {
            "python" => "```{python}".to_owned(),
            "python no-run" => "```{python}\n#| eval: false".to_owned(),
            _ => fence_info(line).into_owned(),
        };
        out.push_str(&head);
        out.push('\n');
        for l in lines.by_ref() {
            out.push_str(l);
            out.push('\n');
            if l.starts_with("```") {
                break;
            }
        }
        after_cell = info == "python";
    }
    out
}

/// An R twin that holds `EXECUTE_MARK` as the body of an article that runs:
/// a hidden first chunk runs the chunks from the repository's root, as
/// `R/prospicio/tests/test-guides.R` does, and loads
/// `docs/guides/site/figures.R`, which draws ggplot2 figures in the site's
/// colours for both modes; the `text` block after a chunk, which that test
/// checks against the chunk's output, is left out because the chunk's own
/// output is shown.
fn executed_r(text: &str, root: &Path) -> String {
    let dir = root.to_string_lossy().replace('\\', "/");
    // The YAML header stays first, where `article` looks for the title.
    let (header, text) = match text.strip_prefix("---").and_then(|r| r.split_once("\n---")) {
        Some((h, rest)) => {
            let rest = rest.split_once('\n').map_or("", |x| x.1);
            (format!("---{h}\n---\n"), rest)
        }
        None => (String::new(), text),
    };
    let mut out = format!(
        "{header}```{{r, include = FALSE}}\nknitr::opts_knit$set(root.dir = \"{dir}\")\n\
         source(\"{dir}/docs/guides/site/figures.R\")\n```\n\n"
    );
    let mut lines = text.lines().peekable();
    let mut after_chunk = false;
    while let Some(line) = lines.next() {
        if line.trim() == EXECUTE_MARK {
            continue;
        }
        let Some(info) = line.strip_prefix("```") else {
            if !line.trim().is_empty() {
                after_chunk = false;
            }
            out.push_str(line);
            out.push('\n');
            continue;
        };
        let info = info.trim();
        if info == "text" && after_chunk {
            for l in lines.by_ref() {
                if l.starts_with("```") {
                    break;
                }
            }
            if lines.peek().is_some_and(|l| l.trim().is_empty()) {
                lines.next();
            }
            after_chunk = false;
            continue;
        }
        out.push_str(line);
        out.push('\n');
        for l in lines.by_ref() {
            out.push_str(l);
            out.push('\n');
            if l.starts_with("```") {
                break;
            }
        }
        after_chunk = info.starts_with("{r") && !info.contains("eval = FALSE");
    }
    out
}

/// An opening fence whose info string is several bare words, as the guides'
/// `python no-run`, written as pandoc attributes (`{.python .no-run}`):
/// pandoc's Markdown reader takes a bare info string of one word only and
/// reads the block as a paragraph otherwise. The guides keep the bare form,
/// which `python/tests/test_guides.py` reads.
fn fence_info(line: &str) -> std::borrow::Cow<'_, str> {
    let Some(info) = line.strip_prefix("```") else {
        return line.into();
    };
    let words: Vec<&str> = info.split_whitespace().collect();
    if words.len() < 2 || info.starts_with('{') || info.starts_with('`') {
        return line.into();
    }
    let classes: Vec<String> = words.iter().map(|w| format!(".{w}")).collect();
    format!("```{{{}}}", classes.join(" ")).into()
}

/// The files in `dir` with extension `ext`, sorted by name.
fn files(dir: &Path, ext: &str) -> Result<Vec<PathBuf>> {
    let mut out: Vec<PathBuf> = fs::read_dir(dir)
        .map_err(|e| format!("reading {}: {e}", dir.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|p| p.is_file() && p.extension() == Some(OsStr::new(ext)))
        .collect();
    out.sort();
    Ok(out)
}

fn file_stem(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

fn read(path: &Path) -> Result<String> {
    fs::read_to_string(path).map_err(|e| format!("reading {}: {e}", path.display()))
}

fn write(path: &Path, text: &str) -> Result {
    fs::write(path, text).map_err(|e| format!("writing {}: {e}", path.display()))
}

fn r(opts: Options) -> Result {
    let root = root();
    // roxygen2 loads the source with pkgload, which tolerates a stale
    // NAMESPACE (say, a renamed export) that `R CMD INSTALL` would reject, so
    // it runs before the install.
    step("r: regenerate man/ and NAMESPACE (roxygen2)");
    rscript(&root, r#"roxygen2::roxygenise("R/prospicio")"#)?;
    r_install_with_wrappers(&root)?;
    if opts.tests {
        // The runner CI's r job uses: every file, one per core with the
        // Bayesian and boosting files alone at the end, each failure
        // reported (#849, #845).
        step("r: test");
        let python = if cfg!(windows) { "python" } else { "python3" };
        run(Command::new(python)
            .arg("validation/scripts/run_r_tests.py")
            .current_dir(&root))?;
    }
    r_doc_check(&root)?;
    step("r: stage the R guides and examples as articles");
    stage_r_articles()?;
    step("r: render the docs site (pkgdown)");
    let mut cmd = Command::new("Rscript");
    cmd.args([
        "-e",
        r#"pkgdown::build_site("R/prospicio", install = FALSE, new_process = FALSE, preview = FALSE)"#,
    ])
    .current_dir(&root);
    // pkgdown needs pandoc; Quarto, which the Python docs need anyway,
    // bundles one.
    if env::var_os("RSTUDIO_PANDOC").is_none()
        && let Some(pandoc) = quarto_pandoc()
    {
        cmd.env("RSTUDIO_PANDOC", pandoc);
    }
    let built = with_generated_articles(&root, &mut cmd);
    built?;
    check_r_examples()?;
    if opts.check {
        unchanged(R_GENERATED)?;
    }
    Ok(())
}

/// Installs the R package, then writes `R/extendr-wrappers.R` from the
/// installed library's own `wrap__make_prospicio_wrappers` (#836). When the
/// file changes, roxygen2 and the install run again, since the file's
/// `@export` tags feed `NAMESPACE`.
fn r_install_with_wrappers(root: &Path) -> Result {
    let install = || {
        step("r: install");
        run(Command::new("R")
            .args(["CMD", "INSTALL", "R/prospicio"])
            .current_dir(root))
    };
    install()?;
    step("r: regenerate R/extendr-wrappers.R from the installed library");
    let raw = root.join("target/extendr-wrappers.raw.R");
    fs::create_dir_all(root.join("target")).map_err(|e| format!("creating target/: {e}"))?;
    rscript(
        root,
        &format!(
            concat!(
                r#"invisible(asNamespace("prospicio")); "#,
                r#"x <- .Call("wrap__make_prospicio_wrappers", use_symbols = TRUE, "#,
                r#"package_name = "prospicio", PACKAGE = "prospicio"); "#,
                r#"writeLines(x, "{}", useBytes = TRUE)"#,
            ),
            raw.display().to_string().replace('\\', "/")
        ),
    )?;
    let wrappers = root.join("R/prospicio/R/extendr-wrappers.R");
    let text = extendr_wrappers(&read(&raw)?)?;
    let before = fs::read_to_string(&wrappers).unwrap_or_default();
    if text != before {
        write(&wrappers, &text)?;
        step("r: R/extendr-wrappers.R changed: regenerate man/ and NAMESPACE, install again");
        rscript(root, r#"roxygen2::roxygenise("R/prospicio")"#)?;
        install()?;
    }
    Ok(())
}

/// One class's chunks in extendr's wrappers: its environment line, its
/// methods by name and its `$` and `[[` methods.
type ClassChunks = (String, Vec<(String, String)>, Vec<String>);

/// The committed `R/extendr-wrappers.R` from extendr's generated wrappers:
/// its doc lines dropped (keeping `@export`), the free functions sorted by
/// name and the impl blocks (each class's environment, its methods sorted by
/// name, and its `$` and `[[` methods) sorted by class, so the file is the
/// same on every platform (on Windows extendr lists the functions in
/// another order).
fn extendr_wrappers(raw: &str) -> Result<String> {
    // Chunks are separated by blank lines; doc lines (`#'`) other than
    // `@export` are dropped, and so are extendr's own comments and its
    // `@useDynLib` block, which the header below replaces.
    let mut chunks: Vec<Vec<String>> = Vec::new();
    let mut current: Vec<String> = Vec::new();
    for line in raw.lines() {
        let line = line.trim_end();
        if line.is_empty() {
            if !current.is_empty() {
                chunks.push(std::mem::take(&mut current));
            }
            continue;
        }
        if line.starts_with("#'") && line != "#' @export" {
            continue;
        }
        if line.starts_with('#') && !line.starts_with("#'") {
            continue;
        }
        current.push(line.to_string());
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    let mut free: Vec<(String, String)> = Vec::new();
    // Class name -> (environment line, methods by name, S3 methods).
    let mut classes: std::collections::BTreeMap<String, ClassChunks> =
        std::collections::BTreeMap::new();
    for chunk in chunks {
        let code: Vec<&String> = chunk.iter().filter(|l| *l != "#' @export").collect();
        let text = chunk.join("\n");
        let Some(first) = code.first() else { continue };
        if *first == "NULL" {
            continue;
        }
        if let Some(class) = first.strip_suffix(" <- new.env(parent = emptyenv())") {
            classes.entry(class.to_string()).or_default().0 = text;
        } else if let Some(rest) = first
            .strip_prefix("`$.")
            .or_else(|| first.strip_prefix("`[[."))
        {
            let class = rest.split('`').next().unwrap_or_default().to_string();
            classes.entry(class).or_default().2.push(text);
        } else if let Some((lhs, _)) = first.split_once(" <- ") {
            match lhs.split_once('$') {
                Some((class, method)) => classes
                    .entry(class.to_string())
                    .or_default()
                    .1
                    .push((method.to_string(), text)),
                None => free.push((lhs.to_string(), text)),
            }
        } else {
            return Err(format!("extendr wrappers: cannot read the chunk {text:?}"));
        }
    }
    free.sort();
    let mut out = String::from(concat!(
        "# Low-level bindings to crates/prospicio-r, generated by `cargo xtask r`\n",
        "# (or `cargo xtask generated`) from extendr's `wrap__make_prospicio_wrappers`,\n",
        "# with the doc lines removed and the functions and classes sorted by name.\n",
        "# Do not edit by hand. Internal: use the functions in distributions.R,\n",
        "# aggregate.R, reinsurance.R, reserving.R and risk.R.\n",
        "\n",
        "#' @useDynLib prospicio, .registration = TRUE\n",
        "NULL\n",
    ));
    for (_, text) in &free {
        out.push('\n');
        out.push_str(text);
        out.push('\n');
    }
    for (class, (env, mut methods, mut s3)) in classes {
        if env.is_empty() {
            return Err(format!(
                "extendr wrappers: methods of {class} without its environment"
            ));
        }
        methods.sort();
        // `$.X` before `[[.X`, as extendr writes them.
        s3.sort_by_key(|t| !t.contains("`$."));
        out.push('\n');
        out.push_str(&env);
        out.push('\n');
        for (_, text) in methods {
            out.push('\n');
            out.push_str(&text);
            out.push('\n');
        }
        for text in s3 {
            out.push('\n');
            out.push_str(&text);
            out.push('\n');
        }
    }
    Ok(out)
}

/// `tools::undoc` and `tools::codoc` on the installed package.
fn r_doc_check(root: &Path) -> Result {
    step("r: check that every export is documented and usage matches code");
    rscript(
        root,
        concat!(
            r#"u <- tools::undoc(package = "prospicio"); "#,
            r#"c <- tools::codoc(package = "prospicio"); "#,
            r#"if (length(unlist(u)) || length(c)) { print(u); print(c); quit(status = 1) }"#,
        ),
    )
}

/// The committed generated files and the checks that need no rendered site
/// (#823): the Python extension and its stub; roxygen2, the install with
/// the extendr wrappers, `undoc`/`codoc`, and pkgdown's check that every
/// export is in the reference index. With `--check`, fails if a committed
/// generated file changed.
fn generated(opts: Options) -> Result {
    let root = root();
    let dir = root.join("python");
    step("python: build the extension and regenerate its type stubs");
    run(Command::new("uv").arg("sync").current_dir(&dir))?;
    // `--release`, as `cargo xtask python` builds it: the docs job's cache
    // (shared with pages.yml) holds release builds, so a debug build here
    // would compile every crate cold.
    run(Command::new("uv")
        .args([
            "run",
            "maturin",
            "develop",
            "--uv",
            "--release",
            "--generate-stubs",
        ])
        .current_dir(&dir))?;
    step("r: regenerate man/ and NAMESPACE (roxygen2)");
    rscript(&root, r#"roxygen2::roxygenise("R/prospicio")"#)?;
    r_install_with_wrappers(&root)?;
    r_doc_check(&root)?;
    // pkgdown's own index builder, the step that failed in the PRs whose new
    // export was missing from `_pkgdown.yml`; `pkgdown::check_pkgdown()`
    // would also demand a site `url`, which this site does not set.
    step("r: check pkgdown's reference index (every export listed)");
    rscript(
        &root,
        r#"pkg <- pkgdown::as_pkgdown("R/prospicio"); invisible(pkgdown:::data_reference_index(pkg))"#,
    )?;
    if opts.check {
        let mut all = PYTHON_GENERATED.to_vec();
        all.extend_from_slice(R_GENERATED);
        unchanged(&all)?;
    }
    Ok(())
}

/// Runs the pkgdown build with the `articles:` index generated from
/// docs/guides/toc.yml. `_pkgdown.yml` holds the rest of the site's
/// configuration and no `articles:` key; the generated block is added to it
/// for this build only (the file is restored afterwards, whether or not the
/// build succeeds). pkgdown aborts on a staged article that the index does
/// not list, so a staged article that the table of contents does not place
/// (an R twin with no guide or notebook entry) is added to a last group,
/// "More articles".
fn with_generated_articles(root: &Path, cmd: &mut Command) -> Result {
    let toc = load_toc(root)?;
    let guide_twins: Vec<String> = files(&root.join("docs/guides"), "qmd")?
        .iter()
        .map(|p| file_stem(p))
        .collect();
    let example_twins: Vec<String> = files(&root.join("examples"), "qmd")?
        .iter()
        .map(|p| file_stem(p))
        .collect();
    let (block, listed) = r_articles_block(&toc, &guide_twins, &example_twins);
    let mut staged = Vec::new();
    for entry in
        fs::read_dir(root.join(R_ARTICLES)).map_err(|e| format!("reading articles: {e}"))?
    {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.extension().is_some_and(|e| e == "Rmd") {
            staged.push(file_stem(&path));
        }
    }
    staged.sort();
    let loose: Vec<&String> = staged.iter().filter(|n| !listed.contains(n)).collect();
    let path = root.join("R/prospicio/_pkgdown.yml");
    let original = read(&path)?;
    if original.lines().any(|l| l.starts_with("articles:")) {
        return Err(
            "R/prospicio/_pkgdown.yml has an `articles:` key; the index is generated from docs/guides/toc.yml"
                .into(),
        );
    }
    let mut text = original.clone();
    if !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(&block);
    if !loose.is_empty() {
        println!(
            "r: not placed by docs/guides/toc.yml, shown under \"More articles\": {}",
            loose
                .iter()
                .map(|s| s.as_str())
                .collect::<Vec<_>>()
                .join(", ")
        );
        text.push_str(concat!(
            "  - title: More articles\n",
            "    navbar: More articles\n",
            "    desc: Articles that docs/guides/toc.yml does not place.\n",
            "    contents:\n",
        ));
        for name in &loose {
            text.push_str(&format!("      - articles/{name}\n"));
        }
    }
    write(&path, &text)?;
    let built = run(cmd);
    write(&path, &original)?;
    built
}

/// The `articles:` block of `_pkgdown.yml` for the table of contents: a group
/// for each section, holding the pages that have an R twin (a guide's
/// `.qmd` in docs/guides, a notebook's in examples/), in order. A section
/// with no such page is left out. Also returns the article names listed.
fn r_articles_block(
    toc: &[TocSection],
    guide_twins: &[String],
    example_twins: &[String],
) -> (String, Vec<String>) {
    let quote = |s: &str| format!("\"{}\"", s.replace('\\', "\\\\").replace('"', "\\\""));
    let mut block = String::from("articles:\n");
    let mut listed = Vec::new();
    for section in toc {
        let names: Vec<&String> = section
            .pages
            .iter()
            .filter(|p| {
                if p.notebook {
                    example_twins.contains(&p.name)
                } else {
                    guide_twins.contains(&p.name)
                }
            })
            .map(|p| &p.name)
            .collect();
        if names.is_empty() {
            continue;
        }
        block.push_str(&format!("  - title: {}\n", quote(&section.title)));
        block.push_str(&format!("    navbar: {}\n", quote(&section.title)));
        block.push_str(&format!("    desc: {}\n", quote(&section.description)));
        block.push_str("    contents:\n");
        for name in names {
            block.push_str(&format!("      - articles/{name}\n"));
            listed.push(name.clone());
        }
    }
    (block, listed)
}

/// Where the R site's articles are staged: pkgdown renders each `.Rmd` in
/// `vignettes/articles` as `articles/<name>.html`. Gitignored and kept out of
/// the package tarball (.Rbuildignore); docs/guides/toc.yml groups the pages.
const R_ARTICLES: &str = "R/prospicio/vignettes/articles";

/// Where the repository's files are on GitHub, for the links in a guide to
/// files that are not pages of the site (the knowledge bundle).
const GITHUB_BLOB: &str = "https://github.com/EKtheSage/prospicio/blob/main/";

/// Stands for the root of the collected site (target/docs-site) while an
/// article's links are rewritten.
const SITE: &str = "<site>/";

/// Stages the R twins as articles of the pkgdown site: each `.qmd` in
/// docs/guides, then each `.qmd` in examples/, copied as R Markdown. The
/// `.qmd` files stay the only source.
///
/// A guide's chunks are not run: each prints nothing on the page (the guide
/// sets `output: false`) and is followed by a `text` block holding what it
/// prints, which R/prospicio/tests/test-guides.R checks. An example is run in
/// full, as the Python site runs the notebooks, since its results are its
/// output; its summary is checked after the build (`check_r_examples`).
///
/// Links are rewritten for the site: a Python guide or notebook points to its
/// page on the Python site (`../../python/user-guide/<name>.html` in the
/// collected site, target/docs-site), a sibling `.qmd` to its article, and a
/// path out of docs/guides to the file on GitHub.
fn stage_r_articles() -> Result {
    let root = root();
    let stage = root.join(R_ARTICLES);
    if stage.exists() {
        fs::remove_dir_all(&stage).map_err(|e| format!("clearing {}: {e}", stage.display()))?;
    }
    fs::create_dir_all(&stage).map_err(|e| format!("creating {}: {e}", stage.display()))?;
    let guides = root.join("docs/guides");
    let examples = root.join("examples");
    let python_guides: Vec<String> = files(&guides, "md")?
        .iter()
        .map(|p| file_stem(p))
        .filter(|s| s != "README")
        .collect();
    let notebooks: Vec<String> = files(&examples, "ipynb")?
        .iter()
        .map(|p| file_stem(p))
        .collect();
    let guide_twins = files(&guides, "qmd")?;
    let example_twins = files(&examples, "qmd")?;
    let twins: Vec<String> = guide_twins
        .iter()
        .chain(&example_twins)
        .map(|p| file_stem(p))
        .collect();
    // Site links are written with a placeholder root, so that the rewrite of
    // the remaining relative links (to GitHub) leaves them alone.
    let python_page = |s: &str| format!("]({SITE}python/user-guide/{s}.html");
    for (sources, run, up) in [
        (&guide_twins, false, "](../../"),
        (&example_twins, true, "](../"),
    ] {
        for source in sources {
            let mut text = read(source)?;
            for s in &python_guides {
                text = text.replace(&format!("]({s}.md"), &python_page(s));
                text = text.replace(&format!("](../docs/guides/{s}.md"), &python_page(s));
            }
            for s in &notebooks {
                text = text.replace(&format!("]({s}.ipynb"), &python_page(s));
                text = text.replace(&format!("](../../examples/{s}.ipynb"), &python_page(s));
            }
            for s in &twins {
                text = text.replace(&format!("]({s}.qmd"), &format!("]({s}.html"));
                text = text.replace(&format!("](../examples/{s}.qmd"), &format!("]({s}.html"));
                text = text.replace(&format!("](../../examples/{s}.qmd"), &format!("]({s}.html"));
            }
            text = text
                .replace(up, &format!("]({GITHUB_BLOB}"))
                .replace(&format!("]({SITE}"), "](../../");
            let marked = text.lines().any(|l| l.trim() == EXECUTE_MARK);
            if marked {
                text = executed_r(&text, &root);
            }
            let run = run || marked;
            let name = file_stem(source);
            let rel = source
                .strip_prefix(&root)
                .unwrap_or(source)
                .to_string_lossy()
                .replace('\\', "/");
            write(
                &stage.join(format!("{name}.Rmd")),
                &article(&text, &rel, &name, run),
            )?;
        }
    }
    Ok(())
}

/// An R Markdown article from a `.qmd`: its title, a first chunk that sets
/// whether the chunks run, a line naming the source, then the body with
/// inline code that knitr would run (`` `r ...` ``) made literal.
fn article(qmd: &str, source: &str, name: &str, run: bool) -> String {
    let (header, body) = qmd
        .strip_prefix("---")
        .and_then(|rest| rest.split_once("\n---"))
        .map_or(("", qmd), |(h, b)| {
            (h, b.split_once('\n').map_or("", |x| x.1))
        });
    let title = header
        .lines()
        .find_map(|l| l.strip_prefix("title:"))
        .map_or(format!("\"{name}\""), |t| t.trim().to_owned());
    let (options, note) = if run {
        (
            "warning = FALSE, message = FALSE",
            "The code runs each time the site is built, so the output below is current.",
        )
    } else {
        (
            "eval = FALSE",
            "The code is not run when the site is built: the block after each chunk is \
             what the chunk prints, and CI checks every one of them \
             (`R/prospicio/tests/test-guides.R`).",
        )
    };
    format!(
        "---\ntitle: {title}\n---\n\n```{{r, include = FALSE}}\nknitr::opts_chunk$set({options})\n```\n\n\
         *Rendered from [`{source}`]({GITHUB_BLOB}{source}). {note}*\n\n{}",
        literal_inline_r(body.trim_start())
    )
}

/// Outside code fences, turns `` `r ...` `` (R code in prose, such as
/// `` `r = 0.15` ``) into a double-backtick span that knitr does not run.
fn literal_inline_r(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut fenced = false;
    for line in text.lines() {
        if line.trim_start().starts_with("```") {
            fenced = !fenced;
        }
        let mut rest = line;
        if !fenced {
            while let Some(start) = rest.find("`r ").or_else(|| rest.find("`r#")) {
                let Some(len) = rest[start + 1..].find('`') else {
                    break;
                };
                let code = &rest[start + 1..start + 1 + len];
                out.push_str(&rest[..start]);
                out.push_str("`` ");
                out.push_str(code);
                out.push_str(" ``");
                rest = &rest[start + 2 + len..];
            }
        }
        out.push_str(rest);
        out.push('\n');
    }
    out
}

/// Fails unless each executed R example that reports a summary of its
/// comparisons ("N of M comparisons are inside their tolerance") has N = M.
/// The same words in the page's code (`%d of %d`) are not counts and are
/// skipped.
fn check_r_examples() -> Result {
    let root = root();
    let marker = " comparisons are inside their tolerance";
    for source in files(&root.join("examples"), "qmd")? {
        let name = file_stem(&source);
        let page = root.join(format!("R/prospicio/docs/articles/{name}.html"));
        let html = read(&page)?;
        for (end, _) in html.match_indices(marker) {
            let line = html[..end].rsplit('\n').next().unwrap_or("");
            let words: Vec<&str> = line.split_whitespace().rev().take(3).collect();
            let number = |w: &str| !w.is_empty() && w.bytes().all(|b| b.is_ascii_digit());
            if words.len() < 3 || words[1] != "of" || !number(words[0]) || !number(words[2]) {
                continue;
            }
            if words[0] != words[2] {
                return Err(format!(
                    "{}: {} of {} comparisons inside their tolerance",
                    page.display(),
                    words[2],
                    words[0]
                ));
            }
            println!(
                "r: {name}: {} of {} comparisons inside their tolerance",
                words[2], words[0]
            );
        }
    }
    Ok(())
}

fn rust() -> Result {
    step("rust: rustdoc");
    run(Command::new(cargo())
        .args(["doc", "--no-deps"])
        .env("RUSTDOCFLAGS", "-D warnings")
        .current_dir(root()))
}

fn docs(opts: Options) -> Result {
    rust()?;
    python(opts)?;
    r(opts)?;
    let root = root();
    let site = root.join("target/docs-site");
    step("docs: collect into target/docs-site");
    if site.exists() {
        fs::remove_dir_all(&site).map_err(|e| format!("clearing {}: {e}", site.display()))?;
    }
    for (from, to) in [
        ("target/doc", "rust"),
        ("python/great-docs/_site", "python"),
        ("R/prospicio/docs", "r"),
    ] {
        copy_dir(&root.join(from), &site.join(to))
            .map_err(|e| format!("copying {from} into the site: {e}"))?;
    }
    let index = INDEX;
    fs::write(site.join("index.html"), index)
        .map_err(|e| format!("writing the site index: {e}"))?;
    println!("docs: open {}", site.join("index.html").display());
    Ok(())
}

/// The landing page at the site's root, linking the three sites. Its look
/// is shared with python/docs-site.css and R/prospicio/pkgdown/extra.css.
const INDEX: &str = include_str!("../site/index.html");

/// Directory of the pandoc bundled with Quarto: `<quarto bin>/tools` on
/// Windows, `<quarto bin>/tools/<arch>` on Linux and macOS.
fn quarto_pandoc() -> Option<PathBuf> {
    let out = Command::new("quarto").arg("--paths").output().ok()?;
    let bin = String::from_utf8(out.stdout)
        .ok()?
        .lines()
        .next()?
        .trim()
        .to_owned();
    let tools = Path::new(&bin).join("tools");
    [tools.join(env::consts::ARCH), tools]
        .into_iter()
        .find(|dir| dir.join("pandoc").is_file() || dir.join("pandoc.exe").is_file())
}

/// Fails if any of `paths` differs from `HEAD` in content, or holds a new
/// untracked file. Content, not `git status`: under `core.autocrlf` a
/// regenerated file with LF endings shows as modified in `git status` even
/// when it matches the commit.
fn unchanged(paths: &[&str]) -> Result {
    let modified = git(&["diff", "HEAD", "--name-status", "--"], paths)?;
    let untracked = git(&["ls-files", "--others", "--exclude-standard", "--"], paths)?;
    let changed = format!("{modified}{untracked}");
    if changed.trim().is_empty() {
        Ok(())
    } else {
        let diff = git(&["--no-pager", "diff", "HEAD", "--"], paths)?;
        Err(format!(
            "generated docs are out of date; commit the regenerated files:
{changed}
{diff}"
        ))
    }
}

fn git(args: &[&str], paths: &[&str]) -> Result<String> {
    let out = Command::new("git")
        .args(args)
        .args(paths)
        .current_dir(root())
        .output()
        .map_err(|e| format!("running git: {e}"))?;
    if !out.status.success() {
        return Err(format!(
            "git {} failed: {}",
            args[0],
            String::from_utf8_lossy(&out.stderr)
        ));
    }
    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

/// Runs one R expression. Keep `expr` on one line: Rscript on Windows drops
/// everything after the first newline of an `-e` argument.
fn rscript(dir: &Path, expr: &str) -> Result {
    debug_assert!(!expr.contains('\n'), "Rscript -e takes one line");
    run(Command::new("Rscript").args(["-e", expr]).current_dir(dir))
}

/// The MinGW C compiler for 64-bit Windows, as named on Linux and in MSYS2.
const MINGW_GCC: &str = "x86_64-w64-mingw32-gcc";

fn found(program: &str) -> bool {
    Command::new(program)
        .arg("--version")
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .status()
        .is_ok()
}

fn xll() -> Result {
    let root = root();
    // MSVC on a Windows host; elsewhere the GNU target with MinGW's linker.
    let target = if cfg!(windows) {
        "x86_64-pc-windows-msvc"
    } else {
        "x86_64-pc-windows-gnu"
    };
    step(&format!("cargo build prospicio-xll ({target})"));
    let mut build = Command::new(cargo());
    build
        .current_dir(&root)
        .args([
            "build",
            "--release",
            "-p",
            "prospicio-xll",
            "--target",
            target,
        ])
        .args(["--config", "profile.release.strip=true"])
        // Link-time optimization across crates and one codegen unit drop the
        // engines' unused code: 16.3 MB to 14.0 MB (xll.md, decision 47).
        .args(["--config", "profile.release.lto=\"fat\""])
        .args(["--config", "profile.release.codegen-units=1"]);
    let linker = "CARGO_TARGET_X86_64_PC_WINDOWS_GNU_LINKER";
    if !cfg!(windows) && env::var_os(linker).is_none() {
        build.env(linker, MINGW_GCC);
    }
    run(&mut build)?;

    let out = root.join("target/xll");
    fs::create_dir_all(&out).map_err(|e| format!("{}: {e}", out.display()))?;
    let dll = root.join(format!("target/{target}/release/prospicio_xll.dll"));
    let xll = out.join("prospicio.xll");
    fs::copy(&dll, &xll).map_err(|e| format!("copy {}: {e}", dll.display()))?;
    println!("built {}", xll.display());

    // The smoke test: a C host standing in for Excel (harness/fake_excel.c).
    let wine = !cfg!(windows);
    if !found(MINGW_GCC) || (wine && !found("wine")) {
        println!(
            "skipping the fake-Excel smoke test: needs {MINGW_GCC}{}",
            if wine { " and wine" } else { "" }
        );
        return Ok(());
    }
    step("fake-Excel smoke test");
    let exe = out.join("fake_excel.exe");
    run(Command::new(MINGW_GCC)
        .arg("-O2")
        .arg("-o")
        .arg(&exe)
        .arg(root.join("crates/prospicio-xll/harness/fake_excel.c")))?;
    let mut smoke = if wine {
        let mut c = Command::new("wine");
        c.arg(&exe).env("WINEDEBUG", "-all");
        c
    } else {
        Command::new(&exe)
    };
    run(smoke.arg("prospicio.xll").current_dir(&out))
}

fn cargo() -> std::ffi::OsString {
    env::var_os("CARGO").unwrap_or_else(|| "cargo".into())
}

fn step(name: &str) {
    println!("\n==> {name}");
}

fn run(cmd: &mut Command) -> Result {
    let program = cmd.get_program().to_owned();
    let status = cmd
        .status()
        .map_err(|e| format!("could not start {}: {e}", show(&program)))?;
    if status.success() {
        Ok(())
    } else {
        Err(format!("{} failed ({status})", show(&program)))
    }
}

fn show(program: &OsStr) -> String {
    program.to_string_lossy().into_owned()
}

fn copy_dir(from: &Path, to: &Path) -> io::Result<()> {
    fs::create_dir_all(to)?;
    for entry in fs::read_dir(from)? {
        let entry = entry?;
        let target = to.join(entry.file_name());
        if entry.file_type()?.is_dir() {
            copy_dir(&entry.path(), &target)?;
        } else {
            fs::copy(entry.path(), target)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        TocPage, TocSection, article, check_toc, executed, executed_r, fence_info, page, parse_toc,
        r_articles_block, staged_name,
    };

    const TOC: &str = "# comment
- section: One
  description: The first.
  pages:
    - guide: a
    - notebook: n

- section: Two
  description: \"The second\"
  pages:
    - guide: b
";

    fn names(v: &[&str]) -> Vec<String> {
        v.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn the_toc_is_parsed_in_order() {
        let toc = parse_toc(TOC).unwrap();
        assert_eq!(toc.len(), 2);
        assert_eq!(toc[0].title, "One");
        assert_eq!(toc[1].description, "The second");
        let page = |name: &str, notebook| TocPage {
            name: name.into(),
            notebook,
        };
        assert_eq!(toc[0].pages, vec![page("a", false), page("n", true)]);
        assert!(parse_toc("- guide: a\n").is_err());
        assert!(parse_toc("- section: A\n  nonsense\n").is_err());
    }

    #[test]
    fn the_toc_must_match_the_files() {
        let toc = parse_toc(TOC).unwrap();
        assert!(check_toc(&toc, &names(&["a", "b"]), &names(&["n"])).is_ok());
        let err = check_toc(&toc, &names(&["a", "b", "c"]), &names(&["n", "m"])).unwrap_err();
        assert!(err.contains("guide docs/guides/c is not listed"), "{err}");
        assert!(err.contains("notebook examples/m is not listed"), "{err}");
        let err = check_toc(&toc, &names(&["a"]), &names(&["n"])).unwrap_err();
        assert!(
            err.contains("guide docs/guides/b is listed but does not exist"),
            "{err}"
        );
        let twice = parse_toc("- section: S\n  pages:\n    - guide: a\n    - guide: a\n").unwrap();
        let err = check_toc(&twice, &names(&["a"]), &[]).unwrap_err();
        assert!(err.contains("is listed twice"), "{err}");
    }

    #[test]
    fn the_committed_toc_matches_the_files() {
        super::load_toc(&super::root()).unwrap();
    }

    #[test]
    fn staged_names_sort_in_toc_order() {
        let mut staged: Vec<String> = (1..=120).map(|n| staged_name(n, "x")).collect();
        let sorted = staged.clone();
        staged.sort();
        assert_eq!(staged, sorted);
        assert_eq!(staged_name(7, "cat-0-overview"), "007-cat-0-overview.md");
    }

    #[test]
    fn the_r_index_lists_pages_with_a_twin_in_order() {
        let toc: Vec<TocSection> = parse_toc(TOC).unwrap();
        let (block, listed) = r_articles_block(&toc, &names(&["a"]), &names(&["n"]));
        assert_eq!(listed, vec!["a", "n"]);
        assert_eq!(
            block,
            "articles:
  - title: \"One\"
    navbar: \"One\"
    desc: \"The first.\"
    contents:
      - articles/a
      - articles/n
"
        );
    }

    #[test]
    fn a_fence_of_several_words_becomes_pandoc_attributes() {
        assert_eq!(fence_info("```python no-run"), "```{.python .no-run}");
        assert_eq!(fence_info("```python"), "```python");
        assert_eq!(fence_info("```"), "```");
        assert_eq!(fence_info("```{r, eval = FALSE}"), "```{r, eval = FALSE}");
        assert_eq!(fence_info("print(1)"), "print(1)");
        let staged = page("# T\n\n```python no-run\nx = 1\n```\n", "S");
        assert!(
            staged.contains("```{.python .no-run}\nx = 1\n```\n"),
            "{staged}"
        );
    }

    #[test]
    fn an_executed_guide_has_cells_and_drops_the_checked_output() {
        let guide = "# T\n\n<!-- site: execute -->\n\nProse.\n\n```python\nprint(1)\n```\n\n```text\n1\n```\n\nMore.\n\n```python no-run\nslow()\n```\n\n```text\nrecorded\n```\n\n```bash\npip install x\n```\n";
        let out = executed(guide, std::path::Path::new("C:\\g"));
        assert!(
            out.starts_with("```{python}\n#| include: false\nimport os\nos.chdir(r\"C:/g\")\n"),
            "{out}"
        );
        assert!(!out.contains("<!-- site: execute -->"), "{out}");
        assert!(out.contains("```{python}\nprint(1)\n```\n\nMore."), "{out}");
        assert!(!out.contains("```text\n1\n```"), "{out}");
        assert!(
            out.contains("```{python}\n#| eval: false\nslow()\n```\n\n```text\nrecorded\n```"),
            "{out}"
        );
        assert!(out.contains("```bash\npip install x\n```"), "{out}");
    }

    #[test]
    fn an_executed_r_twin_runs_from_the_root_and_drops_the_checked_output() {
        let twin = "---\ntitle: \"T\"\n---\n\n<!-- site: execute -->\n\nProse.\n\n```{r}\nprint(1)\n```\n\n```text\n[1] 1\n```\n\nMore.\n\n```{r, eval = FALSE}\nslow()\n```\n\n```text\nrecorded\n```\n";
        let out = executed_r(twin, std::path::Path::new("C:\\repo"));
        assert!(
            out.starts_with(
                "---\ntitle: \"T\"\n---\n```{r, include = FALSE}\n\
                 knitr::opts_knit$set(root.dir = \"C:/repo\")\n\
                 source(\"C:/repo/docs/guides/site/figures.R\")\n```\n"
            ),
            "{out}"
        );
        assert!(!out.contains("<!-- site: execute -->"), "{out}");
        assert!(out.contains("```{r}\nprint(1)\n```\n\nMore."), "{out}");
        assert!(!out.contains("[1] 1"), "{out}");
        assert!(
            out.contains("slow()\n```\n\n```text\nrecorded\n```"),
            "{out}"
        );
        let page = article(&out, "docs/guides/t.qmd", "t", true);
        assert!(page.starts_with("---\ntitle: \"T\"\n---"), "{page}");
        assert!(page.contains("knitr::opts_knit$set(root.dir"), "{page}");
    }
}

#[cfg(test)]
mod wrapper_tests {
    use super::extendr_wrappers;

    #[test]
    fn wrappers_drop_docs_and_sort_functions_and_classes() {
        let raw = "\
#' Doc of zeta.
zeta_rust <- function(x) .Call(wrap__zeta_rust, x)

alpha_rust <- function() .Call(wrap__alpha_rust)

#' @noRd
mid_rust <- function(y) .Call(wrap__mid_rust, y)

Zed <- new.env(parent = emptyenv())

Zed$b <- function() .Call(wrap__Zed__b, self)

Zed$a <- function() .Call(wrap__Zed__a, self)

#' @export
`$.Zed` <- function (self, name) { func <- Zed[[name]]; environment(func) <- environment(); func }

#' @export
`[[.Zed` <- `$.Zed`

#' A class.
#' @section Methods
Abc <- new.env(parent = emptyenv())

Abc$new <- function() .Call(wrap__Abc__new)

#' @export
`$.Abc` <- function (self, name) { func <- Abc[[name]]; environment(func) <- environment(); func }

#' @export
`[[.Abc` <- `$.Abc`
";
        let out = extendr_wrappers(raw).unwrap();
        let body = out.split("NULL\n").nth(1).unwrap();
        let firsts: Vec<&str> = body
            .split("\n\n")
            .map(|c| c.trim())
            .filter(|c| !c.is_empty())
            .map(|c| c.lines().last().unwrap())
            .collect();
        assert_eq!(
            firsts,
            [
                "alpha_rust <- function() .Call(wrap__alpha_rust)",
                "mid_rust <- function(y) .Call(wrap__mid_rust, y)",
                "zeta_rust <- function(x) .Call(wrap__zeta_rust, x)",
                "Abc <- new.env(parent = emptyenv())",
                "Abc$new <- function() .Call(wrap__Abc__new)",
                "`$.Abc` <- function (self, name) { func <- Abc[[name]]; environment(func) <- environment(); func }",
                "`[[.Abc` <- `$.Abc`",
                "Zed <- new.env(parent = emptyenv())",
                "Zed$a <- function() .Call(wrap__Zed__a, self)",
                "Zed$b <- function() .Call(wrap__Zed__b, self)",
                "`$.Zed` <- function (self, name) { func <- Zed[[name]]; environment(func) <- environment(); func }",
                "`[[.Zed` <- `$.Zed`",
            ]
        );
        assert!(!out.contains("Doc of zeta") && !out.contains("@noRd"));
        assert_eq!(out.matches("#' @export").count(), 4);
        // The same input in another order gives the same file.
        let shuffled = raw.replace("alpha_rust <- function() .Call(wrap__alpha_rust)\n\n", "")
            + "\nalpha_rust <- function() .Call(wrap__alpha_rust)\n";
        assert_eq!(extendr_wrappers(&shuffled).unwrap(), out);
    }
}
