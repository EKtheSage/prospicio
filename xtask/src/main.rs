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
//! - `xll`: build the 64-bit Excel add-in as `target/xll/prospicio.xll`
//!   and, when a MinGW compiler (and off Windows, Wine) is found, run the
//!   fake-Excel smoke test on it (docs/design/xll.md).
//!
//! `--check` fails if a build changed a generated file that is committed
//! (the Python stub, R's `man/` and `NAMESPACE`): someone changed a doc
//! comment without rebuilding.

use std::env;
use std::ffi::OsStr;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::process::{Command, ExitCode};

const USAGE: &str = "\
usage: cargo xtask <task> [--check]

tasks:
  python   build the Python extension and stubs, test, render its docs
           (with the guides and the executed notebooks)
  r        install the R package, regenerate man/ and NAMESPACE, test, render its docs
  rust     build rustdoc for the pure-Rust crates
  docs     all of the above, collected into target/docs-site
  xll      build the 64-bit Excel add-in as target/xll/prospicio.xll and
           smoke-test it in a fake Excel when MinGW (and Wine) are found

  --check  fail if the build changed a committed generated file
";

/// Generated files that are committed, per binding: the build must leave
/// them unchanged under `--check`.
const PYTHON_GENERATED: &[&str] = &["python/prospicio/prospicio_native.pyi"];
const R_GENERATED: &[&str] = &["R/prospicio/man", "R/prospicio/NAMESPACE"];

type Result<T = ()> = std::result::Result<T, String>;

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    let check = args.iter().any(|a| a == "--check");
    let tasks: Vec<&str> = args
        .iter()
        .map(String::as_str)
        .filter(|a| *a != "--check")
        .collect();
    let result = match tasks.as_slice() {
        ["python"] => python(check),
        ["r"] => r(check),
        ["rust"] => rust(),
        ["docs"] => docs(check),
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

fn root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .expect("xtask lives one level below the workspace root")
        .to_path_buf()
}

fn python(check: bool) -> Result {
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
    step("python: test");
    run(uv().args(["run", "pytest", "tests"]))?;
    step("python: stage the user guide (the guides, and the notebooks executed)");
    stage_user_guide(&uv)?;
    step("python: render the docs site");
    // UTF-8 mode: great-docs writes _quarto.yml in the locale's encoding,
    // which on Windows cannot hold the reference titles' non-ASCII letters.
    run(uv()
        .args(["run", "great-docs", "build"])
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1"))?;
    if check {
        unchanged(PYTHON_GENERATED)?;
    }
    Ok(())
}

/// Where the Python site's user guide is staged; `user_guide` in
/// python/great-docs.yml points here.
const USER_GUIDE: &str = "target/user-guide";

/// Stages the Python site's user guide: each guide in docs/guides under
/// "Guides", then each notebook in examples/, executed and exported to
/// Markdown with its outputs and charts, under "Worked examples". The
/// notebooks are committed without outputs (.gitattributes), so this is where
/// their results are published. A numeric prefix orders the pages, and
/// great-docs strips it from their URLs: `user-guide/<name>.html`.
fn stage_user_guide(uv: &dyn Fn() -> Command) -> Result {
    let root = root();
    let stage = root.join(USER_GUIDE);
    if stage.exists() {
        fs::remove_dir_all(&stage).map_err(|e| format!("clearing {}: {e}", stage.display()))?;
    }
    fs::create_dir_all(&stage).map_err(|e| format!("creating {}: {e}", stage.display()))?;
    let notebooks = files(&root.join("examples"), "ipynb")?;
    let stems: Vec<String> = notebooks.iter().map(|p| file_stem(p)).collect();
    for guide in files(&root.join("docs/guides"), "md")? {
        let name = file_stem(&guide);
        if name == "README" {
            continue;
        }
        // A guide links to a notebook on GitHub; on the site, to its page.
        let mut text = read(&guide)?;
        for s in &stems {
            text = text.replace(&format!("](../../examples/{s}.ipynb"), &format!("]({s}.md"));
        }
        write(&stage.join(format!("1-{name}.md")), &page(&text, "Guides"))?;
    }
    for (notebook, s) in notebooks.iter().zip(&stems) {
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
            .args(["--output", s])
            .arg(notebook))?;
        let exported = stage.join(format!("{s}.md"));
        let text = read(&exported)?.replace("](../docs/guides/", "](");
        write(
            &stage.join(format!("2-{s}.md")),
            &page(&text, "Worked examples"),
        )?;
        fs::remove_file(&exported).map_err(|e| format!("removing {}: {e}", exported.display()))?;
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
        if title.is_none() && !fenced {
            if let Some(heading) = line.strip_prefix("# ") {
                title = Some(heading.trim().to_owned());
                continue;
            }
        }
        body.push_str(line);
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

fn r(check: bool) -> Result {
    let root = root();
    let pkg = "R/prospicio";
    // roxygen2 loads the source with pkgload, which tolerates a stale
    // NAMESPACE (say, a renamed export) that `R CMD INSTALL` would reject, so
    // it runs before the install.
    step("r: regenerate man/ and NAMESPACE (roxygen2)");
    rscript(&root, r#"roxygen2::roxygenise("R/prospicio")"#)?;
    step("r: install");
    run(Command::new("R")
        .args(["CMD", "INSTALL", pkg])
        .current_dir(&root))?;
    step("r: test");
    let tests = root.join("R/prospicio/tests");
    let mut files: Vec<PathBuf> = std::fs::read_dir(&tests)
        .map_err(|e| format!("cannot read {}: {e}", tests.display()))?
        .filter_map(|entry| entry.ok().map(|e| e.path()))
        .filter(|path| path.extension().is_some_and(|ext| ext == "R"))
        .collect();
    files.sort();
    for file in files {
        run(Command::new("Rscript").arg(&file).current_dir(&root))?;
    }
    step("r: check that every export is documented and usage matches code");
    rscript(
        &root,
        concat!(
            r#"u <- tools::undoc(package = "prospicio"); "#,
            r#"c <- tools::codoc(package = "prospicio"); "#,
            r#"if (length(unlist(u)) || length(c)) { print(u); print(c); quit(status = 1) }"#,
        ),
    )?;
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
    // bundles one. (No let chain: the MSRV is 1.85.)
    if env::var_os("RSTUDIO_PANDOC").is_none() {
        if let Some(pandoc) = quarto_pandoc() {
            cmd.env("RSTUDIO_PANDOC", pandoc);
        }
    }
    let built = with_loose_articles(&root, &mut cmd);
    built?;
    check_r_examples()?;
    if check {
        unchanged(R_GENERATED)?;
    }
    Ok(())
}

/// Runs the pkgdown build. pkgdown aborts on an article that `_pkgdown.yml`
/// does not list, so a staged article that no group takes in is added to a
/// last group, "More articles", in `_pkgdown.yml` for this build only (the
/// file is restored afterwards, whether or not the build succeeds). A new
/// guide can then never break the site; list it in its group to place it.
fn with_loose_articles(root: &Path, cmd: &mut Command) -> Result {
    let loose = uncovered_articles()?;
    if loose.is_empty() {
        return run(cmd);
    }
    println!(
        "r: not listed in _pkgdown.yml, shown under \"More articles\": {}",
        loose.join(", ")
    );
    let path = root.join("R/prospicio/_pkgdown.yml");
    let original = read(&path)?;
    let mut text = original.clone();
    if !text.ends_with('\n') {
        text.push('\n');
    }
    text.push_str(concat!(
        "  - title: More articles\n",
        "    navbar: More articles\n",
        "    desc: Articles not yet placed in a group of _pkgdown.yml.\n",
        "    contents:\n",
    ));
    for name in &loose {
        text.push_str(&format!("      - articles/{name}\n"));
    }
    write(&path, &text)?;
    let built = run(cmd);
    write(&path, &original)?;
    built
}

/// The staged articles that no group of `_pkgdown.yml` lists, by name or by a
/// `starts_with("articles/<prefix>")` line.
fn uncovered_articles() -> Result<Vec<String>> {
    let yml = read(&root().join("R/prospicio/_pkgdown.yml"))?;
    let mut names = Vec::new();
    for entry in
        fs::read_dir(root().join(R_ARTICLES)).map_err(|e| format!("reading articles: {e}"))?
    {
        let path = entry.map_err(|e| e.to_string())?.path();
        if path.extension().is_some_and(|e| e == "Rmd") {
            names.push(file_stem(&path));
        }
    }
    names.sort();
    Ok(uncovered(&yml, &names))
}

fn uncovered(yml: &str, names: &[String]) -> Vec<String> {
    let mut listed = Vec::new();
    let mut prefixes = Vec::new();
    for line in yml.lines() {
        let line = line.trim();
        if line.starts_with('#') {
            continue;
        }
        let Some(item) = line.strip_prefix("- ") else {
            continue;
        };
        let item = item.trim().trim_matches(|c| c == '"' || c == '\'');
        if let Some(rest) = item.strip_prefix("starts_with(") {
            let prefix = rest
                .trim_end_matches(')')
                .trim_matches(|c| c == '"' || c == '\'');
            if let Some(p) = prefix.strip_prefix("articles/") {
                prefixes.push(p.to_owned());
            }
        } else if let Some(name) = item.strip_prefix("articles/") {
            listed.push(name.to_owned());
        }
    }
    names
        .iter()
        .filter(|n| !listed.contains(n) && !prefixes.iter().any(|p| n.starts_with(p.as_str())))
        .cloned()
        .collect()
}

/// Where the R site's articles are staged: pkgdown renders each `.Rmd` in
/// `vignettes/articles` as `articles/<name>.html`. Gitignored and kept out of
/// the package tarball (.Rbuildignore); `_pkgdown.yml` groups the pages.
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

fn docs(check: bool) -> Result {
    rust()?;
    python(check)?;
    r(check)?;
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

const INDEX: &str = r#"<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>prospicio docs</title>
<style>
  body { font: 16px/1.5 system-ui, sans-serif; max-width: 40rem; margin: 3rem auto; padding: 0 1rem; }
</style>
<h1>prospicio</h1>
<p>Actuarial modeling on a Rust core. One kernel, documented for each front end.</p>
<ul>
  <li><a href="python/index.html">Python</a> (great-docs, with the guides and worked examples)</li>
  <li><a href="r/index.html">R</a> (pkgdown, with the guides and worked examples)</li>
  <li><a href="rust/prospicio_prob/index.html">Rust crates</a> (rustdoc)</li>
</ul>
"#;

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
        .args(["--config", "profile.release.strip=true"]);
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
    use super::uncovered;

    #[test]
    fn a_staged_article_no_group_lists_is_uncovered() {
        let yml = "articles:
  - title: A
    contents:
      - articles/one
      - starts_with(\"articles/pir-\")
    # - articles/commented
";
        let names: Vec<String> = ["one", "pir-x", "two", "commented"]
            .map(String::from)
            .to_vec();
        assert_eq!(uncovered(yml, &names), vec!["two", "commented"]);
    }
}
