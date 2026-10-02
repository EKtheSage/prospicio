//! Build tasks that span languages. Run `cargo xtask help`.
//!
//! Building a binding also regenerates its documentation, so the docs can
//! never lag the code they describe (docs/architecture.md, "Documentation"):
//!
//! - `python`: build the extension and its type stubs, run the tests, render
//!   the great-docs site.
//! - `r`: regenerate `man/` and `NAMESPACE` with roxygen2, install the
//!   package, run the tests, render the pkgdown site.
//! - `rust`: rustdoc for the pure-Rust crates, warnings denied.
//! - `docs`: all three, collected into one site under `target/docs-site`.
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
  r        install the R package, regenerate man/ and NAMESPACE, test, render its docs
  rust     build rustdoc for the pure-Rust crates
  docs     all of the above, collected into target/docs-site

  --check  fail if the build changed a committed generated file
";

/// Generated files that are committed, per binding: the build must leave
/// them unchanged under `--check`.
const PYTHON_GENERATED: &[&str] = &["python/actuarialrs/actuarialrs_native.pyi"];
const R_GENERATED: &[&str] = &["R/actuarialrs/man", "R/actuarialrs/NAMESPACE"];

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
    run(uv().args(["sync", "--group", "docs"]))?;
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
    step("python: render the docs site");
    run(uv()
        .args(["run", "great-docs", "build"])
        .env("PYTHONIOENCODING", "utf-8"))?;
    if check {
        unchanged(PYTHON_GENERATED)?;
    }
    Ok(())
}

fn r(check: bool) -> Result {
    let root = root();
    let pkg = "R/actuarialrs";
    // roxygen2 loads the source with pkgload, which tolerates a stale
    // NAMESPACE (say, a renamed export) that `R CMD INSTALL` would reject, so
    // it runs before the install.
    step("r: regenerate man/ and NAMESPACE (roxygen2)");
    rscript(&root, r#"roxygen2::roxygenise("R/actuarialrs")"#)?;
    step("r: install");
    run(Command::new("R")
        .args(["CMD", "INSTALL", pkg])
        .current_dir(&root))?;
    step("r: test");
    run(Command::new("Rscript")
        .arg("R/actuarialrs/tests/test-distributions.R")
        .current_dir(&root))?;
    step("r: check that every export is documented and usage matches code");
    rscript(
        &root,
        concat!(
            r#"u <- tools::undoc(package = "actuarialrs"); "#,
            r#"c <- tools::codoc(package = "actuarialrs"); "#,
            r#"if (length(unlist(u)) || length(c)) { print(u); print(c); quit(status = 1) }"#,
        ),
    )?;
    step("r: render the docs site (pkgdown)");
    let mut cmd = Command::new("Rscript");
    cmd.args([
        "-e",
        r#"pkgdown::build_site("R/actuarialrs", install = FALSE, new_process = FALSE, preview = FALSE)"#,
    ])
    .current_dir(&root);
    // pkgdown needs pandoc; Quarto, which the Python docs need anyway,
    // bundles one. (No let chain: the MSRV is 1.85.)
    if env::var_os("RSTUDIO_PANDOC").is_none() {
        if let Some(pandoc) = quarto_pandoc() {
            cmd.env("RSTUDIO_PANDOC", pandoc);
        }
    }
    run(&mut cmd)?;
    if check {
        unchanged(R_GENERATED)?;
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
        ("R/actuarialrs/docs", "r"),
    ] {
        copy_dir(&root.join(from), &site.join(to))
            .map_err(|e| format!("copying {from} into the site: {e}"))?;
    }
    fs::write(site.join("index.html"), INDEX)
        .map_err(|e| format!("writing the site index: {e}"))?;
    println!("docs: open {}", site.join("index.html").display());
    Ok(())
}

const INDEX: &str = r#"<!doctype html>
<html lang="en">
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>actuarialrs docs</title>
<style>
  body { font: 16px/1.5 system-ui, sans-serif; max-width: 40rem; margin: 3rem auto; padding: 0 1rem; }
</style>
<h1>actuarialrs</h1>
<p>Actuarial modeling on a Rust core. One kernel, documented for each front end.</p>
<ul>
  <li><a href="python/index.html">Python</a> (great-docs)</li>
  <li><a href="r/index.html">R</a> (pkgdown)</li>
  <li><a href="rust/act_prob/index.html">Rust crates</a> (rustdoc)</li>
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
