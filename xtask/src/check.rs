//! `cargo xtask check [--changed] [--fast]`: the checks `ci.yml` runs on a
//! pull request, in one command (#968).
//!
//! It mirrors the `checks` job (formatting and the Python governance
//! scripts) and the `rust` job (clippy, tests, rustdoc, the WASM check), and
//! adds `cargo xtask generated --check` when the bindings' inputs change.
//! The Python and R test suites, the Bayesian shards and the site render are
//! left to CI (`AGENTS.md`, "Checks", says when to run them by hand).
//!
//! - `--changed`: read `git diff --name-only origin/main...HEAD` (plus the
//!   working tree) and run only what CI's `changes` job would run.
//! - `--fast`: skip the slow Rust steps (`cargo test`, rustdoc, WASM) and
//!   the generated-files check; clippy still runs.
//!
//! A step whose tool is missing (R, uv, rustup's WASM target, Python) is
//! skipped with a printed note, and the run goes on after a failed step so
//! one run lists every failure.

use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use crate::{Options, Result, cargo, generated, root, run, rust, show, step};

/// What a diff asks for. Keep the filters below in step with the `changes`
/// job of `.github/workflows/ci.yml` (its `DOCS_ONLY` and the `rust` and
/// `docs` outputs); ci.yml carries the same reminder.
#[derive(Debug, PartialEq, Eq)]
pub(crate) struct Wanted {
    /// The `rust` job: anything but docs, knowledge, fragments, lane status
    /// (the Excel add-in and xtask are Rust).
    pub rust: bool,
    /// The `docs` job's generated files (`cargo xtask generated --check`).
    pub generated: bool,
}

/// ci.yml `DOCS_ONLY`, as a function. With `rust_counts` the Excel add-in and
/// `xtask` are not docs-only (ci.yml's `rust_free`).
fn docs_only(path: &str, rust_counts: bool) -> bool {
    const SCRIPTS: &[&str] = &[
        "check_okf",
        "check_lanes",
        "test_check_lanes",
        "test_rerun_references",
        "check_bindings",
        "test_check_bindings",
        "fold_changes",
        "bayes_shards",
    ];
    let top_level_md = |dir: &str| {
        path.strip_prefix(dir).is_some_and(|rest| {
            !rest.contains('/') && rest.len() > ".md".len() && rest.ends_with(".md")
        })
    };
    if path.starts_with("knowledge/")
        || path.starts_with("changes/")
        || path.starts_with("docs/lanes/")
        || path.starts_with("docs/design/")
        || top_level_md("docs/")
        || top_level_md("")
        || path == "validation/bindings-allowlist.toml"
        || path == "deny.toml"
    {
        return true;
    }
    if !rust_counts && (path.starts_with("crates/prospicio-xll/") || path.starts_with("xtask/")) {
        return true;
    }
    if let Some(name) = path.strip_prefix(".github/workflows/") {
        return ["xll", "release", "pages"]
            .iter()
            .any(|w| name.strip_suffix(".yml") == Some(w));
    }
    if let Some(name) = path.strip_prefix("validation/scripts/") {
        return name
            .strip_suffix(".py")
            .is_some_and(|stem| SCRIPTS.contains(&stem));
    }
    false
}

/// ci.yml's `docs` output: the generated files' inputs.
fn generated_input(path: &str) -> bool {
    const PREFIXES: &[&str] = &[
        "crates/prospicio-python/",
        "crates/prospicio-r/",
        "python/prospicio/",
        "R/prospicio/R/",
        "R/prospicio/man/",
        "R/prospicio/pkgdown/",
        "R/prospicio/src/",
        "xtask/",
    ];
    const FILES: &[&str] = &[
        "python/pyproject.toml",
        "python/uv.lock",
        "python/great-docs.yml",
        "python/docs-site.css",
        "R/prospicio/DESCRIPTION",
        "R/prospicio/NAMESPACE",
        "R/prospicio/_pkgdown.yml",
        ".github/workflows/ci.yml",
    ];
    PREFIXES.iter().any(|p| path.starts_with(p)) || FILES.contains(&path)
}

pub(crate) fn wanted(changed: &[String]) -> Wanted {
    Wanted {
        rust: changed.iter().any(|p| !docs_only(p, true)),
        generated: changed.iter().any(|p| generated_input(p)),
    }
}

/// How to start Python: the command and its leading arguments.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Python(Vec<String>);

impl Python {
    fn command(&self) -> Command {
        let mut c = Command::new(&self.0[0]);
        c.args(&self.0[1..]);
        c
    }
}

fn works(mut cmd: Command) -> bool {
    cmd.stdout(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .is_ok_and(|s| s.success())
}

/// Finds Python the way the scripts expect: `python3`, `python`, then
/// `uv run --no-project python`. The Windows Store's `python3` stub exits
/// non-zero, so it is passed over. `need_yaml` also asks for pyyaml, which
/// `check_okf.py` imports; uv gets it with `--with pyyaml`.
fn find_python(need_yaml: bool) -> Option<Python> {
    let probe = if need_yaml {
        "import yaml"
    } else {
        "import sys"
    };
    for name in ["python3", "python"] {
        let mut c = Command::new(name);
        c.args(["-c", probe]);
        if works(c) {
            return Some(Python(vec![name.into()]));
        }
    }
    let mut args: Vec<String> = ["uv", "run", "--no-project"].map(String::from).into();
    if need_yaml {
        args.extend(["--with".into(), "pyyaml".into()]);
    }
    args.push("python".into());
    let py = Python(args);
    let mut c = py.command();
    c.args(["-c", probe]);
    works(c).then_some(py)
}

enum Outcome {
    Passed,
    Failed(String),
    Skipped(String),
}

struct Report {
    rows: Vec<(String, Duration, Outcome)>,
}

impl Report {
    fn record(&mut self, name: &str, work: impl FnOnce() -> Outcome) {
        step(name);
        let start = Instant::now();
        let outcome = work();
        match &outcome {
            Outcome::Skipped(why) => println!("skipped: {why}"),
            Outcome::Failed(why) => eprintln!("FAILED: {why}"),
            Outcome::Passed => {}
        }
        self.rows.push((name.into(), start.elapsed(), outcome));
    }

    fn cmd(&mut self, name: &str, mut cmd: Command) {
        self.record(name, || match cmd.current_dir(root()).status() {
            Ok(s) if s.success() => Outcome::Passed,
            Ok(s) => Outcome::Failed(format!("{} failed ({s})", show(cmd.get_program()))),
            Err(e) => Outcome::Failed(format!("could not start {}: {e}", show(cmd.get_program()))),
        });
    }

    fn task(&mut self, name: &str, task: impl FnOnce() -> Result) {
        self.record(name, || match task() {
            Ok(()) => Outcome::Passed,
            Err(e) => Outcome::Failed(e),
        });
    }

    fn skip(&mut self, name: &str, why: &str) {
        self.record(name, || Outcome::Skipped(why.into()));
    }
}

/// The Python scripts of the `checks` job: a name, then the commands run in
/// order (`&&` in ci.yml), each a script under `validation/scripts/` and its
/// arguments.
type Script = (&'static str, &'static [&'static [&'static str]]);
const SCRIPTS: &[Script] = &[
    (
        "fold_changes.py",
        &[
            &["fold_changes.py", "--self-test"],
            &["fold_changes.py", "--check"],
        ],
    ),
    (
        "check_lanes.py",
        &[&["test_check_lanes.py"], &["check_lanes.py"]],
    ),
    ("test_rerun_references.py", &[&["test_rerun_references.py"]]),
    (
        "check_bindings.py",
        &[&["test_check_bindings.py"], &["check_bindings.py"]],
    ),
    (
        "check_reference_manifest.py",
        &[&["check_reference_manifest.py"]],
    ),
    ("bayes_shards.py", &[&["bayes_shards.py"]]),
];

fn changed_files() -> Result<Vec<String>> {
    let git = |args: &[&str]| -> Result<String> {
        let out = Command::new("git")
            .args(args)
            .current_dir(root())
            .output()
            .map_err(|e| format!("running git: {e}"))?;
        if !out.status.success() {
            return Err(format!(
                "git {} failed: {}\n(--changed needs `git fetch origin`)",
                args.join(" "),
                String::from_utf8_lossy(&out.stderr).trim()
            ));
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    };
    // The branch's commits as CI sees them, then what is not committed yet.
    let mut files: Vec<String> = Vec::new();
    for args in [
        &["diff", "--name-only", "origin/main...HEAD"][..],
        &["diff", "--name-only", "HEAD"][..],
        &["ls-files", "--others", "--exclude-standard"][..],
    ] {
        for line in git(args)?.lines().filter(|l| !l.is_empty()) {
            if !files.iter().any(|f| f == line) {
                files.push(line.to_string());
            }
        }
    }
    Ok(files)
}

pub(crate) fn check(changed: bool, fast: bool) -> Result {
    let started = Instant::now();
    let (want_rust, want_generated) = if changed {
        let files = changed_files()?;
        let w = wanted(&files);
        println!(
            "{} files changed: rust job {}, generated files {}",
            files.len(),
            if w.rust { "runs" } else { "skipped" },
            if w.generated { "checked" } else { "skipped" }
        );
        (w.rust, w.generated)
    } else {
        (true, true)
    };
    let root = root();
    let mut rep = Report { rows: Vec::new() };

    // The `checks` job: it runs on every pull request.
    let mut fmt = Command::new(cargo());
    fmt.args(["fmt", "--all", "--check"]);
    rep.cmd("cargo fmt --all --check", fmt);

    let scripts = root.join("validation/scripts");
    match find_python(true) {
        Some(py) => {
            let mut c = py.command();
            c.arg(scripts.join("check_okf.py")).arg("knowledge");
            rep.cmd("check_okf.py knowledge", c);
        }
        None => rep.skip(
            "check_okf.py knowledge",
            "no Python with pyyaml (python3, python, uv)",
        ),
    }
    let plain = find_python(false);
    for (name, cmds) in SCRIPTS {
        match &plain {
            Some(py) => rep.task(name, || {
                for args in *cmds {
                    let mut c = py.command();
                    c.arg(scripts.join(args[0]))
                        .args(&args[1..])
                        .current_dir(&root);
                    run(&mut c)?;
                }
                Ok(())
            }),
            None => rep.skip(name, "no Python (python3, python, uv)"),
        }
    }

    // The `rust` job.
    if want_rust {
        let mut clippy = Command::new(cargo());
        clippy.args([
            "clippy",
            "--all-targets",
            "--features",
            "prospicio-validation/slow-bayes",
            "--",
            "-D",
            "warnings",
        ]);
        rep.cmd("cargo clippy", clippy);
        if fast {
            rep.skip("cargo test, rustdoc, WASM check", "--fast");
        } else {
            let mut test = Command::new(cargo());
            test.arg("test");
            rep.cmd("cargo test", test);
            rep.task("cargo xtask rust (rustdoc)", rust);
            if wasm_target_installed() {
                let mut wasm = Command::new(cargo());
                wasm.args(["check", "--target", "wasm32-unknown-unknown"]);
                for p in [
                    "prospicio-core",
                    "prospicio-math",
                    "prospicio-prob",
                    "prospicio-aggregate",
                    "prospicio-pricing",
                    "prospicio-reserving",
                    "prospicio",
                ] {
                    wasm.args(["-p", p]);
                }
                rep.cmd("cargo check (wasm32)", wasm);
            } else {
                rep.skip(
                    "cargo check (wasm32)",
                    "target missing: run `rustup target add wasm32-unknown-unknown`",
                );
            }
        }
    } else {
        rep.skip("Rust job", "no Rust file changed");
    }

    // The `docs` job's generated files.
    if !want_generated {
        rep.skip("cargo xtask generated --check", "no binding input changed");
    } else if fast {
        rep.skip("cargo xtask generated --check", "--fast");
    } else if !tool_runs("Rscript") || !tool_runs("uv") {
        rep.skip(
            "cargo xtask generated --check",
            "needs uv and R (roxygen2, pkgload, pkgdown)",
        );
    } else {
        rep.task("cargo xtask generated --check", || {
            generated(Options {
                check: true,
                tests: false,
            })
        });
    }

    summary(&rep, started.elapsed())
}

fn tool_runs(program: &str) -> bool {
    let mut c = Command::new(program);
    c.arg("--version");
    works(c)
}

fn wasm_target_installed() -> bool {
    Command::new("rustup")
        .args(["target", "list", "--installed"])
        .output()
        .is_ok_and(|o| String::from_utf8_lossy(&o.stdout).contains("wasm32-unknown-unknown"))
}

fn summary(rep: &Report, total: Duration) -> Result {
    println!("\n==> summary");
    let mut failed = Vec::new();
    for (name, took, outcome) in &rep.rows {
        let (mark, note) = match outcome {
            Outcome::Passed => ("ok     ", String::new()),
            Outcome::Skipped(why) => ("skipped", format!(" ({why})")),
            Outcome::Failed(_) => ("FAILED ", String::new()),
        };
        println!("{mark} {:>7.1}s  {name}{note}", took.as_secs_f64());
        if matches!(outcome, Outcome::Failed(_)) {
            failed.push(name.as_str());
        }
    }
    println!("total {:.1}s", total.as_secs_f64());
    if failed.is_empty() {
        Ok(())
    } else {
        Err(format!("failed: {}", failed.join(", ")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn w(paths: &[&str]) -> Wanted {
        wanted(&paths.iter().map(|s| s.to_string()).collect::<Vec<_>>())
    }

    #[test]
    fn docs_only_changes_skip_rust() {
        for p in [
            "knowledge/index.md",
            "changes/changelog/x.md",
            "docs/lanes/reserving.md",
            "docs/design/models.md",
            "docs/release.md",
            "AGENTS.md",
            "validation/scripts/check_okf.py",
            "validation/bindings-allowlist.toml",
            ".github/workflows/pages.yml",
            "deny.toml",
        ] {
            assert!(!w(&[p]).rust, "{p} must not run the rust job");
        }
    }

    #[test]
    fn code_changes_run_rust() {
        for p in [
            "crates/prospicio-core/src/lib.rs",
            "crates/prospicio-xll/src/lib.rs",
            "xtask/src/main.rs",
            "Cargo.lock",
            ".github/workflows/ci.yml",
            "docs/guides/intro.md",
            "docs/sub/dir/x.md",
            "validation/scripts/some_generator.py",
            "validation/tests/aggregate.rs",
        ] {
            assert!(w(&[p]).rust, "{p} must run the rust job");
        }
    }

    #[test]
    fn one_code_file_among_docs_runs_rust() {
        assert!(
            w(&[
                "AGENTS.md",
                "knowledge/log.md",
                "crates/prospicio-prob/src/lib.rs"
            ])
            .rust
        );
        assert!(!w(&[]).rust);
    }

    #[test]
    fn generated_files_follow_the_bindings() {
        for p in [
            "crates/prospicio-python/src/lib.rs",
            "crates/prospicio-r/src/reserving.rs",
            "python/prospicio/reserving.py",
            "python/uv.lock",
            "R/prospicio/R/reserving.R",
            "R/prospicio/man/x.Rd",
            "R/prospicio/NAMESPACE",
            "xtask/src/main.rs",
            ".github/workflows/ci.yml",
        ] {
            assert!(w(&[p]).generated, "{p} must check the generated files");
        }
        for p in [
            "python/tests/test_x.py",
            "R/prospicio/tests/test.R",
            "crates/prospicio-core/src/lib.rs",
            "docs/guides/x.md",
        ] {
            assert!(!w(&[p]).generated, "{p} must not check the generated files");
        }
    }
}
