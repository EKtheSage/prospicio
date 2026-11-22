### Examples

- `cargo xtask check [--changed] [--fast]` runs the checks CI runs on a pull
  request in one command: formatting, the Python governance scripts, clippy,
  tests, rustdoc, the WASM check and, when bindings change, `generated --check`.
  `--changed` runs only what CI's change filters ask for. AGENTS.md "Checks"
  names it and now lists `check_okf.py` and `bayes_shards.py` (#968, #1027).
