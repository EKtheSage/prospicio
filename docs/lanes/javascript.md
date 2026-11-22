# JavaScript lane

**Session:** Cloud (the JavaScript session). **Owns:** `crates/prospicio-wasm/`, `docs/design/wasm.md`, the `wasm` task in `xtask/src/main.rs` and its filter in `xtask/src/check.rs`, the `wasm` job and the `changes` job's `wasm` output in `.github/workflows/ci.yml`

This lane wraps the crates in the WASM scope and changes none of them. JavaScript is only a front end to Rust: a calculation the crates lack goes to the owning lane first

<!-- One bullet per line. A PR edits only the bullets about its own item. This file merges with git's union driver (.gitattributes). -->

Finished items are in [done/javascript.md](done/javascript.md), which agents read when they need the lane's history; it is not imported into every session (#846).

## Next

- [wasm-publish] a `release.yml` job that builds the package and publishes it to npm on a version tag, with the registry setup in `docs/release.md` (`wasm.md` O2)
- [wasm-more-functions] further functions as the user asks, each in its lane's module file (`wasm.md` O5)
