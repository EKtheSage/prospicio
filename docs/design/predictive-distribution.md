# Design note: PredictiveDistribution

Status: **Core implemented** (draws only) · v0.1 · Depends on: `distributions.md` · Blocks: v0.1 ODP bootstrap, Mack output

## What exists

`prospicio_prob::PredictiveDistribution`, with:

- `from_draws(dims, components, draws, provenance)` for draws already laid
  out simulation-major;
- `simulate(dims, components, n_sims, seed, provenance, f)`, which fills row
  `i` in parallel from `StreamRng::new(seed, i)`, and records the seed,
  `SIM_INDEX_SCHEME` (`"chacha20/sim-index/v1"`) and the build's samplers
  (`provenance::SAMPLERS`) in the provenance;
- `marginal(key) -> Option<Sampled>`, `aggregate(keep) -> PredictiveDistribution`
  and `resample(rng, n)`, which draws whole rows;
- `total() -> &Sampled` (row sums, computed on first use). The
  `Distribution` and `Empirical` impls (`mean`, `quantile`, `var`, `tvar`, …)
  describe the total and go through `risk::*`;
- `Provenance` with `model`, `parameters`, `seed`, `stream_scheme`,
  `samplers`, `versions` (starting with `prospicio-prob`) and
  `input_hash`, and the checks `shares_streams` (seed and scheme) and
  `replays_same_draws` (seed, scheme and samplers);
- `prospicio_prob::portfolio`: `join(parts, dim, Pairing)` puts distributions of
  different models side by side under a new leading dimension (the union
  of their dimensions after it, `""` where a part lacks one), pairing
  simulation `i` of every part; `Pairing::Independent` refuses two parts
  with the same seed and stream scheme, which would share random numbers
  whatever their samplers,
  and `Pairing::SameSimulations` is for parts derived from the same
  scenarios. `reorder_groups(dim, correlation, seed)` sets the dependence
  between groups by Iman–Conover on their totals, moving each group's
  simulations as whole rows so its internal joint structure is kept. With
  `Tower::apply_aggregate` this closes the v0.4 gate: a reserve bootstrap
  and a tower feed capital allocation end to end
  (`validation/tests/aggregate.rs`);
- with the `arrow` feature, `write_ipc` / `read_ipc` (Arrow IPC files) and
  `to_record_batch` / `from_record_batch`, in `prospicio_prob::ipc`.

### Decisions

- **Draws only, `f64` only** (open questions 1 and 2): `Joint::Gaussian`
  and `f32` draws wait until a consumer or memory needs them. There is no
  `Joint` enum yet; it is introduced with the second variant.
- **Key values are `Int`, `Text` or `Period`.** `Period` is
  `prospicio_core::Period`, the same type as the Triangle's origins, so a reserve
  component keyed by origin joins back to its triangle row without
  conversion. A `Period` key never equals an `Int` key: the 2019 accident
  year is `Period::year(2019)`, not `2019`.
- **`aggregate(keep)`** keeps the listed dimensions in the listed order and
  sums the rest within each simulation. Groups appear in the order of their
  first component. `aggregate(&[])` is the total as a one-component
  distribution.
- **`draw_matrix()`** is the full simulation-major matrix. `Empirical::draws()`
  is the per-simulation total, like every other `Distribution` method.
- **Errors** use `prospicio_core::Error::InvalidParameter`, with the offending
  index or length as the value. Dedicated `Shape` / `UnknownKey` variants
  would read better, but changing `prospicio-core` needs its own PR (AGENTS.md).
- **`input_hash`** comes from `provenance::InputHasher` (BLAKE3 in
  key-derivation mode, context `INPUT_HASH_CONTEXT` =
  `"risk-rs 2026-09-30 input-hash v1"`). Each field is a one-byte type tag,
  a little-endian `u64` length and the payload, so field boundaries and
  types cannot collide. Floats are hashed by their bits, with `-0.0` as
  `0.0` and one canonical NaN. The hash is written `"blake3:<64 hex>"`.
  Golden values are reproduced independently by
  `validation/scripts/input_hash_golden.py`. Changing the encoding means a
  new context (`v2`). Models choose what to feed it; once Arrow lands, a
  triangle's canonical Arrow IPC bytes go in through `bytes()`.
- **Arrow IPC files** (open question 4), behind prospicio-prob's `arrow`
  feature (off by default; the `validation` crate turns it on, so
  `cargo test` covers it). The layout is **wide**: one non-null `Float64`
  column per component, so each column is a marginal and each row a
  simulation, and pyarrow, R `arrow` and Polars read it as a plain table.
  A long table (`sim, lob, origin, value`) would repeat every key in every
  simulation, and would force one Arrow type per dimension, which
  `KeyValue` does not. Each field's metadata holds its typed key as JSON
  (`{"int": …}`, `{"text": …}`, `{"period": {"start": "2019-01", "grain":
  "Y"}}`). The schema metadata holds the format name, `format_version`
  `"1"`, the dimension names and the provenance as JSON, with the seed as
  a decimal string because JSON numbers lose precision above 2^53.
  Readers reject unknown versions. A new optional provenance key is not a
  new version: readers look keys up by name, ignore the others and read
  a missing one as `null`. `samplers` was added that way on 2026-10-08
  (`rng.md`), so the version-1 files written before it, the pyarrow
  fixture among them, read with `samplers` not recorded. The full spec is in the `ipc` module
  docs. `validation/scripts/predictive_ipc.py` writes a fixture from that
  spec with pyarrow, which `validation/tests/predictive.rs` must read
  back, and it can check a file Rust wrote. Errors use `ipc::IpcError`
  rather than `prospicio_core::Error`, which has no I/O variant. arrow 59.x is
  the newest line that builds on rust-version 1.85.

## Goal

The single object every model returns and every risk measure consumes. It
is **joint by default**: components (origin years, lines, layers) keep
their dependence, because the 99.5% of a total reserve is not the sum of
per-origin 99.5%s.

## Shape

```text
PredictiveDistribution
├── components: Vec<ComponentKey>        e.g. {lob: "Auto", origin: 2019}
├── joint:      Joint                    how the components depend on each other
└── provenance: Provenance               model, params, seed, streams, version, input hash
```

```rust
pub enum Joint {
    /// n_sims × n_components, simulation-major: row i is simulation i.
    Draws(DrawMatrix),
    /// Deferred: multivariate normal / lognormal given mean and covariance
    /// (e.g. Mack with its cross-origin covariance).
    Gaussian { mean: Vec<f64>, cov: Vec<f64> },
}
```

**Simulation-major storage.** The Rayon outer loop runs over simulations,
so each task writes one contiguous row with no synchronization, and row i
is exactly the output of stream i. `aggregate` (row sums) is also
row-contiguous. `marginal` is a strided copy, which is cheap next to the
simulation that produced the draws.

## API

```text
mean() variance() cdf(x) quantile(p) var(p) tvar(p)   on the total (all components)
sample(n, seed)                                       resample whole rows (joint)
marginal(key)            -> Sampled                   one component
aggregate(selector)      -> PredictiveDistribution    sum components per simulation,
                                                      e.g. by lob, by calendar year
components()             -> &[ComponentKey]
provenance()             -> &Provenance
```

Risk measures are **not** implemented here. `var` / `tvar` delegate to the
shared risk-measure functions in `prospicio-prob` (the plan's rule: reserving
never implements its own quantiles).

## Component keys

A `ComponentKey` is an ordered set of `(dimension, value)` pairs with a
shared dimension schema per distribution, e.g. `(lob, origin)`. Aggregation
takes a list of dimensions to keep (`["lob"]` sums over origins). Values are
strings, integers or periods. Periods reuse the Triangle's period type, so an
origin in a reserve distribution and an origin in a triangle compare equal.

## Provenance

| Field | Example |
|---|---|
| `model` | `"odp_bootstrap"` |
| `parameters` | ordered key/value list (serializable) |
| `seed`, `stream_scheme` | `42`, `"chacha20/sim-index/v1"` |
| `samplers` | `[("gamma", "marsaglia-tsang/2026-10")]`, or none when not recorded |
| `versions` | crate version of `prospicio-prob` and the model's crate |
| `input_hash` | hash of the canonical input bytes (Arrow IPC of the triangle) |

`stream_scheme` names the rule in `rng.md` that maps simulations to
streams; `samplers` names the samplers that turned the streams' uniforms
into draws (a family not listed draws by inverse transform). With both,
a result can be replayed years later (`replays_same_draws`); `join`'s
check that independent parts do not share random numbers needs only the
first (`shares_streams`). See `rng.md`, stability policy.

## Memory

10,000 simulations × 100 components × 8 bytes = 8 MB, so v0.1 stores
everything. For 10^6 simulations × 10^3 components (8 GB) we will need
chunked or streaming aggregation. The API above does not preclude it
(`aggregate` and risk measures only need row-wise passes).

## Validation

Bootstrap outputs are checked in `validation/` by comparing moments and
quantiles, and by a KS statistic under fixed seeds, against chainladder-python
and R ChainLadder, with tolerances per test.

## Open questions

All four were decided on 2026-09-30, following the recommendations:

1. ~~`Joint::Gaussian` in v0.1~~: draws only; add `Gaussian` when a
   consumer needs analytic results.
2. ~~`f32` draws~~: `f64` only until memory forces the question.
3. ~~Hash function~~: BLAKE3 for `input_hash` (implemented, see Decisions).
4. ~~Serialization format~~: Arrow IPC, with provenance in the schema
   metadata (implemented, see Decisions).

Also decided:

5. ~~`Period` in `prospicio-core`~~: done in #13 (2026-09-30), and `KeyValue`
   has a `Period` variant.
