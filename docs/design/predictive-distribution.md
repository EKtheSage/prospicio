# Design note: PredictiveDistribution

Status: **Core implemented** (draws only) · v0.1 · Depends on: `distributions.md` · Blocks: v0.1 ODP bootstrap, Mack output

## What exists

`act_prob::PredictiveDistribution`, with:

- `from_draws(dims, components, draws, provenance)` for draws already laid
  out simulation-major;
- `simulate(dims, components, n_sims, seed, provenance, f)`, which fills row
  `i` in parallel from `StreamRng::new(seed, i)`, and records the seed and
  `SIM_INDEX_SCHEME` (`"chacha20/sim-index/v1"`) in the provenance;
- `marginal(key) -> Option<Sampled>`, `aggregate(keep) -> PredictiveDistribution`
  and `resample(rng, n)`, which draws whole rows;
- `total() -> &Sampled` (row sums, computed on first use). The
  `Distribution` and `Empirical` impls (`mean`, `quantile`, `var`, `tvar`, …)
  describe the total and go through `risk::*`;
- `Provenance` with `model`, `parameters`, `seed`, `stream_scheme`,
  `versions` (starting with `act-prob`) and `input_hash`.

### Decisions

- **Draws only, `f64` only** (open questions 1 and 2): `Joint::Gaussian`
  and `f32` draws wait until a consumer or memory needs them. There is no
  `Joint` enum yet; it is introduced with the second variant.
- **Key values are `Int` or `Text`.** The Triangle's `Period` lives in
  `act-reserving`, which `act-prob` cannot depend on. When `Period` moves to
  `act-core`, `KeyValue` gains a `Period` variant so reserve components
  join back to triangle origins without conversion.
- **`aggregate(keep)`** keeps the listed dimensions in the listed order and
  sums the rest within each simulation. Groups appear in the order of their
  first component. `aggregate(&[])` is the total as a one-component
  distribution.
- **`draw_matrix()`** is the full simulation-major matrix. `Empirical::draws()`
  is the per-simulation total, like every other `Distribution` method.
- **Errors** use `act_core::Error::InvalidParameter`, with the offending
  index or length as the value. Dedicated `Shape` / `UnknownKey` variants
  would read better, but changing `act-core` needs its own PR (AGENTS.md).
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
- **Deferred to its own PR:** Arrow IPC persistence with provenance in the
  schema metadata (open question 4), which adds the `arrow` dependency.

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
shared risk-measure functions in `act-prob` (the plan's rule: reserving
never implements its own quantiles).

## Component keys

A `ComponentKey` is an ordered set of `(dimension, value)` pairs with a
shared dimension schema per distribution, e.g. `(lob, origin)`. Aggregation
takes a list of dimensions to keep (`["lob"]` sums over origins). Values are
strings or integers. Periods reuse the Triangle's period type, so an
origin in a reserve distribution and an origin in a triangle compare equal.

## Provenance

| Field | Example |
|---|---|
| `model` | `"odp_bootstrap"` |
| `parameters` | ordered key/value list (serializable) |
| `seed`, `stream_scheme` | `42`, `"chacha20/sim-index/v1"` |
| `package_version` | crate version of `act-prob` and the model's crate |
| `input_hash` | hash of the canonical input bytes (Arrow IPC of the triangle) |

`stream_scheme` names the rule in `rng.md` that maps simulations to
streams, so a result can be replayed years later.

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
   metadata (own PR).

Also decided:

5. ~~`Period` in `act-core`~~: done in #13 (2026-09-30). `KeyValue` gains a
   `Period` variant in its own PR.
