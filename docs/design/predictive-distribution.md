# Design note: PredictiveDistribution

Status: **Draft for review** · Phase 0 · Depends on: `distributions.md` · Blocks: v0.1 ODP bootstrap, Mack output

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

1. **`Joint::Gaussian` in v0.1?** Mack naturally produces mean and
   covariance. Returning it as `Gaussian` keeps Mack analytic; converting to
   draws makes every downstream step uniform. Recommendation: draws only
   for v0.1; add `Gaussian` when a consumer needs analytic results.
2. **f64 only, or allow f32 draws** for very large simulations?
   Recommendation: f64 only until memory forces the question.
3. **Hash function** for `input_hash`: BLAKE3 (fast, a new dependency) vs a
   simpler non-cryptographic hash. Integrity against tampering is not a
   goal, but collision resistance for audit is.
4. **Serialization format** for persisted results: Arrow IPC (fits the data
   plan) vs a custom format. Recommendation: Arrow IPC, with provenance in
   schema metadata.
