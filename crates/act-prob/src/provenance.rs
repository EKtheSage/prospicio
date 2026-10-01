//! Where a result came from: model, parameters, seed and versions.

/// Name of the rule mapping simulations to RNG streams used by
/// [`crate::PredictiveDistribution::simulate`]: simulation `i` draws only
/// from `StreamRng::new(seed, i)`. See `docs/design/rng.md`.
pub const SIM_INDEX_SCHEME: &str = "chacha20/sim-index/v1";

/// Audit record carried by every [`crate::PredictiveDistribution`], so a
/// result can be traced to its model and replayed from its seed.
///
/// ```
/// use act_prob::Provenance;
///
/// let p = Provenance::new("odp_bootstrap").param("n_sims", 10_000);
/// assert_eq!(p.model, "odp_bootstrap");
/// assert_eq!(p.parameters, [("n_sims".to_string(), "10000".to_string())]);
/// assert_eq!(p.versions[0].0, "act-prob");
/// ```
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Provenance {
    /// Model that produced the result, e.g. `"odp_bootstrap"`.
    pub model: String,
    /// Model parameters in the order the model reports them.
    pub parameters: Vec<(String, String)>,
    /// Seed of the simulation, if the result is simulated.
    pub seed: Option<u64>,
    /// Rule mapping simulations to streams, e.g. [`SIM_INDEX_SCHEME`].
    pub stream_scheme: Option<String>,
    /// Crate versions involved, starting with `act-prob`.
    pub versions: Vec<(String, String)>,
    /// Hash of the model's canonical input, set by the model.
    pub input_hash: Option<String>,
}

impl Provenance {
    /// Provenance for `model`, recording this crate's version.
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            parameters: Vec::new(),
            seed: None,
            stream_scheme: None,
            versions: vec![("act-prob".into(), env!("CARGO_PKG_VERSION").into())],
            input_hash: None,
        }
    }

    /// Adds a model parameter.
    pub fn param(mut self, name: impl Into<String>, value: impl ToString) -> Self {
        self.parameters.push((name.into(), value.to_string()));
        self
    }

    /// Records the version of another crate involved, e.g. the model's.
    pub fn version(mut self, krate: impl Into<String>, version: impl Into<String>) -> Self {
        self.versions.push((krate.into(), version.into()));
        self
    }

    /// Records the seed and the stream scheme it was used with.
    pub fn seed(mut self, seed: u64, stream_scheme: impl Into<String>) -> Self {
        self.seed = Some(seed);
        self.stream_scheme = Some(stream_scheme.into());
        self
    }

    /// Records the hash of the model's canonical input.
    pub fn input_hash(mut self, hash: impl Into<String>) -> Self {
        self.input_hash = Some(hash.into());
        self
    }
}
