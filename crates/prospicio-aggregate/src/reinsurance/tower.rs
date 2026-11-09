//! The tower: layers in stages applied to simulated years.

use super::*;

/// Which entries of a year each [scoped](Layer::scope) layer sees: each
/// entry's label combination and, for each layer, whether each
/// combination is in its scope (`None` for an unscoped layer).
#[derive(Debug, Clone, Copy)]
pub(crate) struct ScopeView<'a> {
    /// Each aggregate column's combination.
    pub(crate) aggregate: &'a [usize],
    /// Each event's combination.
    pub(crate) events: &'a [usize],
    pub(crate) masks: &'a [Option<Vec<bool>>],
}

/// A reinsurance programme: layers in inuring stages.
///
/// Layers in the same stage see the same losses: the gross losses in the
/// first stage, and in each later stage the losses net of every earlier
/// stage, event by event (see [`Layer::ceded_by_event`]). Within a stage,
/// layers that overlap would both pay.
#[derive(Debug, Clone, PartialEq)]
pub struct Tower {
    pub layers: Vec<Layer>,
    /// Stage of each layer, starting at 0 and non-decreasing.
    pub stages: Vec<usize>,
    /// Scales read on the loss ratio of several layers together
    /// ([`Tower::deal_scale`]), in the order they were added.
    pub deals: Vec<DealScale>,
    /// How the programme attaches to a term's losses, for
    /// [`Tower::apply_timeline`]: `None` until set with
    /// [`Tower::cover_basis`].
    pub cover_basis: Option<CoverBasis>,
    /// Whether the programme's contracts interlock under risks attaching
    /// ([`Tower::interlocking`]); off by default.
    pub interlocking: bool,
}

impl Tower {
    /// A tower of `layers` in one stage, all seeing the gross losses;
    /// fails if there are none or two share a name.
    pub fn new(layers: Vec<Layer>) -> Result<Self> {
        Self::inuring(vec![layers])
    }

    /// A tower whose stages inure in order: each stage's layers see the
    /// losses net of all earlier stages. Fails if any stage is empty or
    /// two layers share a name.
    ///
    /// ```
    /// use prospicio_aggregate::{Layer, Tower};
    ///
    /// // A 50% quota share inures to the benefit of a 5 xs 5 cover:
    /// // a 30 loss is 15 net of the quota share, so the cover pays 5.
    /// let tower = Tower::inuring(vec![
    ///     vec![Layer::quota_share("QS", 0.5).unwrap()],
    ///     vec![Layer::xol("5x5", 5.0, 5.0).unwrap()],
    /// ])
    /// .unwrap();
    /// assert_eq!(tower.ceded(&[30.0]), [15.0, 5.0]);
    /// ```
    pub fn inuring(stages: Vec<Vec<Layer>>) -> Result<Self> {
        if stages.is_empty() || stages.iter().any(Vec::is_empty) {
            return Err(invalid("layers", 0.0, "must not be empty"));
        }
        let stage_of = stages
            .iter()
            .enumerate()
            .flat_map(|(i, stage)| std::iter::repeat_n(i, stage.len()))
            .collect();
        let layers: Vec<Layer> = stages.into_iter().flatten().collect();
        for (i, layer) in layers.iter().enumerate() {
            if layers[..i].iter().any(|l| l.name == layer.name) {
                return Err(invalid("layers", i as f64, "repeats an earlier layer name"));
            }
        }
        let stage_of: Vec<usize> = stage_of;
        for (i, layer) in layers.iter().enumerate() {
            let Some(d) = &layer.drop_down else { continue };
            let Some(u) = layers.iter().position(|l| l.name == d.underlying) else {
                return Err(Error::Data(format!(
                    "layer {:?} drops down on layer {:?}, which the tower does not have",
                    layer.name, d.underlying
                )));
            };
            if u > i || stage_of[u] != stage_of[i] {
                return Err(Error::Data(format!(
                    "layer {:?} drops down on layer {:?}, which must be earlier in the same stage",
                    layer.name, d.underlying
                )));
            }
            if !layers[u].aggregate_limit.is_finite() {
                return Err(Error::Data(format!(
                    "layer {:?} drops down on layer {:?}, which has no annual limit to use up",
                    layer.name, d.underlying
                )));
            }
        }
        Ok(Self {
            layers,
            stages: stage_of,
            deals: Vec::new(),
            cover_basis: None,
            interlocking: false,
        })
    }

    /// Ceded loss of each layer, in tower order, for one year's losses,
    /// taken as given as [`Layer::ceded`] takes them;
    /// [`try_ceded`](Self::try_ceded) refuses a negative loss.
    pub fn ceded(&self, losses: &[f64]) -> Vec<f64> {
        self.year(losses, None, None)
            .into_iter()
            .map(|(c, _)| c)
            .collect()
    }

    /// The tower's total ceded loss for one occurrence of `loss` on the
    /// loss basis, before any annual terms: each stage sees the loss net of
    /// the earlier stages, and each layer cedes its share of its
    /// per-occurrence recovery. This is the program's ceded function `g` of
    /// an occurrence that [`pml::program_occurrence_pml`](crate::pml::program_occurrence_pml)
    /// reads; annual terms (aggregate deductibles and limits, reinstatements)
    /// do not enter, since they act on a year's losses, not one occurrence.
    /// Fails for a layer on a sum-insured basis or one that needs a year's
    /// events in order.
    ///
    /// ```
    /// use prospicio_aggregate::{Layer, Tower};
    ///
    /// let t = Tower::new(vec![
    ///     Layer::xol("A", 50.0, 50.0).unwrap(),
    ///     Layer::xol("B", 100.0, 100.0).unwrap().share(0.5).unwrap(),
    /// ])
    /// .unwrap();
    /// assert_eq!(t.occurrence_ceded(160.0).unwrap(), 50.0 + 30.0);
    /// ```
    pub fn occurrence_ceded(&self, loss: f64) -> Result<f64> {
        if let Some(l) = self
            .layers
            .iter()
            .find(|l| l.basis != Basis::Loss || l.needs_event_order())
        {
            return Err(Error::Data(format!(
                "layer {:?}: an occurrence's ceded loss needs a layer on the loss basis that reads \
                 no order of events",
                l.name
            )));
        }
        let (mut seen, mut total, mut stage, mut stage_ceded) = (loss, 0.0, 0usize, 0.0);
        for (l, &s) in self.layers.iter().zip(&self.stages) {
            if s != stage {
                seen -= stage_ceded;
                stage_ceded = 0.0;
                stage = s;
            }
            let c = l.share * l.recovery(seen);
            stage_ceded += c;
            total += c;
        }
        Ok(total)
    }

    /// [`ceded`](Self::ceded), refusing a loss that is negative or not
    /// finite (#593).
    pub fn try_ceded(&self, losses: &[f64]) -> Result<Vec<f64>> {
        check_losses(losses)?;
        Ok(self.ceded(losses))
    }

    /// Ceded loss of each layer for one year's losses on risks with the
    /// given sums insured, one per loss. Every stage sees each risk's
    /// original sum insured.
    pub fn ceded_with_sums_insured(&self, losses: &[f64], sums_insured: &[f64]) -> Vec<f64> {
        self.year(losses, Some(sums_insured), None)
            .into_iter()
            .map(|(c, _)| c)
            .collect()
    }

    /// Whether any layer needs each event's sum insured.
    pub fn needs_sums_insured(&self) -> bool {
        self.layers.iter().any(Layer::needs_sums_insured)
    }

    /// Whether any layer needs each event's time.
    pub fn needs_times(&self) -> bool {
        self.layers.iter().any(Layer::needs_times)
    }

    /// An error for a path with no events in order (a grid, an annual
    /// total), naming the layer, when a layer is an nth-event or
    /// top-and-drop cover (#591).
    pub(crate) fn refuse_event_order(&self, path: &str) -> Result<()> {
        match self.layers.iter().find(|l| l.needs_event_order()) {
            Some(l) => Err(Error::Data(format!(
                "layer {:?} is an nth-event or top-and-drop cover, which reads the order of a \
                 year's events; {path} has none, so apply the tower to an event set",
                l.name
            ))),
            None => Ok(()),
        }
    }

    /// An error for a path of single years, naming the layer, when a
    /// layer's no-claims bonus looks back over several contract years,
    /// which only a term has (#592).
    pub(crate) fn refuse_lookback(&self, path: &str) -> Result<()> {
        match self
            .layers
            .iter()
            .find(|l| l.no_claims_bonus.is_some_and(|b| b.years > 1))
        {
            Some(l) => Err(Error::Data(format!(
                "layer {:?} has a no-claims bonus over several years, which {path} cannot \
                 look back over; apply the tower over a term (apply_term)",
                l.name
            ))),
            None => Ok(()),
        }
    }

    /// Whether any layer is [scoped](Layer::scope) to some labels.
    pub fn has_scope(&self) -> bool {
        self.layers.iter().any(|l| l.scope.is_some())
    }

    /// An error for a path with no labelled events, naming the scoped
    /// layer, when the tower has one.
    pub(crate) fn refuse_scope(&self, path: &str) -> Result<()> {
        match self.layers.iter().find(|l| l.scope.is_some()) {
            Some(l) => Err(Error::Data(format!(
                "layer {:?} is scoped to some labels, which {path} does not carry; apply the \
                 tower to labelled events",
                l.name
            ))),
            None => Ok(()),
        }
    }

    /// Each layer's scope over `events`' label combinations: for a scoped
    /// layer, whether each combination is in it; `None` for an unscoped
    /// layer, and for every layer when none is scoped.
    pub(crate) fn scope_masks(&self, events: &EventSet) -> Result<Option<Vec<Option<Vec<bool>>>>> {
        if !self.has_scope() {
            return Ok(None);
        }
        let Some(labels) = events.label_set() else {
            return self.refuse_scope("unlabelled events").map(|_| None);
        };
        self.layers
            .iter()
            .map(|l| {
                let Some(scope) = &l.scope else {
                    return Ok(None);
                };
                let d = labels
                    .dims
                    .iter()
                    .position(|x| *x == scope.label)
                    .ok_or_else(|| {
                        Error::Data(format!(
                            "layer {:?} is scoped to label {:?}, which the events do not carry \
                         (they carry {:?})",
                            l.name, scope.label, labels.dims
                        ))
                    })?;
                let mask: Vec<bool> = labels
                    .combinations
                    .iter()
                    .map(|c| scope.values.contains(&c[d]))
                    .collect();
                if !mask.contains(&true) {
                    return Err(Error::Data(format!(
                        "layer {:?} is scoped to {:?} in {:?}, none of which the events carry",
                        l.name, scope.values, scope.label
                    )));
                }
                Ok(Some(mask))
            })
            .collect::<Result<Vec<_>>>()
            .map(Some)
    }

    /// Ceded loss and reinstatement premium of each layer for one year.
    pub(crate) fn year(
        &self,
        losses: &[f64],
        si: Option<&[f64]>,
        times: Option<&[f64]>,
    ) -> Vec<(f64, f64)> {
        self.year_by_event(&[], losses, si, times, None, None, None, None)
    }

    /// [`year`](Self::year) with each layer's scale key: the ceded loss, or
    /// for a layer with
    /// [`scales_before_annual_terms`](Layer::scales_before_annual_terms) its
    /// share of the year's recoveries before the annual terms.
    pub(crate) fn year_keyed(&self, losses: &[f64]) -> (Vec<(f64, f64)>, Vec<f64>) {
        let mut keys = Vec::with_capacity(self.layers.len());
        let year = self.year_by_event(&[], losses, None, None, None, None, None, Some(&mut keys));
        (year, keys)
    }

    /// [`year`](Self::year) with an aggregate part `aggregate` beside the
    /// events, and with `by_event` each layer's ceded loss by entry (one
    /// vector per layer, in tower order, the aggregate part's entries
    /// first). The aggregate part comes before the events in the year and
    /// only layers that [see it](Layer::sees_aggregate_part) cede from it.
    /// With `keys` each layer's scale key: the ceded loss, or for a layer
    /// with [`scales_before_annual_terms`](Layer::scales_before_annual_terms)
    /// its share of the year's recoveries before the annual terms.
    #[allow(clippy::too_many_arguments)]
    fn year_by_event(
        &self,
        aggregate: &[f64],
        losses: &[f64],
        si: Option<&[f64]>,
        times: Option<&[f64]>,
        lf: Option<&[f64]>,
        scope: Option<ScopeView<'_>>,
        mut by_event: Option<&mut Vec<Vec<f64>>>,
        mut keys: Option<&mut Vec<f64>>,
    ) -> Vec<(f64, f64)> {
        let mut ceded = Vec::with_capacity(self.layers.len());
        let n_agg = aggregate.len();
        // Each entry's label combination, the aggregate part's first, when a
        // layer is scoped.
        let all_of: Option<Vec<usize>> =
            scope.map(|v| v.aggregate.iter().chain(v.events).copied().collect());
        let mut masked: Vec<f64> = Vec::new();
        let mut dropped: Vec<f64> = Vec::new();
        let mut seen: Vec<f64> = aggregate.iter().chain(losses).copied().collect();
        // A layer that sees the aggregate part is on the loss, so needs no
        // sum insured; the part's time is the year's start.
        let all_times: Option<Vec<f64>> = (n_agg > 0)
            .then(|| {
                times.map(|t| {
                    std::iter::repeat_n(0.0, n_agg)
                        .chain(t.iter().copied())
                        .collect()
                })
            })
            .flatten();
        // The aggregate part is not an event, so its limit is not scaled.
        let all_lf: Option<Vec<f64>> = (n_agg > 0)
            .then(|| {
                lf.map(|f| {
                    std::iter::repeat_n(1.0, n_agg)
                        .chain(f.iter().copied())
                        .collect()
                })
            })
            .flatten();
        let last_stage = self.stages.last().copied().unwrap_or(0);
        let mut i = 0;
        while i < self.layers.len() {
            let stage = self.stages[i];
            let end = i + self.stages[i..].iter().take_while(|&&s| s == stage).count();
            let mut stage_by_event = vec![0.0; seen.len()];
            for (li, layer) in self.layers.iter().enumerate().take(end).skip(i) {
                let sees = n_agg > 0 && layer.sees_aggregate_part();
                let (mut x, s, t, f) = if sees {
                    (&seen[..], None, all_times.as_deref(), all_lf.as_deref())
                } else {
                    (&seen[n_agg..], si, times, lf)
                };
                // A scoped layer sees the entries outside its scope as zero.
                if let (Some(view), Some(of)) = (scope, all_of.as_deref())
                    && let Some(mask) = &view.masks[li]
                {
                    let of = if sees { of } else { &of[n_agg..] };
                    masked.clear();
                    masked.extend(
                        x.iter()
                            .zip(of)
                            .map(|(&v, &c)| if mask[c] { v } else { 0.0 }),
                    );
                    x = &masked;
                }
                // A top-and-drop layer sees each event after its underlying
                // layer's annual limit is used up shifted up by the drop, so
                // its own attachment reads as the lower one.
                if let Some(d) = &layer.drop_down {
                    let u = self
                        .layers
                        .iter()
                        .position(|l| l.name == d.underlying)
                        .expect("Tower::inuring checks the underlying layer");
                    let under = &self.layers[u];
                    let mut ux: Vec<f64> = seen[n_agg..].to_vec();
                    if let (Some(view), Some(of)) = (scope, all_of.as_deref())
                        && let Some(mask) = &view.masks[u]
                    {
                        for (v, &c) in ux.iter_mut().zip(&of[n_agg..]) {
                            if !mask[c] {
                                *v = 0.0;
                            }
                        }
                    }
                    let used = under.cumulative_layer_loss(&ux, si, lf);
                    let shift = layer.attachment - d.attachment;
                    let offset = x.len() - used.len();
                    let mut exhausted = false;
                    dropped.clear();
                    dropped.extend_from_slice(x);
                    for (e, cum) in used.iter().enumerate() {
                        if exhausted {
                            dropped[offset + e] += shift;
                        }
                        exhausted |= *cum >= under.aggregate_limit;
                    }
                    x = &dropped;
                }
                let layer_ceded = layer.share * layer.layer_loss(x, s, f);
                ceded.push((layer_ceded, layer.reinstatement_premium_at(x, s, t, f)));
                if let Some(keys) = keys.as_deref_mut() {
                    keys.push(if layer.scales_before_annual_terms {
                        layer.share * layer.before_terms(x, s, f)
                    } else {
                        layer_ceded
                    });
                }
                if stage < last_stage || by_event.is_some() {
                    let mut layer_by_event = layer.ceded_by_event_si(x, s, f);
                    if !sees && n_agg > 0 {
                        layer_by_event.splice(0..0, std::iter::repeat_n(0.0, n_agg));
                    }
                    for (total, c) in stage_by_event.iter_mut().zip(&layer_by_event) {
                        *total += c;
                    }
                    if let Some(by_event) = by_event.as_deref_mut() {
                        by_event.push(layer_by_event);
                    }
                }
            }
            if stage < last_stage {
                for (x, c) in seen.iter_mut().zip(&stage_by_event) {
                    *x -= c;
                }
            }
            i = end;
        }
        ceded
    }

    /// Applies the tower to every simulated year.
    ///
    /// The result has dimensions `["kind", "layer"]` and components
    /// `(gross, ground_up)`, `(ceded, <layer name>)` for each layer, and
    /// `(net, retained)`, kept joint per year, then
    /// `(reinstatement_premium, <layer name>)` for each layer with paid
    /// reinstatements, `(swing_premium, <layer name>)` for each
    /// swing-rated layer, `(ceding_commission, <layer name>)` for each
    /// layer with a flat or sliding commission, `(profit_commission,
    /// <layer name>)` for each with a profit commission, `(brokerage,
    /// <layer name>)` for each with brokerage, `(<label>, <layer name>)`
    /// for each of a layer's [deductions](Layer::deduction), and
    /// `(loss_before_terms, <layer name>)` for each layer whose scales read
    /// it ([`Layer::scales_before_annual_terms`]), then for each deal
    /// ([`Tower::deal_scale`]) its `(ceding_commission, <deal>)`,
    /// `(brokerage, <deal>)` and `(loss_before_terms, <deal>)` where it has
    /// them. A fixed premium
    /// is the layer's `premium` and gets no component;
    /// [`deal_ratios`](Self::deal_ratios) turns a layer's components into
    /// ratios of it.
    /// `aggregate(&["kind"])` gives gross, total ceded and net (and the
    /// totals of the others); `net = gross - Σ ceded` in every year, so
    /// net is a loss, before any premium or commission.
    ///
    /// With labels on the events ([`EventSet::with_labels`]: a peril or a
    /// class) the dimensions are `["kind", "layer", <label dimensions>]`:
    /// gross, each layer's ceded loss and net come once for each
    /// combination of labels in [`EventSet::label_combinations`], each the
    /// sum over that combination's events (a layer's split by event, as
    /// [`Layer::ceded_by_event`] gives it, under the layer's
    /// [`LimitAllocation`]), and the other components, annual by nature,
    /// get `""` in every label dimension. `aggregate(&["kind", "layer"])`
    /// gives back the unlabelled result.
    ///
    /// Take one kind with `select("kind", ...)` before joining it to
    /// another distribution (the ceded part to a book, say): `aggregate(&[])`
    /// or summing over `kind` adds gross, ceded and net together, which is
    /// no quantity at all.
    ///
    /// ```
    /// use prospicio_aggregate::{Layer, Tower, simulate_events};
    /// use prospicio_prob::{Distribution, Lognormal, Poisson};
    ///
    /// let events = simulate_events(
    ///     &Poisson::new(2.0).unwrap(),
    ///     &Lognormal::from_mean_cv(3e6, 1.5).unwrap(),
    ///     10_000,
    ///     7,
    /// )
    /// .unwrap();
    /// let tower = Tower::new(vec![
    ///     Layer::xol("5x5", 5e6, 5e6).unwrap(),
    ///     Layer::xol("15x10", 15e6, 10e6).unwrap(),
    /// ])
    /// .unwrap();
    /// let result = tower.apply(&events).unwrap();
    /// let by_kind = result.aggregate(&["kind"]).unwrap();
    /// assert_eq!(by_kind.n_components(), 3); // gross, ceded, net
    /// ```
    pub fn apply(&self, events: &EventSet) -> Result<PredictiveDistribution> {
        if self.needs_sums_insured() && !events.has_sums_insured() {
            return Err(Error::Data(
                "a surplus treaty needs each event's sum insured: use \
                 EventSet::with_sums_insured or simulate from a risk profile"
                    .into(),
            ));
        }
        if self.needs_times() && !events.has_times() {
            return Err(Error::Data(
                "reinstatements pro rata as to time need each event's time: use \
                 EventSet::with_times, with_uniform_times or with_seasonal_times"
                    .into(),
            ));
        }
        self.refuse_lookback("a single year")?;
        // The events' seed, stream scheme and samplers (none for years
        // from elsewhere).
        let provenance = events.provenance("reinsurance_tower");
        if events.has_labels() {
            return self.apply_labeled(events, provenance);
        }
        self.refuse_scope("unlabelled events")?;
        self.apply_years(
            (0..events.n_sims()).map(|i| {
                (
                    events.aggregate_part(i).unwrap_or(&[]),
                    events.events(i),
                    events.sums_insured(i),
                    events.times(i),
                )
            }),
            events.n_sims(),
            provenance,
        )
    }

    /// Applies the tower to any predictive distribution, taking each
    /// simulation's total as one aggregate loss: an adverse development
    /// cover or loss portfolio transfer on a reserve bootstrap, a stop-loss
    /// or quota share on premium risk from a GLM or a collective model.
    /// Aggregate terms (stop-loss, quota share, annual deductible and
    /// limit) read as usual; an occurrence layer sees the year's total as
    /// one occurrence, so it acts as an aggregate excess of loss. To cover
    /// some components only, `aggregate` or select them first.
    ///
    /// The result has the components of [`apply`](Self::apply), and keeps
    /// the input's seed in its provenance.
    ///
    /// ```
    /// use prospicio_aggregate::{Layer, Tower};
    /// use prospicio_prob::{KeyValue, PredictiveDistribution, Provenance};
    ///
    /// // Reserves by origin in three simulations; an ADC of 50 xs 100.
    /// let reserve = PredictiveDistribution::from_draws(
    ///     vec!["origin".into()],
    ///     vec![vec![KeyValue::Int(2023)], vec![KeyValue::Int(2024)]],
    ///     vec![40.0, 50.0, 60.0, 70.0, 80.0, 100.0],
    ///     Provenance::new("odp_bootstrap"),
    /// )
    /// .unwrap();
    /// let adc = Tower::new(vec![Layer::stop_loss("adc", 50.0, 100.0).unwrap()]).unwrap();
    /// let result = adc.apply_aggregate(&reserve).unwrap();
    /// let ceded = result.marginal(&vec![KeyValue::from("ceded"), KeyValue::from("adc")]).unwrap();
    /// assert_eq!(prospicio_prob::Empirical::draws(&ceded), [0.0, 30.0, 50.0]);
    /// ```
    pub fn apply_aggregate(&self, pd: &PredictiveDistribution) -> Result<PredictiveDistribution> {
        self.refuse_scope("an aggregate loss")?;
        self.refuse_lookback("a single year")?;
        self.refuse_event_order("an annual total")?;
        if self.needs_sums_insured() {
            return Err(Error::Data(
                "a surplus treaty works risk by risk; an aggregate loss has no sum insured".into(),
            ));
        }
        if self.needs_times() {
            return Err(Error::Data(
                "reinstatements pro rata as to time need event times; an aggregate loss has none"
                    .into(),
            ));
        }
        let totals = prospicio_prob::Empirical::draws(pd.total()).to_vec();
        let source = pd.provenance();
        // The input's seed, stream scheme and samplers: the tower draws
        // nothing itself.
        let provenance = Provenance::new("reinsurance_tower")
            .version("prospicio-aggregate", env!("CARGO_PKG_VERSION"))
            .param("loss", format!("total of {}", source.model))
            .draws_from(source);
        self.apply_years(
            totals.chunks(1).map(|t| (&[][..], t, None, None)),
            totals.len(),
            provenance,
        )
    }

    /// [`apply`](Self::apply) on labelled events: gross, each layer's
    /// ceded loss and net by label combination, the other components with
    /// `""` labels.
    fn apply_labeled(&self, events: &EventSet, base: Provenance) -> Result<PredictiveDistribution> {
        let labels = events.label_set().expect("the events carry labels");
        let masks = self.scope_masks(events)?;
        let n_labels = labels.combinations.len();
        let n_layers = self.layers.len();
        let columns = Columns::new(self);
        let unlabeled = columns.keys(self);
        let blank = vec![KeyValue::from(""); labels.dims.len()];
        let with = |key: &[KeyValue], combination: &[String]| -> Vec<KeyValue> {
            key.iter()
                .cloned()
                .chain(combination.iter().map(|v| KeyValue::from(v.as_str())))
                .collect()
        };
        // gross, each layer's ceded and net by label, in that order, then
        // the annual components.
        let mut components = Vec::new();
        for key in &unlabeled[..n_layers + 2] {
            for combination in &labels.combinations {
                components.push(with(key, combination));
            }
        }
        for key in &unlabeled[n_layers + 2..] {
            components.push(key.iter().cloned().chain(blank.iter().cloned()).collect());
        }
        let n_sims = events.n_sims();
        let agg_of = events.aggregate_part_indices().unwrap_or(&[]);
        let mut draws = Vec::with_capacity(n_sims * components.len());
        let mut annual = Vec::new();
        for i in 0..n_sims {
            let losses = events.events(i);
            let aggregate = events.aggregate_part(i).unwrap_or(&[]);
            let (si, times) = (events.sums_insured(i), events.times(i));
            // Each entry's combination: the aggregate part's, then the
            // events'.
            let of: Vec<usize> = agg_of
                .iter()
                .chain(events.label_indices(i).expect("the events carry labels"))
                .copied()
                .collect();
            let mut by_event = Vec::with_capacity(n_layers);
            let mut keys = Vec::with_capacity(n_layers);
            let scope = masks.as_deref().map(|masks| ScopeView {
                aggregate: agg_of,
                events: &of[agg_of.len()..],
                masks,
            });
            let year = self.year_by_event(
                aggregate,
                losses,
                si,
                times,
                None,
                scope,
                Some(&mut by_event),
                Some(&mut keys),
            );
            let by_label = |values: &[f64]| -> Vec<f64> {
                let mut sums = vec![0.0; n_labels];
                for (&x, &c) in values.iter().zip(&of) {
                    sums[c] += x;
                }
                sums
            };
            let all: Vec<f64> = aggregate.iter().chain(losses).copied().collect();
            let gross = by_label(&all);
            let ceded: Vec<Vec<f64>> = by_event.iter().map(|c| by_label(c)).collect();
            draws.extend(&gross);
            for c in &ceded {
                draws.extend(c);
            }
            draws.extend((0..n_labels).map(|j| gross[j] - ceded.iter().map(|c| c[j]).sum::<f64>()));
            // The annual components, as an unlabelled year writes them.
            annual.clear();
            let mut carried = self.carries();
            self.push_terms(&columns, &mut annual, &year, &keys, 0, &mut carried);
            draws.extend(&annual);
        }
        let mut dims = vec!["kind".to_string(), "layer".to_string()];
        dims.extend(labels.dims.iter().cloned());
        PredictiveDistribution::from_draws(dims, components, draws, self.terms_provenance(base))
    }

    /// Each event's ceded loss to the named layers, summed over them, as an
    /// event set: the input to an occurrence retro on a reinsurer's own
    /// cessions (#590). Each layer's split by event is the one
    /// [`apply`](Self::apply)'s labels use ([`Layer::ceded_by_event`],
    /// after the layer's annual terms under its [`LimitAllocation`] and at
    /// its share, behind any earlier inuring stage), so in every year the
    /// set's losses add to the layers' ceded total. The events keep their
    /// times and labels; events that cede nothing are left out, sums
    /// insured are dropped (they belong to the original losses), and when
    /// a named layer [sees](Layer::sees_aggregate_part) the aggregate part
    /// the set keeps its ceded part as its own aggregate part. Several
    /// deals' sets join with [`EventSet::merge`], one occurrence each; to
    /// make one occurrence of a deal's losses from the same event, group
    /// the merged set by an hours clause ([`EventSet::grouped`]).
    ///
    /// ```
    /// use prospicio_aggregate::{EventSet, Layer, Tower};
    ///
    /// // 10 xs 5 with an annual limit of 15 on 20, 8 and 15: 10, 3 and 2.
    /// let events = EventSet::from_years(vec![vec![20.0, 8.0, 15.0], vec![4.0]], 0)
    ///     .unwrap()
    ///     .with_times(vec![0.1, 0.5, 0.9, 0.3])
    ///     .unwrap();
    /// let layer = Layer::xol("10x5", 10.0, 5.0).unwrap().aggregate_limit(15.0).unwrap();
    /// let ceded = Tower::new(vec![layer]).unwrap().ceded_events(&events, &["10x5"]).unwrap();
    /// assert_eq!(ceded.events(0), &[10.0, 3.0, 2.0]);
    /// assert_eq!(ceded.times(0), Some(&[0.1, 0.5, 0.9][..]));
    /// assert!(ceded.events(1).is_empty());
    /// ```
    pub fn ceded_events(&self, events: &EventSet, layers: &[&str]) -> Result<EventSet> {
        if layers.is_empty() {
            return Err(Error::Data("name at least one layer".into()));
        }
        let mut chosen = Vec::with_capacity(layers.len());
        for name in layers {
            let i = self
                .layers
                .iter()
                .position(|l| l.name == *name)
                .ok_or_else(|| Error::Data(format!("no layer named {name:?} in the tower")))?;
            if chosen.contains(&i) {
                return Err(Error::Data(format!("layer {name:?} is named twice")));
            }
            chosen.push(i);
        }
        if self.needs_sums_insured() && !events.has_sums_insured() {
            return Err(Error::Data(
                "a surplus layer needs each event's sum insured".into(),
            ));
        }
        if self.needs_times() && !events.has_times() {
            return Err(Error::Data(
                "reinstatements pro rata as to time need each event's time".into(),
            ));
        }
        let masks = self.scope_masks(events)?;
        let agg_of = events.aggregate_part_indices().unwrap_or(&[]);
        let mut by_year = Vec::with_capacity(events.n_sims());
        for i in 0..events.n_sims() {
            let aggregate = events.aggregate_part(i).unwrap_or(&[]);
            let losses = events.events(i);
            let scope = masks.as_deref().map(|masks| ScopeView {
                aggregate: agg_of,
                events: events.label_indices(i).expect("a scope needs labels"),
                masks,
            });
            let mut by_event = Vec::with_capacity(self.layers.len());
            self.year_by_event(
                aggregate,
                losses,
                events.sums_insured(i),
                events.times(i),
                None,
                scope,
                Some(&mut by_event),
                None,
            );
            let mut row = vec![0.0; aggregate.len() + losses.len()];
            for &j in &chosen {
                for (r, c) in row.iter_mut().zip(&by_event[j]) {
                    *r += c;
                }
            }
            by_year.push(row);
        }
        let sees = chosen.iter().any(|&j| self.layers[j].sees_aggregate_part());
        Ok(events.with_amounts(&by_year, sees))
    }

    /// Applies the tower to plain annual losses, one per simulation, each
    /// taken as one aggregate loss as in [`apply_aggregate`](Self::apply_aggregate),
    /// with no [`EventSet`] or [`PredictiveDistribution`] to build. A
    /// contract stated on the loss ratio is a layer on money amounts: a
    /// loss-ratio cap `c` on subject premium `P` is
    /// `Layer::stop_loss(name, c × P, 0)` with `premium_rate(1, P)` (the
    /// ceded loss ratio is then the capped one), and a sliding scale keyed
    /// on it is [`Layer::sliding_scale`]. To key the scales on the loss
    /// ratio before the cap, state the cap as an
    /// [`aggregate_limit`](Layer::aggregate_limit) and set
    /// [`Layer::scales_before_annual_terms`]. A deal of several sections
    /// is one layer per section, each scale reading its own section's
    /// loss ratio on its own premium, and the amounts add; adding the
    /// scales point by point is right only for one section.
    /// [`Commission::rate`] evaluates a scale on any loss ratio.
    ///
    /// Simulation `i` of the result is `losses[i]`: every component's draws
    /// keep the order of `losses`, so the ceded, net and terms of index `i`
    /// belong with whatever else was computed for simulation `i` (another
    /// line's losses, the cash flows of the same year). A reserving fit's
    /// `total_ultimate_draws` or `loss_ratio_draws` times the premium
    /// (`prospicio_reserving::UltimateDraws`) are in its simulation order;
    /// the losses must be finite and not negative.
    ///
    /// ```
    /// use prospicio_aggregate::{Layer, Tower};
    /// use prospicio_prob::{Empirical, KeyValue};
    ///
    /// // Losses capped at a 120% loss ratio on a premium of 100, with a
    /// // commission sliding from 30% at a 50% loss ratio to 10% at 90%.
    /// let cover = Layer::stop_loss("cover", 120.0, 0.0).unwrap()
    ///     .premium_rate(1.0, 100.0).unwrap()
    ///     .sliding_scale(vec![(0.3, 0.5), (0.1, 0.9)]).unwrap();
    /// let result = Tower::new(vec![cover]).unwrap().apply_annual(&[40.0, 70.0, 150.0]).unwrap();
    /// let at = |k: &str| {
    ///     let key: Vec<KeyValue> = vec![k.into(), "cover".into()];
    ///     Empirical::draws(&result.marginal(&key).unwrap()).to_vec()
    /// };
    /// assert_eq!(at("ceded"), [40.0, 70.0, 120.0]);
    /// let cc = at("ceding_commission");
    /// assert!((cc[0] - 30.0).abs() < 1e-12 && (cc[1] - 20.0).abs() < 1e-12 && (cc[2] - 10.0).abs() < 1e-12);
    /// ```
    pub fn apply_annual(&self, losses: &[f64]) -> Result<PredictiveDistribution> {
        self.refuse_scope("an annual loss")?;
        self.refuse_lookback("a single year")?;
        self.refuse_event_order("an annual total")?;
        if losses.is_empty() {
            return Err(Error::Data("needs at least one annual loss".into()));
        }
        if let Some(&bad) = losses.iter().find(|x| !(x.is_finite() && **x >= 0.0)) {
            return Err(Error::InvalidParameter {
                name: "losses",
                value: bad,
                reason: "must be finite and non-negative",
            });
        }
        if self.needs_sums_insured() {
            return Err(Error::Data(
                "a surplus treaty works risk by risk; an annual loss has no sum insured".into(),
            ));
        }
        if self.needs_times() {
            return Err(Error::Data(
                "reinstatements pro rata as to time need event times; an annual loss has none"
                    .into(),
            ));
        }
        let provenance = Provenance::new("reinsurance_tower")
            .version("prospicio-aggregate", env!("CARGO_PKG_VERSION"))
            .param("loss", "annual losses");
        self.apply_years(
            losses.chunks(1).map(|t| (&[][..], t, None, None)),
            losses.len(),
            provenance,
        )
    }

    /// Indices of the layers that satisfy `f`.
    pub(super) fn indices(&self, f: impl Fn(&Layer) -> bool) -> Vec<usize> {
        (0..self.layers.len())
            .filter(|&i| f(&self.layers[i]))
            .collect()
    }

    fn apply_years<'a>(
        &self,
        years: impl Iterator<Item = (&'a [f64], &'a [f64], Option<&'a [f64]>, Option<&'a [f64]>)>,
        n_sims: usize,
        base: Provenance,
    ) -> Result<PredictiveDistribution> {
        let columns = Columns::new(self);
        let mut draws = Vec::with_capacity(n_sims * columns.len(self));
        for (aggregate, losses, si, times) in years {
            // A single year starts with nothing carried.
            let mut carried = self.carries();
            self.push_year(
                &columns,
                &mut draws,
                aggregate,
                losses,
                si,
                times,
                None,
                None,
                0,
                &mut carried,
            );
        }
        let components = columns.keys(self);
        PredictiveDistribution::from_draws(
            vec!["kind".into(), "layer".into()],
            components,
            draws,
            self.terms_provenance(base),
        )
    }

    /// Appends one year's components (in the order of [`Columns::keys`])
    /// to `draws`: year `t` of a term, with its aggregate part
    /// `aggregate` (empty without one) and what earlier years carried in
    /// `carried` (two per layer: the profit commission's, then the sliding
    /// scale's).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn push_year(
        &self,
        columns: &Columns,
        draws: &mut Vec<f64>,
        aggregate: &[f64],
        losses: &[f64],
        si: Option<&[f64]>,
        times: Option<&[f64]>,
        lf: Option<&[f64]>,
        scope: Option<ScopeView<'_>>,
        t: usize,
        carried: &mut [Carried],
    ) {
        let gross: f64 = aggregate.iter().chain(losses).sum();
        draws.push(gross);
        let mut keys = Vec::with_capacity(self.layers.len());
        let year = self.year_by_event(
            aggregate,
            losses,
            si,
            times,
            lf,
            scope,
            None,
            Some(&mut keys),
        );
        let ceded_total: f64 = year.iter().map(|(c, _)| c).sum();
        draws.extend(year.iter().map(|(c, _)| c));
        draws.push(gross - ceded_total);
        self.push_terms(columns, draws, &year, &keys, t, carried);
    }

    /// Appends one year's components after net: reinstatement and swing
    /// premiums, commissions, brokerage, deductions and the loss before
    /// annual terms, from each layer's ceded loss and reinstatement
    /// premium in `year` and its scale key in `keys`.
    fn push_terms(
        &self,
        columns: &Columns,
        draws: &mut Vec<f64>,
        year: &[(f64, f64)],
        keys: &[f64],
        t: usize,
        carried: &mut [Carried],
    ) {
        draws.extend(
            year.iter()
                .zip(&columns.paid)
                .filter(|(_, p)| **p)
                .map(|((_, rp), _)| rp),
        );
        let ceded = |i: usize| year[i].0;
        draws.extend(
            columns
                .swing
                .iter()
                .map(|&i| self.layers[i].premium_for(ceded(i))),
        );
        for &i in &columns.commission {
            let c =
                self.layers[i].ceding_commission_in(ceded(i), keys[i], t, &mut carried[2 * i + 1]);
            draws.push(c);
        }
        for &i in &columns.profit {
            let c = self.layers[i].profit_commission_in(ceded(i), t, &mut carried[2 * i]);
            draws.push(c);
        }
        // Each layer's no-claims history after the deals' carries.
        let history = 2 * self.layers.len() + self.deals.len();
        for &i in &columns.bonus {
            let c = self.layers[i].no_claims_bonus_in(ceded(i), t, &mut carried[history + i]);
            draws.push(c);
        }
        draws.extend(
            columns
                .brokerage
                .iter()
                .map(|&i| self.layers[i].brokerage_keyed(ceded(i), keys[i])),
        );
        for &i in &columns.deductions {
            let l = &self.layers[i];
            draws.extend(l.deductions.iter().map(|d| l.deduction_for(d, ceded(i))));
        }
        draws.extend(columns.keyed.iter().map(|&i| keys[i]));
        // Each deal's scales on its sections' keys and premiums together,
        // its carry after the layers' two each.
        let n = self.layers.len();
        for (j, (deal, sections)) in self.deals.iter().zip(&columns.deals).enumerate() {
            let premium: f64 = sections.iter().map(|&i| self.layers[i].premium).sum();
            let key: f64 = sections.iter().map(|&i| keys[i]).sum();
            if let Some(scale) = &deal.commission {
                let carry = &mut carried[2 * n + j];
                draws.push(scale_in(
                    scale,
                    premium,
                    key,
                    t,
                    carry,
                    deal.commission_carry.as_ref(),
                ));
            }
            if deal.brokerage.is_some() {
                draws.push(deal.brokerage_at(premium, key));
            }
            if deal_keyed(self, sections) {
                draws.push(key);
            }
        }
    }

    /// What a simulated term carries, empty: two per layer (the profit
    /// commission's, then the sliding scale's), then one per deal, then
    /// each layer's no-claims history.
    pub(crate) fn carries(&self) -> Vec<Carried> {
        vec![Carried::default(); 3 * self.layers.len() + self.deals.len()]
    }

    /// The provenance of a result: `base` with every layer's terms.
    pub(crate) fn terms_provenance(&self, base: Provenance) -> Provenance {
        let mut provenance = base;
        for (l, stage) in self.layers.iter().zip(&self.stages) {
            let mut terms = format!(
                "{} xs {}, share {}, aad {}, aal {}, stage {stage}",
                l.limit, l.attachment, l.share, l.aggregate_deductible, l.aggregate_limit
            );
            if let Basis::Surplus { retention, lines } = l.basis {
                terms += &format!(", surplus retention {retention}, lines {lines}");
            }
            if !l.reinstatement_rates.is_empty() {
                terms += &format!(
                    ", premium {}, reinstatement rates {:?}",
                    l.premium, l.reinstatement_rates
                );
                if l.pro_rata_time {
                    terms += ", pro rata as to time";
                }
            }
            if l.allocation == LimitAllocation::ProRata {
                terms += ", annual terms pro rata across events";
            }
            if l.reinstatement_rates.is_empty() && l.premium > 0.0 {
                terms += &format!(", premium {}", l.premium);
            }
            if let Some(s) = &l.swing {
                terms += &format!(
                    ", swing basic {} lcm {} min {} max {} at 100%",
                    s.basic, s.lcm, s.minimum, s.maximum
                );
            }
            match &l.commission {
                Some(Commission::Flat(c)) => terms += &format!(", ceding commission {c}"),
                Some(Commission::SlidingScale(a)) => {
                    terms += &format!(", sliding scale (commission, loss ratio) {a:?}");
                }
                None => {}
            }
            if let Some(b) = &l.no_claims_bonus {
                terms += &format!(
                    ", no-claims bonus {} over {} years at {}",
                    b.share, b.years, b.threshold
                );
            }
            if let Some(pc) = &l.profit_commission {
                terms += &format!(", profit commission {} after {}", pc.share, pc.allowance);
            }
            match &l.brokerage {
                Some(Commission::Flat(b)) => terms += &format!(", brokerage {b}"),
                Some(Commission::SlidingScale(a)) => {
                    terms += &format!(", brokerage scale (rate, loss ratio) {a:?}");
                }
                None => {}
            }
            for d in &l.deductions {
                terms += &format!(
                    ", deduction {} {} on {}{}",
                    d.label,
                    d.value,
                    d.basis.name(),
                    if d.inside_nmbf { "" } else { " outside nmbf" }
                );
            }
            if l.scales_before_annual_terms {
                terms += ", scales before annual terms";
            }
            if l.brokerage_within_commission_range {
                terms += ", brokerage within the commission's range";
            }
            if let Some(cf) = &l.profit_commission_carry {
                terms += &format!(", profit commission deficits carried {}", carry_text(cf));
            }
            if let Some(cf) = &l.sliding_scale_carry {
                let what = match (cf.deficit, cf.credit) {
                    (true, true) => "deficits and credits",
                    (true, false) => "deficits",
                    _ => "credits",
                };
                terms += &format!(", sliding scale {what} carried {}", carry_text(cf));
            }
            provenance = provenance.param(format!("layer:{}", l.name), terms);
        }
        for d in &self.deals {
            provenance = provenance.param(format!("deal:{}", d.name), d.terms_text());
        }
        provenance
    }
}
