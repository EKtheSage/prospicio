//! Latest diagonals, `LatestDiagonals`.

use super::*;

fn past_pattern_end(name: &str) -> Result<PastPatternEnd> {
    PastPatternEnd::from_name(name)
        .ok_or_else(|| crate::unknown_name("past_pattern_end", name, PastPatternEnd::NAMES))
}

fn past_pattern_end_name(rule: PastPatternEnd) -> &'static str {
    rule.name()
}

fn grain_name(grain: Grain) -> &'static str {
    match grain {
        Grain::Month => "M",
        Grain::Quarter => "Q",
        Grain::Semester => "S",
        Grain::Year => "Y",
    }
}

/// A segment label as R writes it: its key values joined by `" / "`, or
/// `"Total"` without keys.
fn label_from_r(text: &str, n_keys: usize) -> Label {
    if n_keys == 0 && text == "Total" {
        Label::default()
    } else {
        Label::new(text.split(" / ").map(str::to_owned).collect::<Vec<_>>())
    }
}

/// A latest diagonal in one or several segments: each origin's latest
/// cumulative value at its own age, as a bordereau holds it
/// (`docs/design/reserving-v02.md`, decisions 15 to 19). R's `latest_valuation()`
/// and `as_latest_valuation()`.
#[extendr]
pub(crate) struct LatestDiagonals {
    inner: LatestDiagonalsInner,
}

#[extendr]
impl LatestDiagonals {
    /// From a long table. `keys` holds the key columns named `key_names`,
    /// column by column. The ages are `ages`, or with `by_valuation` the
    /// valuation months, one for every row or one per row.
    #[allow(clippy::too_many_arguments)]
    fn from_long(
        key_names: Vec<String>,
        keys: Vec<String>,
        origin_year: &[i32],
        origin_month: &[i32],
        ages: &[f64],
        by_valuation: bool,
        valuation_year: &[i32],
        valuation_month: &[i32],
        latest: &[f64],
        exposure: Nullable<Vec<f64>>,
        origin_grain: &str,
        past_pattern_end: &str,
    ) -> Result<Self> {
        let origin = months(origin_year, origin_month, "origin")?;
        let n = origin.len();
        if keys.len() != n * key_names.len() {
            return Err(Error::Other(format!(
                "keys has {} entries, expected {n} rows times {} keys",
                keys.len(),
                key_names.len()
            )));
        }
        let key_values: Vec<Vec<&str>> = (0..key_names.len())
            .map(|k| {
                keys[k * n..(k + 1) * n]
                    .iter()
                    .map(String::as_str)
                    .collect()
            })
            .collect();
        let key_columns: Vec<(&str, &[&str])> = key_names
            .iter()
            .zip(&key_values)
            .map(|(name, v)| (name.as_str(), v.as_slice()))
            .collect();
        let rule = self::past_pattern_end(past_pattern_end)?;
        let valuations;
        let lags: Vec<Lag>;
        let age = if by_valuation {
            valuations = months(valuation_year, valuation_month, "valuation")?;
            match valuations.as_slice() {
                [one] => DiagonalAge::Valuation(*one),
                _ => DiagonalAge::Valuations(&valuations),
            }
        } else {
            lags = ages.iter().map(|&a| age(a)).collect::<Result<_>>()?;
            DiagonalAge::Ages(&lags)
        };
        let exposure = match exposure {
            Nullable::NotNull(e) => Some(e),
            Nullable::Null => None,
        };
        let inner = LatestDiagonalsInner::from_long(&LongDiagonal {
            keys: &key_columns,
            origin: &origin,
            age,
            latest,
            exposure: exposure.as_deref(),
            origin_grain: grain(origin_grain, "origin_grain")?,
        })
        .map_err(to_r)?
        .with_past_pattern_end(rule);
        Ok(Self { inner })
    }

    /// The latest diagonal of a triangle column in every segment, with the
    /// `exposure` column's latest values when given.
    fn from_triangle(
        triangle: &Triangle,
        column: &str,
        exposure: Nullable<String>,
        past_pattern_end: &str,
    ) -> Result<Self> {
        let rule = self::past_pattern_end(past_pattern_end)?;
        let exposure = match &exposure {
            Nullable::NotNull(e) => Some(e.as_str()),
            Nullable::Null => None,
        };
        let inner = LatestDiagonalsInner::from_triangle(&triangle.inner, column, exposure)
            .map_err(to_r)?
            .with_past_pattern_end(rule);
        Ok(Self { inner })
    }

    /// The same rows with `exposure`, in the order of `origins()`.
    fn with_exposure(&self, exposure: &[f64]) -> Result<Self> {
        let inner = self.inner.clone().with_exposure(exposure).map_err(to_r)?;
        Ok(Self { inner })
    }

    fn with_past_pattern_end(&self, rule: &str) -> Result<Self> {
        let rule = self::past_pattern_end(rule)?;
        Ok(Self {
            inner: self.inner.clone().with_past_pattern_end(rule),
        })
    }

    fn n_segments(&self) -> i32 {
        self.inner.len() as i32
    }

    fn keys(&self) -> Vec<String> {
        self.inner.key_names().to_vec()
    }

    /// The key columns, one row per segment.
    fn index(&self) -> List {
        let names = self.inner.key_names().to_vec();
        let columns: Vec<Vec<String>> = (0..names.len())
            .map(|k| {
                self.inner
                    .labels()
                    .iter()
                    .map(|l| l.parts()[k].clone())
                    .collect()
            })
            .collect();
        List::from_names_and_values(names, columns).expect("one name per key")
    }

    /// The key columns, one row per origin.
    fn row_keys(&self) -> List {
        let names = self.inner.key_names().to_vec();
        let columns: Vec<Vec<String>> = (0..names.len())
            .map(|k| {
                self.inner
                    .iter()
                    .flat_map(|(l, d)| std::iter::repeat_n(l.parts()[k].clone(), d.origins().len()))
                    .collect()
            })
            .collect();
        List::from_names_and_values(names, columns).expect("one name per key")
    }

    /// The segment label (parts joined by " / ") of each row.
    fn row_segments(&self) -> Vec<String> {
        self.inner
            .iter()
            .flat_map(|(l, d)| std::iter::repeat_n(l.to_string(), d.origins().len()))
            .collect()
    }

    fn origin_grain(&self) -> &'static str {
        grain_name(self.inner.origin_grain())
    }

    fn origins(&self) -> Vec<String> {
        let grain = self.inner.origin_grain();
        self.inner
            .diagonals()
            .iter()
            .flat_map(|d| d.origins().iter())
            .map(|&m| Period::containing(m, grain).to_string())
            .collect()
    }

    fn ages(&self) -> Vec<i32> {
        self.inner
            .diagonals()
            .iter()
            .flat_map(|d| d.ages().iter().map(|&a| a as i32))
            .collect()
    }

    fn latest(&self) -> Vec<f64> {
        self.inner
            .diagonals()
            .iter()
            .flat_map(|d| d.latest().iter().copied())
            .collect()
    }

    /// The exposure of each row, or `NULL` when none was given.
    fn exposure(&self) -> Nullable<Vec<f64>> {
        let mut out = Vec::new();
        for d in self.inner.diagonals() {
            match d.exposure() {
                Some(e) => out.extend_from_slice(e),
                None => return Nullable::Null,
            }
        }
        Nullable::NotNull(out)
    }

    fn past_pattern_end(&self) -> &'static str {
        past_pattern_end_name(self.inner.diagonals()[0].past_pattern_end())
    }

    /// The rows of the one segment whose `keys` have `values`.
    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        let inner = self.inner.segment(&choice(&keys, &values)?).map_err(to_r)?;
        Ok(Self { inner })
    }

    /// The chain ladder on each segment; with `development` (a
    /// `ChainLadderFit`) each segment's development is that fit's segment
    /// that `segment_map` names (a named character vector, target to
    /// source, or `NULL` to match by label).
    #[allow(clippy::too_many_arguments)]
    fn chain_ladder(
        &self,
        average: &str,
        sigma_interpolation: &str,
        excl: Robj,
        tail: Robj,
        ldf: Robj,
        development: Robj,
        segment_map: Robj,
    ) -> Result<ChainLadderFit> {
        let m = pattern(
            average,
            sigma_interpolation,
            &excl,
            tail,
            &ldf,
            self.grain(),
        )?;
        let inner = match self.developments(&development, &segment_map)? {
            None => m.fit_diagonal_segments(&self.inner),
            Some(s) => m.fit_diagonal_segments_from(&self.inner, &s),
        }
        .map_err(to_r)?;
        Ok(ChainLadderFit { inner })
    }

    #[allow(clippy::too_many_arguments)]
    fn expected_loss(
        &self,
        apriori: Robj,
        average: &str,
        sigma_interpolation: &str,
        excl: Robj,
        tail: Robj,
        ldf: Robj,
        development: Robj,
        segment_map: Robj,
    ) -> Result<ExpectedLossFit> {
        let m = ExpectedLoss {
            apriori: apriori_arg(&apriori)?,
            chain_ladder: pattern(
                average,
                sigma_interpolation,
                &excl,
                tail,
                &ldf,
                self.grain(),
            )?,
        };
        let inner = match self.developments(&development, &segment_map)? {
            None => m.fit_diagonal_segments(&self.inner),
            Some(s) => m.fit_diagonal_segments_from(&self.inner, &s),
        }
        .map_err(to_r)?;
        Ok(ExpectedLossFit { inner })
    }

    #[allow(clippy::too_many_arguments)]
    fn bornhuetter_ferguson(
        &self,
        apriori: Robj,
        average: &str,
        sigma_interpolation: &str,
        excl: Robj,
        tail: Robj,
        ldf: Robj,
        development: Robj,
        segment_map: Robj,
    ) -> Result<ExpectedLossFit> {
        let m = BornhuetterFerguson {
            apriori: apriori_arg(&apriori)?,
            chain_ladder: pattern(
                average,
                sigma_interpolation,
                &excl,
                tail,
                &ldf,
                self.grain(),
            )?,
        };
        let inner = match self.developments(&development, &segment_map)? {
            None => m.fit_diagonal_segments(&self.inner),
            Some(s) => m.fit_diagonal_segments_from(&self.inner, &s),
        }
        .map_err(to_r)?;
        Ok(ExpectedLossFit { inner })
    }

    #[allow(clippy::too_many_arguments)]
    fn benktander(
        &self,
        apriori: Robj,
        n_iters: f64,
        average: &str,
        sigma_interpolation: &str,
        excl: Robj,
        tail: Robj,
        ldf: Robj,
        development: Robj,
        segment_map: Robj,
    ) -> Result<ExpectedLossFit> {
        let n_iters = usize::try_from(whole(n_iters, "n_iters")?)
            .map_err(|_| Error::Other(format!("n_iters {n_iters} is too large")))?;
        let m = Benktander {
            apriori: apriori_arg(&apriori)?,
            n_iters,
            chain_ladder: pattern(
                average,
                sigma_interpolation,
                &excl,
                tail,
                &ldf,
                self.grain(),
            )?,
        };
        let inner = match self.developments(&development, &segment_map)? {
            None => m.fit_diagonal_segments(&self.inner),
            Some(s) => m.fit_diagonal_segments_from(&self.inner, &s),
        }
        .map_err(to_r)?;
        Ok(ExpectedLossFit { inner })
    }

    #[allow(clippy::too_many_arguments)]
    fn cape_cod(
        &self,
        trend: f64,
        decay: f64,
        average: &str,
        sigma_interpolation: &str,
        excl: Robj,
        tail: Robj,
        ldf: Robj,
        development: Robj,
        segment_map: Robj,
    ) -> Result<CapeCodFit> {
        let m = CapeCod {
            trend,
            decay,
            chain_ladder: pattern(
                average,
                sigma_interpolation,
                &excl,
                tail,
                &ldf,
                self.grain(),
            )?,
        };
        let inner = match self.developments(&development, &segment_map)? {
            None => m.fit_diagonal_segments(&self.inner),
            Some(s) => m.fit_diagonal_segments_from(&self.inner, &s),
        }
        .map_err(to_r)?;
        Ok(CapeCodFit { inner })
    }
}

impl LatestDiagonals {
    fn grain(&self) -> Grain {
        self.inner.origin_grain()
    }

    /// The development of each segment from the `development` and
    /// `segment_map` arguments: `None` without `development`.
    fn developments(
        &self,
        development: &Robj,
        segment_map: &Robj,
    ) -> Result<Option<SegmentFits<ChainLadder>>> {
        if development.is_null() {
            if !segment_map.is_null() {
                return Err(Error::Other(
                    "segment_map is given without development".into(),
                ));
            }
            return Ok(None);
        }
        let source = <&ChainLadderFit>::try_from(development).map_err(|_| {
            Error::Other(
                "development must be a chain_ladder_fit, or a fit with @chain_ladder".into(),
            )
        })?;
        let map = if segment_map.is_null() {
            SegmentMap::ByLabel
        } else {
            let sources = segment_map.as_string_vector().ok_or_else(|| {
                Error::Other(
                    "segment_map must be a named character vector: names are segments of the \
                     latest valuation, values segments of the development"
                        .into(),
                )
            })?;
            let targets = segment_map
                .names()
                .map(|n| n.map(str::to_owned).collect::<Vec<_>>())
                .filter(|n| n.len() == sources.len())
                .ok_or_else(|| {
                    Error::Other(
                        "segment_map needs a name (a segment of the latest valuation) on every value"
                            .into(),
                    )
                })?;
            let (n_target, n_source) = (self.inner.key_names().len(), source.inner.key_names.len());
            SegmentMap::Pairs(
                targets
                    .iter()
                    .zip(&sources)
                    .map(|(t, s)| (label_from_r(t, n_target), label_from_r(s, n_source)))
                    .collect(),
            )
        };
        self.inner
            .developments(&source.inner, &map)
            .map(Some)
            .map_err(to_r)
    }
}

extendr_module! {
    mod latest_diagonal;
    impl LatestDiagonals;
}
