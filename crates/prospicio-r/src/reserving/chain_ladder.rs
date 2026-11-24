//! The chain ladder's fit, `ChainLadderFit`, and the helpers of segmented fits.

use super::*;

/// The one fit of a single-segment result, or an error naming what to use
/// instead: the long table `instead` (if any) or `segment()`.
pub(crate) fn single<'a, T>(fits: &'a SegmentFits<T>, field: &str, instead: &str) -> Result<&'a T> {
    match fits.fits.as_slice() {
        [one] => Ok(one),
        _ => {
            let table = if instead.is_empty() {
                String::new()
            } else {
                format!("{instead} or ")
            };
            Err(Error::Other(format!(
                "{field} needs a single-segment fit, and this one has {} segments; \
                 use {table}segment()",
                fits.len()
            )))
        }
    }
}

/// Per-origin values of every segment, in the order of the long rows.
pub(crate) fn by_origin<T>(fits: &SegmentFits<T>, f: impl Fn(&T) -> Vec<f64>) -> Vec<f64> {
    fits.fits.iter().flat_map(f).collect()
}

fn origin_labels<T: ReserveFit>(fits: &SegmentFits<T>) -> Vec<String> {
    fits.fits
        .iter()
        .flat_map(|f| f.chain_ladder().origins.iter().map(ToString::to_string))
        .collect()
}

/// The segment label (parts joined by " / ") of each per-origin value.
pub(crate) fn row_segments<T: ReserveFit>(fits: &SegmentFits<T>) -> Vec<String> {
    fits.iter()
        .flat_map(|(l, f)| std::iter::repeat_n(l.to_string(), f.chain_ladder().origins.len()))
        .collect()
}

pub(crate) fn ages<T: ReserveFit>(fits: &SegmentFits<T>) -> Vec<i32> {
    let dev = &fits.fits[0].chain_ladder().development.development;
    dev.iter().map(|&a| a as i32).collect()
}

/// The keys choosing one segment as `(key, value)` pairs.
pub(crate) fn choice<'a>(
    keys: &'a [String],
    values: &'a [String],
) -> Result<Vec<(&'a str, &'a str)>> {
    if keys.len() != values.len() {
        return Err(Error::Other("one value is needed per key".into()));
    }
    Ok(keys
        .iter()
        .zip(values)
        .map(|(k, v)| (k.as_str(), v.as_str()))
        .collect())
}

pub(crate) fn pick<T: Clone>(
    fits: &SegmentFits<T>,
    keys: Vec<String>,
    values: Vec<String>,
) -> Result<SegmentFits<T>> {
    fits.segment(&choice(&keys, &values)?).map_err(to_r)
}

/// A long result table: the key columns (a named list), `origin` (labels)
/// or `development` (ages), `NULL` where the table has none, and the value
/// columns with their names.
pub(crate) fn fit_table(table: FitTable) -> List {
    let origin: Robj = match table.origin {
        Some(o) => o.iter().map(ToString::to_string).collect::<Vec<_>>().into(),
        None => ().into(),
    };
    let development: Robj = match table.age {
        Some(a) => a.iter().map(|&a| a as i32).collect::<Vec<_>>().into(),
        None => ().into(),
    };
    let names: Vec<String> = table.values.iter().map(|(n, _)| n.clone()).collect();
    let values = List::from_values(table.values.into_iter().map(|(_, v)| v));
    list!(
        keys = key_columns(table.keys),
        origin = origin,
        development = development,
        names = names,
        values = values
    )
}

/// A chain ladder fitted to every segment of a triangle column. Per-origin
/// vectors run over the origins of each segment in turn.
#[extendr]
pub(crate) struct ChainLadderFit {
    pub(crate) inner: SegmentFits<ChainLadderInner>,
}

#[extendr]
impl ChainLadderFit {
    fn n_segments(&self) -> i32 {
        self.inner.len() as i32
    }

    fn keys(&self) -> Vec<String> {
        self.inner.key_names.clone()
    }

    fn row_segments(&self) -> Vec<String> {
        row_segments(&self.inner)
    }

    fn origins(&self) -> Vec<String> {
        origin_labels(&self.inner)
    }

    fn development(&self) -> Vec<i32> {
        ages(&self.inner)
    }

    /// The selected factors within the triangle: the estimated ones,
    /// replaced by the tail's from its attachment age.
    fn ldf(&self) -> Result<Vec<f64>> {
        let f = single(&self.inner, "ldf", "development_frame()")?;
        Ok(f.ldf().to_vec())
    }

    fn sigma(&self) -> Result<Vec<f64>> {
        let f = single(&self.inner, "sigma", "development_frame()")?;
        Ok(f.development.sigma.clone())
    }

    /// Standard errors of the factors used before the tail: `1 - weight`
    /// times the estimate's.
    fn std_err(&self) -> Result<Vec<f64>> {
        let f = single(&self.inner, "std_err", "development_frame()")?;
        Ok(f.development.blended_std_err())
    }

    fn estimated_std_err(&self) -> Result<Vec<f64>> {
        let f = single(&self.inner, "estimated_std_err", "development_frame()")?;
        Ok(f.development.std_err.clone())
    }

    /// The selected pattern's factor of each link; NaN where none.
    fn selected_ldf(&self) -> Result<Vec<f64>> {
        let f = single(&self.inner, "selected_ldf", "development_frame()")?;
        Ok(f.development
            .selected
            .iter()
            .map(|s| s.unwrap_or(f64::NAN))
            .collect())
    }

    fn ldf_weight(&self) -> Result<Vec<f64>> {
        let f = single(&self.inner, "ldf_weight", "development_frame()")?;
        Ok(f.development.weight.clone())
    }

    fn blended_ldf(&self) -> Result<Vec<f64>> {
        let f = single(&self.inner, "blended_ldf", "development_frame()")?;
        Ok(f.development.blended_ldf())
    }

    /// The rescale of a cumulative pattern; NaN for an age-to-age pattern
    /// or none.
    fn pattern_rescale(&self) -> Result<f64> {
        let f = single(&self.inner, "pattern_rescale", "totals_frame()")?;
        Ok(f.pattern_rescale.unwrap_or(f64::NAN))
    }

    /// The tail a cumulative pattern implies; NaN where it implies none.
    fn pattern_tail(&self) -> Result<f64> {
        let f = single(&self.inner, "pattern_tail", "totals_frame()")?;
        Ok(f.pattern_tail.unwrap_or(f64::NAN))
    }

    fn alpha(&self) -> f64 {
        self.inner.fits[0].development.alpha
    }

    /// The link ratios each factor averaged, after `n_periods`,
    /// `drop_high` and `drop_low`.
    fn n_links(&self) -> Result<Vec<i32>> {
        let f = single(&self.inner, "n_links", "")?;
        Ok(f.development.n_links.iter().map(|&n| n as i32).collect())
    }

    /// Whether `n_periods` or the drops left out a link ratio at some age;
    /// needs a single-segment fit.
    fn excluded(&self) -> Result<bool> {
        let f = single(&self.inner, "excluded", "")?;
        Ok(f.development.excluded)
    }

    /// The factors as estimated, before the tail replaced any.
    fn estimated_ldf(&self) -> Result<Vec<f64>> {
        let f = single(&self.inner, "estimated_ldf", "")?;
        Ok(f.development.ldf.clone())
    }

    /// Age from which `ldf` holds the tail's factors; the oldest age when
    /// the tail replaced none.
    fn tail_attachment_age(&self) -> Result<i32> {
        let f = single(&self.inner, "tail_attachment_age", "")?;
        f.development
            .development
            .get(f.tail.attachment)
            .map(|&a| a as i32)
            .ok_or_else(|| Error::Other("the fit has no development ages".into()))
    }

    fn tail(&self) -> Result<f64> {
        Ok(single(&self.inner, "tail", "totals_frame()")?.tail.factor)
    }

    /// Factors past the oldest age, which multiply to the tail.
    fn tail_ldf(&self) -> Result<Vec<f64>> {
        let f = single(&self.inner, "tail_ldf", "")?;
        Ok(f.tail.ldf[f.development.ldf.len()..].to_vec())
    }

    fn tail_sigma(&self) -> Result<f64> {
        Ok(single(&self.inner, "tail_sigma", "totals_frame()")?
            .tail
            .sigma)
    }

    fn tail_std_err(&self) -> Result<f64> {
        Ok(single(&self.inner, "tail_std_err", "totals_frame()")?
            .tail
            .std_err)
    }

    fn cdf(&self) -> Result<Vec<f64>> {
        Ok(single(&self.inner, "cdf", "development_frame()")?
            .cdf
            .clone())
    }

    fn latest(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.latest.clone())
    }

    fn ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.ultimate.clone())
    }

    fn reserve(&self) -> Vec<f64> {
        by_origin(&self.inner, ChainLadderInner::reserves)
    }

    fn total_ultimate(&self) -> f64 {
        self.inner.total_ultimate()
    }

    fn total_reserve(&self) -> f64 {
        self.inner.total_reserve()
    }

    fn long_table(&self) -> List {
        fit_table(self.inner.to_long())
    }

    fn totals_table(&self) -> List {
        fit_table(self.inner.totals())
    }

    fn development_table(&self) -> List {
        fit_table(self.inner.development_table())
    }

    /// The fit of the one segment whose `keys` have `values`.
    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys, values)?,
        })
    }

    /// The fit frozen as a chain ladder (`ChainLadderFit::to_chain_ladder`):
    /// a list of `ages` and `values`, its age-to-age factors, and `tail`,
    /// a constant factor. Needs a single segment.
    fn to_chain_ladder(&self) -> Result<List> {
        let cl = single(&self.inner, "to_chain_ladder", "")?.to_chain_ladder();
        let pattern = cl
            .development
            .pattern
            .expect("a frozen fit selects its factors");
        let (ages, values): (Vec<f64>, Vec<f64>) = pattern
            .values
            .iter()
            .map(|&(a, v)| (f64::from(a), v))
            .unzip();
        let tail = match cl.tail {
            Tail::Constant(c) => c.factor,
            _ => unreachable!("a frozen fit's tail is a constant"),
        };
        Ok(list!(ages = ages, values = values, tail = tail))
    }
}

extendr_module! {
    mod chain_ladder;
    impl ChainLadderFit;
}
