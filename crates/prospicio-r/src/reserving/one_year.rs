//! The one-year view of either bootstrap, `OneYearFit`.

use super::*;

/// The simulated one-year view of every segment of a triangle column:
/// each segment's bootstrap and opening ultimate, and the joint claims
/// development result, from the ODP bootstrap or Mack's.
#[extendr]
pub(crate) struct OneYearFit {
    pub(crate) inner: OneYear,
}

/// A one-year view from either bootstrap.
pub(crate) enum OneYear {
    Odp(OneYearFits),
    Mack(OneYearFits<MackBootstrapSegment>),
}

/// `$body` on whichever one-year view `$fit` holds, bound to `$f`.
macro_rules! each_one_year {
    ($fit:expr, $f:ident => $body:expr) => {
        match $fit {
            OneYear::Odp($f) => $body,
            OneYear::Mack($f) => $body,
        }
    };
}

#[extendr]
impl OneYearFit {
    /// "odp" or "mack".
    fn model(&self) -> &'static str {
        match self.inner {
            OneYear::Odp(_) => "odp",
            OneYear::Mack(_) => "mack",
        }
    }

    /// The bootstrap's chain ladder: volume-weighted for the ODP, Mack's
    /// averaging for Mack's.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: each_one_year!(&self.inner, f => f
                .segments
                .map(|s| s.bootstrap.chain_ladder().clone())),
        }
    }

    /// Mack's model behind `mack_one_year()`.
    fn mack(&self) -> Result<MackFit> {
        match &self.inner {
            OneYear::Mack(f) => Ok(MackFit {
                inner: f.segments.map(|s| s.bootstrap.mack.clone()),
            }),
            OneYear::Odp(_) => Err(Error::Other(
                "mack is Mack's bootstrap's model; this one-year view is the ODP's".into(),
            )),
        }
    }

    fn opening_ultimate(&self) -> Vec<f64> {
        each_one_year!(&self.inner, f => by_origin(&f.segments, |s| s.opening_ultimate.clone()))
    }

    fn opening_reserve(&self) -> Vec<f64> {
        each_one_year!(&self.inner, f => by_origin(&f.segments, |s| s.opening_reserve.clone()))
    }

    fn scale(&self) -> Result<f64> {
        match &self.inner {
            OneYear::Odp(f) => Ok(single(&f.segments, "scale", "totals_frame()")?
                .bootstrap
                .scale),
            OneYear::Mack(_) => Err(Error::Other(
                "scale is the ODP bootstrap's; this one-year view is Mack's".into(),
            )),
        }
    }

    fn cdr(&self) -> PredictiveDistribution {
        PredictiveDistribution {
            inner: each_one_year!(&self.inner, f => f.cdr.clone()),
        }
    }

    fn long_table(&self) -> List {
        fit_table(each_one_year!(&self.inner, f => f.to_long()))
    }

    fn totals_table(&self) -> List {
        fit_table(each_one_year!(&self.inner, f => f.totals()))
    }

    fn development_table(&self) -> List {
        fit_table(each_one_year!(&self.inner, f => f.segments.development_table()))
    }

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        let keys = choice(&keys, &values)?;
        let inner = match &self.inner {
            OneYear::Odp(f) => OneYear::Odp(f.segment(&keys).map_err(to_r)?),
            OneYear::Mack(f) => OneYear::Mack(f.segment(&keys).map_err(to_r)?),
        };
        Ok(Self { inner })
    }
}

extendr_module! {
    mod one_year;
    impl OneYearFit;
}
