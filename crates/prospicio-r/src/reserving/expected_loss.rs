//! The expected-loss methods' fits, `ExpectedLossFit` and `CapeCodFit`.

use super::*;

/// An expected-loss method (expected loss, Bornhuetter-Ferguson or
/// Benktander) fitted to every segment of a triangle column, each with its
/// exposure from another column of the same triangle.
#[extendr]
pub(crate) struct ExpectedLossFit {
    pub(crate) inner: SegmentFits<ExpectedLossInner>,
}

#[extendr]
impl ExpectedLossFit {
    /// The chain ladder that gives the development pattern.
    fn chain_ladder(&self) -> ChainLadderFit {
        ChainLadderFit {
            inner: self.inner.map(|f| f.chain_ladder.clone()),
        }
    }

    fn exposure(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.exposure.clone())
    }

    fn apriori(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.apriori.clone())
    }

    fn ultimate(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.ultimate.clone())
    }

    fn reserve(&self) -> Vec<f64> {
        by_origin(&self.inner, ExpectedLossInner::reserves)
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

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys, values)?,
        })
    }
}

/// A Cape Cod fitted to every segment of a triangle column: the
/// Bornhuetter-Ferguson fit on the detrended apriori, and the trended
/// apriori it came from.
#[extendr]
pub(crate) struct CapeCodFit {
    pub(crate) inner: SegmentFits<CapeCodInner>,
}

#[extendr]
impl CapeCodFit {
    /// The Bornhuetter-Ferguson fit on the detrended apriori.
    fn expected_loss(&self) -> ExpectedLossFit {
        ExpectedLossFit {
            inner: self.inner.map(|f| f.expected_loss.clone()),
        }
    }

    fn trended_apriori(&self) -> Vec<f64> {
        by_origin(&self.inner, |f| f.trended_apriori.clone())
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

    fn segment(&self, keys: Vec<String>, values: Vec<String>) -> Result<Self> {
        Ok(Self {
            inner: pick(&self.inner, keys, values)?,
        })
    }
}

extendr_module! {
    mod expected_loss;
    impl ExpectedLossFit;
    impl CapeCodFit;
}
