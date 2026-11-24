//! The tails, `ReservingTail`.

use super::*;

/// How development past the oldest age is estimated: a constant factor, a
/// curve fitted to the factors, Bondy's rule or R ChainLadder's log-linear
/// rule. An age that is not given is `NULL`.
#[extendr]
pub(crate) struct ReservingTail {
    inner: Tail,
}

pub(crate) fn tail_arg(tail: &Robj) -> Result<Tail> {
    <&ReservingTail>::try_from(tail)
        .map(|t| t.inner)
        .map_err(|_| Error::Other("tail must be a tail estimator".into()))
}

pub(crate) fn curve_name(c: CurveShape) -> &'static str {
    c.name()
}

/// A curve shape by its name in the bindings.
pub(crate) fn curve_shape(name: &str, what: &str) -> Result<CurveShape> {
    CurveShape::from_name(name).ok_or_else(|| crate::unknown_name(what, name, CurveShape::NAMES))
}

#[extendr]
impl ReservingTail {
    fn constant(factor: f64, decay: f64, attachment_age: Nullable<f64>) -> Result<Self> {
        Ok(Self {
            inner: Tail::Constant(TailConstant {
                factor,
                decay,
                attachment_age: optional_age(attachment_age, "attachment_age")?,
            }),
        })
    }

    /// `curve` is a shape's name (`curve_shape`); the fit period runs from
    /// `fit_from` (inclusive) to `fit_to` (exclusive).
    fn curve(
        curve: &str,
        fit_from: Nullable<f64>,
        fit_to: Nullable<f64>,
        extrap_periods: f64,
        attachment_age: Nullable<f64>,
    ) -> Result<Self> {
        let curve = curve_shape(curve, "curve")?;
        let extrap_periods = usize::try_from(whole(extrap_periods, "extrap_periods")?)
            .map_err(|_| Error::Other("extrap_periods is too large".into()))?;
        Ok(Self {
            inner: Tail::Curve(TailCurve {
                curve,
                fit_period: (
                    optional_age(fit_from, "fit_period")?,
                    optional_age(fit_to, "fit_period")?,
                ),
                extrap_periods,
                attachment_age: optional_age(attachment_age, "attachment_age")?,
            }),
        })
    }

    fn bondy(earliest_age: Nullable<f64>, attachment_age: Nullable<f64>) -> Result<Self> {
        Ok(Self {
            inner: Tail::Bondy(TailBondy {
                earliest_age: optional_age(earliest_age, "earliest_age")?,
                attachment_age: optional_age(attachment_age, "attachment_age")?,
            }),
        })
    }

    fn log_linear() -> Self {
        Self {
            inner: Tail::LogLinear,
        }
    }

    /// "constant", "curve", "bondy" or "log_linear".
    fn kind(&self) -> &'static str {
        match self.inner {
            Tail::Constant(_) => "constant",
            Tail::Curve(_) => "curve",
            Tail::Bondy(_) => "bondy",
            Tail::LogLinear => "log_linear",
        }
    }

    /// The parameters of this kind as a named list, with `NULL` for an age
    /// not given; the curve's fit period is `fit_from` and `fit_to`.
    fn params(&self) -> List {
        let age = |a: Option<Lag>| -> Robj {
            match a {
                Some(a) => (a as i32).into(),
                None => ().into(),
            }
        };
        match self.inner {
            Tail::Constant(t) => list!(
                factor = t.factor,
                decay = t.decay,
                attachment_age = age(t.attachment_age)
            ),
            Tail::Curve(t) => list!(
                curve = curve_name(t.curve),
                fit_from = age(t.fit_period.0),
                fit_to = age(t.fit_period.1),
                extrap_periods = t.extrap_periods as f64,
                attachment_age = age(t.attachment_age)
            ),
            Tail::Bondy(t) => list!(
                earliest_age = age(t.earliest_age),
                attachment_age = age(t.attachment_age)
            ),
            Tail::LogLinear => List::new(0),
        }
    }
}

extendr_module! {
    mod tails;
    impl ReservingTail;
}
