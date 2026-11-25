//! Saving and loading [`Terms`] and [`Coding`]: versioned JSON artifacts in
//! the envelope the model artifacts share (`docs/design/models.md`, "Model
//! artifacts"; #978).
//!
//! `Terms` is the formula before any data is seen; `Coding` is the formula
//! with what training learned (factor levels and references, bins, the
//! orthogonal polynomials' recurrence, the spline's boundary knots), so a
//! loaded coding builds the same design matrix on new data. Numbers
//! round-trip bit for bit. Neither keeps a hash of its data, so the
//! provenance's `input_hash` is empty. A coding read back is checked as
//! [`Terms::fit`] checks what it builds.

use prospicio_core::Result;
use serde_json::{Map, Value, json};

use super::{Coded, Coding, Side, Term, Terms, increasing};
use crate::artifact::{array, bad, field, float, floats, num, nums, object, parse, strings, uint};

const TERMS_FORMAT: &str = "risk_rs.terms";
const CODING_FORMAT: &str = "risk_rs.coding";
/// The newest format version this build reads.
const FORMAT_VERSION: u64 = 1;

fn envelope(format: &str, model: &str, terms: Vec<Value>) -> String {
    json!({
        "format": format,
        "format_version": FORMAT_VERSION,
        "provenance": crate::artifact::provenance(
            model,
            "prospicio-models",
            env!("CARGO_PKG_VERSION"),
            "",
        ),
        "terms": terms,
    })
    .to_string()
}

fn string(o: &Map<String, Value>, key: &str) -> Result<String> {
    field(o, key)?
        .as_str()
        .map(String::from)
        .ok_or_else(|| bad(key))
}

fn optional<'a>(o: &'a Map<String, Value>, key: &str) -> Option<&'a Value> {
    o.get(key).filter(|v| !v.is_null())
}

fn usize_at(o: &Map<String, Value>, key: &str) -> Result<usize> {
    usize::try_from(uint(o, key)?).map_err(|_| bad(key))
}

fn term_json(t: &Term) -> Value {
    match t {
        Term::Intercept => json!({"kind": "intercept"}),
        Term::Numeric(name) => json!({"kind": "numeric", "name": name}),
        Term::Factor { name, reference } => {
            json!({"kind": "factor", "name": name, "reference": reference})
        }
        Term::Interaction { a, b } => json!({"kind": "interaction", "a": a, "b": b}),
        Term::Poly { name, degree } => json!({"kind": "poly", "name": name, "degree": degree}),
        Term::Hinges { name, knots } => {
            json!({"kind": "hinges", "name": name, "knots": nums(knots)})
        }
        Term::Steps { name, knots } => json!({"kind": "steps", "name": name, "knots": nums(knots)}),
        Term::Bins {
            name,
            cuts,
            reference,
        } => json!({"kind": "bins", "name": name, "cuts": nums(cuts), "reference": reference}),
        Term::Log { name, base } => json!({"kind": "log", "name": name, "base": num(*base)}),
        Term::Ordinal { name, levels } => {
            json!({"kind": "ordinal", "name": name, "levels": levels})
        }
        Term::OrthogonalPoly { name, degree } => {
            json!({"kind": "orthogonal_poly", "name": name, "degree": degree})
        }
        Term::NaturalSpline {
            name,
            knots,
            boundary,
        } => json!({
            "kind": "natural_spline",
            "name": name,
            "knots": nums(knots),
            "boundary": boundary.map(|(lo, hi)| nums(&[lo, hi])),
        }),
    }
}

fn term_from(v: &Value) -> Result<Term> {
    let o = object(v, "terms")?;
    let kind = field(o, "kind")?
        .as_str()
        .ok_or_else(|| bad("terms.kind"))?;
    if kind == "intercept" {
        return Ok(Term::Intercept);
    }
    if kind == "interaction" {
        return Ok(Term::Interaction {
            a: string(o, "a")?,
            b: string(o, "b")?,
        });
    }
    let name = string(o, "name")?;
    Ok(match kind {
        "numeric" => Term::Numeric(name),
        "factor" => Term::Factor {
            name,
            reference: optional(o, "reference")
                .map(|r| r.as_str().map(String::from).ok_or_else(|| bad("reference")))
                .transpose()?,
        },
        "poly" => Term::Poly {
            name,
            degree: usize_at(o, "degree")?,
        },
        "hinges" => Term::Hinges {
            name,
            knots: floats(o, "knots")?,
        },
        "steps" => Term::Steps {
            name,
            knots: floats(o, "knots")?,
        },
        "bins" => Term::Bins {
            name,
            cuts: floats(o, "cuts")?,
            reference: optional(o, "reference")
                .map(|r| {
                    r.as_u64()
                        .and_then(|r| usize::try_from(r).ok())
                        .ok_or_else(|| bad("reference"))
                })
                .transpose()?,
        },
        "log" => Term::Log {
            name,
            base: float(field(o, "base")?, "base")?,
        },
        "ordinal" => Term::Ordinal {
            name,
            levels: strings(o, "levels")?,
        },
        "orthogonal_poly" => Term::OrthogonalPoly {
            name,
            degree: usize_at(o, "degree")?,
        },
        "natural_spline" => Term::NaturalSpline {
            name,
            knots: floats(o, "knots")?,
            boundary: match optional(o, "boundary") {
                None => None,
                Some(_) => match floats(o, "boundary")?.as_slice() {
                    [lo, hi] => Some((*lo, *hi)),
                    _ => return Err(bad("boundary")),
                },
            },
        },
        _ => return Err(bad("terms.kind")),
    })
}

fn side_json(s: &Side) -> Value {
    match s {
        Side::Numeric(name) => json!({"kind": "numeric", "name": name}),
        Side::Factor {
            name,
            reference,
            levels,
        } => json!({"kind": "factor", "name": name, "reference": reference, "levels": levels}),
    }
}

fn side_from(v: &Value) -> Result<Side> {
    let o = object(v, "interaction side")?;
    let name = string(o, "name")?;
    match field(o, "kind")?.as_str() {
        Some("numeric") => Ok(Side::Numeric(name)),
        Some("factor") => Ok(Side::Factor {
            name,
            reference: string(o, "reference")?,
            levels: strings(o, "levels")?,
        }),
        _ => Err(bad("interaction side kind")),
    }
}

fn coded_json(c: &Coded) -> Value {
    match c {
        Coded::Intercept => json!({"kind": "intercept"}),
        Coded::Numeric(name) => json!({"kind": "numeric", "name": name}),
        Coded::Factor {
            name,
            reference,
            levels,
        } => json!({"kind": "factor", "name": name, "reference": reference, "levels": levels}),
        Coded::Interaction { a, b } => {
            json!({"kind": "interaction", "a": side_json(a), "b": side_json(b)})
        }
        Coded::Poly { name, degree } => json!({"kind": "poly", "name": name, "degree": degree}),
        Coded::Hinges { name, knots } => {
            json!({"kind": "hinges", "name": name, "knots": nums(knots)})
        }
        Coded::Steps { name, knots } => {
            json!({"kind": "steps", "name": name, "knots": nums(knots)})
        }
        Coded::Bins {
            name,
            cuts,
            reference,
        } => json!({"kind": "bins", "name": name, "cuts": nums(cuts), "reference": reference}),
        Coded::Log { name, base } => json!({"kind": "log", "name": name, "base": num(*base)}),
        Coded::Ordinal { name, levels } => {
            json!({"kind": "ordinal", "name": name, "levels": levels})
        }
        Coded::OrthogonalPoly { name, alpha, norm2 } => json!({
            "kind": "orthogonal_poly",
            "name": name,
            "alpha": nums(alpha),
            "norm2": nums(norm2),
        }),
        Coded::NaturalSpline { name, knots } => {
            json!({"kind": "natural_spline", "name": name, "knots": nums(knots)})
        }
    }
}

fn coded_from(v: &Value) -> Result<Coded> {
    let o = object(v, "terms")?;
    let kind = field(o, "kind")?
        .as_str()
        .ok_or_else(|| bad("terms.kind"))?;
    if kind == "intercept" {
        return Ok(Coded::Intercept);
    }
    if kind == "interaction" {
        return Ok(Coded::Interaction {
            a: side_from(field(o, "a")?)?,
            b: side_from(field(o, "b")?)?,
        });
    }
    let name = string(o, "name")?;
    Ok(match kind {
        "numeric" => Coded::Numeric(name),
        "factor" => Coded::Factor {
            name,
            reference: string(o, "reference")?,
            levels: strings(o, "levels")?,
        },
        "poly" => {
            let degree = usize_at(o, "degree")?;
            if degree == 0 {
                return Err(bad("degree"));
            }
            Coded::Poly { name, degree }
        }
        "hinges" | "steps" => {
            let knots = floats(o, "knots")?;
            increasing(&knots, "knots")?;
            if kind == "hinges" {
                Coded::Hinges { name, knots }
            } else {
                Coded::Steps { name, knots }
            }
        }
        "bins" => {
            let cuts = floats(o, "cuts")?;
            increasing(&cuts, "cuts")?;
            let reference = usize_at(o, "reference")?;
            if reference > cuts.len() {
                return Err(bad("reference"));
            }
            Coded::Bins {
                name,
                cuts,
                reference,
            }
        }
        "log" => {
            let base = float(field(o, "base")?, "base")?;
            if !(base.is_finite() && base > 0.0) {
                return Err(bad("base"));
            }
            Coded::Log { name, base }
        }
        "ordinal" => {
            let levels = strings(o, "levels")?;
            if levels.len() < 2 {
                return Err(bad("levels"));
            }
            Coded::Ordinal { name, levels }
        }
        "orthogonal_poly" => {
            let (alpha, norm2) = (floats(o, "alpha")?, floats(o, "norm2")?);
            if alpha.is_empty()
                || norm2.len() != alpha.len() + 2
                || alpha.iter().any(|a| !a.is_finite())
                || norm2.iter().any(|n| !(n.is_finite() && *n > 0.0))
            {
                return Err(bad("orthogonal_poly"));
            }
            Coded::OrthogonalPoly { name, alpha, norm2 }
        }
        "natural_spline" => {
            let knots = floats(o, "knots")?;
            increasing(&knots, "knots")?;
            if knots.len() < 3 {
                return Err(bad("knots"));
            }
            Coded::NaturalSpline { name, knots }
        }
        _ => return Err(bad("terms.kind")),
    })
}

fn items(text: &str, format: &str) -> Result<Vec<Value>> {
    let doc = parse(text, format, FORMAT_VERSION)?;
    let top = object(&doc, "artifact")?;
    Ok(array(top, "terms")?.clone())
}

impl Terms {
    /// The terms as a versioned JSON artifact (format `risk_rs.terms`),
    /// which [`Terms::from_json`] reads back.
    ///
    /// ```
    /// use prospicio_models::Terms;
    ///
    /// let terms = Terms::new().intercept().numeric("age").factor("region").bins("x", &[1.0, 2.0]);
    /// assert_eq!(Terms::from_json(&terms.to_json()).unwrap(), terms);
    /// ```
    pub fn to_json(&self) -> String {
        envelope(
            TERMS_FORMAT,
            "Terms",
            self.terms.iter().map(term_json).collect(),
        )
    }

    /// Reads terms that [`Terms::to_json`] wrote. Fails on text that is
    /// not such an artifact, or one of a newer format version.
    pub fn from_json(text: &str) -> Result<Self> {
        let terms = items(text, TERMS_FORMAT)?
            .iter()
            .map(term_from)
            .collect::<Result<_>>()?;
        Ok(Self { terms })
    }
}

impl Coding {
    /// The coding as a versioned JSON artifact (format `risk_rs.coding`),
    /// which [`Coding::from_json`] reads back: what training learned, so
    /// the loaded coding builds the same design on new data.
    ///
    /// ```
    /// use prospicio_models::{Coding, Column, Frame, Terms};
    ///
    /// let train = Frame::new(vec![
    ///     ("age".into(), Column::Numeric(vec![30.0, 45.0, 60.0])),
    ///     ("region".into(), Column::Categorical(vec!["N".into(), "S".into(), "W".into()])),
    /// ])
    /// .unwrap();
    /// let coding = Terms::new().intercept().numeric("age").factor("region").fit(&train).unwrap();
    /// let back = Coding::from_json(&coding.to_json()).unwrap();
    /// assert_eq!(back, coding);
    /// assert_eq!(back.design(&train).unwrap(), coding.design(&train).unwrap());
    /// ```
    pub fn to_json(&self) -> String {
        envelope(
            CODING_FORMAT,
            "Coding",
            self.coded.iter().map(coded_json).collect(),
        )
    }

    /// Reads a coding that [`Coding::to_json`] wrote. Fails on text that
    /// is not such an artifact, one of a newer format version, or a term
    /// that training could not have produced.
    pub fn from_json(text: &str) -> Result<Self> {
        let coded = items(text, CODING_FORMAT)?
            .iter()
            .map(coded_from)
            .collect::<Result<_>>()?;
        Ok(Self { coded })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Column, Frame};

    fn frame() -> Frame {
        let x: Vec<f64> = (0..12).map(|i| 1.0 + f64::from(i) * 0.7).collect();
        let cat = |v: &[&str]| Column::Categorical(v.iter().map(|s| (*s).into()).collect());
        Frame::new(vec![
            ("x".into(), Column::Numeric(x)),
            (
                "region".into(),
                cat(&["N", "S", "W", "N", "S", "W", "N", "S", "W", "N", "S", "W"]),
            ),
            (
                "band".into(),
                cat(&[
                    "lo", "mid", "hi", "lo", "mid", "hi", "lo", "mid", "hi", "lo", "mid", "hi",
                ]),
            ),
        ])
        .unwrap()
    }

    fn every_term() -> Terms {
        Terms::new()
            .intercept()
            .numeric("x")
            .factor("region")
            .factor_with_reference("band", "mid")
            .interaction("x", "region")
            .poly("x", 3)
            .hinges("x", &[3.0, 5.0])
            .steps("x", &[4.0])
            .bins("x", &[2.0, 6.0])
            .log("x", 2.0)
            .ordinal("band", &["lo", "mid", "hi"])
            .orthogonal_poly("x", 2)
            .natural_spline("x", &[3.0, 6.0])
            .term(Term::NaturalSpline {
                name: "x".into(),
                knots: vec![4.0],
                boundary: Some((0.5, 10.0)),
            })
    }

    #[test]
    fn terms_and_coding_round_trip_with_every_kind() {
        let terms = every_term();
        assert_eq!(Terms::from_json(&terms.to_json()).unwrap(), terms);
        let coding = terms.fit(&frame()).unwrap();
        let back = Coding::from_json(&coding.to_json()).unwrap();
        assert_eq!(back, coding);
        assert_eq!(back.to_json(), coding.to_json());
        assert_eq!(
            back.design(&frame()).unwrap(),
            coding.design(&frame()).unwrap()
        );
    }

    #[test]
    fn bad_artifacts_are_refused() {
        let coding = every_term().fit(&frame()).unwrap();
        let text = coding.to_json();
        assert!(Terms::from_json(&text).is_err(), "wrong format");
        assert!(
            Coding::from_json(&text.replace("\"format_version\":1", "\"format_version\":2"))
                .is_err()
        );
        let ordinal = text.replace(r#""levels":["lo","mid","hi"]"#, r#""levels":["lo"]"#);
        assert!(Coding::from_json(&ordinal).is_err(), "one ordinal level");
        let bins = text.replace(r#""reference":0"#, r#""reference":7"#);
        assert!(
            Coding::from_json(&bins).is_err(),
            "bin reference out of range"
        );
        assert!(Coding::from_json("not json").is_err());
    }
}
