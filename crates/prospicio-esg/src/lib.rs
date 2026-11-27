//! An economic scenario generator (`docs/design/esg-dfa.md`): short-rate
//! models with their bond prices in closed form, the Nelson-Siegel-Svensson
//! curve, and simulated [`Scenarios`] of the short rate, the stochastic
//! discount factor and the curve at each step.
//!
//! * [`Vasicek`] and [`Cir`] (Brigo and Mercurio, chapter 3), simulated by
//!   their exact transitions;
//! * [`HullWhite`] one factor, which reproduces an [`InitialCurve`] (a
//!   spot curve or a [`Svensson`] fit) exactly;
//! * [`Svensson`], fitted to observed zero-coupon yields.
//!
//! The dynamics are real-world (P) as parametrised: a market price of risk
//! is folded into the parameters (decision 3). Inflation, the superimposed
//! claims inflation and their `IndexPaths` come next (the note's PR 2), and
//! the multi-year projection in `prospicio-aggregate::dfa` after them.
//!
//! In the WASM scope: no threads, files or C dependencies. Paths are drawn
//! one after another, each from its own stream, so a parallel split later
//! keeps their numbers.
//!
//! ```
//! use prospicio_esg::{Esg, HullWhite, InitialCurve, Step};
//!
//! let today = InitialCurve::spot(vec![1.0, 5.0, 10.0], vec![0.03, 0.035, 0.04])?;
//! let model = HullWhite::new(0.1, 0.01, today.clone())?;
//! let paths = Esg::new(model).with_step(Step::Annual).simulate(10, 500, 1)?;
//! // Today's simulated curve is the initial one.
//! let curve = paths.curve(0, 0)?;
//! assert!((curve.rates[0] - 0.03).abs() < 1e-12);
//! # Ok::<(), prospicio_core::Error>(())
//! ```

mod sampling;
mod scenarios;
mod short_rate;
mod svensson;

pub use scenarios::{Curve, Esg, Scenarios, Step};
pub use short_rate::{Cir, HullWhite, InitialCurve, ShortRate, Vasicek};
pub use svensson::Svensson;
