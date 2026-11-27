### Aggregate and reinsurance

- A new crate, `prospicio-esg` (`prospicio::esg`): an economic scenario
  generator with the Vasicek, CIR and Hull-White short-rate models, their
  bond prices in closed form, the Nelson-Siegel-Svensson curve and its fit,
  and `Esg::simulate`, which gives the short rate, the stochastic discount
  factor and the curve on each path at monthly, quarterly or annual steps.
  The transitions are exact, and Hull-White reproduces today's curve.
  Rust only for now; inflation and the Python and R bindings follow (#983).
