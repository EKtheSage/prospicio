---
type: Reference
title: Mildenhall's aggregate package, Pricing Insurance Risk and CAS Monograph 15
description: The Python aggregate package (1.0.1) as the reference for distortions, calibration and natural allocation; where the Pricing Insurance Risk and Monograph 15 examples live, and the API and tolerance quirks met reproducing them.
resource: https://github.com/mynl/aggregate
tags: [pricing, risk-measures, distortions, capital-allocation, aggregate, parity]
status: stable
generated: { by: claude-code/cloud-session, at: 2026-10-08T15:00:00Z }
sources:
  - id: aggregate
    resource: https://github.com/mynl/aggregate
    title: mynl/aggregate, Stephen Mildenhall's aggregate distribution and pricing package (1.0.1, 2026-10-07)
  - id: monograph
    resource: https://github.com/casact/capital-modeling
    title: casact/capital-modeling, the Quarto source of CAS Monograph 15 (Major and Mildenhall, Introduction to Capital Modeling and Portfolio Management, 2026)
  - id: pir
    resource: https://pypi.org/project/aggregate/0.30.1/
    title: aggregate 0.30.1 on PyPI, whose extensions/ reproduce the exhibits of Pricing Insurance Risk (Mildenhall and Major, Wiley 2022)
  - id: script
    resource: ../validation/scripts/aggregate_distortions.py
    title: aggregate_distortions.py
---

# Which source for what

* **aggregate 1.0.1** (Python 3.12 to 3.14) is the reference for the
  distortions, their calibration and the natural allocation: `Distortion`,
  `Portfolio.calibrate_distortions`, `Portfolio.price(p, d, allocation=)`.[^aggregate]
* **Pricing Insurance Risk** exhibits do not reproduce on 1.0: its
  changelog says much of their machinery was removed in the 1.0 alphas.
  They need **aggregate 0.30.1** in its own environment, whose
  `extensions/` hold the case studies (`case_studies.py`, `discrete.py`,
  `tame.py`, `cnc.py`, `hs.py`), the Bodoff exhibit and the figure code.[^pir]
* **CAS Monograph 15** is free, and its full Quarto source, with the
  scripts behind its tables (`basic_example_script.py`, `070_script.py`),
  is on GitHub. The CAS site itself was blocked from the cloud
  environment; GitHub and PyPI were not.[^monograph]

# Quirks

* The monograph's scripts call `calibrate_distortions(Ps=[1], COCs=[.15])`,
  an older signature; 1.0.1 takes `calibrate_distortions(0.15, p=1)` (the
  return first, then the capital standard `p` or the assets `a`).[^monograph]
* `Distortion` takes its parameters by keyword: `Distortion('ccoc', r=)`,
  `('ph', a=)`, `('wang', lam=)`, `('dual', b=)`, `('tvar', p=)`,
  `('bitvar', p0=, p1=, w1=)` with `w1` the weight on `TVaR(p1)`,
  `('clin', r0=, slope=)`, `('cll', r0=, b=)`, `('lep', r0=, r=)`,
  `('ly', r0=, r=)`, `('beta', a=, b=)`, `('wtdtvar', ps=, wts=)`.
  `Distortion.price(ser, a=, kind='ask')` returns a `Price` named tuple;
  the asset level must be one of the series' outcomes.
* Calibration stops Newton at a premium error, reported as `error` in
  `distortion_df`: on InsCo (`P = 53.565`, ι = 15%, `p = 1`) it is
  `3.3e-10` for PH, `1.3e-8` for Wang, `3.4e-7` for dual and `7.6e-6` for
  TVaR. prospicio's bisection reaches the premium to rounding, so its
  parameters agree with aggregate's only within that error over the
  price's slope (`3.2e-6` in TVaR's `p`).[^script]
* InsCo's calibrated parameters at 15% and `p = 1`: CCoC `r = 0.15`, PH
  `0.720479`, Wang `0.342731`, dual `1.595151`, TVaR `0.271287`; the
  monograph's totals are 22, 28, 36, 40 (four times), 55, 65 and 100.
