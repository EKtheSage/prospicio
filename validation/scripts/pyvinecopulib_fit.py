"""Reference values for the Student t pair and for fitting pair copulas,
from pyvinecopulib 1.0.1 (docs/design/risk.md, vine copulas; #568).

- `student`: the Student t pair's cdf, log density, h2 and its inverse at a
  few points, for integer degrees of freedom (pyvinecopulib's bivariate t
  cdf rounds the degrees of freedom to an integer, so non-integer ones are
  checked against quadrature in the Rust tests instead);
- `fit`: data sets of 400 pairs simulated by pyvinecopulib with a seed,
  each with pyvinecopulib's maximum likelihood fit of the true family and
  rotation and its AIC choice (on the row after, the same data) over the Gaussian, Student, Clayton, Gumbel,
  Frank and Joe families.

    uv run --no-project --python 3.12 --with pyvinecopulib==1.0.1 \\
        python validation/scripts/pyvinecopulib_fit.py

writes validation/reference/vine_fit_pyvinecopulib.csv.
"""

from pathlib import Path

import numpy as np
import pyvinecopulib as pv

F = pv.BicopFamily
SOURCE = "pyvinecopulib 1.0.1"
OUT = Path(__file__).resolve().parents[1] / "reference" / "vine_fit_pyvinecopulib.csv"
NAMES = {F.indep: "independence", F.gaussian: "gaussian", F.student: "student", F.clayton: "clayton",
         F.gumbel: "gumbel", F.frank: "frank", F.joe: "joe"}
FAMILIES = [F.gaussian, F.student, F.clayton, F.gumbel, F.frank, F.joe]


def params(b):
    return ";".join(repr(float(x)) for x in b.parameters.ravel())


def main():
    rows = ["case,family,rotation,parameters,quantity,data,expected,source"]
    for rho, nu in [(0.6, 4.0), (-0.4, 8.0), (0.9, 30.0), (0.2, 3.0)]:
        c = pv.Bicop(family=F.student, parameters=np.array([[rho], [nu]]))
        for u, v in [(0.3, 0.6), (0.8, 0.2), (0.55, 0.45), (0.1, 0.9), (0.02, 0.97), (0.999, 0.998)]:
            x = np.array([[u, v]])
            pt = f"{u!r};{v!r}"
            for q, val in [("cdf", c.cdf(x)[0]), ("log_pdf", np.log(c.pdf(x)[0])),
                           ("h2", c.hfunc2(x)[0]), ("h2_inverse", c.hinv2(x)[0])]:
                rows.append(f"student,student,0,{rho!r};{nu!r},{q},{pt},{float(val)!r},{SOURCE}")
    truths = [(F.gaussian, 0, [0.5]), (F.student, 0, [0.6, 5.0]), (F.clayton, 0, [3.0]),
              (F.gumbel, 180, [2.0]), (F.frank, 0, [-4.0]), (F.joe, 90, [2.5]), (F.clayton, 270, [1.5])]
    for k, (fam, rot, par) in enumerate(truths):
        truth = pv.Bicop(family=fam, rotation=rot, parameters=np.array(par).reshape(-1, 1))
        u = truth.sample(400, seeds=[568, k])
        data = ";".join(f"{float(a)!r}|{float(b)!r}" for a, b in u)
        mle = pv.FitControlsBicop(family_set=[fam], parametric_method="mle")
        own = pv.Bicop(family=fam, rotation=rot)
        own.fit(u, controls=mle)
        rows.append(f"fit,{NAMES[fam]},{rot},{params(own)},own,{data},{float(own.loglik(u))!r},{SOURCE}")
        sel = pv.Bicop()
        sel.select(u, controls=pv.FitControlsBicop(family_set=FAMILIES, parametric_method="mle",
                                                    selection_criterion="aic"))
        # The same data as the row before.
        rows.append(f"fit,{NAMES[sel.family]},{sel.rotation},{params(sel)},select,,"
                    f"{float(sel.loglik(u))!r},{SOURCE}")
    OUT.write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
