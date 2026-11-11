"""Regenerate the fit-curves parity reference from SciPy (issue #804).

Run from the repository root, after `scipy_severity_fits.py`:

    python validation/scripts/scipy_fit_curves.py

Reads the `plain` losses of `validation/data/severity_fits.csv` (89 losses
reported above t = 150,000) and writes `validation/reference/
fit_curves_scipy.csv`: at rows 1, 2, 45 and 89 of the sorted losses, the
plotting position `(n - i + 0.5) / n` and each family's survival
conditional on t, `S(x) / S(t)`, from `scipy_severity_fits.py`'s maximum
likelihood fits; and the rows numpy keeps when thinning to 10 points,
`unique(round(linspace(0, n - 1, 10)))`.
"""

import csv
import importlib.util
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent
spec = importlib.util.spec_from_file_location("fits", HERE / "scipy_severity_fits.py")
fits = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fits)

OUT = "validation/reference/fit_curves_scipy.csv"
ROWS = [1, 2, 45, 89]


def main():
    import warnings

    warnings.simplefilter("ignore", RuntimeWarning)
    y, r, c = fits.datasets()["plain"]
    x = np.sort(y)
    n = len(x)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["family", "row", "loss", "expected", "abs_tol", "rel_tol", "source"])
        for i in ROWS:
            w.writerow(["empirical", i, repr(float(x[i - 1])), repr((n - i + 0.5) / n), 1e-15, 0.0, "formula"])
        for family, (names, make) in fits.FAMILIES.items():
            _, p, _ = fits.fit(family, y, r, c)
            d = make(p)
            st = d.sf(fits.T)
            boundary = min(p[1:] if family == "lognormal" else p) < 1e-6
            tol = 1e-4 if boundary else 1e-6
            for i in ROWS:
                v = float(d.sf(x[i - 1]) / st)
                w.writerow([family, i, repr(float(x[i - 1])), repr(v), tol, 0.0, fits.SOURCE])
        keep = np.unique(np.round(np.linspace(0, n - 1, 10))).astype(int)
        for k in keep:
            w.writerow(["thinned_10", int(k) + 1, repr(float(x[k])), int(k) + 1, 0.0, 0.0, "numpy " + np.__version__])
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
