"""Regenerate validation/reference/layer_moments_mpmath.csv.

    pip install mpmath
    python validation/scripts/mpmath_layer_moments.py

Layer means and layer second moments by numerically integrating the
survival function at 30 significant digits, independently of the closed
forms in Rust:

    layer(c xs a)            = integral of S(x) over (a, a + c)
    layer second moment      = 2 * integral of (x - a) S(x) over (a, a + c)
    lev(d), stop_loss(d)     = integral of S over (0, d) and (d, inf)

for the Pareto (optionally truncated), the piecewise Pareto (with both
truncation types) and the lognormal.
"""

import csv
import sys

import mpmath as mp

mp.mp.dps = 30
OUT = "validation/reference/layer_moments_mpmath.csv"
SOURCE = f"mpmath {mp.__version__} quad of survival at 30 digits"

PARETOS = [  # (t, alpha, truncation or None)
    (1000, 0.8, None),
    (1000, 1.0, None),
    (1000, 1.5, None),
    (1000, 2.0, None),
    (1000, 3.5, None),
    (500, 1.2, 20000),
    (500, 2.0, 8000),
]
PIECEWISE = [  # (thresholds, alphas, truncation or None, type)
    ([1000, 2000, 3000], [1, 1.5, 2], None, None),
    ([1000, 2000, 3000], [1, 1.5, 2], 5000, "lp"),
    ([1000, 2000, 3000], [1, 1.5, 2], 5000, "wd"),
    ([1000, 1500, 4000, 10000], [2.5, 0.4, 3, 1.2], 1000000, "wd"),
    ([200, 1000, 50000], [1, 3, 2.5], None, None),
]
LOGNORMALS = [(7.0, 0.5), (10.0, 2.0)]
LAYERS = [  # (cover, attachment); "inf" for unlimited
    ("500", "0"),
    ("4000", "1000"),
    ("5000", "5000"),
    ("1e6", "2e5"),
    ("inf", "3000"),
]


def pareto_survival(t, alpha, tr):
    t, alpha = mp.mpf(t), mp.mpf(alpha)

    def s(x):
        if x < t:
            return mp.mpf(1)
        raw = (t / x) ** alpha
        if tr is None:
            return raw
        tr_ = mp.mpf(tr)
        if x >= tr_:
            return mp.mpf(0)
        q = (t / tr_) ** alpha
        return (raw - q) / (1 - q)

    return s, [t] + ([mp.mpf(tr)] if tr is not None else [])


def piecewise_survival(ts, alphas, tr, kind):
    ts, alphas = [mp.mpf(t) for t in ts], [mp.mpf(a) for a in alphas]

    def base(x):
        if x < ts[0]:
            return mp.mpf(1)
        s = mp.mpf(1)
        for k, (t, a) in enumerate(zip(ts, alphas)):
            nxt = ts[k + 1] if k + 1 < len(ts) else mp.inf
            if x < nxt:
                if kind == "lp" and k == len(ts) - 1:
                    q = (t / tr_) ** a
                    return s * ((t / x) ** a - q) / (1 - q) if x < tr_ else mp.mpf(0)
                return s * (t / x) ** a
            s *= (t / nxt) ** a

    tr_ = mp.mpf(tr) if tr is not None else None
    if kind == "wd":
        s_tr = base(tr_)
        return (lambda x: (base(x) - s_tr) / (1 - s_tr) if x < tr_ else mp.mpf(0)), ts + [tr_]
    return base, ts + ([tr_] if tr_ is not None else [])


def lognormal_survival(mu, sigma):
    mu, sigma = mp.mpf(mu), mp.mpf(sigma)
    return (lambda x: mp.erfc((mp.log(x) - mu) / (sigma * mp.sqrt(2))) / 2 if x > 0 else mp.mpf(1)), [mp.exp(mu)]


def integrate(f, a, b, kinks):
    pts = sorted({a, b} | {k for k in kinks if a < k < b})
    if b == mp.inf:
        finite = [p for p in pts if p != mp.inf]
        top = max(finite)
        pts = finite + [top * 10, top * 1000, mp.inf] if top > 0 else [0, 1, 1000, mp.inf]
    return mp.quad(f, pts)


def moments(s, kinks, cover, att):
    a = mp.mpf(att)
    b = mp.inf if cover == "inf" else a + mp.mpf(cover)
    m1 = integrate(s, a, b, kinks)
    m2 = 2 * integrate(lambda x: (x - a) * s(x), a, b, kinks)
    return m1, m2


def rows():
    for t, alpha, tr in PARETOS:
        s, kinks = pareto_survival(t, alpha, tr)
        params = f"t={t};alpha={alpha}" + (f";truncation={tr}" if tr else "")
        for cover, att in LAYERS:
            infinite_mean = cover == "inf" and tr is None and alpha <= 1
            infinite_m2 = cover == "inf" and tr is None and alpha <= 2
            m1, m2 = moments(s, kinks, cover, att) if not infinite_m2 else (None, None)
            if infinite_m2 and not infinite_mean:
                m1 = integrate(s, mp.mpf(att), mp.inf, kinks)
            if m1 is not None:
                yield "pareto", params, "layer", cover, att, m1, 1e-12
            if m2 is not None:
                yield "pareto", params, "layer_second_moment", cover, att, m2, 1e-11
    for ts, alphas, tr, kind in PIECEWISE:
        s, kinks = piecewise_survival(ts, alphas, tr, kind)
        join = lambda v: "|".join(str(x) for x in v)
        params = f"t={join(ts)};alpha={join(alphas)}" + (f";truncation={tr};type={kind}" if tr else "")
        for cover, att in LAYERS:
            if cover == "inf" and tr is None and alphas[-1] <= 2:
                continue
            m1, m2 = moments(s, kinks, cover, att)
            yield "piecewise_pareto", params, "layer", cover, att, m1, 1e-12
            yield "piecewise_pareto", params, "layer_second_moment", cover, att, m2, 1e-11
    for mu, sigma in LOGNORMALS:
        s, kinks = lognormal_survival(mu, sigma)
        params = f"meanlog={mu};sdlog={sigma}"
        for cover, att in LAYERS:
            m1, m2 = moments(s, kinks, cover, att)
            yield "lognormal", params, "layer", cover, att, m1, 1e-11
            yield "lognormal", params, "layer_second_moment", cover, att, m2, 1e-10


def main():
    out = sys.stdout if "--stdout" in sys.argv else open(OUT, "w", newline="")
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected", "abs_tol", "rel_tol", "source"])
    for dist, params, qty, cover, att, value, rel in rows():
        w.writerow([dist, params, qty, cover, att, mp.nstr(value, 17, strip_zeros=False), 0.0, rel, SOURCE])


if __name__ == "__main__":
    main()
