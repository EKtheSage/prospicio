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
truncation types), the generalized Pareto (with a location; Riegel's
parameterization is xi = 1/alpha_tail, beta = t/alpha_ini, location t), the
log-affine local Pareto, the lognormal, the inverse gamma, the inverse
Gaussian, the Burr (XII), the scaled beta, and a lognormal and a Burr
conditioned on a window (lower, upper].
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
GPDS = [  # (xi, beta, location)
    (0.0, 2000, 0),
    (-0.3, 3000, 500),
    (0.1, 2000, 0),
    (0.5, 1000, 1000),
    (1 / 1.5, 500, 1000),
    (1 / 0.9, 250, 1000),
]
LOG_AFFINE = [(1000, 1.5, 0.3), (1000, 0.6, 1.2), (500, 2.5, 0.05), (1000, 0.2, 2.0)]
LOGNORMALS = [(7.0, 0.5), (10.0, 2.0)]
INVERSE_GAMMAS = [(0.7, 1000), (1.5, 2000), (3.0, 4000)]  # (shape, scale)
INVERSE_GAUSSIANS = [(1000, 250), (1000, 4000), (2000, 100000)]  # (mean, shape)
BURRS = [(0.6, 1.5, 1000), (1.5, 2.0, 1500), (3.0, 1.2, 2000)]  # (alpha, gamma, scale)
BETAS = [(2, 5, 10000), (0.5, 0.8, 6000)]  # (a, b, scale)
TRUNCATED = [  # (inner family, inner params, lower, upper)
    ("lognormal", (7.0, 1.0), 500, 20000),
    ("lognormal", (7.0, 1.0), 0, 5000),
    ("lognormal", (7.0, 1.0), 2000, "inf"),
    ("burr", (1.5, 2.0, 1500), 1000, 100000),
]
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


def gpd_survival(xi, beta, loc):
    xi, beta, loc = mp.mpf(xi), mp.mpf(beta), mp.mpf(loc)

    def s(x):
        if x <= loc:
            return mp.mpf(1)
        z = (x - loc) / beta
        if xi == 0:
            return mp.exp(-z)
        y = 1 + xi * z
        return y ** (-1 / xi) if y > 0 else mp.mpf(0)

    # Breakpoints every few scales keep quad accurate where an exponential
    # tail decays by hundreds of e-folds across a layer.
    end = loc - beta / xi if xi < 0 else None
    steps = [loc + 4 * beta * k for k in range(1, 400)]
    kinks = [loc] + ([end] if end is not None else []) + [k for k in steps if end is None or k < end]
    return s, kinks


def log_affine_survival(t, alpha0, gamma):
    t, alpha0, gamma = mp.mpf(t), mp.mpf(alpha0), mp.mpf(gamma)

    def s(x):
        if x < t:
            return mp.mpf(1)
        l = mp.log(x / t)
        return mp.exp(-alpha0 * l - alpha0 * gamma * l * l / 2)

    # Breakpoints at every half doubling keep quad accurate over long layers.
    return s, [t * mp.mpf(2) ** (j / 2) for j in range(0, 80)]


def lognormal_survival(mu, sigma):
    mu, sigma = mp.mpf(mu), mp.mpf(sigma)
    return (lambda x: mp.erfc((mp.log(x) - mu) / (sigma * mp.sqrt(2))) / 2 if x > 0 else mp.mpf(1)), [mp.exp(mu)]


def inverse_gamma_survival(shape, scale):
    a, t = mp.mpf(shape), mp.mpf(scale)
    return (lambda x: mp.gammainc(a, 0, t / x, regularized=True) if x > 0 else mp.mpf(1)), [
        t * mp.mpf(2) ** (j / 2) for j in range(-10, 60)
    ]


def inverse_gaussian_survival(mean, shape):
    m, lam = mp.mpf(mean), mp.mpf(shape)

    def s(x):
        if x <= 0:
            return mp.mpf(1)
        r = mp.sqrt(lam / x)
        return mp.ncdf(-(x - m) / m * r) - mp.exp(2 * lam / m) * mp.ncdf(-(x + m) / m * r)

    sd = mp.sqrt(m**3 / lam)
    return s, [m + sd * k / 2 for k in range(-8, 400) if m + sd * k / 2 > 0]


def burr_survival(alpha, gamma, scale):
    a, g, t = mp.mpf(alpha), mp.mpf(gamma), mp.mpf(scale)
    return (lambda x: (1 + (x / t) ** g) ** (-a) if x > 0 else mp.mpf(1)), [
        t * mp.mpf(2) ** (j / 2) for j in range(-10, 80)
    ]


def beta_survival(a, b, scale):
    a, b, t = mp.mpf(a), mp.mpf(b), mp.mpf(scale)

    def s(x):
        if x <= 0:
            return mp.mpf(1)
        if x >= t:
            return mp.mpf(0)
        return mp.betainc(b, a, 0, 1 - x / t, regularized=True)

    return s, [t * k / 16 for k in range(1, 17)]


def truncated_survival(family, params, lower, upper):
    inner, kinks = lognormal_survival(*params) if family == "lognormal" else burr_survival(*params)
    lo = mp.mpf(lower)
    hi = mp.inf if upper == "inf" else mp.mpf(upper)
    s_hi = mp.mpf(0) if hi == mp.inf else inner(hi)
    p = inner(lo) - s_hi

    def s(x):
        if x <= lo:
            return mp.mpf(1)
        if x >= hi:
            return mp.mpf(0)
        return (inner(x) - s_hi) / p

    if family == "lognormal":
        kinks = [mp.exp(params[0]) * mp.mpf(2) ** (j / 2) for j in range(-10, 60)]
    return s, [k for k in kinks if lo < k < hi] + [lo] + ([hi] if hi != mp.inf else [])


def integrate(f, a, b, kinks, scale=None):
    # quad's error target is absolute, so scale far-tail integrands by the
    # survival at the attachment.
    scale = scale if scale else (f(a) if f(a) != 0 else mp.mpf(1))
    return scale * integrate_unscaled(lambda x: f(x) / scale, a, b, kinks)


def integrate_unscaled(f, a, b, kinks):
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
    m2 = 2 * integrate(lambda x: (x - a) * s(x), a, b, kinks, scale=s(a))
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
    for xi, beta, loc in GPDS:
        s, kinks = gpd_survival(xi, beta, loc)
        params = f"xi={xi!r};beta={beta};location={loc}"
        for cover, att in LAYERS:
            if cover == "inf" and xi >= 0.5:
                if xi < 1:
                    m1 = integrate(s, mp.mpf(att), mp.inf, kinks)
                    yield "gpd", params, "layer", cover, att, m1, 1e-12
                continue
            m1, m2 = moments(s, kinks, cover, att)
            yield "gpd", params, "layer", cover, att, m1, 1e-12
            yield "gpd", params, "layer_second_moment", cover, att, m2, 1e-11
    for t, alpha0, gamma in LOG_AFFINE:
        s, kinks = log_affine_survival(t, alpha0, gamma)
        params = f"t={t};alpha_0={alpha0};gamma={gamma}"
        for cover, att in LAYERS:
            m1, m2 = moments(s, kinks, cover, att)
            yield "log_affine_pareto", params, "layer", cover, att, m1, 1e-12
            yield "log_affine_pareto", params, "layer_second_moment", cover, att, m2, 1e-11
    for mu, sigma in LOGNORMALS:
        s, kinks = lognormal_survival(mu, sigma)
        params = f"meanlog={mu};sdlog={sigma}"
        for cover, att in LAYERS:
            m1, m2 = moments(s, kinks, cover, att)
            yield "lognormal", params, "layer", cover, att, m1, 1e-11
            yield "lognormal", params, "layer_second_moment", cover, att, m2, 1e-10


def family_rows(name, params, s, kinks, has_mean, has_m2, rel=(1e-12, 1e-11)):
    for cover, att in LAYERS:
        unlimited = cover == "inf"
        if unlimited and not has_mean:
            continue
        a = mp.mpf(att)
        b = mp.inf if unlimited else a + mp.mpf(cover)
        m1 = integrate(s, a, b, kinks)
        if m1 != 0:
            yield name, params, "layer", cover, att, m1, rel[0]
        if unlimited and not has_m2:
            continue
        m2 = 2 * integrate(lambda x: (x - a) * s(x), a, b, kinks, scale=s(a))
        if m2 != 0:
            yield name, params, "layer_second_moment", cover, att, m2, rel[1]


def new_rows():
    for shape, scale in INVERSE_GAMMAS:
        s, kinks = inverse_gamma_survival(shape, scale)
        yield from family_rows("inverse_gamma", f"shape={shape};scale={scale}", s, kinks, shape > 1, shape > 2)
    for mean, shape in INVERSE_GAUSSIANS:
        s, kinks = inverse_gaussian_survival(mean, shape)
        # The survival R(a) - R(b) loses about log10(x / mean) digits, so
        # layers a hundred means out get a looser tolerance.
        for row in family_rows("inverse_gaussian", f"mean={mean};shape={shape}", s, kinks, True, True):
            far = float(row[4]) > 100 * mean
            yield row[:6] + ((1e-9,) if far else row[6:])
    for alpha, gamma, scale in BURRS:
        s, kinks = burr_survival(alpha, gamma, scale)
        params = f"alpha={alpha};gamma={gamma};scale={scale}"
        yield from family_rows("burr", params, s, kinks, alpha * gamma > 1, alpha * gamma > 2)
    for a, b, scale in BETAS:
        s, kinks = beta_survival(a, b, scale)
        yield from family_rows("beta", f"a={a};b={b};scale={scale}", s, kinks, True, True)
    for family, params, lower, upper in TRUNCATED:
        s, kinks = truncated_survival(family, params, lower, upper)
        names = ("meanlog", "sdlog") if family == "lognormal" else ("alpha", "gamma", "scale")
        inner = ";".join(f"{k}={v}" for k, v in zip(names, params))
        p = f"inner={family};{inner};lower={lower};upper={upper}"
        bounded = upper != "inf"
        heavy = family == "burr" and params[0] * params[1] <= 2
        yield from family_rows("truncated", p, s, kinks, True, bounded or not heavy)


def main():
    out = sys.stdout if "--stdout" in sys.argv else open(OUT, "w", newline="")
    w = csv.writer(out, lineterminator="\n")
    w.writerow(["distribution", "params", "quantity", "arg", "arg2", "expected", "abs_tol", "rel_tol", "source"])
    for dist, params, qty, cover, att, value, rel in list(rows()) + list(new_rows()):
        w.writerow([dist, params, qty, cover, att, mp.nstr(value, 17, strip_zeros=False), 0.0, rel, SOURCE])


if __name__ == "__main__":
    main()
