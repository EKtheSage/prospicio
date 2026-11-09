# Stochastic loss reserving with GLMs: Taylor and McGuire reproduced

<!-- site: execute -->

CAS Monograph 3 is *Stochastic Loss Reserving Using Generalized Linear Models*
(Greg Taylor and Gráinne McGuire, 2016). It shows that the chain ladder is a
generalized linear model (GLM), puts a prediction error on the reserve, checks
the model with residuals and improves it with fewer parameters and some
interactions, all on one triangle. This guide runs the examples in prospicio and
shows the monograph's numbers beside prospicio's.

Table and figure numbers are the monograph's. The explanations are new and the
numbers are recomputed. Money is in thousands of dollars. To choose a reserving
method in general, read [`reserving-methods.md`](reserving-methods.md).

`prospicio.reserving` has the chain ladder, Mack, the ODP bootstrap and the ODP
GLM (`OdpGlm`) on a `Triangle`. `OdpGlm` fits the cross-classified model by
default and gives the analytic prediction error, simulated reserves and
residual diagnostics. `OdpGlm.cells` lists the triangle's cells: write any other
design on them and pass it to `fit(..., design=)`. Each model of chapter 7 is
the same call with another design. `prospicio.models` has the general GLM
(`Glm`, `Design`), which fits the ODP Mack model of section 2.

## 1. The data and the chain ladder

The monograph's example (Table 1-1) is the incremental paid triangle of a New
Jersey workers compensation insurer, accident years 1988 to 1997, from the
database of Meyers and Shi (2011). Its age-to-age factors are almost constant
down each column, so the chain ladder fits it well.

```python
import math

import numpy as np

from prospicio.models import Design, Glm
from prospicio.reserving import ChainLadder, Mack, OdpBootstrap, OdpGlm, Triangle

INCREMENTS = [  # Table 1-1, accident years 1988 to 1997, development years 1 to 10
    [41821, 34729, 20147, 15965, 11285, 5924, 4775, 3742, 3435, 2958],
    [48167, 39495, 24444, 18178, 10840, 7379, 5683, 4758, 3959],
    [52058, 47459, 27359, 17916, 11448, 8846, 5869, 5391],
    [57251, 49510, 27036, 20871, 14304, 10552, 7742],
    [59213, 54129, 29566, 22484, 14114, 10000],
    [59475, 52076, 26836, 22332, 14756],
    [65607, 44648, 27062, 22655],
    [56748, 39315, 26748],
    [52212, 40030],
    [43962],
]
K = len(INCREMENTS)
YEARS = list(range(1988, 1988 + K))
cumulative = [np.cumsum(row) for row in INCREMENTS]
tri = Triangle.from_long(
    [YEARS[k] for k, row in enumerate(cumulative) for _ in row],
    [12 * (j + 1) for row in cumulative for j in range(len(row))],
    {"paid": [float(v) for row in cumulative for v in row]},
)
cl = ChainLadder().fit(tri, "paid")
print("factors ", " ".join(f"{f:.3f}" for f in cl.ldf))
print("reserves", [round(r) for r in cl.reserve[1:]])
print("total   ", round(cl.total_reserve))
```

```text
factors  1.815 1.261 1.158 1.088 1.055 1.039 1.030 1.025 1.021
reserves [3398, 8155, 14579, 22645, 31865, 45753, 60093, 80983, 105874]
total    373346
```

The volume-weighted factors are Table 3-1 and the reserves the last column of
Table 3-2, to the unit.

## 2. The chain ladder as a GLM

Chapter 3 gives two models that reproduce the chain ladder's forecasts. The
monograph uses the *cross-classified* ODP model from then on. Each incremental
payment `Y[k, j]` is over-dispersed Poisson (ODP) with mean `a[k] * b[j]`, a row
effect times a column effect, and variance `phi` times the mean.

On the log scale this is a GLM with a log link and an indicator column for
each accident year and development year. The design drops one column effect for
full rank: the monograph sets `ln b[1] = 0`, so the accident-year terms carry
the level.

`OdpGlm().fit(tri, "paid")` uses an intercept and factors from the first level;
the monograph uses one `ln a[k]` per accident year and no intercept. To print
that coding, give `fit` that design on `OdpGlm.cells`: the 100 cells (55
observed, 45 to forecast), in the order a design's rows must follow, with
0-based accident and development indices:

```python
import pandas as pd
from great_tables import GT

cells = OdpGlm.cells(tri, "paid")
k = np.array(cells["origin_index"]) + 1  # accident year, 1 to 10
j = np.array(cells["development_index"]) + 1  # development year, 1 to 10


def fit(columns, names):
    """The ODP GLM of the triangle, one design column per array (or constant) over the cells."""
    x = [np.broadcast_to(np.asarray(c, float), k.shape).tolist() for c in columns]
    return OdpGlm().fit(tri, "paid", design=Design(x, names))


cc = fit([k == a for a in range(1, K + 1)] + [j == b for b in range(2, K + 1)],
         [f"ln a{a}" for a in range(1, K + 1)] + [f"ln b{b}" for b in range(2, K + 1)])

est, se = cc.glm.coefficients, cc.glm.std_errors
table_5_1 = pd.DataFrame({
    "index": [str(i) for i in range(1, K + 1)],
    "ln_a": est[:K],
    "se_a": se[:K],
    "ln_b": [0.0, *est[K:]],
    "se_b": [np.nan, *se[K:]],
})
(GT(table_5_1, rowname_col="index")
 .tab_header("Table 5-1: the cross-classified model",
             "Accident year k for ln a, development year j for ln b; ln b[1] = 0")
 .cols_label(ln_a="ln a", se_a="s.e.", ln_b="ln b", se_b="s.e.")
 .fmt_number(["ln_a", "ln_b"], decimals=3)
 .fmt_number(["se_a", "se_b"], decimals=4)
 .sub_missing(missing_text=""))
```

```text
      ln a    s.e.    ln b    s.e.
1   10.657  0.0316   0.000        
2   10.795  0.0299  -0.205  0.0228
3   10.899  0.0289  -0.747  0.0282
4   10.989  0.0281  -1.017  0.0328
5   11.039  0.0278  -1.452  0.0421
6   11.016  0.0285  -1.833  0.0547
7   11.008  0.0295  -2.140  0.0715
8   10.891  0.0327  -2.348  0.0931
9   10.836  0.0367  -2.513  0.1267
10  10.691  0.0510  -2.664  0.1993
```

This is Table 5-1 and the first two columns of Table 3-5, digit for digit.
`phi` is the Pearson chi-square over the 55 - 19 = 36 residual degrees of
freedom, as SAS GENMOD gives with `PSCALE`: the monograph's 114.5 (section
5.4.2). `cc.glm` is the underlying GLM fit, coefficients in design order.

```python
print(f"phi {cc.dispersion:.1f} on {cc.glm.df_resid:.0f} degrees of freedom")
```

```text
phi 114.5 on 36 degrees of freedom
```

Rescale the effects so that the `b[j]` add to 1 (Table 3-3): `b[j]` is the
share of the ultimate paid in development year `j`, and `a[k]` the ultimate of
accident year `k`. The reserve is the forecast of the future cells by accident
year:

```python
a = np.exp(est[:K])
b = np.exp([0.0, *est[K:]])
gap = max(abs(g - c) for g, c in zip(cc.reserves[1:], cl.reserve[1:]))
table_3_3 = pd.DataFrame({
    "index": [str(i) for i in range(1, K + 1)],
    "ultimate": a * b.sum(),
    "pattern": b / b.sum(),
    "reserve": cc.reserves,
})
(GT(table_3_3, rowname_col="index")
 .tab_header("Table 3-3: ultimates and payment pattern",
             "Ultimate and reserve by accident year k, pattern by development year j")
 .fmt_integer(["ultimate", "reserve"])
 .fmt_number("pattern", decimals=3))
```

```text
    ultimate  pattern  reserve
1    144,781    0.293        0
2    166,301    0.239    3,398
3    184,501    0.139    8,155
4    201,845    0.106   14,579
5    212,151    0.069   22,645
6    207,340    0.047   31,865
7    205,725    0.035   45,753
8    182,904    0.028   60,093
9    173,225    0.024   80,983
10   149,836    0.020  105,874
```

```python
print(f"GLM total reserve {cc.total_reserve:,.0f}")
print("largest gap to the chain ladder below 1e-6:", gap < 1e-6)
```

```text
GLM total reserve 373,346
largest gap to the chain ladder below 1e-6: True
```

The table is Table 3-3. The last column of Table 3-5 differs by one unit in
four places, because the monograph rounds. The GLM reserves equal the chain
ladder's because maximum likelihood under the ODP makes the fitted total equal
the actual total in each row and column, as the chain ladder does (chapter 3).

The other model of chapter 3, *ODP Mack*, works on the age-to-age factors. Each
observed `f[k, j] - 1` is ODP with mean `f[j] - 1` and weight the cumulative
paid `X[k, j]` at the start of the link. It is a GLM on the links, not the
increments, with one column per development year, a log link and prior weights,
so the general `Glm` fits it (`dispersion="pearson"`):

```python
links = [(k, j) for k in range(K - 1) for j in range(K - 1 - k)]
growth = [cumulative[k][j + 1] / cumulative[k][j] - 1.0 for k, j in links]
weight = [float(cumulative[k][j]) for k, j in links]
mack_design = Design(
    [[1.0 if j == d else 0.0 for _, j in links] for d in range(K - 1)],
    [f"f{d + 1}" for d in range(K - 1)],
    weights=weight,
)
odp_mack = Glm("poisson", "log", dispersion="pearson").fit(mack_design, growth)
print("f - 1", " ".join(f"{math.exp(c):.3f}" for c in odp_mack.coefficients))
```

```text
f - 1 0.815 0.261 0.158 0.088 0.055 0.039 0.030 0.025 0.021
```

These are Table 3-4: the chain-ladder factors minus one. The two models give
the same forecast but not the same prediction error, because their variance
assumptions differ (section 4).

## 3. Prediction error by the delta method

Chapter 5 puts a prediction error on the reserve. The mean square error of
prediction (MSEP) of future cells has two parts:

- Process error is the variance of the cells: `phi` times their mean under the
  ODP.
- Parameter error comes from the estimated coefficients. To first order (the
  delta method), a reserve `R = sum of exp(x' beta)` moves by
  `g' (beta_hat - beta)`, with gradient `g = sum of mu * x` over its cells. Its
  variance is `g' Sigma g`, where `Sigma` is the covariance of the coefficients
  from the GLM.

`OdpGlmFit.prediction_error()` does this (equations 5-11 to 5-13) for each
accident year and the total, for any design. The table splits the RMSEP into
its two parts beside Table 5-4:

```python
pe = cc.prediction_error()  # a row per accident year, then "Total"
TABLE_5_4 = {  # forecast, RMSEP, CVP %
    "1989": (3398, 924, 27.2), "1990": (8155, 1363, 16.7), "1991": (14579, 1775, 12.2),
    "1992": (22645, 2169, 9.6), "1993": (31865, 2523, 7.9), "1994": (45753, 3036, 6.6),
    "1995": (60093, 3577, 6.0), "1996": (80983, 4538, 5.6), "1997": (105874, 6786, 6.4),
    "Total": (373346, 14076, 3.8),
}
rows = []
for i, year in enumerate(pe["origin"][1:], start=1):
    forecast, rmsep, cvp = TABLE_5_4[year]
    rows += [(year, pe["reserve"][i], pe["process_se"][i], pe["parameter_se"][i], pe["rmsep"][i], 100 * pe["cv"][i]),
             (f"{year}, book", forecast, np.nan, np.nan, rmsep, cvp)]
table_5_4 = pd.DataFrame(rows, columns=["year", "forecast", "process_se", "parameter_se", "rmsep", "cvp"])
(GT(table_5_4, rowname_col="year")
 .tab_header("Table 5-4: reserve and prediction error by accident year",
             "Money in the units of the triangle; CVP in percent")
 .cols_label(process_se="process SE", parameter_se="parameter SE", rmsep="RMSEP", cvp="CVP %")
 .fmt_integer(["forecast", "process_se", "parameter_se", "rmsep"])
 .fmt_number("cvp", decimals=1)
 .sub_missing(missing_text=""))
```

```text
             forecast  process SE  parameter SE   RMSEP  CVP %
1989            3,398         624           681     924   27.2
1989, book      3,398                               924   27.2
1990            8,155         966           961   1,363   16.7
1990, book      8,155                             1,363   16.7
1991           14,579       1,292         1,217   1,775   12.2
1991, book     14,579                             1,775   12.2
1992           22,645       1,610         1,453   2,169    9.6
1992, book     22,645                             2,169    9.6
1993           31,865       1,910         1,649   2,523    7.9
1993, book     31,865                             2,523    7.9
1994           45,753       2,289         1,994   3,036    6.6
1994, book     45,753                             3,036    6.6
1995           60,093       2,624         2,432   3,577    6.0
1995, book     60,093                             3,577    6.0
1996           80,983       3,046         3,364   4,538    5.6
1996, book     80,983                             4,538    5.6
1997          105,874       3,482         5,824   6,786    6.4
1997, book    105,874                             6,786    6.4
Total         373,346       6,539        12,465  14,076    3.8
Total, book   373,346                            14,076    3.8
```

Table 5-4 agrees in every number. Parameter error is the larger part of the
total's RMSEP. Adding the years' MSEPs as if they were independent gives:

```python
by_year_msep = [e**2 for e in pe["rmsep"][1:-1]]
print(f"total if the years were independent {math.sqrt(sum(by_year_msep)):,.0f}  (book 10,275)")
```

```text
total if the years were independent 10,275  (book 10,275)
```

This is lower than the total because the years are positively correlated: they
share the `b[j]`, so an error in one moves all years the same way.

## 4. Prediction error by simulation

Section 5.4.2 checks the delta method with a parametric bootstrap: draw the
coefficients from their estimated normal distribution, forecast the future
cells, add process error, and repeat 10,000 times.

`OdpGlmFit.predict_distribution` does this. Its process error on each cell is
`phi * Poisson(mu / phi)`; the monograph does not state its process
distribution. `"normal"` draws the coefficients from a plain normal, as the
monograph does, and `aggregate(["origin"])` adds the cells by accident year.

```python
def by_year_draws(fit, seed, n_sims=10_000):
    """Simulated reserves: one row per accident year 1989 to 1997, one column per draw."""
    draws = fit.predict_distribution(n_sims, seed, "normal").aggregate(["origin"])
    return np.array(draws.draw_matrix()).T


def sim_table(sims, book, title):
    """Mean, RMSEP and CVP of simulated reserves by year and in total, beside the book's."""
    keys = [*YEARS[1:], "Total"]
    rows = []
    for key, s in zip(keys, [*sims, sims.sum(axis=0)]):
        rows += [(str(key), s.mean(), s.std(ddof=1), 100 * s.std(ddof=1) / s.mean()),
                 (f"{key}, book", *book[key])]
    return (GT(pd.DataFrame(rows, columns=["year", "mean", "rmsep", "cvp"]), rowname_col="year")
            .tab_header(title, "Money in the units of the triangle; CVP in percent")
            .cols_label(rmsep="RMSEP", cvp="CVP %")
            .fmt_integer(["mean", "rmsep"])
            .fmt_number("cvp", decimals=1))


TABLE_5_5 = {
    1989: (3476, 937, 27.0), 1990: (8269, 1366, 16.5), 1991: (14738, 1794, 12.2),
    1992: (22776, 2186, 9.6), 1993: (32043, 2525, 7.9), 1994: (45963, 3057, 6.7),
    1995: (60273, 3608, 6.0), 1996: (81249, 4589, 5.6), 1997: (106204, 6831, 6.4),
    "Total": (374992, 14286, 3.8),
}
sim_table(by_year_draws(cc, seed=1), TABLE_5_5, "Table 5-5: the parametric bootstrap, 10,000 draws")
```

```text
                mean   RMSEP  CVP %
1989           3,467     943   27.2
1989, book     3,476     937   27.0
1990           8,290   1,397   16.8
1990, book     8,269   1,366   16.5
1991          14,725   1,792   12.2
1991, book    14,738   1,794   12.2
1992          22,792   2,177    9.6
1992, book    22,776   2,186    9.6
1993          32,037   2,528    7.9
1993, book    32,043   2,525    7.9
1994          45,950   3,101    6.7
1994, book    45,963   3,057    6.7
1995          60,313   3,593    6.0
1995, book    60,273   3,608    6.0
1996          81,254   4,582    5.6
1996, book    81,249   4,589    5.6
1997         106,056   6,849    6.5
1997, book   106,204   6,831    6.4
Total        374,883  14,432    3.8
Total, book  374,992  14,286    3.8
```

Means and RMSEPs agree with Table 5-5 within simulation noise: with 10,000
draws a standard deviation has a relative error of about 0.7%. The total RMSEPs
are 1.0% apart, as two seeds also are. Both are slightly above the delta method.

Both sets of means are *above* the chain ladder's forecasts (374,992 in the
monograph against 373,346). This is not noise: a cell `exp(x' beta)` with
`beta` drawn from a normal has the mean `mu * exp(v / 2)`, where
`v = x' Sigma x` is the variance of the linear predictor (the lognormal bias).
`predictive_means()` gives the simulation's exact expectation for each cell:

```python
print(f"chain ladder {cc.total_reserve:9,.0f}")
print(f"E[simulated] {sum(cc.predictive_means()):9,.0f}   (book's bootstrap mean 374,992)")
```

```text
chain ladder   373,346
E[simulated]   374,994   (book's bootstrap mean 374,992)
```

The monograph puts the higher means down to sampling error, but the bias
explains all of the 1,646 difference, to two units. To simulate around the
chain ladder's forecast, use `"mean_preserving"`: it moves each cell's linear
predictor by `-v / 2`, so the draws average to the fitted mean.

**Other methods for the same question.** The ODP bootstrap of England and
Verrall (`OdpBootstrap`) resamples the model's adjusted Pearson residuals,
draws no parameters from a normal and adds gamma process error. Mack's model is
the distribution-free counterpart of ODP Mack (section 2). The bootstrap's
scale is the same Pearson `phi`:

```python
ob = OdpBootstrap(n_sims=10_000, seed=1).fit(tri, "paid")
boot = np.array(ob.reserves.draw_matrix()).T
mack = Mack().fit(tri, "paid")
print(f"ODP bootstrap scale {ob.scale:.1f}")
```

```text
ODP bootstrap scale 114.5
```

```python
boot_years = [*boot[1:], boot.sum(axis=0)]
vs_mack = pd.DataFrame({
    "year": [str(y) for y in YEARS[1:]] + ["Total"],
    "boot_mean": [s.mean() for s in boot_years],
    "boot_sd": [s.std(ddof=1) for s in boot_years],
    "mack_reserve": [*mack.reserve[1:], mack.total_reserve],
    "mack_se": [*mack.standard_error[1:], mack.total_standard_error],
})
(GT(vs_mack, rowname_col="year")
 .tab_header("ODP bootstrap against Mack", "Money in the units of the triangle; 10,000 draws")
 .cols_label(boot_mean="bootstrap mean", boot_sd="bootstrap SD",
             mack_reserve="Mack reserve", mack_se="Mack SE")
 .fmt_integer(["boot_mean", "boot_sd", "mack_reserve", "mack_se"]))
```

```text
       bootstrap mean  bootstrap SD  Mack reserve  Mack SE
1989            3,383           926         3,398       39
1990            8,140         1,372         8,155       44
1991           14,545         1,769        14,579      410
1992           22,600         2,184        22,645      850
1993           31,822         2,549        31,865    1,364
1994           45,712         3,013        45,753    1,959
1995           60,075         3,624        60,093    2,308
1996           80,887         4,545        80,983    3,179
1997          105,942         6,839       105,874    9,192
Total         373,105        14,194       373,346   10,939
```

```python
import matplotlib.pyplot as plt

total_boot = boot.sum(axis=0)
grid = np.linspace(280_000, 470_000, 300)


def normal_pdf(x, mean, sd):
    return np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * math.sqrt(2 * math.pi))


fig, ax = plt.subplots()
ax.hist(total_boot, bins=60, density=True, color="0.75", label="ODP bootstrap draws")
ax.plot(grid, normal_pdf(grid, mack.total_reserve, mack.total_standard_error), label="normal, Mack SE")
ax.plot(grid, normal_pdf(grid, cc.total_reserve, pe["rmsep"][-1]), label="normal, GLM RMSEP")
ax.set_xticks(range(300_000, 500_000, 50_000))
ax.xaxis.set_major_formatter(lambda x, _: f"{x:,.0f}")
ax.set_xlabel("total reserve (units of the triangle)")
ax.set_ylabel("density")
ax.set_title("Total reserve: the bootstrap is wider than Mack's error")
ax.legend()
```

The bootstrap's total RMSEP is near the parametric bootstrap's. Mack's is about
a quarter lower, and very small for the older years. Mack's variance comes from
the scatter of the age-to-age factors, which is little in the later columns;
the ODP ties each payment's variance to its size through one `phi`. Two models
with the same forecast can have different prediction errors (monograph section
5.4.1), and here the difference is large.

## 5. Residual diagnostics

Chapter 6 checks the fit with residuals. A cell's standardized deviance
residual is its deviance residual over `sqrt(phi * (1 - h))`, with `h` the
leverage (the hat-matrix diagonal). `OdpGlmFit.residuals()` gives these for
each observed cell, with Pearson residuals, leverages and actual-to-fitted
ratios; `residual_summary` groups them by development, accident or calendar
year:

```python
dev = cc.residual_summary("development")
cal = cc.residual_summary("calendar")
print("by development year 1 to 8, spread (sd) of the residuals")
print(" ".join(f"{s:.2f}" for s in dev["sd"][:8]))
print("by calendar year 1988 to 1997, mean residual")
print(" ".join(f"{m:+.2f}" for m in cal["mean"]))
```

```text
by development year 1 to 8, spread (sd) of the residuals
1.42 1.68 0.94 0.67 0.91 0.86 0.75 0.49
by calendar year 1988 to 1997, mean residual
-0.38 -0.13 -0.40 +0.56 +0.42 -0.43 +0.23 -0.27 -0.26 +0.32
```

```python
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.5))
ax1.bar(range(1, 9), dev["sd"][:8], color="0.75")
ax1.axhline(1.0, color="0.4", linestyle=":")
ax1.set_xlabel("development year")
ax1.set_ylabel("sd of standardized residuals")
ax1.set_title("Spread by development year")
ax2.bar(range(1988, 1998), cal["mean"], color="0.75")
ax2.axhline(0.0, color="0.4")
ax2.set_xlabel("calendar year")
ax2.set_ylabel("mean standardized residual")
ax2.set_title("Mean by calendar year")
fig.tight_layout()
```

The spreads use divisor `n - 1`. Two cells have leverage one, the last of 1988
and the first of 1997. Each is the only observation of its parameter, so its
residual is undefined. Development year 10 has no residual and year 9 has two,
too few for a spread. Development years 1 and 2 scatter more than later years
(Figure 6-19), and the calendar-year means change from negative to positive
(Figure 6-20).

Actual and fitted totals agree on every row and column, so one-way comparisons
show no problem. The ratio of actual to fitted for each cell can (Figure 6-22):

```python
def heat_map(fit, title):
    """The actual-to-fitted ratio of every observed cell, in percent, a row per accident year."""
    r = fit.residuals()
    rows = []
    for year in map(str, YEARS):
        ratios = [100 * x for o, x in zip(r["origin"], r["actual_to_fitted"]) if o == year]
        rows.append([year, *ratios, *[np.nan] * (K - len(ratios))])
    columns = [str(d) for d in range(1, K + 1)]
    return (GT(pd.DataFrame(rows, columns=["year", *columns]), rowname_col="year")
            .tab_header(title, "Actual over fitted in percent; columns are development years")
            .fmt_integer(columns)
            .sub_missing(missing_text="")
            .data_color(columns, domain=[80, 120], palette=["firebrick", "white", "steelblue"]))


heat_map(cc, "Figure 6-22: the cross-classified model")
```

```text
        1    2    3    4    5    6    7    8    9   10
1988   98  100  100  104  113   87   96   92  100  100
1989   99   99  106  103   95   95   99  102  100     
1990   96  108  107   91   90  102   92  104          
1991   97  103   96   97  103  111  111               
1992   95  107  100  100   97  100                    
1993   98  105   93  101  104                         
1994  109   91   95  104                              
1995  106   90  105                                   
1996  103   97                                        
1997  100                                             
```

In development year 1 the early accident years are below 100% and the later
ones above; development year 2 shows the opposite. The payment pattern changed
over the accident years: an interaction of row and column, which the
cross-classified model cannot represent. Chapter 7 adds it.

## 6. Fewer parameters

Chapter 7 starts from Figure 7-1: the ten `ln a[k]` of section 2 follow a
parabola in `k`. Model 7-2 replaces them with `a1 * k + a2 * k**2` and keeps
all ten `ln b[j]` free. It has 12 parameters instead of 19. In prospicio it is
the same `fit` with other columns:

```python
quad = fit([k, k * k] + [j == b for b in range(1, K + 1)], ["a1", "a2"] + [f"ln b{b}" for b in range(1, K + 1)])
c = quad.glm.coefficients
level = c[2]  # ln b1 carries the level, as the monograph's quadratic does
print(f"accident year effect {level:.3f} + {c[0]:.4f} k {c[1]:.4f} k^2  (book 10.471 + 0.2001 k - 0.0179 k^2)")
```

```text
accident year effect 10.471 + 0.2001 k -0.0179 k^2  (book 10.471 + 0.2001 k - 0.0179 k^2)
```

The development pattern is Table 7-1:

```python
book_7_1 = [-0.206, -0.750, -1.015, -1.452, -1.830, -2.142, -2.353, -2.514, -2.661]
table_7_1 = pd.DataFrame({
    "j": [str(i) for i in range(2, K + 1)],
    "prospicio": [x - level for x in c[3:]],
    "book": book_7_1,
})
(GT(table_7_1, rowname_col="j")
 .tab_header("Table 7-1: ln b[j] - ln b[1] of model 7-2", "By development year j")
 .fmt_number(["prospicio", "book"], decimals=3))
```

```text
    prospicio    book
2      -0.206  -0.206
3      -0.750  -0.750
4      -1.015  -1.015
5      -1.452  -1.452
6      -1.830  -1.830
7      -2.142  -2.142
8      -2.353  -2.353
9      -2.514  -2.514
10     -2.661  -2.661
```

Model 7-10 (section 7.4, six parameters) replaces the development effects with
a straight line in `j` with a kink at 7.5 and adds a term for development year
2. Model 7-14 (section 7.6, ten parameters) adds the interactions that the heat
map showed: a shift of development years 1 and 2 for accident years 1 to 6, a
trend in accident year for development year 3 and a term for development year
4.

```python
early = k <= 6  # accident years 1 to 6
SMOOTH = [1.0, k, k * k, j - 1, np.maximum(0.0, j - 7.5), j == 2]
smooth = fit(SMOOTH, ["a0", "a1", "a2", "b1", "b2", "c"])
interact = fit(SMOOTH + [j == 4, (j == 1) & early, (j == 2) & early, (j == 3) * k],
               ["a0", "a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2", "d3"])
book_7_3 = [10.469, 0.200, -0.018, -0.358, 0.236, 0.155]
book_7_5 = [10.4900, 0.2066, -0.0183, -0.3685, 0.2720, 0.0375, 0.0528, -0.0671, 0.1273, -0.0113]
missing = [np.nan] * 4  # model 7-10 has no c2 and d1 to d3
table_7_5 = pd.DataFrame({
    "coefficient": ["a0", "a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2", "d3"],
    "m710": [*smooth.glm.coefficients, *missing],
    "t73": [*book_7_3, *missing],
    "m714": interact.glm.coefficients,
    "t75": book_7_5,
})
(GT(table_7_5, rowname_col="coefficient")
 .tab_header("Models 7-10 and 7-14 against Tables 7-3 and 7-5",
             "Model 7-10's c is c1 of model 7-14")
 .cols_label(m710="model 7-10", t73="Table 7-3", m714="model 7-14", t75="Table 7-5")
 .fmt_number(["m710", "m714", "t75"], decimals=4)
 .fmt_number("t73", decimals=3)
 .sub_missing(missing_text=""))
```

```text
    model 7-10  Table 7-3  model 7-14  Table 7-5
a0     10.4687     10.469     10.4904    10.4900
a1      0.2001      0.200      0.2066     0.2066
a2     -0.0179     -0.018     -0.0183    -0.0183
b1     -0.3577     -0.358     -0.3685    -0.3685
b2      0.2356      0.236      0.2720     0.2720
c1      0.1545      0.155      0.0375     0.0375
c2                             0.0528     0.0528
d1                            -0.0671    -0.0671
d2                             0.1273     0.1273
d3                            -0.0113    -0.0113
```

```python
print(f"phi: 7-2 {quad.dispersion:.1f}, 7-10 {smooth.dispersion:.1f}, 7-14 {interact.dispersion:.1f}")
```

```text
phi: 7-2 102.6, 7-10 107.3, 7-14 53.9
```

Both fits agree with the tables. Model 7-10's `c1` is 0.1544995, which rounds
to 0.154 where Table 7-3 prints 0.155 (SAS differs in the seventh decimal).
Table 7-5 prints 10.490 for `a0` with an added trailing zero. Model 7-14's `c1`
is for development year 2 and `c2` for year 4, then `d1` to `d3`, the
monograph's order. `phi` falls from 114.5 to about 54 with the interactions,
the main reason for their lower prediction error.

The heat map of the six-parameter model (Figure 7-5) shows the pattern that the
interactions correct:

```python
heat_map(smooth, "Figure 7-5: the six-parameter model")
```

```text
        1    2    3    4    5    6    7    8    9  10
1988   99  101   98  111  112   84   97   96  100  97
1989   99   99  102  109   93   90   99  106   99    
1990   95  107  102   96   88   97   92  107         
1991   97  103   94  104  102  107  113              
1992   97  108   99  108   97   98                   
1993   97  104   89  106  101                        
1994  110   92   93  112                             
1995  102   87   99                                  
1996  105   98                                       
1997  101                                            
```

## 7. Forecasts and prediction error of the smaller models

Tables 7-2, 7-4 and 7-6 give the monograph's parametric bootstrap of each
model, run as in section 4. The table adds the delta method of section 3 (the
total row of `prediction_error()`), which applies to every design and has no
simulation noise:

```python
BOOK = {  # total forecast and RMSEP of the parametric bootstrap
    "7-2 quadratic": (373641, 13086), "7-10 smooth": (373403, 13248), "7-14 interactions": (371559, 10907),
}
rows = []
for (name, book), model in zip(BOOK.items(), [quad, smooth, interact]):
    total = by_year_draws(model, seed=1).sum(axis=0)
    rows.append((name, model.total_reserve, model.prediction_error()["rmsep"][-1],
                 total.mean(), total.std(ddof=1), *book))
totals = pd.DataFrame(rows, columns=["model", "delta_forecast", "delta_rmsep", "boot_mean",
                                     "boot_rmsep", "book_mean", "book_rmsep"])
(GT(totals, rowname_col="model")
 .tab_header("Total reserve of the smaller models", "Money in the units of the triangle")
 .cols_label(delta_forecast="delta forecast", delta_rmsep="delta RMSEP", boot_mean="bootstrap mean",
             boot_rmsep="bootstrap RMSEP", book_mean="book mean", book_rmsep="book RMSEP")
 .fmt_integer(["delta_forecast", "delta_rmsep", "boot_mean", "boot_rmsep", "book_mean", "book_rmsep"]))
```

```text
                   delta forecast  delta RMSEP  bootstrap mean  bootstrap RMSEP  book mean  book RMSEP
7-2 quadratic             372,532       13,030         373,866           13,131    373,641      13,086
7-10 smooth               373,006       13,215         373,675           13,392    373,403      13,248
7-14 interactions         370,493       11,021         370,965           11,135    371,559      10,907
```

The bootstrap totals are within simulation noise of the monograph, with the
lognormal bias of section 4. As in the monograph, the quadratic accident-year
effect lowers the total RMSEP from about 14,100 to 13,000 with little change in
the forecast, and the smooth development pattern raises it a little. The
interactions lower it to about 11,000 (CV near 3%) because they model the
change in the payment pattern, which the chain ladder treats as noise.

By year, the interaction model (Table 7-6):

```python
TABLE_7_6 = {
    1989: (3630, 569, 15.7), 1990: (8557, 935, 10.9), 1991: (14563, 1203, 8.3),
    1992: (22193, 1418, 6.4), 1993: (32505, 1677, 5.2), 1994: (45771, 2018, 4.4),
    1995: (62998, 2459, 3.9), 1996: (79601, 3079, 3.9), 1997: (101742, 4094, 4.0),
    "Total": (371559, 10907, 2.9),
}
sim_table(by_year_draws(interact, seed=1), TABLE_7_6, "Table 7-6: model 7-14, parametric bootstrap")
```

```text
                mean   RMSEP  CVP %
1989           3,635     560   15.4
1989, book     3,630     569   15.7
1990           8,573     937   10.9
1990, book     8,557     935   10.9
1991          14,576   1,204    8.3
1991, book    14,563   1,203    8.3
1992          22,218   1,418    6.4
1992, book    22,193   1,418    6.4
1993          32,521   1,694    5.2
1993, book    32,505   1,677    5.2
1994          45,714   1,992    4.4
1994, book    45,771   2,018    4.4
1995          63,015   2,423    3.8
1995, book    62,998   2,459    3.9
1996          79,385   3,143    4.0
1996, book    79,601   3,079    3.9
1997         101,327   4,191    4.1
1997, book   101,742   4,094    4.0
Total        370,965  11,135    3.0
Total, book  371,559  10,907    2.9
```

Each year agrees within noise. The forecast for 1997, whose only payment is in
development year 1, where the pattern changed, falls from 105,874 to about
101,300 (101,742 in Table 7-6).

## 8. Choosing between the models

Table 7-7 compares the four models by AIC, BIC and generalized
cross-validation (GCV). The monograph's log-likelihood is the Poisson kernel
`sum(y log(mu) - mu) / phi`, with `phi` held at the interaction model's value
for all models, so the comparison measures fit, not a change of scale. GCV
(section 4.4) is `n * RSS / (n - p)**2`, with `RSS` the plain sum of squared
differences between actual and fitted.

```python
yy = np.array(cc.residuals()["actual"])  # the 55 observed increments
n = len(yy)
phi_fixed = interact.dispersion
TABLE_7_7 = {  # AIC, BIC, GCV
    "chain ladder": (-509392, -509354, 6685428), "7-2 quadratic": (-509400, -509376, 5075351),
    "7-10 smooth": (-509397, -509385, 4311874), "7-14 interactions": (-509441, -509421, 1733202),
}
rows = []
for (name, book), model in zip(TABLE_7_7.items(), [cc, quad, smooth, interact]):
    mu, p = np.array(model.glm.fitted), len(model.glm.coefficients)
    loglik = np.sum(yy * np.log(mu) - mu) / phi_fixed
    gcv = n * np.sum((yy - mu) ** 2) / (n - p) ** 2
    rows += [(name, -2 * loglik + 2 * p, -2 * loglik + p * math.log(n), gcv), (f"{name}, book", *book)]
criteria = pd.DataFrame(rows, columns=["model", "aic", "bic", "gcv"])
(GT(criteria, rowname_col="model")
 .tab_header("Table 7-7: AIC, BIC and GCV", "Lower is better")
 .cols_label(aic="AIC", bic="BIC", gcv="GCV")
 .fmt_integer(["aic", "bic", "gcv"]))
```

```text
                              AIC       BIC        GCV
chain ladder             -509,391  -509,353  6,685,459
chain ladder, book       -509,392  -509,354  6,685,428
7-2 quadratic            -509,399  -509,375  5,075,358
7-2 quadratic, book      -509,400  -509,376  5,075,351
7-10 smooth              -509,396  -509,384  4,311,876
7-10 smooth, book        -509,397  -509,385  4,311,874
7-14 interactions        -509,440  -509,420  1,733,200
7-14 interactions, book  -509,441  -509,421  1,733,202
```

GCV agrees to five significant figures. AIC and BIC are 0.7 to 1.4 above the
monograph's, a difference of level only: differences between models agree
within the printed rounding, and every criterion ranks the models as the
monograph does, with the interaction model best.

The level is sensitive: the log-likelihood is about 254,700 in these units, so
a relative difference of `2e-6` in `phi`, or a looser convergence test
elsewhere, moves the AIC by one. The code uses each fit's `glm.fitted` and
`glm.coefficients` because the monograph defines its own criteria; the GLM's
`aic` is the usual Poisson AIC and answers a different question.

## 9. A gamma cross-classified model

Chapter 2 of the monograph puts the ODP in the Tweedie family. Section 6.5
gives a gamma error (variance proportional to the mean squared) as the
alternative to try when the residual spread grows with the fitted value. The
monograph prints no gamma results for this triangle. In prospicio the fit needs
one argument, `family="gamma"` (or `family="tweedie"` with a `power` between 1
and 2). This uses the default cross-classified design:

```python
gamma = OdpGlm(family="gamma").fit(tri, "paid")
print(f"gamma phi {gamma.dispersion:.4f}")
```

```text
gamma phi 0.0047
```

```python
by_gamma = pd.DataFrame({
    "year": [str(y) for y in YEARS[1:]] + ["Total"],
    "chain_ladder": [*cc.reserves[1:], cc.total_reserve],
    "gamma": [*gamma.reserves[1:], gamma.total_reserve],
})
(GT(by_gamma, rowname_col="year")
 .tab_header("Reserves of the cross-classified model", "Poisson (the chain ladder) against gamma")
 .cols_label(chain_ladder="chain ladder")
 .fmt_integer(["chain_ladder", "gamma"]))
```

```text
       chain ladder    gamma
1989          3,398    3,423
1990          8,155    8,134
1991         14,579   15,086
1992         22,645   22,667
1993         31,865   31,820
1994         45,753   45,434
1995         60,093   60,216
1996         80,983   80,829
1997        105,874  105,857
Total       373,346  373,466
```

The gamma fit weights each cell's relative error equally and does not keep the
row and column totals, so its forecasts differ slightly from the chain
ladder's. `gamma.prediction_error()` applies the delta method of section 3 with
process variance `phi * mu**2`.

## 10. What matched

Every table and figure above agrees with the monograph to the printed digits,
apart from these known differences:

- The parametric bootstraps (Tables 5-5, 7-2, 7-4 and 7-6) differ by
  simulation noise and by the process distribution, which the monograph does
  not state. Their means have the lognormal bias.
- The AIC and BIC of Table 7-7 differ by about one unit in 509,000, from `phi`
  or convergence. GCV agrees to five figures.
- Three renormalised ultimates of Table 3-5 are one unit different from Table
  3-3, because the monograph rounds.
