# Stochastic loss reserving with GLMs: Taylor and McGuire reproduced

CAS Monograph 3 is *Stochastic Loss Reserving Using Generalized Linear Models*
(Greg Taylor and Gráinne McGuire, 2016). It shows that the chain ladder is a
generalized linear model (GLM). It uses this fact to put a prediction error on
the reserve. Then it checks the model with residuals and improves it with fewer
parameters and some interactions. All examples use one triangle. This guide
runs the examples in prospicio and prints the monograph's numbers next to
prospicio's.

The text of the monograph is the authors'. This guide explains it in new words
and recomputes the numbers. Table and figure numbers are the monograph's. Money
is in thousands of dollars, as in the monograph. To select a reserving method
in general, read [`reserving-methods.md`](reserving-methods.md).

Two modules do the work:

- `prospicio.reserving` has the chain ladder, Mack, the ODP bootstrap and the
  ODP GLM (`OdpGlm`) on a `Triangle`. `OdpGlm` fits the monograph's models to
  the incremental cells of the triangle. It gives the analytic prediction
  error, simulated reserves and residual diagnostics.
- `prospicio.models` has the general GLM (`Glm`, `Design`). `Design` holds the
  design columns. `Glm` fits the ODP Mack model of section 2, which is the one
  model that does not use the cells of the triangle.

By default, `OdpGlm` fits the cross-classified model. `OdpGlm.cells` lists the
cells of the triangle. You can write any other design on these cells and give
it to `fit(..., design=)`. Each model in chapter 7 of the monograph is the same
call with a different design.

## 1. The data and the chain ladder

The example of the monograph (Table 1-1) is the incremental paid loss triangle
of a New Jersey workers compensation insurer. It has accident years 1988 to
1997 and comes from the database of Meyers and Shi (2011). The authors chose it
because its age-to-age factors are almost constant down each column. Thus the
chain ladder fits it well.

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

The volume-weighted factors are Table 3-1 (1.815, 1.261, 1.158, 1.088, 1.055,
1.039, 1.030, 1.025, 1.021). The reserves by accident year are the last column
of Table 3-2: 3,398 for 1989 up to 105,874 for 1997, with a total of 373,346.
They agree to the unit.

## 2. The chain ladder as a GLM

Chapter 3 of the monograph gives two models that both reproduce the forecasts
of the chain ladder. The monograph uses the *cross-classified* ODP model from
then on. In this model, each incremental payment `Y[k, j]` is over-dispersed
Poisson (ODP). Its mean is `a[k] * b[j]`, a row effect times a column effect.
Its variance is `phi` times the mean.

On the log scale the mean is `ln a[k] + ln b[j]`. This is a GLM with a log link
and one indicator column for each accident year and each development year. The
design must drop one column effect to be full rank. The monograph sets
`ln b[1] = 0`, so its accident-year terms carry the level.

`OdpGlm().fit(tri, "paid")` fits this model with an intercept, and with origin
and development factors measured from the first of each. The monograph codes
the model with one `ln a[k]` for each accident year and no intercept. To print
this coding, write that design on the cells and give it to `fit`. `OdpGlm.cells` gives the 100
cells of the triangle: 55 observed cells and 45 cells to forecast. The cells
are in the order that the rows of a design must follow. Each cell has its
accident year and development year as a 0-based index:

```python
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
print(" j or k   ln a    s.e.    ln b    s.e.")
print(f"{1:6d} {est[0]:7.3f} {se[0]:7.4f}   0.000")
for i in range(1, K):
    print(f"{i + 1:6d} {est[i]:7.3f} {se[i]:7.4f} {est[K + i - 1]:7.3f} {se[K + i - 1]:7.4f}")
print(f"phi {cc.dispersion:.1f} on {cc.glm.df_resid:.0f} degrees of freedom")
```

```text
 j or k   ln a    s.e.    ln b    s.e.
     1  10.657  0.0316   0.000
     2  10.795  0.0299  -0.205  0.0228
     3  10.899  0.0289  -0.747  0.0282
     4  10.989  0.0281  -1.017  0.0328
     5  11.039  0.0278  -1.452  0.0421
     6  11.016  0.0285  -1.833  0.0547
     7  11.008  0.0295  -2.140  0.0715
     8  10.891  0.0327  -2.348  0.0931
     9  10.836  0.0367  -2.513  0.1267
    10  10.691  0.0510  -2.664  0.1993
phi 114.5 on 36 degrees of freedom
```

This is Table 5-1 (and the first two columns of Table 3-5) digit for digit.
`ln a` goes from 10.657 to 10.691 and `ln b` from -0.205 to -2.664. The
standard errors go from 0.0316 and 0.0228 up to 0.0510 and 0.1993. The scale
parameter is the 114.5 of the monograph (section 5.4.2).

`OdpGlm` makes the Poisson over-dispersed with the Pearson dispersion. `phi` is
the Pearson chi-square divided by the 55 - 19 = 36 residual degrees of freedom.
SAS GENMOD gives the same value with its `PSCALE` option. `cc.glm` is the
underlying GLM fit, with the coefficients in the order of the design.

The monograph rescales the effects so that the `b[j]` add to 1 (Table 3-3).
Then `b[j]` is the share of the ultimate paid in development year `j`, and
`a[k]` is the ultimate of accident year `k`:

```python
a = np.exp(est[:K])
b = np.exp([0.0, *est[K:]])
print("a (ultimates)", [round(x) for x in a * b.sum()])
print("b (pattern)  ", " ".join(f"{x:.3f}" for x in b / b.sum()))
print("GLM reserves ", [round(r) for r in cc.reserves[1:]], round(cc.total_reserve))
print("largest gap to the chain ladder below 1e-6:", max(abs(g - c) for g, c in zip(cc.reserves[1:], cl.reserve[1:])) < 1e-6)
```

```text
a (ultimates) [144781, 166301, 184501, 201845, 212151, 207340, 205725, 182904, 173225, 149836]
b (pattern)   0.293 0.239 0.139 0.106 0.069 0.047 0.035 0.028 0.024 0.020
GLM reserves  [3398, 8155, 14579, 22645, 31865, 45753, 60093, 80983, 105874] 373346
largest gap to the chain ladder below 1e-6: True
```

The ultimates are the `a` column of Table 3-3 (144,781, 166,301, ...,
149,836). The last column of Table 3-5 is one unit different in four places,
because the monograph rounds. The pattern is the `b` column (0.293, 0.239,
0.139, ..., 0.020).

The fit's `reserves` are the forecasts of the 45 future cells, added by
accident year. They are equal to the reserves of the chain ladder, to rounding
noise. Thus the GLM and the chain ladder are the same estimator here. Chapter 3
of the monograph proves this equivalence. Maximum likelihood under the ODP makes
the fitted total equal to the actual total for each row and each column. The
chain ladder keeps the same marginal totals.

The other model of chapter 3 is the *ODP Mack* model. It uses the age-to-age
factors. Each observed factor minus one, `f[k, j] - 1`, is ODP with mean
`f[j] - 1`. Its weight is the cumulative paid `X[k, j]` at the start of the
link. This is also a GLM, with one column for each development year, a log link
and prior weights. Its cells are the age-to-age links, not the increments of
the triangle. Thus the general `Glm` fits it, and `dispersion="pearson"` makes
the Poisson over-dispersed:

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

These are Table 3-4 (0.815, 0.261, 0.158, ..., 0.021): the chain-ladder
factors minus one. A mean of the factors, weighted by the cumulative values, is
the volume-weighted factor. The two models give the same forecast. They do not
give the same prediction error, because their variance assumptions are
different (section 4).

## 3. Prediction error by the delta method

Chapter 5 puts a prediction error on the reserve. The mean square error of
prediction (MSEP) of a sum of future cells has two parts:

- Process error is the variance of the cells. Under the ODP it is `phi` times
  their mean.
- Parameter error comes from the estimated coefficients. To first order (the
  delta method), a reserve `R = sum of exp(x' beta)` moves by
  `g' (beta_hat - beta)`, with gradient `g = sum of mu * x` over its cells.
  Its variance is `g' Sigma g`, where `Sigma` is the covariance of the
  coefficients from the GLM.

Equations (5-11) to (5-13) of the monograph do this calculation in matrix form.
`OdpGlmFit.prediction_error()` does it for each accident year and for the
total, for any design.

```python
pe = cc.prediction_error()  # a row per accident year, then "Total"
TABLE_5_4 = {  # forecast, RMSEP, CVP %
    "1989": (3398, 924, 27.2), "1990": (8155, 1363, 16.7), "1991": (14579, 1775, 12.2),
    "1992": (22645, 2169, 9.6), "1993": (31865, 2523, 7.9), "1994": (45753, 3036, 6.6),
    "1995": (60093, 3577, 6.0), "1996": (80983, 4538, 5.6), "1997": (105874, 6786, 6.4),
    "Total": (373346, 14076, 3.8),
}
print("year    forecast  RMSEP   CVP |  book: forecast  RMSEP   CVP")
for key, r, rmsep, cv in zip(pe["origin"][1:], pe["reserve"][1:], pe["rmsep"][1:], pe["cv"][1:]):
    f, e, c = TABLE_5_4[key]
    print(f"{key:6s} {r:9,.0f} {rmsep:6,.0f} {100 * cv:5.1f} |  {f:12,} {e:6,} {c:5.1f}")
```

```text
year    forecast  RMSEP   CVP |  book: forecast  RMSEP   CVP
1989       3,398    924  27.2 |         3,398    924  27.2
1990       8,155  1,363  16.7 |         8,155  1,363  16.7
1991      14,579  1,775  12.2 |        14,579  1,775  12.2
1992      22,645  2,169   9.6 |        22,645  2,169   9.6
1993      31,865  2,523   7.9 |        31,865  2,523   7.9
1994      45,753  3,036   6.6 |        45,753  3,036   6.6
1995      60,093  3,577   6.0 |        60,093  3,577   6.0
1996      80,983  4,538   5.6 |        80,983  4,538   5.6
1997     105,874  6,786   6.4 |       105,874  6,786   6.4
Total    373,346 14,076   3.8 |       373,346 14,076   3.8
```

All numbers in Table 5-4 agree. The RMSEP of the total, 14,076, is a CV of
3.8%. The parameter error is the larger part:

```python
print(f"process   {pe['process_se'][-1]:7,.0f}")
print(f"parameter {pe['parameter_se'][-1]:7,.0f}")
by_year_msep = [e**2 for e in pe["rmsep"][1:-1]]
print(f"total if the years were independent {math.sqrt(sum(by_year_msep)):7,.0f}  (book 10,275)")
```

```text
process     6,539
parameter  12,465
total if the years were independent  10,275  (book 10,275)
```

The monograph adds the MSEPs of the accident years as if the years were
independent. This gives 10,275 instead of 14,076, and prospicio gives the same
value. The difference is the positive correlation between years. The years
share the development-year parameters, so an error in a `b[j]` moves the
forecasts of all years in the same direction.

## 4. Prediction error by simulation

Section 5.4.2 checks the delta method with a parametric bootstrap. The
procedure is:

1. Draw the coefficients from their estimated normal distribution.
2. Forecast the future cells with these coefficients.
3. Add process error.
4. Do steps 1 to 3 10,000 times.

`OdpGlmFit.predict_distribution` does this procedure. Its process error on each
cell is `phi * Poisson(mu / phi)`. The monograph does not state which process
distribution it used. `aggregate(["origin"])` adds the cells by accident year.
`"normal"` gives plain normal draws of the coefficients, as in the monograph.
The seed sets the draws.

```python
def by_year_draws(fit, seed, n_sims=10_000):
    """Simulated reserves: one row per accident year 1989 to 1997, one column per draw."""
    draws = fit.predict_distribution(n_sims, seed, "normal").aggregate(["origin"])
    return np.array(draws.draw_matrix()).T


def show(sims, book):
    print("year    mean    RMSEP   CVP |  book: mean   RMSEP   CVP")
    for key, s in [*zip(YEARS[1:], sims), ("Total", sims.sum(axis=0))]:
        m, e = s.mean(), s.std(ddof=1)
        bm, be, bc = book[key]
        print(f"{key!s:6s} {m:8,.0f} {e:7,.0f} {100 * e / m:5.1f} |  {bm:9,} {be:7,} {bc:5.1f}")


TABLE_5_5 = {
    1989: (3476, 937, 27.0), 1990: (8269, 1366, 16.5), 1991: (14738, 1794, 12.2),
    1992: (22776, 2186, 9.6), 1993: (32043, 2525, 7.9), 1994: (45963, 3057, 6.7),
    1995: (60273, 3608, 6.0), 1996: (81249, 4589, 5.6), 1997: (106204, 6831, 6.4),
    "Total": (374992, 14286, 3.8),
}
show(by_year_draws(cc, seed=1), TABLE_5_5)
```

```text
year    mean    RMSEP   CVP |  book: mean   RMSEP   CVP
1989      3,467     943  27.2 |      3,476     937  27.0
1990      8,290   1,397  16.8 |      8,269   1,366  16.5
1991     14,725   1,792  12.2 |     14,738   1,794  12.2
1992     22,792   2,177   9.6 |     22,776   2,186   9.6
1993     32,037   2,528   7.9 |     32,043   2,525   7.9
1994     45,950   3,101   6.7 |     45,963   3,057   6.7
1995     60,313   3,593   6.0 |     60,273   3,608   6.0
1996     81,254   4,582   5.6 |     81,249   4,589   5.6
1997    106,056   6,849   6.5 |    106,204   6,831   6.4
Total   374,883  14,432   3.8 |    374,992  14,286   3.8
```

The means and RMSEPs agree with Table 5-5 within simulation noise. With 10,000
draws, a standard deviation has a relative error of about 0.7%. The total RMSEP
is 14,286 in the monograph and 14,432 in prospicio (1.0% apart). Two runs with
different seeds show a difference of this size. Both RMSEPs are slightly above
the delta method.

Both sets of means are *above* the forecasts of the chain ladder (374,992 in
the monograph against 373,346). This is not noise. When you draw `beta` from a
normal and calculate `exp(x' beta)`, each cell has the mean `mu * exp(v / 2)`.
Here `v = x' Sigma x` is the variance of the linear predictor. This is the
lognormal bias. You can calculate the expected value of the simulation exactly.
`predictive_means()` gives it for each cell:

```python
print(f"chain ladder {cc.total_reserve:9,.0f}")
print(f"E[simulated] {sum(cc.predictive_means()):9,.0f}   (book's bootstrap mean 374,992)")
```

```text
chain ladder   373,346
E[simulated]   374,994   (book's bootstrap mean 374,992)
```

The monograph says that the higher means come from sampling error. But the bias
accounts for all of the 1,646 difference, to two units. The mean of the
bootstrap in the monograph is within noise of this expectation. To simulate
around the forecast of the chain ladder, use `"mean_preserving"`. It moves the
linear predictor of each cell by `-v / 2`, so that the draws of the cell
average to its fitted mean.

**Other methods for the same question.** The ODP bootstrap of England and
Verrall (`OdpBootstrap`) resamples the adjusted Pearson residuals of the model.
It does not draw parameters from a normal, and it adds gamma process error.
Mack's model is the distribution-free equivalent of the ODP Mack model of
section 2:

```python
ob = OdpBootstrap(n_sims=10_000, seed=1).fit(tri, "paid")
boot = np.array(ob.reserves.draw_matrix()).T
mack = Mack().fit(tri, "paid")
print(f"ODP bootstrap scale {ob.scale:.1f}")
print(f"ODP bootstrap total mean {boot.sum(axis=0).mean():9,.0f}  RMSEP {boot.sum(axis=0).std(ddof=1):7,.0f}")
print(f"Mack total               {mack.total_reserve:9,.0f}  RMSEP {mack.total_standard_error:7,.0f}")
print("Mack by year", [round(s) for s in mack.standard_error[1:]])
```

```text
ODP bootstrap scale 114.5
ODP bootstrap total mean   373,105  RMSEP  14,194
Mack total                 373,346  RMSEP  10,939
Mack by year [39, 44, 410, 850, 1364, 1959, 2308, 3179, 9192]
```

The scale of the ODP bootstrap is the same Pearson `phi`. Its total RMSEP is
near the RMSEP of the parametric bootstrap. The total standard error of Mack is
approximately a quarter lower. Its spread across years is also very different:
very small for the older years and larger for 1997.

The variance of Mack comes from the scatter of the age-to-age factors in each
column. The authors chose this triangle for its stable factors, and the factors
in the later columns have very little scatter. The ODP connects the variance of
each payment to its size through one `phi`. The monograph (section 5.4.1) shows
that two models with the same forecast can have different prediction errors.
This triangle shows how large that difference can be.

## 5. Residual diagnostics

Chapter 6 checks the fit with residuals. The standardized deviance residual of
a cell is its deviance residual divided by `sqrt(phi * (1 - h))`. Here `h` is
the leverage of the cell, the diagonal of the hat matrix of the GLM.
`OdpGlmFit.residuals()` gives these residuals for each observed cell. It also
gives the deviance and Pearson residuals, the leverages and the ratio of actual
to fitted. `residual_summary` groups the standardized deviance residuals by
development year, accident year or calendar year:

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

The spreads are sample standard deviations, with divisor `n - 1`. Two cells
have leverage one: the last year of 1988 and the first year of 1997. Each of
these cells is the only observation of its parameter. Thus its fitted value is
equal to its actual value, and its residual is not defined. Development year 10
has no residual, and year 9 has two, which is not sufficient to measure a
spread.

In the other cells, development years 1 and 2 have more scatter than the later
years. The monograph shows this pattern in Figure 6-19. The calendar-year means
change from negative to positive, which is the pattern of Figure 6-20.

The model has a parameter for each row and each column. Thus the actual and
fitted totals agree on each row and column, and these one-way comparisons
cannot show a problem. The ratio of actual to fitted for each cell can show it:

```python
def heat_map(fit):
    """The actual-to-fitted ratio of every observed cell, in percent, a row per accident year."""
    r = fit.residuals()
    for year in map(str, YEARS):
        print(year, " ".join(f"{round(100 * x):4d}" for o, x in zip(r["origin"], r["actual_to_fitted"]) if o == year))


heat_map(cc)
```

```text
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

These are the percentages of the heat map of the monograph, Figure 6-22. Read
down the first two columns. In development year 1, the early accident years are
below 100% and the later years are above 100%. Development year 2 shows the
opposite. The payment pattern changed over the accident years. This is an
interaction between row and column, which a cross-classified model cannot
show. Chapter 7 adds it.

## 6. Fewer parameters

Chapter 7 starts from Figure 7-1: the ten `ln a[k]` of section 2 follow a
parabola in `k`. Model 7-2 of the monograph replaces them with
`a1 * k + a2 * k**2` and keeps all ten `ln b[j]` free. This model has 12
parameters instead of 19. In prospicio it is the same `fit` with different
columns:

```python
quad = fit([k, k * k] + [j == b for b in range(1, K + 1)], ["a1", "a2"] + [f"ln b{b}" for b in range(1, K + 1)])
c = quad.glm.coefficients
level = c[2]  # ln b1 carries the level, as the monograph's quadratic does
print(f"accident year effect {level:.3f} + {c[0]:.4f} k {c[1]:.4f} k^2  (book 10.471 + 0.2001 k - 0.0179 k^2)")
book_7_1 = [-0.206, -0.750, -1.015, -1.452, -1.830, -2.142, -2.353, -2.514, -2.661]
print("ln b - ln b1", " ".join(f"{x - level:.3f}" for x in c[3:]))
print("book        ", " ".join(f"{x:.3f}" for x in book_7_1))
```

```text
accident year effect 10.471 + 0.2001 k -0.0179 k^2  (book 10.471 + 0.2001 k - 0.0179 k^2)
ln b - ln b1 -0.206 -0.750 -1.015 -1.452 -1.830 -2.142 -2.353 -2.514 -2.661
book         -0.206 -0.750 -1.015 -1.452 -1.830 -2.142 -2.353 -2.514 -2.661
```

The quadratic and the development pattern are those of Table 7-1. The next two
models are:

- Model 7-10 (section 7.4, six parameters) replaces the development effects
  with a straight line in `j` with a kink at 7.5. It adds a separate term for
  development year 2.
- Model 7-14 (section 7.6, ten parameters) adds the interactions that the heat
  map showed. It shifts development years 1 and 2 for accident years 1 to 6. It
  adds a trend in accident year for development year 3 and a term for
  development year 4.

```python
early = k <= 6  # accident years 1 to 6
SMOOTH = [1.0, k, k * k, j - 1, np.maximum(0.0, j - 7.5), j == 2]
smooth = fit(SMOOTH, ["a0", "a1", "a2", "b1", "b2", "c"])
interact = fit(SMOOTH + [j == 4, (j == 1) & early, (j == 2) & early, (j == 3) * k],
               ["a0", "a1", "a2", "b1", "b2", "c1", "c2", "d1", "d2", "d3"])
book_7_3 = [10.469, 0.200, -0.018, -0.358, 0.236, 0.155]
book_7_5 = [10.4900, 0.2066, -0.0183, -0.3685, 0.2720, 0.0375, 0.0528, -0.0671, 0.1273, -0.0113]
print("model 7-10  ", " ".join(f"{x:7.4f}" for x in smooth.glm.coefficients))
print("Table 7-3   ", " ".join(f"{x:7.3f}" for x in book_7_3))
print("model 7-14  ", " ".join(f"{x:7.4f}" for x in interact.glm.coefficients))
print("Table 7-5   ", " ".join(f"{x:7.4f}" for x in book_7_5))
print(f"phi: 7-2 {quad.dispersion:.1f}, 7-10 {smooth.dispersion:.1f}, 7-14 {interact.dispersion:.1f}")
```

```text
model 7-10   10.4687  0.2001 -0.0179 -0.3577  0.2356  0.1545
Table 7-3     10.469   0.200  -0.018  -0.358   0.236   0.155
model 7-14   10.4904  0.2066 -0.0183 -0.3685  0.2720  0.0375  0.0528 -0.0671  0.1273 -0.0113
Table 7-5    10.4900  0.2066 -0.0183 -0.3685  0.2720  0.0375  0.0528 -0.0671  0.1273 -0.0113
phi: 7-2 102.6, 7-10 107.3, 7-14 53.9
```

Both models agree. In model 7-10, `c` is 0.1544995, which is a very small
amount below 0.1545. Thus it rounds to 0.154, but Table 7-3 prints 0.155. The
estimate from SAS needs to be different only in the seventh decimal. Table 7-5
prints 10.490 for `a0` with an added trailing zero.

The 7-14 coefficients are in the order of the monograph: `c1` for development
year 2, `c2` for year 4, then `d1` to `d3`. The scale parameter decreases from
114.5 for the chain ladder to approximately 54 with the interactions. This is
the main cause of the lower prediction error below.

The heat map of the six-parameter model is Figure 7-5 of the monograph. It
shows the pattern that the interactions correct:

```python
heat_map(smooth)
```

```text
1988   99  101   98  111  112   84   97   96  100   97
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

Tables 7-2, 7-4 and 7-6 of the monograph give the parametric bootstrap of each
model, run as in section 4. The delta method of section 3 applies to all
designs, so the table also prints it (the total row of `prediction_error()`).
It has no cost and no simulation noise.

```python
BOOK = {  # total forecast and RMSEP of the parametric bootstrap
    "7-2 quadratic": (373641, 13086), "7-10 smooth": (373403, 13248), "7-14 interactions": (371559, 10907),
}
print("model              delta: forecast  RMSEP | bootstrap: mean  RMSEP |  book: mean  RMSEP")
for (name, book), model in zip(BOOK.items(), [quad, smooth, interact]):
    rmsep = model.prediction_error()["rmsep"][-1]
    total = by_year_draws(model, seed=1).sum(axis=0)
    print(f"{name:18s} {model.total_reserve:15,.0f} {rmsep:6,.0f} | {total.mean():15,.0f} {total.std(ddof=1):6,.0f} |"
          f" {book[0]:10,} {book[1]:6,}")
```

```text
model              delta: forecast  RMSEP | bootstrap: mean  RMSEP |  book: mean  RMSEP
7-2 quadratic              372,532 13,030 |         373,866 13,131 |    373,641 13,086
7-10 smooth                373,006 13,215 |         373,675 13,392 |    373,403 13,248
7-14 interactions          370,493 11,021 |         370,965 11,135 |    371,559 10,907
```

The bootstrap totals are within simulation noise of the monograph. The means
have the same lognormal bias as in section 4. The results are those of the
monograph:

- The quadratic accident-year effect decreases the total RMSEP from
  approximately 14,100 to 13,000. It almost does not change the forecast.
- The smooth development pattern increases the RMSEP a little.
- The interactions decrease it to approximately 11,000, a CV near 3%. They
  model the change in the payment pattern, which the chain ladder treats as
  noise.

```python
interact_by_year = by_year_draws(interact, seed=1)
TABLE_7_6 = {
    1989: (3630, 569, 15.7), 1990: (8557, 935, 10.9), 1991: (14563, 1203, 8.3),
    1992: (22193, 1418, 6.4), 1993: (32505, 1677, 5.2), 1994: (45771, 2018, 4.4),
    1995: (62998, 2459, 3.9), 1996: (79601, 3079, 3.9), 1997: (101742, 4094, 4.0),
    "Total": (371559, 10907, 2.9),
}
show(interact_by_year, TABLE_7_6)
```

```text
year    mean    RMSEP   CVP |  book: mean   RMSEP   CVP
1989      3,635     560  15.4 |      3,630     569  15.7
1990      8,573     937  10.9 |      8,557     935  10.9
1991     14,576   1,204   8.3 |     14,563   1,203   8.3
1992     22,218   1,418   6.4 |     22,193   1,418   6.4
1993     32,521   1,694   5.2 |     32,505   1,677   5.2
1994     45,714   1,992   4.4 |     45,771   2,018   4.4
1995     63,015   2,423   3.8 |     62,998   2,459   3.9
1996     79,385   3,143   4.0 |     79,601   3,079   3.9
1997    101,327   4,191   4.1 |    101,742   4,094   4.0
Total   370,965  11,135   3.0 |    371,559  10,907   2.9
```

For each year, the interaction model agrees with Table 7-6 within noise. Its
forecasts are different from those of the chain ladder where the interactions
apply. The only payment of 1997 is in development year 1, where the pattern
changed. Its forecast decreases from 105,874 to approximately 101,300 here
(101,742 in Table 7-6).

## 8. Choosing between the models

Table 7-7 compares the four models by AIC, BIC and generalized cross-validation
(GCV). A footnote of the monograph defines its log-likelihood as the Poisson
kernel `sum(y log(mu) - mu) / phi`. It holds `phi` at the value of the
interaction model for all models. Thus the comparison measures fit, not a
change of scale. GCV (section 4.4) is `n * RSS / (n - p)**2`, where `RSS` is
the plain sum of squared differences between actual and fitted.

```python
yy = np.array(cc.residuals()["actual"])  # the 55 observed increments
n = len(yy)
phi_fixed = interact.dispersion
TABLE_7_7 = {  # AIC, BIC, GCV
    "chain ladder": (-509392, -509354, 6685428), "7-2 quadratic": (-509400, -509376, 5075351),
    "7-10 smooth": (-509397, -509385, 4311874), "7-14 interactions": (-509441, -509421, 1733202),
}
print("model                  AIC        BIC       GCV |  book:  AIC        BIC       GCV")
for (name, book), model in zip(TABLE_7_7.items(), [cc, quad, smooth, interact]):
    mu, p = np.array(model.glm.fitted), len(model.glm.coefficients)
    loglik = np.sum(yy * np.log(mu) - mu) / phi_fixed
    gcv = n * np.sum((yy - mu) ** 2) / (n - p) ** 2
    print(f"{name:18s} {-2 * loglik + 2 * p:10,.0f} {-2 * loglik + p * math.log(n):10,.0f} {gcv:9,.0f} |"
          f" {book[0]:10,} {book[1]:10,} {book[2]:9,}")
```

```text
model                  AIC        BIC       GCV |  book:  AIC        BIC       GCV
chain ladder         -509,391   -509,353 6,685,459 |   -509,392   -509,354 6,685,428
7-2 quadratic        -509,399   -509,375 5,075,358 |   -509,400   -509,376 5,075,351
7-10 smooth          -509,396   -509,384 4,311,876 |   -509,397   -509,385 4,311,874
7-14 interactions    -509,440   -509,420 1,733,200 |   -509,441   -509,421 1,733,202
```

GCV agrees to five significant figures. The AIC and BIC are each 0.7 to 1.4
above the values of the monograph. This difference is in the level, not in the
comparison. The differences between models agree within the rounding of the
printed values. Thus all criteria rank the models as the monograph does, with
the interaction model best.

The level is sensitive. The log-likelihood is approximately 254,700 in these
units. A relative difference of `2e-6` in `phi`, or in fitted values from
software with a looser convergence test, moves the AIC by one. The code
calculates the criteria from the GLM of each fit (`fit.glm.fitted` and
`fit.glm.coefficients`), because the monograph uses its own definition. The
`aic` of the GLM is the usual Poisson AIC, which answers a different question.

## 9. A gamma cross-classified model

Chapter 2 of the monograph puts the ODP in the Tweedie family. Section 6.5
gives a gamma error (variance proportional to the mean squared) as an
alternative. It is the model to try when the spread of the residuals increases
with the fitted value.
The monograph prints no gamma results for this triangle. In prospicio the fit
needs one argument, `family="gamma"` (or `family="tweedie"` with a `power`
between 1 and 2). Here it uses the default cross-classified design:

```python
gamma = OdpGlm(family="gamma").fit(tri, "paid")
print(f"gamma phi {gamma.dispersion:.4f}")
print("gamma reserves", [round(r) for r in gamma.reserves[1:]])
print(f"gamma total {gamma.total_reserve:,.0f}  (chain ladder {cc.total_reserve:,.0f})")
```

```text
gamma phi 0.0047
gamma reserves [3423, 8134, 15086, 22667, 31820, 45434, 60216, 80829, 105857]
gamma total 373,466  (chain ladder 373,346)
```

The gamma fit gives the same weight to the relative error of each cell. Thus it
does not keep the row and column totals, and its forecasts are not those of the
chain ladder. On this stable triangle they are near. The delta method of
section 3 applies with process variance `phi * mu**2` instead of `phi * mu`, and
`gamma.prediction_error()` uses it.

## 10. What matched

**Exact agreement**:

- the chain-ladder factors and reserves (Tables 3-1 and 3-2)
- the coefficients and standard errors of the cross-classified GLM (Tables 3-5
  and 5-1), and its scale 114.5
- the ultimates and payment pattern (Table 3-3)
- the factors of the ODP Mack model (Table 3-4)
- each forecast, RMSEP and CVP of the delta method (Table 5-4)
- the coefficients of the three reduced models (Tables 7-1, 7-3 and 7-5)
- the two heat maps (Figures 6-22 and 7-5)

**Agreement with known differences**:

- The independent-years RMSEP is one unit different, because the monograph
  adds rounded values.
- The parametric bootstraps (Tables 5-5, 7-2, 7-4 and 7-6) differ by
  simulation noise and by the process distribution, which the monograph does
  not state. Their means have the lognormal bias.
- The AIC and BIC of Table 7-7 differ by approximately one unit in 509,000,
  from `phi` or convergence. GCV agrees to five figures.
- Three renormalised ultimates of Table 3-5 are one unit different from Table
  3-3, because the monograph rounds.
