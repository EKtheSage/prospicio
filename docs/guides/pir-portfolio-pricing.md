# Pricing Insurance Risk, Part II: portfolio pricing in prospicio

<!-- site: execute -->

Part II of Mildenhall and Major's *Pricing Insurance Risk: Theory and Practice*
(Wiley, 2022) is chapters 8 to 11. It prices a whole portfolio in two ways:

- The classical approach (chapters 8 and 9). It applies premium calculation
  principles to each unit or to the total. It also covers the portfolio
  constant cost of capital (CCoC), ruin theory and option pricing.
- The modern approach (chapters 10 and 11). The premium is a distortion of the
  loss capped at the assets, which is a spectral risk measure (SRM). The book
  calibrates five representative distortions to one target and compares them.

This guide runs those chapters in prospicio and puts the book's numbers beside
prospicio's.

The book's text belongs to the authors, and this guide explains it in new
words. Exhibit numbers are the book's. Related guides:

- Risk measures (VaR, TVaR, coherence, the representations of an SRM):
  [Pricing Insurance Risk, Part I](pir-risk-measures.md).
- The split of a portfolio price over its units (the natural allocation, Part
  III of the book): [the case studies guide](pricing-insurance-risk-case-studies.md)
  and [CAS Monograph 15 reproduced](capital-modeling-monograph-15.md).

The code prices a unit on its own (stand-alone) or the total as a whole.

## 1. Two books to price

The practice chapters use the book's case studies. Two of them run fast enough
for a guide:

- **Tame** has two gamma units with mean 50 and CVs 0.10 and 0.15, at a 99.99%
  VaR capital standard. In the net view, an aggregate cover of 12.91 excess of
  56.17 applies to unit B.
- **Cat/Non-Cat** has a gamma Non-Cat unit (mean 80, CV 0.15) and a lognormal
  Cat unit (mean 20, CV 1), at 99.9%. In the net view, 79.64 excess of 41.11
  applies to Cat (Table 7.3).

The code puts each unit on the grid of `aggregate`, with a step of 1/64 and
2^16 points. A net unit is the gross unit minus the recovery of the cover,
outcome by outcome:

```python
import math
import time

from prospicio.distributions import Gamma, Grid, Lognormal, Sampled
from prospicio.pricing import Portfolio, calibrate_classical, classical_premium
from prospicio.risk import Distortion

STEP, POINTS = 1 / 64, 2**16
started = time.perf_counter()


def discretize(severity):
    """A severity on aggregate's grid: rounded, the mass above the top spread by renormalizing."""
    probs = Grid.rounding(severity, STEP, 2 * POINTS)[0].probs[:POINTS]
    total = sum(probs)
    return Grid(STEP, [q / total for q in probs])


def net_of(grid, limit, attachment):
    """The grid's loss less an aggregate cover of limit excess of attachment."""
    return grid.map(lambda x: x - min(max(x - attachment, 0.0), limit))[0]


def total_grid(book):
    """The total of a portfolio as a grid."""
    probs = [0.0] * (round(max(book.totals) / STEP) + 1)
    for x, q in zip(book.totals, book.probs):
        probs[round(x / STEP)] += q
    return Grid(STEP, probs)


a_unit, b_unit = discretize(Gamma.from_mean_cv(50.0, 0.10)), discretize(Gamma.from_mean_cv(50.0, 0.15))
b_net = net_of(b_unit, 12.91, 56.17)
tame = Portfolio.from_independent(["A", "B"], [a_unit, b_unit])
tame_net = Portfolio.from_independent(["A", "B"], [a_unit, b_net])

cat, non_cat = discretize(Lognormal.from_mean_cv(20.0, 1.0)), discretize(Gamma.from_mean_cv(80.0, 0.15))
cat_net = net_of(cat, 79.64, 41.11)
cnc = Portfolio.from_independent(["Cat", "NonCat"], [cat, non_cat])
cnc_net = Portfolio.from_independent(["Cat", "NonCat"], [cat_net, non_cat])

print(f"means: B net {b_net.mean():.3f} (book 49.08), Cat net {cat_net.mean():.3f} (book 17.79)")
```

```text
means: B net 49.084 (book 49.08), Cat net 17.786 (book 17.79)
```

The attachments and limits of the covers come from the quantiles of the gross
units: the 80% and 99% quantiles of B (56.17, 69.08) and the 90% and 99.5%
quantiles of Cat (41.11, 120.75). `Grid.map` makes the net unit exactly on the
grid. A `Tower` with `on_grid` gives the same distribution by FFT and sets the
FFT round-off (about 1e-19 where the probability must be zero) to zero. Section
3 shows why that matters.

## 2. The portfolio CCoC price

Chapter 8 gives the classical portfolio price, and every practice chapter uses
it as the benchmark. The price holds assets at the VaR of the total and charges
a constant cost of capital on the capital. With a 10% return, the premium is
P = v E[min(X, a)] + d a, where v = 1 / 1.1 and d = 1 - v. The insurance
pentagon then gives the margin, capital, loss ratio and leverage.

Table 7.2 prints the result for each book, gross and net. The book's leverage
is premium to capital (`premium_to_capital`). The loss is the expected loss
within the assets, so it includes the effect of default: Cat/Non-Cat gross has
a loss of 99.95, not 100. All 24 figures agree:

```python
import pandas as pd
from great_tables import GT


def pair_table(frame, decimals):
    """A table whose rows alternate prospicio and the book: prospicio at fixed decimals, the book as printed."""
    mine, book = list(range(0, len(frame), 2)), list(range(1, len(frame), 2))
    table = GT(frame, rowname_col=frame.columns[0])
    for column, places in decimals.items():
        table = table.fmt_number(column, rows=mine, decimals=places, use_seps=False)
        table = table.fmt_number(column, rows=book, decimals=4, drop_trailing_zeros=True, use_seps=False)
    return table.sub_missing(missing_text="")


ccoc = Distortion.ccoc(0.10)
BOOK_72 = {  # Table 7.2: loss, margin, premium, capital, assets, leverage
    ("Tame", "gross"): (100.00, 3.42, 103.4, 34.23, 137.7, 3.02),
    ("Tame", "net"): (99.08, 2.46, 101.5, 24.64, 126.2, 4.12),
    ("Cat/Non-Cat", "gross"): (99.95, 15.21, 115.2, 152.1, 267.2, 0.757),
    ("Cat/Non-Cat", "net"): (97.73, 8.17, 105.9, 81.68, 187.6, 1.3),
}
BOOKS = {("Tame", "gross"): (tame, 0.9999), ("Tame", "net"): (tame_net, 0.9999),
         ("Cat/Non-Cat", "gross"): (cnc, 0.999), ("Cat/Non-Cat", "net"): (cnc_net, 0.999)}
COLUMNS_72 = ["loss", "margin", "premium", "capital", "assets", "P/Q"]
rows = []
for key, (book, p) in BOOKS.items():
    t = book.price(ccoc, p=p).total
    rows.append((" ".join(key), t.loss, t.margin, t.premium, t.capital, t.assets, t.premium_to_capital))
    rows.append((" ".join(key) + ", book", *BOOK_72[key]))
table_72 = pd.DataFrame(rows, columns=["book", *COLUMNS_72])
(pair_table(table_72, {**{c: 2 for c in COLUMNS_72[:5]}, "P/Q": 3})
 .tab_header("Table 7.2: the portfolio CCoC price", "Currency units; P/Q is premium over capital"))
```

```text
                           loss  margin  premium  capital  assets    P/Q
Tame gross               100.00    3.42   103.42    34.23  137.66  3.021
Tame gross, book            100    3.42    103.4    34.23   137.7   3.02
Tame net                  99.08    2.46   101.55    24.64  126.19  4.121
Tame net, book            99.08    2.46    101.5    24.64   126.2   4.12
Cat/Non-Cat gross         99.95   15.21   115.15   152.05  267.20  0.757
Cat/Non-Cat gross, book   99.95   15.21    115.2    152.1   267.2  0.757
Cat/Non-Cat net           97.73    8.17   105.90    81.68  187.58  1.297
Cat/Non-Cat net, book     97.73    8.17    105.9    81.68   187.6    1.3
```

Stand-alone, each unit holds its own 99.99% or 99.9% VaR. The units together
need more capital than the book (Tables 9.12, 9.13 and 9.11). B and Cat are net
of their cover in the net rows. P is premium and Q is capital:

```python
BOOK_SA = {  # Tables 9.12, 9.13: stand-alone premium and capital by unit; Table 9.11's sums
    ("Tame", "gross"): ([51.89, 52.98], [18.88, 29.82], 104.9, 48.69),
    ("Tame", "net"): ([51.89, 50.98], [18.88, 18.91], 102.9, 37.79),
    ("Cat/Non-Cat", "gross"): ([34.98, 83.84], [150.3, 38.41], 118.8, 188.7),
    ("Cat/Non-Cat", "net"): ([25.72, 83.84], [79.92, 38.41], 109.6, 118.3),
}
NAN = float("nan")
rows = []
for key, (book, p) in BOOKS.items():
    sa = book.price_stand_alone(ccoc, p)
    (p1, p2), (q1, q2), sum_p, sum_q = BOOK_SA[key]
    u1, u2 = sa.stand_alone
    rows.append((" ".join(key), u1.premium, u1.capital, u2.premium, u2.capital,
                 sa.sum.premium, sa.total.premium, sa.total.premium / sa.sum.premium - 1,
                 sa.sum.capital, sa.total.capital, sa.total.capital / sa.sum.capital - 1))
    rows.append((" ".join(key) + ", book", p1, q1, p2, q2, sum_p, NAN, NAN, sum_q, NAN, NAN))
table_sa = pd.DataFrame(rows, columns=["book", "p1", "q1", "p2", "q2", "sum_p", "total_p", "saved_p",
                                       "sum_q", "total_q", "saved_q"])
(pair_table(table_sa, {c: 2 for c in ["p1", "q1", "p2", "q2", "sum_p", "total_p", "sum_q", "total_q"]})
 .fmt_percent(["saved_p", "saved_q"], rows=list(range(0, len(rows), 2)), decimals=1)
 .tab_header("Tables 9.12, 9.13 and 9.11: stand-alone against pooled",
             "Units A, B (Tame) and Cat, Non-Cat (Cat/Non-Cat); currency units")
 .tab_spanner("unit 1", ["p1", "q1"])
 .tab_spanner("unit 2", ["p2", "q2"])
 .tab_spanner("premium", ["sum_p", "total_p", "saved_p"])
 .tab_spanner("capital", ["sum_q", "total_q", "saved_q"])
 .cols_label(p1="P", q1="Q", p2="P", q2="Q", sum_p="sum of units", total_p="total", saved_p="pooling saves",
             sum_q="sum of units", total_q="total", saved_q="pooling saves"))
```

```text
                             P       Q      P      Q  sum of units   total  pooling saves  sum of units   total  pooling saves
Tame gross               51.89   18.88  52.98  29.82        104.87  103.42          -1.4%         48.69   34.23         -29.7%
Tame gross, book         51.89   18.88  52.98  29.82         104.9                                48.69                       
Tame net                 51.89   18.88  50.98  18.92        102.86  101.55          -1.3%         37.79   24.64         -34.8%
Tame net, book           51.89   18.88  50.98  18.91         102.9                                37.79                       
Cat/Non-Cat gross        34.98  150.31  83.84  38.41        118.81  115.15          -3.1%        188.72  152.05         -19.4%
Cat/Non-Cat gross, book  34.98   150.3  83.84  38.41         118.8                                188.7                       
Cat/Non-Cat net          25.72   79.92  83.84  38.41        109.56  105.90          -3.3%        118.33   81.68         -31.0%
Cat/Non-Cat net, book    25.72   79.92  83.84  38.41         109.6                                118.3                       
```

Premium falls less than capital, because most of the premium is expected loss
and pooling does not change the expected loss. The thin tails of Tame give the
most diversification.

## 3. Classical premium principles

Section 8.4 describes the classical premium calculation principles (PCPs).
Section 9.1 applies ten of them to the case studies in two steps:

1. Calibrate each principle so that the gross total costs the CCoC premium
   above.
2. Apply the principle stand-alone to each unit, to the net units and to the
   net total.

prospicio has all of them as `classical_premium(principle, x, loading)`, and
`calibrate_classical` finds a loading. The book's Fischer principle uses the
power 2, which is the prospicio default.

Table 9.2 lists the Tame premiums, sorted by the net total from the lowest
premium to the highest. Beside each loading is the book's. The last column is
the largest gap to the book's premiums in that row:

```python
PRINCIPLES = ["expected_value", "variance", "esscher", "exponential", "semi_variance",
              "var", "dutch", "standard_deviation", "fischer"]
BOOK_92 = {  # Table 9.2, Tame: loading; A, B net, B gross, total net, total gross
    "expected_value": (0.034, 51.712, 50.765, 51.712, 102.476, 103.423),
    "variance": (0.042, 51.053, 50.620, 52.370, 101.673, 103.423),
    "esscher": (0.041, 51.034, 50.468, 52.389, 101.503, 103.423),
    "exponential": (0.080, 51.028, 50.420, 52.395, 101.447, 103.423),
    "semi_variance": (0.080, 51.051, 50.330, 52.425, 101.411, 103.423),
    "var": (0.659, 51.906, 52.750, 52.750, 102.672, 103.422),
    "dutch": (0.953, 51.899, 51.498, 52.846, 102.096, 103.423),
    "standard_deviation": (0.380, 51.899, 51.377, 52.848, 102.061, 103.423),
    "fischer": (0.523, 51.897, 51.149, 52.881, 101.907, 103.423),
}


def classical_frame(gross_total, grids, labels, target, book_values):
    """Each principle's loading and premiums on the grids, with the book's loading and the largest gap."""
    rows = []
    for principle, (book_t, *book_p) in book_values.items():
        t = calibrate_classical(principle, gross_total, target)
        got = [classical_premium(principle, g, t) for g in grids]
        rows.append((principle.replace("_", " "), t, book_t, *got, max(abs(a - b) for a, b in zip(got, book_p))))
    return pd.DataFrame(rows, columns=["principle", "loading", "book loading", *labels, "gap"]).sort_values("net")


def classical_gt(frame, title, subtitle, spanned):
    table = (GT(frame, rowname_col="principle")
             .tab_header(title, subtitle)
             .tab_spanner("calibrated loading", ["loading", "book loading"])
             .fmt_number(["loading", "book loading"], decimals=3)
             .fmt_number([c for c in frame.columns[3:]], decimals=3)
             .cols_label(loading="prospicio", **{"book loading": "book"}))
    for label, columns in spanned.items():
        table = table.tab_spanner(label, columns)
    return table


tame_target = tame.price(ccoc, p=0.9999).total.premium
tame_total, tame_net_total = total_grid(tame), total_grid(tame_net)
tame_92 = classical_frame(tame_total, [a_unit, b_net, b_unit, tame_net_total, tame_total],
                          ["A", "B net", "B", "net", "total"], tame_target, BOOK_92)
classical_gt(tame_92, "Table 9.2: Tame premiums by principle, each calibrated to the CCoC total",
             "Currency units; the loading is the parameter of the principle",
             {"units": ["A", "B net", "B"], "totals": ["net", "total"]})
```

```text
                    prospicio   book       A   B net       B      net    total    gap
semi variance           0.080  0.080  51.051  50.330  52.425  101.411  103.423  0.000
exponential             0.080  0.080  51.028  50.419  52.395  101.447  103.423  0.001
esscher                 0.041  0.041  51.034  50.468  52.389  101.502  103.423  0.001
variance                0.042  0.042  51.053  50.619  52.370  101.672  103.423  0.001
fischer                 0.523  0.523  51.897  51.149  52.881  101.906  103.423  0.001
standard deviation      0.380  0.380  51.899  51.376  52.848  102.060  103.423  0.001
dutch                   0.953  0.953  51.899  51.498  52.846  102.096  103.423  0.000
expected value          0.034  0.034  51.712  50.764  51.712  102.476  103.423  0.001
var                     0.659  0.659  51.906  52.750  52.750  102.672  103.422  0.000
```

Every loading and premium agrees to the third decimal. The largest gap, 0.001,
is rounding.

Table 9.3 subtracts the premium of the total from the sum of the stand-alone
premiums (SoP). The book prints the gross SoP for every principle but
expected value:

```python
BOOK_93 = {"variance": 103.423, "esscher": 103.423, "exponential": 103.423, "semi_variance": 103.477,
           "var": 104.656, "dutch": 104.745, "standard_deviation": 104.747, "fischer": 104.778}
rows = []
for principle in PRINCIPLES:
    t = calibrate_classical(principle, tame_total, tame_target)
    sop = classical_premium(principle, a_unit, t) + classical_premium(principle, b_unit, t)
    sop_net = classical_premium(principle, a_unit, t) + classical_premium(principle, b_net, t)
    net = classical_premium(principle, tame_net_total, t)
    rows.append((principle.replace("_", " "), sop, sop - tame_target, BOOK_93.get(principle, float("nan")),
                 sop_net, sop_net - net))
table_93 = pd.DataFrame(rows, columns=["principle", "sop", "delta", "book", "sop_net", "delta_net"])
(GT(table_93, rowname_col="principle")
 .tab_header("Table 9.3: stand-alone sum less the pooled premium", "Tame; currency units")
 .tab_spanner("gross", ["sop", "delta", "book"])
 .tab_spanner("net", ["sop_net", "delta_net"])
 .fmt_number(["sop", "delta", "book", "sop_net", "delta_net"], decimals=3)
 .sub_missing(missing_text="")
 .cols_label(sop="SoP", delta="delta", book="book SoP", sop_net="SoP", delta_net="delta"))
```

```text
                        SoP  delta  book SoP      SoP   delta
expected value      103.423  0.000            102.476   0.000
variance            103.423  0.000   103.423  101.672   0.000
esscher             103.423  0.000   103.423  101.502   0.000
exponential         103.423  0.000   103.423  101.447   0.000
semi variance       103.477  0.054   103.477  101.381  -0.030
var                 104.656  1.233   104.656  104.656   1.984
dutch               104.745  1.322   104.745  103.397   1.301
standard deviation  104.747  1.324   104.747  103.275   1.214
fischer             104.778  1.355   104.778  103.046   1.140
```

The delta columns show the book's main point:

- Expected value, variance, Esscher and exponential are additive for
  independent units, so pooling earns nothing.
- VaR and the deviation principles (Dutch, standard deviation, Fischer) are
  subadditive. They charge the pool approximately 1.3 less than the parts.
- The semivariance is neither. Its delta changes sign between gross and net
  (0.054 and -0.029 in the book).

The formula of a principle sets its view of diversification, not the data.

Two principles are sensitive to how the code makes the net B grid of Tame. The
exponential and Esscher premiums weight an outcome x by exp(t x). On a grid
that goes to 1,024, exp(0.08 x) is approximately 1e35 at the top, so an FFT
round-off of 1e-19 there would give a premium of hundreds. `Tower.on_grid` sets
probabilities below machine epsilon to zero, as `remove_fuzz` in `aggregate`
does. Thus its net grid gives the same price as the exact grid:

```python
from prospicio.distributions import Count
from prospicio.reinsurance import Layer, Tower

b_net_fft = Tower([Layer.stop_loss("B", 12.91, 56.17)]).on_grid(Count.fixed(1), b_unit, POINTS).net
t_exp = calibrate_classical("exponential", tame_total, tame_target)
fft_premium = classical_premium('exponential', b_net_fft, t_exp)
noise = max(b_net_fft.probs[150 * 64:])
print(f"exponential premium of net B: by map {classical_premium('exponential', b_net, t_exp):.3f}, "
      f"from the FFT {fft_premium:.3f} (book 50.420)")
print(f"largest probability above 150: FFT {'exactly zero' if noise == 0 else noise}, "
      "map exactly zero")
```

```text
exponential premium of net B: by map 50.419, from the FFT 50.419 (book 50.420)
largest probability above 150: FFT exactly zero, map exactly zero
```

To clean a grid from a different source (for example, your own FFT) in the
same way, use `Grid.remove_fuzz()` before you price with these two principles.

On Cat/Non-Cat, the principles give very different results (Tables 9.5 and
9.6):

```python
BOOK_95 = {  # Table 9.5, Cat/Non-Cat: loading; Cat net, Cat, Non-Cat, total net, total gross
    "expected_value": (0.152, 20.480, 23.030, 92.121, 112.601, 115.151),
    "var": (0.818, 30.109, 30.109, 90.734, 113.391, 115.141),
    "variance": (0.028, 22.573, 31.141, 84.013, 106.585, 115.151),
    "dutch": (1.859, 27.170, 32.000, 88.882, 110.638, 115.151),
    "semi_variance": (0.040, 22.513, 32.785, 83.092, 105.194, 115.151),
    "standard_deviation": (0.650, 26.302, 32.992, 87.797, 109.332, 115.151),
    "fischer": (0.776, 26.249, 33.918, 86.844, 108.380, 115.151),
}
cnc_target = cnc.price(ccoc, p=0.999).total.premium
cnc_total, cnc_net_total = total_grid(cnc), total_grid(cnc_net)
cnc_95 = classical_frame(cnc_total, [cat_net, cat, non_cat, cnc_net_total, cnc_total],
                         ["Cat net", "Cat", "Non-Cat", "net", "total"], cnc_target, BOOK_95)
classical_gt(cnc_95, "Tables 9.5 and 9.6: Cat/Non-Cat premiums by principle, each calibrated to the CCoC total",
             "Currency units; the loading is the parameter of the principle",
             {"units": ["Cat net", "Cat", "Non-Cat"], "totals": ["net", "total"]})
```

```text
                    prospicio   book  Cat net     Cat  Non-Cat      net    total    gap
semi variance           0.040  0.040   22.512  32.782   83.091  105.193  115.151  0.003
variance                0.028  0.028   22.572  31.139   84.012  106.584  115.151  0.002
fischer                 0.776  0.776   26.248  33.916   86.843  108.380  115.151  0.002
standard deviation      0.650  0.650   26.301  32.991   87.797  109.331  115.151  0.001
dutch                   1.859  1.859   27.170  32.000   88.882  110.638  115.151  0.000
expected value          0.152  0.152   20.480  23.030   92.121  112.601  115.151  0.000
var                     0.818  0.818   30.109  30.109   90.734  113.391  115.141  0.000
```

Seven principles agree to within 0.003, the size of difference that two
truncations of a lognormal tail give. The Esscher and exponential principles do
not agree, and you should not expect them to:

```python
BOOK_EXP = {"esscher": (0.011, 23.575, 35.944, 81.561, 105.054),
            "exponential": (0.014, 24.760, 39.697, 81.059, 105.619)}
rows = []
for principle, (book_t, *book_p) in BOOK_EXP.items():
    t = calibrate_classical(principle, cnc_total, cnc_target)
    got = [classical_premium(principle, g, t) for g in (cat_net, cat, non_cat, cnc_net_total)]
    rows.append((principle, t, *got))
    rows.append((principle + ", book", book_t, *book_p))
table_exp = pd.DataFrame(rows, columns=["principle", "loading", "Cat net", "Cat", "Non-Cat", "net"])
(pair_table(table_exp, {"loading": 4, "Cat net": 3, "Cat": 3, "Non-Cat": 3, "net": 3})
 .tab_header("Esscher and exponential premiums on Cat/Non-Cat", "Currency units; the book's values are as printed"))
```

```text
                   loading  Cat net     Cat  Non-Cat      net
esscher             0.0103   22.760  33.639   81.512  104.272
esscher, book        0.011   23.575  35.944   81.561  105.054
exponential         0.0138   22.820  34.139   81.012  103.832
exponential, book    0.014    24.76  39.697   81.059  105.619
```

```python
lower = cat.probs[: POINTS // 2]
scale = sum(lower)
top_half = Grid(STEP, [q / scale for q in lower])
print(f"exponential premium of Cat at loading 0.014: grid to 1,024 {classical_premium('exponential', cat, 0.014):.3f}, "
      f"grid to 512 {classical_premium('exponential', top_half, 0.014):.3f}")
```

```text
exponential premium of Cat at loading 0.014: grid to 1,024 35.497, grid to 512 25.189
```

A lognormal has no exponential moment: E[exp(t X)] is infinite for every t > 0.
Both premiums are artifacts of the point where the grid stops. When the grid
stops at 512, not 1,024, the Cat premium moves by several units. The book does
not apply these principles to the Hurricane/SCS case for this reason. Do not
use them for a loss without a light tail.

## 4. Ruin theory and the natural scale of a market

Section 9.3 uses the Pollaczeck-Khinchine formula for the probability of
eventual ruin of a compound Poisson surplus process. In prospicio this is
`eventual_ruin`. The example is a liability line with lognormal claims of mean
50,000 and CV 10, limited at 1M, 5M or 10M.

With a margin r on expected loss, the formula gives the starting surplus u at
which eventual ruin has probability p. The annual margin, n r E[X] for n claims
a year, must pay a cost of capital delta on that surplus. Thus a market needs
at least n = delta u / (r E[X]) claims a year (Eq. (9.2)). Table 9.15
tabulates n with delta = 0.1. The last three columns are the book's n over
prospicio's:

```python
from prospicio.aggregate import eventual_ruin

claim = Lognormal.from_mean_cv(50_000.0, 10.0)
BOOK_915 = {  # Table 9.15: limit -> {p: (n at r = 0.05, 0.1, 0.25)}
    1e6: {0.01: (10_172, 2_655, 476.1), 0.1: (5_021, 1_294, 224.5)},
    1e7: {0.01: (33_610, 9_087, 1_765), 0.1: (16_384, 4_324, 793.6)},
}
rows = []
for limit, by_p in BOOK_915.items():
    # 10,000 steps up to the limit; the claim's mass above the limit lands on it
    capped = Grid.rounding(claim, limit / 10_000, 10_001)[0].map(lambda x: min(x, limit))[0]
    for p, book in by_p.items():
        counts = []
        for r in (0.05, 0.1, 0.25):
            u = eventual_ruin(capped, rho=r, points=2**18).capital(p, interpolate=True)
            counts.append(0.1 * u / (r * capped.mean()))
        rows.append((f"{limit:,.0f}, p {p}", *counts, *book, *(b / n for n, b in zip(counts, book))))
table_915 = pd.DataFrame(rows, columns=["case", "n1", "n2", "n3", "b1", "b2", "b3", "x1", "x2", "x3"])
(GT(table_915, rowname_col="case")
 .tab_header("Table 9.15: claims a year a market needs", "By limit and ruin probability p, at margins r = 5%, 10%, 25%")
 .tab_spanner("prospicio", ["n1", "n2", "n3"])
 .tab_spanner("book", ["b1", "b2", "b3"])
 .tab_spanner("book / prospicio", ["x1", "x2", "x3"])
 .fmt_number(["n1", "n2", "n3"], decimals=1, use_seps=False)
 .fmt_number(["b1", "b2", "b3"], decimals=1, drop_trailing_zeros=True, use_seps=False)
 .fmt_number(["x1", "x2", "x3"], decimals=2)
 .cols_label(n1="5%", n2="10%", n3="25%", b1="5%", b2="10%", b3="25%", x1="5%", x2="10%", x3="25%"))
```

```text
                        5%    10%    25%     5%   10%    25%     5%    10%    25%
1,000,000, p 0.01   1019.0  266.0   47.7  10172  2655  476.1   9.98   9.98   9.98
1,000,000, p 0.1     503.0  129.7   22.5   5021  1294  224.5   9.98   9.98   9.99
10,000,000, p 0.01  3362.0  909.0  176.5  33610  9087   1765  10.00  10.00  10.00
10,000,000, p 0.1   1639.0  432.5   79.4  16384  4324  793.6  10.00  10.00  10.00
```

Every prospicio count is one tenth of the book's. The book's table is
u / (r E[X]), which is Eq. (9.2) with delta = 1, so it seems to leave out the
cost of capital of 0.1 that its caption gives. The example in the text has the
same factor: 39,776 claims, approximately 800,000 policies, for a 1M limit at
p = 0.01 and r = 0.025. The factor is the same in every cell, so the book's
conclusions about how the scale changes with the limit, the margin and the ruin
probability are correct. (The grid causes the remaining 0.2%.)

## 5. The modern view: a price for each layer of assets

Chapter 10 starts from the layer. The risk that losses consume the l-th unit of
assets is a Bernoulli with probability s = S(l). A market prices this risk at
g(s), where g is a distortion. The sum over the layers gives the premium for a
loss X with assets a: the integral of g(S(x)) from 0 to a. This is the SRM of
the capped loss X ^ a.

A Bernoulli layer has expected loss s, premium g(s), margin g(s) - s and
capital 1 - g(s). Thus its loss ratio is s / g(s), its return on capital is
(g(s) - s) / (1 - g(s)), and its premium leverage is g(s) / (1 - g(s)).
Exercise 142 tabulates these for the proportional hazard distortions s^rho
(Table 10.3). The last column is the largest gap to the book's table:

```python
BOOK_103 = {  # Table 10.3 at s = 0.01, 0.1, 0.5: loss ratio, risk return, premium leverage
    0.4: ((0.0631, 0.2512, 0.6598), (0.1765, 0.4953, 1.0649), (0.1883, 0.6614, 3.1298)),
    0.5: ((0.1000, 0.3162, 0.7071), (0.1000, 0.3162, 0.7071), (0.1111, 0.4625, 2.4142)),
    0.7: ((0.2512, 0.5012, 0.8123), (0.0310, 0.1243, 0.3006), (0.0415, 0.2493, 1.6013)),
    0.9: ((0.6310, 0.7943, 0.9330), (0.0059, 0.0296, 0.0773), (0.0161, 0.1440, 1.1546)),
}
rows = []
for rho, book in BOOK_103.items():
    g = Distortion.proportional_hazard(rho).g
    stats = [[s / g(s) for s in (0.01, 0.1, 0.5)],
             [(g(s) - s) / (1 - g(s)) for s in (0.01, 0.1, 0.5)],
             [g(s) / (1 - g(s)) for s in (0.01, 0.1, 0.5)]]
    worst = max(abs(a - b) for row, brow in zip(stats, book) for a, b in zip(row, brow))
    rows.append((f"rho {rho}", *(v for row in stats for v in row), worst))
names = [f"{kind}{i}" for kind in ("lr", "roc", "lev") for i in range(3)]
table_103 = pd.DataFrame(rows, columns=["rho", *names, "gap"])
(GT(table_103, rowname_col="rho")
 .tab_header("Table 10.3: Bernoulli layers under the proportional hazard", "At s = 0.01, 0.1 and 0.5")
 .tab_spanner("loss ratio", names[:3])
 .tab_spanner("return on capital", names[3:6])
 .tab_spanner("premium leverage", names[6:])
 .fmt_number(names + ["gap"], decimals=4)
 .cols_label(**{n: f"s = {('0.01', '0.1', '0.5')[int(n[-1])]}" for n in names}))
```

```text
         s = 0.01  s = 0.1  s = 0.5  s = 0.01  s = 0.1  s = 0.5  s = 0.01  s = 0.1  s = 0.5     gap
rho 0.4    0.0631   0.2512   0.6598    0.1765   0.4953   1.0649    0.1883   0.6614   3.1298  0.0000
rho 0.5    0.1000   0.3162   0.7071    0.1000   0.3162   0.7071    0.1111   0.4625   2.4142  0.0000
rho 0.7    0.2512   0.5012   0.8123    0.0310   0.1243   0.3006    0.0415   0.2493   1.6013  0.0000
rho 0.9    0.6310   0.7943   0.9330    0.0059   0.0296   0.0773    0.0161   0.1440   1.1546  0.0000
```

Table 10.3 agrees in every cell.

The pattern is the main lesson of the chapter. A concave g charges remote
layers (small s) a low loss ratio. But these layers need much capital for each
unit of premium, so their return is also low. The return on capital increases
with s. Only the CCoC distortion, a straight line from (0, d) to (1, 1), earns
the same return on every layer. For this reason it is "constant".

The figure shows the return on capital of a layer against its exceedance
probability s:

```python
import matplotlib.pyplot as plt

grid_s = [10 ** (-3 + 2.99 * i / 99) for i in range(100)]
fig, ax = plt.subplots()
for label, dist in (("PH, rho 0.5", Distortion.proportional_hazard(0.5)),
                    ("PH, rho 0.7", Distortion.proportional_hazard(0.7)),
                    ("CCoC, 10%", Distortion.ccoc(0.10))):
    ax.plot(grid_s, [(dist.g(s) - s) / (1 - dist.g(s)) for s in grid_s], label=label)
ax.set_xscale("log")
ax.set_xlabel("exceedance probability s of the layer")
ax.set_ylabel("return on capital of the layer")
ax.set_title("Return on capital by layer")
ax.legend()
```

The Simple Discrete Example of Part I shows the capped premium as a function of
the assets. In Tables 11.1 and 11.3, the PH with rho 0.5 prices the uncapped
loss at 51.156 and the loss capped at 80 at 42.828:

```python
OUTCOMES = [0, 1, 8, 9, 10, 11, 90, 98, 100]
SIXTEENTHS = [4, 2, 2, 1, 2, 1, 2, 1, 1]
discrete = Sampled([float(x) for x, k in zip(OUTCOMES, SIXTEENTHS) for _ in range(k)])
ph = Distortion.proportional_hazard(0.5)
rows = []
for a in (10, 50, 80, 90, 100):
    capped = Sampled([min(x, a) for x in discrete.draws])
    premium = ph.measure(capped)
    book = {80: 42.828, 100: 51.156}.get(a, float("nan"))
    rows.append((str(a), capped.mean(), premium, book, premium - capped.mean(), a - premium))
table_cap = pd.DataFrame(rows, columns=["assets", "mean", "premium", "book", "margin", "capital"])
(GT(table_cap, rowname_col="assets")
 .tab_header("Premium of the loss capped at the assets", "PH with rho 0.5 on the Simple Discrete Example")
 .fmt_number(["mean", "premium", "book", "margin", "capital"], decimals=3)
 .sub_missing(missing_text="")
 .cols_label(mean="E[X ^ a]", premium="PH premium", book="book premium"))
```

```text
     E[X ^ a]  PH premium  book premium  margin  capital
10      6.062       7.769                 1.706    2.231
50     16.125      27.828                11.703   22.172
80     23.625      42.828        42.828  19.203   37.172
90     26.125      47.828                21.703   42.172
100    27.250      51.156        51.156  23.906   48.844
```

Premium funds most of the low layers, because losses consume them in most
outcomes. From 11 to 90, losses consume each added unit of assets only in the
25% of outcomes above 11, so each unit adds g(0.25) = 0.5 of premium and 0.5 of
capital. The top layers add mostly capital.

## 6. Calibrating the five distortions

Chapter 11 compares SRMs that charge the same total premium. The five families
are CCoC, proportional hazard (PH), Wang, dual power and TVaR, calibrated to
the gross CCoC premium of section 2. A sixth distortion, "Blend", is piecewise
linear and built from layer costs of capital. Table 11.5 prints the calibrated
parameters:

```python
FAMILIES = ["ccoc", "ph", "wang", "dual", "tvar"]
BOOK_115 = {"Tame": (0.10, 0.683, 0.375, 1.576, 0.227), "Cat/Non-Cat": (0.10, 0.596, 0.611, 2.463, 0.482)}
fitted = {}
rows = []
for name, (book, p) in (("Tame", (tame, 0.9999)), ("Cat/Non-Cat", (cnc, 0.999))):
    fitted[name] = {f: book.calibrate(f, p=p, return_on_capital=0.10) for f in FAMILIES}
    rows.append((name, *(next(iter(d.parameters.values())) for d in fitted[name].values())))
    rows.append((name + ", book", *BOOK_115[name]))
table_115 = pd.DataFrame(rows, columns=["book", *FAMILIES])
(pair_table(table_115, {f: 3 for f in FAMILIES})
 .tab_header("Table 11.5: calibrated parameters", "Each family calibrated to the gross CCoC premium"))
```

```text
                    ccoc     ph   wang   dual   tvar
Tame               0.100  0.683  0.375  1.576  0.227
Tame, book           0.1  0.683  0.375  1.576  0.227
Cat/Non-Cat        0.100  0.596  0.611  2.463  0.482
Cat/Non-Cat, book    0.1  0.596  0.611  2.463  0.482
```

All ten parameters agree. In prospicio, each family takes the parameter that
the book uses:

- `ccoc`: the return r (0.10)
- PH: the exponent rho
- Wang: the shift lambda
- dual power: the exponent beta
- TVaR: the level p

The case studies guide also has the Cat/Non-Cat parameters, run at 99.9%
against `aggregate` 0.30.1. That guide also runs Tame at 99.9%, but the book's
Tame uses 99.99%.

The Blend distortion of Exercise 197 (Table 11.6) sets a layer cost of capital
nu at some exceedance probabilities s. The values go from 2% for the most
remote layers to 10% at s = 0.2. At each of these points g(s) =
(s + nu) / (1 + nu), and the Blend interpolates between the points.
`Distortion.convex` builds the smallest concave distortion through such points.
The book's point at s = 0.03 is below the line through its neighbours, so the
concave version lifts it a small amount:

```python
LAYERS = {0.0: 0.02, 0.01: 0.03, 0.03: 0.04, 0.05: 0.06, 0.10: 0.08, 0.20: 0.10}
blend = Distortion.convex([(s, (s + nu) / (1 + nu)) for s, nu in LAYERS.items()] + [(1.0, 1.0)])
table_blend = pd.DataFrame({
    "s": [str(s) for s in LAYERS],
    "nu": list(LAYERS.values()),
    "specified": [(s + nu) / (1 + nu) for s, nu in LAYERS.items()],
    "blend": [blend.g(s) for s in LAYERS],
})
(GT(table_blend, rowname_col="s")
 .tab_header("Table 11.6: the Blend through layer costs of capital", "g(s) at the specified points")
 .fmt_percent("nu", decimals=0)
 .fmt_number(["specified", "blend"], decimals=4)
 .cols_label(nu="cost of capital", specified="specified g", blend="Blend g"))
```

```text
      cost of capital  specified g  Blend g
0.0                2%       0.0196   0.0000
0.01               3%       0.0388   0.0388
0.03               4%       0.0673   0.0713
0.05               6%       0.1038   0.1038
0.1                8%       0.1667   0.1667
0.2               10%       0.2727   0.2727
```

At s = 0 a distortion is 0 by definition. The specified 0.0196 is the mass at
zero (the rate on line of the most remote layer), which `blend.mass` reports.
Only the point at s = 0.03 moves. The Blend is not calibrated, so it is the
same distortion in both cases.

```python
print(f"mass at zero of the Blend: {blend.mass:.4f}")
```

```text
mass at zero of the Blend: 0.0196
```

The figure draws the six distortions of the Tame book. The grey line is
g(s) = s, which prices a layer at its expected loss. The distance above it is
the margin:

```python
tame_g = {**fitted["Tame"], "blend": blend}
fig, axes = plt.subplots(2, 3, sharex=True, sharey=True, figsize=(7.0, 4.6))
points = [i / 100 for i in range(101)]
for ax, (name, dist) in zip(axes.flat, tame_g.items()):
    ax.plot(points, points, color="0.75")
    ax.plot(points, [dist.g(s) for s in points])
    ax.set_title(name)
for ax in axes[1]:
    ax.set_xlabel("exceedance probability s")
for ax in axes[:, 0]:
    ax.set_ylabel("g(s)")
fig.suptitle("The six distortions of the Tame book")
fig.tight_layout()
```

## 7. Pricing each unit with the five distortions

Tables 11.7 and 11.8 price each unit stand-alone with each calibrated
distortion, at its own VaR. They also price the book total, gross and net. The
tables also print margin, loss ratio, capital, return and leverage, which the
pentagon derives from the premiums, so the code compares the premiums only.
The last column is the largest relative gap to the book in the row:

```python
BOOK_117 = {  # Table 11.7 (Tame) premiums: A, B, SoP, total; net B, net SoP, net total
    "ccoc": (51.89, 52.98, 104.9, 103.4, 50.98, 102.9, 101.5),
    "ph": (51.90, 52.89, 104.8, 103.4, 50.91, 102.8, 101.8),
    "wang": (51.90, 52.86, 104.8, 103.4, 51.17, 103.1, 102.0),
    "dual": (51.90, 52.83, 104.7, 103.4, 51.44, 103.3, 102.1),
    "tvar": (51.90, 52.81, 104.7, 103.4, 51.63, 103.5, 102.2),
    "blend": (51.05, 51.62, 102.7, 101.9, 50.03, 101.1, 100.5),
}
BOOK_118 = {  # Table 11.8 (Cat/Non-Cat) premiums: Cat, Non-Cat, SoP, total; net Cat, net SoP, net total
    "ccoc": (34.98, 83.84, 118.8, 115.2, 25.72, 109.6, 105.9),
    "ph": (33.75, 86.36, 120.1, 115.2, 25.11, 111.5, 107.6),
    "wang": (32.82, 87.50, 120.3, 115.2, 25.68, 113.2, 108.8),
    "dual": (31.83, 88.59, 120.4, 115.2, 26.57, 115.2, 110.1),
    "tvar": (31.15, 89.17, 120.3, 115.2, 26.88, 116.0, 110.9),
    "blend": (26.43, 82.39, 108.8, 106.8, 21.17, 103.6, 101.8),
}


def by_unit(gross, net, p, distortions, book_values, reinsured, labels, title):
    rows = []
    for name, d in distortions.items():
        g, n = gross.price_stand_alone(d, p), net.price_stand_alone(d, p)
        got = [u.premium for u in g.stand_alone] + [g.sum.premium, g.total.premium,
                                                    n.stand_alone[reinsured].premium, n.sum.premium, n.total.premium]
        book = book_values[name]
        rows.append((name, *got, max(abs(a / b - 1) for a, b in zip(got, book))))
        rows.append((name + ", book", *book, float("nan")))
    frame = pd.DataFrame(rows, columns=["distortion", *labels, "gap"])
    return (pair_table(frame, {c: 2 for c in labels})
            .fmt_percent("gap", rows=list(range(0, len(rows), 2)), decimals=2)
            .tab_header(title, "Premiums in currency units")
            .tab_spanner("gross", labels[:4])
            .tab_spanner("net", labels[4:]))


by_unit(tame, tame_net, 0.9999, {**fitted["Tame"], "blend": blend}, BOOK_117, 1,
        ["A", "B", "SoP", "total", "B net", "SoP net", "net total"], "Table 11.7: Tame premiums by distortion")
```

```text
                 A      B     SoP   total  B net  SoP net  net total    gap
ccoc         51.89  52.98  104.87  103.42  50.98   102.86     101.55  0.05%
ccoc, book   51.89  52.98   104.9   103.4  50.98    102.9      101.5       
ph           51.90  52.89  104.79  103.42  50.91   102.80     101.85  0.05%
ph, book      51.9  52.89   104.8   103.4  50.91    102.8      101.8       
wang         51.90  52.86  104.76  103.42  51.17   103.07     101.99  0.04%
wang, book    51.9  52.86   104.8   103.4  51.17    103.1        102       
dual         51.90  52.83  104.73  103.42  51.44   103.34     102.11  0.04%
dual, book    51.9  52.83   104.7   103.4  51.44    103.3      102.1       
tvar         51.90  52.81  104.71  103.42  51.62   103.53     102.24  0.04%
tvar, book    51.9  52.81   104.7   103.4  51.63    103.5      102.2       
blend        51.05  51.62  102.67  101.89  50.03   101.08     100.53  0.03%
blend, book  51.05  51.62   102.7   101.9  50.03    101.1      100.5       
```

```python
by_unit(cnc, cnc_net, 0.999, {**fitted["Cat/Non-Cat"], "blend": blend}, BOOK_118, 0,
        ["Cat", "Non-Cat", "SoP", "total", "Cat net", "SoP net", "net total"],
        "Table 11.8: Cat/Non-Cat premiums by distortion")
```

```text
               Cat  Non-Cat     SoP   total  Cat net  SoP net  net total    gap
ccoc         34.98    83.84  118.81  115.15    25.72   109.56     105.90  0.04%
ccoc, book   34.98    83.84   118.8   115.2    25.72    109.6      105.9       
ph           33.75    86.36  120.11  115.15    25.11   111.47     107.55  0.04%
ph, book     33.75    86.36   120.1   115.2    25.11    111.5      107.6       
wang         32.82    87.50  120.32  115.15    25.68   113.18     108.79  0.04%
wang, book   32.82     87.5   120.3   115.2    25.68    113.2      108.8       
dual         31.83    88.59  120.42  115.15    26.57   115.16     110.13  0.04%
dual, book   31.83    88.59   120.4   115.2    26.57    115.2      110.1       
tvar         31.15    89.17  120.32  115.15    26.88   116.05     110.88  0.04%
tvar, book   31.15    89.17   120.3   115.2    26.88      116      110.9       
blend        26.43    82.39  108.82  106.76    21.17   103.56     101.82  0.04%
blend, book  26.43    82.39   108.8   106.8    21.17    103.6      101.8       
```

Every premium agrees with the book's four significant figures, to within one in
the last digit.

All five calibrated distortions charge the gross book the same 103.4 or 115.2,
by construction. They disagree about the units alone and about the net book. In
Cat/Non-Cat, the CCoC charges Cat alone 34.98 and the TVaR 31.15, and Non-Cat
goes the other way. The net book costs 105.9 under the CCoC and 110.9 under
TVaR, so the Cat cover (gross minus net premium) is worth 9.3 or 4.3. The value
depends on the distortion that the capital providers use.

The figure shows both points for Cat/Non-Cat, with the premiums of Table 11.8:

```python
cnc_names = list(BOOK_118)
fig, (left, right) = plt.subplots(1, 2, sharex=True, figsize=(7.0, 3.6))
left.bar(cnc_names, [BOOK_118[n][0] for n in cnc_names])
left.set_title("Cat alone")
left.set_ylabel("premium (currency units)")
right.bar(cnc_names, [BOOK_118[n][3] - BOOK_118[n][6] for n in cnc_names])
right.set_title("Value of the Cat cover")
right.set_ylabel("gross less net premium (currency units)")
for ax in (left, right):
    ax.set_xlabel("distortion")
fig.tight_layout()
```

Exercise 200 asks what reinsurance price is attractive. There is no answer
until you choose the distortion. The Blend is not calibrated, and it charges
less than all of them because the cost of its remote capital is low (2% to
10%).

## 8. When an indicator is priced below its expected value

Section 11.7 resolves an apparent paradox. In a portfolio priced by g, the
risk-adjusted probability of "the loss is at most x" is 1 - g(S(x)). This is
below its true probability F(x). Thus a policy that pays 1 in good years seems
to have a price below its expected loss.

Stand-alone, this is not true. As its own risk, the indicator is a Bernoulli
with probability F(x), and its price is g(F(x)), which is above F(x). The book
uses a lognormal with mean 10 and CV 0.5 and a PH with rho 0.7:

```python
x_loss = Lognormal.from_mean_cv(10.0, 0.5)
ph7 = Distortion.proportional_hazard(0.7)
for p in (0.5, 0.9):
    x0 = x_loss.quantile(p)
    print(f"x = {x0:6.3f}: P(X <= x) {p:.2f}, stand-alone price {ph7.g(p):.3f}, "
          f"price as a part of X {1 - ph7.g(1 - p):.3f}")
print(f"(run time of this guide: {'under' if time.perf_counter() - started < 15 else 'over'} 15 s)")
```

```text
x =  8.944: P(X <= x) 0.50, stand-alone price 0.616, price as a part of X 0.384
x = 16.385: P(X <= x) 0.90, stand-alone price 0.929, price as a part of X 0.800
(run time of this guide: under 15 s)
```

In the portfolio, the indicator pays in the states that the distortion gives
less weight. Thus it is worth less than its expected value. It is a hedge, and
to sell it is like to buy reinsurance. Alone, the underwriter orders the states
by the outcome of the indicator and charges a margin. The same difference,
price in a portfolio against price alone, explains the stand-alone columns of
section 7.
