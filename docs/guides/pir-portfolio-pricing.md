# Pricing Insurance Risk, Part II: portfolio pricing in prospicio

Part II of Mildenhall and Major's *Pricing Insurance Risk: Theory and Practice*
(Wiley, 2022) is chapters 8 to 11. It prices a whole portfolio in two ways:

- The classical approach (chapters 8 and 9). It applies premium calculation
  principles to each unit or to the total. It also covers the portfolio
  constant cost of capital (CCoC), ruin theory and option pricing.
- The modern approach (chapters 10 and 11). The premium is a distortion of the
  loss capped at the assets, which is a spectral risk measure (SRM). The book
  calibrates five representative distortions to one target and compares them.

This guide runs those chapters in prospicio and prints the book's numbers
beside the prospicio numbers.

The book's text belongs to the authors, and this guide explains it in new
words. Exhibit numbers are the book's. Related guides:

- Risk measures (VaR, TVaR, coherence, the representations of an SRM):
  [Pricing Insurance Risk, Part I](pir-risk-measures.md).
- The split of a portfolio price over its units (the natural allocation, Part
  III of the book): [the case studies guide](pricing-insurance-risk-case-studies.md)
  and [CAS Monograph 15 reproduced](capital-modeling-monograph-15.md).

In this guide, the code always prices a unit on its own (stand-alone) or prices
the total as a whole.

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
units. The 80% and 99% quantiles of B are 56.17 and 69.08. The 90% and 99.5%
quantiles of Cat are 41.11 and 120.75. `Grid.map` makes the net unit exactly on
the grid. A `Tower` with `on_grid` gives the same distribution by FFT. It sets
the FFT round-off (approximately 1e-19 where the probability must be zero) to
zero, and section 3 shows why this is important.

## 2. The portfolio CCoC price

Chapter 8 gives the classical portfolio price, and every practice chapter uses
it as the benchmark. This price holds assets at the VaR of the total and
charges a constant cost of capital on the capital. With a 10% return, the
premium is P = v E[min(X, a)] + d a, where v = 1 / 1.1 and d = 1 - v. The
insurance pentagon then gives the margin, capital, loss ratio and leverage.

Table 7.2 prints the result for each book, gross and net. Table 9.11 compares
the total with the sum of the units priced stand-alone, each at its own VaR:

```python
ccoc = Distortion.ccoc(0.10)
BOOK_72 = {  # Table 7.2: loss, margin, premium, capital, assets, leverage
    ("Tame", "gross"): (100.00, 3.42, 103.4, 34.23, 137.7, 3.02),
    ("Tame", "net"): (99.08, 2.46, 101.5, 24.64, 126.2, 4.12),
    ("Cat/Non-Cat", "gross"): (99.95, 15.21, 115.2, 152.1, 267.2, 0.757),
    ("Cat/Non-Cat", "net"): (97.73, 8.17, 105.9, 81.68, 187.6, 1.3),
}
BOOKS = {("Tame", "gross"): (tame, 0.9999), ("Tame", "net"): (tame_net, 0.9999),
         ("Cat/Non-Cat", "gross"): (cnc, 0.999), ("Cat/Non-Cat", "net"): (cnc_net, 0.999)}
print(f"{'':19s} {'loss':>7s} {'margin':>7s} {'premium':>8s} {'capital':>8s} {'assets':>7s} {'P/Q':>6s}")
for key, (book, p) in BOOKS.items():
    t = book.price(ccoc, p=p).total
    print(f"{key[0]:11s} {key[1]:5s}   {t.loss:7.2f} {t.margin:7.2f} {t.premium:8.2f} {t.capital:8.2f} "
          f"{t.assets:7.2f} {t.premium_to_capital:6.3f}")
    print(f"{'':19s} " + " ".join(f"{b:>7}" for b in BOOK_72[key]) + "   (book)")
```

```text
                       loss  margin  premium  capital  assets    P/Q
Tame        gross    100.00    3.42   103.42    34.23  137.66  3.021
                      100.0    3.42   103.4   34.23   137.7    3.02   (book)
Tame        net       99.08    2.46   101.55    24.64  126.19  4.121
                      99.08    2.46   101.5   24.64   126.2    4.12   (book)
Cat/Non-Cat gross     99.95   15.21   115.15   152.05  267.20  0.757
                      99.95   15.21   115.2   152.1   267.2   0.757   (book)
Cat/Non-Cat net       97.73    8.17   105.90    81.68  187.58  1.297
                      97.73    8.17   105.9   81.68   187.6     1.3   (book)
```

All 24 figures agree. The book's leverage is premium to capital, which is the
`premium_to_capital` of the pentagon. The loss is the expected loss within the
assets, so it includes the effect of default. For example, Cat/Non-Cat gross
has a loss of 99.95, not 100.

Stand-alone, each unit holds its own 99.99% or 99.9% VaR. The units together
need more capital than the book (Tables 9.12, 9.13 and 9.11):

```python
BOOK_SA = {  # Tables 9.12, 9.13: stand-alone premium and capital by unit; Table 9.11's sums
    ("Tame", "gross"): ([51.89, 52.98], [18.88, 29.82], 104.9, 48.69),
    ("Tame", "net"): ([51.89, 50.98], [18.88, 18.91], 102.9, 37.79),
    ("Cat/Non-Cat", "gross"): ([34.98, 83.84], [150.3, 38.41], 118.8, 188.7),
    ("Cat/Non-Cat", "net"): ([25.72, 83.84], [79.92, 38.41], 109.6, 118.3),
}
for key, (book, p) in BOOKS.items():
    sa = book.price_stand_alone(ccoc, p)
    prem, cap, sum_p, sum_q = BOOK_SA[key]
    units = ", ".join(f"{u.premium:.2f}/{u.capital:.2f} ({bp}/{bq})" for u, bp, bq in zip(sa.stand_alone, prem, cap))
    print(f"{key[0]:11s} {key[1]:5s} by unit P/Q {units}")
    print(f"{'':17s} sum P {sa.sum.premium:6.2f} ({sum_p}), total {sa.total.premium:6.2f}; "
          f"sum Q {sa.sum.capital:6.2f} ({sum_q}), total {sa.total.capital:6.2f}; "
          f"premium {sa.total.premium / sa.sum.premium - 1:+.1%}, capital {sa.total.capital / sa.sum.capital - 1:+.1%}")
```

```text
Tame        gross by unit P/Q 51.89/18.88 (51.89/18.88), 52.98/29.82 (52.98/29.82)
                  sum P 104.87 (104.9), total 103.42; sum Q  48.69 (48.69), total  34.23; premium -1.4%, capital -29.7%
Tame        net   by unit P/Q 51.89/18.88 (51.89/18.88), 50.98/18.92 (50.98/18.91)
                  sum P 102.86 (102.9), total 101.55; sum Q  37.79 (37.79), total  24.64; premium -1.3%, capital -34.8%
Cat/Non-Cat gross by unit P/Q 34.98/150.31 (34.98/150.3), 83.84/38.41 (83.84/38.41)
                  sum P 118.81 (118.8), total 115.15; sum Q 188.72 (188.7), total 152.05; premium -3.1%, capital -19.4%
Cat/Non-Cat net   by unit P/Q 25.72/79.92 (25.72/79.92), 83.84/38.41 (83.84/38.41)
                  sum P 109.56 (109.6), total 105.90; sum Q 118.33 (118.3), total  81.68; premium -3.3%, capital -31.0%
```

These are the book's figures, with the diversification that Table 9.11
reports. The pool of the Tame units saves 1.4% of premium and 29.7% of capital.
The pool of Cat with Non-Cat saves 3.0% and 19.4%. Premium decreases less than
capital, because most of the premium is expected loss, and pooling does not
change the expected loss. The thin tails of Tame give the most diversification.

## 3. Classical premium principles

Section 8.4 describes the classical premium calculation principles (PCPs).
Section 9.1 applies ten of them to the case studies in two steps:

1. Calibrate each principle so that the gross total costs the CCoC premium
   above.
2. Apply the principle stand-alone to each unit, to the net units and to the
   net total.

prospicio has all of them as `classical_premium(principle, x, loading)`, and
`calibrate_classical` finds a loading. The book's Fischer principle uses the
power 2, which is the prospicio default:

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


def classical_table(gross_total, grids, target, book_values):
    largest = 0.0
    for principle, (book_t, *book_p) in book_values.items():
        t = calibrate_classical(principle, gross_total, target)
        got = [classical_premium(principle, g, t) for g in grids]
        largest = max([largest] + [abs(a - b) for a, b in zip(got, book_p)])
        print(f"{principle:18s} {t:5.3f} ({book_t:5.3f}) " + " ".join(f"{g:8.3f}" for g in got))
    print(f"largest difference from the book: {largest:.3f}")


tame_target = tame.price(ccoc, p=0.9999).total.premium
tame_total, tame_net_total = total_grid(tame), total_grid(tame_net)
print(f"{'':18s} loading         {'A':>8s} {'B net':>8s} {'B':>8s} {'net':>8s} {'total':>8s}")
classical_table(tame_total, [a_unit, b_net, b_unit, tame_net_total, tame_total], tame_target, BOOK_92)
```

```text
                   loading                A    B net        B      net    total
expected_value     0.034 (0.034)   51.712   50.764   51.712  102.476  103.423
variance           0.042 (0.042)   51.053   50.619   52.370  101.672  103.423
esscher            0.041 (0.041)   51.034   50.468   52.389  101.502  103.423
exponential        0.080 (0.080)   51.028   50.419   52.395  101.447  103.423
semi_variance      0.080 (0.080)   51.051   50.330   52.425  101.411  103.423
var                0.659 (0.659)   51.906   52.750   52.750  102.672  103.422
dutch              0.953 (0.953)   51.899   51.498   52.846  102.096  103.423
standard_deviation 0.380 (0.380)   51.899   51.376   52.848  102.060  103.423
fischer            0.523 (0.523)   51.897   51.149   52.881  101.906  103.423
largest difference from the book: 0.001
```

Every loading and premium in Table 9.2 agrees to the third decimal. The largest
difference, 0.001, is rounding. Table 9.3 also agrees. It subtracts the premium
of the total from the sum of the stand-alone premiums.

```python
print(f"{'':18s} {'SoP gross':>9s} {'delta':>7s} {'SoP net':>8s} {'delta':>7s}")
for principle in PRINCIPLES:
    t = calibrate_classical(principle, tame_total, tame_target)
    sop = classical_premium(principle, a_unit, t) + classical_premium(principle, b_unit, t)
    sop_net = classical_premium(principle, a_unit, t) + classical_premium(principle, b_net, t)
    net = classical_premium(principle, tame_net_total, t)
    print(f"{principle:18s} {sop:9.3f} {sop - tame_target:7.3f} {sop_net:8.3f} {sop_net - net:7.3f}")
print("book SoP gross: variance, Esscher, exponential 103.423; semivariance 103.477; VaR 104.656; "
      "Dutch 104.745; standard deviation 104.747; Fischer 104.778")
```

```text
                   SoP gross   delta  SoP net   delta
expected_value       103.423   0.000  102.476   0.000
variance             103.423   0.000  101.672   0.000
esscher              103.423   0.000  101.502   0.000
exponential          103.423   0.000  101.447   0.000
semi_variance        103.477   0.054  101.381  -0.030
var                  104.656   1.233  104.656   1.984
dutch                104.745   1.322  103.397   1.301
standard_deviation   104.747   1.324  103.275   1.214
fischer              104.778   1.355  103.046   1.140
book SoP gross: variance, Esscher, exponential 103.423; semivariance 103.477; VaR 104.656; Dutch 104.745; standard deviation 104.747; Fischer 104.778
```

The delta columns show the book's main point:

- Expected value, variance, Esscher and exponential are additive for
  independent units, so pooling earns nothing.
- VaR and the deviation principles (Dutch, standard deviation, Fischer) are
  subadditive. They charge the pool approximately 1.3 less than the parts.
- The semivariance is neither. Its delta changes sign between gross and net
  (0.054 and -0.029 in the book).

The formula of a principle sets its view of diversification. The data does not
change that view.

Two principles are sensitive to how the code makes the net B grid of Tame. The
exponential and Esscher premiums weight an outcome x by exp(t x). On a grid
that goes to 1,024, the factor exp(0.08 x) is approximately 1e35 at the top.
Thus a round-off of 1e-19 from an FFT at the top of the grid would give a
premium of hundreds. `Tower.on_grid` sets probabilities below machine epsilon
to zero, as `remove_fuzz` in `aggregate` does. Thus its net grid gives the same
price as the exact grid:

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
9.6). The columns are Cat net, Cat gross, Non-Cat, total net and total gross:

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
print(f"{'':18s} loading         {'Cat net':>8s} {'Cat':>8s} {'Non-Cat':>8s} {'net':>8s} {'total':>8s}")
classical_table(cnc_total, [cat_net, cat, non_cat, total_grid(cnc_net), cnc_total], cnc_target, BOOK_95)
```

```text
                   loading          Cat net      Cat  Non-Cat      net    total
expected_value     0.152 (0.152)   20.480   23.030   92.121  112.601  115.151
var                0.818 (0.818)   30.109   30.109   90.734  113.391  115.141
variance           0.028 (0.028)   22.572   31.139   84.012  106.584  115.151
dutch              1.859 (1.859)   27.170   32.000   88.882  110.638  115.151
semi_variance      0.040 (0.040)   22.512   32.782   83.091  105.193  115.151
standard_deviation 0.650 (0.650)   26.301   32.991   87.797  109.331  115.151
fischer            0.776 (0.776)   26.248   33.916   86.843  108.380  115.151
largest difference from the book: 0.003
```

Seven principles agree to within 0.003. This is the size of difference that two
different truncations of a lognormal tail give. The Esscher and exponential
principles do not agree, and you should not expect them to agree:

```python
BOOK_EXP = {"esscher": (0.011, 23.575, 35.944, 81.561, 105.054),
            "exponential": (0.014, 24.760, 39.697, 81.059, 105.619)}
for principle, (book_t, *book_p) in BOOK_EXP.items():
    t = calibrate_classical(principle, cnc_total, cnc_target)
    got = [classical_premium(principle, g, t) for g in (cat_net, cat, non_cat, cnc_net_total)]
    print(f"{principle:11s} {t:.4f} ({book_t}) " + " ".join(f"{g:7.3f} ({b:7.3f})" for g, b in zip(got, book_p)))
lower = cat.probs[: POINTS // 2]
scale = sum(lower)
top_half = Grid(STEP, [q / scale for q in lower])
print(f"exponential premium of Cat at loading 0.014: grid to 1,024 {classical_premium('exponential', cat, 0.014):.3f}, "
      f"grid to 512 {classical_premium('exponential', top_half, 0.014):.3f}")
```

```text
esscher     0.0103 (0.011)  22.760 ( 23.575)  33.639 ( 35.944)  81.512 ( 81.561) 104.272 (105.054)
exponential 0.0138 (0.014)  22.820 ( 24.760)  34.139 ( 39.697)  81.012 ( 81.059) 103.832 (105.619)
exponential premium of Cat at loading 0.014: grid to 1,024 35.497, grid to 512 25.189
```

A lognormal has no exponential moment: E[exp(t X)] is infinite for every t > 0.
Thus both premiums are artifacts of the point where the grid stops. When the
grid stops at 512, not 1,024, the Cat premium moves by several units. The book
makes the same point for the Hurricane/SCS case, and it does not apply these
principles there. On Cat/Non-Cat, its numbers are what its grid gave. Do not
use these two principles for a loss without a light tail.

## 4. Ruin theory and the natural scale of a market

Section 9.3 uses the Pollaczeck-Khinchine formula for the probability of
eventual ruin of a compound Poisson surplus process. In prospicio this is
`eventual_ruin`. The example is a liability line with lognormal claims of mean
50,000 and CV 10, limited at 1M, 5M or 10M.

With a margin r on expected loss, the formula gives the starting surplus u at
which eventual ruin has probability p. The annual margin, n r E[X] for n claims
a year, must pay a cost of capital delta on that surplus. Thus a market needs
at least n = delta u / (r E[X]) claims a year (Eq. (9.2)). Table 9.15
tabulates n with delta = 0.1:

```python
from prospicio.aggregate import eventual_ruin

claim = Lognormal.from_mean_cv(50_000.0, 10.0)
BOOK_915 = {  # Table 9.15: limit -> {p: (n at r = 0.05, 0.1, 0.25)}
    1e6: {0.01: (10_172, 2_655, 476.1), 0.1: (5_021, 1_294, 224.5)},
    1e7: {0.01: (33_610, 9_087, 1_765), 0.1: (16_384, 4_324, 793.6)},
}
for limit, rows in BOOK_915.items():
    # 10,000 steps up to the limit; the claim's mass above the limit lands on it
    capped = Grid.rounding(claim, limit / 10_000, 10_001)[0].map(lambda x: min(x, limit))[0]
    for p, book in rows.items():
        counts = []
        for r in (0.05, 0.1, 0.25):
            u = eventual_ruin(capped, rho=r, points=2**18).capital(p, interpolate=True)
            counts.append(0.1 * u / (r * capped.mean()))
        print(f"limit {limit:>10,.0f} p {p:4}: n " + " ".join(f"{n:8.1f}" for n in counts)
              + "   book " + " ".join(f"{b:>8}" for b in book)
              + "   ratio " + " ".join(f"{b / n:.2f}" for n, b in zip(counts, book)))
```

```text
limit  1,000,000 p 0.01: n   1019.0    266.0     47.7   book    10172     2655    476.1   ratio 9.98 9.98 9.98
limit  1,000,000 p  0.1: n    503.0    129.7     22.5   book     5021     1294    224.5   ratio 9.98 9.98 9.99
limit 10,000,000 p 0.01: n   3362.0    909.0    176.5   book    33610     9087     1765   ratio 10.00 10.00 10.00
limit 10,000,000 p  0.1: n   1639.0    432.5     79.4   book    16384     4324    793.6   ratio 10.00 10.00 10.00
```

The counts are a constant factor of 10 below the counts in Table 9.15, at every
limit, probability and margin. The book's table is exactly u / (r E[X]), which
is Eq. (9.2) with delta = 1. Thus the table appears to leave out the cost of
capital of 0.1 that its caption gives. The example in the text has the same
factor. It gives 39,776 claims, approximately 800,000 policies, for a 1M limit
at p = 0.01 and r = 0.025.

Trust the ratio of the two columns. The factor is the same in every cell. Thus
the book's conclusions about how the scale changes with the limit, the margin
and the ruin probability are correct. (The grid causes the remaining 0.2%.)

## 5. The modern view: a price for each layer of assets

Chapter 10 starts from the layer. The risk that losses consume the l-th unit of
assets is a Bernoulli with probability s = S(l). A market prices this risk at
g(s), where g is a distortion. The sum over the layers gives the premium for a loss X with
assets a: the integral of g(S(x)) from 0 to a. This is the SRM of the capped
loss X ^ a.

A Bernoulli layer has expected loss s, premium g(s), margin g(s) - s and
capital 1 - g(s). Thus its loss ratio is s / g(s), its return on capital is
(g(s) - s) / (1 - g(s)), and its premium leverage is g(s) / (1 - g(s)).
Exercise 142 tabulates these for the proportional hazard distortions s^rho
(Table 10.3):

```python
BOOK_103 = {  # Table 10.3 at s = 0.01, 0.1, 0.5: loss ratio, risk return, premium leverage
    0.4: ((0.0631, 0.2512, 0.6598), (0.1765, 0.4953, 1.0649), (0.1883, 0.6614, 3.1298)),
    0.5: ((0.1000, 0.3162, 0.7071), (0.1000, 0.3162, 0.7071), (0.1111, 0.4625, 2.4142)),
    0.7: ((0.2512, 0.5012, 0.8123), (0.0310, 0.1243, 0.3006), (0.0415, 0.2493, 1.6013)),
    0.9: ((0.6310, 0.7943, 0.9330), (0.0059, 0.0296, 0.0773), (0.0161, 0.1440, 1.1546)),
}
for rho, book in BOOK_103.items():
    g = Distortion.proportional_hazard(rho).g
    stats = [[s / g(s) for s in (0.01, 0.1, 0.5)],
             [(g(s) - s) / (1 - g(s)) for s in (0.01, 0.1, 0.5)],
             [g(s) / (1 - g(s)) for s in (0.01, 0.1, 0.5)]]
    worst = max(abs(a - b) for row, brow in zip(stats, book) for a, b in zip(row, brow))
    print(f"rho {rho}: loss ratio {[round(v, 4) for v in stats[0]]}, return {[round(v, 4) for v in stats[1]]}, "
          f"leverage {[round(v, 4) for v in stats[2]]}  largest difference {worst:.4f}")
```

```text
rho 0.4: loss ratio [0.0631, 0.2512, 0.6598], return [0.1765, 0.4953, 1.0649], leverage [0.1883, 0.6614, 3.1298]  largest difference 0.0000
rho 0.5: loss ratio [0.1, 0.3162, 0.7071], return [0.1, 0.3162, 0.7071], leverage [0.1111, 0.4625, 2.4142]  largest difference 0.0000
rho 0.7: loss ratio [0.2512, 0.5012, 0.8123], return [0.031, 0.1243, 0.3006], leverage [0.0415, 0.2493, 1.6013]  largest difference 0.0000
rho 0.9: loss ratio [0.631, 0.7943, 0.933], return [0.0059, 0.0296, 0.0773], leverage [0.0161, 0.144, 1.1546]  largest difference 0.0000
```

Table 10.3 agrees in every cell.

The pattern is the main lesson of the chapter. A concave g charges remote
layers (small s) a low loss ratio. But these layers need much capital for each
unit of premium, so their return is also low. The return on capital increases
with s. Only the CCoC distortion, a straight line from (0, d) to (1, 1), earns
the same return on every layer. For this reason it is "constant".

The Simple Discrete Example of Part I shows the capped premium as a function of
the assets. In Tables 11.1 and 11.3, the PH with rho 0.5 prices the uncapped
loss at 51.156 and the loss capped at 80 at 42.828:

```python
OUTCOMES = [0, 1, 8, 9, 10, 11, 90, 98, 100]
SIXTEENTHS = [4, 2, 2, 1, 2, 1, 2, 1, 1]
discrete = Sampled([float(x) for x, k in zip(OUTCOMES, SIXTEENTHS) for _ in range(k)])
ph = Distortion.proportional_hazard(0.5)
print("assets   E[X ^ a]   PH premium   margin   capital")
for a in (10, 50, 80, 90, 100):
    capped = Sampled([min(x, a) for x in discrete.draws])
    premium = ph.measure(capped)
    print(f"{a:6d} {capped.mean():10.3f} {premium:12.3f} {premium - capped.mean():8.3f} {a - premium:9.3f}")
print("book: a = 80 premium 42.828 (Table 11.3); a = 100 premium 51.156 (Table 11.1)")
```

```text
assets   E[X ^ a]   PH premium   margin   capital
    10      6.062        7.769    1.706     2.231
    50     16.125       27.828   11.703    22.172
    80     23.625       42.828   19.203    37.172
    90     26.125       47.828   21.703    42.172
   100     27.250       51.156   23.906    48.844
book: a = 80 premium 42.828 (Table 11.3); a = 100 premium 51.156 (Table 11.1)
```

Premium funds most of the low layers. With assets of 10, the premium is 7.8 and
the capital is 2.2, because losses consume those layers in most outcomes. From
11 to 90, losses consume each added unit of assets only in the 25% of outcomes
above 11. Each such unit adds g(0.25) = 0.5 of premium and 0.5 of capital. The
top layers add mostly capital.

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
for name, (book, p) in (("Tame", (tame, 0.9999)), ("Cat/Non-Cat", (cnc, 0.999))):
    fitted[name] = {f: book.calibrate(f, p=p, return_on_capital=0.10) for f in FAMILIES}
    values = [next(iter(d.parameters.values())) for d in fitted[name].values()]
    print(f"{name:11s} " + "  ".join(f"{f} {v:.3f} ({b})" for f, v, b in zip(FAMILIES, values, BOOK_115[name])))
```

```text
Tame        ccoc 0.100 (0.1)  ph 0.683 (0.683)  wang 0.375 (0.375)  dual 1.576 (1.576)  tvar 0.227 (0.227)
Cat/Non-Cat ccoc 0.100 (0.1)  ph 0.596 (0.596)  wang 0.611 (0.611)  dual 2.463 (2.463)  tvar 0.482 (0.482)
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
print("g at the specified s:", [round(blend.g(s), 4) for s in LAYERS], " mass at zero", round(blend.mass, 4))
print("specified           :", [round((s + nu) / (1 + nu), 4) for s, nu in LAYERS.items()])
```

```text
g at the specified s: [0.0, 0.0388, 0.0713, 0.1038, 0.1667, 0.2727]  mass at zero 0.0196
specified           : [0.0196, 0.0388, 0.0673, 0.1038, 0.1667, 0.2727]
```

At s = 0 a distortion is 0 by definition. The specified 0.0196 is the mass at
zero, which is the rate on line of the most remote layer. `mass` reports it.
Only the point at s = 0.03 moves (0.0673 to 0.0713). The Blend is not
calibrated, so it is the same distortion in both cases.

## 7. Pricing each unit with the five distortions

Tables 11.7 and 11.8 price each unit stand-alone with each calibrated
distortion, at its own VaR. They also price the book total, gross and net. The
code compares the premiums. The tables also print margin, loss ratio, capital,
return and leverage, which the pentagon derives from the premiums:

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


def by_unit(gross, net, p, distortions, book_values, reinsured):
    largest = 0.0
    for name, d in distortions.items():
        g, n = gross.price_stand_alone(d, p), net.price_stand_alone(d, p)
        got = [u.premium for u in g.stand_alone] + [g.sum.premium, g.total.premium,
                                                    n.stand_alone[reinsured].premium, n.sum.premium, n.total.premium]
        book = book_values[name]
        largest = max([largest] + [abs(a / b - 1) for a, b in zip(got, book)])
        print(f"{name:5s} " + " ".join(f"{a:7.2f}" for a in got) + "   book " + " ".join(f"{b}" for b in book))
    print(f"largest relative difference from the book: {largest:.2%}")


print("Tame:   A       B     SoP   total   B net SoP net net total")
by_unit(tame, tame_net, 0.9999, {**fitted["Tame"], "blend": blend}, BOOK_117, reinsured=1)
print("Cat/Non-Cat: Cat Non-Cat SoP  total  Cat net SoP net net total")
by_unit(cnc, cnc_net, 0.999, {**fitted["Cat/Non-Cat"], "blend": blend}, BOOK_118, reinsured=0)
```

```text
Tame:   A       B     SoP   total   B net SoP net net total
ccoc    51.89   52.98  104.87  103.42   50.98  102.86  101.55   book 51.89 52.98 104.9 103.4 50.98 102.9 101.5
ph      51.90   52.89  104.79  103.42   50.91  102.80  101.85   book 51.9 52.89 104.8 103.4 50.91 102.8 101.8
wang    51.90   52.86  104.76  103.42   51.17  103.07  101.99   book 51.9 52.86 104.8 103.4 51.17 103.1 102.0
dual    51.90   52.83  104.73  103.42   51.44  103.34  102.11   book 51.9 52.83 104.7 103.4 51.44 103.3 102.1
tvar    51.90   52.81  104.71  103.42   51.62  103.53  102.24   book 51.9 52.81 104.7 103.4 51.63 103.5 102.2
blend   51.05   51.62  102.67  101.89   50.03  101.08  100.53   book 51.05 51.62 102.7 101.9 50.03 101.1 100.5
largest relative difference from the book: 0.05%
Cat/Non-Cat: Cat Non-Cat SoP  total  Cat net SoP net net total
ccoc    34.98   83.84  118.81  115.15   25.72  109.56  105.90   book 34.98 83.84 118.8 115.2 25.72 109.6 105.9
ph      33.75   86.36  120.11  115.15   25.11  111.47  107.55   book 33.75 86.36 120.1 115.2 25.11 111.5 107.6
wang    32.82   87.50  120.32  115.15   25.68  113.18  108.79   book 32.82 87.5 120.3 115.2 25.68 113.2 108.8
dual    31.83   88.59  120.42  115.15   26.57  115.16  110.13   book 31.83 88.59 120.4 115.2 26.57 115.2 110.1
tvar    31.15   89.17  120.32  115.15   26.88  116.05  110.88   book 31.15 89.17 120.3 115.2 26.88 116.0 110.9
blend   26.43   82.39  108.82  106.76   21.17  103.56  101.82   book 26.43 82.39 108.8 106.8 21.17 103.6 101.8
largest relative difference from the book: 0.04%
```

Every premium in both tables agrees with the four significant figures of the
book, to within one in the last digit. For example, the PH net total is 101.85
against a printed 101.8, and the TVaR net B is 51.62 against 51.63.

Read across the rows. All five calibrated distortions charge the gross book the
same 103.4 or 115.2, by construction. But they disagree about the units alone
and about the net book. In Cat/Non-Cat, the CCoC charges Cat alone 34.98 and
the TVaR charges 31.15, and Non-Cat goes the other way. The net book costs
105.9 under the CCoC and 110.9 under TVaR. Thus the value of the Cat cover to
the insurer (gross minus net premium) is from 9.3 to 4.3. The value depends on
the distortion that the capital providers use.

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
