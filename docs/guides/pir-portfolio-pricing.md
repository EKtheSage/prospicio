# Pricing Insurance Risk, Part II: portfolio pricing in prospicio

Part II of Mildenhall and Major's *Pricing Insurance Risk: Theory and Practice*
(Wiley, 2022), chapters 8 to 11, prices a whole portfolio. Chapters 8 and 9 are
the classical approach: premium calculation principles applied to each unit or
to the total, the portfolio constant cost of capital (CCoC), ruin theory and
option pricing. Chapters 10 and 11 are the modern one: the premium is a
distortion of the loss capped at the assets, a spectral risk measure (SRM), and
the five representative distortions are calibrated to one target and compared.
This guide runs those chapters in prospicio and prints the book's numbers as
literals beside prospicio's.

The text is the authors'; what follows explains it in new words. Exhibit numbers
are the book's. Risk measures themselves (VaR, TVaR, coherence, the
representations of an SRM) are in
[Pricing Insurance Risk, Part I](pir-risk-measures.md). Splitting a portfolio's
price over its units (the natural allocation, Part III of the book) is in
[the case studies guide](pricing-insurance-risk-case-studies.md) and
[CAS Monograph 15 reproduced](capital-modeling-monograph-15.md); here a unit is
always priced on its own (stand-alone) or the total as a whole.

## 1. Two books to price

The practice chapters use the book's case studies. Two of them are cheap enough
to run in a guide. **Tame** is two gamma units with mean 50 and CVs 0.10 and
0.15, at a 99.99% VaR capital standard, with an aggregate cover of 12.91 excess
of 56.17 on unit B in the net view. **Cat/Non-Cat** is a gamma Non-Cat unit
(mean 80, CV 0.15) and a lognormal Cat unit (mean 20, CV 1), at 99.9%, with
79.64 excess of 41.11 on Cat (Table 7.3). Every unit goes on `aggregate`'s grid,
a step of 1/64 and 2^16 points, and a net unit is the gross unit less the
cover's recovery, outcome by outcome:

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

The covers' attachments and limits are read off the gross units' quantiles: B's
80% and 99% quantiles are 56.17 and 69.08, Cat's 90% and 99.5% are 41.11 and
120.75. Making the net unit with `Grid.map` keeps it exact on the grid. A
`Tower` with `on_grid` gives the same distribution by FFT, with round-off of
about 1e-19 where the probability should be zero; section 3 shows when that
matters (issue #367).

## 2. The portfolio CCoC price

Chapter 8's classical portfolio price, used as the benchmark in every practice
chapter, holds assets at the VaR of the total and charges a constant cost of
capital on the capital: with a 10% return, the premium is P = v E[min(X, a)] + d
a, where v = 1 / 1.1 and d = 1 - v. The insurance pentagon then gives the
margin, capital, loss ratio and leverage. Table 7.2 prints the result for each
book, gross and net; Table 9.11 compares the total with the sum of the units
priced stand-alone (each at its own VaR):

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

All 24 figures agree. The book's leverage is premium to capital, the pentagon's
`premium_to_capital`. The loss is the expected loss within the assets, so it
allows for default (99.95 rather than 100 for Cat/Non-Cat gross).

Stand-alone, each unit holds its own 99.99% or 99.9% VaR, and the units together
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

These are the book's figures, including the diversification it reports in
Table 9.11: pooling the Tame units saves 1.4% of premium and 29.7% of capital,
pooling Cat with Non-Cat 3.0% and 19.4%. Premium falls less than capital because
most of the premium is expected loss, which pooling does not change. The thin
tails of Tame diversify best.

## 3. Classical premium principles

Section 8.4 surveys the classical premium calculation principles (PCPs) and
Section 9.1 applies ten of them to the case studies: each is calibrated so that
the gross total costs the CCoC premium above, and then applied stand-alone to
each unit, to the net units and to the net total. They are all in prospicio as
`classical_premium(principle, x, loading)`, with `calibrate_classical` to find a
loading; the book's Fischer principle uses the power 2, prospicio's default:

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

Every loading and premium in Table 9.2 agrees to its third decimal (the largest
difference, 0.001, is rounding). So does Table 9.3, which subtracts: the sum of
the stand-alone premiums against the total's.

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

The delta columns are the book's point. Expected value, variance, Esscher and
exponential are additive for independent units, so pooling earns nothing; VaR
and the deviation-based principles (Dutch, standard deviation, Fischer) are
subadditive and charge the pool about 1.3 less than the parts. The semivariance
is neither: its delta changes sign between gross and net (0.054 and -0.029 in
the book). A principle's view of diversification is built into its formula, not
learned from the data.

Two principles misbehave on Tame's net B as soon as the net grid comes from an
FFT. The exponential and Esscher premiums weight an outcome x by exp(t x), and
on a grid that reaches 1,024 the factor exp(0.08 x) is about 1e35 at the top,
so the 1e-19 round-off a `Tower` grid carries there becomes a premium of
hundreds:

```python
from prospicio.distributions import Count
from prospicio.reinsurance import Layer, Tower

b_net_fft = Tower([Layer.stop_loss("B", 12.91, 56.17)]).on_grid(Count.fixed(1), b_unit, POINTS).net
t_exp = calibrate_classical("exponential", tame_total, tame_target)
print(f"exponential premium of net B: by map {classical_premium('exponential', b_net, t_exp):.3f}, "
      f"from the FFT {classical_premium('exponential', b_net_fft, t_exp):.1f} (book 50.420)")
print(f"largest probability above 150: FFT {max(b_net_fft.probs[150 * 64:]):.1e}, map exactly zero")
```

```text
exponential premium of net B: by map 50.419, from the FFT 509.4 (book 50.420)
largest probability above 150: FFT 7.0e-20, map exactly zero
```

Use the exact net (`Grid.map`) for these two principles, or set the round-off
to zero before pricing.

Cat/Non-Cat is where the principles spread out (Tables 9.5 and 9.6). The columns
are Cat net, Cat gross, Non-Cat, total net and total gross:

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

Seven of the principles agree to within 0.003, the size of the difference
between two grids' truncation of a lognormal tail. The Esscher and exponential
principles do not, and should not be expected to:

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

A lognormal has no exponential moment: E[exp(t X)] is infinite for every t > 0,
so both premiums are artifacts of where the grid stops, and cutting the grid
from 1,024 to 512 moves the Cat premium by several units. The book makes the
same point for the Hurricane/SCS case, where it does not apply them at all; on
Cat/Non-Cat its numbers are what its grid gave. Do not use these two principles
for a loss without a light tail.

## 4. Ruin theory and the natural scale of a market

Section 9.3 uses the Pollaczeck-Khinchine formula for the probability of
eventual ruin of a compound Poisson surplus process, here `eventual_ruin`. The
example is a liability line with lognormal claims of mean 50,000 and CV 10,
limited at 1M, 5M or 10M. With a margin r on expected loss, the formula gives
the starting surplus u for which eventual ruin has probability p, and since the
annual margin, n r E[X] for n claims a year, must pay a cost of capital delta
on that surplus, a market needs at least n = delta u / (r E[X]) claims a year
(Eq. (9.2)). Table 9.15 tabulates n with delta = 0.1:

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

The counts are a constant factor of 10 below Table 9.15's, at every limit,
probability and margin. The book's table is exactly u / (r E[X]), Eq. (9.2) with
delta = 1, so it appears to leave out the cost of capital of 0.1 that its
caption says it uses; the text's example (39,776 claims, about 800,000 policies,
for a 1M limit at p = 0.01 and r = 0.025) carries the same factor. The ratio of
the two columns is what to trust: the factor is the same in every cell, so the
book's conclusions about how the scale moves with the limit, the margin and the
ruin probability stand. (The remaining 0.2% is the grid.)

## 5. Option pricing models: not in prospicio

Sections 8.8 and 9.4 price insurance as a put option on the insurer's assets
(Table 9.16: the value of insurance and of equity for lognormal assets and
liabilities at several volatilities and asset-to-liability ratios). prospicio
has no option pricing model (issue #367); the formulas are short and a spreadsheet or scipy
reproduces them. The discounted cash flow and IRR models of Section 8.7 are
likewise outside prospicio, apart from the time value of money in
`price_over_payout` (see [Reinsurance price](reinsurance-price.md)).

## 6. The modern view: a price for each layer of assets

Chapter 10 starts from the layer: the risk that the l-th unit of assets is
consumed is a Bernoulli with probability s = S(l), and a market prices it at
g(s), a distortion. Summing over layers, the premium for a loss X backed by
assets a is the integral of g(S(x)) from 0 to a, the SRM of the capped loss
X ^ a. A Bernoulli layer has expected loss s, premium g(s), margin g(s) - s and
capital 1 - g(s), so its loss ratio is s / g(s), its return on capital
(g(s) - s) / (1 - g(s)) and its premium leverage g(s) / (1 - g(s)). Exercise 142
tabulates these for proportional hazard distortions s^rho (Table 10.3):

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

Table 10.3 agrees in every cell. The pattern is the chapter's main lesson: a
concave g charges remote layers (small s) a low loss ratio but, because they
need so much capital per unit of premium, a low return; the return on capital
rises with s. Only the CCoC distortion, a straight line from (0, d) to (1, 1),
earns the same return on every layer, which is the sense in which it is
"constant".

The Simple Discrete Example of Part I shows the capped premium as a function of
the assets (Tables 11.1 and 11.3: PH with rho 0.5 prices the uncapped loss at
51.156 and the loss capped at 80 at 42.828):

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

The low layers are funded mostly by premium: with assets of 10 the premium is
7.8 and the capital 2.2, because those layers are consumed in most outcomes.
From 11 to 90 each extra unit of assets is consumed only in the 25% of outcomes
above 11, and adds g(0.25) = 0.5 of premium and 0.5 of capital. The top layers
add mostly capital.

## 7. Calibrating the five distortions

Chapter 11 compares SRMs that charge the same total premium: the CCoC, the
proportional hazard (PH), Wang, dual power and TVaR families, calibrated to the
gross CCoC premium of section 2, plus a sixth, "Blend", a piecewise linear
distortion built from layer costs of capital. Table 11.5 prints the calibrated
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

All ten parameters agree. prospicio's `ccoc` takes the return r (0.10), the PH
the exponent rho, Wang the shift lambda, dual power the exponent beta and TVaR
the level p, as the book writes them. (The Cat/Non-Cat parameters are also in
the case studies guide, run at 99.9% against `aggregate` 0.30.1; Tame there is
at 99.9% too, while the book's Tame uses 99.99%.)

The Blend distortion of Exercise 197 (Table 11.6) sets a layer cost of capital
nu at a few exceedance probabilities s, from 2% for the remotest layers to 10% at
s = 0.2, and interpolates g(s) = (s + nu) / (1 + nu) between them.
`Distortion.convex` builds the smallest concave distortion through such points;
the book's point at s = 0.03 lies below the line through its neighbours, so the
concave version lifts it slightly:

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

At s = 0 a distortion is 0 by definition; the specified 0.0196 is the mass at
zero, the rate on line of the remotest layer, which `mass` reports. Only the
point at s = 0.03 moves (0.0673 to 0.0713). The Blend is not calibrated: it is
the same distortion in both cases.

## 8. Pricing each unit with the five distortions

Tables 11.7 and 11.8 price each unit stand-alone with each calibrated
distortion, at its own VaR, and the book total, gross and net. The premiums
(the tables also print margin, loss ratio, capital, return and leverage, all
derived from these through the pentagon):

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

Every premium in both tables agrees with the book's four significant figures,
to within one in the last digit (PH's net total is 101.85 against a printed
101.8, TVaR's net B 51.62 against 51.63). Read across the
rows: all five calibrated distortions charge the gross book the same 103.4 or
115.2 (by construction), but they disagree about the units alone and about the
net book. In Cat/Non-Cat, the CCoC charges Cat alone 34.98 and the TVaR 31.15;
Non-Cat the other way. The net book costs 105.9 under the CCoC and 110.9 under
TVaR, so the value of the Cat cover to the insurer (gross less net premium)
ranges from 9.3 to 4.3 depending on which distortion the capital providers
use. Exercise 200's question, what reinsurance price is attractive, has no
answer until the distortion is chosen. Blend, not calibrated, charges less
than all of them because its remote capital is cheap (2% to 10%).

## 9. When an indicator is priced below its expected value

Section 11.7 resolves an apparent paradox: in a portfolio priced by g, the
risk-adjusted probability of "the loss is at most x" is 1 - g(S(x)), which is
below its true probability F(x), so a policy paying 1 in good years seems to be
priced below its expected loss. Stand-alone it is not: as its own risk, the
indicator is a Bernoulli with probability F(x) and costs g(F(x)), above F(x).
The book's setting is a lognormal with mean 10 and CV 0.5 and a PH with rho 0.7:

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

Within the portfolio, the indicator pays in the states the distortion
down-weights, so it is worth less than its expected value: it is a hedge, and
selling it is like buying reinsurance. Alone, the underwriter orders the states
by the indicator's own outcome and charges a margin. The same distinction, price
in a portfolio against price alone, is behind the stand-alone columns of
section 8.

## 10. What this guide leaves out

- **Hurricane/SCS** (Tables 7.2, 9.8 to 9.10, 9.14, 11.9): it needs the
  2^19-point grid and the occurrence cover of the
  [case studies guide](pricing-insurance-risk-case-studies.md), which runs it at
  aggregate 0.30.1's settings; the book's Table 9.8 also prints its parameters
  one row out of line.
- **Option pricing and DCF models** (Sections 8.7, 8.8, 9.4, Table 9.16): not in
  prospicio (section 5).
- **Fitting distortions to cat bond spreads** (Section 11.6): the book's
  regressions run on its cat bond data, which it does not print.
  `Distortion.convex` builds a distortion from (expected loss, spread) points,
  and `risk-loads-and-distortions.md` calibrates to quoted layers.
- **Allocating the price to units** (Chapter 11's portfolio resolution onward,
  Part III): see the case studies and Monograph 15 guides.
- **Remaining figures** (11.6 to 11.11: loss ratio, return and leverage by
  exceedance probability and by asset level): `d.g` and section 6's loop give
  every curve; there are no printed values to compare.
