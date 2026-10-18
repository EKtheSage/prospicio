# Basic ratemaking: Werner and Modlin's worked indications

This is the second half of the Werner and Modlin series. The first,
[`ratemaking-werner-modlin.md`](ratemaking-werner-modlin.md), takes the steps
of *Basic Ratemaking* (5th edition, CAS, 2016) one at a time: the
parallelogram method, premium and loss trend, loss development, expenses and
the permissible loss ratio, and credibility. This one runs the book's three
worked overall indications end to end in prospicio, Appendix A (auto property
damage, loss ratio method), Appendix B (homeowners, pure premium method) and
Appendix C (medical malpractice, loss ratio method with a net trend), then
covers chapter 11's increased limits and deductibles and lists what prospicio
does not do.

As before, the book's exhibits are the authors' and the text here is new; each
printed number is shown next to prospicio's, money is in dollars, and where
the book rounds a cell before the next step the code applies the same
rounding by hand (`half_up`) and says so. The rounding rules are collected in
`knowledge/references/werner-modlin-ratemaking.md`, and the library's parity
suite is `validation/tests/pricing.rs`.

## 1. The factors, recomputed

The indications use the factors the first guide derives. This block
recomputes them in one place, rounded to four places as the exhibits print
them: Appendix A's on-level factors (from the book's printed cumulative rate
level indices), its two-step premium trend, its development factors (the
straight average of link ratios excluding the highest and lowest, chained to
ultimate) and its two-step loss trend; and Appendix C's on-level factors and
net trend (8.9% over trend lengths the book rounds to two places).

```python
from decimal import Decimal, ROUND_HALF_UP

from prospicio.credibility import LimitedFluctuation
from prospicio.pricing import (
    LossRatioExperience,
    Period,
    Provisions,
    PurePremiumExperience,
    RateHistory,
    credibility_weighted,
    premium_based_expenses,
    trend_factor,
    trended_present_rates,
)
from prospicio.reserving import ChainLadder, Triangle


def half_up(x, dp):
    """Round half up to dp places, as the book's exhibits do (after clearing binary noise at 10 places)."""
    return float(Decimal(f"{x:.10f}").quantize(Decimal(1).scaleb(-dp), rounding=ROUND_HALF_UP))


def calendar(year):
    return Period.calendar(float(year), float(year + 1))


# Appendix A, Current Rate Level: six-month policies, the book's cumulative indices.
crl = RateHistory.from_levels([2011.25, 2012.5, 2013.75, 2014.5, 2015.75, 2016.0],
                              [0.9500, 1.0450, 1.0973, 1.0754, 1.1292, 1.1857])
a_on_level = [half_up(1.1857 / half_up(crl.rate_level(calendar(y), 0.5).average_level, 4), 4)
              for y in range(2011, 2016)]

# Premium Trend-3: to the latest average written premium of 115.35, then 2% for 2 years.
ep_crl = [1_364_916.59, 1_405_728.94, 1_448_424.45, 1_492_177.86, 1_536_267.03]
exposures = [12_900, 13_020, 13_130, 13_258, 13_380]
projected = half_up(trend_factor(0.02, 2.0), 4)
a_premium_trend = [half_up(half_up(115.35 / half_up(ep / x, 2), 4) * projected, 4)
                   for ep, x in zip(ep_crl, exposures)]

# Loss Development: reported losses and paid ALAE, accident years 2009 to 2015.
columns = [
    [705_088, 712_475, 714_196, 764_101, 774_384, 785_068, 797_866],
    [725_592, 753_295, 763_913, 861_114, 846_167, 821_509],
    [738_686, 782_248, 855_150, 884_498, 835_120],
    [753_027, 800_258, 874_106, 867_184],
    [732_239, 813_949, 856_495],
]
ages = [15, 27, 39, 51, 63]
tri = Triangle.from_long([2009 + i for col in columns for i in range(len(col))],
                         [a for a, col in zip(ages, columns) for _ in col],
                         {"reported": [float(x) for col in columns for x in col]})
links = [[half_up(c1[i] / c0[i], 4) for i in range(len(c1))] for c0, c1 in zip(columns, columns[1:])]
hi_lo = [half_up((sum(c) - max(c) - min(c)) / (len(c) - 2), 4) for c in links]
cdf = ChainLadder(ldf=dict(zip(ages, hi_lo)), ldf_kind="ldf").fit(tri, "reported").cdf
a_development = [half_up(c, 4) for c in cdf][::-1]

# Loss Trend-5: -0.5% to 6/30/2015, then +0.5% for 2.25 years to 9/30/2017.
a_loss_trend = [half_up(half_up(trend_factor(-0.005, 2015 - y), 4) * half_up(trend_factor(0.005, 2.25), 4), 4)
                for y in range(2011, 2016)]

# Appendix C: annual policies, and the net trend over lengths rounded to 2 places.
mm = RateHistory([2011.75, 2014 + 2 / 12, 2015.5], [-0.076, 0.146, 0.136])
c_on_level = []
for y in range(2010, 2015):
    lv = mm.rate_level(calendar(y), 1.0)
    c_on_level.append(half_up(half_up(lv.current_level, 4) / half_up(lv.average_level, 4), 4))
c_net_trend = [half_up(trend_factor(0.089, t), 4) for t in (6.83, 5.83, 4.83, 3.83, 2.83)]

book = {
    "A on-level": "1.2161 1.2176 1.1311 1.0892 1.0991",
    "A premium trend": "1.1342 1.1116 1.0879 1.0663 1.0452",
    "A development": "1.0000 0.9799 1.0003 1.0282 1.0966",
    "A loss trend": "0.9912 0.9962 1.0012 1.0062 1.0113",
    "C on-level": "1.2029 1.2058 1.2724 1.3018 1.2391",
    "C net trend": "1.7902 1.6439 1.5095 1.3862 1.2729",
}
ours = dict(zip(book, [a_on_level, a_premium_trend, a_development, a_loss_trend, c_on_level, c_net_trend]))
for name, values in ours.items():
    row = " ".join(f"{v:.4f}" for v in values)
    print(f"{name:16s} {row}  {'same as the book' if row == book[name] else 'book ' + book[name]}")
```

```text
A on-level       1.2161 1.2176 1.1311 1.0892 1.0991  same as the book
A premium trend  1.1342 1.1116 1.0879 1.0663 1.0452  same as the book
A development    1.0000 0.9799 1.0003 1.0282 1.0966  same as the book
A loss trend     0.9912 0.9962 1.0012 1.0062 1.0113  same as the book
C on-level       1.2029 1.2058 1.2724 1.3018 1.2391  same as the book
C net trend      1.7902 1.6439 1.5095 1.3862 1.2729  same as the book
```

Each row is the column of the indication exhibit it feeds, digit for digit.

## 2. Appendix A: auto property damage by the loss ratio method

Appendix A's indication (LR Indication exhibit) puts the pieces together for
five calendar-accident years of private passenger auto property damage:
earned premium brought to the current rate level and trended, reported
losses and paid ALAE developed, trended and loaded for ULAE, and the ratio of
the totals. `LossRatioExperience` takes each factor as a list with one value
per year (or one number for all years). The factors are section 1's; the
ULAE factor of 1.143 comes from the appendix's ULAE Ratio exhibit:

```python
a = LossRatioExperience(
    earned_premium=[1_122_372, 1_154_508, 1_280_545, 1_369_976, 1_397_750],
    losses=[856_495, 867_184, 835_120, 821_509, 797_866],
    on_level_factor=a_on_level,
    premium_trend=a_premium_trend,
    development=a_development,
    loss_trend=a_loss_trend,
    lae_factor=1.143,
)
book_premium = [1_548_088, 1_562_608, 1_575_741, 1_591_109, 1_605_706]
book_losses = [970_359, 967_578, 955_974, 971_450, 1_011_357]
book_ratio = [62.7, 61.9, 60.7, 61.1, 63.0]
for k, year in enumerate(range(2011, 2016)):
    print(f"{year}  premium {a.projected_premium_by_year[k]:12,.0f} (book {book_premium[k]:,})"
          f"  losses {a.projected_losses_by_year[k]:10,.0f} (book {book_losses[k]:,})"
          f"  ratio {100 * a.loss_ratio_by_year[k]:.1f}% (book {book_ratio[k]}%)")
rounded_rows = sum(half_up(x, 0) for x in a.projected_premium_by_year)
print(f"total premium {a.projected_premium:,.2f}, sum of rounded rows {rounded_rows:,.0f} (book 7,883,252)")
print(f"total losses {a.projected_losses:,.0f} (book 4,876,718)"
      f"  loss and LAE ratio {100 * a.loss_ratio:.2f}% (book 61.9%)")
```

```text
2011  premium    1,548,088 (book 1,548,088)  losses    970,359 (book 970,359)  ratio 62.7% (book 62.7%)
2012  premium    1,562,608 (book 1,562,608)  losses    967,578 (book 967,578)  ratio 61.9% (book 61.9%)
2013  premium    1,575,741 (book 1,575,741)  losses    955,974 (book 955,974)  ratio 60.7% (book 60.7%)
2014  premium    1,591,109 (book 1,591,109)  losses    971,450 (book 971,450)  ratio 61.1% (book 61.1%)
2015  premium    1,605,706 (book 1,605,706)  losses  1,011,357 (book 1,011,357)  ratio 63.0% (book 63.0%)
total premium 7,883,253.19, sum of rounded rows 7,883,252 (book 7,883,252)
total losses 4,876,718 (book 4,876,718)  loss and LAE ratio 61.86% (book 61.9%)
```

Every row agrees to the dollar. The book's total premium is the sum of its
rounded rows, a dollar below the exact total.

The Expense exhibit splits five selected expense ratios by their fixed shares
(general 6.3% and other acquisition 8.5%, each 75% fixed; licenses and fees
0.2%, all fixed; commission and brokerage 11.2% and taxes 2.1%, all
variable). The book rounds each part to a tenth before adding. The indicated
change follows from the selected 61.9% loss and LAE ratio:

```python
ratios = [0.063, 0.085, 0.002, 0.112, 0.021]
shares = [0.75, 0.75, 1.00, 0.00, 0.00]
parts = [premium_based_expenses([r], [s]) for r, s in zip(ratios, shares)]
fixed = sum(half_up(100 * f, 1) for f, _ in parts)
variable = sum(half_up(100 * v, 1) for _, v in parts)
exact = premium_based_expenses(ratios, shares)
print(f"fixed {fixed:.1f}% (book 11.3%)  variable {variable:.1f}% (book 17.0%)"
      f"  unrounded {100 * exact[0]:.2f}% and {100 * exact[1]:.2f}%")
pa = Provisions(variable / 100, 0.05)
change = pa.loss_ratio_change(0.619, fixed / 100)
print(f"variable permissible loss ratio {100 * pa.variable_permissible_loss_ratio:.1f}% (book 78.0%)")
print(f"indicated change {100 * change:.2f}% (book -6.2%);"
      f" from the unrounded loss ratio {100 * pa.loss_ratio_change(a.loss_ratio, fixed / 100):.2f}%")
```

```text
fixed 11.3% (book 11.3%)  variable 17.0% (book 17.0%)  unrounded 11.30% and 17.00%
variable permissible loss ratio 78.0% (book 78.0%)
indicated change -6.15% (book -6.2%); from the unrounded loss ratio -6.20%
```

The book computes the change from the selected, rounded 61.9%; from the
exact 61.86% it would be -6.20%, the same printed figure.

The Credibility exhibit counts 3,612 claims against the 1,082 standard, so the
indication is fully credible. It still computes the complement, trended
present rates: last time's 13.2% indication with 5.0% taken, and a net trend
of the 0.5% loss trend over the 2.0% premium trend for one year:

```python
za = LimitedFluctuation(0.90, 0.05).credibility(3_612, 1_082)
complement = trended_present_rates(0.132, 0.05, 1.005 / 1.02)
final = credibility_weighted(half_up(change, 3), za, complement)
print(f"Z {za:.3f} (book 100.0%)  complement {100 * complement:.2f}% (book 6.2%)"
      f"  credibility-weighted {100 * final:.1f}% (book -6.2%)")
```

```text
Z 1.000 (book 100.0%)  complement 6.22% (book 6.2%)  credibility-weighted -6.2% (book -6.2%)
```

## 3. Appendix B: homeowners by the pure premium method

Appendix B prices homeowners per exposure. Non-catastrophe losses are
developed, trended and loaded for ULAE over earned exposures
(`PurePremiumExperience`); catastrophes enter separately, as a non-modelled
load (from a long history of catastrophe losses to amount of insurance) and a
hurricane model's modelled load; the net cost of reinsurance and fixed
expenses are amounts per exposure; and the rate is the fundamental equation
with variable expense and profit. The development and trend factors are the
exhibit's (its development triangle and trend fits are not reproduced
here):

```python
b = PurePremiumExperience(
    exposure=[12_760, 12_766, 12_805, 12_834, 13_411],
    losses=[5_161_624, 4_820_968, 4_112_172, 5_052_052, 6_559_224],
    development=[1.0000, 1.0012, 1.0055, 1.0186, 1.0555],
    loss_trend=[1.1939, 1.1705, 1.1476, 1.1251, 1.1030],
    lae_factor=1.012,
)
book_losses = [6_236_412, 5_717_511, 4_802_025, 5_859_265, 7_727_993]
book_pp = [488.75, 447.87, 375.01, 456.54, 576.24]
for k, year in enumerate(range(2011, 2016)):
    print(f"{year}  losses {b.projected_losses_by_year[k]:11,.0f} (book {book_losses[k]:,})"
          f"  pure premium {b.pure_premium_by_year[k]:.2f} (book {book_pp[k]:.2f})")
print(f"total {b.projected_losses:,.0f} (book 30,343,206)  pure premium {b.pure_premium:.2f} (book 469.88)")
```

```text
2011  losses   6,236,412 (book 6,236,412)  pure premium 488.75 (book 488.75)
2012  losses   5,717,511 (book 5,717,511)  pure premium 447.87 (book 447.87)
2013  losses   4,802,025 (book 4,802,025)  pure premium 375.01 (book 375.01)
2014  losses   5,859,265 (book 5,859,265)  pure premium 456.54 (book 456.54)
2015  losses   7,727,993 (book 7,727,993)  pure premium 576.24 (book 576.24)
total 30,343,206 (book 30,343,206)  pure premium 469.88 (book 469.88)
```

The credibility is 683 claims against 1,082; the book rounds it to 79.5%
before weighting with the regional pure premium of $585.75. The reinsurance
cost is the projected net cost of $214,575 over the latest exposures grown 1%
a year for two years, rounded to whole exposures; the fixed expense of $72.79
per exposure is trended 3.4% a year for two years with the factor rounded to
four places:

```python
zb = LimitedFluctuation(0.90, 0.05).credibility(683, 1_082)
weighted = credibility_weighted(469.88, half_up(zb, 3), 585.75)
print(f"Z {zb:.4f} (book 79.5%)  weighted {weighted:.2f} (book 493.63);"
      f" with the unrounded Z {credibility_weighted(b.pure_premium, zb, 585.75):.2f}")
reinsurance = 214_575 / half_up(13_411 * trend_factor(0.01, 2.0), 0)
fixed_pp = 72.79 * half_up(trend_factor(0.034, 2.0), 4)
print(f"reinsurance per exposure {reinsurance:.2f} (book 15.68)  fixed expense {fixed_pp:.2f} (book 77.83)")
total_pp = weighted + 29.11 + 74.57  # non-modelled and modelled catastrophe loads, as printed
pb = Provisions(0.138, 0.05)
rate = pb.pure_premium_rate(half_up(total_pp, 2) + half_up(reinsurance, 2), half_up(fixed_pp, 2))
print(f"total pure premium {total_pp:.2f} (book 597.31)"
      f"  variable permissible {100 * pb.variable_permissible_loss_ratio:.1f}% (book 81.2%)")
print(f"indicated rate {rate:.2f} (book 850.76)")
```

```text
Z 0.7945 (book 79.5%)  weighted 493.63 (book 493.63); with the unrounded Z 493.69
reinsurance per exposure 15.68 (book 15.68)  fixed expense 77.83 (book 77.83)
total pure premium 597.31 (book 597.31)  variable permissible 81.2% (book 81.2%)
indicated rate 850.76 (book 850.76)
```

The $29.11 non-modelled catastrophe load is taken as printed: the
appendix's Non-Modeled Cat exhibit rounds its ratios to three places at each
step, and the unrounded chain gives $29.28. The $74.57 modelled load comes
from a hurricane model outside the book.

## 4. Appendix C: medical malpractice with a net trend

Appendix C works on a small, long-tailed book of occurrence policies:
ultimate losses and ALAE (developed in the appendix by several methods, taken
here as the selected ultimates), premium at current rate level and
one net trend of 8.9% applied to losses with no separate premium trend.
ULAE sits with the expenses, so the losses carry no LAE factor:

```python
c = LossRatioExperience(
    earned_premium=[14_904_664, 14_494_543, 14_442_449, 14_834_605, 18_265_093],
    losses=[11_673_500, 11_200_835, 6_290_368, 18_254_793, 23_371_444],
    on_level_factor=c_on_level,
    loss_trend=c_net_trend,
)
book_premium = [17_928_820, 17_477_520, 18_376_572, 19_311_689, 22_632_277]
book_losses = [20_897_900, 18_413_053, 9_495_310, 25_304_794, 29_749_511]
book_ratio = [116.6, 105.4, 51.7, 131.0, 131.4]
for k, year in enumerate(range(2010, 2015)):
    print(f"{year}  premium {c.projected_premium_by_year[k]:11,.0f} (book {book_premium[k]:,})"
          f"  losses {c.projected_losses_by_year[k]:11,.0f} (book {book_losses[k]:,})"
          f"  ratio {100 * c.loss_ratio_by_year[k]:.1f}% (book {book_ratio[k]}%)")
print(f"totals {c.projected_premium:,.0f} (book 95,726,878) and {c.projected_losses:,.0f}"
      f" (book 103,860,568): {100 * c.loss_ratio:.2f}% (book 108.5%)")
```

```text
2010  premium  17,928,820 (book 17,928,820)  losses  20,897,900 (book 20,897,900)  ratio 116.6% (book 116.6%)
2011  premium  17,477,520 (book 17,477,520)  losses  18,413,053 (book 18,413,053)  ratio 105.4% (book 105.4%)
2012  premium  18,376,572 (book 18,376,572)  losses   9,495,310 (book 9,495,310)  ratio 51.7% (book 51.7%)
2013  premium  19,311,689 (book 19,311,689)  losses  25,304,794 (book 25,304,794)  ratio 131.0% (book 131.0%)
2014  premium  22,632,277 (book 22,632,277)  losses  29,749,511 (book 29,749,511)  ratio 131.4% (book 131.4%)
totals 95,726,878 (book 95,726,878) and 103,860,568 (book 103,860,568): 108.50% (book 108.5%)
```

All expenses are variable here (general 6.4%, other acquisition 2.6%, taxes,
licenses and fees 2.1%, commission and brokerage 21.2%, and ULAE 2.4%), and
the underwriting profit provision is -5% because the investment income
expected in this long-tailed line covers the target return. The credibility standard is stricter than
Appendix A's (within 7.5% with 95% probability, 683 claims) and the
complement is the countrywide indication of +18.5%; the book rounds the
indication and the credibility to a tenth of a percent before weighting:

```python
variable = premium_based_expenses([0.064, 0.026, 0.021, 0.212, 0.024], [0.0] * 5)[1]
pc = Provisions(variable, -0.05)
indicated = pc.loss_ratio_change(1.085)
lfc = LimitedFluctuation(0.95, 0.075)
zc = lfc.credibility(283, round(lfc.frequency_standard()))
print(f"expenses {100 * variable:.1f}% (book 34.7%)  permissible {100 * pc.permissible_loss_ratio():.1f}% (book 70.3%)")
print(f"indicated {100 * indicated:.2f}% (book 54.3%)  standard {lfc.frequency_standard():.2f} (book 683)"
      f"  Z {100 * zc:.2f}% (book 64.4%)")
weighted = credibility_weighted(half_up(indicated, 3), half_up(zc, 3), 0.185)
print(f"credibility-weighted {100 * weighted:.2f}% (book 41.6%);"
      f" unrounded {100 * credibility_weighted(indicated, zc, 0.185):.2f}%")
```

```text
expenses 34.7% (book 34.7%)  permissible 70.3% (book 70.3%)
indicated 54.34% (book 54.3%)  standard 682.93 (book 683)  Z 64.37% (book 64.4%)
credibility-weighted 41.56% (book 41.6%); unrounded 41.57%
```

The selected loss and ALAE ratio of 108.5% is the rounded total above. The
book's exhibits treat 2010 to 2014 as the experience period; its earned
premium table also lists 2008 with the same figure as 2014 (18,265,093),
probably a copy, which does not enter the indication.

Appendix D (workers compensation) is a fourth worked indication; its
benefit-level and wage adjustments are chains of factors a caller multiplies
with `trend_factor`, and it is not reproduced here.

## 5. Increased limits and deductibles

Chapter 11 prices limits and deductibles from a size-of-loss table: the
increased limit factor is the limited average severity at the limit over that
at the basic limit, and a deductible's loss elimination ratio is the share of
losses below it. For a fitted or assumed severity, `prospicio.pricing.ilf` and
`loss_elimination_ratio` compute both exactly (see
[`distributions-for-actuaries-claim-size.md`](distributions-for-actuaries-claim-size.md)).
The book's tables, though, give only claim counts and loss totals per size
band, and prospicio has no severity built from such a table, so the book's
figures are a few lines of arithmetic. For the record, here they are, in plain
Python:

```python
# Table 11.3: 5,000 uncensored claims, (count, losses) by band up to 100K, 250K, 500K, 1M.
counts = [2_324, 1_923, 680, 73]
losses = [117_629_223, 307_599_929, 222_793_514, 43_047_470]
las_100 = (losses[0] + 100_000 * sum(counts[1:])) / sum(counts)
las_250 = (losses[0] + losses[1] + 250_000 * sum(counts[2:])) / sum(counts)
print(f"LAS(100K) {las_100:,.0f} (book 77,046)  LAS(250K) {las_250:,.0f} (book 122,696)"
      f"  ILF {las_250 / las_100:.2f} (book 1.59)")

# Table 11.6: the same claims censored at policy limits of 100K, 250K and 500K.
layer_1 = (156_657_898 + 34_903_214 + 35_768_111 + 100_000 * (773 + 574 + 232)) / 5_000
layer_2 = ((142_767_479 - 773 * 100_000) + (90_009_422 - 574 * 100_000) + 232 * 150_000) / (1_463 + 1_518)
layer_3 = (81_092_725 - 232 * 250_000) / 1_518
l1, l2, l3 = (half_up(x, 0) for x in (layer_1, layer_2, layer_3))  # the book adds rounded layers
print(f"censored: layers {l1:,.0f} + {l2:,.0f} + {l3:,.0f} (book 77,046 + 44,575 + 15,213)")
print(f"  LAS(250K) {l1 + l2:,.0f} (book 121,621)  LAS(500K) {l1 + l2 + l3:,.0f} (book 136,834)"
      f"  unrounded {layer_1 + layer_2:,.1f} and {layer_1 + layer_2 + layer_3:,.1f}")

# Table 11.8: loss elimination ratio of a 250 deductible, ground-up homeowners losses.
counts = [3_200, 1_225, 1_187, 1_845, 2_543]
losses = [240_365, 207_588, 463_954, 1_551_938, 11_140_545]
eliminated = losses[0] + losses[1] + 250 * sum(counts[2:])
print(f"LER(250) {eliminated / sum(losses):.3f} (book 0.135)")
```

```text
LAS(100K) 77,046 (book 77,046)  LAS(250K) 122,696 (book 122,696)  ILF 1.59 (book 1.59)
censored: layers 77,046 + 44,575 + 15,213 (book 77,046 + 44,575 + 15,213)
  LAS(250K) 121,621 (book 121,621)  LAS(500K) 136,834 (book 136,834)  unrounded 121,620.5 and 136,833.0
LER(250) 0.135 (book 0.135)
```

The censored method builds each layer only from policies whose limit is at
least the top of the layer, and scales it by the share of those policies'
claims that reach the layer. It gives a lower LAS(250K) than the uncensored
table, as the book's numbers show.

## 6. What prospicio does not cover

The overall indication is covered end to end. These parts of the book are
not, and some are filed as issues for the lane that owns the code:

- **Exposure and premium aggregation from policies** (chapters 4 and 5:
  written, earned, unearned and in-force amounts, the 24ths rule, the
  extension of exposures through a rating algorithm) and premium
  development. prospicio starts from aggregated figures.
- **Development averages other than volume weighted, simple or regression**,
  such as the straight average excluding the highest and lowest link ratio of
  Appendix A, and averages over the latest n years: compute them and pass
  them to `ChainLadder` as a selected pattern, as in section 1 here and
  section 4 of the first guide (issue #339).
- **Size-of-loss tables** (chapter 11): increased limit factors, the censored
  layer method and loss elimination ratios from grouped counts and losses.
  prospicio computes them from a severity distribution, not from a table
  (section 5; issue #338).
- **Traditional risk classification** (chapter 9: univariate pure premium
  and loss ratio relativities, adjusted for distributional bias; Appendix E),
  **multivariate classification and GLMs** (chapter 10 and Appendix F),
  **territorial ratemaking** and rate level indications by territory or
  class, and the other special classifications of chapter 11 (insurance to
  value and coinsurance, workers compensation size of risk and premium
  discounts). prospicio has a GLM library (`prospicio.models`), but the
  classification workflow of these chapters is the subject of Goldburd, Khare
  and Tevet's CAS Monograph 5, *Generalized Linear Models for Insurance
  Rating*, which a later guide in this series will follow.
- **Implementation** (chapter 14: rate capping, base rate offsetting,
  dislocation analysis), **commercial lines rating mechanisms** (chapter 15:
  experience and schedule rating, large deductibles, retrospective rating;
  reinsurance-style experience rating is in
  [`experience-and-exposure-rating.md`](experience-and-exposure-rating.md)),
  and **claims-made ratemaking** (chapter 16).
- **Catastrophe loads from amount of insurance years** and **workers
  compensation benefit chains** (Appendices B and D) are arithmetic with
  `trend_factor`; the hurricane model behind Appendix B's modelled load is
  outside the book.

## Where the book's printed numbers differ from exact arithmetic

Every difference between a printed figure and prospicio's exact value comes
from the book rounding a cell before the next step; once that rounding is
applied, as in the code above, each printed figure is reproduced:

| Where | Book | Exact | Cause |
|---|---|---|---|
| Table 5.19, cumulative level | 1.1435 | 1.14345 | index rounded to 4 places |
| Table 5.21, policy-year average | 1.1464 | 1.14634 | uses the rounded 1.1435 |
| Appendix A, 2012 on-level factor | 1.2176 | 1.2175 | indices rounded at each step |
| Table 6.12, 27-39 arithmetic average | 1.30 | 1.2932 | average of link ratios rounded to 2 places |
| Table 6.13, ultimate losses | 14,095 | 14,093.7 | factors to ultimate rounded to 2 places |
| Appendix A, 15-27 all-year average | 1.0704 | 1.0703 | link ratios rounded to 4 places |
| Appendix C, 2010 net trend factor | 1.7902 | 1.7907 | trend length rounded to 6.83 |
| Table 12.1, 5% and 99% standard | 2,652 | 2,654 | z rounded to 2.575 |
| Appendix A, total projected premium | 7,883,252 | 7,883,253.19 | sum of rounded rows |
| Appendix B, weighted pure premium | 493.63 | 493.69 | credibility rounded to 79.5% |
| Appendix B, non-modelled cat load | 29.11 | 29.28 | ratios rounded to 3 places |
| Appendix C, credibility-weighted change | 41.6% | 41.57% | indication and Z rounded first (same printed tenth) |
| Table 11.6, censored LAS(250K) | 121,621 | 121,620.4 | layers rounded to the dollar before adding |

Separately, three slips in the book itself, none of which changes an
indication: Appendix A's Premium Trend-3 footnote
gives the trend period as 6/30/2017 to 6/30/2017 (it is from 6/30/2015);
Table 7.5 uses 2015 earned premium of 545,250,000 where Table 7.4 has
530,000,000; and Appendix C lists the same earned premium for 2008 and 2014.
