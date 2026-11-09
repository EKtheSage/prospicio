# Basic ratemaking: Werner and Modlin's worked indications

<!-- site: execute -->

This guide is the second part of the Werner and Modlin series. The first part,
[`ratemaking-werner-modlin.md`](ratemaking-werner-modlin.md), shows the steps
of *Basic Ratemaking* (5th edition, CAS, 2016) one at a time. This part runs
the book's three overall indications end to end, then chapter 11's increased
limits and deductibles:

- Appendix A: auto property damage, by the loss ratio method.
- Appendix B: homeowners, by the pure premium method.
- Appendix C: medical malpractice, by the loss ratio method with a net trend.

The book's exhibits are the authors', and the text here is new. Each table
sets the book's numbers beside prospicio's, and money is in dollars. Where the
book rounds a cell before the next step, the code does the same by hand
(`half_up`); `knowledge/references/werner-modlin-ratemaking.md` lists the
rules and `validation/tests/pricing.rs` is the parity suite.

## 1. The factors, recomputed

This block recomputes the first guide's factors and rounds them to four
places, as the exhibits print them. Appendix A needs four sets:

- on-level factors, from the book's cumulative rate level indices
- the two-step premium trend
- development factors: the straight average of link ratios without the
  highest and lowest (`ChainLadder(average="simple", drop_high=1,
  drop_low=1)`), rounded to four places before it is chained to ultimate, as
  section 4 of the first guide explains
- the two-step loss trend

Appendix C needs the on-level factors and the net trend (8.9% over trend
lengths that the book rounds to two places). Each pair of rows is prospicio,
then the book. The columns run from the oldest year to the newest (A: 2011 to
2015, C: 2010 to 2014).

```python
from decimal import Decimal, ROUND_HALF_UP

import pandas as pd
from great_tables import GT

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


def row(item, ours, d, book, bd=None, alt=None, ad=None):
    """One line of a comparison: prospicio at d decimals, the book at bd, an optional unrounded value at ad."""
    return (item, ours, d, book, d if bd is None else bd, float("nan") if alt is None else alt, d if ad is None else ad)


def compare(rows, title, subtitle, alt=None):
    """A table of prospicio against the book; `alt` names an extra column for an unrounded value."""
    frame = pd.DataFrame({"item": [r[0] for r in rows], "prospicio": [r[1] for r in rows],
                          "book": [r[3] for r in rows], "alt": [r[5] for r in rows]})
    table = GT(frame if alt else frame.drop(columns="alt"), rowname_col="item").tab_header(title, subtitle)
    for k, (_, _, d, _, bd, _, ad) in enumerate(rows):
        table = table.fmt_number("prospicio", rows=[k], decimals=d).fmt_number("book", rows=[k], decimals=bd)
        if alt:
            table = table.fmt_number("alt", rows=[k], decimals=ad)
    return table.sub_missing(missing_text="").cols_label(alt=alt) if alt else table


def year_table(years, fit, book, total, title, subtitle, total_decimals):
    """Premium, losses and loss ratio by year with a total row, prospicio beside the book."""
    n = len(years)
    frame = pd.DataFrame({
        "year": [str(y) for y in years] + ["total"],
        "premium": [*fit.projected_premium_by_year, fit.projected_premium],
        "book_premium": [*book[0], total[0]],
        "losses": [*fit.projected_losses_by_year, fit.projected_losses],
        "book_losses": [*book[1], total[1]],
        "ratio": [*(100 * x for x in fit.loss_ratio_by_year), 100 * fit.loss_ratio],
        "book_ratio": [*book[2], total[2]],
    })
    by_year = list(range(n))
    return (GT(frame, rowname_col="year")
            .tab_header(title, subtitle)
            .fmt_number(["premium", "losses"], rows=by_year, decimals=0)
            .fmt_number("premium", rows=[n], decimals=total_decimals)
            .fmt_number("losses", rows=[n], decimals=0)
            .fmt_integer(["book_premium", "book_losses"])
            .fmt_number("ratio", rows=by_year, decimals=1)
            .fmt_number("ratio", rows=[n], decimals=2)
            .fmt_number("book_ratio", decimals=1)
            .tab_spanner("premium", ["premium", "book_premium"])
            .tab_spanner("losses", ["losses", "book_losses"])
            .tab_spanner("loss ratio (%)", ["ratio", "book_ratio"])
            .cols_label(premium="prospicio", book_premium="book", losses="prospicio", book_losses="book",
                        ratio="prospicio", book_ratio="book"))


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
# The straight average excluding the highest and lowest link ratio, selected to 4 places.
hi_lo = [half_up(f, 4) for f in ChainLadder(average="simple", drop_high=1, drop_low=1).fit(tri, "reported").ldf]
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
rows = []
for name, values in ours.items():
    rows += [(name, *values), (name + ", book", *map(float, book[name].split()))]
factors = pd.DataFrame(rows, columns=["factor", *[f"y{k}" for k in range(1, 6)]])
(GT(factors, rowname_col="factor")
 .tab_header("The indications' factors, recomputed", "Rounded to four places; each pair is prospicio, then the book")
 .fmt_number([f"y{k}" for k in range(1, 6)], decimals=4)
 .cols_label(**{f"y{k}": f"year {k}" for k in range(1, 6)}))
```

```text
                       year 1  year 2  year 3  year 4  year 5
A on-level             1.2161  1.2176  1.1311  1.0892  1.0991
A on-level, book       1.2161  1.2176  1.1311  1.0892  1.0991
A premium trend        1.1342  1.1116  1.0879  1.0663  1.0452
A premium trend, book  1.1342  1.1116  1.0879  1.0663  1.0452
A development          1.0000  0.9799  1.0003  1.0282  1.0966
A development, book    1.0000  0.9799  1.0003  1.0282  1.0966
A loss trend           0.9912  0.9962  1.0012  1.0062  1.0113
A loss trend, book     0.9912  0.9962  1.0012  1.0062  1.0113
C on-level             1.2029  1.2058  1.2724  1.3018  1.2391
C on-level, book       1.2029  1.2058  1.2724  1.3018  1.2391
C net trend            1.7902  1.6439  1.5095  1.3862  1.2729
C net trend, book      1.7902  1.6439  1.5095  1.3862  1.2729
```

## 2. Appendix A: auto property damage by the loss ratio method

Appendix A (LR Indication exhibit) covers five accident years of private
passenger auto property damage. Premium goes to the current rate level and is
trended. Reported losses and paid ALAE are developed, trended and loaded for
ULAE (factor 1.143, from the ULAE Ratio exhibit). The indicated loss ratio is
the ratio of the totals. `LossRatioExperience` takes each factor as a list
with one value per year, or one number for all years:

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
year_table(range(2011, 2016), a,
           ([1_548_088, 1_562_608, 1_575_741, 1_591_109, 1_605_706],
            [970_359, 967_578, 955_974, 971_450, 1_011_357], [62.7, 61.9, 60.7, 61.1, 63.0]),
           (7_883_252, 4_876_718, 61.9),
           "Appendix A: auto property damage, loss ratio method",
           "Projected premium and losses in dollars (losses include ALAE and ULAE)", 2)
```

```text
          prospicio       book  prospicio       book  prospicio  book
2011      1,548,088  1,548,088    970,359    970,359       62.7  62.7
2012      1,562,608  1,562,608    967,578    967,578       61.9  61.9
2013      1,575,741  1,575,741    955,974    955,974       60.7  60.7
2014      1,591,109  1,591,109    971,450    971,450       61.1  61.1
2015      1,605,706  1,605,706  1,011,357  1,011,357       63.0  63.0
total  7,883,253.19  7,883,252  4,876,718  4,876,718      61.86  61.9
```

The book's total premium is the sum of its rounded rows, a dollar below the
exact total.

The Expense exhibit splits five selected expense ratios into fixed and
variable parts: general 6.3% and other acquisition 8.5% (each 75% fixed),
licenses and fees 0.2% (all fixed), commission and brokerage 11.2% and taxes
2.1% (all variable). The book rounds each part to a tenth before it adds the
parts. The indicated change follows from the selected 61.9% loss and LAE
ratio:

```python
ratios = [0.063, 0.085, 0.002, 0.112, 0.021]
shares = [0.75, 0.75, 1.00, 0.00, 0.00]
parts = [premium_based_expenses([r], [s]) for r, s in zip(ratios, shares)]
fixed = sum(half_up(100 * f, 1) for f, _ in parts)
variable = sum(half_up(100 * v, 1) for _, v in parts)
exact = premium_based_expenses(ratios, shares)
pa = Provisions(variable / 100, 0.05)
change = pa.loss_ratio_change(0.619, fixed / 100)
compare([
    row("fixed expense (%)", fixed, 1, 11.3, alt=100 * exact[0], ad=2),
    row("variable expense (%)", variable, 1, 17.0, alt=100 * exact[1], ad=2),
    row("variable permissible loss ratio (%)", 100 * pa.variable_permissible_loss_ratio, 1, 78.0),
    row("indicated change (%)", 100 * change, 2, -6.2, bd=1,
        alt=100 * pa.loss_ratio_change(a.loss_ratio, fixed / 100), ad=2),
], "Appendix A: expenses and the indicated change",
   "Percent; unrounded: before the parts are rounded, and from the unrounded loss ratio", alt="unrounded")
```

```text
                                     prospicio  book  unrounded
fixed expense (%)                         11.3  11.3      11.30
variable expense (%)                      17.0  17.0      17.00
variable permissible loss ratio (%)       78.0  78.0           
indicated change (%)                     -6.15  -6.2      -6.20
```

The book starts from the rounded 61.9%. The exact 61.86% gives the same tenth.

With 3,612 claims against a standard of 1,082, the indication is fully
credible. The exhibit still computes the complement, trended present rates
(last indication 13.2%, rate taken 5.0%, net trend 0.5% loss over 2.0%
premium, one year):

```python
za = LimitedFluctuation(0.90, 0.05).credibility(3_612, 1_082)
complement = trended_present_rates(0.132, 0.05, 1.005 / 1.02)
final = credibility_weighted(half_up(change, 3), za, complement)
compare([
    row("credibility Z", za, 3, 1.0),
    row("complement (%)", 100 * complement, 2, 6.2, bd=1),
    row("credibility-weighted change (%)", 100 * final, 1, -6.2),
], "Appendix A: credibility", "Z is a fraction; the rest in percent")
```

```text
                                 prospicio   book
credibility Z                        1.000  1.000
complement (%)                        6.22    6.2
credibility-weighted change (%)       -6.2   -6.2
```

## 3. Appendix B: homeowners by the pure premium method

Appendix B prices homeowners per exposure. `PurePremiumExperience` develops and
trends the non-catastrophe losses, loads them for ULAE and divides them by
earned exposures. Catastrophes enter separately, as a non-modelled load (from
a long history of catastrophe losses to amount of insurance) and a modelled
load (from a hurricane model). Reinsurance and fixed expenses are amounts per
exposure, and the rate follows from the fundamental insurance equation. The
development and trend factors are the exhibit's printed values:

```python
b = PurePremiumExperience(
    exposure=[12_760, 12_766, 12_805, 12_834, 13_411],
    losses=[5_161_624, 4_820_968, 4_112_172, 5_052_052, 6_559_224],
    development=[1.0000, 1.0012, 1.0055, 1.0186, 1.0555],
    loss_trend=[1.1939, 1.1705, 1.1476, 1.1251, 1.1030],
    lae_factor=1.012,
)
exposures_b = [12_760, 12_766, 12_805, 12_834, 13_411]
frame_b = pd.DataFrame({
    "year": [str(y) for y in range(2011, 2016)] + ["total"],
    "exposures": [*exposures_b, sum(exposures_b)],
    "losses": [*b.projected_losses_by_year, b.projected_losses],
    "book_losses": [6_236_412, 5_717_511, 4_802_025, 5_859_265, 7_727_993, 30_343_206],
    "pure_premium": [*b.pure_premium_by_year, b.pure_premium],
    "book_pp": [488.75, 447.87, 375.01, 456.54, 576.24, 469.88],
})
(GT(frame_b, rowname_col="year")
 .tab_header("Appendix B: homeowners non-catastrophe pure premium",
             "Projected losses in dollars, pure premium in dollars per exposure")
 .fmt_integer(["exposures", "losses", "book_losses"])
 .fmt_number(["pure_premium", "book_pp"], decimals=2)
 .tab_spanner("losses", ["losses", "book_losses"])
 .tab_spanner("pure premium", ["pure_premium", "book_pp"])
 .cols_label(losses="prospicio", book_losses="book", pure_premium="prospicio", book_pp="book"))
```

```text
       exposures   prospicio        book  prospicio    book
2011      12,760   6,236,412   6,236,412     488.75  488.75
2012      12,766   5,717,511   5,717,511     447.87  447.87
2013      12,805   4,802,025   4,802,025     375.01  375.01
2014      12,834   5,859,265   5,859,265     456.54  456.54
2015      13,411   7,727,993   7,727,993     576.24  576.24
total     64,576  30,343,206  30,343,206     469.88  469.88
```

The credibility is 683 claims against 1,082. The book rounds it to 79.5%
before it weights with the regional pure premium of $585.75. Reinsurance is
the net cost of $214,575 over the exposures grown 1% a year for two years,
rounded to whole exposures. The fixed expense of $72.79 trends at 3.4% a year
for two years, with the factor rounded to four places:

```python
zb = LimitedFluctuation(0.90, 0.05).credibility(683, 1_082)
weighted = credibility_weighted(469.88, half_up(zb, 3), 585.75)
reinsurance = 214_575 / half_up(13_411 * trend_factor(0.01, 2.0), 0)
fixed_pp = 72.79 * half_up(trend_factor(0.034, 2.0), 4)
total_pp = weighted + 29.11 + 74.57  # non-modelled and modelled catastrophe loads, as printed
pb = Provisions(0.138, 0.05)
rate = pb.pure_premium_rate(half_up(total_pp, 2) + half_up(reinsurance, 2), half_up(fixed_pp, 2))
compare([
    row("credibility Z", zb, 4, 0.795, bd=3),
    row("weighted pure premium", weighted, 2, 493.63, alt=credibility_weighted(b.pure_premium, zb, 585.75)),
    row("reinsurance per exposure", reinsurance, 2, 15.68),
    row("fixed expense per exposure", fixed_pp, 2, 77.83),
    row("total pure premium", total_pp, 2, 597.31),
    row("variable permissible loss ratio (%)", 100 * pb.variable_permissible_loss_ratio, 1, 81.2),
    row("indicated rate", rate, 2, 850.76),
], "Appendix B: from the pure premium to the indicated rate",
   "Dollars per exposure unless marked; unrounded: the weighting with the unrounded Z", alt="unrounded")
```

```text
                                     prospicio    book  unrounded
credibility Z                           0.7945   0.795           
weighted pure premium                   493.63  493.63     493.69
reinsurance per exposure                 15.68   15.68           
fixed expense per exposure               77.83   77.83           
total pure premium                      597.31  597.31           
variable permissible loss ratio (%)       81.2    81.2           
indicated rate                          850.76  850.76
```

The code takes the $29.11 non-modelled catastrophe load as printed: the
Non-Modeled Cat exhibit rounds its ratios to three places at each step, and
the unrounded chain gives $29.28. The $74.57 modelled load comes from a
hurricane model, not from the book's data, so the code takes it as printed too.

## 4. Appendix C: medical malpractice with a net trend

Appendix C is a small, long-tailed book of occurrence policies. The inputs are
the selected ultimate losses and ALAE (the appendix develops them by several
methods), premium at current rate level and one net trend of 8.9%. The net
trend applies to losses, with no separate premium trend. ULAE is part of the
expenses, so the losses have no LAE factor:

```python
c = LossRatioExperience(
    earned_premium=[14_904_664, 14_494_543, 14_442_449, 14_834_605, 18_265_093],
    losses=[11_673_500, 11_200_835, 6_290_368, 18_254_793, 23_371_444],
    on_level_factor=c_on_level,
    loss_trend=c_net_trend,
)
year_table(range(2010, 2015), c,
           ([17_928_820, 17_477_520, 18_376_572, 19_311_689, 22_632_277],
            [20_897_900, 18_413_053, 9_495_310, 25_304_794, 29_749_511], [116.6, 105.4, 51.7, 131.0, 131.4]),
           (95_726_878, 103_860_568, 108.5),
           "Appendix C: medical malpractice, loss ratio method with a net trend",
           "Projected premium and losses in dollars (losses include ALAE)", 0)
```

```text
        prospicio        book    prospicio         book  prospicio   book
2010   17,928,820  17,928,820   20,897,900   20,897,900      116.6  116.6
2011   17,477,520  17,477,520   18,413,053   18,413,053      105.4  105.4
2012   18,376,572  18,376,572    9,495,310    9,495,310       51.7   51.7
2013   19,311,689  19,311,689   25,304,794   25,304,794      131.0  131.0
2014   22,632,277  22,632,277   29,749,511   29,749,511      131.4  131.4
total  95,726,878  95,726,878  103,860,568  103,860,568     108.50  108.5
```



```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots()
ax.bar([str(y) for y in range(2010, 2015)], [100 * r for r in c.loss_ratio_by_year],
       label="loss and ALAE ratio by year")
ax.axhline(100 * c.loss_ratio, color="0.5", linestyle="--", label="total, all five years")
ax.set_xlabel("accident year")
ax.set_ylabel("projected loss and ALAE ratio (%)")
ax.set_title("Appendix C: projected loss ratio by year")
ax.set_ylim(0, 160)
ax.legend()
```

All expenses are variable here: general 6.4%, other acquisition 2.6%, taxes,
licenses and fees 2.1%, commission and brokerage 21.2%, and ULAE 2.4%. The
underwriting profit provision is -5%, because the investment income of this
long-tailed line covers the target return. The credibility standard is
stricter than Appendix A's: within 7.5% with 95% probability, 683 claims. The
complement is the countrywide indication of +18.5%. The book rounds the
indication and the credibility to a tenth of a percent before it weights them:

```python
variable = premium_based_expenses([0.064, 0.026, 0.021, 0.212, 0.024], [0.0] * 5)[1]
pc = Provisions(variable, -0.05)
indicated = pc.loss_ratio_change(1.085)
lfc = LimitedFluctuation(0.95, 0.075)
zc = lfc.credibility(283, round(lfc.frequency_standard()))
weighted = credibility_weighted(half_up(indicated, 3), half_up(zc, 3), 0.185)
compare([
    row("expenses (%)", 100 * variable, 1, 34.7),
    row("permissible loss ratio (%)", 100 * pc.permissible_loss_ratio(), 1, 70.3),
    row("indicated change (%)", 100 * indicated, 2, 54.3, bd=1),
    row("credibility standard (claims)", lfc.frequency_standard(), 2, 683, bd=0),
    row("credibility Z (%)", 100 * zc, 2, 64.4, bd=1),
    row("credibility-weighted change (%)", 100 * weighted, 2, 41.6, bd=1,
        alt=100 * credibility_weighted(indicated, zc, 0.185)),
], "Appendix C: from the loss ratio to the credibility-weighted change",
   "Percent unless marked; unrounded: the weighting without the rounding", alt="unrounded")
```

```text
                                 prospicio  book  unrounded
expenses (%)                          34.7  34.7           
permissible loss ratio (%)            70.3  70.3           
indicated change (%)                 54.34  54.3           
credibility standard (claims)       682.93   683           
credibility Z (%)                    64.37  64.4           
credibility-weighted change (%)      41.56  41.6      41.57
```

The selected loss and ALAE ratio, 108.5%, is the rounded total above. The
book's earned premium table also lists 2008 with the same figure as 2014
(18,265,093), probably a copy error outside the indication.

## 5. Increased limits and deductibles

Chapter 11 prices limits and deductibles from a size-of-loss table. The
increased limit factor (ILF) is the limited average severity (LAS) at the
limit over the LAS at the basic limit. The loss elimination ratio (LER) of a
deductible is the share of losses below it. For a fitted or assumed severity,
`prospicio.pricing.ilf` and `loss_elimination_ratio` calculate both exactly
(see [`distributions-for-actuaries-claim-size.md`](distributions-for-actuaries-claim-size.md)).

The book's tables give claim counts and loss totals for each size band.
`SizeOfLoss` reads such a table as a severity. It puts each band's claims at
the band's average, so its limited expected value is exact at each band edge,
and `ilf`, `loss_elimination_ratio` and the layer functions take it as any
other severity. `censored_las` applies the censored method of Table 11.6, with
one table for each policy limit:

```python
from prospicio.distributions import SizeOfLoss
from prospicio.pricing import censored_las, ilf, loss_elimination_ratio

# Table 11.3: 5,000 uncensored claims, (count, losses) by band up to 100K, 250K, 500K, 1M.
table = SizeOfLoss([0, 100e3, 250e3, 500e3, 1e6], [2_324, 1_923, 680, 73],
                   [117_629_223, 307_599_929, 222_793_514, 43_047_470])

# Table 11.6: the same claims censored at policy limits of 100K, 250K and 500K.
by_limit = [
    SizeOfLoss([0, 100e3], [2_019], [156_657_898]),
    SizeOfLoss([0, 100e3, 250e3], [690, 773], [34_903_214, 142_767_479]),
    SizeOfLoss([0, 100e3, 250e3, 500e3], [712, 574, 232], [35_768_111, 90_009_422, 81_092_725]),
]
layers = censored_las(by_limit)
l1, l2, l3 = (half_up(layer["severity"], 0) for layer in layers)  # the book adds rounded layers

# Table 11.8: loss elimination ratio of a 250 deductible, ground-up homeowners losses.
# Only the edge at 250 matters; the other edges are chosen to hold each band's average.
homeowners = SizeOfLoss([0, 100, 250, 500, 1_000, float("inf")], [3_200, 1_225, 1_187, 1_845, 2_543],
                        [240_365, 207_588, 463_954, 1_551_938, 11_140_545])
compare([
    row("uncensored LAS(100K)", table.lev(100e3), 0, 77_046),
    row("uncensored LAS(250K)", table.lev(250e3), 0, 122_696),
    row("ILF at 250K", ilf(table, 250e3, 100e3), 2, 1.59),
    row("censored layer 1", l1, 0, 77_046),
    row("censored layer 2", l2, 0, 44_575),
    row("censored layer 3", l3, 0, 15_213),
    row("censored LAS(250K)", l1 + l2, 0, 121_621, alt=layers[1]["las"], ad=1),
    row("censored LAS(500K)", l1 + l2 + l3, 0, 136_834, alt=layers[2]["las"], ad=1),
    row("LER of a 250 deductible", loss_elimination_ratio(homeowners, 250), 3, 0.135),
], "Chapter 11: increased limits and a deductible",
   "Dollars, except the factor and the ratio; unrounded: the layers added without rounding", alt="unrounded")
```

```text
                         prospicio     book  unrounded
uncensored LAS(100K)        77,046   77,046           
uncensored LAS(250K)       122,696  122,696           
ILF at 250K                   1.59     1.59           
censored layer 1            77,046   77,046           
censored layer 2            44,575   44,575           
censored layer 3            15,213   15,213           
censored LAS(250K)         121,621  121,621  121,620.5
censored LAS(500K)         136,834  136,834  136,833.0
LER of a 250 deductible      0.135    0.135
```



```python
limits = [100, 250, 500, 1000]
fig, ax = plt.subplots()
ax.plot(limits, [ilf(table, 1e3 * m, 100e3) for m in limits], marker="o",
        label="uncensored table (Table 11.3)")
ax.plot(limits[:3], [layer["las"] / layers[0]["las"] for layer in layers], marker="o",
        label="censored method (Table 11.6)")
ax.set_xlabel("policy limit (thousands of dollars)")
ax.set_ylabel("increased limit factor")
ax.set_title("Increased limit factors, basic limit 100K")
ax.legend()
```

The censored method builds each layer only from policies whose limit is at
least the top of the layer, and scales the layer by the share of those
policies' claims that reach it. It gives a slightly lower factor at 250K and
500K than the uncensored table.

## Where the book's printed numbers differ from exact arithmetic

Each difference comes from the book rounding a cell before the next step. With
that rounding, as above, the code reproduces each printed figure:

| Where | Book | Exact | Cause |
|---|---|---|---|
| Table 5.19, cumulative level | 1.1435 | 1.14345 | index rounded to 4 places |
| Table 5.21, policy-year average | 1.1464 | 1.14634 | uses the rounded 1.1435 |
| Appendix A, 2012 on-level factor | 1.2176 | 1.2175 | indices rounded at each step |
| Table 6.12, 27-39 arithmetic average | 1.30 | 1.2932 | average of link ratios rounded to 2 places |
| Table 6.13, ultimate losses | 14,095 | 14,093.7 | factors to ultimate rounded to 2 places |
| Appendix A, 15-27 all-year average | 1.0704 | 1.0703 | link ratios rounded to 4 places |
| Appendix A, 15-ultimate development factor | 1.0966 | 1.0965 | selected factors rounded to 4 places before chaining |
| Appendix C, 2010 net trend factor | 1.7902 | 1.7907 | trend length rounded to 6.83 |
| Table 12.1, 5% and 99% standard | 2,652 | 2,654 | z rounded to 2.575 |
| Appendix A, total projected premium | 7,883,252 | 7,883,253.19 | sum of rounded rows |
| Appendix B, weighted pure premium | 493.63 | 493.69 | credibility rounded to 79.5% |
| Appendix B, non-modelled cat load | 29.11 | 29.28 | ratios rounded to 3 places |
| Appendix C, credibility-weighted change | 41.6% | 41.57% | indication and Z rounded first (same printed tenth) |
| Table 11.6, censored LAS(250K) | 121,621 | 121,620.5 | layers rounded to the dollar before adding |

The book also has three slips. None of them changes an indication:

- Appendix A's Premium Trend-3 footnote gives the trend period as 6/30/2017 to
  6/30/2017. The period starts at 6/30/2015.
- Table 7.5 uses 2015 earned premium of 545,250,000, where Table 7.4 has
  530,000,000.
- Appendix C lists the same earned premium for 2008 and 2014.
