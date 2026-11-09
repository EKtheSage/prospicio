# Applications and the choice of a distortion: CAS Monograph 15, chapters 5 and 6

<!-- site: execute -->

CAS Monograph 15 is *Introduction to Capital Modeling and Portfolio
Management* (John Major and Stephen Mildenhall, 2026). Chapters 5 and 6 apply
the natural allocation to the InsCo example. This guide recomputes their
tables with prospicio and shows the monograph's value beside prospicio's.
Table numbers are the monograph's.

[Capital adequacy and the natural
allocation](capital-modeling-monograph-15-theory.md) builds the method in
chapters 3 and 4. [Capital modeling and portfolio
pricing](capital-modeling-monograph-15.md) reproduces chapter 9.

## 1. The book and its target

InsCo has three units over ten equally likely years, plan premiums of 13.9,
18.7 and 19.6, and assets of 100. Investors want a return of 15% on capital.
That needs a premium of 53.565, against a plan of 52.2. Chapter 5 allocates
the required premium with a Wang distortion calibrated to it. Chapter 6 uses
all five representative families:

```python
from prospicio.distributions import Sampled
from prospicio.pricing import Pentagon, Portfolio
from prospicio.risk import Distortion

UNITS = ["A", "B", "C"]
ROWS = [
    [5, 20, 11], [7, 33, 0], [15, 13, 0], [15, 7, 0], [13, 20, 7],
    [5, 27, 8], [15, 16, 9], [26, 19, 10], [17, 8, 40], [16, 20, 64],
]
PLAN = [13.9, 18.7, 19.6]
insco = Portfolio(UNITS, ROWS)
totals = [float(sum(row)) for row in ROWS]

FAMILIES = ["ccoc", "ph", "wang", "dual", "tvar"]
distortions = {f: insco.calibrate(f, assets=100, return_on_capital=0.15) for f in FAMILIES}
wang = distortions["wang"]
required = insco.price(wang, assets=100)
print(f"required premium {required.total.premium:.3f}, plan {sum(PLAN):.1f}")
```

```text
required premium 53.565, plan 52.2
```

## 2. Performance assessment

Section 5.1 asks which units meet the target. The portfolio's shortfall of
1.365 is the same under every calibrated distortion. The split over the units
is not. With Wang (Tables 5.1 and 5.2), the excess loss ratio is the loss
ratio at the required premium less the plan's:

```python
import pandas as pd
from great_tables import GT

NAN = float("nan")


def pair_table(frame, places, percent=()):
    """Rows labelled ", book" are the monograph's, shown as printed; the others at fixed decimals."""
    book = [i for i, label in enumerate(frame.iloc[:, 0]) if label.endswith(", book")]
    mine = [i for i in range(len(frame)) if i not in book]
    table = GT(frame, rowname_col=frame.columns[0])
    for column, n in places.items():
        fmt = "fmt_percent" if column in percent else "fmt_number"
        table = getattr(table, fmt)(column, rows=mine, decimals=n, use_seps=False)
        table = getattr(table, fmt)(column, rows=book, decimals=n, drop_trailing_zeros=True, use_seps=False)
    return table.sub_missing(missing_text="")


BOOK_52 = {  # Tables 5.1 and 5.2: Wang premium, excess, excess as share of plan margin, excess loss ratio
    "A": (14.109, -0.209, -0.418, -0.0143), "B": (18.637, 0.063, 0.156, 0.0033),
    "C": (20.819, -1.219, -0.259, -0.0445),
}
rows = []
for u, plan, unit in zip(UNITS, PLAN, required.allocated):
    excess = plan - unit.premium
    rows.append((u, plan, unit.premium, excess, excess / (plan - unit.loss),
                 unit.loss / unit.premium - unit.loss / plan))
    rows.append((u + ", book", NAN, *BOOK_52[u]))
gross_excess = sum(PLAN) - required.total.premium
rows.append(("total", sum(PLAN), required.total.premium, gross_excess,
             gross_excess / (sum(PLAN) - 46.6), NAN))
rows.append(("total, book", NAN, NAN, -1.365, -0.244, NAN))
perf = pd.DataFrame(rows, columns=["unit", "plan", "wang", "excess", "of_plan", "lr"])
(pair_table(perf, {"plan": 1, "wang": 3, "excess": 3, "of_plan": 1, "lr": 2}, percent=("of_plan", "lr"))
 .tab_header("Table 5.2: plan premium against the required premium under Wang",
             "Premium in currency units; excess = plan less Wang")
 .cols_label(plan="plan", wang="Wang", excess="excess", of_plan="% of plan margin",
             lr="excess loss ratio"))
```

```text
             plan    Wang  excess  % of plan margin  excess loss ratio
A            13.9  14.109  -0.209            -41.8%             -1.43%
A, book            14.109  -0.209            -41.8%             -1.43%
B            18.7  18.637   0.063             15.6%              0.33%
B, book            18.637   0.063             15.6%              0.33%
C            19.6  20.819  -1.219            -25.9%             -4.45%
C, book            20.819  -1.219            -25.9%             -4.45%
total        52.2  53.565  -1.365            -24.4%                   
total, book                -1.365            -24.4%                   
```

Only unit B's plan meets its target. Unit C has most of the deficit in money.
Unit A is furthest off relative to its own margin: its plan margin of 0.5 must
rise by 42%. This holds for Wang only. Section 8 shows other distortions.

## 3. Reinsurance decision-making

Section 5.2 considers an aggregate stop loss of 35 excess of 65 at a premium
of 6.5. It pays 35 in the worst year only (expected loss 3.5). Two ways of
judging it give opposite answers.

The first prices the net portfolio alone with the pentagon. The cover cuts the
assets needed from 100 to 65, so the net premium at 15% is 7.609 less than
gross. That saves more than the cost of 6.5, so buy it:

```python
gross = Pentagon.solve(loss=46.6, assets=100.0, return_on_capital=0.15)
net = Pentagon.solve(loss=43.1, assets=65.0, return_on_capital=0.15)
saving = gross.premium - net.premium
print(f"net premium {net.premium:.4f} (book 45.956), saving {saving:.3f} (book 7.609), cover costs 6.5")
print(f"cover: loss on line {3.5 / 35:.0%}, rate on line {6.5 / 35:.1%} (book 10%, 18.6%)")
```

```text
net premium 45.9565 (book 45.956), saving 7.609 (book 7.609), cover costs 6.5
cover: loss on line 10%, rate on line 18.6% (book 10%, 18.6%)
```

The monograph rounds the discount factors to five decimals (0.86956 and
0.13043), which gives 45.956 where the unrounded premium is 45.9565.

That method assumes the net portfolio still costs 15% of its capital. But the
cover removes the top layer of capital, the cheapest under every distortion
except the CCoC. The second method prices the cover and the net portfolio as
two units of one portfolio under Wang:

```python
def stop_loss(t, attach=65.0, limit=35.0):
    return min(max(t - attach, 0.0), limit)


split = Portfolio(["net", "ceded"], [[t - stop_loss(t), stop_loss(t)] for t in totals])
net_unit, ceded_unit = split.price(wang, assets=100).allocated
print(f"ceded: loss {ceded_unit.loss:.3f}, Wang premium {ceded_unit.premium:.3f} (book 6.087), "
      f"quoted 6.5")
net_return = (net_unit.premium - net_unit.loss) / (65.0 - net_unit.premium)
print(f"net: premium {net_unit.premium:.3f} (book 47.478), return on 65 of assets "
      f"{net_return:.1%} (book 25%)")
```

```text
ceded: loss 3.500, Wang premium 6.087 (book 6.087), quoted 6.5
net: premium 47.478 (book 47.478), return on 65 of assets 25.0% (book 25%)
```

Under Wang the cover is worth 6.087 and costs 6.5, so do not buy it. The net
portfolio earns 25%, more than 15%, because its remaining capital is the most
likely to be used and so the most expensive. Section 9 gives the cover's value
under all five distortions.

## 4. New business pricing

Section 5.3 prices a small new account, too small to change the order of the
years. It is priced as a fourth unit with the current Wang weights `q`. The
monograph regresses the account's simulated losses on the portfolio total
(LOWESS). Those losses are not printed, so this guide starts from the
conditional means at the seven totals (Table 5.3):

```python
NEW = [0.02028, 0.01965, 0.01849, 0.01855, 0.02402, 0.02910, 0.06339]  # Table 5.3, at each total
Q_BOOK = [0.0522, 0.0660, 0.0748, 0.3791, 0.1190, 0.1350, 0.1739]      # the Wang q, Table 5.3


def wang_q(g, probs):
    s, gs = 1.0, [1.0]
    for p in probs:
        s -= p
        gs.append(g.g(s) if s > 1e-12 else 0.0)
    return [gs[j] - gs[j + 1] for j in range(len(probs))]


qw = wang_q(wang, insco.probs)
states = [str(int(t)) for t in sorted(set(totals))]
table_53 = pd.DataFrame({"total": states, "prob": list(insco.probs), "new": NEW, "q": qw, "q_book": Q_BOOK})
(GT(table_53, rowname_col="total")
 .tab_header("Table 5.3: the new account's conditional mean and the Wang weights",
             "Rows are the seven portfolio totals")
 .cols_label(prob="probability", new="account loss", q="q", q_book="q, book")
 .fmt_number(["prob"], decimals=1)
 .fmt_number(["new"], decimals=5)
 .fmt_number(["q", "q_book"], decimals=4))
```

```text
     probability  account loss       q  q, book
22           0.1       0.02028  0.0522   0.0522
28           0.1       0.01965  0.0660   0.0660
36           0.1       0.01849  0.0748   0.0748
40           0.4       0.01855  0.3791   0.3791
55           0.1       0.02402  0.1190   0.1190
65           0.1       0.02910  0.1350   0.1350
100          0.1       0.06339  0.1739   0.1739
```

A fourth unit in the portfolio gives the same premium as the weights:

```python
loss = sum(p * x for p, x in zip(insco.probs, NEW))
premium = sum(w * x for w, x in zip(qw, NEW))
print(f"new business: loss {loss:.5f} (book 0.02491), premium {premium:.5f} (book 0.02858), "
      f"margin {premium - loss:.5f} (book 0.00367)")

rows4 = [[*k, x] for k, x in zip(zip(*(insco.kappa(u) for u in UNITS)), NEW)]
with_new = Portfolio(UNITS + ["new"], rows4, insco.probs)
print(f"as a fourth unit: premium {with_new.price(wang).allocated[3].premium:.5f}")
```

```text
new business: loss 0.02491 (book 0.02491), premium 0.02858 (book 0.02858), margin 0.00367 (book 0.00367)
as a fourth unit: premium 0.02858
```

The two agree because the account does not reorder the totals. Its premium is
its reservation price: the lowest premium that keeps the portfolio on target.

## 5. Mergers and acquisitions

An acquisition is too large to price with the current weights. Section 5.4
scales the new account up about 2,000 times, to a mean of 48.35. At each
portfolio total, two points with half the probability each represent the
acquired loss: the conditional mean plus and minus one conditional standard
deviation. The merged portfolio has fourteen scenarios (Table 5.4). The Wang
distortion prices it, with assets equal to its worst total:

```python
MERGER = [  # (probability, InsCo total, acquired loss), Table 5.4
    (0.05, 22, 29.37), (0.05, 28, 25.47), (0.05, 36, 21.55), (0.2, 40, 22.74),
    (0.05, 22, 49.35), (0.05, 28, 50.79), (0.05, 55, 30.83), (0.05, 36, 50.22),
    (0.2, 40, 49.27), (0.05, 65, 29.89), (0.05, 55, 62.38), (0.05, 100, 47.73),
    (0.05, 65, 83.06), (0.05, 100, 198.30),
]
merged = Portfolio(["InsCo", "acquired"], [[x, y] for _, x, y in MERGER], [p for p, _, _ in MERGER])
m = merged.price(wang)
(old, new), tot = m.allocated, m.total
parts = pd.DataFrame([
    ("expected loss", old.loss, new.loss, tot.loss),
    ("expected loss, book", 46.60, 48.35, 94.95),
    ("Wang premium", old.premium, new.premium, tot.premium),
    ("Wang premium, book", 52.74, 59.09, 111.83),
], columns=["item", "InsCo", "acquired", "merged"])
(pair_table(parts, {"InsCo": 2, "acquired": 2, "merged": 2})
 .tab_header("Table 5.4: the merged portfolio under Wang", "Currency units"))
```

```text
                     InsCo  acquired  merged
expected loss        46.60     48.35   94.95
expected loss, book   46.6     48.35   94.95
Wang premium         52.74     59.09  111.83
Wang premium, book   52.74     59.09  111.83
```

```python
extra_premium = tot.premium - required.total.premium
extra_capital = tot.capital - required.total.capital
extra_margin = tot.margin - required.total.margin
effects = pd.DataFrame([
    ("acquisition", extra_premium, tot.assets, tot.capital, tot.return_on_capital,
     extra_capital, extra_margin, extra_margin / extra_capital),
    ("acquisition, book", 58.26, 298.30, 186.46, 0.0906, 140.03, 9.92, 0.0708),
], columns=["item", "premium", "assets", "capital", "ret", "add_cap", "add_margin", "add_ret"])
(pair_table(effects, {"premium": 3, "assets": 2, "capital": 3, "ret": 2, "add_cap": 2,
                      "add_margin": 2, "add_ret": 2}, percent=("ret", "add_ret"))
 .tab_header("The merged portfolio's premium, capital and return", "Currency units")
 .cols_label(premium="premium needed", ret="return", add_cap="added capital",
             add_margin="added margin", add_ret="return on added"))
```

```text
                   premium needed  assets  capital  return  added capital  added margin  return on added
acquisition                58.269  298.30  186.466   9.06%         140.03          9.92            7.08%
acquisition, book           58.26   298.3   186.46   9.06%         140.03          9.92            7.08%
```

All values agree with the printed solution to two decimals, with two
exceptions. The monograph's premium needed (58.26) subtracts the rounded 53.57
from 111.83, and its capital (186.46) truncates the unrounded value.

The split of the merged premium between the parts does not matter, because the
acquisition changes the weights for both. The acquired business must supply
the increase in portfolio premium. The return falls to about 9%, below the 15%
that set the Wang parameter, because the extra assets are far in the tail,
where capital is cheap under Wang.

## 6. Portfolio optimization

Section 5.5 asks the reverse question: at the given prices, which mix of
business is best? The objective is economic value added (EVA), plan premium
less required premium. The parameters are a scale for each unit between -50%
and +50%. Plan premium scales with volume, and capital (assets less plan
premium) must stay at 47.8. If everything scaled linearly, each unit's
allocated quantities would be its marginal ones (Table 5.5):

```python
eva = [plan - u.premium for plan, u in zip(PLAN, required.allocated)]
assets = [insco.kappa(u)[-1] for u in UNITS]           # coTVaR: the worst year, 16, 20, 64
capital = [a - plan for a, plan in zip(assets, PLAN)]
table_55 = pd.DataFrame({
    "unit": [*UNITS, "total"], "eva": [*eva, sum(eva)], "capital": [*capital, sum(capital)],
    "ratio": [*(e / c for e, c in zip(eva, capital)), NAN],
})
(GT(table_55, rowname_col="unit")
 .tab_header("Table 5.5: EVA and plan capital by unit", "Currency units")
 .cols_label(eva="EVA", capital="plan capital", ratio="EVA per capital")
 .fmt_number(["eva", "capital", "ratio"], decimals=4)
 .sub_missing(missing_text=""))
```

```text
           EVA  plan capital  EVA per capital
A      -0.2092        2.1000          -0.0996
B       0.0625        1.3000           0.0481
C      -1.2186       44.4000          -0.0274
total  -1.3652       47.8000                 
```

A linear step takes the best ratios to their limits: shrink A by 50%, grow B
by 50%, and grow C only enough to use the released capital (0.4 / 44.4 =
0.9%). The predicted EVA is -1.2403 (Table 5.6). The rerun at those scales
keeps the Wang distortion without a new calibration, because the distortion
describes investors' appetite and the target return is now an output:

```python
def rescale(scales):
    """InsCo with each unit's losses and plan premium scaled by 1 + s."""
    rows = [[x * (1 + s) for x, s in zip(row, scales)] for row in ROWS]
    book = Portfolio(UNITS, rows).price(wang)
    plan = [p * (1 + s) for p, s in zip(PLAN, scales)]
    return book, plan


def summary(scales):
    book, plan = rescale(scales)
    return (f"EVA {sum(plan) - book.total.premium:.4f}, assets {book.total.assets:.4f}, "
            f"capital {book.total.assets - sum(plan):.4f}")


def scale_table(cases, digits, title):
    """Required premium by unit at each case's scales, the book's row under it."""
    rows = []
    for label, scales, book_values in cases:
        book, _ = rescale(scales)
        rows.append((label, *(u.premium for u in book.allocated), book.total.premium))
        rows.append((label + ", book", *book_values))
    frame = pd.DataFrame(rows, columns=["case", *UNITS, "total"])
    return (pair_table(frame, {c: digits for c in [*UNITS, "total"]})
            .tab_header(title, "Required premium in currency units"))


linear_step = [-0.5, 0.5, 0.4 / 44.4]
print("linear prediction of EVA:", round(sum(eva) + sum(e * s for e, s in zip(eva, linear_step)), 4),
      "(book -1.2403)")
print(summary(linear_step))
```

```text
linear prediction of EVA: -1.2404 (book -1.2403)
EVA -1.4660, assets 102.5766, capital 47.8000
```

The linear prediction is -1.2404 here and -1.2403 in Table 5.6, which adds
steps rounded to four decimals (0.1046, 0.0313, -0.0110). The rerun agrees
with Table 5.8:

```python
scale_table([("linear step", linear_step, [6.8115, 28.6040, 20.8271, 56.2426])], 4,
            "Table 5.8: the premium after the linear step")
```

```text
                        A        B        C    total
linear step        6.8115  28.6040  20.8271  56.2426
linear step, book  6.8115   28.604  20.8271  56.2426
```

The EVA of -1.466 is worse than the start: scaling the units changes the order
of the years (Table 5.7), so Wang weights different years than the linear
prediction assumed. With A held at -50%, B shows how uneven the surface is
(Figure 5.4):

```python
def eva_at(sa, sb):
    """EVA with unit C scaled to hold capital at 47.8 (the worst year stays the worst)."""
    sc = -(2.1 * sa + 1.3 * sb) / 44.4
    book, plan = rescale([sa, sb, sc])
    return sum(plan) - book.total.premium


SWEEP = [-0.5, -0.3, -0.21, -0.2, -0.1, 0.0, 0.25, 0.5]
sweep = pd.DataFrame({"b": [f"{sb:+.2f}" for sb in SWEEP], "eva": [eva_at(-0.5, sb) for sb in SWEEP]})
(GT(sweep, rowname_col="b")
 .tab_header("Figure 5.4: EVA as the scale of B changes", "A at -50%, C holds capital at 47.8")
 .cols_label(eva="EVA")
 .fmt_number("eva", decimals=4))
```

```text
           EVA
-0.50  -1.4321
-0.30  -1.3694
-0.21  -1.3477
-0.20  -1.3484
-0.10  -1.3604
+0.00  -1.3724
+0.25  -1.4058
+0.50  -1.4660
```

The monograph's Nelder-Mead search ends at A -0.478165 and B -0.209790. On
this kinked surface its end point depends on the start and the implementation,
so this guide searches a grid instead, in steps of 0.01 and then 0.0005 around
the best point, with `price` in a loop:

```python
coarse = max((eva_at(a / 100, b / 100), a / 100, b / 100)
             for a in range(-50, 51) for b in range(-50, 51))
_, a0, b0 = coarse
fine = max((eva_at(a0 + i / 2000, b0 + j / 2000), a0 + i / 2000, b0 + j / 2000)
           for i in range(-20, 21) for j in range(-20, 21))
print(f"grid: A {fine[1]:+.4f}, B {fine[2]:+.4f}, EVA {fine[0]:.6f}")

POINTS = [(-0.209790, [7.269164, 14.816395, 21.453658, 43.539217]),   # Table 5.9
          (-0.209795, [7.334042, 14.607535, 21.597550, 43.539127])]   # Table 5.10


def at_point(sb):
    return [-0.478165, sb, -(2.1 * -0.478165 + 1.3 * sb) / 44.4]


for sb, _ in POINTS:
    print(f"at A -0.478165, B {sb}: EVA {eva_at(-0.478165, sb):.6f} (book -1.345119)")
    print(summary(at_point(sb)))
```

```text
grid: A -0.4780, B -0.2095, EVA -1.345142
at A -0.478165, B -0.20979: EVA -1.345139 (book -1.345119)
EVA -1.3451, assets 89.9941, capital 47.8000
at A -0.478165, B -0.209795: EVA -1.345139 (book -1.345119)
EVA -1.3451, assets 89.9940, capital 47.8000
```

The grid ends next to the monograph's optimum, a small gain on the start of
-1.3652. At the monograph's own point the EVA agrees to about 2e-5. Two scales
for B only 0.000005 apart give the same portfolio EVA and very different
allocations (Tables 5.9 and 5.10):

```python
scale_table([(f"B {sb}", at_point(sb), book_values) for sb, book_values in POINTS], 6,
            "Tables 5.9 and 5.10: the premium at the monograph's two optima")
```

```text
                          A          B          C      total
B -0.20979         7.269165  14.816395  21.453677  43.539237
B -0.20979, book   7.269164  14.816395  21.453658  43.539217
B -0.209795        7.334043  14.607535  21.597569  43.539147
B -0.209795, book  7.334042  14.607535   21.59755  43.539127
```

Two years almost tie for their place in the order, and the year that comes
first decides which unit gets the Wang weight. Units A and B agree to the six
printed decimals. Unit C is 2e-5 higher here, within the unprinted digits of
the scales and of the Wang parameter. The monograph's conclusion holds: the
optimum is flat, kinked and sensitive, so read it as a direction, not a
target.

## 7. The range of possible prices

Chapter 6 asks which distortion to use. Section 6.1 starts from the extremes,
the mean (TVaR at 0) and the maximum (TVaR at 1). The CCoC is the blend that
puts weight `d = 0.15 / 1.15` on the maximum and the rest on the mean
(Table 6.1):

```python
ccoc = distortions["ccoc"]
blend = Distortion.bitvar(0.0, 1.0, 0.15 / 1.15)
table_61 = pd.DataFrame({"total": states, "ccoc": wang_q(ccoc, insco.probs),
                         "blend": wang_q(blend, insco.probs)})
(GT(table_61, rowname_col="total")
 .tab_header("Table 6.1: the CCoC weights and the blend of mean and maximum",
             "Rows are the seven portfolio totals")
 .cols_label(ccoc="q, CCoC", blend="q, blend")
 .fmt_number(["ccoc", "blend"], decimals=3))
```

```text
     q, CCoC  q, blend
22     0.087     0.087
28     0.087     0.087
36     0.087     0.087
40     0.348     0.348
55     0.087     0.087
65     0.087     0.087
100    0.217     0.217
```

Infinitely many distortions price the portfolio at 53.565, and they disagree
about the units. The monograph spans them with five representatives.
`Portfolio.premium_bounds` finds the true extremes for each unit, always
bi-TVaRs (Section 11.2 of *Pricing Insurance Risk*):

```python
target = required.total.premium
by_family = {f: [u.premium for u in insco.price(g, assets=100).allocated]
             for f, g in distortions.items()}
ranges = pd.DataFrame([
    (b["unit"], b["lower"], min(by_family[f][i] for f in FAMILIES),
     max(by_family[f][i] for f in FAMILIES), b["upper"])
    for i, b in enumerate(insco.premium_bounds(target, assets=100))
], columns=["unit", "lowest", "five_low", "five_high", "highest"])
(GT(ranges, rowname_col="unit")
 .tab_header("Premium of each unit: all distortions against the five", "Currency units")
 .cols_label(lowest="lowest", five_low="five, low", five_high="five, high", highest="highest")
 .fmt_number(["lowest", "five_low", "five_high", "highest"], decimals=3))
```

```text
   lowest  five, low  five, high  highest
A  13.098     13.739      14.127   15.033
B  17.466     18.349      20.412   20.412
C  19.255     19.371      21.304   22.098
```

Grey: all distortions. Coloured: the five. Dot: the plan.

```python
import matplotlib.pyplot as plt

bounds = insco.premium_bounds(target, assets=100)
fig, ax = plt.subplots()
for y, b in enumerate(bounds):
    five = [by_family[f][y] for f in FAMILIES]
    ax.plot([b["lower"], b["upper"]], [y, y], color="0.75", linewidth=10, solid_capstyle="butt",
            label="all distortions" if y == 0 else None)
    ax.plot([min(five), max(five)], [y, y], color="C0", linewidth=4, solid_capstyle="butt",
            label="the five" if y == 0 else None)
    ax.plot(PLAN[y], y, "o", color="C1", label="plan" if y == 0 else None)
ax.set_yticks(range(len(UNITS)), [f"unit {u}" for u in UNITS])
ax.set_xlabel("premium (currency units)")
ax.set_title(f"Range of the premium of each unit at a total premium of {target:.3f}")
ax.legend()
```

The five cover most of each unit's range, not all of it: a distortion outside
the five can price unit A as low as 13.098. The monograph warns that unless
one distortion lies above another everywhere, the two rank some pair of risks
in opposite orders. So distortions have no single order by price.

## 8. Required premium by distortion

Section 6.2 gives the required premium and loss ratio of each unit under each
distortion (Table 6.2):

```python
BOOK_62 = {  # Table 6.2, premium by unit
    "ccoc": (13.739, 18.522, 21.304), "ph": (14.060, 18.349, 21.156),
    "wang": (14.109, 18.637, 20.818), "dual": (14.127, 19.117, 20.321),
    "tvar": (13.783, 20.412, 19.371),
}
loss = insco.expected()
rows = []
for f in FAMILIES:
    rows.append((f, *by_family[f], *(l / p for l, p in zip(loss, by_family[f]))))
    rows.append((f + ", book", *BOOK_62[f], NAN, NAN, NAN))
rows.append(("plan", *PLAN, *(l / p for l, p in zip(loss, PLAN))))
table_62 = pd.DataFrame(rows, columns=["distortion", "a", "b", "c", "a_lr", "b_lr", "c_lr"])
(pair_table(table_62, {"a": 3, "b": 3, "c": 3, "a_lr": 1, "b_lr": 1, "c_lr": 1},
            percent=("a_lr", "b_lr", "c_lr"))
 .tab_header("Table 6.2: required premium and loss ratio by distortion", "Premium in currency units")
 .cols_label(a="A premium", b="B premium", c="C premium", a_lr="A loss ratio",
             b_lr="B loss ratio", c_lr="C loss ratio"))
```

```text
            A premium  B premium  C premium  A loss ratio  B loss ratio  C loss ratio
ccoc           13.739     18.522     21.304         97.5%         98.8%         69.9%
ccoc, book     13.739     18.522     21.304                                          
ph             14.060     18.349     21.156         95.3%         99.7%         70.4%
ph, book        14.06     18.349     21.156                                          
wang           14.109     18.637     20.819         95.0%         98.2%         71.6%
wang, book     14.109     18.637     20.818                                          
dual           14.127     19.117     20.322         94.9%         95.7%         73.3%
dual, book     14.127     19.117     20.321                                          
tvar           13.783     20.412     19.371         97.2%         89.7%         76.9%
tvar, book     13.783     20.412     19.371                                          
plan           13.900     18.700     19.600         96.4%         97.9%         76.0%
```

Section 6.3 shows which distortion is cheapest for each unit (Table 6.3):

```python
for i, u in enumerate(UNITS):
    cheapest = min(FAMILIES, key=lambda f: by_family[f][i])
    within = min(by_family[f][i] for f in FAMILIES) <= PLAN[i]
    print(f"{u}: cheapest {cheapest}, plan meets at least one distortion: {within}")
```

```text
A: cheapest ccoc, plan meets at least one distortion: True
B: cheapest ph, plan meets at least one distortion: True
C: cheapest tvar, plan meets at least one distortion: True
```

```python
fig, axes = plt.subplots(1, 3, figsize=(7.0, 3.4))
for i, ax in enumerate(axes):
    ax.plot(FAMILIES, [by_family[f][i] for f in FAMILIES], "o", color="C0", label="required")
    ax.axhline(PLAN[i], color="C1", label="plan")
    ax.set_title(f"Unit {UNITS[i]}")
    ax.tick_params(axis="x", labelrotation=45)
axes[0].set_ylabel("premium (currency units)")
axes[1].legend(loc="upper left")
fig.suptitle("Required premium by distortion, against the plan")
fig.tight_layout()
```

Unit C's two small differences are rounding. The dual power premium is
20.32158, printed as 20.321 in Table 6.2 and 20.322 in Table 9.8 (the
monograph's dual parameter is a few millionths lower, on the rounding point).
The Wang premium is 20.81858, printed as 20.818 in Table 6.2 and 20.819 in
Table 5.1. Each unit's plan meets at least one distortion, but no single
distortion is met by all three plans.

## 9. Zeroing in, or embracing the range

Section 6.3 suggests calibration to the market where possible. Suppose InsCo's
Baa bonds have a real default probability of 1.5% to 2%, and the market prices
them as if it were 2.5% to 4%. A candidate distortion should map the first
range into the second:

```python
rows = []
for f, g in distortions.items():
    lo, hi = g.g(0.015), g.g(0.02)
    rows.append((f, lo, hi, "yes" if hi >= 0.025 and lo <= 0.04 else "no"))
bonds = pd.DataFrame(rows, columns=["distortion", "lo", "hi", "fits"])
(GT(bonds, rowname_col="distortion")
 .tab_header("Each distortion on the Baa bond's default probability", "g(s) at 1.5% and 2%")
 .cols_label(lo="g(1.5%)", hi="g(2%)", fits="overlaps 2.5% to 4%")
 .fmt_percent(["lo", "hi"], decimals=2))
```

```text
      g(1.5%)   g(2%)  overlaps 2.5% to 4%
ccoc   14.35%  14.78%                   no
ph      4.85%   5.97%                   no
wang    3.38%   4.35%                  yes
dual    2.38%   3.17%                  yes
tvar    2.06%   2.74%                  yes
```

Wang and the dual power fit the bond, as the monograph's "Dual and Wang, with
TVaR a distant third" says. TVaR touches the window only at its top. PH and the
CCoC overprice the bond's risk.

Section 6.4 takes the other path: no distortion is right, but the range gives
information. The stop loss of section 3 is worth 7.6 under the CCoC and 4.8
under TVaR, as wide as the reinsurance cycle:

```python
rows = []
for f, g in distortions.items():
    _, ceded = split.price(g, assets=100).allocated
    rows.append((f, ceded.premium, ceded.loss / ceded.premium, "yes" if ceded.premium > 6.5 else "no"))
cover = pd.DataFrame(rows, columns=["distortion", "worth", "lr", "buy"])
(GT(cover, rowname_col="distortion")
 .tab_header("The stop loss to InsCo by distortion", "Value in currency units; quoted premium 6.5")
 .cols_label(worth="cover worth", lr="loss ratio", buy="buy at 6.5")
 .fmt_number("worth", decimals=3)
 .fmt_percent("lr", decimals=0))
```

```text
      cover worth  loss ratio  buy at 6.5
ccoc        7.609         46%         yes
ph          6.662         53%         yes
wang        6.087         58%          no
dual        5.415         65%          no
tvar        4.803         73%          no
```

```python
worth = [split.price(distortions[f], assets=100).allocated[1].premium for f in FAMILIES]
fig, ax = plt.subplots()
ax.bar(FAMILIES, worth, color="C0", label="value of the cover")
ax.axhline(6.5, color="C1", label="quoted premium")
ax.set_xlabel("distortion")
ax.set_ylabel("premium (currency units)")
ax.set_title("Value of the stop loss to InsCo by distortion")
ax.legend()
```

A plan inside a unit's range, as all three are here, is consistent with some
risk appetite. A plan outside the range of every distortion is consistent with
none and needs scrutiny. `Portfolio.premium_bounds` shows such a plan.

For which distortion to use and how to calibrate it to quotes, see [Risk loads
and distortions](risk-loads-and-distortions.md). For capital by unit (section
8.3 of the monograph), see section 5 of [the chapter 9
guide](capital-modeling-monograph-15.md).
