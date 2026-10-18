# Applications and the choice of a distortion: CAS Monograph 15, chapters 5 and 6

Chapters 5 and 6 of CAS Monograph 15, *Introduction to Capital Modeling and
Portfolio Management* (John Major and Stephen Mildenhall, 2026), put the
natural allocation to work on the InsCo example: judging units against a
profit target, deciding on a reinsurance cover, pricing new business and an
acquisition, rebalancing the portfolio, and choosing the distortion that does
the allocating. This guide recomputes each of their tables with prospicio and
prints the monograph's value beside prospicio's. It continues
[Capital adequacy and the natural allocation](capital-modeling-monograph-15-theory.md),
which builds the method in chapters 3 and 4; the monograph's code, chapter 9,
is reproduced in
[Capital modeling and portfolio pricing](capital-modeling-monograph-15.md).
The reasoning is the authors'; what follows explains it in new words and
recomputes the numbers. Table numbers are the monograph's.

## 1. The book and its target

InsCo has three units over ten equally likely years, plan premiums of 13.9,
18.7 and 19.6, and assets of 100. Investors want 15% on capital, which needs
a premium of 53.565 against a plan of 52.2. Chapter 5 allocates that
requirement with a Wang distortion calibrated to it, and chapter 6 with all
five representative families:

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
print("Wang by unit", [round(u.premium, 3) for u in required.allocated],
      " (book 14.109, 18.637, 20.819)")
```

```text
required premium 53.565, plan 52.2
Wang by unit [14.109, 18.637, 20.819]  (book 14.109, 18.637, 20.819)
```

## 2. Performance assessment

Section 5.1 asks which units meet the target. The shortfall of 1.365 for the
book does not depend on the distortion, since every calibrated distortion
charges 53.565 in total; how it splits over the units does. With Wang
(Table 5.2):

```python
BOOK_52 = {  # Table 5.2: excess margin, as % of plan margin, excess loss ratio
    "A": (-0.209, -0.418, -0.0143), "B": (0.063, 0.156, 0.0033), "C": (-1.219, -0.259, -0.0445),
}
print(f"{'':2s} {'plan':>6s} {'Wang':>7s} {'excess':>7s} {'of plan':>8s} {'excess LR':>10s}")
for u, plan, unit in zip(UNITS, PLAN, required.allocated):
    excess = plan - unit.premium
    plan_margin = plan - unit.loss
    excess_lr = unit.loss / unit.premium - unit.loss / plan
    b = BOOK_52[u]
    print(f"{u:2s} {plan:6.1f} {unit.premium:7.3f} {excess:7.3f} {excess / plan_margin:8.1%} "
          f"{excess_lr:10.2%}   book {b[0]:.3f} {b[1]:.1%} {b[2]:.2%}")
gross_excess = sum(PLAN) - required.total.premium
print(f"book total: excess {gross_excess:.3f} (book -1.365), "
      f"{gross_excess / (sum(PLAN) - 46.6):.1%} of plan margin (book -24.4%)")
```

```text
     plan    Wang  excess  of plan  excess LR
A    13.9  14.109  -0.209   -41.8%     -1.43%   book -0.209 -41.8% -1.43%
B    18.7  18.637   0.063    15.6%      0.33%   book 0.063 15.6% 0.33%
C    19.6  20.819  -1.219   -25.9%     -4.45%   book -1.219 -25.9% -4.45%
book total: excess -1.365 (book -1.365), -24.4% of plan margin (book -24.4%)
```

Only unit B's plan covers its target, and by 0.063. Unit C carries most of
the deficit in money (1.219), while unit A is furthest off relative to its
own margin: its plan margin of 0.5 would have to rise by 42%. The loss ratio
gap is the target loss ratio less the plan's, so a negative value means the
plan runs too high. All of this is conditional on Wang; section 7 shows how
much the verdict moves with the distortion.

## 3. Reinsurance decision-making

Section 5.2 considers an aggregate stop loss of 35 excess of 65 offered for a
premium of 6.5. It pays 35 in the worst year only, an expected 3.5. Two ways
to judge it give opposite answers. The first prices the net book alone with
the pentagon: with the cover, the assets needed fall from 100 to 65, and at
15% the net book needs a premium of 45.956, 7.609 less than gross. The cover
saves more than its 6.5, so buy it:

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

The book's 45.956 comes from discount factors rounded to five decimals
(0.86956 and 0.13043); unrounded, the premium is 45.9565, a hair above the
rounding point.

That argument assumes the net book still costs 15% of its capital, but the
cover removes the top layer of capital, the cheapest under any distortion but
the CCoC, so the remaining capital is dearer. The second way prices the
cover and the net book as two units of the same book, with the same Wang
distortion:

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

The cover is worth 6.087 to InsCo under Wang and costs 6.5, so do not buy.
The net book, keeping 47.478 of the premium on 65 of assets, earns 25% on its
remaining capital, more than 15%, because what is left is the capital most
likely to be used, and that is the most expensive kind. Section 7 gives the
range of the cover's value over all five distortions.

## 4. New business pricing

Section 5.3 prices a small new account from a surrogate model. The
monograph regresses the account's simulated losses on the portfolio total
(LOWESS) and reads off its expected loss at each of the seven totals. The
account is too small to reorder the years, so it is priced as a fourth unit
with the existing Wang weights. Starting from those conditional means (Table
5.3):

```python
NEW = [0.02028, 0.01965, 0.01849, 0.01855, 0.02402, 0.02910, 0.06339]  # Table 5.3, at each total
q = [0.0522, 0.0660, 0.0748, 0.3791, 0.1190, 0.1350, 0.1739]           # the Wang q, Table 5.3


def wang_q(g, probs):
    s, gs = 1.0, [1.0]
    for p in probs:
        s -= p
        gs.append(g.g(s) if s > 1e-12 else 0.0)
    return [gs[j] - gs[j + 1] for j in range(len(probs))]


qw = wang_q(wang, insco.probs)
print("q here", [round(x, 4) for x in qw])
loss = sum(p * x for p, x in zip(insco.probs, NEW))
premium = sum(w * x for w, x in zip(qw, NEW))
print(f"new business: loss {loss:.5f} (book 0.02491), premium {premium:.5f} (book 0.02858), "
      f"margin {premium - loss:.5f} (book 0.00367)")

rows4 = [[*k, x] for k, x in zip(zip(*(insco.kappa(u) for u in UNITS)), NEW)]
with_new = Portfolio(UNITS + ["new"], rows4, insco.probs)
print(f"as a fourth unit: premium {with_new.price(wang).allocated[3].premium:.5f}")
```

```text
q here [0.0522, 0.066, 0.0748, 0.3791, 0.119, 0.135, 0.1739]
new business: loss 0.02491 (book 0.02491), premium 0.02858 (book 0.02858), margin 0.00367 (book 0.00367)
as a fourth unit: premium 0.02858
```

The two calculations agree because adding the account does not change the
order of the totals; on a large simulation it would reorder a few years, by
too little to matter. The account's premium is its reservation price: the
least that keeps the book on target.

## 5. Mergers and acquisitions

An acquisition is too large to price with the current weights. Section 5.4
scales the new account up about 2,000 times (a mean of 48.35) and represents
its loss at each portfolio total by two points, the conditional mean plus and
minus one conditional standard deviation, each with half the probability.
The merged book has fourteen scenarios (Table 5.4), and the same Wang
distortion prices it at assets equal to its worst total:

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
print(f"expected loss {old.loss:.2f}, {new.loss:.2f}, {tot.loss:.2f}  (book 46.60, 48.35, 94.95)")
print(f"Wang premium  {old.premium:.2f}, {new.premium:.2f}, {tot.premium:.2f}  (book 52.74, 59.09, 111.83)")
extra_premium = tot.premium - required.total.premium
extra_capital = tot.capital - required.total.capital
extra_margin = tot.margin - required.total.margin
print(f"premium needed for the acquisition {extra_premium:.3f} (book 58.26)")
print(f"assets {tot.assets:.2f} (book 298.30), capital {tot.capital:.3f} (book 186.46), "
      f"return {tot.return_on_capital:.2%} (book 9.06%)")
print(f"added capital {extra_capital:.2f} (book 140.03), added margin {extra_margin:.2f} "
      f"(book 9.92), return on it {extra_margin / extra_capital:.2%} (book 7.08%)")
```

```text
expected loss 46.60, 48.35, 94.95  (book 46.60, 48.35, 94.95)
Wang premium  52.74, 59.09, 111.83  (book 52.74, 59.09, 111.83)
premium needed for the acquisition 58.269 (book 58.26)
assets 298.30 (book 298.30), capital 186.466 (book 186.46), return 9.06% (book 9.06%)
added capital 140.03 (book 140.03), added margin 9.92 (book 9.92), return on it 7.08% (book 7.08%)
```

Everything agrees with the printed solution to its two decimals except the
premium needed (58.269 against 58.26) and the capital (186.466 against
186.46): the book subtracts the rounded 53.57 from 111.83, and its capital is
the unrounded figure cut off rather than rounded. The acquired losses are
themselves printed to two decimals, so no closer agreement is possible.

The split of the merged premium between the two books (52.74 and 59.09) is
beside the point: the acquisition changes the weights for both. What InsCo
needs from the acquired business is the increase in the book's premium,
58.26. The return on the merged book falls to about 9%, below the 15% that
fixed the Wang parameter, because the extra assets sit far in the tail, where
capital is cheap under Wang; the monograph keeps the distortion, as a
statement of what investors charge for risk, rather than the return.

## 6. Portfolio optimization

Section 5.5 turns the question around: given the prices, which mix of
business is best? The objective is economic value added (EVA), plan premium
less required premium, the parameters are a scale for each unit between -50%
and +50%, plan premium scales with volume, and capital (assets less plan
premium) must stay at 47.8. If assets and premiums scaled linearly, each
unit's allocated quantities would be its marginal ones (Table 5.5):

```python
eva = [plan - u.premium for plan, u in zip(PLAN, required.allocated)]
assets = [insco.kappa(u)[-1] for u in UNITS]           # coTVaR: the worst year, 16, 20, 64
capital = [a - plan for a, plan in zip(assets, PLAN)]
for u, e, c in zip(UNITS, eva, capital):
    print(f"{u}: EVA {e:8.4f}  plan capital {c:6.4f}  EVA per capital {e / c:8.4f}")
print(f"book: EVA {sum(eva):.4f} (book -1.3652), capital {sum(capital):.4f} (book 47.8000)")
```

```text
A: EVA  -0.2092  plan capital 2.1000  EVA per capital  -0.0996
B: EVA   0.0625  plan capital 1.3000  EVA per capital   0.0481
C: EVA  -1.2186  plan capital 44.4000  EVA per capital  -0.0274
book: EVA -1.3652 (book -1.3652), capital 47.8000 (book 47.8000)
```

These are Table 5.5's -0.2092, 0.0625 and -1.2186, and the ratios -0.0996,
0.0481 and -0.0274. A linear step takes the best ratios to their limits:
shrink A by 50%, grow B by 50%, then grow C just enough to use the freed
capital, 0.4 / 44.4 = 0.9%, predicting an EVA of -1.2403 (Table 5.6). Rerun
the model at those scales, with the same Wang distortion (not recalibrated:
it describes investors' appetite, and the target return is an output now) and
assets at the new worst year:

```python
def rescale(scales):
    """InsCo with each unit's losses and plan premium scaled by 1 + s."""
    rows = [[x * (1 + s) for x, s in zip(row, scales)] for row in ROWS]
    book = Portfolio(UNITS, rows).price(wang)
    plan = [p * (1 + s) for p, s in zip(PLAN, scales)]
    return book, plan


def show(scales, book_values, digits=4):
    book, plan = rescale(scales)
    req = [u.premium for u in book.allocated]
    w = digits + 5
    print("required", " ".join(f"{r:{w}.{digits}f}" for r in req), f" total {book.total.premium:.{digits}f}")
    print("book    ", " ".join(f"{r:{w}.{digits}f}" for r in book_values[:3]), f" total {book_values[3]:.{digits}f}")
    print(f"EVA {sum(plan) - book.total.premium:.4f}, assets {book.total.assets:.4f}, "
          f"capital {book.total.assets - sum(plan):.4f}")


linear_step = [-0.5, 0.5, 0.4 / 44.4]
print("linear prediction of EVA:", round(sum(eva) + sum(e * s for e, s in zip(eva, linear_step)), 4),
      "(book -1.2403)")
show(linear_step, [6.8115, 28.6040, 20.8271, 56.2426])  # Table 5.8
```

```text
linear prediction of EVA: -1.2404 (book -1.2403)
required    6.8115   28.6040   20.8271  total 56.2426
book        6.8115   28.6040   20.8271  total 56.2426
EVA -1.4660, assets 102.5766, capital 47.8000
```

The linear prediction is -1.2404 here and -1.2403 in Table 5.6, which adds
the four-decimal steps (0.1046, 0.0313, -0.0110) rather than the unrounded
ones. The rerun (Table 5.8) agrees to every printed digit and has an EVA of -1.466, worse than where it started.
Scaling the units reorders the years (Table 5.7), so the Wang distortion
weights different years than the linear prediction assumed. Holding A at -50%
and moving B shows how uneven the surface is (Figure 5.4):

```python
def eva_at(sa, sb):
    """EVA with unit C scaled to hold capital at 47.8 (the worst year stays the worst)."""
    sc = -(2.1 * sa + 1.3 * sb) / 44.4
    book, plan = rescale([sa, sb, sc])
    return sum(plan) - book.total.premium


for sb in (-0.5, -0.3, -0.21, -0.2, -0.1, 0.0, 0.25, 0.5):
    print(f"B {sb:+.2f}: EVA {eva_at(-0.5, sb):.4f}")
```

```text
B -0.50: EVA -1.4321
B -0.30: EVA -1.3694
B -0.21: EVA -1.3477
B -0.20: EVA -1.3484
B -0.10: EVA -1.3604
B +0.00: EVA -1.3724
B +0.25: EVA -1.4058
B +0.50: EVA -1.4660
```

The monograph's Nelder-Mead search ends at A -0.478165 and B -0.209790. A
plain grid over both scales, in steps of 0.01 and then 0.0005 around the
best, needs nothing but `price` in a loop:

```python
coarse = max((eva_at(a / 100, b / 100), a / 100, b / 100)
             for a in range(-50, 51) for b in range(-50, 51))
_, a0, b0 = coarse
fine = max((eva_at(a0 + i / 2000, b0 + j / 2000), a0 + i / 2000, b0 + j / 2000)
           for i in range(-20, 21) for j in range(-20, 21))
print(f"grid: A {fine[1]:+.4f}, B {fine[2]:+.4f}, EVA {fine[0]:.6f}")
for sb, book_values in ((-0.209790, [7.269164, 14.816395, 21.453658, 43.539217]),   # Table 5.9
                        (-0.209795, [7.334042, 14.607535, 21.597550, 43.539127])):  # Table 5.10
    print(f"at A -0.478165, B {sb}: EVA {eva_at(-0.478165, sb):.6f} (book -1.345119)")
    show([-0.478165, sb, -(2.1 * -0.478165 + 1.3 * sb) / 44.4], book_values, digits=6)
```

```text
grid: A -0.4780, B -0.2095, EVA -1.345142
at A -0.478165, B -0.20979: EVA -1.345139 (book -1.345119)
required    7.269165   14.816395   21.453677  total 43.539237
book        7.269164   14.816395   21.453658  total 43.539217
EVA -1.3451, assets 89.9941, capital 47.8000
at A -0.478165, B -0.209795: EVA -1.345139 (book -1.345119)
required    7.334043   14.607535   21.597569  total 43.539147
book        7.334042   14.607535   21.597550  total 43.539127
EVA -1.3451, assets 89.9940, capital 47.8000
```

The grid lands next to the monograph's optimum with an EVA of about -1.3451,
a small improvement on the start of -1.3652, and at the monograph's own point
the EVA agrees to about 2e-5. Two near-identical scales for B, 0.000005
apart, give the same EVA for the book and very different allocations to the
units (Tables 5.9 and 5.10): two years almost tie for their place in the
order, and which comes first decides which unit the Wang weight lands on. Units A
and B agree to the six decimals printed, or within 1e-6; unit C's premium is
2e-5 higher here, within what the unprinted digits of the scales and of the
book's calibrated Wang parameter can move. The monograph's conclusion stands: the
optimum is flat, kinked and sensitive, so read it as a direction rather than
a target.

## 7. The range of possible prices

Chapter 6 asks which distortion to use. Section 6.1 starts from the extremes:
the mean, TVaR at 0, and the maximum, TVaR at 1. Any blend of TVaRs is again a
spectral measure, and the CCoC is the blend that puts weight `d = 0.15 / 1.15`
on the maximum and the rest on the mean (Table 6.1):

```python
ccoc = distortions["ccoc"]
blend = Distortion.bitvar(0.0, 1.0, 0.15 / 1.15)
print("q CCoC ", [round(x, 3) for x in wang_q(ccoc, insco.probs)])
print("q blend", [round(x, 3) for x in wang_q(blend, insco.probs)])
```

```text
q CCoC  [0.087, 0.087, 0.087, 0.348, 0.087, 0.087, 0.217]
q blend [0.087, 0.087, 0.087, 0.348, 0.087, 0.087, 0.217]
```

Infinitely many distortions price the book at 53.565, and they disagree about
the units. The monograph spans them with its five representatives;
`Portfolio.premium_bounds` finds the true extremes, the lowest and highest
premium each unit can be charged by any distortion that prices the book at
53.565, which are always bi-TVaRs (Section 11.2 of *Pricing Insurance Risk*):

```python
target = required.total.premium
by_family = {f: [u.premium for u in insco.price(g, assets=100).allocated]
             for f, g in distortions.items()}
print(f"{'':2s} {'lowest':>7s} {'five, low':>10s} {'five, high':>11s} {'highest':>8s}")
for i, b in enumerate(insco.premium_bounds(target, assets=100)):
    five = [by_family[f][i] for f in FAMILIES]
    print(f"{b['unit']:2s} {b['lower']:7.3f} {min(five):10.3f} {max(five):11.3f} {b['upper']:8.3f}")
```

```text
    lowest  five, low  five, high  highest
A   13.098     13.739      14.127   15.033
B   17.466     18.349      20.412   20.412
C   19.255     19.371      21.304   22.098
```

The five representatives cover most of each unit's range but not all of it:
unit B's highest possible premium is the TVaR's, while unit A can be priced
as low as 13.098 by a distortion outside the five. The monograph's warning
applies: unless one distortion lies above another everywhere, they will rank
some pair of risks in opposite orders, so there is no single ordering of
distortions by price.

## 8. Required premium by distortion

Section 6.2 tabulates the required premium and loss ratio for each unit
under each distortion (Table 6.2), and section 6.3 which distortion is
cheapest for each unit (Table 6.3):

```python
BOOK_62 = {  # Table 6.2, premium by unit
    "ccoc": (13.739, 18.522, 21.304), "ph": (14.060, 18.349, 21.156),
    "wang": (14.109, 18.637, 20.818), "dual": (14.127, 19.117, 20.321),
    "tvar": (13.783, 20.412, 19.371),
}
loss = insco.expected()
print(f"{'':5s} " + " ".join(f"{u + ' premium':>17s} {'LR':>6s}" for u in UNITS))
for f in FAMILIES:
    cells = [f"{p:7.3f} ({b:7.3f}) {l / p:6.1%}" for p, b, l in zip(by_family[f], BOOK_62[f], loss)]
    print(f"{f:5s} " + " ".join(cells))
print(f"{'plan':5s} " + " ".join(f"{p:7.3f} {'':9s} {l / p:6.1%}" for p, l in zip(PLAN, loss)))
for i, u in enumerate(UNITS):
    cheapest = min(FAMILIES, key=lambda f: by_family[f][i])
    within = min(by_family[f][i] for f in FAMILIES) <= PLAN[i]
    print(f"{u}: cheapest {cheapest}, plan meets at least one distortion: {within}")
```

```text
              A premium     LR         B premium     LR         C premium     LR
ccoc   13.739 ( 13.739)  97.5%  18.522 ( 18.522)  98.8%  21.304 ( 21.304)  69.9%
ph     14.060 ( 14.060)  95.3%  18.349 ( 18.349)  99.7%  21.156 ( 21.156)  70.4%
wang   14.109 ( 14.109)  95.0%  18.637 ( 18.637)  98.2%  20.819 ( 20.818)  71.6%
dual   14.127 ( 14.127)  94.9%  19.117 ( 19.117)  95.7%  20.322 ( 20.321)  73.3%
tvar   13.783 ( 13.783)  97.2%  20.412 ( 20.412)  89.7%  19.371 ( 19.371)  76.9%
plan   13.900            96.4%  18.700            97.9%  19.600            76.0%
A: cheapest ccoc, plan meets at least one distortion: True
B: cheapest ph, plan meets at least one distortion: True
C: cheapest tvar, plan meets at least one distortion: True
```

The dual power premium for unit C is 20.32158 here; the book prints the same
quantity as 20.321 in Table 6.2 and 20.322 in Table 9.8 (its dual parameter,
a few millionths lower, gives 20.3215, on the rounding point), and Wang's is 20.81858, printed
20.818 in Table 6.2 and 20.819 in Table 5.1: the same values rounded
differently, not a difference between the programs. The cheapest distortion
is CCoC for A, PH for B and TVaR for C, and each unit's plan meets at least
one of them, but no one distortion is met by all three.

## 9. Zeroing in, or embracing the range

Section 6.3 suggests calibrating to the market where possible. If InsCo's Baa
bonds have a real default probability of 1.5% to 2% and are priced as if it
were 2.5% to 4%, a candidate distortion should map the first range into the
second:

```python
print(f"{'':5s} {'g(1.5%)':>8s} {'g(2%)':>7s}  overlaps 2.5% to 4%")
for f, g in distortions.items():
    lo, hi = g.g(0.015), g.g(0.02)
    print(f"{f:5s} {lo:8.2%} {hi:7.2%}  {hi >= 0.025 and lo <= 0.04}")
```

```text
       g(1.5%)   g(2%)  overlaps 2.5% to 4%
ccoc    14.35%  14.78%  False
ph       4.85%   5.97%  False
wang     3.38%   4.35%  True
dual     2.38%   3.17%  True
tvar     2.06%   2.74%  True
```

Wang and the dual power fit the bond; TVaR touches the window only at its
top, and PH and the CCoC overprice the bond's risk. This is the monograph's
"Dual and Wang, with TVaR a distant third".

Section 6.4 takes the other path: no distortion is right, but the range
teaches something. The stop loss of section 3 is worth 7.6 to InsCo under
the CCoC and 4.8 under TVaR, a range as wide as the reinsurance cycle:

```python
for f, g in distortions.items():
    _, ceded = split.price(g, assets=100).allocated
    print(f"{f:5s} cover worth {ceded.premium:.3f}, loss ratio {ceded.loss / ceded.premium:.0%}, "
          f"buy at 6.5: {ceded.premium > 6.5}")
```

```text
ccoc  cover worth 7.609, loss ratio 46%, buy at 6.5: True
ph    cover worth 6.662, loss ratio 53%, buy at 6.5: True
wang  cover worth 6.087, loss ratio 58%, buy at 6.5: False
dual  cover worth 5.415, loss ratio 65%, buy at 6.5: False
tvar  cover worth 4.803, loss ratio 73%, buy at 6.5: False
```

A CCoC view buys the cover at 6.5, as would PH; Wang, the dual power and TVaR
do not (the monograph quotes the ends, 7.6 at a 46% loss ratio and 4.8 at
73%). A plan inside a unit's range, as all three are here, is consistent with
some risk appetite; a plan outside the range of every distortion, which
`premium_bounds` would show, is inconsistent with any and needs scrutiny.

## 10. What this guide leaves out

- **The surrogate and LOWESS steps** of sections 5.3 and 5.4: the scatter of
  simulated losses behind Figures 5.1 to 5.3 is not printed, so the guide
  starts from the conditional means and the two-point samples the monograph
  tabulates. prospicio has no LOWESS smoother; any regression of a unit's
  losses on the total gives the same input.
- **Nelder-Mead**: section 6 searches by grid; `scipy.optimize.minimize`
  with `method="Nelder-Mead"` on `eva_at` works too, but its end point
  depends on the start and the implementation, which on this kinked surface
  matters.
- **Capital by unit** (section 8.3): see section 5 of
  [the chapter 9 guide](capital-modeling-monograph-15.md); chapters 5 and 6
  allocate premium and margin only.

For which distortion to use and how to calibrate it to quotes, see
[Risk loads and distortions](risk-loads-and-distortions.md).
