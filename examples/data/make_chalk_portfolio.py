"""Write the simulated motor portfolio of the CAS Monograph 16 guides as CSV files.

Run from anywhere:  python examples/data/make_chalk_portfolio.py
It needs numpy only. docs/guides/glm-to-comprehensive-pricing-chalk.md builds
the portfolio in Python with numpy's generator (seed 2025); R cannot replay
numpy's draws, so the R twins of the guides (the .qmd files next to them)
read the same draws from the files this script writes:

  chalk_portfolio.csv   the 20,000 policies: features, exposure, claims and
                        `shuffle`, the permutation part 2 of the guide uses
                        on each fold: row i of fold k takes the values of the
                        fold's row shuffle[i], counted from 0 in row order
                        (np.random.default_rng(k)); -1 on test rows
  chalk_toys.csv        the true effect of each of the 300 makes, and the
                        small examples of part 1 drawn after the portfolio
                        from the same generator (ridge in metres, soft
                        thresholding, ridge on dummies)

The code below is the guide's, draw for draw, so the files are the same every
run; they are committed, and this script is how they were made.
"""
from pathlib import Path

import numpy as np

here = Path(__file__).resolve().parent

rng = np.random.default_rng(2025)
n = 20_000
age = rng.integers(18, 81, n)
veh_age = rng.integers(0, 21, n)
power = rng.choice(["low", "mid", "high"], n, p=[0.5, 0.35, 0.15])
region = rng.choice(list("ABCDEF"), n, p=[0.3, 0.2, 0.15, 0.15, 0.1, 0.1])
zipf = 1 / np.arange(1, 301) ** 1.1
make = rng.choice(300, n, p=zipf / zipf.sum())
make_effect = rng.normal(0, 0.4, 300)
noise = rng.normal(size=n)
exper = np.maximum(age - 17 - rng.poisson(2, n), 0)
exposure = rng.uniform(0.5, 1.0, n).round(3)
true_lp = (
    np.log(0.25)
    + 0.9 * np.exp(-(age - 18) / 6)
    + 0.0015 * np.maximum(age - 60, 0) ** 1.5
    - 0.08 * np.minimum(veh_age, 6)
    + np.select([region == "B", region == "C", region == "F"], [0.15, -0.2, 0.3], 0.0)
    + make_effect[make]
    + np.where(power == "high", 0.25, 0.0)
    + np.where((power == "high") & (age < 25), 0.6, 0.0)
)
claims = rng.poisson(np.exp(true_lp) * exposure).astype(float)
fold = np.arange(n) % 10

# Part 1's small examples, drawn next from the same generator.
x_m = rng.uniform(1.0, 5.0, 200)
y_len = 2.0 * x_m + rng.normal(0, 1, 200)
x_std = rng.normal(size=500)
x_std = (x_std - x_std.mean()) / x_std.std()
y_soft = 0.3 * x_std + rng.normal(0, 1, 500)
y_soft -= y_soft.mean()
sizes = [1, 5, 50, 400]
levels = np.repeat(np.arange(4), sizes)
y_lvl = rng.normal(np.array([2.0, -1.0, 0.5, 0.0])[levels], 1.0)

# Part 2's permutation of each cross-validation fold.
shuffle = np.full(n, -1)
for k in range(7):
    shuffle[fold == k] = np.random.default_rng(k).permutation((fold == k).sum())


def main(out=here):
    """Write both files to the folder `out`."""
    with open(Path(out) / "chalk_portfolio.csv", "w", newline="\n") as f:
        f.write("# Simulated motor portfolio of docs/guides/glm-to-comprehensive-pricing-chalk.md (numpy seed 2025),\n")
        f.write("# written by examples/data/make_chalk_portfolio.py. One row per policy; the policy number is the row number\n")
        f.write("# counted from 0 and its last digit the fold. shuffle is -1 on test rows.\n")
        f.write("age,veh_age,power,region,make,noise,exper,exposure,claims,shuffle\n")
        for i in range(n):
            f.write(f"{age[i]},{veh_age[i]},{power[i]},{region[i]},{make[i]},{float(noise[i])!r},{exper[i]},"
                    f"{float(exposure[i])!r},{claims[i]:.0f},{shuffle[i]}\n")

    with open(Path(out) / "chalk_toys.csv", "w", newline="\n") as f:
        f.write("# Part 1's small examples of docs/guides/glm-to-comprehensive-pricing-chalk.md, drawn after the portfolio\n")
        f.write("# from the same numpy generator, written by examples/data/make_chalk_portfolio.py. x is the feature\n")
        f.write("# (for 'dummies' the level, 0 to 3) and y the response; a 'make_effect' row holds the true log effect y of make x.\n")
        f.write("example,x,y\n")
        for j, e in enumerate(make_effect):
            f.write(f"make_effect,{j},{float(e)!r}\n")
        for x, y in zip(x_m, y_len):
            f.write(f"metres,{float(x)!r},{float(y)!r}\n")
        for x, y in zip(x_std, y_soft):
            f.write(f"soft,{float(x)!r},{float(y)!r}\n")
        for x, y in zip(levels, y_lvl):
            f.write(f"dummies,{int(x)},{float(y)!r}\n")


if __name__ == "__main__":
    main()
