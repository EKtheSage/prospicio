"""Write validation/reference/sobol_scipy.csv: unscrambled Sobol points from SciPy.

Rows: SciPy 1.14.1 stats.qmc.Sobol(d, scramble=False, bits=32).random(256),
each coordinate times 2^32 (an integer), for d in 1, 2, 7, 40 and 1111;
every coordinate for d up to 7, every third for 40 and every 101st (and
the last) for 1111.
python3 validation/scripts/sobol_scipy.py
"""

from pathlib import Path

import scipy
from scipy.stats import qmc

OUT = Path(__file__).resolve().parents[1] / "reference/sobol_scipy.csv"


def main():
    lines = ["dim,coord,points,source"]
    for d in (1, 2, 7, 40, 1111):
        x = qmc.Sobol(d, scramble=False, bits=32).random(256)
        step = 1 if d <= 7 else 3 if d <= 40 else 101
        coords = sorted(set(range(0, d, step)) | {d - 1})
        for c in coords:
            ints = ";".join(str(int(v * 2**32)) for v in x[:, c])
            lines.append(f"{d},{c},{ints},SciPy {scipy.__version__}")
    OUT.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
