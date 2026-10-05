"""Download freMTPL2freq from OpenML and write the prepared CSV the
freMTPL2 parity uses.

    python validation/scripts/fetch_fremtpl2.py

freMTPL2freq is the French motor third-party liability portfolio of the R
package CASdatasets (Charpentier), 678,013 policies, as OpenML dataset
41214 (version 1). It is downloaded, checked against its SHA-256 and
prepared as in Noll, Salzmann and Wüthrich (2018), "Case Study: French
Motor Third-Party Liability Claims" (SSRN 3164764), section on the GLM:

- ClaimNb capped at 4 and Exposure at 1;
- Area as its rank 1 (A) to 6 (F), a numeric term;
- VehPower capped at 9, a factor;
- VehAge in the classes 0-5, 6-12, 13+ (reference 6-12);
- DrivAge in 18-20, 21-25, 26-30, 31-40, 41-50, 51-70, 71+ (reference
  41-50);
- BonusMalus capped at 150, numeric;
- Density on the log scale, numeric;
- VehBrand, VehGas and Region as factors (reference R24 for Region).

The output, validation/data/external/freMTPL2freq_glm.csv, is not
committed (36 MB in, about 30 MB out): the parity test reads it when it is
there, and CI fetches it with this script and caches it.
"""

import csv
import hashlib
import math
import os
import sys
import urllib.request

URL = "https://www.openml.org/data/v1/download/20649148/freMTPL2freq.arff"
SHA256 = "a45363e056e2ea56408b38eeb9d4d04d7f6c6982eb7a14ed5e807c7c71807cdd"
OUT = "validation/data/external/freMTPL2freq_glm.csv"
COLUMNS = ["ClaimNb", "Exposure", "Area", "VehPower", "VehAge", "DrivAge",
           "BonusMalus", "VehBrand", "VehGas", "Density", "Region"]


def veh_age(a):
    return "0-5" if a <= 5 else "6-12" if a <= 12 else "13+"


def driv_age(a):
    for hi, label in [(20, "18-20"), (25, "21-25"), (30, "26-30"), (40, "31-40"),
                      (50, "41-50"), (70, "51-70")]:
        if a <= hi:
            return label
    return "71+"


def main():
    raw = urllib.request.urlopen(URL, timeout=300).read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != SHA256:
        sys.exit(f"freMTPL2freq.arff has SHA-256 {digest}, expected {SHA256}")
    lines = raw.decode().splitlines()
    start = lines.index("@data") + 1
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = 0
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(COLUMNS)
        for line in lines[start:]:
            if not line:
                continue
            v = [x.strip("'") for x in line.split(",")]
            _id, claims, exposure, area, power, vage, dage, bm, brand, gas, density, region = v
            w.writerow([
                min(int(float(claims)), 4),
                repr(min(float(exposure), 1.0)),
                "ABCDEF".index(area) + 1,
                str(min(int(float(power)), 9)),
                veh_age(int(float(vage))),
                driv_age(int(float(dage))),
                min(int(float(bm)), 150),
                brand,
                gas,
                repr(math.log(float(density))),
                region,
            ])
            n += 1
    print(f"wrote {n} policies to {OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
