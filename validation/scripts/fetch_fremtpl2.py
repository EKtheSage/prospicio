"""Download freMTPL2freq and freMTPL2sev from OpenML and write the prepared
CSVs the freMTPL2 parity uses.

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

freMTPL2sev (OpenML dataset 41215, version 1) has one row per claim:
26,639 claim amounts keyed by IDpol. It is checked against its SHA-256,
aggregated per policy (NSev, the number of amounts, and AvgClaim, their
mean) and inner-joined on IDpol to the prepared freMTPL2freq rows, in
freMTPL2freq's order: the severity data has the same columns plus NSev and
AvgClaim. Policies with claims but no amounts, and amounts whose IDpol is
not in freMTPL2freq, drop out of the join.

The outputs, validation/data/external/freMTPL2freq_glm.csv and
freMTPL2sev_glm.csv, are not committed (37 MB in, about 40 MB out): the
parity test reads them when they are there, and CI fetches them with this
script and caches them.
"""

import csv
import hashlib
import math
import os
import sys
import urllib.request

URL = "https://www.openml.org/data/v1/download/20649148/freMTPL2freq.arff"
SHA256 = "a45363e056e2ea56408b38eeb9d4d04d7f6c6982eb7a14ed5e807c7c71807cdd"
SEV_URL = "https://www.openml.org/data/v1/download/20649149/freMTPL2sev.arff"
SEV_SHA256 = "047632016b87e132247124f449a95d6d0e9f8ba05f2fe0947ef76ab8aaa10e0a"
OUT = "validation/data/external/freMTPL2freq_glm.csv"
SEV_OUT = "validation/data/external/freMTPL2sev_glm.csv"
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


def download(url, sha256, name):
    """The ARFF's data lines, after checking its SHA-256."""
    raw = urllib.request.urlopen(url, timeout=300).read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != sha256:
        sys.exit(f"{name} has SHA-256 {digest}, expected {sha256}")
    lines = raw.decode().splitlines()
    return [line for line in lines[lines.index("@data") + 1:] if line]


def severity():
    """Claim amounts per policy: IDpol -> (NSev, AvgClaim)."""
    amounts = {}
    for line in download(SEV_URL, SEV_SHA256, "freMTPL2sev.arff"):
        idpol, amount = line.split(",")
        amounts.setdefault(int(float(idpol)), []).append(float(amount))
    return {k: (len(v), math.fsum(v) / len(v)) for k, v in amounts.items()}


def main():
    lines = download(URL, SHA256, "freMTPL2freq.arff")
    sev = severity()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n = n_sev = 0
    with open(OUT, "w", newline="") as f, open(SEV_OUT, "w", newline="") as fs:
        w = csv.writer(f, lineterminator="\n")
        ws = csv.writer(fs, lineterminator="\n")
        w.writerow(COLUMNS)
        ws.writerow(COLUMNS + ["NSev", "AvgClaim"])
        for line in lines:
            v = [x.strip("'") for x in line.split(",")]
            idpol, claims, exposure, area, power, vage, dage, bm, brand, gas, density, region = v
            row = [
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
            ]
            w.writerow(row)
            n += 1
            amounts = sev.get(int(float(idpol)))
            if amounts is not None:
                ws.writerow(row + [amounts[0], repr(amounts[1])])
                n_sev += 1
    print(f"wrote {n} policies to {OUT} and {n_sev} to {SEV_OUT}", file=sys.stderr)


if __name__ == "__main__":
    main()
