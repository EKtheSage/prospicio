# Datasets

* [CAS loss reserve database, six lines](clrd-lines.md) - Paid triangles of comauto, medmal, othliab, ppauto, prodliab and wkcomp summed over companies (chainladder-python 0.10.1), 1988-1997; same shape, so they can be synchronized.
* [freMTPL2freq and freMTPL2sev](fremtpl2.md) - 678,013 French motor TPL policies with claim counts and exposure, and their 26,639 claim amounts; the GLM parity portfolio, fetched from OpenML, not committed.
* [RAA triangle](raa.md) - Mack's (1993) 10x10 cumulative paid triangle; has one negative increment, which shapes how ODP models must handle it.
* [Meyers' illustrative insurer, group 353 Commercial Auto](clrd-comauto-353.md) - The full 10x10 square (paid, case incurred, net earned premium) of CAS group 353 from the CAS Schedule P data model release 20260613_041006; matches Meyers' Tables 2.1-2.3 but for one misprint, outcomes paid 40,000 and incurred 40,061.
