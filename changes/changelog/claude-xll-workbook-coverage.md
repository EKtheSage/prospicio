### Excel

- `PRO.EXPOSURE.CURVE` and `PRO.EXPOSURE.QUANTILE` tabulate several
  curves at once: a column of x against a row of Swiss Re c values (or
  MBBEFD b or g) gives one column per curve.
- The test workbook (`harness/make_workbook.py`) calls every one of the
  96 functions with a check beside it; the ODP sheet's P95 checks read
  the P95 column.
