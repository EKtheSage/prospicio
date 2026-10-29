### Reserving

- Triangles read from files (#552): `prospicio-reserving`'s optional `io`
  feature (off by default; the `csv` and `calamine` crates) adds
  `FileLayout` (long or wide), `Triangle::from_csv` on any reader,
  `Triangle::read_csv` and `Triangle::read_excel` (`.xlsx`, `.xlsm`,
  `.xlsb`, `.xls`, `.ods`), with errors that name the line or cell and the
  column. Python `Triangle.read_csv` and `Triangle.read_excel`, and
  `prospicio.reserving.schedule_p()`, CAS Schedule P triangles by line and
  company from the `cas-schedule-p` package, with the held-out lower
  triangle on request. R `read_triangle()` and `schedule_p_triangle()`, the
  latter on the CAS loss reserve database's CSV files. **Breaking** in
  Rust: `Error` gains the variant `File`.
