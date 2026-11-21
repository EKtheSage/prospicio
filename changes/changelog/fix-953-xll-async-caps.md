### Excel

- Asynchronous functions share a pool of threads instead of starting one per call, so a sheet of hundreds of cells no longer starts hundreds of threads. A result over 4 million cells is returned and not kept, so it no longer empties the kept results, and the least recently used result is evicted first, not the oldest. `PRO.CAP.COPULA.SIM` and `PRO.CAP.NESTED.SIM` return an error text for more than 10,000,000 cells (rows times columns) instead of risking an out-of-memory crash of Excel. Numbers do not change.
