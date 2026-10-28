# Report templates

Quarto templates that turn the results of an analysis into a page an
investor or a committee can read without the code. A template holds no data
and runs no model: it reads a folder of tidy tables, named by a parameter in
its header, and stops with the name of any table or column it cannot find.

| Template | What it shows |
|---|---|
| [`deal_report.qmd`](deal_report.qmd) (R) and [`deal_report_py.qmd`](deal_report_py.qmd) (Python) | A reinsurance deal for an investor: the structures' terms, the loss ratio and the ratios that make up the result, the gross, ceded and net loss, the investor's IRR and MOIC with the probability of loss, the structures side by side, and actual against expected for monitoring |

The two deal report templates are twins, one per language, and show the same
tables. The tables they read, their columns and how to make them from a deal
analysis are in the guide
[`docs/guides/submission-schema.md`](../../docs/guides/submission-schema.md)
("The report's tables"); [`../data/deal_report/`](../data/deal_report/) is
the guide's output for the sample submission. From this folder:

```sh
quarto render deal_report.qmd -P tables:../data/deal_report
quarto render deal_report_py.qmd -P tables:../data/deal_report
```

The R template needs `knitr` and `rmarkdown`. The Python one needs Jupyter,
and `papermill` for `-P`:

```sh
uv run --no-project --with jupyter --with papermill quarto render deal_report_py.qmd -P tables:../data/deal_report
```

Neither needs prospicio. `money_unit` and `money_label` (by default 1,000,000
and `m`) set how amounts are shown. CI's `deal-report` job renders both
templates on the sample tables when a pull request changes them, the sample
tables or the guide. Rendered `.html` is not committed.
