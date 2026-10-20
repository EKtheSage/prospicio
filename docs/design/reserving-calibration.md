# Design note: the 200-triangle calibration study (Meyers' ultimate percentile test)

Status: **Accepted** (2026-10-20; issue #392; the user decided the
points marked "the user decided"; two points the user did not decide are
taken as this note's recommendation and marked so) · Lane: Reserving
(`validation/` only; `crates/prospicio-reserving/` only if a thin helper
is needed) · Depends on: `reserving-bayes.md` (decision 7 and open
decision 7, the data source; decision 8, the diagonal backtest),
`reserving-compartmental.md` (decision 8, PR E, merged as #447),
`reserving-v02.md` (the bootstraps), `predictive-distribution.md`
(`Sampled::cdf`)

## Goal

Run Meyers' own calibration test on prospicio's reserving models: fit
each model on the upper triangle (accident years 1988 to 1997, the 1997
statement) of 200 Schedule P triangles, 50 companies in each of
commercial auto (CA), personal auto (PA), workers' compensation (WC) and
other liability (OL); take the percentile of the actual outcome (from the
lower triangle) under the model's predictive distribution; and test the
200 percentiles, and the 50 of each line, for uniformity with a p-p plot
and the Kolmogorov-Smirnov (KS) bound Meyers uses (Meyers (2019), CAS
Monograph 8, sec. 3 and 4).

What it adds over what the repo has:

* the **ultimate**, not a diagonal. The diagonal backtest
  (`diagonal_backtest`, `crates/prospicio-reserving/src/backtest.rs:462`)
  scores held-out calendar diagonals of an observed triangle; its
  `total_percentile` (backtest.rs:298-305) is the share of simulated
  diagonal totals at or below the actual one. It never sees the age-10
  value of an origin, which is what Meyers tests;
* **many companies, four lines.** Every parity test so far is one
  triangle (group 353 CA, company 337 WC, RAA, GenIns). One triangle
  checks the machinery (`knowledge/references/meyers-monograph-8-reproduced.md`
  reproduces every printed group 353 percentile); it cannot say whether a
  model is calibrated;
* a **comparison with Meyers' published D statistics** for the same models
  on (as far as the screen allows) the same panel.

Out of scope: Meyers' SCC and IPI models (not in prospicio), the
Anderson-Darling test (Meyers reports that almost all models fail it,
p.8; it can be added to the output later at no cost), one-year views,
tails (Meyers' outcome has none) and the Bayesian Bornhuetter-Ferguson
(no Meyers counterpart; a later arm on request).

## Pull requests

Each PR is small and lands in `validation/` and docs. No binding changes:
this is a validation study, not a user-facing feature (#342, the backtest
bindings, is not a prerequisite). The models are staged (the user
decided, 2026-10-20).

| PR | Branch | Holds | Base |
|---|---|---|---|
| 1 | `claude/calibration-study` | this note; the lane, roadmap and knowledge entries | `main` |
| 2 | `claude/calibration-harness` | the panel script `validation/scripts/calibration_panel.py` (decision 2), the Rust panel loader, the percentile and KS code (decision 5), the harness (decision 6) with Mack (paid and incurred) and the ODP bootstrap (paid), the `#[ignore]`d study, its outputs (decision 7); the panel accepted only if Mack reproduces Meyers' D (decision 3); a timing probe (the models built by then; see "As built") | PR 1 |
| 3 | `claude/calibration-bayes` | `BayesOdp` on paid, and Meyers' CRC, CSR and CAY on his bases; the comparison table with Meyers' D (decision 8) | PR 2 |
| 4 | `claude/calibration-compartmental` | Compartmental model 1 (paid), and model 2 as an optional, slower arm (decision 6) | PR 3 |

After a run the user reviews, the D values are pinned within a tolerance
(decision 7) and the knowledge finding is written; that happens in the
PR whose run was reviewed. The PR that finishes the study (PR 4) also
moves the bullet `[calibration-study]` in `docs/lanes/reserving.md` to
Done, updates `docs/roadmap.md` (the line that names the study),
`reserving-bayes.md` (the lines that call the study "next"), and this
note's status. No changelog fragment: the harness exposes nothing to
users. The old PR 0 (a note-only branch) and the vendored-CSV PR are gone.

Note on the issue text: #392 says Compartmental can join "once PR E gives
it `fit_segments` and `TriangleModel`". PR E is merged (#447,
`1f425a37`): `Compartmental::fit_segments` (compartmental.rs:924) and
`CompartmentalCandidate` (bayes/compartmental/backtest.rs:732) exist. The
only reason to defer Compartmental now is cost (decision 4).

## References

| What | Primary reference | Secondary |
|---|---|---|
| The test, the panel, the results | Meyers (2019), *Stochastic Loss Reserving Using Bayesian MCMC Models*, CAS Monograph 8, 2nd ed.: sec. 3 (Mack on incurred, p.7-11), 4 (Mack and ODP on paid, p.12-15), 5 (CRC, p.19-20), 7 (CSR, p.35), 8 (CAY, p.42); Appendix (screen, Tables A.1 and A.2, p.83-85) | `knowledge/references/meyers-monograph-8.md`, `-reproduced.md`; the guide `docs/guides/reserving-meyers.md` (sec. 3, the KS mechanics) |
| Meyers' code and summary statistics | `meyers-appendix.zip` (sha256 in `knowledge/references/meyers-monograph-8.md:12`); per the monograph (p.3) it holds "key summary statistics from all the models" | not downloaded: the user decided (2026-10-20) on no download of the zip; the panel is the rebuilt screen |
| The data | the CAS Schedule P loss reserving database (NAIC Schedule P, Meyers and Shi), as rebuilt in [EKtheSage/cas-schedule-p-data-model](https://github.com/EKtheSage/cas-schedule-p-data-model) and bundled as parquet in the pip package `cas-schedule-p==2026.6.13` (the repo's other panels use release `20260613_041006`, table `mart_reserving_model_training`) | the CAS originals under `https://www.casact.org/sites/default/files/2021-04` |
| The screen | `cas_schedule_p.screens.select_companies`, shipped in the pip package `cas-schedule-p==2026.6.13` (the user decided, 2026-10-20), rebuilt from Meyers' appendix criteria, not his list (`reserving-bayes.md` open decision 7) | Table A.2 |

Page numbers are the monograph's printed pages (the PDF page less 10).
The D values quoted in this note were read from the figure text of
`knowledge/references/08-Meyers.pdf` (untracked in the main checkout) on
2026-10-20 and match the scouts' reading.

## Decisions

### 1. What is tested, model by model

The outcome of a triangle is Meyers' (p.8, Table 3.1): the sum over the
ten origins of the cumulative value at age 10 (120 months), including the
1988 origin, already at age 10 in the training data. No tail. The
percentile is the share of simulated totals at or below the outcome,
`Sampled::cdf(x) = #{d <= x} / n` (`crates/prospicio-prob/src/sampled.rs:149-151`),
which is exactly Meyers' statistic (`meyers-monograph-8-reproduced.md:47-48`).

The grid of models and bases mirrors Meyers', so that each cell has a
published D to compare with (decision 8). "Incurred" is case incurred,
`incurred_loss - bulk_loss`, and premium is `earned_prem_net`, as in
`clrd_comauto_353.csv`.

| Arm | Basis | API | Percentile of | Meyers |
|---|---|---|---|---|
| Mack | paid, incurred | `Mack::default().fit(tri, col)` (mack.rs:102) | a log-normal matched to `sum(chain_ladder.ultimate)` and `total_standard_error`, at the outcome (Meyers p.7; the guide's `lognormal_percentile`, reserving-meyers.md:103-107, needs a Rust twin in the harness) | yes, both |
| ODP bootstrap | paid | `OdpBootstrap { n_sims: 10_000, seed, .. }.fit(tri, "paid")` (bootstrap.rs:459) | `reserves.total().cdf(outcome - sum(chain_ladder.latest))` | yes (10,000 sims, p.12) |
| `BayesOdp` | paid (basis open, see the end) | `BayesOdp::default().fit(tri, "paid")` (bayes/odp.rs:290) | as the ODP bootstrap | no counterpart |
| CRC | paid, incurred | `Meyers { model: Crc, seed, .. }.fit(tri, col, "premium")` (meyers.rs:457) | `ultimates.total().cdf(outcome)` (meyers.rs:264, no shift needed) | yes, both |
| CSR | paid | as CRC with `Csr` | as CRC | yes |
| CAY | incurred | as CRC with `Cay` | as CRC | yes |
| Compartmental model 1 | paid (with outstanding from case incurred) | `Compartmental::new(CompartmentalPriors::gesmann_morris()).fit(tri, "paid", Outstanding::FromIncurred("incurred"), "premium")` (compartmental.rs:683, 734) | `reserves.total().cdf(outcome_paid - sum(latest_paid))` (compartmental.rs:1549, 1583) | no published result |
| Compartmental model 2 | paid | `Compartmental::model_2(CompartmentalPriors::gesmann_morris_2())` (compartmental.rs:711) | as model 1 | no published result |

Compartmental's reserves are paid to the square's last age, so it scores
against the paid outcome only. Its priors are the paper's presets, tuned
to WC company 337 (`reserving-compartmental.md` open decision 2); on CA,
PA and OL they are a poor prior by inference, and the study reports that
as a property of the preset, not of the model. Its `Outstanding::FromIncurred`
on case incurred differs from Gesmann and Morris' outstanding (with bulk
and IBNR); that matters for parity with the paper, not for calibration.

The triangle is built from the 55 upper cells only, with
`Triangle::from_long` (triangle.rs:180), so no model can see the lower
triangle; the outcomes are read from the panel separately. The per-origin
percentiles are recorded too (decision 7), but the test is on the total,
as Meyers'.

### 2. The data and how it reaches the study

The user decided (2026-10-20): the study uses the `cas-schedule-p` data
package at run time (pip `cas-schedule-p==2026.6.13`, which bundles the
parquet and ships `cas_schedule_p.screens.select_companies`). The panel
is **not stored in the repo** (no CSV), and Meyers' appendix zip is not
downloaded. The panel is the rebuilt screen, diffed against Table A.2's
175 printed codes and checked by Mack reproducing Meyers' D (decision 3).

No company-level panel is in the repo today: `validation/data/` holds
group 353 (`clrd_comauto_353.csv`), company 337 (`clrd_wkcomp_337.csv`)
and industry sums with upper triangles only (`clrd_lines.csv`,
`clrd_wkcomp.csv`).

The data path:

1. **The script** `validation/scripts/calibration_panel.py`, run as

   ```
   uv run --no-project --with "cas-schedule-p[screens]==2026.6.13" python -I validation/scripts/calibration_panel.py
   ```

   (PR 2 confirmed the package: Python 3.11 or newer, and the `screens`
   extra, which brings duckdb and pandas, is needed for
   `select_companies(mart_path=None, line=None, per_line=50)`.) It
   applies the screen, asserts group 353's square against
   `validation/data/clrd_comauto_353.csv` cell by cell, diffs the chosen
   codes against Table A.2's 175 printed codes (transcribed into the
   script as constants with the page cited; the differences are printed,
   a mismatch in PA, WC or OL is an error), and writes one long CSV with
   columns `line, group_code, origin, development, paid, incurred,
   case_incurred, premium, premium_direct` for the full 10x10 square of
   each of the 200 triangles (about 20,000 rows, by inference).
2. **The location.** The default output is `target/calibration/panel.csv`
   under the workspace root. `/target` is gitignored (`.gitignore`, line
   1; checked 2026-10-20), so nothing is committed by a run. The script
   takes `--out PATH`, and the Rust study reads the environment variable
   `RISK_RS_CALIBRATION_PANEL` first, then the default path (from
   `CARGO_MANIFEST_DIR/../target/calibration/panel.csv`, so a
   `CARGO_TARGET_DIR` elsewhere needs the variable).
3. **The study** (decision 6) reads the CSV. If the file is absent it
   fails at once with a message that holds the exact command above and
   the path it looked for; it never reaches for the network itself.

Licence: the data repo's README calls the Schedule P data public research
data from the CAS and NAIC, outside its MPL-2.0 code licence, and asks for
credit to Meyers and Shi and the CAS. That is a README statement; the CAS
site's terms were not checked. Nothing is redistributed, since the panel
is built on the user's machine at run time.

### 3. The panel and how it is checked

Meyers' screen (p.83-84): the four lines; premium above 20 and incurred
loss above 4 (Schedule P is in USD thousands); CV1 (the coefficient of
variation of net earned premium over the ten years) and CV2 (the CV of
the net-to-direct premium ratio) under the limits of Table A.1 (CA 0.795 /
0.125, PA 1.003 / 0.125, WC 0.772 / 0.300, OL 0.628 / 0.15, verified in
the PDF text); then the 50 triangles with the smallest CV1; group 38997
removed by hand after provisional testing. The data repo's
`select_companies` encodes the same limits (scout reading of its
`screens.py`; to be confirmed in PR 1).

**Table A.2 is incomplete for CA.** The PDF text of p.84-85 shows 25 CA
codes (18 on p.84, 8672 to 14974, and 7 on p.85, 15024 to 19780) and 50
each for PA, WC and OL: 175 codes, not 200 (verified 2026-10-20). That the
missing 25 are the lower-numbered CA codes is an inference: the printed
CA column starts at 8672 while the other lines start at 43, 86 and 620,
and group 353, Meyers' own illustrative CA insurer (Tables 2.1 to 2.3), is
not in the printed CA column.

So the panel is checked two ways:

1. **Codes.** Diff the rebuilt screen against Table A.2: an exact check
   for PA, WC and OL, half a check for CA. The printed codes are
   transcribed into the generator as constants with the page cited.
2. **Mack's D (the acceptance test).** Mack is deterministic and costs
   nothing, so on the rebuilt panel Mack incurred must reproduce Meyers'
   D (16.4, 16.6, 27, 14.2 by line; 15.4 on all 200; Figure 3.2, p.10) and
   Mack paid his (24.6, 44.6, 30.4, 8.5; 23.1; Figure 4.2, p.15), within a
   tolerance set in PR 2: 8 points per line and 4 on all 200. The
   tolerance is **empirical**, not a bound derived from the panel. A
   swapped triangle can move a line's D by up to 100/50 = 2 points (0.5 on
   all 200), and PR 2's panel is known to miss 26 of the 175 printed codes
   (9 CA, 6 PA, 2 WC, 9 OL), with 25 more CA codes that cannot be checked,
   so the known differences alone could move a line's D by more than 8.
   The first run was within 3.7 points per line and 1.5 on all 200; that
   the swapped triangles move D this little is observed, and why is an
   inference (see "As built"). This is the only full check of CA. The
   ODP bootstrap (Figure 4.1, p.14) is a seed-dependent second check.

If the codes or Mack's D disagree, the study still runs on the rebuilt
panel (the user decided on it) and every output says that the panel
differs and where; the difference is then a finding, not a blocker.

### 4. Cost

Measured single-fit times, release, local, with every total here an
inference from them:

| Arm | Per fit | Source | Fits | Serial total |
|---|---|---|---|---|
| Mack | under 1 ms (analytic, an inference) | | 400 | seconds |
| ODP bootstrap, 10,000 sims | a fraction of a second, by inference (the Mack bootstrap's lifetime view takes 0.02 s at 20,000 sims on ABC) | mack-bootstrap-lifetime-vs-mack.md:114 | 200 | under a minute |
| `BayesOdp` | about 0.1 s fit; the predictive 0.56 s for 4,000 × 55 in parallel | bayes-odp-vs-odp-bootstrap.md:63-65 | 200 (paid) | about 3 min |
| CRC | 0.3 s | meyers-crc-csr-parity.md:80 | 400 | 2 min |
| CSR | 0.5 s | meyers-crc-csr-parity.md:80 | 200 | 1.7 min |
| CAY | 0.5 s | meyers-cay-parity.md:73 | 200 | 1.7 min |
| Compartmental model 1 | 5.5 s (4 chains in parallel) | compartmental-gm1-parity.md:70 | 200 | 18 min |
| Compartmental model 2 | about 61 s | compartmental-gm2-parity.md:71 | 200 | 3.4 h |

Meyers' four arms: 200 × (0.3 + 0.3 + 0.5 + 0.5) s = 320 s, about 5.3
min; with a 10,000-sim predictive and elpd about 1 s a fit, up to 13 min.
Without Compartmental the study is about 20 minutes of fits locally;
model 1 adds 18 minutes (200 × 5.5 s), model 2 3.4 hours (200 × 61 s).
These are group 353, GenIns and company 337 times; real companies are
larger or messier and some fits will need a retry at a higher
`target_accept` (decision 5).

Run time: the study is run by hand (decision 6), so there is no CI
budget. The local estimates above are for a developer machine; the PR 2
probe (8 triangles per model, two per line) replaces them with measured
times before the first full run. Compartmental model 2 at about 3.4 hours
is an optional arm, run on its own selection (decision 6).

### 5. Percentiles, failures and the uniformity test

Per triangle and arm the harness records the percentile, the predictive
mean and sd of the total, the outcome, the per-origin percentiles, the
seed, the wall time, and for NUTS fits the diagnostics (`Diagnostics`:
max R-hat, min bulk and tail ESS, divergences, warnings; Compartmental's
`rate_swaps`).

Failures are recorded, never dropped silently:

* an `Err` from `fit` (a non-positive cell for Meyers, meyers.rs:114,
  where Meyers' own code floors at 1; a non-positive column sum for
  `BayesOdp`; Compartmental's rate-swap guard, compartmental.rs:3974) is
  a row with the error text and no percentile;
* a fit whose diagnostics fail (R-hat above 1.05, or divergences) is
  refitted once at `target_accept` 0.99, the remedy the Meyers backtest
  sweep already uses (`meyers-segments-backtest.md`); both results are
  recorded, and the KS test uses the refit;
* the KS test is reported on the triangles every arm scored (the common
  set) and on each arm's own set, with the counts. The policy for
  non-positive cells is decision 5 of the list at the end.

Seeds: `seed = hash(line, group_code, model, basis)`, fixed in the
harness, so each triangle's fit is reproducible and independent of the
order or sharding of the run. NUTS draws at a fixed seed differ between
Windows and Linux (`meyers-monograph-8-reproduced.md`), so the recorded
percentiles carry the platform; KS conclusions are expected to be stable,
the last digits not.

The test: with the sorted percentiles `p_(i)` in percent,

* **Meyers' D**, `max_i |p_(i) - 100 i / n|` (his figures; the guide,
  reserving-meyers.md:181-184), reported for comparability;
* the **two-sided KS statistic**, `max_i max(100 i / n - p_(i), p_(i) - 100 (i - 1) / n)`,
  which is at least Meyers' D and at most 100/n above it, reported beside
  it;
* the 5% bound `1.36 / sqrt(n)` in probability, which is `136 / sqrt(n)`
  in percentage points: 19.2 for a line (n = 50), 9.6 for all 200. On a
  common set smaller than 50 or 200 the bound uses the actual n.

Per line and on all 200, as Meyers. The p-p plot is written as a table
(decision 7); no plotting dependency in Rust.

### 6. The harness and where it runs

The user decided (2026-10-20): **manual only.** The study is an
`#[ignore]`d test run by hand, like the seed sweeps. There is no CI job,
no new CI variable (no `RISK_RS_CALIBRATION`), and no schedule.

The file is `validation/tests/reserving_calibration_meyers.rs` (the
Reserving lane owns `validation/tests/reserving*.rs`), with a `[[test]]`
entry in `validation/Cargo.toml`. PR 2's Mack and ODP arms need only the
default features. PR 3 and 4 add the Bayesian arms behind `slow-bayes`;
the `bayes` job scans `validation/Cargo.toml` for `slow-bayes` entries,
so PR 3 puts those arms where that job does not run the study (it does
not pass `--include-ignored`; PR 3 confirms this in ci.yml and says so).
The study never runs on a pull request or on a schedule.

Run by hand (the file's header and the loader's failure message carry the
same commands):

```
uv run --no-project --with "cas-schedule-p[screens]==2026.6.13" python -I validation/scripts/calibration_panel.py
cargo test --release -p prospicio-validation --test reserving_calibration_meyers -- --ignored --nocapture
```

Selection is by environment variables read by the test itself, with
`if`, not `cfg`, so clippy lints every arm: `RISK_RS_CALIBRATION_LINES`
(for example `ca,pa`; default all four) and `RISK_RS_CALIBRATION_ARMS`
(for example `mack,odp`; default the arms built so far, model 2 only when
named). These are options of a manual run, not CI variables. Triangles
run in parallel with rayon (`par_iter`; NUTS chains already share the
global pool, prospicio-bayes/src/nuts.rs:196-197).

Because the study is not in CI, the repo's checks cover it only by
compiling and linting it (`cargo clippy --all-targets`). PR 2 adds a
cheap test that is not ignored, for the pure pieces (the percentile, D
and KS arithmetic against hand-computed cases, and the loader's error
message when the file is absent), which does run in CI.

### 7. Outputs

Per run, under `target/calibration/` (gitignored), nothing in the repo:

* `percentiles.csv`: one row per line, group, arm and basis: percentile,
  mean, sd, outcome, seed, platform, wall time, diagnostics, refit flag,
  error;
* `pp.csv`: per line (and `all`) and arm, the sorted percentiles beside
  `e_i = 100 i / (n + 1)`, the expected percentile on Meyers' horizontal
  axis (p.8), and `f_i = 100 i / n`, the points his D and the
  `136 / sqrt(n)` band are measured from (his footnote 6 calls plotting
  against `e_i` an approximation of that): the table a p-p plot is drawn
  from;
* `ks.csv`: per line and arm, n, the number skipped (decision 5),
  Meyers' D, the two-sided D, the bound, pass or fail, and Meyers'
  published D where one exists;
* the same KS table printed to the test log.

What the test asserts (the user decided, 2026-10-20: record, then
regress):

* **First runs** assert only the harness and the panel: the loader's
  shape (200 triangles, 100 cells each), group 353's square, and Mack's D
  near Meyers' (decision 3). The arms' D values are recorded, not
  asserted.
* **After the user reviews a run**, each arm's D is pinned within a
  tolerance (set from that run, expected near 3 points per line and 1.5
  on all 200, to absorb seed and platform noise), so the knowledge
  finding can be verified. Because the study is manual, "verified" means
  confirmed by the test when run by hand; the finding's frontmatter says
  that rather than claim a CI check.

The comparison table and the D values of a reviewed run are transcribed
by hand into a knowledge finding,
`knowledge/findings/meyers-calibration-panel.md`, listed in
`knowledge/findings/index.md`, with a fragment in `changes/knowledge/`.
The panel's description is `knowledge/datasets/clrd-panel.md` (added in
this PR).

### 8. Comparison with Meyers' published results

Meyers' D, in percentage points, read from the figure text of the PDF
(a star marks a failure at 5%; bounds 19.2 per line, 9.6 on all 200):

| Model | Basis | CA | PA | WC | OL | All 200 | Figure, page |
|---|---|---|---|---|---|---|---|
| Mack | incurred | 16.4 | 16.6 | 27* | 14.2 | 15.4* | 3.2, p.10 |
| Mack | paid | 24.6* | 44.6* | 30.4* | 8.5 | 23.1* | 4.2, p.15 |
| ODP bootstrap | paid | 23.1* | 44.9* | 28.4* | 6.8 | 24.1* | 4.1, p.14 |
| CRC | paid | 22.8* | 41.7* | 30.5* | 25.3* | 25.5* | 5.1, p.19 |
| CRC | incurred | 10.2 | 16.7 | 14.4 | 20.7* | 11.9* | 5.2, p.20 |
| CSR | paid | 5.9 | 18.5 | 12 | 10.4 | 3.1 | 7.4, p.35 |
| CAY | incurred | 8.3 | 15.9 | 12.5 | 20.8* | 10.8* | 8.5, p.42 |

What a match means: Mack's rows are the panel's acceptance test
(decision 3). For ODP, CRC, CSR and CAY the expected agreement is in the
conclusions (pass or fail per line, and the shape of the p-p plot: paid
Mack and ODP percentiles piled low, Mack incurred too light in the tails,
CSR passing on paid), not in the digits, since the simulations differ.
The finding states, arm by arm, whether prospicio reaches Meyers'
conclusion. The comparison is by line and in total only: the appendix zip, which may
hold his per-triangle percentiles, is not downloaded (the user decided).

## Risks

* **The panel differs from Meyers'** in every line (PR 2: 26 of the 175
  printed codes missing, and 25 CA codes uncheckable), so a D differs for
  reasons that are not the model's. Mitigated, not removed, by decision
  3's Mack check, whose tolerance is empirical.
* **Real triangles break fits**: zero or negative paid cells (Meyers
  errors, his code floors), negative increments (the ODP bootstrap,
  `knowledge/findings/odp-negative-increments.md`), non-convergence on
  messy companies, the rate-swap guard and model 2's 87 parameters on a
  55-cell paid triangle (`compartmental-segments-backtest.md`). Mitigated
  by recording every failure and the common-set test (decision 5).
* **Cost overruns**: the times in decision 4 are from clean triangles.
  Mitigated by the PR 2 probe and the line and arm selection of decision 6.
* **Compartmental's presets are WC priors**; a poor D on CA or OL may say
  more about the preset than the model (decision 1).
* **Platform-dependent NUTS draws** make the recorded percentiles
  reproducible only on the platform that made them (decision 5).

## Decisions taken, and what stays open

The user decided these on 2026-10-20 (issue #392):

1. **Data.** The `cas-schedule-p` data package at run time; no CSV in
   the repo; no download of Meyers' appendix zip; the panel is the
   rebuilt screen, diffed against Table A.2's 175 printed codes and
   checked by Mack reproducing Meyers' D (decisions 2 and 3).
2. **Where it runs.** Manual only: an `#[ignore]`d study run by hand,
   like the seed sweeps; no CI job, no new CI variable, no schedule
   (decision 6).
3. **Models.** Staged: PR 2 Mack and the ODP bootstrap; PR 3 `BayesOdp`
   on paid plus Meyers' CRC, CSR and CAY on his bases; PR 4 Compartmental
   model 1, with model 2 as an optional slower arm (decisions 1 and 4).
4. **Assertions.** Record, then regress (decision 7).

Taken as this note's recommendation, **not decided by the user**; say so
if you disagree:

5. **Non-positive cells.** A triangle with a non-positive cell is skipped
   for the arm that errors, and the count is recorded in `ks.csv`; the
   test runs on the common set and on each arm's own set (decision 5).
   Flooring at 1 as Meyers' code does (meyers.rs:114) is a second run
   only if the skips are many.
6. **Compartmental model 2** is an optional arm (about 3.4 hours of fits;
   it runs only when named in `RISK_RS_CALIBRATION_ARMS`). Both
   compartmental models use the WC company 337 presets on every line, so
   their D on CA, PA and OL partly tests the preset (decision 1).

Open: `BayesOdp`'s basis (paid only is planned, beside the ODP
bootstrap; paid and incurred is cheap and can be added in PR 3 on
request).

## As built (PR 2, 2026-10-20)

PR 2 (`claude/calibration-harness`) holds the panel script
`validation/scripts/calibration_panel.py`, the harness and its unit tests
`validation/tests/reserving_calibration_meyers.rs` (the `#[ignore]`d
`calibration_study`, plus fourteen cheap tests that run in a plain `cargo
test`), the Mack (paid and incurred) and ODP bootstrap (paid, 10,000
simulations) arms, and `percentiles.csv`, `pp.csv` and `ks.csv` under
`target/calibration/`. The script also writes `panel_check.csv`, the diff
against Table A.2, one row per code with `status` (`both`, `chosen_only`,
`printed_only`, or `unchecked` for a CA code chosen but not printed),
`rank` (the code's place by increasing CV1 among the triangles that pass
the screen, empty if it fails) and `screen_pass` (how many pass in that
line); the study folds the counts into `ks.csv` as the column
`panel_vs_table_a2` (`both/chosen_only/printed_only/unchecked`; on all 200
it is `149/17/26/34`). It writes `panel_meta.csv` too: the Python,
cas-schedule-p, duckdb and pandas versions and the sha256 of `panel.csv`,
which the study prints in its header. The first run's panel has sha256
`075b392525b26c505c251ea1d94f1641b4c6ab4216ee085655e5ec678fff59d2`
(cas-schedule-p 2026.6.13, duckdb 1.5.6, pandas 3.0.6, Python 3.14.4);
the PR that pins the D values compares a rebuild with it. Differences
from the plan above:

* **The command needs the extra**: `cas-schedule-p[screens]==2026.6.13`
  (duckdb and pandas). The package requires Python 3.11 or newer, so no
  `--python` flag.
* **Panel**: the rebuilt screen has 50 triangles in each line and agrees
  with Table A.2 on 149 of its 175 printed codes. The details are in
  `knowledge/findings/meyers-calibration-panel.md`.
* **The script's exit status** departs from decision 2, which makes only a
  mismatch in PA, WC or OL an error: the script exits with status 2 on any
  printed code it did not choose, CA included (a CA code chosen but not
  printed is not an error). It writes the panel either way. On the
  current package every run exits 2, so the two commands of decision 6
  are run one after the other, not chained with `&&`.
* **No `[[test]]` entry**: decision 6 planned one in
  `validation/Cargo.toml`. The file needs no feature, so Cargo discovers
  it; `validation/Cargo.toml` has only a comment that says so. PR 3, whose
  arms need `slow-bayes`, decides whether an entry is needed then.
* **Group 353** is asserted, as decision 7 plans: the study requires group
  353 commercial auto in the panel (no fallback to the other lines' group
  353, different insurers) and its square equal to
  `validation/data/clrd_comauto_353.csv` cell by cell. A unit test that
  runs in every `cargo test`, `the_arms_reproduce_meyers_on_group_353`,
  scores that square with the arms: Mack gives 72.0 (paid) and 86.1
  (incurred), Meyers' 72.02 and 86.03, and the ODP bootstrap's mean is
  within 1% of Mack's with its percentile near 74 (74.06 at 10,000
  simulations). So the arms' wiring, not only the study, is checked in CI.
* **Mack's acceptance test** asserts Mack's D within 8 points per line and
  4 on all 200 of Meyers' (decision 3), after the outputs are written, and
  only on the whole panel; with `RISK_RS_CALIBRATION_LINES` or `_LIMIT` set
  it is skipped. The tolerance was picked before the run and is empirical
  (decision 3): the run was within 3.7 points per line and 1.5 on all 200,
  although 26 printed triangles are known to be swapped. Why they move D so
  little is not tested: a swapped triangle moves D only if its percentile
  differs from the one it replaces near where the maximum is taken. A later
  package version or panel could fail the assertion for a reason that is
  the panel's, not a bug's; the assertion message says so.
* **The ODP bootstrap's D** is 1.9 to 3.9 points above Meyers' per line and
  2.2 on all 200. That is within what the swapped triangles alone can
  produce (2 points of a line's D each, up to 13 points on all 200 for 26),
  so it is no evidence of a difference in the model. A run on the printed
  codes only (PA, WC and OL, where Table A.2 is complete) would separate
  the two; it is not built.
* **`pp.csv`** holds both `e_i = 100 i / (n + 1)`, Meyers' plotting axis,
  and `f_i = 100 i / n`, the centre of the band as in his D (decision 7).
* **Failures**: a fit that errors or panics is a row with the error text
  (`percentiles.csv`, column `error`) and counts as skipped in `ks.csv`; the
  KS tables are written for each arm's own set and for the common set.
  `diagnostics` and `refit` are empty columns until the NUTS arms (PR 3).
* **Seeds**: FNV-1a of `line|group|arm|basis`, pinned in a unit test.
* **Timing probe**: the whole study (Mack on two bases and the ODP bootstrap,
  600 fits) takes 1.6 s in release on this machine (Windows, parallel over
  the triangles; the ODP's mean fit is 0.13 s under that parallelism and its
  slowest 0.8 s). The probe of the Bayesian models belongs to PR 3 and PR 4,
  whose arms are not built yet; `RISK_RS_CALIBRATION_LIMIT=2` runs the first two
  triangles of each line for it.
* **Not done here**: pinning the arms' D values (after the user reviews the
  run), the Bayesian arms, Compartmental.
