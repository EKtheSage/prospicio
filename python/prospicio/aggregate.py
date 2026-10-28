"""Aggregate loss: compound distributions by Panjer and FFT, and simulated
events, the occurrence exceedance (OEP) curve, compound moments to third
order with the approximations fitted to them, eventual ruin by the
Pollaczeck–Khinchine formula and, for renewal claims, by Wiener–Hopf
(wiener_hopf), ruin within a finite horizon (finite_horizon_ruin), a severity reshaped to layer loss picks, and a catastrophe model's
event loss table with secondary uncertainty (EventLossTable), and two lines hit by
the same events: their joint loss by a 2-D FFT (fft_bivariate, JointGrid) and the
clash model (Clash), a vendor's year loss table read and expanded to a run's
simulations (read_ylt, EventSet.from_ylt), and the P&L ledger (Ledger, Group, Leg, and stack to set
ledgers' results side by side; Book, Expense and derive_premium build the ledger of
a book and its annual covers, and occurrence_source the joint gross and ceded loss
its occurrence covers are walked over; pnl_program prices a book and its covers from
target combined ratios), and a term of accident months shared by its underwriting
years (Timeline, Exposure). Layers and towers are in
``prospicio.reinsurance``."""

from .prospicio_native import (
    Approximation,
    BivariateReport,
    Book,
    Clash,
    CompoundMoments,
    CompoundReport,
    EventLossTable,
    EventSet,
    Expense,
    Exposure,
    FiniteRuin,
    GridAudit,
    GridSize,
    Group,
    JointGrid,
    Ledger,
    Leg,
    OccurrencePml,
    Picks,
    RenewalRuin,
    Ruin,
    Timeline,
    audit_grid,
    audit_portfolio_grid,
    compound_moments,
    derive_premium,
    eventual_ruin,
    fft,
    fft_auto,
    fft_bivariate,
    finite_horizon_ruin,
    layer_loss_picks,
    occurrence_exceedance,
    occurrence_pml,
    occurrence_source,
    panjer,
    pnl_program,
    pnl_program_stated,
    recommend_grid,
    recommend_portfolio_grid,
    round_bucket,
    round_premium,
    simulate_events,
    size_premium,
    stack,
    wiener_hopf,
    year_correlation_for_ratio,
    ylt_hash,
)

__all__ = [
    "panjer",
    "fft",
    "fft_auto",
    "recommend_grid",
    "recommend_portfolio_grid",
    "round_bucket",
    "GridSize",
    "audit_grid",
    "audit_portfolio_grid",
    "GridAudit",
    "CompoundReport",
    "simulate_events",
    "EventSet",
    "EventLossTable",
    "Timeline",
    "Exposure",
    "read_ylt",
    "year_correlation_for_ratio",
    "ylt_hash",
    "occurrence_exceedance",
    "occurrence_pml",
    "OccurrencePml",
    "compound_moments",
    "CompoundMoments",
    "Approximation",
    "eventual_ruin",
    "Ruin",
    "finite_horizon_ruin",
    "FiniteRuin",
    "wiener_hopf",
    "RenewalRuin",
    "layer_loss_picks",
    "Picks",
    "fft_bivariate",
    "JointGrid",
    "BivariateReport",
    "Clash",
    "Ledger",
    "Group",
    "Leg",
    "stack",
    "Book",
    "Expense",
    "derive_premium",
    "occurrence_source",
    "pnl_program",
    "pnl_program_stated",
    "size_premium",
    "round_premium",
]


def _whole(text, column, row):
    """A whole number from a CSV cell, written ``3`` or ``3.0``."""
    try:
        return int(text)
    except ValueError:
        pass
    try:
        value = float(text)
    except ValueError:
        value = float("nan")
    if not value.is_integer():
        raise ValueError(f"row {row}: {column} {text!r} is not a whole number")
    return int(value)


def read_ylt(
    path,
    *,
    year="year",
    loss="value",
    slot="slot",
    labels=None,
    n_years=None,
    n_sims=None,
    expand="tile",
    seed=0,
    unique_by=None,
    expected=None,
    file_format=None,
):
    """Reads a year loss table from a CSV or Parquet file into years of
    events, checked and expanded to a run's simulations.

    The file has one row per non-zero event loss. Its rows are checked
    (losses finite and non-negative, years from 1, no repeated year and
    slot within the ``unique_by`` labels, labels among the ``expected``
    values), ordered by year then slot (the file's order breaking ties),
    and expanded as ``EventSet.from_ylt`` describes. The file's bytes are
    hashed (``ylt_hash``) into the events' ``input_hash``, which results
    keep in their provenance. Rows sharing a slot, one event hitting
    several classes, stay separate losses.

    Parameters
    ----------
    path : str or path-like
    year, loss : str, default "year", "value"
        The columns of each row's year, from 1, and its loss.
    slot : str or None, default "slot"
        The column of each row's event slot, or ``None`` to keep the file's
        order within a year and skip the check for repeats.
    labels : list of str, optional
        Columns kept as labels, ``["peril", "class"]`` say, by which
        ``Tower.apply`` splits its results.
    n_years : int, optional
        The table's years, at least its largest year; by default the
        largest year.
    n_sims : int, optional
        The simulations to expand to, by default ``n_years``.
    expand : {"tile", "resample"}, default "tile"
    seed : int, default 0
        The seed of ``"resample"``.
    unique_by : list of str, optional
        With slots, no two rows may share a year, a slot and their values
        in these labels (``["peril"]``); by default every label.
    expected : dict of str to list of str, optional
        The values allowed in each named label (``{"peril": ["WS", "EQ"]}``).
    file_format : {"csv", "parquet"}, optional
        By default Parquet for a ``.parquet`` or ``.pq`` file and CSV
        otherwise. Parquet needs ``pyarrow``.

    Returns
    -------
    EventSet

    Examples
    --------
    >>> import os, tempfile
    >>> from prospicio.aggregate import read_ylt
    >>> path = os.path.join(tempfile.mkdtemp(), "ylt.csv")
    >>> _ = open(path, "w").write("year,peril,slot,value\\n2,wind,9,5\\n1,quake,1,1\\n2,wind,3,7\\n")
    >>> e = read_ylt(path, labels=["peril"], n_sims=4)
    >>> e.events(1), e.labels(1)
    ([7.0, 5.0], {'peril': ['wind', 'wind']})
    >>> e.input_hash.startswith("blake3:")
    True
    """
    import io
    from pathlib import Path

    path = Path(path)
    data = path.read_bytes()
    labels = list(labels or [])
    columns = [year, loss] + ([slot] if slot is not None else []) + labels
    if file_format is None:
        file_format = "parquet" if path.suffix.lower() in (".parquet", ".pq") else "csv"
    if file_format == "parquet":
        try:
            import pyarrow.parquet as pq
        except ImportError as e:
            raise ImportError("reading a Parquet year loss table needs pyarrow: pip install pyarrow") from e
        table = pq.read_table(io.BytesIO(data))
        missing = [c for c in columns if c not in table.column_names]
        if missing:
            raise ValueError(f"{path}: no column {missing[0]!r}")
        cols = table.select(columns).to_pydict()
        years = [int(y) for y in cols[year]]
        slots = [int(s) for s in cols[slot]] if slot is not None else None
        losses = [float(x) for x in cols[loss]]
        label_values = {d: [str(v) for v in cols[d]] for d in labels}
    elif file_format == "csv":
        import csv

        reader = csv.DictReader(io.StringIO(data.decode("utf-8-sig")))
        missing = [c for c in columns if c not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{path}: no column {missing[0]!r}")
        years, slots, losses = [], [] if slot is not None else None, []
        label_values = {d: [] for d in labels}
        for row, r in enumerate(reader, start=1):
            years.append(_whole(r[year], year, row))
            if slot is not None:
                slots.append(_whole(r[slot], slot, row))
            losses.append(float(r[loss]))
            for d in labels:
                label_values[d].append(r[d])
    else:
        raise ValueError(f'file_format must be "csv" or "parquet", not {file_format!r}')
    return EventSet.from_ylt(
        years,
        losses,
        slots=slots,
        labels=label_values or None,
        n_years=n_years,
        n_sims=n_sims,
        expand=expand,
        seed=seed,
        unique_by=unique_by,
        expected=expected,
        source_hash=ylt_hash(data),
    )
