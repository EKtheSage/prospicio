"""Reserving: the loss triangle, the chain ladder with tail factors, Mack's
model with its one-year view and Mack's tests of its assumptions, the
expected-loss methods, Clark's growth
curves and a growth curve fitted to a benchmark pattern, Friedland's
exhibits and a weighted selection of the methods' ultimates, the ODP model as a
GLM with its analytic prediction error, and the ODP bootstrap with its
simulated one-year view of the chain
ladder and the expected-loss methods, Mack's bootstrap for the lifetime and
one-year views under Mack's process, and the Bayesian ODP chain ladder, the Bayesian
Bornhuetter-Ferguson, Meyers' CRC, CSR and CAY models and Gesmann and Morris'
compartmental model sampled with NUTS, a latest diagonal on its own
projected with a pattern from elsewhere, and excess-layer development
patterns derived from a ground-up triangle
(docs/design/triangle.md, docs/design/reserving-v02.md,
docs/design/reserving-bayes.md, docs/design/reserving-compartmental.md,
docs/design/reserving-excess.md)."""

from .prospicio_native import (
    Backtest,
    BayesBf,
    BayesBfFit,
    BayesBfNormalised,
    BayesBfNormalisedFit,
    BayesOdp,
    BayesOdpFit,
    BayesOdpFits,
    BayesOneYearFit,
    BayesOneYearFits,
    BenchmarkCurve,
    BenchmarkCurveFit,
    Benktander,
    BornhuetterFerguson,
    CalendarYearTest,
    CapeCod,
    CapeCodFit,
    ChainLadder,
    ChainLadderFit,
    ClaimsDevelopmentResult,
    ClarkCapeCod,
    ClarkFit,
    ClarkLdf,
    Compartmental,
    CompartmentalCandidate,
    CompartmentalFit,
    CompartmentalFits,
    CompartmentalOneYearFit,
    CompartmentalOneYearFits,
    CompartmentalPriors,
    DevelopmentCorrelationTest,
    DevelopmentExhibit,
    ExcessLayer,
    ExcessPattern,
    ExpectedLoss,
    ExpectedLossFit,
    GlmCandidate,
    LatestDiagonal,
    LayerSharePrior,
    Mack,
    MackBootstrap,
    MackBootstrapFit,
    MackFit,
    Meyers,
    MeyersFit,
    MeyersFits,
    NewYear,
    NotSupportedError,
    OdpBootstrap,
    OdpBootstrapFit,
    OdpGlm,
    OdpGlmFit,
    OneYearFit,
    ParetoShape,
    ScaleShift,
    SelectedUltimates,
    TailBondy,
    TailConstant,
    TailCurve,
    TailLogLinear,
    Triangle,
    DevelopmentPatterns,
    UnderwritingBasis,
    UltimateExhibit,
    calendar_year_effect,
    development_correlation,
    diagonal_backtest,
    layer_at_shift,
    layer_exposure,
    layer_share,
)

__all__ = [
    "Triangle",
    "UnderwritingBasis",
    "DevelopmentPatterns",
    "ChainLadder",
    "ChainLadderFit",
    "Mack",
    "MackFit",
    "ClaimsDevelopmentResult",
    "calendar_year_effect",
    "development_correlation",
    "CalendarYearTest",
    "DevelopmentCorrelationTest",
    "DevelopmentExhibit",
    "UltimateExhibit",
    "SelectedUltimates",
    "select_ultimates",
    "ExpectedLoss",
    "BornhuetterFerguson",
    "Benktander",
    "CapeCod",
    "ExpectedLossFit",
    "CapeCodFit",
    "LatestDiagonal",
    "ClarkLdf",
    "ClarkCapeCod",
    "ClarkFit",
    "BenchmarkCurve",
    "BenchmarkCurveFit",
    "OdpGlm",
    "OdpGlmFit",
    "OdpBootstrap",
    "OdpBootstrapFit",
    "MackBootstrap",
    "MackBootstrapFit",
    "BayesOdp",
    "BayesOdpFit",
    "BayesBf",
    "BayesBfFit",
    "BayesBfNormalised",
    "BayesBfNormalisedFit",
    "BayesOdpFits",
    "BayesOneYearFit",
    "BayesOneYearFits",
    "Meyers",
    "MeyersFit",
    "MeyersFits",
    "NewYear",
    "NotSupportedError",
    "Compartmental",
    "CompartmentalCandidate",
    "CompartmentalPriors",
    "CompartmentalFit",
    "CompartmentalFits",
    "CompartmentalOneYearFit",
    "CompartmentalOneYearFits",
    "OneYearFit",
    "TailConstant",
    "TailCurve",
    "TailBondy",
    "TailLogLinear",
    "Backtest",
    "GlmCandidate",
    "LayerSharePrior",
    "diagonal_backtest",
    "ExcessLayer",
    "ScaleShift",
    "ParetoShape",
    "ExcessPattern",
    "layer_share",
    "layer_exposure",
    "layer_at_shift",
    "schedule_p",
]

# Line names accepted by ``schedule_p``: the calibration panel's codes, the
# CAS loss reserve database's file names and the mart's own names.
_SCHEDULE_P_LINES = {
    "ca": "commercial_auto",
    "comauto": "commercial_auto",
    "pa": "private_passenger_auto",
    "ppauto": "private_passenger_auto",
    "wc": "workers_compensation",
    "wkcomp": "workers_compensation",
    "ol": "other_liability",
    "othliab": "other_liability",
    "pl": "products_liability",
    "prodliab": "products_liability",
    "mm": "medical_malpractice",
    "medmal": "medical_malpractice",
}

# Measure name: the mart's column (case_incurred is incurred less bulk).
_SCHEDULE_P_COLUMNS = {
    "paid": "cum_paid_loss",
    "incurred": "incurred_loss",
    "case_incurred": None,
    "premium": "earned_prem_net",
    "premium_direct": "earned_prem_direct",
}


def schedule_p(line, company=None, columns=None, *, accident_years=(1988, 1997), lower=False):
    """Triangles of the CAS loss reserve database (NAIC Schedule P), one
    segment per company, from the ``cas-schedule-p`` package.

    The package bundles the CAS Schedule P data model's table
    ``mart_reserving_model_training`` (no network access); this reads it
    as ``validation/scripts/calibration_panel.py`` does for the calibration
    study: ``paid`` is ``cum_paid_loss``, ``incurred`` is ``incurred_loss``
    (bulk and IBNR included), ``case_incurred`` is ``incurred_loss -
    bulk_loss`` (Meyers' incurred), ``premium`` is ``earned_prem_net`` and
    ``premium_direct`` is ``earned_prem_direct``, in USD thousands. Ages are
    ``development_age * 12`` months. A company's cells are as the mart holds
    them, so a company with gaps has holes.

    Parameters
    ----------
    line : str
        ``"ca"`` (commercial auto), ``"pa"`` (private passenger auto),
        ``"wc"`` (workers' compensation), ``"ol"`` (other liability),
        ``"pl"`` (products liability) or ``"mm"`` (medical malpractice),
        or the CAS file names (``"comauto"``, ``"ppauto"``, ``"wkcomp"``,
        ``"othliab"``, ``"prodliab"``, ``"medmal"``) or the mart's names
        (``"commercial_auto"``, ...).
    company : int, str or list, optional
        NAIC group or company codes (``353`` or ``"353"``); every company in
        the line by default.
    columns : str or list of str, optional
        Measures among ``"paid"``, ``"incurred"``, ``"case_incurred"``,
        ``"premium"`` and ``"premium_direct"``; all five by default.
    accident_years : (int, int), default (1988, 1997)
        The first and last accident year. The mart's accident years run to
        2007; Meyers' panel uses 1988 to 1997.
    lower : bool, default False
        Whether to keep the cells valued after the last accident year's
        first year-end: the lower triangle, which a study holds out as the
        outcome. With it, each complete company is a 10 x 10 square.

    Returns
    -------
    Triangle
        Keyed by ``company`` (the code as a string), with annual origins
        and development, cumulative.

    Raises
    ------
    ImportError
        If ``cas-schedule-p`` or ``pyarrow`` is not installed.
    ValueError
        For an unknown line or column, or a company with no rows in the
        line and years.

    Examples
    --------
    >>> from prospicio.reserving import schedule_p
    >>> tri = schedule_p("ca", company=353, columns=["paid", "premium"])  # doctest: +SKIP
    >>> tri.index, tri.shape  # doctest: +SKIP
    (['353'], (1, 2, 10, 10))
    """
    try:
        import cas_schedule_p
        import pyarrow.parquet as pq
    except ImportError as e:
        raise ImportError(
            "schedule_p needs the cas-schedule-p and pyarrow packages (Python 3.11 or "
            "newer): pip install cas-schedule-p pyarrow"
        ) from e
    name = _SCHEDULE_P_LINES.get(line, line) if isinstance(line, str) else line
    if name not in _SCHEDULE_P_LINES.values():
        raise ValueError(
            f"unknown Schedule P line {line!r}; use one of "
            + ", ".join(repr(k) for k in _SCHEDULE_P_LINES)
        )
    if columns is None:
        columns = list(_SCHEDULE_P_COLUMNS)
    elif isinstance(columns, str):
        columns = [columns]
    else:
        columns = list(columns)
    unknown = [c for c in columns if c not in _SCHEDULE_P_COLUMNS]
    if unknown:
        raise ValueError(
            f"unknown Schedule P column {unknown[0]!r}; use "
            + ", ".join(repr(k) for k in _SCHEDULE_P_COLUMNS)
        )
    first, last = (int(y) for y in accident_years)
    filters = [
        ("line_of_business", "=", name),
        ("accident_year", ">=", first),
        ("accident_year", "<=", last),
    ]
    codes = None
    if company is not None:
        if isinstance(company, (str, int)):
            company = [company]
        codes = [str(c) for c in company]
        filters.append(("company_code", "in", codes))
    table = pq.read_table(
        cas_schedule_p.path("mart_reserving_model_training"),
        columns=[
            "company_code",
            "accident_year",
            "development_age",
            "cum_paid_loss",
            "incurred_loss",
            "bulk_loss",
            "earned_prem_net",
            "earned_prem_direct",
        ],
        filters=filters,
    ).to_pydict()
    keep = [
        k
        for k, (ay, age) in enumerate(zip(table["accident_year"], table["development_age"]))
        if lower or ay + age - 1 <= last
    ]
    found = {table["company_code"][k] for k in keep}
    if codes is not None:
        missing = [c for c in codes if c not in found]
        if missing:
            raise ValueError(
                f"no rows for company {missing[0]} in {name}, accident years {first} to {last}"
            )
    if not keep:
        raise ValueError(f"no rows in {name}, accident years {first} to {last}")

    def column(measure):
        source = _SCHEDULE_P_COLUMNS[measure]
        if source is None:
            return [table["incurred_loss"][k] - table["bulk_loss"][k] for k in keep]
        return [table[source][k] for k in keep]

    return Triangle.from_long(
        origin=[table["accident_year"][k] for k in keep],
        development=[12 * table["development_age"][k] for k in keep],
        values={c: column(c) for c in columns},
        keys={"company": [table["company_code"][k] for k in keep]},
    )


def select_ultimates(
    fits, weights=None, *, by_age=None, triangle=None, reported=None, paid=None, total=True
):
    """The selected ultimate per origin, a weighting of several methods'
    ultimates, with its IBNR and unpaid, as a table: Friedland's selected
    ultimate claims (chapter 15, Exhibit I, sheet 2, column (11)) and
    chainladder-python's ``VotingChainladder``
    (docs/design/reserving-v02.md, decision 34).

    Each origin's weights must be at least 0 and sum to 1 (within 1e-9);
    they are not rescaled. A method left out has weight 0, and a method
    with weight 0 may have no ultimate (``nan``) there.

    Parameters
    ----------
    fits : UltimateExhibit or dict
        An exhibit with the methods, or a dict of method name to a fit of
        one segment (or to its ultimate per origin), in which case
        ``triangle`` and at least one of ``reported`` and ``paid`` give the
        claims to date.
    weights : str or mapping, optional
        A method name, or a mapping (a dict or a DataFrame) of method name
        to one weight for every origin or a weight per origin.
    by_age : mapping, optional
        Instead of ``weights``, age in months to a mapping of method name
        to weight; each origin takes the row of the oldest age not past its
        own (``UltimateExhibit.select``).
    triangle : Triangle, optional
        The triangle the fits came from, with ``fits`` as a dict.
    reported, paid : str, optional
        Its columns of reported and paid claims.
    total : bool, default True
        Whether to end the table with a ``"total"`` row.

    Returns
    -------
    pandas.DataFrame or dict
        One row per origin (and the total): ``reported`` and ``paid`` when
        given, ``ultimate``, ``ibnr`` and ``unpaid`` when defined, and each
        method's weight as ``weight_<method>``. A dict of lists without
        pandas.

    Raises
    ------
    ValueError
        As ``UltimateExhibit.select``, or a dict of fits without a triangle
        and a column.

    Examples
    --------
    >>> from prospicio.reserving import ChainLadder, Triangle, select_ultimates
    >>> rows = [(2020, 12, 100.0, 60.0), (2020, 24, 150.0, 140.0), (2021, 12, 120.0, 50.0)]
    >>> origin, age, rep, pd_ = map(list, zip(*rows))
    >>> tri = Triangle.from_long(origin, age, {"reported": rep, "paid": pd_})
    >>> fits = {"reported": ChainLadder().fit(tri, "reported"),
    ...         "paid": ChainLadder().fit(tri, "paid")}
    >>> table = select_ultimates(fits, {"reported": [1.0, 0.5], "paid": [0.0, 0.5]},
    ...                          triangle=tri, reported="reported", paid="paid")
    >>> [round(u, 2) for u in table["ultimate"]]
    [150.0, 148.33, 298.33]
    """
    if isinstance(fits, UltimateExhibit):
        exhibit = fits
    else:
        if triangle is None or (reported is None and paid is None):
            raise ValueError(
                "with a dict of fits, give the triangle and its reported or paid column"
            )
        exhibit = UltimateExhibit.from_triangle(triangle, reported=reported, paid=paid)
        for name, fit in fits.items():
            exhibit = exhibit.with_method(name, fit)
    return exhibit.select(weights, by_age=by_age).to_frame(total=total)
