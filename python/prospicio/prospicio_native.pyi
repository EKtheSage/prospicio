from collections.abc import Sequence
from datetime import date
from typing import Any, final

@final
class AggregateLayerStatistics:
    """
    An aggregate cover's statistics, from ``aggregate_layer_statistics``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def aggregate(self, /) -> CompoundMoments:
        """
        The annual ceded loss, at the share.
        """
    @property
    def attachment(self, /) -> float:
        """
        The aggregate retention.
        """
    @property
    def limit(self, /) -> float:
        """
        The aggregate limit.
        """
    @property
    def loss_on_line(self, /) -> float:
        """
        The ceded mean over ``share * limit``; 0 for an unlimited cover.
        """
    @property
    def pr_attach(self, /) -> float:
        """
        ``P(S > attachment)``: the chance of a loss to the cover.
        """
    @property
    def pr_detach(self, /) -> float:
        """
        ``P(S >= attachment + limit)``; 0 for an unlimited cover.
        """
    @property
    def share(self, /) -> float:
        """
        The placed share.
        """

@final
class Allocation:
    """
    Capital allocation of a distortion risk measure, from ``capital``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def allocated(self, /) -> list[float]:
        """
        Allocated capital of each component.
        """
    def diversification(self, /) -> list[float]:
        """
        ``standalone - allocated`` per component: each one's share of the
        diversification benefit.
        
        Returns
        -------
        list of float
        """
    def diversification_benefit(self, /) -> float:
        """
        ``sum(standalone) - total``: the capital saved by holding the
        components together.
        
        Returns
        -------
        float
        """
    @property
    def method(self, /) -> str:
        """
        The allocation method, as passed to ``capital``.
        """
    @property
    def standalone(self, /) -> list[float]:
        """
        Stand-alone measure ``rho(X_j)`` of each component.
        """
    @property
    def total(self, /) -> float:
        """
        The portfolio's measure, ``rho(S)``.
        """

@final
class AnnualGrids:
    """
    A tower applied to an annual loss distribution, from
    ``Tower.on_annual_grid``. Ceded grids are at the placed share and after
    annual terms; a layer with share ``c`` has step ``c * h``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def ceded(self, /) -> list[Grid]:
        """
        Annual ceded loss of each layer, in tower order.
        """
    @property
    def expected_ceding_commission(self, /) -> list[float]:
        """
        Expected ceding commission (flat or sliding) of each layer.
        """
    @property
    def expected_premium(self, /) -> list[float]:
        """
        Expected premium of each layer: the swing-rated premium's mean, or
        else the fixed premium.
        """
    @property
    def expected_profit_commission(self, /) -> list[float]:
        """
        Expected profit commission of each layer.
        """
    @property
    def expected_reinstatement_premium(self, /) -> list[float]:
        """
        Expected reinstatement premium of each layer.
        """
    @property
    def gross(self, /) -> Grid:
        """
        The annual loss given.
        """
    @property
    def net(self, /) -> Grid:
        """
        Annual net loss.
        """
    @property
    def on_points(self, /) -> bool:
        """
        Whether every ceded and net loss fell on a grid point. The
        expectations are exact either way.
        """

@final
class Approximation:
    """
    A distribution fitted to compound moments, from
    ``CompoundMoments.approximate``.
    """
    def __repr__(self, /) -> str: ...
    def cdf(self, /, x: float) -> float:
        """
        ``P(S <= x)``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @property
    def kind(self, /) -> str:
        """
        The family: ``"normal"``, ``"gamma"``, ``"lognormal"``,
        ``"shifted_gamma"`` or ``"shifted_lognormal"``.
        """
    def mean(self, /) -> float:
        """
        The mean.
        
        Returns
        -------
        float
        """
    @property
    def parameters(self, /) -> list[tuple[str, float]]:
        """
        The parameters by name: ``mean`` and ``sd``; ``shape`` and
        ``scale``; ``meanlog`` and ``sdlog``; with ``shift`` for the
        shifted families.
        """
    def quantile(self, /, p: float) -> float:
        """
        The ``p`` quantile.
        
        Parameters
        ----------
        p : float
        
        Returns
        -------
        float
        """

@final
class ArchimedeanCopula:
    """
    An exchangeable Archimedean copula: Clayton, Gumbel, Frank or Joe.
    
    Parameters
    ----------
    family : {"clayton", "gumbel", "frank", "joe"}
    theta : float
        Positive for Clayton and Frank; at least 1 for Gumbel and Joe.
    dim : int
    
    Raises
    ------
    ValueError
        If the family is unknown or ``theta`` is out of range.
    
    Examples
    --------
    >>> from prospicio.risk import ArchimedeanCopula
    >>> c = ArchimedeanCopula("clayton", 2.0, 3)  # Kendall's tau 0.5
    >>> c.dim, c.family
    (3, 'clayton')
    """
    def __new__(cls, /, family: str, theta: float, dim: int) -> ArchimedeanCopula: ...
    def cdf(self, /, u: Sequence[float]) -> float:
        """
        The copula's distribution function at the point ``u``, each
        coordinate in ``[0, 1]``.
        
        Parameters
        ----------
        u : list of float
            One value per dimension.
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``u`` has the wrong length or a value outside ``[0, 1]``,
            or the copula has no closed-form distribution function here.
        
        Examples
        --------
        >>> from prospicio.risk import ArchimedeanCopula
        >>> c = ArchimedeanCopula("clayton", 2.0, 2)
        >>> round(c.cdf([0.3, 0.6]), 12) == round((0.3**-2 + 0.6**-2 - 1) ** -0.5, 12)
        True
        """
    @property
    def dim(self, /) -> int:
        """
        Number of dimensions.
        """
    @property
    def family(self, /) -> str:
        """
        Family name.
        """
    def sample(self, /, n: int, seed: int) -> list[list[float]]:
        """
        ``n`` draws of uniforms; draw ``i`` uses stream ``i`` of ``seed``.
        
        Parameters
        ----------
        n : int
        seed : int
        
        Returns
        -------
        list of list of float
        """
    @property
    def theta(self, /) -> float:
        """
        Copula parameter.
        """

@final
class AsIf:
    """
    Each year's cessions under its own program and under the proposed one,
    from ``PreparedLosses.as_if``.
    
    Examples
    --------
    >>> from prospicio.pricing import LargeLossExperience
    >>> from prospicio.reinsurance import Layer, Tower
    >>> p = LargeLossExperience(["a"], [2019.5], [300.0]).prepare(factors=[1.2])
    >>> then = Tower([Layer("150 xs 150", 150.0, 150.0)])
    >>> now = Tower([Layer("200 xs 200", 200.0, 200.0)])
    >>> a = p.as_if([then], [2019], now)
    >>> a.to_dict()["historical"], [round(x, 6) for x in a.to_dict()["proposed"][0]]
    ([[150.0]], [160.0])
    """
    def __repr__(self, /) -> str: ...
    @property
    def proposed_layers(self, /) -> list[str]:
        """
        The proposed tower's layer names.
        """
    def to_dict(self, /) -> dict:
        """
        The years as columns: ``year``, ``reported_losses``,
        ``trended_losses``, and per year the lists ``historical_layers``,
        ``historical`` (one value per layer in force that year) and
        ``proposed`` (one per proposed layer).
        
        Returns
        -------
        dict of str to list
        """

@final
class BayesBf:
    """
    The Bayesian Bornhuetter–Ferguson of England, Verrall and Wüthrich
    (2012, Model 3.1): the over-dispersed Poisson model of a cumulative
    triangle's incremental values ``X`` with mean ``mu[w] * gamma[d]``, an
    informative gamma prior on each origin's row parameter ``mu[w]`` with
    mean ``apriori * exposure[w]`` and coefficient of variation ``cv``, a
    vague normal prior on each ``log gamma[d]`` (the pattern is not
    normalised), and the
    chain ladder's Pearson scale ``phi`` (``OdpBootstrapFit.scale``) plugged
    in. The posterior is sampled with NUTS, and each draw projects the
    future cells up to the oldest observed age (no tail) with the process
    error of ``process``, as ``BayesOdp`` does.
    
    The prior's strength moves the reserve between two limits. With
    ``cv=float("inf")`` (a flat prior on ``log mu[w]``) the model is the
    Bayesian chain ladder, and the reserves agree with ``BayesOdp``'s within
    Monte Carlo error. As ``cv`` falls to 0, ``mu[w]`` is fixed at the prior
    ultimate ``m[w]`` and each origin's mean reserve tends to ``m[w]`` times
    the sum of the future ``S[d] / M[d]`` (``S[d]`` the observed increments
    of column ``d``, ``M[d]`` the prior ultimates of the origins observed
    there): EVW's strong-prior limit, Mack's (2008) Bornhuetter–Ferguson
    with the raw pattern. That equals the classical
    ``BornhuetterFerguson`` reserve ``(1 - 1 / cdf) * m[w]`` only when the
    prior ultimates are the chain ladder's.
    
    ``mu[w]`` is EVW's row parameter, not the origin's expected ultimate:
    that is ``mu[w]`` times the sum of the ``gamma[d]``, which under a tight
    prior is ``m[w]`` times the sum of ``S[d] / M[d]`` over all columns. On
    EVW's triangle that sum is 0.885, so the implied ultimate loss ratio is
    about 11.5% below ``apriori``.
    
    Every development column's observed increments must sum to more than
    zero, and every origin's to more than ``-phi / cv**2``; negative cells
    are allowed. One ``seed`` replays a fit exactly, whatever the number of
    threads. Failed MCMC diagnostics (R-hat above 1.01, bulk or tail ESS
    below 100 per chain, any divergent transition) do not fail the fit:
    ``fit`` raises one ``UserWarning`` listing them, and they stay in
    ``BayesBfFit.diagnostics``.
    
    Parameters
    ----------
    apriori : float, default 1.0
        Prior mean of each origin's ``mu[w]`` per unit of exposure;
        positive and finite. It is the expected loss ratio only when the
        pattern ``gamma`` sums to 1 (see above).
    cv : float, default 0.1
        Coefficient of variation of the gamma prior on each ``mu[w]`` (shape
        ``1 / cv**2``, EVW's ``a``); positive, ``float("inf")`` for a flat
        prior.
    development_sd : float, default 10.0
        Standard deviation of the normal prior on each ``log gamma[d]``,
        centred on 0. With a flat ``cv``, a much vaguer prior (1,000, say)
        leaves divergent transitions.
    process : {"gamma", "none"}, default "gamma"
        Process error on each simulated future increment: Gamma with mean
        ``mu`` and variance ``phi * mu``, or none for parameter error only.
    chains, tune, draws : int, default 4, 1000, 1000
        NUTS chains, warm-up draws per chain and kept draws per chain.
    n_sims : int, default 10000
        Number of predictive simulations, independent of the number of
        posterior draws; positive.
    seed : int, default 0
        Seed of the chains and of the simulations, from 0 to ``2**64 - 1``.
    target_accept : float, default 0.95
        Target mean acceptance rate of the step-size adaptation.
    max_depth : int, default 10
        Largest NUTS tree depth.
    
    Raises
    ------
    ValueError
        If ``n_sims`` is zero or ``process`` is unknown.
    OverflowError
        If an integer setting is negative or too large.
    
    Examples
    --------
    >>> from prospicio.reserving import BayesBf, Triangle
    >>> tri = Triangle.from_long(
    ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
    ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
    ...     {
    ...         "paid": [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
    ...         "premium": [200.0] * 4 + [220.0] * 3 + [240.0] * 2 + [260.0],
    ...     },
    ... )
    >>> fit = BayesBf(apriori=0.85, cv=0.1, n_sims=2000, seed=1).fit(tri, "paid", "premium")
    >>> fit.reserves.components()
    [('2020',), ('2021',), ('2022',), ('2023',)]
    >>> fit.prior_ultimate[3] == 0.85 * 260.0
    True
    >>> fit.parameter_names[:2]
    ['alpha[1]', 'alpha[2]']
    """
    def __new__(cls, /, apriori: float = 1.0, cv: float = 0.1, development_sd: float = 10.0, process: str = "gamma", chains: int = 4, tune: int = 1000, draws: int = 1000, n_sims: int = 10000, seed: int = 0, target_accept: float = 0.95, max_depth: int = 10) -> BayesBf: ...
    def __repr__(self, /) -> str: ...
    @property
    def apriori(self, /) -> float:
        """
        Prior mean of each origin's ``mu[w]`` per unit of exposure.
        """
    @property
    def chains(self, /) -> int:
        """
        Number of NUTS chains.
        """
    @property
    def cv(self, /) -> float:
        """
        Coefficient of variation of each origin's prior; ``inf`` is flat.
        """
    @property
    def development_sd(self, /) -> float:
        """
        Standard deviation of each development effect's prior.
        """
    @property
    def draws(self, /) -> int:
        """
        Kept draws per chain.
        """
    def fit(self, /, triangle: Triangle, column: str, exposure: str) -> BayesBfFit:
        """
        Samples the posterior of one loss column of a single-segment
        cumulative triangle, with each origin's exposure from another
        column, and simulates its reserves. Every origin must be observed
        from the first age up to its latest and there must be more observed
        cells than chain-ladder parameters. Raises one ``UserWarning``
        listing the failed diagnostics, if any.
        
        Parameters
        ----------
        triangle : Triangle
            Cumulative, with one segment.
        column : str
            The losses to project.
        exposure : str
            The exposure column; each origin's latest observed cumulative
            value is its exposure, as for ``BornhuetterFerguson``.
        
        Returns
        -------
        BayesBfFit
        
        Raises
        ------
        ValueError
            As ``OdpBootstrap.fit``; if the triangle has several segments; if
            an origin has no observed, finite, positive exposure; if
            ``apriori``, ``cv`` or ``development_sd`` is invalid or a sampler
            setting is; if a development column's increments do not sum to
            more than zero; or if an origin's sum to ``-phi / cv**2`` or
            less.
        """
    @property
    def max_depth(self, /) -> int:
        """
        Largest NUTS tree depth.
        """
    @property
    def n_sims(self, /) -> int:
        """
        Number of predictive simulations.
        """
    @property
    def process(self, /) -> str:
        """
        Process error: ``"gamma"`` or ``"none"``.
        """
    @property
    def seed(self, /) -> int:
        """
        Seed of the chains and of the simulations.
        """
    @property
    def target_accept(self, /) -> float:
        """
        Target mean acceptance rate.
        """
    @property
    def tune(self, /) -> int:
        """
        Warm-up draws per chain.
        """

@final
class BayesBfFit:
    """
    A fitted ``BayesBf``: the posterior of ``alpha[1..n]`` (``log mu[w]``)
    and ``beta[1..m]`` (``log gamma[d]``), its MCMC diagnostics, and the
    posterior predictive distribution of the reserves, laid out as
    ``BayesOdpFit.reserves``. The frames' ``latest``, ``ultimate`` and
    ``reserve`` are the volume-weighted chain ladder's (``chain_ladder``),
    with each origin's ``exposure`` and ``prior_ultimate``; ``mean`` and
    ``std_dev`` are the posterior predictive reserve's.
    """
    def __repr__(self, /) -> str: ...
    @property
    def chain_ladder(self, /) -> ChainLadderFit:
        """
        The volume-weighted chain ladder of the triangle, whose latest
        values and Pearson scale the model uses.
        """
    @property
    def chains(self, /) -> int:
        """
        Number of chains.
        """
    @property
    def development(self, /) -> list[int]:
        """
        Development ages in months.
        """
    def development_frame(self, /) -> Any:
        """
        The chain ladder's development factors, one row per age, as
        ``ChainLadderFit.development_frame``. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def diagnostics(self, /) -> dict:
        """
        MCMC diagnostics: a dict with ``max_rhat``, ``min_ess_bulk``,
        ``min_ess_tail``, ``divergences`` and ``warnings`` (one message per
        failed threshold; empty when every threshold holds).
        """
    @property
    def draws(self, /) -> list[list[float]]:
        """
        Posterior draws, one row per draw (chain by chain), one column per
        parameter in the order of ``parameter_names``.
        """
    @property
    def exposure(self, /) -> list[float]:
        """
        Exposure per origin: the exposure column's latest observed value.
        """
    def loo(self, /) -> Elpd:
        """
        PSIS-LOO of the observed increments, as ``BayesOdpFit.loo``: the
        pointwise term is the normalised over-dispersed Poisson density at
        the cell's mean and ``scale``, so the result compares with
        ``BayesOdp``'s on the same column.
        
        Returns
        -------
        Elpd
        
        Raises
        ------
        ValueError
            If an observed increment is negative (the density does not
            exist there).
        """
    @property
    def origins(self, /) -> list[str]:
        """
        Origin period of each reserve component.
        """
    @property
    def parameter_names(self, /) -> list[str]:
        """
        Parameter names: ``alpha[1]`` to ``alpha[n]``, ``beta[1]`` to
        ``beta[m]``.
        """
    @property
    def prior_ultimate(self, /) -> list[float]:
        """
        The prior mean of each origin's row parameter ``mu[w]``,
        ``apriori * exposure`` (EVW's prior ultimate). The model's expected
        ultimate is ``mu[w]`` times the sum of the ``gamma[d]``, not
        ``mu[w]``.
        """
    @property
    def reserves(self, /) -> PredictiveDistribution:
        """
        Joint posterior predictive distribution of the reserve (the sum of
        future incremental values) by origin: dimension ``"origin"``, one
        component per origin, one row per simulation.
        """
    @property
    def scale(self, /) -> float:
        """
        The plugged-in scale ``phi``, the chain ladder's Pearson scale.
        """
    def summary(self, /) -> Any:
        """
        Posterior summary, one row per parameter: ``parameter``, ``mean``,
        ``sd``, ``q05``, ``q50``, ``q95``, ``rhat``, ``ess_bulk`` and
        ``ess_tail``, as ``BayesOdpFit.summary``. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    def to_frame(self, /) -> Any:
        """
        One row per origin: ``origin``, the chain ladder's ``latest``,
        ``ultimate`` and ``reserve``, the ``exposure`` and
        ``prior_ultimate``, and the ``mean`` and ``std_dev`` of the
        posterior predictive reserve. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    def totals_frame(self, /) -> Any:
        """
        One row: the chain ladder's totals, the total ``exposure`` and
        ``prior_ultimate``, the ``scale``, and the ``mean`` and ``std_dev``
        of the posterior predictive total reserve. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """

@final
class BayesGlm:
    """
    A Bayesian GLM sampled with NUTS (nuts-rs, the Rust core of nutpie).
    
    Normal priors with mean 0 on the coefficients: standard deviation
    ``intercept_sd`` for an all-ones column, ``prior_sd`` for the others
    (on the link scale; standardize covariates). For the Gaussian, gamma
    and inverse Gaussian the dispersion is sampled too, with a half-normal
    prior of scale ``dispersion_scale``, unless ``dispersion`` fixes it.
    Chains run in parallel, start near the maximum-likelihood fit, and
    replay exactly from ``seed``.
    
    Parameters
    ----------
    family : str
    link : str, optional
    prior_sd : float, default 2.5
    intercept_sd : float, default 10.0
    dispersion : float, optional
        A fixed dispersion; 1 by default for the Poisson, binomial and
        negative binomial. A Tweedie needs one.
    dispersion_scale : float, default 10.0
    chains, tune, draws : int, default 4, 1000, 1000
    seed : int, default 0
    target_accept : float, default 0.8
    max_depth : int, default 10
    theta, power, link_power : float, optional
    
    Examples
    --------
    >>> from prospicio.models import BayesGlm, Design
    >>> x = [(i % 4) - 1.5 for i in range(40)]
    >>> y = [[1.0, 2.0, 3.0, 5.0][i % 4] for i in range(40)]
    >>> d = Design([[1.0] * 40, x], ["(Intercept)", "x"])
    >>> fit = BayesGlm("poisson", chains=2, tune=300, draws=300).fit(d, y)
    >>> all(s["rhat"] < 1.05 for s in fit.summary())
    True
    """
    def __new__(cls, /, family: str, link: str |None = None, prior_sd: float = 2.5, intercept_sd: float = 10.0, dispersion: float |None = None, dispersion_scale: float = 10.0, chains: int = 4, tune: int = 1000, draws: int = 1000, seed: int = 0, target_accept: float = 0.8, max_depth: int = 10, theta: float |None = None, power: float |None = None, link_power: float |None = None) -> BayesGlm: ...
    def fit(self, /, design: Design, y: Sequence[float]) -> BayesGlmFit:
        """
        Samples the posterior.
        
        Parameters
        ----------
        design : Design
        y : list of float
        
        Returns
        -------
        BayesGlmFit
        """

@final
class BayesGlmFit:
    """
    A sampled Bayesian GLM, from ``BayesGlm.fit``.
    """
    @property
    def chains(self, /) -> int:
        """
        Number of chains.
        """
    @property
    def coefficient_draws(self, /) -> list[list[float]]:
        """
        Coefficient draws, one row per draw (chain by chain).
        """
    @property
    def dispersion_draws(self, /) -> list[float]:
        """
        Dispersion draws, one per draw (constant when fixed).
        """
    @property
    def divergences(self, /) -> int:
        """
        Divergent transitions among the kept draws.
        """
    def log_likelihood(self, /, design: Design, y: Sequence[float]) -> list[list[float]]:
        """
        Pointwise log-likelihood of ``y`` given ``design``: one row per draw,
        one column per observation, for ``elpd_loo`` or ``elpd_waic``.
        
        Returns
        -------
        list of list of float
        """
    def loo(self, /, design: Design, y: Sequence[float]) -> Elpd:
        """
        PSIS-LOO of ``y`` given ``design``, with each observation's relative
        efficiency estimated from the chains.
        
        Returns
        -------
        Elpd
        """
    @property
    def names(self, /) -> list[str]:
        """
        Coefficient names.
        """
    @property
    def posterior_mean(self, /) -> list[float]:
        """
        Posterior means of the coefficients.
        """
    def predict(self, /, design: Design) -> list[float]:
        """
        Posterior mean of each row's mean.
        
        Returns
        -------
        list of float
        """
    def predict_distribution(self, /, design: Design, n_sims: int, seed: int) -> PredictiveDistribution:
        """
        Posterior predictive draws across the rows, keyed ``row = 0, 1, ...``.
        
        Returns
        -------
        PredictiveDistribution
        """
    def summary(self, /) -> list[dict]:
        """
        Posterior summary: one dict per parameter with ``name``, ``mean``,
        ``sd``, ``q05``, ``q50``, ``q95``, ``rhat``, ``ess_bulk`` and
        ``ess_tail``; the dispersion last when it was sampled.
        
        Returns
        -------
        list of dict
        """

@final
class BayesOdp:
    """
    The Bayesian chain ladder of England and Verrall (2002, 2006): the
    over-dispersed Poisson model of a cumulative triangle's incremental
    values ``X`` with mean ``exp(c + a[w] + b[d])`` (``a[1] = b[1] = 0``),
    normal priors on the log scale, and the chain ladder's Pearson scale
    ``phi`` (``OdpBootstrapFit.scale``) plugged in. The posterior is sampled
    with NUTS, and each draw projects the future cells up to the oldest
    observed age (no tail) with the process error of ``process``, as
    ``OdpBootstrap`` does. Every origin's and every development column's
    observed increments must sum to more than zero; negative cells are
    allowed.
    
    One ``seed`` replays a fit exactly, whatever the number of threads: the
    chains and the simulations each take a seed derived from it.
    
    Failed MCMC diagnostics (R-hat above 1.01, bulk or tail ESS below 100
    per chain, any divergent transition) do not fail the fit: ``fit``
    raises one ``UserWarning`` listing them, and they stay in
    ``BayesOdpFit.diagnostics``.
    
    Parameters
    ----------
    intercept_sd : float, default 10.0
        Standard deviation of the normal prior on the intercept ``c``,
        centred on the log of the mean observed increment.
    origin_sd : float, default 10.0
        Standard deviation of the normal prior on each origin effect
        ``a[w]``, ``w >= 2``, centred on 0.
    development_sd : float, default 10.0
        Standard deviation of the normal prior on each development effect
        ``b[d]``, ``d >= 2``, centred on 0.
    process : {"gamma", "none"}, default "gamma"
        Process error on each simulated future increment: Gamma with mean
        ``mu`` and variance ``phi * mu``, or none for parameter error only.
    chains, tune, draws : int, default 4, 1000, 1000
        NUTS chains, warm-up draws per chain and kept draws per chain.
    n_sims : int, default 10000
        Number of predictive simulations, independent of the number of
        posterior draws; positive.
    seed : int, default 0
        Seed of the chains and of the simulations, from 0 to ``2**64 - 1``.
    target_accept : float, default 0.95
        Target mean acceptance rate of the step-size adaptation; above
        NUTS's usual 0.8, which on triangles such as RAA usually leaves a
        few divergent transitions.
    max_depth : int, default 10
        Largest NUTS tree depth.
    dependence : {"independent", "rank_correlation", "synchronized"}, default "independent"
        How the segments of a multi-segment triangle depend on each other
        in ``fit_segments`` (``fit`` has one segment and does not use it):
        each segment's own posterior and simulations, with seeds derived
        from ``seed`` and the segment's position so that the segments share
        no random numbers; or the segments' simulations reordered as whole
        rows (Iman–Conover on the segment totals) to the Spearman matrix
        ``spearman`` and then put in a random order, as
        ``OdpBootstrap``'s. ``"synchronized"`` is accepted here, as the
        bootstraps take it, but ``fit_segments`` raises: it resamples the
        residuals of the same cells in every segment, and a posterior has no
        residuals to share.
    spearman : list of list of float, optional
        With ``dependence="rank_correlation"`` only: Spearman's rho between
        the segments' totals, as ``OdpBootstrap``'s.
    
    Raises
    ------
    ValueError
        If ``n_sims`` is zero, ``process`` or ``dependence`` is unknown, or
        ``spearman`` is missing for ``"rank_correlation"``, given for another
        dependence or not square.
    OverflowError
        If an integer setting is negative or too large.
    
    Examples
    --------
    >>> from prospicio.reserving import BayesOdp, Triangle
    >>> tri = Triangle.from_long(
    ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
    ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
    ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
    ... )
    >>> fit = BayesOdp(n_sims=2000, seed=1).fit(tri, "values")
    >>> fit.reserves.components()
    [('2020',), ('2021',), ('2022',), ('2023',)]
    >>> fit.reserves.mean() > 0
    True
    >>> fit.parameter_names[:2]
    ['c', 'a[2]']
    """
    def __new__(cls, /, intercept_sd: float = 10.0, origin_sd: float = 10.0, development_sd: float = 10.0, process: str = "gamma", chains: int = 4, tune: int = 1000, draws: int = 1000, n_sims: int = 10000, seed: int = 0, target_accept: float = 0.95, max_depth: int = 10, dependence: str = "independent", spearman: Sequence[Sequence[float]] |None = None) -> BayesOdp: ...
    def __repr__(self, /) -> str: ...
    @property
    def chains(self, /) -> int:
        """
        Number of NUTS chains.
        """
    @property
    def dependence(self, /) -> str:
        """
        How segments depend on each other in ``fit_segments``:
        ``"independent"``, ``"rank_correlation"`` or ``"synchronized"``.
        """
    @property
    def development_sd(self, /) -> float:
        """
        Standard deviation of each development effect's prior.
        """
    @property
    def draws(self, /) -> int:
        """
        Kept draws per chain.
        """
    def fit(self, /, triangle: Triangle, column: str) -> BayesOdpFit:
        """
        Samples the posterior of one measure column of a single-segment
        cumulative triangle and simulates its reserves. Every origin must be
        observed from the first age up to its latest, there must be more
        observed cells than parameters, and every origin's and every
        development column's increments must sum to more than zero. Raises
        one ``UserWarning`` listing the failed diagnostics, if any.
        
        Parameters
        ----------
        triangle : Triangle
            Cumulative, with one segment.
        column : str
        
        Returns
        -------
        BayesOdpFit
        
        Raises
        ------
        ValueError
            As ``OdpBootstrap.fit``; if the triangle has several segments; if
            a prior standard deviation is not positive and finite or a
            sampler setting is invalid; or if an origin's or a development
            column's increments do not sum to more than zero.
        """
    def fit_segments(self, /, triangle: Triangle, column: str) -> BayesOdpFits:
        """
        Samples the posterior of one measure column in every segment of a
        cumulative triangle, each segment with its own posterior (the same
        model and settings, sampled independently), into one joint
        distribution of the reserves, the segments depending on each other
        as ``dependence`` says. Segment ``k`` (in index order) takes seeds
        derived from ``seed`` and ``k`` for its chains and its simulations,
        so independent segments share no random numbers, and a segment's
        draws differ from ``fit`` of that segment alone. Raises one
        ``UserWarning`` listing the failed diagnostics of every segment,
        each message prefixed with its segment's label.
        
        Parameters
        ----------
        triangle : Triangle
            Cumulative, with any number of segments.
        column : str
        
        Returns
        -------
        BayesOdpFits
        
        Raises
        ------
        ValueError
            As ``fit`` for any segment (the message names the segment); if
            ``dependence`` is ``"synchronized"`` (before any sampling); or if
            ``spearman`` does not fit the segments.
        
        Examples
        --------
        >>> import warnings
        >>> from prospicio.reserving import BayesOdp, Triangle
        >>> paid = [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0]
        >>> tri = Triangle.from_long(
        ...     ([2020] * 4 + [2021] * 3 + [2022] * 2 + [2023]) * 2,
        ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12] * 2,
        ...     {"paid": paid + [2 * v for v in paid]},
        ...     keys={"lob": ["Auto"] * 10 + ["Home"] * 10},
        ... )
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
        ...     fits = BayesOdp(tune=300, draws=300, n_sims=1000).fit_segments(tri, "paid")
        >>> fits.index, fits.reserves.dims
        (['Auto', 'Home'], ['lob', 'origin'])
        >>> fits.segment(lob="Home").scale > 0
        True
        """
    @property
    def intercept_sd(self, /) -> float:
        """
        Standard deviation of the intercept's prior.
        """
    @property
    def max_depth(self, /) -> int:
        """
        Largest NUTS tree depth.
        """
    @property
    def n_sims(self, /) -> int:
        """
        Number of predictive simulations.
        """
    @property
    def origin_sd(self, /) -> float:
        """
        Standard deviation of each origin effect's prior.
        """
    @property
    def process(self, /) -> str:
        """
        Process error: ``"gamma"`` or ``"none"``.
        """
    @property
    def seed(self, /) -> int:
        """
        Seed of the chains and of the simulations.
        """
    @property
    def spearman(self, /) -> list[list[float]] |None:
        """
        The Spearman matrix of ``dependence="rank_correlation"``; ``None``
        otherwise.
        """
    @property
    def target_accept(self, /) -> float:
        """
        Target mean acceptance rate.
        """
    @property
    def tune(self, /) -> int:
        """
        Warm-up draws per chain.
        """

@final
class BayesOdpFit:
    """
    A fitted ``BayesOdp``: the posterior of ``c``, ``a[2..n]`` and
    ``b[2..m]``, its MCMC diagnostics, and the posterior predictive
    distribution of the reserves, laid out as ``OdpBootstrapFit.reserves``.
    The frames' ``latest``, ``ultimate`` and ``reserve`` are the
    volume-weighted chain ladder's (``chain_ladder``); ``mean`` and
    ``std_dev`` are the posterior predictive reserve's.
    """
    def __repr__(self, /) -> str: ...
    @property
    def chain_ladder(self, /) -> ChainLadderFit:
        """
        The volume-weighted chain ladder of the triangle, whose latest
        values and Pearson scale the model uses.
        """
    @property
    def chains(self, /) -> int:
        """
        Number of chains.
        """
    @property
    def development(self, /) -> list[int]:
        """
        Development ages in months.
        """
    def development_frame(self, /) -> Any:
        """
        The chain ladder's development factors, one row per age, as
        ``ChainLadderFit.development_frame``. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def diagnostics(self, /) -> dict:
        """
        MCMC diagnostics: a dict with ``max_rhat``, ``min_ess_bulk``,
        ``min_ess_tail``, ``divergences`` and ``warnings`` (one message per
        failed threshold; empty when every threshold holds).
        """
    @property
    def draws(self, /) -> list[list[float]]:
        """
        Posterior draws, one row per draw (chain by chain), one column per
        parameter in the order of ``parameter_names``.
        """
    def loo(self, /) -> Elpd:
        """
        PSIS-LOO of the observed increments, with each cell's relative
        efficiency from the chains, as ``BayesGlmFit.loo``. The pointwise
        term is the normalised over-dispersed Poisson density at the cell's
        mean and ``scale``, so compare the result only with models of the
        same incremental values.
        
        Returns
        -------
        Elpd
        
        Raises
        ------
        ValueError
            If an observed increment is negative (the density does not
            exist there).
        """
    @property
    def origins(self, /) -> list[str]:
        """
        Origin period of each reserve component.
        """
    @property
    def parameter_names(self, /) -> list[str]:
        """
        Parameter names: ``c``, ``a[2]`` to ``a[n]``, ``b[2]`` to ``b[m]``.
        """
    @property
    def reserves(self, /) -> PredictiveDistribution:
        """
        Joint posterior predictive distribution of the reserve (the sum of
        future incremental values) by origin: dimension ``"origin"``, one
        component per origin, one row per simulation.
        """
    @property
    def scale(self, /) -> float:
        """
        The plugged-in scale ``phi``, the chain ladder's Pearson scale (the
        ODP bootstrap's).
        """
    def summary(self, /) -> Any:
        """
        Posterior summary, one row per parameter: ``parameter``, ``mean``,
        ``sd``, ``q05``, ``q50``, ``q95``, ``rhat``, ``ess_bulk`` and
        ``ess_tail``, the fields of ``BayesGlmFit.summary``. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    def to_frame(self, /) -> Any:
        """
        One row per origin: ``origin``, the chain ladder's ``latest``,
        ``ultimate`` and ``reserve``, and the ``mean`` and ``std_dev`` of
        the posterior predictive reserve. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    def totals_frame(self, /) -> Any:
        """
        One row: the chain ladder's totals, the ``scale``, and the ``mean``
        and ``std_dev`` of the posterior predictive total reserve. Needs
        pandas.
        
        Returns
        -------
        pandas.DataFrame
        """

@final
class BayesOdpFits:
    """
    ``BayesOdp.fit_segments``: each segment's own posterior and the joint
    posterior predictive distribution of the reserves, laid out as
    ``OdpBootstrapFit``.
    
    ``reserves`` has the triangle's keys and ``"origin"`` as dimensions, so
    ``reserves.aggregate(["lob"])`` keeps the dependence between segments.
    Per-origin lists run over the origins of each segment in turn, like the
    rows of ``to_frame()`` and the components of ``reserves``. The
    posterior of one segment (``scale``, ``diagnostics``,
    ``parameter_names``, ``draws``, ``chains``, ``summary()`` and
    ``loo()``) needs a single-segment fit: for several segments use
    ``segment(...)``, ``totals_frame()`` (the scales) or
    ``diagnostics_frame()``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def chain_ladder(self, /) -> ChainLadderFit:
        """
        The volume-weighted chain ladder of every segment, whose latest
        values and Pearson scales the model uses.
        """
    @property
    def chains(self, /) -> int:
        """
        Number of chains of a single-segment fit.
        """
    @property
    def development(self, /) -> list[int]:
        """
        Development ages in months of the first segment.
        """
    def development_frame(self, /) -> Any:
        """
        The chain ladders' development factors, one row per segment and
        age, as ``ChainLadderFit.development_frame``. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def diagnostics(self, /) -> dict:
        """
        MCMC diagnostics of a single-segment fit, as
        ``BayesOdpFit.diagnostics``.
        """
    def diagnostics_frame(self, /) -> Any:
        """
        One row per segment: the key columns, then ``max_rhat``,
        ``min_ess_bulk``, ``min_ess_tail``, ``divergences`` and
        ``warnings`` (the segment's messages joined by ``"; "``, empty when
        every threshold holds). Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def draws(self, /) -> list[list[float]]:
        """
        Posterior draws of a single-segment fit, as ``BayesOdpFit.draws``.
        """
    @property
    def index(self, /) -> list[Any]:
        """
        Label of each segment, as ``Triangle.index``.
        """
    @property
    def keys(self, /) -> list[str]:
        """
        Names of the triangle's key columns; empty without keys.
        """
    def loo(self, /) -> Elpd:
        """
        PSIS-LOO of a single-segment fit's observed increments, as
        ``BayesOdpFit.loo``.
        
        Returns
        -------
        Elpd
        
        Raises
        ------
        ValueError
            If the fit has several segments, or an observed increment is
            negative.
        """
    @property
    def origins(self, /) -> list[str]:
        """
        Origin period of each per-origin value and reserve component.
        """
    @property
    def parameter_names(self, /) -> list[str]:
        """
        Parameter names of a single-segment fit, as
        ``BayesOdpFit.parameter_names``.
        """
    @property
    def reserves(self, /) -> PredictiveDistribution:
        """
        Joint posterior predictive distribution of the reserve by segment
        and origin: the triangle's keys and ``"origin"`` are its dimensions,
        one component per segment and origin, one row per simulation.
        """
    @property
    def scale(self, /) -> float:
        """
        The plugged-in scale ``phi`` of a single-segment fit.
        """
    def segment(self, /, **keys) -> BayesOdpFits:
        """
        The fit of one segment, chosen by key values as
        ``ChainLadderFit.segment``, with its part of the joint reserves
        (same dimensions).
        
        Returns
        -------
        BayesOdpFits
        """
    def summary(self, /) -> Any:
        """
        Posterior summary of a single-segment fit, as
        ``BayesOdpFit.summary``. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    def to_frame(self, /) -> Any:
        """
        One row per segment and origin: the key columns, ``origin``, the
        chain ladder's ``latest``, ``ultimate`` and ``reserve``, and the
        ``mean`` and ``std_dev`` of the posterior predictive reserve. Needs
        pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    def totals_frame(self, /) -> Any:
        """
        One row per segment: the key columns, the chain ladder's totals,
        the ``scale``, and the ``mean`` and ``std_dev`` of the segment's
        posterior predictive total reserve. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """

@final
class BayesStacking:
    """
    Bayesian stacking: a posterior for the stacking weights, with a
    Dirichlet prior, sampled by NUTS from pointwise held-out log densities
    (Yao et al., 2018). ``stacking_weights`` gives the optimum alone.
    
    Parameters
    ----------
    concentration : list of float, optional
        Dirichlet concentration, one per model (default 1, uniform).
    chains, tune, draws : int, default 4, 1000, 1000
    seed : int, default 0
    """
    def __new__(cls, /, concentration: Sequence[float] |None = None, chains: int = 4, tune: int = 1000, draws: int = 1000, seed: int = 0) -> BayesStacking: ...
    def fit(self, /, lpd: Sequence[Sequence[float]]) -> StackingFit:
        """
        Samples the weights.
        
        Parameters
        ----------
        lpd : list of list of float
            One list per model, one held-out log density per observation.
        
        Returns
        -------
        StackingFit
        """

@final
class Benktander:
    """
    The Benktander (iterated Bornhuetter–Ferguson) method: starting from
    ``U(0) = apriori * exposure``, ``U(k) = latest + (1 - 1 / cdf) * U(k-1)``
    for ``n_iters`` steps, as chainladder-python's ``Benktander``.
    ``n_iters=0`` is the expected loss method, 1 is Bornhuetter–Ferguson,
    and many iterations approach the chain ladder. The steps are summed in
    closed form, so a large ``n_iters`` is cheap; where an origin's ``cdf``
    is below 1/2 they diverge instead.
    
    Parameters
    ----------
    apriori : float, default 1.0
        Expected loss ratio of the starting ultimate; positive.
    n_iters : int, default 1
        Number of Bornhuetter–Ferguson steps.
    average : {"volume", "simple", "regression"}, default "volume"
        How link ratios are averaged, as in ``ChainLadder``.
    sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
    tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
        As ``ChainLadder``; no tail by default.
    
    Examples
    --------
    >>> from prospicio.reserving import Benktander, Triangle
    >>> tri = Triangle.from_long(
    ...     [2020, 2020, 2021], [12, 24, 12],
    ...     {"paid": [100.0, 150.0, 200.0], "premium": [250.0, 250.0, 400.0]},
    ... )
    >>> fit = Benktander(apriori=0.5, n_iters=2).fit(tri, "paid", "premium")
    >>> [round(u, 2) for u in fit.ultimate]
    [150.0, 288.89]
    """
    def __new__(cls, /, apriori: float = 1.0, n_iters: int = 1, average: str = "volume", sigma_interpolation: str = "log-linear", tail: Any |None = None) -> Benktander: ...
    def __repr__(self, /) -> str: ...
    @property
    def apriori(self, /) -> float:
        """
        Expected loss ratio of the starting ultimate.
        """
    @property
    def average(self, /) -> str:
        """
        How link ratios are averaged.
        """
    def fit(self, /, triangle: Triangle, column: str, exposure: str) -> ExpectedLossFit:
        """
        Fits one loss column in every segment of a triangle, each with its
        own exposure.
        
        Parameters
        ----------
        triangle : Triangle
        column : str
            The losses to project.
        exposure : str
            The exposure column; each origin's latest observed cumulative
            value is its exposure (an incremental triangle's is cumulated).
        
        Returns
        -------
        ExpectedLossFit
        
        Raises
        ------
        ValueError
            As ``ExpectedLoss.fit``.
        """
    @property
    def n_iters(self, /) -> int:
        """
        Number of Bornhuetter–Ferguson steps.
        """
    @property
    def sigma_interpolation(self, /) -> str:
        """
        How unestimable variance parameters are filled in.
        """
    @property
    def tail(self, /) -> Any:
        """
        The tail: a constant factor as a number, otherwise its estimator.
        """

@final
class Beta:
    """
    Beta distribution with shapes ``a`` and ``b`` on ``[0, scale]``, as
    SciPy's ``beta(a, b, scale=scale)``: a bounded severity, such as a
    damage ratio times a sum insured.
    
    Parameters
    ----------
    a : float
    b : float
    scale : float, default 1.0
    
    Raises
    ------
    ValueError
        If a parameter is not finite and positive.
    
    Examples
    --------
    >>> from prospicio.distributions import Beta
    >>> Beta(1.0, 1.0, 10.0).mean()
    5.0
    """
    def __getnewargs__(self, /) -> tuple[float, float, float]: ...
    def __new__(cls, /, a: float, b: float, scale: float = 1.0) -> Beta: ...
    def __repr__(self, /) -> str: ...
    @property
    def a(self, /) -> float:
        """
        First shape ``a``.
        """
    @property
    def b(self, /) -> float:
        """
        Second shape ``b``.
        """
    def cdf(self, /, x: float) -> float:
        """
        Distribution function ``P(X <= x)``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    def layer(self, /, limit: float, attachment: float) -> float:
        """
        Expected loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
            ``inf`` for an unlimited layer.
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_second_moment(self, /, limit: float, attachment: float) -> float:
        """
        Second moment of the loss to the layer ``limit`` xs
        ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_variance(self, /, limit: float, attachment: float) -> float:
        """
        Variance of the loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def lev(self, /, limit: float) -> float:
        """
        Limited expected value ``E[min(X, limit)]``.
        
        Parameters
        ----------
        limit : float
        
        Returns
        -------
        float
        """
    def mean(self, /) -> float:
        """
        Mean of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """
    def quantile(self, /, p: float) -> float:
        """
        Quantile: the smallest ``x`` with ``P(X <= x) >= p``.
        
        Parameters
        ----------
        p : float
            Probability in ``[0, 1]``.
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def sample(self, /, n: int, seed: int, stream: int = 0) -> list[float]:
        """
        ``n`` draws from stream ``stream`` of the generator keyed by
        ``seed``.
        
        Parameters
        ----------
        n : int
        seed : int
        stream : int, default 0
        
        Returns
        -------
        list of float
        """
    @property
    def scale(self, /) -> float:
        """
        Scale, the top of the support.
        """
    def std(self, /) -> float:
        """
        Standard deviation of the distribution.
        
        Returns
        -------
        float
        """
    def stop_loss(self, /, retention: float) -> float:
        """
        Expected excess over a retention, ``E[max(X - retention, 0)]``.
        
        Parameters
        ----------
        retention : float
        
        Returns
        -------
        float
        """
    def survival(self, /, x: float) -> float:
        """
        Survival function ``P(X > x)``, accurate far into the tail.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    def variance(self, /) -> float:
        """
        Variance of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """

@final
class Binomial:
    """
    Binomial claim counts: ``n`` risks, each claiming with probability
    ``p``.
    
    Parameters
    ----------
    n : int
        Number of trials.
    p : float
        Claim probability in ``[0, 1)``.
    
    Examples
    --------
    >>> from prospicio.distributions import Binomial
    >>> Binomial(10, 0.3).mean()
    3.0
    """
    def __getnewargs__(self, /) -> tuple[int, float]: ...
    def __new__(cls, /, n: int, p: float) -> Binomial: ...
    def __repr__(self, /) -> str: ...
    def cdf(self, /, k: int) -> float:
        """
        ``P(N <= k)``.
        
        Parameters
        ----------
        k : int
        
        Returns
        -------
        float
        """
    def mean(self, /) -> float:
        """
        Mean of the claim count.
        
        Returns
        -------
        float
        """
    @property
    def n(self, /) -> int:
        """
        Number of trials.
        """
    @property
    def p(self, /) -> float:
        """
        Claim probability per trial.
        """
    def pmf(self, /, k: int) -> float:
        """
        ``P(N = k)``.
        
        Parameters
        ----------
        k : int
        
        Returns
        -------
        float
        """
    def quantile(self, /, p: float) -> int:
        """
        Smallest ``k`` with ``P(N <= k) >= p``.
        
        Parameters
        ----------
        p : float
        
        Returns
        -------
        int
        """
    def sample(self, /, n: int, seed: int, stream: int = 0) -> list[int]:
        """
        ``n`` claim counts from stream ``stream`` of the generator keyed by
        ``seed``.
        
        Parameters
        ----------
        n : int
        seed : int
        stream : int, default 0
        
        Returns
        -------
        list of int
        """
    def variance(self, /) -> float:
        """
        Variance of the claim count.
        
        Returns
        -------
        float
        """

@final
class BivariateReport:
    """
    What ``fft_bivariate`` produced and the error it introduced.
    """
    def __repr__(self, /) -> str: ...
    @property
    def aliasing_mass(self, /) -> tuple[float, float]:
        """
        On each axis, the mass past the end of the FFT buffer, which wraps
        around; their sum bounds the probability misplaced. Request more
        points when it is not negligible.
        """
    @property
    def expected_means(self, /) -> tuple[float, float]:
        """
        ``E[N] E[X]`` and ``E[N] E[Y]``: the means before truncation.
        """
    @property
    def grid_means(self, /) -> tuple[float, float]:
        """
        The means of the result.
        """
    @property
    def points(self, /) -> tuple[int, int]:
        """
        Points on each axis.
        """
    @property
    def tail_mass(self, /) -> tuple[float, float]:
        """
        The mass past the last row and past the last column, carried on them.
        """

@final
class Book:
    """
    A seller's book, ``aggregate``'s ``pnl`` and ``xpnl`` builders: a gross
    premium, expenses and the labels of its legs, laid out over a source of
    annual gross losses with the annual covers of a tower.
    
    :meth:`ledger` gives one of three faces:
    
    - ``consolidated`` (``pnl``): one group, the net position. With a tower
      it books net premium (gross less ceded premiums) less the net loss
      less the commission received less expenses, over the net loss; a
      loss-basis expense books as its rate times the expected gross loss.
    - ``walk`` (``xpnl``): the gross book, then one buy group per stage of
      the tower, with the running net and the covers' total impact.
    - ``peel-top-down`` and ``peel-bottom-up`` (``xpnl ... peel``): one buy
      group per layer, with an ``All aggregate`` subtotal.
    
    Every layer's terms are taken at the annual total. The builders book a
    fixed premium and a flat ceding commission; a layer with loss-sensitive
    terms is refused (write its legs with :class:`Leg`).
    
    Parameters
    ----------
    name : str
        The ledger's name.
    premium : float
        The gross premium.
    expenses : list of Expense, optional
    premium_label : str, optional
        ``Premium`` by default.
    loss_label : str, optional
        ``Loss`` by default; also the gross book's margin on a walk.
    step_label : str, optional
        The gross book's step on a walk, ``Gross`` by default.
    
    Examples
    --------
    >>> from prospicio.aggregate import Book, Expense
    >>> from prospicio.reinsurance import Layer, Tower
    >>> gross = ([2, 3, 4, 5, 6, 8], [0.25, 0.3, 0.09, 0.2, 0.12, 0.04])
    >>> cover = Layer.stop_loss("agg 3 xs 5", 3.0, 5.0).with_deposit_premium(1.0)
    >>> cover = cover.with_ceding_commission(0.1)
    >>> book = Book("Ex3", 10.0, [Expense("premium", 0.1)])
    >>> walk = book.ledger(gross, Tower([cover]), "walk")
    >>> round(walk.result.mean(), 12), walk.sheet()["step"][-1]
    (4.54, 'All')
    """
    def __new__(cls, /, name: str, premium: float, expenses: Sequence[Expense] |None = None, *, premium_label: str |None = None, loss_label: str |None = None, step_label: str |None = None) -> Book: ...
    def __repr__(self, /) -> str: ...
    def ledger(self, /, source: Any, tower: Tower |None = None, face: str = "consolidated") -> Ledger:
        """
        The ledger over the annual gross losses ``source``.
        
        Parameters
        ----------
        source : Grid or tuple of (list of float, list of float)
        tower : prospicio.reinsurance.Tower, optional
            Annual covers; without one, the plain book.
        face : str, default "consolidated"
            ``consolidated``, ``walk``, ``peel-top-down`` or
            ``peel-bottom-up``.
        
        Returns
        -------
        Ledger
        
        Raises
        ------
        ValueError
            A peel without a tower, a layer that needs each event, or one
            with loss-sensitive terms.
        """
    @property
    def premium(self, /) -> float:
        """
        The gross premium.
        """

@final
class BornhuetterFerguson:
    """
    The Bornhuetter–Ferguson method: each origin's latest value plus the
    expected loss ``apriori * exposure`` times the share still to develop,
    ``1 - 1 / cdf``, as chainladder-python's ``BornhuetterFerguson``.
    
    The exposure is a measure column of the same triangle (premium, say):
    each origin's latest observed cumulative value in the segment fitted.
    
    Parameters
    ----------
    apriori : float, default 1.0
        Expected loss ratio: the expected ultimate per unit of exposure;
        positive.
    average : {"volume", "simple", "regression"}, default "volume"
        How link ratios are averaged, as in ``ChainLadder``.
    sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
    tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
        As ``ChainLadder``; no tail by default.
    
    Examples
    --------
    >>> from prospicio.reserving import BornhuetterFerguson, Triangle
    >>> tri = Triangle.from_long(
    ...     [2020, 2020, 2021], [12, 24, 12],
    ...     {"paid": [100.0, 150.0, 200.0], "premium": [250.0, 250.0, 400.0]},
    ... )
    >>> fit = BornhuetterFerguson(apriori=0.5).fit(tri, "paid", "premium")
    >>> [round(u, 2) for u in fit.ultimate]
    [150.0, 266.67]
    """
    def __new__(cls, /, apriori: float = 1.0, average: str = "volume", sigma_interpolation: str = "log-linear", tail: Any |None = None) -> BornhuetterFerguson: ...
    def __repr__(self, /) -> str: ...
    @property
    def apriori(self, /) -> float:
        """
        Expected loss ratio.
        """
    @property
    def average(self, /) -> str:
        """
        How link ratios are averaged.
        """
    def fit(self, /, triangle: Triangle, column: str, exposure: str) -> ExpectedLossFit:
        """
        Fits one loss column in every segment of a triangle, each with its
        own exposure.
        
        Parameters
        ----------
        triangle : Triangle
        column : str
            The losses to project.
        exposure : str
            The exposure column; each origin's latest observed cumulative
            value is its exposure (an incremental triangle's is cumulated).
        
        Returns
        -------
        ExpectedLossFit
        
        Raises
        ------
        ValueError
            As ``ExpectedLoss.fit``.
        """
    @property
    def sigma_interpolation(self, /) -> str:
        """
        How unestimable variance parameters are filled in.
        """
    @property
    def tail(self, /) -> Any:
        """
        The tail: a constant factor as a number, otherwise its estimator.
        """

@final
class BuhlmannStraubFit:
    """
    A fitted Bühlmann–Straub model, from ``buhlmann_straub``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def between_variance(self, /) -> float:
        """
        The between-group variance ``a`` (may be negative; the factors are
        then 0).
        """
    @property
    def collective_mean(self, /) -> float:
        """
        The credibility-weighted collective mean.
        """
    @property
    def credibility(self, /) -> list[float]:
        """
        Each group's credibility factor.
        """
    @property
    def group_means(self, /) -> list[float]:
        """
        Each group's weighted mean.
        """
    @property
    def group_weights(self, /) -> list[float]:
        """
        Each group's total weight.
        """
    @property
    def iterations(self, /) -> int:
        """
        Iterations of the iterative estimator (0 for the unbiased one).
        """
    @property
    def k(self, /) -> float:
        """
        Bühlmann's ``k = s2 / a`` (``inf`` when ``a <= 0``).
        """
    @property
    def premiums(self, /) -> list[float]:
        """
        Each group's credibility premium.
        """
    @property
    def within_variance(self, /) -> float:
        """
        The within-group variance ``s2``.
        """

@final
class BurningCost:
    """
    The burning cost by accident year and overall, from
    ``PreparedLosses.burning_cost``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def loss_cost(self, /) -> list[float]:
        """
        Each layer's developed losses over the premium, summed over the
        years: the premium-weighted loss cost.
        """
    @property
    def mean_loss_cost(self, /) -> list[float]:
        """
        Each layer's loss cost averaged over the years with equal weight.
        """
    def to_dict(self, /) -> dict:
        """
        The years as columns: ``year``, ``premium``, ``losses`` (the count
        through the tower), ``development``, and per layer ``layer_loss``,
        ``developed`` and ``loss_cost`` (a list per year, one value per
        layer).
        
        Returns
        -------
        dict of str to list
        """
    @property
    def years(self, /) -> list[int]:
        """
        The accident years.
        """

@final
class Burr:
    """
    Burr (type XII) distribution with tail shape ``alpha``, power ``gamma``
    and scale ``theta``: ``P(X > x) = (1 + (x / theta)**gamma)**(-alpha)``,
    as SciPy's ``burr12(c=gamma, d=alpha, scale=theta)``. With ``alpha = 1``
    it is the ``Loglogistic``; with ``gamma = 1``, the Lomax. Moments exist
    below ``alpha * gamma``; limited and layer moments always do.
    
    Parameters
    ----------
    alpha : float
    gamma : float
    scale : float
    
    Raises
    ------
    ValueError
        If a parameter is not finite and positive.
    
    Examples
    --------
    >>> from prospicio.distributions import Burr
    >>> round(Burr(3.0, 1.0, 2000.0).survival(2000.0), 12)
    0.125
    """
    def __getnewargs__(self, /) -> tuple[float, float, float]: ...
    def __new__(cls, /, alpha: float, gamma: float, scale: float) -> Burr: ...
    def __repr__(self, /) -> str: ...
    @property
    def alpha(self, /) -> float:
        """
        Tail shape ``alpha``.
        """
    def cdf(self, /, x: float) -> float:
        """
        Distribution function ``P(X <= x)``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @property
    def gamma(self, /) -> float:
        """
        Power ``gamma``.
        """
    def layer(self, /, limit: float, attachment: float) -> float:
        """
        Expected loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
            ``inf`` for an unlimited layer.
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_second_moment(self, /, limit: float, attachment: float) -> float:
        """
        Second moment of the loss to the layer ``limit`` xs
        ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_variance(self, /, limit: float, attachment: float) -> float:
        """
        Variance of the loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def lev(self, /, limit: float) -> float:
        """
        Limited expected value ``E[min(X, limit)]``.
        
        Parameters
        ----------
        limit : float
        
        Returns
        -------
        float
        """
    def mean(self, /) -> float:
        """
        Mean of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """
    def quantile(self, /, p: float) -> float:
        """
        Quantile: the smallest ``x`` with ``P(X <= x) >= p``.
        
        Parameters
        ----------
        p : float
            Probability in ``[0, 1]``.
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def sample(self, /, n: int, seed: int, stream: int = 0) -> list[float]:
        """
        ``n`` draws from stream ``stream`` of the generator keyed by
        ``seed``.
        
        Parameters
        ----------
        n : int
        seed : int
        stream : int, default 0
        
        Returns
        -------
        list of float
        """
    @property
    def scale(self, /) -> float:
        """
        Scale ``theta``.
        """
    def std(self, /) -> float:
        """
        Standard deviation of the distribution.
        
        Returns
        -------
        float
        """
    def stop_loss(self, /, retention: float) -> float:
        """
        Expected excess over a retention, ``E[max(X - retention, 0)]``.
        
        Parameters
        ----------
        retention : float
        
        Returns
        -------
        float
        """
    def survival(self, /, x: float) -> float:
        """
        Survival function ``P(X > x)``, accurate far into the tail.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    def variance(self, /) -> float:
        """
        Variance of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """

@final
class CapeCod:
    """
    The Cape Cod (Stanard–Bühlmann) method: Bornhuetter–Ferguson with each
    origin's apriori estimated from the triangle, as chainladder-python's
    ``CapeCod``.
    
    Origin ``j``'s used-up exposure is ``exposure[j] / cdf[j]`` and its
    latest value is trended to the triangle's valuation by
    ``(1 + trend) ** (months / 12)``, the months running from the end of
    the origin period. Origin ``i``'s trended apriori is the sum of the
    trended latest values weighted by ``decay ** abs(i - j)`` over the same
    weighted sum of used-up exposures; dividing by its own trend factor
    gives the apriori of its Bornhuetter–Ferguson ultimate.
    
    Parameters
    ----------
    trend : float, default 0.0
        Annual trend of the loss ratio; above -1.
    decay : float, default 1.0
        Weight of an origin ``n`` periods away, ``decay ** n``; from 0 to 1.
        With 1 every origin shares one loss ratio.
    average : {"volume", "simple", "regression"}, default "volume"
        How link ratios are averaged, as in ``ChainLadder``.
    sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
    tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
        As ``ChainLadder``; no tail by default.
    
    Examples
    --------
    >>> from prospicio.reserving import CapeCod, Triangle
    >>> tri = Triangle.from_long(
    ...     [2020, 2020, 2021], [12, 24, 12],
    ...     {"paid": [100.0, 150.0, 200.0], "premium": [250.0, 250.0, 400.0]},
    ... )
    >>> fit = CapeCod().fit(tri, "paid", "premium")
    >>> [round(a, 4) for a in fit.apriori], [round(u, 2) for u in fit.ultimate]
    ([0.6774, 0.6774], [150.0, 290.32])
    """
    def __new__(cls, /, trend: float = 0.0, decay: float = 1.0, average: str = "volume", sigma_interpolation: str = "log-linear", tail: Any |None = None) -> CapeCod: ...
    def __repr__(self, /) -> str: ...
    @property
    def average(self, /) -> str:
        """
        How link ratios are averaged.
        """
    @property
    def decay(self, /) -> float:
        """
        Weight of an origin one period away.
        """
    def fit(self, /, triangle: Triangle, column: str, exposure: str) -> CapeCodFit:
        """
        Fits one loss column in every segment of a triangle, each with its
        own exposure and apriori. Trend runs to the triangle's valuation.
        
        Parameters
        ----------
        triangle : Triangle
        column : str
            The losses to project.
        exposure : str
            The exposure column; each origin's latest observed cumulative
            value is its exposure (an incremental triangle's is cumulated).
        
        Returns
        -------
        CapeCodFit
        
        Raises
        ------
        ValueError
            As ``ChainLadder.fit``, if ``trend`` or ``decay`` is out of
            range, or if an origin has no observed, finite, positive exposure.
        """
    @property
    def sigma_interpolation(self, /) -> str:
        """
        How unestimable variance parameters are filled in.
        """
    @property
    def tail(self, /) -> Any:
        """
        The tail: a constant factor as a number, otherwise its estimator.
        """
    @property
    def trend(self, /) -> float:
        """
        Annual trend of the loss ratio.
        """

@final
class CapeCodFit:
    """
    A fitted Cape Cod of every segment of a triangle column: the fields of
    ``ExpectedLossFit``, with ``apriori`` the detrended loss ratio applied
    to each origin (chainladder-python's ``detrended_apriori_``), plus
    ``trended_apriori`` before detrending (its ``apriori_``).
    
    Per-origin lists run over the origins of each segment in turn, like the
    rows of ``to_frame()``.
    
    Examples
    --------
    >>> from prospicio.reserving import CapeCod, Triangle
    >>> tri = Triangle.from_long(
    ...     [2020, 2020, 2021], [12, 24, 12],
    ...     {"paid": [100.0, 150.0, 200.0], "premium": [250.0, 250.0, 400.0]},
    ... )
    >>> fit = CapeCod(trend=0.1).fit(tri, "paid", "premium")
    >>> round(fit.trended_apriori[0] / fit.apriori[0], 10)
    1.1
    """
    def __repr__(self, /) -> str: ...
    @property
    def apriori(self, /) -> list[float]:
        """
        Detrended expected loss ratio applied per origin.
        """
    @property
    def cdf(self, /) -> list[float]:
        """
        Age-to-ultimate factors, one per age, including the tail.
        """
    @property
    def chain_ladder(self, /) -> ChainLadderFit:
        """
        The underlying chain-ladder projection, with the chain ladder's
        ultimate.
        """
    @property
    def development(self, /) -> list[int]:
        """
        Development ages in months.
        """
    def development_frame(self, /) -> Any:
        """
        One row per segment and age, as ``ChainLadderFit.development_frame``.
        Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def expected_loss(self, /) -> ExpectedLossFit:
        """
        The expected-loss fit: ultimates, exposures and the detrended
        apriori.
        """
    @property
    def exposure(self, /) -> list[float]:
        """
        Exposure per origin: the exposure column's latest observed cumulative
        value.
        """
    @property
    def index(self, /) -> list[Any]:
        """
        Label of each segment, as ``Triangle.index``.
        """
    @property
    def keys(self, /) -> list[str]:
        """
        Names of the triangle's key columns; empty without keys.
        """
    @property
    def latest(self, /) -> list[float]:
        """
        Latest observed cumulative value per origin.
        """
    @property
    def ldf(self, /) -> list[float]:
        """
        Age-to-age factors; factor ``k`` links age ``k`` to ``k + 1``.
        """
    @property
    def origins(self, /) -> list[str]:
        """
        Origin period of each per-origin value.
        """
    @property
    def reserve(self, /) -> list[float]:
        """
        Reserve (ultimate minus latest) per origin.
        """
    def segment(self, /, **keys) -> CapeCodFit:
        """
        The fit of one segment, chosen by key values as
        ``ChainLadderFit.segment``.
        
        Returns
        -------
        CapeCodFit
        """
    def to_frame(self, /) -> Any:
        """
        One row per segment and origin: the key columns, ``origin``,
        ``latest``, ``ultimate``, ``reserve``, ``exposure``, ``apriori`` and
        ``trended_apriori``. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def total_reserve(self, /) -> float:
        """
        Total reserve across segments and origins.
        """
    @property
    def total_ultimate(self, /) -> float:
        """
        Total ultimate across segments and origins.
        """
    def totals_frame(self, /) -> Any:
        """
        One row per segment: the key columns and the segment's total
        ``latest``, ``ultimate``, ``reserve`` and ``exposure``. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def trended_apriori(self, /) -> list[float]:
        """
        Expected loss ratio per origin at the valuation's cost level, before
        detrending.
        """
    @property
    def ultimate(self, /) -> list[float]:
        """
        Cape Cod ultimate per origin.
        """

@final
class CessionStatistics:
    """
    The total cession of several per-occurrence layers, from
    ``occurrence_cession_statistics``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def aggregate(self, /) -> CompoundMoments:
        """
        The annual ceded loss.
        """
    @property
    def loss_on_line(self, /) -> float:
        """
        The annual ceded mean over the placed limits; 0 when one is
        unlimited.
        """
    @property
    def pr_attach(self, /) -> float:
        """
        ``P(X > a)`` at the lowest attachment, per claim.
        """
    @property
    def pr_loss(self, /) -> float:
        """
        The chance of a loss to the cession in the year.
        """
    @property
    def severity(self, /) -> CompoundMoments:
        """
        The ceded loss of a claim, every claim counted.
        """

@final
class ChainLadder:
    """
    The chain-ladder method: each origin's latest value projected to
    ultimate with age-to-age factors estimated from the triangle and a tail.
    
    Parameters
    ----------
    average : {"volume", "simple", "regression"}, default "volume"
        How link ratios are averaged into one factor per age: volume
        weighted, their mean, or least squares through the origin (Mack's
        ``alpha`` of 1, 0 and 2).
    sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
        How a variance parameter with a single link ratio is filled in.
    tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
        Development past the oldest age: a number is a constant factor from
        the oldest age to ultimate. No tail (a factor of 1) by default.
    
    Examples
    --------
    >>> from prospicio.reserving import ChainLadder, Triangle
    >>> tri = Triangle.from_long([2020, 2020, 2021], [12, 24, 12], {"paid": [100.0, 150.0, 200.0]})
    >>> fit = ChainLadder().fit(tri, "paid")
    >>> fit.ldf, fit.ultimate, fit.total_reserve
    ([1.5], [150.0, 300.0], 100.0)
    """
    def __new__(cls, /, average: str = "volume", sigma_interpolation: str = "log-linear", tail: Any |None = None) -> ChainLadder: ...
    def __repr__(self, /) -> str: ...
    @property
    def average(self, /) -> str:
        """
        How link ratios are averaged.
        """
    def fit(self, /, triangle: Triangle, column: str) -> ChainLadderFit:
        """
        Fits one measure column in every segment of a triangle, each on its
        own.
        
        Parameters
        ----------
        triangle : Triangle
            Cumulative or incremental, with any number of segments.
        column : str
        
        Returns
        -------
        ChainLadderFit
        
        Raises
        ------
        ValueError
            If the column is unknown, a factor cannot be estimated or the tail
            cannot be fitted (a constant that is not positive, a curve with
            fewer than two factors above 1 to fit); with keys, the message
            names the segment.
        """
    @property
    def sigma_interpolation(self, /) -> str:
        """
        How unestimable variance parameters are filled in.
        """
    @property
    def tail(self, /) -> Any:
        """
        The tail: a constant factor as a number, otherwise its estimator.
        """

@final
class ChainLadderFit:
    """
    A fitted chain-ladder projection of every segment of a triangle column.
    
    Per-origin lists (``origins``, ``latest``, ``ultimate``, ``reserve``)
    run over the origins of each segment in turn, like the rows of
    ``to_frame()``, so a single-segment fit has one value per origin.
    Per-age lists (``ldf``, ``cdf``, ``sigma``, ``std_err``) and the tail
    need a single-segment fit; for several segments use
    ``development_frame()`` (per age), ``totals_frame()`` (``tail``,
    ``tail_sigma``, ``tail_std_err``) or ``segment(...)``.
    
    Examples
    --------
    >>> from prospicio.reserving import ChainLadder, Triangle
    >>> tri = Triangle.from_long(
    ...     [2020, 2020, 2021] * 2,
    ...     [12, 24, 12] * 2,
    ...     {"paid": [100.0, 150.0, 200.0, 10.0, 20.0, 30.0]},
    ...     keys={"lob": ["Auto"] * 3 + ["Home"] * 3},
    ... )
    >>> fit = ChainLadder().fit(tri, "paid")
    >>> fit.index, fit.reserve
    (['Auto', 'Home'], [0.0, 100.0, 0.0, 30.0])
    >>> fit.segment(lob="Home").ldf
    [2.0]
    """
    def __repr__(self, /) -> str: ...
    @property
    def cdf(self, /) -> list[float]:
        """
        Age-to-ultimate factors, one per age, including the tail.
        """
    @property
    def development(self, /) -> list[int]:
        """
        Development ages in months.
        """
    def development_frame(self, /) -> Any:
        """
        One row per segment and age: the key columns, ``development``,
        ``ldf`` (the selected factor to the next age), ``cdf`` (to ultimate,
        with the tail), ``sigma`` and ``std_err``; the oldest age has ``nan``
        for ``ldf``, ``sigma`` and ``std_err``, and the tail factor as its
        ``cdf``. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def estimated_ldf(self, /) -> list[float]:
        """
        Age-to-age factors as estimated, before the tail replaced any.
        """
    @property
    def index(self, /) -> list[Any]:
        """
        Label of each segment, as ``Triangle.index``.
        """
    @property
    def keys(self, /) -> list[str]:
        """
        Names of the triangle's key columns; empty without keys.
        """
    @property
    def latest(self, /) -> list[float]:
        """
        Latest observed cumulative value per origin.
        """
    @property
    def ldf(self, /) -> list[float]:
        """
        Selected age-to-age factors, which the projection uses: the
        estimated ones, replaced by the tail's from its attachment age.
        Factor ``k`` links age ``k`` to ``k + 1``.
        """
    @property
    def origins(self, /) -> list[str]:
        """
        Origin period of each per-origin value.
        """
    @property
    def reserve(self, /) -> list[float]:
        """
        Reserve (ultimate minus latest) per origin.
        """
    def segment(self, /, **keys) -> ChainLadderFit:
        """
        The fit of one segment, chosen by key values (compared as ``str()``
        of each value). Keys not named may take any value, so a fit with one
        segment needs none.
        
        Returns
        -------
        ChainLadderFit
        
        Raises
        ------
        ValueError
            If a key or value is unknown, or the choice matches several
            segments.
        """
    @property
    def sigma(self, /) -> list[float]:
        """
        Variance parameter of each factor, with unestimable ones
        interpolated (``nan`` where that is impossible).
        """
    @property
    def std_err(self, /) -> list[float]:
        """
        Standard error of each factor.
        """
    @property
    def tail(self, /) -> float:
        """
        Tail factor from the oldest age to ultimate.
        """
    @property
    def tail_attachment_age(self, /) -> int:
        """
        Age from which ``ldf`` holds the tail's factors rather than the
        estimated ones; the oldest age when the tail replaced none.
        """
    @property
    def tail_ldf(self, /) -> list[float]:
        """
        Factors past the oldest age, which multiply to ``tail``: one per
        development period of the following year and one to ultimate, as
        chainladder-python's ``ldf_`` (a single factor for
        ``TailLogLinear``).
        """
    @property
    def tail_sigma(self, /) -> float:
        """
        The tail's variance parameter, extrapolated log-linearly; 0 without
        a tail (a factor of 1), ``nan`` if it cannot be extrapolated. A tail
        below 1 is read where a tail of 1.001 would be, as chainladder-python
        does.
        """
    @property
    def tail_std_err(self, /) -> float:
        """
        Standard error of the tail factor, extrapolated log-linearly.
        """
    def to_frame(self, /) -> Any:
        """
        One row per segment and origin: the key columns, ``origin``,
        ``latest``, ``ultimate`` and ``reserve``. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def total_reserve(self, /) -> float:
        """
        Total reserve across segments and origins.
        """
    @property
    def total_ultimate(self, /) -> float:
        """
        Total ultimate across segments and origins.
        """
    def totals_frame(self, /) -> Any:
        """
        One row per segment: the key columns, the segment's total
        ``latest``, ``ultimate`` and ``reserve``, and its ``tail``,
        ``tail_sigma`` and ``tail_std_err``. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def ultimate(self, /) -> list[float]:
        """
        Projected ultimate per origin.
        """

@final
class ClaimsDevelopmentResult:
    """
    Merz and Wüthrich's (2008) one-year view of a Mack fit: standard errors
    of the claims development result (CDR), the change in the chain-ladder
    ultimate over a calendar year, per origin and in total. The total
    includes the covariance between origins. Year ``k`` of the run-off is
    R ChainLadder's ``CDR(k)S.E.``; summed in square over the years, the
    run-off gives back Mack's standard error.
    
    Examples
    --------
    >>> from prospicio.reserving import Mack, Triangle
    >>> tri = Triangle.from_long(
    ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
    ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
    ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
    ... )
    >>> mack = Mack().fit(tri, "values")
    >>> cdr = mack.claims_development_result()
    >>> len(cdr.by_calendar_year), cdr.one_year_standard_error[0]
    (3, 0.0)
    >>> abs(cdr.total_run_off_standard_error - mack.total_standard_error) < 1e-9
    True
    """
    def __repr__(self, /) -> str: ...
    @property
    def by_calendar_year(self, /) -> list[list[float]]:
        """
        Standard error of each origin's CDR in each future calendar year:
        ``by_calendar_year[k - 1][i]`` is year ``k`` (R's ``CDR(k)S.E.``) of
        origin ``i``, zero once the origin is fully developed. One year per
        age-to-age factor.
        """
    @property
    def one_year_standard_error(self, /) -> list[float]:
        """
        Standard error of each origin's CDR in the next calendar year, R's
        ``CDR(1)S.E.``.
        """
    @property
    def origins(self, /) -> list[str]:
        """
        Origin periods, oldest first.
        """
    @property
    def run_off_standard_error(self, /) -> list[float]:
        """
        Standard error of each origin's full run-off, the square root of
        the sum of its yearly mean squared errors; equals Mack's.
        """
    def to_frame(self, /) -> Any:
        """
        One row per origin: ``origin``, then ``cdr_1``, ``cdr_2``, ... the
        standard error of the CDR in each future calendar year (R's
        ``CDR(k)S.E.``), and ``run_off``, that of the full run-off. Needs
        pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def total_by_calendar_year(self, /) -> list[float]:
        """
        Standard error of the total CDR in each future calendar year.
        """
    @property
    def total_one_year_standard_error(self, /) -> float:
        """
        Standard error of the total CDR in the next calendar year.
        """
    @property
    def total_run_off_standard_error(self, /) -> float:
        """
        Standard error of the total full run-off; equals Mack's.
        """

@final
class ClarkCapeCod:
    """
    Clark's Cape Cod method (Clark 2003), as R ChainLadder's
    ``ClarkCapeCod``: one expected loss ratio times each origin's exposure
    and a growth curve are fitted to the incremental losses by
    over-dispersed Poisson maximum likelihood, with ages measured from the
    average date of loss.
    
    The reserve is the fitted ``elr * exposure * (G(max_age) - G(age))``;
    process and parameter risk are as in ``ClarkLdf``.
    
    Parameters
    ----------
    curve : {"loglogistic", "weibull"}, default "loglogistic"
        The growth curve, as in ``ClarkLdf``.
    max_age : float, optional
        Age in months at which development stops; at least the triangle's
        last age. ``None`` develops to infinity.
    
    Raises
    ------
    ValueError
        If ``curve`` is unknown.
    
    Examples
    --------
    >>> from prospicio.reserving import ClarkCapeCod, Triangle
    >>> rows = [[110.0, 290.0, 370.0, 420.0, 440.0], [95.0, 300.0, 390.0, 425.0],
    ...         [130.0, 320.0, 410.0], [105.0, 305.0], [120.0]]
    >>> tri = Triangle.from_long(
    ...     [2020 + i for i, row in enumerate(rows) for _ in row],
    ...     [12 * (d + 1) for row in rows for d in range(len(row))],
    ...     {"paid": [v for row in rows for v in row],
    ...      "premium": [800.0 for row in rows for _ in row]},
    ... )
    >>> fit = ClarkCapeCod().fit(tri, "paid", "premium")
    >>> 0 < fit.elr < 1 and fit.expected_ultimate == [fit.elr * 800.0] * 5
    True
    """
    def __new__(cls, /, curve: str = "loglogistic", max_age: float |None = None) -> ClarkCapeCod: ...
    def __repr__(self, /) -> str: ...
    @property
    def curve(self, /) -> str:
        """
        The growth curve.
        """
    def fit(self, /, triangle: Triangle, column: str, exposure: str) -> ClarkFit:
        """
        Fits one loss column in every segment of a triangle, each with its
        own exposure.
        
        Parameters
        ----------
        triangle : Triangle
        column : str
            The losses to project.
        exposure : str
            The exposure column; each origin's latest observed value is its
            exposure.
        
        Returns
        -------
        ClarkFit
        
        Raises
        ------
        ValueError
            As ``ClarkLdf.fit``, and if an origin has no observed, finite,
            positive exposure.
        """
    @property
    def max_age(self, /) -> float |None:
        """
        Age in months at which development stops; ``None`` for infinity.
        """

@final
class ClarkFit:
    """
    A fitted Clark LDF or Cape Cod model of every segment of a triangle
    column.
    
    Per-origin lists (``origins``, ``latest``, ``expected_ultimate``,
    ``ultimate``, ``reserve`` and the standard errors) run over the origins
    of each segment in turn, like the rows of ``to_frame()``. The fitted
    parameters and the standard errors of the total need a single-segment
    fit; for several segments use ``totals_frame()`` or ``segment(...)``.
    ``total_ultimate`` and ``total_reserve`` sum over every segment.
    
    Examples
    --------
    >>> from prospicio.reserving import ClarkLdf, Triangle
    >>> rows = [[110.0, 290.0, 370.0, 420.0, 440.0], [95.0, 300.0, 390.0, 425.0],
    ...         [130.0, 320.0, 410.0], [105.0, 305.0], [120.0]]
    >>> tri = Triangle.from_long(
    ...     [2020 + i for i, row in enumerate(rows) for _ in row],
    ...     [12 * (d + 1) for row in rows for d in range(len(row))],
    ...     [v for row in rows for v in row],
    ... )
    >>> fit = ClarkLdf().fit(tri, "values")
    >>> len(fit.covariance), fit.elr, fit.growth(float("inf"))
    (7, None, 1.0)
    """
    def __repr__(self, /) -> str: ...
    @property
    def chain_ladder(self, /) -> ChainLadderFit:
        """
        The volume-weighted chain ladder of the same column, with the chain
        ladder's ultimate.
        """
    @property
    def covariance(self, /) -> list[list[float]]:
        """
        Covariance of the parameters: the expected ultimates (LDF) or the
        expected loss ratio (Cape Cod), then ``omega`` and ``theta``. NaN if
        the Fisher information is singular.
        """
    @property
    def curve(self, /) -> str:
        """
        The growth curve.
        """
    @property
    def elr(self, /) -> float |None:
        """
        Expected loss ratio (Cape Cod), or ``None`` (LDF).
        """
    @property
    def expected_ultimate(self, /) -> list[float]:
        """
        Expected ultimate per origin, developed to infinity: fitted (LDF)
        or ``elr * exposure`` (Cape Cod).
        """
    @property
    def exposure(self, /) -> list[float] |None:
        """
        Exposure per origin (Cape Cod), or ``None`` (LDF).
        """
    def growth(self, /, age: float) -> float:
        """
        Share of the expected ultimate developed by a development age.
        
        Parameters
        ----------
        age : float
            Development age in months, before the shift to the average date
            of loss; ``inf`` gives 1.
        
        Returns
        -------
        float
        """
    @property
    def index(self, /) -> list[Any]:
        """
        Label of each segment, as ``Triangle.index``.
        """
    @property
    def keys(self, /) -> list[str]:
        """
        Names of the triangle's key columns; empty without keys.
        """
    @property
    def latest(self, /) -> list[float]:
        """
        Latest observed cumulative value per origin.
        """
    @property
    def max_age(self, /) -> float |None:
        """
        Age in months at which development stops; ``None`` for infinity.
        """
    @property
    def n_observations(self, /) -> int:
        """
        Number of observed incremental values fitted; ``scale`` divides by
        this less the number of parameters.
        """
    @property
    def omega(self, /) -> float:
        """
        Fitted shape of the growth curve.
        """
    @property
    def origin_width(self, /) -> float:
        """
        Length of the origin period in months; ages are shifted by half of
        it to the average date of loss.
        """
    @property
    def origins(self, /) -> list[str]:
        """
        Origin period of each per-origin value.
        """
    @property
    def parameter_risk(self, /) -> list[float]:
        """
        Parameter standard error per origin.
        """
    @property
    def process_risk(self, /) -> list[float]:
        """
        Process standard error per origin.
        """
    @property
    def reserve(self, /) -> list[float]:
        """
        Reserve per origin.
        """
    @property
    def scale(self, /) -> float:
        """
        Over-dispersion ``sigma**2``: squared Pearson residuals over the
        observed incremental values less the number of parameters.
        """
    def segment(self, /, **keys) -> ClarkFit:
        """
        The fit of one segment, chosen by key values as
        ``ChainLadderFit.segment``.
        
        Returns
        -------
        ClarkFit
        """
    @property
    def standard_error(self, /) -> list[float]:
        """
        Standard error per origin: ``sqrt(process**2 + parameter**2)``.
        """
    @property
    def theta(self, /) -> float:
        """
        Fitted scale of the growth curve, in months.
        """
    def to_frame(self, /) -> Any:
        """
        One row per segment and origin: the key columns, ``origin``,
        ``latest``, ``ultimate``, ``reserve``, (Cape Cod) ``exposure``,
        ``expected_ultimate``, ``process_risk``, ``parameter_risk`` and
        ``standard_error``. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def total_parameter_risk(self, /) -> float:
        """
        Parameter standard error of the total reserve, with the covariance
        between origins.
        """
    @property
    def total_process_risk(self, /) -> float:
        """
        Process standard error of the total reserve.
        """
    @property
    def total_reserve(self, /) -> float:
        """
        Total reserve across segments and origins.
        """
    @property
    def total_standard_error(self, /) -> float:
        """
        Standard error of the total reserve.
        """
    @property
    def total_ultimate(self, /) -> float:
        """
        Total ultimate across segments and origins.
        """
    def totals_frame(self, /) -> Any:
        """
        One row per segment: the key columns, the segment's total
        ``latest``, ``ultimate`` and ``reserve``, the ``process_risk``,
        ``parameter_risk`` and ``standard_error`` of its total reserve, and
        its ``omega``, ``theta``, ``scale`` and (Cape Cod) ``elr``. Needs
        pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def ultimate(self, /) -> list[float]:
        """
        Ultimate per origin: the latest value plus the reserve.
        """

@final
class ClarkLdf:
    """
    Clark's LDF method (Clark 2003), as R ChainLadder's ``ClarkLDF``: each
    origin's expected ultimate and a growth curve are fitted to the
    incremental losses by over-dispersed Poisson maximum likelihood, with
    ages measured from the average date of loss (the middle of the origin
    period, R's ``adol = TRUE``).
    
    The ultimate is the latest value developed by the fitted curve to
    ``max_age``. Process risk is the scale times the fitted reserve, and
    parameter risk the delta method on the parameters' covariance, the
    scale times the inverse Fisher information.
    
    Parameters
    ----------
    curve : {"loglogistic", "weibull"}, default "loglogistic"
        The growth curve ``G``: ``x**omega / (x**omega + theta**omega)`` or
        ``1 - exp(-(x / theta)**omega)``.
    max_age : float, optional
        Age in months at which development stops; at least the triangle's
        last age. ``None`` develops to infinity.
    
    Raises
    ------
    ValueError
        If ``curve`` is unknown.
    
    Examples
    --------
    >>> from prospicio.reserving import ClarkLdf, Triangle
    >>> rows = [[110.0, 290.0, 370.0, 420.0, 440.0], [95.0, 300.0, 390.0, 425.0],
    ...         [130.0, 320.0, 410.0], [105.0, 305.0], [120.0]]
    >>> tri = Triangle.from_long(
    ...     [2020 + i for i, row in enumerate(rows) for _ in row],
    ...     [12 * (d + 1) for row in rows for d in range(len(row))],
    ...     [v for row in rows for v in row],
    ... )
    >>> fit = ClarkLdf(curve="weibull", max_age=120).fit(tri, "values")
    >>> fit.omega > 0 and fit.total_standard_error > fit.total_process_risk
    True
    >>> round(fit.ultimate[2] * fit.growth(36) / fit.growth(120), 6)
    410.0
    """
    def __new__(cls, /, curve: str = "loglogistic", max_age: float |None = None) -> ClarkLdf: ...
    def __repr__(self, /) -> str: ...
    @property
    def curve(self, /) -> str:
        """
        The growth curve.
        """
    def fit(self, /, triangle: Triangle, column: str) -> ClarkFit:
        """
        Fits one loss column in every segment of a triangle, each on its
        own.
        
        Parameters
        ----------
        triangle : Triangle
        column : str
        
        Returns
        -------
        ClarkFit
        
        Raises
        ------
        ValueError
            As ``ChainLadder.fit``, if the triangle has fewer than four ages,
            ``max_age`` is before its last age, an origin's latest value is
            not positive, or the likelihood search does not converge.
        """
    @property
    def max_age(self, /) -> float |None:
        """
        Age in months at which development stops; ``None`` for infinity.
        """

@final
class Clash:
    """
    Two lines hit by the same events, each line independently within an
    event: ``aggregate``'s ``clash``.
    
    Given the expected events a year hitting line A only, line B only and
    both (the clash count), independence within an event fixes those hitting
    neither, ``only_a * only_b / both``. Each event hits A with probability
    ``trigger_a`` and B with ``trigger_b``; the dependence between the lines
    comes only from the shared count.
    
    Parameters
    ----------
    only_a, only_b : float
        Expected events a year hitting one line only, non-negative.
    both : float
        Expected events a year hitting both lines, positive.
    
    Examples
    --------
    >>> from prospicio.aggregate import Clash
    >>> from prospicio.distributions import Grid, Poisson
    >>> clash = Clash(2.0, 3.0, 1.0)
    >>> clash.events, clash.trigger_a
    (12.0, 0.25)
    >>> a = Grid(1.0, [0.0, 0.5, 0.5])
    >>> b = Grid(1.0, [0.0, 1.0])
    >>> joint, report = clash.aggregate(Poisson(clash.events), a, b, (128, 128))
    >>> round(joint.covariance(), 9)  # Poisson: both * E[X] * E[Y]
    1.5
    """
    def __new__(cls, /, only_a: float, only_b: float, both: float) -> Clash: ...
    def __repr__(self, /) -> str: ...
    def aggregate(self, /, frequency: Any, a: Grid, b: Grid, points: tuple[int, int]) -> tuple[JointGrid, BivariateReport]:
        """
        The joint annual loss to the two lines by ``fft_bivariate``.
        
        Parameters
        ----------
        frequency : Poisson, NegativeBinomial, Binomial or Count
            The shared count; its mean must be ``events``.
        a, b : Grid
            The loss to each line given that the event hits it.
        points : tuple of (int, int)
        
        Returns
        -------
        tuple of (JointGrid, BivariateReport)
        """
    @property
    def both(self, /) -> float:
        """
        Expected events a year hitting both lines: the clash count.
        """
    @property
    def events(self, /) -> float:
        """
        Expected events a year: the mean of the shared count.
        """
    @property
    def neither(self, /) -> float:
        """
        Expected events a year hitting neither line.
        """
    @property
    def only_a(self, /) -> float:
        """
        Expected events a year hitting line A only.
        """
    @property
    def only_b(self, /) -> float:
        """
        Expected events a year hitting line B only.
        """
    def severity(self, /, a: Grid, b: Grid) -> JointGrid:
        """
        The joint per-event severity: ``a`` with probability ``trigger_a``
        on line A and ``b`` with ``trigger_b`` on line B, independently.
        
        Parameters
        ----------
        a, b : Grid
            The loss to each line given that the event hits it.
        
        Returns
        -------
        JointGrid
        """
    @property
    def trigger_a(self, /) -> float:
        """
        The probability that an event hits line A.
        """
    @property
    def trigger_b(self, /) -> float:
        """
        The probability that an event hits line B.
        """

@final
class Coding:
    """
    Terms with factor levels learned from training data, from
    ``Terms.fit``.
    """
    def design(self, /, data: Any, offset: Sequence[float] |None = None, weights: Sequence[float] |None = None) -> Design:
        """
        The design matrix for ``data``.
        
        Parameters
        ----------
        data : dict of str to list
        offset : list of float, optional
        weights : list of float, optional
        
        Returns
        -------
        Design
        
        Raises
        ------
        ValueError
            If a column is missing or a factor level was not seen in training.
        """
    @property
    def names(self, /) -> list[str]:
        """
        Design matrix column names.
        """

@final
class CollectiveModel:
    """
    The collective risk model: a claim count and a severity, with layer
    moments in closed form.
    
    For the layer ``limit`` xs ``attachment`` applied to each loss, with
    ``Y`` the loss to the layer from one claim, the aggregate has mean
    ``E[N] E[Y]`` and variance ``E[N] Var[Y] + Var[N] E[Y]**2``.
    
    Parameters
    ----------
    frequency : Poisson, NegativeBinomial or Binomial
    severity : Lognormal, Grid, Pareto, PiecewisePareto, LogAffinePareto or GeneralizedPareto
    
    Examples
    --------
    >>> from prospicio.distributions import Pareto, claim_count
    >>> from prospicio.pricing import CollectiveModel
    >>> m = CollectiveModel(claim_count(2.0, 1.5), Pareto(1e6, 2.0))
    >>> round(m.layer_mean(4e6, 1e6))
    1600000
    >>> m.excess_frequency(2e6)
    0.5
    """
    def __new__(cls, /, frequency: Any, severity: Any) -> CollectiveModel: ...
    def excess_frequency(self, /, x: float) -> float:
        """
        Expected number of losses above ``x``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    def layer_mean(self, /, limit: float, attachment: float) -> float:
        """
        Expected aggregate loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
            ``inf`` for an unlimited layer.
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_std(self, /, limit: float, attachment: float) -> float:
        """
        Standard deviation of the aggregate loss to the layer.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_variance(self, /, limit: float, attachment: float) -> float:
        """
        Variance of the aggregate loss to the layer.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def mean(self, /) -> float:
        """
        Expected aggregate loss.
        
        Returns
        -------
        float
        """
    def simulate(self, /, n_sims: int, seed: int) -> EventSet:
        """
        ``n_sims`` simulated years of individual losses.
        
        Parameters
        ----------
        n_sims : int
        seed : int
        
        Returns
        -------
        EventSet
        """
    def variance(self, /) -> float:
        """
        Variance of the aggregate loss.
        
        Returns
        -------
        float
        """

@final
class CompoundMoments:
    """
    Compound moments to third order, from ``compound_moments``.
    """
    def __repr__(self, /) -> str: ...
    def approximate(self, /, kind: str) -> Approximation:
        """
        The moment-matched approximation: ``"normal"``, ``"gamma"``,
        ``"lognormal"`` (mean and cv), ``"shifted_gamma"`` or
        ``"shifted_lognormal"`` (mean, cv and skewness).
        
        Parameters
        ----------
        kind : str
        
        Returns
        -------
        Approximation
        
        Raises
        ------
        ValueError
            For an unknown kind, a non-positive mean (gamma, lognormal) or a
            non-positive skewness (the shifted fits).
        """
    @property
    def cv(self, /) -> float:
        """
        The coefficient of variation.
        """
    @property
    def mean(self, /) -> float:
        """
        ``E[S]``.
        """
    @property
    def skew(self, /) -> float:
        """
        The skewness.
        """
    @property
    def third_cumulant(self, /) -> float:
        """
        The third cumulant, ``E[(S - E[S])^3]``.
        """
    @property
    def variance(self, /) -> float:
        """
        ``Var[S]``.
        """

@final
class CompoundReport:
    """
    What a compound calculation produced and the error it introduced.
    
    Returned with the aggregate grid by ``panjer`` and ``fft``. Read
    ``aliasing_error`` first for FFT results: when it is not negligible the
    grid is unreliable, including ``tail_mass``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def aliasing_error(self, /) -> float:
        """
        Largest change in any probability when the FFT buffer doubles; 0 for
        Panjer.
        """
    @property
    def expected_mean(self, /) -> float:
        """
        ``E[N] * E[X]`` on the severity grid.
        """
    @property
    def grid_mean(self, /) -> float:
        """
        Mean of the aggregate grid.
        """
    def mean_error(self, /) -> float:
        """
        ``grid_mean - expected_mean``.
        
        Returns
        -------
        float
        """
    @property
    def method(self, /) -> str:
        """
        Method: ``"panjer"`` or ``"fft"``.
        """
    @property
    def points(self, /) -> int:
        """
        Number of points in the aggregate grid.
        """
    @property
    def tail_mass(self, /) -> float:
        """
        Aggregate probability above the last point, lumped onto it.
        """

@final
class Count:
    """
    A claim count from one of the families beyond the Poisson, negative
    binomial and binomial: zero-modified and zero-truncated counts, the
    logarithmic, mixed Poisson counts (negative binomial, Delaporte,
    Poisson-inverse Gaussian and its shifted version), Poisson-stopped sums
    (Neyman type A, Pólya-Aeppli) and empirical counts. These are
    ``aggregate``'s ``zm``, ``zt``, ``logarithmic``, ``mixed``,
    ``neymana`` and ``dfreq``. Make one with a static constructor; every
    aggregation function takes it like any other count.
    
    Examples
    --------
    >>> from prospicio.distributions import Count, Poisson
    >>> n = Count.zero_truncated(Poisson(2.0))
    >>> n.pmf(0)
    0.0
    >>> pig = Count.mixed_poisson(10.0, 0.5, mixing="inverse_gaussian")
    >>> round(pig.variance(), 9)
    35.0
    """
    def __repr__(self, /) -> str: ...
    @staticmethod
    def bernoulli(p: float) -> Count:
        """
        A Bernoulli count, one claim with probability ``p`` in ``[0, 1)``,
        ``aggregate``'s ``bernoulli``.
        
        Parameters
        ----------
        p : float
        
        Returns
        -------
        Count
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1)``.
        """
    def cdf(self, /, k: int) -> float:
        """
        ``P(N <= k)``.
        
        Parameters
        ----------
        k : int
        
        Returns
        -------
        float
        """
    @staticmethod
    def compound_poisson(rate: float, secondary: Any) -> Count:
        """
        A Poisson(``rate``) number of clusters, each a ``secondary`` count:
        Poisson secondaries give the Neyman type A.
        
        Parameters
        ----------
        rate : float
        secondary : Poisson, NegativeBinomial, Binomial or Count
        
        Returns
        -------
        Count
        """
    @staticmethod
    def empirical(probs: Sequence[float]) -> Count:
        """
        An empirical count, ``P(N = k) = probs[k]``.
        
        Parameters
        ----------
        probs : list of float
            Non-negative, summing to 1.
        
        Returns
        -------
        Count
        """
    @staticmethod
    def fixed(n: int) -> Count:
        """
        A fixed count, ``N = n``, ``aggregate``'s ``fixed``.
        
        Parameters
        ----------
        n : int
        
        Returns
        -------
        Count
        
        Examples
        --------
        >>> from prospicio.distributions import Count
        >>> n = Count.fixed(3)
        >>> n.mean(), n.variance()
        (3.0, 0.0)
        """
    @staticmethod
    def geometric(mean: float) -> Count:
        """
        A geometric count on ``0, 1, ...`` with this mean, a negative
        binomial with ``r = 1``, ``aggregate``'s ``geometric``.
        
        Parameters
        ----------
        mean : float
        
        Returns
        -------
        Count
        
        Raises
        ------
        ValueError
            If ``mean`` is not finite and positive.
        
        Examples
        --------
        >>> from prospicio.distributions import Count
        >>> round(Count.geometric(3.0).pmf(0), 12)
        0.25
        """
    @staticmethod
    def logarithmic(p: float) -> Count:
        """
        The logarithmic count on ``1, 2, ...``:
        ``P(N = k) = -p**k / (k log(1 - p))``.
        
        Parameters
        ----------
        p : float
            In ``(0, 1)``.
        
        Returns
        -------
        Count
        """
    def mean(self, /) -> float:
        """
        Mean of the claim count.
        
        Returns
        -------
        float
        """
    @staticmethod
    def mixed_poisson(mean: float, cv: float, mixing: str = "gamma", shift: float = 0.0) -> Count:
        """
        A Poisson count whose mean is ``mean * theta``, with the mixing
        variable ``theta = shift + (1 - shift) G`` of mean 1 and coefficient
        of variation ``cv``; variance ``mean + mean**2 cv**2``.
        
        Parameters
        ----------
        mean : float
        cv : float
            Coefficient of variation of ``theta``, fixed part included.
        mixing : {"gamma", "inverse_gaussian"}, default "gamma"
            The distribution of ``G``. Gamma gives the negative binomial
            (``shift = 0``) or the Delaporte; inverse Gaussian the
            Poisson-inverse Gaussian or its shifted version.
        shift : float, default 0.0
            The fixed part of ``theta``, in ``[0, 1)``.
        
        Returns
        -------
        Count
        """
    def panjer_ab(self, /) -> tuple[float, float] |None:
        """
        ``(a, b)`` of the ``(a, b, 1)`` class, or ``None`` when Panjer's
        recursion does not apply (use FFT).
        
        Returns
        -------
        tuple of (float, float) or None
        """
    def pmf(self, /, k: int) -> float:
        """
        ``P(N = k)``.
        
        Parameters
        ----------
        k : int
        
        Returns
        -------
        float
        """
    def quantile(self, /, p: float) -> int:
        """
        Smallest ``k`` with ``P(N <= k) >= p``.
        
        Parameters
        ----------
        p : float
        
        Returns
        -------
        int
        """
    @staticmethod
    def renewal(wait: Any, horizon: float, *, step: float |None = None, points: int |None = None, closed: bool = False) -> Count:
        """
        The number of claims by ``horizon`` when the waits between claims
        are independent draws from ``wait``: a renewal (Sparre Andersen)
        count, ``aggregate``'s ``years ... wait ...`` frequency. Exponential
        waits give the Poisson.
        
        The wait is put on a grid and the count read off by FFT; waits of
        zero arrive in clusters, and waits whose probabilities sum to less
        than 1 end the process. The grid is ``aggregate``'s by default.
        
        Parameters
        ----------
        wait : distribution or tuple of (list of float, list of float)
            The wait's distribution, or its values and their probabilities
            (``aggregate``'s ``dwait``), which may sum to less than 1.
        horizon : float
            The period, in the wait's units.
        step : float, optional
            The wait grid's step; give it with ``points``.
        points : int, optional
            The wait grid's number of points.
        closed : bool, default False
            With ``step``: whether waits summing to exactly ``horizon``
            count in full (waits on the grid's lattice), not half.
        
        Returns
        -------
        Count
        
        Examples
        --------
        >>> from prospicio.distributions import Count, Gamma
        >>> n = Count.renewal(Gamma(2.0, 1.0), 5.0)
        >>> round(n.pmf(0), 6)  # the first wait is beyond 5: 6 e^-5
        0.040428
        >>> round(Count.renewal(([1.0], [1.0]), 3.0).pmf(3), 12)  # waits of 1 by 3
        1.0
        """
    def sample(self, /, n: int, seed: int, stream: int = 0) -> list[int]:
        """
        ``n`` claim counts from stream ``stream`` of the generator keyed by
        ``seed``.
        
        Parameters
        ----------
        n : int
        seed : int
        stream : int, default 0
        
        Returns
        -------
        list of int
        """
    def variance(self, /) -> float:
        """
        Variance of the claim count.
        
        Returns
        -------
        float
        """
    @staticmethod
    def zero_modified(base: Any, p0: float) -> Count:
        """
        The base count with ``P(N = 0) = p0``; the other probabilities are
        scaled to sum to ``1 - p0``.
        
        Parameters
        ----------
        base : Poisson, NegativeBinomial, Binomial or Count
        p0 : float
            In ``[0, 1)``.
        
        Returns
        -------
        Count
        """
    @staticmethod
    def zero_truncated(base: Any) -> Count:
        """
        The base count conditioned on at least one claim.
        
        Parameters
        ----------
        base : Poisson, NegativeBinomial, Binomial or Count
        
        Returns
        -------
        Count
        """

@final
class Custom:
    """
    A loss severity defined by your own distribution function: the slow
    path for a distribution the library does not have.
    
    Give the cdf, and the quantile function if you have one (sampling
    inverts the cdf by bisection otherwise, about a hundred cdf calls per
    draw). The mean, variance, limited expected values and layer moments
    are computed by Gauss–Legendre quadrature of the survival function
    between the distribution's own quantiles, ignoring the probability
    above the ``1 - 1e-12`` quantile. A ``Custom`` goes anywhere a severity
    does (layers, compound distributions, simulated events, copula
    marginals, mixtures); calculations that meet one run single-threaded,
    since every value calls back into Python.
    
    Parameters
    ----------
    cdf : callable
        ``cdf(x) -> float``: ``P(X <= x)`` for ``x >= 0``, in ``[0, 1]``
        and non-decreasing. Losses are non-negative.
    quantile : callable, optional
        ``quantile(p) -> float``: the smallest ``x`` with ``cdf(x) >= p``.
    name : str, default "custom"
        Shown in errors and ``repr``.
    
    Raises
    ------
    ValueError
        If a callable raises or returns a value out of range, or the cdf
        never reaches ``1 - 1e-12`` at a finite loss. An error in a later
        call makes that value ``nan``; ``last_error`` says why.
    
    Examples
    --------
    >>> import math
    >>> from prospicio.distributions import Custom
    >>> d = Custom(lambda x: 1 - math.exp(-x / 100), name="exponential")
    >>> round(d.mean(), 6)
    100.0
    >>> round(d.lev(50), 6) == round(100 * (1 - math.exp(-0.5)), 6)
    True
    """
    def __getnewargs__(self, /) -> tuple[Any, Any |None, str]: ...
    def __new__(cls, /, cdf: Any, quantile: Any |None = None, name: str = "custom") -> Custom: ...
    def __repr__(self, /) -> str: ...
    def cdf(self, /, x: float) -> float:
        """
        Distribution function ``P(X <= x)``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @property
    def has_quantile(self, /) -> bool:
        """
        Whether a quantile function was given.
        """
    @property
    def last_error(self, /) -> str |None:
        """
        The first error raised by a callable since construction, or ``None``.
        """
    def layer(self, /, limit: float, attachment: float) -> float:
        """
        Expected loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
            ``inf`` for an unlimited layer.
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_second_moment(self, /, limit: float, attachment: float) -> float:
        """
        Second moment of the loss to the layer ``limit`` xs
        ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_variance(self, /, limit: float, attachment: float) -> float:
        """
        Variance of the loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def lev(self, /, limit: float) -> float:
        """
        Limited expected value ``E[min(X, limit)]``.
        
        Parameters
        ----------
        limit : float
        
        Returns
        -------
        float
        """
    def mean(self, /) -> float:
        """
        Mean of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """
    @property
    def name(self, /) -> str:
        """
        The name given at construction.
        """
    def quantile(self, /, p: float) -> float:
        """
        Quantile: the smallest ``x`` with ``P(X <= x) >= p``.
        
        Parameters
        ----------
        p : float
            Probability in ``[0, 1]``.
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def sample(self, /, n: int, seed: int, stream: int = 0) -> list[float]:
        """
        ``n`` draws from stream ``stream`` of the generator keyed by
        ``seed``.
        
        Parameters
        ----------
        n : int
        seed : int
        stream : int, default 0
        
        Returns
        -------
        list of float
        """
    def std(self, /) -> float:
        """
        Standard deviation of the distribution.
        
        Returns
        -------
        float
        """
    def stop_loss(self, /, retention: float) -> float:
        """
        Expected excess over a retention, ``E[max(X - retention, 0)]``.
        
        Parameters
        ----------
        retention : float
        
        Returns
        -------
        float
        """
    def survival(self, /, x: float) -> float:
        """
        Survival function ``P(X > x)``, accurate far into the tail.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @property
    def upper(self, /) -> float:
        """
        The ``1 - 1e-12`` quantile, where the moment integrals stop.
        """
    def variance(self, /) -> float:
        """
        Variance of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """

@final
class CvPath:
    """
    Cross-validated scores along an elastic-net path, from
    ``ElasticNet.cross_validate``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def fold_scores(self, /) -> list[list[float]]:
        """
        Each fold's mean deviance per ``lam``.
        """
    @property
    def lam_1se(self, /) -> float:
        """
        The largest ``lam`` within one standard error of the lowest.
        """
    @property
    def lam_min(self, /) -> float:
        """
        ``lam`` with the lowest mean deviance.
        """
    @property
    def lams(self, /) -> list[float]:
        """
        Penalty strengths, in the order given.
        """
    @property
    def mean(self, /) -> list[float]:
        """
        Mean deviance over folds (weighted by fold weight), per ``lam``.
        """
    @property
    def se(self, /) -> list[float]:
        """
        Standard error of ``mean``, per ``lam``.
        """

@final
class Design:
    """
    A design matrix with an offset and prior weights.
    
    Parameters
    ----------
    columns : list of list of float
        One list per column.
    names : list of str
    offset : list of float, optional
    weights : list of float, optional
    
    Examples
    --------
    >>> from prospicio.models import Design
    >>> d = Design([[1.0, 1.0], [0.0, 2.0]], ["(Intercept)", "x"])
    >>> d.n_rows, d.names
    (2, ['(Intercept)', 'x'])
    """
    def __new__(cls, /, columns: Sequence[Sequence[float]], names: Sequence[str], offset: Sequence[float] |None = None, weights: Sequence[float] |None = None) -> Design: ...
    def column(self, /, j: int) -> list[float]:
        """
        Column ``j``.
        
        Parameters
        ----------
        j : int
        
        Returns
        -------
        list of float
        """
    @property
    def n_rows(self, /) -> int:
        """
        Number of rows.
        """
    @property
    def names(self, /) -> list[str]:
        """
        Column names.
        """
    @property
    def offset(self, /) -> list[float]:
        """
        Offset per row.
        """
    def select(self, /, rows: Sequence[int]) -> Design:
        """
        The rows ``rows``, in that order.
        
        Parameters
        ----------
        rows : list of int
        
        Returns
        -------
        Design
        """
    @property
    def weights(self, /) -> list[float]:
        """
        Prior weight per row.
        """

@final
class DevelopmentPattern:
    """
    Development factors to ultimate by age.
    
    ``at(age)`` interpolates linearly in age between the given ages, takes
    the first factor at younger ages and the last at older ones. Use one to
    develop each known large loss to ultimate (IBNER) in
    ``LargeLossExperience.prepare``, or as a count pattern for the claims not
    yet reported in ``PreparedLosses.burning_cost``.
    
    Parameters
    ----------
    ages : list of float
        Strictly increasing ages, in years since the accident date.
    factors : list of float
        The factor to ultimate at each age.
    
    Examples
    --------
    >>> from prospicio.pricing import DevelopmentPattern
    >>> p = DevelopmentPattern.from_history(
    ...     ["a", "a", "b", "b"], [1.0, 2.0, 1.0, 2.0], [100.0, 150.0, 50.0, 50.0], ages=[1.0, 2.0]
    ... )
    >>> [round(f, 4) for f in p.factors]
    [1.3333, 1.0]
    >>> round(p.at(1.5), 4)
    1.1667
    """
    def __new__(cls, /, ages: Sequence[float], factors: Sequence[float]) -> DevelopmentPattern: ...
    def __repr__(self, /) -> str: ...
    @property
    def age_to_age(self, /) -> list[float]:
        """
        The age-to-age factors.
        """
    @property
    def ages(self, /) -> list[float]:
        """
        The ages.
        """
    def at(self, /, age: float) -> float:
        """
        The factor to ultimate at ``age``.
        
        Parameters
        ----------
        age : float
        
        Returns
        -------
        float
        """
    @property
    def factors(self, /) -> list[float]:
        """
        The factors to ultimate at each age.
        """
    @staticmethod
    def from_history(ids: Sequence[str], age: Sequence[float], amount: Sequence[float], *, ages: Sequence[float]) -> DevelopmentPattern:
        """
        Estimated from each claim's valuations.
        
        The age-to-age factor from ``ages[k]`` to ``ages[k + 1]`` is the sum
        of the amounts at the later age over the sum at the earlier, over the
        claims valued at both with a positive earlier amount (a claim first
        seen later is IBNR, not IBNER). The factor at the last age is 1. With
        every claim valued at every age up to its latest, this is the
        volume-weighted chain ladder on the summed triangle.
        
        Parameters
        ----------
        ids : list of str
            The claim of each valuation.
        age : list of float
            The claim's age at each valuation, in years; valuations at an age
            not in ``ages`` are ignored.
        amount : list of float
            The claim's amount at each valuation.
        ages : list of float
            The ages of the pattern.
        
        Returns
        -------
        DevelopmentPattern
        """

@final
class Discrete:
    """
    A distribution on finitely many values, of either sign and any spacing:
    a row of a P&L (premium less loss less expenses) or any other signed
    outcome, as the Python package ``aggregate``'s ``GridDistribution`` on an
    irregular support.
    
    Values may repeat and come in any order: equal values (to ``1e-12``) are
    merged, zero probabilities dropped, and the probabilities rescaled to
    sum to 1. Quantiles and TVaR follow ``aggregate`` exactly; ``tvar`` is
    the upper tail even when the values are a payoff, whose adverse tail is
    the low one.
    
    Parameters
    ----------
    values : list of float
    probs : list of float
        Non-negative, with a positive total.
    
    Raises
    ------
    ValueError
        If the lengths differ, a value is not finite, a probability is
        negative or none is positive.
    
    Examples
    --------
    >>> from prospicio.distributions import Discrete
    >>> # A margin: 9 less the loss of two claims of 1, 2 or 4.
    >>> m = Discrete([7, 6, 5, 4, 3, 1], [0.25, 0.3, 0.09, 0.2, 0.12, 0.04])
    >>> round(m.mean(), 12), m.quantile(0.01)
    (5.2, 1.0)
    >>> round(m.tvar(0.01), 6)
    5.242424
    """
    def __len__(self, /) -> int: ...
    def __new__(cls, /, values: Sequence[float], probs: Sequence[float]) -> Discrete: ...
    def __repr__(self, /) -> str: ...
    def cdf(self, /, x: float) -> float:
        """
        ``P(X <= x)``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    def mean(self, /) -> float:
        """
        The mean.
        
        Returns
        -------
        float
        """
    def pmf(self, /, x: float) -> float:
        """
        The probability of exactly ``x`` (to ``1e-12``).
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @property
    def probs(self, /) -> list[float]:
        """
        Their probabilities, summing to 1.
        """
    def quantile(self, /, p: float) -> float:
        """
        The lower quantile, the smallest value with ``F >= p``.
        
        Parameters
        ----------
        p : float
            In ``[0, 1]``.
        
        Returns
        -------
        float
        """
    def quantile_upper(self, /, p: float) -> float |None:
        """
        The upper quantile, the smallest value with ``F > p``; None at
        ``p = 1``.
        
        Parameters
        ----------
        p : float
            In ``[0, 1]``.
        
        Returns
        -------
        float or None
        """
    def skewness(self, /) -> float |None:
        """
        The skewness, or None when the variance is 0.
        
        Returns
        -------
        float or None
        """
    def std_dev(self, /) -> float:
        """
        The standard deviation.
        
        Returns
        -------
        float
        """
    def survival(self, /, x: float) -> float:
        """
        ``P(X > x)``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    def tail(self, /, periods: Sequence[float] |None = None) -> dict:
        """
        The return-period table, as ``aggregate``'s ``tail_df``: each period
        ``T`` read from both tails, at ``P = 1 / T`` and ``P = 1 - 1 / T``. A
        payoff (a P&L) is read off the low rows, a loss off the high ones.
        
        Parameters
        ----------
        periods : list of float, optional
            At least 1; ``aggregate``'s 2, 5, 10, 25, 50, 100, 200, 250, 500,
            1000 by default.
        
        Returns
        -------
        dict
            Columns ``p``, ``period``, ``var``, ``tvar``, ``excess`` (VaR less
            the mean) and ``var_to_mean`` (None near a zero mean).
        """
    def tvar(self, /, p: float) -> float:
        """
        Tail value at risk at ``p``: the mean of the upper ``1 - p`` of the
        distribution, the upper tail even for a payoff.
        
        Parameters
        ----------
        p : float
            In ``[0, 1]``.
        
        Returns
        -------
        float
        """
    @property
    def values(self, /) -> list[float]:
        """
        The distinct values, ascending.
        """
    def variance(self, /) -> float:
        """
        The variance.
        
        Returns
        -------
        float
        """

@final
class DiscretizationReport:
    """
    How a distribution was discretized, and the error that introduced.
    
    Returned with the grid by ``Grid.local_moment``, ``Grid.rounding`` and
    ``Grid.lower``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def grid_mean(self, /) -> float:
        """
        Mean of the grid.
        """
    def mean_error(self, /) -> float:
        """
        ``grid_mean - source_mean``.
        
        Returns
        -------
        float
        """
    @property
    def method(self, /) -> str:
        """
        Method: ``"local_moment"``, ``"rounding"`` or ``"lower"``.
        """
    @property
    def points(self, /) -> int:
        """
        Number of grid points.
        """
    @property
    def source_mean(self, /) -> float:
        """
        Mean of the source distribution.
        """
    @property
    def step(self, /) -> float:
        """
        Grid step.
        """
    @property
    def tail_mass(self, /) -> float:
        """
        Probability the source puts above the last grid point, lumped onto it.
        """

@final
class Distortion:
    """
    A distortion risk measure: ``rho(X) = integral of g(S(x)) dx`` for a
    concave distortion ``g`` of the survival function.
    
    Make one with ``Distortion.tvar``, ``Distortion.wang``,
    ``Distortion.proportional_hazard``, ``Distortion.dual_power``,
    ``Distortion.exponential``, ``Distortion.ccoc`` (constant cost of
    capital), ``Distortion.bitvar``, ``Distortion.weighted_tvar``,
    ``Distortion.capped_linear``, ``Distortion.capped_log_linear``,
    ``Distortion.lep``, ``Distortion.linear_yield``, ``Distortion.beta``,
    ``Distortion.mixture``, ``Distortion.minimum`` or ``Distortion.convex``.
    Every one is concave, so every measure is coherent. ``calibrate``
    solves for the parameter that gives a target price.
    
    Examples
    --------
    >>> from prospicio.distributions import Sampled
    >>> from prospicio.risk import Distortion
    >>> x = Sampled([1.0, 2.0, 3.0, 4.0])
    >>> Distortion.tvar(0.5).measure(x)
    3.5
    >>> Distortion.tvar(0.5).weights(4)
    [0.0, 0.0, 0.5, 0.5]
    """
    def __repr__(self, /) -> str: ...
    @staticmethod
    def beta(a: float, b: float) -> Distortion:
        """
        The Beta(a, b) distribution function, concave for ``a <= 1 <= b``.
        
        Parameters
        ----------
        a : float
            In ``(0, 1]``.
        b : float
            ``>= 1``.
        
        Returns
        -------
        Distortion
        """
    @staticmethod
    def bitvar(p0: float, p1: float, w: float) -> Distortion:
        """
        Weight ``1 - w`` on ``TVaR(p0)`` and ``w`` on ``TVaR(p1)``.
        
        Parameters
        ----------
        p0, p1 : float
            Levels with ``0 <= p0 <= p1 <= 1``.
        w : float
            Weight on ``TVaR(p1)``, in ``[0, 1]``.
        
        Returns
        -------
        Distortion
        """
    @staticmethod
    def capped_linear(r0: float, slope: float) -> Distortion:
        """
        Capped linear: ``g(s) = min(1, r0 + slope s)`` for ``s > 0``.
        
        Parameters
        ----------
        r0 : float
            Mass on the largest outcome, in ``[0, 1)``.
        slope : float
            With ``r0 + slope >= 1``.
        
        Returns
        -------
        Distortion
        """
    @staticmethod
    def capped_log_linear(r0: float, b: float) -> Distortion:
        """
        Capped log-linear: ``g(s) = min(1, exp(r0) s**b)``.
        
        Parameters
        ----------
        r0 : float
            ``>= 0``.
        b : float
            In ``(0, 1]``.
        
        Returns
        -------
        Distortion
        """
    @staticmethod
    def ccoc(r: float) -> Distortion:
        """
        Constant cost of capital ``r``: ``g(s) = min(1, d + (1 - d) s)`` for
        ``s > 0``, ``d = r / (1 + r)``. The price is ``(E[X] + r max X) / (1 + r)``.
        
        Parameters
        ----------
        r : float
            Return on capital, ``>= 0``.
        
        Returns
        -------
        Distortion
        """
    @staticmethod
    def convex(points: Sequence[tuple[float, float]]) -> Distortion:
        """
        The smallest concave distortion above the points ``(s, g)``, for
        example layers' exceedance probabilities and their prices per unit
        of limit (a cat bond's expected loss and spread).
        
        Parameters
        ----------
        points : list of (float, float)
            In the unit square, with ``g >= s``.
        
        Returns
        -------
        Distortion
        
        Examples
        --------
        >>> from prospicio.risk import Distortion
        >>> Distortion.convex([(0.1, 0.3), (0.5, 0.55)]).g(0.05)
        0.15
        """
    @staticmethod
    def dual_power(beta: float) -> Distortion:
        """
        Dual power transform: ``g(s) = 1 - (1 - s)**beta``.
        
        Parameters
        ----------
        beta : float
            ``>= 1``.
        
        Returns
        -------
        Distortion
        """
    @staticmethod
    def exponential(k: float) -> Distortion:
        """
        Exponential spectral measure: ``g(s) = (1 - exp(-k s)) / (1 - exp(-k))``,
        risk aversion that grows exponentially towards the worst outcomes.
        
        Parameters
        ----------
        k : float
            Risk aversion, positive; the mean as ``k -> 0``.
        
        Returns
        -------
        Distortion
        """
    @property
    def family(self, /) -> str:
        """
        The family, as its constructor is named, for example
        ``"proportional_hazard"`` or ``"capped_linear"``.
        """
    def g(self, /, s: float) -> float:
        """
        The distortion ``g(s)`` of a survival probability ``s``.
        
        Parameters
        ----------
        s : float
        
        Returns
        -------
        float
        """
    def g_dual(self, /, s: float) -> float:
        """
        The dual distortion ``1 - g(1 - s)``, which gives the bid.
        
        Parameters
        ----------
        s : float
        
        Returns
        -------
        float
        """
    def g_inv(self, /, y: float) -> float:
        """
        The smallest ``s`` with ``g(s) >= y``.
        
        Parameters
        ----------
        y : float
        
        Returns
        -------
        float
        """
    @staticmethod
    def lep(r0: float, r: float) -> Distortion:
        """
        Leverage-equivalent pricing:
        ``g(s) = min(1, d + (1 - d) s + (delta - d) sqrt(s (1 - s)))`` with
        ``d = r0 / (1 + r0)`` and ``delta = r / (1 + r)``.
        
        Parameters
        ----------
        r0 : float
            Minimum rate on line, ``>= 0``.
        r : float
            Target return, ``>= r0``.
        
        Returns
        -------
        Distortion
        """
    @staticmethod
    def linear_yield(r0: float, r: float) -> Distortion:
        """
        Linear yield: ``g(s) = (r0 + (1 + r) s) / (1 + r0 + r s)``.
        
        Parameters
        ----------
        r0, r : float
            Both ``>= 0``.
        
        Returns
        -------
        Distortion
        """
    @property
    def mass(self, /) -> float:
        """
        The probability mass on the largest outcome, ``g(0+)``.
        """
    def measure(self, /, dist: Any) -> float:
        """
        The risk measure of a distribution.
        
        Parameters
        ----------
        dist : Sampled, Grid or PredictiveDistribution
            A predictive distribution is measured on its total.
        
        Returns
        -------
        float
        """
    @staticmethod
    def minimum(distortions: Sequence[Distortion]) -> Distortion:
        """
        The pointwise minimum of distortions.
        
        Parameters
        ----------
        distortions : list of Distortion
        
        Returns
        -------
        Distortion
        """
    @staticmethod
    def mixture(distortions: Sequence[Distortion], weights: Sequence[float]) -> Distortion:
        """
        A weighted average of distortions.
        
        Parameters
        ----------
        distortions : list of Distortion
        weights : list of float
            Non-negative, summing to 1.
        
        Returns
        -------
        Distortion
        """
    @property
    def parameters(self, /) -> dict[str, float]:
        """
        The scalar parameters by name, as the constructor takes them, for
        example ``{"rho": 0.7}``; empty for weighted TVaR, mixtures, minimums
        and convex envelopes, whose inputs are lists.
        
        Examples
        --------
        >>> from prospicio.risk import Distortion
        >>> d = Distortion.proportional_hazard(0.7)
        >>> d.family, d.parameters
        ('proportional_hazard', {'rho': 0.7})
        """
    @staticmethod
    def proportional_hazard(rho: float) -> Distortion:
        """
        Proportional hazard transform: ``g(s) = s**rho``.
        
        Parameters
        ----------
        rho : float
            In ``(0, 1]``.
        
        Returns
        -------
        Distortion
        """
    @staticmethod
    def tvar(p: float) -> Distortion:
        """
        Tail value at risk at level ``p``: ``g(s) = min(s / (1 - p), 1)``.
        
        Parameters
        ----------
        p : float
            In ``[0, 1]``.
        
        Returns
        -------
        Distortion
        """
    @staticmethod
    def wang(lam: float) -> Distortion:
        """
        Wang transform: ``g(s) = Phi(Phi^-1(s) + lambda)``.
        
        Parameters
        ----------
        lam : float
            Market price of risk, ``>= 0``.
        
        Returns
        -------
        Distortion
        """
    @staticmethod
    def weighted_tvar(ps: Sequence[float], wts: Sequence[float]) -> Distortion:
        """
        A weighted average of TVaRs.
        
        Parameters
        ----------
        ps : list of float
            Strictly ascending levels in ``[0, 1]``.
        wts : list of float
            Non-negative weights summing to 1.
        
        Returns
        -------
        Distortion
        """
    def weights(self, /, n: int) -> list[float]:
        """
        Weights for ``n`` equally likely values sorted ascending.
        
        Parameters
        ----------
        n : int
        
        Returns
        -------
        list of float
            Non-negative, summing to 1.
        """

@final
class ElasticNet:
    """
    An elastic-net GLM: the lasso (``alpha=1``), ridge (``alpha=0``) and
    everything between, minimizing glmnet's objective
    ``sum(w * d) / (2 * sum(w)) + lam * sum(pf * ((1 - alpha) / 2 * b**2 + alpha * |b|))``
    over coefficients ``b`` of standardized columns. The design's first
    all-ones column is the unpenalized intercept; coefficients are reported
    on the design's scale.
    
    Parameters
    ----------
    family : str
        As ``Glm``.
    link : str, optional
        As ``Glm``; the canonical link by default.
    alpha : float, default 1.0
        Mixing between ridge (0) and the lasso (1).
    lam : float, default 0.0
        Penalty strength (``lambda`` in glmnet).
    standardize : bool, default True
        Penalize the coefficients of columns scaled to unit standard
        deviation.
    penalty_factor : list of float, optional
        One factor per design column (the intercept's is ignored); 0 leaves
        a column unpenalized.
    theta : float, optional
    power : float, optional
    link_power : float, optional
    
    Examples
    --------
    >>> from prospicio.models import Design, ElasticNet
    >>> x = [float(i) for i in range(6)]
    >>> d = Design([[1.0] * 6, x, [1.0, 0.0] * 3], ["(Intercept)", "x1", "x2"])
    >>> y = [1.0, 3.1, 4.9, 7.2, 9.0, 10.8]
    >>> net = ElasticNet("gaussian", alpha=1.0)
    >>> top = net.lambda_max(d, y)
    >>> net.with_lam(1.01 * top).fit(d, y).coefficients[1:]
    [0.0, 0.0]
    """
    def __new__(cls, /, family: str, link: str |None = None, alpha: float = 1.0, lam: float = 0.0, standardize: bool = True, penalty_factor: Sequence[float] |None = None, theta: float |None = None, power: float |None = None, link_power: float |None = None) -> ElasticNet: ...
    def __repr__(self, /) -> str: ...
    @property
    def alpha(self, /) -> float:
        """
        Mixing parameter.
        """
    def cross_validate(self, /, design: Design, y: Sequence[float], lams: Sequence[float], splits: Sequence[tuple[Sequence[int], Sequence[int]]]) -> CvPath:
        """
        Cross-validates the path, like glmnet's ``cv.glmnet``: on each
        split, fits ``lams`` (warm starts) to the training rows and scores
        the mean deviance on the test rows. Folds run in parallel.
        
        Parameters
        ----------
        design : Design
        y : list of float
        lams : list of float
            Penalty strengths, largest first (from ``lambda_path``).
        splits : list of (list of int, list of int)
            ``(train, test)`` rows, as ``k_fold`` returns.
        
        Returns
        -------
        CvPath
        """
    def fit(self, /, design: Design, y: Sequence[float]) -> ElasticNetFit:
        """
        Fits at ``lam``.
        
        Parameters
        ----------
        design : Design
        y : list of float
        
        Returns
        -------
        ElasticNetFit
        
        Raises
        ------
        ValueError
            If a parameter is out of range, a response is outside the
            family's range, or the fit does not converge.
        """
    @property
    def lam(self, /) -> float:
        """
        Penalty strength.
        """
    def lambda_max(self, /, design: Design, y: Sequence[float]) -> float:
        """
        The smallest ``lam`` at which every penalized coefficient is zero.
        
        Parameters
        ----------
        design : Design
        y : list of float
        
        Returns
        -------
        float
        """
    def lambda_path(self, /, design: Design, y: Sequence[float], n: int = 100, min_ratio: float = 1e-4) -> list[float]:
        """
        ``n`` penalty strengths, log-spaced from ``lambda_max`` down to
        ``min_ratio`` times it.
        
        Parameters
        ----------
        design : Design
        y : list of float
        n : int, default 100
        min_ratio : float, default 1e-4
        
        Returns
        -------
        list of float
        """
    def path(self, /, design: Design, y: Sequence[float], lams: Sequence[float]) -> list[ElasticNetFit]:
        """
        Fits at each of ``lams`` in turn, each from the previous solution.
        
        Parameters
        ----------
        design : Design
        y : list of float
        lams : list of float
        
        Returns
        -------
        list of ElasticNetFit
        """
    def with_lam(self, /, lam: float) -> ElasticNet:
        """
        The same spec at another penalty strength.
        
        Parameters
        ----------
        lam : float
        
        Returns
        -------
        ElasticNet
        """

@final
class ElasticNetFit:
    """
    A fitted elastic net, from ``ElasticNet.fit`` or ``ElasticNet.path``.
    """
    def __reduce__(self, /) -> tuple[Any, tuple[str]]: ...
    def __repr__(self, /) -> str: ...
    @property
    def coefficients(self, /) -> list[float]:
        """
        Coefficients on the design's scale; exact zeros where the penalty
        dropped a column.
        """
    @property
    def deviance(self, /) -> float:
        """
        Residual deviance.
        """
    @property
    def deviance_ratio(self, /) -> float:
        """
        Share of the null deviance explained (glmnet's ``dev.ratio``).
        """
    @property
    def df(self, /) -> int:
        """
        Number of non-zero coefficients, intercept excluded.
        """
    @property
    def dispersion(self, /) -> float:
        """
        Dispersion.
        """
    @property
    def fitted(self, /) -> list[float]:
        """
        Fitted means on the training data.
        """
    @staticmethod
    def from_json(text: str) -> ElasticNetFit:
        """
        Reads an artifact written by ``to_json``.
        
        Parameters
        ----------
        text : str
        
        Returns
        -------
        ElasticNetFit
        
        Raises
        ------
        ValueError
            For malformed JSON, another format, a newer format version or
            inconsistent fields.
        """
    @property
    def input_hash(self, /) -> str:
        """
        Hash of the training data (design, offset, weights, response).
        """
    @property
    def lam(self, /) -> float:
        """
        Penalty strength.
        """
    @property
    def names(self, /) -> list[str]:
        """
        Coefficient names.
        """
    @property
    def null_deviance(self, /) -> float:
        """
        Deviance with only the intercept and unpenalized columns.
        """
    def predict(self, /, design: Design) -> list[float]:
        """
        Expected response for each row.
        
        Parameters
        ----------
        design : Design
            Same columns as the training design.
        
        Returns
        -------
        list of float
        """
    def predict_distribution(self, /, design: Design, n_sims: int, seed: int) -> PredictiveDistribution:
        """
        Joint predictive distribution across the rows, keyed
        ``row = 0, 1, ...``: process uncertainty only (penalized
        coefficients have no standard errors; bootstrap the fit for
        parameter uncertainty).
        
        Parameters
        ----------
        design : Design
        n_sims : int
        seed : int
        
        Returns
        -------
        PredictiveDistribution
        """
    def to_json(self, /) -> str:
        """
        The fit as a versioned JSON artifact (spec with its lambda and alpha, coefficients, fit statistics), with provenance (crate
        version and a hash of the training data). ``ElasticNetFit.from_json`` reads
        it back exactly; pickling uses it too.
        
        Returns
        -------
        str
        """

@final
class Elpd:
    """
    An ELPD estimate from ``elpd_loo`` or ``elpd_waic``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def elpd(self, /) -> float:
        """
        Expected log pointwise predictive density, summed.
        """
    @property
    def ic(self, /) -> float:
        """
        The information criterion, ``-2 elpd`` (LOOIC or WAIC).
        """
    @property
    def k_threshold(self, /) -> float |None:
        """
        PSIS-LOO only: ``min(1 - 1/log10(S), 0.7)``; observations with a
        larger ``pareto_k`` are unreliable.
        """
    @property
    def p(self, /) -> float:
        """
        Effective number of parameters, ``lppd - elpd``.
        """
    @property
    def pareto_k(self, /) -> list[float] |None:
        """
        PSIS-LOO only: the fitted Pareto shape per observation.
        """
    @property
    def pointwise(self, /) -> list[float]:
        """
        ELPD per observation.
        """
    @property
    def se(self, /) -> float:
        """
        Its standard error, ``sqrt(N var(pointwise))``.
        """

@final
class EventLossTable:
    """
    A catastrophe model's event loss table with secondary uncertainty.
    
    Each row is an event with an annual rate, a mean loss, independent and
    correlated standard deviations, and an exposure, the most it can lose.
    Events occur as independent Poisson processes; an event's loss, given
    that it occurs, is a beta distribution on ``[0, exposure]`` with the
    row's mean and standard deviation ``sd_independent + sd_correlated``
    (summed, as catastrophe models report them for one portfolio), or the
    mean exactly when that is 0. A row whose standard deviation no beta on
    its exposure can have is an error naming the row.
    
    Parameters
    ----------
    rate, mean, sd_independent, sd_correlated, exposure : list of float
        One value per event, as the table's columns.
    
    Examples
    --------
    >>> from prospicio.aggregate import EventLossTable
    >>> elt = EventLossTable([0.01, 0.05], [800.0, 200.0], [200.0, 0.0], [100.0, 0.0], [2000.0, 500.0])
    >>> round(elt.rate, 12), round(elt.expected_annual_loss, 12)
    (0.06, 18.0)
    >>> events, ids = elt.simulate(1000, seed=1)
    >>> events.n_sims
    1000
    """
    def __len__(self, /) -> int: ...
    def __new__(cls, /, rate: Sequence[float], mean: Sequence[float], sd_independent: Sequence[float], sd_correlated: Sequence[float], exposure: Sequence[float]) -> EventLossTable: ...
    def __repr__(self, /) -> str: ...
    def exceedance_frequency(self, /, x: float) -> float:
        """
        The annual frequency of events losing more than ``x``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @property
    def expected_annual_loss(self, /) -> float:
        """
        The expected annual loss, the sum of rate times mean.
        """
    def occurrence_pml(self, /, p: float) -> float:
        """
        The occurrence PML at annual probability ``p``: the loss whose
        occurrence exceedance probability is ``p``.
        
        Parameters
        ----------
        p : float
        
        Returns
        -------
        float
        """
    def oep(self, /, x: float) -> float:
        """
        The occurrence exceedance probability at ``x``,
        ``1 - exp(-exceedance_frequency(x))``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @property
    def rate(self, /) -> float:
        """
        The mean number of events a year, the rates' sum.
        """
    def simulate(self, /, n_years: int, seed: int = 0) -> tuple[EventSet, list[list[int]]]:
        """
        Simulated years of events, with each loss's event (row index).
        
        Year ``i`` draws from its own stream of the seed: a Poisson count,
        then for each event the row by its rate and its loss, so any year
        replays alone.
        
        Parameters
        ----------
        n_years : int
        seed : int, default 0
        
        Returns
        -------
        tuple of (EventSet, list of list of int)
        """

@final
class EventSet:
    """
    Simulated years of individual losses, for applying per-loss terms such
    as reinsurance layers.
    
    Created by ``simulate_events``. Year ``i`` was drawn from stream ``i`` of
    the generator keyed by ``seed``, so results do not depend on the number of
    threads.
    """
    def __repr__(self, /) -> str: ...
    def counts(self, /) -> list[int]:
        """
        Number of losses in each year.
        
        Returns
        -------
        list of int
        """
    def events(self, /, sim: int) -> list[float]:
        """
        Year ``sim``'s individual losses, in the order they were drawn.
        
        Parameters
        ----------
        sim : int
        
        Returns
        -------
        list of float
        
        Raises
        ------
        IndexError
            If ``sim`` is not a simulated year.
        """
    @staticmethod
    def from_long(years: Sequence[int], losses: Sequence[float], n_years: int, seed: int = 0) -> EventSet:
        """
        Years of losses in long form, as a year loss table lists them: one
        row per loss with its year, an index from 0 to ``n_years - 1``.
        
        Within a year the losses keep their order; years with no row are
        empty.
        
        Parameters
        ----------
        years : list of int
        losses : list of float
        n_years : int
            The number of simulated years, including those with no loss.
        seed : int, default 0
        
        Returns
        -------
        EventSet
        
        Examples
        --------
        >>> from prospicio.aggregate import EventSet
        >>> e = EventSet.from_long([2, 0, 2], [5.0, 1.0, 7.0], 4)
        >>> e.counts(), e.events(2)
        ([1, 0, 2, 0], [5.0, 7.0])
        """
    @staticmethod
    def from_years(years: Sequence[Sequence[float]], sums_insured: Sequence[Sequence[float]] |None = None, seed: int = 0, times: Sequence[Sequence[float]] |None = None) -> EventSet:
        """
        Years of losses from elsewhere (your own simulation, or a
        catastrophe model's event loss table by year), optionally with the
        sum insured of the risk each loss hit, which a surplus treaty needs.
        
        Parameters
        ----------
        years : list of list of float
            Each year's losses, in order.
        sums_insured : list of list of float, optional
            The same shape: each loss's sum insured, at least the loss.
        seed : int, default 0
            Recorded in results' provenance; the samplers are not, since
            the losses were not drawn here.
        times : list of list of float, optional
            The same shape: each loss's time, as the fraction of the year
            elapsed (in ``[0, 1]``, non-decreasing within a year), which
            reinstatements pro rata as to time need.
        
        Returns
        -------
        EventSet
        
        Examples
        --------
        >>> from prospicio.aggregate import EventSet
        >>> e = EventSet.from_years([[5.0, 2.0], [], [9.0]], [[10.0, 2.0], [], [50.0]])
        >>> e.counts(), e.sums_insured(2)
        ([2, 0, 1], [50.0])
        """
    def grouped(self, /, window_hours: float) -> EventSet:
        """
        Each year's losses grouped into occurrences by an hours clause.
        
        Losses within ``window_hours`` of the first loss of a window are one
        occurrence, whose loss is their sum and whose time is the window's
        start; the next loss after the window opens the next one. A year is
        8,760 hours. Windows open at each first loss in turn, not where they
        would recover most. Needs times; sums insured are not carried.
        
        Parameters
        ----------
        window_hours : float
            Such as 72 for a windstorm, 168 for a flood.
        
        Returns
        -------
        EventSet
        
        Examples
        --------
        >>> from prospicio.aggregate import EventSet
        >>> h = 1 / 8760
        >>> e = EventSet.from_years([[10.0, 20.0, 5.0]], times=[[0.5, 0.5 + 30 * h, 0.5 + 80 * h]])
        >>> e.grouped(72.0).events(0)
        [30.0, 5.0]
        """
    @property
    def has_sums_insured(self, /) -> bool:
        """
        Whether the losses carry sums insured.
        """
    @property
    def has_times(self, /) -> bool:
        """
        Whether the losses carry times.
        """
    def maxima(self, /) -> PredictiveDistribution:
        """
        Each year's largest loss (0 in a year with none): its survival
        function is the simulated occurrence exceedance (OEP) curve, and its
        quantiles the occurrence PMLs.
        
        Returns
        -------
        PredictiveDistribution
        """
    def merge(self, /, other: EventSet) -> EventSet:
        """
        Both sets' losses in each year: a catastrophe event set with a
        non-catastrophe one, say.
        
        The sets need the same number of years. With times on both, each
        year's losses are merged in time order (this set's first on a tie);
        with times on neither, this set's losses come first. Sums insured are
        kept when both sets carry them.
        
        Parameters
        ----------
        other : EventSet
        
        Returns
        -------
        EventSet
        
        Examples
        --------
        >>> from prospicio.aggregate import EventSet
        >>> cat = EventSet.from_years([[100.0], []], times=[[0.5], []])
        >>> other = EventSet.from_years([[3.0, 4.0], [5.0]], times=[[0.2, 0.7], [0.1]])
        >>> cat.merge(other).events(0)
        [3.0, 100.0, 4.0]
        """
    @property
    def n_sims(self, /) -> int:
        """
        Number of simulated years.
        """
    @property
    def seed(self, /) -> int:
        """
        The seed the years were drawn from.
        """
    def sums_insured(self, /, sim: int) -> list[float] |None:
        """
        Year ``sim``'s sums insured, one per loss, or ``None``.
        
        Parameters
        ----------
        sim : int
        
        Returns
        -------
        list of float or None
        """
    def times(self, /, sim: int) -> list[float] |None:
        """
        Year ``sim``'s times, one per loss, or ``None``.
        
        Parameters
        ----------
        sim : int
        
        Returns
        -------
        list of float or None
        """
    def totals(self, /) -> PredictiveDistribution:
        """
        Each year's total loss.
        
        Returns
        -------
        PredictiveDistribution
        """
    def with_seasonal_times(self, /, weights: Sequence[float]) -> EventSet:
        """
        The same events at times drawn from a seasonal density.
        
        The year is cut into ``len(weights)`` equal periods (12 for months,
        52 for weeks) starting at the contract's inception, and a loss
        falls in period ``k`` with probability ``weights[k] / sum(weights)``,
        uniformly within it. A zero weight means no losses in that period.
        The draws are those of ``with_uniform_times``, mapped through the
        season's quantile, so equal weights give the uniform times.
        
        Parameters
        ----------
        weights : list of float
            Each period's relative weight: non-negative, not all zero.
        
        Returns
        -------
        EventSet
        
        Examples
        --------
        >>> from prospicio.aggregate import EventSet
        >>> e = EventSet.from_years([[5.0, 2.0, 7.0]], seed=3)
        >>> t = e.with_seasonal_times([0.0, 1.0]).times(0)
        >>> all(x >= 0.5 for x in t) and t == sorted(t)
        True
        """
    def with_uniform_times(self, /) -> EventSet:
        """
        The same events at times spread uniformly over the year.
        
        Year ``i``'s losses take sorted uniform draws, in their order, from
        a stream of the generator keyed by the set's seed apart from the
        losses' own, so the losses are unchanged and any year replays alone.
        
        Returns
        -------
        EventSet
        
        Examples
        --------
        >>> from prospicio.aggregate import EventSet
        >>> e = EventSet.from_years([[5.0, 2.0, 7.0]], seed=3).with_uniform_times()
        >>> t = e.times(0)
        >>> t == sorted(t) and e.has_times
        True
        """

@final
class ExpectedLoss:
    """
    The expected loss ratio method: each origin's ultimate is ``apriori``
    times its exposure, whatever has been observed. The chain ladder is
    still fitted for the development pattern the fit reports.
    
    The exposure is a measure column of the same triangle (premium, say):
    each origin's latest observed cumulative value in the segment fitted.
    
    Parameters
    ----------
    apriori : float, default 1.0
        Expected loss ratio: the ultimate per unit of exposure; positive.
    average : {"volume", "simple", "regression"}, default "volume"
        How link ratios are averaged, as in ``ChainLadder``.
    sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
    tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
        As ``ChainLadder``; no tail by default.
    
    Examples
    --------
    >>> from prospicio.reserving import ExpectedLoss, Triangle
    >>> tri = Triangle.from_long(
    ...     [2020, 2020, 2021], [12, 24, 12],
    ...     {"paid": [100.0, 150.0, 200.0], "premium": [250.0, 250.0, 400.0]},
    ... )
    >>> fit = ExpectedLoss(apriori=0.5).fit(tri, "paid", "premium")
    >>> fit.ultimate, fit.reserve
    ([125.0, 200.0], [-25.0, 0.0])
    """
    def __new__(cls, /, apriori: float = 1.0, average: str = "volume", sigma_interpolation: str = "log-linear", tail: Any |None = None) -> ExpectedLoss: ...
    def __repr__(self, /) -> str: ...
    @property
    def apriori(self, /) -> float:
        """
        Expected loss ratio.
        """
    @property
    def average(self, /) -> str:
        """
        How link ratios are averaged.
        """
    def fit(self, /, triangle: Triangle, column: str, exposure: str) -> ExpectedLossFit:
        """
        Fits one loss column in every segment of a triangle, each with its
        own exposure.
        
        Parameters
        ----------
        triangle : Triangle
        column : str
            The losses to project.
        exposure : str
            The exposure column; each origin's latest observed cumulative
            value is its exposure (an incremental triangle's is cumulated).
        
        Returns
        -------
        ExpectedLossFit
        
        Raises
        ------
        ValueError
            As ``ChainLadder.fit``, if ``apriori`` is not positive, or if an
            origin has no observed, finite, positive exposure (the message
            names it and, with keys, its segment).
        """
    @property
    def sigma_interpolation(self, /) -> str:
        """
        How unestimable variance parameters are filled in.
        """
    @property
    def tail(self, /) -> Any:
        """
        The tail: a constant factor as a number, otherwise its estimator.
        """

@final
class ExpectedLossFit:
    """
    A fitted expected-loss method (``ExpectedLoss``,
    ``BornhuetterFerguson`` or ``Benktander``) of every segment of a
    triangle column.
    
    Per-origin lists (``origins``, ``latest``, ``exposure``, ``apriori``,
    ``ultimate``, ``reserve``) run over the origins of each segment in turn,
    like the rows of ``to_frame()``. ``ultimate`` and ``reserve`` are this
    method's; ``chain_ladder`` holds the chain ladder's. Per-age lists need
    a single-segment fit; for several segments use ``development_frame()``
    or ``segment(...)``.
    
    Examples
    --------
    >>> from prospicio.reserving import BornhuetterFerguson, Triangle
    >>> tri = Triangle.from_long(
    ...     [2020, 2020, 2021] * 2,
    ...     [12, 24, 12] * 2,
    ...     {"paid": [100.0, 150.0, 200.0, 10.0, 20.0, 30.0],
    ...      "premium": [250.0, 250.0, 400.0, 500.0, 500.0, 800.0]},
    ...     keys={"lob": ["Auto"] * 3 + ["Home"] * 3},
    ... )
    >>> fit = BornhuetterFerguson(apriori=0.5).fit(tri, "paid", "premium")
    >>> fit.exposure, fit.segment(lob="Home").ultimate
    ([250.0, 400.0, 500.0, 800.0], [20.0, 230.0])
    """
    def __repr__(self, /) -> str: ...
    @property
    def apriori(self, /) -> list[float]:
        """
        Expected loss ratio applied per origin.
        """
    @property
    def cdf(self, /) -> list[float]:
        """
        Age-to-ultimate factors, one per age, including the tail.
        """
    @property
    def chain_ladder(self, /) -> ChainLadderFit:
        """
        The underlying chain-ladder projection, with the chain ladder's
        ultimate.
        """
    @property
    def development(self, /) -> list[int]:
        """
        Development ages in months.
        """
    def development_frame(self, /) -> Any:
        """
        One row per segment and age, as ``ChainLadderFit.development_frame``.
        Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def exposure(self, /) -> list[float]:
        """
        Exposure per origin: the exposure column's latest observed cumulative
        value.
        """
    @property
    def index(self, /) -> list[Any]:
        """
        Label of each segment, as ``Triangle.index``.
        """
    @property
    def keys(self, /) -> list[str]:
        """
        Names of the triangle's key columns; empty without keys.
        """
    @property
    def latest(self, /) -> list[float]:
        """
        Latest observed cumulative value per origin.
        """
    @property
    def ldf(self, /) -> list[float]:
        """
        Age-to-age factors; factor ``k`` links age ``k`` to ``k + 1``.
        """
    @property
    def origins(self, /) -> list[str]:
        """
        Origin period of each per-origin value.
        """
    @property
    def reserve(self, /) -> list[float]:
        """
        Reserve (ultimate minus latest) per origin.
        """
    def segment(self, /, **keys) -> ExpectedLossFit:
        """
        The fit of one segment, chosen by key values as
        ``ChainLadderFit.segment``.
        
        Returns
        -------
        ExpectedLossFit
        """
    def to_frame(self, /) -> Any:
        """
        One row per segment and origin: the key columns, ``origin``,
        ``latest``, ``ultimate``, ``reserve``, ``exposure`` and ``apriori``.
        Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def total_reserve(self, /) -> float:
        """
        Total reserve across segments and origins.
        """
    @property
    def total_ultimate(self, /) -> float:
        """
        Total ultimate across segments and origins.
        """
    def totals_frame(self, /) -> Any:
        """
        One row per segment: the key columns and the segment's total
        ``latest``, ``ultimate``, ``reserve`` and ``exposure``. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def ultimate(self, /) -> list[float]:
        """
        This method's ultimate per origin.
        """

@final
class Expense:
    """
    One expense leg of a :class:`Book`: an amount (``fixed``), a fraction of
    the gross premium (``premium``) or of the gross loss (``loss``, loss
    adjustment expense), or several such terms in one leg.
    
    Unlabelled, a book's only expense is the leg ``expense``; with several,
    a single-basis one is ``<basis> expense`` and a mixed one ``expense``,
    with ``(2)``, ``(3)`` … added to a name already taken, as ``aggregate``.
    
    Parameters
    ----------
    basis : str or list of tuple of (str, float)
        ``fixed``, ``premium`` or ``loss``; or a list of ``(basis, value)``
        terms.
    value : float, optional
        The amount or rate, with a single ``basis``.
    label : str, optional
    
    Examples
    --------
    >>> from prospicio.aggregate import Expense
    >>> Expense("premium", 0.25, label="Acq")
    Expense("Acq", [("premium", 0.25)])
    """
    def __new__(cls, /, basis: Any, value: float |None = None, *, label: str |None = None) -> Expense: ...
    def __repr__(self, /) -> str: ...
    @property
    def label(self, /) -> str |None:
        """
        The label, or None.
        """
    @property
    def terms(self, /) -> list[tuple[str, float]]:
        """
        The ``(basis, value)`` terms.
        """

@final
class Gam:
    """
    A generalized additive model: a ``Glm`` plus P-spline smooths of
    numeric design columns, with smoothing chosen by GCV or UBRE.
    
    Parameters
    ----------
    glm : Glm
        Family, link and dispersion.
    smooths : list of str or (str, int)
        The design columns to smooth, optionally with the number of basis
        functions (10 by default).
    smoothing : str or list of float, default "auto"
        ``"auto"`` (UBRE for a fixed dispersion, GCV otherwise), ``"gcv"``,
        ``"ubre"``, or fixed smoothing parameters, one per smooth.
    
    Examples
    --------
    >>> import math
    >>> from prospicio.models import Design, Gam, Glm
    >>> x = [i / 99 for i in range(100)]
    >>> y = [math.sin(6 * v) for v in x]
    >>> d = Design([[1.0] * 100, x], ["(Intercept)", "x"])
    >>> fit = Gam(Glm("gaussian"), ["x"]).fit(d, y)
    >>> abs(fit.predict(d)[50] - y[50]) < 0.01
    True
    """
    def __new__(cls, /, glm: Glm, smooths: Sequence[Any], smoothing: Any |None = None) -> Gam: ...
    def fit(self, /, design: Design, y: Sequence[float]) -> GamFit:
        """
        Fits the model.
        
        Parameters
        ----------
        design : Design
            Includes the raw columns to smooth.
        y : list of float
        
        Returns
        -------
        GamFit
        """

@final
class GamFit:
    """
    A fitted GAM, from ``Gam.fit``.
    """
    def __reduce__(self, /) -> tuple[Any, tuple[str]]: ...
    @property
    def coefficients(self, /) -> list[float]:
        """
        Coefficients.
        """
    @property
    def deviance(self, /) -> float:
        """
        Residual deviance.
        """
    @property
    def dispersion(self, /) -> float:
        """
        Dispersion.
        """
    @property
    def edf(self, /) -> float:
        """
        Effective degrees of freedom.
        """
    @property
    def fitted(self, /) -> list[float]:
        """
        Fitted means on the training data.
        """
    @staticmethod
    def from_json(text: str) -> GamFit:
        """
        Reads an artifact written by ``to_json``.
        
        Parameters
        ----------
        text : str
        
        Returns
        -------
        GamFit
        
        Raises
        ------
        ValueError
            For malformed JSON, another format, a newer format version or
            inconsistent fields.
        """
    @property
    def input_hash(self, /) -> str:
        """
        Hash of the training data (design, offset, weights, response).
        """
    @property
    def lambdas(self, /) -> list[float]:
        """
        Smoothing parameter of each smooth.
        """
    @property
    def names(self, /) -> list[str]:
        """
        Coefficient names: parametric columns, then ``s(x).1``, ...
        """
    def predict(self, /, design: Design) -> list[float]:
        """
        Expected response for each row.
        
        Parameters
        ----------
        design : Design
            Same columns as the training design, raw smooth columns included.
        
        Returns
        -------
        list of float
        """
    def predict_distribution(self, /, design: Design, n_sims: int, seed: int) -> PredictiveDistribution:
        """
        Joint predictive distribution across the rows, keyed ``row``.
        
        Parameters
        ----------
        design : Design
        n_sims : int
        seed : int
        
        Returns
        -------
        PredictiveDistribution
        """
    @property
    def score(self, /) -> float:
        """
        The minimized GCV or UBRE score.
        """
    def to_json(self, /) -> str:
        """
        The fit as a versioned JSON artifact (GLM spec, smooths with their knots and constraints, smoothing parameters, estimates), with provenance (crate
        version and a hash of the training data). ``GamFit.from_json`` reads
        it back exactly; pickling uses it too.
        
        Returns
        -------
        str
        """

@final
class Gamma:
    """
    Gamma distribution with shape ``alpha`` and scale ``theta``: mean
    ``alpha * theta``, variance ``alpha * theta**2``.
    
    Parameters
    ----------
    shape : float
    scale : float
    
    Raises
    ------
    ValueError
        If a parameter is not finite and positive.
    
    Examples
    --------
    >>> from prospicio.distributions import Gamma
    >>> g = Gamma.from_mean_cv(1000.0, 0.5)
    >>> g.shape, round(g.std(), 9)
    (4.0, 500.0)
    """
    def __getnewargs__(self, /) -> tuple[float, float]: ...
    def __new__(cls, /, shape: float, scale: float) -> Gamma: ...
    def __repr__(self, /) -> str: ...
    def cdf(self, /, x: float) -> float:
        """
        Distribution function ``P(X <= x)``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @staticmethod
    def from_mean_cv(mean: float, cv: float) -> Gamma:
        """
        Gamma with the given mean and coefficient of variation.
        
        Parameters
        ----------
        mean : float
        cv : float
        
        Returns
        -------
        Gamma
        """
    @staticmethod
    def from_mean_dispersion(mean: float, dispersion: float) -> Gamma:
        """
        Gamma with mean ``mu`` and GLM dispersion ``phi`` (variance
        ``phi * mu**2``): shape ``1 / phi``.
        
        Parameters
        ----------
        mean : float
        dispersion : float
        
        Returns
        -------
        Gamma
        """
    def layer(self, /, limit: float, attachment: float) -> float:
        """
        Expected loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
            ``inf`` for an unlimited layer.
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_second_moment(self, /, limit: float, attachment: float) -> float:
        """
        Second moment of the loss to the layer ``limit`` xs
        ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_variance(self, /, limit: float, attachment: float) -> float:
        """
        Variance of the loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def lev(self, /, limit: float) -> float:
        """
        Limited expected value ``E[min(X, limit)]``.
        
        Parameters
        ----------
        limit : float
        
        Returns
        -------
        float
        """
    def ln_pdf(self, /, x: float) -> float:
        """
        Log density at ``x``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    def mean(self, /) -> float:
        """
        Mean of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """
    def quantile(self, /, p: float) -> float:
        """
        Quantile: the smallest ``x`` with ``P(X <= x) >= p``.
        
        Parameters
        ----------
        p : float
            Probability in ``[0, 1]``.
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def sample(self, /, n: int, seed: int, stream: int = 0) -> list[float]:
        """
        ``n`` draws from stream ``stream`` of the generator keyed by
        ``seed``.
        
        Parameters
        ----------
        n : int
        seed : int
        stream : int, default 0
        
        Returns
        -------
        list of float
        """
    @property
    def scale(self, /) -> float:
        """
        Scale ``theta``.
        """
    @property
    def shape(self, /) -> float:
        """
        Shape ``alpha``.
        """
    def std(self, /) -> float:
        """
        Standard deviation of the distribution.
        
        Returns
        -------
        float
        """
    def stop_loss(self, /, retention: float) -> float:
        """
        Expected excess over a retention, ``E[max(X - retention, 0)]``.
        
        Parameters
        ----------
        retention : float
        
        Returns
        -------
        float
        """
    def survival(self, /, x: float) -> float:
        """
        Survival function ``P(X > x)``, accurate far into the tail.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    def variance(self, /) -> float:
        """
        Variance of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """

@final
class GaussianCopula:
    """
    The Gaussian copula with correlation matrix ``correlation``.
    
    Parameters
    ----------
    correlation : list of list of float
        Symmetric, unit diagonal, positive definite.
    
    Raises
    ------
    ValueError
        If the matrix is not a valid correlation matrix.
    
    Examples
    --------
    >>> from prospicio.risk import GaussianCopula
    >>> c = GaussianCopula([[1.0, 0.5], [0.5, 1.0]])
    >>> u = c.sample(3, seed=1)
    >>> len(u), all(0.0 < x < 1.0 for row in u for x in row)
    (3, True)
    """
    def __new__(cls, /, correlation: Sequence[Sequence[float]]) -> GaussianCopula: ...
    def cdf(self, /, u: Sequence[float]) -> float:
        """
        The copula's distribution function at the point ``u``, each
        coordinate in ``[0, 1]``.
        
        Parameters
        ----------
        u : list of float
            One value per dimension.
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``u`` has the wrong length or a value outside ``[0, 1]``,
            or the copula has no closed-form distribution function here.
        
        Examples
        --------
        >>> from prospicio.risk import GaussianCopula
        >>> round(GaussianCopula([[1.0, 0.5], [0.5, 1.0]]).cdf([0.5, 0.5]), 12)
        0.333333333333
        """
    @property
    def dim(self, /) -> int:
        """
        Number of dimensions.
        """
    def sample(self, /, n: int, seed: int) -> list[list[float]]:
        """
        ``n`` draws of uniforms; draw ``i`` uses stream ``i`` of ``seed``.
        
        Parameters
        ----------
        n : int
        seed : int
        
        Returns
        -------
        list of list of float
        """

@final
class GeneralizedPareto:
    """
    Generalized Pareto severity with a location (Riegel's parameterization
    via :meth:`GeneralizedPareto.riegel`): ``P(X > x) =
    (1 + xi (x - location) / beta) ** (-1 / xi)`` above the location.
    
    For tail estimation from draws, see :class:`prospicio.risk.Gpd`;
    this class is the same distribution as a pricing severity.
    
    Parameters
    ----------
    xi : float
        Shape.
    beta : float
        Scale; finite and positive.
    location : float, default 0.0
    
    Examples
    --------
    >>> from prospicio.distributions import GeneralizedPareto
    >>> g = GeneralizedPareto.riegel(1000.0, 2.0, 1.5)
    >>> round(g.survival(2000.0), 12) == round((7 / 3) ** -1.5, 12)
    True
    """
    def __getnewargs__(self, /) -> tuple[float, float, float]: ...
    def __new__(cls, /, xi: float, beta: float, location: float = 0.0) -> GeneralizedPareto: ...
    def __repr__(self, /) -> str: ...
    @property
    def beta(self, /) -> float:
        """
        Scale ``beta``.
        """
    def cdf(self, /, x: float) -> float:
        """
        Distribution function ``P(X <= x)``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @staticmethod
    def fit_riegel(losses: Sequence[float], t: float, reporting_thresholds: Sequence[float] |None = None, censored: Sequence[bool] |None = None, weights: Sequence[float] |None = None) -> GeneralizedPareto:
        """
        Maximum likelihood fit of Riegel's generalized Pareto with threshold
        ``t`` to large losses at or above ``t``.
        
        Parameters
        ----------
        losses : list of float
        t : float
        reporting_thresholds : list of float, optional
        censored : list of bool, optional
        weights : list of float, optional
        
        Returns
        -------
        GeneralizedPareto
            Read the alphas as ``t / beta`` (initial) and ``1 / xi`` (tail).
        """
    def layer(self, /, limit: float, attachment: float) -> float:
        """
        Expected loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
            ``inf`` for an unlimited layer.
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_second_moment(self, /, limit: float, attachment: float) -> float:
        """
        Second moment of the loss to the layer ``limit`` xs
        ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_variance(self, /, limit: float, attachment: float) -> float:
        """
        Variance of the loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def lev(self, /, limit: float) -> float:
        """
        Limited expected value ``E[min(X, limit)]``.
        
        Parameters
        ----------
        limit : float
        
        Returns
        -------
        float
        """
    @property
    def location(self, /) -> float:
        """
        Location, where the support starts.
        """
    def mean(self, /) -> float:
        """
        Mean of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """
    def quantile(self, /, p: float) -> float:
        """
        Quantile: the smallest ``x`` with ``P(X <= x) >= p``.
        
        Parameters
        ----------
        p : float
            Probability in ``[0, 1]``.
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    @staticmethod
    def riegel(t: float, alpha_ini: float, alpha_tail: float) -> GeneralizedPareto:
        """
        Riegel's generalized Pareto: local alpha ``alpha_ini`` at the
        threshold ``t``, tending to ``alpha_tail`` far out.
        
        Parameters
        ----------
        t : float
        alpha_ini : float
        alpha_tail : float
        
        Returns
        -------
        GeneralizedPareto
        """
    def sample(self, /, n: int, seed: int, stream: int = 0) -> list[float]:
        """
        ``n`` draws from stream ``stream`` of the generator keyed by
        ``seed``.
        
        Parameters
        ----------
        n : int
        seed : int
        stream : int, default 0
        
        Returns
        -------
        list of float
        """
    def std(self, /) -> float:
        """
        Standard deviation of the distribution.
        
        Returns
        -------
        float
        """
    def stop_loss(self, /, retention: float) -> float:
        """
        Expected excess over a retention, ``E[max(X - retention, 0)]``.
        
        Parameters
        ----------
        retention : float
        
        Returns
        -------
        float
        """
    def survival(self, /, x: float) -> float:
        """
        Survival function ``P(X > x)``, accurate far into the tail.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    def variance(self, /) -> float:
        """
        Variance of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """
    @property
    def xi(self, /) -> float:
        """
        Shape ``xi``.
        """

@final
class Glm:
    """
    A generalized linear model, fitted by IRLS.
    
    Parameters
    ----------
    family : str
        ``"gaussian"``, ``"poisson"``, ``"gamma"``, ``"inverse_gaussian"``,
        ``"binomial"``, ``"negative_binomial"`` (needs ``theta``) or
        ``"tweedie"`` (needs ``power``).
    link : str, optional
        ``"identity"``, ``"log"``, ``"logit"``, ``"probit"``,
        ``"cloglog"``, ``"inverse"``, ``"inverse_squared"`` or ``"power"``
        (needs ``link_power``); the family's canonical link by default.
    dispersion : str or float, optional
        ``"pearson"``, ``"deviance"`` or a fixed value. By default 1 for the
        Poisson, binomial and negative binomial and Pearson's estimate
        otherwise; ``"pearson"`` with the Poisson is the over-dispersed
        (quasi-) Poisson, which accepts negative responses as long as the
        fitted means stay positive.
    theta : float, optional
    power : float, optional
    link_power : float, optional
    
    Examples
    --------
    >>> from prospicio.models import Design, Glm
    >>> d = Design([[1.0] * 4, [0.0, 0.0, 1.0, 1.0]], ["(Intercept)", "young"],
    ...            offset=[0.0, 0.0, 0.0, 0.0])
    >>> fit = Glm("poisson", "log").fit(d, [1.0, 3.0, 4.0, 6.0])
    >>> round(fit.coefficients[1], 10) == round(__import__("math").log(5 / 2), 10)
    True
    """
    def __new__(cls, /, family: str, link: str |None = None, dispersion: Any |None = None, theta: float |None = None, power: float |None = None, link_power: float |None = None) -> Glm: ...
    def __repr__(self, /) -> str: ...
    def fit(self, /, design: Design, y: Sequence[float]) -> GlmFit:
        """
        Fits the model.
        
        Parameters
        ----------
        design : Design
        y : list of float
        
        Returns
        -------
        GlmFit
        
        Raises
        ------
        ValueError
            If the design is collinear, a response is out of the family's
            range, or IRLS does not converge.
        """

@final
class GlmFit:
    """
    A fitted GLM, from ``Glm.fit``.
    """
    def __reduce__(self, /) -> tuple[Any, tuple[str]]: ...
    def __repr__(self, /) -> str: ...
    @property
    def aic(self, /) -> float:
        """
        AIC, ``-2 loglik + 2 p``.
        """
    @property
    def coefficients(self, /) -> list[float]:
        """
        Estimated coefficients.
        """
    @property
    def covariance(self, /) -> list[list[float]]:
        """
        Covariance of the coefficients, as a list of rows.
        """
    @property
    def deviance(self, /) -> float:
        """
        Residual deviance.
        """
    @property
    def df_resid(self, /) -> float:
        """
        Residual degrees of freedom.
        """
    @property
    def dispersion(self, /) -> float:
        """
        Dispersion.
        """
    @property
    def fitted(self, /) -> list[float]:
        """
        Fitted means on the training data.
        """
    @staticmethod
    def from_json(text: str) -> GlmFit:
        """
        Reads an artifact written by ``to_json``.
        
        Parameters
        ----------
        text : str
        
        Returns
        -------
        GlmFit
        
        Raises
        ------
        ValueError
            For malformed JSON, another format, a newer format version or
            inconsistent fields.
        """
    @property
    def input_hash(self, /) -> str:
        """
        Hash of the training data (design, offset, weights, response).
        """
    @property
    def iterations(self, /) -> int:
        """
        IRLS iterations used.
        """
    @property
    def log_likelihood(self, /) -> float:
        """
        Log-likelihood.
        """
    @property
    def names(self, /) -> list[str]:
        """
        Coefficient names.
        """
    @property
    def null_deviance(self, /) -> float:
        """
        Deviance of the intercept-and-offset model.
        """
    @property
    def p_values(self, /) -> list[float]:
        """
        Two-sided p-values (normal for a fixed dispersion, Student's t when
        it is estimated).
        """
    def predict(self, /, design: Design) -> list[float]:
        """
        Expected response for each row.
        
        Parameters
        ----------
        design : Design
            Same columns as the training design.
        
        Returns
        -------
        list of float
        """
    def predict_distribution(self, /, design: Design, n_sims: int, seed: int, parameters: str = "normal") -> PredictiveDistribution:
        """
        Joint predictive distribution across the rows, with parameter and
        process uncertainty, keyed ``row = 0, 1, ...``.
        
        Parameters
        ----------
        design : Design
        n_sims : int
        seed : int
        parameters : {"normal", "mean_preserving", "fixed"}, default "normal"
            How the coefficients are drawn: ``beta ~ N(beta_hat, Sigma)``
            (through a log link the draws' mean is
            ``mu_hat * exp(x' Sigma x / 2)``); the same with each row's linear
            predictor shifted so its draws average the fitted mean exactly
            (log or identity link); or fixed at ``beta_hat`` (process
            uncertainty only).
        
        Returns
        -------
        PredictiveDistribution
        """
    def robust_covariance(self, /, design: Design, y: Sequence[float], kind: str = "HC0", groups: Sequence[int |str] |None = None) -> list[list[float]]:
        """
        Sandwich (heteroskedasticity- or cluster-robust) covariance of the
        coefficients, as statsmodels' ``cov_type="HC0"`` and ``"cluster"``.
        
        It stays valid when the variance function or dispersion is wrong,
        as long as the mean is right. The dispersion cancels. For a
        non-canonical link it uses the observed information, as
        statsmodels does (R's ``sandwich`` uses the expected).
        
        Parameters
        ----------
        design : Design
            The design the model was fitted on.
        y : list of float
            The response the model was fitted on.
        kind : {"HC0", "HC1", "cluster"}, default "HC0"
            ``"HC1"`` scales HC0 by ``n / (n - p)``; ``"cluster"`` sums the
            scores within each cluster and scales by
            ``G / (G - 1) * (n - 1) / (n - p)``.
        groups : list of int or str, optional
            One cluster label per row, for ``kind="cluster"``: a policy or
            an event, say.
        
        Returns
        -------
        list of list of float
        
        Examples
        --------
        >>> from prospicio.models import Design, Glm
        >>> d = Design([[1.0] * 6, [0.0, 0.0, 0.0, 1.0, 1.0, 1.0]], ["(Intercept)", "x"])
        >>> y = [1.0, 2.0, 6.0, 1.0, 4.0, 2.0]
        >>> fit = Glm("poisson").fit(d, y)
        >>> round(fit.robust_covariance(d, y)[0][0] * 81, 10)
        14.0
        >>> se = fit.robust_std_errors(d, y, "cluster", groups=[1, 1, 2, 2, 3, 3])
        """
    def robust_std_errors(self, /, design: Design, y: Sequence[float], kind: str = "HC0", groups: Sequence[int |str] |None = None) -> list[float]:
        """
        Square roots of the diagonal of ``robust_covariance``, with the same
        arguments.
        
        Parameters
        ----------
        design : Design
        y : list of float
        kind : {"HC0", "HC1", "cluster"}, default "HC0"
        groups : list of int or str, optional
        
        Returns
        -------
        list of float
        """
    @property
    def std_errors(self, /) -> list[float]:
        """
        Standard errors.
        """
    def to_json(self, /) -> str:
        """
        The fit as a versioned JSON artifact: spec, estimates, covariance,
        fit statistics, fitted values and provenance (crate version and a
        hash of the training data). ``GlmFit.from_json`` reads it back
        exactly; pickling uses it too.
        
        Returns
        -------
        str
        
        Examples
        --------
        >>> import pickle
        >>> from prospicio.models import Design, Glm, GlmFit
        >>> d = Design([[1.0] * 4, [0.0, 1.0, 2.0, 3.0]], ["(Intercept)", "x"])
        >>> fit = Glm("poisson").fit(d, [1.0, 2.0, 2.0, 5.0])
        >>> GlmFit.from_json(fit.to_json()).coefficients == fit.coefficients
        True
        >>> pickle.loads(pickle.dumps(fit)).input_hash == fit.input_hash
        True
        """

@final
class Gpd:
    """
    The generalized Pareto distribution, as SciPy's
    ``genpareto(c=xi, scale=beta)``.
    
    ``P(X > x) = (1 + xi x / beta)**(-1 / xi)`` for ``x >= 0``.
    
    Parameters
    ----------
    xi : float
        Shape; moments of order ``1 / xi`` and above are infinite.
    beta : float
        Scale, positive.
    
    Examples
    --------
    >>> from prospicio.risk import Gpd
    >>> g = Gpd(0.5, 2.0)
    >>> g.mean()
    4.0
    >>> fit = Gpd.fit([g.quantile((i - 0.5) / 1000) for i in range(1, 1001)])
    >>> round(fit.xi, 2), round(fit.beta, 2)
    (0.5, 2.0)
    """
    def __new__(cls, /, xi: float, beta: float) -> Gpd: ...
    def __repr__(self, /) -> str: ...
    @property
    def beta(self, /) -> float:
        """
        Scale.
        """
    def cdf(self, /, x: float) -> float:
        """
        Distribution function.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @staticmethod
    def fit(exceedances: Sequence[float]) -> Gpd:
        """
        Maximum likelihood fit to exceedances (values over a threshold,
        minus the threshold).
        
        Parameters
        ----------
        exceedances : list of float
            At least 3, non-negative, not all equal.
        
        Returns
        -------
        Gpd
        """
    def mean(self, /) -> float:
        """
        Mean, ``beta / (1 - xi)``; infinite for ``xi >= 1``.
        """
    def quantile(self, /, p: float) -> float:
        """
        Quantile function.
        
        Parameters
        ----------
        p : float
        
        Returns
        -------
        float
        """
    @property
    def xi(self, /) -> float:
        """
        Shape.
        """

@final
class Grid:
    """
    A distribution on the points ``0, step, 2*step, ...``: the discretized
    representation that FFT and Panjer aggregation work on.
    
    Parameters
    ----------
    step : float
        Grid step; must be positive.
    probs : list of float
        Probabilities at ``0, step, ...``; non-negative, summing to 1.
    
    Raises
    ------
    ValueError
        If the step is not positive or the probabilities are invalid.
    
    Examples
    --------
    >>> from prospicio.distributions import Grid, Lognormal
    >>> grid, report = Grid.local_moment(Lognormal(7.0, 0.5), 100.0, 200)
    >>> report.tail_mass < 1e-8
    True
    """
    def __getnewargs__(self, /) -> tuple[float, list[float]]: ...
    def __len__(self, /) -> int: ...
    def __new__(cls, /, step: float, probs: Sequence[float]) -> Grid: ...
    def __repr__(self, /) -> str: ...
    def cdf(self, /, x: float) -> float:
        """
        Distribution function ``P(X <= x)``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    def layer(self, /, limit: float, attachment: float) -> float:
        """
        Expected loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def lev(self, /, limit: float) -> float:
        """
        Limited expected value ``E[min(X, limit)]``, exact on the grid.
        
        Parameters
        ----------
        limit : float
        
        Returns
        -------
        float
        """
    @staticmethod
    def local_moment(severity: Any, step: float, points: int) -> tuple[Grid, DiscretizationReport]:
        """
        Discretizes a severity by local moment matching on the mean.
        
        Parameters
        ----------
        severity : Lognormal, Grid, Pareto, PiecewisePareto, LogAffinePareto or GeneralizedPareto
        step : float
        points : int
        
        Returns
        -------
        tuple of (Grid, DiscretizationReport)
        
        Raises
        ------
        ValueError
            If ``step`` is not positive or ``points`` is 0.
        """
    @staticmethod
    def lower(severity: Any, step: float, points: int) -> tuple[Grid, DiscretizationReport]:
        """
        Discretizes a severity by moving each cell's mass to its left end: a
        stochastic lower bound.
        
        Parameters
        ----------
        severity : Lognormal, Grid, Pareto, PiecewisePareto, LogAffinePareto or GeneralizedPareto
        step : float
        points : int
        
        Returns
        -------
        tuple of (Grid, DiscretizationReport)
        
        Raises
        ------
        ValueError
            If ``step`` is not positive or ``points`` is 0.
        """
    def map(self, /, f: Any) -> tuple[Grid, bool]:
        """
        The distribution of ``f(X)`` on the same step.
        
        Each point's mass moves to ``f(x)``. A value between two points is
        split between them so its mean is kept, so the mean is always exact
        and the whole distribution is exact when every value lands on a
        point. ``f`` is a Python callable, evaluated once per point with
        mass.
        
        Parameters
        ----------
        f : callable
            Maps a loss to a finite, non-negative value.
        
        Returns
        -------
        tuple of (Grid, bool)
            The grid and whether every value landed on a grid point.
        
        Raises
        ------
        ValueError
            If ``f`` returns a negative or non-finite value.
        
        Examples
        --------
        >>> from prospicio.distributions import Grid
        >>> x = Grid(1.0, [0.2, 0.3, 0.3, 0.2])
        >>> layer, exact = x.map(lambda v: min(max(v - 1.0, 0.0), 1.0))
        >>> layer.probs, exact
        ([0.5, 0.5], True)
        """
    def mean(self, /) -> float:
        """
        Mean of the distribution.
        
        Returns
        -------
        float
        """
    @property
    def probs(self, /) -> list[float]:
        """
        Probabilities at ``0, step, 2*step, ...``.
        """
    def quantile(self, /, p: float) -> float:
        """
        Smallest grid point ``x`` with ``P(X <= x) >= p``.
        
        Parameters
        ----------
        p : float
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    @staticmethod
    def rounding(severity: Any, step: float, points: int) -> tuple[Grid, DiscretizationReport]:
        """
        Discretizes a severity by rounding each loss to the nearest point.
        
        Parameters
        ----------
        severity : Lognormal, Grid, Pareto, PiecewisePareto, LogAffinePareto or GeneralizedPareto
        step : float
        points : int
        
        Returns
        -------
        tuple of (Grid, DiscretizationReport)
        
        Raises
        ------
        ValueError
            If ``step`` is not positive or ``points`` is 0.
        """
    @property
    def step(self, /) -> float:
        """
        Grid step.
        """
    def stop_loss(self, /, retention: float) -> float:
        """
        Expected excess ``E[max(X - retention, 0)]``, exact on the grid.
        
        Parameters
        ----------
        retention : float
        
        Returns
        -------
        float
        """
    def tvar(self, /, p: float) -> float:
        """
        Tail value at risk at ``p``: the mean of the worst ``1 - p`` of the
        distribution, the point at the VaR counted with its mass above
        ``p``.
        
        Parameters
        ----------
        p : float
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def tvar_threshold(self, /, p: float) -> float:
        """
        The level ``t`` with ``TVaR(t) = VaR(p)`` (the lower VaR), as
        ``aggregate``'s ``tvar_threshold``; ``t <= p``, and ``p = 0`` gives 0.
        
        Parameters
        ----------
        p : float
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        
        Examples
        --------
        >>> from prospicio.distributions import Grid
        >>> g = Grid(1.0, [0.4, 0.3, 0.2, 0.1])
        >>> t = g.tvar_threshold(0.85)
        >>> round(g.tvar(t), 12) == g.quantile(0.85)
        True
        """
    def variance(self, /) -> float:
        """
        Variance of the distribution.
        
        Returns
        -------
        float
        """

@final
class GridSize:
    """
    A recommended FFT grid, from ``recommend_grid`` or ``fft_auto``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def extent(self, /) -> float:
        """
        The extent the grid was sized to cover.
        """
    @property
    def jump_extent(self, /) -> float |None:
        """
        The single-big-jump extent at ``p_star``, or ``None``.
        """
    @property
    def method(self, /) -> str:
        """
        What sized it: ``"moments"`` or ``"single_big_jump"``.
        """
    @property
    def moment_extent(self, /) -> float |None:
        """
        The moment extent, or ``None`` with an infinite variance.
        """
    @property
    def points(self, /) -> int:
        """
        Number of points, a power of two.
        """
    @property
    def step(self, /) -> float:
        """
        Bucket size, a ``round_bucket`` rung.
        """
    @property
    def tail_estimate(self, /) -> float:
        """
        ``min(1, E[N] S_X(top))``: roughly the probability beyond the grid
        from one claim alone. Raise ``log2`` when it is not small.
        """

@final
class Group:
    """
    A book of the ledger: ``sell`` (the insurer of the subject business,
    booking ``+consideration - obligation``) or ``buy`` (a cession, booking
    the contra: ceded premium paid, recoveries and commission received).
    
    Parameters
    ----------
    label : str
    role : str
        ``sell`` or ``buy``.
    consideration : list of Leg, optional
    obligation : list of Leg, optional
    margin_label : str, optional
        The label of a seller's own margin on a ledger that also buys
        cover; ``Gross`` by default.
    """
    def __new__(cls, /, label: str, role: str, consideration: Sequence[Leg] |None = None, obligation: Sequence[Leg] |None = None, margin_label: str |None = None) -> Group: ...
    def __repr__(self, /) -> str: ...
    @property
    def label(self, /) -> str:
        """
        The label.
        """

@final
class HierarchicalStacking:
    """
    Hierarchical stacking (Yao, Pirš, Vehtari and Gelman, 2022): model
    weights that vary with covariates, ``w = softmax(alpha + B x)`` against
    the last model as reference, so a model can be trusted in one part of
    the portfolio and not another. The priors are those of BayesBlend's
    ``HierarchicalBayesStacking``; sampled by NUTS.
    
    Scale continuous covariates (BayesBlend divides by twice the standard
    deviation) and dummy-code discrete ones before fitting, with the
    dummies first.
    
    With ``partial_pooling``, each model's slopes on the discrete
    covariates, and separately on the continuous ones, are drawn around a
    model-level mean, itself drawn around a global mean. A scale of 0
    removes a level: ``tau_mu_global=0`` fixes the global mean at 0,
    ``tau_mu_*=0`` pools completely and ``tau_sigma_*=0`` sets every slope
    to its model's mean. BayesBlend warns that pooling needs at least three
    covariates. ``adaptive`` multiplies the prior scales by ``N**lambda``
    with ``lambda ~ Exponential(adaptive)``, weakening them as the data
    grow.
    
    Parameters
    ----------
    discrete : int, default 0
        Number of leading covariates that are dummy codes.
    alpha_loc, alpha_scale : float, default 0.0, 1.0
    beta_loc, beta_scale : float, default 0.0, 1.0
        Slope prior without pooling.
    partial_pooling : bool, default False
    tau_mu_global, tau_mu_discrete, tau_mu_continuous : float, default 1.0
    tau_sigma_discrete, tau_sigma_continuous : float, default 1.0
        Pooling scales, BayesBlend's defaults.
    adaptive : float, optional
        Rate of the exponential prior on ``lambda`` (BayesBlend uses 4).
    chains, tune, draws : int, default 4, 1000, 1000
    seed : int, default 0
    
    Examples
    --------
    >>> from prospicio.models import HierarchicalStacking
    >>> x = [i / 99 - 0.5 for i in range(100)]
    >>> a = [-0.5 if v < 0 else -2.0 for v in x]
    >>> b = [-2.0 if v < 0 else -0.5 for v in x]
    >>> fit = HierarchicalStacking(chains=2, tune=300, draws=300).fit([a, b], [x])
    >>> w = fit.weights([[-0.4, 0.4]])
    >>> w[0][0] > 0.7 and w[1][0] < 0.3
    True
    """
    def __new__(cls, /, discrete: int = 0, alpha_loc: float = 0.0, alpha_scale: float = 1.0, beta_loc: float = 0.0, beta_scale: float = 1.0, partial_pooling: bool = False, tau_mu_global: float = 1.0, tau_mu_discrete: float = 1.0, tau_mu_continuous: float = 1.0, tau_sigma_discrete: float = 1.0, tau_sigma_continuous: float = 1.0, adaptive: float |None = None, chains: int = 4, tune: int = 1000, draws: int = 1000, seed: int = 0) -> HierarchicalStacking: ...
    def fit(self, /, lpd: Sequence[Sequence[float]], covariates: Sequence[Sequence[float]]) -> StackingFit:
        """
        Samples the intercepts and slopes.
        
        Parameters
        ----------
        lpd : list of list of float
            One list per model, one held-out log density per observation.
        covariates : list of list of float
            One list per covariate, one value per observation.
        
        Returns
        -------
        StackingFit
        """

@final
class IndexClause:
    """
    An index (stability) clause: the treaty's retention and limit move with
    an index, by the ratio ``r`` of the index at a loss's payment to the
    index at the treaty's base date, within a franchise.
    
    With ``kind="full"``, nothing moves while ``|r - 1|`` is within the
    franchise, and beyond it the terms move by all of ``r``, up or down.
    With ``kind="severe"`` (a severe inflation clause), the terms move only
    by the rise beyond the franchise, ``r / (1 + franchise)``, and never
    down.
    
    Parameters
    ----------
    franchise : float
        A fraction, such as 0.1 for 10%.
    kind : {"full", "severe"}, default "full"
    
    Examples
    --------
    >>> from prospicio.pricing import IndexClause
    >>> IndexClause(0.1).adjustment(1.25)
    1.25
    >>> round(IndexClause(0.1, kind="severe").adjustment(1.21), 12)
    1.1
    """
    def __new__(cls, /, franchise: float, kind: str = "full") -> IndexClause: ...
    def __repr__(self, /) -> str: ...
    def adjustment(self, /, ratio: float) -> float:
        """
        The factor the terms move by at index ratio ``ratio``.
        
        Parameters
        ----------
        ratio : float
        
        Returns
        -------
        float
        """
    @property
    def franchise(self, /) -> float:
        """
        The franchise.
        """
    @property
    def kind(self, /) -> str:
        """
        ``"full"`` or ``"severe"``.
        """

@final
class InverseGamma:
    """
    Inverse gamma distribution with shape ``alpha`` and scale ``theta``:
    ``X = theta / G`` for ``G`` a unit-scale gamma, as SciPy's ``invgamma``.
    The tail is Pareto-like: the mean is infinite for ``alpha <= 1`` and the
    variance for ``alpha <= 2``; limited and layer moments always exist.
    
    Parameters
    ----------
    shape : float
    scale : float
    
    Raises
    ------
    ValueError
        If a parameter is not finite and positive.
    
    Examples
    --------
    >>> from prospicio.distributions import InverseGamma
    >>> round(InverseGamma(3.0, 2000.0).mean(), 9)
    1000.0
    """
    def __getnewargs__(self, /) -> tuple[float, float]: ...
    def __new__(cls, /, shape: float, scale: float) -> InverseGamma: ...
    def __repr__(self, /) -> str: ...
    def cdf(self, /, x: float) -> float:
        """
        Distribution function ``P(X <= x)``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @staticmethod
    def from_mean_cv(mean: float, cv: float) -> InverseGamma:
        """
        Inverse gamma with the given mean and coefficient of variation:
        ``alpha = 2 + 1 / cv**2``.
        
        Parameters
        ----------
        mean : float
        cv : float
        
        Returns
        -------
        InverseGamma
        """
    def layer(self, /, limit: float, attachment: float) -> float:
        """
        Expected loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
            ``inf`` for an unlimited layer.
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_second_moment(self, /, limit: float, attachment: float) -> float:
        """
        Second moment of the loss to the layer ``limit`` xs
        ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_variance(self, /, limit: float, attachment: float) -> float:
        """
        Variance of the loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def lev(self, /, limit: float) -> float:
        """
        Limited expected value ``E[min(X, limit)]``.
        
        Parameters
        ----------
        limit : float
        
        Returns
        -------
        float
        """
    def mean(self, /) -> float:
        """
        Mean of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """
    def quantile(self, /, p: float) -> float:
        """
        Quantile: the smallest ``x`` with ``P(X <= x) >= p``.
        
        Parameters
        ----------
        p : float
            Probability in ``[0, 1]``.
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def sample(self, /, n: int, seed: int, stream: int = 0) -> list[float]:
        """
        ``n`` draws from stream ``stream`` of the generator keyed by
        ``seed``.
        
        Parameters
        ----------
        n : int
        seed : int
        stream : int, default 0
        
        Returns
        -------
        list of float
        """
    @property
    def scale(self, /) -> float:
        """
        Scale ``theta``.
        """
    @property
    def shape(self, /) -> float:
        """
        Shape ``alpha``.
        """
    def std(self, /) -> float:
        """
        Standard deviation of the distribution.
        
        Returns
        -------
        float
        """
    def stop_loss(self, /, retention: float) -> float:
        """
        Expected excess over a retention, ``E[max(X - retention, 0)]``.
        
        Parameters
        ----------
        retention : float
        
        Returns
        -------
        float
        """
    def survival(self, /, x: float) -> float:
        """
        Survival function ``P(X > x)``, accurate far into the tail.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    def variance(self, /) -> float:
        """
        Variance of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """

@final
class InverseGaussian:
    """
    Inverse Gaussian distribution with mean ``mu`` and shape ``lam``:
    variance ``mu**3 / lam``. SciPy's ``invgauss(mu / lam, scale=lam)``.
    
    Parameters
    ----------
    mean : float
    shape : float
    
    Raises
    ------
    ValueError
        If a parameter is not finite and positive.
    
    Examples
    --------
    >>> from prospicio.distributions import InverseGaussian
    >>> d = InverseGaussian.from_mean_cv(1000.0, 0.5)
    >>> d.shape, round(d.std(), 9)
    (4000.0, 500.0)
    """
    def __getnewargs__(self, /) -> tuple[float, float]: ...
    def __new__(cls, /, mean: float, shape: float) -> InverseGaussian: ...
    def __repr__(self, /) -> str: ...
    def cdf(self, /, x: float) -> float:
        """
        Distribution function ``P(X <= x)``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @staticmethod
    def from_mean_cv(mean: float, cv: float) -> InverseGaussian:
        """
        Inverse Gaussian with the given mean and coefficient of variation:
        ``lam = mean / cv**2``.
        
        Parameters
        ----------
        mean : float
        cv : float
        
        Returns
        -------
        InverseGaussian
        """
    def layer(self, /, limit: float, attachment: float) -> float:
        """
        Expected loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
            ``inf`` for an unlimited layer.
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_second_moment(self, /, limit: float, attachment: float) -> float:
        """
        Second moment of the loss to the layer ``limit`` xs
        ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_variance(self, /, limit: float, attachment: float) -> float:
        """
        Variance of the loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def lev(self, /, limit: float) -> float:
        """
        Limited expected value ``E[min(X, limit)]``.
        
        Parameters
        ----------
        limit : float
        
        Returns
        -------
        float
        """
    def mean(self, /) -> float:
        """
        Mean of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """
    def quantile(self, /, p: float) -> float:
        """
        Quantile: the smallest ``x`` with ``P(X <= x) >= p``.
        
        Parameters
        ----------
        p : float
            Probability in ``[0, 1]``.
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def sample(self, /, n: int, seed: int, stream: int = 0) -> list[float]:
        """
        ``n`` draws from stream ``stream`` of the generator keyed by
        ``seed``.
        
        Parameters
        ----------
        n : int
        seed : int
        stream : int, default 0
        
        Returns
        -------
        list of float
        """
    @property
    def shape(self, /) -> float:
        """
        Shape ``lam``.
        """
    def std(self, /) -> float:
        """
        Standard deviation of the distribution.
        
        Returns
        -------
        float
        """
    def stop_loss(self, /, retention: float) -> float:
        """
        Expected excess over a retention, ``E[max(X - retention, 0)]``.
        
        Parameters
        ----------
        retention : float
        
        Returns
        -------
        float
        """
    def survival(self, /, x: float) -> float:
        """
        Survival function ``P(X > x)``, accurate far into the tail.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    def variance(self, /) -> float:
        """
        Variance of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """

@final
class JointGrid:
    """
    A joint distribution of two losses on a grid, ``P(A = i * step_a, B = j
    * step_b)``: rows are line A and columns line B. The mass past the last
    row or column is carried on it.
    
    Parameters
    ----------
    steps : tuple of (float, float)
    probs : list of list of float
        One row per point of A, one column per point of B, summing to 1.
    """
    def __new__(cls, /, steps: tuple[float, float], probs: Sequence[Sequence[float]]) -> JointGrid: ...
    def __repr__(self, /) -> str: ...
    def correlation(self, /) -> float:
        """
        Pearson's correlation of ``A`` and ``B``; NaN if either is constant.
        """
    def covariance(self, /) -> float:
        """
        ``Cov(A, B)``.
        """
    @staticmethod
    def from_copula(a: Grid, b: Grid, copula: Any, *, trigger_a: float = 1.0, trigger_b: float = 1.0) -> JointGrid:
        """
        The joint per-event severity of two losses coupled by a copula.
        
        Each line is hit with its ``trigger`` probability and then loses
        from its grid (``a`` for line A, the rows; ``b`` for B), so mass at 0
        is an event that misses the line; the two per-event marginals are
        joined by the copula's rectangle probabilities over their cdfs, as
        ``aggregate``'s copula ``bivariate``. The marginals come back
        exactly. Give the result to ``fft_bivariate``.
        
        Parameters
        ----------
        a, b : Grid
            The loss to each line given that the event hits it.
        copula : GaussianCopula or ArchimedeanCopula
            Two-dimensional, with a distribution function.
        trigger_a, trigger_b : float, default 1.0
            The probability that an event hits each line.
        
        Returns
        -------
        JointGrid
        
        Examples
        --------
        >>> from prospicio.aggregate import JointGrid
        >>> from prospicio.distributions import Grid
        >>> from prospicio.risk import ArchimedeanCopula
        >>> a = Grid(1.0, [0.25, 0.5, 0.25])
        >>> b = Grid(1.0, [0.5, 0.5])
        >>> s = JointGrid.from_copula(a, b, ArchimedeanCopula("gumbel", 2.0, 2), trigger_a=0.5)
        >>> s.correlation() > 0
        True
        """
    @staticmethod
    def independent(a: Grid, b: Grid) -> JointGrid:
        """
        The joint law of two independent losses.
        
        Parameters
        ----------
        a, b : Grid
        
        Returns
        -------
        JointGrid
        """
    def marginal(self, /, axis: int) -> Grid:
        """
        The marginal distribution of line A (``axis=0``) or B (``axis=1``).
        
        Parameters
        ----------
        axis : int
        
        Returns
        -------
        Grid
        """
    def mean(self, /, axis: int) -> float:
        """
        The mean of line A (``axis=0``) or B (``axis=1``).
        """
    def moment(self, /, p: int, q: int) -> float:
        """
        The raw mixed moment ``E[A^p B^q]``.
        
        Parameters
        ----------
        p, q : int
        
        Returns
        -------
        float
        """
    def prob(self, /, i: int, j: int) -> float:
        """
        ``P(A = i * step_a, B = j * step_b)``.
        
        Parameters
        ----------
        i, j : int
        
        Returns
        -------
        float
        """
    @property
    def probs(self, /) -> list[list[float]]:
        """
        The probabilities, one list per row.
        """
    @property
    def shape(self, /) -> tuple[int, int]:
        """
        The number of rows (line A) and columns (line B).
        """
    @property
    def steps(self, /) -> tuple[float, float]:
        """
        The step on each axis.
        """
    def total(self, /) -> Grid:
        """
        The distribution of the total ``A + B``: exact when the steps are
        equal, otherwise on the larger step with each cell's mass split to
        keep the mean.
        
        Returns
        -------
        Grid
        """
    def variance(self, /, axis: int) -> float:
        """
        The variance of line A (``axis=0``) or B (``axis=1``).
        """

@final
class LargeLossExperience:
    """
    A record of individual large losses, for experience rating.
    
    Develop each known loss to ultimate (optional) and trend it to the
    prospective period with ``prepare``, then apply each year's treaty and
    develop the layer totals with ``PreparedLosses.burning_cost``. The steps
    follow Clark's *Basics of Reinsurance Pricing* (CAS study note).
    
    Parameters
    ----------
    ids : list of str
        An identifier per loss.
    accident : list of float
        Each loss's accident date as a decimal year (see ``year_fraction``).
    amount : list of float
        Each loss's amount, incurred at the latest valuation.
    threshold : float, default 0.0
        The reporting threshold of the record, in nominal terms. With a
        threshold, losses trended below it times the largest trend factor
        are left out, so every year is complete on the same basis.
    valuation : float, optional
        The date the amounts were valued at, a decimal year; needed to
        develop the losses by a ``DevelopmentPattern`` at their ages.
    fx : float or list of float, optional
        Converts the losses to the pricing currency before development and
        trend: a stated rate, or a rate per loss (units of the pricing
        currency per unit of the loss's). The threshold stays in the
        reported currency.
    
    Examples
    --------
    >>> from prospicio.pricing import LargeLossExperience
    >>> from prospicio.reinsurance import Layer, Tower
    >>> e = LargeLossExperience(["a", "b"], [1995.5, 1996.5], [150.0, 400.0])
    >>> p = e.prepare(factors=[1.0, 1.0])
    >>> tower = Tower([Layer("200x100", 200.0, 100.0)])
    >>> b = p.burning_cost(tower, [1995, 1996], premium=[1000.0, 1000.0], development=[1.0, 1.5])
    >>> b.to_dict()["developed"]
    [[50.0], [300.0]]
    >>> round(b.loss_cost[0], 4)
    0.175
    """
    def __len__(self, /) -> int: ...
    def __new__(cls, /, ids: Sequence[str], accident: Sequence[float], amount: Sequence[float], *, threshold: float = 0.0, valuation: float |None = None, fx: Any |None = None) -> LargeLossExperience: ...
    def __repr__(self, /) -> str: ...
    def prepare(self, /, *, rate: float |None = None, to: float |None = None, factors: Sequence[float] |None = None, development: Any |None = None) -> PreparedLosses:
        """
        Develop every loss to ultimate, if asked, and trend it to the
        prospective period.
        
        Pass either an annual ``rate`` with the date ``to`` (the average
        accident date of the prospective period, a decimal year), for factors
        ``(1 + rate) ** (to - accident)``, or a factor per loss.
        
        Parameters
        ----------
        rate : float, optional
        to : float, optional
        factors : list of float, optional
        development : DevelopmentPattern or list of float, optional
            Develops each known loss to ultimate (IBNER): a pattern, read at
            each loss's age at the valuation date, or a factor per loss.
            Then give ``burning_cost`` only the count IBNR factors, not an
            excess development factor that already holds IBNER.
        
        Returns
        -------
        PreparedLosses
        
        Raises
        ------
        TypeError
            If neither or both of ``rate`` (with ``to``) and ``factors`` are
            given.
        """
    @property
    def threshold(self, /) -> float:
        """
        The reporting threshold.
        """
    @property
    def valuation(self, /) -> float |None:
        """
        The valuation date, or ``None``.
        """

@final
class Layer:
    """
    A per-occurrence excess-of-loss layer: ``limit`` xs ``attachment`` on each
    loss, then annual terms.
    
    For one year, ``ceded = share * min(max(sum of per-loss recoveries -
    aggregate_deductible, 0), aggregate_limit)``, less any loss corridor
    (``with_loss_corridor``) before the annual limit.
    
    Parameters
    ----------
    name : str
    limit : float
        Per-occurrence limit; may be ``inf``.
    attachment : float
    share : float, default 1.0
        Placed share, in ``(0, 1]``.
    aggregate_deductible : float, default 0.0
    aggregate_limit : float, default inf
    reinstatements : int, optional
        Free reinstatements: sets ``aggregate_limit`` to
        ``limit * (reinstatements + 1)``; cannot be combined with
        ``aggregate_limit``.
    premium : float, default 0.0
        Upfront premium for the placed share: the base of paid
        reinstatements, ceding and profit commissions and a sliding scale's
        loss ratio. ``with_rate_on_line``, ``with_premium_rate`` and
        ``with_deposit_premium`` set it from a quote.
    reinstatement_rates : list of float, optional
        Paid reinstatements, one rate per reinstatement as a fraction of
        ``premium`` (1.0 is 100%), pro rata as to amount. Sets
        ``aggregate_limit`` to ``limit * (len(reinstatement_rates) + 1)``;
        cannot be combined with ``aggregate_limit`` or ``reinstatements``.
    pro_rata_time : bool, default False
        Paid reinstatements also pro rata as to time: the limit a loss at
        time ``t`` (the fraction of the year elapsed) uses up is charged at
        ``1 - t``. Needs ``reinstatement_rates``, and events with times
        (``EventSet.with_uniform_times``, or ``times=`` in
        ``EventSet.from_years``).
    
    Raises
    ------
    ValueError
        If a term is out of range, or more than one of ``aggregate_limit``,
        ``reinstatements`` and ``reinstatement_rates`` is given.
    
    Examples
    --------
    >>> from prospicio.reinsurance import Layer
    >>> layer = Layer("5x5", 5e6, 5e6, reinstatements=1)
    >>> layer.ceded([7e6])
    2000000.0
    >>> layer.ceded([12e6, 20e6, 30e6])
    10000000.0
    >>> paid = Layer("10x10", 10.0, 10.0, premium=2.0, reinstatement_rates=[1.0, 0.5])
    >>> paid.reinstatement_premium([22.0, 12.0])
    2.2
    >>> timed = Layer("10x10", 10.0, 10.0, premium=2.0, reinstatement_rates=[1.0, 0.5],
    ...               pro_rata_time=True)
    >>> round(timed.reinstatement_premium([22.0, 12.0], times=[0.25, 0.5]), 12)
    1.6
    """
    def __new__(cls, /, name: str, limit: float, attachment: float, share: float = 1.0, aggregate_deductible: float = 0.0, aggregate_limit: float |None = None, reinstatements: int |None = None, premium: float = 0.0, reinstatement_rates: Sequence[float] |None = None, pro_rata_time: bool = False) -> Layer: ...
    def __repr__(self, /) -> str: ...
    @property
    def aggregate_deductible(self, /) -> float:
        """
        Annual aggregate deductible.
        """
    @property
    def aggregate_limit(self, /) -> float:
        """
        Annual aggregate limit.
        """
    @property
    def attachment(self, /) -> float:
        """
        Per-occurrence attachment.
        """
    def ceded(self, /, losses: Sequence[float]) -> float:
        """
        Ceded loss for one year's losses.
        
        Parameters
        ----------
        losses : list of float
        
        Returns
        -------
        float
        """
    def ceded_by_event(self, /, losses: Sequence[float]) -> list[float]:
        """
        Ceded loss per event for one year, taking losses as chronological.
        
        The annual deductible absorbs the first recoveries and the annual
        limit stops the last ones; the entries sum to ``ceded(losses)``.
        
        Parameters
        ----------
        losses : list of float
        
        Returns
        -------
        list of float
        
        Examples
        --------
        >>> from prospicio.reinsurance import Layer
        >>> layer = Layer("L", 10.0, 5.0, aggregate_deductible=4.0, aggregate_limit=15.0)
        >>> layer.ceded_by_event([8.0, 20.0, 12.0])
        [0.0, 9.0, 6.0]
        """
    def ceded_with_sums_insured(self, /, losses: Sequence[float], sums_insured: Sequence[float]) -> float:
        """
        Ceded loss for one year's losses on risks with the given sums
        insured, one per loss.
        
        Parameters
        ----------
        losses : list of float
        sums_insured : list of float
        
        Returns
        -------
        float
        """
    @property
    def ceding_commission(self, /) -> float |None:
        """
        The flat ceding commission rate, or ``None`` (also when the
        commission slides).
        """
    def ceding_commission_for(self, /, ceded: float) -> float:
        """
        The year's ceding commission (flat or sliding) given its ceded loss.
        
        Parameters
        ----------
        ceded : float
        
        Returns
        -------
        float
        """
    @property
    def limit(self, /) -> float:
        """
        Per-occurrence limit.
        """
    @property
    def loss_corridor(self, /) -> tuple[float, float, float] |None:
        """
        The loss corridor as ``(lower, upper, retained)``, or ``None``.
        """
    @property
    def name(self, /) -> str:
        """
        Layer name.
        """
    @property
    def needs_sums_insured(self, /) -> bool:
        """
        Whether the layer is a surplus treaty, which needs sums insured.
        """
    @property
    def premium(self, /) -> float:
        """
        Upfront premium for the placed share.
        """
    def premium_for(self, /, ceded: float) -> float:
        """
        The year's premium given its ceded loss: the swing-rated premium,
        or else the fixed ``premium``.
        
        Parameters
        ----------
        ceded : float
            The layer's ceded loss for the year, at the placed share.
        
        Returns
        -------
        float
        """
    @property
    def pro_rata_time(self, /) -> bool:
        """
        Whether paid reinstatements are pro rata as to time.
        """
    @property
    def profit_commission(self, /) -> tuple[float, float] |None:
        """
        The profit commission as ``(share, allowance)``, or ``None``.
        """
    def profit_commission_for(self, /, ceded: float) -> float:
        """
        The year's profit commission given its ceded loss.
        
        Parameters
        ----------
        ceded : float
        
        Returns
        -------
        float
        """
    @staticmethod
    def quota_share(name: str, cession: float) -> Layer:
        """
        A quota share ceding ``cession`` of every loss.
        
        Unlimited cover from the first unit with ``share = cession``.
        
        Parameters
        ----------
        name : str
        cession : float
            In ``(0, 1]``.
        
        Returns
        -------
        Layer
        
        Examples
        --------
        >>> from prospicio.reinsurance import Layer
        >>> Layer.quota_share("QS", 0.4).ceded([10.0, 5.0])
        6.0
        """
    def reinstatement_premium(self, /, losses: Sequence[float], times: Sequence[float] |None = None) -> float:
        """
        Reinstatement premium for one year's losses.
        
        With layer loss ``L`` at 100% after annual terms, ``premium *
        sum(rate_k * min(max(L - k * limit, 0), limit) / limit)``; zero when
        reinstatements are free. Pro rata as to time, the limit each loss
        uses up is charged at ``1 - t``, its time's share of the year left.
        
        Parameters
        ----------
        losses : list of float
            In time order.
        times : list of float, optional
            Each loss's time, as the fraction of the year elapsed; needed
            (and only used) when the layer is pro rata as to time.
        
        Returns
        -------
        float
            NaN for a layer pro rata as to time without ``times``.
        """
    @property
    def reinstatement_rates(self, /) -> list[float]:
        """
        Rate of each paid reinstatement; empty when reinstatements are free.
        """
    @property
    def share(self, /) -> float:
        """
        Placed share.
        """
    @property
    def sliding_scale(self, /) -> list[tuple[float, float]] |None:
        """
        The sliding scale's ``(commission, loss_ratio)`` anchors in
        increasing loss ratio, or ``None``.
        """
    @staticmethod
    def stop_loss(name: str, limit: float, retention: float) -> Layer:
        """
        An aggregate stop-loss: ``limit`` xs ``retention`` on the year's total.
        
        Covers the total of the losses it sees: gross, or net of earlier
        stages in an inuring ``Tower``.
        
        Parameters
        ----------
        name : str
        limit : float
            Annual limit; may be ``inf``.
        retention : float
        
        Returns
        -------
        Layer
        
        Examples
        --------
        >>> from prospicio.reinsurance import Layer
        >>> Layer.stop_loss("SL", 50.0, 100.0).ceded([60.0, 70.0])
        30.0
        """
    @staticmethod
    def surplus(name: str, retention: float, lines: float) -> Layer:
        """
        A surplus treaty: each risk cedes the part of its sum insured above
        the retention line ``retention``, up to ``lines`` lines, and the
        same share of every loss on it.
        
        With a retention of 1m and 9 lines (a capacity of 9m), a 5m risk
        cedes 80% and a 20m risk 45%. The events must carry sums insured
        (``EventSet.from_years(..., sums_insured=...)``); it can inure to a
        per-risk excess of loss in a later stage of a ``Tower``.
        
        Parameters
        ----------
        name : str
        retention : float
            The retention line; positive.
        lines : float
            Number of lines of capacity; positive.
        
        Returns
        -------
        Layer
        
        Examples
        --------
        >>> from prospicio.reinsurance import Layer
        >>> s = Layer.surplus("surplus", 1e6, 9.0)
        >>> round(s.ceded_with_sums_insured([2e6, 2e6], [5e6, 20e6]))
        2500000
        """
    @property
    def swing_rating(self, /) -> tuple[float, float, float, float] |None:
        """
        The swing rating at 100% as ``(basic, lcm, minimum, maximum)``, or
        ``None``.
        """
    def with_ceding_commission(self, /, rate: float) -> Layer:
        """
        The same layer with a flat ceding commission: ``rate`` of the
        year's premium (the swing-rated premium, if any) is paid back to
        the cedant. Reinstatement premiums carry none.
        
        Parameters
        ----------
        rate : float
            In ``[0, 1]``.
        
        Returns
        -------
        Layer
        """
    def with_deposit_premium(self, /, amount: float) -> Layer:
        """
        The same layer with its premium set from a deposit quoted for 100%
        of the layer: ``premium = share * amount``.
        
        Parameters
        ----------
        amount : float
        
        Returns
        -------
        Layer
        
        Examples
        --------
        >>> from prospicio.reinsurance import Layer
        >>> Layer("5x5", 5e6, 5e6, share=0.4).with_deposit_premium(1e6).premium
        400000.0
        """
    def with_loss_corridor(self, /, lower: float, upper: float, retained: float = 1.0) -> Layer:
        """
        The same layer with a loss corridor.
        
        Of the annual layer loss at 100% after the annual deductible, the
        cedant keeps ``retained`` of the part between ``lower`` and
        ``upper``; the annual limit then caps what is left, so the reinsurer
        still pays up to the full annual limit. Reinstatement premiums
        follow the loss after the corridor. A corridor quoted as loss
        ratios ``lr`` on the reinsurer's premium ``P`` for a placed share
        ``s`` is ``lr * P / s`` (for a quota share, ``P / s`` is the
        subject premium).
        
        Parameters
        ----------
        lower : float
            Non-negative.
        upper : float
            Finite, above ``lower``.
        retained : float, default 1.0
            Share of the band the cedant keeps, in ``(0, 1]``.
        
        Returns
        -------
        Layer
        
        Examples
        --------
        >>> from prospicio.reinsurance import Layer
        >>> qs = Layer.quota_share("QS", 0.3).with_loss_corridor(70.0, 90.0)
        >>> round(qs.ceded([50.0, 30.0]), 12), round(qs.ceded([120.0]), 12)
        (21.0, 30.0)
        """
    def with_premium_rate(self, /, rate: float, subject_premium: float) -> Layer:
        """
        The same layer with its premium set as a rate on the subject
        premium: ``premium = share * rate * subject_premium``. A quota
        share's ceded premium is ``with_premium_rate(1.0, subject_premium)``.
        
        Parameters
        ----------
        rate : float
        subject_premium : float
        
        Returns
        -------
        Layer
        
        Examples
        --------
        >>> from prospicio.reinsurance import Layer
        >>> Layer.quota_share("QS", 0.3).with_premium_rate(1.0, 1000.0).premium
        300.0
        """
    def with_profit_commission(self, /, share: float, allowance: float = 0.0) -> Layer:
        """
        The same layer with a profit commission: ``share`` of ``max(premium
        * (1 - allowance) - ceded, 0)``, as ``aggregate``'s ``pc <share>
        after <allowance>``.
        
        The allowance is the reinsurer's expenses and margin as a fraction
        of premium, including any ceding commission the contract deducts
        before profit. Set the premium first; it cannot be combined with
        swing rating or paid reinstatements.
        
        Parameters
        ----------
        share : float
            In ``[0, 1]``.
        allowance : float, default 0.0
        
        Returns
        -------
        Layer
        
        Examples
        --------
        >>> from prospicio.reinsurance import Layer
        >>> qs = (Layer.quota_share("QS", 0.5).with_premium_rate(1.0, 200.0)
        ...       .with_profit_commission(0.25, 0.1))
        >>> round(qs.profit_commission_for(60.0), 12)
        7.5
        """
    def with_rate_on_line(self, /, rol: float) -> Layer:
        """
        The same layer with its premium set from a rate on line:
        ``premium = share * rol * limit``. Needs a finite limit.
        
        Parameters
        ----------
        rol : float
        
        Returns
        -------
        Layer
        
        Examples
        --------
        >>> from prospicio.reinsurance import Layer
        >>> Layer("10x10", 10e6, 10e6).with_rate_on_line(0.125).premium
        1250000.0
        """
    def with_sliding_scale(self, /, anchors: Sequence[tuple[float, float]]) -> Layer:
        """
        The same layer with a sliding-scale ceding commission.
        
        The commission rate at the year's ceded loss ratio (ceded loss over
        premium) is interpolated linearly between ``(commission,
        loss_ratio)`` anchors and flat beyond the first and last, as
        ``aggregate``'s ``slide``. Set the premium first; it cannot be
        combined with swing rating or paid reinstatements, and replaces a
        flat commission.
        
        Parameters
        ----------
        anchors : list of (float, float)
            ``(commission, loss_ratio)`` pairs; the commission must not rise
            with the loss ratio.
        
        Returns
        -------
        Layer
        
        Examples
        --------
        >>> from prospicio.reinsurance import Layer
        >>> qs = (Layer.quota_share("QS", 0.5).with_premium_rate(1.0, 200.0)
        ...       .with_sliding_scale([(0.45, 0.60), (0.25, 0.70), (0.19, 0.80)]))
        >>> round(qs.ceding_commission_for(65.0), 12)
        35.0
        """
    def with_swing_rating(self, /, basic: float, lcm: float, minimum: float |None = None, maximum: float |None = None) -> Layer:
        """
        The same layer, swing rated: the year's premium is ``clip(share *
        basic + lcm * ceded, share * minimum, share * maximum)``.
        
        The terms are quoted for 100% of the layer, as in ``aggregate``; the
        ceded loss is already at the placed share. The swing premium
        replaces the fixed premium in the tower's results. It cannot be
        combined with paid reinstatements, a sliding scale or a profit
        commission.
        
        Parameters
        ----------
        basic : float
        lcm : float
            Loss conversion factor, such as ``100 / 80``.
        minimum : float, optional
            Defaults to ``basic``.
        maximum : float, optional
            Defaults to no cap.
        
        Returns
        -------
        Layer
        
        Examples
        --------
        >>> from prospicio.reinsurance import Layer
        >>> layer = Layer("L", 1000.0, 0.0).with_swing_rating(0.0, 1.25, 100.0, 300.0)
        >>> [layer.premium_for(x) for x in (40.0, 200.0, 400.0)]
        [100.0, 250.0, 300.0]
        """

@final
class LayerStatistics:
    """
    A per-occurrence layer's statistics, from
    ``occurrence_layer_statistics``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def aggregate(self, /) -> CompoundMoments:
        """
        The annual ceded loss, at the share.
        """
    @property
    def attachment(self, /) -> float:
        """
        The attachment.
        """
    @property
    def count(self, /) -> CompoundMoments:
        """
        The number of claims that reach the layer.
        """
    @property
    def limit(self, /) -> float:
        """
        The per-occurrence limit.
        """
    @property
    def loss_on_line(self, /) -> float:
        """
        The annual ceded mean over ``share * limit``.
        """
    @property
    def pr_attach(self, /) -> float:
        """
        ``P(X > attachment)``, per claim.
        """
    @property
    def pr_detach(self, /) -> float:
        """
        ``P(X >= attachment + limit)``, per claim.
        """
    @property
    def pr_loss(self, /) -> float:
        """
        The chance of a loss to the layer in the year.
        """
    @property
    def severity(self, /) -> CompoundMoments:
        """
        The ceded loss of a claim that reaches the layer, at the share.
        """
    @property
    def share(self, /) -> float:
        """
        The placed share.
        """

@final
class Ledger:
    """
    A P&L ledger, as ``aggregate``'s ``PnL``: a source (a :class:`Grid`, or
    values and their probabilities) and its groups, with every row (each
    leg, each side's total, each group's result, running nets, tier and
    net-of blocks, and with several groups the grand rows and the total
    impact) a sum of signed legs at each atom, and its exact distribution.
    
    The scenario columns ``k01`` to ``k99`` (``aggregate``'s ``κ``) are each
    row's conditional mean given that the grand result equals its quantile
    at that level, so each column is one state of the world and foots down
    the sheet. Rows are in the payoff convention: ``k01`` is the bad year.
    The source's probabilities are rescaled to sum to 1.
    
    Parameters
    ----------
    source : Grid or tuple of (list of float, list of float)
    groups : list of Group
    name : str, default "ledger"
    result_name : str, default "result"
    spans : list of tuple of (str, int, int), optional
        ``(label, lo, hi)``: groups ``lo`` to ``hi - 1`` with their own
        subtotal block, and a net-of block when a later group follows.
    force_tower : bool, default False
        Present one group as a tower, with a step level.
    
    Examples
    --------
    >>> from prospicio.aggregate import Group, Ledger, Leg
    >>> book = Group("Book", "sell", [Leg("Premium", 10.0)], [Leg("Loss", lambda x: x)])
    >>> pnl = Ledger(([2.0, 8.0], [0.5, 0.5]), [book], name="Book")
    >>> pnl.result.mean()
    5.0
    >>> s = pnl.sheet()
    >>> s["label"], s["k01"]
    (['Premium', 'Loss', 'Total'], [10.0, -8.0, 2.0])
    """
    def __new__(cls, /, source: Any, groups: Sequence[Group], *, name: str = ..., result_name: str = ..., spans: Sequence[tuple[str, int, int]] |None = None, force_tower: bool = False) -> Ledger: ...
    def __repr__(self, /) -> str: ...
    def card(self, /) -> dict:
        """
        The card, ``aggregate``'s ``summary_df``: each block's consideration,
        obligation and margin (and on a tower each step's running net and
        the covers' total impact), with ``EX``, ``SD``, ``CV``, ``Skew``,
        ``P01``, ``Median`` and ``P99`` of each row's own distribution.
        
        Returns
        -------
        dict
        """
    def legs(self, /) -> dict:
        """
        The declared legs, ``aggregate``'s ``legs_df``: step, side, label,
        kind, ``EX`` and ``SD``.
        
        Returns
        -------
        dict
        """
    def marginal(self, /) -> dict:
        """
        Each row's own quantiles, ``aggregate``'s ``economic_marginal_df``:
        ``P01`` to ``P99``, which do not add down the sheet.
        
        Returns
        -------
        dict
        """
    @property
    def name(self, /) -> str:
        """
        The ledger's name.
        """
    @property
    def prob_eq_0(self, /) -> float:
        """
        The chance that the result is 0.
        """
    def ratios(self, /) -> dict:
        """
        The ratio exhibit, ``aggregate``'s ``economic_ratios_df``: for each
        group, span, net-of block and ``All``, premium ``P``, loss ``L`` net
        of recoveries, expense ``E`` net of commission and the result ``M``
        (mean amounts, signed for the block's own view), the result's ``SD``,
        the ratios of means ``LR``, ``ER``, ``CR``, the means of ratios
        ``E_LR``, ``E_ER``, ``E_CR`` (where the premium never vanishes), and
        ``P_share`` and ``M_share`` of the first block. Legs without a kind
        count as premium on the consideration side and loss on the other.
        
        Returns
        -------
        dict
        """
    @property
    def result(self, /) -> Discrete:
        """
        The (grand) result's distribution.
        """
    def row(self, /, label: str) -> Discrete:
        """
        The distribution of the row labelled ``label``.
        
        Parameters
        ----------
        label : str
        
        Returns
        -------
        Discrete
        """
    @property
    def rows(self, /) -> list[str]:
        """
        The rows' labels, in sheet order.
        """
    def sheet(self, /) -> dict:
        """
        The sheet, ``aggregate``'s ``economic_df``: every row with its mean,
        standard deviation, coefficient of variation, skewness and scenario
        columns ``k01`` to ``k99``.
        
        Returns
        -------
        dict
            Columns ``step`` (on a tower), ``side``, ``label``, ``row`` (the
            row's key), ``EX``, ``SD``, ``CV``, ``Skew`` and the scenarios.
        """
    def validation(self, /) -> dict:
        """
        Each rebucketed leg's exact mean against its grid's,
        ``aggregate``'s ``validation_df``. Empty when no leg has a ``step``.
        
        Returns
        -------
        dict
            Columns ``label``, ``EX``, ``Est``, ``abs_err`` and ``rel_err``.
        """

@final
class Leg:
    """
    One cash flow of a :class:`Group`: a constant, or a function of the
    source's value, written as a positive magnitude; the group's role signs
    it. A function is called once with all the source's values (a numpy
    array when numpy is installed, else a list) and returns one value per
    value, or a single number.
    
    Parameters
    ----------
    label : str
        Unique across the ledger.
    value : float or callable
    kind : str, optional
        ``premium``, ``loss``, ``expense``, ``recovery`` or ``commission``.
    step : float, optional
        Put the leg's own distribution on a grid of this step (``aggregate``'s
        ``bs``), scattering each value linearly to its two neighbours, which
        keeps its mean. Only its marginal quantiles change; its mean, the
        sums down the sheet and the scenario columns stay exact.
        :meth:`Ledger.validation` compares the two means.
    
    Examples
    --------
    >>> from prospicio.aggregate import Leg
    >>> Leg("Loss", lambda x: x, kind="loss")
    Leg("Loss")
    """
    def __new__(cls, /, label: str, value: Any, kind: str |None = None, step: float |None = None) -> Leg: ...
    def __repr__(self, /) -> str: ...
    @property
    def label(self, /) -> str:
        """
        The label.
        """

@final
class LimitedFluctuation:
    """
    Limited fluctuation (classical) credibility: the expected number of
    claims for the observed value to lie within ``k`` of its mean with
    probability ``p``, under a Poisson count and the normal approximation.
    
    With ``z = Phi^-1((1 + p) / 2)`` and ``n_F = (z / k)**2``, the full
    standards in claims are ``n_F`` for the frequency, ``n_F * cv**2`` for
    the severity and ``n_F * (1 + cv**2)`` for the pure premium; partial
    credibility is ``min(1, sqrt(n / standard))``.
    
    Parameters
    ----------
    p : float, default 0.90
    k : float, default 0.05
    
    Raises
    ------
    ValueError
        If ``p`` is not in ``(0, 1)`` or ``k`` is not positive.
    
    Examples
    --------
    >>> from prospicio.credibility import LimitedFluctuation
    >>> lf = LimitedFluctuation(0.90, 0.05)
    >>> round(lf.frequency_standard())
    1082
    """
    def __new__(cls, /, p: float = 0.90, k: float = 0.05) -> LimitedFluctuation: ...
    def __repr__(self, /) -> str: ...
    def credibility(self, /, n: float, standard: float) -> float:
        """
        Credibility ``min(1, sqrt(n / standard))`` of ``n`` claims.
        
        Parameters
        ----------
        n : float
        standard : float
        
        Returns
        -------
        float
        """
    def frequency_standard(self, /) -> float:
        """
        Full standard for the frequency, in expected claims.
        
        Returns
        -------
        float
        """
    def pure_premium_standard(self, /, cv: float) -> float:
        """
        Full standard for the pure premium, in expected claims.
        
        Parameters
        ----------
        cv : float
            The severity's coefficient of variation.
        
        Returns
        -------
        float
        """
    def severity_standard(self, /, cv: float) -> float:
        """
        Full standard for the severity, in claims.
        
        Parameters
        ----------
        cv : float
            The severity's coefficient of variation.
        
        Returns
        -------
        float
        """

@final
class LogAffinePareto:
    """
    Log-affine local Pareto: the local alpha
    ``alpha0 * (1 + gamma * ln(x / t))`` rises linearly in the log of the
    amount, so ``P(X > x) = exp(-alpha0 L - alpha0 gamma L**2 / 2)`` with
    ``L = ln(x / t)``.
    
    Parameters
    ----------
    t : float
        Threshold; finite and positive.
    alpha0 : float
        Local alpha at ``t``; finite and positive.
    gamma : float
        Non-negative; 0 gives the Pareto.
    
    Examples
    --------
    >>> from prospicio.distributions import LogAffinePareto
    >>> d = LogAffinePareto.from_delta(1e6, 1.5, 0.5)
    >>> round(d.local_alpha(2e6), 12)
    2.0
    """
    def __getnewargs__(self, /) -> tuple[float, float, float]: ...
    def __new__(cls, /, t: float, alpha0: float, gamma: float) -> LogAffinePareto: ...
    def __repr__(self, /) -> str: ...
    @property
    def alpha0(self, /) -> float:
        """
        Local alpha at the threshold.
        """
    def cdf(self, /, x: float) -> float:
        """
        Distribution function ``P(X <= x)``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @property
    def delta(self, /) -> float:
        """
        ``delta = alpha0 * gamma * ln 2``.
        """
    @staticmethod
    def from_delta(t: float, alpha0: float, delta: float) -> LogAffinePareto:
        """
        The distribution from ``delta = alpha0 * gamma * ln 2``, the rise
        in the local alpha each time the amount doubles.
        
        Parameters
        ----------
        t : float
        alpha0 : float
        delta : float
        
        Returns
        -------
        LogAffinePareto
        """
    @property
    def gamma(self, /) -> float:
        """
        ``gamma``.
        """
    def layer(self, /, limit: float, attachment: float) -> float:
        """
        Expected loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
            ``inf`` for an unlimited layer.
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_second_moment(self, /, limit: float, attachment: float) -> float:
        """
        Second moment of the loss to the layer ``limit`` xs
        ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_variance(self, /, limit: float, attachment: float) -> float:
        """
        Variance of the loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def lev(self, /, limit: float) -> float:
        """
        Limited expected value ``E[min(X, limit)]``.
        
        Parameters
        ----------
        limit : float
        
        Returns
        -------
        float
        """
    def local_alpha(self, /, x: float) -> float:
        """
        The local Pareto alpha ``-x S'(x) / S(x)`` at ``x``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    def mean(self, /) -> float:
        """
        Mean of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """
    def quantile(self, /, p: float) -> float:
        """
        Quantile: the smallest ``x`` with ``P(X <= x) >= p``.
        
        Parameters
        ----------
        p : float
            Probability in ``[0, 1]``.
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def sample(self, /, n: int, seed: int, stream: int = 0) -> list[float]:
        """
        ``n`` draws from stream ``stream`` of the generator keyed by
        ``seed``.
        
        Parameters
        ----------
        n : int
        seed : int
        stream : int, default 0
        
        Returns
        -------
        list of float
        """
    def std(self, /) -> float:
        """
        Standard deviation of the distribution.
        
        Returns
        -------
        float
        """
    def stop_loss(self, /, retention: float) -> float:
        """
        Expected excess over a retention, ``E[max(X - retention, 0)]``.
        
        Parameters
        ----------
        retention : float
        
        Returns
        -------
        float
        """
    def survival(self, /, x: float) -> float:
        """
        Survival function ``P(X > x)``, accurate far into the tail.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @property
    def t(self, /) -> float:
        """
        Threshold ``t``.
        """
    def variance(self, /) -> float:
        """
        Variance of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """

@final
class Loglogistic:
    """
    Loglogistic (Fisk) distribution with shape ``alpha`` and scale
    ``theta`` (the median): ``F(x) = (x/theta)**alpha / (1 + (x/theta)**alpha)``,
    as SciPy's ``fisk``. Its ``cdf`` is Clark's loglogistic growth curve.
    The mean is infinite for ``alpha <= 1`` and the variance for
    ``alpha <= 2``; limited and layer moments always exist.
    
    Parameters
    ----------
    shape : float
    scale : float
    
    Examples
    --------
    >>> from prospicio.distributions import Loglogistic
    >>> Loglogistic(1.0, 2.0).cdf(3.0)
    0.6
    """
    def __getnewargs__(self, /) -> tuple[float, float]: ...
    def __new__(cls, /, shape: float, scale: float) -> Loglogistic: ...
    def __repr__(self, /) -> str: ...
    def cdf(self, /, x: float) -> float:
        """
        Distribution function ``P(X <= x)``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    def layer(self, /, limit: float, attachment: float) -> float:
        """
        Expected loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
            ``inf`` for an unlimited layer.
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_second_moment(self, /, limit: float, attachment: float) -> float:
        """
        Second moment of the loss to the layer ``limit`` xs
        ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_variance(self, /, limit: float, attachment: float) -> float:
        """
        Variance of the loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def lev(self, /, limit: float) -> float:
        """
        Limited expected value ``E[min(X, limit)]``.
        
        Parameters
        ----------
        limit : float
        
        Returns
        -------
        float
        """
    def mean(self, /) -> float:
        """
        Mean of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """
    def quantile(self, /, p: float) -> float:
        """
        Quantile: the smallest ``x`` with ``P(X <= x) >= p``.
        
        Parameters
        ----------
        p : float
            Probability in ``[0, 1]``.
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def sample(self, /, n: int, seed: int, stream: int = 0) -> list[float]:
        """
        ``n`` draws from stream ``stream`` of the generator keyed by
        ``seed``.
        
        Parameters
        ----------
        n : int
        seed : int
        stream : int, default 0
        
        Returns
        -------
        list of float
        """
    @property
    def scale(self, /) -> float:
        """
        Scale ``theta``, the median.
        """
    @property
    def shape(self, /) -> float:
        """
        Shape ``alpha``.
        """
    def std(self, /) -> float:
        """
        Standard deviation of the distribution.
        
        Returns
        -------
        float
        """
    def stop_loss(self, /, retention: float) -> float:
        """
        Expected excess over a retention, ``E[max(X - retention, 0)]``.
        
        Parameters
        ----------
        retention : float
        
        Returns
        -------
        float
        """
    def survival(self, /, x: float) -> float:
        """
        Survival function ``P(X > x)``, accurate far into the tail.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    def variance(self, /) -> float:
        """
        Variance of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """

@final
class Lognormal:
    """
    Lognormal distribution: ``ln X ~ Normal(meanlog, sdlog**2)``.
    
    Parameters
    ----------
    meanlog : float
        Mean of ``ln X``.
    sdlog : float
        Standard deviation of ``ln X``; must be positive.
    
    Raises
    ------
    ValueError
        If ``meanlog`` is not finite or ``sdlog`` is not positive and finite.
    
    Examples
    --------
    >>> from prospicio.distributions import Lognormal
    >>> d = Lognormal.from_mean_cv(1000.0, 0.5)
    >>> round(d.mean(), 6)
    1000.0
    """
    def __getnewargs__(self, /) -> tuple[float, float]: ...
    def __new__(cls, /, meanlog: float, sdlog: float) -> Lognormal: ...
    def __repr__(self, /) -> str: ...
    def cdf(self, /, x: float) -> float:
        """
        Distribution function ``P(X <= x)``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @staticmethod
    def from_mean_cv(mean: float, cv: float) -> Lognormal:
        """
        Lognormal with the given mean and coefficient of variation.
        
        Parameters
        ----------
        mean : float
            Mean of ``X``; must be positive.
        cv : float
            Coefficient of variation of ``X``; must be positive.
        
        Returns
        -------
        Lognormal
        
        Raises
        ------
        ValueError
            If ``mean`` or ``cv`` is not positive and finite.
        """
    def layer(self, /, limit: float, attachment: float) -> float:
        """
        Expected loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        
        Examples
        --------
        >>> from prospicio.distributions import Lognormal
        >>> d = Lognormal(7.0, 0.5)
        >>> abs(d.layer(1000.0, 0.0) - d.lev(1000.0)) < 1e-9
        True
        """
    def lev(self, /, limit: float) -> float:
        """
        Limited expected value ``E[min(X, limit)]``.
        
        Parameters
        ----------
        limit : float
        
        Returns
        -------
        float
        """
    def mean(self, /) -> float:
        """
        Mean of the distribution.
        
        Returns
        -------
        float
        """
    @property
    def meanlog(self, /) -> float:
        """
        Mean of ``ln X``.
        """
    def quantile(self, /, p: float) -> float:
        """
        Quantile: the smallest ``x`` with ``P(X <= x) >= p``.
        
        Parameters
        ----------
        p : float
            Probability in ``[0, 1]``; ``quantile(1.0)`` is ``inf``.
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def sample(self, /, n: int, seed: int, stream: int = 0) -> list[float]:
        """
        ``n`` draws from stream ``stream`` of the generator keyed by ``seed``.
        
        The same ``(seed, stream)`` gives the same draws in Python, R and Rust.
        
        Parameters
        ----------
        n : int
            Number of draws.
        seed : int
            Generator seed.
        stream : int, default 0
            Stream id; distinct streams are independent.
        
        Returns
        -------
        list of float
        """
    @property
    def sdlog(self, /) -> float:
        """
        Standard deviation of ``ln X``.
        """
    def std(self, /) -> float:
        """
        Standard deviation of the distribution.
        
        Returns
        -------
        float
        """
    def stop_loss(self, /, retention: float) -> float:
        """
        Expected excess over a retention, ``E[max(X - retention, 0)]``,
        accurate far into the tail.
        
        Parameters
        ----------
        retention : float
        
        Returns
        -------
        float
        """
    def variance(self, /) -> float:
        """
        Variance of the distribution.
        
        Returns
        -------
        float
        """

@final
class LossRatioExperience:
    """
    Experience years for the loss ratio method: earned premium brought to
    the current rate level and trended, and losses developed, trended and
    loaded for LAE, combined as the ratio of their totals (Werner and
    Modlin, ch. 8).
    
    Every factor is a list with one value per year, one number for all
    years, or ``None`` for 1.
    
    Parameters
    ----------
    earned_premium : list of float
    losses : list of float
        Reported or paid losses, with whatever LAE they include.
    on_level_factor, premium_trend, development, loss_trend, lae_factor : list of float or float, optional
        ``lae_factor`` is ``1 +`` the LAE ratio not in ``losses``; a net
        trend goes in ``loss_trend``.
    
    Raises
    ------
    ValueError
        For no years, mismatched lengths, or a premium or factor that is not
        positive.
    
    Examples
    --------
    >>> from prospicio.pricing import LossRatioExperience
    >>> e = LossRatioExperience([100.0, 300.0], [50.0, 250.0])
    >>> e.loss_ratio
    0.75
    """
    def __new__(cls, /, earned_premium: Sequence[float], losses: Sequence[float], on_level_factor: Any |None = None, premium_trend: Any |None = None, development: Any |None = None, loss_trend: Any |None = None, lae_factor: Any |None = None) -> LossRatioExperience: ...
    def __repr__(self, /) -> str: ...
    @property
    def loss_ratio(self, /) -> float:
        """
        The projected loss and LAE ratio, the ratio of the totals.
        """
    @property
    def loss_ratio_by_year(self, /) -> list[float]:
        """
        Each year's projected loss ratio.
        """
    @property
    def projected_losses(self, /) -> float:
        """
        Total projected ultimate losses and LAE.
        """
    @property
    def projected_losses_by_year(self, /) -> list[float]:
        """
        Each year's projected losses and LAE.
        """
    @property
    def projected_premium(self, /) -> float:
        """
        Total projected premium at the current rate level.
        """
    @property
    def projected_premium_by_year(self, /) -> list[float]:
        """
        Each year's projected premium.
        """

@final
class LossSensitivePremium:
    """
    A premium that depends on the year's loss ``x``: ``clip(basic + lcm *
    x, minimum, maximum)``.
    
    The premium of a retrospectively rated policy, with ``x`` the account's
    loss (net of any reinsurance that inures to it), and of a swing-rated
    layer (``Layer.with_swing_rating``). For a retro plan, ``basic`` and
    ``lcm`` include the tax multiplier.
    
    Parameters
    ----------
    basic : float
    lcm : float
        Loss conversion factor.
    minimum : float, optional
        Defaults to ``basic``.
    maximum : float, optional
        Defaults to no cap.
    
    Examples
    --------
    >>> from prospicio.reinsurance import LossSensitivePremium
    >>> retro = LossSensitivePremium(1000.0, 1.1, maximum=2500.0)
    >>> [round(p, 9) for p in retro.premium([0.0, 500.0, 1500.0])]
    [1000.0, 1550.0, 2500.0]
    """
    def __new__(cls, /, basic: float, lcm: float, minimum: float |None = None, maximum: float |None = None) -> LossSensitivePremium: ...
    def __repr__(self, /) -> str: ...
    @property
    def basic(self, /) -> float:
        """
        The basic premium.
        """
    @property
    def lcm(self, /) -> float:
        """
        The loss conversion factor.
        """
    @property
    def maximum(self, /) -> float:
        """
        The maximum premium; ``inf`` when uncapped.
        """
    @property
    def minimum(self, /) -> float:
        """
        The minimum premium.
        """
    def premium(self, /, losses: Sequence[float]) -> list[float]:
        """
        The premium for each loss.
        
        Parameters
        ----------
        losses : list of float
        
        Returns
        -------
        list of float
        """

@final
class Mack:
    """
    Mack's distribution-free chain ladder: the chain-ladder projection plus
    the standard error of each origin's reserve and of the total, split into
    process and parameter risk (Mack 1993, 1999).
    
    A tail other than 1 is one more development step, from the oldest age
    to ultimate, with its own sigma and standard error, as R ChainLadder's
    ``MackChainLadder(tail = ...)``; unless given, both are extrapolated
    log-linearly. Every origin, the oldest included, carries the tail's risk.
    A tail below 1 follows chainladder-python: it scales the ultimates and
    carries the risk read where a tail of 1.001 would be. R's
    ``MackChainLadder`` ignores a tail below 1 altogether.
    
    Parameters
    ----------
    average : {"volume", "simple", "regression"}, default "volume"
    sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
    tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
        As ``ChainLadder``; no tail by default.
    tail_sigma : float, optional
        The tail's sigma (R's ``tail.sigma``); extrapolated if not given.
        Unused when the tail factor is 1.
    tail_std_err : float, optional
        The tail factor's standard error (R's ``tail.se``); extrapolated if
        not given. Unused when the tail factor is 1.
    
    Examples
    --------
    >>> from prospicio.reserving import Mack, Triangle
    >>> tri = Triangle.from_long(
    ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
    ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
    ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
    ... )
    >>> fit = Mack().fit(tri, "values")
    >>> fit.total_standard_error > 0 and fit.standard_error[0] == 0
    True
    """
    def __new__(cls, /, average: str = "volume", sigma_interpolation: str = "log-linear", tail: Any |None = None, tail_sigma: float |None = None, tail_std_err: float |None = None) -> Mack: ...
    def __repr__(self, /) -> str: ...
    @property
    def average(self, /) -> str:
        """
        How link ratios are averaged.
        """
    def fit(self, /, triangle: Triangle, column: str) -> MackFit:
        """
        Fits one measure column in every segment of a triangle, each on its
        own.
        
        Parameters
        ----------
        triangle : Triangle
        column : str
        
        Returns
        -------
        MackFit
        
        Raises
        ------
        ValueError
            As ``ChainLadder.fit``, and if the triangle has fewer than three
            ages, a variance parameter can be neither estimated nor
            interpolated, or the tail's sigma or standard error can neither be
            extrapolated nor is given.
        """
    @property
    def sigma_interpolation(self, /) -> str:
        """
        How unestimable variance parameters are filled in.
        """
    @property
    def tail(self, /) -> Any:
        """
        The tail: a constant factor as a number, otherwise its estimator.
        """
    @property
    def tail_sigma(self, /) -> float |None:
        """
        The given tail sigma, or ``None`` to extrapolate it.
        """
    @property
    def tail_std_err(self, /) -> float |None:
        """
        The given standard error of the tail factor, or ``None`` to
        extrapolate it.
        """

@final
class MackBootstrap:
    """
    Mack's bootstrap for the lifetime and one-year views (England, Verrall
    and Wüthrich 2019, Appendix 1): the scaled bias-adjusted residuals of
    the link ratios are resampled into pseudo factors, and each future
    cumulative value, to the last age (``fit``) or over the coming year
    (``one_year``), is drawn from the one before ``C`` (the observed latest
    value for the first) with mean ``f* C`` and Mack's variance
    ``sigma**2 * abs(C)**(2 - alpha)``. The lifetime view's standard
    deviation approximates Mack's analytic standard error. Beside
    ``OdpBootstrap`` (variance ``scale`` times the mean increment), it gives
    the one-year view under Mack's process: with the volume-weighted chain
    ladder and no tail, its standard deviation is
    ``MackFit.claims_development_result()``'s (Merz and Wüthrich) within
    Monte Carlo error. The residuals are centred by default
    (``centre_residuals``), so the mean CDR is Merz and Wüthrich's zero and
    the lifetime mean reserve the chain ladder's, and EVW's Table 4 expected
    reserves agree. Uncentred, as EVW's Appendix 1 is written, the pool's
    non-zero mean biases the pseudo factors: the mean CDR is about -0.2
    (RAA), -0.04 (GenIns) and +0.18 (ABC) times its standard deviation, the
    lifetime mean reserve about +17%, +0.7% and -0.8% off the chain
    ladder's, and the one-year standard deviation up to 1.3% wide on RAA.
    Simulation ``i`` uses random stream ``i`` of ``seed`` for every segment
    in turn.
    
    Parameters
    ----------
    n_sims : int, default 10000
        Number of simulations; positive.
    seed : int, default 0
        Seed of the simulation streams, from 0 to ``2**64 - 1``.
    process : {"gamma", "lognormal", "residuals", "normal", "none"}, default "gamma"
        Process error on each next cumulative value: Gamma or lognormal
        (negated for a negative mean) or normal, with Mack's mean and
        variance; the mean plus a resampled residual times the standard
        deviation, which carries the residuals' mean and variance; or none
        for parameter error only.
    average : {"volume", "simple", "regression"}, default "volume"
        How Mack's model averages the link ratios (its ``alpha``).
    sigma_interpolation : {"log-linear", "mack"}, default "log-linear"
        How a sigma behind a single link ratio is filled in.
    centre_residuals : bool, default True
        Subtract the residuals' mean before resampling them, so that the
        pseudo factors are unbiased, the mean CDR is about zero and the
        lifetime mean reserve is the chain ladder's (without a tail or with
        a constant one). ``False`` resamples them uncentred, as EVW's
        Appendix 1 is written.
    tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
        Development past the oldest age in the lifetime view (``fit``), as
        ``Mack``'s: one more step, to ultimate, with the process error of
        the tail's sigma (Mack 1999). An estimated tail (``TailCurve``,
        ``TailBondy``, ``TailLogLinear``) is refitted on each simulation's
        pseudo factors, which gives its parameter error; a constant one is
        drawn from the lognormal with the factor as mean and the tail's
        standard error. No tail by default. ``one_year`` takes the step for
        the origins whose step falls in the coming year, as
        ``OdpBootstrap``'s does; ``method``'s tail must then be this one.
    tail_sigma : float, optional
        The tail's sigma (R's ``tail.sigma``); extrapolated if not given.
        Unused when the tail factor is 1.
    tail_std_err : float, optional
        The constant tail factor's standard error (R's ``tail.se``);
        extrapolated if not given, unlike ``OdpBootstrap``, which then keeps
        the factor fixed. Unused when the tail factor is 1; an error with an
        estimated tail.
    dependence : {"independent", "synchronized", "rank_correlation"}, default "independent"
        How the segments of a multi-segment triangle depend on each other
        in ``fit`` and ``one_year``: each resamples its own residuals
        independently; every segment resamples the link-ratio residuals of
        the same origins and ages in each simulation (the synchronous bootstrap of
        Taylor and McGuire 2007 and Kirschner, Kerley and Isaacs 2008), so
        the lines' parameter error takes the correlation of their residuals
        and process error stays independent, which needs the same origins,
        ages and observed cells in every segment; or each is bootstrapped
        independently and the segments' simulations are reordered as whole
        rows (Iman–Conover on the segment totals) to the Spearman matrix
        ``spearman`` and then put in a random order, so that row ``i`` is
        no longer simulation ``i`` but any subset of rows is a fair sample.
    spearman : list of list of float, optional
        With ``dependence="rank_correlation"`` only: Spearman's rho between
        the segments' totals, one row and column per segment in index
        order, symmetric with a unit diagonal. It is converted to the normal
        scores' correlation ``2 sin(pi rho / 6)``, which must be positive
        definite.
    
    Raises
    ------
    ValueError
        If ``n_sims`` is zero, a setting is unknown, or ``spearman`` is
        missing for ``"rank_correlation"``, given for another dependence or
        not square.
    TypeError
        If ``tail`` is not a number or a tail estimator.
    OverflowError
        If ``n_sims`` or ``seed`` is negative or too large.
    
    Examples
    --------
    >>> from prospicio.reserving import ChainLadder, Mack, MackBootstrap, Triangle
    >>> tri = Triangle.from_long(
    ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
    ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
    ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
    ... )
    >>> fit = MackBootstrap(n_sims=2000, seed=42).one_year(tri, "values", ChainLadder())
    >>> fit.model
    'mack'
    >>> fit.cdr.variance() ** 0.5 < Mack().fit(tri, "values").total_standard_error
    True
    """
    def __new__(cls, /, n_sims: int = 10000, seed: int = 0, process: str = "gamma", average: str = "volume", sigma_interpolation: str = "log-linear", centre_residuals: bool = True, tail: Any |None = None, tail_sigma: float |None = None, tail_std_err: float |None = None, dependence: str = "independent", spearman: Sequence[Sequence[float]] |None = None) -> MackBootstrap: ...
    def __repr__(self, /) -> str: ...
    @property
    def average(self, /) -> str:
        """
        How Mack's model averages the link ratios.
        """
    @property
    def centre_residuals(self, /) -> bool:
        """
        Whether the residuals are centred before they are resampled.
        """
    @property
    def dependence(self, /) -> str:
        """
        How the segments depend on each other: ``"independent"``,
        ``"synchronized"`` or ``"rank_correlation"``.
        """
    def fit(self, /, triangle: Triangle, column: str) -> MackBootstrapFit:
        """
        The lifetime view: bootstraps one measure column in every segment of
        a cumulative triangle, each with its own Mack model and residuals,
        into one joint distribution of the reserves (EVW's Appendix 1), the
        segments depending on each other as ``dependence`` says. Each
        simulation resamples the residuals into pseudo factors and draws
        every cumulative value from the latest observed one to the last age,
        each from the one before, with Mack's mean and variance; an origin's
        reserve is its last drawn value less its latest. With a ``tail``
        each origin takes one more step, to ultimate, with the tail's sigma,
        the tail factor refitted (an estimated tail) or drawn (a constant
        one) per simulation. The standard deviation approximates Mack's
        analytic standard error (``Mack.fit``, with the same tail; for an
        estimated tail the parameter error is the refit's, not Mack's
        extrapolated one); the mean is the chain ladder's reserve with
        ``centre_residuals`` (the default), without a tail or with a
        constant one, about 17% above it on RAA without centring.
        
        Parameters
        ----------
        triangle : Triangle
            Cumulative, with any number of segments and any development
            grain, every origin observed from the first age to its latest
            with no negative value.
        column : str
        
        Returns
        -------
        MackBootstrapFit
        
        Raises
        ------
        ValueError
            As ``Mack.fit``, if an origin has a gap before its latest age, if
            a cumulative value is negative, if synchronized segments differ in
            their origins, ages or observed cells, if ``spearman`` does not
            fit the segments, if ``tail_std_err`` is given with an estimated
            tail, or if the tail cannot be refitted on some simulation's
            pseudo factors (the message counts them and gives one).
        
        Examples
        --------
        >>> from prospicio.reserving import MackBootstrap, Triangle
        >>> tri = Triangle.from_long(
        ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
        ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
        ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
        ... )
        >>> fit = MackBootstrap(n_sims=2000, seed=42).fit(tri, "values")
        >>> fit.reserves.components()
        [('2020',), ('2021',), ('2022',), ('2023',)]
        >>> abs(fit.reserves.mean() / fit.chain_ladder.total_reserve - 1) < 0.05
        True
        """
    @property
    def n_sims(self, /) -> int:
        """
        Number of simulations.
        """
    def one_year(self, /, triangle: Triangle, column: str, method: Any, exposure: str |None = None) -> OneYearFit:
        """
        The one-year view of any reserving method under Mack's process, as
        ``OdpBootstrap.one_year``: each simulation draws the cells of the
        coming twelve months from Mack's bootstrap, each from the one before
        with the same pseudo factors, appends them to the triangle, refits
        ``method`` and records ``CDR = opening ultimate - closing
        ultimate``. Without a bootstrap ``tail``, development past the
        oldest age moves only through ``method``'s refitted tail; with one,
        which must be ``method``'s, the origins whose tail step falls in
        the coming year take Mack's step to ultimate and are closed, as in
        ``OdpBootstrap.one_year``.
        
        Parameters
        ----------
        triangle : Triangle
            Cumulative, with any number of segments and any development
            grain, every origin observed from the first age to its latest
            with no negative value.
        column : str
        method : ChainLadder, ExpectedLoss, BornhuetterFerguson, Benktander or CapeCod
            The method refitted at the start and at the end of the year.
        exposure : str, optional
            The exposure column; required by the expected-loss methods, not
            taken by ``ChainLadder``.
        
        Returns
        -------
        OneYearFit
            With ``model == "mack"``.
        
        Raises
        ------
        TypeError
            If ``method`` is not one of the classes above.
        ValueError
            As ``OdpBootstrap.one_year`` and ``Mack.fit``, if a cumulative
            value is negative, and if the bootstrap has a ``tail`` that is
            not ``method``'s.
        """
    @property
    def process(self, /) -> str:
        """
        Process error: ``"gamma"``, ``"lognormal"``, ``"residuals"``,
        ``"normal"`` or ``"none"``.
        """
    @property
    def seed(self, /) -> int:
        """
        Seed of the simulation streams.
        """
    @property
    def sigma_interpolation(self, /) -> str:
        """
        How a sigma behind a single link ratio is filled in.
        """
    @property
    def spearman(self, /) -> list[list[float]] |None:
        """
        The Spearman matrix of ``dependence="rank_correlation"``; ``None``
        otherwise.
        """
    @property
    def tail(self, /) -> Any:
        """
        The lifetime view's tail: a constant factor as a number, otherwise
        its estimator.
        """
    @property
    def tail_sigma(self, /) -> float |None:
        """
        The given tail sigma, or ``None`` to extrapolate it.
        """
    @property
    def tail_std_err(self, /) -> float |None:
        """
        The given standard error of a constant tail factor, or ``None`` to
        extrapolate it.
        """

@final
class MackBootstrapFit:
    """
    A fitted bootstrap of Mack's model, the lifetime view, of every
    segment (``MackBootstrap.fit``).
    
    ``reserves`` is one joint distribution with the triangle's keys and
    ``"origin"`` as dimensions, so ``reserves.aggregate(["lob"])`` keeps the
    dependence between segments. Per-origin lists run over the origins of
    each segment in turn, like the rows of ``to_frame()`` and the
    components of ``reserves``. ``residuals`` needs a single-segment fit;
    for several segments use ``segment(...)``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def chain_ladder(self, /) -> ChainLadderFit:
        """
        The chain ladder of Mack's model (its averaging), with the
        bootstrap's tail: the reserves the bootstrap's mean equals with
        ``centre_residuals=True``, the default, without a tail or with a
        constant one (an estimated tail's refit has its own mean).
        """
    @property
    def development(self, /) -> list[int]:
        """
        Development ages in months.
        """
    def development_frame(self, /) -> Any:
        """
        The chain ladders' development factors, one row per segment and
        age, as ``ChainLadderFit.development_frame``. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def index(self, /) -> list[Any]:
        """
        Label of each segment, as ``Triangle.index``.
        """
    @property
    def keys(self, /) -> list[str]:
        """
        Names of the triangle's key columns; empty without keys.
        """
    @property
    def mack(self, /) -> MackFit:
        """
        Mack's model on the observed triangle, with the bootstrap's tail:
        the factors and sigmas the simulation uses, and the analytic
        standard errors the simulated standard deviations approximate (with
        an estimated tail, up to the refit's own parameter error).
        """
    @property
    def origins(self, /) -> list[str]:
        """
        Origin period of each per-origin value and reserve component.
        """
    @property
    def reserves(self, /) -> PredictiveDistribution:
        """
        Joint distribution of the reserve (each origin's last simulated
        cumulative value less its latest) by segment and origin: the
        triangle's keys and ``"origin"`` are its dimensions, one component
        per segment and origin, one row per simulation. Its ``mean`` and
        ``quantile`` describe the total reserve. Columns of
        ``draw_matrix()`` follow ``origins``.
        """
    @property
    def residuals(self, /) -> list[list[float]]:
        """
        Mack's scaled bias-adjusted residuals of the link ratios,
        ``[origin][development]``, ``[o][k]`` the link from age ``k`` to
        ``k + 1``; ``nan`` where there is none, from a zero, or behind a
        factor with a single link ratio or a zero sigma. Never centred.
        """
    def segment(self, /, **keys) -> MackBootstrapFit:
        """
        The bootstrap of one segment, chosen by key values as
        ``ChainLadderFit.segment``, with its part of the joint reserves
        (same dimensions).
        
        Returns
        -------
        MackBootstrapFit
        """
    def to_frame(self, /) -> Any:
        """
        One row per segment and origin: the key columns, ``origin``, the
        chain ladder's ``latest``, ``ultimate`` and ``reserve``, and the
        ``mean`` and ``std_dev`` of the bootstrapped reserve. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    def totals_frame(self, /) -> Any:
        """
        One row per segment: the key columns, the chain ladder's totals, and
        the ``mean`` and ``std_dev`` of the segment's bootstrapped total
        reserve. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """

@final
class MackFit:
    """
    A fitted Mack model of every segment: the chain-ladder fields, plus
    standard errors of each origin's reserve and of each segment's total.
    
    Per-origin lists run over the origins of each segment in turn, like the
    rows of ``to_frame()``. Per-age lists, the tail and the totals'
    standard errors need a single-segment fit; for several segments use
    ``development_frame()``, ``totals_frame()`` (the totals' standard errors
    and the tail) or ``segment(...)``.
    ``total_ultimate`` and ``total_reserve`` sum over every segment.
    """
    def __repr__(self, /) -> str: ...
    @property
    def cdf(self, /) -> list[float]:
        """
        Age-to-ultimate factors, including the tail.
        """
    @property
    def chain_ladder(self, /) -> ChainLadderFit:
        """
        The underlying chain-ladder projection.
        """
    def claims_development_result(self, /) -> ClaimsDevelopmentResult:
        """
        Merz and Wüthrich's (2008) one-year view: the standard error of the
        claims development result of each origin and in total, in the next
        calendar year and in every later one, as R ChainLadder's
        ``CDR(MackChainLadder(x), dev = "all")``.
        
        Returns
        -------
        ClaimsDevelopmentResult
        
        Raises
        ------
        ValueError
            If the fit has several segments (use ``segment(...)``), the
            factors are not volume-weighted, the fit has a tail (a factor
            other than 1, or one that replaces estimated factors), or the
            latest values do not lie on one calendar diagonal with one new
            origin per period.
        """
    @property
    def development(self, /) -> list[int]:
        """
        Development ages in months.
        """
    def development_frame(self, /) -> Any:
        """
        One row per segment and age, as ``ChainLadderFit.development_frame``.
        Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def estimated_ldf(self, /) -> list[float]:
        """
        Factors as estimated, as ``ChainLadderFit.estimated_ldf``.
        """
    @property
    def index(self, /) -> list[Any]:
        """
        Label of each segment, as ``Triangle.index``.
        """
    @property
    def keys(self, /) -> list[str]:
        """
        Names of the triangle's key columns; empty without keys.
        """
    @property
    def latest(self, /) -> list[float]:
        """
        Latest observed cumulative value per origin.
        """
    @property
    def ldf(self, /) -> list[float]:
        """
        Selected age-to-age factors, as ``ChainLadderFit.ldf``.
        """
    @property
    def origins(self, /) -> list[str]:
        """
        Origin period of each per-origin value.
        """
    @property
    def parameter_risk(self, /) -> list[float]:
        """
        Parameter (estimation) standard error per origin.
        """
    @property
    def process_risk(self, /) -> list[float]:
        """
        Process standard error per origin.
        """
    @property
    def reserve(self, /) -> list[float]:
        """
        Reserve per origin.
        """
    def segment(self, /, **keys) -> MackFit:
        """
        The fit of one segment, chosen by key values as
        ``ChainLadderFit.segment``.
        
        Returns
        -------
        MackFit
        """
    @property
    def sigma(self, /) -> list[float]:
        """
        Variance parameter of each factor.
        """
    @property
    def standard_error(self, /) -> list[float]:
        """
        Mack standard error per origin: ``sqrt(process**2 + parameter**2)``.
        """
    @property
    def std_err(self, /) -> list[float]:
        """
        Standard error of each factor.
        """
    @property
    def tail(self, /) -> float:
        """
        Tail factor from the oldest age to ultimate.
        """
    @property
    def tail_attachment_age(self, /) -> int:
        """
        Age from which ``ldf`` holds the tail's factors, as
        ``ChainLadderFit.tail_attachment_age``.
        """
    @property
    def tail_ldf(self, /) -> list[float]:
        """
        Factors past the oldest age, as ``ChainLadderFit.tail_ldf``.
        """
    @property
    def tail_sigma(self, /) -> float:
        """
        The tail's sigma used in the process risk: given, or extrapolated
        log-linearly; 0 without a tail (a factor of 1).
        """
    @property
    def tail_std_err(self, /) -> float:
        """
        The tail factor's standard error used in the parameter risk: given,
        or extrapolated log-linearly; 0 without a tail (a factor of 1).
        """
    def to_frame(self, /) -> Any:
        """
        One row per segment and origin: the key columns, ``origin``,
        ``latest``, ``ultimate``, ``reserve``, ``process_risk``,
        ``parameter_risk`` and ``standard_error``. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def total_cv(self, /) -> float:
        """
        Coefficient of variation of the total reserve.
        """
    @property
    def total_parameter_risk(self, /) -> float:
        """
        Parameter standard error of the total reserve, including the
        correlation between origins that share estimated factors.
        """
    @property
    def total_process_risk(self, /) -> float:
        """
        Process standard error of the total reserve.
        """
    @property
    def total_reserve(self, /) -> float:
        """
        Total reserve across segments and origins.
        """
    @property
    def total_standard_error(self, /) -> float:
        """
        Mack standard error of the total reserve.
        """
    @property
    def total_ultimate(self, /) -> float:
        """
        Total ultimate across segments and origins.
        """
    def totals_frame(self, /) -> Any:
        """
        One row per segment: the key columns, the segment's total
        ``latest``, ``ultimate`` and ``reserve``, and the ``process_risk``,
        ``parameter_risk`` and ``standard_error`` of its total reserve.
        Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def ultimate(self, /) -> list[float]:
        """
        Projected ultimate per origin.
        """

@final
class Mbbefd:
    """
    The MBBEFD exposure curve and destruction-rate distribution (Bernegger,
    1997), with ``b >= 0`` and ``g >= 1``; ``1/g`` is the probability of a
    total loss.
    
    ``G(x)`` is the share of a risk's expected loss below the fraction
    ``x`` of its maximum possible loss (MPL). ``Mbbefd.swiss_re(c)`` gives
    Bernegger's one-parameter family: ``c = 1.5, 2, 3, 4`` are the Swiss Re
    curves and ``c = 5`` the Lloyd's curve.
    
    Parameters
    ----------
    b : float
    g : float
    
    Examples
    --------
    >>> from prospicio.pricing import Mbbefd
    >>> c3 = Mbbefd.swiss_re(3.0)
    >>> top = c3.layer_share(5e6, 5e6, 10e6)
    >>> bottom = c3.layer_share(5e6, 0.0, 10e6)
    >>> round(top + bottom, 12), top < bottom
    (1.0, True)
    """
    def __new__(cls, /, b: float, g: float) -> Mbbefd: ...
    def __repr__(self, /) -> str: ...
    @property
    def b(self, /) -> float:
        """
        Parameter ``b``.
        """
    def cdf(self, /, x: Sequence[float]) -> list[float]:
        """
        Distribution function of the destruction rate at each ``x``.
        
        Parameters
        ----------
        x : list of float
        
        Returns
        -------
        list of float
        """
    def curve(self, /, x: Sequence[float]) -> list[float]:
        """
        The exposure curve ``G(x)`` at each ``x`` (clamped to [0, 1]).
        
        Parameters
        ----------
        x : list of float
        
        Returns
        -------
        list of float
        """
    @property
    def g(self, /) -> float:
        """
        Parameter ``g``.
        """
    def layer_share(self, /, limit: float, attachment: float, mpl: float) -> float:
        """
        Share of a risk's expected loss in the layer ``limit`` xs
        ``attachment``, for a risk with maximum possible loss ``mpl``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        mpl : float
        
        Returns
        -------
        float
        """
    def mean(self, /) -> float:
        """
        Mean destruction rate, ``1 / G'(0)``.
        
        Returns
        -------
        float
        """
    def rate_quantile(self, /, u: Sequence[float]) -> list[float]:
        """
        Destruction rate (loss over MPL) at each probability ``u`` in
        ``(0, 1)``: draws with this curve as their exposure curve.
        
        Parameters
        ----------
        u : list of float
        
        Returns
        -------
        list of float
        """
    @staticmethod
    def swiss_re(c: float) -> Mbbefd:
        """
        Bernegger's curve ``c``: ``b = exp(3.1 - 0.15 (1 + c) c)``,
        ``g = exp((0.78 + 0.12 c) c)``.
        
        Parameters
        ----------
        c : float
            Non-negative; 0 is the straight line.
        
        Returns
        -------
        Mbbefd
        """
    def total_loss_probability(self, /) -> float:
        """
        Probability of a total loss, ``1/g``.
        
        Returns
        -------
        float
        """

@final
class Meyers:
    """
    Meyers' (2019, CAS Monograph 8) log-normal models of a cumulative
    triangle ``C`` with premium ``P``, for origins ``w = 1..n`` and ages
    ``d = 1..m``: ``log C[w, d]`` is normal with mean
    ``log P[w] + logelr + alpha[w] + beta[d] g[w]`` (``alpha[1] = 0``,
    ``beta[m] = 0``) and sd ``sigma[d]``, decreasing with age. ``g[w]`` is 1
    for the correlated chain ladder without its correlation between origins
    (``"crc"``) and ``(1 - gamma)^(w - 1)`` for the changing settlement rate
    (``"csr"``). The correlated accident year model (``"cay"``) keeps
    ``g[w] = 1`` and adds ``rho`` times the previous origin's residual at
    the same age, ``log C[w - 1, d]`` less its mean, to the mean of every
    origin after the first, so neighbouring origins are correlated; Meyers
    fits it to incurred data. The priors are those of Meyers' Stan code:
    ``logelr`` normal with mean ``logelr_mean`` and sd ``logelr_sd``,
    ``alpha`` and ``beta`` normal around 0, ``gamma`` normal around 0 with
    sd ``gamma_sd``, ``(1 - rho) / 2`` beta with both shapes ``rho_shape``,
    and each step of ``sigma[d]^2`` uniform on ``(1 - exp(-1e-5), 1)``.
    
    The ultimate is the value at the oldest age (no tail): each posterior
    draw gives each open origin a log-normal ultimate; an origin observed at
    the oldest age keeps its value; under ``"cay"`` the ultimates are drawn
    origin by origin, each from the previous origin's residual at the oldest
    age, observed for a closed origin and simulated otherwise.
    The reserve is the ultimate less the latest value of the fitted column
    (an outstanding reserve on paid data, IBNR on incurred data) and can be
    negative. Every observed cell must be positive, and under ``"cay"``
    every observed cell after the first origin needs the previous origin's
    cell at the same age.
    
    One ``seed`` replays a fit exactly, whatever the number of threads: the
    chains and the simulations each take a seed derived from it.
    
    Failed MCMC diagnostics (R-hat above 1.01, bulk or tail ESS below 100
    per chain, any divergent transition) do not fail the fit: ``fit``
    raises one ``UserWarning`` listing them, and they stay in
    ``MeyersFit.diagnostics``. With the defaults, all 120 fits of a seed
    sweep on Meyers' group 353 data (paid and incurred, CRC and CSR, 30
    seeds) passed every threshold, and CAY on incurred passed at 199 of 200
    seeds (one divergent transition in 8,000 draws at the other).
    
    Parameters
    ----------
    model : {"crc", "csr", "cay"}, default "crc"
        The correlated chain ladder without its correlation between origins,
        the changing settlement rate model, or the correlated accident year
        model.
    logelr_mean : float, default -0.4
        Mean of the normal prior on ``logelr``, the log expected loss ratio.
    logelr_sd : float, default 3.162
        Standard deviation of the normal prior on ``logelr`` (3.162 is
        Meyers' rounding of ``sqrt(10)``).
    origin_sd : float, default 3.162
        Standard deviation of the normal prior on each origin effect
        ``alpha[w]``, ``w >= 2``, centred on 0.
    development_sd : float, default 3.162
        Standard deviation of the normal prior on each development effect
        ``beta[d]``, ``d < m``, centred on 0.
    gamma_sd : float, default 0.05
        Standard deviation of the normal prior on ``gamma``, centred on 0
        (``"csr"`` only).
    rho_shape : float, default 2.0
        Both shapes of the beta prior on ``(1 - rho) / 2`` (``"cay"``
        only); Meyers' 2 makes the prior density of ``rho`` proportional to
        ``1 - rho^2``.
    chains, tune, draws : int, default 4, 1000, 2000
        NUTS chains, warm-up draws per chain and kept draws per chain; with
        1,000 kept draws the late ``sigma[d]`` sometimes miss the R-hat
        threshold.
    n_sims : int, default 10000
        Number of predictive simulations, independent of the number of
        posterior draws; positive.
    seed : int, default 0
        Seed of the chains and of the simulations, from 0 to ``2**64 - 1``.
    target_accept : float, default 0.95
        Target mean acceptance rate of the step-size adaptation; at NUTS's
        usual 0.8 the paid fits of group 353 leave divergent transitions.
    max_depth : int, default 10
        Largest NUTS tree depth.
    dependence : {"independent", "rank_correlation", "synchronized"}, default "independent"
        How the segments of a multi-segment triangle depend on each other
        in ``fit_segments`` (``fit`` has one segment and does not use it),
        as ``BayesOdp``'s: each segment's own posterior and simulations, with
        seeds derived from ``seed`` and the segment's position; or the
        segments' simulations reordered as whole rows (Iman–Conover on the
        segments' total reserves, each segment's ultimates moving with its
        reserves) to the Spearman matrix ``spearman`` and then put in a
        random order. ``"synchronized"`` is accepted here, as the bootstraps
        take it, but ``fit_segments`` raises: a posterior has no residuals to
        share.
    spearman : list of list of float, optional
        With ``dependence="rank_correlation"`` only: Spearman's rho between
        the segments' total reserves, as ``OdpBootstrap``'s.
    
    Raises
    ------
    ValueError
        If ``n_sims`` is zero, ``model`` or ``dependence`` is unknown, or
        ``spearman`` is missing for ``"rank_correlation"``, given for another
        dependence or not square.
    OverflowError
        If an integer setting is negative or too large.
    
    Examples
    --------
    >>> import warnings
    >>> from prospicio.reserving import Meyers, Triangle
    >>> paid = [[952, 1529, 2813, 3647, 3724, 3832], [849, 1564, 2202, 2432, 2468],
    ...         [983, 2211, 2830, 3832], [1657, 2685, 3169], [932, 1940], [1162]]
    >>> premium = [5812, 4908, 5454, 5165, 5214, 5230]
    >>> rows = [(1988 + w, 12 * (d + 1), float(v), float(premium[w]))
    ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
    >>> origin, age, values, prem = map(list, zip(*rows))
    >>> tri = Triangle.from_long(origin, age, {"paid": values, "premium": prem})
    >>> with warnings.catch_warnings():
    ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
    ...     fit = Meyers("csr", tune=300, draws=300, n_sims=2000, seed=1).fit(
    ...         tri, "paid", "premium")
    >>> fit.reserves.components()
    [('1988',), ('1989',), ('1990',), ('1991',), ('1992',), ('1993',)]
    >>> fit.parameter_names[0], fit.parameter_names[-1]
    ('logelr', 'sigma[6]')
    >>> fit.premium[:2]
    [5812.0, 4908.0]
    """
    def __new__(cls, /, model: str = "crc", logelr_mean: float = ..., logelr_sd: float = 3.162, origin_sd: float = 3.162, development_sd: float = 3.162, gamma_sd: float = 0.05, rho_shape: float = 2.0, chains: int = 4, tune: int = 1000, draws: int = 2000, n_sims: int = 10000, seed: int = 0, target_accept: float = 0.95, max_depth: int = 10, dependence: str = "independent", spearman: Sequence[Sequence[float]] |None = None) -> Meyers: ...
    def __repr__(self, /) -> str: ...
    @property
    def chains(self, /) -> int:
        """
        Number of NUTS chains.
        """
    @property
    def dependence(self, /) -> str:
        """
        How segments depend on each other in ``fit_segments``:
        ``"independent"``, ``"rank_correlation"`` or ``"synchronized"``.
        """
    @property
    def development_sd(self, /) -> float:
        """
        Standard deviation of each development effect's prior.
        """
    @property
    def draws(self, /) -> int:
        """
        Kept draws per chain.
        """
    def fit(self, /, triangle: Triangle, column: str, premium: str) -> MeyersFit:
        """
        Samples the posterior of one measure column of a single-segment
        cumulative triangle, with each origin's premium from another column
        (its latest value, as ``CapeCod`` reads exposure), and simulates the
        ultimates and reserves. Every origin must be observed from the first
        age up to its latest, every observed cell must be positive, and
        there must be more observed cells than parameters. Raises one
        ``UserWarning`` listing the failed diagnostics, if any.
        
        Parameters
        ----------
        triangle : Triangle
            Cumulative, with one segment.
        column : str
            The losses: paid, or case incurred.
        premium : str
            The premium of each origin.
        
        Returns
        -------
        MeyersFit
        
        Raises
        ------
        ValueError
            If a column is missing, the triangle has several segments or an
            origin has a gap; if a premium is missing or not positive; if a
            prior standard deviation or ``rho_shape`` is not positive and
            finite, ``logelr_mean`` is not finite or a sampler setting is
            invalid; if an observed cell is not positive; if there are no more
            observed cells than parameters; or, under ``"cay"``, if a cell
            lacks the previous origin's cell at the same age.
        """
    def fit_segments(self, /, triangle: Triangle, column: str, premium: str) -> MeyersFits:
        """
        Samples the posterior of one measure column in every segment of a
        cumulative triangle, each segment with its own posterior (the same
        model and settings, sampled independently) and its premium from its
        own ``premium`` column as ``fit`` reads it, into one joint
        distribution of the ultimates and one of the reserves, the segments
        depending on each other as ``dependence`` says. Segment ``k`` (in
        index order) takes seeds derived from ``seed`` and ``k`` for its
        chains and its simulations, so independent segments share no random
        numbers, and a segment's draws differ from ``fit`` of that segment
        alone. Raises one ``UserWarning`` listing the failed diagnostics of
        every segment, each message prefixed with its segment's label.
        
        Parameters
        ----------
        triangle : Triangle
            Cumulative, with any number of segments.
        column : str
            The losses: paid, or case incurred.
        premium : str
            The premium of each origin.
        
        Returns
        -------
        MeyersFits
        
        Raises
        ------
        ValueError
            As ``fit`` for any segment (the message names the segment); if
            ``dependence`` is ``"synchronized"`` (before any sampling); or if
            ``spearman`` does not fit the segments.
        
        Examples
        --------
        >>> import warnings
        >>> from prospicio.reserving import Meyers, Triangle
        >>> paid = [[952, 1529, 2813, 3647, 3724], [849, 1564, 2202, 2432],
        ...         [983, 2211, 2830], [1657, 2685], [932]]
        >>> premium = [5812, 4908, 5454, 5165, 5214]
        >>> rows = [(lob, 1988 + w, 12 * (d + 1), k * v, k * premium[w])
        ...         for lob, k in [("Auto", 1.0), ("Home", 2.0)]
        ...         for w, row in enumerate(paid) for d, v in enumerate(row)]
        >>> lob, origin, age, values, prem = map(list, zip(*rows))
        >>> tri = Triangle.from_long(
        ...     origin, age, {"paid": values, "premium": prem}, keys={"lob": lob})
        >>> with warnings.catch_warnings():
        ...     warnings.simplefilter("ignore")  # short chains may miss a threshold
        ...     fits = Meyers(tune=200, draws=200, n_sims=1000).fit_segments(
        ...         tri, "paid", "premium")
        >>> fits.index, fits.reserves.dims
        (['Auto', 'Home'], ['lob', 'origin'])
        >>> fits.segment(lob="Home").premium[:2]
        [11624.0, 9816.0]
        """
    @property
    def gamma_sd(self, /) -> float:
        """
        Standard deviation of the prior on ``gamma`` (``"csr"`` only).
        """
    @property
    def logelr_mean(self, /) -> float:
        """
        Mean of the prior on ``logelr``.
        """
    @property
    def logelr_sd(self, /) -> float:
        """
        Standard deviation of the prior on ``logelr``.
        """
    @property
    def max_depth(self, /) -> int:
        """
        Largest NUTS tree depth.
        """
    @property
    def model(self, /) -> str:
        """
        The model: ``"crc"``, ``"csr"`` or ``"cay"``.
        """
    @property
    def n_sims(self, /) -> int:
        """
        Number of predictive simulations.
        """
    @property
    def origin_sd(self, /) -> float:
        """
        Standard deviation of each origin effect's prior.
        """
    @property
    def rho_shape(self, /) -> float:
        """
        Both shapes of the beta prior on ``(1 - rho) / 2`` (``"cay"`` only).
        """
    @property
    def seed(self, /) -> int:
        """
        Seed of the chains and of the simulations.
        """
    @property
    def spearman(self, /) -> list[list[float]] |None:
        """
        The Spearman matrix of ``dependence="rank_correlation"``; ``None``
        otherwise.
        """
    @property
    def target_accept(self, /) -> float:
        """
        Target mean acceptance rate.
        """
    @property
    def tune(self, /) -> int:
        """
        Warm-up draws per chain.
        """

@final
class MeyersFit:
    """
    A fitted ``Meyers``: the posterior of ``logelr``, ``alpha[2..n]``,
    ``beta[1..m-1]``, ``gamma`` (``"csr"`` only) or ``rho`` (``"cay"``
    only), and ``sigma[1..m]``, its
    MCMC diagnostics, and the posterior predictive distributions of the
    ultimates and reserves by origin. The frames' ``latest``, ``ultimate``
    and ``reserve`` are the volume-weighted chain ladder's
    (``chain_ladder``); ``mean`` and ``std_dev`` are the posterior
    predictive reserve's, and ``ultimate_mean`` and ``ultimate_std_dev``
    the posterior predictive ultimate's.
    """
    def __repr__(self, /) -> str: ...
    @property
    def chain_ladder(self, /) -> ChainLadderFit:
        """
        The volume-weighted chain ladder of the fitted column, whose latest
        values the reserves are measured from.
        """
    @property
    def chains(self, /) -> int:
        """
        Number of chains.
        """
    @property
    def development(self, /) -> list[int]:
        """
        Development ages in months.
        """
    def development_frame(self, /) -> Any:
        """
        The chain ladder's development factors, one row per age, as
        ``ChainLadderFit.development_frame``. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def diagnostics(self, /) -> dict:
        """
        MCMC diagnostics: a dict with ``max_rhat``, ``min_ess_bulk``,
        ``min_ess_tail``, ``divergences`` and ``warnings`` (one message per
        failed threshold; empty when every threshold holds).
        """
    @property
    def draws(self, /) -> list[list[float]]:
        """
        Posterior draws, one row per draw (chain by chain), one column per
        parameter in the order of ``parameter_names``.
        """
    def loo(self, /) -> Elpd:
        """
        PSIS-LOO of the observed cells, with each cell's relative
        efficiency from the chains, as ``BayesGlmFit.loo``. The pointwise
        term is Meyers', the normal density of ``log C[w, d]``, so the
        result is on the log scale and comparable only between ``"crc"``,
        ``"csr"`` and ``"cay"`` fits of the same column (Meyers 2019,
        Tables 6.4, 7.3 and 8.3). Under ``"cay"`` each cell's density is
        conditional on the previous origin's cell, as in Meyers' code.
        
        Returns
        -------
        Elpd
        """
    @property
    def model(self, /) -> str:
        """
        The model: ``"crc"``, ``"csr"`` or ``"cay"``.
        """
    @property
    def origins(self, /) -> list[str]:
        """
        Origin period of each component.
        """
    @property
    def parameter_names(self, /) -> list[str]:
        """
        Parameter names: ``logelr``, ``alpha[2]`` to ``alpha[n]``,
        ``beta[1]`` to ``beta[m-1]``, ``gamma`` (``"csr"`` only) or ``rho``
        (``"cay"`` only), and ``sigma[1]`` to ``sigma[m]``.
        """
    @property
    def premium(self, /) -> list[float]:
        """
        The premium of each origin.
        """
    @property
    def reserves(self, /) -> PredictiveDistribution:
        """
        The ultimates less the latest values, simulation by simulation,
        laid out as ``ultimates`` and as ``OdpBootstrapFit.reserves``.
        """
    def summary(self, /) -> Any:
        """
        Posterior summary, one row per parameter: ``parameter``, ``mean``,
        ``sd``, ``q05``, ``q50``, ``q95``, ``rhat``, ``ess_bulk`` and
        ``ess_tail``, the fields of ``BayesGlmFit.summary``. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    def to_frame(self, /) -> Any:
        """
        One row per origin: ``origin``, the chain ladder's ``latest``,
        ``ultimate`` and ``reserve``, the ``premium``, the ``mean`` and
        ``std_dev`` of the posterior predictive reserve, and the
        ``ultimate_mean`` and ``ultimate_std_dev`` of the posterior
        predictive ultimate. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    def totals_frame(self, /) -> Any:
        """
        One row: the chain ladder's totals, the total ``premium``, the
        ``mean`` and ``std_dev`` of the posterior predictive total reserve,
        and the ``ultimate_mean`` and ``ultimate_std_dev`` of the total
        ultimate. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def ultimates(self, /) -> PredictiveDistribution:
        """
        Joint posterior predictive distribution of the ultimate (the value
        at the oldest age) by origin: dimension ``"origin"``, one component
        per origin, one row per simulation. An origin observed at the
        oldest age keeps its value.
        """

@final
class MeyersFits:
    """
    ``Meyers.fit_segments``: each segment's own posterior and the joint
    posterior predictive distributions of the ultimates and the reserves,
    laid out as ``BayesOdpFits``.
    
    ``ultimates`` and ``reserves`` have the triangle's keys and ``"origin"``
    as dimensions, so ``reserves.aggregate(["lob"])`` keeps the dependence
    between segments. Per-origin lists (``origins``, ``premium``) run over
    the origins of each segment in turn, like the rows of ``to_frame()``
    and the components of ``reserves``. The posterior of one segment
    (``diagnostics``, ``parameter_names``, ``draws``, ``chains``,
    ``summary()`` and ``loo()``) needs a single-segment fit: for several
    segments use ``segment(...)`` or ``diagnostics_frame()``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def chain_ladder(self, /) -> ChainLadderFit:
        """
        The volume-weighted chain ladder of every segment's fitted column,
        whose latest values the reserves are measured from.
        """
    @property
    def chains(self, /) -> int:
        """
        Number of chains of a single-segment fit.
        """
    @property
    def development(self, /) -> list[int]:
        """
        Development ages in months of the first segment.
        """
    def development_frame(self, /) -> Any:
        """
        The chain ladders' development factors, one row per segment and
        age, as ``ChainLadderFit.development_frame``. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def diagnostics(self, /) -> dict:
        """
        MCMC diagnostics of a single-segment fit, as
        ``MeyersFit.diagnostics``.
        """
    def diagnostics_frame(self, /) -> Any:
        """
        One row per segment: the key columns, then ``max_rhat``,
        ``min_ess_bulk``, ``min_ess_tail``, ``divergences`` and
        ``warnings`` (the segment's messages joined by ``"; "``, empty when
        every threshold holds). Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def draws(self, /) -> list[list[float]]:
        """
        Posterior draws of a single-segment fit, as ``MeyersFit.draws``.
        """
    @property
    def index(self, /) -> list[Any]:
        """
        Label of each segment, as ``Triangle.index``.
        """
    @property
    def keys(self, /) -> list[str]:
        """
        Names of the triangle's key columns; empty without keys.
        """
    def loo(self, /) -> Elpd:
        """
        PSIS-LOO of a single-segment fit's observed cells on Meyers' log
        scale, as ``MeyersFit.loo``.
        
        Returns
        -------
        Elpd
        
        Raises
        ------
        ValueError
            If the fit has several segments.
        """
    @property
    def model(self, /) -> str:
        """
        The model: ``"crc"``, ``"csr"`` or ``"cay"``.
        """
    @property
    def origins(self, /) -> list[str]:
        """
        Origin period of each per-origin value and component.
        """
    @property
    def parameter_names(self, /) -> list[str]:
        """
        Parameter names of a single-segment fit, as
        ``MeyersFit.parameter_names``.
        """
    @property
    def premium(self, /) -> list[float]:
        """
        The premium of each segment's origins, in the order of ``origins``.
        """
    @property
    def reserves(self, /) -> PredictiveDistribution:
        """
        The ultimates less the latest values, simulation by simulation,
        laid out as ``ultimates``.
        """
    def segment(self, /, **keys) -> MeyersFits:
        """
        The fit of one segment, chosen by key values as
        ``ChainLadderFit.segment``, with its part of the joint ultimates and
        reserves (same dimensions).
        
        Returns
        -------
        MeyersFits
        """
    def summary(self, /) -> Any:
        """
        Posterior summary of a single-segment fit, as
        ``MeyersFit.summary``. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    def to_frame(self, /) -> Any:
        """
        One row per segment and origin: the key columns, ``origin``, the
        chain ladder's ``latest``, ``ultimate`` and ``reserve``, the
        ``premium``, the ``mean`` and ``std_dev`` of the posterior
        predictive reserve, and the ``ultimate_mean`` and
        ``ultimate_std_dev`` of the posterior predictive ultimate. Needs
        pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    def totals_frame(self, /) -> Any:
        """
        One row per segment: the key columns, the chain ladder's totals,
        the total ``premium``, the ``mean`` and ``std_dev`` of the segment's
        posterior predictive total reserve, and the ``ultimate_mean`` and
        ``ultimate_std_dev`` of its total ultimate. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def ultimates(self, /) -> PredictiveDistribution:
        """
        Joint posterior predictive distribution of the ultimate (the value
        at the oldest age) by segment and origin: the triangle's keys and
        ``"origin"`` are its dimensions, one component per segment and
        origin, one row per simulation.
        """

@final
class Mixture:
    """
    A finite mixture of severities: component ``i`` with probability
    ``w_i``, such as attritional plus large losses.
    
    Parameters
    ----------
    components : list of (float, severity)
        Weights (positive, summing to 1) and severities.
    
    Examples
    --------
    >>> from prospicio.distributions import Lognormal, Mixture, Pareto
    >>> m = Mixture([(0.9, Lognormal.from_mean_cv(1e4, 1.0)), (0.1, Pareto(1e5, 2.0))])
    >>> round(m.mean(), 6)
    29000.0
    """
    def __new__(cls, /, components: Sequence[tuple[float, Any]]) -> Mixture: ...
    def __repr__(self, /) -> str: ...
    def cdf(self, /, x: float) -> float:
        """
        Distribution function ``P(X <= x)``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    def layer(self, /, limit: float, attachment: float) -> float:
        """
        Expected loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
            ``inf`` for an unlimited layer.
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_second_moment(self, /, limit: float, attachment: float) -> float:
        """
        Second moment of the loss to the layer ``limit`` xs
        ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_variance(self, /, limit: float, attachment: float) -> float:
        """
        Variance of the loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def lev(self, /, limit: float) -> float:
        """
        Limited expected value ``E[min(X, limit)]``.
        
        Parameters
        ----------
        limit : float
        
        Returns
        -------
        float
        """
    def mean(self, /) -> float:
        """
        Mean of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """
    def quantile(self, /, p: float) -> float:
        """
        Quantile: the smallest ``x`` with ``P(X <= x) >= p``.
        
        Parameters
        ----------
        p : float
            Probability in ``[0, 1]``.
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def sample(self, /, n: int, seed: int, stream: int = 0) -> list[float]:
        """
        ``n`` draws from stream ``stream`` of the generator keyed by
        ``seed``.
        
        Parameters
        ----------
        n : int
        seed : int
        stream : int, default 0
        
        Returns
        -------
        list of float
        """
    @staticmethod
    def splice(components: Sequence[tuple[float, Any]], breaks: Sequence[float]) -> Mixture:
        """
        A splice: component ``i`` conditioned on
        ``(breaks[i], breaks[i + 1]]`` with weight ``w_i``, such as a body
        below a threshold and a Pareto tail above it.
        
        Parameters
        ----------
        components : list of (float, severity)
            Weights (positive, summing to 1) and severities.
        breaks : list of float
            One more than the components: 0, the joins, and the top (which
            may be ``inf``).
        
        Returns
        -------
        Mixture
        
        Raises
        ------
        ValueError
            If the breaks do not fit the components or a piece's window has
            no probability.
        
        Examples
        --------
        >>> from prospicio.distributions import Lognormal, Mixture, Pareto
        >>> body, tail = Lognormal.from_mean_cv(50.0, 1.0), Pareto(100.0, 1.8)
        >>> s = Mixture.splice([(0.9, body), (0.1, tail)], [0.0, 100.0, float("inf")])
        >>> round(s.cdf(100.0), 12)
        0.9
        """
    def std(self, /) -> float:
        """
        Standard deviation of the distribution.
        
        Returns
        -------
        float
        """
    def stop_loss(self, /, retention: float) -> float:
        """
        Expected excess over a retention, ``E[max(X - retention, 0)]``.
        
        Parameters
        ----------
        retention : float
        
        Returns
        -------
        float
        """
    def survival(self, /, x: float) -> float:
        """
        Survival function ``P(X > x)``, accurate far into the tail.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    def variance(self, /) -> float:
        """
        Variance of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """
    @property
    def weights(self, /) -> list[float]:
        """
        Component weights.
        """

@final
class NaturalPrice:
    """
    A portfolio's price and its natural allocation, from ``Portfolio.price``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def allocated(self, /) -> list[Pentagon]:
        """
        Each unit's share, a list of Pentagon; the amounts add up to ``total``.
        """
    def to_dict(self, /) -> dict:
        """
        The table as a dict of columns: ``unit`` (the units, then
        ``"total"``), ``loss``, ``margin``, ``premium``, ``capital``,
        ``assets``, ``loss_ratio``, ``premium_to_capital`` and
        ``return_on_capital``. ``pandas.DataFrame(price.to_dict())`` makes
        a frame of it.
        
        Returns
        -------
        dict of str to list
        """
    @property
    def total(self, /) -> Pentagon:
        """
        The portfolio, a Pentagon.
        """
    @property
    def units(self, /) -> list[str]:
        """
        Unit names.
        """

@final
class NegativeBinomial:
    """
    Negative binomial claim counts: mean ``r * beta``, variance
    ``r * beta * (1 + beta)`` (Klugman, Panjer & Willmot).
    
    SciPy's ``nbinom(n=r, p=1/(1+beta))`` is the same distribution.
    
    Parameters
    ----------
    r : float
        Shape; must be positive.
    beta : float
        Scale; must be positive.
    
    Raises
    ------
    ValueError
        If ``r`` or ``beta`` is not positive and finite.
    
    Examples
    --------
    >>> from prospicio.distributions import NegativeBinomial
    >>> n = NegativeBinomial.from_mean_variance(10.0, 30.0)
    >>> round(n.variance(), 9)
    30.0
    """
    def __getnewargs__(self, /) -> tuple[float, float]: ...
    def __new__(cls, /, r: float, beta: float) -> NegativeBinomial: ...
    def __repr__(self, /) -> str: ...
    @property
    def beta(self, /) -> float:
        """
        Scale ``beta``.
        """
    def cdf(self, /, k: int) -> float:
        """
        ``P(N <= k)``.
        
        Parameters
        ----------
        k : int
        
        Returns
        -------
        float
        """
    @staticmethod
    def from_mean_variance(mean: float, variance: float) -> NegativeBinomial:
        """
        The negative binomial with this mean and variance.
        
        Parameters
        ----------
        mean : float
            Must be positive.
        variance : float
            Must exceed the mean.
        
        Returns
        -------
        NegativeBinomial
        
        Raises
        ------
        ValueError
            If the mean is not positive or the variance does not exceed it.
        """
    def mean(self, /) -> float:
        """
        Mean of the claim count.
        
        Returns
        -------
        float
        """
    def pmf(self, /, k: int) -> float:
        """
        ``P(N = k)``.
        
        Parameters
        ----------
        k : int
        
        Returns
        -------
        float
        """
    def quantile(self, /, p: float) -> int:
        """
        Smallest ``k`` with ``P(N <= k) >= p``.
        
        Parameters
        ----------
        p : float
            Probability in ``[0, 1)``.
        
        Returns
        -------
        int
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    @property
    def r(self, /) -> float:
        """
        Shape ``r``.
        """
    def sample(self, /, n: int, seed: int, stream: int = 0) -> list[int]:
        """
        ``n`` claim counts from stream ``stream`` of the generator keyed by
        ``seed``.
        
        Parameters
        ----------
        n : int
        seed : int
        stream : int, default 0
        
        Returns
        -------
        list of int
        """
    def variance(self, /) -> float:
        """
        Variance of the claim count.
        
        Returns
        -------
        float
        """

@final
class OccurrencePml:
    """
    One point of the occurrence PML curve, from ``occurrence_pml``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def annual_return_period(self, /) -> float:
        """
        ``1 / probability``, the average time between years with such an
        occurrence.
        """
    @property
    def loss(self, /) -> float:
        """
        The loss whose occurrence exceedance is the probability.
        """
    @property
    def occurrence_return_period(self, /) -> float:
        """
        ``1 / (E[N] P(X > loss))``, the average time between occurrences
        above the loss; it can be shorter than a year.
        """
    @property
    def probability(self, /) -> float:
        """
        The annual probability asked for.
        """
    @property
    def severity_survival(self, /) -> float:
        """
        The severity's exceedance ``P(X > loss)``.
        """

@final
class OdpBootstrap:
    """
    Over-dispersed Poisson bootstrap of the chain ladder (England and
    Verrall 2002), as R ChainLadder's ``BootChainLadder``: adjusted Pearson
    residuals of the volume-weighted chain ladder are resampled into pseudo
    triangles, each is re-projected, and process error is added to every
    future incremental value. Simulation ``i`` uses random stream ``i`` of
    ``seed`` for every segment in turn, so results do not depend on the
    number of threads.
    
    Parameters
    ----------
    n_sims : int, default 10000
        Number of simulations; positive.
    seed : int, default 0
        Seed of the simulation streams, from 0 to ``2**64 - 1``. R accepts
        seeds below ``2**53``; a seed in both ranges gives the same draws.
    process : {"gamma", "none"}, default "gamma"
        Process error on each simulated future incremental value: Gamma with
        the expected value as mean and variance ``scale * |mean|`` (R's
        ``process.distr = "gamma"``), or none for parameter error only.
    tail : float, TailConstant, TailCurve, TailBondy or TailLogLinear, optional
        Development past the oldest age in the lifetime view (``fit``), as
        ``ChainLadder``'s; no tail by default, as R's ``BootChainLadder``.
        An estimated tail (``TailCurve``, ``TailBondy``, ``TailLogLinear``)
        is refitted on each simulation's pseudo factors, which gives its
        parameter error; a constant one is fixed unless ``tail_std_err`` is
        given. The step from the oldest age to ultimate is one more future
        increment, with mean the pseudo value at the oldest age times the
        tail factor less 1 and the same process error. ``one_year`` values
        that step one development period after the oldest age and takes it
        for the origins whose step falls in the coming year, closing them;
        ``method``'s tail must then be this one. R's rule
        (``TailLogLinear``) refitted on the ODP's pseudo factors is often
        exactly 1 (5.7% of simulations on RAA) and on average high (RAA's
        oldest origin's mean reserve 78% above the plug-in); a constant tail
        with ``tail_std_err`` has neither.
    tail_std_err : float, optional
        Standard error of a constant tail factor: each simulation draws the
        factor from the lognormal with the factor as mean and this standard
        deviation. Not given, the factor is fixed, unlike
        ``MackBootstrap``, which extrapolates Mack's standard error. Unused
        when the factor is 1; an error with an estimated tail.
    dependence : {"independent", "synchronized", "rank_correlation"}, default "independent"
        How the segments of a multi-segment triangle depend on each other
        in ``fit`` and ``one_year``: each resamples its own residuals
        independently; every segment resamples the residuals of the same
        origins and ages in each simulation (the synchronous bootstrap of
        Taylor and McGuire 2007 and Kirschner, Kerley and Isaacs 2008), so
        the lines' parameter error takes the correlation of their residuals
        and process error stays independent, which needs the same origins,
        ages and observed cells in every segment; or each is bootstrapped
        independently and the segments' simulations are reordered as whole
        rows (Iman–Conover on the segment totals) to the Spearman matrix
        ``spearman`` and then put in a random order, so that row ``i`` is
        no longer simulation ``i`` but any subset of rows is a fair sample.
    spearman : list of list of float, optional
        With ``dependence="rank_correlation"`` only: Spearman's rho between
        the segments' totals, one row and column per segment in index
        order, symmetric with a unit diagonal. It is converted to the normal
        scores' correlation ``2 sin(pi rho / 6)``, which must be positive
        definite.
    
    Raises
    ------
    ValueError
        If ``n_sims`` is zero, ``process`` or ``dependence`` is unknown, or
        ``spearman`` is missing for ``"rank_correlation"``, given for another
        dependence or not square.
    TypeError
        If ``tail`` is not a number or a tail estimator.
    OverflowError
        If ``n_sims`` or ``seed`` is negative or too large.
    
    Examples
    --------
    >>> from prospicio.reserving import OdpBootstrap, Triangle
    >>> tri = Triangle.from_long(
    ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
    ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
    ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
    ... )
    >>> fit = OdpBootstrap(n_sims=2000, seed=42).fit(tri, "values")
    >>> fit.reserves.components()
    [('2020',), ('2021',), ('2022',), ('2023',)]
    >>> fit.reserves.mean() > 0
    True
    """
    def __new__(cls, /, n_sims: int = 10000, seed: int = 0, process: str = "gamma", tail: Any |None = None, tail_std_err: float |None = None, dependence: str = "independent", spearman: Sequence[Sequence[float]] |None = None) -> OdpBootstrap: ...
    def __repr__(self, /) -> str: ...
    @property
    def dependence(self, /) -> str:
        """
        How the segments depend on each other: ``"independent"``,
        ``"synchronized"`` or ``"rank_correlation"``.
        """
    def fit(self, /, triangle: Triangle, column: str) -> OdpBootstrapFit:
        """
        Bootstraps one measure column in every segment of a cumulative
        triangle, each with its own residuals and scale, into one joint
        distribution of the reserves, the segments depending on each other
        as ``dependence`` says. Every origin must be observed from the first
        age up to its latest.
        
        Parameters
        ----------
        triangle : Triangle
            Cumulative, with any number of segments.
        column : str
        
        Returns
        -------
        OdpBootstrapFit
        
        Raises
        ------
        ValueError
            As ``ChainLadder.fit``, and if an origin has a gap before its
            latest age, a segment has too few observed cells for the
            degrees of freedom to be positive, synchronized segments differ
            in their origins, ages or observed cells, or ``spearman`` does
            not fit the segments; with an estimated ``tail``, if it cannot be
            refitted on some simulation's pseudo factors (the message counts
            them and gives one).
        """
    @property
    def n_sims(self, /) -> int:
        """
        Number of simulations.
        """
    def one_year(self, /, triangle: Triangle, column: str, method: Any, exposure: str |None = None) -> OneYearFit:
        """
        The one-year view of any reserving method: the claims development
        result over the coming year, by re-reserving on the bootstrap
        ("actuary in the box"). The coming year is every cell valued in the
        twelve months after the segment's valuation: one per origin for an
        annual development grain, four for a quarterly one (fewer for an
        origin that reaches the last age). Each simulation resamples the
        residuals for the volume-weighted factors, projects the increments
        of those cells in turn from the origin's resampled latest value with
        the bootstrap's process error, as ``fit`` projects, adds them to the
        observed latest value, appends the cells to the triangle, refits
        ``method`` and records ``CDR = opening ultimate - closing
        ultimate``; a negative CDR is an adverse development. An origin whose
        remaining cells all fall in the year thus has its lifetime bootstrap
        reserve as its one-year view; an origin at the last age gets no new
        cell; an origin short of the latest diagonal develops from its own
        latest cell, and only the year's cells are appended. Unlike
        ``MackFit.claims_development_result()`` (Merz and Wüthrich), any
        averaging, tail and development grain are allowed.
        
        Without a bootstrap ``tail``, development past the oldest age moves
        only through ``method``'s refitted tail. With one, which must be
        ``method``'s tail, the tail is drawn in each simulation as ``fit``
        draws it, and its step to ultimate is valued one development period
        after the oldest age: an origin whose step falls in the coming year
        (with an annual grain, one already at the oldest age) takes it and
        is closed at that simulated ultimate. The step is the whole tail: a
        curve or a decayed constant tail is taken as the product of its
        factors in that one step, so an origin that takes it has its
        lifetime tail risk in the coming year, an upper bound on its
        one-year risk.
        
        Parameters
        ----------
        triangle : Triangle
            Cumulative, with any number of segments and any development
            grain.
        column : str
        method : ChainLadder, ExpectedLoss, BornhuetterFerguson, Benktander or CapeCod
            The method refitted at the start and at the end of the year.
        exposure : str, optional
            The exposure column; required by the expected-loss methods, not
            taken by ``ChainLadder``. Its latest value per origin is kept for
            the end of the year.
        
        Returns
        -------
        OneYearFit
        
        Raises
        ------
        TypeError
            If ``method`` is not one of the classes above.
        ValueError
            As ``fit`` and the method's own ``fit``; if ``exposure`` is
            missing for an expected-loss method or given for
            ``ChainLadder``; if the bootstrap has a ``tail`` that is not
            ``method``'s; or if the refit (or the bootstrap's tail refit)
            fails in any simulation (the message counts them and gives one).
        
        Examples
        --------
        >>> from prospicio.reserving import BornhuetterFerguson, OdpBootstrap, Triangle
        >>> tri = Triangle.from_long(
        ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
        ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
        ...     {
        ...         "paid": [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
        ...         "premium": [250.0] * 4 + [260.0] * 3 + [270.0] * 2 + [280.0],
        ...     },
        ... )
        >>> boot = OdpBootstrap(n_sims=2000, seed=42)
        >>> fit = boot.one_year(tri, "paid", BornhuetterFerguson(apriori=0.7), exposure="premium")
        >>> fit.cdr.components()
        [('2020',), ('2021',), ('2022',), ('2023',)]
        >>> fit.opening_reserve[0]
        0.0
        >>> fit.cdr.variance() < boot.fit(tri, "paid").reserves.variance()
        True
        """
    @property
    def process(self, /) -> str:
        """
        Process error: ``"gamma"`` or ``"none"``.
        """
    @property
    def seed(self, /) -> int:
        """
        Seed of the simulation streams.
        """
    @property
    def spearman(self, /) -> list[list[float]] |None:
        """
        The Spearman matrix of ``dependence="rank_correlation"``; ``None``
        otherwise.
        """
    @property
    def tail(self, /) -> Any:
        """
        The lifetime view's tail: a constant factor as a number, otherwise
        its estimator.
        """
    @property
    def tail_std_err(self, /) -> float |None:
        """
        The standard error a constant tail factor is drawn with, or
        ``None`` to keep it fixed.
        """

@final
class OdpBootstrapFit:
    """
    A fitted ODP bootstrap of every segment.
    
    ``reserves`` is one joint distribution with the triangle's keys and
    ``"origin"`` as dimensions, so ``reserves.aggregate(["lob"])`` keeps the
    dependence between segments. Per-origin lists run over the origins of
    each segment in turn, like the rows of ``to_frame()`` and the
    components of ``reserves``. ``fitted``, ``residuals`` and ``scale``
    need a single-segment fit; for several segments use ``segment(...)`` or
    ``totals_frame()``. ``fitted`` and ``residuals`` are nested lists
    indexed ``[origin][development]``, like one segment of
    ``Triangle.values``, with ``nan`` where the triangle is not observed.
    """
    def __repr__(self, /) -> str: ...
    @property
    def chain_ladder(self, /) -> ChainLadderFit:
        """
        The deterministic volume-weighted chain ladder the bootstrap is
        centred on, with the bootstrap's tail.
        """
    @property
    def development(self, /) -> list[int]:
        """
        Development ages in months.
        """
    def development_frame(self, /) -> Any:
        """
        The chain ladders' development factors, one row per segment and
        age, as ``ChainLadderFit.development_frame``. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def fitted(self, /) -> list[list[float]]:
        """
        Fitted incremental values, ``[origin][development]``.
        """
    @property
    def index(self, /) -> list[Any]:
        """
        Label of each segment, as ``Triangle.index``.
        """
    @property
    def keys(self, /) -> list[str]:
        """
        Names of the triangle's key columns; empty without keys.
        """
    @property
    def origins(self, /) -> list[str]:
        """
        Origin period of each per-origin value and reserve component.
        """
    @property
    def reserves(self, /) -> PredictiveDistribution:
        """
        Joint distribution of the reserve (the sum of future incremental
        values) by segment and origin: the triangle's keys and ``"origin"``
        are its dimensions, one component per segment and origin, one row
        per simulation. Its ``mean`` and ``quantile`` describe the total
        reserve. Columns of ``draw_matrix()`` follow ``origins``.
        """
    @property
    def residuals(self, /) -> list[list[float]]:
        """
        Adjusted Pearson residuals ``(x - m) / sqrt(|m|) * sqrt(n / (n - p))``,
        ``[origin][development]``; ``nan`` where not observed or where the
        fitted value is zero.
        """
    @property
    def scale(self, /) -> float:
        """
        The scale parameter ``phi``: the sum of squared unadjusted residuals
        over the degrees of freedom ``n - p``.
        """
    def segment(self, /, **keys) -> OdpBootstrapFit:
        """
        The bootstrap of one segment, chosen by key values as
        ``ChainLadderFit.segment``, with its part of the joint reserves
        (same dimensions).
        
        Returns
        -------
        OdpBootstrapFit
        """
    def to_frame(self, /) -> Any:
        """
        One row per segment and origin: the key columns, ``origin``, the
        chain ladder's ``latest``, ``ultimate`` and ``reserve``, and the
        ``mean`` and ``std_dev`` of the bootstrapped reserve. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    def totals_frame(self, /) -> Any:
        """
        One row per segment: the key columns, the chain ladder's totals, the
        ``scale``, and the ``mean`` and ``std_dev`` of the segment's
        bootstrapped total reserve. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """

@final
class OneYearFit:
    """
    The simulated one-year view of every segment, from
    ``OdpBootstrap.one_year`` or ``MackBootstrap.one_year`` (``model``
    says which).
    
    ``cdr`` is one joint distribution of the claims development result
    with the triangle's keys and ``"origin"`` as dimensions, so
    ``cdr.aggregate(["lob"])`` keeps the dependence between segments, and
    ``cdr.quantile(0.005)`` is minus the one-year value at risk at 99.5%.
    Per-origin lists run over the origins of each segment in turn, like the
    rows of ``to_frame()`` and the components of ``cdr``. ``fitted``,
    ``residuals`` and ``scale`` need a single-segment fit; for several
    segments use ``segment(...)`` or ``totals_frame()``. ``fitted`` and
    ``scale`` are the ODP bootstrap's (``OdpBootstrapFit``'s), ``mack`` is
    Mack's bootstrap's model (a ``MackFit`` of every segment, as
    ``chain_ladder``), and ``residuals`` are either's.
    """
    def __repr__(self, /) -> str: ...
    @property
    def cdr(self, /) -> PredictiveDistribution:
        """
        Joint distribution of the claims development result (opening less
        closing ultimate) by segment and origin: the triangle's keys and
        ``"origin"`` are its dimensions, one component per segment and
        origin, one row per simulation. Its ``mean``, ``variance`` and
        ``quantile`` describe the total. Columns of ``draw_matrix()`` follow
        ``origins``.
        """
    @property
    def chain_ladder(self, /) -> ChainLadderFit:
        """
        The bootstrap's chain ladder: the factors the simulated next cells
        develop with (volume-weighted for the ODP, Mack's averaging for
        Mack's).
        """
    @property
    def development(self, /) -> list[int]:
        """
        Development ages in months.
        """
    def development_frame(self, /) -> Any:
        """
        The bootstrap's chain ladders' development factors, one row per
        segment and age, as ``ChainLadderFit.development_frame``. Needs
        pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    @property
    def fitted(self, /) -> list[list[float]]:
        """
        The ODP bootstrap's fitted incremental values,
        ``[origin][development]``.
        """
    @property
    def index(self, /) -> list[Any]:
        """
        Label of each segment, as ``Triangle.index``.
        """
    @property
    def keys(self, /) -> list[str]:
        """
        Names of the triangle's key columns; empty without keys.
        """
    @property
    def latest(self, /) -> list[float]:
        """
        Latest observed value per origin.
        """
    @property
    def mack(self, /) -> MackFit:
        """
        Mack's model behind ``MackBootstrap.one_year``, with its lifetime
        standard errors and ``claims_development_result()``.
        """
    @property
    def model(self, /) -> str:
        """
        The bootstrap's model: ``"odp"`` or ``"mack"``.
        """
    @property
    def opening_reserve(self, /) -> list[float]:
        """
        The opening ultimate less the latest value, per origin.
        """
    @property
    def opening_ultimate(self, /) -> list[float]:
        """
        The method's ultimate per origin on the observed triangle.
        """
    @property
    def origins(self, /) -> list[str]:
        """
        Origin period of each per-origin value and CDR component.
        """
    @property
    def residuals(self, /) -> list[list[float]]:
        """
        The residuals the bootstrap resamples, ``[origin][development]``:
        the ODP's adjusted Pearson residuals, as
        ``OdpBootstrapFit.residuals``, or Mack's scaled bias-adjusted
        residuals of the link ratios, ``[o][k]`` the link from age ``k`` to
        ``k + 1`` (``nan`` where there is none, from a zero, or behind a
        factor with a single link ratio).
        """
    @property
    def scale(self, /) -> float:
        """
        The ODP bootstrap's scale parameter ``phi``.
        """
    def segment(self, /, **keys) -> OneYearFit:
        """
        The one-year view of one segment, chosen by key values as
        ``ChainLadderFit.segment``, with its part of the joint claims
        development result (same dimensions).
        
        Returns
        -------
        OneYearFit
        """
    def to_frame(self, /) -> Any:
        """
        One row per segment and origin: the key columns, ``origin``,
        ``latest``, the method's ``opening_ultimate`` and
        ``opening_reserve``, and the ``cdr_mean`` and ``cdr_std_dev`` of the
        simulated claims development result. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    def totals_frame(self, /) -> Any:
        """
        One row per segment: the key columns, the totals of ``to_frame()``'s
        columns, the ODP bootstrap's ``scale`` (not for Mack's), and the
        ``cdr_mean`` and ``cdr_std_dev`` of the segment's total claims
        development result. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """

@final
class Pareto:
    """
    Single-parameter Pareto: ``P(X > x) = (t / x) ** alpha`` for ``x >= t``,
    optionally truncated (conditioned on ``X < truncation``).
    
    Parameters
    ----------
    t : float
        Threshold; finite and positive.
    alpha : float
        Pareto alpha; finite and positive.
    truncation : float, optional
        Truncation point above ``t``.
    
    Raises
    ------
    ValueError
        If a parameter is out of range.
    
    Examples
    --------
    >>> from prospicio.distributions import Pareto
    >>> p = Pareto(500.0, 2.0)
    >>> round(p.layer(4000.0, 1000.0), 9)
    200.0
    """
    def __getnewargs__(self, /) -> tuple[float, float, float |None]: ...
    def __new__(cls, /, t: float, alpha: float, truncation: float |None = None) -> Pareto: ...
    def __repr__(self, /) -> str: ...
    @property
    def alpha(self, /) -> float:
        """
        Pareto alpha.
        """
    def cdf(self, /, x: float) -> float:
        """
        Distribution function ``P(X <= x)``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @staticmethod
    def fit(losses: Sequence[float], t: float, reporting_thresholds: Sequence[float] |None = None, censored: Sequence[bool] |None = None, weights: Sequence[float] |None = None, truncation: float |None = None) -> Pareto:
        """
        Maximum likelihood fit of the alpha to large losses at or above
        ``t``.
        
        Parameters
        ----------
        losses : list of float
        t : float
            Threshold of the fitted Pareto.
        reporting_thresholds : list of float, optional
            Per-loss thresholds below which a loss would not have been
            reported; raised to ``t``.
        censored : list of bool, optional
            ``True`` where a loss was capped by a policy limit.
        weights : list of float, optional
        truncation : float, optional
        
        Returns
        -------
        Pareto
        
        Examples
        --------
        >>> from prospicio.distributions import Pareto
        >>> round(Pareto.fit([1500.0, 2500.0, 4000.0], 1000.0).alpha, 6)
        1.10524
        """
    def layer(self, /, limit: float, attachment: float) -> float:
        """
        Expected loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
            ``inf`` for an unlimited layer.
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_second_moment(self, /, limit: float, attachment: float) -> float:
        """
        Second moment of the loss to the layer ``limit`` xs
        ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_variance(self, /, limit: float, attachment: float) -> float:
        """
        Variance of the loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def lev(self, /, limit: float) -> float:
        """
        Limited expected value ``E[min(X, limit)]``.
        
        Parameters
        ----------
        limit : float
        
        Returns
        -------
        float
        """
    def mean(self, /) -> float:
        """
        Mean of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """
    def quantile(self, /, p: float) -> float:
        """
        Quantile: the smallest ``x`` with ``P(X <= x) >= p``.
        
        Parameters
        ----------
        p : float
            Probability in ``[0, 1]``.
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def sample(self, /, n: int, seed: int, stream: int = 0) -> list[float]:
        """
        ``n`` draws from stream ``stream`` of the generator keyed by
        ``seed``.
        
        Parameters
        ----------
        n : int
        seed : int
        stream : int, default 0
        
        Returns
        -------
        list of float
        """
    def std(self, /) -> float:
        """
        Standard deviation of the distribution.
        
        Returns
        -------
        float
        """
    def stop_loss(self, /, retention: float) -> float:
        """
        Expected excess over a retention, ``E[max(X - retention, 0)]``.
        
        Parameters
        ----------
        retention : float
        
        Returns
        -------
        float
        """
    def survival(self, /, x: float) -> float:
        """
        Survival function ``P(X > x)``, accurate far into the tail.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @property
    def t(self, /) -> float:
        """
        Threshold ``t``.
        """
    @property
    def truncation(self, /) -> float |None:
        """
        Truncation point, or ``None``.
        """
    def variance(self, /) -> float:
        """
        Variance of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """

@final
class PaymentPattern:
    """
    The share of the ultimate loss paid at each time, in years from the
    treaty's inception.
    
    With a pattern that does not depend on the size of the loss, every
    simulated year's present value is its loss times ``discount_factor``, so
    a loss distribution is discounted by scaling it.
    
    Parameters
    ----------
    times : list of float
        Non-negative, strictly increasing times, in years from inception.
    shares : list of float
        The share paid at each time; they sum to 1.
    
    Examples
    --------
    >>> from prospicio.pricing import PaymentPattern
    >>> p = PaymentPattern.from_development([1.0, 2.0, 3.0], [4.0, 1.25, 1.0])
    >>> [round(s, 12) for s in p.shares]
    [0.25, 0.55, 0.2]
    >>> round(p.duration, 12), round(p.discount_factor(0.05), 6)
    (1.95, 0.909729)
    """
    def __new__(cls, /, times: Sequence[float], shares: Sequence[float]) -> PaymentPattern: ...
    def __repr__(self, /) -> str: ...
    def capital_years(self, /, rate: float) -> float:
        """
        The discounted capital years per unit of initial capital: capital
        held in each year in proportion to the share outstanding at its
        start, charged at the start of the year.
        
        Parameters
        ----------
        rate : float
        
        Returns
        -------
        float
        """
    def discount_factor(self, /, rate: float) -> float:
        """
        The present value at inception of a unit of ultimate loss, at an
        annual effective ``rate``.
        
        Parameters
        ----------
        rate : float
        
        Returns
        -------
        float
        """
    @property
    def duration(self, /) -> float:
        """
        The mean time to payment, undiscounted.
        """
    @staticmethod
    def from_development(ages: Sequence[float], factors: Sequence[float]) -> PaymentPattern:
        """
        From development factors to ultimate by age, such as a paid chain
        ladder's ``cdf``: the share paid by each age is one over its factor.
        
        The last factor must be 1 (add any tail as a last age). Ages are read
        as times from inception; ``shifted`` moves the pattern when losses
        start developing later.
        
        Parameters
        ----------
        ages : list of float
        factors : list of float
        
        Returns
        -------
        PaymentPattern
        """
    def outstanding(self, /, t: float) -> float:
        """
        The share not yet paid at time ``t``.
        
        Parameters
        ----------
        t : float
        
        Returns
        -------
        float
        """
    @property
    def shares(self, /) -> list[float]:
        """
        The share paid at each time.
        """
    def shifted(self, /, by: float) -> PaymentPattern:
        """
        The pattern ``by`` years later.
        
        Parameters
        ----------
        by : float
        
        Returns
        -------
        PaymentPattern
        """
    @property
    def times(self, /) -> list[float]:
        """
        The times.
        """

@final
class PayoutPrice:
    """
    A price restated over a payout, from ``price_over_payout``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def assets(self, /) -> float:
        """
        Assets, the one-year risk measure.
        """
    @property
    def capital(self, /) -> float:
        """
        Initial capital, assets less premium.
        """
    @property
    def capital_years(self, /) -> float:
        """
        The discounted capital years (1 for a distortion price).
        """
    @property
    def discount_factor(self, /) -> float:
        """
        The losses' discount factor.
        """
    @property
    def discounted_loss(self, /) -> float:
        """
        Expected loss discounted to inception.
        """
    @property
    def expected_loss(self, /) -> float:
        """
        Expected loss, undiscounted.
        """
    @property
    def margin(self, /) -> float:
        """
        Margin over the discounted loss.
        """
    @property
    def premium(self, /) -> float:
        """
        Premium at inception.
        """

@final
class Pentagon:
    """
    Loss, margin, premium, capital and assets, with ``P = L + M`` and
    ``a = P + Q``.
    
    Build one from any three known quantities with ``Pentagon.solve``.
    
    Examples
    --------
    >>> from prospicio.pricing import Pentagon
    >>> p = Pentagon.solve(loss=46.6, assets=100.0, return_on_capital=0.15)
    >>> round(p.premium, 6)
    53.565217
    """
    def __repr__(self, /) -> str: ...
    @property
    def assets(self, /) -> float:
        """
        Assets ``a``.
        """
    @property
    def capital(self, /) -> float:
        """
        Capital ``Q = a - P``.
        """
    @property
    def discount(self, /) -> float:
        """
        Discount ``M / (a - L)``; ``P = (1 - discount) L + discount a``.
        """
    @property
    def loss(self, /) -> float:
        """
        Expected loss paid, ``L``.
        """
    @property
    def loss_ratio(self, /) -> float:
        """
        Loss ratio ``L / P``.
        """
    @property
    def margin(self, /) -> float:
        """
        Margin ``M = P - L``.
        """
    @property
    def premium(self, /) -> float:
        """
        Premium ``P``.
        """
    @property
    def premium_to_capital(self, /) -> float:
        """
        Premium leverage ``P / Q``.
        """
    @property
    def return_on_capital(self, /) -> float:
        """
        Return on capital ``M / Q``.
        """
    @staticmethod
    def solve(*, loss: float |None = None, margin: float |None = None, premium: float |None = None, capital: float |None = None, assets: float |None = None, loss_ratio: float |None = None, premium_to_capital: float |None = None, return_on_capital: float |None = None) -> Pentagon:
        """
        The pentagon fixed by exactly three of its quantities.
        
        Parameters
        ----------
        loss, margin, premium, capital, assets : float, optional
        loss_ratio, premium_to_capital, return_on_capital : float, optional
        
        Returns
        -------
        Pentagon
        
        Raises
        ------
        ValueError
            Unless exactly three are given and they determine the rest.
        """

@final
class Period:
    """
    An experience or forecast period ``[start, end)`` in year fractions:
    premium earned (losses occurring) in a calendar window, or policies
    written in a policy window. A forecast period is the policy period of
    the policies written while new rates are in effect.
    
    Build one with ``Period.calendar`` or ``Period.policy``.
    
    Examples
    --------
    >>> from prospicio.pricing import Period
    >>> cy = Period.calendar(2011.0, 2012.0)
    >>> future = Period.policy(2013.0, 2014.0)
    >>> future.average_written_date(1.0) - cy.average_written_date(1.0)
    2.5
    """
    def __repr__(self, /) -> str: ...
    def average_earned_date(self, /, term: float) -> float:
        """
        The average date the period's premium is earned, the average
        accident date of its losses: losses are trended from and to these
        dates.
        
        Parameters
        ----------
        term : float
        
        Returns
        -------
        float
        """
    def average_written_date(self, /, term: float) -> float:
        """
        The average date the period's premium was written, for policies of
        ``term`` years: premium is trended from and to these dates.
        
        Parameters
        ----------
        term : float
        
        Returns
        -------
        float
        """
    @staticmethod
    def calendar(start: float, end: float) -> Period:
        """
        Premium earned (losses occurring) in ``[start, end)``.
        
        Parameters
        ----------
        start : float
        end : float
        
        Returns
        -------
        Period
        """
    @property
    def kind(self, /) -> str:
        """
        ``"calendar"`` or ``"policy"``.
        """
    @staticmethod
    def policy(start: float, end: float) -> Period:
        """
        Policies written in ``[start, end)``, over their whole term.
        
        Parameters
        ----------
        start : float
        end : float
        
        Returns
        -------
        Period
        """

@final
class Picks:
    """
    A severity reshaped to layer loss picks, from ``layer_loss_picks``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def severity(self, /) -> Grid:
        """
        The adjusted severity, on the same grid.
        """
    @property
    def straight_bottom(self, /) -> bool:
        """
        Whether the bottom layer took the straight-line shape.
        """
    def to_dict(self, /) -> dict:
        """
        Each layer, bottom up, as columns: ``bottom``, ``top``, ``before``
        and ``target`` (the expected loss per claim before and after),
        ``weight`` (on the survival function) and ``above`` (the adjusted
        chance of a loss above the layer).
        """

@final
class PiecewisePareto:
    """
    Piecewise Pareto: alpha ``alpha[k]`` above threshold ``t[k]``, the
    general large-loss model and the result of tower matching.
    
    Parameters
    ----------
    t : list of float
        Strictly increasing positive thresholds.
    alpha : list of float
        One alpha per threshold; interior ones may be 0, the last must be
        positive.
    truncation : float, optional
        Truncation point above the last threshold.
    truncation_type : {"lp", "wd"}, default "lp"
        Truncate the last piece only, or the whole distribution.
    
    Raises
    ------
    ValueError
        If a parameter is out of range.
    
    Examples
    --------
    >>> from prospicio.distributions import PiecewisePareto
    >>> pp = PiecewisePareto([1000.0, 2000.0], [1.0, 2.0])
    >>> round(pp.survival(4000.0), 12)
    0.125
    """
    def __getnewargs__(self, /) -> tuple[list[float], list[float], float |None, str]: ...
    def __new__(cls, /, t: Sequence[float], alpha: Sequence[float], truncation: float |None = None, truncation_type: str = "lp") -> PiecewisePareto: ...
    def __repr__(self, /) -> str: ...
    @property
    def alpha(self, /) -> list[float]:
        """
        Alphas, one per threshold.
        """
    def cdf(self, /, x: float) -> float:
        """
        Distribution function ``P(X <= x)``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @staticmethod
    def fit(losses: Sequence[float], t: Sequence[float], reporting_thresholds: Sequence[float] |None = None, censored: Sequence[bool] |None = None, weights: Sequence[float] |None = None, truncation: float |None = None, truncation_type: str = "lp") -> PiecewisePareto:
        """
        Maximum likelihood fit of the alphas for thresholds ``t`` to large
        losses at or above ``t[0]``.
        
        Parameters
        ----------
        losses : list of float
        t : list of float
            Thresholds of the fitted distribution.
        reporting_thresholds : list of float, optional
        censored : list of bool, optional
        weights : list of float, optional
        truncation : float, optional
        truncation_type : {"lp", "wd"}, default "lp"
            Truncate the last piece only (each alpha a closed form or a
            one-dimensional solve), or the whole distribution (the alphas
            are coupled and solved together).
        
        Returns
        -------
        PiecewisePareto
        """
    def layer(self, /, limit: float, attachment: float) -> float:
        """
        Expected loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
            ``inf`` for an unlimited layer.
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_second_moment(self, /, limit: float, attachment: float) -> float:
        """
        Second moment of the loss to the layer ``limit`` xs
        ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_variance(self, /, limit: float, attachment: float) -> float:
        """
        Variance of the loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def lev(self, /, limit: float) -> float:
        """
        Limited expected value ``E[min(X, limit)]``.
        
        Parameters
        ----------
        limit : float
        
        Returns
        -------
        float
        """
    def mean(self, /) -> float:
        """
        Mean of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """
    def quantile(self, /, p: float) -> float:
        """
        Quantile: the smallest ``x`` with ``P(X <= x) >= p``.
        
        Parameters
        ----------
        p : float
            Probability in ``[0, 1]``.
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def sample(self, /, n: int, seed: int, stream: int = 0) -> list[float]:
        """
        ``n`` draws from stream ``stream`` of the generator keyed by
        ``seed``.
        
        Parameters
        ----------
        n : int
        seed : int
        stream : int, default 0
        
        Returns
        -------
        list of float
        """
    def std(self, /) -> float:
        """
        Standard deviation of the distribution.
        
        Returns
        -------
        float
        """
    def stop_loss(self, /, retention: float) -> float:
        """
        Expected excess over a retention, ``E[max(X - retention, 0)]``.
        
        Parameters
        ----------
        retention : float
        
        Returns
        -------
        float
        """
    def survival(self, /, x: float) -> float:
        """
        Survival function ``P(X > x)``, accurate far into the tail.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @property
    def t(self, /) -> list[float]:
        """
        Thresholds.
        """
    @property
    def truncation(self, /) -> float |None:
        """
        Truncation point, or ``None``.
        """
    @property
    def truncation_type(self, /) -> str |None:
        """
        ``"lp"`` or ``"wd"`` when truncated, else ``None``.
        """
    def variance(self, /) -> float:
        """
        Variance of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """

@final
class Poisson:
    """
    Poisson claim counts with mean ``lam``.
    
    Parameters
    ----------
    lam : float
        Mean number of claims; must be finite and non-negative.
    
    Raises
    ------
    ValueError
        If ``lam`` is negative or not finite.
    
    Examples
    --------
    >>> from prospicio.distributions import Poisson
    >>> n = Poisson(3.0)
    >>> round(n.pmf(0), 6)
    0.049787
    """
    def __getnewargs__(self, /) -> tuple[float]: ...
    def __new__(cls, /, lam: float) -> Poisson: ...
    def __repr__(self, /) -> str: ...
    def cdf(self, /, k: int) -> float:
        """
        ``P(N <= k)``.
        
        Parameters
        ----------
        k : int
        
        Returns
        -------
        float
        """
    @property
    def lam(self, /) -> float:
        """
        The mean number of claims.
        """
    def mean(self, /) -> float:
        """
        Mean of the claim count.
        
        Returns
        -------
        float
        """
    def pmf(self, /, k: int) -> float:
        """
        ``P(N = k)``.
        
        Parameters
        ----------
        k : int
        
        Returns
        -------
        float
        """
    def quantile(self, /, p: float) -> int:
        """
        Smallest ``k`` with ``P(N <= k) >= p``.
        
        Parameters
        ----------
        p : float
            Probability in ``[0, 1)``.
        
        Returns
        -------
        int
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def sample(self, /, n: int, seed: int, stream: int = 0) -> list[int]:
        """
        ``n`` claim counts from stream ``stream`` of the generator keyed by
        ``seed``.
        
        Parameters
        ----------
        n : int
        seed : int
        stream : int, default 0
        
        Returns
        -------
        list of int
        """
    def variance(self, /) -> float:
        """
        Variance of the claim count.
        
        Returns
        -------
        float
        """

@final
class Portfolio:
    """
    A portfolio as the distribution of its total and each unit's
    conditional expectation given the total, ``kappa_i(x) = E[X_i | X = x]``:
    the representation of Mildenhall and Major (*Pricing Insurance Risk*,
    2022) and CAS Monograph 15, for pricing with limited liability and the
    natural allocation.
    
    Parameters
    ----------
    units : list of str
    rows : list of list of float
        Each scenario's loss by unit; non-negative.
    probs : list of float, optional
        The scenarios' probabilities; equal when omitted.
    
    Examples
    --------
    >>> from prospicio.pricing import Portfolio
    >>> from prospicio.risk import Distortion
    >>> rows = [[15, 7, 0], [15, 13, 0], [5, 20, 11], [7, 33, 0], [13, 20, 7],
    ...         [5, 27, 8], [15, 16, 9], [26, 19, 10], [17, 8, 40], [16, 20, 64]]
    >>> insco = Portfolio(["A", "B", "C"], rows)
    >>> price = insco.price(Distortion.ccoc(0.15), assets=100)
    >>> round(price.total.premium, 6)
    53.565217
    >>> [round(u.assets, 6) for u in price.allocated]
    [16.0, 20.0, 64.0]
    """
    def __new__(cls, /, units: Sequence[str], rows: Sequence[Sequence[float]], probs: Sequence[float] |None = None) -> Portfolio: ...
    def __repr__(self, /) -> str: ...
    def assets(self, /, p: float) -> float:
        """
        Assets at the capital standard ``p``: the lower ``p`` quantile of
        the total.
        
        Parameters
        ----------
        p : float
        
        Returns
        -------
        float
        """
    def assets_for_epd(self, /, epd: float) -> float:
        """
        The smallest assets whose total EPD ratio is at most ``epd``.
        
        Parameters
        ----------
        epd : float
        
        Returns
        -------
        float
        """
    def bodoff(self, /, assets: float |None = None, p: float |None = None) -> list[float]:
        """
        Bodoff's percentile layer of capital: each unit's share of the
        assets.
        
        Parameters
        ----------
        assets : float, optional
        p : float, optional
        
        Returns
        -------
        list of float
        """
    def calibrate(self, /, family: str, assets: float |None = None, p: float |None = None, *, premium: float |None = None, return_on_capital: float |None = None, loss_ratio: float |None = None, r0: float = 0.0) -> Distortion:
        """
        The member of a distortion family that prices the loss capped at
        the assets at a target: give exactly one of ``premium``,
        ``return_on_capital`` and ``loss_ratio``.
        
        Parameters
        ----------
        family : str
            As for ``prospicio.risk.calibrate``.
        assets : float, optional
        p : float, optional
        premium, return_on_capital, loss_ratio : float, optional
        r0 : float, default 0.0
        
        Returns
        -------
        Distortion
        """
    def comonotonic_kappa(self, /) -> dict[str, list[float]]:
        """
        Each unit's comonotonic allocation at each total, a non-decreasing
        function of the total that still adds up to it: Denuit et al.'s
        (2025) improvement of ``kappa``, ``aggregate``'s
        ``make_comonotonic_allocations``.
        
        Returns
        -------
        dict of str to list of float
            Unit name to its allocation at each of ``totals()``.
        """
    def epd(self, /, assets: float) -> tuple[float, list[float]]:
        """
        The expected policyholder deficit ratio with these assets: in
        total, and by unit under equal priority.
        
        Parameters
        ----------
        assets : float
        
        Returns
        -------
        tuple of (float, list of float)
        """
    def expected(self, /) -> list[float]:
        """
        Each unit's expected loss.
        
        Returns
        -------
        list of float
        """
    @staticmethod
    def from_independent(units: Sequence[str], grids: Sequence[Grid]) -> Portfolio:
        """
        From independent units, each a Grid with the same step; the total
        and the conditional expectations are computed by FFT.
        
        Parameters
        ----------
        units : list of str
        grids : list of Grid
        
        Returns
        -------
        Portfolio
        """
    @staticmethod
    def from_predictive(pd: PredictiveDistribution) -> Portfolio:
        """
        From a joint simulation: each component is a unit, each simulation
        an equally likely scenario.
        
        Parameters
        ----------
        pd : PredictiveDistribution
        
        Returns
        -------
        Portfolio
        """
    def kappa(self, /, unit: str) -> list[float]:
        """
        A unit's conditional expectation at each total.
        
        Parameters
        ----------
        unit : str
        
        Returns
        -------
        list of float
        """
    def premium_bounds(self, /, premium: float, assets: float |None = None, p: float |None = None) -> list[dict]:
        """
        The range of each unit's premium over every distortion that prices
        the total, capped at the assets, at ``premium`` (linear allocation).
        The extremes are BiTVaR distortions, found exactly.
        
        Parameters
        ----------
        premium : float
        assets : float, optional
        p : float, optional
        
        Returns
        -------
        list of dict
            One per unit: ``unit``, ``lower``, ``upper``, and the Distortion
            giving each, ``lower_distortion`` and ``upper_distortion``.
        """
    def price(self, /, distortion: Distortion, assets: float |None = None, p: float |None = None, allocation: str = "linear") -> NaturalPrice:
        """
        The premium of the loss capped at the assets, and its natural
        allocation to the units.
        
        Parameters
        ----------
        distortion : Distortion
        assets : float, optional
            The asset level; or give ``p``. The largest total by default.
        p : float, optional
            The capital standard: assets at the total's lower ``p`` quantile.
        allocation : {"linear", "lifted"}, default "linear"
            The unit share of the assets above the asset level: expected
            (linear) or distorted (lifted).
        
        Returns
        -------
        NaturalPrice
        """
    def price_stand_alone(self, /, distortion: Distortion, p: float) -> StandAlone:
        """
        Each unit priced as if it were the whole book, at its own lower
        ``p`` quantile, against the book priced together at the VaR of its
        total, ``aggregate``'s ``Portfolio.price_stand_alone``. The excess
        of ``sum`` over ``total`` is the diversification benefit.
        
        Parameters
        ----------
        distortion : Distortion
        p : float
            The VaR capital standard, in ``(0, 1)``.
        
        Returns
        -------
        StandAlone
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``(0, 1)``.
        
        Examples
        --------
        >>> from prospicio.pricing import Portfolio
        >>> from prospicio.risk import Distortion
        >>> port = Portfolio(["a", "b"], [[0, 0], [1, 4], [4, 1], [5, 5]])
        >>> s = port.price_stand_alone(Distortion.proportional_hazard(0.5), 0.75)
        >>> [u.assets for u in s.stand_alone], s.total.assets
        ([4.0, 4.0], 5.0)
        """
    def priority(self, /, assets: float |None = None, p: float |None = None) -> dict:
        """
        The priority ladder: each unit's recovery from the assets when it
        ranks ahead of the other units pooled (senior), pro rata (equal), or
        behind them (junior), as ``aggregate``'s ``priority_epd_df``. Each is
        unit against the rest pooled, so on three or more units only the
        equal recoveries add up to the total. Exact under any dependence for
        a portfolio built from scenarios.
        
        Parameters
        ----------
        assets : float, optional
            The largest total by default.
        p : float, optional
            The capital standard: assets at the total's lower ``p`` quantile.
        
        Returns
        -------
        dict
            Columns, one row per unit and a last ``"total"`` row: ``unit``,
            ``mean``, the recoveries ``senior``, ``equal`` and ``junior``, and
            the EPD ratios ``senior_epd``, ``equal_epd`` and ``junior_epd``
            (the total row is ``E[X ∧ a]`` under every rule); and ``assets``.
        
        Examples
        --------
        >>> from prospicio.pricing import Portfolio
        >>> port = Portfolio(["A", "B"], [[15, 7], [5, 20], [26, 19], [17, 48]])
        >>> ladder = port.priority(assets=50)
        >>> round(ladder["senior"][0] + ladder["junior"][1], 9) == ladder["senior"][2]
        True
        """
    def priority_kappa(self, /, unit: str, assets: float |None = None, p: float |None = None) -> tuple[list[float], list[float]]:
        """
        Where in the total the shortfall ``(X - a)+`` falls when ``unit`` is
        junior to the rest pooled: at each of ``totals``, the junior unit's
        expected part ``E[min((X - a)+, X_i) | X = x]`` and the senior
        pool's, which add up to ``(x - a)+``. As ``aggregate``'s
        ``priority_kappa``.
        
        Parameters
        ----------
        unit : str
        assets : float, optional
        p : float, optional
        
        Returns
        -------
        tuple of (list of float, list of float)
            The junior and the senior parts.
        """
    @property
    def probs(self, /) -> list[float]:
        """
        The probability of each total.
        """
    @property
    def totals(self, /) -> list[float]:
        """
        The distinct totals, ascending.
        """
    def tvar_threshold(self, /, p: float) -> float:
        """
        The level ``t`` at which the total's TVaR reaches its VaR at ``p``,
        ``aggregate``'s ``Portfolio.tvar_threshold(p, 'lower')``.
        
        Parameters
        ----------
        p : float
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    @property
    def units(self, /) -> list[str]:
        """
        Unit names.
        """

@final
class PortfolioPrice:
    """
    Prices of a portfolio's components and of the portfolio as a whole.
    
    Returned by ``price_portfolio``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def allocated(self, /) -> list[Price]:
        """
        Each component's share of the portfolio price; these add up to
        ``total``.
        """
    def components(self, /) -> list[Any]:
        """
        Component keys, one tuple per component.
        
        Returns
        -------
        list of tuple
        """
    def diversification(self, /) -> float:
        """
        Premium saved by writing the components together: the sum of the
        standalone premiums less the portfolio premium.
        
        Returns
        -------
        float
        """
    @property
    def standalone(self, /) -> list[Price]:
        """
        Each component priced on its own.
        """
    @property
    def total(self, /) -> Price:
        """
        The portfolio, priced on the total of its components.
        """

@final
class PotTail:
    """
    A peaks-over-threshold tail: draws above a threshold modelled by a
    fitted generalized Pareto distribution, for VaR and TVaR beyond the
    draws.
    
    Make one with ``PotTail.fit(draws, level)``, which takes the threshold
    at the empirical ``level`` quantile.
    
    Examples
    --------
    >>> from prospicio.distributions import Lognormal, Sampled
    >>> from prospicio.risk import PotTail
    >>> d = Lognormal(0.0, 1.0)
    >>> s = Sampled([d.quantile((i - 0.5) / 100_000) for i in range(1, 100_001)])
    >>> tail = PotTail.fit(s, 0.95)
    >>> abs(tail.var(0.999) / d.quantile(0.999) - 1) < 0.02
    True
    """
    def __repr__(self, /) -> str: ...
    @staticmethod
    def fit(draws: Sampled, level: float) -> PotTail:
        """
        Fits a tail to the draws above their empirical ``level`` quantile.
        
        Parameters
        ----------
        draws : Sampled
        level : float
            For example 0.95 for the top 5%.
        
        Returns
        -------
        PotTail
        """
    @property
    def gpd(self, /) -> Gpd:
        """
        The fitted GPD for the exceedances.
        """
    @property
    def p_exceed(self, /) -> float:
        """
        Share of draws above the threshold.
        """
    @property
    def threshold(self, /) -> float:
        """
        Threshold ``u``.
        """
    def tvar(self, /, p: float) -> float:
        """
        TVaR at ``p >= 1 - p_exceed``; infinite when ``xi >= 1``.
        
        Parameters
        ----------
        p : float
        
        Returns
        -------
        float
        """
    def var(self, /, p: float) -> float:
        """
        VaR at ``p >= 1 - p_exceed``.
        
        Parameters
        ----------
        p : float
        
        Returns
        -------
        float
        """

@final
class PredictiveDistribution:
    """
    The joint result every model returns: draws for each simulation (row)
    and component (column), keyed by dimension values.
    
    The ``mean``, ``quantile``, ``var`` and ``tvar`` methods describe the
    total over all components, computed from row sums.
    
    Parameters
    ----------
    dims : list of str
        Dimension names, e.g. ``["lob", "origin"]``.
    components : list of tuple
        One key per component, with one ``int`` or ``str`` per dimension.
    draws : list of list of float
        One row per simulation, one value per component.
    
    Raises
    ------
    ValueError
        If the keys or the draws do not fit together.
    
    Examples
    --------
    >>> from prospicio.distributions import PredictiveDistribution
    >>> pd = PredictiveDistribution(["line"], [("A",), ("B",)],
    ...                             [[0.0, 0.0], [0.0, 0.0], [0.0, 100.0], [100.0, 0.0]])
    >>> pd.var(0.75)
    100.0
    >>> pd.marginal(("A",)).var(0.75)
    0.0
    """
    def __new__(cls, /, dims: Sequence[str], components: Sequence[Sequence[int |str]], draws: Sequence[Sequence[float]]) -> PredictiveDistribution: ...
    def __repr__(self, /) -> str: ...
    def aggregate(self, /, keep: Sequence[str]) -> PredictiveDistribution:
        """
        Sums the components within each simulation over every dimension not
        in ``keep``, keeping the joint structure.
        
        Parameters
        ----------
        keep : list of str
        
        Returns
        -------
        PredictiveDistribution
        
        Raises
        ------
        ValueError
            If ``keep`` names an unknown dimension or repeats one.
        """
    @staticmethod
    def blend(models: Sequence[PredictiveDistribution], weights: Sequence[float], seed: int) -> PredictiveDistribution:
        """
        Blends several models' predictive distributions: simulation ``i`` is
        simulation ``i`` of model ``k``, with ``k`` drawn with probability
        ``weights[k]`` from stream ``i`` of ``seed``. Rows stay whole, so sums
        across components remain coherent. Use weights from
        ``stacking_weights`` or ``pseudo_bma_weights``.
        
        Parameters
        ----------
        models : list of PredictiveDistribution
            Same dimensions, components and number of simulations.
        weights : list of float
            Non-negative, not all zero; normalized.
        seed : int
        
        Returns
        -------
        PredictiveDistribution
        
        Examples
        --------
        >>> from prospicio.distributions import PredictiveDistribution
        >>> a = PredictiveDistribution(["lob"], [("x",)], [[0.0]] * 1000)
        >>> b = PredictiveDistribution(["lob"], [("x",)], [[1.0]] * 1000)
        >>> mix = PredictiveDistribution.blend([a, b], [0.25, 0.75], seed=7)
        >>> abs(mix.mean() - 0.75) < 0.05
        True
        """
    @staticmethod
    def blend_by_component(models: Sequence[PredictiveDistribution], weights: Sequence[Sequence[float]], seed: int) -> PredictiveDistribution:
        """
        Blends models with weights that differ by component, as
        ``HierarchicalStacking`` gives them: in simulation ``i`` every
        component draws its model from the same uniform against its own
        cumulative weights, so components with equal weights take the same
        model and dependence is kept as far as the weights allow.
        
        Parameters
        ----------
        models : list of PredictiveDistribution
        weights : list of list of float
            One weight vector per component (in ``components()`` order), one
            weight per model.
        seed : int
        
        Returns
        -------
        PredictiveDistribution
        """
    def components(self, /) -> list[Any]:
        """
        Component keys, one tuple per column.
        
        Returns
        -------
        list of tuple
        """
    @property
    def dims(self, /) -> list[str]:
        """
        Dimension names.
        """
    def draw_matrix(self, /) -> list[list[float]]:
        """
        All draws, one row per simulation.
        
        Returns
        -------
        list of list of float
        """
    @staticmethod
    def join(parts: Sequence[tuple[str, PredictiveDistribution]], dim: str, same_simulations: bool = False) -> PredictiveDistribution:
        """
        Joins distributions of different models into one portfolio, with a
        leading dimension ``dim`` holding each part's label, followed by the
        union of the parts' dimensions (``""`` where a part lacks one).
        Simulation ``i`` of the result is simulation ``i`` of every part.
        
        Parameters
        ----------
        parts : list of (str, PredictiveDistribution)
        dim : str
        same_simulations : bool, default False
            ``False``: the parts were simulated separately, and two with the
            same seed and stream scheme (which would share random numbers)
            are refused. ``True``: the parts come from the same scenarios (a
            cover applied to a reserve) and keep their pairing.
        
        Returns
        -------
        PredictiveDistribution
        
        Examples
        --------
        >>> from prospicio.distributions import PredictiveDistribution
        >>> a = PredictiveDistribution(["origin"], [(2023,), (2024,)], [[10.0, 20.0], [12.0, 25.0]])
        >>> b = PredictiveDistribution(["lob"], [("motor",)], [[50.0], [40.0]])
        >>> p = PredictiveDistribution.join([("reserve", a), ("premium", b)], "risk")
        >>> p.dims, p.total().draws
        (['risk', 'origin', 'lob'], [80.0, 77.0])
        """
    def marginal(self, /, key: Sequence[int |str]) -> Sampled |None:
        """
        One component's draws, or ``None`` if no component has this key.
        
        An origin period is named by its label, as a string or an integer:
        ``("2021",)`` or ``(2021,)`` for a year, ``("2021Q3",)`` for a
        quarter.
        
        Parameters
        ----------
        key : tuple
        
        Returns
        -------
        Sampled or None
        """
    def mean(self, /) -> float:
        """
        Mean of the total.
        
        Returns
        -------
        float
        """
    @property
    def n_components(self, /) -> int:
        """
        Number of components (columns).
        """
    @property
    def n_sims(self, /) -> int:
        """
        Number of simulations (rows).
        """
    def provenance(self, /) -> dict:
        """
        Where this result came from: model, parameters, seed, stream scheme,
        samplers, crate versions and input hash.
        
        ``stream_scheme`` names how simulations map to random-number
        streams; ``samplers`` lists ``(family, sampler id)`` pairs, e.g.
        ``[("gamma", "marsaglia-tsang/2026-10")]``: the sampler table of the
        build that made the draws, so every sampler they may have used. A
        family not listed uses its first sampler (inverse transform for a
        distribution, the documented method for a copula's frailty). A
        result replays only with the same seed, stream scheme and samplers.
        ``samplers`` is ``None`` when not recorded: a result without draws,
        one computed from draws made elsewhere, or one made by a build from
        before samplers were recorded (2026-10-08).
        
        Returns
        -------
        dict
        """
    def quantile(self, /, p: float) -> float:
        """
        Quantile of the total.
        
        Parameters
        ----------
        p : float
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def reorder_groups(self, /, dim: str, correlation: Sequence[Sequence[float]], seed: int) -> PredictiveDistribution:
        """
        Sets the dependence between the groups of dimension ``dim`` by
        Iman–Conover on the groups' totals, moving each group's simulations
        as whole rows: every group keeps its distribution and internal joint
        structure, and the group totals take a rank correlation close to
        ``correlation``.
        
        Parameters
        ----------
        dim : str
        correlation : list of list of float
            One row and column per group, in order of first appearance.
        seed : int
        
        Returns
        -------
        PredictiveDistribution
        """
    def total(self, /) -> Sampled:
        """
        The total over all components, one value per simulation.
        
        Returns
        -------
        Sampled
        """
    def tvar(self, /, p: float) -> float:
        """
        Tail value at risk of the total at level ``p``.
        
        Parameters
        ----------
        p : float
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def var(self, /, p: float) -> float:
        """
        Value at risk of the total at level ``p``.
        
        Parameters
        ----------
        p : float
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def variance(self, /) -> float:
        """
        Variance of the total.
        
        Returns
        -------
        float
        """

@final
class PreparedLosses:
    """
    Large losses trended to the prospective period, from
    ``LargeLossExperience.prepare``.
    """
    def __len__(self, /) -> int: ...
    def __repr__(self, /) -> str: ...
    def as_if(self, /, historical: Sequence[Tower], years: Sequence[int], proposed: Tower) -> AsIf:
        """
        Each year's losses under the program in force that year, next to the
        year's trended losses under the proposed tower.
        
        The historical side takes every loss of the year as reported (in its
        own currency, untrended and undeveloped), so it can be checked
        against the cedant's recoveries; the proposed side is the as-if
        layer loss of ``burning_cost`` before development by year. The
        historical towers may differ in layers from year to year.
        
        Parameters
        ----------
        historical : list of Tower
            The program in force in each of ``years``.
        years : list of int
        proposed : Tower
        
        Returns
        -------
        AsIf
        """
    def burning_cost(self, /, towers: Any, years: Sequence[int], *, premium: Sequence[float], development: Any, index: IndexClause |None = None, ratios: Sequence[float] |None = None) -> BurningCost:
        """
        The as-if loss to each layer by accident year, developed and divided
        by the premium.
        
        Parameters
        ----------
        towers : Tower or list of Tower
            The treaty in force each year, or one tower for every year (to
            price the prospective terms). Every tower needs the same number of
            layers.
        years : list of int
            The accident years to cost; a loss belongs to the year of its
            accident date.
        premium : list of float
            Each year's on-level subject premium.
        development : list of float or DevelopmentPattern
            Each year's development factor for the layer losses: Clark's
            excess factor, or, when the losses were developed one by one in
            ``prepare``, the count IBNR factor only. A count
            ``DevelopmentPattern`` is read at each year's average age at the
            record's valuation date.
        index : IndexClause, optional
            An index clause on the treaty: each loss meets the retention and
            limit moved by the clause at its own ratio in ``ratios``. Towers
            with annual terms (an aggregate deductible or limit, a corridor)
            or a surplus treaty are an error.
        ratios : list of float, optional
            With ``index``, each loss's index ratio, in the record's order:
            the index at its payment over the index at the treaty's base date.
        
        Returns
        -------
        BurningCost
        """
    @property
    def complete_above(self, /) -> float:
        """
        The trended size above which every year's record is complete: the
        threshold times the largest trend factor.
        """
    def to_dict(self, /) -> dict:
        """
        The trended losses as columns: ``id``, ``year``, ``accident``,
        ``amount``, ``fx``, ``development``, ``ultimate``, ``factor`` and
        ``trended``.
        
        Returns
        -------
        dict of str to list
        """
    @property
    def valuation(self, /) -> float |None:
        """
        The record's valuation date, or ``None``.
        """

@final
class Price:
    """
    The risk-loaded price of a cover, or of one component's share of a
    portfolio: expected loss, premium and the assets backing the loss.
    
    Returned by ``price`` and ``price_portfolio``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def assets(self, /) -> float:
        """
        Assets ``a`` backing the loss.
        """
    @property
    def capital(self, /) -> float:
        """
        Capital ``a - P``: the assets the premium does not fund.
        """
    @property
    def expected_loss(self, /) -> float:
        """
        Expected loss ``E[X]``.
        """
    @property
    def loss_ratio(self, /) -> float:
        """
        Loss ratio ``E[X] / P``.
        """
    @property
    def margin(self, /) -> float:
        """
        Margin ``P - E[X]``.
        """
    @property
    def premium(self, /) -> float:
        """
        Premium ``P``.
        """
    @property
    def return_on_capital(self, /) -> float:
        """
        Return on capital, margin over capital.
        """

@final
class PricingBounds:
    """
    The range of other risks' prices over every distortion that prices a
    reference risk ``X`` at a premium: ``aggregate``'s ``PricingBounds``,
    exact. Prices are law-invariant, so only each risk's distribution
    matters. The ends are attained by BiTVaRs, given with each bound.
    
    Parameters
    ----------
    x : Sampled, Grid or PredictiveDistribution
        The reference risk (a predictive distribution's total).
    risks : dict of str to Sampled, Grid or PredictiveDistribution
        The risks to bound.
    assets : float, optional
        Cap every risk at the assets.
    
    Examples
    --------
    >>> from prospicio.distributions import Sampled
    >>> from prospicio.pricing import PricingBounds
    >>> x = Sampled([22, 28, 36, 40, 40, 40, 40, 55, 65, 100])
    >>> y = Sampled([0, 0, 11, 0, 7, 8, 9, 10, 40, 64])
    >>> pb = PricingBounds(x, {"X3": y})
    >>> b = pb.bounds(53.565217391304344)[0]
    >>> round(b["lower"], 6), round(b["upper"], 6)
    (20.070992, 22.098038)
    """
    def __new__(cls, /, x: Any, risks: dict, assets: float |None = None) -> PricingBounds: ...
    def __repr__(self, /) -> str: ...
    def bounds(self, /, premium: float) -> list[dict]:
        """
        Each risk's lowest and highest price when ``X`` is priced at
        ``premium``.
        
        Parameters
        ----------
        premium : float
        
        Returns
        -------
        list of dict
            One per risk: ``risk``, ``lower``, ``upper``, and a Distortion
            attaining each, ``lower_distortion`` and ``upper_distortion``.
        """
    @property
    def premium_range(self, /) -> tuple[float, float]:
        """
        The premiums ``X`` can be priced at, ``(E[X], max X)``.
        """

@final
class Provisions:
    """
    The provisions proportional to premium: variable expenses ``V`` and the
    target underwriting profit ``Q`` (Werner and Modlin, ch. 7 and 8).
    
    Parameters
    ----------
    variable : float
        In ``[0, 1)``.
    profit : float
        May be negative (investment income); ``variable + profit < 1``.
    
    Examples
    --------
    >>> from prospicio.pricing import Provisions
    >>> p = Provisions(0.25, 0.10)
    >>> round(p.pure_premium_rate(300.0, 25.0), 6)
    500.0
    >>> round(p.loss_ratio_change(0.65, 0.065), 6)
    0.1
    """
    def __new__(cls, /, variable: float, profit: float) -> Provisions: ...
    def __repr__(self, /) -> str: ...
    def loss_ratio_change(self, /, loss_ratio: float, fixed_ratio: float = 0.0) -> float:
        """
        The loss ratio method's indicated change, ``(LR + F) / (1 - V - Q) - 1``.
        
        Parameters
        ----------
        loss_ratio : float
            The projected loss and LAE ratio at current rates.
        fixed_ratio : float, default 0.0
        
        Returns
        -------
        float
        """
    def permissible_loss_ratio(self, /, fixed_ratio: float = 0.0) -> float:
        """
        ``1 - F - V - Q``.
        
        Parameters
        ----------
        fixed_ratio : float, default 0.0
        
        Returns
        -------
        float
        """
    @property
    def profit(self, /) -> float:
        """
        The target profit provision.
        """
    def pure_premium_rate(self, /, pure_premium: float, fixed_per_exposure: float = 0.0) -> float:
        """
        The pure premium method's indicated rate, ``(PP + fixed) / (1 - V - Q)``.
        Other costs per exposure, such as net reinsurance, add to either.
        
        Parameters
        ----------
        pure_premium : float
        fixed_per_exposure : float, default 0.0
        
        Returns
        -------
        float
        """
    @property
    def variable(self, /) -> float:
        """
        The variable expense provision.
        """
    @property
    def variable_permissible_loss_ratio(self, /) -> float:
        """
        ``1 - V - Q``.
        """

@final
class PurePremiumExperience:
    """
    Experience years for the pure premium method: losses developed, trended
    and loaded for LAE over earned exposure, combined as the ratio of their
    totals.
    
    Parameters
    ----------
    exposure : list of float
    losses : list of float
    development, loss_trend, lae_factor : list of float or float, optional
    
    Raises
    ------
    ValueError
        For no years, mismatched lengths, or an exposure or factor that is
        not positive.
    
    Examples
    --------
    >>> from prospicio.pricing import PurePremiumExperience
    >>> e = PurePremiumExperience([10.0, 30.0], [1000.0, 2000.0], loss_trend=1.1)
    >>> round(e.pure_premium, 6)
    82.5
    """
    def __new__(cls, /, exposure: Sequence[float], losses: Sequence[float], development: Any |None = None, loss_trend: Any |None = None, lae_factor: Any |None = None) -> PurePremiumExperience: ...
    def __repr__(self, /) -> str: ...
    @property
    def exposure(self, /) -> float:
        """
        Total earned exposure.
        """
    @property
    def projected_losses(self, /) -> float:
        """
        Total projected ultimate losses and LAE.
        """
    @property
    def projected_losses_by_year(self, /) -> list[float]:
        """
        Each year's projected losses and LAE.
        """
    @property
    def pure_premium(self, /) -> float:
        """
        The projected pure premium, the ratio of the totals.
        """
    @property
    def pure_premium_by_year(self, /) -> list[float]:
        """
        Each year's projected pure premium.
        """

@final
class Quote:
    """
    A cover's quote: the premium the cedant pays, built up from a
    risk-loaded price, with the ratios the market quotes.
    
    ``Quote.technical(price, ...)`` grosses the risk premium ``R`` (the
    expected loss plus the margin for capital, from ``price``) up for
    internal expenses ``e`` and brokerage ``b``, both fractions of the
    premium, and a fixed expense ``f``: ``P = (R + f) / (1 - e - b)``.
    ``quote.at(premium)`` reads a quoted premium the other way: the
    reinsurer keeps ``P (1 - e - b) - f`` as its risk premium, and the margin
    and return on capital follow with the same expected loss and assets.
    
    Examples
    --------
    >>> from prospicio.distributions import Sampled
    >>> from prospicio.pricing import Quote, price
    >>> from prospicio.risk import Distortion
    >>> p = price(Sampled([0.0, 0.0, 2.0, 6.0]), Distortion.tvar(0.5), cost_of_capital=0.1)
    >>> q = Quote.technical(p, expenses=0.03, brokerage=0.10, limit=10.0)
    >>> round(q.premium, 6), round(q.rate_on_line, 6), round(q.return_on_capital, 6)
    (2.507837, 0.250784, 0.1)
    >>> round(q.at(3.0).return_on_capital, 4)
    0.4388
    """
    def __repr__(self, /) -> str: ...
    @property
    def assets(self, /) -> float:
        """
        Assets backing the loss.
        """
    def at(self, /, premium: float) -> Quote:
        """
        The same cover at a quoted premium: what the quote leaves after the
        loads is the risk premium.
        
        Parameters
        ----------
        premium : float
        
        Returns
        -------
        Quote
        
        Raises
        ------
        ValueError
            If the risk premium the quote leaves reaches the assets.
        """
    @property
    def brokerage(self, /) -> float:
        """
        Brokerage, an amount.
        """
    @property
    def capital(self, /) -> float:
        """
        Capital, assets less the risk premium.
        """
    @property
    def expected_loss(self, /) -> float:
        """
        Expected loss.
        """
    @property
    def expenses(self, /) -> float:
        """
        Internal expenses, an amount.
        """
    @property
    def limit(self, /) -> float |None:
        """
        The limit, or ``None``.
        """
    @property
    def loss_ratio(self, /) -> float:
        """
        Expected loss over the premium.
        """
    @property
    def margin(self, /) -> float:
        """
        Margin, risk premium less expected loss.
        """
    @property
    def multiple(self, /) -> float:
        """
        Premium over expected loss.
        """
    @property
    def net_premium(self, /) -> float:
        """
        The premium net of brokerage.
        """
    @property
    def payback(self, /) -> float |None:
        """
        Limit over premium, or ``None`` without a limit.
        """
    @property
    def premium(self, /) -> float:
        """
        The premium the cedant pays.
        """
    @property
    def rate_on_line(self, /) -> float |None:
        """
        Premium over the limit, or ``None`` without a limit.
        """
    @property
    def return_on_capital(self, /) -> float:
        """
        Return on capital, margin over capital.
        """
    @property
    def risk_premium(self, /) -> float:
        """
        The risk premium: what the reinsurer keeps for the loss and capital.
        """
    @staticmethod
    def technical(price: Price, *, expenses: float = 0.0, brokerage: float = 0.0, fixed: float = 0.0, limit: float |None = None) -> Quote:
        """
        The technical premium of a risk-loaded price.
        
        Parameters
        ----------
        price : Price
            The risk-loaded price, from ``price``; its premium is the risk
            premium.
        expenses : float, default 0.0
            Internal expenses, a fraction of the premium.
        brokerage : float, default 0.0
            Brokerage, a fraction of the premium.
        fixed : float, default 0.0
            A fixed expense per contract, an amount.
        limit : float, optional
            The cover's limit at the placed share, for the rate on line;
            omit it for a cover without one, such as a quota share.
        
        Returns
        -------
        Quote
        
        Raises
        ------
        ValueError
            If a load is negative, ``expenses + brokerage`` reaches 1, or the
            limit is not positive.
        """
    def to_dict(self, /) -> dict:
        """
        The build-up and ratios as a dict, for a table.
        
        Returns
        -------
        dict of str to float
        """

@final
class RateHistory:
    """
    A history of rate changes from an initial level of 1, and the
    parallelogram method: the portion of a period's premium at each rate
    level under uniform writing and earning, and the factor to the current
    level (Werner and Modlin, ch. 5).
    
    Parameters
    ----------
    effective : list of float
        Effective dates as year fractions, in order.
    changes : list of float
        Overall average changes, ``0.05`` for +5%.
    applies : list of str, optional
        ``"written"`` (policies written on or after the date; the default)
        or ``"in_force"`` (all premium earned on or after it, a law change).
    
    Raises
    ------
    ValueError
        For dates out of order, a change of -100% or less, or mismatched
        lengths.
    
    Examples
    --------
    >>> from prospicio.pricing import Period, RateHistory
    >>> h = RateHistory([2010.5, 2011.0, 2012.25], [0.05, 0.10, -0.01])
    >>> level = h.rate_level(Period.calendar(2011.0, 2012.0), term=1.0)
    >>> level.shares
    [0.125, 0.375, 0.5]
    >>> round(level.on_level_factor, 4)
    1.0431
    """
    def __new__(cls, /, effective: Sequence[float], changes: Sequence[float], applies: Sequence[str] |None = None) -> RateHistory: ...
    def __repr__(self, /) -> str: ...
    @property
    def current_level(self, /) -> float:
        """
        The level after every change.
        """
    @staticmethod
    def from_levels(effective: Sequence[float], levels: Sequence[float], applies: Sequence[str] |None = None) -> RateHistory:
        """
        A history from the cumulative level after each change, as an
        exhibit prints them; rounded printed levels carry into the factors.
        
        Parameters
        ----------
        effective : list of float
        levels : list of float
        applies : list of str, optional
        
        Returns
        -------
        RateHistory
        """
    def rate_level(self, /, period: Period, term: float) -> RateLevel:
        """
        The portion of ``period``'s premium at each rate level, for
        policies of ``term`` years.
        
        Parameters
        ----------
        period : Period
        term : float
        
        Returns
        -------
        RateLevel
        """

@final
class RateLevel:
    """
    A period's premium by rate level, from ``RateHistory.rate_level``. Each
    list has one entry per cell of constant level with a positive share.
    """
    def __repr__(self, /) -> str: ...
    @property
    def average_level(self, /) -> float:
        """
        ``sum(shares * levels)``.
        """
    @property
    def current_level(self, /) -> float:
        """
        The level after every change in the history.
        """
    @property
    def in_force_changes(self, /) -> list[int]:
        """
        The number of in-force changes in effect for each cell.
        """
    @property
    def levels(self, /) -> list[float]:
        """
        Each cell's cumulative rate level.
        """
    @property
    def on_level_factor(self, /) -> float:
        """
        ``current_level / average_level``.
        """
    @property
    def shares(self, /) -> list[float]:
        """
        Each cell's share of the period's premium.
        """
    @property
    def written_changes(self, /) -> list[int]:
        """
        The number of written-policy changes in effect for each cell.
        """

@final
class RenewalRuin:
    """
    The probability of eventual ruin at each initial surplus on the grid
    when the waits between claims follow any law, by Wiener–Hopf, from
    ``wiener_hopf``.
    """
    def __repr__(self, /) -> str: ...
    def capital(self, /, p: float, interpolate: bool = False) -> float |None:
        """
        The initial surplus for ruin probability ``p``: the smallest grid
        point with ``psi(u) < p``, or linear between grid points.
        
        Parameters
        ----------
        p : float
        interpolate : bool, default False
        
        Returns
        -------
        float or None
            None when the grid does not reach it.
        """
    @property
    def maximum(self, /) -> list[float]:
        """
        The probabilities of the claim walk's all-time maximum at each grid
        point.
        """
    @property
    def mean_step(self, /) -> float:
        """
        The grid's mean step of the claim walk ``X - cW``, negative.
        """
    @property
    def premium_rate(self, /) -> float:
        """
        The premium rate ``c = (1 + rho) E[X] / E[W]``.
        """
    @property
    def psi(self, /) -> list[float]:
        """
        ``psi(u)`` at each grid point; the last shows the mass beyond the
        grid.
        """
    def psi_at(self, /, u: float) -> float:
        """
        ``psi(u)`` at any initial surplus (constant between grid points).
        
        Parameters
        ----------
        u : float
        
        Returns
        -------
        float
        """
    @property
    def rho(self, /) -> float:
        """
        The margin ``rho``.
        """
    @property
    def severity_mean(self, /) -> float:
        """
        The severity's mean on the grid.
        """
    @property
    def u(self, /) -> list[float]:
        """
        The initial surplus at each grid point.
        """

@final
class RiskProfile:
    """
    A risk profile for property per-risk business: bands of sum insured,
    each with an expected loss (given, or premium times a loss ratio) and
    its own exposure curve.
    
    Each band's representative risk has sum insured ``SI`` (its total sum
    insured over its number of risks, say), taken as its MPL. The band
    expects ``EL / (SI * curve.mean_rate)`` losses a year; each simulated
    loss is the band's ``SI`` times a destruction rate from the band's
    curve, and carries that ``SI``, so a surplus treaty (``Layer.surplus``)
    and the per-risk excess of loss it inures to apply to the events. The
    exposure-rated expectations (``expected_layer_loss``,
    ``expected_surplus_loss``) check the simulation.
    
    Parameters
    ----------
    sums_insured : list of float
        One per band.
    risks : list of float
        Number of risks per band (for reference).
    curves : Mbbefd or TabulatedCurve, or a list of them
        One curve for every band, or one per band.
    expected_losses : list of float, optional
        Expected annual loss per band. Give this, or ``premiums``.
    premiums : list of float, optional
        Premium per band, with ``loss_ratio``.
    loss_ratio : float or list of float, optional
        Expected loss ratio, one for all bands or one per band.
    lower, upper : list of float or None, optional
        Bounds of each band's sums insured (``None`` for a band without).
        A band with bounds spreads its risks' sums insured uniformly between
        them: its mean ``SI`` is ``(lower + upper) / 2`` (in place of
        ``sums_insured``), each simulated loss draws its own ``SI`` between
        the bounds, and the exposure-rated expectations average over the
        band, weighted by sum insured.
    spread : {"uniform", "tilted"}, default "uniform"
        How a band with bounds spreads its sums insured. ``"tilted"`` keeps
        ``sums_insured`` as the mean, so the spread matches both the bounds
        and the band's total sum insured: the density ``∝ exp(θ s)`` on the
        bounds with ``θ`` solved for that mean (``sums_insured`` must lie
        strictly between the bounds).
    
    Examples
    --------
    >>> from prospicio.pricing import Mbbefd, RiskProfile
    >>> p = RiskProfile([1e6, 10e6], [800, 50], Mbbefd.swiss_re(3.0),
    ...                 premiums=[2e6, 1e6], loss_ratio=0.6)
    >>> round(p.expected_loss())
    1800000
    >>> events = p.simulate(1000, 7)
    >>> events.has_sums_insured
    True
    """
    def __new__(cls, /, sums_insured: Sequence[float], risks: Sequence[float], curves: Any, expected_losses: Sequence[float] |None = None, premiums: Sequence[float] |None = None, loss_ratio: Any |None = None, lower: Sequence[float |None] |None = None, upper: Sequence[float |None] |None = None, spread: str = "uniform") -> RiskProfile: ...
    def __repr__(self, /) -> str: ...
    def expected_claims(self, /) -> list[float]:
        """
        Expected number of losses a year, per band.
        
        Returns
        -------
        list of float
        """
    def expected_layer_loss(self, /, limit: float, attachment: float, surplus_retention: float |None = None, surplus_lines: float |None = None) -> float:
        """
        Exposure-rated expected loss to a per-risk layer ``limit`` xs
        ``attachment``, optionally on each risk net of a surplus treaty.
        
        Parameters
        ----------
        limit : float
            ``inf`` for unlimited.
        attachment : float
        surplus_retention, surplus_lines : float, optional
            A surplus treaty the layer inures to.
        
        Returns
        -------
        float
        """
    def expected_loss(self, /) -> float:
        """
        Expected annual loss, all bands.
        
        Returns
        -------
        float
        """
    def expected_surplus_loss(self, /, retention: float, lines: float) -> float:
        """
        Expected annual loss ceded to a surplus treaty.
        
        Parameters
        ----------
        retention : float
        lines : float
        
        Returns
        -------
        float
        """
    def simulate(self, /, n_sims: int, seed: int) -> EventSet:
        """
        ``n_sims`` years of losses, each with its risk's sum insured.
        
        Parameters
        ----------
        n_sims : int
        seed : int
        
        Returns
        -------
        EventSet
        """

@final
class Ruin:
    """
    The probability of eventual ruin at each initial surplus on the grid, by
    the Pollaczeck–Khinchine formula, from ``eventual_ruin``.
    """
    def __repr__(self, /) -> str: ...
    def capital(self, /, p: float, interpolate: bool = False) -> float |None:
        """
        The initial surplus for ruin probability ``p``: the smallest grid
        point with ``psi(u) < p``, or linear between grid points.
        
        Parameters
        ----------
        p : float
        interpolate : bool, default False
        
        Returns
        -------
        float or None
            None when the grid does not reach it.
        """
    @property
    def equilibrium(self, /) -> Grid:
        """
        The equilibrium (integrated tail) severity.
        """
    @property
    def lundberg_exponent(self, /) -> float |None:
        """
        The Lundberg exponent ``R`` with ``psi(u) <= exp(-R u)``, or None
        when the grid's severity has no root below ``700 / max(x)``.
        """
    @property
    def psi(self, /) -> list[float]:
        """
        ``psi(u)`` at each grid point; the last is the mass beyond the grid.
        """
    def psi_at(self, /, u: float) -> float:
        """
        ``psi(u)`` at any initial surplus (constant between grid points).
        
        Parameters
        ----------
        u : float
        
        Returns
        -------
        float
        """
    @property
    def report(self, /) -> CompoundReport:
        """
        The compound FFT's report; read ``aliasing_error``.
        """
    @property
    def rho(self, /) -> float:
        """
        The margin ``rho``; the loss ratio is ``1 / (1 + rho)``.
        """
    @property
    def severity_mean(self, /) -> float:
        """
        The severity's mean on the grid, after any cap, excess or stop loss.
        """
    @property
    def u(self, /) -> list[float]:
        """
        The initial surplus at each grid point.
        """

@final
class Sampled:
    """
    A distribution known only through equally weighted draws.
    
    Parameters
    ----------
    draws : list of float
        Non-empty, all finite.
    
    Raises
    ------
    ValueError
        If ``draws`` is empty or holds a value that is not finite.
    
    Examples
    --------
    >>> from prospicio.distributions import Sampled
    >>> s = Sampled([1.0, 2.0, 3.0, 4.0])
    >>> s.tvar(0.5)
    3.5
    """
    def __getnewargs__(self, /) -> tuple[list[float]]: ...
    def __len__(self, /) -> int: ...
    def __new__(cls, /, draws: Sequence[float]) -> Sampled: ...
    def __repr__(self, /) -> str: ...
    def cdf(self, /, x: float) -> float:
        """
        Empirical distribution function ``P(X <= x)``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @property
    def draws(self, /) -> list[float]:
        """
        The draws, in simulation order.
        """
    def mean(self, /) -> float:
        """
        Mean of the draws.
        
        Returns
        -------
        float
        """
    def quantile(self, /, p: float) -> float:
        """
        Inverted empirical cdf (R ``type = 1``).
        
        Parameters
        ----------
        p : float
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def tvar(self, /, p: float) -> float:
        """
        Tail value at risk at level ``p``: the mean of the worst ``1 - p``.
        
        Parameters
        ----------
        p : float
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def var(self, /, p: float) -> float:
        """
        Value at risk at level ``p``.
        
        Parameters
        ----------
        p : float
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def variance(self, /) -> float:
        """
        Variance of the draws (dividing by ``n``).
        
        Returns
        -------
        float
        """

@final
class StackingFit:
    """
    Posterior stacking weights, from ``BayesStacking.fit`` or
    ``HierarchicalStacking.fit``.
    """
    @property
    def alpha_draws(self, /) -> list[float]:
        """
        Intercept draws (the logits for Bayesian stacking), one row per draw,
        one per model but the reference (last).
        """
    @property
    def beta_draws(self, /) -> list[float]:
        """
        Slope draws, flattened draw by draw, model by model, covariate by
        covariate.
        """
    @property
    def divergences(self, /) -> int:
        """
        Divergent transitions among the kept draws.
        """
    def rhat_ess(self, /) -> list[tuple[float, float]]:
        """
        R-hat and bulk ESS of each sampled parameter.
        
        Returns
        -------
        list of (float, float)
        """
    def weights(self, /, covariates: Sequence[Sequence[float]] |None = None) -> list[list[float]]:
        """
        Posterior mean weights: one row per observation of ``covariates``
        (one list per covariate), one weight per model. For Bayesian
        stacking leave ``covariates`` empty: one row.
        
        Parameters
        ----------
        covariates : list of list of float, optional
        
        Returns
        -------
        list of list of float
        """

@final
class StandAlone:
    """
    Each unit priced alone, the book priced together and the sum of the
    parts, from ``Portfolio.price_stand_alone``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def stand_alone(self, /) -> list[Pentagon]:
        """
        Each unit as if it were the whole book, at its own VaR: a list of
        Pentagon.
        """
    @property
    def sum(self, /) -> Pentagon:
        """
        The stand-alone loss, premium and assets added up: a Pentagon.
        """
    def to_dict(self, /) -> dict:
        """
        The table as a dict of columns, as ``NaturalPrice.to_dict``, with
        rows for the units, ``"total"`` and ``"sum"``.
        
        Returns
        -------
        dict of str to list
        """
    @property
    def total(self, /) -> Pentagon:
        """
        The book, at the VaR of its total: a Pentagon.
        """
    @property
    def units(self, /) -> list[str]:
        """
        Unit names.
        """

@final
class StudentTCopula:
    """
    The Student t copula with correlation matrix ``correlation`` and ``nu``
    degrees of freedom: Gaussian-like correlation with joint extremes.
    
    Parameters
    ----------
    correlation : list of list of float
        Symmetric, unit diagonal, positive definite.
    nu : float
        Degrees of freedom, positive.
    
    Raises
    ------
    ValueError
        If the matrix or ``nu`` is invalid.
    
    Examples
    --------
    >>> from prospicio.risk import StudentTCopula
    >>> StudentTCopula([[1.0, 0.5], [0.5, 1.0]], 4.0).dim
    2
    """
    def __new__(cls, /, correlation: Sequence[Sequence[float]], nu: float) -> StudentTCopula: ...
    def cdf(self, /, u: Sequence[float]) -> float:
        """
        The copula's distribution function at the point ``u``, each
        coordinate in ``[0, 1]``.
        
        Parameters
        ----------
        u : list of float
            One value per dimension.
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``u`` has the wrong length or a value outside ``[0, 1]``,
            or the copula has no closed-form distribution function here.
        
        The t copula has none yet: this always raises ``ValueError``.
        """
    @property
    def dim(self, /) -> int:
        """
        Number of dimensions.
        """
    @property
    def nu(self, /) -> float:
        """
        Degrees of freedom.
        """
    def sample(self, /, n: int, seed: int) -> list[list[float]]:
        """
        ``n`` draws of uniforms; draw ``i`` uses stream ``i`` of ``seed``.
        
        Parameters
        ----------
        n : int
        seed : int
        
        Returns
        -------
        list of list of float
        """

@final
class TabulatedCurve:
    """
    A tabulated exposure curve: points ``(x, G(x))`` from ``(0, 0)`` to
    ``(1, 1)``, interpolated linearly, as published curves are given
    (Salzmann's homeowners scale, Ludwig's curves, ISO PSOLD tables, a
    reinsurer's own).
    
    The table must be concave (its slopes never increase). Its destruction
    rate is discrete: the points' ``x`` with probabilities from the drops in
    slope, and a total loss with probability last slope over first. Its
    mean rate is the first chord's, ``x1 / G(x1)``, so a table needs fine
    first points for the expected loss to be right.
    
    Parameters
    ----------
    x : list of float
        Increasing from 0 to 1.
    g : list of float
        ``G(x)``, from 0 to 1.
    
    Raises
    ------
    ValueError
        If the points do not run from ``(0, 0)`` to ``(1, 1)``, or are not
        increasing and concave.
    
    Examples
    --------
    >>> from prospicio.pricing import TabulatedCurve
    >>> t = TabulatedCurve([0.0, 0.1, 0.5, 1.0], [0.0, 0.4, 0.8, 1.0])
    >>> round(t.curve([0.3])[0], 12), t.mean_rate()
    (0.6, 0.25)
    """
    def __getnewargs__(self, /) -> tuple[list[float], list[float]]: ...
    def __new__(cls, /, x: Sequence[float], g: Sequence[float]) -> TabulatedCurve: ...
    def __repr__(self, /) -> str: ...
    def curve(self, /, x: Sequence[float]) -> list[float]:
        """
        The exposure curve ``G(x)`` at each ``x`` (clamped to [0, 1]).
        
        Parameters
        ----------
        x : list of float
        
        Returns
        -------
        list of float
        """
    @property
    def g(self, /) -> list[float]:
        """
        The table's ``G(x)``.
        """
    def layer_share(self, /, limit: float, attachment: float, mpl: float) -> float:
        """
        Share of a risk's expected loss in the layer ``limit`` xs
        ``attachment``, for a risk with maximum possible loss ``mpl``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        mpl : float
        
        Returns
        -------
        float
        """
    def mean_rate(self, /) -> float:
        """
        Mean destruction rate, ``x1 / G(x1)``.
        
        Returns
        -------
        float
        """
    def rate_quantile(self, /, u: Sequence[float]) -> list[float]:
        """
        Destruction rate at each probability ``u`` in ``(0, 1)``.
        
        Parameters
        ----------
        u : list of float
        
        Returns
        -------
        list of float
        """
    @property
    def x(self, /) -> list[float]:
        """
        The table's ``x``.
        """

@final
class TailBondy:
    """
    The Bondy tail, as chainladder-python's ``TailBondy``.
    
    Each log factor from ``earliest_age`` on is taken as ``b`` times the one
    before it, ``b`` fitted by least squares. The fitted factors are
    ``f0 ** (b ** j)`` from the factor ``f0`` at ``earliest_age``, and those
    past the next one multiply to the last fitted factor raised to
    ``b / (1 - b)``. With the default ``earliest_age`` (the age of the last
    factor) ``b`` is 1/2 and the tail repeats the last factor.
    
    Parameters
    ----------
    earliest_age : int, optional
        First age in months whose factor enters the fit (the last age at or
        before it, as chainladder-python reads it); the age of the last
        factor by default.
    attachment_age : int, optional
        The factor from this age (the last age at or before it) to the next
        is kept and the fitted ones replace those after it; the age of the
        last factor by default. Not before ``earliest_age``.
    
    Examples
    --------
    >>> from prospicio.reserving import ChainLadder, TailBondy, Triangle
    >>> tri = Triangle.from_long(
    ...     [2020, 2020, 2020, 2021, 2021, 2022],
    ...     [12, 24, 36, 12, 24, 12],
    ...     [100.0, 150.0, 165.0, 110.0, 170.0, 120.0],
    ... )
    >>> round(ChainLadder(tail=TailBondy()).fit(tri, "values").tail, 12)
    1.1
    """
    def __new__(cls, /, earliest_age: int |None = None, attachment_age: int |None = None) -> TailBondy: ...
    def __repr__(self, /) -> str: ...
    @property
    def attachment_age(self, /) -> int |None:
        """
        Age after which the fitted factors replace the estimated ones;
        ``None`` is the age of the last factor.
        """
    @property
    def earliest_age(self, /) -> int |None:
        """
        First age whose factor enters the fit; ``None`` is the age of the
        last factor.
        """

@final
class TailConstant:
    """
    A given tail factor, as chainladder-python's ``TailConstant``.
    
    The factor applies from the attachment age to ultimate. Past the
    attachment it is spread over the following periods as
    ``1 + x * decay**k``, the last factor making up the difference; this
    shapes the factors past the attachment, not the factor to ultimate. An
    attachment before the oldest age replaces the estimated factors from
    there.
    
    Parameters
    ----------
    factor : float, default 1.0
        Factor from the attachment age to ultimate; finite and positive.
    decay : float, default 0.5
        Share of each period's development kept in the next, from 0 to 1.
    attachment_age : int, optional
        Age in months the factor attaches at (the first age at or after
        it); the oldest age by default. An age at or before the youngest
        replaces every estimated factor (chainladder-python ignores such an
        attachment).
    
    Examples
    --------
    >>> from prospicio.reserving import ChainLadder, TailConstant, Triangle
    >>> tri = Triangle.from_long([2020, 2020, 2021], [12, 24, 12], {"paid": [100.0, 150.0, 200.0]})
    >>> fit = ChainLadder(tail=TailConstant(1.05)).fit(tri, "paid")
    >>> fit.tail, round(fit.ultimate[1], 6)
    (1.05, 315.0)
    """
    def __new__(cls, /, factor: float = 1.0, decay: float = 0.5, attachment_age: int |None = None) -> TailConstant: ...
    def __repr__(self, /) -> str: ...
    @property
    def attachment_age(self, /) -> int |None:
        """
        Age in months the factor attaches at; ``None`` is the oldest age.
        """
    @property
    def decay(self, /) -> float:
        """
        Share of each period's development kept in the next.
        """
    @property
    def factor(self, /) -> float:
        """
        Factor from the attachment age to ultimate.
        """

@final
class TailCurve:
    """
    A curve fitted to the estimated factors and extrapolated, as
    chainladder-python's ``TailCurve``.
    
    Factors above 1.00001 in the fit period are regressed by least squares:
    ``ln(f - 1)`` on the 1-based development index ``k`` (exponential) or on
    ``ln(k)`` (inverse power). The fitted curve replaces the factors from the
    attachment age on and runs ``extrap_periods`` periods past the oldest
    age.
    
    Parameters
    ----------
    curve : {"exponential", "inverse_power"}, default "exponential"
    fit_period : tuple of (int or None, int or None), default (None, None)
        Ages in months whose factors enter the fit: from the last age at or
        before the first (inclusive) to the last age at or before the second
        (exclusive), as chainladder-python reads them; ``None`` is
        open-ended.
    extrap_periods : int, default 100
        Number of periods past the oldest age the curve is extrapolated.
    attachment_age : int, optional
        Age in months the curve attaches at (the first age at or after it);
        the oldest age by default.
    
    Examples
    --------
    >>> from prospicio.reserving import ChainLadder, TailCurve, Triangle
    >>> tri = Triangle.from_long(
    ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
    ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
    ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
    ... )
    >>> fit = ChainLadder(tail=TailCurve()).fit(tri, "values")
    >>> 1.0 < fit.tail < 1.05
    True
    """
    def __new__(cls, /, curve: str = "exponential", fit_period: tuple[int |None, int |None] = ..., extrap_periods: int = 100, attachment_age: int |None = None) -> TailCurve: ...
    def __repr__(self, /) -> str: ...
    @property
    def attachment_age(self, /) -> int |None:
        """
        Age in months the curve attaches at; ``None`` is the oldest age.
        """
    @property
    def curve(self, /) -> str:
        """
        The curve fitted to ``f - 1``.
        """
    @property
    def extrap_periods(self, /) -> int:
        """
        Number of periods past the oldest age the curve is extrapolated.
        """
    @property
    def fit_period(self, /) -> tuple[int |None, int |None]:
        """
        Ages whose factors enter the fit, from (inclusive) and to
        (exclusive).
        """

@final
class TailLogLinear:
    """
    R ChainLadder's ``tail = TRUE`` rule (its ``tailfactor`` function).
    
    When the third- and second-last factors multiply to more than 1.0001,
    ``ln(f - 1)`` is regressed on the development index over the factors
    above 1 and the next 100 extrapolated factors are multiplied; otherwise
    the tail is 1. A tail above 2 is reset to 1, as R does.
    
    Examples
    --------
    >>> from prospicio.reserving import Mack, TailLogLinear, Triangle
    >>> tri = Triangle.from_long(
    ...     [2020] * 4 + [2021] * 3 + [2022] * 2 + [2023],
    ...     [12, 24, 36, 48, 12, 24, 36, 12, 24, 12],
    ...     [100.0, 150.0, 165.0, 170.0, 110.0, 170.0, 180.0, 120.0, 175.0, 130.0],
    ... )
    >>> fit = Mack(tail=TailLogLinear()).fit(tri, "values")
    >>> fit.tail > 1.0 and fit.standard_error[0] > 0.0
    True
    """
    def __new__(cls, /) -> TailLogLinear: ...
    def __repr__(self, /) -> str: ...

@final
class Terms:
    """
    The terms of a model: an intercept, numeric columns and factors.
    
    Build them up, then ``fit`` them to training data to learn the factor
    levels; the result builds the same design matrix on any data.
    
    Examples
    --------
    >>> from prospicio.models import Terms
    >>> data = {"age": [30.0, 45.0, 60.0], "region": ["N", "S", "W"]}
    >>> coding = Terms().intercept().numeric("age").factor("region").fit(data)
    >>> coding.names
    ['(Intercept)', 'age', 'region[S]', 'region[W]']
    """
    def __new__(cls, /) -> Terms: ...
    def factor(self, /, name: str, reference: str |None = None) -> Terms:
        """
        Adds a factor in treatment coding.
        
        Parameters
        ----------
        name : str
        reference : str, optional
            Reference level; the first in sorted order by default.
        
        Returns
        -------
        Terms
        """
    def fit(self, /, data: Any) -> Coding:
        """
        Learns factor levels from training data.
        
        Parameters
        ----------
        data : dict of str to list
            Numeric columns as lists of numbers, factors as lists of strings.
        
        Returns
        -------
        Coding
        
        Raises
        ------
        ValueError
            If a column is missing or has the wrong kind.
        """
    def intercept(self, /) -> Terms:
        """
        Adds an intercept.
        
        Returns
        -------
        Terms
        """
    def numeric(self, /, name: str) -> Terms:
        """
        Adds a numeric column.
        
        Parameters
        ----------
        name : str
        
        Returns
        -------
        Terms
        """

@final
class Tower:
    """
    A reinsurance programme: layers in inuring stages.
    
    ``Tower(layers)`` is one stage: every layer sees the gross losses.
    ``Tower.inuring(stages)`` applies stages in order, each seeing the losses
    net of all earlier stages, event by event.
    
    Parameters
    ----------
    layers : list of Layer
        At least one; names must be unique.
    
    Raises
    ------
    ValueError
        If there are no layers or two share a name.
    
    Examples
    --------
    >>> from prospicio.aggregate import simulate_events
    >>> from prospicio.reinsurance import Layer, Tower
    >>> from prospicio.distributions import Lognormal, Poisson
    >>> events = simulate_events(Poisson(2.0), Lognormal.from_mean_cv(3e6, 1.5), 1_000, 7)
    >>> tower = Tower([Layer("5x5", 5e6, 5e6), Layer("15x10", 15e6, 10e6)])
    >>> result = tower.apply(events)
    >>> [k[0] for k in result.aggregate(["kind"]).components()]
    ['gross', 'ceded', 'net']
    """
    def __new__(cls, /, layers: Sequence[Layer]) -> Tower: ...
    def __reduce__(self, /) -> tuple[Any, tuple[str]]: ...
    def __repr__(self, /) -> str: ...
    def apply(self, /, events: EventSet) -> PredictiveDistribution:
        """
        Applies the tower to every simulated year.
        
        The result has dimensions ``["kind", "layer"]``: ``("gross",
        "ground_up")``, ``("ceded", name)`` per layer, ``("net",
        "retained")``, then ``("reinstatement_premium", name)`` per layer with
        paid reinstatements, ``("swing_premium", name)`` per swing-rated
        layer, ``("ceding_commission", name)`` per layer with a flat or
        sliding commission and ``("profit_commission", name)`` per layer with
        a profit commission. ``aggregate(["kind"])`` gives gross, total ceded
        and net; net is a loss, before premiums and commissions.
        
        Parameters
        ----------
        events : EventSet
        
        Returns
        -------
        PredictiveDistribution
        """
    def apply_aggregate(self, /, losses: PredictiveDistribution) -> PredictiveDistribution:
        """
        Applies the tower to any predictive distribution, each simulation's
        total taken as one aggregate loss: an adverse development cover on a
        reserve bootstrap, a stop-loss or quota share on modelled premium
        risk. An occurrence layer sees the total as one occurrence, so it
        acts as an aggregate excess of loss. Components as ``apply``.
        
        Parameters
        ----------
        losses : PredictiveDistribution
        
        Returns
        -------
        PredictiveDistribution
        """
    def ceded(self, /, losses: Sequence[float]) -> list[float]:
        """
        Ceded loss of each layer, in order, for one year's losses.
        
        Parameters
        ----------
        losses : list of float
        
        Returns
        -------
        list of float
        """
    @staticmethod
    def from_json(text: str) -> Tower:
        """
        A tower whose stages inure in order.
        
        Each stage's layers see the losses net of all earlier stages, event
        by event, with annual terms used up in event order.
        
        Parameters
        ----------
        stages : list of list of Layer
            No stage may be empty; names must be unique across stages.
        
        Returns
        -------
        Tower
        
        Examples
        --------
        >>> from prospicio.reinsurance import Layer, Tower
        >>> tower = Tower.inuring([[Layer.quota_share("QS", 0.5)], [Layer("5x5", 5.0, 5.0)]])
        >>> tower.ceded([30.0])
        [15.0, 5.0]
        Reads a document written by ``Tower.to_json``.
        
        Parameters
        ----------
        text : str
        
        Returns
        -------
        Tower
        
        Raises
        ------
        ValueError
            On malformed JSON, another format, a newer format version, or a
            term a layer refuses.
        """
    @staticmethod
    def inuring(stages: Sequence[Sequence[Layer]]) -> Tower:
        """
        A tower whose stages inure in order.
        
        Each stage's layers see the losses net of all earlier stages, event
        by event, with annual terms used up in event order.
        
        Parameters
        ----------
        stages : list of list of Layer
            No stage may be empty; names must be unique across stages.
        
        Returns
        -------
        Tower
        
        Examples
        --------
        >>> from prospicio.reinsurance import Layer, Tower
        >>> tower = Tower.inuring([[Layer.quota_share("QS", 0.5)], [Layer("5x5", 5.0, 5.0)]])
        >>> tower.ceded([30.0])
        [15.0, 5.0]
        """
    @property
    def layer_names(self, /) -> list[str]:
        """
        Layer names, in order.
        """
    def on_annual_grid(self, /, annual: Grid) -> AnnualGrids:
        """
        The tower applied to an annual loss on a grid, each year's total
        treated as one loss (as ``apply_aggregate`` does for draws): a quota
        share or stop-loss on a loss-ratio or annual-loss distribution, with
        its expected sliding-scale and profit commission. Inuring stages and
        annual terms are all allowed, and the net always exists.
        
        Parameters
        ----------
        annual : Grid
            The annual loss.
        
        Returns
        -------
        AnnualGrids
        
        Raises
        ------
        ValueError
            For a surplus treaty or reinstatements pro rata as to time, which
            need risks or event times.
        
        Examples
        --------
        >>> from prospicio.reinsurance import Layer, Tower
        >>> from prospicio.distributions import Grid
        >>> probs = [0.0] * 81
        >>> for k in (50, 60, 70, 80):
        ...     probs[k] = 0.25
        >>> qs = Layer.quota_share("QS", 0.4).with_deposit_premium(100.0).with_ceding_commission(0.3)
        >>> r = Tower([qs]).on_annual_grid(Grid(1.0, probs))
        >>> round(r.ceded[0].mean(), 12), round(r.expected_ceding_commission[0], 12)
        (26.0, 12.0)
        """
    def on_grid(self, /, frequency: Any, severity: Grid, points: int) -> TowerGrids:
        """
        Gross, ceded and net annual distributions on the grid, by FFT.
        
        Each layer's per-occurrence recoveries form a severity grid, which
        is compounded with the same claim count; annual terms and the share
        then apply to the total. With boundaries on multiples of the step
        the grids are exact for the discretized problem, with no sampling
        error. Grids are marginal (use ``apply`` on simulated events for
        joint results). ``net`` is given when no layer has annual terms, or
        when the last stage is a single aggregate cover such as a
        stop-loss; otherwise it is ``None``.
        
        Parameters
        ----------
        frequency : Poisson, NegativeBinomial or Binomial
        severity : Grid
        points : int
            Points in every aggregate grid.
        
        Returns
        -------
        TowerGrids
        
        Raises
        ------
        ValueError
            If a layer with annual terms inures to a later stage, or
            ``points`` is 0.
        
        Examples
        --------
        >>> from prospicio.reinsurance import Layer, Tower
        >>> from prospicio.distributions import Grid, Poisson
        >>> sev = Grid(1.0, [0.0, 0.4, 0.3, 0.2, 0.1])
        >>> r = Tower([Layer("2x2", 2.0, 2.0)]).on_grid(Poisson(3.0), sev, 200)
        >>> round(r.ceded[0].mean(), 12), r.on_points
        (1.2, True)
        """
    @property
    def stages(self, /) -> list[int]:
        """
        Stage of each layer, in order, starting at 0.
        """
    def to_json(self, /) -> str:
        """
        The programme as a versioned JSON document: every stage and layer
        with all its terms, numbers bit for bit. ``Tower.from_json`` reads it
        back to an equal tower, rebuilding each layer through the same
        checks; towers also pickle this way.
        
        Returns
        -------
        str
        
        Examples
        --------
        >>> from prospicio.reinsurance import Layer, Tower
        >>> tower = Tower.inuring([[Layer.surplus("S", 1e6, 4.0)], [Layer("xl", 2e6, 1e6)]])
        >>> back = Tower.from_json(tower.to_json())
        >>> back.to_json() == tower.to_json()
        True
        """

@final
class TowerGrids:
    """
    A tower's annual distributions on the grid, from ``Tower.on_grid``.
    
    Every grid is a marginal distribution. Ceded grids are at the placed
    share and after annual terms; a layer with share ``c`` has step
    ``c * h``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def ceded(self, /) -> list[Grid]:
        """
        Annual ceded loss of each layer, in tower order.
        """
    @property
    def ceded_reports(self, /) -> list[CompoundReport]:
        """
        The compound calculation behind each layer: its annual recovery at
        100%, before annual terms.
        """
    @property
    def expected_ceding_commission(self, /) -> list[float]:
        """
        Expected ceding commission (flat or sliding) of each layer.
        """
    @property
    def expected_premium(self, /) -> list[float]:
        """
        Expected premium of each layer: the swing-rated premium's mean, or
        else the fixed premium (0 when none is set).
        """
    @property
    def expected_profit_commission(self, /) -> list[float]:
        """
        Expected profit commission of each layer.
        """
    @property
    def expected_reinstatement_premium(self, /) -> list[float]:
        """
        Expected reinstatement premium of each layer; 0 without paid
        reinstatements.
        """
    @property
    def gross(self, /) -> Grid:
        """
        Annual gross loss.
        """
    @property
    def gross_report(self, /) -> CompoundReport:
        """
        The compound calculation behind ``gross``.
        """
    @property
    def net(self, /) -> Grid |None:
        """
        Annual net loss, or ``None`` when it is not a single compound total.
        """
    @property
    def on_points(self, /) -> bool:
        """
        Whether every boundary and net loss fell on a grid point. When
        ``False``, means are still exact but shapes are smeared by up to a
        step.
        """
    def price(self, /, distortion: Distortion) -> TowerPrice:
        """
        Each view priced on its own by a distortion: the stand-alone price
        of the gross, of every layer's cession and of the net, as
        ``aggregate``'s ``reins_price_df``.
        
        Parameters
        ----------
        distortion : Distortion
        
        Returns
        -------
        TowerPrice
        
        Examples
        --------
        >>> from prospicio.reinsurance import Layer, Tower
        >>> from prospicio.distributions import Grid, Poisson
        >>> from prospicio.risk import Distortion
        >>> sev = Grid(1.0, [0.0, 0.4, 0.3, 0.2, 0.1])
        >>> r = Tower([Layer("2x2", 2.0, 2.0)]).on_grid(Poisson(3.0), sev, 200)
        >>> p = r.price(Distortion.proportional_hazard(0.7))
        >>> round(p.ceded[0].loss, 12), p.ceded[0].premium > p.ceded[0].loss
        (1.2, True)
        """
    @property
    def stage_ceded(self, /) -> list[Grid |None]:
        """
        Annual ceded loss of each stage, its layers together: the total
        cession ``aggregate`` reports as ``Ceded``. ``None`` for a stage of
        several layers one of which has annual terms.
        """

@final
class TowerModel:
    """
    A frequency and a piecewise Pareto severity that reproduce a tower,
    a PML curve or a set of references.
    """
    def __repr__(self, /) -> str: ...
    def excess_frequency(self, /, x: float) -> float:
        """
        Expected number of losses a year above ``x``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @property
    def frequency(self, /) -> float:
        """
        Expected number of losses a year above the lowest threshold.
        """
    def layer_loss(self, /, limit: float, attachment: float) -> float:
        """
        Expected loss a year to ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    @property
    def severity(self, /) -> PiecewisePareto:
        """
        The fitted severity.
        """

@final
class TowerPrice:
    """
    A tower's views priced by a distortion, from ``TowerGrids.price``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def ceded(self, /) -> list[ViewPrice]:
        """
        Each layer's cession, in tower order.
        """
    @property
    def gross(self, /) -> ViewPrice:
        """
        The gross.
        """
    @property
    def net(self, /) -> ViewPrice |None:
        """
        The net, or ``None`` when the tower has no net grid.
        """
    @property
    def stages(self, /) -> list[ViewPrice |None]:
        """
        Each stage's total cession, ``None`` where ``stage_ceded`` is.
        """

@final
class TrendFit:
    """
    A trend fitted by least squares to values at times in years:
    ``log(y)`` on time for an exponential trend, ``y`` for a linear one. For
    quarterly points, pass times a quarter apart.
    
    Build one with ``TrendFit.exponential`` or ``TrendFit.linear``.
    
    Examples
    --------
    >>> from prospicio.pricing import TrendFit
    >>> t = [0.0, 0.25, 0.5, 0.75]
    >>> fit = TrendFit.exponential(t, [100 * 1.03 ** x for x in t])
    >>> round(fit.annual_change, 6)
    0.03
    """
    def __repr__(self, /) -> str: ...
    @property
    def annual_change(self, /) -> float:
        """
        The annual trend: ``exp(slope) - 1`` for an exponential fit, the
        slope (an amount a year) for a linear one.
        """
    @staticmethod
    def exponential(times: Sequence[float], values: Sequence[float]) -> TrendFit:
        """
        Fit ``log(y) = a + b t``; every value must be positive.
        
        Parameters
        ----------
        times : list of float
        values : list of float
        
        Returns
        -------
        TrendFit
        """
    def fitted(self, /, time: float) -> float:
        """
        The fitted value at ``time``.
        
        Parameters
        ----------
        time : float
        
        Returns
        -------
        float
        """
    @property
    def intercept(self, /) -> float:
        """
        The intercept, on the log scale for an exponential fit.
        """
    @property
    def kind(self, /) -> str:
        """
        ``"exponential"`` or ``"linear"``.
        """
    @staticmethod
    def linear(times: Sequence[float], values: Sequence[float]) -> TrendFit:
        """
        Fit ``y = a + b t``.
        
        Parameters
        ----------
        times : list of float
        values : list of float
        
        Returns
        -------
        TrendFit
        """
    @property
    def n(self, /) -> int:
        """
        The number of points.
        """
    @property
    def r_squared(self, /) -> float:
        """
        The coefficient of determination, on the fitted scale.
        """
    @property
    def slope(self, /) -> float:
        """
        The slope per year, on the log scale for an exponential fit.
        """

@final
class Triangle:
    """
    A loss triangle with four axes: index (segment), column (measure), origin
    and development age, in chainladder-python's order.
    
    Segments are named by key columns such as ``"lob"`` and ``"state"``:
    ``keys`` gives their names and ``index`` one label per segment. A
    triangle without keys has one segment, ``"Total"``.
    
    Build one from a long table with ``from_long`` or ``from_frame``. Ages
    are whole months from the start of the origin period, so age 12 on a
    2021 accident year is valued at December 2021. Cells that were not
    observed are ``nan`` in ``values``; an observed zero stays zero.
    
    Examples
    --------
    >>> from prospicio.reserving import Triangle
    >>> tri = Triangle.from_long(
    ...     origin=[2020, 2020, 2021],
    ...     development=[12, 24, 12],
    ...     values={"paid": [100.0, 150.0, 110.0]},
    ... )
    >>> tri.shape
    (1, 1, 2, 2)
    >>> tri.origins, tri.development, tri.valuation
    (['2020', '2021'], [12, 24], datetime.date(2021, 12, 31))
    >>> tri.values[0][0]
    [[100.0, 150.0], [110.0, nan]]
    """
    def __eq__(self, other: object, /) -> bool: ...
    def __repr__(self, /) -> str: ...
    def _repr_html_(self, /) -> str: ...
    @property
    def columns(self, /) -> list[str]:
        """
        Measure column names.
        """
    @property
    def development(self, /) -> list[int]:
        """
        Development ages in months, youngest first.
        """
    @property
    def development_grain(self, /) -> str:
        """
        Development grain: ``"Y"``, ``"S"``, ``"Q"`` or ``"M"``.
        """
    @staticmethod
    def from_frame(data: Any, origin: str, development: str, columns: Any, keys: Any |None = None, origin_grain: str = "Y", development_grain: str = "Y", cumulative: bool = True, development_is_valuation: bool = False) -> Triangle:
        """
        Builds a triangle from a data frame in long format.
        
        Columns are looked up with ``data[name]``, so a pandas or Polars
        DataFrame works, as does a dict of columns.
        
        Parameters
        ----------
        data : DataFrame or dict
        origin : str
            Name of the origin column (dates or integer years).
        development : str
            Name of the development column (ages in months, or valuation
            dates when ``development_is_valuation`` is true).
        columns : str or list of str
            Names of the measure columns.
        keys : str or list of str, optional
            Names of the key columns, such as ``["lob", "state"]``. By
            default every row is in one segment, ``"Total"``.
        origin_grain : {"Y", "S", "Q", "M"}, default "Y"
        development_grain : {"Y", "S", "Q", "M"}, default "Y"
        cumulative : bool, default True
        development_is_valuation : bool, default False
        
        Returns
        -------
        Triangle
        
        Raises
        ------
        ValueError
            As for ``from_long``.
        
        Examples
        --------
        >>> from prospicio.reserving import Triangle
        >>> df = {
        ...     "lob": ["Auto", "Auto", "Auto", "Home"],
        ...     "year": [2020, 2020, 2021, 2020],
        ...     "age": [12, 24, 12, 12],
        ...     "paid": [100.0, 150.0, 110.0, 50.0],
        ... }
        >>> tri = Triangle.from_frame(df, "year", "age", "paid", keys="lob")
        >>> tri.keys, tri.index, tri.shape
        (['lob'], ['Auto', 'Home'], (2, 1, 2, 2))
        """
    @staticmethod
    def from_long(origin: Any, development: Any, values: Any, keys: Any |None = None, origin_grain: str = "Y", development_grain: str = "Y", cumulative: bool = True, development_is_valuation: bool = False) -> Triangle:
        """
        Builds a triangle from the columns of a long table, one row per
        (keys, origin, development).
        
        Origins span every period from the earliest to the latest row and
        ages every development period from the youngest to the oldest. Rows
        with the same (keys, origin, age) are summed; ``nan`` values are
        missing. Incremental input treats a missing row as a period without
        movement, as chainladder-python does.
        
        Parameters
        ----------
        origin : array-like
            Any date in each row's origin period (numpy ``datetime64``,
            ``datetime.date``, pandas ``Timestamp``), or integer years.
        development : array-like
            Development age of each row in months (12, 24, ...), or its
            valuation date when ``development_is_valuation`` is true.
        values : dict of str to array-like, or array-like
            Measure columns by name. A single array-like is one column named
            ``"values"``.
        keys : dict of str to array-like, optional
            Key columns by name, such as ``{"lob": [...], "state": [...]}``,
            in key order. Values are stored as strings (``str()`` of each);
            ``None`` and ``nan`` are not allowed. Each distinct combination is
            a segment. By default every row is in one segment, ``"Total"``.
        origin_grain : {"Y", "S", "Q", "M"}, default "Y"
            Length of an origin period.
        development_grain : {"Y", "S", "Q", "M"}, default "Y"
            Spacing of development ages; must divide the origin grain.
        cumulative : bool, default True
            Whether the values are cumulative (otherwise incremental).
        development_is_valuation : bool, default False
            Whether ``development`` holds valuation dates instead of ages.
        
        Returns
        -------
        Triangle
        
        Raises
        ------
        ValueError
            If columns differ in length, an age is not on the development
            grid, a value is infinite, the grains are incompatible, a key name
            is repeated or also a value column, or a key has missing values.
        
        Examples
        --------
        >>> from prospicio.reserving import Triangle
        >>> tri = Triangle.from_long(
        ...     origin=[2020, 2020, 2021, 2020],
        ...     development=[12, 24, 12, 12],
        ...     values={"paid": [100.0, 150.0, 110.0, 50.0]},
        ...     keys={"lob": ["Auto", "Auto", "Auto", "Home"], "state": ["CA", "CA", "CA", "NY"]},
        ... )
        >>> tri.keys, tri.index
        (['lob', 'state'], [('Auto', 'CA'), ('Home', 'NY')])
        
        Valuation dates instead of ages:
        
        >>> import datetime
        >>> from prospicio.reserving import Triangle
        >>> d = datetime.date
        >>> tri = Triangle.from_long(
        ...     origin=[d(2021, 2, 1), d(2021, 2, 1), d(2021, 5, 1)],
        ...     development=[d(2021, 3, 31), d(2021, 6, 30), d(2021, 6, 30)],
        ...     values=[10.0, 25.0, 7.0],
        ...     origin_grain="Q",
        ...     development_grain="Q",
        ...     development_is_valuation=True,
        ... )
        >>> tri.origins, tri.development
        (['2021Q1', '2021Q2'], [3, 6])
        """
    def grain(self, /, origin_grain: str, development_grain: str |None = None) -> Triangle:
        """
        The triangle at a coarser origin and/or development grain.
        
        Parameters
        ----------
        origin_grain : {"Y", "S", "Q", "M"}
        development_grain : {"Y", "S", "Q", "M"}, optional
            By default ``origin_grain``.
        
        Returns
        -------
        Triangle
        
        Raises
        ------
        ValueError
            If a grain is finer than the current one, or the development
            grain does not divide the origin grain.
        
        Examples
        --------
        >>> from prospicio.reserving import Triangle
        >>> q = Triangle.from_long(
        ...     [2020, 2020], [3, 6], [1.0, 2.0], origin_grain="Q", development_grain="Q"
        ... )
        >>> y = q.grain("Y")
        >>> y.origins, y.development_grain
        (['2020'], 'Y')
        """
    def group_by(self, /, keys: Any) -> Triangle:
        """
        Sums the segments that share the values of ``keys``, dropping the
        other keys.
        
        Cumulative values are summed cell by cell, and a cell is observed if
        any segment in the group observes it. An incremental triangle is
        summed as cumulative values and returned incremental.
        
        Parameters
        ----------
        keys : str or list of str
            The keys to keep, in the order the result has them. ``[]`` sums
            every segment into one, labelled ``"Total"``.
        
        Returns
        -------
        Triangle
        
        Raises
        ------
        ValueError
            If a key is unknown or named twice.
        
        Examples
        --------
        >>> from prospicio.reserving import Triangle
        >>> tri = Triangle.from_long(
        ...     [2020, 2020, 2020],
        ...     [12, 12, 12],
        ...     {"paid": [1.0, 2.0, 3.0]},
        ...     keys={"lob": ["Auto", "Auto", "Home"], "state": ["CA", "NY", "NY"]},
        ... )
        >>> by_lob = tri.group_by("lob")
        >>> by_lob.index, by_lob.to_long()["paid"]
        (['Auto', 'Home'], [3.0, 3.0])
        >>> tri.group_by([]).to_long()["paid"]
        [6.0]
        """
    @property
    def index(self, /) -> list[Any]:
        """
        Segment labels: a str each with one key, a tuple of key values with
        several, and ``["Total"]`` without keys.
        """
    @property
    def is_cumulative(self, /) -> bool:
        """
        Whether the values are cumulative (otherwise incremental).
        """
    @property
    def keys(self, /) -> list[str]:
        """
        Names of the key columns, in key order; empty without keys.
        """
    def latest_diagonal(self, /) -> list[list[list[float]]]:
        """
        The latest observed value of each origin, as nested lists indexed
        ``[index][column][origin]``, ``nan`` for an origin with no value.
        
        Returns
        -------
        list of list of list of float
        """
    def link_ratios(self, /) -> Triangle:
        """
        Age-to-age link ratios of the cumulative values. Development
        position ``d`` holds the ratio from age ``d`` to age ``d + 1``,
        observed where both ages are observed and the earlier value is not
        zero.
        
        Returns
        -------
        Triangle
        """
    @property
    def origin_grain(self, /) -> str:
        """
        Origin grain: ``"Y"``, ``"S"``, ``"Q"`` or ``"M"``.
        """
    @property
    def origins(self, /) -> list[str]:
        """
        Origin periods, oldest first: ``"2021"``, ``"2021H1"``,
        ``"2021Q3"`` or ``"2021-07"`` by grain.
        """
    def select(self, /, columns: Any |None = None, **keys) -> Triangle:
        """
        The segments whose key values match, and the measure columns named.
        
        Each keyword names a key and gives one value or a list of values to
        keep; segments must match every keyword, and keep their order.
        Values are compared as strings, as keys are stored (``str()`` of a
        value). A key named ``columns`` cannot be selected this way.
        
        Parameters
        ----------
        columns : str or list of str, optional
            Measure columns to keep, in this order. By default every column.
        **keys : value or list of values
            For example ``lob="Auto"`` or ``state=["CA", "NY"]``; any
            iterable that is not a string (a tuple, set, NumPy array or
            pandas Series) is a list of values.
        
        Returns
        -------
        Triangle
        
        Raises
        ------
        ValueError
            If a key, value or column is unknown or given twice, a list of
            values is empty, or no segment matches.
        
        Examples
        --------
        >>> from prospicio.reserving import Triangle
        >>> tri = Triangle.from_long(
        ...     [2020, 2020, 2020],
        ...     [12, 12, 12],
        ...     {"paid": [1.0, 2.0, 3.0], "incurred": [2.0, 3.0, 4.0]},
        ...     keys={"lob": ["Auto", "Auto", "Home"], "state": ["CA", "NY", "NY"]},
        ... )
        >>> tri.select(state="NY").index
        [('Auto', 'NY'), ('Home', 'NY')]
        >>> tri.select(lob="Auto", state=["CA", "NY"], columns="paid").shape
        (2, 1, 1, 1)
        """
    @property
    def shape(self, /) -> tuple[int, int, int, int]:
        """
        Axis lengths: ``(index, column, origin, development)``.
        """
    def summary(self, /) -> Any:
        """
        One row per segment and measure: the key values, ``column``,
        ``n_origins`` (origins with an observed value), ``first_origin`` and
        ``last_origin`` of those, ``valuation`` (the last day of the latest
        valuation with an observed value), ``latest`` (the sum over origins of
        the latest cumulative value, so for an incremental triangle the sum
        of every increment) and ``cumulative``. An origin or valuation is
        missing (``None``, which pandas may show as ``NaN``) when the segment
        has no observed value of the measure.
        
        Returns
        -------
        pandas.DataFrame or dict
            A DataFrame with pandas installed, a dict of lists otherwise.
        
        Raises
        ------
        ValueError
            If a key has the name of one of the summary's columns.
        
        Examples
        --------
        >>> from prospicio.reserving import Triangle
        >>> tri = Triangle.from_long(
        ...     [2020, 2020, 2021, 2020],
        ...     [12, 24, 12, 12],
        ...     {"paid": [100.0, 150.0, 110.0, 50.0]},
        ...     keys={"lob": ["Auto", "Auto", "Auto", "Home"]},
        ... )
        >>> s = tri.summary()
        >>> s["lob"].tolist(), s["n_origins"].tolist(), s["latest"].tolist()
        (['Auto', 'Home'], [2, 1], [260.0, 50.0])
        """
    def to_cumulative(self, /) -> Triangle:
        """
        Cumulative values: running sums of the observed increments.
        
        Returns
        -------
        Triangle
        """
    def to_frame(self, /) -> Any:
        """
        The long table of ``to_long`` as a pandas DataFrame. Needs pandas.
        
        Returns
        -------
        pandas.DataFrame
        """
    def to_incremental(self, /) -> Triangle:
        """
        Incremental values: each observed value minus the previous observed
        value in its row.
        
        Returns
        -------
        Triangle
        """
    def to_long(self, /) -> dict:
        """
        The triangle as a long table: a dict of equal-length lists with one
        entry per key column (by name), ``"origin"`` (start of the origin
        period, a ``datetime.date``), ``"development"`` (age in months) and
        one per measure column, with a row per (segment, origin, age) that
        has an observed measure. It feeds back into ``from_frame`` (with
        ``keys=tri.keys``) or ``pandas.DataFrame``.
        
        Returns
        -------
        dict of str to list
        
        Raises
        ------
        ValueError
            If a key or measure column is named ``origin`` or
            ``development``.
        
        Examples
        --------
        >>> from prospicio.reserving import Triangle
        >>> tri = Triangle.from_long(
        ...     [2020, 2020], [12, 12], {"paid": [1.0, 2.0]}, keys={"lob": ["Auto", "Home"]}
        ... )
        >>> long = tri.to_long()
        >>> list(long), long["lob"]
        (['lob', 'origin', 'development', 'paid'], ['Auto', 'Home'])
        """
    def to_string(self, /, max_rows: int = ..., max_cols: int = ...) -> str:
        """
        The printout as text: the origin × development grid for a triangle
        with one segment and one measure (as ``view``), otherwise the
        ``summary`` table. Numbers are rounded for reading; ``view`` and
        ``summary`` give exact values.
        
        Parameters
        ----------
        max_rows : int, default 20
            Rows shown before the middle ones are left out; 0 for no limit.
        max_cols : int, default 12
            Development ages shown before the middle ones are left out; 0
            for no limit.
        
        Returns
        -------
        str
        
        Examples
        --------
        >>> from prospicio.reserving import Triangle
        >>> tri = Triangle.from_long([2020, 2020, 2021], [12, 24, 12], {"paid": [1000.0, 1500.0, 1100.0]})
        >>> print(tri.to_string())
        Triangle: paid (cumulative, valuation 2021-12)
                 12     24
        2020  1,000  1,500
        2021  1,100
        """
    @property
    def valuation(self, /) -> date:
        """
        Valuation date of the latest diagonal: the last day of its month,
        as a ``datetime.date``.
        """
    @property
    def values(self, /) -> list[list[list[list[float]]]]:
        """
        Values as nested lists indexed ``[index][column][origin][development]``,
        ``nan`` where unobserved. ``numpy.asarray`` gives the 4-D array.
        """
    def view(self, /, column: str |None = None, **keys) -> Any:
        """
        One segment and measure as an origin × development table.
        
        Parameters
        ----------
        column : str, optional
            The measure; may be left out when the triangle has one column.
        **keys : value
            One value per key, such as ``lob="Auto"``, compared as ``str()``
            of it. Keys not named may take any value, but the choice must
            leave one segment; a triangle with one segment needs none. A key
            named ``column`` cannot be chosen this way (``select`` it first).
        
        Returns
        -------
        pandas.DataFrame or dict
            With pandas installed, a DataFrame with the origin labels as its
            index (named ``origin``), the ages in months as its columns
            (named ``development``) and ``nan`` where a cell is not observed.
            Without pandas, a dict of lists as the other tables of this
            module: ``"origin"``, then one list per age keyed by the age.
        
        Raises
        ------
        ValueError
            If a key, value or column is unknown, the keys match several
            segments, or the column is left out and there are several.
        
        Examples
        --------
        >>> from prospicio.reserving import Triangle
        >>> tri = Triangle.from_long(
        ...     [2020, 2020, 2021, 2020],
        ...     [12, 24, 12, 12],
        ...     {"paid": [100.0, 150.0, 110.0, 50.0]},
        ...     keys={"lob": ["Auto", "Auto", "Auto", "Home"]},
        ... )
        >>> v = tri.view(lob="Auto")
        >>> list(v.index), list(v.columns), float(v.loc["2021", 12])
        (['2020', '2021'], [12, 24], 110.0)
        """

@final
class Truncated:
    """
    A severity conditioned on ``lower < X <= upper``, as ``aggregate``'s
    ``sev_lb`` and ``sev_ub``: its distribution function is
    ``(F(x) - F(lower)) / (F(upper) - F(lower))`` on the window. Layer
    moments are exact, from the inner severity's.
    
    Parameters
    ----------
    severity : a severity
        Any distribution but ``Sampled``.
    lower : float, default 0.0
    upper : float, default inf
    
    Raises
    ------
    ValueError
        If ``0 <= lower < upper`` fails or the window has no probability.
    
    Examples
    --------
    >>> from prospicio.distributions import Lognormal, Truncated
    >>> t = Truncated(Lognormal.from_mean_cv(1000.0, 1.0), upper=5000.0)
    >>> t.cdf(5000.0), t.mean() < 1000.0
    (1.0, True)
    """
    def __new__(cls, /, severity: Any, lower: float = 0.0, upper: float = ...) -> Truncated: ...
    def __repr__(self, /) -> str: ...
    def cdf(self, /, x: float) -> float:
        """
        Distribution function ``P(X <= x)``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    def layer(self, /, limit: float, attachment: float) -> float:
        """
        Expected loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
            ``inf`` for an unlimited layer.
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_second_moment(self, /, limit: float, attachment: float) -> float:
        """
        Second moment of the loss to the layer ``limit`` xs
        ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_variance(self, /, limit: float, attachment: float) -> float:
        """
        Variance of the loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def lev(self, /, limit: float) -> float:
        """
        Limited expected value ``E[min(X, limit)]``.
        
        Parameters
        ----------
        limit : float
        
        Returns
        -------
        float
        """
    @property
    def lower(self, /) -> float:
        """
        Lower end of the window (excluded).
        """
    def mean(self, /) -> float:
        """
        Mean of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """
    @property
    def probability(self, /) -> float:
        """
        ``P(lower < X <= upper)`` under the inner severity.
        """
    def quantile(self, /, p: float) -> float:
        """
        Quantile: the smallest ``x`` with ``P(X <= x) >= p``.
        
        Parameters
        ----------
        p : float
            Probability in ``[0, 1]``.
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def sample(self, /, n: int, seed: int, stream: int = 0) -> list[float]:
        """
        ``n`` draws from stream ``stream`` of the generator keyed by
        ``seed``.
        
        Parameters
        ----------
        n : int
        seed : int
        stream : int, default 0
        
        Returns
        -------
        list of float
        """
    @property
    def severity(self, /) -> Any:
        """
        The severity before conditioning.
        """
    def std(self, /) -> float:
        """
        Standard deviation of the distribution.
        
        Returns
        -------
        float
        """
    def stop_loss(self, /, retention: float) -> float:
        """
        Expected excess over a retention, ``E[max(X - retention, 0)]``.
        
        Parameters
        ----------
        retention : float
        
        Returns
        -------
        float
        """
    def survival(self, /, x: float) -> float:
        """
        Survival function ``P(X > x)``, accurate far into the tail.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @property
    def upper(self, /) -> float:
        """
        Upper end of the window (included).
        """
    def variance(self, /) -> float:
        """
        Variance of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """

@final
class Tweedie:
    """
    Tweedie distribution with mean ``mu``, dispersion ``phi`` and power
    ``1 < p < 2``: variance ``phi * mu**p``, a point mass at 0 and a
    continuous density above it.
    
    It is a Poisson number of gamma losses, the GLM family for pure
    premium. ``P(Y = 0) = exp(-lambda_)``.
    
    Parameters
    ----------
    mean : float
    dispersion : float
    power : float
        In ``(1, 2)``.
    
    Raises
    ------
    ValueError
        If a parameter is out of range.
    
    Examples
    --------
    >>> import math
    >>> from prospicio.distributions import Tweedie
    >>> y = Tweedie(500.0, 40.0, 1.6)
    >>> abs(y.cdf(0.0) - math.exp(-y.lambda_)) < 1e-15
    True
    """
    def __getnewargs__(self, /) -> tuple[float, float, float]: ...
    def __new__(cls, /, mean: float, dispersion: float, power: float) -> Tweedie: ...
    def __repr__(self, /) -> str: ...
    def cdf(self, /, x: float) -> float:
        """
        Distribution function ``P(X <= x)``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    @property
    def dispersion(self, /) -> float:
        """
        Dispersion ``phi``.
        """
    @staticmethod
    def from_poisson_gamma(lambda_: float, shape: float, scale: float) -> Tweedie:
        """
        The Tweedie equal to a Poisson(``lambda_``) number of
        Gamma(``shape``, ``scale``) losses.
        
        Parameters
        ----------
        lambda_ : float
        shape : float
        scale : float
        
        Returns
        -------
        Tweedie
        """
    @property
    def lambda_(self, /) -> float:
        """
        Poisson mean of the number of losses.
        """
    def layer(self, /, limit: float, attachment: float) -> float:
        """
        Expected loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
            ``inf`` for an unlimited layer.
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_second_moment(self, /, limit: float, attachment: float) -> float:
        """
        Second moment of the loss to the layer ``limit`` xs
        ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_variance(self, /, limit: float, attachment: float) -> float:
        """
        Variance of the loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def lev(self, /, limit: float) -> float:
        """
        Limited expected value ``E[min(X, limit)]``.
        
        Parameters
        ----------
        limit : float
        
        Returns
        -------
        float
        """
    def ln_pdf(self, /, y: float) -> float:
        """
        Log density at ``y > 0``; at ``y = 0``, the log of the point mass.
        
        Parameters
        ----------
        y : float
        
        Returns
        -------
        float
        """
    def mean(self, /) -> float:
        """
        Mean of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """
    @property
    def power(self, /) -> float:
        """
        Power ``p``.
        """
    def quantile(self, /, p: float) -> float:
        """
        Quantile: the smallest ``x`` with ``P(X <= x) >= p``.
        
        Parameters
        ----------
        p : float
            Probability in ``[0, 1]``.
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def sample(self, /, n: int, seed: int, stream: int = 0) -> list[float]:
        """
        ``n`` draws from stream ``stream`` of the generator keyed by
        ``seed``.
        
        Parameters
        ----------
        n : int
        seed : int
        stream : int, default 0
        
        Returns
        -------
        list of float
        """
    @property
    def severity(self, /) -> Gamma:
        """
        The gamma distribution of each loss.
        """
    def std(self, /) -> float:
        """
        Standard deviation of the distribution.
        
        Returns
        -------
        float
        """
    def stop_loss(self, /, retention: float) -> float:
        """
        Expected excess over a retention, ``E[max(X - retention, 0)]``.
        
        Parameters
        ----------
        retention : float
        
        Returns
        -------
        float
        """
    def survival(self, /, x: float) -> float:
        """
        Survival function ``P(X > x)``, accurate far into the tail.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    def variance(self, /) -> float:
        """
        Variance of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """

@final
class ViewPrice:
    """
    One view's price under a distortion, from ``TowerGrids.price``: the
    expected loss ``loss``, the premium, the margin ``premium - loss`` and
    the loss ratio ``loss / premium``.
    """
    def __repr__(self, /) -> str: ...
    @property
    def loss(self, /) -> float:
        """
        The expected loss.
        """
    @property
    def loss_ratio(self, /) -> float:
        """
        ``loss / premium``.
        """
    @property
    def margin(self, /) -> float:
        """
        ``premium - loss``.
        """
    @property
    def premium(self, /) -> float:
        """
        The distorted expectation.
        """

@final
class Weibull:
    """
    Weibull distribution with shape ``k`` and scale ``lam``:
    ``P(X > x) = exp(-(x / lam) ** k)``, as SciPy's ``weibull_min``.
    
    Parameters
    ----------
    shape : float
    scale : float
    
    Examples
    --------
    >>> from prospicio.distributions import Weibull
    >>> Weibull(1.0, 2.0).mean()
    2.0
    """
    def __getnewargs__(self, /) -> tuple[float, float]: ...
    def __new__(cls, /, shape: float, scale: float) -> Weibull: ...
    def __repr__(self, /) -> str: ...
    def cdf(self, /, x: float) -> float:
        """
        Distribution function ``P(X <= x)``.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    def layer(self, /, limit: float, attachment: float) -> float:
        """
        Expected loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
            ``inf`` for an unlimited layer.
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_second_moment(self, /, limit: float, attachment: float) -> float:
        """
        Second moment of the loss to the layer ``limit`` xs
        ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def layer_variance(self, /, limit: float, attachment: float) -> float:
        """
        Variance of the loss to the layer ``limit`` xs ``attachment``.
        
        Parameters
        ----------
        limit : float
        attachment : float
        
        Returns
        -------
        float
        """
    def lev(self, /, limit: float) -> float:
        """
        Limited expected value ``E[min(X, limit)]``.
        
        Parameters
        ----------
        limit : float
        
        Returns
        -------
        float
        """
    def mean(self, /) -> float:
        """
        Mean of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """
    def quantile(self, /, p: float) -> float:
        """
        Quantile: the smallest ``x`` with ``P(X <= x) >= p``.
        
        Parameters
        ----------
        p : float
            Probability in ``[0, 1]``.
        
        Returns
        -------
        float
        
        Raises
        ------
        ValueError
            If ``p`` is outside ``[0, 1]``.
        """
    def sample(self, /, n: int, seed: int, stream: int = 0) -> list[float]:
        """
        ``n`` draws from stream ``stream`` of the generator keyed by
        ``seed``.
        
        Parameters
        ----------
        n : int
        seed : int
        stream : int, default 0
        
        Returns
        -------
        list of float
        """
    @property
    def scale(self, /) -> float:
        """
        Scale ``lam``.
        """
    @property
    def shape(self, /) -> float:
        """
        Shape ``k``.
        """
    def std(self, /) -> float:
        """
        Standard deviation of the distribution.
        
        Returns
        -------
        float
        """
    def stop_loss(self, /, retention: float) -> float:
        """
        Expected excess over a retention, ``E[max(X - retention, 0)]``.
        
        Parameters
        ----------
        retention : float
        
        Returns
        -------
        float
        """
    def survival(self, /, x: float) -> float:
        """
        Survival function ``P(X > x)``, accurate far into the tail.
        
        Parameters
        ----------
        x : float
        
        Returns
        -------
        float
        """
    def variance(self, /) -> float:
        """
        Variance of the distribution (``inf`` if it does not exist).
        
        Returns
        -------
        float
        """

def actual_vs_expected(periods: Sequence[int |str], family: str, y: Sequence[float], mu: Sequence[float], weights: Sequence[float] |None = None, dispersion: float = 1.0, theta: float |None = None, power: float |None = None) -> dict:
    """
    Actual against expected by period for a stored model's predictions on
    new data, with each period's z-score under the model and a test for
    drift.
    
    ``A = sum(w * y)`` and ``E = sum(w * mu)`` per period; the z-score is
    ``(A - E) / sqrt(dispersion * sum(w * V(mu)))`` with the family's
    variance function ``V``, about standard normal while the model holds.
    ``trend`` is the slope of ``A / E - 1`` per period step (periods in
    sorted order), weighted by each period's precision.
    
    Parameters
    ----------
    periods : list of int or str
        One period label per row.
    family : str
    y : list of float
        Actuals.
    mu : list of float
        The model's predicted means.
    weights : list of float, optional
        Prior weights as fitted (exposure for a rate); leave out for counts
        with exposure in the offset.
    dispersion : float, default 1.0
    theta, power : float, optional
        Negative binomial ``theta``, Tweedie ``power``.
    
    Returns
    -------
    dict
        ``periods`` (a list of dicts with ``period``, ``n``, ``weight``,
        ``actual``, ``expected``, ``ratio``, ``std_dev`` and ``z``),
        ``total`` (the same without ``period``), ``trend``,
        ``trend_std_error`` and ``trend_z``.
    
    Examples
    --------
    >>> from prospicio.models import actual_vs_expected
    >>> m = actual_vs_expected([2023, 2023, 2024, 2024], "poisson",
    ...                        [1.0, 3.0, 2.0, 6.0], [2.0, 2.0, 2.0, 2.0])
    >>> m["periods"][1]["ratio"], m["periods"][1]["z"]
    (2.0, 2.0)
    """

def aggregate_layer_statistics(annual: Grid, share: float, limit: float, attachment: float) -> AggregateLayerStatistics:
    """
    The statistics of ``share`` of ``limit`` xs ``attachment`` on an annual
    total, an aggregate cover such as a stop loss, as ``aggregate``'s
    ``reins_stats_df`` for its ``agg`` layers: the chances of reaching and
    exhausting the cover, the loss on line and the ceded loss's first three
    cumulants, exact for the grid.
    
    Parameters
    ----------
    annual : Grid
        The annual total the cover sees: a compound grid, or a tower's net.
    share : float
    limit : float
        May be ``inf``.
    attachment : float
    
    Returns
    -------
    AggregateLayerStatistics
    
    Examples
    --------
    >>> from prospicio.reinsurance import aggregate_layer_statistics
    >>> from prospicio.distributions import Grid
    >>> s = aggregate_layer_statistics(Grid(1.0, [0.2, 0.3, 0.3, 0.2]), 1.0, 1.0, 1.0)
    >>> round(s.pr_attach, 12), round(s.aggregate.mean, 12)
    (0.5, 0.5)
    """

def allocate(pd: PredictiveDistribution, distortion: Distortion) -> list[float]:
    """
    Allocates a distortion risk measure of the total to the components.
    
    Euler allocation by co-measure: simulations are ranked by their total
    and each component gets the distortion-weighted sum of its own draws.
    The contributions sum to ``distortion.measure(pd)``; for
    ``Distortion.tvar(p)`` they are the CoTVaRs. Components must add up to
    the portfolio being allocated.
    
    Parameters
    ----------
    pd : PredictiveDistribution
    distortion : Distortion
    
    Returns
    -------
    list of float
        One contribution per component, in ``pd.components()`` order.
    
    Examples
    --------
    >>> from prospicio.distributions import PredictiveDistribution
    >>> from prospicio.risk import Distortion, allocate
    >>> pd = PredictiveDistribution(["lob"], [("motor",), ("property",)],
    ...                             [[1.0, 2.0], [4.0, 1.0], [2.0, 5.0], [3.0, 6.0]])
    >>> allocate(pd, Distortion.tvar(0.5))
    [2.5, 5.5]
    """

def alpha_between_frequencies(threshold_1: float, frequency_1: float, threshold_2: float, frequency_2: float, truncation: float |None = None) -> float:
    """
    The Pareto alpha between two excess frequencies.
    
    Parameters
    ----------
    threshold_1 : float
    frequency_1 : float
    threshold_2 : float
    frequency_2 : float
    truncation : float, optional
    
    Returns
    -------
    float
    
    Examples
    --------
    >>> from prospicio.pricing import alpha_between_frequencies
    >>> round(alpha_between_frequencies(1e6, 4.0, 2e6, 1.0), 12)
    2.0
    """

def alpha_between_frequency_and_layer(threshold: float, frequency: float, limit: float, attachment: float, expected_loss: float, truncation: float |None = None) -> float:
    """
    The Pareto alpha at which ``frequency`` losses a year above
    ``threshold`` give the layer an expected loss of ``expected_loss``.
    
    Parameters
    ----------
    threshold : float
    frequency : float
    limit : float
    attachment : float
    expected_loss : float
    truncation : float, optional
    
    Returns
    -------
    float
    """

def alpha_between_layers(a: tuple[float, float, float], b: tuple[float, float, float], truncation: float |None = None) -> float:
    """
    The Pareto alpha at which two layers have the given expected losses.
    
    Parameters
    ----------
    a : tuple of float
        ``(limit, attachment, expected_loss)``.
    b : tuple of float
        ``(limit, attachment, expected_loss)``; one layer must lie above
        the other.
    truncation : float, optional
    
    Returns
    -------
    float
    """

def buhlmann_straub(ratios: Sequence[Sequence[float]], weights: Sequence[Sequence[float]] |None = None, estimator: str = "unbiased", tol: float = 1.4901161193847656e-8, max_iter: int = 100) -> BuhlmannStraubFit:
    """
    The Bühlmann–Straub credibility model, as R's ``actuar::cm(~group)``:
    group ``i`` in period ``j`` has ratio ``X_ij`` (loss per unit of
    exposure) with weight ``w_ij``. Without weights it is Bühlmann's model.
    
    Parameters
    ----------
    ratios : list of list of float
        One row per group, one column per period; NaN for a missing cell.
    weights : list of list of float, optional
        The same shape, NaN in the same places; equal weights if omitted.
    estimator : {"unbiased", "iterative"}, default "unbiased"
        The between variance: Bühlmann and Straub's unbiased estimator
        (``actuar``'s ``"Buhlmann-Gisler"`` and ``"Ohlsson"``), or Bichsel
        and Straub's iterative one.
    tol : float, default 1.49e-8
        Relative tolerance of the iterative estimator.
    max_iter : int, default 100
    
    Returns
    -------
    BuhlmannStraubFit
    
    Raises
    ------
    ValueError
        For fewer than two groups, too few observations, mismatched shapes
        or missing cells, or missing ratios without weights.
    
    Examples
    --------
    >>> from prospicio.credibility import buhlmann_straub
    >>> fit = buhlmann_straub([[1.0, 2.0, 3.0], [4.0, 5.0, 9.0]])
    >>> fit.within_variance
    4.0
    """

def calibrate(family: str, dist: Any, premium: float, assets: float |None = None, r0: float = 0.0) -> Distortion:
    """
    The member of a distortion family whose price of ``dist`` is ``premium``.
    
    Parameters
    ----------
    family : str
        ``"ccoc"``, ``"ph"``, ``"wang"``, ``"dual"``, ``"tvar"``, ``"exp"``,
        ``"clin"``, ``"cll"``, ``"lep"`` or ``"ly"``.
    dist : Sampled, Grid or PredictiveDistribution
        A predictive distribution is priced on its total.
    premium : float
        The target price, strictly between the (capped) mean and maximum.
    assets : float, optional
        Price the loss capped at the assets, ``min(X, assets)``.
    r0 : float, default 0.0
        The fixed ``r0`` of the ``clin``, ``cll``, ``lep`` and ``ly`` families.
    
    Returns
    -------
    Distortion
    
    Examples
    --------
    >>> from prospicio.distributions import Sampled
    >>> from prospicio.risk import calibrate
    >>> x = Sampled([22, 28, 36, 40, 40, 40, 40, 55, 65, 100])
    >>> premium = (46.6 + 0.15 * 100) / 1.15
    >>> d = calibrate("proportional_hazard", x, premium)
    >>> round(d.measure(x), 9) == round(premium, 9)
    True
    """

def calibrate_classical(principle: str, dist: Any, premium: float, q: float = 2.0) -> float:
    """
    The loading of a classical premium principle that gives ``premium``;
    see ``classical_premium``.
    
    Parameters
    ----------
    principle : str
    dist : Sampled, Grid or PredictiveDistribution
    premium : float
    q : float, default 2.0
    
    Returns
    -------
    float
    """

def capital(pd: PredictiveDistribution, distortion: Distortion, method: str = "euler") -> Allocation:
    """
    Allocates the distortion risk measure of a portfolio's total to its
    components.
    
    Methods:
    
    - ``"euler"``: co-measure (for TVaR, the CoTVaRs); consistent with
      marginal changes to the portfolio.
    - ``"covariance"``: ``rho(S) Cov(X_j, S) / Var(S)``.
    - ``"proportional"``: stand-alone measures scaled to ``rho(S)``.
    - ``"marginal"``: ``rho(S) - rho(S - X_j)`` (Merton-Perold); does not
      add up to ``rho(S)``.
    - ``"shapley"``: Shapley value of ``v(T) = rho(sum of T)``; at most 12
      components.
    
    Parameters
    ----------
    pd : PredictiveDistribution
        Components that add up to the portfolio.
    distortion : Distortion
    method : str, default "euler"
    
    Returns
    -------
    Allocation
    
    Raises
    ------
    ValueError
        For an unknown method, a constant total (``"covariance"``),
        stand-alone measures summing to 0 (``"proportional"``) or more than
        12 components (``"shapley"``).
    
    Examples
    --------
    >>> from prospicio.distributions import PredictiveDistribution
    >>> from prospicio.risk import Distortion, capital
    >>> pd = PredictiveDistribution(["lob"], [("motor",), ("property",)],
    ...                             [[1.0, 2.0], [4.0, 1.0], [2.0, 5.0], [3.0, 6.0]])
    >>> a = capital(pd, Distortion.tvar(0.5))
    >>> a.total, a.standalone, a.allocated
    (8.0, [3.5, 5.5], [2.5, 5.5])
    >>> a.diversification_benefit()
    1.0
    """

def claim_count(mean: float, dispersion: float) -> Any:
    """
    The claim count with this mean and dispersion ``Var[N] / E[N]``:
    binomial below 1, Poisson at 1, negative binomial above 1.
    
    A binomial needs a whole number of trials, so below 1 the trials are
    ``mean / (1 - dispersion)`` rounded up: the mean is kept and the
    dispersion moves up to the nearest attainable value.
    
    Parameters
    ----------
    mean : float
    dispersion : float
        Positive.
    
    Returns
    -------
    Binomial or Poisson or NegativeBinomial
    
    Examples
    --------
    >>> from prospicio.distributions import claim_count
    >>> claim_count(4.0, 2.5)
    NegativeBinomial(r=2.6666666666666665, beta=1.5)
    """

def classical_premium(principle: str, dist: Any, loading: float, q: float = 2.0) -> float:
    """
    The premium of a distribution under a classical premium principle.
    
    Parameters
    ----------
    principle : str
        ``"expected_value"`` (``(1 + t) mean``), ``"variance"``
        (``mean + t var``), ``"standard_deviation"``, ``"semi_variance"``
        (``mean + t E[(X - mean)+**2]``), ``"exponential"``
        (``log E[exp(t X)] / t``), ``"esscher"``
        (``E[X exp(t X)] / E[exp(t X)]``), ``"dutch"``
        (``mean + t E[(X - mean)+]``), ``"fischer"``
        (``mean + t E[(X - mean)+**q]**(1/q)``) or ``"var"`` (the lower ``t``
        quantile).
    dist : Sampled, Grid or PredictiveDistribution
        A predictive distribution is priced on its total.
    loading : float
    q : float, default 2.0
        The Fischer power.
    
    Returns
    -------
    float
    
    Examples
    --------
    >>> from prospicio.distributions import Sampled
    >>> from prospicio.pricing import classical_premium
    >>> classical_premium("standard_deviation", Sampled([0.0, 10.0]), 0.2)
    6.0
    """

def compound_moments(frequency: Any, severity: Any) -> CompoundMoments:
    """
    Compound moments to third order: the mean, variance and third cumulant
    of ``S = X_1 + ... + X_N``, from the count's cumulants and the
    severity's first three moments, and the approximations fitted to them
    (the Python package ``aggregate``'s ``approximate``).
    
    Parameters
    ----------
    frequency : a claim count
    severity : a severity with a closed-form third moment
    
    Returns
    -------
    CompoundMoments
    
    Raises
    ------
    ValueError
        If the severity has no closed-form third moment, or an infinite one.
    
    Examples
    --------
    >>> from prospicio.aggregate import compound_moments
    >>> from prospicio.distributions import Gamma, Poisson
    >>> m = compound_moments(Poisson(10.0), Gamma.from_mean_cv(50.0, 0.7))
    >>> round(m.mean, 6), round(m.cv, 6), round(m.skew, 6)
    (500.0, 0.386005, 0.512946)
    >>> fit = m.approximate("shifted_gamma")
    >>> [(k, round(v, 4)) for k, v in fit.parameters]
    [('shape', 15.2025), ('scale', 49.5), ('shift', -252.5253)]
    """

def covar(pd: PredictiveDistribution, key: Sequence[int |str], p: float, q: float) -> float:
    """
    CoVaR of a component: the total's VaR at level ``q`` over the
    simulations where the component is at or above its own VaR at ``p``.
    
    Compare it with the total's unconditional VaR at ``q`` to see how much
    one segment's bad years drag the portfolio.
    
    Parameters
    ----------
    pd : PredictiveDistribution
    key : tuple
        The component's key.
    p : float
        The component's distress level.
    q : float
        The level of the total's VaR.
    
    Returns
    -------
    float
    
    Raises
    ------
    ValueError
        If there is no component ``key``.
    
    Examples
    --------
    >>> from prospicio.distributions import PredictiveDistribution
    >>> from prospicio.risk import covar
    >>> pd = PredictiveDistribution(["lob"], [("a",), ("b",)],
    ...                             [[1.0, 0.0], [2.0, 1.0], [3.0, 5.0], [4.0, 1.0]])
    >>> covar(pd, ("a",), 0.75, 0.5)
    5.0
    """

def credibility_weighted(estimate: float, credibility: float, complement: float) -> float:
    """
    ``Z * estimate + (1 - Z) * complement``.
    
    Parameters
    ----------
    estimate : float
    credibility : float
    complement : float
    
    Returns
    -------
    float
    
    Examples
    --------
    >>> from prospicio.pricing import credibility_weighted
    >>> round(credibility_weighted(200.0, 0.3, 300.0), 6)
    270.0
    """

def crps(draws: Sequence[float], y: float) -> float:
    """
    Continuous ranked probability score of equally likely draws for an
    outcome; lower is better.
    
    Parameters
    ----------
    draws : list of float
    y : float
    
    Returns
    -------
    float
    """

def derive_premium(technical: float, expenses: Sequence[Expense]) -> float:
    """
    The gross premium whose own expenses leave the technical premium
    ``technical``, ``aggregate``'s ``derive premium``: ``(T + F) / (1 - r)``
    with ``F`` the fixed amounts and ``r`` the premium rates.
    
    Parameters
    ----------
    technical : float
    expenses : list of Expense
    
    Returns
    -------
    float
    
    Raises
    ------
    ValueError
        On a loss-basis term, or premium rates totalling 1 or more.
    
    Examples
    --------
    >>> from prospicio.aggregate import Expense, derive_premium
    >>> round(derive_premium(5.0, [Expense("premium", 0.1), Expense("fixed", 1.0)]), 12)
    6.666666666667
    """

def deviance(family: str, y: Sequence[float], mu: Sequence[float], weights: Sequence[float] |None = None, theta: float |None = None, power: float |None = None) -> float:
    """
    Deviance ``sum w d(y, mu)`` of a family.
    
    Parameters
    ----------
    family : str
    y : list of float
    mu : list of float
    weights : list of float, optional
    theta : float, optional
    power : float, optional
    
    Returns
    -------
    float
    """

def distortion_envelope(x: Any, premium: float, s: Sequence[float], assets: float |None = None) -> tuple[list[float], list[float]]:
    """
    The smallest and largest ``g(s)`` at each ``s`` over every distortion
    that prices ``x`` at ``premium``: ``aggregate``'s ``Bounds`` envelopes,
    exact (each is the price range of a Bernoulli risk with ``P(1) = s``).
    
    Parameters
    ----------
    x : Sampled, Grid or PredictiveDistribution
    premium : float
    s : list of float
        Levels in [0, 1].
    assets : float, optional
        Cap ``x`` at the assets.
    
    Returns
    -------
    tuple of (list of float, list of float)
        The lower envelope (a concave distortion) and the upper.
    """

def elpd_loo(log_lik: Sequence[Sequence[float]], r_eff: Sequence[float] |None = None) -> Elpd:
    """
    Leave-one-out cross-validation by Pareto-smoothed importance sampling
    (PSIS-LOO), from one fit's pointwise log-likelihood draws. Matches the
    R package ``loo``.
    
    Parameters
    ----------
    log_lik : list of list of float
        One row per posterior draw, one column per observation:
        ``log p(y_i | theta_s)``.
    r_eff : list of float, optional
        Relative efficiency of the draws per observation (1 for
        independent draws).
    
    Returns
    -------
    Elpd
        With ``pareto_k`` and ``k_threshold``.
    """

def elpd_waic(log_lik: Sequence[Sequence[float]]) -> Elpd:
    """
    WAIC from pointwise log-likelihood draws: ``lppd`` less the variance of
    each observation's log-likelihood.
    
    Parameters
    ----------
    log_lik : list of list of float
        One row per posterior draw, one column per observation.
    
    Returns
    -------
    Elpd
    """

def entropic(dist: Any, theta: float) -> float:
    """
    Entropic risk measure ``(1 / theta) log E[exp(theta X)]``: the certainty
    equivalent of a loss under exponential utility.
    
    It rises from the mean (``theta -> 0``) to the largest draw
    (``theta -> inf``); for a normal loss it is ``mu + theta sigma**2 / 2``.
    
    Parameters
    ----------
    dist : Sampled, PredictiveDistribution or list of float
        A predictive distribution is measured on its total.
    theta : float
        Risk aversion, positive.
    
    Returns
    -------
    float
    
    Examples
    --------
    >>> import math
    >>> from prospicio.risk import entropic
    >>> round(entropic([0.0, 1.0], math.log(2.0)), 12) == round(math.log2(1.5), 12)
    True
    """

def esscher(dist: Any, h: float) -> float:
    """
    Esscher premium ``E[X exp(h X)] / E[exp(h X)]``: the mean after tilting
    probability towards large losses.
    
    The mean at ``h = 0``; ``mu + h sigma**2`` for a normal loss.
    
    Parameters
    ----------
    dist : Sampled, PredictiveDistribution or list of float
        A predictive distribution is measured on its total.
    h : float
    
    Returns
    -------
    float
    
    Examples
    --------
    >>> import math
    >>> from prospicio.risk import esscher
    >>> round(esscher([0.0, 1.0], math.log(3.0)), 12)
    0.75
    """

def esscher_allocation(pd: PredictiveDistribution, h: float) -> list[float]:
    """
    Esscher allocation: each component's mean under the Esscher transform
    of the total, ``E[X_j exp(h S)] / E[exp(h S)]``.
    
    The contributions sum to ``esscher(pd, h)``; at ``h = 0`` they are the
    means.
    
    Parameters
    ----------
    pd : PredictiveDistribution
    h : float
    
    Returns
    -------
    list of float
        One per component, in ``pd.components()`` order.
    
    Examples
    --------
    >>> from prospicio.distributions import PredictiveDistribution
    >>> from prospicio.risk import esscher_allocation
    >>> pd = PredictiveDistribution(["lob"], [("motor",), ("property",)],
    ...                             [[1.0, 2.0], [4.0, 1.0], [2.0, 5.0], [3.0, 6.0]])
    >>> esscher_allocation(pd, 0.0)
    [2.5, 3.5]
    """

def eventual_ruin(severity: Grid, rho: float |None = None, *, loss_ratio: float |None = None, points: int |None = None, cap: float |None = None, excess: float |None = None, stop_loss: float |None = None) -> Ruin:
    """
    The probability of eventual ruin of the classical surplus process by
    the Pollaczeck–Khinchine formula, as ``aggregate``'s
    ``pollaczeck_khinchine`` and ``eventual_ruin``: Poisson claims with this
    severity and a premium at ``1 + rho`` times the expected loss. The
    Poisson rate cancels. The maximal aggregate loss is a compound geometric
    of the equilibrium severity, by FFT.
    
    Parameters
    ----------
    severity : Grid
    rho : float, optional
        The margin to loss, positive. Or give ``loss_ratio``.
    loss_ratio : float, optional
        ``1 / (1 + rho)``, in (0, 1).
    points : int, optional
        Grid points for ``psi``; the severity's by default.
    cap : float, optional
        Use ``X | X <= cap``.
    excess : float, optional
        Use ``X | X > excess`` (not shifted).
    stop_loss : float, optional
        Use ``X`` with the mass above ``stop_loss`` moved onto it.
    
    Returns
    -------
    Ruin
    
    Examples
    --------
    >>> from prospicio.aggregate import eventual_ruin
    >>> from prospicio.distributions import Gamma, Grid
    >>> sev, _ = Grid.rounding(Gamma.from_mean_cv(100.0, 1.0), 1.0, 1 << 14)
    >>> ruin = eventual_ruin(sev, rho=0.2)
    >>> # Exponential claims: exp(-u / 600) / 1.2 = 0.157; the step costs 1%.
    >>> round(ruin.psi_at(1000.0), 3)
    0.156
    """

def fft(frequency: Any, severity: Grid, points: int) -> tuple[Grid, CompoundReport]:
    """
    Aggregate loss ``S = X_1 + ... + X_N`` by fast Fourier transform.
    
    Works for large claim counts that make ``panjer`` underflow. Check
    ``report.aliasing_error`` before using the result.
    
    Parameters
    ----------
    frequency : Poisson, NegativeBinomial or Binomial
    severity : Grid
    points : int
    
    Returns
    -------
    tuple of (Grid, CompoundReport)
    
    Raises
    ------
    ValueError
        If ``points`` is 0.
    """

def fft_auto(frequency: Any, severity: Any, log2: int = 16, p: float = ..., p_star: float = ...) -> tuple[Grid, CompoundReport, GridSize]:
    """
    The compound distribution by FFT on the grid ``recommend_grid``
    chooses, the severity discretized by rounding.
    
    Parameters
    ----------
    frequency : a claim count
    severity : a severity
    log2 : int, default 16
    p : float, default 1 - 1e-5
    p_star : float, default 1 - 1e-12
    
    Returns
    -------
    tuple of (Grid, CompoundReport, GridSize)
    
    Raises
    ------
    ValueError
        As ``recommend_grid``.
    
    Examples
    --------
    >>> from prospicio.aggregate import fft_auto
    >>> from prospicio.distributions import Lognormal, Poisson
    >>> agg, report, size = fft_auto(Poisson(5.0), Lognormal.from_mean_cv(100.0, 1.0))
    >>> round(agg.mean()), report.aliasing_error < 1e-12
    (500, True)
    """

def fft_bivariate(frequency: Any, severity: JointGrid, points: tuple[int, int]) -> tuple[JointGrid, BivariateReport]:
    """
    The joint annual loss to two lines hit by the same events, by a 2-D FFT:
    ``(A, B) = sum over N events of (X_k, Y_k)``, with ``severity`` the joint
    law of one event's pair of losses (mass at 0 on an axis is an event that
    misses that line).
    
    Parameters
    ----------
    frequency : Poisson, NegativeBinomial, Binomial or Count
        The shared count of events.
    severity : JointGrid
    points : tuple of (int, int)
        Points on each axis of the result.
    
    Returns
    -------
    tuple of (JointGrid, BivariateReport)
    
    Examples
    --------
    >>> from prospicio.aggregate import JointGrid, fft_bivariate
    >>> from prospicio.distributions import Grid, Poisson
    >>> a = Grid(1.0, [0.0, 0.6, 0.4])
    >>> b = Grid(1.0, [0.5, 0.5])
    >>> joint, report = fft_bivariate(Poisson(2.0), JointGrid.independent(a, b), (64, 64))
    >>> round(joint.covariance(), 9)  # Var(N) E[X] E[Y]
    1.4
    """

def fit_pml_curve(return_periods: Sequence[float], amounts: Sequence[float], tail_alpha: float = 2.0, truncation: float |None = None) -> TowerModel:
    """
    The model through the points of a PML curve: ``amounts[j]`` is
    exceeded once in ``return_periods[j]`` years.
    
    Parameters
    ----------
    return_periods : list of float
    amounts : list of float
    tail_alpha : float, default 2.0
        Alpha above the largest amount.
    truncation : float, optional
        Truncation of the last piece.
    
    Returns
    -------
    TowerModel
    """

def fit_references(layers: Sequence[tuple[float, float, float]] = ..., frequencies: Sequence[tuple[float, float]] = ..., default_alpha: float = 2.0, rule: str = "minimize") -> TowerModel:
    """
    A model that reproduces every reference: expected layer losses (which
    may overlap or leave gaps) and excess frequencies.
    
    Parameters
    ----------
    layers : list of tuple of float, optional
        ``(limit, attachment, expected_loss)`` per layer.
    frequencies : list of tuple of float, optional
        ``(threshold, frequency)`` per excess frequency.
    default_alpha : float, default 2.0
        Alpha above the highest point, unless an unlimited layer sets it.
    rule : {"minimize", "midpoint"}, default "minimize"
    
    Returns
    -------
    TowerModel
    
    Examples
    --------
    >>> from prospicio.pricing import fit_references
    >>> m = fit_references([(1000.0, 1000.0, 150.0), (3000.0, 1500.0, 160.0)], [(1000.0, 0.3)])
    >>> round(m.layer_loss(3000.0, 1500.0), 6)
    160.0
    """

def from_json(text: str) -> Any:
    """
    A distribution from a document written by ``to_json``, as the class of
    its family.
    
    Parameters
    ----------
    text : str
    
    Returns
    -------
    a distribution
    
    Raises
    ------
    ValueError
        If the document is malformed, of another format or a newer
        version, or its parameters are out of range.
    """

def gini(y: Sequence[float], pred: Sequence[float], exposure: Sequence[float] |None = None) -> float:
    """
    Gini index of the ordered Lorenz curve.
    
    Parameters
    ----------
    y : list of float
    pred : list of float
    exposure : list of float, optional
    
    Returns
    -------
    float
    """

def group_k_fold(groups: Sequence[str], k: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """
    Grouped ``k``-fold splits: each group's rows stay in one fold.
    
    Parameters
    ----------
    groups : list of str
    k : int
    seed : int
    
    Returns
    -------
    list of (list of int, list of int)
    """

def hill(draws: Sequence[float], ks: Sequence[int]) -> list[float]:
    """
    Hill estimates of the tail index ``xi`` (``1 / alpha``) from the ``k``
    largest draws, for each ``k``.
    
    Parameters
    ----------
    draws : list of float
    ks : list of int
    
    Returns
    -------
    list of float
    
    Raises
    ------
    ValueError
        If a ``k`` is 0 or not below the number of draws, or the ``k + 1``
        largest draws are not all positive.
    """

def ilf(severity: Any, limit: float, basic_limit: float) -> float:
    """
    Increased limit factor ``LEV(limit) / LEV(basic_limit)``.
    
    Parameters
    ----------
    severity : a severity
    limit : float
    basic_limit : float
    
    Returns
    -------
    float
    
    Examples
    --------
    >>> from prospicio.distributions import Pareto
    >>> from prospicio.pricing import ilf
    >>> round(ilf(Pareto(100.0, 2.0), 1000.0, 200.0), 12)
    1.266666666667
    """

def iman_conover(pd: PredictiveDistribution, correlation: Sequence[Sequence[float]], seed: int) -> PredictiveDistribution:
    """
    Reorders each component's draws to a target correlation (Iman-Conover).
    
    Every component keeps exactly its own draws; only their pairing across
    simulations changes. The correlation of the result's normal scores is
    close to ``correlation``, and Spearman's rho close to
    ``(6 / pi) asin(correlation / 2)``.
    
    Parameters
    ----------
    pd : PredictiveDistribution
    correlation : list of list of float
        One row and column per component.
    seed : int
    
    Returns
    -------
    PredictiveDistribution
    
    Examples
    --------
    >>> from prospicio.distributions import PredictiveDistribution
    >>> from prospicio.risk import iman_conover
    >>> rows = [[float(i), float((i * 7919) % 1000)] for i in range(1000)]
    >>> pd = PredictiveDistribution(["lob"], [(0,), (1,)], rows)
    >>> joined = iman_conover(pd, [[1.0, 0.7], [0.7, 1.0]], seed=3)
    >>> sorted(joined.marginal((1,)).draws) == sorted(pd.marginal((1,)).draws)
    True
    """

def k_fold(n: int, k: int, seed: int) -> list[tuple[list[int], list[int]]]:
    """
    ``k``-fold splits of ``n`` rows, shuffled with ``seed``.
    
    Parameters
    ----------
    n : int
    k : int
    seed : int
    
    Returns
    -------
    list of (list of int, list of int)
        ``(train, test)`` row indices per fold.
    """

def ks_uniform(values: Sequence[float]) -> float:
    """
    Kolmogorov-Smirnov distance from the uniform on ``[0, 1]``; about
    ``1.36 / sqrt(n)`` or less 95% of the time under uniformity.
    
    Parameters
    ----------
    values : list of float
    
    Returns
    -------
    float
    """

def layer_loss_picks(severity: Grid, attachments: Sequence[float], picks: Sequence[float], claims: float = 1.0) -> Picks:
    """
    Reshapes a severity so that its expected loss to each layer hits a
    pick, as ``aggregate``'s ``Aggregate.picks``: in each layer the survival
    function is rescaled about its value at the layer's top, keeping its
    shape there; above the top attachment the severity is kept.
    
    Parameters
    ----------
    severity : Grid
    attachments : list of float
        Increasing grid points; the layers are 0 to the first, then between
        consecutive attachments.
    picks : list of float
        The loss to each layer for ``claims`` expected ground-up claims.
    claims : float, default 1.0
    
    Returns
    -------
    Picks
    
    Raises
    ------
    ValueError
        If an attachment is not a grid point or the picks are infeasible (a
        pick below what the layers above force through its layer).
    
    Examples
    --------
    >>> from prospicio.aggregate import layer_loss_picks
    >>> from prospicio.distributions import Gamma, Grid
    >>> sev, _ = Grid.rounding(Gamma.from_mean_cv(100.0, 1.5), 1.0, 1 << 14)
    >>> picked = layer_loss_picks(sev, [100, 500, 1000], [480, 420, 60], claims=10)
    >>> [round(10 * t, 6) for t in picked.to_dict()["target"]]
    [480.0, 420.0, 60.0]
    """

def lift(y: Sequence[float], pred: Sequence[float], exposure: Sequence[float] |None = None, bands: int = 10) -> list[dict[str, float]]:
    """
    Lift table: rows sorted by predicted rate, cut into bands of about
    equal exposure.
    
    Parameters
    ----------
    y : list of float
    pred : list of float
    exposure : list of float, optional
    bands : int, default 10
    
    Returns
    -------
    list of dict
        ``exposure``, ``expected`` and ``actual`` per band.
    """

def local_pareto_to_piecewise(t: float, alpha: Any, rel_tolerance: float = 1e-4, stop_survival: float = 1e-9, stop_at: float = ...) -> tuple[PiecewisePareto, float, float]:
    """
    Converts the local Pareto distribution with local alpha ``alpha(x)``
    above ``t`` to a piecewise Pareto that matches its survival function
    exactly at the thresholds and within ``rel_tolerance`` between them.
    
    Parameters
    ----------
    t : float
        Threshold; ``P(X > x) = 1`` below it.
    alpha : callable
        ``alpha(x) -> float``, finite and non-negative, positive where the
        conversion stops.
    rel_tolerance : float, default 1e-4
    stop_survival : float, default 1e-9
        Stop once the survival function falls below this.
    stop_at : float, default inf
        Stop at this amount.
    
    Returns
    -------
    tuple of (PiecewisePareto, float, float)
        The approximation, the largest relative error found, and where the
        approximated range ends (the last alpha continues above it).
    
    Examples
    --------
    >>> import math
    >>> from prospicio.distributions import local_pareto_to_piecewise
    >>> pp, err, end = local_pareto_to_piecewise(1000.0, lambda x: 1.5 + 0.3 * math.log(x / 1000.0))
    >>> err <= 1e-4
    True
    """

def log_score(family: str, y: Sequence[float], mu: Sequence[float], dispersion: float = 1.0, weights: Sequence[float] |None = None, theta: float |None = None, power: float |None = None) -> float:
    """
    Mean log score ``-(1/n) sum log f(y_i)`` of the outcomes under the
    family's predictive distribution; lower is better.
    
    Parameters
    ----------
    family : str
    y : list of float
    mu : list of float
    dispersion : float, default 1.0
    weights : list of float, optional
    theta : float, optional
    power : float, optional
    
    Returns
    -------
    float
    
    Examples
    --------
    >>> from prospicio.models import log_score
    >>> round(log_score("poisson", [0.0], [1.0]), 12)
    1.0
    """

def loss_elimination_ratio(severity: Any, deductible: float) -> float:
    """
    Loss elimination ratio of a deductible, ``LEV(deductible) / E[X]``.
    
    Parameters
    ----------
    severity : a severity
    deductible : float
    
    Returns
    -------
    float
    """

def lppd(log_lik: Sequence[Sequence[float]]) -> float:
    """
    In-sample log pointwise predictive density,
    ``sum_i log(mean_s p(y_i | theta_s))``.
    
    Parameters
    ----------
    log_lik : list of list of float
        One row per posterior draw, one column per observation.
    
    Returns
    -------
    float
    
    Examples
    --------
    >>> import math
    >>> from prospicio.models import lppd
    >>> round(lppd([[math.log(0.5)], [math.log(0.25)]]), 12) == round(math.log(0.375), 12)
    True
    """

def marginal_expected_shortfall(pd: PredictiveDistribution, p: float) -> list[float]:
    """
    Marginal expected shortfall of each component at level ``p``: its mean
    over the simulations where the total is in its worst ``1 - p``.
    
    The same as ``allocate(pd, Distortion.tvar(p))``; it sums to the
    total's TVaR.
    
    Parameters
    ----------
    pd : PredictiveDistribution
    p : float
    
    Returns
    -------
    list of float
        One per component, in ``pd.components()`` order.
    
    Examples
    --------
    >>> from prospicio.distributions import PredictiveDistribution
    >>> from prospicio.risk import marginal_expected_shortfall
    >>> pd = PredictiveDistribution(["lob"], [("motor",), ("property",)],
    ...                             [[1.0, 2.0], [4.0, 1.0], [2.0, 5.0], [3.0, 6.0]])
    >>> marginal_expected_shortfall(pd, 0.5)
    [2.5, 5.5]
    """

def match_tower(attachments: Sequence[float], layer_losses: Sequence[float], frequencies: Sequence[float |None] |None = None, rule: str = "minimize") -> TowerModel:
    """
    Matches a tower of contiguous layers, the last unlimited, with one
    frequency and a piecewise Pareto severity (Riegel 2018).
    
    Parameters
    ----------
    attachments : list of float
        Increasing attachment points; layer ``i`` runs to the next one, the
        last is unlimited.
    layer_losses : list of float
        Expected loss a year of each layer.
    frequencies : list of float or None, optional
        Expected losses a year above each attachment point; ``None`` (or a
        ``None`` entry) to derive them.
    rule : {"minimize", "midpoint"}, default "minimize"
        How the free threshold inside each layer is chosen.
    
    Returns
    -------
    TowerModel
    
    Examples
    --------
    >>> from prospicio.pricing import match_tower
    >>> m = match_tower([1000.0, 1500.0, 2000.0], [100.0, 90.0, 120.0], [0.25, None, None])
    >>> round(m.layer_loss(500.0, 1500.0), 9)
    90.0
    """

def mcmc_diagnostics(chains: Sequence[Sequence[float]]) -> dict[str, float]:
    """
    MCMC diagnostics of chains of draws (Vehtari et al. 2021, as R's
    ``posterior``): rank-normalized split R-hat, bulk and tail effective
    sample sizes, the effective sample size of the mean and its Monte Carlo
    standard error.
    
    Parameters
    ----------
    chains : list of list of float
        Equal-length chains, at least 4 draws each.
    
    Returns
    -------
    dict
        ``rhat``, ``ess_bulk``, ``ess_tail``, ``ess_mean``, ``mcse_mean``.
    
    Examples
    --------
    >>> from prospicio.models import mcmc_diagnostics
    >>> a = [float((i * 37) % 101) for i in range(400)]
    >>> b = [float((i * 53 + 7) % 101) for i in range(400)]
    >>> mcmc_diagnostics([a, b])["rhat"] < 1.01
    True
    """

def mean_excess(draws: Sequence[float], thresholds: Sequence[float]) -> list[tuple[float, float, int]]:
    """
    The empirical mean-excess function ``e(u) = E[X - u | X > u]`` at each
    threshold, linear above a threshold where a GPD fits.
    
    Parameters
    ----------
    draws : list of float
    thresholds : list of float
    
    Returns
    -------
    list of (float, float, int)
        ``(u, e(u), number of draws above u)``; ``e(u)`` is NaN when none are.
    
    Examples
    --------
    >>> from prospicio.risk import mean_excess
    >>> mean_excess([1.0, 2.0, 3.0, 4.0], [2.0])
    [(2.0, 1.5, 2)]
    """

def occurrence_cession_statistics(frequency: Any, severity: Grid, layers: Sequence[tuple[float, float, float]]) -> CessionStatistics:
    """
    The statistics of the total cession of several per-occurrence layers
    side by side, as ``aggregate``'s ``Ceded`` column of
    ``reins_stats_df``: a claim cedes the sum of ``share * min(max(x - a,
    0), limit)`` over the layers. Moments are exact for the grid.
    
    Parameters
    ----------
    frequency : Poisson, NegativeBinomial or Binomial
    severity : Grid
    layers : list of (share, limit, attachment)
    
    Returns
    -------
    CessionStatistics
    
    Examples
    --------
    >>> from prospicio.reinsurance import occurrence_cession_statistics
    >>> from prospicio.distributions import Grid, Poisson
    >>> sev = Grid(1.0, [0.0, 0.4, 0.3, 0.2, 0.1])
    >>> c = occurrence_cession_statistics(Poisson(3.0), sev, [(1.0, 1.0, 1.0), (1.0, 2.0, 2.0)])
    >>> round(c.aggregate.mean, 12)
    3.0
    """

def occurrence_exceedance(frequency: Any, severity: Any, x: float) -> float:
    """
    The chance that the year's largest occurrence exceeds ``x``,
    ``1 - G_N(F(x))`` with ``G_N`` the count's generating function: the
    occurrence exceedance (OEP) curve.
    
    Parameters
    ----------
    frequency : a claim count
    severity : a severity
    x : float
    
    Returns
    -------
    float
    
    Examples
    --------
    >>> import math
    >>> from prospicio.aggregate import occurrence_exceedance
    >>> from prospicio.distributions import Lognormal, Poisson
    >>> sev = Lognormal.from_mean_cv(100.0, 2.0)
    >>> oep = occurrence_exceedance(Poisson(2.0), sev, 500.0)
    >>> abs(oep - (1 - math.exp(-2 * (1 - sev.cdf(500.0))))) < 1e-12
    True
    """

def occurrence_layer_statistics(frequency: Any, severity: Grid, share: float, limit: float, attachment: float) -> LayerStatistics:
    """
    The statistics of ``share`` of ``limit`` xs ``attachment`` per
    occurrence on a severity grid, as ``aggregate``'s ``reins_stats_df``.
    
    The chances that a claim reaches the layer (``P(X > a)``, exclusive)
    and exhausts it (``P(X >= a + l)``, inclusive), and that the year has a
    loss to it; the excess count (the count thinned to the claims that
    reach the layer), the layer severity given a loss to it and the annual
    ceded loss, each by its first three cumulants; and the loss on line.
    The share scales the moments exactly.
    
    Parameters
    ----------
    frequency : Poisson, NegativeBinomial or Binomial
    severity : Grid
    share : float
        In ``(0, 1]``.
    limit : float
        May be ``inf``; the loss on line is then 0.
    attachment : float
    
    Returns
    -------
    LayerStatistics
    
    Raises
    ------
    ValueError
        For a share outside ``(0, 1]``, a non-positive limit or a negative
        attachment.
    
    Examples
    --------
    >>> from prospicio.reinsurance import occurrence_layer_statistics
    >>> from prospicio.distributions import Grid, Poisson
    >>> sev = Grid(1.0, [0.0, 0.4, 0.3, 0.2, 0.1])
    >>> s = occurrence_layer_statistics(Poisson(3.0), sev, 1.0, 2.0, 2.0)
    >>> round(s.pr_attach, 12), round(s.count.mean, 12), round(s.aggregate.mean, 12)
    (0.3, 0.9, 1.2)
    """

def occurrence_pml(frequency: Any, severity: Any, p: float) -> OccurrencePml:
    """
    The occurrence PML at an annual probability ``p``: the loss whose
    occurrence exceedance is ``p``, for any claim count (the Python package
    ``aggregate``'s ``oep`` covers Poisson counts).
    
    Parameters
    ----------
    frequency : a claim count
    severity : a severity
    p : float
        In ``(0, 1 - P(N = 0))``.
    
    Returns
    -------
    OccurrencePml
    
    Raises
    ------
    ValueError
        If ``p`` is outside ``(0, 1 - P(N = 0))``.
    
    Examples
    --------
    >>> from prospicio.aggregate import occurrence_pml
    >>> from prospicio.distributions import Lognormal, Poisson
    >>> pml = occurrence_pml(Poisson(2.0), Lognormal.from_mean_cv(100.0, 2.0), 0.01)
    >>> round(pml.loss, 4), pml.annual_return_period
    (1171.4979, 100.0)
    """

def panjer(frequency: Any, severity: Grid, points: int) -> tuple[Grid, CompoundReport]:
    """
    Aggregate loss ``S = X_1 + ... + X_N`` by Panjer's recursion.
    
    Parameters
    ----------
    frequency : Poisson, NegativeBinomial or Binomial
    severity : Grid
        Severity on a grid; the result uses its step.
    points : int
        Points in the aggregate grid.
    
    Returns
    -------
    tuple of (Grid, CompoundReport)
    
    Raises
    ------
    ValueError
        If ``points`` is 0 or ``P(S = 0)`` underflows (use ``fft``).
    
    Examples
    --------
    >>> from prospicio.aggregate import panjer
    >>> from prospicio.distributions import Grid, Poisson
    >>> sev = Grid(1.0, [0.1, 0.3, 0.25, 0.2, 0.1, 0.05])
    >>> agg, report = panjer(Poisson(3.0), sev, 100)
    >>> round(agg.mean(), 6)
    6.15
    """

def pareto_extrapolation(from_: tuple[float, float], to: tuple[float, float], alpha: float, truncation: float |None = None) -> float:
    """
    Expected loss of layer ``to`` per unit of expected loss of layer
    ``from_``, under a Pareto with this alpha (and truncation).
    
    Parameters
    ----------
    from_ : tuple of float
        ``(limit, attachment)``.
    to : tuple of float
        ``(limit, attachment)``.
    alpha : float
    truncation : float, optional
    
    Returns
    -------
    float
    
    Examples
    --------
    >>> from prospicio.pricing import pareto_extrapolation
    >>> round(pareto_extrapolation((1e6, 1e6), (2e6, 2e6), 2.0), 12)
    0.5
    """

def pinball_loss(y: Sequence[float], pred: Sequence[float], alpha: float, weights: Sequence[float] |None = None) -> float:
    """
    Weighted mean pinball (quantile) loss of predictions of the ``alpha``
    quantile: ``sum(w * rho(y - q)) / sum(w)`` with
    ``rho(u) = u * (alpha - (u < 0))``. Lowest in expectation at the true
    ``alpha`` quantile; lower is better.
    
    Parameters
    ----------
    y : list of float
    pred : list of float
        Predicted ``alpha`` quantiles.
    alpha : float
        In ``(0, 1)``.
    weights : list of float, optional
    
    Returns
    -------
    float
    
    Examples
    --------
    >>> from prospicio.models import pinball_loss
    >>> pinball_loss([1.0, 0.0], [0.0, 1.0], 0.9)
    0.5
    """

def pit(family: str, y: Sequence[float], mu: Sequence[float], dispersion: float = 1.0, weights: Sequence[float] |None = None, seed: int = 0, theta: float |None = None, power: float |None = None) -> list[float]:
    """
    Probability integral transform of each outcome under the family's
    predictive distribution, randomized where it has atoms (counts, a
    Tweedie's zero); uniform when the model is calibrated.
    
    Parameters
    ----------
    family : str
    y : list of float
    mu : list of float
    dispersion : float, default 1.0
    weights : list of float, optional
    seed : int, default 0
        Seeds the randomization.
    theta : float, optional
    power : float, optional
    
    Returns
    -------
    list of float
    """

def pit_from_draws(draws: Sequence[float], y: float, u: float = 0.5) -> float:
    """
    The PIT of ``y`` under the empirical distribution of ``draws``,
    randomized over ties by ``u``.
    
    Parameters
    ----------
    draws : list of float
    y : float
    u : float, default 0.5
    
    Returns
    -------
    float
    """

def pit_histogram(pit: Sequence[float], bins: int = 10) -> list[int]:
    """
    Counts of PIT values in ``bins`` equal-width bins of ``[0, 1]``.
    
    Parameters
    ----------
    pit : list of float
    bins : int, default 10
    
    Returns
    -------
    list of int
    """

def premium_based_expenses(ratios: Sequence[float], fixed_shares: Sequence[float]) -> tuple[float, float]:
    """
    The premium-based projection of expenses: each expense ratio split by
    its fixed share and summed, as ``(fixed, variable)``. With every share 0
    it is the all-variable method.
    
    Parameters
    ----------
    ratios : list of float
        Expense ratios to premium.
    fixed_shares : list of float
        The share of each that is fixed, in ``[0, 1]``.
    
    Returns
    -------
    tuple of float
    
    Examples
    --------
    >>> from prospicio.pricing import premium_based_expenses
    >>> fixed, variable = premium_based_expenses([0.06, 0.10], [0.5, 0.0])
    >>> round(fixed, 6), round(variable, 6)
    (0.03, 0.13)
    """

def price(losses: Any, assets: Distortion, *, cost_of_capital: float |None = None, distortion: Distortion |None = None) -> Price:
    """
    Risk-loaded price of a cover from its simulated losses.
    
    The assets backing the loss are a distortion risk measure of it. The
    premium is either a pricing distortion of the loss, or set by a
    constant cost of capital ``r`` on the capital ``a - P``, which gives
    ``P = (E[X] + r a) / (1 + r)``.
    
    Parameters
    ----------
    losses : Sampled, PredictiveDistribution or Grid
        Loss draws (for a ``PredictiveDistribution``, its total), or a loss
        distribution on a grid, such as a layer's exact annual loss from
        ``Tower.on_grid``.
    assets : Distortion
        The measure that sets the assets, for example ``Distortion.tvar(0.99)``.
    cost_of_capital : float, optional
        Positive rate. Give this or ``distortion``.
    distortion : Distortion, optional
        Pricing distortion; it must load less than ``assets``.
    
    Returns
    -------
    Price
    
    Raises
    ------
    ValueError
        Unless exactly one rule is given, or if the premium exceeds the
        assets.
    
    Examples
    --------
    >>> from prospicio.distributions import Sampled
    >>> from prospicio.pricing import price
    >>> from prospicio.risk import Distortion
    >>> p = price(Sampled([0.0, 0.0, 2.0, 6.0]), Distortion.tvar(0.5), cost_of_capital=0.25)
    >>> p.premium, p.capital
    (2.4, 1.6)
    >>> from prospicio.distributions import Grid
    >>> g = price(Grid(2.0, [0.5, 0.25, 0.0, 0.25]), Distortion.tvar(0.5), cost_of_capital=0.25)
    >>> round(g.premium, 12)
    2.4
    """

def price_over_payout(price: Price, pattern: PaymentPattern, *, rate: float = 0.0, cost_of_capital: float |None = None, distortion: Distortion |None = None) -> PayoutPrice:
    """
    A one-period price restated over a payout.
    
    With a cost of capital ``r``, the premium solves
    ``P = E[X] d + r (a - P) A``: the loss discounted by the pattern's
    ``discount_factor`` ``d``, plus the cost of the initial capital held while
    losses are outstanding over ``A = capital_years``. With a pricing
    distortion, the premium is discounted with the losses. Give the same rule
    the price was made with. A pattern that pays everything within the first
    year, at a zero rate, gives back the one-period price.
    
    Parameters
    ----------
    price : Price
        The one-period price, from ``price``.
    pattern : PaymentPattern
    rate : float, default 0.0
        Annual effective discount rate.
    cost_of_capital : float, optional
    distortion : Distortion, optional
    
    Returns
    -------
    PayoutPrice
    
    Examples
    --------
    >>> from prospicio.distributions import Sampled
    >>> from prospicio.pricing import PaymentPattern, price, price_over_payout
    >>> from prospicio.risk import Distortion
    >>> p = price(Sampled([0.0, 0.0, 2.0, 6.0]), Distortion.tvar(0.5), cost_of_capital=0.25)
    >>> slow = PaymentPattern([1.0, 2.0, 3.0], [0.2, 0.3, 0.5])
    >>> q = price_over_payout(p, slow, rate=0.0, cost_of_capital=0.25)
    >>> round(q.capital_years, 12), round(q.premium, 6)
    (2.3, 2.730159)
    """

def price_portfolio(pd: PredictiveDistribution, assets: Distortion, *, cost_of_capital: float |None = None, distortion: Distortion |None = None) -> PortfolioPrice:
    """
    Prices a portfolio and allocates the price to its components.
    
    Premium and assets are each allocated by co-measure (the natural
    allocation): component prices add up to the portfolio's, and a
    component that diversifies the portfolio is priced below its
    standalone price. With a cost of capital, every component earns the
    rate on its allocated capital.
    
    Parameters
    ----------
    pd : PredictiveDistribution
        Components that add up to the portfolio: segments or covers, not
        gross, ceded and net side by side.
    assets : Distortion
    cost_of_capital : float, optional
    distortion : Distortion, optional
        Exactly one of ``cost_of_capital`` and ``distortion``, as in
        ``price``.
    
    Returns
    -------
    PortfolioPrice
    
    Examples
    --------
    >>> from prospicio.distributions import PredictiveDistribution
    >>> from prospicio.pricing import price_portfolio
    >>> from prospicio.risk import Distortion
    >>> pd = PredictiveDistribution(["cover"], [("a",), ("b",)],
    ...                             [[0.0, 2.0], [1.0, 1.0], [4.0, 0.0], [8.0, 0.0]])
    >>> p = price_portfolio(pd, Distortion.tvar(0.5), cost_of_capital=0.1)
    >>> [round(c.premium, 6) for c in p.allocated]
    [3.5, 0.681818]
    >>> p.allocated[1].margin < 0  # the second cover hedges the first
    True
    """

def pseudo_bma_weights(lpd: Sequence[Sequence[float]], bootstrap: bool = True, n_draws: int = 1000, seed: int = 0) -> list[float]:
    """
    Pseudo-BMA weights, ``w_k`` proportional to ``exp(elpd_k)``; with
    ``bootstrap=True``, pseudo-BMA+ weights averaged over Bayesian-bootstrap
    replicates of the observations, which keeps a model that is only
    slightly better from taking all the weight.
    
    Parameters
    ----------
    lpd : list of list of float
        One list per model, as for ``stacking_weights``.
    bootstrap : bool, default True
    n_draws : int, default 1000
        Bootstrap replicates.
    seed : int, default 0
        Replicate ``b`` uses stream ``b`` of ``seed``.
    
    Returns
    -------
    list of float
    """

def recommend_grid(frequency: Any, severity: Any, log2: int = 16, p: float = ..., p_star: float = ...) -> GridSize:
    """
    A grid for the compound distribution of ``frequency`` claims of
    ``severity``, as ``aggregate`` sizes one when no bucket is given: the
    larger of a lognormal or gamma fitted to the aggregate's mean and
    variance at ``p``, and one big claim on a typical bulk at ``p``; then
    one big claim at ``p_star`` when it fits at the same bucket.
    
    Parameters
    ----------
    frequency : a claim count
    severity : a severity
        Any distribution but ``Sampled``.
    log2 : int, default 16
        At most ``2**log2`` points.
    p : float, default 1 - 1e-5
    p_star : float, default 1 - 1e-12
    
    Returns
    -------
    GridSize
    
    Raises
    ------
    ValueError
        If the aggregate mean is not finite and positive, or ``log2`` is not
        in ``1..=30``.
    
    Examples
    --------
    >>> from prospicio.aggregate import recommend_grid
    >>> from prospicio.distributions import Gamma, Poisson
    >>> g = recommend_grid(Poisson(10.0), Gamma.from_mean_cv(50.0, 0.7))
    >>> g.step, g.points, g.method
    (0.0625, 65536, 'moments')
    """

def round_bucket(bs: float) -> float:
    """
    Rounds a bucket size up to a "nice" value, as ``aggregate``'s
    ``round_bucket``: ``{1, 2, 4, 5, 8} * 10**k`` at 1 and above, a power of
    two below.
    
    Parameters
    ----------
    bs : float
        Positive and finite.
    
    Returns
    -------
    float
    
    Raises
    ------
    ValueError
        If ``bs`` is not positive and finite.
    
    Examples
    --------
    >>> from prospicio.aggregate import round_bucket
    >>> round_bucket(3.4), round_bucket(0.3)
    (4.0, 0.5)
    """

def severity_exposure_curve(severity: Any, mpl: float, x: Sequence[float]) -> list[float]:
    """
    The exposure curve of a severity capped at the maximum possible loss
    ``mpl``: ``G(x) = LEV(x mpl) / LEV(mpl)`` at each ``x``.
    
    Parameters
    ----------
    severity : a severity
    mpl : float
    x : list of float
    
    Returns
    -------
    list of float
    
    Examples
    --------
    >>> from prospicio.distributions import Pareto
    >>> from prospicio.pricing import severity_exposure_curve
    >>> g = severity_exposure_curve(Pareto(1e5, 1.5), 1e7, [0.0, 0.5, 1.0])
    >>> g[0], round(g[2], 12), g[1] > 0.5
    (0.0, 1.0, True)
    """

def simulate(copula: Any, marginals: Sequence[Any], n_sims: int, seed: int, keys: Sequence[Sequence[int |str]] |None = None, dims: Sequence[str] |None = None) -> PredictiveDistribution:
    """
    Simulates marginals joined by a copula.
    
    In simulation ``i``, draws uniforms from ``copula`` with stream ``i`` of
    ``seed`` and applies each marginal's quantile function.
    
    Parameters
    ----------
    copula : GaussianCopula, StudentTCopula or ArchimedeanCopula
    marginals : list of Lognormal, Grid or Pareto-family severities
        One per copula dimension.
    n_sims : int
    seed : int
    keys : list of tuple, optional
        One component key per marginal; defaults to ``(0,), (1,), ...``.
    dims : list of str, default ["component"]
    
    Returns
    -------
    PredictiveDistribution
    
    Examples
    --------
    >>> from prospicio.distributions import Lognormal
    >>> from prospicio.risk import GaussianCopula, simulate
    >>> c = GaussianCopula([[1.0, 0.4], [0.4, 1.0]])
    >>> pd = simulate(c, [Lognormal.from_mean_cv(100.0, 0.2), Lognormal.from_mean_cv(50.0, 1.0)],
    ...               10_000, 42, keys=[("motor",), ("property",)], dims=["lob"])
    >>> abs(pd.mean() - 150.0) < 3.0
    True
    """

def simulate_events(frequency: Any, severity: Any, n_sims: int, seed: int) -> EventSet:
    """
    Simulates ``n_sims`` years of claims: a count from ``frequency``, then that
    many independent losses from ``severity``.
    
    Parameters
    ----------
    frequency : Poisson, NegativeBinomial or Binomial
    severity : Lognormal, Grid, Pareto, PiecewisePareto, LogAffinePareto or GeneralizedPareto
    n_sims : int
    seed : int
    
    Returns
    -------
    EventSet
    
    Raises
    ------
    ValueError
        If ``n_sims`` is 0.
    
    Examples
    --------
    >>> from prospicio.aggregate import simulate_events
    >>> from prospicio.distributions import Lognormal, Poisson
    >>> events = simulate_events(Poisson(5.0), Lognormal.from_mean_cv(1000.0, 1.0), 20_000, 42)
    >>> abs(events.totals().mean() - 5000.0) < 75.0
    True
    """

def simulate_from_means(family: str, means: Sequence[Sequence[float]], n_sims: int, seed: int, dispersion: float |Sequence[float] |None = None, weights: Sequence[float] |None = None, theta: float |None = None, power: float |None = None) -> PredictiveDistribution:
    """
    Joint predictive draws from fitted means, for engines that give only
    a mean per row (the boosting adapters): the family adds process noise
    and several mean vectors (bootstrap refits) add parameter uncertainty.
    
    Simulation ``i`` uses stream ``i`` of ``seed``: it picks one mean vector
    uniformly, then draws each row's response from the family with that
    mean, the row's dispersion and the row's weight. Components are keyed
    ``row = 0, 1, ...``, as ``GlmFit.predict_distribution`` keys them.
    
    Parameters
    ----------
    family : str
        As in ``Glm``.
    means : list of list of float
        One or more mean vectors, one value per row each.
    n_sims : int
    seed : int
    dispersion : float or list of float, optional
        One value for every row, or one per row (from a dispersion model);
        1 by default.
    weights : list of float, optional
        Prior weights; 1 by default.
    theta, power : float, optional
        Negative binomial ``theta``, Tweedie ``power``.
    
    Returns
    -------
    PredictiveDistribution
    
    Examples
    --------
    >>> from prospicio.models import simulate_from_means
    >>> pd = simulate_from_means("poisson", [[0.1, 0.4]], 20_000, 7)
    >>> round(pd.mean(), 1)
    0.5
    """

def stack(ledgers: dict, impacts: Sequence[tuple[str, str, str]] |None = None) -> dict:
    """
    Stack ledgers' results, ``aggregate``'s ``stack_marginal_pnls``: one row
    per ledger with the mean, standard deviation, coefficient of variation,
    skewness and quantiles ``P01`` to ``P99`` of its result, then one row per
    impact, the difference of two stacked rows statistic by statistic.
    
    Parameters
    ----------
    ledgers : dict of str to Ledger
        In row order.
    impacts : list of tuple of (str, str, str), optional
        ``(label, target, base)``: the row ``target`` less the row ``base``.
    
    Returns
    -------
    dict
        Columns ``label``, ``EX``, ``SD``, ``CV``, ``Skew`` and ``P01`` to
        ``P99``.
    
    Raises
    ------
    ValueError
        If an impact names a row that is not stacked.
    
    Examples
    --------
    >>> from prospicio.aggregate import Group, Ledger, Leg, stack
    >>> src = ([2.0, 8.0], [0.5, 0.5])
    >>> a = Ledger(src, [Group("A", "sell", [Leg("P", 10.0)], [Leg("L", lambda x: x)])])
    >>> b = Ledger(src, [Group("B", "sell", [Leg("P", 12.0)], [Leg("L", lambda x: x)])])
    >>> stack({"a": a, "b": b}, [("b less a", "b", "a")])["EX"]
    [5.0, 7.0, 2.0]
    """

def stacking_weights(lpd: Sequence[Sequence[float]]) -> list[float]:
    """
    Stacking weights from pointwise held-out log predictive densities
    (Yao et al., 2018): the weights on the simplex that maximize the log
    score of the mixture of the models' predictive distributions.
    
    Works for any model: pass PSIS-LOO pointwise values (``Elpd.pointwise``)
    for a Bayesian fit, or cross-validated log densities for any other. A
    model that adds nothing gets weight exactly 0.
    
    Parameters
    ----------
    lpd : list of list of float
        One list per model, each with one log density per observation.
    
    Returns
    -------
    list of float
        One weight per model, summing to 1.
    
    Examples
    --------
    >>> from prospicio.models import stacking_weights
    >>> w = stacking_weights([[-0.1, -0.1, -3.0, -3.0], [-3.0, -3.0, -0.1, -0.1]])
    >>> [round(x, 9) for x in w]
    [0.5, 0.5]
    """

def time_ordered(periods: Sequence[int], n_test: int) -> list[tuple[list[int], list[int]]]:
    """
    Time-ordered splits: for each of the last ``n_test`` periods, train on
    earlier periods and test on that one (for a triangle, the calendar
    diagonal backtest).
    
    Parameters
    ----------
    periods : list of int
    n_test : int
    
    Returns
    -------
    list of (list of int, list of int)
    """

def to_json(dist: Any) -> str:
    """
    A distribution as a JSON document: the family and the parameters its
    constructor takes, versioned, numbers bit for bit. ``from_json`` reads
    it back to an equal distribution of the same class.
    
    A ``Custom`` cannot be saved: it is a Python function.
    
    Parameters
    ----------
    dist : a distribution
        Any distribution class, ``Sampled`` and ``Mixture`` included.
    
    Returns
    -------
    str
    
    Raises
    ------
    ValueError
        For a ``Custom``.
    
    Examples
    --------
    >>> from prospicio.distributions import Lognormal, from_json, to_json
    >>> text = to_json(Lognormal(7.0, 0.5))
    >>> from_json(text).mean() == Lognormal(7.0, 0.5).mean()
    True
    """

def trend_factor(rate: float, years: float) -> float:
    """
    ``(1 + rate) ** years``, the factor of an exponential trend.
    
    Parameters
    ----------
    rate : float
        The annual trend, ``0.02`` for +2%.
    years : float
    
    Returns
    -------
    float
    
    Examples
    --------
    >>> from prospicio.pricing import trend_factor
    >>> round(trend_factor(0.02, 2.5), 4)
    1.0508
    """

def trended_present_rates(prior_indicated: float, prior_implemented: float, trend: float) -> float:
    """
    The trended present rates complement as an indicated change:
    ``(1 + prior_indicated) / (1 + prior_implemented) * trend - 1``, with
    ``trend`` the factor from the last review's target effective date to the
    new one (the loss trend over the premium trend for a loss ratio).
    
    Parameters
    ----------
    prior_indicated : float
    prior_implemented : float
    trend : float
    
    Returns
    -------
    float
    
    Examples
    --------
    >>> from prospicio.pricing import trend_factor, trended_present_rates
    >>> c = trended_present_rates(0.10, 0.06, trend_factor(0.05, 2.0))
    >>> round(200 * (1 + c), 2)
    228.82
    """

def wiener_hopf(severity: Grid, wait: Any, rho: float |None = None, *, loss_ratio: float |None = None, points: int |None = None) -> RenewalRuin:
    """
    The probability of eventual ruin when the waits between claims follow
    any law (a renewal, or Sparre Andersen, process), as ``aggregate``'s
    ``wiener_hopf``: claims of this severity, waits drawn from ``wait``,
    and the premium at the rate ``(1 + rho) E[X] / E[W]``. Ruin comes at a
    claim, so ``psi(u)`` is the chance that the walk with steps ``X - cW``
    ever rises above ``u``, found by the cepstral Wiener–Hopf
    factorization. Exponential waits give ``eventual_ruin``.
    
    Parameters
    ----------
    severity : Grid
    wait : distribution
        The law of the wait between claims, with a finite mean.
    rho : float, optional
        The margin to loss, positive. Or give ``loss_ratio``.
    loss_ratio : float, optional
        ``1 / (1 + rho)``, in (0, 1).
    points : int, optional
        Grid points for ``psi``, at least the severity's (the default).
        Raise it when ``psi`` at the last point is not small.
    
    Returns
    -------
    RenewalRuin
    
    Examples
    --------
    >>> from prospicio.aggregate import eventual_ruin, wiener_hopf
    >>> from prospicio.distributions import Gamma, Grid
    >>> sev, _ = Grid.rounding(Gamma.from_mean_cv(10.0, 0.3), 1 / 32, 1 << 14)
    >>> # Gamma waits (shape 2) are more regular than Poisson arrivals.
    >>> wh = wiener_hopf(sev, Gamma(2.0, 1.0), rho=0.2)
    >>> pk = eventual_ruin(sev, rho=0.2)
    >>> wh.psi_at(50.0) < pk.psi_at(50.0)
    True
    """

def year_fraction(year: int, month: int, day: int) -> float:
    """
    The year fraction of a date on Werner and Modlin's half-month grid: the
    first of a month is its start, a day near the middle is half way, and
    the last day is the next month's start (6/30 is 7/1).
    
    Parameters
    ----------
    year : int
    month : int
    day : int
    
    Returns
    -------
    float
    
    Raises
    ------
    ValueError
        If the month or day does not exist.
    
    Examples
    --------
    >>> from prospicio.pricing import year_fraction
    >>> year_fraction(2013, 6, 30) - year_fraction(2011, 1, 1)
    2.5
    """
