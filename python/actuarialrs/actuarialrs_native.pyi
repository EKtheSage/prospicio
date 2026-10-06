from collections.abc import Sequence
from datetime import date
from typing import Any, final

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
    >>> from actuarialrs.risk import ArchimedeanCopula
    >>> c = ArchimedeanCopula("clayton", 2.0, 3)  # Kendall's tau 0.5
    >>> c.dim, c.family
    (3, 'clayton')
    """
    def __new__(cls, /, family: str, theta: float, dim: int) -> ArchimedeanCopula: ...
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
    >>> from actuarialrs.models import BayesGlm, Design
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
    >>> from actuarialrs.distributions import Binomial
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
    >>> from actuarialrs.reserving import ChainLadder, Triangle
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
    >>> from actuarialrs.reserving import ChainLadder, Triangle
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
    >>> from actuarialrs.distributions import Pareto, claim_count
    >>> from actuarialrs.pricing import CollectiveModel
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
    >>> from actuarialrs.models import Design
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
    ``Distortion.proportional_hazard``, ``Distortion.dual_power`` or
    ``Distortion.exponential``. Every one is coherent, and each has a
    parameter value that gives the mean (or a limit that does).
    
    Examples
    --------
    >>> from actuarialrs.distributions import Sampled
    >>> from actuarialrs.risk import Distortion
    >>> x = Sampled([1.0, 2.0, 3.0, 4.0])
    >>> Distortion.tvar(0.5).measure(x)
    3.5
    >>> Distortion.tvar(0.5).weights(4)
    [0.0, 0.0, 0.5, 0.5]
    """
    def __repr__(self, /) -> str: ...
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
    >>> from actuarialrs.models import Design, ElasticNet
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
    def totals(self, /) -> PredictiveDistribution:
        """
        Each year's total loss.
        
        Returns
        -------
        PredictiveDistribution
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
    >>> from actuarialrs.models import Design, Gam, Glm
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
    >>> from actuarialrs.distributions import Gamma
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
    >>> from actuarialrs.risk import GaussianCopula
    >>> c = GaussianCopula([[1.0, 0.5], [0.5, 1.0]])
    >>> u = c.sample(3, seed=1)
    >>> len(u), all(0.0 < x < 1.0 for row in u for x in row)
    (3, True)
    """
    def __new__(cls, /, correlation: Sequence[Sequence[float]]) -> GaussianCopula: ...
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
    
    For tail estimation from draws, see :class:`actuarialrs.risk.Gpd`;
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
    >>> from actuarialrs.distributions import GeneralizedPareto
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
    >>> from actuarialrs.models import Design, Glm
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
        >>> from actuarialrs.models import Design, Glm
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
        >>> from actuarialrs.models import Design, Glm, GlmFit
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
    >>> from actuarialrs.risk import Gpd
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
    >>> from actuarialrs.distributions import Grid, Lognormal
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
        >>> from actuarialrs.distributions import Grid
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
    def variance(self, /) -> float:
        """
        Variance of the distribution.
        
        Returns
        -------
        float
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
    >>> from actuarialrs.models import HierarchicalStacking
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
class Layer:
    """
    A per-occurrence excess-of-loss layer: ``limit`` xs ``attachment`` on each
    loss, then annual terms.
    
    For one year, ``ceded = share * min(max(sum of per-loss recoveries -
    aggregate_deductible, 0), aggregate_limit)``.
    
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
        Upfront premium for the placed share; used only by
        ``reinstatement_rates``.
    reinstatement_rates : list of float, optional
        Paid reinstatements, one rate per reinstatement as a fraction of
        ``premium`` (1.0 is 100%), pro rata as to amount. Sets
        ``aggregate_limit`` to ``limit * (len(reinstatement_rates) + 1)``;
        cannot be combined with ``aggregate_limit`` or ``reinstatements``.
    
    Raises
    ------
    ValueError
        If a term is out of range, or more than one of ``aggregate_limit``,
        ``reinstatements`` and ``reinstatement_rates`` is given.
    
    Examples
    --------
    >>> from actuarialrs.aggregate import Layer
    >>> layer = Layer("5x5", 5e6, 5e6, reinstatements=1)
    >>> layer.ceded([7e6])
    2000000.0
    >>> layer.ceded([12e6, 20e6, 30e6])
    10000000.0
    >>> paid = Layer("10x10", 10.0, 10.0, premium=2.0, reinstatement_rates=[1.0, 0.5])
    >>> paid.reinstatement_premium([22.0, 12.0])
    2.2
    """
    def __new__(cls, /, name: str, limit: float, attachment: float, share: float = 1.0, aggregate_deductible: float = 0.0, aggregate_limit: float |None = None, reinstatements: int |None = None, premium: float = 0.0, reinstatement_rates: Sequence[float] |None = None) -> Layer: ...
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
        >>> from actuarialrs.aggregate import Layer
        >>> layer = Layer("L", 10.0, 5.0, aggregate_deductible=4.0, aggregate_limit=15.0)
        >>> layer.ceded_by_event([8.0, 20.0, 12.0])
        [0.0, 9.0, 6.0]
        """
    @property
    def limit(self, /) -> float:
        """
        Per-occurrence limit.
        """
    @property
    def name(self, /) -> str:
        """
        Layer name.
        """
    @property
    def premium(self, /) -> float:
        """
        Upfront premium for the placed share.
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
        >>> from actuarialrs.aggregate import Layer
        >>> Layer.quota_share("QS", 0.4).ceded([10.0, 5.0])
        6.0
        """
    def reinstatement_premium(self, /, losses: Sequence[float]) -> float:
        """
        Reinstatement premium for one year's losses.
        
        With layer loss ``L`` at 100% after annual terms, ``premium *
        sum(rate_k * min(max(L - k * limit, 0), limit) / limit)``; zero when
        reinstatements are free.
        
        Parameters
        ----------
        losses : list of float
        
        Returns
        -------
        float
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
        >>> from actuarialrs.aggregate import Layer
        >>> Layer.stop_loss("SL", 50.0, 100.0).ceded([60.0, 70.0])
        30.0
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
    >>> from actuarialrs.distributions import LogAffinePareto
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
    >>> from actuarialrs.distributions import Loglogistic
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
    >>> from actuarialrs.distributions import Lognormal
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
        >>> from actuarialrs.distributions import Lognormal
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
    >>> from actuarialrs.reserving import Mack, Triangle
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
    >>> from actuarialrs.distributions import Lognormal, Mixture, Pareto
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
    >>> from actuarialrs.distributions import NegativeBinomial
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
    
    Raises
    ------
    ValueError
        If ``n_sims`` is zero or ``process`` is unknown.
    OverflowError
        If ``n_sims`` or ``seed`` is negative or too large.
    
    Examples
    --------
    >>> from actuarialrs.reserving import OdpBootstrap, Triangle
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
    def __new__(cls, /, n_sims: int = 10000, seed: int = 0, process: str = "gamma") -> OdpBootstrap: ...
    def __repr__(self, /) -> str: ...
    def fit(self, /, triangle: Triangle, column: str) -> OdpBootstrapFit:
        """
        Bootstraps one measure column in every segment of a cumulative
        triangle, each with its own residuals and scale, into one joint
        distribution of the reserves. Every origin must be observed from the
        first age up to its latest.
        
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
            latest age or a segment has too few observed cells for the
            degrees of freedom to be positive.
        """
    @property
    def n_sims(self, /) -> int:
        """
        Number of simulations.
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
        centred on.
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
    >>> from actuarialrs.distributions import Pareto
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
        >>> from actuarialrs.distributions import Pareto
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
    >>> from actuarialrs.distributions import PiecewisePareto
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
    >>> from actuarialrs.distributions import Poisson
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
    >>> from actuarialrs.distributions import Lognormal, Sampled
    >>> from actuarialrs.risk import PotTail
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
    >>> from actuarialrs.distributions import PredictiveDistribution
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
        >>> from actuarialrs.distributions import PredictiveDistribution
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
        >>> from actuarialrs.distributions import PredictiveDistribution
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
        crate versions and input hash.
        
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
    >>> from actuarialrs.distributions import Sampled
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
    >>> from actuarialrs.risk import StudentTCopula
    >>> StudentTCopula([[1.0, 0.5], [0.5, 1.0]], 4.0).dim
    2
    """
    def __new__(cls, /, correlation: Sequence[Sequence[float]], nu: float) -> StudentTCopula: ...
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
    >>> from actuarialrs.reserving import ChainLadder, TailBondy, Triangle
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
    >>> from actuarialrs.reserving import ChainLadder, TailConstant, Triangle
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
    >>> from actuarialrs.reserving import ChainLadder, TailCurve, Triangle
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
    >>> from actuarialrs.reserving import Mack, TailLogLinear, Triangle
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
    >>> from actuarialrs.models import Terms
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
    >>> from actuarialrs.aggregate import Layer, Tower, simulate_events
    >>> from actuarialrs.distributions import Lognormal, Poisson
    >>> events = simulate_events(Poisson(2.0), Lognormal.from_mean_cv(3e6, 1.5), 1_000, 7)
    >>> tower = Tower([Layer("5x5", 5e6, 5e6), Layer("15x10", 15e6, 10e6)])
    >>> result = tower.apply(events)
    >>> [k[0] for k in result.aggregate(["kind"]).components()]
    ['gross', 'ceded', 'net']
    """
    def __new__(cls, /, layers: Sequence[Layer]) -> Tower: ...
    def __repr__(self, /) -> str: ...
    def apply(self, /, events: EventSet) -> PredictiveDistribution:
        """
        Applies the tower to every simulated year.
        
        The result has dimensions ``["kind", "layer"]``: ``("gross",
        "ground_up")``, ``("ceded", name)`` per layer, ``("net",
        "retained")``, then ``("reinstatement_premium", name)`` per layer with
        paid reinstatements. ``aggregate(["kind"])`` gives gross, total ceded
        and net; net is a loss, before premiums.
        
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
        >>> from actuarialrs.aggregate import Layer, Tower
        >>> tower = Tower.inuring([[Layer.quota_share("QS", 0.5)], [Layer("5x5", 5.0, 5.0)]])
        >>> tower.ceded([30.0])
        [15.0, 5.0]
        """
    @property
    def layer_names(self, /) -> list[str]:
        """
        Layer names, in order.
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
        >>> from actuarialrs.aggregate import Layer, Tower
        >>> from actuarialrs.distributions import Grid, Poisson
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
    >>> from actuarialrs.reserving import Triangle
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
        >>> from actuarialrs.reserving import Triangle
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
        >>> from actuarialrs.reserving import Triangle
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
        >>> from actuarialrs.reserving import Triangle
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
        >>> from actuarialrs.reserving import Triangle
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
        >>> from actuarialrs.reserving import Triangle
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
        >>> from actuarialrs.reserving import Triangle
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
        >>> from actuarialrs.reserving import Triangle
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
        >>> from actuarialrs.reserving import Triangle
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
        >>> from actuarialrs.reserving import Triangle
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
        >>> from actuarialrs.reserving import Triangle
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
    >>> from actuarialrs.distributions import Tweedie
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
    >>> from actuarialrs.distributions import Weibull
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
    >>> from actuarialrs.models import actual_vs_expected
    >>> m = actual_vs_expected([2023, 2023, 2024, 2024], "poisson",
    ...                        [1.0, 3.0, 2.0, 6.0], [2.0, 2.0, 2.0, 2.0])
    >>> m["periods"][1]["ratio"], m["periods"][1]["z"]
    (2.0, 2.0)
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
    >>> from actuarialrs.distributions import PredictiveDistribution
    >>> from actuarialrs.risk import Distortion, allocate
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
    >>> from actuarialrs.pricing import alpha_between_frequencies
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
    >>> from actuarialrs.distributions import PredictiveDistribution
    >>> from actuarialrs.risk import Distortion, capital
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
    >>> from actuarialrs.distributions import claim_count
    >>> claim_count(4.0, 2.5)
    NegativeBinomial(r=2.6666666666666665, beta=1.5)
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
    >>> from actuarialrs.distributions import PredictiveDistribution
    >>> from actuarialrs.risk import covar
    >>> pd = PredictiveDistribution(["lob"], [("a",), ("b",)],
    ...                             [[1.0, 0.0], [2.0, 1.0], [3.0, 5.0], [4.0, 1.0]])
    >>> covar(pd, ("a",), 0.75, 0.5)
    5.0
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
    >>> from actuarialrs.risk import entropic
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
    >>> from actuarialrs.risk import esscher
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
    >>> from actuarialrs.distributions import PredictiveDistribution
    >>> from actuarialrs.risk import esscher_allocation
    >>> pd = PredictiveDistribution(["lob"], [("motor",), ("property",)],
    ...                             [[1.0, 2.0], [4.0, 1.0], [2.0, 5.0], [3.0, 6.0]])
    >>> esscher_allocation(pd, 0.0)
    [2.5, 3.5]
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
    >>> from actuarialrs.pricing import fit_references
    >>> m = fit_references([(1000.0, 1000.0, 150.0), (3000.0, 1500.0, 160.0)], [(1000.0, 0.3)])
    >>> round(m.layer_loss(3000.0, 1500.0), 6)
    160.0
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
    >>> from actuarialrs.distributions import Pareto
    >>> from actuarialrs.pricing import ilf
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
    >>> from actuarialrs.distributions import PredictiveDistribution
    >>> from actuarialrs.risk import iman_conover
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
    >>> from actuarialrs.distributions import local_pareto_to_piecewise
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
    >>> from actuarialrs.models import log_score
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
    >>> from actuarialrs.models import lppd
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
    >>> from actuarialrs.distributions import PredictiveDistribution
    >>> from actuarialrs.risk import marginal_expected_shortfall
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
    >>> from actuarialrs.pricing import match_tower
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
    >>> from actuarialrs.models import mcmc_diagnostics
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
    >>> from actuarialrs.risk import mean_excess
    >>> mean_excess([1.0, 2.0, 3.0, 4.0], [2.0])
    [(2.0, 1.5, 2)]
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
    >>> from actuarialrs.aggregate import panjer
    >>> from actuarialrs.distributions import Grid, Poisson
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
    >>> from actuarialrs.pricing import pareto_extrapolation
    >>> round(pareto_extrapolation((1e6, 1e6), (2e6, 2e6), 2.0), 12)
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

def price(losses: Any, assets: Distortion, *, cost_of_capital: float |None = None, distortion: Distortion |None = None) -> Price:
    """
    Risk-loaded price of a cover from its simulated losses.
    
    The assets backing the loss are a distortion risk measure of it. The
    premium is either a pricing distortion of the loss, or set by a
    constant cost of capital ``r`` on the capital ``a - P``, which gives
    ``P = (E[X] + r a) / (1 + r)``.
    
    Parameters
    ----------
    losses : Sampled or PredictiveDistribution
        Loss draws; for a ``PredictiveDistribution``, its total.
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
    >>> from actuarialrs.distributions import Sampled
    >>> from actuarialrs.pricing import price
    >>> from actuarialrs.risk import Distortion
    >>> p = price(Sampled([0.0, 0.0, 2.0, 6.0]), Distortion.tvar(0.5), cost_of_capital=0.25)
    >>> p.premium, p.capital
    (2.4, 1.6)
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
    >>> from actuarialrs.distributions import PredictiveDistribution
    >>> from actuarialrs.pricing import price_portfolio
    >>> from actuarialrs.risk import Distortion
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
    >>> from actuarialrs.distributions import Lognormal
    >>> from actuarialrs.risk import GaussianCopula, simulate
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
    >>> from actuarialrs.aggregate import simulate_events
    >>> from actuarialrs.distributions import Lognormal, Poisson
    >>> events = simulate_events(Poisson(5.0), Lognormal.from_mean_cv(1000.0, 1.0), 20_000, 42)
    >>> abs(events.totals().mean() - 5000.0) < 75.0
    True
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
    >>> from actuarialrs.models import stacking_weights
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
