from collections.abc import Sequence
from typing import Any, final

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
    ``Distortion.proportional_hazard`` or ``Distortion.dual_power``. Every
    one is coherent, and each has a parameter value that gives the mean.
    
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
    def marginal(self, /, key: Sequence[int |str]) -> Sampled |None:
        """
        One component's draws, or ``None`` if no component has this key.
        
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
    @property
    def stages(self, /) -> list[int]:
        """
        Stage of each layer, in order, starting at 0.
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
