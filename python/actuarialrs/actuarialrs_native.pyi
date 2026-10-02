from collections.abc import Sequence
from typing import Any, final

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
        severity : Lognormal or Grid
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
        severity : Lognormal or Grid
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
        severity : Lognormal or Grid
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
