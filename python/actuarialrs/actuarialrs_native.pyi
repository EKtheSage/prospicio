from typing import final

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
    def variance(self, /) -> float:
        """
        Variance of the distribution.
        
        Returns
        -------
        float
        """
