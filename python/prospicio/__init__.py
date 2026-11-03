"""Actuarial modeling on a Rust core.

User-facing namespaces follow docs/architecture.md: ``distributions``,
``aggregate``, ``reinsurance``, ``models`` (with ``boosting``), ``pricing``,
``credibility``, ``reserving`` and ``risk`` so far.
"""

from . import (
    aggregate,
    boosting,
    credibility,
    distributions,
    models,
    pricing,
    reinsurance,
    reserving,
    risk,
)
from .reserving import NotSupportedError

__all__ = ["NotSupportedError", "aggregate", "boosting", "credibility", "distributions", "models",
           "pricing", "reinsurance", "reserving", "risk"]
