"""Actuarial modeling on a Rust core.

User-facing namespaces follow docs/architecture.md: ``distributions``,
``aggregate``, ``models`` (with ``boosting``), ``pricing``, ``reserving`` and ``risk``
so far.
"""

from . import aggregate, boosting, distributions, models, pricing, reserving, risk

__all__ = ["aggregate", "boosting", "distributions", "models", "pricing", "reserving", "risk"]
