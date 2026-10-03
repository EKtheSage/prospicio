"""Actuarial modeling on a Rust core.

User-facing namespaces follow docs/architecture.md: ``distributions``,
``aggregate``, ``models``, ``pricing`` and ``risk`` so far.
"""

from . import aggregate, distributions, models, pricing, risk

__all__ = ["aggregate", "distributions", "models", "pricing", "risk"]
