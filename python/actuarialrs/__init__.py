"""Actuarial modeling on a Rust core.

User-facing namespaces follow docs/architecture.md: ``distributions``,
``aggregate``, ``pricing`` and ``risk`` so far.
"""

from . import aggregate, distributions, pricing, risk

__all__ = ["aggregate", "distributions", "pricing", "risk"]
