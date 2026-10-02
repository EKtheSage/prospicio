"""Actuarial modeling on a Rust core.

User-facing namespaces follow docs/architecture.md: ``distributions``,
``aggregate`` and ``risk`` so far.
"""

from . import aggregate, distributions, risk

__all__ = ["aggregate", "distributions", "risk"]
