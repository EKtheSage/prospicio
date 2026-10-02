"""Actuarial modeling on a Rust core.

User-facing namespaces follow docs/architecture.md: ``distributions`` and
``aggregate`` so far.
"""

from . import aggregate, distributions

__all__ = ["aggregate", "distributions"]
