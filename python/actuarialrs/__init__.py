"""Actuarial modeling on a Rust core.

User-facing namespaces follow docs/architecture.md; Phase 0 exposes
``distributions`` only.
"""

from . import distributions

__all__ = ["distributions"]
