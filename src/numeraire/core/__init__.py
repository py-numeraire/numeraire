"""numeraire.core — the stable, method-agnostic spine.

Contains only code that depends on no specific method and that every method depends on.
Dependency arrows point *toward* core; core never imports ``numeraire.methods``,
``numeraire.adapters``, or any reference library. This is enforced in CI by import-linter.
"""

from __future__ import annotations

# Importing evaluators registers the bundled native evaluators (open registry, §2.5).
from numeraire.core import evaluators as evaluators

__all__ = ["evaluators"]
