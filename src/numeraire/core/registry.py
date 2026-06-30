"""Open registries over closed enums (SPEC §2.5).

A minimal evaluator registry. Bundled native evaluators register here; external packages
register the same way (directly or via the ``numeraire.methods`` entry-point group), so a
new method/evaluator never requires editing core.
"""

from __future__ import annotations

from numeraire.core.protocols import Evaluator

_EVALUATORS: dict[str, Evaluator] = {}


def register_evaluator(name: str, evaluator: Evaluator, *, overwrite: bool = False) -> None:
    """Register ``evaluator`` under ``name``. Raises if the name exists unless ``overwrite``."""
    if not overwrite and name in _EVALUATORS:
        raise KeyError(f"evaluator {name!r} already registered")
    _EVALUATORS[name] = evaluator


def get_evaluator(name: str) -> Evaluator:
    """Return the evaluator registered under ``name``."""
    try:
        return _EVALUATORS[name]
    except KeyError:
        raise KeyError(f"no evaluator registered as {name!r}") from None


def available_evaluators() -> tuple[str, ...]:
    """Return the names of all registered evaluators, sorted."""
    return tuple(sorted(_EVALUATORS))
