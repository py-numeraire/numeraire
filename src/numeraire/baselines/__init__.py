"""Universal benchmarks bundled in ``numeraire`` — the reference rules every method is compared to.

Four estimators, all framework citizens (they pass ``numeraire.testing.check_estimator``) and all
registered via the ``numeraire.methods`` entry-point group (dogfooding the open discovery):

- :class:`EqualWeight` — 1/N ``to_weights`` (the naive benchmark).
- :class:`MinVariance` — global minimum-variance ``to_weights`` (sample covariance + window).
- :class:`MeanVariance` — plug-in mean-variance ``to_weights`` (``S^-1 mu``, explicit norm).
- :class:`HistoricalMean` — prevailing-historical-mean ``to_forecast`` (Goyal-Welch benchmark).

The pure weight functions (:func:`equal_weights`, :func:`minimum_variance_weights`,
:func:`mean_variance_weights`) are the single source of truth for these formulae — method packages
(e.g. the naive-diversification reproduction) build on them rather than re-deriving the algebra.

Serious constrained optimizers are **not** re-implemented here; they arrive through the optional
skfolio adapter (``numeraire.adapters``). These baselines are the always-available floor.
"""

from __future__ import annotations

from numeraire.baselines.forecast import HistoricalMean
from numeraire.baselines.weights import (
    EqualWeight,
    MeanVariance,
    MinVariance,
    equal_weights,
    mean_variance_weights,
    minimum_variance_weights,
)

__all__ = [
    "EqualWeight",
    "HistoricalMean",
    "MeanVariance",
    "MinVariance",
    "equal_weights",
    "mean_variance_weights",
    "minimum_variance_weights",
]
