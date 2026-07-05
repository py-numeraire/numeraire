"""skfolio adapter — wrap a skfolio portfolio optimizer as a numeraire ``to_weights`` estimator.

`skfolio <https://skfolio.org>`_ (BSD-3) provides mean-risk / hierarchical-risk-parity /
risk-budgeting optimizers as scikit-learn estimators. This adapter makes one conform to the
numeraire ``Estimator`` / ``Model`` protocol so it plugs into the walk-forward engine as a peer of
any native method — **without** adopting skfolio's own cross-validation or walk-forward machinery
(numeraire owns the out-of-sample loop).

The contract (why this stays leak-free):

- ``fit(view)`` fits the skfolio estimator on ``view.returns_frame()`` — the exact training window
  the engine hands it — and stores the fitted ``weights_``.
- ``to_weights(view)`` **broadcasts** those fitted weights across the view's calendar. It never
  calls ``estimator.predict(X_test)``: skfolio's ``predict`` scores a weight vector *on the returns
  it is given*, so feeding it the test window would pour realized test returns into the position —
  a structural look-ahead. Weights come only from ``.weights_`` (a function of the fit window).

Through ``backtest_weights`` the estimator is re-fit at each origin on that origin's PIT window and
the resulting weights are applied to the next period, so the broadcast is per-origin and
point-in-time.
The optional ``window`` caps the lookback to the most recent ``window`` rows of whatever the engine
hands ``fit`` (e.g. a rolling estimation window under an expanding split).

skfolio is an **optional** dependency (the ``[skfolio]`` extra); it is imported lazily inside
``fit`` so this module imports with or without it installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from numeraire.core import capabilities
from numeraire.core.data import TimeSeriesView
from numeraire.core.protocols import DataView


def _as_tsv(view: DataView) -> TimeSeriesView:
    if not isinstance(view, TimeSeriesView):
        raise TypeError("the skfolio adapter runs on a TimeSeriesView (asset returns block)")
    return view


class _SkfolioModel:
    """A fitted skfolio portfolio: a single optimal weight vector, broadcast across a calendar."""

    def __init__(self, weights: pd.Series, meta: dict[str, Any]) -> None:
        self._weights = weights  # index = fitted asset labels
        self.meta = meta

    def capabilities(self) -> set[str]:
        return {capabilities.TO_WEIGHTS}

    def to_weights(self, view: DataView) -> pd.DataFrame:
        tsv = _as_tsv(view)
        assets = [str(a) for a in tsv.assets]
        w = self._weights.reindex(assets).to_numpy(dtype=np.float64)
        if np.isnan(w).any():
            missing = [a for a, x in zip(assets, w, strict=True) if np.isnan(x)]
            raise ValueError(f"fitted skfolio weights do not cover view assets {missing}")
        vals = np.repeat(w[None, :], len(tsv.calendar), axis=0)
        return pd.DataFrame(vals, index=tsv.calendar, columns=tsv.assets)


class SkfolioWeights:
    """Adapt a skfolio optimizer to numeraire ``to_weights``.

    ``estimator`` is a skfolio estimator instance (e.g. ``MeanRisk()``, ``RiskBudgeting()``,
    ``HierarchicalRiskParity()``); it is cloned per fit so each origin gets a fresh optimization.
    When ``None``, a default ``skfolio.optimization.MeanRisk`` is used. ``window`` optionally caps
    the estimation lookback to the most recent rows of the fit view.
    """

    def __init__(self, estimator: Any | None = None, *, window: int | None = None) -> None:
        self.estimator = estimator
        self.window = window

    def fit(self, view: DataView) -> _SkfolioModel:
        import skfolio
        from sklearn.base import clone

        tsv = _as_tsv(view)
        if self.estimator is None:
            from skfolio.optimization import MeanRisk

            est = MeanRisk()
        else:
            est = clone(self.estimator)
        returns = tsv.returns_frame()
        if self.window is not None:
            returns = returns.tail(self.window)
        est.fit(returns)
        weights = pd.Series(
            np.asarray(est.weights_, dtype=np.float64).ravel(),
            index=[str(a) for a in tsv.assets],
        )
        meta = {
            "adapter": "skfolio",
            "skfolio_version": skfolio.__version__,
            "estimator": type(est).__name__,
            "solver": getattr(est, "solver", None),
            "n_train": len(returns),
        }
        return _SkfolioModel(weights, meta)
