"""The historical-mean forecast baseline — the Goyal-Welch OOS reference.

The prevailing (expanding) historical mean of the returns block is the benchmark every predictive
regression is scored against: Goyal-Welch (2008) show it is a stubbornly hard forecast to beat out
of sample, and the OOS R^2 in :class:`~numeraire.core.evaluators.OutOfSampleR2Evaluator` measures
MSE
improvement *relative to it*. The walk-forward forecast engine already computes exactly this
benchmark column for free at each origin (``train.returns_frame().mean(axis=0)``); this estimator
exposes the very same quantity as a first-class ``to_forecast`` citizen, so it can be compared,
registered and run through the engine like any other method (its OOS R^2 against the engine
benchmark is ~0 by construction — it *is* the benchmark).

``window`` gives the rolling variant (last ``k`` observations); ``None`` (default) is the expanding
prevailing mean, matching the engine's benchmark convention.
"""

from __future__ import annotations

import pandas as pd

from numeraire.core import capabilities
from numeraire.core.data import TimeSeriesView
from numeraire.core.protocols import DataView


def _as_tsv(view: DataView) -> TimeSeriesView:
    if not isinstance(view, TimeSeriesView):
        raise TypeError("HistoricalMean requires a TimeSeriesView (asset-returns block)")
    return view


class _HistoricalMeanModel:
    """Fitted historical-mean model: forecasts the per-asset sample mean over the fit window."""

    def __init__(self, window: int | None) -> None:
        self._window = window

    def capabilities(self) -> set[str]:
        return {capabilities.TO_FORECAST}

    def forecast(self, view: DataView) -> pd.Series:
        tsv = _as_tsv(view)
        rets = tsv.returns_frame()
        if self._window is not None:
            rets = rets.tail(self._window)
        # Per-asset prevailing mean, computed exactly as the engine's benchmark column so the two
        # stay bit-identical (OOS R^2 vs the benchmark is exactly zero by construction). For a
        # multi-period view the target is the compounded (t, t+h] return, so the mean is compounded
        # to the same horizon ((1 + mu)^h - 1); h = 1 returns the mean itself, untouched.
        mu = rets.to_numpy(dtype=float).mean(axis=0)
        h = tsv.horizon
        values = mu if h == 1 else (1.0 + mu) ** h - 1.0
        return pd.Series(values, index=[str(c) for c in rets.columns])


class HistoricalMean:
    """The prevailing-historical-mean forecaster (Goyal-Welch benchmark).

    Parameters
    ----------
    window:
        Trailing window (in calendar steps) for the mean; ``None`` (default) is the expanding
        prevailing mean — the same quantity the walk-forward engine uses as its OOS R^2 benchmark.
    """

    def __init__(self, *, window: int | None = None) -> None:
        if window is not None and window < 1:
            raise ValueError("window must be >= 1")
        self.window = window

    def fit(self, view: DataView) -> _HistoricalMeanModel:
        _as_tsv(view)
        return _HistoricalMeanModel(self.window)
