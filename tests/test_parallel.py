"""Parallel execution must be bit-for-bit identical to serial.

The walk-forward folds are pure functions of ``(estimator, train, test)`` and results are
reassembled in fold order, so ``n_jobs != 1`` (thread pool) reproduces ``n_jobs=1`` exactly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import make_monthly_view, toy_panel_wide
from numeraire.core import capabilities
from numeraire.core.data import CrossSectionView, TimeSeriesView
from numeraire.core.engine import (
    _even_chunks,
    _map_folds,
    _resolve_workers,
    backtest_forecast,
    backtest_panel,
    backtest_weights,
)
from numeraire.core.splitter import WalkForwardSplitter


class _OLSModel:
    def __init__(self, beta: np.ndarray) -> None:
        self._beta = beta

    def capabilities(self) -> set[str]:
        return {capabilities.TO_WEIGHTS}

    def to_weights(self, view: TimeSeriesView) -> pd.DataFrame:
        rows = [
            np.sign(np.concatenate([[1.0], view.features_asof(t)]) @ self._beta)
            for t in view.calendar
        ]
        return pd.DataFrame(np.vstack(rows), index=view.calendar, columns=view.assets)


class _OLSTimingEstimator:
    def fit(self, view: TimeSeriesView) -> _OLSModel:
        _, x, y = view.aligned()
        xi = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(xi, y, rcond=None)
        return _OLSModel(beta)


class _MeanModel:
    def capabilities(self) -> set[str]:
        return {capabilities.TO_FORECAST}

    def forecast(self, view: TimeSeriesView) -> pd.Series:
        return view.returns_frame().mean()


class _MeanEstimator:
    def fit(self, view: TimeSeriesView) -> _MeanModel:
        _ = view
        return _MeanModel()


class _XSModel:
    def __init__(self, beta: np.ndarray) -> None:
        self._beta = beta

    def capabilities(self) -> set[str]:
        return {capabilities.TO_WEIGHTS}

    def to_weights(self, view: CrossSectionView) -> pd.Series:
        dates: list[pd.Timestamp] = []
        assets: list[object] = []
        vals: list[float] = []
        for t in view.calendar:
            ids, x = view.features_asof(t)
            w = x @ self._beta
            w = w - w.mean()
            norm = float(np.abs(w).sum())
            if norm > 0:
                w = w / norm
            for a, wi in zip(ids, w, strict=True):
                dates.append(t)
                assets.append(a)
                vals.append(float(wi))
        idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex(dates), assets], names=["date", "asset"])
        return pd.Series(vals, index=idx, name="weight")


class _XSEstimator:
    def fit(self, view: CrossSectionView) -> _XSModel:
        _keys, x, y = view.aligned()
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        return _XSModel(beta)


def test_even_chunks_partition_is_contiguous_and_complete() -> None:
    items = list(range(23))
    for k in (1, 3, 4, 8, 50):
        chunks = _even_chunks(items, k)
        assert [x for c in chunks for x in c] == items  # order-preserving, no drops/dups
        assert all(len(c) > 0 for c in chunks)  # no empty chunks
        sizes = [len(c) for c in chunks]
        assert max(sizes) - min(sizes) <= 1  # near-even
        assert len(chunks) == min(len(items), k)


def test_map_folds_batched_matches_serial() -> None:
    items = list(range(100))
    square = lambda x: x * x  # noqa: E731
    serial = _map_folds(square, items, n_jobs=1)
    for n_jobs in (2, 4, -1):
        assert _map_folds(square, items, n_jobs=n_jobs) == serial  # order + values preserved
    assert _map_folds(square, [], n_jobs=4) == []  # empty is a no-op


def test_resolve_workers() -> None:
    assert _resolve_workers(4) == 4
    assert _resolve_workers(-1) >= 1
    import pytest

    with pytest.raises(ValueError, match="n_jobs"):
        _resolve_workers(0)


def test_walk_forward_parallel_matches_serial() -> None:
    v = make_monthly_view(n=180, n_features=2, seed=7)
    sp = WalkForwardSplitter(min_train=60, test_size=12)
    serial = backtest_weights(_OLSTimingEstimator(), v, sp, method="ols")
    parallel = backtest_weights(_OLSTimingEstimator(), v, sp, method="ols", n_jobs=4)
    pd.testing.assert_frame_equal(serial.weights, parallel.weights)
    pd.testing.assert_frame_equal(serial.realized, parallel.realized)
    np.testing.assert_array_equal(
        serial.strategy_returns().to_numpy(), parallel.strategy_returns().to_numpy()
    )


def test_walk_forward_forecast_parallel_matches_serial() -> None:
    v = make_monthly_view(n=90, n_assets=1, seed=2)
    serial = backtest_forecast(_MeanEstimator(), v, min_train=24, method="mean")
    parallel = backtest_forecast(_MeanEstimator(), v, min_train=24, method="mean", n_jobs=-1)
    pd.testing.assert_frame_equal(serial.forecasts, parallel.forecasts)
    pd.testing.assert_frame_equal(serial.realized, parallel.realized)
    pd.testing.assert_frame_equal(serial.benchmark, parallel.benchmark)


def test_walk_forward_panel_parallel_matches_serial() -> None:
    v = CrossSectionView(toy_panel_wide(), chars=["size", "bm", "mom"], horizon=1)
    sp = WalkForwardSplitter(min_train=24, test_size=6)
    serial = backtest_panel(_XSEstimator(), v, sp, method="fm")
    parallel = backtest_panel(_XSEstimator(), v, sp, method="fm", n_jobs=3)
    pd.testing.assert_series_equal(serial.weights, parallel.weights)
    pd.testing.assert_series_equal(serial.realized, parallel.realized)
