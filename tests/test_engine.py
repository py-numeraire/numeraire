"""Integration tests for the walk-forward OOS engine on a synthetic timing estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import make_monthly_view
from numeraire.core import capabilities
from numeraire.core.data import TimeSeriesView
from numeraire.core.engine import (
    ForecastOutput,
    WeightsOutput,
    config_hash,
    walk_forward,
    walk_forward_forecast,
)
from numeraire.core.splitter import WalkForwardSplitter


class _OLSModel:
    """Sign-of-forecast timing model: long when the OLS forecast is positive."""

    def __init__(self, beta: np.ndarray) -> None:
        self._beta = beta

    def capabilities(self) -> set[str]:
        return {capabilities.TO_WEIGHTS}

    def to_weights(self, view: TimeSeriesView) -> pd.DataFrame:
        rows: list[np.ndarray] = []
        for t in view.calendar:
            x = view.features_asof(t)
            pred = np.concatenate([[1.0], x]) @ self._beta
            rows.append(np.sign(pred))
        return pd.DataFrame(np.vstack(rows), index=view.calendar, columns=view.assets)


class _OLSTimingEstimator:
    """Fits an OLS forecast of the (t, t+h] return on features over the train window."""

    def fit(self, view: TimeSeriesView) -> _OLSModel:
        _, x, y = view.aligned()
        xi = np.column_stack([np.ones(len(x)), x])
        beta, _res, _rank, _sv = np.linalg.lstsq(xi, y, rcond=None)
        return _OLSModel(beta)


def test_config_hash_is_deterministic() -> None:
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_walk_forward_produces_aligned_output() -> None:
    v = make_monthly_view(n=120, n_assets=1, n_features=2, horizon=1, seed=7)
    sp = WalkForwardSplitter(min_train=60, test_size=12)
    out = walk_forward(_OLSTimingEstimator(), v, sp, method="ols_timing", config={"h": 1})

    assert isinstance(out, WeightsOutput)
    assert out.capability == capabilities.TO_WEIGHTS
    assert out.method == "ols_timing"
    assert out.run_id == f"ols_timing-{out.config_hash}"
    # weights and realized share an index; nothing unrealized survives
    assert out.weights.index.equals(out.realized.index)
    assert not out.realized.isna().to_numpy().any()
    assert out.weights.index.is_monotonic_increasing


def test_strategy_returns_match_manual() -> None:
    v = make_monthly_view(n=90, horizon=1, seed=3)
    sp = WalkForwardSplitter(min_train=48, test_size=12)
    out = walk_forward(_OLSTimingEstimator(), v, sp, method="ols_timing")
    sr = out.strategy_returns()
    manual = (out.weights.to_numpy() * out.realized.to_numpy()).sum(axis=1)
    np.testing.assert_allclose(sr.to_numpy(), manual)


def test_horizon_aware_realized_uses_compounded_return() -> None:
    v = make_monthly_view(n=90, horizon=3, seed=1)
    sp = WalkForwardSplitter(min_train=48, test_size=12)
    out = walk_forward(_OLSTimingEstimator(), v, sp, method="ols_timing")
    # each realized entry must equal the view's 3-period target for that date
    for t in out.realized.index:
        np.testing.assert_allclose(out.realized.loc[t].to_numpy(), v.target_asof(t, horizon=3))


class _MeanModel:
    """Forecasts the window historical mean (the GW benchmark) — for engine plumbing tests."""

    def capabilities(self) -> set[str]:
        return {capabilities.TO_FORECAST}

    def forecast(self, view: TimeSeriesView) -> pd.Series:
        return view.returns_frame().mean()


class _MeanEstimator:
    def fit(self, view: TimeSeriesView) -> _MeanModel:
        _ = view
        return _MeanModel()


def test_walk_forward_forecast_expanding_origin_alignment() -> None:
    v = make_monthly_view(n=60, n_assets=1, horizon=1, seed=2)
    out = walk_forward_forecast(_MeanEstimator(), v, min_train=24, method="mean")
    assert isinstance(out, ForecastOutput)
    # expanding warm-up of 24, h=1 -> origins at index 23..58 -> 36 forecasts
    assert len(out.forecasts) == 36
    assert out.forecasts.index.equals(out.benchmark.index)
    # this model forecasts exactly the window mean, so forecast == engine benchmark everywhere
    np.testing.assert_allclose(out.forecasts.to_numpy(), out.benchmark.to_numpy())
    assert not out.realized.isna().to_numpy().any()


def test_walk_forward_forecast_rolling_window_is_bounded() -> None:
    v = make_monthly_view(n=80, horizon=1, seed=5)
    out = walk_forward_forecast(_MeanEstimator(), v, window=12, method="mean")
    # rolling 12, h=1 -> origins index 11..78 -> 68 forecasts; benchmark is the 12-obs mean
    assert len(out.forecasts) == 68
    assert out.forecasts.index.min() == v.calendar[11]
