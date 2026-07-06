"""Combination + invariant tests over the toy-data catalog (conftest ``toy_*``).

Where the per-feature unit tests (``test_data`` / ``test_multiblock_view`` / ``test_vintaged``)
each exercise one mechanism, these mount several *data shapes* together through the real
:class:`TimeSeriesView` (and the walk-forward engine) and assert the cross-cutting invariants:
concatenation order across heterogeneous blocks, horizon pairing, raw->excess conversion, and the
no-look-ahead property that the whole stack must preserve.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import (
    toy_assets,
    toy_macro_block,
    toy_market,
    toy_predictors,
    toy_vintaged_block,
    toy_vintaged_table,
)
from numeraire.core import capabilities
from numeraire.core.data import FeatureBlock, TimeSeriesView, VintagedBlock
from numeraire.core.engine import WeightsOutput, backtest_weights
from numeraire.core.splitter import WalkForwardSplitter


class _SignTimingModel:
    """Long/short the single asset on the sign of an OLS forecast (toy timing model)."""

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
    """Fits OLS of the (t, t+h] return on the lag-aware features over the train window."""

    def fit(self, view: TimeSeriesView) -> _SignTimingModel:
        _, x, y = view.aligned()
        xi = np.column_stack([np.ones(len(x)), x])
        beta, *_ = np.linalg.lstsq(xi, y, rcond=None)
        return _SignTimingModel(beta)


def _combined_view(horizon: int = 1) -> TimeSeriesView:
    """The grand combination: multi-asset excess returns + three heterogeneous feature blocks.

    predictors (lag=0, shared calendar) + a publication-lagged macro (lag=2) + a vintaged FRED-like
    panel (VintagedBlock, timestamp asof). Exercises all three block flavours in one view.
    """
    returns = toy_assets()
    blocks = [
        FeatureBlock(toy_predictors(), lag=0, name="pred"),
        toy_macro_block(lag=2),
        toy_vintaged_block(),
    ]
    return TimeSeriesView(returns, blocks=blocks, horizon=horizon)


# --- market timing with excess conversion, full stack --------------------------------------------


def test_market_timing_excess_end_to_end() -> None:
    raw, rf = toy_market()
    preds = toy_predictors()
    view = TimeSeriesView(raw, preds, risk_free=rf, horizon=1)  # raw -> excess internally
    out = backtest_weights(
        _OLSTimingEstimator(),
        view,
        WalkForwardSplitter(min_train=36, test_size=12),
        method="toy_ols",
    )
    assert isinstance(out, WeightsOutput)
    assert out.capability == capabilities.TO_WEIGHTS
    assert out.weights.index.equals(out.realized.index)  # only realized pairs survive
    assert not out.realized.isna().to_numpy().any()  # no look-ahead / no unrealized tail
    # excess returns are raw - rf everywhere
    np.testing.assert_allclose(view.returns_frame()["mkt"].to_numpy(), (raw["mkt"] - rf).to_numpy())


def test_walk_forward_multi_asset_end_to_end() -> None:
    # a fixed 3-asset universe with shared predictors (the MV / portfolio shape): to_weights emits
    # a 3-column weight each period and the engine sums w*r across the 3 assets
    view = TimeSeriesView(toy_assets(), toy_predictors(), horizon=1)
    out = backtest_weights(
        _OLSTimingEstimator(), view, WalkForwardSplitter(min_train=30, test_size=6), method="multi"
    )
    assert list(out.weights.columns) == ["size", "value", "mom"]
    assert out.weights.shape[1] == 3
    assert out.weights.index.equals(out.realized.index)
    assert not out.realized.isna().to_numpy().any()
    manual = (out.weights.to_numpy() * out.realized.to_numpy()).sum(axis=1)
    np.testing.assert_allclose(out.strategy_returns().to_numpy(), manual)


def test_excess_log_vs_simple_differ() -> None:
    raw, rf = toy_market()
    preds = toy_predictors()
    simple = TimeSeriesView(raw, preds, risk_free=rf, excess="simple").returns_frame()
    log = TimeSeriesView(raw, preds, risk_free=rf, excess="log").returns_frame()
    # both valid excess series, but the log construction is not identical to the simple one
    assert not np.allclose(simple.to_numpy(), log.to_numpy())


# --- multi-asset + predictors: shapes and horizon ------------------------------------------------


def test_multiasset_predictors_shapes_and_horizon() -> None:
    view = TimeSeriesView(toy_assets(), toy_predictors(), horizon=3)
    assert view.assets == ["size", "value", "mom"]
    assert view.feature_names == ["dp", "tbl"]
    dates, x, y = view.aligned()
    assert x.shape == (len(dates), 2)  # two predictors
    assert y.shape == (len(dates), 3)  # three assets
    # horizon-3 target compounds the next three returns, per asset
    t = dates[0]
    pos = int(view.calendar.searchsorted(t))
    fut = toy_assets().to_numpy()[pos + 1 : pos + 4]
    np.testing.assert_allclose(y[0], np.prod(1.0 + fut, axis=0) - 1.0)


# --- the grand combination: block order + readiness ----------------------------------------------


def test_combined_feature_name_order_and_asof() -> None:
    view = _combined_view()
    # concatenation order follows block order, across all three block flavours
    assert view.feature_names == ["dp", "tbl", "cpi", "INDPRO", "UNRATE"]
    vec = view.features_asof("2003-06-30")
    assert vec.shape == (5,)
    assert np.isfinite(vec).all()


def test_combined_aligned_purges_warmup_and_horizon() -> None:
    view = _combined_view(horizon=2)
    dates, x, y = view.aligned()
    assert x.shape[1] == 5
    assert y.shape[1] == 3
    # early dates are dropped while the lag-2 macro / lag-1 vintage warm up; late dates while the
    # horizon-2 target is unrealized. So the kept span sits strictly inside the calendar.
    assert dates.min() > view.calendar[0]
    assert dates.max() < view.calendar[-1]


# --- the no-look-ahead property (the headline invariant) -----------------------------------------


def test_asof_is_invariant_to_future_data() -> None:
    """features_asof(t) must not change when future data is truncated away (no leak, #1)."""
    view = _combined_view()
    for t in view.calendar[30::6]:  # sample dates where all blocks are ready
        full = view.features_asof(t)
        windowed = view.window(t).features_asof(t)  # future rows/vintages removed
        np.testing.assert_array_equal(full, windowed)


def test_target_is_invariant_to_extra_future_data() -> None:
    """The realized (t, t+h] target is identical whether or not later data exists in the view."""
    short = TimeSeriesView(toy_assets(n=40), toy_predictors(n=40), horizon=1)
    long = TimeSeriesView(toy_assets(n=72), toy_predictors(n=72), horizon=1)
    t = short.calendar[20]
    np.testing.assert_allclose(short.target_asof(t), long.target_asof(t))


# --- determinism ---------------------------------------------------------------------------------


def test_toy_catalog_is_deterministic() -> None:
    np.testing.assert_array_equal(toy_assets().to_numpy(), toy_assets().to_numpy())
    a, _ = toy_market()
    b, _ = toy_market()
    np.testing.assert_array_equal(a.to_numpy(), b.to_numpy())


def test_walk_forward_with_vintaged_block_end_to_end() -> None:
    # a multi-block TS view (lag-0 predictors + vintaged FRED) through the real engine: the vintage
    # warm-up is purged via is_ready inside aligned(), and the fit sees the concatenated features
    mkt, rf = toy_market()
    view = TimeSeriesView(
        mkt,
        risk_free=rf,
        blocks=[
            FeatureBlock(toy_predictors(), lag=0, name="pred"),
            VintagedBlock(toy_vintaged_table(n_refs=72), name="fred"),
        ],
    )
    out = backtest_weights(
        _OLSTimingEstimator(),
        view,
        WalkForwardSplitter(min_train=30, test_size=6),
        method="voc_fred",
    )
    assert isinstance(out, WeightsOutput)
    assert not out.weights.empty
    assert not out.realized.isna().to_numpy().any()  # warm-up + unrealized purged end-to-end
