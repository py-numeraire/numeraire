"""Multi-block PIT features: heterogeneous macro sources with per-block availability lag.

Covers the motivating case — combining sources with different publication lags and different
calendars as predictors: FRED (lag=1) + another vintage-like source (lag=2) + a no-vintage
source (lag=0) — and the no-look-ahead invariant a lag must enforce.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from numeraire.core.data import FeatureBlock, TimeSeriesView


def _monthly(vals: list[float], *, start: str = "2000-01-31", name: str = "x") -> pd.DataFrame:
    idx = pd.date_range(start, periods=len(vals), freq="ME")
    return pd.DataFrame({name: vals}, index=idx)


def _returns(n: int = 12) -> pd.DataFrame:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    return pd.DataFrame({"mkt": np.zeros(n)}, index=idx)


def test_multiblock_per_block_lag_and_concat() -> None:
    # Three macro sources, distinct lags; B starts later (its own calendar, not the returns one).
    fred = FeatureBlock(_monthly([float(m) for m in range(1, 13)], name="fred"), lag=1, name="FRED")
    a = FeatureBlock(_monthly([100.0 + m for m in range(1, 13)], name="a"), lag=2, name="A")
    b = FeatureBlock(
        _monthly([1000.0 + m for m in range(3, 13)], start="2000-03-31", name="b"), lag=0, name="B"
    )
    view = TimeSeriesView(_returns(), blocks=[fred, a, b], horizon=1)

    assert view.feature_names == ["fred", "a", "b"]  # concatenated in block order
    # Decision end-of-June (pos 5): FRED lag1 -> May(5), A lag2 -> Apr(104), B lag0 -> Jun(1006).
    np.testing.assert_array_equal(view.features_asof("2000-06-30"), [5.0, 104.0, 1006.0])


def test_lag_makes_look_ahead_impossible() -> None:
    fred = FeatureBlock(_monthly([float(m) for m in range(1, 13)], name="fred"), lag=1, name="FRED")
    view = TimeSeriesView(_returns(), blocks=[fred], horizon=1)
    # At end of June the newest knowable value is May's (5.0), never June's (6.0)...
    assert view.features_asof("2000-06-30")[0] == 5.0
    # ...June's value only becomes available a month later.
    assert view.features_asof("2000-07-31")[0] == 6.0


def test_aligned_X_is_lag_aware() -> None:
    # The supervised design matrix must carry the lagged feature, not the contemporaneous one.
    fred = FeatureBlock(_monthly([float(m) for m in range(1, 13)], name="fred"), lag=1, name="FRED")
    view = TimeSeriesView(_returns(), blocks=[fred], horizon=1)
    dates, x, _ = view.aligned()
    # feature paired with the May rebalance is April's value (lag 1): X[date=May] == 4.0
    row = int(np.where(dates == pd.Timestamp("2000-05-31"))[0][0])
    assert x[row, 0] == 4.0


def test_features_arg_is_a_lag0_block_backcompat() -> None:
    feats = pd.DataFrame(
        {"x": np.arange(6.0)}, index=pd.date_range("2000-01-31", periods=6, freq="ME")
    )
    legacy = TimeSeriesView(_returns(6), feats, horizon=1)
    explicit = TimeSeriesView(_returns(6), blocks=[FeatureBlock(feats, lag=0)], horizon=1)
    t = pd.Timestamp("2000-04-30")
    np.testing.assert_array_equal(legacy.features_asof(t), explicit.features_asof(t))
    np.testing.assert_array_equal(legacy.features_asof(t), [3.0])


def test_features_and_blocks_are_mutually_exclusive() -> None:
    feats = pd.DataFrame(
        {"x": np.zeros(3)}, index=pd.date_range("2000-01-31", periods=3, freq="ME")
    )
    with pytest.raises(ValueError, match="at most one"):
        TimeSeriesView(_returns(3), feats, blocks=[FeatureBlock(feats)])
    # neither features nor blocks is a valid returns-only view (no predictors)
    assert TimeSeriesView(_returns(3)).feature_names == []
