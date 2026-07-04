"""Unit tests for TimeSeriesView: PIT windowing, explicit horizon, alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_monthly_view
from numeraire.core.data import TimeSeriesView
from numeraire.core.protocols import DataView


def _hand_view() -> TimeSeriesView:
    index = pd.date_range("2000-01-31", periods=6, freq="ME")
    returns = pd.DataFrame({"r0": [0.01, 0.02, -0.03, 0.04, 0.05, -0.01]}, index=index)
    features = pd.DataFrame({"x0": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}, index=index)
    return TimeSeriesView(returns, features, horizon=1)


def test_is_dataview() -> None:
    assert isinstance(_hand_view(), DataView)


def test_horizon_zero_rejected() -> None:
    index = pd.date_range("2000-01-31", periods=3, freq="ME")
    df = pd.DataFrame({"a": [0.0, 0.0, 0.0]}, index=index)
    with pytest.raises(ValueError, match="contemporaneous"):
        TimeSeriesView(df, df, horizon=0)


def test_misaligned_index_rejected() -> None:
    idx_a = pd.date_range("2000-01-31", periods=3, freq="ME")
    idx_b = pd.date_range("2001-01-31", periods=3, freq="ME")
    with pytest.raises(ValueError, match="identical DatetimeIndex"):
        TimeSeriesView(
            pd.DataFrame({"a": [0.0, 0.0, 0.0]}, index=idx_a),
            pd.DataFrame({"x": [0.0, 0.0, 0.0]}, index=idx_b),
        )


def test_window_truncates_to_end() -> None:
    v = _hand_view()
    cutoff = v.calendar[2]
    w = v.window(cutoff)
    assert w.calendar.max() == cutoff
    assert len(w.calendar) == 3


def test_features_asof_is_latest_known() -> None:
    v = _hand_view()
    t = v.calendar[3]
    np.testing.assert_array_equal(v.features_asof(t), np.array([4.0]))


def test_target_asof_h1_is_next_return() -> None:
    v = _hand_view()
    # target at t=index[1] over (t, t+1] is the return at index[2] = -0.03
    got = v.target_asof(v.calendar[1])
    np.testing.assert_allclose(got, np.array([-0.03]))


def test_target_asof_h2_compounds() -> None:
    index = pd.date_range("2000-01-31", periods=4, freq="ME")
    returns = pd.DataFrame({"r0": [0.0, 0.10, 0.20, 0.0]}, index=index)
    features = pd.DataFrame({"x0": [1.0, 2.0, 3.0, 4.0]}, index=index)
    v = TimeSeriesView(returns, features, horizon=2)
    # target at index[0] over the next two periods: (1.10 * 1.20) - 1
    np.testing.assert_allclose(v.target_asof(index[0]), np.array([(1.10 * 1.20) - 1.0]))


def test_target_unrealized_is_nan() -> None:
    v = _hand_view()
    assert np.isnan(v.target_asof(v.calendar[-1])).all()


def test_aligned_drops_unrealized_tail() -> None:
    v = _hand_view()  # 6 dates, h=1
    dates, x, y = v.aligned()
    # last feature date cannot be the final calendar point (its target is unrealized)
    assert len(dates) == 5
    assert dates.max() == v.calendar[4]
    assert x.shape == (5, 1)
    assert y.shape == (5, 1)
    np.testing.assert_allclose(y.ravel(), np.array([0.02, -0.03, 0.04, 0.05, -0.01]))


def test_window_alignment_purges_for_horizon() -> None:
    v = make_monthly_view(n=120, horizon=3)
    cutoff = v.calendar[80]
    dates, _, _ = v.window(cutoff).aligned()
    pos_cut = int(v.calendar.searchsorted(cutoff))
    pos_last = int(v.calendar.searchsorted(dates.max()))
    # last kept feature is at least `horizon` steps before the cutoff (target realized by cutoff)
    assert pos_cut - pos_last >= v.horizon


def test_risk_free_converts_raw_to_excess() -> None:
    index = pd.date_range("2000-01-31", periods=4, freq="ME")
    raw = pd.DataFrame({"r0": [0.05, 0.03, 0.07, 0.02]}, index=index)
    feats = pd.DataFrame({"x0": [1.0, 2.0, 3.0, 4.0]}, index=index)
    rf = pd.Series([0.01, 0.01, 0.02, 0.00], index=index)
    v = TimeSeriesView(raw, feats, risk_free=rf)  # raw - rf, broadcast across assets
    np.testing.assert_allclose(v.returns_frame()["r0"].to_numpy(), [0.04, 0.02, 0.05, 0.02])


def test_risk_free_missing_dates_rejected() -> None:
    index = pd.date_range("2000-01-31", periods=3, freq="ME")
    raw = pd.DataFrame({"r0": [0.0, 0.0, 0.0]}, index=index)
    rf = pd.Series([0.01, 0.01], index=index[:2])  # missing the last date
    with pytest.raises(ValueError, match="risk_free is missing"):
        TimeSeriesView(raw, raw, risk_free=rf)


# -- returns-only view (no features / no blocks) --------------------------------


def _returns_only_view() -> TimeSeriesView:
    index = pd.date_range("2000-01-31", periods=6, freq="ME")
    returns = pd.DataFrame({"r0": [0.01, 0.02, -0.03, 0.04, 0.05, -0.01]}, index=index)
    return TimeSeriesView(returns, horizon=1)


def test_returns_only_has_no_features() -> None:
    v = _returns_only_view()
    assert v.feature_names == []
    ff = v.features_frame()
    assert ff.shape == (6, 0)
    assert list(ff.columns) == []
    assert isinstance(v, DataView)


def test_returns_only_aligned_zero_width_x() -> None:
    v = _returns_only_view()
    dates, x, y = v.aligned()
    assert x.shape == (5, 0)  # one row per realizable origin, zero predictors
    assert y.shape == (5, 1)
    assert len(dates) == 5
    # the returns target is unchanged by the absence of features
    np.testing.assert_allclose(y.ravel(), [0.02, -0.03, 0.04, 0.05, -0.01])


def test_returns_only_windows_and_ejects() -> None:
    v = _returns_only_view()
    w = v.window(v.calendar[3])
    assert w.features_frame().shape == (4, 0)
    assert w.returns_frame().shape == (4, 1)
    # empty calendar slice still yields a (0, 0) feature frame, not an error
    empty = v.between(v.calendar[-1], v.calendar[-1])
    assert empty.features_frame().shape == (0, 0)


def test_features_and_blocks_are_mutually_exclusive() -> None:
    index = pd.date_range("2000-01-31", periods=3, freq="ME")
    df = pd.DataFrame({"r0": [0.0, 0.0, 0.0]}, index=index)
    with pytest.raises(ValueError, match="at most one"):
        TimeSeriesView(df, df, blocks=[], horizon=1)
