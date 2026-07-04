"""The conformance suite catches leaks and passes well-behaved estimators.

Exercises ``numeraire.testing.check_estimator`` on trivial in-repo estimators: a correct
constant-weight ``to_weights`` model, a correct historical-mean ``to_forecast`` model, and a
deliberately leaky ``to_weights`` model that peeks at the end of whatever view it is handed. The
zoo runs the same suite against the six real reproductions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from numeraire.core import capabilities
from numeraire.core.data import CrossSectionView, TimeSeriesView
from numeraire.testing import (
    _perturb_after,
    check_capabilities,
    check_estimator,
    check_no_lookahead,
)


def _ts_returns_only() -> TimeSeriesView:
    idx = pd.date_range("2000-01-31", periods=48, freq="ME")
    rng = np.random.default_rng(11)
    ret = pd.DataFrame(rng.normal(0.01, 0.04, (48, 4)), index=idx, columns=["a", "b", "c", "d"])
    return TimeSeriesView(ret, horizon=1)


def _ts_with_features() -> TimeSeriesView:
    idx = pd.date_range("2000-01-31", periods=48, freq="ME")
    rng = np.random.default_rng(22)
    ret = pd.DataFrame(rng.normal(0.01, 0.04, (48, 1)), index=idx, columns=["mkt"])
    feat = pd.DataFrame(rng.normal(0.0, 1.0, (48, 2)), index=idx, columns=["x0", "x1"])
    return TimeSeriesView(ret, feat, horizon=1)


# -- correct estimators ---------------------------------------------------------


class _EqualWeightModel:
    def capabilities(self) -> set[str]:
        return {capabilities.TO_WEIGHTS}

    def to_weights(self, view: TimeSeriesView) -> pd.DataFrame:
        n = len(view.assets)
        vals = np.full((len(view.calendar), n), 1.0 / n)
        return pd.DataFrame(vals, index=view.calendar, columns=view.assets)


class _EqualWeight:
    def fit(self, view: TimeSeriesView) -> _EqualWeightModel:
        _ = view
        return _EqualWeightModel()


class _HistMeanModel:
    def capabilities(self) -> set[str]:
        return {capabilities.TO_FORECAST}

    def forecast(self, view: TimeSeriesView) -> pd.Series:
        return view.returns_frame().mean()


class _HistMean:
    def fit(self, view: TimeSeriesView) -> _HistMeanModel:
        _ = view
        return _HistMeanModel()


# -- leaky estimator (peeks at the end of the passed view) ----------------------


class _LeakyModel:
    def capabilities(self) -> set[str]:
        return {capabilities.TO_WEIGHTS}

    def to_weights(self, view: TimeSeriesView) -> pd.DataFrame:
        last = view.returns_frame().to_numpy(np.float64)[-1]  # LEAK: the view's final row
        vals = np.tile(last, (len(view.calendar), 1))
        return pd.DataFrame(vals, index=view.calendar, columns=view.assets)


class _Leaky:
    def fit(self, view: TimeSeriesView) -> _LeakyModel:
        _ = view
        return _LeakyModel()


# -- panel (cross-sectional) to_weights ----------------------------------------


def _panel_view() -> CrossSectionView:
    rng = np.random.default_rng(33)
    dates = pd.date_range("2000-01-31", periods=40, freq="ME")
    rows: list[tuple[object, ...]] = []
    for d in dates:
        for a in ("s1", "s2", "s3", "s4", "s5"):
            rows.append((d, a, rng.normal(), rng.normal(), rng.normal(0.01, 0.05)))
    df = pd.DataFrame(rows, columns=["date", "asset", "c0", "c1", "ret"])
    return CrossSectionView(df, chars=["c0", "c1"], horizon=1)


class _PanelModel:
    def __init__(self, beta: np.ndarray) -> None:
        self._beta = beta

    def capabilities(self) -> set[str]:
        return {capabilities.TO_WEIGHTS}

    def to_weights(self, view: CrossSectionView) -> pd.Series:
        dates: list[pd.Timestamp] = []
        assets: list[object] = []
        vals: list[float] = []
        for t in view.calendar:
            ids, x = view.features_asof(t)  # PIT: cross-section known as of t
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


class _Panel:
    def fit(self, view: CrossSectionView) -> _PanelModel:
        _keys, x, y = view.aligned()
        beta, *_ = np.linalg.lstsq(x, y, rcond=None)
        return _PanelModel(beta)


# -- capability-only (to_pricing) ----------------------------------------------


class _PricingModel:
    def capabilities(self) -> set[str]:
        return {capabilities.TO_PRICING}


class _Pricing:
    def fit(self, view: TimeSeriesView) -> _PricingModel:
        _ = view
        return _PricingModel()


def test_correct_weights_estimator_passes() -> None:
    check_estimator(_EqualWeight(), _ts_returns_only)
    # a to_weights estimator on a view *with* features exercises the feature-perturbation branch
    check_no_lookahead(_EqualWeight(), _ts_with_features)


def test_correct_panel_weights_estimator_passes() -> None:
    check_estimator(_Panel(), _panel_view)


def test_capability_only_pricing_estimator_passes() -> None:
    # to_pricing is not crystallized: capabilities passes, the weight/forecast checks no-op
    check_estimator(_Pricing(), _ts_returns_only)


def test_perturb_after_rejects_unknown_view() -> None:
    class _Fake:
        pass

    with pytest.raises(TypeError, match="cannot perturb"):
        _perturb_after(_Fake(), pd.Timestamp("2000-01-31"))


def test_correct_forecast_estimator_passes() -> None:
    check_estimator(_HistMean(), _ts_with_features)


def test_leaky_estimator_fails_no_lookahead() -> None:
    with pytest.raises(AssertionError, match="look-ahead"):
        check_no_lookahead(_Leaky(), _ts_returns_only)
    # and the full suite surfaces the same failure
    with pytest.raises(AssertionError, match="look-ahead"):
        check_estimator(_Leaky(), _ts_returns_only)


def test_capabilities_rejects_empty() -> None:
    class _NoCapModel:
        def capabilities(self) -> set[str]:
            return set()

    class _NoCap:
        def fit(self, view: TimeSeriesView) -> _NoCapModel:
            _ = view
            return _NoCapModel()

    with pytest.raises(AssertionError, match="at least one"):
        check_capabilities(_NoCap(), _ts_returns_only)
