"""Baselines: conformance suite, algebraic identities, and engine round-trips.

Synthetic data only — the point here is protocol conformance and the closed-form identities
(1/N, the min-variance first-order condition, the mean-variance direction), not paper reproduction.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from numeraire.baselines import (
    EqualWeight,
    HistoricalMean,
    MeanVariance,
    MinVariance,
    equal_weights,
    mean_variance_weights,
    minimum_variance_weights,
)
from numeraire.core.data import TimeSeriesView
from numeraire.core.engine import walk_forward, walk_forward_forecast
from numeraire.core.splitter import WalkForwardSplitter
from numeraire.testing import check_estimator


def _view(n: int = 48, n_assets: int = 4, seed: int = 0) -> TimeSeriesView:
    idx = pd.date_range("1990-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(seed)
    cols = [f"a{i}" for i in range(n_assets)]
    ret = pd.DataFrame(rng.normal(0.01, 0.04, (n, n_assets)), index=idx, columns=cols)
    return TimeSeriesView(ret, horizon=1)


# --------------------------------------------------------------------------- conformance


def test_equal_weight_conformance() -> None:
    check_estimator(EqualWeight(), lambda: _view(seed=1), min_train=24)


def test_min_variance_conformance() -> None:
    check_estimator(MinVariance(window=12), lambda: _view(seed=2), min_train=24)


def test_mean_variance_conformance() -> None:
    check_estimator(MeanVariance(window=18), lambda: _view(seed=3), min_train=24)


def test_historical_mean_conformance() -> None:
    check_estimator(HistoricalMean(), lambda: _view(n_assets=1, seed=4), min_train=24)


# --------------------------------------------------------------------------- identities


def test_equal_weights_identity() -> None:
    w = equal_weights(5)
    assert w == pytest.approx(np.full(5, 0.2))
    assert w.sum() == pytest.approx(1.0)


def test_min_variance_first_order_condition() -> None:
    rng = np.random.default_rng(0)
    block = rng.normal(0.01, 0.04, (200, 6))
    cov = np.cov(block, rowvar=False)
    w = minimum_variance_weights(cov)
    assert w.sum() == pytest.approx(1.0)
    # FOC of min w'Sw s.t. 1'w = 1: 2 S w = lambda 1, so S w is proportional to the ones vector.
    sw = cov @ w
    assert sw == pytest.approx(np.full(6, sw[0]))
    # and it has no larger sample variance than 1/N (the defining optimality)
    ew = equal_weights(6)
    assert float(w @ cov @ w) <= float(ew @ cov @ ew) + 1e-12


def test_mean_variance_direction_and_normalization() -> None:
    rng = np.random.default_rng(1)
    block = rng.normal(0.01, 0.04, (200, 5))
    mu = block.mean(axis=0)
    cov = np.cov(block, rowvar=False)
    # raw direction: S w = mu exactly (w = S^-1 mu)
    w_none = mean_variance_weights(mu, cov, normalization="none")
    assert cov @ w_none == pytest.approx(mu)
    # budget: sums to one and S w stays proportional to mu (same direction, rescaled)
    w_budget = mean_variance_weights(mu, cov, normalization="budget")
    assert w_budget.sum() == pytest.approx(1.0)
    ratio = (cov @ w_budget) / mu
    assert ratio == pytest.approx(np.full(5, ratio[0]))
    assert w_budget == pytest.approx(w_none / w_none.sum())


def test_mean_variance_rejects_bad_normalization() -> None:
    rng = np.random.default_rng(2)
    block = rng.normal(0.0, 1.0, (50, 3))
    with pytest.raises(ValueError, match="normalization must be"):
        mean_variance_weights(
            block.mean(axis=0),
            np.cov(block, rowvar=False),
            normalization="bogus",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="normalization must be"):
        MeanVariance(normalization="bogus")  # type: ignore[arg-type]


# --------------------------------------------------------------------------- engine round-trips


def test_equal_weight_engine_roundtrip() -> None:
    view = _view(seed=5)
    out = walk_forward(
        EqualWeight(),
        view,
        WalkForwardSplitter(min_train=24, test_size=1, expanding=True),
        method="ew",
    )
    assert not out.weights.empty
    # every rebalance is exactly 1/N across the four assets
    assert out.weights.to_numpy() == pytest.approx(0.25)


def test_min_variance_engine_equals_vectorized() -> None:
    # closed-form + deterministic: walk-forward must match a hand-rolled rolling min-var on commons.
    view = _view(n=120, seed=6)
    r = view.returns_frame().to_numpy(dtype=np.float64)
    cal = view.calendar
    win = 36
    hand = {
        cal[i]: minimum_variance_weights(np.cov(r[i - win + 1 : i + 1], rowvar=False))
        for i in range(win - 1, len(cal))
    }
    out = walk_forward(
        MinVariance(window=win),
        view,
        WalkForwardSplitter(min_train=win, test_size=1, expanding=True),
        method="minvar",
    )
    assert len(out.weights) > 40
    for t, w in out.weights.iterrows():
        np.testing.assert_allclose(w.to_numpy(), hand[t], atol=1e-10)


def test_historical_mean_is_the_engine_benchmark() -> None:
    # HistoricalMean's forecast IS the engine's prevailing-mean benchmark, so they coincide exactly.
    view = _view(n_assets=1, seed=7)
    out = walk_forward_forecast(HistoricalMean(), view, min_train=24, method="hm")
    assert not out.forecasts.empty
    np.testing.assert_allclose(out.forecasts.to_numpy(), out.benchmark.to_numpy(), atol=1e-12)


def test_weight_rules_require_timeseries_view() -> None:
    model = EqualWeight().fit(_view())
    with pytest.raises(TypeError, match="TimeSeriesView"):
        model.to_weights(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="TimeSeriesView"):
        HistoricalMean().fit(_view()).forecast(object())  # type: ignore[arg-type]


def test_constructor_and_input_guards() -> None:
    with pytest.raises(ValueError, match="at least one asset"):
        equal_weights(0)
    for bad in (MinVariance, MeanVariance):
        with pytest.raises(ValueError, match="window must be"):
            bad(window=1)
        with pytest.raises(ValueError, match="min_obs must be"):
            bad(min_obs=1)
    with pytest.raises(ValueError, match="window must be"):
        HistoricalMean(window=0)


def test_mean_variance_none_normalization_engine() -> None:
    # the raw (unnormalized) direction runs through the engine and its weights need not sum to one
    view = _view(n=80, seed=8)
    out = walk_forward(
        MeanVariance(normalization="none", window=36),
        view,
        WalkForwardSplitter(min_train=36, test_size=1, expanding=True),
        method="mv_raw",
    )
    assert not out.weights.empty


def test_windowed_historical_mean_is_rolling() -> None:
    view = _view(n_assets=1, seed=9)
    win = 12
    fc = HistoricalMean(window=win).fit(view).forecast(view)
    assert float(fc.iloc[0]) == pytest.approx(float(view.returns_frame().to_numpy()[-win:].mean()))
