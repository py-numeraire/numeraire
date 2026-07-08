"""The dispatching ``backtest`` entry point and the deprecated-alias shims.

``backtest`` routes on the fitted model's capability + the view type to the right typed driver and
returns the matching Output; ``in_sample=True`` selects the in-sample pricing path. Every old name
(``walk_forward*``, ``pricing_in_sample``, ``adjust_tests``, ``clark_west``, ``make_sorts``,
``OOSR2Evaluator``) keeps working for one release but emits a ``DeprecationWarning``.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterator

import numpy as np
import pandas as pd
import pytest

from conftest import make_monthly_view, toy_panel_wide
from numeraire import (
    ForecastOutput,
    OOSR2Evaluator,
    OutOfSampleR2Evaluator,
    PanelWeightsOutput,
    PricingOutput,
    WeightsOutput,
    adjust_pvalues,
    adjust_tests,
    backtest,
    clark_west,
    clark_west_test,
    make_sorts,
    pricing_in_sample,
    sort_portfolios,
    walk_forward,
    walk_forward_forecast,
    walk_forward_panel,
    walk_forward_pricing,
)
from numeraire.core import capabilities
from numeraire.core.data import CrossSectionView, TimeSeriesView
from numeraire.core.engine import (
    _FORECAST_DEFAULT_MIN_TRAIN,
    backtest_forecast,
    backtest_weights,
)
from numeraire.core.splitter import WalkForwardSplitter

# -- toy estimators (one per dispatchable capability + edge cases) ---------------------------------


class _WeightsModel:
    def capabilities(self) -> set[str]:
        return {capabilities.TO_WEIGHTS}

    def to_weights(self, view: TimeSeriesView) -> pd.DataFrame:
        assets = view.assets
        w = np.full((len(view.calendar), len(assets)), 1.0 / len(assets))
        return pd.DataFrame(w, index=view.calendar, columns=assets)


class _WeightsEst:
    def fit(self, view: TimeSeriesView) -> _WeightsModel:
        return _WeightsModel()


class _ForecastModel:
    def capabilities(self) -> set[str]:
        return {capabilities.TO_FORECAST}

    def forecast(self, view: TimeSeriesView) -> pd.Series:
        return view.returns_frame().mean()


class _ForecastEst:
    def fit(self, view: TimeSeriesView) -> _ForecastModel:
        return _ForecastModel()


class _PanelModel:
    def capabilities(self) -> set[str]:
        return {capabilities.TO_WEIGHTS}

    def to_weights(self, view: CrossSectionView) -> pd.Series:
        dates: list[pd.Timestamp] = []
        assets: list[object] = []
        vals: list[float] = []
        for t in view.calendar:
            ids, _x = view.features_asof(t)
            n = len(ids)
            for a in ids:
                dates.append(t)
                assets.append(a)
                vals.append(1.0 / n if n else 0.0)
        idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex(dates), assets], names=["date", "asset"])
        return pd.Series(vals, index=idx, name="weight")


class _PanelEst:
    def fit(self, view: CrossSectionView) -> _PanelModel:
        return _PanelModel()


class _PricingModel:
    def __init__(self, mu: pd.Series) -> None:
        self._mu = mu

    def capabilities(self) -> set[str]:
        return {capabilities.TO_PRICING}

    def expected_returns(self, view: TimeSeriesView) -> pd.DataFrame:
        row = self._mu.reindex([str(a) for a in view.assets]).to_numpy(np.float64)
        vals = np.tile(row, (len(view.calendar), 1))
        return pd.DataFrame(vals, index=view.calendar, columns=view.assets)


class _PricingEst:
    def fit(self, view: TimeSeriesView) -> _PricingModel:
        mu = view.returns_frame().mean()
        mu.index = [str(c) for c in mu.index]
        return _PricingModel(mu)


class _MultiCapModel(_WeightsModel):
    def capabilities(self) -> set[str]:
        return {capabilities.TO_WEIGHTS, capabilities.TO_FORECAST}

    def forecast(self, view: TimeSeriesView) -> pd.Series:
        return view.returns_frame().mean()


class _MultiCapEst:
    def fit(self, view: TimeSeriesView) -> _MultiCapModel:
        return _MultiCapModel()


class _NoCapModel:
    def capabilities(self) -> set[str]:
        return {capabilities.TO_DENSITY}


class _NoCapEst:
    def fit(self, view: TimeSeriesView) -> _NoCapModel:
        return _NoCapModel()


class _RecordingWeightsEst:
    """Records the max calendar date of every view it is fitted on (to inspect the probe fit)."""

    def __init__(self) -> None:
        self.fit_max_dates: list[pd.Timestamp] = []

    def fit(self, view: TimeSeriesView) -> _WeightsModel:
        self.fit_max_dates.append(view.calendar.max())
        return _WeightsModel()


class _RecordingForecastEst:
    def __init__(self) -> None:
        self.fit_max_dates: list[pd.Timestamp] = []

    def fit(self, view: TimeSeriesView) -> _ForecastModel:
        self.fit_max_dates.append(view.calendar.max())
        return _ForecastModel()


def _ts_view() -> TimeSeriesView:
    return make_monthly_view(n=120, n_assets=3)


def _panel_view() -> CrossSectionView:
    return CrossSectionView(toy_panel_wide(), chars=["size", "bm", "mom"], horizon=1)


def _splitter() -> WalkForwardSplitter:
    return WalkForwardSplitter(min_train=60, test_size=12)


# -- dispatch: capability + view -> the right Output -----------------------------------------------


def test_backtest_dispatches_weights_timeseries_to_weights_output() -> None:
    out = backtest(_WeightsEst(), _ts_view(), _splitter(), method="w")
    assert isinstance(out, WeightsOutput)
    assert out.capability == capabilities.TO_WEIGHTS
    assert not out.weights.empty


def test_backtest_dispatches_weights_crosssection_to_panel_output() -> None:
    out = backtest(
        _PanelEst(),
        _panel_view(),
        WalkForwardSplitter(min_train=24, test_size=6),
        method="panel",
    )
    assert isinstance(out, PanelWeightsOutput)
    assert out.capability == capabilities.TO_WEIGHTS


def test_backtest_dispatches_forecast_to_forecast_output() -> None:
    out = backtest(_ForecastEst(), _ts_view(), method="f", min_train=24)
    assert isinstance(out, ForecastOutput)
    assert out.capability == capabilities.TO_FORECAST
    assert not out.forecasts.empty


def test_backtest_dispatches_pricing_walk_forward() -> None:
    out = backtest(_PricingEst(), _ts_view(), _splitter(), method="p")
    assert isinstance(out, PricingOutput)
    assert out.protocol == "walk_forward"


def test_backtest_in_sample_selects_in_sample_pricing() -> None:
    out = backtest(_PricingEst(), _ts_view(), method="p", in_sample=True)
    assert isinstance(out, PricingOutput)
    assert out.protocol == "in_sample"


def test_backtest_raises_when_no_dispatchable_capability() -> None:
    with pytest.raises(TypeError, match="none of the dispatchable capabilities"):
        backtest(_NoCapEst(), _ts_view(), _splitter(), method="none")


def test_backtest_raises_on_ambiguous_multiple_capabilities() -> None:
    with pytest.raises(TypeError, match="multiple dispatchable capabilities"):
        backtest(_MultiCapEst(), _ts_view(), _splitter(), method="ambiguous")


def test_backtest_in_sample_requires_pricing() -> None:
    with pytest.raises(TypeError, match="does not support 'to_pricing'"):
        backtest(_WeightsEst(), _ts_view(), _splitter(), method="w", in_sample=True)


# -- probe fit sees only the driver's first train window, never the full sample -------------------


def test_probe_fit_sees_only_first_fold_train_weights() -> None:
    view = _ts_view()
    sp = _splitter()  # min_train=60, test_size=12
    est = _RecordingWeightsEst()
    out = backtest(est, view, sp, method="w")
    assert isinstance(out, WeightsOutput)
    # The probe fit (the first recorded fit) sees exactly the first fold's train window ...
    first_train_end = next(iter(sp.split(view)))[0].calendar.max()
    probe_max = est.fit_max_dates[0]
    assert probe_max == first_train_end
    # ... and never the full sample (that would be a look-ahead channel for a stateful estimator).
    assert probe_max < view.calendar.max()


def test_probe_fit_sees_only_warmup_prefix_forecast() -> None:
    view = _ts_view()
    est = _RecordingForecastEst()
    out = backtest(est, view, method="f", min_train=24)
    assert isinstance(out, ForecastOutput)
    probe_max = est.fit_max_dates[0]
    # The probe window is exactly the first ``min_train`` calendar steps.
    assert probe_max == view.calendar[24 - 1]
    assert probe_max < view.calendar.max()


def test_probe_fit_window_kwarg_rolls_the_prefix() -> None:
    view = _ts_view()
    est = _RecordingForecastEst()
    backtest(est, view, method="f", window=18)
    # With a rolling ``window`` the warm-up ends at the window-th step (mirrors backtest_forecast).
    assert est.fit_max_dates[0] == view.calendar[18 - 1]


class _OneShotSplitter:
    """Adversarial splitter: ``split()`` returns one stored iterator, usable exactly once.

    A second ``split()`` call would find the iterator already exhausted — the probe must therefore
    materialize the folds once and replay them to the driver, or fold 0 silently disappears.
    """

    def __init__(self, inner: WalkForwardSplitter, view: TimeSeriesView) -> None:
        self._it: Iterator[tuple[TimeSeriesView, TimeSeriesView]] = inner.split(view)
        self.calls = 0

    def split(self, view: TimeSeriesView) -> Iterator[tuple[TimeSeriesView, TimeSeriesView]]:
        self.calls += 1
        return self._it


def test_one_shot_splitter_loses_no_folds() -> None:
    view = _ts_view()
    one_shot = _OneShotSplitter(_splitter(), view)
    out = backtest(_WeightsEst(), view, one_shot, method="w")
    assert isinstance(out, WeightsOutput)
    # the user splitter's split() ran exactly once (probe + driver share the materialized folds)
    assert one_shot.calls == 1
    # every fold survives: identical output to the direct driver run with a fresh splitter
    direct = backtest_weights(_WeightsEst(), _ts_view(), _splitter(), method="w")
    assert len(out.weights) == len(direct.weights)
    pd.testing.assert_frame_equal(out.weights, direct.weights)


def test_probe_short_view_matches_driver_empty_output() -> None:
    # 12 dates < the default min_train=20: the driver yields zero forecast origins (empty output);
    # the probe must not die on the warm-up indexing where the driver itself would succeed.
    view = make_monthly_view(n=12, n_assets=3)
    out = backtest(_ForecastEst(), view, method="f")
    assert isinstance(out, ForecastOutput)
    direct = backtest_forecast(_ForecastEst(), make_monthly_view(n=12, n_assets=3), method="f")
    assert out.forecasts.empty and direct.forecasts.empty


def test_panel_window_kwarg_without_splitter_still_raises_cleanly() -> None:
    # CrossSectionView has window() but no tail(): the probe must not crash on the rolling-tail
    # step before the dispatch surfaces the real error (walk-forward weights need a splitter).
    with pytest.raises(TypeError, match="requires a `splitter`"):
        backtest(_PanelEst(), _panel_view(), method="panel", window=6)


def test_forecast_default_min_train_constant_stays_in_sync() -> None:
    # drift guard: the probe's fallback warm-up must equal backtest_forecast's min_train default
    default = inspect.signature(backtest_forecast).parameters["min_train"].default
    assert default == _FORECAST_DEFAULT_MIN_TRAIN


# -- deprecated aliases: still work AND emit DeprecationWarning ------------------------------------


def test_walk_forward_alias_warns_and_returns_weights_output() -> None:
    with pytest.warns(DeprecationWarning, match="backtest_weights"):
        out = walk_forward(_WeightsEst(), _ts_view(), _splitter(), method="w")
    assert isinstance(out, WeightsOutput)


def test_walk_forward_forecast_alias_warns_and_returns_forecast_output() -> None:
    with pytest.warns(DeprecationWarning, match="backtest_forecast"):
        out = walk_forward_forecast(_ForecastEst(), _ts_view(), min_train=24, method="f")
    assert isinstance(out, ForecastOutput)


def test_walk_forward_panel_alias_warns_and_returns_panel_output() -> None:
    with pytest.warns(DeprecationWarning, match="backtest_panel"):
        out = walk_forward_panel(
            _PanelEst(), _panel_view(), WalkForwardSplitter(min_train=24, test_size=6), method="p"
        )
    assert isinstance(out, PanelWeightsOutput)


def test_walk_forward_pricing_alias_warns_and_returns_pricing_output() -> None:
    with pytest.warns(DeprecationWarning, match="backtest_pricing"):
        out = walk_forward_pricing(_PricingEst(), _ts_view(), _splitter(), method="p")
    assert isinstance(out, PricingOutput)
    assert out.protocol == "walk_forward"


def test_pricing_in_sample_alias_warns_and_returns_pricing_output() -> None:
    with pytest.warns(DeprecationWarning, match="backtest_pricing_in_sample"):
        out = pricing_in_sample(_PricingEst(), _ts_view(), method="p")
    assert isinstance(out, PricingOutput)
    assert out.protocol == "in_sample"


def test_adjust_tests_alias_warns_and_matches_new_name() -> None:
    p = np.array([0.001, 0.02, 0.3, 0.8])
    with pytest.warns(DeprecationWarning, match="adjust_pvalues"):
        old = adjust_tests(p)
    new = adjust_pvalues(p)
    np.testing.assert_allclose(old.adjusted_p, new.adjusted_p)


def test_clark_west_alias_warns_and_matches_new_name() -> None:
    rng = np.random.default_rng(0)
    r = rng.normal(0.0, 0.04, 80)
    f = 0.5 * r
    b = np.zeros_like(r)
    with pytest.warns(DeprecationWarning, match="clark_west_test"):
        old = clark_west(r, f, b)
    new = clark_west_test(r, f, b)
    assert old.t_stat == pytest.approx(new.t_stat)


def test_make_sorts_alias_warns_and_matches_new_name() -> None:
    idx = pd.date_range("2000-01-31", periods=12, freq="ME")
    cols = [f"a{i}" for i in range(6)]
    rng = np.random.default_rng(1)
    signal = pd.DataFrame(rng.normal(size=(12, 6)), index=idx, columns=cols)
    returns = pd.DataFrame(rng.normal(0.0, 0.05, size=(12, 6)), index=idx, columns=cols)
    with pytest.warns(DeprecationWarning, match="sort_portfolios"):
        old = make_sorts(signal, returns, n_bins=3)
    new = sort_portfolios(signal, returns, n_bins=3)
    pd.testing.assert_series_equal(old.long_short, new.long_short)


def test_oosr2_evaluator_alias_warns_on_construction_and_is_subclass() -> None:
    with pytest.warns(DeprecationWarning, match="OutOfSampleR2Evaluator"):
        ev = OOSR2Evaluator()
    assert isinstance(ev, OutOfSampleR2Evaluator)
