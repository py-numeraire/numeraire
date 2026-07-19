"""Per-fold estimator isolation and forecast/pricing fold-calendar containment.

Two acceptance criteria for the fold-isolation work package:

* A deliberately **stateful** warm-start estimator must produce bit-identical output under
  ``n_jobs=1`` and ``n_jobs=4`` (and through ``check_fold_isolation``). On the previous engine the
  drivers fitted **one shared instance** across parallel folds, so a stateful estimator's results
  depended on the thread schedule and serial and parallel diverged (a fit-returns-self estimator
  drifted by roughly ~0.047). Each fold now fits an isolated ``copy.deepcopy``, so the property
  holds.
* A **malicious** estimator that emits dates before its test fold, duplicated dates, or a phantom
  asset must now raise at the forecast/pricing driver boundary. Both passed **silently** before:
  the pricing driver pooled the out-of-fold / duplicated cross-sections as real OOS observations,
  and the forecast driver dropped a phantom-asset forecast (scoring it as if the model abstained).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_monthly_view
from numeraire.core import capabilities
from numeraire.core.data import TimeSeriesView
from numeraire.core.engine import (
    backtest_forecast,
    backtest_pricing,
    backtest_weights,
)
from numeraire.core.splitter import WalkForwardSplitter
from numeraire.testing import check_fold_isolation, check_output_shapes

# -- a genuinely stateful (warm-start) weights estimator ------------------------------------------


class _WarmStartWeightsModel:
    def __init__(self, w: np.ndarray) -> None:
        self._w = w

    def capabilities(self) -> set[str]:
        return {capabilities.TO_WEIGHTS}

    def to_weights(self, view: TimeSeriesView) -> pd.DataFrame:
        rows = np.tile(self._w, (len(view.calendar), 1))
        return pd.DataFrame(rows, index=view.calendar, columns=view.assets)


class _WarmStartWeights:
    """Blends the PREVIOUS fit's mean return into this fit — stateful across fits.

    On a shared-instance engine the per-fold weights depend on the order folds were fitted (and race
    under threads); with per-fold deepcopy isolation every fold starts from the same pristine state
    (``_prev is None``), so serial and parallel are bit-identical.
    """

    def __init__(self) -> None:
        self._prev: np.ndarray | None = None

    def fit(self, view: TimeSeriesView) -> _WarmStartWeightsModel:
        mean = view.returns_frame().to_numpy(np.float64).mean(axis=0)
        blended = mean if self._prev is None else 0.5 * mean + 0.5 * self._prev
        self._prev = mean
        w = np.sign(blended)
        norm = float(np.abs(w).sum())
        if norm > 0:
            w = w / norm
        return _WarmStartWeightsModel(w)


def test_warm_start_weights_serial_equals_parallel() -> None:
    # Bite (a): on the old shared-instance engine n_jobs=1 and n_jobs=4 DIFFERED (~0.047) for this
    # stateful estimator; per-fold deepcopy isolation makes them bit-identical.
    v = make_monthly_view(n=120, n_assets=4, seed=5)
    sp = WalkForwardSplitter(min_train=36, test_size=12)  # several folds -> real thread parallelism
    serial = backtest_weights(_WarmStartWeights(), v, sp, method="ws", n_jobs=1)
    parallel = backtest_weights(_WarmStartWeights(), v, sp, method="ws", n_jobs=4)
    pd.testing.assert_frame_equal(serial.weights, parallel.weights)
    pd.testing.assert_frame_equal(serial.realized, parallel.realized)


def test_warm_start_weights_passes_fold_isolation() -> None:
    # The conformance check must certify the same property for a stateful estimator.
    check_fold_isolation(_WarmStartWeights(), lambda: make_monthly_view(n=96, n_assets=4, seed=6))


# -- malicious pricing estimators: out-of-fold / duplicate expected-return dates -------------------


class _PrefoldPricingModel:
    def capabilities(self) -> set[str]:
        return {capabilities.TO_PRICING}

    def expected_returns(self, view: TimeSeriesView) -> pd.DataFrame:
        # A fabricated date strictly before any test fold, prepended to two real fold dates.
        idx = pd.DatetimeIndex([pd.Timestamp("1970-01-31"), *view.calendar[:2]])
        return pd.DataFrame(0.01, index=idx, columns=view.assets)


class _PrefoldPricing:
    def fit(self, view: TimeSeriesView) -> _PrefoldPricingModel:
        _ = view
        return _PrefoldPricingModel()


class _DupDatePricingModel:
    def capabilities(self) -> set[str]:
        return {capabilities.TO_PRICING}

    def expected_returns(self, view: TimeSeriesView) -> pd.DataFrame:
        cal = view.calendar
        idx = pd.DatetimeIndex([cal[0], cal[0], cal[1]])  # a duplicated in-fold date
        return pd.DataFrame(0.01, index=idx, columns=view.assets)


class _DupDatePricing:
    def fit(self, view: TimeSeriesView) -> _DupDatePricingModel:
        _ = view
        return _DupDatePricingModel()


def _pricing_view() -> TimeSeriesView:
    idx = pd.date_range("2000-01-31", periods=48, freq="ME")
    rng = np.random.default_rng(7)
    ret = pd.DataFrame(rng.normal(0.01, 0.04, (48, 3)), index=idx, columns=["a", "b", "c"])
    return TimeSeriesView(ret, horizon=1)


def test_pricing_rejects_out_of_fold_dates() -> None:
    # Bite (b): a pre-fold expected-return date was silently pooled as a real OOS observation.
    v = _pricing_view()
    sp = WalkForwardSplitter(min_train=24, test_size=12)
    with pytest.raises(ValueError, match="outside the current test fold"):
        backtest_pricing(_PrefoldPricing(), v, sp, method="mal_pricer")


def test_pricing_rejects_duplicate_dates() -> None:
    v = _pricing_view()
    sp = WalkForwardSplitter(min_train=24, test_size=12)
    with pytest.raises(ValueError, match="expected-return dates must be unique"):
        backtest_pricing(_DupDatePricing(), v, sp, method="mal_pricer")


def test_check_output_shapes_catches_malicious_pricing() -> None:
    # The conformance shape check surfaces the same violations before an engine run does.
    with pytest.raises(AssertionError, match="calendar"):
        check_output_shapes(_PrefoldPricing(), _pricing_view)
    with pytest.raises(AssertionError, match="unique dates"):
        check_output_shapes(_DupDatePricing(), _pricing_view)


# -- malicious forecast estimators: phantom / duplicate asset labels -------------------------------
#
# Forecast *origins* (dates) are engine-assigned from the view calendar, so the forecast driver
# cannot emit an out-of-fold date; the containment a model can violate is on the asset axis. A
# phantom asset was silently dropped by the reindex before (scored as an abstention); a duplicate
# asset label made the reindex raise cryptically. Both now raise with the method name.


class _PhantomAssetForecastModel:
    def capabilities(self) -> set[str]:
        return {capabilities.TO_FORECAST}

    def forecast(self, view: TimeSeriesView) -> pd.Series:
        labels = [*view.assets, "PHANTOM"]  # an asset absent from the view
        return pd.Series(np.full(len(labels), 0.01), index=labels)


class _PhantomAssetForecast:
    def fit(self, view: TimeSeriesView) -> _PhantomAssetForecastModel:
        _ = view
        return _PhantomAssetForecastModel()


class _DupAssetForecastModel:
    def capabilities(self) -> set[str]:
        return {capabilities.TO_FORECAST}

    def forecast(self, view: TimeSeriesView) -> pd.Series:
        labels = [view.assets[0], *view.assets]  # a duplicated asset label
        return pd.Series(np.full(len(labels), 0.01), index=labels)


class _DupAssetForecast:
    def fit(self, view: TimeSeriesView) -> _DupAssetForecastModel:
        _ = view
        return _DupAssetForecastModel()


def test_forecast_rejects_phantom_asset() -> None:
    v = _pricing_view()
    with pytest.raises(ValueError, match="assets absent from the view"):
        backtest_forecast(_PhantomAssetForecast(), v, min_train=24, method="mal_fc")


def test_forecast_rejects_duplicate_asset_labels() -> None:
    v = _pricing_view()
    with pytest.raises(ValueError, match="forecast asset labels must be unique"):
        backtest_forecast(_DupAssetForecast(), v, min_train=24, method="mal_fc")
