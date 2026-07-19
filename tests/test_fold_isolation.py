"""Per-fold estimator isolation and forecast/pricing fold-calendar containment.

Two acceptance criteria for the fold-isolation work package:

* A deliberately **stateful** warm-start estimator must produce bit-identical output under
  ``n_jobs=1`` and ``n_jobs=4``, match a deterministic fresh-estimator-per-fold oracle, and pass
  ``check_fold_isolation``. On the previous engine the drivers fitted **one shared instance**
  across all folds, so a stateful estimator's serial folds chained state (a deterministic
  divergence from the oracle) and its parallel folds raced on the shared object (a fit-returns-self
  estimator drifted by roughly ~0.047). Each fold now fits an isolated ``copy.deepcopy``, so both
  properties hold. The estimator's weights are a *continuous* function of the carried state (no
  sign/threshold collapse), so any chained state moves the output.
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
from numeraire.comparison import ComparisonEntry, compare
from numeraire.core import capabilities
from numeraire.core.data import TimeSeriesView
from numeraire.core.engine import (
    backtest_forecast,
    backtest_pricing,
    backtest_pricing_in_sample,
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

    The blend is a **continuous** function of the carried state and the state advances
    deterministically at every fit (``_prev`` becomes the blended vector), so any cross-fit state
    chaining moves the output — no sign/threshold step can collapse two different states onto the
    same weights. On a shared-instance engine the serial folds therefore diverge deterministically
    from a fresh-estimator-per-fold oracle; with per-fold deepcopy isolation every fold starts from
    the same pristine state (``_prev is None``) and the oracle is reproduced exactly.
    """

    def __init__(self) -> None:
        self._prev: np.ndarray | None = None

    def fit(self, view: TimeSeriesView) -> _WarmStartWeightsModel:
        mean = view.returns_frame().to_numpy(np.float64).mean(axis=0)
        blended = mean if self._prev is None else 0.5 * mean + 0.5 * self._prev
        self._prev = blended.copy()  # deterministic state advancement, deeper every fit
        w = blended / float(np.abs(blended).sum())  # continuous in the state, unit gross exposure
        return _WarmStartWeightsModel(w)


def _oracle_weights(v: TimeSeriesView, sp: WalkForwardSplitter) -> pd.DataFrame:
    """Deterministic fresh-estimator-per-fold reference: what exact fold isolation must produce."""
    parts: list[pd.DataFrame] = []
    for train, test in sp.split(v):
        model = _WarmStartWeights().fit(train)
        parts.append(model.to_weights(test))
    return pd.concat(parts).sort_index()


def test_warm_start_weights_matches_fresh_per_fold_oracle() -> None:
    # Bite (a): on the old shared-instance engine even the SERIAL run chained state across folds,
    # so from fold 2 on the weights diverged deterministically from this oracle (no thread timing
    # involved); per-fold deepcopy isolation reproduces the oracle exactly, at any n_jobs.
    v = make_monthly_view(n=120, n_assets=4, seed=5)
    sp = WalkForwardSplitter(min_train=36, test_size=12)  # several folds -> several chained fits
    oracle = _oracle_weights(v, sp)
    for n_jobs in (1, 4):
        out = backtest_weights(_WarmStartWeights(), v, sp, method="ws", n_jobs=n_jobs)
        expected = oracle.reindex(index=out.weights.index)
        pd.testing.assert_frame_equal(out.weights, expected)


def test_warm_start_weights_serial_equals_parallel() -> None:
    # And the two engine runs agree with each other bit-for-bit (the oracle test pins the level).
    v = make_monthly_view(n=120, n_assets=4, seed=5)
    sp = WalkForwardSplitter(min_train=36, test_size=12)
    serial = backtest_weights(_WarmStartWeights(), v, sp, method="ws", n_jobs=1)
    parallel = backtest_weights(_WarmStartWeights(), v, sp, method="ws", n_jobs=4)
    pd.testing.assert_frame_equal(serial.weights, parallel.weights)
    pd.testing.assert_frame_equal(serial.realized, parallel.realized)


def test_warm_start_weights_passes_fold_isolation() -> None:
    # The conformance check must certify the same property for a stateful estimator. (On the old
    # shared-instance engine its serial-vs-fresh-serial comparison failed deterministically: run 1
    # left the estimator's state advanced, so run 3's first fold fit started from different state.)
    check_fold_isolation(_WarmStartWeights(), lambda: make_monthly_view(n=96, n_assets=4, seed=6))


def test_check_fold_isolation_rejects_single_fold_splitter() -> None:
    # A single-fold splitter never dispatches to the thread pool, so serial-vs-parallel identity
    # would be vacuous — the check demands at least two folds instead of certifying nothing.
    single = WalkForwardSplitter(min_train=40, test_size=20)  # 60 dates -> exactly one fold
    with pytest.raises(ValueError, match="at least two folds"):
        check_fold_isolation(
            _WarmStartWeights(),
            lambda: make_monthly_view(n=60, n_assets=3, seed=8),
            splitter=single,
        )


class _UncopyableEstimator:
    """Simulates an estimator pinning an un-copyable resource (a DB handle, an open socket)."""

    def __deepcopy__(self, memo: dict[int, object]) -> _UncopyableEstimator:
        raise RuntimeError("this resource cannot be copied")

    def fit(self, view: TimeSeriesView) -> _WarmStartWeightsModel:
        raise AssertionError("fit must never be reached when the deepcopy fails")


def test_non_deepcopyable_estimator_raises_contextual_typeerror() -> None:
    # The deepcopy failure surfaces as a TypeError naming the method and estimator type and citing
    # the contract, with the original exception chained as its cause.
    v = make_monthly_view(n=60, n_assets=3, seed=8)
    sp = WalkForwardSplitter(min_train=24, test_size=12)
    with pytest.raises(
        TypeError, match=r"db_method.*_UncopyableEstimator.*not deepcopy-able"
    ) as ei:
        backtest_weights(_UncopyableEstimator(), v, sp, method="db_method")
    assert isinstance(ei.value.__cause__, RuntimeError)


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


# -- cross-fold label-type mixing: validation is on str labels, pooling must be too ----------------


def _numeric_label_view(n: int = 48) -> TimeSeriesView:
    """A view whose asset labels are numeric strings — the ones an int label str-normalizes onto."""
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(17)
    ret = pd.DataFrame(rng.normal(0.01, 0.04, (n, 2)), index=idx, columns=["1", "2"])
    return TimeSeriesView(ret, horizon=1)


class _MixedLabelPricingModel:
    def __init__(self, as_int: bool) -> None:
        self._as_int = as_int

    def capabilities(self) -> set[str]:
        return {capabilities.TO_PRICING}

    def expected_returns(self, view: TimeSeriesView) -> pd.DataFrame:
        cols = [1, 2] if self._as_int else ["1", "2"]
        return pd.DataFrame(0.01, index=view.calendar, columns=cols)


class _MixedLabelPricing:
    """Emits int column labels on the first (shortest-train) fold and str labels on later ones."""

    def fit(self, view: TimeSeriesView) -> _MixedLabelPricingModel:
        return _MixedLabelPricingModel(as_int=len(view.calendar) <= 24)


def test_pricing_pools_mixed_type_labels_as_one_asset() -> None:
    # Bite: one fold emitting the int column 1 and another the str column "1" each passed the
    # str-based validation individually, but the pooled panels previously concatenated the ORIGINAL
    # labels — one asset silently became two half-empty columns. The driver now pools the
    # str-normalized labels, so the output carries exactly the view's assets, fully populated.
    v = _numeric_label_view()
    sp = WalkForwardSplitter(min_train=24, test_size=8)  # trains 24, 32, 40 -> int, str, str
    out = backtest_pricing(_MixedLabelPricing(), v, sp, method="mixed_labels")
    assert list(out.predicted.columns) == v.assets
    assert not bool(out.predicted.isna().to_numpy().any())


# -- empty pricing panels cannot smuggle malformed labels ------------------------------------------


class _EmptyPhantomPricingModel:
    def capabilities(self) -> set[str]:
        return {capabilities.TO_PRICING}

    def expected_returns(self, view: TimeSeriesView) -> pd.DataFrame:
        # Zero rows, but a column absent from the view: label validation must still fire.
        return pd.DataFrame(columns=[*view.assets, "PHANTOM"])


class _EmptyPhantomPricing:
    def fit(self, view: TimeSeriesView) -> _EmptyPhantomPricingModel:
        _ = view
        return _EmptyPhantomPricingModel()


def test_empty_pricing_panel_still_validates_columns() -> None:
    # The emptiness short-circuit previously ran before label validation, so a zero-row panel
    # bypassed it entirely; column checks now run first (the date checks stay vacuous on 0 rows).
    v = _pricing_view()
    sp = WalkForwardSplitter(min_train=24, test_size=12)
    with pytest.raises(ValueError, match="assets absent from the view"):
        backtest_pricing(_EmptyPhantomPricing(), v, sp, method="empty_phantom")
    with pytest.raises(ValueError, match="assets absent from the view"):
        backtest_pricing_in_sample(_EmptyPhantomPricing(), v, method="empty_phantom")


# -- check_output_shapes mirrors the driver guard exactly ------------------------------------------


class _ObjectIndexPricingModel:
    def capabilities(self) -> set[str]:
        return {capabilities.TO_PRICING}

    def expected_returns(self, view: TimeSeriesView) -> pd.DataFrame:
        # Timestamps in a plain object Index: set-containment in view.calendar still holds, but
        # the drivers demand a DatetimeIndex — the shape check must too.
        idx = pd.Index(list(view.calendar), dtype=object)
        return pd.DataFrame(0.01, index=idx, columns=view.assets)


class _ObjectIndexPricing:
    def fit(self, view: TimeSeriesView) -> _ObjectIndexPricingModel:
        _ = view
        return _ObjectIndexPricingModel()


class _StrCollidingPricingModel:
    def capabilities(self) -> set[str]:
        return {capabilities.TO_PRICING}

    def expected_returns(self, view: TimeSeriesView) -> pd.DataFrame:
        # [1, "1"]: unique as raw labels, colliding after the str normalization the engine pools on.
        return pd.DataFrame(0.01, index=view.calendar, columns=[1, "1"])


class _StrCollidingPricing:
    def fit(self, view: TimeSeriesView) -> _StrCollidingPricingModel:
        _ = view
        return _StrCollidingPricingModel()


def test_check_output_shapes_rejects_object_typed_date_index() -> None:
    with pytest.raises(AssertionError, match="DatetimeIndex"):
        check_output_shapes(_ObjectIndexPricing(), _pricing_view)


def test_check_output_shapes_rejects_str_colliding_columns() -> None:
    with pytest.raises(AssertionError, match="str normalization"):
        check_output_shapes(_StrCollidingPricing(), _numeric_label_view)


# -- the comparison harness fits an isolated copy, like the drivers --------------------------------


class _CountingPricingModel:
    def __init__(self, mu: pd.Series) -> None:
        self._mu = mu

    def capabilities(self) -> set[str]:
        return {capabilities.TO_PRICING}

    def expected_returns(self, view: TimeSeriesView) -> pd.DataFrame:
        row = self._mu.reindex([str(a) for a in view.assets]).to_numpy(np.float64)
        vals = np.tile(row, (len(view.calendar), 1))
        return pd.DataFrame(vals, index=view.calendar, columns=view.assets)


class _CountingPricing:
    """Counts fits on THIS instance — a fit routed through a deepcopy leaves it untouched."""

    def __init__(self) -> None:
        self.n_fits = 0

    def fit(self, view: TimeSeriesView) -> _CountingPricingModel:
        self.n_fits += 1
        mu = view.returns_frame().mean()
        mu.index = [str(c) for c in mu.index]
        return _CountingPricingModel(mu)


def test_compare_fits_isolated_copy() -> None:
    # compare() previously fitted the entry's estimator directly — the last user-fit path outside
    # the isolation contract. It now fits a deepcopy, so the caller's instance is never fitted.
    v = _pricing_view()
    est = _CountingPricing()
    rows = compare([ComparisonEntry(name="cp", estimator=est, train_view=v)], v)
    assert len(rows) >= 1
    assert est.n_fits == 0


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
