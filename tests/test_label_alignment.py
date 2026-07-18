"""Label-alignment + dispatch-guard regressions for the walk-forward engine.

The headline test (:func:`test_weights_scored_by_label_not_position`) pins the correctness fix:
a ``to_weights`` model that emits its columns in a permuted / subset order relative to
``view.assets`` must score **identically** to the same asset→weight assignment in canonical order.
Under the old positional pairing (``weights.to_numpy() * realized.to_numpy()``) the permuted model
was silently mis-scored; the engine now reindexes returned weights to ``view.assets`` before
pairing them with realized returns.
"""

from __future__ import annotations

import typing

import numpy as np
import pandas as pd
import pytest

from conftest import make_monthly_view
from numeraire import (
    SupportsWeights,
    backtest,
    backtest_weights,
)
from numeraire.core import capabilities
from numeraire.core.data import TimeSeriesView
from numeraire.core.splitter import WalkForwardSplitter


class _AssetWeightModel:
    """Emits a fixed asset→weight assignment, with the returned column ORDER configurable.

    Weights are asset-specific (not equal), so a positional vs label pairing with realized returns
    give *different* strategy returns unless the engine aligns by label.
    """

    def __init__(self, weight_map: dict[str, float], order: list[str]) -> None:
        self._wm = weight_map
        self._order = order

    def capabilities(self) -> set[str]:
        return {capabilities.TO_WEIGHTS}

    def to_weights(self, view: TimeSeriesView) -> pd.DataFrame:
        row = [self._wm[c] for c in self._order]
        rows = np.tile(row, (len(view.calendar), 1))
        return pd.DataFrame(rows, index=view.calendar, columns=self._order)


class _AssetWeightEst:
    def __init__(self, weight_map: dict[str, float], order: list[str]) -> None:
        self._wm = weight_map
        self._order = order

    def fit(self, view: TimeSeriesView) -> _AssetWeightModel:
        return _AssetWeightModel(self._wm, self._order)


def _view() -> TimeSeriesView:
    return make_monthly_view(n=120, n_assets=3)


def _splitter() -> WalkForwardSplitter:
    return WalkForwardSplitter(min_train=60, test_size=12)


def test_weights_scored_by_label_not_position() -> None:
    """Permuted-column weights score IDENTICALLY to canonical-order weights (fails pre-fix)."""
    assets = _view().assets  # ["r0", "r1", "r2"]
    wm = {"r0": 0.5, "r1": 0.3, "r2": 0.2}  # asset-specific so order matters

    canonical = backtest_weights(
        _AssetWeightEst(wm, list(assets)), _view(), _splitter(), method="canonical"
    )
    reversed_ = backtest_weights(
        _AssetWeightEst(wm, list(reversed(assets))), _view(), _splitter(), method="reversed"
    )

    # After the fix the two runs are bit-identical strategy P&L; pre-fix the reversed model paired
    # r2's weight against r0's realized return (etc.) and diverged.
    pd.testing.assert_series_equal(
        canonical.strategy_returns(),
        reversed_.strategy_returns(),
        check_names=False,
    )
    # And the stored weights are in canonical asset order regardless of what the model emitted.
    assert list(reversed_.weights.columns) == list(assets)


def test_weights_subset_columns_scored_by_label() -> None:
    """A model omitting an asset scores like a full model that zero-weights it (label alignment)."""
    assets = _view().assets
    wm_full = {"r0": 0.6, "r1": 0.4, "r2": 0.0}
    full = backtest_weights(
        _AssetWeightEst(wm_full, list(assets)), _view(), _splitter(), method="full"
    )
    # Same non-zero assignment but the model never mentions r2 (dropped column, not zero).
    subset = backtest_weights(
        _AssetWeightEst({"r0": 0.6, "r1": 0.4}, ["r0", "r1"]),
        _view(),
        _splitter(),
        method="subset",
    )
    pd.testing.assert_series_equal(
        full.strategy_returns(), subset.strategy_returns(), check_names=False
    )
    assert bool((subset.weights["r2"] == 0.0).all())


# -- dispatch guards (items 2 & 3) -----------------------------------------------------------------


def test_backtest_weights_route_requires_splitter() -> None:
    with pytest.raises(TypeError, match="requires a `splitter`"):
        backtest(
            _AssetWeightEst({"r0": 0.5, "r1": 0.3, "r2": 0.2}, ["r0", "r1", "r2"]),
            _view(),
            method="w",
        )


class _ForecastModel:
    def capabilities(self) -> set[str]:
        return {capabilities.TO_FORECAST}

    def forecast(self, view: TimeSeriesView) -> pd.Series:
        return view.returns_frame().mean()


class _ForecastEst:
    def fit(self, view: TimeSeriesView) -> _ForecastModel:
        return _ForecastModel()


def test_backtest_forecast_route_rejects_splitter() -> None:
    with pytest.raises(TypeError, match="not a splitter"):
        backtest(_ForecastEst(), _view(), _splitter(), method="f", min_train=24)


# -- widened type contract (item 4) + exports (item 5) ---------------------------------------------


def test_supports_weights_return_annotation_allows_series() -> None:
    hints = typing.get_type_hints(SupportsWeights.to_weights)
    args = typing.get_args(hints["return"])
    assert pd.DataFrame in args and pd.Series in args


def test_block_helpers_importable_from_top_level() -> None:
    import numeraire

    for name in ["FeatureBlock", "VintagedBlock", "CharBlock", "PanelTensor"]:
        assert name in numeraire.__all__
        assert getattr(numeraire, name, None) is not None


def test_check_output_shapes_exported() -> None:
    from numeraire import testing

    assert "check_output_shapes" in testing.__all__
    assert callable(testing.check_output_shapes)


def test_check_output_shapes_rejects_duplicate_weight_columns() -> None:
    """The strengthened shape check flags duplicate weight labels (ambiguous alignment)."""
    from numeraire import testing

    class _DupModel:
        def capabilities(self) -> set[str]:
            return {capabilities.TO_WEIGHTS}

        def to_weights(self, view: TimeSeriesView) -> pd.DataFrame:
            n = len(view.calendar)
            return pd.DataFrame(np.full((n, 2), 0.5), index=view.calendar, columns=["r0", "r0"])

    class _DupEst:
        def fit(self, view: TimeSeriesView) -> _DupModel:
            return _DupModel()

    with pytest.raises(AssertionError, match="unique labels"):
        testing.check_output_shapes(_DupEst(), _view)


def test_check_output_shapes_rejects_nonfinite_target_weights() -> None:
    """NaN is not an implicit zero-weight convention in the estimator contract."""
    from numeraire import testing

    assets = _view().assets
    weights = {assets[0]: np.nan, assets[1]: 0.5, assets[2]: 0.5}
    with pytest.raises(AssertionError, match="finite"):
        testing.check_output_shapes(_AssetWeightEst(weights, assets), _view)
