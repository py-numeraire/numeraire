"""Missing-return scoring is explicit, fail-closed, and target-weight preserving.

These tests pin the distinction between a model decision and the ex-post weights used to score it.
Ordinary missing returns never rewrite target exposures or disappear from the sample; only a
mechanically unrealized horizon tail is removed by a driver.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from numeraire import (
    CrossSectionView,
    ExposureEvaluator,
    PanelWeightsOutput,
    TimeSeriesView,
    WalkForwardSplitter,
    WeightsOutput,
    backtest_panel,
    backtest_weights,
)


def _wide_output(returns: list[float], *, missing_returns: str = "error") -> WeightsOutput:
    date = pd.DatetimeIndex(["2000-01-31"])
    columns = ["long_observed", "long_missing", "short_a", "short_b"]
    weights = pd.DataFrame([[0.5, 0.5, -0.5, -0.5]], index=date, columns=columns)
    realized = pd.DataFrame([returns], index=date, columns=columns)
    return WeightsOutput(
        weights=weights,
        realized=realized,
        method="toy",
        config_hash="cfg",
        data_vintage="synthetic",
        run_id="toy-cfg",
        missing_returns=missing_returns,  # type: ignore[arg-type]
    )


def _as_panel(output: WeightsOutput) -> PanelWeightsOutput:
    index = pd.MultiIndex.from_product(
        [output.weights.index, output.weights.columns], names=["date", "asset"]
    )
    weights = pd.Series(output.weights.to_numpy().reshape(-1), index=index, name="weight")
    realized = pd.Series(output.realized.to_numpy().reshape(-1), index=index, name="realized")
    return PanelWeightsOutput(
        weights=weights,
        realized=realized,
        method=output.method,
        config_hash=output.config_hash,
        data_vintage=output.data_vintage,
        run_id=output.run_id,
        missing_returns=output.missing_returns,
    )


def test_default_policy_fails_on_held_missing_return() -> None:
    output = _wide_output([0.10, np.nan, 0.02, 0.04])
    with pytest.raises(ValueError, match="long_missing"):
        output.strategy_returns()


def test_renormalize_legs_preserves_target_and_scores_seven_percent() -> None:
    output = _wide_output([0.10, np.nan, 0.02, 0.04], missing_returns="renormalize_legs")
    target = output.weights.copy()
    expected = pd.DataFrame([[1.0, 0.0, -0.5, -0.5]], index=target.index, columns=target.columns)

    pd.testing.assert_frame_equal(output.scoring_weights(), expected)
    pd.testing.assert_frame_equal(output.weights, target)
    np.testing.assert_allclose(output.strategy_returns().iloc[0], 0.07)

    exposure = ExposureEvaluator().evaluate(output).set_index("metric")["value"]
    np.testing.assert_allclose(exposure["gross_leverage"], 2.0)
    np.testing.assert_allclose(exposure["net_exposure"], 0.0)
    np.testing.assert_allclose(exposure["hhi"], 1.0)  # effective scoring weights would be 1.5


def test_renormalize_legs_rejects_an_unidentified_whole_leg() -> None:
    output = _wide_output([np.nan, np.nan, 0.02, 0.04], missing_returns="renormalize_legs")
    with pytest.raises(ValueError, match="all returns in the positive leg"):
        output.strategy_returns()


def test_missing_return_on_zero_weight_is_harmless() -> None:
    date = pd.DatetimeIndex(["2000-01-31"])
    output = WeightsOutput(
        weights=pd.DataFrame([[1.0, 0.0]], index=date, columns=["held", "unheld"]),
        realized=pd.DataFrame([[0.03, np.nan]], index=date, columns=["held", "unheld"]),
        method="toy",
        config_hash="cfg",
        data_vintage="synthetic",
        run_id="toy-cfg",
    )
    pd.testing.assert_frame_equal(output.scoring_weights(), output.weights)
    np.testing.assert_allclose(output.strategy_returns().iloc[0], 0.03)


def test_wide_and_panel_outputs_apply_the_same_policy() -> None:
    wide = _wide_output([0.10, np.nan, 0.02, 0.04], missing_returns="renormalize_legs")
    panel = _as_panel(wide)
    np.testing.assert_allclose(
        panel.scoring_weights().to_numpy(),
        wide.scoring_weights().to_numpy().reshape(-1),
    )
    pd.testing.assert_series_equal(
        panel.strategy_returns(), wide.strategy_returns(), check_names=False
    )


def test_zero_policy_is_explicit_and_keeps_target_weights() -> None:
    output = _wide_output([0.10, np.nan, 0.02, 0.04], missing_returns="zero")
    pd.testing.assert_frame_equal(output.scoring_weights(), output.weights)
    np.testing.assert_allclose(output.strategy_returns().iloc[0], 0.02)


def test_nonfinite_target_weight_is_rejected() -> None:
    output = _wide_output([0.10, 0.08, 0.02, 0.04])
    output.weights.iloc[0, 0] = np.nan
    with pytest.raises(ValueError, match="non-finite target weight"):
        output.scoring_weights()


def test_nonfinite_target_weight_is_rejected_at_construction() -> None:
    date = pd.DatetimeIndex(["2000-01-31"])
    with pytest.raises(ValueError, match="target weights must all be finite"):
        WeightsOutput(
            weights=pd.DataFrame([[np.inf]], index=date, columns=["a"]),
            realized=pd.DataFrame([[0.01]], index=date, columns=["a"]),
            method="toy",
            config_hash="cfg",
            data_vintage="synthetic",
            run_id="toy-cfg",
        )


def test_invalid_policy_and_misaligned_axes_are_rejected() -> None:
    with pytest.raises(ValueError, match="missing_returns must be one of"):
        _wide_output([0.10, 0.08, 0.02, 0.04], missing_returns="guess")

    date = pd.DatetimeIndex(["2000-01-31"])
    with pytest.raises(ValueError, match="identical indexes and columns"):
        WeightsOutput(
            weights=pd.DataFrame([[1.0]], index=date, columns=["a"]),
            realized=pd.DataFrame([[0.01]], index=date, columns=["b"]),
            method="toy",
            config_hash="cfg",
            data_vintage="synthetic",
            run_id="toy-cfg",
        )

    duplicate = pd.MultiIndex.from_tuples([(date[0], "a"), (date[0], "a")], names=["date", "asset"])
    with pytest.raises(ValueError, match="unique"):
        PanelWeightsOutput(
            weights=pd.Series([0.5, 0.5], index=duplicate),
            realized=pd.Series([0.01, 0.01], index=duplicate),
            method="toy",
            config_hash="cfg",
            data_vintage="synthetic",
            run_id="toy-cfg",
        )


def test_new_policy_field_preserves_existing_positional_output_fields() -> None:
    date = pd.DatetimeIndex(["2000-01-31"])
    weights = pd.DataFrame([[1.0]], index=date, columns=["a"])
    realized = pd.DataFrame([[0.01]], index=date, columns=["a"])
    output = WeightsOutput(
        weights,
        realized,
        "toy",
        "cfg",
        "synthetic",
        "toy-cfg",
        "custom_capability",
        {"source": "manual"},
    )
    assert output.capability == "custom_capability"
    assert output.meta == {"source": "manual"}
    assert output.missing_returns == "error"


class _OneModel:
    def capabilities(self) -> set[str]:
        return {"to_weights"}

    def to_weights(self, view: TimeSeriesView) -> pd.DataFrame:
        return pd.DataFrame(1.0, index=view.calendar, columns=view.assets)


class _OneEstimator:
    def fit(self, view: TimeSeriesView) -> _OneModel:
        return _OneModel()


class _OutsideFoldModel:
    def __init__(self, date: pd.Timestamp) -> None:
        self._date = date

    def capabilities(self) -> set[str]:
        return {"to_weights"}

    def to_weights(self, view: TimeSeriesView) -> pd.DataFrame:
        return pd.DataFrame(1.0, index=pd.DatetimeIndex([self._date]), columns=view.assets)


class _OutsideFoldEstimator:
    def __init__(self, date: pd.Timestamp) -> None:
        self._date = date

    def fit(self, view: TimeSeriesView) -> _OutsideFoldModel:
        return _OutsideFoldModel(self._date)


class _PanelOneModel:
    def capabilities(self) -> set[str]:
        return {"to_weights"}

    def to_weights(self, view: CrossSectionView) -> pd.Series:
        dates: list[pd.Timestamp] = []
        assets: list[object] = []
        for date in view.calendar:
            ids, _features = view.features_asof(date)
            dates.extend([date] * len(ids))
            assets.extend(ids)
        index = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex(dates), assets], names=["date", "asset"]
        )
        return pd.Series(1.0, index=index, name="weight")


class _PanelOneEstimator:
    def fit(self, view: CrossSectionView) -> _PanelOneModel:
        return _PanelOneModel()


class _GhostPanelModel(_PanelOneModel):
    def to_weights(self, view: CrossSectionView) -> pd.Series:
        weights = super().to_weights(view)
        ghost = pd.MultiIndex.from_tuples([(view.calendar[0], "GHOST")], names=["date", "asset"])
        return pd.concat([weights, pd.Series([1.0], index=ghost, name="weight")])


class _GhostPanelEstimator:
    def fit(self, view: CrossSectionView) -> _GhostPanelModel:
        return _GhostPanelModel()


class _TailGhostPanelModel(_PanelOneModel):
    def to_weights(self, view: CrossSectionView) -> pd.Series:
        weights = super().to_weights(view)
        ghost = pd.MultiIndex.from_tuples([(view.calendar[-1], "GHOST")], names=["date", "asset"])
        return pd.concat([weights, pd.Series([1.0], index=ghost, name="weight")])


class _TailGhostPanelEstimator:
    def fit(self, view: CrossSectionView) -> _TailGhostPanelModel:
        return _TailGhostPanelModel()


class _StringPanelModel(_PanelOneModel):
    def to_weights(self, view: CrossSectionView) -> pd.Series:
        weights = super().to_weights(view)
        index = pd.MultiIndex.from_arrays(
            [
                weights.index.get_level_values("date"),
                [str(a) for a in weights.index.get_level_values("asset")],
            ],
            names=["date", "asset"],
        )
        return pd.Series(weights.to_numpy(), index=index, name="weight")


class _StringPanelEstimator:
    def fit(self, view: CrossSectionView) -> _StringPanelModel:
        return _StringPanelModel()


class _PanelFrameModel:
    def capabilities(self) -> set[str]:
        return {"to_weights"}

    def to_weights(self, view: CrossSectionView) -> pd.DataFrame:
        return _PanelOneModel().to_weights(view).to_frame("weight")


class _PanelFrameEstimator:
    def fit(self, view: CrossSectionView) -> _PanelFrameModel:
        return _PanelFrameModel()


class _EmptySeriesModel:
    def capabilities(self) -> set[str]:
        return {"to_weights"}

    def to_weights(self, view: TimeSeriesView) -> pd.Series:
        return pd.Series(dtype=np.float64)


class _EmptySeriesEstimator:
    def fit(self, view: TimeSeriesView) -> _EmptySeriesModel:
        return _EmptySeriesModel()


class _MalformedEmptyPanelModel:
    def capabilities(self) -> set[str]:
        return {"to_weights"}

    def to_weights(self, view: CrossSectionView) -> pd.Series:
        return pd.Series(dtype=np.float64)


class _MalformedEmptyPanelEstimator:
    def fit(self, view: CrossSectionView) -> _MalformedEmptyPanelModel:
        return _MalformedEmptyPanelModel()


def _missing_view() -> TimeSeriesView:
    dates = pd.date_range("2000-01-31", periods=5, freq="ME")
    # The missing return at dates[3] is the held outcome for the scoreable dates[2] decision.
    returns = pd.DataFrame(
        {
            "missing_asset": [0.01, 0.02, 0.03, np.nan, 0.05],
            "observed_asset": [0.02, 0.01, 0.04, 0.06, 0.03],
        },
        index=dates,
    )
    return TimeSeriesView(returns)


def _missing_panel_view() -> CrossSectionView:
    dates = pd.date_range("2000-01-31", periods=5, freq="ME")
    rows: list[dict[str, object]] = []
    for date, a_return, b_return in zip(
        dates,
        [0.01, 0.02, 0.03, np.nan, 0.05],
        [0.02, 0.01, 0.04, 0.06, 0.03],
        strict=True,
    ):
        rows.extend(
            [
                {"date": date, "asset": "A", "x": 1.0, "ret": a_return},
                {"date": date, "asset": "B", "x": 1.0, "ret": b_return},
            ]
        )
    return CrossSectionView(pd.DataFrame(rows), chars=["x"])


def _integer_asset_panel_view() -> CrossSectionView:
    frame = _missing_panel_view().panel_frame().reset_index()
    frame["asset"] = frame["asset"].map({"A": 1, "B": 2})
    return CrossSectionView(frame, chars=["x"])


def test_driver_drops_only_structural_tail_and_errors_on_earlier_missing() -> None:
    view = _missing_view()
    splitter = WalkForwardSplitter(min_train=2, test_size=3)
    with pytest.raises(ValueError, match="2000-03-31"):
        backtest_weights(_OneEstimator(), view, splitter, method="one")

    output = backtest_weights(_OneEstimator(), view, splitter, method="one", missing_returns="zero")
    assert output.weights.index.equals(view.calendar[2:4])
    assert output.realized.index.equals(output.weights.index)
    assert np.isnan(output.realized.iloc[0, 0])
    assert view.calendar[-1] not in output.weights.index


def test_driver_rejects_weight_dates_outside_the_current_test_fold() -> None:
    view = _missing_view()
    with pytest.raises(ValueError, match="outside the current test fold"):
        backtest_weights(
            _OutsideFoldEstimator(view.calendar[0]),
            view,
            WalkForwardSplitter(min_train=2, test_size=3),
            method="outside",
            missing_returns="zero",
        )


def test_panel_driver_preserves_ordinary_missing_keys_and_drops_only_tail() -> None:
    view = _missing_panel_view()
    splitter = WalkForwardSplitter(min_train=2, test_size=3)
    with pytest.raises(ValueError, match="2000-03-31"):
        backtest_panel(_PanelOneEstimator(), view, splitter, method="panel_one")

    output = backtest_panel(
        _PanelOneEstimator(),
        view,
        splitter,
        method="panel_one",
        missing_returns="zero",
    )
    decision_dates = output.weights.index.get_level_values("date").unique()
    assert decision_dates.equals(view.calendar[2:4])
    assert output.weights.index.equals(output.realized.index)
    assert np.isnan(output.realized.loc[(view.calendar[2], "A")])
    assert (view.calendar[-1], "A") not in output.weights.index


def test_panel_driver_rejects_asset_absent_from_the_formation_universe() -> None:
    with pytest.raises(ValueError, match="formation universe"):
        backtest_panel(
            _GhostPanelEstimator(),
            _missing_panel_view(),
            WalkForwardSplitter(min_train=2, test_size=3),
            method="ghost",
            missing_returns="zero",
        )

    # Structural-tail filtering is not a schema escape hatch: invalid target keys are rejected
    # even when they occur only at the final, mechanically unscoreable origin.
    with pytest.raises(ValueError, match="formation universe"):
        backtest_panel(
            _TailGhostPanelEstimator(),
            _missing_panel_view(),
            WalkForwardSplitter(min_train=2, test_size=3),
            method="tail_ghost",
            missing_returns="zero",
        )


def test_panel_driver_aligns_string_labels_to_nonstring_view_assets() -> None:
    view = _integer_asset_panel_view()
    splitter = WalkForwardSplitter(min_train=2, test_size=3)
    original = backtest_panel(
        _PanelOneEstimator(), view, splitter, method="original_ids", missing_returns="zero"
    )
    strings = backtest_panel(
        _StringPanelEstimator(), view, splitter, method="string_ids", missing_returns="zero"
    )
    assert set(original.weights.index.get_level_values("asset")) == {1, 2}
    assert set(strings.weights.index.get_level_values("asset")) == {"1", "2"}
    np.testing.assert_allclose(original.weights.to_numpy(), strings.weights.to_numpy())
    np.testing.assert_allclose(original.realized.to_numpy(), strings.realized.to_numpy())
    pd.testing.assert_series_equal(original.strategy_returns(), strings.strategy_returns())


def test_panel_one_column_frame_matches_driver_and_conformance_contract() -> None:
    from numeraire.testing import check_output_shapes

    check_output_shapes(_PanelFrameEstimator(), _missing_panel_view)
    output = backtest_panel(
        _PanelFrameEstimator(),
        _missing_panel_view(),
        WalkForwardSplitter(min_train=2, test_size=3),
        method="panel_frame",
        missing_returns="zero",
    )
    assert not output.weights.empty


def test_drivers_collect_scoring_metadata_without_materializing_effective_outputs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numeraire.core.engine as engine

    def _unexpected(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("driver eagerly materialized effective scoring weights")

    monkeypatch.setattr(engine, "_wide_scoring", _unexpected)
    monkeypatch.setattr(engine, "_panel_scoring", _unexpected)

    wide = backtest_weights(
        _OneEstimator(),
        _missing_view(),
        WalkForwardSplitter(min_train=2, test_size=3),
        method="wide_stats",
        missing_returns="zero",
    )
    panel = backtest_panel(
        _PanelOneEstimator(),
        _missing_panel_view(),
        WalkForwardSplitter(min_train=2, test_size=3),
        method="panel_stats",
        missing_returns="zero",
    )
    assert wide.meta["missing_held"] == 1
    assert panel.meta["missing_held"] == 1


def test_driver_metadata_validation_rejects_a_whole_missing_leg() -> None:
    wide_returns = _missing_view().returns_frame()
    wide_returns.loc[wide_returns.index[3]] = np.nan
    wide_view = TimeSeriesView(wide_returns)
    with pytest.raises(ValueError, match="all returns in the positive leg"):
        backtest_weights(
            _OneEstimator(),
            wide_view,
            WalkForwardSplitter(min_train=2, test_size=3),
            method="wide_whole_leg",
            missing_returns="renormalize_legs",
        )

    panel_frame = _missing_panel_view().panel_frame().reset_index()
    panel_frame.loc[panel_frame["date"].eq(panel_frame["date"].unique()[3]), "ret"] = np.nan
    panel_view = CrossSectionView(panel_frame, chars=["x"])
    with pytest.raises(ValueError, match="all returns in the positive leg"):
        backtest_panel(
            _PanelOneEstimator(),
            panel_view,
            WalkForwardSplitter(min_train=2, test_size=3),
            method="panel_whole_leg",
            missing_returns="renormalize_legs",
        )


def test_empty_outputs_must_still_obey_their_driver_schema() -> None:
    splitter = WalkForwardSplitter(min_train=2, test_size=3)
    with pytest.raises(TypeError, match="date x asset DataFrame"):
        backtest_weights(_EmptySeriesEstimator(), _missing_view(), splitter, method="empty_wide")
    with pytest.raises(TypeError, match="MultiIndex"):
        backtest_panel(
            _MalformedEmptyPanelEstimator(),
            _missing_panel_view(),
            splitter,
            method="empty_panel",
        )


def test_policy_is_hashed_and_recorded_in_metadata() -> None:
    view = _missing_view()
    splitter = WalkForwardSplitter(min_train=2, test_size=3)
    zero = backtest_weights(_OneEstimator(), view, splitter, method="one", missing_returns="zero")
    renormalized = backtest_weights(
        _OneEstimator(), view, splitter, method="one", missing_returns="renormalize_legs"
    )
    assert zero.config_hash != renormalized.config_hash
    assert zero.meta == {
        "missing_returns": "zero",
        "missing_held": 1,
        "missing_dates": 1,
        "renormalized_dates": 0,
        "frequency": "ME",  # target-contract metadata now travels alongside the scoring stats
    }
    assert renormalized.meta["renormalized_dates"] == 1


def test_config_cannot_conflict_with_explicit_policy() -> None:
    with pytest.raises(ValueError, match="conflicts"):
        backtest_weights(
            _OneEstimator(),
            _missing_view(),
            WalkForwardSplitter(min_train=2, test_size=3),
            method="one",
            config={"missing_returns": "error"},
            missing_returns="zero",
        )
