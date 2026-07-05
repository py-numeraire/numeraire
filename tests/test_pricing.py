"""Tests for the pricing capability: drivers, output, and the native pricing evaluators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import toy_panel_wide
from numeraire.core import capabilities
from numeraire.core.data import CrossSectionView, TimeSeriesView
from numeraire.core.engine import (
    PricingOutput,
    backtest_pricing,
    backtest_pricing_in_sample,
)
from numeraire.core.evaluators import (
    AverageAbsAlphaEvaluator,
    CrossSectionalR2Evaluator,
)
from numeraire.core.registry import available_evaluators, get_evaluator
from numeraire.core.schema import RESULT_COLUMNS, validate_result

# -- toy pricing estimators -----------------------------------------------------


class _TSPricingModel:
    """Unconditional pricer: each asset's train mean, broadcast across the view's calendar."""

    def __init__(self, mu: pd.Series) -> None:
        self._mu = mu

    def capabilities(self) -> set[str]:
        return {capabilities.TO_PRICING}

    def expected_returns(self, view: TimeSeriesView) -> pd.DataFrame:
        row = self._mu.reindex([str(a) for a in view.assets]).to_numpy(np.float64)
        vals = np.tile(row, (len(view.calendar), 1))
        return pd.DataFrame(vals, index=view.calendar, columns=view.assets)


class _TSPricing:
    def fit(self, view: TimeSeriesView) -> _TSPricingModel:
        mu = view.returns_frame().mean()
        mu.index = [str(c) for c in mu.index]
        return _TSPricingModel(mu)


class _CSPricingModel:
    """Cross-sectional pricer: the train grand-mean forward return, broadcast over date x asset."""

    def __init__(self, mu: float) -> None:
        self._mu = mu

    def capabilities(self) -> set[str]:
        return {capabilities.TO_PRICING}

    def expected_returns(self, view: CrossSectionView) -> pd.DataFrame:
        vals = np.full((len(view.calendar), len(view.assets)), self._mu, dtype=np.float64)
        return pd.DataFrame(vals, index=view.calendar, columns=view.assets)


class _CSPricing:
    def fit(self, view: CrossSectionView) -> _CSPricingModel:
        _keys, _x, y = view.aligned()
        return _CSPricingModel(float(y.mean()) if len(y) else 0.0)


def _ts_view(n: int = 60, n_assets: int = 5, seed: int = 4) -> TimeSeriesView:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(seed)
    ret = pd.DataFrame(
        rng.normal(0.01, 0.04, (n, n_assets)),
        index=idx,
        columns=[f"a{i}" for i in range(n_assets)],
    )
    return TimeSeriesView(ret, horizon=1)


# -- drivers --------------------------------------------------------------------


def test_walk_forward_pricing_pooled_panels() -> None:
    from numeraire.core.splitter import WalkForwardSplitter

    v = _ts_view()
    sp = WalkForwardSplitter(min_train=24, test_size=12)
    out = backtest_pricing(_TSPricing(), v, sp, method="ts_pricer", config={"h": 1})
    assert isinstance(out, PricingOutput)
    assert out.capability == capabilities.TO_PRICING
    assert out.protocol == "walk_forward"
    assert out.run_id == f"ts_pricer-{out.config_hash}"
    assert out.predicted.index.equals(out.realized.index)
    assert list(out.predicted.columns) == v.assets
    # every scored date has at least one realized asset (unrealized tail dropped)
    assert not out.realized.isna().to_numpy().all(axis=1).any()
    assert out.predicted.index.is_monotonic_increasing


def test_pricing_in_sample_labels_protocol() -> None:
    v = _ts_view()
    out = backtest_pricing_in_sample(_TSPricing(), v, method="ts_pricer")
    assert out.protocol == "in_sample"
    assert not out.predicted.empty
    # a single full-sample fit broadcasts one row -> all prediction rows identical
    first = out.predicted.iloc[0].to_numpy()
    assert np.allclose(out.predicted.to_numpy(), first)


def test_walk_forward_pricing_on_cross_section() -> None:
    from numeraire.core.splitter import WalkForwardSplitter

    panel = toy_panel_wide(n_months=48, n_assets=8, seed=9)
    v = CrossSectionView(panel, chars=["size", "bm", "mom"], horizon=1)
    sp = WalkForwardSplitter(min_train=24, test_size=12)
    out = backtest_pricing(_CSPricing(), v, sp, method="cs_pricer")
    assert isinstance(out, PricingOutput)
    assert out.protocol == "walk_forward"
    assert not out.predicted.empty
    # ragged: realized carries NaN where an asset is absent, but no fully-empty scored date
    assert not out.realized.isna().to_numpy().all(axis=1).any()


def test_walk_forward_pricing_rejects_non_pricing_model() -> None:
    from numeraire.core.splitter import WalkForwardSplitter

    class _NotPricingModel:
        def capabilities(self) -> set[str]:
            return {capabilities.TO_WEIGHTS}

    class _NotPricing:
        def fit(self, view: TimeSeriesView) -> _NotPricingModel:
            _ = view
            return _NotPricingModel()

    v = _ts_view(n=40)
    sp = WalkForwardSplitter(min_train=20, test_size=10)
    with pytest.raises(TypeError, match="does not support 'to_pricing'"):
        backtest_pricing(_NotPricing(), v, sp, method="bad")


# -- evaluators -----------------------------------------------------------------


def _pricing_output(
    predicted: pd.DataFrame, realized: pd.DataFrame, protocol: str
) -> PricingOutput:
    return PricingOutput(
        predicted=predicted,
        realized=realized,
        method="toy",
        config_hash="deadbeef",
        data_vintage="2026-07",
        run_id="toy-deadbeef",
        protocol=protocol,
    )


def test_pricing_evaluators_registered() -> None:
    assert "xs_r2" in available_evaluators()
    assert "avg_abs_alpha" in available_evaluators()
    assert get_evaluator("xs_r2").requires == {capabilities.TO_PRICING}


def test_xs_r2_is_one_when_priced_exactly() -> None:
    idx = pd.date_range("2000-01-31", periods=6, freq="ME")
    cols = ["a", "b", "c", "d"]
    rng = np.random.default_rng(1)
    panel = pd.DataFrame(rng.normal(0.01, 0.03, (6, 4)), index=idx, columns=cols)
    out = _pricing_output(panel, panel, "in_sample")  # predicted == realized => exact pricing
    r2 = CrossSectionalR2Evaluator().evaluate(out)
    validate_result(r2)
    assert r2.iloc[0]["metric"] == "xs_r2"
    assert r2.iloc[0]["protocol"] == "in_sample"
    np.testing.assert_allclose(r2.iloc[0]["value"], 1.0)
    aaa = AverageAbsAlphaEvaluator().evaluate(out)
    np.testing.assert_allclose(aaa.iloc[0]["value"], 0.0, atol=1e-12)


def test_xs_r2_matches_manual_ols_and_ignores_time_ordering() -> None:
    idx = pd.date_range("2000-01-31", periods=8, freq="ME")
    cols = ["a", "b", "c", "d", "e"]
    rng = np.random.default_rng(2)
    predicted = pd.DataFrame(rng.normal(0.01, 0.02, (8, 5)), index=idx, columns=cols)
    realized = pd.DataFrame(rng.normal(0.01, 0.05, (8, 5)), index=idx, columns=cols)
    out = _pricing_output(predicted, realized, "walk_forward")
    mp = predicted.mean().to_numpy()
    mr = realized.mean().to_numpy()
    x = np.column_stack([np.ones(5), mp])
    coef, *_ = np.linalg.lstsq(x, mr, rcond=None)
    resid = mr - x @ coef
    expected = 1.0 - float(resid @ resid) / float(((mr - mr.mean()) ** 2).sum())
    got = CrossSectionalR2Evaluator().evaluate(out).iloc[0]["value"]
    np.testing.assert_allclose(got, expected)
    # broadcast (constant-in-time) predicted reduces to the same per-asset means -> same R^2
    broadcast = pd.DataFrame(np.tile(mp, (8, 1)), index=idx, columns=cols)
    out_b = _pricing_output(broadcast, realized, "walk_forward")
    np.testing.assert_allclose(
        CrossSectionalR2Evaluator().evaluate(out_b).iloc[0]["value"], expected
    )


def test_avg_abs_alpha_matches_manual() -> None:
    idx = pd.date_range("2000-01-31", periods=5, freq="ME")
    cols = ["a", "b", "c"]
    predicted = pd.DataFrame(0.01, index=idx, columns=cols)
    realized = pd.DataFrame({"a": 0.02, "b": 0.00, "c": 0.03}, index=idx)
    out = _pricing_output(predicted, realized, "in_sample")
    got = AverageAbsAlphaEvaluator().evaluate(out).iloc[0]["value"]
    expected = np.mean(np.abs(np.array([0.02, 0.00, 0.03]) - 0.01))
    np.testing.assert_allclose(got, expected)


def test_xs_r2_handles_all_nan_asset_column() -> None:
    idx = pd.date_range("2000-01-31", periods=4, freq="ME")
    predicted = pd.DataFrame({"a": [0.01] * 4, "b": [0.02] * 4, "c": [np.nan] * 4}, index=idx)
    realized = pd.DataFrame({"a": [0.02] * 4, "b": [0.01] * 4, "c": [np.nan] * 4}, index=idx)
    out = _pricing_output(predicted, realized, "walk_forward")
    # asset c (all-NaN) is dropped; the metric still computes over a and b
    val = CrossSectionalR2Evaluator().evaluate(out).iloc[0]["value"]
    assert np.isfinite(val)


def test_pricing_evaluators_reject_wrong_output() -> None:
    with pytest.raises(TypeError):
        CrossSectionalR2Evaluator().evaluate(object())
    with pytest.raises(TypeError):
        AverageAbsAlphaEvaluator().evaluate(object())


class _EmptyPricingModel:
    def capabilities(self) -> set[str]:
        return {capabilities.TO_PRICING}

    def expected_returns(self, view: TimeSeriesView) -> pd.DataFrame:
        return pd.DataFrame(columns=view.assets)  # prices nothing


class _EmptyPricing:
    def fit(self, view: TimeSeriesView) -> _EmptyPricingModel:
        _ = view
        return _EmptyPricingModel()


def test_walk_forward_pricing_empty_when_nothing_priced() -> None:
    from numeraire.core.splitter import WalkForwardSplitter

    v = _ts_view(n=40)
    sp = WalkForwardSplitter(min_train=20, test_size=10)
    out = backtest_pricing(_EmptyPricing(), v, sp, method="empty")
    assert out.predicted.empty
    assert list(out.predicted.columns) == v.assets


def test_pricing_in_sample_empty_when_nothing_priced() -> None:
    v = _ts_view(n=40)
    out = backtest_pricing_in_sample(_EmptyPricing(), v, method="empty")
    assert out.predicted.empty
    assert out.protocol == "in_sample"


def test_pricing_realized_and_finalize_helpers() -> None:
    from numeraire.core.engine import _finalize_pricing, _pricing_realized

    idx = pd.date_range("2000-01-31", periods=3, freq="ME")
    predicted = pd.DataFrame({"a": [0.0, 0.0, 0.0]}, index=idx)
    # unknown view type is rejected
    with pytest.raises(TypeError, match="cannot align realized returns"):
        _pricing_realized(object(), predicted)
    # empty predicted short-circuits finalize
    empty = pd.DataFrame()
    p, r = _finalize_pricing(empty, empty)
    assert p.empty and r.empty
    # all-unrealized rows are dropped
    realized = pd.DataFrame({"a": [np.nan, np.nan, np.nan]}, index=idx)
    p2, r2 = _finalize_pricing(predicted, realized)
    assert p2.empty and r2.empty


def test_protocol_column_in_schema_and_existing_rows_walk_forward() -> None:
    assert "protocol" in RESULT_COLUMNS
    # a weights output (no protocol field) reports the intrinsic "walk_forward" protocol
    from numeraire.core.engine import WeightsOutput
    from numeraire.core.evaluators import SharpeEvaluator

    idx = pd.date_range("2000-01-31", periods=4, freq="ME")
    w = WeightsOutput(
        weights=pd.DataFrame({"r0": [1.0] * 4}, index=idx),
        realized=pd.DataFrame({"r0": [0.02, -0.01, 0.03, 0.01]}, index=idx),
        method="toy",
        config_hash="dead",
        data_vintage="2026",
        run_id="toy-dead",
    )
    row = SharpeEvaluator().evaluate(w).iloc[0]
    assert row["protocol"] == "walk_forward"
