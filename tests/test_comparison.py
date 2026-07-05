"""Tests for the within-capability comparison harness on a common test-asset panel."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from numeraire.comparison import ComparisonEntry, compare
from numeraire.core import capabilities
from numeraire.core.data import CrossSectionView, TimeSeriesView
from numeraire.core.schema import validate_result

ASSETS = ["s1", "s2", "s3", "s4", "s5"]


def _canonical(n: int = 24, seed: int = 7) -> pd.DataFrame:
    idx = pd.date_range("2000-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(seed)
    return pd.DataFrame(rng.normal(0.01, 0.04, (n, len(ASSETS))), index=idx, columns=ASSETS)


# -- toy pricing estimators (both view shapes, priced on shared labels) ----------


class _TSMeanModel:
    def __init__(self, mu: pd.Series) -> None:
        self._mu = mu

    def capabilities(self) -> set[str]:
        return {capabilities.TO_PRICING}

    def expected_returns(self, view: TimeSeriesView) -> pd.DataFrame:
        row = self._mu.reindex([str(a) for a in view.assets]).to_numpy(np.float64)
        return pd.DataFrame(
            np.tile(row, (len(view.calendar), 1)), index=view.calendar, columns=view.assets
        )


class _TSMean:
    def fit(self, view: TimeSeriesView) -> _TSMeanModel:
        mu = view.returns_frame().mean()
        mu.index = [str(c) for c in mu.index]
        return _TSMeanModel(mu)


class _CSMeanModel:
    def __init__(self, mu: float) -> None:
        self._mu = mu

    def capabilities(self) -> set[str]:
        return {capabilities.TO_PRICING}

    def expected_returns(self, view: CrossSectionView) -> pd.DataFrame:
        vals = np.full((len(view.calendar), len(view.assets)), self._mu, dtype=np.float64)
        return pd.DataFrame(vals, index=view.calendar, columns=view.assets)


class _CSMean:
    def fit(self, view: CrossSectionView) -> _CSMeanModel:
        _keys, _x, y = view.aligned()
        return _CSMeanModel(float(y.mean()) if len(y) else 0.0)


def _cross_section_view(canonical: pd.DataFrame, seed: int = 3) -> CrossSectionView:
    """A CrossSectionView over the same dates and asset labels as ``canonical`` (a stray view)."""
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for d in canonical.index:
        for a in ASSETS:
            rows.append(
                {
                    "date": d,
                    "asset": a,
                    "c0": float(rng.normal()),
                    "c1": float(rng.normal()),
                    "ret": float(canonical.loc[d, a]),
                }
            )
    return CrossSectionView(pd.DataFrame(rows), chars=["c0", "c1"], horizon=1)


class _ExactPricingModel:
    """Prices each date's ``(t, t+h]`` return exactly (expected returns == realized returns)."""

    def __init__(self, target: pd.DataFrame) -> None:
        self._target = target  # indexed at t, holding the (t, t+1] realized return

    def capabilities(self) -> set[str]:
        return {capabilities.TO_PRICING}

    def expected_returns(self, view: TimeSeriesView) -> pd.DataFrame:
        return self._target.reindex(view.calendar)


class _ExactPricing:
    def __init__(self, target: pd.DataFrame) -> None:
        self._target = target

    def fit(self, view: TimeSeriesView) -> _ExactPricingModel:
        _ = view
        return _ExactPricingModel(self._target)


def test_compare_pairs_predicted_with_next_period_realized() -> None:
    """The PricingOutput convention: predicted.loc[t] scores against the (t, t+h] return.

    A model whose expected returns at ``t`` exactly equal the return realized over ``(t, t+1]``
    must price perfectly through ``compare`` (xs_r2 == 1, avg_abs_alpha == 0). Pairing predicted
    with the same-date (t-1, t] return instead would break this identity.
    """
    canonical = _canonical()
    ts_view = TimeSeriesView(canonical, horizon=1)
    target = canonical.shift(-1)  # row t = the (t, t+1] realized return
    entry = ComparisonEntry("exact", _ExactPricing(target), ts_view)

    # bare-frame path (documented horizon-1 convention)
    result = compare([entry], canonical)
    r2 = float(result.loc[result["metric"] == "xs_r2", "value"].iloc[0])
    aaa = float(result.loc[result["metric"] == "avg_abs_alpha", "value"].iloc[0])
    np.testing.assert_allclose(r2, 1.0)
    np.testing.assert_allclose(aaa, 0.0, atol=1e-12)

    # view path (horizon-aware target_asof pairing) gives the identical numbers
    result_v = compare([entry], ts_view)
    np.testing.assert_allclose(
        float(result_v.loc[result_v["metric"] == "xs_r2", "value"].iloc[0]), 1.0
    )
    np.testing.assert_allclose(
        float(result_v.loc[result_v["metric"] == "avg_abs_alpha", "value"].iloc[0]),
        0.0,
        atol=1e-12,
    )


def test_compare_drops_unrealized_horizon_tail() -> None:
    """The last calendar date (whose (t, t+1] return is unrealized in-sample) is not scored."""
    canonical = _canonical()
    ts_view = TimeSeriesView(canonical, horizon=1)
    result = compare([ComparisonEntry("ts", _TSMean(), ts_view)], canonical)
    # every emitted date is strictly before the final calendar date (the unrealized tail)
    assert (result["date"] < canonical.index[-1]).all()


def test_compare_two_methods_common_test_assets() -> None:
    canonical = _canonical()
    ts_view = TimeSeriesView(canonical, horizon=1)
    entries = [
        ComparisonEntry(name="ts_mean", estimator=_TSMean(), train_view=ts_view),
        ComparisonEntry(
            name="cs_mean",
            estimator=_CSMean(),
            train_view=_cross_section_view(canonical),
            test_view=_cross_section_view(canonical),
        ),
    ]
    result = compare(entries, canonical, data_vintage="2026-07")
    validate_result(result)
    assert set(result["method"]) == {"ts_mean", "cs_mean"}
    assert set(result["metric"]) == {"xs_r2", "avg_abs_alpha"}
    assert (result["protocol"] == "in_sample").all()
    assert (result["data_vintage"] == "2026-07").all()


def test_compare_defaults_test_view_to_train_view() -> None:
    canonical = _canonical()
    ts_view = TimeSeriesView(canonical, horizon=1)
    # no test_view -> priced through train_view (the method trains on the test assets themselves)
    result = compare([ComparisonEntry("ts", _TSMean(), ts_view)], canonical)
    assert set(result["metric"]) == {"xs_r2", "avg_abs_alpha"}


def test_compare_accepts_view_as_test_assets() -> None:
    canonical = _canonical()
    ts_view = TimeSeriesView(canonical, horizon=1)
    # test_assets may be a view exposing returns_frame()
    result = compare([ComparisonEntry("ts", _TSMean(), ts_view)], ts_view)
    validate_result(result)


def test_compare_custom_evaluator_list() -> None:
    from numeraire.core.evaluators import CrossSectionalR2Evaluator

    canonical = _canonical()
    ts_view = TimeSeriesView(canonical, horizon=1)
    result = compare(
        [ComparisonEntry("ts", _TSMean(), ts_view)],
        canonical,
        evaluators=[CrossSectionalR2Evaluator()],
    )
    assert set(result["metric"]) == {"xs_r2"}


def test_compare_rejects_stray_asset() -> None:
    class _StrayModel:
        def capabilities(self) -> set[str]:
            return {capabilities.TO_PRICING}

        def expected_returns(self, view: TimeSeriesView) -> pd.DataFrame:
            cols = [*ASSETS, "ZZZ"]  # ZZZ is not in the canonical panel
            return pd.DataFrame(0.01, index=view.calendar, columns=cols)

    class _Stray:
        def fit(self, view: TimeSeriesView) -> _StrayModel:
            _ = view
            return _StrayModel()

    canonical = _canonical()
    ts_view = TimeSeriesView(canonical, horizon=1)
    with pytest.raises(ValueError, match="absent from the common test_assets panel"):
        compare([ComparisonEntry("stray", _Stray(), ts_view)], canonical)


def test_compare_rejects_stray_date() -> None:
    class _StrayDateModel:
        def capabilities(self) -> set[str]:
            return {capabilities.TO_PRICING}

        def expected_returns(self, view: TimeSeriesView) -> pd.DataFrame:
            idx = view.calendar.append(pd.DatetimeIndex([pd.Timestamp("2099-12-31")]))
            return pd.DataFrame(0.01, index=idx, columns=ASSETS)

    class _StrayDate:
        def fit(self, view: TimeSeriesView) -> _StrayDateModel:
            _ = view
            return _StrayDateModel()

    canonical = _canonical()
    ts_view = TimeSeriesView(canonical, horizon=1)
    with pytest.raises(ValueError, match="dates absent from the common"):
        compare([ComparisonEntry("straydate", _StrayDate(), ts_view)], canonical)


def test_compare_rejects_non_pricing_model() -> None:
    class _WModel:
        def capabilities(self) -> set[str]:
            return {capabilities.TO_WEIGHTS}

    class _W:
        def fit(self, view: TimeSeriesView) -> _WModel:
            _ = view
            return _WModel()

    canonical = _canonical()
    ts_view = TimeSeriesView(canonical, horizon=1)
    with pytest.raises(TypeError, match="does not support 'to_pricing'"):
        compare([ComparisonEntry("w", _W(), ts_view)], canonical)


def test_compare_rejects_empty_entries() -> None:
    with pytest.raises(ValueError, match="at least one entry"):
        compare([], _canonical())


def test_compare_rejects_bad_test_assets_type() -> None:
    canonical = _canonical()
    ts_view = TimeSeriesView(canonical, horizon=1)
    with pytest.raises(TypeError, match="DataFrame or a view"):
        compare([ComparisonEntry("ts", _TSMean(), ts_view)], object())
