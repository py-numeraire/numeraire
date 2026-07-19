"""The horizon / target-contract (WP-C): horizon>=1, output horizon+frequency+overlap metadata,
one source of truth for the horizon, the annualization guard, and the same-horizon benchmark.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_monthly_view, toy_panel_wide
from numeraire.baselines import EqualWeight, HistoricalMean
from numeraire.core import capabilities
from numeraire.core.data import CrossSectionView, TimeSeriesView
from numeraire.core.engine import (
    ForecastOutput,
    PanelWeightsOutput,
    PricingOutput,
    WeightsOutput,
    backtest_forecast,
    backtest_panel,
    backtest_pricing,
    backtest_pricing_in_sample,
    backtest_weights,
)
from numeraire.core.evaluators import MeanReturnEvaluator, SharpeEvaluator
from numeraire.core.splitter import WalkForwardSplitter

# -- small toy estimators (reused across the driver tests) --------------------------


class _CSPricingModel:
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


class _PanelEqualWeightModel:
    def capabilities(self) -> set[str]:
        return {capabilities.TO_WEIGHTS}

    def to_weights(self, view: CrossSectionView) -> pd.Series:
        dates: list[pd.Timestamp] = []
        assets: list[object] = []
        vals: list[float] = []
        for t in view.calendar:
            ids, _x = view.features_asof(t)
            if len(ids) == 0:
                continue
            w = 1.0 / len(ids)
            for a in ids:
                dates.append(t)
                assets.append(a)
                vals.append(w)
        idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex(dates), assets], names=["date", "asset"])
        return pd.Series(vals, index=idx, name="weight")


class _PanelEqualWeight:
    def fit(self, view: CrossSectionView) -> _PanelEqualWeightModel:
        return _PanelEqualWeightModel()


def _panel_view(horizon: int = 1) -> CrossSectionView:
    return CrossSectionView(
        toy_panel_wide(n_months=48, n_assets=8), chars=["size", "bm"], horizon=horizon
    )


# ==========================================================================================
# 1. horizon <= 0 rejected everywhere
# ==========================================================================================


@pytest.mark.parametrize("bad", [0, -1, -5])
def test_time_series_view_rejects_non_positive_horizon(bad: int) -> None:
    idx = pd.date_range("2000-01-31", periods=6, freq="ME")
    df = pd.DataFrame({"mkt": np.zeros(6)}, index=idx)
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        TimeSeriesView(df, df, horizon=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_cross_section_view_rejects_non_positive_horizon(bad: int) -> None:
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        _panel_view(horizon=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_time_series_per_call_overrides_reject_non_positive_horizon(bad: int) -> None:
    view = make_monthly_view(n=24)
    t = view.calendar[5]
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        view.target_asof(t, horizon=bad)
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        view.aligned(horizon=bad)


@pytest.mark.parametrize("bad", [0, -1])
def test_cross_section_per_call_overrides_reject_non_positive_horizon(bad: int) -> None:
    view = _panel_view()
    t = view.calendar[5]
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        view.target_asof(t, horizon=bad)
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        view.aligned(horizon=bad)


def test_output_dataclasses_reject_non_positive_horizon() -> None:
    idx = pd.date_range("2000-01-31", periods=3, freq="ME")
    w = pd.DataFrame({"r0": [1.0, 1.0, 1.0]}, index=idx)
    with pytest.raises(ValueError, match="horizon must be >= 1"):
        WeightsOutput(
            weights=w,
            realized=w * 0.0,
            method="m",
            config_hash="c",
            data_vintage="v",
            run_id="r",
            horizon=0,
        )


@pytest.mark.parametrize("bad", [1.5, float("nan"), float("inf"), 2.0, True])
def test_horizon_must_be_a_plain_integer(bad: object) -> None:
    # A malformed horizon (fractional, non-finite, boolean) is a type error everywhere — it must
    # not sit inside a directly constructed output indefinitely.
    idx = pd.date_range("2000-01-31", periods=6, freq="ME")
    df = pd.DataFrame({"mkt": np.zeros(6)}, index=idx)
    with pytest.raises(TypeError, match="finite integer"):
        TimeSeriesView(df, df, horizon=bad)  # type: ignore[arg-type]
    w = pd.DataFrame({"r0": np.ones(3)}, index=idx[:3])
    with pytest.raises(TypeError, match="finite integer"):
        WeightsOutput(
            weights=w,
            realized=w * 0.0,
            method="m",
            config_hash="c",
            data_vintage="v",
            run_id="r",
            horizon=bad,  # type: ignore[arg-type]
        )


# ==========================================================================================
# 2. outputs carry the effective target contract (all four drivers)
# ==========================================================================================


def test_weights_driver_stamps_horizon_and_frequency() -> None:
    view = make_monthly_view(n=48, n_assets=3, seed=1)
    n = len(view.calendar)
    sp = WalkForwardSplitter(min_train=24, test_size=n - 24, expanding=True)
    out = backtest_weights(EqualWeight(), view, sp, method="ew")
    assert out.horizon == 1
    assert out.meta["frequency"] == "ME"
    assert "overlap" not in out.meta  # horizon 1 -> no overlap key


def test_weights_driver_horizon2_stamps_overlap() -> None:
    view = make_monthly_view(n=48, n_assets=3, seed=1, horizon=2)
    n = len(view.calendar)
    sp = WalkForwardSplitter(min_train=24, test_size=n - 24, expanding=True)
    out = backtest_weights(EqualWeight(), view, sp, method="ew")
    assert out.horizon == 2
    assert out.meta["frequency"] == "ME"
    assert out.meta["overlap"] == 1  # horizon - 1


def test_panel_driver_stamps_contract() -> None:
    view = _panel_view(horizon=1)
    n = len(view.calendar)
    sp = WalkForwardSplitter(min_train=24, test_size=n - 24, expanding=True)
    out = backtest_panel(_PanelEqualWeight(), view, sp, method="ew", missing_returns="zero")
    assert out.horizon == 1
    assert out.meta["frequency"] == "ME"


def test_forecast_driver_stamps_contract() -> None:
    view = make_monthly_view(n=60, n_assets=1, seed=2)
    out = backtest_forecast(HistoricalMean(), view, min_train=24, method="hm")
    assert out.horizon == 1
    assert out.meta["frequency"] == "ME"


class _QuarterlyEqualWeightModel:
    """Emits equal weights only at quarter-end months of the test fold (a sparser cadence)."""

    def capabilities(self) -> set[str]:
        return {capabilities.TO_WEIGHTS}

    def to_weights(self, view: TimeSeriesView) -> pd.DataFrame:
        cal = view.calendar
        qdates = cal[cal.month.isin([3, 6, 9, 12])]
        n = len(view.assets)
        return pd.DataFrame(np.full((len(qdates), n), 1.0 / n), index=qdates, columns=view.assets)


class _QuarterlyEqualWeight:
    def fit(self, view: TimeSeriesView) -> _QuarterlyEqualWeightModel:
        return _QuarterlyEqualWeightModel()


def test_sparse_quarterly_output_from_monthly_view_gets_quarterly_scaling() -> None:
    # The stamp follows the FINALIZED OUTPUT dates, not the input view calendar: a model emitting
    # quarterly decisions from a monthly view must be annualized at 4 periods/year, not 12.
    view = make_monthly_view(n=60, n_assets=3, seed=6)
    n = len(view.calendar)
    sp = WalkForwardSplitter(min_train=12, test_size=n - 12, expanding=True)
    out = backtest_weights(_QuarterlyEqualWeight(), view, sp, method="qew")
    assert str(out.meta["frequency"]).startswith("QE")  # quarterly, from the output dates
    via_stamp = SharpeEvaluator().evaluate(out).iloc[0]["value"]
    explicit = SharpeEvaluator(periods_per_year=4).evaluate(out).iloc[0]["value"]
    np.testing.assert_array_equal(via_stamp, explicit)
    # and via the inference FALLBACK: the same panels rebuilt directly with no meta derive 4 too
    direct = WeightsOutput(
        weights=out.weights,
        realized=out.realized,
        method="qew",
        config_hash="c",
        data_vintage="v",
        run_id="r",
    )
    via_fallback = SharpeEvaluator().evaluate(direct).iloc[0]["value"]
    np.testing.assert_array_equal(via_fallback, explicit)


def test_pricing_drivers_stamp_contract() -> None:
    view = _panel_view(horizon=1)
    n = len(view.calendar)
    sp = WalkForwardSplitter(min_train=24, test_size=n - 24, expanding=True)
    wf = backtest_pricing(_CSPricing(), view, sp, method="csp")
    assert wf.horizon == 1
    assert wf.meta["frequency"] == "ME"
    ins = backtest_pricing_in_sample(_CSPricing(), view, method="csp")
    assert ins.horizon == 1
    assert ins.meta["frequency"] == "ME"


# ==========================================================================================
# 3. one source of truth for the horizon (driver override must not disagree with the view)
# ==========================================================================================


def test_forecast_horizon_override_may_agree() -> None:
    view = make_monthly_view(n=60, n_assets=1, seed=3, horizon=2)
    a = backtest_forecast(HistoricalMean(), view, min_train=24, method="hm")
    b = backtest_forecast(HistoricalMean(), view, min_train=24, horizon=2, method="hm")  # asserts h
    pd.testing.assert_frame_equal(a.forecasts, b.forecasts)
    assert a.horizon == b.horizon == 2


def test_forecast_horizon_override_disagreement_raises() -> None:
    view = make_monthly_view(n=60, n_assets=1, seed=3, horizon=2)
    with pytest.raises(ValueError, match=r"disagrees with view\.horizon"):
        backtest_forecast(HistoricalMean(), view, min_train=24, horizon=1, method="hm")


# ==========================================================================================
# 4. annualization guard
# ==========================================================================================


def _monthly_weights_output(meta: dict[str, object]) -> WeightsOutput:
    idx = pd.date_range("2000-01-31", periods=12, freq="ME")
    rng = np.random.default_rng(0)
    return WeightsOutput(
        weights=pd.DataFrame({"s": np.ones(12)}, index=idx),
        realized=pd.DataFrame({"s": rng.normal(0.01, 0.03, 12)}, index=idx),
        method="m",
        config_hash="c",
        data_vintage="v",
        run_id="r",
        meta=meta,
    )


def test_derived_monthly_matches_explicit_twelve() -> None:
    out = _monthly_weights_output({"frequency": "ME"})
    derived = SharpeEvaluator().evaluate(out).iloc[0]["value"]
    explicit = SharpeEvaluator(periods_per_year=12).evaluate(out).iloc[0]["value"]
    np.testing.assert_array_equal(derived, explicit)  # bit-identical: derivation == old default


def test_business_daily_derives_252_calendar_daily_365() -> None:
    for freq, ppy in (("B", 252), ("D", 365)):
        idx = pd.date_range("2000-01-03", periods=20, freq=freq)
        out = WeightsOutput(
            weights=pd.DataFrame({"s": np.ones(20)}, index=idx),
            realized=pd.DataFrame({"s": np.full(20, 0.001)}, index=idx),
            method="m",
            config_hash="c",
            data_vintage="v",
            run_id="r",
            meta={"frequency": freq},
        )
        derived = MeanReturnEvaluator().evaluate(out).iloc[0]["value"]
        explicit = MeanReturnEvaluator(periods_per_year=ppy).evaluate(out).iloc[0]["value"]
        np.testing.assert_array_equal(derived, explicit)


def _irregular_weights_output(meta: dict[str, object]) -> WeightsOutput:
    """A directly-built output on genuinely irregular dates (mixed gaps; nothing inferable)."""
    idx = pd.DatetimeIndex(
        ["2000-01-31", "2000-02-11", "2000-05-02", "2000-05-19", "2000-09-01", "2001-01-03"]
    )
    rng = np.random.default_rng(0)
    return WeightsOutput(
        weights=pd.DataFrame({"s": np.ones(len(idx))}, index=idx),
        realized=pd.DataFrame({"s": rng.normal(0.01, 0.03, len(idx))}, index=idx),
        method="m",
        config_hash="c",
        data_vintage="v",
        run_id="r",
        meta=meta,
    )


def test_irregular_dates_refuse_without_explicit() -> None:
    # Stamped-None (a driver's irregular verdict) and no meta at all both end refused: the
    # fallback inference runs on the same irregular dates and cannot produce a code either.
    for meta in ({"frequency": None}, {}):
        out = _irregular_weights_output(dict(meta))
        with pytest.raises(ValueError, match="no inferable prediction-date frequency"):
            SharpeEvaluator().evaluate(out)
        # explicit argument always wins
        assert np.isfinite(SharpeEvaluator(periods_per_year=12).evaluate(out).iloc[0]["value"])


def test_direct_construction_regular_index_needs_no_meta() -> None:
    # The inference fallback: a directly built output with a regular monthly index and NO stamped
    # metadata derives 12 from its own prediction dates — exactly as if `meta={"frequency": "ME"}`.
    out = _monthly_weights_output({})
    derived = SharpeEvaluator().evaluate(out).iloc[0]["value"]
    explicit = SharpeEvaluator(periods_per_year=12).evaluate(out).iloc[0]["value"]
    np.testing.assert_array_equal(derived, explicit)


def test_overlap_refuses_without_explicit() -> None:
    out = _monthly_weights_output({"frequency": "ME", "overlap": 1})
    with pytest.raises(ValueError, match="overlapping"):
        SharpeEvaluator().evaluate(out)
    # explicit periods_per_year overrides the refusal
    assert np.isfinite(SharpeEvaluator(periods_per_year=12).evaluate(out).iloc[0]["value"])


def test_direct_horizon2_refuses_even_with_frequency_meta() -> None:
    # The refusal keys on the canonical `horizon` field: a direct construction that stamps a
    # frequency but no `overlap` entry must still be refused when its horizon is > 1.
    idx = pd.date_range("2000-01-31", periods=12, freq="ME")
    rng = np.random.default_rng(1)
    out = WeightsOutput(
        weights=pd.DataFrame({"s": np.ones(12)}, index=idx),
        realized=pd.DataFrame({"s": rng.normal(0.01, 0.03, 12)}, index=idx),
        method="m",
        config_hash="c",
        data_vintage="v",
        run_id="r",
        horizon=2,
        meta={"frequency": "ME"},
    )
    with pytest.raises(ValueError, match="overlapping"):
        SharpeEvaluator().evaluate(out)
    assert np.isfinite(SharpeEvaluator(periods_per_year=12).evaluate(out).iloc[0]["value"])


def test_overlap_end_to_end_from_driver_refuses() -> None:
    view = make_monthly_view(n=48, n_assets=3, seed=4, horizon=2)
    n = len(view.calendar)
    sp = WalkForwardSplitter(min_train=24, test_size=n - 24, expanding=True)
    out = backtest_weights(EqualWeight(), view, sp, method="ew")
    with pytest.raises(ValueError, match="overlapping"):
        SharpeEvaluator().evaluate(out)


# ==========================================================================================
# 5. benchmark on the same target (historical mean compounded to h)
# ==========================================================================================


def test_h1_benchmark_is_single_period_mean_exactly() -> None:
    # Bit-identical, not merely close: for h=1 the engine uses the mean itself, never a
    # float-non-associative (1+mu)^1 - 1 detour.
    view = make_monthly_view(n=60, n_assets=1, seed=5)
    out = backtest_forecast(HistoricalMean(), view, min_train=24, method="hm")
    origin = out.benchmark.index[0]
    mu = float(view.window(origin).returns_frame().to_numpy(dtype=np.float64).mean(axis=0)[0])
    np.testing.assert_array_equal(float(out.benchmark.loc[origin, "r0"]), mu)


def test_h2_benchmark_is_compounded_two_period_mean() -> None:
    view = make_monthly_view(n=60, n_assets=1, seed=5, horizon=2)
    out = backtest_forecast(HistoricalMean(), view, min_train=24, method="hm")
    origin = out.benchmark.index[0]
    mu = float(view.window(origin).returns_frame().to_numpy().mean())
    expected = (1.0 + mu) ** 2 - 1.0
    got = float(out.benchmark.loc[origin, "r0"])
    np.testing.assert_allclose(got, expected)
    # the oracle: the compounded benchmark is NOT the single-period mean it used to be
    assert not np.isclose(got, mu)


def test_historical_mean_scores_exactly_zero_against_benchmark_at_h2() -> None:
    # The baseline IS the benchmark, at every horizon: with the h-compounded HistoricalMean the
    # forecast and the engine benchmark are bit-identical, so OOS R^2 is exactly 0.0 (before the
    # fix the baseline stayed single-period and scored spuriously negative at h=2).
    from numeraire.core.evaluators import OutOfSampleR2Evaluator

    view = make_monthly_view(n=72, n_assets=1, seed=7, horizon=2)
    out = backtest_forecast(HistoricalMean(), view, min_train=24, method="hm")
    pd.testing.assert_frame_equal(out.forecasts, out.benchmark)
    r2 = OutOfSampleR2Evaluator().evaluate(out).iloc[0]["value"]
    assert r2 == 0.0


def _forecast_output_direct() -> ForecastOutput:
    """A directly-built ForecastOutput to confirm the horizon field defaults sanely."""
    idx = pd.date_range("2000-01-31", periods=3, freq="ME")
    f = pd.DataFrame({"mkt": [0.1, 0.2, 0.0]}, index=idx)
    return ForecastOutput(
        forecasts=f,
        realized=f,
        benchmark=f * 0.0,
        method="m",
        config_hash="c",
        data_vintage="v",
        run_id="r",
    )


def test_direct_outputs_default_horizon_one() -> None:
    assert _forecast_output_direct().horizon == 1
    idx = pd.date_range("2000-01-31", periods=2, freq="ME")
    keys = pd.MultiIndex.from_tuples([(idx[0], "A"), (idx[1], "A")], names=["date", "asset"])
    pw = PanelWeightsOutput(
        weights=pd.Series([1.0, 1.0], index=keys, name="weight"),
        realized=pd.Series([0.0, 0.0], index=keys, name="realized"),
        method="m",
        config_hash="c",
        data_vintage="v",
        run_id="r",
    )
    assert pw.horizon == 1
    pr = PricingOutput(
        predicted=pd.DataFrame({"a": [0.1]}, index=idx[:1]),
        realized=pd.DataFrame({"a": [0.1]}, index=idx[:1]),
        method="m",
        config_hash="c",
        data_vintage="v",
        run_id="r",
        protocol="in_sample",
    )
    assert pr.horizon == 1
