"""Risk-adjusted performance, information-coefficient, and exposure evaluators.

Correctness is pinned through closed-form identities on hand-set or noiseless synthetic data
(Treynor = excess/beta on a known beta; M-squared = annualized Sharpe x annualized benchmark vol;
Sortino penalizes only downside; IC = 1 for a monotone forecast; exposure leverage/HHI/turnover on
hand-set weights), determinism, degenerate guards, and result-schema conformance — never snapshots.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import make_monthly_view
from numeraire import (
    ExposureEvaluator,
    ForecastOutput,
    ICEvaluator,
    InformationRatioEvaluator,
    M2Evaluator,
    PanelWeightsOutput,
    SharpeEvaluator,
    SortinoEvaluator,
    TreynorEvaluator,
    WalkForwardSplitter,
    WeightsOutput,
    available_evaluators,
    backtest_weights,
    capabilities,
    get_evaluator,
    validate_result,
)
from numeraire.baselines import EqualWeight


def _weights_output(returns: np.ndarray, index: pd.DatetimeIndex) -> WeightsOutput:
    """A single-asset ``WeightsOutput`` whose strategy returns are exactly ``returns``."""
    return WeightsOutput(
        weights=pd.DataFrame({"s": np.ones(len(returns))}, index=index),
        realized=pd.DataFrame({"s": returns}, index=index),
        method="toy",
        config_hash="cfg",
        data_vintage="synthetic",
        run_id="toy-cfg",
        meta={"frequency": "ME"},  # monthly calendar → annualizing evaluators derive 12/sqrt(12)
    )


# --------------------------------------------------------------------------- registration


def test_default_evaluators_registered() -> None:
    for name in ("sortino", "ic", "exposure"):
        assert name in available_evaluators()
    assert get_evaluator("sortino").requires == {capabilities.TO_WEIGHTS}
    assert get_evaluator("ic").requires == {capabilities.TO_FORECAST}
    assert get_evaluator("exposure").requires == {capabilities.TO_WEIGHTS}


# --------------------------------------------------------------------------- Treynor


def test_treynor_equals_excess_over_beta_on_known_beta() -> None:
    rng = np.random.default_rng(0)
    idx = pd.date_range("2000-01-31", periods=240, freq="ME")
    mkt = pd.DataFrame({"mkt": rng.normal(0.006, 0.04, 240)}, index=idx)
    beta_true = 1.5
    # noiseless: the regression recovers beta exactly, so Treynor = periods * mean(r_p) / beta
    strat = beta_true * mkt["mkt"] + 0.003
    out = _weights_output(strat.to_numpy(), idx)
    df = TreynorEvaluator(mkt, periods_per_year=12).evaluate(out)
    validate_result(df)
    assert df.iloc[0]["metric"] == "treynor"
    expected = float(strat.mean()) * 12.0 / beta_true
    np.testing.assert_allclose(df.iloc[0]["value"], expected, rtol=1e-6)


def test_treynor_market_column_selection() -> None:
    rng = np.random.default_rng(1)
    idx = pd.date_range("2000-01-31", periods=180, freq="ME")
    factors = pd.DataFrame(
        {"smb": rng.normal(0.0, 0.02, 180), "mkt": rng.normal(0.006, 0.04, 180)}, index=idx
    )
    strat = 1.2 * factors["mkt"] + 0.002
    out = _weights_output(strat.to_numpy(), idx)
    df = TreynorEvaluator(factors, market="mkt").evaluate(out)
    expected = float(strat.mean()) * 12.0 / 1.2
    np.testing.assert_allclose(df.iloc[0]["value"], expected, rtol=1e-5)


def test_treynor_rejects_wrong_output() -> None:
    idx = pd.date_range("2000-01-31", periods=10, freq="ME")
    with pytest.raises(TypeError):
        TreynorEvaluator(pd.DataFrame({"mkt": np.zeros(10)}, index=idx)).evaluate(object())


# --------------------------------------------------------------------------- Information ratio


def test_information_ratio_matches_manual_formula() -> None:
    rng = np.random.default_rng(2)
    idx = pd.date_range("2000-01-31", periods=200, freq="ME")
    strat = rng.normal(0.01, 0.04, 200)
    bench = pd.Series(rng.normal(0.008, 0.03, 200), index=idx)
    out = _weights_output(strat, idx)
    df = InformationRatioEvaluator(bench, periods_per_year=12).evaluate(out)
    validate_result(df)
    assert df.iloc[0]["metric"] == "information_ratio"
    active = strat - bench.to_numpy()
    expected = active.mean() / active.std(ddof=1) * np.sqrt(12)
    np.testing.assert_allclose(df.iloc[0]["value"], expected)


def test_information_ratio_zero_when_tracking_benchmark() -> None:
    idx = pd.date_range("2000-01-31", periods=60, freq="ME")
    rng = np.random.default_rng(3)
    strat = rng.normal(0.01, 0.04, 60)
    out = _weights_output(strat, idx)
    # active return identically zero -> tracking error 0 -> undefined IR
    df = InformationRatioEvaluator(pd.Series(strat, index=idx)).evaluate(out)
    assert np.isnan(df.iloc[0]["value"])


# --------------------------------------------------------------------------- M-squared


def test_m2_equals_annualized_sharpe_times_annualized_benchmark_vol() -> None:
    rng = np.random.default_rng(4)
    idx = pd.date_range("2000-01-31", periods=200, freq="ME")
    strat = rng.normal(0.01, 0.05, 200)
    bench = pd.Series(rng.normal(0.006, 0.03, 200), index=idx)
    out = _weights_output(strat, idx)
    m2 = M2Evaluator(bench, periods_per_year=12).evaluate(out).iloc[0]["value"]
    sharpe = SharpeEvaluator(periods_per_year=12).evaluate(out).iloc[0]["value"]
    ann_bench_vol = float(bench.std(ddof=1)) * np.sqrt(12)
    np.testing.assert_allclose(m2, sharpe * ann_bench_vol, rtol=1e-9)


def test_m2_schema_and_metric() -> None:
    idx = pd.date_range("2000-01-31", periods=50, freq="ME")
    rng = np.random.default_rng(5)
    out = _weights_output(rng.normal(0.01, 0.04, 50), idx)
    df = M2Evaluator(pd.Series(rng.normal(0.0, 0.03, 50), index=idx)).evaluate(out)
    validate_result(df)
    assert df.iloc[0]["metric"] == "m2"


# --------------------------------------------------------------------------- Sortino


def test_sortino_penalizes_only_downside() -> None:
    idx = pd.date_range("2000-01-31", periods=6, freq="ME")
    # returns above the MAR contribute 0 to downside deviation; only the negatives count
    rets = np.array([0.05, -0.02, 0.03, -0.04, 0.06, 0.01])
    out = _weights_output(rets, idx)
    df = SortinoEvaluator(mar=0.0, periods_per_year=12).evaluate(out)
    validate_result(df)
    assert df.iloc[0]["metric"] == "sortino"
    downside = np.minimum(rets - 0.0, 0.0)
    dd = np.sqrt(np.mean(downside**2))
    expected = rets.mean() / dd * np.sqrt(12)
    np.testing.assert_allclose(df.iloc[0]["value"], expected)


def test_sortino_no_downside_is_nan() -> None:
    idx = pd.date_range("2000-01-31", periods=5, freq="ME")
    out = _weights_output(np.array([0.01, 0.02, 0.03, 0.04, 0.05]), idx)
    # every return exceeds the MAR -> downside deviation 0 -> undefined
    assert np.isnan(SortinoEvaluator(mar=0.0).evaluate(out).iloc[0]["value"])


def test_sortino_mar_shifts_threshold() -> None:
    idx = pd.date_range("2000-01-31", periods=5, freq="ME")
    out = _weights_output(np.array([0.01, 0.02, 0.03, 0.04, 0.05]), idx)
    # a MAR above every return makes all periods downside -> finite ratio (and negative numerator)
    val = SortinoEvaluator(mar=0.10).evaluate(out).iloc[0]["value"]
    assert np.isfinite(val)
    assert val < 0.0


# ----------------------------------------------------------------------- information coefficient


def _forecast_output(f: np.ndarray, r: np.ndarray, idx: pd.DatetimeIndex) -> ForecastOutput:
    cols = [f"a{i}" for i in range(f.shape[1])]
    frame = lambda m: pd.DataFrame(m, index=idx, columns=cols)  # noqa: E731
    return ForecastOutput(
        forecasts=frame(f),
        realized=frame(r),
        benchmark=frame(np.zeros_like(f)),
        method="toy",
        config_hash="cfg",
        data_vintage="synthetic",
        run_id="toy-cfg",
    )


def test_ic_is_one_for_monotone_forecast() -> None:
    idx = pd.date_range("2000-12-31", periods=4, freq="YE")
    f = np.array([[1.0, 2, 3, 4], [4, 3, 2, 1], [2, 4, 1, 3], [1, 3, 2, 4]])
    r = 2.0 * f  # realized is a positive monotone transform of the forecast each period
    df = ICEvaluator().evaluate(_forecast_output(f, r, idx))
    validate_result(df)
    ic = float(df.loc[df["metric"] == "ic", "value"].iloc[0])
    np.testing.assert_allclose(ic, 1.0)


def test_ic_is_minus_one_for_antimonotone_forecast() -> None:
    idx = pd.date_range("2000-12-31", periods=3, freq="YE")
    f = np.array([[1.0, 2, 3, 4], [4, 3, 2, 1], [2, 4, 1, 3]])
    r = -1.0 * f  # rank order is exactly reversed
    ic = float(ICEvaluator().evaluate(_forecast_output(f, r, idx)).iloc[0]["value"])
    np.testing.assert_allclose(ic, -1.0)


def test_ic_ir_and_tstat_from_period_ics() -> None:
    idx = pd.date_range("2000-12-31", periods=3, freq="YE")
    # three periods: perfect (+1), reversed (-1), and a middling positive ordering
    f = np.array([[1.0, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    r = np.array([[1.0, 2, 3, 4], [4, 3, 2, 1], [1, 3, 2, 4]])
    df = ICEvaluator().evaluate(_forecast_output(f, r, idx))
    ics = np.array([1.0, -1.0, 0.8])  # spearman per row
    ic = float(df.loc[df["metric"] == "ic", "value"].iloc[0])
    ic_ir = float(df.loc[df["metric"] == "ic_ir", "value"].iloc[0])
    ic_t = float(df.loc[df["metric"] == "ic_t", "value"].iloc[0])
    np.testing.assert_allclose(ic, ics.mean())
    np.testing.assert_allclose(ic_ir, ics.mean() / ics.std(ddof=1))
    np.testing.assert_allclose(ic_t, ic_ir * np.sqrt(3))


def test_ic_single_asset_is_nan() -> None:
    idx = pd.date_range("2000-12-31", periods=5, freq="YE")
    f = np.arange(5.0).reshape(5, 1)
    df = ICEvaluator().evaluate(_forecast_output(f, f * 2.0, idx))
    assert np.isnan(df.loc[df["metric"] == "ic", "value"].iloc[0])


def test_ic_rejects_wrong_output() -> None:
    with pytest.raises(TypeError):
        ICEvaluator().evaluate(object())


# --------------------------------------------------------------------------- Exposure


def _wide_weights(rows: list[list[float]], cols: list[str]) -> WeightsOutput:
    idx = pd.date_range("2001-01-31", periods=len(rows), freq="ME")
    w = pd.DataFrame(rows, index=idx, columns=cols)
    return WeightsOutput(
        weights=w,
        realized=pd.DataFrame(np.zeros_like(w.to_numpy()), index=idx, columns=cols),
        method="toy",
        config_hash="cfg",
        data_vintage="synthetic",
        run_id="toy-cfg",
    )


def test_exposure_scalars_on_hand_set_weights() -> None:
    out = _wide_weights([[0.5, 0.5, -0.3], [0.5, -0.5, 0.0]], ["x", "y", "z"])
    df = ExposureEvaluator().evaluate(out)
    validate_result(df)
    assert set(df["metric"]) == {"gross_leverage", "net_exposure", "turnover", "hhi"}
    first = df[df["date"] == df["date"].min()].set_index("metric")["value"]
    np.testing.assert_allclose(first["gross_leverage"], 1.3)  # |0.5|+|0.5|+|-0.3|
    np.testing.assert_allclose(first["net_exposure"], 0.7)  # 0.5+0.5-0.3
    np.testing.assert_allclose(first["hhi"], 0.59)  # 0.25+0.25+0.09
    np.testing.assert_allclose(first["turnover"], 1.3)  # opening trade from all cash == gross
    last = df[df["date"] == df["date"].max()].set_index("metric")["value"]
    np.testing.assert_allclose(last["net_exposure"], 0.0)  # dollar-neutral
    np.testing.assert_allclose(last["hhi"], 0.5)
    np.testing.assert_allclose(last["turnover"], 1.3)  # |0|+|-1.0|+|0.3|


def test_exposure_equal_weight_hhi_is_one_over_n() -> None:
    out = _wide_weights([[0.25, 0.25, 0.25, 0.25]], ["a", "b", "c", "d"])
    val = ExposureEvaluator().evaluate(out).set_index("metric")["value"]
    np.testing.assert_allclose(val["hhi"], 0.25)  # 1/N for an equal-weight book of N=4
    np.testing.assert_allclose(val["gross_leverage"], 1.0)


def test_exposure_panel_turnover_aligns_ragged_universe() -> None:
    # entering/exiting universe: turnover must align on the union of names across dates
    idx = pd.date_range("2001-01-31", periods=2, freq="ME")
    keys = pd.MultiIndex.from_tuples(
        [(idx[0], "A"), (idx[0], "B"), (idx[1], "B"), (idx[1], "C")], names=["date", "asset"]
    )
    weights = pd.Series([0.6, 0.4, 0.5, 0.5], index=keys, name="weight")
    out = PanelWeightsOutput(
        weights=weights,
        realized=pd.Series(np.zeros(4), index=keys, name="realized"),
        method="toy",
        config_hash="cfg",
        data_vintage="synthetic",
        run_id="toy-cfg",
    )
    df = ExposureEvaluator().evaluate(out)
    validate_result(df)
    last = df[df["date"] == idx[1]].set_index("metric")["value"]
    # A exits (0.6 -> 0), B 0.4 -> 0.5, C enters (0 -> 0.5): L1 = 0.6 + 0.1 + 0.5
    np.testing.assert_allclose(last["turnover"], 1.2)
    np.testing.assert_allclose(last["gross_leverage"], 1.0)


def test_exposure_rejects_wrong_output() -> None:
    with pytest.raises(TypeError):
        ExposureEvaluator().evaluate(object())


# --------------------------------------------------------------------------- determinism + engine


def test_evaluators_are_deterministic() -> None:
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(7)
    strat = rng.normal(0.01, 0.04, 120)
    bench = pd.Series(rng.normal(0.006, 0.03, 120), index=idx)
    out = _weights_output(strat, idx)
    for ev in (M2Evaluator(bench), InformationRatioEvaluator(bench), SortinoEvaluator()):
        a = ev.evaluate(out)["value"].to_numpy()
        b = ev.evaluate(out)["value"].to_numpy()
        np.testing.assert_array_equal(a, b)


def test_weights_evaluators_run_through_engine() -> None:
    view = make_monthly_view(n=48, n_assets=4, seed=9)
    n = len(view.calendar)
    sp = WalkForwardSplitter(min_train=24, test_size=n - 24, expanding=True)
    out = backtest_weights(EqualWeight(), view, sp, method="equal_weight")
    bench = out.strategy_returns() * 0.5  # a trivial benchmark series on the same calendar
    factors = view.returns_frame().rename(columns={view.assets[0]: "mkt"})[["mkt"]]
    for ev in (
        TreynorEvaluator(factors, market="mkt"),
        InformationRatioEvaluator(bench),
        M2Evaluator(bench),
        SortinoEvaluator(),
        ExposureEvaluator(),
    ):
        rows = ev.evaluate(out)
        validate_result(rows)
        assert len(rows) >= 1
        assert (rows["capability"] == capabilities.TO_WEIGHTS).all()
