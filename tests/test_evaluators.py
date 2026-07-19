"""Tests for native evaluators and capability-dispatched result-schema output."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from numeraire.core import capabilities
from numeraire.core.engine import ForecastOutput, WeightsOutput
from numeraire.core.evaluators import (
    ClarkWestEvaluator,
    MeanReturnEvaluator,
    OutOfSampleR2Evaluator,
    SharpeEvaluator,
    SquaredErrorDiffEvaluator,
    StrategyReturnEvaluator,
)
from numeraire.core.registry import available_evaluators, get_evaluator
from numeraire.core.schema import validate_result
from numeraire.core.stats import newey_west_lrv


def _output(weights: list[float], realized: list[float]) -> WeightsOutput:
    idx = pd.date_range("2000-01-31", periods=len(weights), freq="ME")
    return WeightsOutput(
        weights=pd.DataFrame({"r0": weights}, index=idx),
        realized=pd.DataFrame({"r0": realized}, index=idx),
        method="toy",
        config_hash="deadbeef",
        data_vintage="2026-06",
        run_id="toy-deadbeef",
    )


def test_schema_allows_and_bounds_attrition_columns() -> None:
    from numeraire.core.schema import ATTRITION_COLUMNS

    out = _output([1.0, 1.0], [0.02, -0.01])
    df = SharpeEvaluator().evaluate(out)
    # attrition columns are optional: a plain weights evaluator omits them and still validates
    validate_result(df)
    assert not any(c in df.columns for c in ATTRITION_COLUMNS)
    # when present they must be non-negative counts
    df = df.assign(n_obs=[2], n_dropped=[0])
    validate_result(df)
    with pytest.raises(ValueError, match="non-negative counts"):
        validate_result(df.assign(n_dropped=[-1]))


def test_bundled_evaluators_registered() -> None:
    assert "sharpe" in available_evaluators()
    assert "mean_return" in available_evaluators()
    assert get_evaluator("sharpe").requires == {capabilities.TO_WEIGHTS}


def test_sharpe_emits_valid_result_row() -> None:
    out = _output([1.0, 1.0, 1.0, 1.0], [0.02, -0.01, 0.03, 0.01])
    df = SharpeEvaluator(periods_per_year=12).evaluate(out)
    validate_result(df)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["metric"] == "sharpe"
    assert row["capability"] == capabilities.TO_WEIGHTS
    assert row["config_hash"] == "deadbeef"
    assert np.isfinite(row["value"])


def test_sharpe_matches_manual_annualization() -> None:
    rets = [0.02, -0.01, 0.03, 0.01, 0.00, 0.015]
    out = _output([1.0] * len(rets), rets)
    df = SharpeEvaluator(periods_per_year=12).evaluate(out)
    arr = np.array(rets)
    expected = arr.mean() / arr.std(ddof=1) * np.sqrt(12)
    np.testing.assert_allclose(df.iloc[0]["value"], expected)


def test_mean_return_annualizes() -> None:
    rets = [0.01, 0.02, 0.03]
    out = _output([1.0] * 3, rets)
    df = MeanReturnEvaluator(periods_per_year=12).evaluate(out)
    np.testing.assert_allclose(df.iloc[0]["value"], np.mean(rets) * 12)


def test_sharpe_handles_degenerate_series() -> None:
    out = _output([1.0], [0.02])  # single point -> undefined std
    df = SharpeEvaluator().evaluate(out)
    assert np.isnan(df.iloc[0]["value"])


def test_evaluator_rejects_wrong_output() -> None:
    with pytest.raises(TypeError):
        SharpeEvaluator().evaluate(object())


def _forecast_output(
    forecasts: list[float], realized: list[float], benchmark: list[float]
) -> ForecastOutput:
    idx = pd.date_range("2000-12-31", periods=len(forecasts), freq="YE")
    f = pd.DataFrame({"mkt": forecasts}, index=idx)
    r = pd.DataFrame({"mkt": realized}, index=idx)
    b = pd.DataFrame({"mkt": benchmark}, index=idx)
    return ForecastOutput(
        forecasts=f,
        realized=r,
        benchmark=b,
        method="toy",
        config_hash="cafef00d",
        data_vintage="2026",
        run_id="toy-cafef00d",
    )


def test_oos_r2_registered_and_dispatches_on_forecast() -> None:
    assert "oos_r2" in available_evaluators()
    assert get_evaluator("oos_r2").requires == {capabilities.TO_FORECAST}


def test_oos_r2_perfect_forecast_is_100() -> None:
    out = _forecast_output([0.1, 0.2, 0.0], [0.1, 0.2, 0.0], [0.05, 0.05, 0.05])
    df = OutOfSampleR2Evaluator().evaluate(out)
    validate_result(df)
    assert df.iloc[0]["metric"] == "oos_r2_pct"
    np.testing.assert_allclose(df.iloc[0]["value"], 100.0)


def test_oos_r2_equal_to_benchmark_is_zero() -> None:
    out = _forecast_output([0.05, 0.05, 0.05], [0.1, 0.2, 0.0], [0.05, 0.05, 0.05])
    df = OutOfSampleR2Evaluator().evaluate(out)
    np.testing.assert_allclose(df.iloc[0]["value"], 0.0)


def test_oos_r2_matches_manual_formula() -> None:
    f = [0.08, 0.10, -0.02]
    r = [0.10, 0.05, 0.00]
    b = [0.04, 0.04, 0.04]
    out = _forecast_output(f, r, b)
    sse_m = np.sum((np.array(r) - np.array(f)) ** 2)
    sse_b = np.sum((np.array(r) - np.array(b)) ** 2)
    expected = (1 - sse_m / sse_b) * 100
    np.testing.assert_allclose(OutOfSampleR2Evaluator().evaluate(out).iloc[0]["value"], expected)


def test_oos_r2_rejects_wrong_output() -> None:
    with pytest.raises(TypeError):
        OutOfSampleR2Evaluator().evaluate(object())


def test_oos_r2_zero_benchmark_gkx_convention() -> None:
    # zero benchmark: SSE_bench = sum r^2, ignoring the historical mean carried in the output
    f = [0.08, 0.10, -0.02]
    r = [0.10, 0.05, 0.00]
    out = _forecast_output(f, r, [0.04, 0.04, 0.04])
    sse_m = np.sum((np.array(r) - np.array(f)) ** 2)
    sse_zero = np.sum(np.array(r) ** 2)
    expected = (1 - sse_m / sse_zero) * 100
    np.testing.assert_allclose(
        OutOfSampleR2Evaluator(benchmark="zero").evaluate(out).iloc[0]["value"], expected
    )
    # differs from the historical-mean number
    hist = OutOfSampleR2Evaluator().evaluate(out).iloc[0]["value"]
    assert hist != pytest.approx(expected)


def test_oos_r2_rejects_unknown_benchmark() -> None:
    with pytest.raises(ValueError, match="benchmark must be one of"):
        OutOfSampleR2Evaluator(benchmark="bogus")


def test_strategy_return_emits_one_row_per_date() -> None:
    rets = [0.02, -0.01, 0.03, 0.01]
    out = _output([1.0] * len(rets), rets)
    df = StrategyReturnEvaluator().evaluate(out)
    validate_result(df)
    assert len(df) == len(rets)  # per-period: one row per date (the time series)
    assert (df["metric"] == "strategy_return").all()
    assert df["date"].is_monotonic_increasing
    np.testing.assert_allclose(df["value"].to_numpy(), rets)


def test_sed_per_origin_cumsum_is_cdspe() -> None:
    f = [0.08, 0.10, -0.02]
    r = [0.10, 0.05, 0.00]
    b = [0.04, 0.04, 0.04]
    out = _forecast_output(f, r, b)
    df = SquaredErrorDiffEvaluator().evaluate(out)
    assert len(df) == 3
    assert (df["metric"] == "sed").all()
    rr, ff, bb = np.array(r), np.array(f), np.array(b)
    expected = (rr - bb) ** 2 - (rr - ff) ** 2  # cumsum of this is the CDSPE curve
    np.testing.assert_allclose(df["value"].to_numpy(), expected)


def test_per_period_evaluators_registered() -> None:
    assert "strategy_return" in available_evaluators()
    assert "sed" in available_evaluators()


# --- joint finite mask: a selectively-missing forecast can no longer manufacture skill -----------


def test_oos_r2_joint_mask_does_not_manufacture_skill() -> None:
    # Adversarial forecast: zero where it is present and NaN elsewhere, scored against a zero
    # benchmark. The OLD separate-``nansum`` denominators gave the model a smaller SSE base (only
    # the present half) than the benchmark (the whole sample), reporting large false skill: here
    # that OLD number is ``(1 - 0.05/0.135) * 100 ~= 62.96``. The joint finite mask scores both on
    # the *same* present half, where forecast == benchmark == 0, so honest R^2 is 0.
    out = _forecast_output(
        [0.0, 0.0, float("nan"), float("nan")], [0.1, 0.2, 0.15, 0.25], [0.0] * 4
    )
    df = OutOfSampleR2Evaluator(benchmark="zero").evaluate(out)
    validate_result(df)
    np.testing.assert_allclose(df.iloc[0]["value"], 0.0)
    assert int(df.iloc[0]["n_obs"]) == 2
    assert int(df.iloc[0]["n_dropped"]) == 2


def test_oos_r2_raises_when_majority_missing() -> None:
    # Joint mask drops 3 of 4 candidates (> 50%) -> fail closed rather than score a rump sample.
    out = _forecast_output(
        [0.0, float("nan"), float("nan"), float("nan")], [0.1, 0.2, 0.15, 0.25], [0.0] * 4
    )
    with pytest.raises(ValueError, match="majority-missing"):
        OutOfSampleR2Evaluator(benchmark="zero").evaluate(out)


def test_sed_missing_origin_scores_nan_not_zero() -> None:
    # A forecast missing at an origin used to make ``nansum`` yield a spurious 0 there (a flat point
    # on the CDSPE curve). It now scores NaN with n_obs=0, so the missing origin is visibly empty.
    out = _forecast_output([0.08, float("nan"), -0.02], [0.10, 0.05, 0.00], [0.04, 0.04, 0.04])
    df = SquaredErrorDiffEvaluator().evaluate(out)
    validate_result(df)
    assert np.isnan(df.iloc[1]["value"])
    assert int(df.iloc[1]["n_obs"]) == 0
    assert int(df.iloc[1]["n_dropped"]) == 1
    # the present origins still score, over their joint-finite cell
    assert np.isfinite(df.iloc[0]["value"]) and int(df.iloc[0]["n_obs"]) == 1


def test_clark_west_drops_missing_origins_not_scores_them_as_zero() -> None:
    # A forecast present on the first half and NaN on the second. The OLD ``nansum`` carried each
    # missing origin as a phantom adj=0 observation, diluting the mean and inflating the effective
    # sample of the Newey-West statistic. The NEW code drops those origins, so the statistic equals
    # the one computed on the honest present-only sample and differs from the OLD phantom-zero one.
    r = [0.10, 0.05, 0.08, 0.12, 0.09, 0.07, 0.11, 0.06]
    f = [0.10, 0.05, 0.08, 0.12, float("nan"), float("nan"), float("nan"), float("nan")]
    b = [0.04] * 8
    out = _forecast_output(f, r, b)
    df = ClarkWestEvaluator().evaluate(out)
    validate_result(df)
    cw_t_new = float(df[df["metric"] == "cw_t"].iloc[0]["value"])

    rr, ff, bb = np.array(r), np.array(f), np.array(b)
    per_cell = (rr - bb) ** 2 - ((rr - ff) ** 2 - (bb - ff) ** 2)
    present = np.array([0, 1, 2, 3])

    def _cw_t(adj: np.ndarray) -> float:
        n = adj.size
        se = float(np.sqrt(newey_west_lrv(adj, 0) / n))
        return float(adj.mean() / se)

    adj_new = per_cell[present]
    adj_old = np.concatenate([adj_new, np.zeros(4)])  # OLD: missing origins as phantom zeros
    np.testing.assert_allclose(cw_t_new, _cw_t(adj_new))
    assert cw_t_new != pytest.approx(_cw_t(adj_old))
    assert int(df.iloc[0]["n_obs"]) == 4 and int(df.iloc[0]["n_dropped"]) == 4
