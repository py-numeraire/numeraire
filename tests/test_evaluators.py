"""Tests for native evaluators and capability-dispatched result-schema output."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from numeraire.core import capabilities
from numeraire.core.engine import ForecastOutput, WeightsOutput
from numeraire.core.evaluators import (
    MeanReturnEvaluator,
    OOSR2Evaluator,
    SharpeEvaluator,
    SquaredErrorDiffEvaluator,
    StrategyReturnEvaluator,
)
from numeraire.core.registry import available_evaluators, get_evaluator
from numeraire.core.schema import validate_result


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
    df = OOSR2Evaluator().evaluate(out)
    validate_result(df)
    assert df.iloc[0]["metric"] == "oos_r2_pct"
    np.testing.assert_allclose(df.iloc[0]["value"], 100.0)


def test_oos_r2_equal_to_benchmark_is_zero() -> None:
    out = _forecast_output([0.05, 0.05, 0.05], [0.1, 0.2, 0.0], [0.05, 0.05, 0.05])
    df = OOSR2Evaluator().evaluate(out)
    np.testing.assert_allclose(df.iloc[0]["value"], 0.0)


def test_oos_r2_matches_manual_formula() -> None:
    f = [0.08, 0.10, -0.02]
    r = [0.10, 0.05, 0.00]
    b = [0.04, 0.04, 0.04]
    out = _forecast_output(f, r, b)
    sse_m = np.sum((np.array(r) - np.array(f)) ** 2)
    sse_b = np.sum((np.array(r) - np.array(b)) ** 2)
    expected = (1 - sse_m / sse_b) * 100
    np.testing.assert_allclose(OOSR2Evaluator().evaluate(out).iloc[0]["value"], expected)


def test_oos_r2_rejects_wrong_output() -> None:
    with pytest.raises(TypeError):
        OOSR2Evaluator().evaluate(object())


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
