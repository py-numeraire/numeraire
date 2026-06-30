"""Golden test for the 1/A method: reproduce Li-Li-Lyu-Yu (RFS 2025) Table 3 (dp).

Golden discipline (SPEC §6.1.5): assert an invariant (OOS R^2 strictly decreasing in the
conservatism A) **and** the headline scalars within tolerance — never bit-equality. The paper's
dp Table 3 OOS R^2 (%) is {50: 3.4, 100: 2.1, 200: 1.1, 500: 0.5, 1000: 0.2}; our clean-room
reproduction on public Goyal-Welch annual data lands within <= 0.15pp of every entry.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from conftest import load_gw_annual_view
from numeraire.core import capabilities
from numeraire.core.data import TimeSeriesView
from numeraire.core.engine import ForecastOutput, walk_forward_forecast
from numeraire.core.evaluators import OOSR2Evaluator
from numeraire.core.schema import validate_result
from numeraire.methods.conservative_slope import ConservativeSlope

# The GW fixture is NOT committed (kept local pending a data-redistribution audit; see .gitignore).
# This whole golden module skips when it is absent (e.g. in CI), and runs locally where it exists.
_FIXTURE = Path(__file__).parent / "fixtures" / "gw_annual_1a.csv"
pytestmark = pytest.mark.skipif(
    not _FIXTURE.exists(), reason="GW fixture not committed (pending data audit)"
)

# Paper Table 3 (dp), OOS R^2 in percent.
PAPER_TABLE3 = {50: 3.4, 100: 2.1, 200: 1.1, 500: 0.5, 1000: 0.2}


def _oos_r2(a: int) -> float:
    view = load_gw_annual_view()
    out = walk_forward_forecast(
        ConservativeSlope(a=a, sign=1.0, predictor="dp"),
        view,
        window=20,
        method="conservative_slope",
        config={"A": a, "predictor": "dp", "window": 20},
        data_vintage="GW-annual-2024",
    )
    df = OOSR2Evaluator().evaluate(out)
    return float(df.iloc[0]["value"])


def test_golden_table3_dp_within_tolerance() -> None:
    got = {a: _oos_r2(a) for a in PAPER_TABLE3}
    for a, paper in PAPER_TABLE3.items():
        assert abs(got[a] - paper) <= 0.16, f"A={a}: {got[a]:.3f}% vs paper {paper}%"


def test_golden_invariant_r2_decreases_in_a() -> None:
    # Theorem: more conservative (larger A) shrinks the slope toward HM -> lower OOS R^2 here.
    a_sorted = sorted(PAPER_TABLE3)
    vals = [_oos_r2(a) for a in a_sorted]
    assert vals == sorted(vals, reverse=True)
    assert all(v > 0 for v in vals)  # 1/A dominates the historical average OOS


def test_forecast_count_matches_protocol() -> None:
    # 20-year window, annual 1872-2017 (146 obs) -> first forecast 20y in -> 126 OOS forecasts.
    view = load_gw_annual_view()
    out = walk_forward_forecast(
        ConservativeSlope(a=100, predictor="dp"),
        view,
        window=20,
        method="conservative_slope",
    )
    assert len(out.forecasts) == 126
    assert out.capability == capabilities.TO_FORECAST
    # benchmark equals the window historical mean, and the forecast is HM + a tiny shrunk tilt
    assert (out.forecasts.to_numpy() != out.benchmark.to_numpy()).all()


def test_result_row_is_schema_valid() -> None:
    out = walk_forward_forecast(
        ConservativeSlope(a=200, predictor="dp"),
        load_gw_annual_view(),
        window=20,
        method="conservative_slope",
        config={"A": 200},
    )
    df = OOSR2Evaluator().evaluate(out)
    validate_result(df)
    assert df.iloc[0]["metric"] == "oos_r2_pct"
    assert df.iloc[0]["capability"] == capabilities.TO_FORECAST


def test_forecast_uses_last_in_window_predictor() -> None:
    # A 1-asset, single-predictor sanity check against a hand computation on a tiny window.
    index = pd.date_range("2000-12-31", periods=4, freq="YE")
    returns = pd.DataFrame({"mkt": [0.10, 0.20, -0.10, 0.05]}, index=index)
    features = pd.DataFrame({"dp": [-3.0, -2.0, -1.0, 0.0]}, index=index)
    view = TimeSeriesView(returns, features, horizon=1)
    model = ConservativeSlope(a=50, sign=1.0, predictor="dp").fit(view.tail(3))
    assert isinstance(model.capabilities(), set)
    f = model.forecast(view.tail(3))  # window = last 3 obs: dp [-2,-1,0], y [0.20,-0.10,0.05]
    dp = np.array([-2.0, -1.0, 0.0])
    x_std = (dp[-1] - dp.mean()) / dp.std(ddof=1)
    expected = np.mean([0.20, -0.10, 0.05]) + (1.0 / 50) * x_std
    np.testing.assert_allclose(f.to_numpy(), [expected])


def test_is_forecast_output() -> None:
    out = walk_forward_forecast(
        ConservativeSlope(a=100, predictor="dp"),
        load_gw_annual_view(),
        window=20,
        method="x",
    )
    assert isinstance(out, ForecastOutput)
    assert out.forecasts.index.equals(out.realized.index)
    assert not out.realized.isna().to_numpy().any()
