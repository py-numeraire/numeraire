"""Inference primitives: GRS, paired Sharpe, Clark-West, HAC alpha — identity-pinned tests.

Correctness is asserted through independent algebra where one exists (the GRS statistic must
equal its max-Sharpe-geometry form), exact invariances (scale-invariant Sharpe, antisymmetry),
degenerate cases, and power under an obvious alternative — never through snapshot values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from numeraire import (
    AlphaEvaluator,
    ClarkWestEvaluator,
    WeightsOutput,
    alpha_regression,
    clark_west,
    grs_test,
    newey_west_lrv,
    sharpe_diff_test,
    validate_result,
)
from numeraire.core.engine import ForecastOutput


def _factor_panel(seed: int, t_n: int = 240, n: int = 8, k: int = 2, alpha_shift: float = 0.0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("1990-01-31", periods=t_n, freq="ME")
    f = rng.normal(0.005, 0.03, size=(t_n, k))
    beta = rng.normal(1.0, 0.3, size=(k, n))
    eps = rng.normal(0.0, 0.02, size=(t_n, n))
    y = alpha_shift + f @ beta + eps
    factors = pd.DataFrame(f, index=idx, columns=[f"F{j}" for j in range(k)])
    assets = pd.DataFrame(y, index=idx, columns=[f"P{j}" for j in range(n)])
    return assets, factors


def test_grs_matches_max_sharpe_geometry() -> None:
    # GRS (1989) eq. 7: the quadratic form equals (1 + sh*^2) / (1 + sh_F^2) - 1, where sh*^2 is
    # the max squared Sharpe over assets+factors and sh_F^2 over factors (MLE moments). The
    # regression algebra and the tangency algebra must agree to machine precision.
    assets, factors = _factor_panel(0)
    res = grs_test(assets, factors)
    both = np.column_stack([assets.to_numpy(), factors.to_numpy()])
    mu = both.mean(axis=0)
    omega = ((both - mu).T @ (both - mu)) / len(both)
    sh2_all = float(mu @ np.linalg.solve(omega, mu))
    t_n, n = assets.shape
    k = factors.shape[1]
    expect = (t_n - n - k) / n * ((1.0 + sh2_all) / (1.0 + res.sh2_factors) - 1.0)
    np.testing.assert_allclose(res.f_stat, expect, rtol=1e-8)
    assert 0.0 <= res.p_value <= 1.0


def test_grs_detects_inflated_alphas() -> None:
    null_assets, factors = _factor_panel(1)
    shifted, _ = _factor_panel(1, alpha_shift=0.02)
    assert grs_test(shifted, factors).p_value < 0.01
    assert grs_test(shifted, factors).f_stat > grs_test(null_assets, factors).f_stat


def test_grs_requires_enough_observations() -> None:
    assets, factors = _factor_panel(2, t_n=10, n=8, k=2)
    with pytest.raises(ValueError, match="T > N"):
        grs_test(assets, factors)


def test_sharpe_diff_scale_invariance_and_antisymmetry() -> None:
    rng = np.random.default_rng(3)
    a = rng.normal(0.01, 0.05, size=400)
    # perfectly collinear series (b = 2a) have theta = 0 exactly -> the statistic is undefined
    degenerate = sharpe_diff_test(a, 2.0 * a)
    assert np.isnan(degenerate.z_stat)
    # near-identical Sharpe (scaled + small independent noise) -> z near 0, p large
    near = sharpe_diff_test(a, 2.0 * a + rng.normal(0.0, 0.005, size=400))
    assert abs(near.z_stat) < 1.0
    assert near.p_value > 0.3
    b = rng.normal(0.005, 0.05, size=400)
    ab, ba = sharpe_diff_test(a, b), sharpe_diff_test(b, a)
    np.testing.assert_allclose(ab.z_stat, -ba.z_stat)
    np.testing.assert_allclose(ab.sharpe_a, ba.sharpe_b)


def test_sharpe_diff_detects_dominant_series() -> None:
    rng = np.random.default_rng(4)
    base = rng.normal(0.0, 0.04, size=2000)
    res = sharpe_diff_test(base + 0.01, base * 1.02 + 0.001)  # a clearly higher Sharpe
    assert res.z_stat > 2.0
    assert res.p_value < 0.05


def test_newey_west_lags_zero_is_plain_variance() -> None:
    rng = np.random.default_rng(5)
    x = rng.normal(size=500)
    np.testing.assert_allclose(newey_west_lrv(x, 0), float(np.var(x)))


def test_newey_west_grows_under_positive_autocorrelation() -> None:
    rng = np.random.default_rng(6)
    e = rng.normal(size=3000)
    x = np.empty_like(e)
    x[0] = e[0]
    for i in range(1, len(e)):
        x[i] = 0.8 * x[i - 1] + e[i]
    assert newey_west_lrv(x, 12) > 2.0 * newey_west_lrv(x, 0)


def test_clark_west_power_and_degenerate_case() -> None:
    rng = np.random.default_rng(7)
    t_n = 500
    signal = rng.normal(0.0, 0.02, size=t_n)
    y = signal + rng.normal(0.0, 0.02, size=t_n)
    bench = np.zeros(t_n)  # nested restricted model
    good = clark_west(y, 0.9 * signal, bench)
    assert good.t_stat > 1.645  # one-sided 5%
    assert good.mspe_model < good.mspe_benchmark
    degenerate = clark_west(y, bench, bench)  # model == benchmark
    assert np.isnan(degenerate.t_stat)


def test_clark_west_matches_manual_computation() -> None:
    rng = np.random.default_rng(8)
    y, f, b = (rng.normal(size=50) for _ in range(3))
    res = clark_west(y, f, b)
    adj = (y - b) ** 2 - ((y - f) ** 2 - (b - f) ** 2)
    se = np.sqrt(np.var(adj) / len(adj))
    np.testing.assert_allclose(res.t_stat, adj.mean() / se)


def test_alpha_regression_recovers_coefficients() -> None:
    rng = np.random.default_rng(9)
    idx = pd.date_range("2000-01-31", periods=360, freq="ME")
    f = pd.DataFrame({"mkt": rng.normal(0.006, 0.04, size=360)}, index=idx)
    y = pd.Series(0.004 + 0.6 * f["mkt"] + rng.normal(0.0, 0.005, size=360), index=idx)
    res = alpha_regression(y, f, nw_lags=3)
    np.testing.assert_allclose(res.alpha, 0.004, atol=5e-4)
    np.testing.assert_allclose(res.betas[0], 0.6, atol=0.02)
    assert res.alpha_t > 3.0
    assert res.r2 > 0.9


def test_alpha_regression_inner_joins_and_validates() -> None:
    rng = np.random.default_rng(10)
    idx = pd.date_range("2000-01-31", periods=100, freq="ME")
    f = pd.DataFrame({"mkt": rng.normal(size=100)}, index=idx)
    y = pd.Series(rng.normal(size=80), index=idx[:80])  # shorter — inner join
    assert alpha_regression(y, f).n_obs == 80
    with pytest.raises(ValueError, match="overlapping"):
        alpha_regression(y.iloc[:3], f)


def _toy_forecast_output(seed: int) -> ForecastOutput:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2010-01-31", periods=200, freq="ME")
    signal = rng.normal(0.0, 0.02, size=200)
    frame = lambda v: pd.DataFrame({"mkt": v}, index=idx)  # noqa: E731
    return ForecastOutput(
        forecasts=frame(0.9 * signal),
        realized=frame(signal + rng.normal(0.0, 0.02, size=200)),
        benchmark=frame(np.zeros(200)),
        method="toy",
        config_hash="cfg",
        data_vintage="synthetic",
        run_id="toy-cfg",
    )


def test_clark_west_evaluator_emits_schema_rows() -> None:
    out = _toy_forecast_output(11)
    rows = ClarkWestEvaluator().evaluate(out)
    validate_result(rows)
    assert list(rows["metric"]) == ["cw_t", "cw_p"]
    t_val = float(rows.loc[rows["metric"] == "cw_t", "value"].iloc[0])
    assert t_val > 1.645


def test_alpha_evaluator_emits_schema_rows() -> None:
    rng = np.random.default_rng(12)
    idx = pd.date_range("2010-01-31", periods=240, freq="ME")
    factors = pd.DataFrame({"mkt": rng.normal(0.006, 0.04, size=240)}, index=idx)
    realized = pd.DataFrame(
        {"strat": 0.003 + 0.5 * factors["mkt"] + rng.normal(0.0, 0.004, size=240)}, index=idx
    )
    out = WeightsOutput(
        weights=pd.DataFrame({"strat": np.ones(240)}, index=idx),
        realized=realized,
        method="toy",
        config_hash="cfg",
        data_vintage="synthetic",
        run_id="toy-cfg",
    )
    rows = AlphaEvaluator(factors, nw_lags=3).evaluate(out)
    validate_result(rows)
    alpha_ann = float(rows.loc[rows["metric"] == "alpha_ann", "value"].iloc[0])
    alpha_t = float(rows.loc[rows["metric"] == "alpha_t", "value"].iloc[0])
    np.testing.assert_allclose(alpha_ann, 0.003 * 12, atol=0.01)
    assert alpha_t > 3.0
