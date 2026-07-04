"""Weight-stream simulator: drift/rebalance accounting identities, costs, schedules, policies.

Each accounting property is pinned by a hand-computable identity rather than a snapshot:
buy-and-hold wealth equals the weighted sum of asset wealths, constant-mix equals per-row dot
products, turnover matches the drifted pre-trade formula, and cost enters as a one-off NAV haircut
on the trade row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from numeraire.core.simulate import RebalanceSchedule, SimulationResult, simulate_weights

IDX = pd.date_range("2020-01-31", periods=8, freq="ME")


def _returns(seed: int = 0, n: int = 8, assets: tuple[str, ...] = ("A", "B")) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.01, 0.04, size=(n, len(assets))), index=IDX[:n], columns=list(assets)
    )


def test_constant_mix_equals_per_row_dot_product() -> None:
    r = _returns()
    w = pd.DataFrame({"A": [0.6], "B": [0.4]}, index=IDX[:1])
    res = simulate_weights(r, w, mode="target_weight")
    expect = r.iloc[1:].to_numpy() @ np.array([0.6, 0.4])
    np.testing.assert_allclose(res.gross.to_numpy(), expect)
    np.testing.assert_allclose(res.net.to_numpy(), expect)  # no costs configured


def test_buy_and_hold_wealth_identity() -> None:
    # drifted_holdings with a single rebalance = buy-and-hold: terminal wealth must equal the
    # weighted sum of per-asset cumulated wealths (cash weight 0 here).
    r = _returns(1)
    w0 = np.array([0.3, 0.7])
    w = pd.DataFrame({"A": [w0[0]], "B": [w0[1]]}, index=IDX[:1])
    res = simulate_weights(r, w, mode="drifted_holdings")
    wealth = float(np.prod(1.0 + res.gross.to_numpy()))
    asset_wealth = (1.0 + r.iloc[1:]).prod(axis=0).to_numpy()
    np.testing.assert_allclose(wealth, float(w0 @ asset_wealth))


def test_modes_coincide_with_one_row_per_span() -> None:
    # monthly decisions on monthly data (one return row per rebalance): the two accounting modes
    # must produce identical gross returns.
    r = _returns(2)
    w = pd.DataFrame({"A": np.linspace(0.2, 0.8, 7), "B": np.linspace(0.8, 0.2, 7)}, index=IDX[:7])
    a = simulate_weights(r, w, mode="target_weight")
    b = simulate_weights(r, w, mode="drifted_holdings")
    np.testing.assert_allclose(a.gross.to_numpy(), b.gross.to_numpy())


def test_turnover_uses_pretrade_drifted_weights() -> None:
    # two rebalances, one row apart: pre-trade weights at the 2nd = 1st target drifted one row.
    r = _returns(3)
    t0, t1 = np.array([0.5, 0.5]), np.array([0.2, 0.8])
    w = pd.DataFrame([t0, t1], index=IDX[:2], columns=["A", "B"])
    res = simulate_weights(r, w, mode="drifted_holdings")
    r1 = r.iloc[1].to_numpy()
    port = float(t0 @ r1)  # fully invested
    drifted = t0 * (1.0 + r1) / (1.0 + port)
    np.testing.assert_allclose(res.turnover.iloc[0], np.abs(t0).sum())  # funded from cash
    np.testing.assert_allclose(res.turnover.iloc[1], np.abs(t1 - drifted).sum())


def test_cost_is_one_off_nav_haircut_on_trade_row() -> None:
    r = _returns(4)
    w = pd.DataFrame({"A": [1.0]}, index=IDX[:1])
    bps = 50.0
    res = simulate_weights(r[["A"]], w, cost_bps=bps)
    tau = float(res.turnover.iloc[0])
    g0 = float(res.gross.iloc[0])
    np.testing.assert_allclose(float(res.net.iloc[0]), (1.0 - bps / 1e4 * tau) * (1.0 + g0) - 1.0)
    np.testing.assert_allclose(res.net.iloc[1:].to_numpy(), res.gross.iloc[1:].to_numpy())


def test_cash_leg_earns_rf() -> None:
    r = _returns(5)
    rf = pd.Series(0.002, index=IDX)
    w = pd.DataFrame({"A": [0.5], "B": [0.0]}, index=IDX[:1])  # 50% cash
    res = simulate_weights(r, w, rf=rf)
    expect = 0.5 * r["A"].iloc[1:].to_numpy() + 0.5 * 0.002
    np.testing.assert_allclose(res.gross.to_numpy(), expect)


def test_long_short_book_is_not_silently_renormalized() -> None:
    r = _returns(6)
    w = pd.DataFrame({"A": [1.0], "B": [-1.0]}, index=IDX[:1])  # net 0, gross 2
    res = simulate_weights(r, w)
    expect = r["A"].iloc[1:].to_numpy() - r["B"].iloc[1:].to_numpy()
    np.testing.assert_allclose(res.gross.to_numpy(), expect)
    gross_b = simulate_weights(r, w, normalize="gross_budget")
    np.testing.assert_allclose(gross_b.gross.to_numpy(), expect / 2.0)
    with pytest.raises(ValueError, match="net_budget"):
        simulate_weights(r, w, normalize="net_budget")  # net ~0 cannot be scaled to 1


def test_missing_return_policy() -> None:
    r = _returns(7)
    r.iloc[2, r.columns.get_loc("B")] = np.nan
    w = pd.DataFrame({"A": [0.5], "B": [0.5]}, index=IDX[:1])
    with pytest.raises(ValueError, match="missing return"):
        simulate_weights(r, w)
    res = simulate_weights(r, w, missing="zero")
    assert np.isfinite(res.gross.to_numpy()).all()


def test_month_end_schedule_on_daily_data() -> None:
    daily = pd.date_range("2020-01-01", periods=45, freq="B")
    rng = np.random.default_rng(8)
    r = pd.DataFrame({"A": rng.normal(0.0005, 0.01, size=45)}, index=daily)
    sched = RebalanceSchedule.from_rule(daily, rule="month_end")
    # signals = last business day per month; each governs the rows strictly after it
    months = pd.PeriodIndex(daily, freq="M")
    expect_signals = [daily[months == m][-1] for m in months.unique()]
    assert list(sched.signal_dates) == [d for d in expect_signals if d != daily[-1]] + (
        [daily[-1]] if daily[-1] in sched.signal_dates else []
    )
    w = pd.DataFrame({"A": np.full(len(sched.signal_dates), 1.0)}, index=sched.signal_dates)
    res = simulate_weights(r, w, schedule=sched, mode="drifted_holdings")
    assert isinstance(res, SimulationResult)
    lo0 = sched.spans[0][0]
    assert res.gross.index[0] == daily[lo0]
    np.testing.assert_allclose(res.gross.to_numpy(), r["A"].iloc[lo0:].to_numpy())


def test_signal_collision_raises() -> None:
    daily = pd.date_range("2020-01-01", periods=10, freq="B")
    sigs = pd.DatetimeIndex([daily[2] - pd.Timedelta(hours=2), daily[2] - pd.Timedelta(hours=1)])
    with pytest.raises(ValueError, match="same first data row"):
        RebalanceSchedule.from_signals(daily, sigs)


def test_meta_records_conventions() -> None:
    r = _returns(9)
    w = pd.DataFrame({"A": [1.0]}, index=IDX[:1])
    res = simulate_weights(r[["A"]], w, cost_bps=10.0)
    assert res.meta["turnover_convention"] == "l1_pretrade_drift"
    assert res.meta["cost_timing"] == "at_rebalance_before_period_return"
    assert res.meta["initial"] == "from_cash"
