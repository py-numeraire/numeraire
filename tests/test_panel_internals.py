"""CrossSectionView internals contract: zero-copy PIT windows + vectorized targets + panel metrics.

The int-coded / row-index-matrix redesign must be invisible at the API and strictly cheaper
underneath: PIT sub-views share memory with the parent (no per-fold copy), forward-return
resolution matches a brute-force per-cell reference on randomized ragged panels (entry/exit,
mid-life gaps, multi-step horizons), and the bundled to_weights evaluators accept the panel output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from numeraire import (
    PanelWeightsOutput,
    SharpeEvaluator,
    StrategyReturnEvaluator,
    backtest_panel,
    validate_result,
)
from numeraire.core.data import CrossSectionView
from numeraire.core.splitter import WalkForwardSplitter

CHARS = ["c1", "c2"]


def _random_ragged_panel(seed: int, n_dates: int = 14, n_assets: int = 7) -> pd.DataFrame:
    """Ragged panel with entry/exit and mid-life gaps; asset A0 always alive (non-empty dates)."""
    rng = np.random.default_rng(seed)
    cal = pd.date_range("2001-01-31", periods=n_dates, freq="ME")
    rows: list[dict[str, object]] = []
    for t in cal:
        for i in range(n_assets):
            if i == 0 or rng.random() < 0.7:
                rows.append(
                    {
                        "date": t,
                        "asset": f"A{i}",
                        "c1": float(rng.normal()),
                        "c2": float(rng.normal()),
                        "ret": float(rng.normal(0.01, 0.05)),
                    }
                )
    return pd.DataFrame(rows)


def _brute_targets(df: pd.DataFrame, t: pd.Timestamp, h: int) -> dict[str, float]:
    """Reference target_asof: per alive asset at t, compound (t, t+h] iff present at every step."""
    cal = pd.DatetimeIndex(df["date"].unique()).sort_values()
    p = int(cal.searchsorted(t, side="right")) - 1
    out: dict[str, float] = {}
    for a in sorted(df[df["date"] == cal[p]]["asset"]):
        if p + h >= len(cal):
            out[a] = float("nan")
            continue
        prod, ok = 1.0, True
        for step in range(1, h + 1):
            row = df[(df["date"] == cal[p + step]) & (df["asset"] == a)]
            if row.empty:
                ok = False
                break
            prod *= 1.0 + float(row["ret"].iloc[0])
        out[a] = prod - 1.0 if ok else float("nan")
    return out


def test_target_asof_matches_bruteforce() -> None:
    for seed in (0, 1, 2):
        df = _random_ragged_panel(seed)
        cal = pd.DatetimeIndex(df["date"].unique()).sort_values()
        for h in (1, 3):
            v = CrossSectionView(df, chars=CHARS, horizon=h)
            for t in (cal[0], cal[len(cal) // 2], cal[-2], cal[-1]):
                ids, y = v.target_asof(t)
                expect = _brute_targets(df, t, h)
                assert [str(a) for a in ids] == list(expect)
                np.testing.assert_allclose(y, np.array(list(expect.values())), equal_nan=True)


def test_aligned_matches_bruteforce() -> None:
    for seed in (3, 4):
        df = _random_ragged_panel(seed)
        cal = pd.DatetimeIndex(df["date"].unique()).sort_values()
        for h in (1, 2):
            v = CrossSectionView(df, chars=CHARS, horizon=h)
            keys, x, y = v.aligned()
            expect: dict[tuple[pd.Timestamp, str], float] = {}
            for t in cal:
                for a, tgt in _brute_targets(df, t, h).items():
                    if np.isfinite(tgt):
                        expect[(t, a)] = tgt
            got_keys = set(
                zip(keys.get_level_values("date"), keys.get_level_values("asset"), strict=True)
            )
            assert got_keys == set(expect)
            for (t, a), xi, yi in zip(keys, x, y, strict=True):
                np.testing.assert_allclose(yi, expect[(t, a)])
                src = df[(df["date"] == t) & (df["asset"] == a)]
                np.testing.assert_allclose(xi, src[CHARS].to_numpy(dtype=np.float64).ravel())


def test_window_and_between_are_zero_copy_views() -> None:
    df = _random_ragged_panel(5)
    cal = pd.DatetimeIndex(df["date"].unique()).sort_values()
    v = CrossSectionView(df, chars=CHARS)
    w = v.window(cal[8])
    b = v.between(cal[3], cal[8])
    for sub in (w, b):
        assert np.shares_memory(sub._x, v._x)  # pyright: ignore[reportPrivateUsage]
        assert np.shares_memory(sub._ret, v._ret)  # pyright: ignore[reportPrivateUsage]
        assert np.shares_memory(sub._rowmat, v._rowmat)  # pyright: ignore[reportPrivateUsage]


def test_windowed_assets_exclude_future_entrants() -> None:
    # an asset first appearing after the cutoff must not be in the windowed union axis
    cal = pd.date_range("2001-01-31", periods=6, freq="ME")
    rows = [{"date": t, "asset": "AAA", "c1": 1.0, "c2": 1.0, "ret": 0.01} for t in cal]
    rows += [
        {"date": t, "asset": "ZZZ", "c1": 1.0, "c2": 1.0, "ret": 0.02} for t in cal[4:]
    ]  # late entrant
    v = CrossSectionView(pd.DataFrame(rows), chars=CHARS)
    assert v.assets == ["AAA", "ZZZ"]
    assert v.window(cal[2]).assets == ["AAA"]
    assert v.window(cal[2]).to_tensor().assets == ["AAA"]


class _EW:
    """Equal-weight-the-cross-section toy estimator (panel to_weights)."""

    def fit(self, view: CrossSectionView) -> _EW:
        return self

    def capabilities(self) -> set[str]:
        return {"to_weights"}

    def to_weights(self, view: CrossSectionView) -> pd.Series:
        dates: list[pd.Timestamp] = []
        assets: list[object] = []
        vals: list[float] = []
        for t in view.calendar:
            ids, _x = view.features_asof(t)
            for a in ids:
                dates.append(t)
                assets.append(a)
                vals.append(1.0 / len(ids))
        idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex(dates), assets], names=["date", "asset"])
        return pd.Series(vals, index=idx, name="weight")


def test_panel_output_flows_through_bundled_evaluators() -> None:
    v = CrossSectionView(_random_ragged_panel(6, n_dates=20), chars=CHARS)
    out = backtest_panel(
        _EW(),
        v,
        WalkForwardSplitter(min_train=8, test_size=4),
        method="ew_panel",
        missing_returns="zero",
    )
    assert isinstance(out, PanelWeightsOutput)
    assert out.universe.startswith("n=")
    sharpe = SharpeEvaluator().evaluate(out)
    validate_result(sharpe)
    assert sharpe.iloc[0]["metric"] == "sharpe"
    assert np.isfinite(float(sharpe.iloc[0]["value"]))
    per_period = StrategyReturnEvaluator().evaluate(out)
    validate_result(per_period)
    assert len(per_period) == len(out.strategy_returns())
