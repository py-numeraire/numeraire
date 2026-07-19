"""Log-return input contract: declared log returns are converted to simple returns at ingestion.

The audit defect: ``return_type="log"`` inputs flowed into simple-return algebra (the forward
target ``prod(1 + r) - 1`` and weighted-sum strategy scoring), so declared log returns produced
numerically wrong targets and portfolio P&L. For log returns ``ln(1.1)``/``ln(1.2)`` (the two-period
10%/20% case) the old mixed algebra gave ``prod(1 + ln(1.1))(1 + ln(1.2)) - 1 ≈ 0.29525``; the
correct compounded simple return is ``1.1 * 1.2 - 1 = 0.32``.

The fix converts once at the door (``r = expm1(x)``) and keeps every downstream path on a single
simple-return representation. These bites assert the oracle value on the forward-target path and
end to end through a weights backtest, and that the conversion is recorded in hash-visible
provenance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from numeraire.baselines import EqualWeight
from numeraire.core.data import CrossSectionView, TimeSeriesView
from numeraire.core.engine import backtest_weights, config_hash
from numeraire.core.splitter import WalkForwardSplitter

# The two log returns whose simple compounding is the audit's oracle: 10% then 20%.
_LOG_R1 = float(np.log(1.1))
_LOG_R2 = float(np.log(1.2))
_SIMPLE_COMPOUNDED = 0.32  # 1.1 * 1.2 - 1
_OLD_MIXED = (1.0 + _LOG_R1) * (1.0 + _LOG_R2) - 1.0  # ~0.29525, the pre-fix mixed-algebra value


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2000-01-31", periods=n, freq="ME")


def test_forward_target_compounds_converted_log_returns_to_simple() -> None:
    # Anchor at t0; the (t0, t0+2] target compounds the two log returns declared at t1, t2.
    idx = _idx(3)
    raw = pd.DataFrame({"a": [0.0, _LOG_R1, _LOG_R2]}, index=idx)
    view = TimeSeriesView(raw, horizon=2, return_type="log")

    target = view.target_asof(idx[0], horizon=2)
    np.testing.assert_allclose(target, [_SIMPLE_COMPOUNDED])
    # The old mixed-algebra value (~0.29525) must be unreachable.
    assert not np.isclose(target[0], _OLD_MIXED)
    assert abs(_OLD_MIXED - 0.295) < 1e-3  # pin the documented pre-fix mixed-algebra number


def test_aligned_target_matches_simple_view() -> None:
    # A log-declared view is numerically identical to the equivalent simple-declared view.
    idx = _idx(6)
    log_vals = np.log1p(np.array([0.05, -0.02, 0.10, 0.03, -0.04, 0.06]))
    log_view = TimeSeriesView(
        pd.DataFrame({"a": log_vals}, index=idx), horizon=2, return_type="log"
    )
    simple_view = TimeSeriesView(
        pd.DataFrame({"a": np.expm1(log_vals)}, index=idx), horizon=2, return_type="simple"
    )
    _d1, _x1, y_log = log_view.aligned()
    _d2, _x2, y_simple = simple_view.aligned()
    np.testing.assert_allclose(y_log, y_simple)


def test_cross_section_forward_target_converts_log_returns() -> None:
    # The panel `ret` column is converted at ingestion too: the (t0, t0+2] compounded target is the
    # simple 0.32, not the mixed-algebra 0.29525.
    panel = pd.DataFrame(
        {
            "date": np.repeat(_idx(3), 1),
            "asset": ["a", "a", "a"],
            "size": [1.0, 1.0, 1.0],
            "ret": [0.0, _LOG_R1, _LOG_R2],
        }
    )
    view = CrossSectionView(panel, chars=["size"], horizon=2, return_type="log")
    _ids, target = view.target_asof(_idx(3)[0], horizon=2)
    np.testing.assert_allclose(target, [_SIMPLE_COMPOUNDED])
    assert not np.isclose(target[0], _OLD_MIXED)
    assert view.provenance == {"return_input": "log", "converted": "simple"}


def test_weights_backtest_strategy_return_end_to_end() -> None:
    # End to end through backtest_weights: a log-declared view scores identically to the
    # equivalent simple-declared view (conversion happens once at ingestion; engine stays simple).
    idx = _idx(60)
    rng = np.random.default_rng(0)
    simple = pd.DataFrame(rng.normal(0.005, 0.03, size=(60, 3)), index=idx, columns=["a", "b", "c"])
    log = pd.DataFrame(np.log1p(simple.to_numpy()), index=idx, columns=simple.columns)

    sp = WalkForwardSplitter(min_train=24, test_size=12)
    out_log = backtest_weights(
        EqualWeight(),
        TimeSeriesView(log, horizon=1, return_type="log"),
        sp,
        method="ew",
    )
    out_simple = backtest_weights(
        EqualWeight(),
        TimeSeriesView(simple, horizon=1, return_type="simple"),
        sp,
        method="ew",
    )
    np.testing.assert_allclose(
        out_log.strategy_returns().to_numpy(), out_simple.strategy_returns().to_numpy()
    )
    # The log-declared view realizes the SAME simple targets as the simple view (conversion at
    # ingestion), and the strategy return is their equal-weighted mean...
    np.testing.assert_allclose(out_log.realized.to_numpy(), out_simple.realized.to_numpy())
    np.testing.assert_allclose(
        out_log.strategy_returns().to_numpy(), out_log.realized.to_numpy().mean(axis=1)
    )
    # ...never the mean of the raw log realized values (the pre-fix mixed-algebra path).
    log_of_realized = np.log1p(out_simple.realized.to_numpy())
    assert not np.allclose(out_log.strategy_returns().to_numpy(), log_of_realized.mean(axis=1))


def test_provenance_and_config_hash_record_the_conversion() -> None:
    # Same numeric array, two declared conventions: the recorded provenance (and its config hash)
    # must differ, so the conversion is auditable and hash-visible.
    idx = _idx(6)
    arr = pd.DataFrame({"a": [0.01, 0.02, -0.01, 0.03, 0.0, 0.02]}, index=idx)
    v_log = TimeSeriesView(arr, return_type="log")
    v_simple = TimeSeriesView(arr, return_type="simple")

    assert v_log.provenance == {"return_input": "log", "converted": "simple"}
    assert v_simple.provenance == {}
    assert v_log.provenance != v_simple.provenance
    assert config_hash(v_log.provenance) != config_hash(v_simple.provenance)
    # Provenance survives PIT sub-viewing (window / between carry the stamp).
    assert v_log.window(idx[3]).provenance == v_log.provenance
