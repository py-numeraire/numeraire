"""Log-return input contract: declared log returns are converted to simple returns at ingestion.

The audit defect: ``return_type="log"`` inputs flowed into simple-return algebra (the forward
target ``prod(1 + r) - 1`` and weighted-sum strategy scoring), so declared log returns produced
numerically wrong targets and portfolio P&L. For log returns ``ln(1.1)``/``ln(1.2)`` (the
two-period 10%/20% case) the old mixed algebra gave
``(1 + ln(1.1)) * (1 + ln(1.2)) - 1 = 0.295008836958516``; the correct compounded simple return is
``1.1 * 1.2 - 1 = 0.32``.

The fix converts once at the door (``r = expm1(x)``) and keeps every downstream path — including
the stored representation the ejects read — on a single simple-return representation. The
conversion is recorded on the view's ``provenance`` property; merging that into a backtest
``config`` (opt-in) makes it part of the run's ``config_hash``. These bites assert the oracle value
on the forward-target path and end to end through a weights backtest, that every eject exposes the
converted (simple) representation, and that the conformance twin carries the provenance stamp.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from numeraire.baselines import EqualWeight
from numeraire.core.data import CrossSectionView, TimeSeriesView
from numeraire.core.engine import WeightsOutput, backtest_weights
from numeraire.core.splitter import WalkForwardSplitter
from numeraire.testing import _perturb_after

# The two log returns whose simple compounding is the audit's oracle: 10% then 20%.
_LOG_R1 = float(np.log(1.1))
_LOG_R2 = float(np.log(1.2))
_SIMPLE_COMPOUNDED = 0.32  # 1.1 * 1.2 - 1
# The pre-fix mixed-algebra value: log returns fed straight into prod(1 + r) - 1.
_OLD_MIXED = (1.0 + _LOG_R1) * (1.0 + _LOG_R2) - 1.0  # = 0.295008836958516
_LOG_STAMP = {"return_input": "log", "converted": "simple"}


def _idx(n: int) -> pd.DatetimeIndex:
    return pd.date_range("2000-01-31", periods=n, freq="ME")


def _log_panel() -> pd.DataFrame:
    """A one-asset three-date panel whose ``ret`` column holds the oracle log returns."""
    return pd.DataFrame(
        {
            "date": np.repeat(_idx(3), 1),
            "asset": ["a", "a", "a"],
            "size": [1.0, 1.0, 1.0],
            "ret": [0.0, _LOG_R1, _LOG_R2],
        }
    )


def test_forward_target_compounds_converted_log_returns_to_simple() -> None:
    # Anchor at t0; the (t0, t0+2] target compounds the two log returns declared at t1, t2.
    idx = _idx(3)
    raw = pd.DataFrame({"a": [0.0, _LOG_R1, _LOG_R2]}, index=idx)
    view = TimeSeriesView(raw, horizon=2, return_type="log")

    target = view.target_asof(idx[0], horizon=2)
    np.testing.assert_allclose(target, [_SIMPLE_COMPOUNDED])
    # The old mixed-algebra value must be unreachable.
    assert not np.isclose(target[0], _OLD_MIXED)
    np.testing.assert_allclose(_OLD_MIXED, 0.295008836958516)  # pin the pre-fix number


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
    # simple 0.32, not the mixed-algebra 0.295008836958516.
    view = CrossSectionView(_log_panel(), chars=["size"], horizon=2, return_type="log")
    _ids, target = view.target_asof(_idx(3)[0], horizon=2)
    np.testing.assert_allclose(target, [_SIMPLE_COMPOUNDED])
    assert not np.isclose(target[0], _OLD_MIXED)
    assert view.provenance == _LOG_STAMP


def test_cross_section_stores_converted_simple_representation() -> None:
    # The conversion happens at ingestion, not lazily at compounding: every eject of the stored
    # representation (panel_frame / to_tensor / aligned) must already expose SIMPLE returns,
    # identical to the equivalent simple-declared view. A lazy implementation that kept `ret`
    # logarithmic and converted only inside the compounding loop would fail here.
    log_view = CrossSectionView(_log_panel(), chars=["size"], horizon=2, return_type="log")
    simple_panel = _log_panel().assign(ret=np.expm1(_log_panel()["ret"].to_numpy()))
    simple_view = CrossSectionView(simple_panel, chars=["size"], horizon=2, return_type="simple")

    np.testing.assert_allclose(
        log_view.panel_frame()["ret"].to_numpy(), simple_view.panel_frame()["ret"].to_numpy()
    )
    np.testing.assert_allclose(
        log_view.to_tensor().returns, simple_view.to_tensor().returns, equal_nan=True
    )
    k_log, _x_log, y_log = log_view.aligned()
    k_simple, _x_simple, y_simple = simple_view.aligned()
    assert k_log.equals(k_simple)
    np.testing.assert_allclose(y_log, y_simple)
    # ... and the simple values genuinely differ from the raw (log) input column.
    assert not np.allclose(log_view.panel_frame()["ret"].to_numpy(), _log_panel()["ret"].to_numpy())


def test_cross_section_provenance_survives_windowing() -> None:
    idx = _idx(3)
    view = CrossSectionView(_log_panel(), chars=["size"], horizon=1, return_type="log")
    assert view.window(idx[1]).provenance == _LOG_STAMP
    assert view.between(idx[0], idx[2]).provenance == _LOG_STAMP


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


def test_provenance_records_conversion_and_survives_windowing() -> None:
    # Same numeric array, two declared conventions: the recorded provenance must differ, so the
    # conversion is auditable; the stamp survives PIT sub-viewing.
    idx = _idx(6)
    arr = pd.DataFrame({"a": [0.01, 0.02, -0.01, 0.03, 0.0, 0.02]}, index=idx)
    v_log = TimeSeriesView(arr, return_type="log")
    v_simple = TimeSeriesView(arr, return_type="simple")

    assert v_log.provenance == _LOG_STAMP
    assert v_simple.provenance == {}
    assert v_log.window(idx[3]).provenance == _LOG_STAMP
    assert v_log.between(idx[0], idx[4]).provenance == _LOG_STAMP


def test_config_hash_records_conversion_when_provenance_is_merged() -> None:
    # Provenance enters `config_hash` when (and only when) the caller merges `view.provenance`
    # into the backtest `config` — the recording is opt-in, not automatic.
    idx = _idx(60)
    rng = np.random.default_rng(1)
    simple = pd.DataFrame(rng.normal(0.005, 0.03, size=(60, 2)), index=idx, columns=["a", "b"])
    v_log = TimeSeriesView(
        pd.DataFrame(np.log1p(simple.to_numpy()), index=idx, columns=["a", "b"]), return_type="log"
    )
    v_simple = TimeSeriesView(simple, return_type="simple")
    sp = WalkForwardSplitter(min_train=24, test_size=12)

    def _run(view: TimeSeriesView, config: dict[str, str] | None) -> WeightsOutput:
        return backtest_weights(EqualWeight(), view, sp, method="ew", config=config)

    # merged: the two runs' config hashes differ — the conversion is recorded in provenance
    merged_log = _run(v_log, {**v_log.provenance})
    merged_simple = _run(v_simple, {**v_simple.provenance})
    assert merged_log.config_hash != merged_simple.config_hash
    # not merged (default): the hashes are identical — pins the opt-in nature explicitly
    default_log = _run(v_log, None)
    default_simple = _run(v_simple, None)
    assert default_log.config_hash == default_simple.config_hash


def test_conformance_twin_carries_log_provenance() -> None:
    # The check_no_lookahead twin is rebuilt from ejected (already-simple) frames; it must carry
    # the source view's provenance so an estimator reading the public property sees identical
    # values on both twins (no false leak signal), and must not double-convert the returns.
    idx = _idx(12)
    rng = np.random.default_rng(2)
    simple = pd.DataFrame(rng.normal(0.005, 0.03, size=(12, 2)), index=idx, columns=["a", "b"])
    ts_view = TimeSeriesView(
        pd.DataFrame(np.log1p(simple.to_numpy()), index=idx, columns=["a", "b"]), return_type="log"
    )
    ts_twin = _perturb_after(ts_view, idx[6])
    assert ts_twin.provenance == ts_view.provenance == _LOG_STAMP
    # identical data on <= t: the twin was not re-converted
    np.testing.assert_allclose(
        ts_twin.returns_frame().loc[: idx[6]].to_numpy(),
        ts_view.returns_frame().loc[: idx[6]].to_numpy(),
    )

    cs_view = CrossSectionView(_log_panel(), chars=["size"], horizon=1, return_type="log")
    cs_twin = _perturb_after(cs_view, _idx(3)[1])
    assert cs_twin.provenance == cs_view.provenance == _LOG_STAMP
