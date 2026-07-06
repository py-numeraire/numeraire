"""CharBlock: heterogeneous per-asset ([t,i]) characteristic sources concatenated into a panel.

Covers the two PIT modes (per-asset lag; per-asset vintage edge with no revision leak), multi-source
concatenation (two vendors), missing-asset -> nan, and backward compatibility (no blocks == today).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import toy_panel, toy_vintaged_chars
from numeraire.core.data import CharBlock, CrossSectionView

DATES = pd.date_range("2000-01-31", periods=4, freq="ME")


def _panel() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for d_i, d in enumerate(DATES):
        for a in ("A", "B"):
            rows.append({"date": d, "asset": a, "base": float(d_i), "ret": 0.01 * d_i})
    return pd.DataFrame(rows)


def _osap() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for d_i, d in enumerate(DATES):
        rows.append({"date": d, "asset": "A", "osap": 10.0 + d_i})
        rows.append({"date": d, "asset": "B", "osap": 20.0 + d_i})
    return pd.DataFrame(rows)


def _vint() -> pd.DataFrame:
    # asset A, ref Jan: first released in the Feb vintage (100), revised in the Mar vintage (101)
    return pd.DataFrame(
        {
            "ref_date": pd.to_datetime(["2000-01-31", "2000-01-31"]),
            "asset": ["A", "A"],
            "vintage": pd.to_datetime(["2000-02-29", "2000-03-31"]),
            "macro": [100.0, 101.0],
        }
    )


def _asof(v: CrossSectionView, t: object, asset: str, col: int) -> float:
    ids, x = v.features_asof(t)
    return float(x[list(ids).index(asset), col])


def test_lagged_char_block_pulls_back_per_asset() -> None:
    blk = CharBlock(_osap(), ["osap"], lag=1)
    v = CrossSectionView(_panel(), chars=["base"], char_blocks=[blk], horizon=1)
    assert v.char_names == ["base", "osap"]
    # at Feb, osap lagged 1 = Jan's value (10 / 20); inline base at Feb = 1.0
    ids, x = v.features_asof(DATES[1])
    row = {a: x[i] for i, a in enumerate(ids)}
    np.testing.assert_allclose(row["A"], [1.0, 10.0])
    np.testing.assert_allclose(row["B"], [1.0, 20.0])


def test_lagged_char_block_warmup_is_nan() -> None:
    blk = CharBlock(_osap(), ["osap"], lag=1)
    v = CrossSectionView(_panel(), chars=["base"], char_blocks=[blk], horizon=1)
    _ids, x = v.features_asof(DATES[0])  # Jan: nothing one step before -> nan
    assert np.isnan(x[:, 1]).all()


def test_vintaged_char_block_no_revision_leak() -> None:
    blk = CharBlock(_vint(), ["macro"], vintage_col="vintage")
    v = CrossSectionView(_panel(), chars=["base"], char_blocks=[blk], horizon=1)
    assert np.isnan(_asof(v, DATES[0], "A", 1))  # Jan: Feb vintage not yet available
    np.testing.assert_allclose(_asof(v, DATES[1], "A", 1), 100.0)  # Feb: first release
    np.testing.assert_allclose(_asof(v, DATES[2], "A", 1), 101.0)  # Mar: revision now visible


def test_block_missing_asset_is_nan() -> None:
    blk = CharBlock(_vint(), ["macro"], vintage_col="vintage")  # only asset A
    v = CrossSectionView(_panel(), chars=["base"], char_blocks=[blk], horizon=1)
    assert np.isnan(_asof(v, DATES[2], "B", 1))  # B not in the block


def test_multi_source_concatenation() -> None:
    v = CrossSectionView(
        _panel(),
        chars=["base"],
        char_blocks=[
            CharBlock(_osap(), ["osap"], lag=0),
            CharBlock(_vint(), ["macro"], vintage_col="vintage"),
        ],
        horizon=1,
    )
    assert v.char_names == ["base", "osap", "macro"]  # concatenated along the char axis, in order
    _keys, x, _y = v.aligned()
    assert x.shape[1] == 3


def test_char_blocks_none_is_todays_behaviour() -> None:
    a = CrossSectionView(_panel(), chars=["base"], horizon=1)
    b = CrossSectionView(_panel(), chars=["base"], char_blocks=None, horizon=1)
    _ka, xa, _ya = a.aligned()
    _kb, xb, _yb = b.aligned()
    assert a.char_names == b.char_names == ["base"]
    np.testing.assert_array_equal(xa, xb)


def test_block_chars_survive_windowing() -> None:
    blk = CharBlock(_osap(), ["osap"], lag=0)
    v = CrossSectionView(_panel(), chars=["base"], char_blocks=[blk], horizon=1)
    w = v.window(DATES[2])
    np.testing.assert_allclose(_asof(w, DATES[1], "A", 1), 11.0)  # osap at Feb, unchanged by window


def test_vintaged_chars_on_ragged_panel() -> None:
    # the [t,i] vintage case on the ragged toy panel: per-asset edge, release timing, no leak
    blk = CharBlock(toy_vintaged_chars(), ["acc"], vintage_col="vintage", lag=0)
    v = CrossSectionView(toy_panel(), chars=["size", "bm", "mom"], char_blocks=[blk], horizon=1)
    assert v.char_names == ["size", "bm", "mom", "acc"]
    cal = pd.date_range("2000-01-31", periods=8, freq="ME")

    def acc(t: object, a: str) -> float:
        ids, x = v.features_asof(t)
        return float(x[list(ids).index(a), 3])

    assert np.isnan(acc(cal[0], "AAA"))  # Jan: Jan-ref releases in Feb -> nothing available yet
    np.testing.assert_allclose(acc(cal[1], "AAA"), 10.0)  # Feb: Jan-ref first release
    np.testing.assert_allclose(acc(cal[1], "CCC"), 30.0)  # per-asset (differs from AAA)
    # Mar: Feb-ref is now the newest available (11.0); the Jan revision (10.5) is superseded and
    # Feb's own revision (11.5, vintage Apr) is not yet visible -> no early-revision leak
    np.testing.assert_allclose(acc(cal[2], "AAA"), 11.0)
    assert np.isnan(acc(cal[3], "DDD"))  # DDD absent from the vintaged source -> nan


# --- timestamp availability: sub-monthly data must not leak within a month ------------------------


def test_lagged_intra_month_row_is_not_visible_early() -> None:
    # Daily char panel for asset A stamped Mar-10 / Mar-20; decision date Mar-5 with lag=0. Nothing
    # is known yet (contract: known at its row date) — the month-ordinal resolution used to return
    # the Mar-20 value, a 15-day intra-month look-ahead.
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-03-10", "2020-03-20"]),
            "asset": ["A", "A"],
            "x": [1.0, 2.0],
        }
    )
    blk = CharBlock(panel, ["x"], lag=0)
    out = blk.resolve(
        pd.DatetimeIndex(["2020-03-05"]), np.array(["A"], dtype=object), np.array([0], np.int64)
    )
    assert np.isnan(out[0, 0])


def test_lagged_month_end_row_is_not_visible_at_month_start() -> None:
    # Monthly panel stamped month-end (rows Feb-29 / Mar-31) read on a daily calendar; decision
    # Mar-2 must use the Feb row, not the not-yet-known Mar-31 row.
    panel = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-02-29", "2020-03-31"]),
            "asset": ["A", "A"],
            "x": [10.0, 20.0],
        }
    )
    blk = CharBlock(panel, ["x"], lag=0)
    out = blk.resolve(
        pd.DatetimeIndex(["2020-03-02"]), np.array(["A"], dtype=object), np.array([0], np.int64)
    )
    np.testing.assert_allclose(out[0, 0], 10.0)


def test_vintaged_release_is_not_visible_before_its_stamp() -> None:
    # Vintaged mode: ref Feb-29 released Mar-25; decision Mar-1 — the release has not happened yet.
    panel = pd.DataFrame(
        {
            "ref_date": pd.to_datetime(["2020-02-29"]),
            "vintage": pd.to_datetime(["2020-03-25"]),
            "asset": ["A"],
            "x": [7.0],
        }
    )
    blk = CharBlock(panel, ["x"], vintage_col="vintage", lag=0)
    out = blk.resolve(
        pd.DatetimeIndex(["2020-03-01"]), np.array(["A"], dtype=object), np.array([0], np.int64)
    )
    assert np.isnan(out[0, 0])


def test_vintaged_mode_rejects_nonzero_lag() -> None:
    # Buffers belong in the vintage timestamps, not a row-step lag; vintaged mode forbids lag != 0.
    with pytest.raises(ValueError, match="vintaged mode takes no lag"):
        CharBlock(_vint(), ["macro"], vintage_col="vintage", lag=1)
