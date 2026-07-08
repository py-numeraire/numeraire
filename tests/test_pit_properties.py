"""Property-based no-look-ahead coverage for the timestamp-``asof`` availability layer.

The availability rule is unit-free: a reference/vintage/release stamp is visible at decision time
``t`` exactly when ``stamp <= t``, and per reference date the winning value is the one carried by
the latest vintage that is itself available (``vintage <= t``). These properties pin that contract
against an **independent brute-force oracle** written here from the documented behaviour (not from
the block internals), over irregular calendars, long publication lags, intra-month stamps and
multiple vintages per reference.

Both concrete carriers are covered: :class:`~numeraire.core.data.VintagedBlock` (a shared-series
time-series panel) and :class:`~numeraire.core.data.CharBlock` in its per-asset vintaged mode.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pandas as pd
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from numeraire.core.data import CharBlock, VintagedBlock

_BASE = pd.Timestamp("2000-01-01")

# --- record model + independent oracle -----------------------------------------------------------

# A record is (ref_offset_days, vintage_lag_days, value): the reference date is ``_BASE + ref``,
# and its vintage (release) stamp is ``ref_date + lag`` — so a lag can be zero (period-end-known),
# a few days (intra-month), or hundreds of days (a long publication delay).
Record = tuple[int, int, float]


def _ref_ts(rec: Record) -> pd.Timestamp:
    return _BASE + pd.Timedelta(days=rec[0])


def _vint_ts(rec: Record) -> pd.Timestamp:
    return _BASE + pd.Timedelta(days=rec[0] + rec[1])


def _oracle_asof(records: list[Record], t: pd.Timestamp) -> float | None:
    """Winning value visible at ``t``, or ``None`` if nothing is released yet (from the contract).

    Visible set = records whose vintage stamp is ``<= t``. Among them the real-time edge is the
    latest reference date; within that reference date it is the latest available vintage. Encodes
    the documented rule directly, with no reference to how the block computes it.
    """
    visible = [r for r in records if _vint_ts(r) <= t]
    if not visible:
        return None
    edge_ref = max(_ref_ts(r) for r in visible)
    at_edge = [r for r in visible if _ref_ts(r) == edge_ref]
    winner = max(at_edge, key=_vint_ts)  # latest vintage for that reference date
    return winner[2]


@st.composite
def _record_sets(draw: st.DrawFn) -> list[Record]:
    """A valid record set: unique reference dates, 1-3 unique-lag vintages each, finite values.

    Reference offsets are drawn from a wide range with mixed spacings (so the implied calendar is
    irregular, not month-end-regular); lags span 0 (same-day) to ~500 days (long lag).
    """
    n_refs = draw(st.integers(min_value=1, max_value=6))
    ref_offsets = draw(
        st.lists(
            st.integers(min_value=0, max_value=900),
            min_size=n_refs,
            max_size=n_refs,
            unique=True,
        )
    )
    records: list[Record] = []
    for ref in sorted(ref_offsets):
        n_vint = draw(st.integers(min_value=1, max_value=3))
        lags = draw(
            st.lists(
                st.integers(min_value=0, max_value=500),
                min_size=n_vint,
                max_size=n_vint,
                unique=True,  # unique lags → unique (ref, vintage) pairs within a ref group
            )
        )
        for lag in lags:
            value = draw(
                st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
            )
            records.append((ref, lag, value))
    return records


def _query_times(records: list[Record]) -> list[pd.Timestamp]:
    """A spread of decision times: every stamp, one day either side, midpoints, and out-of-range."""
    stamps = sorted({_vint_ts(r) for r in records} | {_ref_ts(r) for r in records})
    times: set[pd.Timestamp] = set()
    for s in stamps:
        times.add(s)
        times.add(s - pd.Timedelta(days=1))
        times.add(s + pd.Timedelta(days=1))
    for a, b in pairwise(stamps):
        times.add(a + (b - a) / 2)  # a midpoint strictly between two events
    times.add(stamps[0] - pd.Timedelta(days=30))  # before the first release
    times.add(stamps[-1] + pd.Timedelta(days=30))  # after the last release
    return sorted(times)


def _vintaged_table(records: list[Record]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ref_date": [_ref_ts(r) for r in records],
            "vintage": [_vint_ts(r) for r in records],
            "v": [r[2] for r in records],
        }
    )


# --- VintagedBlock (shared-series) ---------------------------------------------------------------


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(records=_record_sets())
def test_vintaged_block_matches_oracle(records: list[Record]) -> None:
    block = VintagedBlock(_vintaged_table(records))
    for t in _query_times(records):
        expected = _oracle_asof(records, t)
        if expected is None:
            assert not block.is_ready(t)
            with pytest.raises(KeyError):
                block.asof(t)
        else:
            assert block.is_ready(t)
            # exact equality: PIT resolution transports a stored value, it does no arithmetic
            assert float(block.asof(t)[0]) == expected


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(records=_record_sets())
def test_vintaged_block_publication_is_monotone(records: list[Record]) -> None:
    """Moving the decision time forward never un-publishes data (the visible set only grows)."""
    block = VintagedBlock(_vintaged_table(records))
    times = _query_times(records)
    seen_ready = False
    prev_visible = -1
    for t in times:
        ready = block.is_ready(t)
        if seen_ready:
            assert ready, "a release visible earlier became invisible at a later decision time"
        seen_ready = seen_ready or ready
        visible = sum(1 for r in records if _vint_ts(r) <= t)
        assert visible >= prev_visible
        prev_visible = visible


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    ref_off=st.integers(min_value=0, max_value=900),
    intra=st.integers(min_value=1, max_value=27),  # release later in the SAME month as t
    val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_vintaged_block_intra_period_release_not_visible_early(
    ref_off: int, intra: int, val: float
) -> None:
    """A release stamped later in ``t``'s own month is invisible at ``t`` (the killed leak class).

    Under the old month-ordinal comparison a same-month-but-later vintage counted as available;
    the timestamp rule must keep it dead.
    """
    ref = pd.Timestamp("2000-06-01") + pd.Timedelta(days=ref_off)
    t = pd.Timestamp(year=ref.year, month=ref.month, day=1)  # start of the reference month
    vintage = t + pd.Timedelta(days=intra)  # released later in the same month, strictly after t
    table = pd.DataFrame({"ref_date": [ref], "vintage": [vintage], "v": [val]})
    block = VintagedBlock(table)
    assert not block.is_ready(t)
    with pytest.raises(KeyError):
        block.asof(t)


# --- CharBlock (per-asset vintaged mode) ---------------------------------------------------------


def _char_panel(records_by_asset: dict[str, list[Record]]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for asset, records in records_by_asset.items():
        for r in records:
            rows.append(
                {
                    "ref_date": _ref_ts(r),
                    "asset": asset,
                    "vintage": _vint_ts(r),
                    "x": r[2],
                }
            )
    return pd.DataFrame(rows)


@st.composite
def _two_asset_records(draw: st.DrawFn) -> dict[str, list[Record]]:
    return {"A": draw(_record_sets()), "B": draw(_record_sets())}


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(records_by_asset=_two_asset_records())
def test_char_block_vintaged_matches_oracle_per_asset(
    records_by_asset: dict[str, list[Record]],
) -> None:
    """Vintaged CharBlock resolves each asset independently against the same per-asset oracle."""
    block = CharBlock(_char_panel(records_by_asset), ["x"], vintage_col="vintage")
    all_records = [r for recs in records_by_asset.values() for r in recs]
    for t in _query_times(all_records):
        for asset, records in records_by_asset.items():
            out = block.resolve(
                pd.DatetimeIndex([t]), np.array([asset], dtype=object), np.array([0], np.int64)
            )
            got = float(out[0, 0])
            expected = _oracle_asof(records, t)
            if expected is None:
                assert np.isnan(got), f"{asset} at {t}: expected nothing available, got {got}"
            else:
                # exact equality: PIT resolution transports a stored value, no arithmetic
                assert got == expected


@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(
    ref_off=st.integers(min_value=0, max_value=900),
    intra=st.integers(min_value=1, max_value=27),
    val=st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
)
def test_char_block_vintaged_intra_period_release_not_visible_early(
    ref_off: int, intra: int, val: float
) -> None:
    """Per-asset analog: a same-month-but-later release stays invisible at ``t`` (leak dead)."""
    ref = pd.Timestamp("2000-06-01") + pd.Timedelta(days=ref_off)
    t = pd.Timestamp(year=ref.year, month=ref.month, day=1)
    vintage = t + pd.Timedelta(days=intra)
    panel = pd.DataFrame({"ref_date": [ref], "asset": ["A"], "vintage": [vintage], "x": [val]})
    block = CharBlock(panel, ["x"], vintage_col="vintage")
    out = block.resolve(
        pd.DatetimeIndex([t]), np.array(["A"], dtype=object), np.array([0], np.int64)
    )
    assert np.isnan(out[0, 0])


# --- invalid inputs always raise (the documented rejections) -------------------------------------


@settings(max_examples=60, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(records=_record_sets())
def test_duplicate_ref_vintage_always_rejected(records: list[Record]) -> None:
    """Duplicating any ``(ref_date, vintage)`` pair (order-dependent edge) must raise."""
    table = _vintaged_table(records)
    dup = pd.concat([table, table.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        VintagedBlock(dup)


def test_nat_vintage_always_rejected() -> None:
    table = _vintaged_table([(0, 5, 1.0), (30, 10, 2.0)])
    table.loc[0, "vintage"] = pd.NaT
    with pytest.raises(ValueError, match="NaT"):
        VintagedBlock(table)


def test_tz_aware_vintage_always_rejected() -> None:
    table = _vintaged_table([(0, 5, 1.0), (30, 10, 2.0)])
    table["vintage"] = table["vintage"].dt.tz_localize("UTC")
    with pytest.raises(TypeError, match="tz-naive"):
        VintagedBlock(table)
