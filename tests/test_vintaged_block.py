"""VintagedBlock: real-time asof over a (ref_date, vintage) panel — revisions + no look-ahead.

Availability is a plain ``vintage <= t`` timestamp comparison (no month lag): a release stamped on
a given day becomes visible on that day and not before.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from numeraire.core.data import FeatureBlock, TimeSeriesView, VintagedBlock


def _table() -> pd.DataFrame:
    # Jan first seen in the Feb vintage (10), then revised to 11 in the Mar vintage; Feb seen in
    # Mar (20); Mar in Apr (30); Feb revised in Apr (21).
    return pd.DataFrame(
        {
            "ref_date": pd.to_datetime(
                ["2020-01-31", "2020-02-29", "2020-01-31", "2020-03-31", "2020-02-29"]
            ),
            "vintage": pd.to_datetime(
                ["2020-02-29", "2020-03-31", "2020-03-31", "2020-04-30", "2020-04-30"]
            ),
            "x": [10.0, 20.0, 11.0, 30.0, 21.0],
        }
    )


def test_asof_edge_advances_with_releases() -> None:
    b = VintagedBlock(_table(), name="fred")
    assert b.names == ["x"]
    np.testing.assert_array_equal(b.asof("2020-03-15"), [10.0])  # only Feb vintage → edge Jan=10
    np.testing.assert_array_equal(b.asof("2020-04-15"), [20.0])  # +Mar vintage → edge Feb=20
    np.testing.assert_array_equal(b.asof("2020-05-15"), [30.0])  # +Apr vintage → edge Mar=30


def test_no_lookahead_on_release_and_revision() -> None:
    b = VintagedBlock(_table())
    # In March, Feb's value (20, only in the Mar vintage) isn't visible; nor the Jan revision to 11.
    assert b.asof("2020-03-15")[0] == 10.0
    assert b.asof("2020-04-15")[0] == 20.0  # edge is Feb; the revised Jan (11) never leaks in early


def test_warmup_not_ready() -> None:
    b = VintagedBlock(_table())
    assert b.is_ready("2020-01-15") is False
    with pytest.raises(KeyError):
        b.asof("2020-01-15")
    assert b.is_ready("2020-03-15") is True


def test_vintage_not_visible_before_its_stamp() -> None:
    # The first vintage is stamped 2020-02-29; under the timestamp rule it is NOT visible earlier
    # that month (the old month-ordinal resolution wrongly returned it any time within February).
    b = VintagedBlock(_table())
    assert b.is_ready("2020-02-15") is False
    with pytest.raises(KeyError):
        b.asof("2020-02-15")
    np.testing.assert_array_equal(b.asof("2020-02-29"), [10.0])  # visible exactly on its stamp


def test_truncate_drops_future_vintages() -> None:
    b = VintagedBlock(_table()).truncate("2020-03-31")  # drop the Apr vintage
    np.testing.assert_array_equal(
        b.asof("2020-05-15"), [20.0]
    )  # edge now Feb (Mar vintage), not Mar


def test_mixed_with_feature_block_in_view() -> None:
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    returns = pd.DataFrame({"mkt": np.zeros(6)}, index=idx)
    vint = VintagedBlock(_table(), name="fred")
    feat = FeatureBlock(pd.DataFrame({"z": np.arange(6.0)}, index=idx), lag=0, name="z")
    view = TimeSeriesView(returns, blocks=[vint, feat], horizon=1)
    assert view.feature_names == ["x", "z"]
    # at 2020-04-30 the Apr-30 vintage is available (same-day release), so the edge is the Mar ref
    np.testing.assert_array_equal(view.features_asof("2020-04-30"), [30.0, 3.0])


def test_nat_vintage_rejected() -> None:
    # A missing vintage stamp would cast to int64 minimum ("available since the beginning of time"),
    # a silent look-ahead: the row would count as ready at any decision date. Reject it instead.
    table = _table()
    table.loc[0, "vintage"] = pd.NaT
    with pytest.raises(ValueError, match="NaT"):
        VintagedBlock(table)


def test_nat_ref_rejected() -> None:
    table = _table()
    table.loc[0, "ref_date"] = pd.NaT
    with pytest.raises(ValueError, match="NaT"):
        VintagedBlock(table)


def test_tz_aware_vintage_rejected() -> None:
    # A tz-aware source compared against a tz-naive decision calendar shifts the availability
    # boundary by the offset; require the caller to normalize to tz-naive first.
    table = _table()
    table["vintage"] = table["vintage"].dt.tz_localize("UTC")
    with pytest.raises(TypeError, match="tz-naive"):
        VintagedBlock(table)


def test_tz_aware_ref_rejected() -> None:
    table = _table()
    table["ref_date"] = table["ref_date"].dt.tz_localize("UTC")
    with pytest.raises(TypeError, match="tz-naive"):
        VintagedBlock(table)


def test_duplicate_ref_vintage_rejected() -> None:
    # Two rows sharing (ref_date, vintage) make the real-time edge order-dependent; reject at build.
    table = pd.DataFrame(
        {
            "ref_date": pd.to_datetime(["2020-01-31", "2020-01-31"]),
            "vintage": pd.to_datetime(["2020-02-29", "2020-02-29"]),
            "x": [10.0, 11.0],
        }
    )
    with pytest.raises(ValueError, match="duplicate"):
        VintagedBlock(table)
