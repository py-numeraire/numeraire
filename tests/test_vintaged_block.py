"""VintagedBlock: real-time asof over a (ref_date, vintage) panel — revisions + no look-ahead."""

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


def test_asof_edge_advances_with_lag() -> None:
    b = VintagedBlock(_table(), lag=1, name="fred")
    assert b.names == ["x"]
    np.testing.assert_array_equal(b.asof("2020-03-15"), [10.0])  # only Feb vintage → edge Jan=10
    np.testing.assert_array_equal(b.asof("2020-04-15"), [20.0])  # +Mar vintage → edge Feb=20
    np.testing.assert_array_equal(b.asof("2020-05-15"), [30.0])  # +Apr vintage → edge Mar=30


def test_no_lookahead_on_release_and_revision() -> None:
    b = VintagedBlock(_table(), lag=1)
    # In March, Feb's value (20, only in the Mar vintage) isn't visible; nor the Jan revision to 11.
    assert b.asof("2020-03-15")[0] == 10.0
    assert b.asof("2020-04-15")[0] == 20.0  # edge is Feb; the revised Jan (11) never leaks in early


def test_warmup_not_ready() -> None:
    b = VintagedBlock(_table(), lag=1)
    assert b.is_ready("2020-01-15") is False
    with pytest.raises(KeyError):
        b.asof("2020-01-15")
    assert b.is_ready("2020-03-15") is True


def test_lag_zero_is_more_aggressive() -> None:
    b = VintagedBlock(_table(), lag=0)
    np.testing.assert_array_equal(b.asof("2020-02-15"), [10.0])  # Feb vintage usable within Feb


def test_truncate_drops_future_vintages() -> None:
    b = VintagedBlock(_table(), lag=1).truncate("2020-03-31")  # drop the Apr vintage
    np.testing.assert_array_equal(
        b.asof("2020-05-15"), [20.0]
    )  # edge now Feb (Mar vintage), not Mar


def test_mixed_with_feature_block_in_view() -> None:
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    returns = pd.DataFrame({"mkt": np.zeros(6)}, index=idx)
    vint = VintagedBlock(_table(), lag=1, name="fred")
    feat = FeatureBlock(pd.DataFrame({"z": np.arange(6.0)}, index=idx), lag=0, name="z")
    view = TimeSeriesView(returns, blocks=[vint, feat], horizon=1)
    assert view.feature_names == ["x", "z"]
    np.testing.assert_array_equal(view.features_asof("2020-04-30"), [20.0, 3.0])
