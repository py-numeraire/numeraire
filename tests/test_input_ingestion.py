"""Input-ingestion contract: how data enters the views/blocks, and what is validated at the door.

The ingestion boundary is where PIT correctness starts and where user mistakes surface, so every
constructor's validation + normalization is pinned here: TimeSeriesView, FeatureBlock,
VintagedBlock, CrossSectionView.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from numeraire.core.data import (
    CrossSectionView,
    FeatureBlock,
    TimeSeriesView,
    VintagedBlock,
)


def _idx(n: int, start: str = "2000-01-31") -> pd.DatetimeIndex:
    return pd.date_range(start, periods=n, freq="ME")


# --- TimeSeriesView returns block ----------------------------------------------------------------


def test_returns_index_must_be_datetime() -> None:
    df = pd.DataFrame({"r": [0.0, 0.1, 0.2]}, index=[0, 1, 2])
    with pytest.raises(TypeError, match="DatetimeIndex"):
        TimeSeriesView(df, df)


def test_returns_index_must_be_sorted() -> None:
    idx = _idx(3)[::-1]  # descending
    df = pd.DataFrame({"r": [0.0, 0.1, 0.2]}, index=idx)
    with pytest.raises(ValueError, match="sorted and unique"):
        TimeSeriesView(df, df)


def test_returns_index_must_be_unique() -> None:
    idx = pd.DatetimeIndex([_idx(1)[0]] * 2)
    df = pd.DataFrame({"r": [0.0, 0.1]}, index=idx)
    with pytest.raises(ValueError, match="sorted and unique"):
        TimeSeriesView(df, df)


def test_multi_asset_returns_ingest() -> None:
    idx = _idx(6)
    returns = pd.DataFrame({"a": np.zeros(6), "b": np.ones(6) * 0.01}, index=idx)
    feats = pd.DataFrame({"x": np.arange(6.0)}, index=idx)
    v = TimeSeriesView(returns, feats, horizon=1)
    assert v.assets == ["a", "b"]
    _dates, x, y = v.aligned()
    assert y.shape[1] == 2  # one target column per asset
    assert x.shape[1] == 1


# --- excess conversion ---------------------------------------------------------------------------


def test_excess_method_must_be_valid() -> None:
    idx = _idx(3)
    df = pd.DataFrame({"r": [0.0, 0.0, 0.0]}, index=idx)
    rf = pd.Series([0.0, 0.0, 0.0], index=idx)
    with pytest.raises(ValueError, match="must be 'simple' or 'log'"):
        TimeSeriesView(df, df, risk_free=rf, excess="geometric")


def test_excess_log_values_are_exact() -> None:
    idx = _idx(2)
    raw = pd.DataFrame({"r": [0.05, 0.03]}, index=idx)
    feats = pd.DataFrame({"x": [1.0, 2.0]}, index=idx)
    rf = pd.Series([0.01, 0.02], index=idx)
    v = TimeSeriesView(raw, feats, risk_free=rf, excess="log")
    expected = np.log1p([0.05, 0.03]) - np.log1p([0.01, 0.02])
    np.testing.assert_allclose(v.returns_frame()["r"].to_numpy(), expected)


def test_risk_free_broadcasts_across_assets() -> None:
    idx = _idx(3)
    raw = pd.DataFrame({"a": [0.05, 0.03, 0.07], "b": [0.02, 0.02, 0.02]}, index=idx)
    rf = pd.Series([0.01, 0.01, 0.02], index=idx)
    v = TimeSeriesView(raw, raw, risk_free=rf)  # rf subtracted from every asset column
    np.testing.assert_allclose(v.returns_frame()["a"].to_numpy(), [0.04, 0.02, 0.05])
    np.testing.assert_allclose(v.returns_frame()["b"].to_numpy(), [0.01, 0.01, 0.00])


# --- FeatureBlock --------------------------------------------------------------------------------


def test_feature_block_rejects_negative_lag() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0]}, index=_idx(2))
    with pytest.raises(ValueError, match="lag must be >= 0"):
        FeatureBlock(df, lag=-1)


def test_feature_block_index_must_be_datetime() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0]}, index=[0, 1])
    with pytest.raises(TypeError, match="DatetimeIndex"):
        FeatureBlock(df)


def test_feature_block_index_must_be_sorted_unique() -> None:
    df = pd.DataFrame({"x": [1.0, 2.0]}, index=_idx(2)[::-1])
    with pytest.raises(ValueError, match="sorted and unique"):
        FeatureBlock(df)


# --- VintagedBlock -------------------------------------------------------------------------------


def _vintage_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ref_date": pd.to_datetime(["2020-01-31", "2020-02-29"]),
            "vintage": pd.to_datetime(["2020-02-29", "2020-03-31"]),
            "x": [10.0, 20.0],
            "y": [1.0, 2.0],
        }
    )


def test_vintaged_block_rejects_negative_lag() -> None:
    with pytest.raises(ValueError, match="lag must be >= 0"):
        VintagedBlock(_vintage_table(), lag=-1)


def test_vintaged_block_series_subset_selects_columns() -> None:
    b = VintagedBlock(_vintage_table(), series=["x"])
    assert b.names == ["x"]
    np.testing.assert_array_equal(b.asof("2020-03-15"), [10.0])  # only Feb vintage -> Jan edge


def test_vintaged_block_custom_column_names() -> None:
    tbl = _vintage_table().rename(columns={"ref_date": "period", "vintage": "release"})
    b = VintagedBlock(tbl, ref_col="period", vintage_col="release")
    assert b.names == ["x", "y"]
    np.testing.assert_array_equal(b.asof("2020-03-15"), [10.0, 1.0])


# --- CrossSectionView ----------------------------------------------------------------------------


def _panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": pd.to_datetime(["2000-01-31", "2000-01-31", "2000-02-29"]),
            "asset": ["AAA", "BBB", "AAA"],
            "size": [1.0, 2.0, 1.1],
            "ret": [0.01, 0.02, 0.03],
        }
    )


def test_panel_horizon_zero_rejected() -> None:
    with pytest.raises(ValueError, match="contemporaneous"):
        CrossSectionView(_panel(), chars=["size"], horizon=0)


def test_panel_rejects_duplicate_observations() -> None:
    dup = pd.concat([_panel(), _panel().iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        CrossSectionView(dup, chars=["size"])


def test_panel_custom_column_names() -> None:
    tbl = _panel().rename(columns={"date": "d", "asset": "ticker", "ret": "r"})
    v = CrossSectionView(tbl, chars=["size"], date_col="d", asset_col="ticker", ret="r")
    assert v.char_names == ["size"]
    assert sorted(v.universe(pd.Timestamp("2000-01-31"))) == ["AAA", "BBB"]


def test_panel_chars_subset_ignores_other_columns() -> None:
    tbl = _panel().assign(bm=[0.5, 0.6, 0.7], mom=[0.0, 0.1, 0.2])
    v = CrossSectionView(tbl, chars=["size"])  # bm/mom present but not selected
    assert v.char_names == ["size"]
    _ids, x = v.features_asof(pd.Timestamp("2000-01-31"))
    assert x.shape == (2, 1)


def test_panel_ingests_unsorted_rows() -> None:
    shuffled = _panel().iloc[[2, 0, 1]].reset_index(drop=True)  # out of (date, asset) order
    v = CrossSectionView(shuffled, chars=["size"])
    ids, x = v.features_asof(pd.Timestamp("2000-01-31"))
    assert list(ids) == ["AAA", "BBB"]  # sorted at the door
    np.testing.assert_array_equal(x.ravel(), [1.0, 2.0])


def test_panel_parses_string_dates() -> None:
    tbl = _panel().assign(date=["2000-01-31", "2000-01-31", "2000-02-29"])
    v = CrossSectionView(tbl, chars=["size"])
    assert isinstance(v.calendar, pd.DatetimeIndex)
    assert len(v.calendar) == 2
