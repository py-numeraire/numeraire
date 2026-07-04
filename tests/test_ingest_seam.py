"""The frame-ingestion seam: non-pandas frames normalize to pandas at the view constructors' door.

Covers ``numeraire.core._ingest.to_pandas`` directly (pandas passthrough, duck-type ``.to_pandas()``
fallback, error paths) and its wiring into ``TimeSeriesView`` / ``CrossSectionView`` — a frame
handed in via a non-pandas container produces a view identical to the pandas path. The polars
round-trip runs only when polars is installed (an optional, non-hard dependency).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from numeraire.core._ingest import to_pandas
from numeraire.core.data import CrossSectionView, TimeSeriesView


class _DuckFrame:
    """A minimal non-pandas frame that only exposes ``.to_pandas()`` (narwhals declines it)."""

    def __init__(self, df: pd.DataFrame) -> None:
        self._df = df

    def to_pandas(self) -> pd.DataFrame:
        return self._df


# --------------------------------------------------------------------------------- to_pandas unit


def test_pandas_passthrough_is_identity() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0]})
    s = pd.Series([1.0, 2.0])
    assert to_pandas(df) is df  # no copy, index preserved
    assert to_pandas(s) is s


def test_duck_type_to_pandas_fallback() -> None:
    df = pd.DataFrame({"a": [1.0, 2.0]})
    out = to_pandas(_DuckFrame(df))
    assert out is df  # narwhals declines the duck frame; .to_pandas() fallback returns it


def test_unsupported_object_raises() -> None:
    with pytest.raises(TypeError, match="cannot ingest returns"):
        to_pandas(object(), what="returns")


def test_to_pandas_returning_non_frame_raises() -> None:
    class _Bad:
        def to_pandas(self) -> object:
            return object()

    with pytest.raises(TypeError, match="cannot ingest frame"):
        to_pandas(_Bad())


# --------------------------------------------------------------------------------- view wiring


def _pandas_ts() -> pd.DataFrame:
    idx = pd.date_range("2000-01-31", periods=24, freq="ME")
    rng = np.random.default_rng(0)
    return pd.DataFrame(rng.normal(0.01, 0.03, (24, 3)), index=idx, columns=["a", "b", "c"])


def test_timeseriesview_accepts_duck_frame() -> None:
    df = _pandas_ts()
    v_pd = TimeSeriesView(df, horizon=1)
    v_duck = TimeSeriesView(_DuckFrame(df), horizon=1)
    assert v_duck.assets == v_pd.assets
    assert v_duck.calendar.equals(v_pd.calendar)
    pd.testing.assert_frame_equal(v_duck.returns_frame(), v_pd.returns_frame())


def _pandas_panel() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows: list[tuple[object, ...]] = []
    for d in pd.date_range("2000-01-31", periods=8, freq="ME"):
        for a in ("s1", "s2", "s3"):
            rows.append((d, a, rng.normal(), rng.normal(0.0, 0.03)))
    return pd.DataFrame(rows, columns=["date", "asset", "c0", "ret"])


def test_crosssectionview_accepts_duck_frame() -> None:
    panel = _pandas_panel()
    v_pd = CrossSectionView(panel, chars=["c0"], horizon=1)
    v_duck = CrossSectionView(_DuckFrame(panel), chars=["c0"], horizon=1)
    assert v_duck.assets == v_pd.assets
    assert v_duck.calendar.equals(v_pd.calendar)


def test_polars_panel_round_trip() -> None:
    pl = pytest.importorskip("polars")  # optional; skips when polars is absent
    panel = _pandas_panel()
    v_pd = CrossSectionView(panel, chars=["c0"], horizon=1)
    v_pl = CrossSectionView(pl.from_pandas(panel), chars=["c0"], horizon=1)
    assert v_pl.assets == v_pd.assets
    assert v_pl.calendar.equals(v_pd.calendar)
