"""CrossSectionView: ragged individual-stock panel — cross-section access, forward returns, PIT.

The sibling of the time-series tests: here predictors vary by ``(date, asset)`` and the universe
enters/exits, so the invariants are cross-section-shaped — ``features_asof`` returns a matrix (not a
vector), delisting yields ``nan`` targets, and ``aligned`` stacks a panel design matrix.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from conftest import toy_panel
from numeraire.core.data import CrossSectionView
from numeraire.core.protocols import DataView

CAL = pd.date_range("2000-01-31", periods=8, freq="ME")
CHARS = ["size", "bm", "mom"]


def _view(horizon: int = 1) -> CrossSectionView:
    return CrossSectionView(toy_panel(), chars=CHARS, horizon=horizon)


def test_is_dataview() -> None:
    assert isinstance(_view(), DataView)


def test_features_asof_returns_cross_section_matrix() -> None:
    v = _view()
    ids, x = v.features_asof(CAL[0])  # month 0: AAA, CCC alive
    assert sorted(ids) == ["AAA", "CCC"]
    assert x.shape == (2, 3)  # (n_alive x K), a matrix not a vector


def test_universe_is_ragged() -> None:
    v = _view()
    assert v.universe(CAL[0]) == ["AAA", "CCC"]  # before BBB/DDD arrive
    assert sorted(v.universe(CAL[3])) == ["AAA", "BBB", "CCC", "DDD"]  # all four present
    assert sorted(v.universe(CAL[5])) == ["AAA", "BBB", "DDD"]  # CCC has delisted
    assert v.assets == ["AAA", "BBB", "CCC", "DDD"]  # union axis


def test_target_asof_is_nan_on_delisting() -> None:
    v = _view()
    ids, y = v.target_asof(CAL[4])  # CCC present at m4 but gone at m5 -> nan
    row = {a: y[i] for i, a in enumerate(ids)}
    assert np.isnan(row["CCC"])  # delisted before the horizon closes
    assert np.isfinite(row["AAA"])  # AAA survives -> realized forward return


def test_target_asof_h1_matches_next_return() -> None:
    df = toy_panel()
    v = _view()
    ids, y = v.target_asof(CAL[0], horizon=1)
    got = {a: y[i] for i, a in enumerate(ids)}
    aaa_next = float(df[(df["asset"] == "AAA") & (df["date"] == CAL[1])]["ret"].iloc[0])
    np.testing.assert_allclose(got["AAA"], aaa_next)


def test_target_asof_h2_compounds() -> None:
    df = toy_panel()
    v = _view(horizon=2)
    ids, y = v.target_asof(CAL[0])
    got = {a: y[i] for i, a in enumerate(ids)}
    r1 = float(df[(df["asset"] == "AAA") & (df["date"] == CAL[1])]["ret"].iloc[0])
    r2 = float(df[(df["asset"] == "AAA") & (df["date"] == CAL[2])]["ret"].iloc[0])
    np.testing.assert_allclose(got["AAA"], (1 + r1) * (1 + r2) - 1)


def test_aligned_stacks_panel_and_purges() -> None:
    v = _view()
    keys, x, y = v.aligned()
    assert list(keys.names) == ["date", "asset"]
    assert x.shape == (len(keys), 3)
    assert y.shape == (len(keys),)
    assert not np.isnan(y).any()  # unrealized / delisting targets purged
    assert not np.isnan(x).any()  # missing-char rows dropped
    # the injected NaN-char cell (BBB @ month 2) must not survive into the design matrix
    assert (CAL[2], "BBB") not in set(keys)


def test_window_is_pit() -> None:
    v = _view()
    w = v.window(CAL[3])
    assert w.calendar.max() == CAL[3]
    assert len(w.calendar) == 4
    # aligned on the window realizes no target past the cutoff (h=1 -> last feature date <= m2)
    keys, _, _ = w.aligned()
    assert keys.get_level_values("date").max() <= CAL[2]


def test_asof_is_invariant_to_future_data() -> None:
    """features_asof(t) cross-section must not change when future dates are truncated (no leak)."""
    v = _view()
    for t in (CAL[2], CAL[4], CAL[6]):
        ids_full, x_full = v.features_asof(t)
        ids_win, x_win = v.window(t).features_asof(t)
        np.testing.assert_array_equal(ids_full, ids_win)
        np.testing.assert_array_equal(x_full, x_win)


def test_panel_frame_round_trips_shape() -> None:
    v = _view()
    pf = v.panel_frame()
    assert list(pf.columns) == ["size", "bm", "mom", "ret"]
    assert list(pf.index.names) == ["date", "asset"]
    # every (date, asset) observation in the ragged panel is present (full view keeps all rows)
    assert len(pf) == len(toy_panel())


def test_to_tensor_shapes_and_mask() -> None:
    v = _view()
    tsr = v.to_tensor()
    T, N, K = len(CAL), 4, 3
    assert tsr.features.shape == (T, N, K)
    assert tsr.returns.shape == (T, N)
    assert tsr.mask.shape == (T, N)
    assert tsr.assets == ["AAA", "BBB", "CCC", "DDD"]
    # mask marks exactly the present observations; padding stays nan
    assert int(tsr.mask.sum()) == len(toy_panel())
    assert np.isnan(tsr.features[~tsr.mask]).all()
    assert np.isfinite(
        tsr.features[tsr.mask]
    ).any()  # present cells carry values (bar the 1 NaN char)


def test_to_tensor_matches_features_asof() -> None:
    v = _view()
    tsr = v.to_tensor()
    t_idx = 3  # month 3: all four assets present
    ids, x = v.features_asof(CAL[t_idx])
    for a, row in zip(ids, x, strict=True):
        j = tsr.assets.index(a)
        assert tsr.mask[t_idx, j]
        np.testing.assert_array_equal(tsr.features[t_idx, j], row)


def test_to_tensor_absent_cells_are_masked_off() -> None:
    v = _view()
    tsr = v.to_tensor()
    j_ddd = tsr.assets.index("DDD")  # DDD arrives at month 3
    assert not tsr.mask[0, j_ddd]
    assert np.isnan(tsr.returns[0, j_ddd])
    j_ccc = tsr.assets.index("CCC")  # CCC delists after month 4
    assert not tsr.mask[5, j_ccc]
