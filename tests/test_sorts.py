"""The portfolio-sort constructor: binning, value/equal weighting, NYSE breakpoints, long-short.

Synthetic, mechanical checks — a panel with a known signal produces known bin assignments and
weighted returns; the NYSE-breakpoint path is checked to move the cutoffs (so small non-NYSE names
don't set them). The French/WRDS decile cross-check rides the credentialed reference tier (W3-4).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from numeraire.core.sorts import make_sorts


def _wide(rows: dict[str, list[float]], dates) -> pd.DataFrame:
    return pd.DataFrame(rows, index=dates)


def test_two_bin_split_and_equal_weight_returns() -> None:
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    # signal 1,2,3,4 -> median split: {1,2} low, {3,4} high
    signal = _wide({"a": [1.0], "b": [2.0], "c": [3.0], "d": [4.0]}, dates)
    returns = _wide({"a": [0.01], "b": [0.03], "c": [0.05], "d": [0.09]}, dates)
    res = make_sorts(signal, returns, n_bins=2)
    assert res.portfolios.loc[dates[0], 0] == np.mean([0.01, 0.03])
    assert res.portfolios.loc[dates[0], 1] == np.mean([0.05, 0.09])
    assert res.long_short.iloc[0] == np.mean([0.05, 0.09]) - np.mean([0.01, 0.03])
    assert list(res.counts.loc[dates[0]]) == [2, 2]


def test_value_weighting_uses_weights() -> None:
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    signal = _wide({"a": [1.0], "b": [2.0], "c": [3.0], "d": [4.0]}, dates)
    returns = _wide({"a": [0.0], "b": [0.10], "c": [0.0], "d": [0.10]}, dates)
    weights = _wide({"a": [1.0], "b": [9.0], "c": [1.0], "d": [9.0]}, dates)
    res = make_sorts(signal, returns, n_bins=2, weights=weights)
    # low bin {a,b}: (1*0 + 9*0.10)/10 = 0.09
    assert res.portfolios.loc[dates[0], 0] == 0.09


def test_direction_flips_long_short_sign() -> None:
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    signal = _wide({"a": [1.0], "b": [2.0], "c": [3.0], "d": [4.0]}, dates)
    returns = _wide({"a": [0.01], "b": [0.02], "c": [0.03], "d": [0.10]}, dates)
    up = make_sorts(signal, returns, n_bins=2, direction=1).long_short.iloc[0]
    down = make_sorts(signal, returns, n_bins=2, direction=-1).long_short.iloc[0]
    assert up == -down


def test_nyse_breakpoints_shift_cutoffs() -> None:
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    # one big NYSE name at 100 and many small non-NYSE names at 1..4: all-name median ~ 2.5, but
    # NYSE-only breakpoints are set by the single NYSE name -> all small names land in the low bin.
    cols = ["nyse", "s1", "s2", "s3", "s4"]
    signal = _wide({c: [v] for c, v in zip(cols, [100.0, 1.0, 2.0, 3.0, 4.0], strict=True)}, dates)
    returns = _wide({c: [0.0] for c in cols}, dates)
    universe = _wide({c: [c == "nyse"] for c in cols}, dates)  # only 'nyse' defines the breakpoints
    res = make_sorts(signal, returns, n_bins=2, breakpoint_universe=universe)
    # with the cutoff at ~100, the 4 small names are all in the low bin, nyse alone in the high bin
    assert list(res.counts.loc[dates[0]]) == [4, 1]


def test_thin_universe_falls_back_to_all_names() -> None:
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    cols = ["a", "b", "c", "d"]
    signal = _wide({c: [v] for c, v in zip(cols, [1.0, 2.0, 3.0, 4.0], strict=True)}, dates)
    returns = _wide({c: [0.01 * i] for i, c in enumerate(cols)}, dates)
    universe = _wide({c: [False] for c in cols}, dates)  # empty universe -> fall back
    res = make_sorts(signal, returns, n_bins=2, breakpoint_universe=universe)
    assert list(res.counts.loc[dates[0]]) == [2, 2]  # all-name median split still happens


def test_insufficient_names_skips_period() -> None:
    dates = pd.date_range("2000-01-31", periods=2, freq="ME")
    signal = _wide({"a": [1.0, 1.0], "b": [np.nan, 2.0]}, dates)
    returns = _wide({"a": [0.01, 0.01], "b": [0.02, 0.02]}, dates)
    res = make_sorts(signal, returns, n_bins=2)
    assert np.isnan(res.long_short.iloc[0])  # only 1 valid name in period 0
    assert np.isfinite(res.long_short.iloc[1])


def test_validation_guards() -> None:
    dates = pd.date_range("2000-01-31", periods=1, freq="ME")
    s = _wide({"a": [1.0], "b": [2.0]}, dates)
    import pytest

    with pytest.raises(ValueError, match="n_bins must be >= 2"):
        make_sorts(s, s, n_bins=1)
    with pytest.raises(ValueError, match="direction must be"):
        make_sorts(s, s, n_bins=2, direction=0)
    with pytest.raises(ValueError, match="aligned on the same"):
        make_sorts(s, s.rename(columns={"a": "z"}), n_bins=2)
