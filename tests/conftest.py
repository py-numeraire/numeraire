"""Shared synthetic-data helpers (public/synthetic only — no WRDS, SPEC §6)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from numeraire.core.data import TimeSeriesView

FIXTURES = Path(__file__).parent / "fixtures"


def load_gw_annual_view() -> TimeSeriesView:
    """Public Goyal-Welch annual view (excess return ``y`` + dividend-price ``dp``), 1872-2017.

    Derived from the public GW predictor file (redistributable); used for the 1/A dp golden test.
    """
    df = pd.read_csv(FIXTURES / "gw_annual_1a.csv")
    index = pd.DatetimeIndex(pd.to_datetime(df["yyyy"].astype(str) + "-12-31"))
    returns = pd.DataFrame({"mkt": df["y"].to_numpy()}, index=index)
    features = pd.DataFrame({"dp": df["dp"].to_numpy()}, index=index)
    return TimeSeriesView(returns, features, horizon=1)


def make_monthly_view(
    n: int = 240,
    n_assets: int = 1,
    n_features: int = 2,
    horizon: int = 1,
    seed: int = 0,
) -> TimeSeriesView:
    """A synthetic monthly :class:`TimeSeriesView` with a deterministic, explicit RNG."""
    rng = np.random.default_rng(seed)
    index = pd.date_range("1990-01-31", periods=n, freq="ME")
    returns = pd.DataFrame(
        rng.normal(0.005, 0.04, size=(n, n_assets)),
        index=index,
        columns=[f"r{i}" for i in range(n_assets)],
    )
    features = pd.DataFrame(
        rng.normal(0.0, 1.0, size=(n, n_features)),
        index=index,
        columns=[f"x{i}" for i in range(n_features)],
    )
    return TimeSeriesView(returns, features, horizon=horizon)
