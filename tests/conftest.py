"""Shared synthetic-data helpers (public/synthetic only — no WRDS)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from numeraire.core.data import TimeSeriesView


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
