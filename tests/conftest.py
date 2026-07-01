"""Shared synthetic-data helpers (public/synthetic only — no WRDS, SPEC §6).

Two flavours live here:

- ``load_gw_annual_view`` reads the (git-excluded, audit-pending) public GW slice — used by the
  1/A golden, which therefore *skips* in CI until the fixture is committed.
- The **toy-data catalog** (``toy_*``) is in-code and deterministic (explicit RNG per landmine #4),
  so combination tests over it always run in CI without any committed file. It is a small reusable
  set of *data shapes* — single-asset raw+rf, multi-asset excess, aggregate predictors, a
  publication-lagged macro block, and a vintaged ``(ref_date, vintage)`` panel — that the view can
  be mounted from in any combination.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from numeraire.core.data import FeatureBlock, TimeSeriesView, VintagedBlock

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


# --- toy-data catalog (in-code, deterministic; reusable data shapes) ------------------------------

TOY_N = 72  # 2000-01 .. 2005-12, monthly


def toy_index(n: int = TOY_N) -> pd.DatetimeIndex:
    """The shared monthly decision calendar for the toy catalog (month-end)."""
    return pd.date_range("2000-01-31", periods=n, freq="ME")


def toy_market(n: int = TOY_N, seed: int = 11) -> tuple[pd.DataFrame, pd.Series]:
    """Single-asset **raw** market return + a matching risk-free series (for excess timing)."""
    rng = np.random.default_rng(seed)
    idx = toy_index(n)
    mkt = pd.DataFrame({"mkt": rng.normal(0.006, 0.045, n)}, index=idx)
    rf = pd.Series(np.abs(rng.normal(0.002, 0.0006, n)), index=idx, name="rf")
    return mkt, rf


def toy_assets(
    n: int = TOY_N, seed: int = 12, cols: tuple[str, ...] = ("size", "value", "mom")
) -> pd.DataFrame:
    """Multi-asset **excess** returns ``(date x asset)`` — a small factor-like panel."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.004, 0.05, size=(n, len(cols))), index=toy_index(n), columns=list(cols)
    )


def toy_predictors(
    n: int = TOY_N, seed: int = 13, cols: tuple[str, ...] = ("dp", "tbl")
) -> pd.DataFrame:
    """Aggregate time-series predictors sharing the toy calendar (mount as a ``lag=0`` block)."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        rng.normal(0.0, 1.0, size=(n, len(cols))), index=toy_index(n), columns=list(cols)
    )


def toy_macro_block(
    n: int = TOY_N, seed: int = 14, lag: int = 2, name: str = "macro"
) -> FeatureBlock:
    """A slow-moving, publication-lagged macro level (no vintage) as a ``lag>=1`` FeatureBlock."""
    rng = np.random.default_rng(seed)
    cpi = 100.0 + np.cumsum(rng.normal(0.2, 0.1, n))
    return FeatureBlock(pd.DataFrame({"cpi": cpi}, index=toy_index(n)), lag=lag, name=name)


def toy_vintaged_table(n_refs: int = 24, seed: int = 15) -> pd.DataFrame:
    """A tidy ``[ref_date, vintage, INDPRO, UNRATE]`` panel: ref released next month, then revised.

    Shape mirrors a FRED-MD build: first release at ``ref+1`` month, a small revision at ``ref+2``.
    """
    rng = np.random.default_rng(seed)
    refs = pd.date_range("2000-01-31", periods=n_refs, freq="ME")
    rows: list[tuple[pd.Timestamp, pd.Timestamp, float, float]] = []
    for i, ref in enumerate(refs):
        indpro = 100.0 + i * 0.3 + float(rng.normal(0.0, 0.1))
        unrate = 4.0 + 0.5 * float(np.sin(i / 3.0))
        v1 = ref + pd.offsets.MonthEnd(1)  # first release
        rows.append((ref, v1, round(indpro, 3), round(unrate, 2)))
        v2 = ref + pd.offsets.MonthEnd(2)  # revision
        rows.append((ref, v2, round(indpro + float(rng.normal(0.0, 0.05)), 3), round(unrate, 2)))
    return pd.DataFrame(rows, columns=["ref_date", "vintage", "INDPRO", "UNRATE"])


def toy_vintaged_block(lag: int = 1, name: str = "fred") -> VintagedBlock:
    """The vintaged panel wrapped as a :class:`VintagedBlock` (asof by ``vintage + lag``)."""
    return VintagedBlock(toy_vintaged_table(), lag=lag, name=name)


# presence windows (global-month indices) → a ragged universe with entry, exit, and a late arrival
TOY_PANEL_PRESENCE: dict[str, range] = {
    "AAA": range(0, 8),  # full history
    "BBB": range(2, 8),  # enters at month 2
    "CCC": range(0, 5),  # delists after month 4
    "DDD": range(3, 8),  # late arrival
}


def toy_panel(seed: int = 21) -> pd.DataFrame:
    """A tidy, **ragged** individual-stock panel ``[date, asset, size, bm, mom, ret]`` (8 months).

    Universe enters/exits (see ``TOY_PANEL_PRESENCE``) and one characteristic cell is missing
    (``BBB`` @ month 2 has ``NaN`` ``bm``) to exercise delisting targets + missing-char handling.
    """
    rng = np.random.default_rng(seed)
    cal = pd.date_range("2000-01-31", periods=8, freq="ME")
    rows: list[dict[str, object]] = []
    for asset, months in TOY_PANEL_PRESENCE.items():
        for m in months:
            rows.append(
                {
                    "date": cal[m],
                    "asset": asset,
                    "size": round(float(rng.normal(5.0, 1.0)), 3),
                    "bm": round(float(rng.normal(0.5, 0.2)), 3),
                    "mom": round(float(rng.normal(0.0, 0.1)), 3),
                    "ret": round(float(rng.normal(0.01, 0.05)), 4),
                }
            )
    df = pd.DataFrame(rows)
    df.loc[(df["asset"] == "BBB") & (df["date"] == cal[2]), "bm"] = np.nan
    return df
