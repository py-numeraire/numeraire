"""Concrete ``DataView`` implementations and the PIT / horizon machinery.

This module ships the first concrete view, :class:`TimeSeriesView`, covering the
market-timing / aggregate-predictor case (VoC, 1/A): a ``returns`` block ``(date x asset)``
with cardinality 1..N and a time-series ``features`` block ``(date x feature)`` sharing one
calendar. It implements the :class:`numeraire.core.protocols.DataView` protocol and adds the
explicit-horizon pairing the OOS engine relies on:

    ``features_asof(t)`` (info known <= t)  <->  ``target_asof(t, h)`` (realized over (t, t+h])

The pairing is **engine-owned** so a method never indexes returns itself — this makes the
SOF-style one-period contemporaneous leak structurally impossible.

Scope note (v0): features and returns are required to share one sorted calendar (the common
monthly-aligned case, e.g. Goyal-Welch). Block-level vintage with separate release dates
(``asof`` over revised sources, §4A) and the panel view crystallize with the next adapters.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Float = NDArray[np.float64]


def _as_2d(frame: pd.DataFrame) -> Float:
    """Return ``frame`` as a contiguous float64 array, raising on non-finite dtypes."""
    return np.ascontiguousarray(frame.to_numpy(dtype=np.float64))


def _to_excess(returns: pd.DataFrame, risk_free: pd.Series, method: str) -> pd.DataFrame:
    """Convert raw per-asset returns to excess returns by subtracting the risk-free rate.

    ``risk_free`` is one ``(date,)`` series broadcast across every asset column. ``method``:
    ``"simple"`` → ``r - rf``; ``"log"`` → ``log(1+r) - log(1+rf)``. The rf must cover every
    return date (reindexed to the returns index; a missing rf date is an error, not a silent NaN).
    """
    if method not in ("simple", "log"):
        raise ValueError(f"excess method must be 'simple' or 'log'; got {method!r}")
    rf = risk_free.reindex(returns.index)
    if rf.isna().to_numpy().any():
        raise ValueError("risk_free is missing values for some return dates")
    if method == "simple":
        return returns.sub(rf, axis=0)
    log_r = np.log1p(returns.to_numpy(dtype=np.float64))
    log_rf = np.log1p(rf.to_numpy(dtype=np.float64))
    return pd.DataFrame(log_r - log_rf[:, None], index=returns.index, columns=returns.columns)


class TimeSeriesView:
    """A point-in-time view: a shared calendar + aligned ``features`` and ``returns`` blocks.

    Parameters
    ----------
    returns:
        ``(date x asset)`` returns. **Excess** returns by default; if ``risk_free`` is given they
        are treated as **raw** and converted to excess internally (convenience for the common case
        of having raw stock returns + one rf series). A return indexed at ``t`` is realized over the
        period ending at ``t`` (standard finance convention). One column = market timing.
    features:
        ``(date x feature)`` predictors known as of their index date.
    horizon:
        Forecast horizon ``h`` in calendar steps. Features at ``t`` pair with the return
        realized over ``(t, t+h]``. ``h >= 1``; ``h = 0`` (contemporaneous) is disallowed for
        predictive use and is rejected here.
    risk_free:
        Optional ``(date,)`` risk-free rate. When given, ``returns`` are raw and the view stores
        ``excess = returns - risk_free`` (broadcast across assets); downstream always sees excess.
    excess:
        Excess-return convention when ``risk_free`` is given: ``"simple"`` (``r - rf``, default) or
        ``"log"`` (``log(1+r) - log(1+rf)``).

    Notes
    -----
    ``returns`` and ``features`` must share an identical, sorted, unique :class:`DatetimeIndex`.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        features: pd.DataFrame,
        *,
        horizon: int = 1,
        risk_free: pd.Series | None = None,
        excess: str = "simple",
    ) -> None:
        if horizon < 1:
            raise ValueError(
                f"horizon must be >= 1 (h=0 is a contemporaneous look-ahead); got {horizon}"
            )
        if risk_free is not None:
            returns = _to_excess(returns, risk_free, excess)
        if not returns.index.equals(features.index):
            raise ValueError("returns and features must share one identical DatetimeIndex")
        index = returns.index
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError("view index must be a DatetimeIndex")
        if not index.is_monotonic_increasing or not index.is_unique:
            raise ValueError("view index must be sorted and unique")

        self._dates: pd.DatetimeIndex = index
        self._assets: list[str] = [str(c) for c in returns.columns]
        self._features: list[str] = [str(c) for c in features.columns]
        self._ret: Float = _as_2d(returns)
        self._feat: Float = _as_2d(features)
        self.horizon: int = horizon
        # Calendar = the subset of dates at which predictions/rebalances happen. Defaults to
        # the full data index; `window`/`between` carve out train/test sub-calendars.
        self._cal: pd.DatetimeIndex = index

    # -- construction helpers -------------------------------------------------

    def _spawn(self, *, data_hi: int, cal: pd.DatetimeIndex) -> TimeSeriesView:
        """Build a sub-view sharing this view's metadata, with data truncated to ``[:data_hi]``."""
        view = object.__new__(TimeSeriesView)
        view._dates = self._dates[:data_hi]
        view._assets = self._assets
        view._features = self._features
        view._ret = self._ret[:data_hi]
        view._feat = self._feat[:data_hi]
        view.horizon = self.horizon
        view._cal = cal
        return view

    # -- DataView protocol ----------------------------------------------------

    @property
    def calendar(self) -> pd.DatetimeIndex:
        """Rebalancing / observation timestamps (the prediction calendar)."""
        return self._cal

    def window(self, end: object) -> TimeSeriesView:
        """View restricted to information available up to ``end`` (data and calendar both <= end).

        No look-ahead: every block is truncated to dates ``<= end``. Used for train folds —
        :meth:`aligned` inside the result then only forms pairs whose target is realized by
        ``end`` (the horizon purge).
        """
        ts = pd.Timestamp(end)  # pyright: ignore[reportArgumentType]
        data_hi = int(self._dates.searchsorted(ts, side="right"))
        cal = self._dates[:data_hi]
        return self._spawn(data_hi=data_hi, cal=cal)

    # -- horizon / PIT pairing (engine-owned) ---------------------------------

    def between(self, start: object, end: object) -> TimeSeriesView:
        """Test-fold view: data truncated to ``<= end``, calendar restricted to ``(start, end]``.

        Predictions are formed only at calendar dates strictly after ``start``; each uses
        ``features_asof(t)`` (data ``<= t``). Realized P&L is computed by the engine from the
        full view, so the model never sees future returns.
        """
        lo = pd.Timestamp(start)  # pyright: ignore[reportArgumentType]
        hi = pd.Timestamp(end)  # pyright: ignore[reportArgumentType]
        data_hi = int(self._dates.searchsorted(hi, side="right"))
        dates = self._dates[:data_hi]
        mask = dates > lo
        cal = dates[mask]
        return self._spawn(data_hi=data_hi, cal=cal)

    @property
    def assets(self) -> list[str]:
        """Asset (returns-column) names."""
        return list(self._assets)

    @property
    def feature_names(self) -> list[str]:
        """Feature (predictor-column) names."""
        return list(self._features)

    def tail(self, k: int) -> TimeSeriesView:
        """Restrict the calendar to its last ``k`` observations (rolling window; data unchanged)."""
        if k < 1:
            raise ValueError("k must be >= 1")
        return self._spawn(data_hi=len(self._dates), cal=self._cal[-k:])

    def returns_frame(self) -> pd.DataFrame:
        """The ``(date x asset)`` returns block over the calendar (raw eject)."""
        pos = np.asarray(self._dates.searchsorted(self._cal))
        return pd.DataFrame(self._ret[pos], index=self._cal, columns=self._assets)

    def features_frame(self) -> pd.DataFrame:
        """The ``(date x feature)`` features block over the calendar (raw eject)."""
        pos = np.asarray(self._dates.searchsorted(self._cal))
        return pd.DataFrame(self._feat[pos], index=self._cal, columns=self._features)

    def features_asof(self, t: object) -> Float:
        """Feature vector known as of ``t`` (the latest data row with date ``<= t``)."""
        ts = pd.Timestamp(t)  # pyright: ignore[reportArgumentType]
        pos = int(self._dates.searchsorted(ts, side="right")) - 1
        if pos < 0:
            raise KeyError(f"no features available as of {ts}")
        return self._feat[pos]

    def target_asof(self, t: object, horizon: int | None = None) -> Float:
        """Return realized over ``(t, t+h]`` per asset, or ``nan`` if not yet realized in-view.

        Compounds simple returns over the ``h`` data periods strictly after ``t``.
        """
        h = self.horizon if horizon is None else horizon
        ts = pd.Timestamp(t)  # pyright: ignore[reportArgumentType]
        pos = int(self._dates.searchsorted(ts, side="right")) - 1
        nan = np.full(len(self._assets), np.nan, dtype=np.float64)
        if pos < 0 or pos + h >= len(self._dates):
            return nan
        fut = self._ret[pos + 1 : pos + 1 + h]
        return np.prod(1.0 + fut, axis=0) - 1.0

    def aligned(self, horizon: int | None = None) -> tuple[pd.DatetimeIndex, Float, Float]:
        """Supervised ``(dates, X, Y)`` over the calendar: ``X`` at ``t`` paired with ``(t, t+h]``.

        Only pairs whose target is fully realized within this view's data are kept — so on a
        ``window(end)`` view the last usable feature date ``t`` satisfies ``t + h <= end``
        (the horizon purge that kills the contemporaneous leak).
        """
        h = self.horizon if horizon is None else horizon
        n_data = len(self._dates)
        last_ok = n_data - h - 1  # max feature position whose target lands within data
        rows_x: list[Float] = []
        rows_y: list[Float] = []
        kept: list[pd.Timestamp] = []
        for t in self._cal:
            pos = int(self._dates.searchsorted(t, side="left"))
            if pos > last_ok:
                continue
            fut = self._ret[pos + 1 : pos + 1 + h]
            rows_x.append(self._feat[pos])
            rows_y.append(np.prod(1.0 + fut, axis=0) - 1.0)
            kept.append(t)
        if not kept:
            x = np.empty((0, len(self._features)), dtype=np.float64)
            y = np.empty((0, len(self._assets)), dtype=np.float64)
            return pd.DatetimeIndex([]), x, y
        return pd.DatetimeIndex(kept), np.vstack(rows_x), np.vstack(rows_y)
