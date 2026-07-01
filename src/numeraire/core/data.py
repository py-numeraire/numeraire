"""Concrete ``DataView`` implementations and the PIT / horizon machinery (SPEC §4A).

This module ships :class:`TimeSeriesView`, covering the market-timing / aggregate-predictor case
(VoC, 1/A): a ``returns`` block ``(date x asset)`` with cardinality 1..N and one or more
time-series ``features`` blocks ``(date x feature)``. It implements the
:class:`numeraire.core.protocols.DataView` protocol and adds the explicit-horizon pairing the OOS
engine relies on:

    ``features_asof(t)`` (info known <= t)  <->  ``target_asof(t, h)`` (realized over (t, t+h])

The pairing is **engine-owned** so a method never indexes returns itself — this makes the
SOF-style one-period contemporaneous leak structurally impossible (landmine #1, SPEC §6.1).

Multi-block / availability (SPEC §4A). Each feature source enters as its own :class:`FeatureBlock`
with its **own calendar** and an **availability lag** (in the block's own periods): a row dated
``tau`` is usable at decision time ``t`` only after ``lag`` periods have elapsed. ``lag=0`` =
period-end-known (prices, Goyal-Welch predictors); ``lag>=1`` = a publication-lagged macro source
that ships no vintage panel (the conservative no-vintage fallback, e.g. FRED used at lag=1). Blocks
are aligned to the returns (decision) calendar independently and concatenated, so heterogeneous
sources — different lags, different calendars — coexist as macro inputs. Block-level **vintage**
(a real ``(tau, v)`` panel resolved by ``asof`` over release dates) crystallizes next, as a third
``FeatureBlock`` flavour; the view shape here is built to take it without further reshaping.

Backward compatibility: ``TimeSeriesView(returns, features=df)`` wraps ``df`` as a single ``lag=0``
block sharing the returns calendar — identical to the original single-block behaviour.
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


class FeatureBlock:
    """One time-series feature block: a ``(date x feature)`` frame + an availability ``lag``.

    A row dated ``tau`` becomes usable only ``lag`` of the block's own periods later, so
    ``asof(t)`` returns the row at the latest ``tau`` whose position is at or before
    ``(latest tau <= t) - lag``. ``lag=0`` = period-end-known (prices, GW predictors); ``lag>=1`` =
    a publication-lagged macro source with no vintage panel (e.g. FRED at ``lag=1``). The lag is
    counted in *this block's own index steps* (v0); contiguous monthly data — the common macro
    case — makes a step exactly one month.

    ``name`` labels the source (for errors / provenance); it does not affect alignment.
    """

    def __init__(self, frame: pd.DataFrame, *, lag: int = 0, name: str | None = None) -> None:
        if lag < 0:
            raise ValueError(f"feature-block lag must be >= 0; got {lag}")
        index = frame.index
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError("feature-block index must be a DatetimeIndex")
        if not index.is_monotonic_increasing or not index.is_unique:
            raise ValueError("feature-block index must be sorted and unique")
        self._dates: pd.DatetimeIndex = index
        self._names: list[str] = [str(c) for c in frame.columns]
        self._vals: Float = _as_2d(frame)
        self.lag: int = lag
        self.name: str | None = name

    @property
    def names(self) -> list[str]:
        """Feature (column) names of this block."""
        return list(self._names)

    def _pos_asof(self, t: object) -> int:
        """Index of the latest row usable at ``t`` (latest ``tau <= t`` pulled back by ``lag``)."""
        ts = pd.Timestamp(t)  # pyright: ignore[reportArgumentType]
        return int(self._dates.searchsorted(ts, side="right")) - 1 - self.lag

    def is_ready(self, t: object) -> bool:
        """Whether any lag-aware row is available at ``t`` (False during the lag warm-up)."""
        return self._pos_asof(t) >= 0

    def asof(self, t: object) -> Float:
        """Feature vector known as of ``t`` (lag-aware; the block's real-time edge row)."""
        pos = self._pos_asof(t)
        if pos < 0:
            raise KeyError(f"no features available as of {t} for block {self.name!r}")
        return self._vals[pos]

    def truncate(self, end: object) -> FeatureBlock:
        """A copy holding only rows dated ``<= end`` (raw data truncation; lag applied at asof)."""
        ts = pd.Timestamp(end)  # pyright: ignore[reportArgumentType]
        hi = int(self._dates.searchsorted(ts, side="right"))
        blk = object.__new__(FeatureBlock)
        blk._dates = self._dates[:hi]
        blk._names = self._names
        blk._vals = self._vals[:hi]
        blk.lag = self.lag
        blk.name = self.name
        return blk


class TimeSeriesView:
    """A point-in-time view: a returns (decision) calendar + one or more aligned feature blocks.

    Parameters
    ----------
    returns:
        ``(date x asset)`` returns; its index is the decision/rebalance calendar. **Excess** by
        default; if ``risk_free`` is given they are treated as **raw** and converted to excess
        internally. A return indexed at ``t`` is realized over the period ending at ``t``. One
        column = market timing.
    features:
        Convenience single-block input: a ``(date x feature)`` frame sharing the returns index,
        wrapped as one ``lag=0`` :class:`FeatureBlock`. Mutually exclusive with ``blocks``.
    blocks:
        Explicit list of :class:`FeatureBlock` — each with its own calendar and availability lag.
        Use this to combine heterogeneous macro sources (e.g. FRED ``lag=1`` + another ``lag=2`` +
        a no-vintage source) as predictors. Mutually exclusive with ``features``.
    horizon:
        Forecast horizon ``h`` in calendar steps; features at ``t`` pair with the return realized
        over ``(t, t+h]``. ``h >= 1``; ``h = 0`` (contemporaneous) is rejected.
    risk_free, excess:
        Optional raw→excess conversion (see :func:`_to_excess`).

    Notes
    -----
    With ``features``, that frame must share the returns ``DatetimeIndex`` (the original behaviour).
    With ``blocks``, each block keeps its own calendar and is aligned to the returns calendar by its
    own lag-aware :meth:`FeatureBlock.asof`.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        features: pd.DataFrame | None = None,
        *,
        blocks: list[FeatureBlock] | None = None,
        horizon: int = 1,
        risk_free: pd.Series | None = None,
        excess: str = "simple",
    ) -> None:
        if horizon < 1:
            raise ValueError(
                f"horizon must be >= 1 (h=0 is a contemporaneous look-ahead); got {horizon}"
            )
        if (features is None) == (blocks is None):
            raise ValueError("provide exactly one of `features` (shared-calendar) or `blocks`")
        if risk_free is not None:
            returns = _to_excess(returns, risk_free, excess)
        index = returns.index
        if not isinstance(index, pd.DatetimeIndex):
            raise TypeError("view index must be a DatetimeIndex")
        if not index.is_monotonic_increasing or not index.is_unique:
            raise ValueError("view index must be sorted and unique")

        if features is not None:
            if not returns.index.equals(features.index):
                raise ValueError("returns and `features` must share one identical DatetimeIndex")
            blocks = [FeatureBlock(features, lag=0, name=None)]
        assert blocks is not None

        self._dates: pd.DatetimeIndex = index
        self._assets: list[str] = [str(c) for c in returns.columns]
        self._ret: Float = _as_2d(returns)
        self._blocks: list[FeatureBlock] = blocks
        self.horizon: int = horizon
        # Calendar = the subset of dates at which predictions/rebalances happen. Defaults to
        # the full returns index; `window`/`between` carve out train/test sub-calendars.
        self._cal: pd.DatetimeIndex = index

    # -- construction helpers -------------------------------------------------

    def _spawn(self, *, end: object, cal: pd.DatetimeIndex) -> TimeSeriesView:
        """Sub-view with info ``<= end``: returns and blocks truncated, calendar set to ``cal``."""
        ts = pd.Timestamp(end)  # pyright: ignore[reportArgumentType]
        data_hi = int(self._dates.searchsorted(ts, side="right"))
        view = object.__new__(TimeSeriesView)
        view._dates = self._dates[:data_hi]
        view._assets = self._assets
        view._ret = self._ret[:data_hi]
        view._blocks = [b.truncate(ts) for b in self._blocks]
        view.horizon = self.horizon
        view._cal = cal
        return view

    def _features_vec(self, t: object) -> Float:
        """Concatenated lag-aware feature vector across all blocks, as known at ``t``."""
        return np.concatenate([b.asof(t) for b in self._blocks])

    # -- DataView protocol ----------------------------------------------------

    @property
    def calendar(self) -> pd.DatetimeIndex:
        """Rebalancing / observation timestamps (the prediction calendar)."""
        return self._cal

    def window(self, end: object) -> TimeSeriesView:
        """View restricted to information available up to ``end`` (data and calendar both <= end).

        No look-ahead: returns and every feature block are truncated to dates ``<= end``. Used for
        train folds — :meth:`aligned` then only forms pairs whose target is realized by ``end``.
        """
        ts = pd.Timestamp(end)  # pyright: ignore[reportArgumentType]
        data_hi = int(self._dates.searchsorted(ts, side="right"))
        cal = self._dates[:data_hi]
        return self._spawn(end=ts, cal=cal)

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
        return self._spawn(end=hi, cal=cal)

    @property
    def assets(self) -> list[str]:
        """Asset (returns-column) names."""
        return list(self._assets)

    @property
    def feature_names(self) -> list[str]:
        """Feature (predictor-column) names, concatenated across blocks in order."""
        return [n for b in self._blocks for n in b.names]

    def tail(self, k: int) -> TimeSeriesView:
        """Restrict the calendar to its last ``k`` observations (rolling window; data unchanged)."""
        if k < 1:
            raise ValueError("k must be >= 1")
        end = self._dates[-1] if len(self._dates) else self._dates
        return self._spawn(end=end, cal=self._cal[-k:])

    def returns_frame(self) -> pd.DataFrame:
        """The ``(date x asset)`` returns block over the calendar (raw eject; SPEC §4A)."""
        pos = np.asarray(self._dates.searchsorted(self._cal))
        return pd.DataFrame(self._ret[pos], index=self._cal, columns=self._assets)

    def features_frame(self) -> pd.DataFrame:
        """The ``(date x feature)`` features block over the calendar (lag-aware; raw eject, §4A)."""
        if not len(self._cal):
            return pd.DataFrame(np.empty((0, len(self.feature_names))), columns=self.feature_names)
        rows = np.vstack([self._features_vec(t) for t in self._cal])
        return pd.DataFrame(rows, index=self._cal, columns=self.feature_names)

    def features_asof(self, t: object) -> Float:
        """Feature vector known as of ``t``, concatenated lag-aware across all blocks."""
        return self._features_vec(t)

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
        (the horizon purge that kills the contemporaneous leak). ``X`` rows are lag-aware and
        concatenated across feature blocks.
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
                continue  # target not realized in-view (late-side horizon purge)
            if any(not b.is_ready(t) for b in self._blocks):
                continue  # features not yet available (early-side lag warm-up purge)
            fut = self._ret[pos + 1 : pos + 1 + h]
            rows_x.append(self._features_vec(t))
            rows_y.append(np.prod(1.0 + fut, axis=0) - 1.0)
            kept.append(t)
        if not kept:
            x = np.empty((0, len(self.feature_names)), dtype=np.float64)
            y = np.empty((0, len(self._assets)), dtype=np.float64)
            return pd.DatetimeIndex([]), x, y
        return pd.DatetimeIndex(kept), np.vstack(rows_x), np.vstack(rows_y)
