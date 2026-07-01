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

Cross-section. :class:`CrossSectionView` is the sibling for the *cross-sectional* family
(Fama-MacBeth / IPCA / characteristic sorts): a ragged panel of many assets whose predictors vary
by ``(date, asset)`` rather than being shared across assets. It shares the ``DataView`` protocol
(``window`` + ``calendar``) but exposes a cross-section-shaped ``features_asof`` / ``target_asof`` /
``aligned`` — see its own docstring. The naming follows the field's own dichotomy (Fama 2015,
"Cross-Section *Versus* Time-Series Tests"): time-series tests (GRS) vs cross-sectional tests (FM).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

Float = NDArray[np.float64]


@runtime_checkable
class Block(Protocol):
    """A tagged feature block the view aligns to the decision calendar and concatenates.

    Both :class:`FeatureBlock` (time-series + lag) and :class:`VintagedBlock` (a ``(ref, vintage)``
    panel) satisfy it, so heterogeneous macro sources coexist in one :class:`TimeSeriesView`.
    """

    @property
    def names(self) -> list[str]: ...
    def is_ready(self, t: object) -> bool: ...
    def asof(self, t: object) -> Float: ...
    def truncate(self, end: object) -> Block: ...


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


class VintagedBlock:
    """A vintaged (point-in-time) block: a ``(ref_date x vintage)`` panel resolved by ``asof``.

    Built from a tidy table ``[ref_date, vintage, <series...>]`` (e.g. what a FRED-MD build yields).
    ``asof(t)`` returns the real-time edge: among vintages available at ``t`` (``vintage`` month +
    ``lag`` <= ``t``'s month), the most recent ``ref_date``'s value taken from its latest available
    vintage. Revisions are respected — an earlier vintage's number is used until a later one is
    available, so no future revision leaks in (landmine #1).

    ``lag`` (whole months, default 1) is the availability buffer. The ``vintage`` label already *is*
    the release month, so one month suffices; sweep it up for a more conservative real-time cut.
    """

    def __init__(
        self,
        table: pd.DataFrame,
        *,
        series: list[str] | None = None,
        lag: int = 1,
        name: str | None = None,
        ref_col: str = "ref_date",
        vintage_col: str = "vintage",
    ) -> None:
        if lag < 0:
            raise ValueError(f"vintaged-block lag must be >= 0; got {lag}")
        cols = (
            [c for c in table.columns if c not in (ref_col, vintage_col)]
            if series is None
            else list(series)
        )
        ref = pd.to_datetime(table[ref_col])
        vint = pd.to_datetime(table[vintage_col])
        self._names: list[str] = [str(c) for c in cols]
        # month ordinals (year*12 + month) → cheap availability / edge comparisons
        self._ref: NDArray[np.int64] = np.asarray(ref.dt.year * 12 + ref.dt.month, dtype=np.int64)
        self._vint: NDArray[np.int64] = np.asarray(
            vint.dt.year * 12 + vint.dt.month, dtype=np.int64
        )
        self._vals: Float = _as_2d(table[cols])
        self.lag: int = lag
        self.name: str | None = name

    @property
    def names(self) -> list[str]:
        """Feature (series) names of this block."""
        return list(self._names)

    def _edge(self, t: object) -> int:
        """Row index of the real-time edge at ``t`` (latest ref_date, latest available vintage)."""
        ts = pd.Timestamp(t)  # pyright: ignore[reportArgumentType]
        t_ord = ts.year * 12 + ts.month
        avail = np.flatnonzero(self._vint + self.lag <= t_ord)
        if avail.size == 0:
            return -1
        edge_ref = int(self._ref[avail].max())
        at_edge = avail[self._ref[avail] == edge_ref]
        return int(at_edge[np.argmax(self._vint[at_edge])])

    def is_ready(self, t: object) -> bool:
        """Whether any vintage is available at ``t`` (False before the first release + lag)."""
        return self._edge(t) >= 0

    def asof(self, t: object) -> Float:
        """Real-time vector at ``t``: latest ref_date's value from its latest available vintage."""
        i = self._edge(t)
        if i < 0:
            raise KeyError(f"no vintage available as of {t} for block {self.name!r}")
        return self._vals[i]

    def truncate(self, end: object) -> VintagedBlock:
        """A copy holding only vintages released by ``end`` (``vintage`` month <= ``end`` month)."""
        ts = pd.Timestamp(end)  # pyright: ignore[reportArgumentType]
        keep = self._vint <= ts.year * 12 + ts.month
        blk = object.__new__(VintagedBlock)
        blk._names = self._names
        blk._ref = self._ref[keep]
        blk._vint = self._vint[keep]
        blk._vals = self._vals[keep]
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
        blocks: Sequence[Block] | None = None,
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
        self._blocks: list[Block] = list(blocks)
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


class CrossSectionView:
    """A cross-sectional (panel) view: many assets with per-asset characteristics, ragged over time.

    The sibling of :class:`TimeSeriesView` for the *cross-sectional* asset-pricing family
    (Fama-MacBeth, IPCA, characteristic sorts), where the predictor ``z_{i,t}`` varies by both date
    **and** asset — unlike the aggregate/shared predictors of a time-series view. Built from a tidy
    long panel ``[date, asset, <chars...>, ret]``; the universe may enter/exit (ragged).

    Shapes (the reason this is a sibling, not a retrofit):

    - ``features_asof(t)`` returns the whole cross-section at ``t`` — ``(ids, X)`` with ``X`` shaped
      ``(n_alive x K)`` — rather than one shared vector.
    - ``target_asof(t, h)`` returns each alive asset's return realized over ``(t, t+h]`` (``nan`` if
      the asset delists before the horizon closes).
    - ``aligned`` stacks every ``(date, asset)`` observation into a panel design matrix
      ``(N_obs x K)`` with a ``(N_obs,)`` target — the shape Fama-MacBeth / IPCA consume.

    Internally stored as a tidy panel sorted by ``(date, asset)`` (date-outer, so a cross-section
    is a contiguous slice). Characteristics are taken as **known at their row date** (period-end
    info set at ``t``); any reporting/publication lag must be applied to the panel before it is
    handed in (PIT is the provider's contract, mirroring a time-series block's ``lag=0``).
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        *,
        chars: Sequence[str],
        ret: str = "ret",
        date_col: str = "date",
        asset_col: str = "asset",
        horizon: int = 1,
    ) -> None:
        if horizon < 1:
            raise ValueError(
                f"horizon must be >= 1 (h=0 is a contemporaneous look-ahead); got {horizon}"
            )
        names = list(chars)
        frame = panel[[date_col, asset_col, *names, ret]].copy()
        frame[date_col] = pd.to_datetime(frame[date_col])
        frame = frame.sort_values([date_col, asset_col], kind="stable").reset_index(drop=True)
        cal = pd.DatetimeIndex(frame[date_col].drop_duplicates())
        self._dates: pd.DatetimeIndex = cal
        self._chars: list[str] = [str(c) for c in names]
        self._asset: NDArray[np.object_] = frame[asset_col].to_numpy().astype(object)
        self._x: Float = _as_2d(frame[names])
        self._ret: Float = frame[ret].to_numpy(dtype=np.float64)
        self._dpos: NDArray[np.int64] = np.asarray(
            cal.searchsorted(frame[date_col].to_numpy()), dtype=np.int64
        )
        self.horizon: int = horizon
        self._cal: pd.DatetimeIndex = cal
        # (date-position, asset) -> row index; powers ragged forward-return lookups over delistings
        self._cell: dict[tuple[int, object], int] = {
            (int(d), a): i for i, (d, a) in enumerate(zip(self._dpos, self._asset, strict=True))
        }

    # -- DataView protocol ----------------------------------------------------

    @property
    def calendar(self) -> pd.DatetimeIndex:
        """Rebalancing / observation timestamps (the prediction calendar)."""
        return self._cal

    @property
    def assets(self) -> list[str]:
        """Sorted union of every asset id that ever appears (the report / tensor column axis)."""
        return sorted({str(a) for a in self._asset})

    @property
    def char_names(self) -> list[str]:
        """Characteristic (per-asset predictor) column names."""
        return list(self._chars)

    # -- cross-section access -------------------------------------------------

    def _pos_asof(self, t: object) -> int:
        """Position of the latest calendar date ``<= t`` (the cross-section observed at ``t``)."""
        ts = pd.Timestamp(t)  # pyright: ignore[reportArgumentType]
        return int(self._dates.searchsorted(ts, side="right")) - 1

    def _row_slice(self, p: int) -> tuple[int, int]:
        """Half-open row range of the (contiguous) cross-section at date-position ``p``."""
        lo = int(np.searchsorted(self._dpos, p, side="left"))
        hi = int(np.searchsorted(self._dpos, p, side="right"))
        return lo, hi

    def universe(self, t: object) -> list[str]:
        """Asset ids alive as of ``t`` (empty before the first date)."""
        p = self._pos_asof(t)
        if p < 0:
            return []
        lo, hi = self._row_slice(p)
        return [str(a) for a in self._asset[lo:hi]]

    def features_asof(self, t: object) -> tuple[NDArray[np.object_], Float]:
        """The cross-section known as of ``t``: ``(ids, X)`` with ``X`` shaped ``(n_alive x K)``."""
        p = self._pos_asof(t)
        if p < 0:
            raise KeyError(f"no cross-section available as of {t}")
        lo, hi = self._row_slice(p)
        return self._asset[lo:hi], self._x[lo:hi]

    def target_asof(
        self, t: object, horizon: int | None = None
    ) -> tuple[NDArray[np.object_], Float]:
        """Per-asset return over ``(t, t+h]`` for the ``t`` cross-section; ``nan`` if it delists."""
        h = self.horizon if horizon is None else horizon
        p = self._pos_asof(t)
        if p < 0:
            return np.empty(0, dtype=object), np.empty(0, dtype=np.float64)
        lo, hi = self._row_slice(p)
        ids = self._asset[lo:hi]
        out = np.full(len(ids), np.nan, dtype=np.float64)
        if p + h < len(self._dates):
            for k in range(len(ids)):
                a = ids[k]
                prod = 1.0
                ok = True
                for step in range(1, h + 1):
                    r = self._cell.get((p + step, a))
                    if r is None:  # asset absent at t+step (delisted / gap) -> no clean h-return
                        ok = False
                        break
                    prod *= 1.0 + self._ret[r]
                if ok:
                    out[k] = prod - 1.0
        return ids, out

    # -- PIT windowing --------------------------------------------------------

    def _spawn(self, *, end: object, cal: pd.DatetimeIndex) -> CrossSectionView:
        """Sub-view with info ``<= end``: rows truncated, calendar set to ``cal``."""
        ts = pd.Timestamp(end)  # pyright: ignore[reportArgumentType]
        hi = int(self._dates.searchsorted(ts, side="right"))
        mask = self._dpos < hi
        v = object.__new__(CrossSectionView)
        v._dates = self._dates[:hi]
        v._chars = self._chars
        v._asset = self._asset[mask]
        v._x = self._x[mask]
        v._ret = self._ret[mask]
        v._dpos = self._dpos[mask]
        v.horizon = self.horizon
        v._cal = cal
        v._cell = {(int(d), a): i for i, (d, a) in enumerate(zip(v._dpos, v._asset, strict=True))}
        return v

    def window(self, end: object) -> CrossSectionView:
        """View restricted to info available up to ``end`` (data and calendar both ``<= end``)."""
        ts = pd.Timestamp(end)  # pyright: ignore[reportArgumentType]
        hi = int(self._dates.searchsorted(ts, side="right"))
        return self._spawn(end=ts, cal=self._dates[:hi])

    def between(self, start: object, end: object) -> CrossSectionView:
        """Test-fold view: data truncated to ``<= end``, calendar restricted to ``(start, end]``."""
        lo = pd.Timestamp(start)  # pyright: ignore[reportArgumentType]
        hi = pd.Timestamp(end)  # pyright: ignore[reportArgumentType]
        data_hi = int(self._dates.searchsorted(hi, side="right"))
        dates = self._dates[:data_hi]
        return self._spawn(end=hi, cal=dates[dates > lo])

    # -- supervised design + ejects -------------------------------------------

    def aligned(self, horizon: int | None = None) -> tuple[pd.MultiIndex, Float, Float]:
        """Panel design over the calendar: stacked ``keys[(date,asset)], X (Nobs x K), y (Nobs,)``.

        Keeps only observations whose target is realized in-view (horizon purge / no delisting gap)
        and whose characteristics are all finite (missing-char imputation is the method's job).
        """
        h = self.horizon if horizon is None else horizon
        d_keys: list[pd.Timestamp] = []
        a_keys: list[object] = []
        rows_x: list[Float] = []
        ys: list[float] = []
        for t in self._cal:
            ids, x = self.features_asof(t)
            _ids, y = self.target_asof(t, h)
            for k in range(len(ids)):
                if not np.isfinite(y[k]) or not bool(np.isfinite(x[k]).all()):
                    continue
                d_keys.append(t)
                a_keys.append(ids[k])
                rows_x.append(x[k])
                ys.append(float(y[k]))
        if not ys:
            empty = pd.MultiIndex.from_arrays([pd.DatetimeIndex([]), []], names=["date", "asset"])
            return empty, np.empty((0, len(self._chars)), dtype=np.float64), np.empty(0, np.float64)
        keys = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex(d_keys), a_keys], names=["date", "asset"]
        )
        return keys, np.vstack(rows_x), np.asarray(ys, dtype=np.float64)

    def panel_frame(self) -> pd.DataFrame:
        """Tidy long eject: ``[<chars>, ret]`` on a ``(date,asset)`` MultiIndex over calendar."""
        cal_pos = {int(x) for x in self._dates.searchsorted(self._cal)}
        keep = [i for i, d in enumerate(self._dpos) if int(d) in cal_pos]
        idx = pd.MultiIndex.from_arrays(
            [self._dates[self._dpos[keep]], self._asset[keep]], names=["date", "asset"]
        )
        data = np.column_stack([self._x[keep], self._ret[keep]])
        return pd.DataFrame(data, index=idx, columns=[*self._chars, "ret"])
