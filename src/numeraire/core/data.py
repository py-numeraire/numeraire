"""Concrete ``DataView`` implementations and the PIT / horizon machinery.

This module ships :class:`TimeSeriesView`, covering the market-timing / aggregate-predictor case
(VoC, 1/A): a ``returns`` block ``(date x asset)`` with cardinality 1..N and one or more
time-series ``features`` blocks ``(date x feature)``. It implements the
:class:`numeraire.core.protocols.DataView` protocol and adds the explicit-horizon pairing the OOS
engine relies on:

    ``features_asof(t)`` (info known <= t)  <->  ``target_asof(t, h)`` (realized over (t, t+h])

The pairing is **engine-owned** so a method never indexes returns itself — this makes the
SOF-style one-period contemporaneous leak structurally impossible.

Multi-block / availability. Each feature source enters as its own :class:`FeatureBlock`
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

Time model. Two rules keep availability unambiguous across mixed data frequencies. First, inside
the decision calendar everything is counted in *steps*: the forecast horizon, the walk-forward
windows, and a block's own availability ``lag`` are position arithmetic on whatever calendar the
caller supplied — the framework never interprets a calendar unit such as "month". Second, at every
source boundary availability is a *timestamp* comparison: a vintage, release, or reference date is
an event on the real timeline, so a row is usable at decision time ``t`` exactly when its stamp is
``<= t`` (never "same month but later"). Controlling frequency — and any publication buffer — is
the data provider's job: stamp the true availability into the data, or shift a coarse label to a
conservative release date, before handing it to a block.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, cast, runtime_checkable

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from numeraire.core._ingest import to_pandas

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


def _to_ns(values: pd.Series | pd.DatetimeIndex, *, context: str) -> NDArray[np.int64]:
    """Datetime values → int64 nanoseconds since the epoch, at a fixed ``ns`` resolution.

    Availability everywhere is a plain ``stamp <= t`` integer comparison. Forcing ``ns`` makes it
    independent of the input's parsed datetime resolution (pandas may land on ``s``/``us``/``ns``)
    and keeps it on the same scale as a scalar ``pd.Timestamp(t).value`` used at query time.

    Two inputs would corrupt that comparison and are rejected rather than silently mis-scaled:

    - **tz-aware stamps** convert to epoch ``ns`` in UTC, while a naive query ``t`` is compared as
      wall-clock ``ns`` — the boundary would shift by the offset. The whole framework assumes
      tz-naive timestamps, so tz-aware input raises ``TypeError``.
    - **missing stamps** (``NaT``) cast to ``int64`` minimum, i.e. "available since the beginning of
      time" — a silent look-ahead. A missing availability stamp raises ``ValueError``.

    ``context`` names the offending column/source in the error message.
    """
    idx = pd.DatetimeIndex(values)
    if idx.tz is not None:
        raise TypeError(
            f"{context}: timestamps must be tz-naive; convert with "
            ".dt.tz_localize(None) / .tz_localize(None) after aligning sources to one timezone"
        )
    if idx.hasnans:
        raise ValueError(
            f"{context}: timestamp column contains missing values (NaT); "
            "every availability stamp must be a real timestamp"
        )
    arr = idx.to_numpy(dtype="datetime64[ns]")
    return np.asarray(arr).astype(np.int64)


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
    ``asof(t)`` returns the real-time edge: among rows whose ``vintage`` timestamp is at or before
    ``t`` (``vintage <= t``), the most recent ``ref_date``'s value taken from its latest available
    vintage. Availability is a plain timestamp comparison — a release stamped on a given day becomes
    visible on that day and not before — so revisions are respected (an earlier vintage's number is
    used until a later one is released) and no future vintage leaks in.

    Availability is read straight off the ``vintage`` column, which must already carry the true
    release timestamp (the data provider's job). To model a more conservative real-time cut — e.g. a
    feed that arrives some time after its stamped release — shift the vintage column before building
    the block::

        table = table.assign(vintage=table["vintage"] + pd.DateOffset(months=1))
        block = VintagedBlock(table)
    """

    def __init__(
        self,
        table: pd.DataFrame,
        *,
        series: list[str] | None = None,
        name: str | None = None,
        ref_col: str = "ref_date",
        vintage_col: str = "vintage",
    ) -> None:
        cols = (
            [c for c in table.columns if c not in (ref_col, vintage_col)]
            if series is None
            else list(series)
        )
        ref = pd.to_datetime(table[ref_col])
        vint = pd.to_datetime(table[vintage_col])
        self._names: list[str] = [str(c) for c in cols]
        # integer nanoseconds since the epoch → cheap real-timestamp availability / edge comparisons
        self._ref: NDArray[np.int64] = _to_ns(
            ref, context=f"VintagedBlock ref_date column {ref_col!r}"
        )
        self._vint: NDArray[np.int64] = _to_ns(
            vint, context=f"VintagedBlock vintage column {vintage_col!r}"
        )
        # A duplicate (ref_date, vintage) pair makes the real-time edge order-dependent (two rows
        # tie on both keys, so which value wins would depend on input row order); reject it.
        if pd.MultiIndex.from_arrays([self._ref, self._vint]).has_duplicates:
            raise ValueError(
                f"VintagedBlock has duplicate ({ref_col!r}, {vintage_col!r}) rows; "
                "each (ref_date, vintage) pair must be unique"
            )
        self._vals: Float = _as_2d(table[cols])
        self.name: str | None = name

    @property
    def names(self) -> list[str]:
        """Feature (series) names of this block."""
        return list(self._names)

    def _edge(self, t: object) -> int:
        """Row index of the real-time edge at ``t`` (latest ref_date, latest available vintage)."""
        t_ns = int(pd.Timestamp(t).value)  # pyright: ignore[reportArgumentType]
        avail = np.flatnonzero(self._vint <= t_ns)
        if avail.size == 0:
            return -1
        edge_ref = int(self._ref[avail].max())
        at_edge = avail[self._ref[avail] == edge_ref]
        return int(at_edge[np.argmax(self._vint[at_edge])])

    def is_ready(self, t: object) -> bool:
        """Whether any vintage is available at ``t`` (False before the first release timestamp)."""
        return self._edge(t) >= 0

    def asof(self, t: object) -> Float:
        """Real-time vector at ``t``: latest ref_date's value from its latest available vintage."""
        i = self._edge(t)
        if i < 0:
            raise KeyError(f"no vintage available as of {t} for block {self.name!r}")
        return self._vals[i]

    def truncate(self, end: object) -> VintagedBlock:
        """A copy holding only vintages released by ``end`` (``vintage <= end``)."""
        end_ns = int(pd.Timestamp(end).value)  # pyright: ignore[reportArgumentType]
        keep = self._vint <= end_ns
        blk = object.__new__(VintagedBlock)
        blk._names = self._names
        blk._ref = self._ref[keep]
        blk._vint = self._vint[keep]
        blk._vals = self._vals[keep]
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
        Omit both ``features`` and ``blocks`` for a **returns-only** view (market-timing /
        moment-based strategies that read only the returns block): ``feature_names`` is then empty
        and :meth:`features_frame` / :meth:`aligned` yield a zero-column ``X``.
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
    own lag-aware :meth:`FeatureBlock.asof`. With neither, the view is returns-only: no feature
    blocks, so every ``X`` is shaped ``(T x 0)`` and ``aligned`` yields the returns targets alone.
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
        returns = cast("pd.DataFrame", to_pandas(returns, what="returns"))
        if features is not None:
            features = cast("pd.DataFrame", to_pandas(features, what="features"))
        if features is not None and blocks is not None:
            raise ValueError("provide at most one of `features` (shared-calendar) or `blocks`")
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
        elif blocks is None:
            blocks = []  # returns-only view: no predictors, X has zero columns

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
        if not self._blocks:
            return np.empty(0, dtype=np.float64)  # returns-only view
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
        """The ``(date x asset)`` returns block over the calendar (raw eject)."""
        pos = np.asarray(self._dates.searchsorted(self._cal))
        return pd.DataFrame(self._ret[pos], index=self._cal, columns=self._assets)

    def features_frame(self) -> pd.DataFrame:
        """The ``(date x feature)`` features block over the calendar (lag-aware; raw eject)."""
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


@dataclass(frozen=True)
class PanelTensor:
    """A dense ``(T x N x K)`` materialization of a ragged panel — the eject for tensor/NN methods.

    ``features[t, j]`` is asset ``assets[j]``'s characteristic vector at ``dates[t]`` (``nan`` where
    the asset is absent), ``returns[t, j]`` its period return, and ``mask[t, j]`` whether it is
    present. Long stays the source of truth (ragged, ecosystem-native); this is derived on demand —
    dense + a mask is exactly how deep asset-pricing models (Gu-Kelly-Xiu, Chen-Pelger-Zhu) ingest
    an unbalanced panel. Padding is ``nan`` (not ``0``) so imputation stays the method's choice.
    """

    dates: pd.DatetimeIndex
    assets: list[str]
    chars: list[str]
    features: Float  # (T, N, K)
    returns: Float  # (T, N)
    mask: NDArray[np.bool_]  # (T, N)


class _AssetChars:
    """One asset's characteristic history + the per-asset PIT edge/lag lookups CharBlock uses.

    ``ref``/``vint`` are integer nanoseconds since the epoch (real timestamps), so availability is a
    plain ``<= t`` comparison in both modes.
    """

    def __init__(self, ref: NDArray[np.int64], vint: NDArray[np.int64] | None, vals: Float) -> None:
        self.ref = ref
        self.vint = vint
        self.vals = vals

    def at_lag(self, t_ns: int, lag: int) -> int:
        """Row of the latest ref-date ``<= t`` stepped back ``lag`` rows (lagged); -1 if none."""
        pos = int(np.searchsorted(self.ref, t_ns, side="right")) - 1 - lag
        return pos if pos >= 0 else -1

    def edge(self, t_ns: int) -> int:
        """Row of the real-time edge at ``t`` (latest ref, latest available vintage); -1 if none."""
        assert self.vint is not None
        avail = np.flatnonzero(self.vint <= t_ns)
        if avail.size == 0:
            return -1
        edge_ref = int(self.ref[avail].max())
        at_edge = avail[self.ref[avail] == edge_ref]
        return int(at_edge[np.argmax(self.vint[at_edge])])


class CharBlock:
    """A per-asset ``[t, i]`` characteristic source with its own PIT, joined into a panel view.

    The cross-sectional analog of a time-series :class:`FeatureBlock`: several heterogeneous
    per-stock predictor panels (e.g. two vendors' characteristic sets) coexist, each with its own
    availability, and concatenate along the characteristic axis. Two modes:

    - **lagged** (default): a tidy ``[date, asset, <chars...>]`` panel; availability is the row's
      own ``date`` timestamp (``date <= t``), then asset ``i``'s value is stepped back ``lag`` rows
      in ``i``'s own series (per-asset lag, counted in row steps).
    - **vintaged** (``vintage_col`` given): a ``[ref_date, asset, vintage, <chars...>]`` panel;
      asset ``i``'s value at ``t`` is its real-time edge — latest ``ref_date`` whose vintage is
      available (``vintage <= t`` by timestamp), from that ref's latest vintage (per-asset
      :class:`VintagedBlock`). This makes per-stock characteristic revisions PIT-safe mechanically.
      ``lag`` is not meaningful here (buffers belong in the vintage timestamps), so passing a
      non-zero ``lag`` together with ``vintage_col`` is an error.

    Resolved against a view's decision dates at construction (each date uses only info available by
    it), so downstream needs no special-casing. Align identifiers to a common id before building it.
    """

    def __init__(
        self,
        panel: pd.DataFrame,
        chars: Sequence[str],
        *,
        lag: int = 0,
        date_col: str = "date",
        asset_col: str = "asset",
        vintage_col: str | None = None,
        ref_col: str = "ref_date",
    ) -> None:
        if lag < 0:
            raise ValueError(f"char-block lag must be >= 0; got {lag}")
        if vintage_col is not None and lag != 0:
            raise ValueError(
                "char-block vintaged mode takes no lag (availability is the vintage timestamp); "
                "apply any publication buffer to the vintage column before building the block"
            )
        names = list(chars)
        self.names: list[str] = [str(c) for c in names]
        self.lag: int = lag
        self._vintaged: bool = vintage_col is not None
        keycol = ref_col if self._vintaged else date_col
        need = [keycol, asset_col, *names, *([vintage_col] if vintage_col is not None else [])]
        missing = [c for c in need if c not in panel.columns]
        if missing:
            raise ValueError(f"char-block panel is missing column(s) {missing}")
        df = panel[need].copy()
        ref = pd.to_datetime(df[keycol])
        ref_ns = _to_ns(ref, context=f"CharBlock {keycol!r} column")
        assets = df[asset_col].to_numpy().astype(object)
        vals = _as_2d(df[names])
        vint_ns = None
        if vintage_col is not None:
            vint = pd.to_datetime(df[vintage_col])
            vint_ns = _to_ns(vint, context=f"CharBlock vintage column {vintage_col!r}")
            # A duplicate (asset, ref_date, vintage) triple makes an asset's real-time edge
            # order-dependent (the tie is broken by input row order); reject it.
            if pd.MultiIndex.from_arrays([assets, ref_ns, vint_ns]).has_duplicates:
                raise ValueError(
                    f"CharBlock has duplicate ({asset_col!r}, {ref_col!r}, {vintage_col!r}) rows; "
                    "each (asset, ref_date, vintage) triple must be unique"
                )
        # per-asset timestamp (ns) + value arrays, so each asset resolves independently
        self._by_asset: dict[object, _AssetChars] = {}
        for a in np.unique(assets):
            m = assets == a
            if self._vintaged:
                assert vint_ns is not None
                self._by_asset[a] = _AssetChars(ref_ns[m], vint_ns[m], vals[m])
            else:
                order = np.argsort(ref_ns[m], kind="stable")
                self._by_asset[a] = _AssetChars(ref_ns[m][order], None, vals[m][order])

    def resolve(
        self, dates: pd.DatetimeIndex, assets: NDArray[np.object_], dpos: NDArray[np.int64]
    ) -> Float:
        """Values known at each row's decision date: ``(len(assets) x K)``, ``nan`` where absent.

        Row ``r`` is asset ``assets[r]`` at decision date ``dates[dpos[r]]``.
        """
        cal_ns = _to_ns(dates, context="CharBlock decision calendar")
        out = np.full((len(assets), len(self.names)), np.nan, dtype=np.float64)
        for r in range(len(assets)):
            entry = self._by_asset.get(assets[r])
            if entry is None:
                continue
            t_ns = int(cal_ns[dpos[r]])
            i = entry.edge(t_ns) if self._vintaged else entry.at_lag(t_ns, self.lag)
            if i >= 0:
                out[r] = entry.vals[i]
        return out


class CrossSectionView:
    """A cross-sectional (panel) view: many assets with per-asset characteristics, ragged over time.

    The sibling of :class:`TimeSeriesView` for the *cross-sectional* asset-pricing family
    (Fama-MacBeth, IPCA, characteristic sorts), where the predictor ``z_{i,t}`` varies by both date
    **and** asset — unlike the aggregate/shared predictors of a time-series view. Built from a tidy
    long panel ``[date, asset, <chars...>, ret]``; the universe may enter/exit (ragged).

    Shapes (the reason this is a sibling, not a retrofit):

    - ``features_asof(t)`` returns the whole cross-section at ``t`` — ``(ids, X)`` with ``X`` shaped
      ``(n_alive x K)`` — rather than one shared vector.
    - ``target_asof(t, h)`` returns each alive asset's return realized over ``(t, t+h]`` (``nan`` on
      an absent row, an input missing return, or an unrealized horizon tail).
    - ``aligned`` stacks every ``(date, asset)`` observation into a panel design matrix
      ``(N_obs x K)`` with a ``(N_obs,)`` target — the shape Fama-MacBeth / IPCA consume.

    Internally stored as a tidy panel sorted by ``(date, asset)`` (date-outer, so a cross-section
    is a contiguous slice), with assets int-coded against a fixed label axis and a ``(T x N)``
    row-index matrix (``-1`` = absent) in place of any per-cell lookup: ragged forward-return
    resolution is a vectorized gather, and PIT ``window``/``between`` sub-views are **zero-copy
    prefix slices** (a date cutoff keeps a contiguous prefix of the date-sorted rows).
    Characteristics are taken as **known at their row date** (period-end
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
        char_blocks: Sequence[CharBlock] | None = None,
    ) -> None:
        if horizon < 1:
            raise ValueError(
                f"horizon must be >= 1 (h=0 is a contemporaneous look-ahead); got {horizon}"
            )
        panel = cast("pd.DataFrame", to_pandas(panel, what="panel"))
        names = list(chars)
        required = [date_col, asset_col, ret, *names]
        missing = [c for c in required if c not in panel.columns]
        if missing:
            raise ValueError(
                f"panel is missing required column(s) {missing}; "
                f"needs date_col={date_col!r}, asset_col={asset_col!r}, ret={ret!r}, chars={names}"
            )
        frame = panel[[date_col, asset_col, *names, ret]].copy()
        frame[date_col] = pd.to_datetime(frame[date_col])
        frame = frame.sort_values([date_col, asset_col], kind="stable").reset_index(drop=True)
        if bool(frame.duplicated([date_col, asset_col]).any()):
            raise ValueError(
                "panel has duplicate (date, asset) rows; each observation must be unique"
            )
        cal = pd.DatetimeIndex(frame[date_col].drop_duplicates())
        if len(cal) <= horizon:
            raise ValueError(
                f"panel has {len(cal)} date(s) but horizon={horizon} needs at least {horizon + 1} "
                "to form any forward return; no usable (t, t+h] window exists"
            )
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
        # Heterogeneous per-asset char sources (e.g. two vendors, some vintaged) are resolved to
        # each row's decision date here (PIT-safe) and concatenated along the char axis — so every
        # downstream path (features_asof/aligned/window/to_tensor) sees them with no special-casing.
        for blk in char_blocks or ():
            resolved = blk.resolve(self._dates, self._asset, self._dpos)
            self._x = np.ascontiguousarray(np.column_stack([self._x, resolved]))
            self._chars = self._chars + blk.names
        # Int-coded asset axis + a (T x N_labels) row-index matrix (-1 = absent): forward-return
        # lookups over the ragged universe become vectorized gathers (no per-cell dict), and the
        # matrix slices along with the dates, keeping PIT windows zero-copy.
        labels, inverse = np.unique(self._asset, return_inverse=True)
        self._labels: NDArray[np.object_] = labels
        self._codes: NDArray[np.int32] = np.asarray(inverse, dtype=np.int32)
        self._rowmat: NDArray[np.int32] = np.full((len(cal), len(labels)), -1, dtype=np.int32)
        self._rowmat[self._dpos, self._codes] = np.arange(len(self._codes), dtype=np.int32)

    # -- DataView protocol ----------------------------------------------------

    @property
    def calendar(self) -> pd.DatetimeIndex:
        """Rebalancing / observation timestamps (the prediction calendar)."""
        return self._cal

    @property
    def assets(self) -> list[str]:
        """Sorted union of every asset id appearing in this view (report / tensor column axis)."""
        return [str(self._labels[c]) for c in np.unique(self._codes)]

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

    def _compound(self, dpos: NDArray[np.int64], codes: NDArray[np.int32], h: int) -> Float:
        """Compounded ``(t, t+h]`` return per (date-position, asset-code) row; ``nan`` on any gap.

        Vectorized over rows via the row-index matrix — an asset absent at any forward step (exit or
        ordinary gap), or with a non-finite input return, yields ``nan``. The engine cannot infer a
        delisting payoff from absence; data providers must merge it upstream. The only Python loop
        is the horizon.
        """
        out = np.full(len(codes), np.nan, dtype=np.float64)
        live = np.flatnonzero(dpos + h < len(self._dates))
        if live.size == 0:
            return out
        prod = np.ones(live.size, dtype=np.float64)
        ok = np.ones(live.size, dtype=bool)
        for step in range(1, h + 1):
            rows = self._rowmat[dpos[live] + step, codes[live]]
            present = rows >= 0
            ok &= present
            prod *= np.where(present, 1.0 + self._ret[rows], 1.0)
        out[live[ok]] = prod[ok] - 1.0
        return out

    def target_asof(
        self, t: object, horizon: int | None = None
    ) -> tuple[NDArray[np.object_], Float]:
        """Per-asset ``(t, t+h]`` return; ``nan`` on a gap, missing input, or horizon tail."""
        h = self.horizon if horizon is None else horizon
        p = self._pos_asof(t)
        if p < 0:
            return np.empty(0, dtype=object), np.empty(0, dtype=np.float64)
        lo, hi = self._row_slice(p)
        dpos = np.full(hi - lo, p, dtype=np.int64)
        return self._asset[lo:hi], self._compound(dpos, self._codes[lo:hi], h)

    # -- PIT windowing --------------------------------------------------------

    def _spawn(self, *, end: object, cal: pd.DatetimeIndex) -> CrossSectionView:
        """Sub-view with info ``<= end`` — zero-copy: rows dated ``<= end`` are a contiguous
        prefix of the date-sorted panel, so truncation is a numpy slice (view) of every array,
        including the row-index matrix (whose surviving entries all point into the kept prefix).
        """
        ts = pd.Timestamp(end)  # pyright: ignore[reportArgumentType]
        hi = int(self._dates.searchsorted(ts, side="right"))
        row_hi = int(np.searchsorted(self._dpos, hi, side="left"))
        v = object.__new__(CrossSectionView)
        v._dates = self._dates[:hi]
        v._chars = self._chars
        v._asset = self._asset[:row_hi]
        v._x = self._x[:row_hi]
        v._ret = self._ret[:row_hi]
        v._dpos = self._dpos[:row_hi]
        v._labels = self._labels
        v._codes = self._codes[:row_hi]
        v._rowmat = self._rowmat[:hi]
        v.horizon = self.horizon
        v._cal = cal
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
        y = self._compound(self._dpos, self._codes, h)
        in_cal = np.zeros(len(self._dates), dtype=bool)
        in_cal[np.asarray(self._dates.searchsorted(self._cal))] = True
        keep = np.flatnonzero(
            in_cal[self._dpos] & np.isfinite(y) & np.isfinite(self._x).all(axis=1)
        )
        if keep.size == 0:
            empty = pd.MultiIndex.from_arrays([pd.DatetimeIndex([]), []], names=["date", "asset"])
            return empty, np.empty((0, len(self._chars)), dtype=np.float64), np.empty(0, np.float64)
        keys = pd.MultiIndex.from_arrays(
            [pd.DatetimeIndex(self._dates[self._dpos[keep]]), self._asset[keep]],
            names=["date", "asset"],
        )
        return keys, np.ascontiguousarray(self._x[keep]), y[keep]

    def panel_frame(self) -> pd.DataFrame:
        """Tidy long eject: ``[<chars>, ret]`` on a ``(date,asset)`` MultiIndex over calendar."""
        in_cal = np.zeros(len(self._dates), dtype=bool)
        in_cal[np.asarray(self._dates.searchsorted(self._cal))] = True
        keep = np.flatnonzero(in_cal[self._dpos])
        idx = pd.MultiIndex.from_arrays(
            [self._dates[self._dpos[keep]], self._asset[keep]], names=["date", "asset"]
        )
        data = np.column_stack([self._x[keep], self._ret[keep]])
        return pd.DataFrame(data, index=idx, columns=[*self._chars, "ret"])

    def to_tensor(self) -> PanelTensor:
        """Dense ``(T x N x K)`` eject + an ``(T x N)`` presence mask (nan-padded; see PanelTensor).

        ``T`` = this view's dates, ``N`` = the union asset axis
        (:attr:`~numeraire.core.data.CrossSectionView.assets`), ``K`` = chars.
        """
        present, cols = np.unique(self._codes, return_inverse=True)
        assets = [str(self._labels[c]) for c in present]
        t_n, n_n, k_n = len(self._dates), len(assets), len(self._chars)
        features = np.full((t_n, n_n, k_n), np.nan, dtype=np.float64)
        returns = np.full((t_n, n_n), np.nan, dtype=np.float64)
        mask = np.zeros((t_n, n_n), dtype=bool)
        features[self._dpos, cols] = self._x
        returns[self._dpos, cols] = self._ret
        mask[self._dpos, cols] = True
        return PanelTensor(self._dates, assets, list(self._chars), features, returns, mask)
