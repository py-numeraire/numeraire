"""Walk-forward OOS engine. The most-reused, most-bug-prone, method-agnostic core.

The driver is deliberately small: for each ``(train, test)`` fold it fits the estimator on the
train view and asks the fitted model for its capability output on the test view, then computes
realized P&L **from the original full view** so the model never touches future returns. Output
is one tidy container carrying the preprocessing/vintage provenance every result row needs
(``config_hash`` + ``data_vintage``).
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np
import pandas as pd

from numeraire.core import capabilities
from numeraire.core.data import CrossSectionView, Float, TimeSeriesView
from numeraire.core.protocols import Estimator, SupportsForecast, SupportsWeights

_T = TypeVar("_T")
_R = TypeVar("_R")


def config_hash(config: dict[str, Any] | None) -> str:
    """Stable short hash of a JSON-serializable config dict (preprocessing provenance)."""
    payload = json.dumps(config or {}, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def _resolve_workers(n_jobs: int) -> int:
    """Resolve an sklearn-style ``n_jobs`` to a positive worker count (``-1`` = all cores)."""
    if n_jobs == 0:
        raise ValueError("n_jobs must be >= 1 or negative (-1 = all cores); got 0")
    if n_jobs < 0:
        return max(1, (os.cpu_count() or 1) + 1 + n_jobs)
    return n_jobs


def _map_folds(fn: Callable[[_T], _R], items: Sequence[_T], n_jobs: int) -> list[_R]:
    """Map ``fn`` over independent fold work, order-preserving (so results stay deterministic).

    ``n_jobs=1`` runs serially in-process — identical to the sequential path, zero overhead. Any
    other value uses a **thread** pool, chosen over processes because (a) it works with any
    estimator (no pickling / closure limits, so it is safe to turn on by default) and (b)
    asset-pricing fits are BLAS-bound and NumPy/SciPy release the GIL during the heavy linear
    algebra, so threads parallelize the real work. Determinism is preserved: each
    fold is a pure function of ``(estimator, train, test)`` and :meth:`ThreadPoolExecutor.map`
    yields results in input order, so a parallel run reassembles bit-for-bit like the serial one.
    """
    if n_jobs == 1:
        return [fn(it) for it in items]
    with ThreadPoolExecutor(max_workers=_resolve_workers(n_jobs)) as pool:
        return list(pool.map(fn, items))


@dataclass(frozen=True)
class WeightsOutput:
    """OOS output for a ``to_weights`` method: realized weights aligned with realized returns.

    ``weights`` and ``realized`` are both ``(date x asset)`` indexed by the prediction dates,
    where ``realized.loc[t]`` is the return over ``(t, t+h]`` (so ``strategy_returns`` is the
    realized, no-look-ahead P&L of holding ``weights.loc[t]`` over that period).
    """

    weights: pd.DataFrame
    realized: pd.DataFrame
    method: str
    config_hash: str
    data_vintage: str
    run_id: str
    capability: str = capabilities.TO_WEIGHTS
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def universe(self) -> str:
        """Compact universe label (``n=<#assets>`` for panels, the name for a single asset)."""
        cols = [str(c) for c in self.weights.columns]
        return cols[0] if len(cols) == 1 else f"n={len(cols)}"

    def strategy_returns(self) -> pd.Series:
        """Realized portfolio return per date: ``sum_a weights[a] * realized[a]``."""
        prod = self.weights.to_numpy(dtype=np.float64) * self.realized.to_numpy(dtype=np.float64)
        return pd.Series(np.nansum(prod, axis=1), index=self.weights.index, name="strategy_return")


@dataclass(frozen=True)
class ForecastOutput:
    """OOS output for a ``to_forecast`` method: per-origin forecast, realized return, benchmark.

    All three are ``(origin x asset)`` indexed by the forecast origin ``t``: ``forecasts.loc[t]``
    predicts the return over ``(t, t+h]``, ``realized.loc[t]`` is that realized return, and
    ``benchmark.loc[t]`` is the prevailing/window historical-mean forecast the engine computes
    for free at each origin (the Goyal-Welch OOS R^2 reference).
    """

    forecasts: pd.DataFrame
    realized: pd.DataFrame
    benchmark: pd.DataFrame
    method: str
    config_hash: str
    data_vintage: str
    run_id: str
    capability: str = capabilities.TO_FORECAST
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def universe(self) -> str:
        """Compact universe label (``n=<#assets>`` for panels, the name for a single asset)."""
        cols = [str(c) for c in self.forecasts.columns]
        return cols[0] if len(cols) == 1 else f"n={len(cols)}"


def walk_forward_forecast(
    estimator: Estimator,
    view: TimeSeriesView,
    *,
    min_train: int = 20,
    window: int | None = None,
    horizon: int | None = None,
    method: str,
    config: dict[str, Any] | None = None,
    data_vintage: str = "unknown",
    run_id: str | None = None,
    n_jobs: int = 1,
) -> ForecastOutput:
    """Walk-forward pseudo-OOS forecast (forecast-origin convention; GW2008 / 1-A / VoC).

    At each origin ``t`` the model is fit on the window of data ending at and **including** ``t``
    (rolling if ``window`` is given, else expanding from the start with ``min_train`` warm-up) and
    asked to forecast the return over ``(t, t+h]``; the engine records the realized return and the
    window historical-mean benchmark. No look-ahead: the forecast uses only data ``<= t`` and the
    target is strictly future. ``n_jobs`` fans the independent origins over a thread pool
    (``-1`` = all cores); results are order-preserved, so the output is identical to ``n_jobs=1``.
    """
    h = view.horizon if horizon is None else horizon
    chash = config_hash(config)
    rid = run_id if run_id is not None else f"{method}-{chash}"
    assets = view.assets
    cal = view.calendar
    n = len(cal)
    warmup = window if window is not None else min_train
    if warmup < 1:
        raise ValueError("need a positive window / min_train warm-up")

    def _run(j: int) -> tuple[pd.Timestamp, Float, Float, Float]:
        origin = cal[j]
        train = view.window(origin)
        if window is not None:
            train = train.tail(window)
        model = estimator.fit(train)
        if capabilities.TO_FORECAST not in model.capabilities() or not isinstance(
            model, SupportsForecast
        ):
            raise TypeError(f"{method}: fitted model does not support 'to_forecast'")
        f = model.forecast(train)
        bench = train.returns_frame().to_numpy(dtype=np.float64).mean(axis=0)
        return origin, f.to_numpy(dtype=np.float64), bench, view.target_asof(origin, horizon=h)

    rows = _map_folds(_run, list(range(warmup - 1, n - h)), n_jobs)
    idx: list[pd.Timestamp] = [r[0] for r in rows]
    f_rows: list[Float] = [r[1] for r in rows]
    b_rows: list[Float] = [r[2] for r in rows]
    r_rows: list[Float] = [r[3] for r in rows]

    index = pd.DatetimeIndex(idx)
    forecasts = pd.DataFrame(_stack(f_rows, len(assets)), index=index, columns=assets)
    benchmark = pd.DataFrame(_stack(b_rows, len(assets)), index=index, columns=assets)
    realized = pd.DataFrame(_stack(r_rows, len(assets)), index=index, columns=assets)
    return ForecastOutput(
        forecasts=forecasts,
        realized=realized,
        benchmark=benchmark,
        method=method,
        config_hash=chash,
        data_vintage=data_vintage,
        run_id=rid,
    )


def _stack(rows: list[Float], n_cols: int) -> Float:
    """Vertically stack forecast rows, or an empty ``(0, n_cols)`` array if there are none."""
    if not rows:
        return np.empty((0, n_cols), dtype=np.float64)
    return np.vstack(rows)


def walk_forward(
    estimator: Estimator,
    view: TimeSeriesView,
    splitter: Any,
    *,
    method: str,
    config: dict[str, Any] | None = None,
    data_vintage: str = "unknown",
    run_id: str | None = None,
    n_jobs: int = 1,
) -> WeightsOutput:
    """Run a walk-forward OOS backtest of a ``to_weights`` estimator over ``view``.

    Parameters
    ----------
    estimator:
        Anything conforming to :class:`~numeraire.core.protocols.Estimator`; the fitted model
        must support :class:`~numeraire.core.protocols.SupportsWeights`.
    splitter:
        Any object with ``split(view) -> Iterator[(train, test)]`` (e.g.
        :class:`~numeraire.core.splitter.WalkForwardSplitter`).
    config:
        Preprocessing/method config, hashed into every result row's ``config_hash``.
    n_jobs:
        Fan the independent ``(train, test)`` folds over a thread pool (``-1`` = all cores).
        Order-preserving, so the result is identical to the serial ``n_jobs=1`` default.
    """
    chash = config_hash(config)
    rid = run_id if run_id is not None else f"{method}-{chash}"
    assets = view.assets

    def _run(
        fold: tuple[TimeSeriesView, TimeSeriesView],
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        train, test = fold
        model = estimator.fit(train)
        if capabilities.TO_WEIGHTS not in model.capabilities() or not isinstance(
            model, SupportsWeights
        ):
            raise TypeError(f"{method}: fitted model does not support 'to_weights'")
        w = model.to_weights(test)
        if w.empty:
            return None
        realized = np.vstack([view.target_asof(t) for t in w.index])
        # Drop prediction dates whose target is not yet realized in-sample (the unrealized
        # tail near the end of data) — they cannot be scored without look-ahead.
        keep = ~np.isnan(realized).all(axis=1)
        if not bool(keep.any()):
            return None
        return w.iloc[keep], pd.DataFrame(realized[keep], index=w.index[keep], columns=assets)

    results = _map_folds(_run, list(splitter.split(view)), n_jobs)
    w_rows: list[pd.DataFrame] = [r[0] for r in results if r is not None]
    r_rows: list[pd.DataFrame] = [r[1] for r in results if r is not None]

    if w_rows:
        weights = pd.concat(w_rows).sort_index()
        realized_df = pd.concat(r_rows).sort_index()
    else:
        weights = pd.DataFrame(columns=assets)
        realized_df = pd.DataFrame(columns=assets)

    return WeightsOutput(
        weights=weights,
        realized=realized_df,
        method=method,
        config_hash=chash,
        data_vintage=data_vintage,
        run_id=rid,
    )


@dataclass(frozen=True)
class PanelWeightsOutput:
    """OOS output for a cross-sectional ``to_weights`` method over a ragged panel.

    ``weights`` and ``realized`` are long ``pd.Series`` on a ``(date, asset)`` MultiIndex; the wide,
    fixed-universe :class:`WeightsOutput` can't represent an entering/exiting universe, so the panel
    path carries the long form. ``realized`` is each name's ``(t, t+h]`` return, aligned by key.
    """

    weights: pd.Series
    realized: pd.Series
    method: str
    config_hash: str
    data_vintage: str
    run_id: str
    capability: str = capabilities.TO_WEIGHTS
    meta: dict[str, Any] = field(default_factory=dict)

    def strategy_returns(self) -> pd.Series:
        """Cross-sectional portfolio return per date: ``sum_a weights[t, a] * realized[t, a]``."""
        prod = self.weights * self.realized
        return prod.groupby(level="date").sum().rename("strategy_return")


def _panel_realized(view: CrossSectionView, keys: pd.MultiIndex, horizon: int) -> pd.Series:
    """Forward return over ``(t, t+h]`` for each ``(date, asset)`` key (``nan`` on delisting)."""
    dates = keys.get_level_values("date")
    assets = keys.get_level_values("asset")
    out = np.full(len(keys), np.nan, dtype=np.float64)
    for t in pd.DatetimeIndex(dates).unique():
        ids, y = view.target_asof(t, horizon=horizon)
        by_asset = dict(zip(ids, y, strict=True))
        for pos in np.flatnonzero(dates == t):
            out[int(pos)] = by_asset.get(assets[int(pos)], np.nan)
    return pd.Series(out, index=keys, name="realized")


def _as_weight_series(w: pd.Series | pd.DataFrame) -> pd.Series:
    """Normalize a panel model's weights to a long ``(date, asset)`` Series."""
    if isinstance(w, pd.DataFrame):
        if w.shape[1] != 1:
            raise TypeError(
                "panel to_weights must return a long (date, asset) Series or 1-col frame"
            )
        return w.iloc[:, 0]
    return w


def walk_forward_panel(
    estimator: Estimator,
    view: CrossSectionView,
    splitter: Any,
    *,
    method: str,
    config: dict[str, Any] | None = None,
    data_vintage: str = "unknown",
    run_id: str | None = None,
    n_jobs: int = 1,
) -> PanelWeightsOutput:
    """Walk-forward OOS backtest of a cross-sectional ``to_weights`` estimator over a ragged panel.

    Mirrors :func:`walk_forward` but for :class:`~numeraire.core.data.CrossSectionView`: the fitted
    model returns long ``(date, asset)`` weights, realized forward returns are aligned by key (so an
    entering/exiting universe is handled), and any name whose horizon is unrealized in-view (or that
    delists first) is dropped before scoring. The time-series engine is left untouched. ``n_jobs``
    fans the folds over a thread pool (``-1`` = all cores); order-preserving, so identical output.
    """
    chash = config_hash(config)
    rid = run_id if run_id is not None else f"{method}-{chash}"

    def _run(fold: tuple[CrossSectionView, CrossSectionView]) -> tuple[pd.Series, pd.Series] | None:
        train, test = fold
        model = estimator.fit(train)
        if capabilities.TO_WEIGHTS not in model.capabilities() or not isinstance(
            model, SupportsWeights
        ):
            raise TypeError(f"{method}: fitted model does not support 'to_weights'")
        w = _as_weight_series(model.to_weights(test))
        if w.empty:
            return None
        keys = w.index
        if not isinstance(keys, pd.MultiIndex):
            raise TypeError(f"{method}: panel weights need a (date, asset) MultiIndex")
        realized = _panel_realized(view, keys, view.horizon)
        keep = realized.notna().to_numpy()
        if not bool(keep.any()):
            return None
        return w[keep], realized[keep]

    results = _map_folds(_run, list(splitter.split(view)), n_jobs)
    w_parts: list[pd.Series] = [r[0] for r in results if r is not None]
    r_parts: list[pd.Series] = [r[1] for r in results if r is not None]

    if w_parts:
        weights = pd.concat(w_parts).sort_index()
        realized_s = pd.concat(r_parts).sort_index()
    else:
        empty_idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex([]), []], names=["date", "asset"])
        weights = pd.Series(dtype=np.float64, index=empty_idx, name="weight")
        realized_s = pd.Series(dtype=np.float64, index=empty_idx, name="realized")

    return PanelWeightsOutput(
        weights=weights,
        realized=realized_s,
        method=method,
        config_hash=chash,
        data_vintage=data_vintage,
        run_id=rid,
    )
