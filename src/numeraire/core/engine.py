"""Walk-forward OOS engine. The most-reused, most-bug-prone, method-agnostic core.

The driver is deliberately small: for each ``(train, test)`` fold it fits the estimator on the
train view and asks the fitted model for its capability output on the test view, then computes
realized P&L **from the original full view** so the model never touches future returns. Output
is one tidy container carrying the preprocessing/vintage provenance every result row needs
(``config_hash`` + ``data_vintage``).

Naming convention: ``*Output`` classes (``WeightsOutput``, ``ForecastOutput``, ``PricingOutput``,
``PanelWeightsOutput``) are the engine's capability *artifacts* — the OOS panels an ``Evaluator``
consumes to produce result rows. ``*Result`` classes elsewhere (``SimulationResult``,
``SortResult``, the ``stats`` ``*Result`` records) are the return values of one-shot computations.
If it feeds an evaluator it is an ``Output``; if it is a computed answer it is a ``Result``.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import warnings
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, ParamSpec, TypeVar, cast

import numpy as np
import pandas as pd

from numeraire.core import capabilities
from numeraire.core.data import CrossSectionView, Float, TimeSeriesView
from numeraire.core.protocols import (
    Estimator,
    SupportsForecast,
    SupportsPricing,
    SupportsWeights,
)

_T = TypeVar("_T")
_R = TypeVar("_R")
_P = ParamSpec("_P")
_RT = TypeVar("_RT")


def _deprecated_alias(replacement: Callable[_P, _RT], *, old: str, new: str) -> Callable[_P, _RT]:
    """Wrap ``replacement`` in a thin forwarder that emits a ``DeprecationWarning`` naming ``new``.

    The old public name keeps working for one release (non-breaking rename); the warning tells
    callers to migrate. Signature and return type are preserved for type-checkers via ``ParamSpec``.
    """

    @functools.wraps(replacement)
    def _alias(*args: _P.args, **kwargs: _P.kwargs) -> _RT:
        warnings.warn(
            f"{old}() is deprecated and will be removed in a future release; use {new}() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return replacement(*args, **kwargs)

    return _alias


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


def _even_chunks(items: Sequence[_T], k: int) -> list[Sequence[_T]]:
    """Split ``items`` into ``k`` contiguous, near-even chunks (order preserved, no empties)."""
    n = len(items)
    base, extra = divmod(n, k)
    chunks: list[Sequence[_T]] = []
    start = 0
    for i in range(k):
        size = base + (1 if i < extra else 0)
        if size == 0:
            continue  # more chunks requested than items — skip the empty tail
        chunks.append(items[start : start + size])
        start += size
    return chunks


def _map_folds(fn: Callable[[_T], _R], items: Sequence[_T], n_jobs: int) -> list[_R]:
    """Map ``fn`` over independent fold work, order-preserving (so results stay deterministic).

    ``n_jobs=1`` runs serially in-process — identical to the sequential path, zero overhead. Any
    other value uses a **thread** pool, chosen over processes because (a) it works with any
    estimator (no pickling / closure limits, so it is safe to turn on by default) and (b)
    asset-pricing fits are BLAS-bound and NumPy/SciPy release the GIL during the heavy linear
    algebra, so threads parallelize the real work.

    Work is **batched** into ~4x-workers contiguous chunks rather than submitted one task per
    fold: with many short folds the per-submit / future bookkeeping cost dominates, so amortizing
    it over a handful of folds per task cuts overhead, while 4x-workers (rather than exactly
    workers) keeps the pool load-balanced when fold costs vary. Determinism is preserved: each
    fold is a pure function of ``(estimator, train, test)``, chunks are contiguous, and
    :meth:`ThreadPoolExecutor.map` yields chunk results in input order, so the flattened output
    reassembles bit-for-bit like the serial one.
    """
    if n_jobs == 1 or len(items) <= 1:
        return [fn(it) for it in items]
    workers = _resolve_workers(n_jobs)
    if workers == 1:
        return [fn(it) for it in items]
    chunks = _even_chunks(items, min(len(items), 4 * workers))

    def _run_chunk(chunk: Sequence[_T]) -> list[_R]:
        return [fn(it) for it in chunk]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return [r for chunk_res in pool.map(_run_chunk, chunks) for r in chunk_res]


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


@dataclass(frozen=True)
class PricingOutput:
    """Output for a ``to_pricing`` method: predicted expected returns vs realized, on test assets.

    ``predicted`` and ``realized`` are both ``(date x asset)``: ``predicted.loc[t]`` is the model's
    cross-section of expected returns for the return realized over ``(t, t+h]`` and
    ``realized.loc[t]`` is that realized return (``nan`` for an asset absent / not yet realized at
    ``t``). ``protocol`` records the discipline the panels were produced under — ``"walk_forward"``
    (per-fold PIT refits, :func:`backtest_pricing`) or ``"in_sample"`` (one full-sample fit,
    :func:`backtest_pricing_in_sample`) — and flows straight through to every result row so an
    explanatory in-sample R^2 stays distinguishable from an out-of-sample one.
    """

    predicted: pd.DataFrame
    realized: pd.DataFrame
    method: str
    config_hash: str
    data_vintage: str
    run_id: str
    protocol: str
    capability: str = capabilities.TO_PRICING
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def universe(self) -> str:
        """Compact universe label (``n=<#assets>`` for panels, the name for a single asset)."""
        cols = [str(c) for c in self.predicted.columns]
        return cols[0] if len(cols) == 1 else f"n={len(cols)}"


def backtest_forecast(
    estimator: Estimator,
    view: TimeSeriesView,
    *,
    min_train: int = 20,
    window: int | None = None,
    horizon: int | None = None,
    refit_every: int = 1,
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
    target is strictly future.

    ``refit_every`` decouples the refit cadence from the prediction cadence (the ML-cross-section
    protocol: e.g. annual refits with monthly predictions = ``refit_every=12`` on a monthly
    calendar): the model is re-fit on every ``refit_every``-th origin and reused for the origins in
    between, whose forecasts still consume each origin's own up-to-date PIT window (fresh features,
    stale parameters — never stale information). The benchmark stays the per-origin prevailing
    mean regardless. Include ``refit_every`` in ``config`` for provenance if you sweep it.

    ``n_jobs`` fans the independent refit blocks over a thread pool (``-1`` = all cores);
    results are order-preserved, so the output is identical to ``n_jobs=1``.
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
    if refit_every < 1:
        raise ValueError(f"refit_every must be >= 1; got {refit_every}")

    def _train_at(j: int) -> TimeSeriesView:
        train = view.window(cal[j])
        return train.tail(window) if window is not None else train

    def _run_block(block: list[int]) -> list[tuple[pd.Timestamp, Float, Float, Float]]:
        model = estimator.fit(_train_at(block[0]))
        if capabilities.TO_FORECAST not in model.capabilities() or not isinstance(
            model, SupportsForecast
        ):
            raise TypeError(f"{method}: fitted model does not support 'to_forecast'")
        rows: list[tuple[pd.Timestamp, Float, Float, Float]] = []
        for j in block:
            origin = cal[j]
            train = _train_at(j)
            # Reindex the forecast to ``view.assets`` order so it pairs BY LABEL with the realized
            # target and the benchmark (both built in ``view.assets`` order); a forecast Series
            # whose index is permuted relative to ``view.assets`` would otherwise be scored
            # positionally against the wrong asset's realized return. Index labels are str-normed
            # first (the shape contract str-maps the forecast index) so a str-match still aligns.
            fc = model.forecast(train)
            f = fc.set_axis([str(i) for i in fc.index]).reindex(assets)
            bench = train.returns_frame().to_numpy(dtype=np.float64).mean(axis=0)
            rows.append(
                (origin, f.to_numpy(dtype=np.float64), bench, view.target_asof(origin, horizon=h))
            )
        return rows

    origins = list(range(warmup - 1, n - h))
    blocks = [origins[i : i + refit_every] for i in range(0, len(origins), refit_every)]
    rows = [r for block_rows in _map_folds(_run_block, blocks, n_jobs) for r in block_rows]
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


def backtest_weights(
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
        # Reindex to the canonical ``view.assets`` order so the model's returned columns pair with
        # ``realized`` (built in ``view.assets`` order) BY LABEL, not by position. A model that
        # returns weight columns permuted or subset relative to ``view.assets`` would otherwise be
        # silently mis-scored downstream (``strategy_returns`` multiplies positionally by column).
        # Assets the model omits become NaN and drop out of the ``nansum`` P&L consistently with the
        # existing unrealized-tail handling. (The wide path expects a DataFrame; the panel path,
        # which handles a long Series, is ``backtest_panel``.) Column labels are str-normed first so
        # a model whose columns only str-match ``view.assets`` (the shape contract str-maps) aligns.
        wide = cast("pd.DataFrame", w)
        wide = wide.set_axis([str(c) for c in wide.columns], axis=1)
        w = wide.reindex(columns=assets)
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

    @property
    def universe(self) -> str:
        """Compact universe label (``n=<#assets>`` over the OOS panel; the name if single)."""
        names = self.weights.index.get_level_values("asset").unique()
        return str(names[0]) if len(names) == 1 else f"n={len(names)}"

    def strategy_returns(self) -> pd.Series:
        """Cross-sectional portfolio return per date: ``sum_a weights[t, a] * realized[t, a]``."""
        prod = self.weights * self.realized
        return prod.groupby(level="date").sum().rename("strategy_return")


def _panel_realized(view: CrossSectionView, keys: pd.MultiIndex, horizon: int) -> pd.Series:
    """Forward return over ``(t, t+h]`` for each ``(date, asset)`` key (``nan`` on delisting)."""
    dates = keys.get_level_values("date")
    assets = keys.get_level_values("asset").to_numpy()
    out = np.full(len(keys), np.nan, dtype=np.float64)
    for t in pd.DatetimeIndex(dates).unique():
        ids, y = view.target_asof(t, horizon=horizon)
        pos = np.flatnonzero(dates == t)
        cross = pd.Series(y, index=pd.Index(ids))
        out[pos] = cross.reindex(assets[pos]).to_numpy(dtype=np.float64)
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


def backtest_panel(
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

    Mirrors :func:`backtest_weights` but for :class:`~numeraire.core.data.CrossSectionView`: the
    fitted model returns long ``(date, asset)`` weights, realized forward returns are aligned by key
    (so an entering/exiting universe is handled), and any name whose horizon is unrealized in-view
    (or that delists first) is dropped before scoring. The time-series engine is left untouched.
    ``n_jobs`` fans the folds over a thread pool (``-1`` = all cores); order-preserving, so
    identical output.
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


def _pricing_realized(view: Any, predicted: pd.DataFrame) -> pd.DataFrame:
    """Realized ``(t, t+h]`` returns aligned to ``predicted``'s ``(date x asset)`` shape.

    Pulled from the same view the model was fit on, so the model never touches future returns:
    ``realized.loc[t, a]`` is asset ``a``'s return over ``(t, t+h]`` (``nan`` where ``a`` is absent
    at ``t`` or the horizon is not yet realized in-view). Handles both concrete view shapes — a
    :class:`~numeraire.core.data.TimeSeriesView` returns block or a ragged
    :class:`~numeraire.core.data.CrossSectionView` cross-section.
    """
    index = predicted.index
    columns = [str(c) for c in predicted.columns]
    col_pos = {c: j for j, c in enumerate(columns)}
    out = np.full((len(index), len(columns)), np.nan, dtype=np.float64)
    if isinstance(view, TimeSeriesView):
        assets = view.assets
        for i, t in enumerate(index):
            y = view.target_asof(t)
            for a, val in zip(assets, y, strict=True):
                j = col_pos.get(a)
                if j is not None:
                    out[i, j] = val
    elif isinstance(view, CrossSectionView):
        for i, t in enumerate(index):
            ids, y = view.target_asof(t)
            for a, val in zip(ids, y, strict=True):
                j = col_pos.get(str(a))
                if j is not None:
                    out[i, j] = val
    else:
        raise TypeError(
            f"pricing driver cannot align realized returns for a {type(view).__name__}; "
            "pass a TimeSeriesView or CrossSectionView"
        )
    return pd.DataFrame(out, index=index, columns=predicted.columns)


def _finalize_pricing(
    predicted: pd.DataFrame, realized: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Drop prediction dates whose realized cross-section is fully unrealized (the horizon tail)."""
    if predicted.empty:
        return predicted, realized
    keep = ~realized.isna().to_numpy().all(axis=1)
    return predicted.iloc[keep], realized.iloc[keep]


def backtest_pricing(
    estimator: Estimator,
    view: Any,
    splitter: Any,
    *,
    method: str,
    config: dict[str, Any] | None = None,
    data_vintage: str = "unknown",
    run_id: str | None = None,
    n_jobs: int = 1,
) -> PricingOutput:
    """Walk-forward OOS pricing of a ``to_pricing`` estimator: pooled predicted vs realized panels.

    Mirrors :func:`backtest_weights`: for each ``(train, test)`` fold the estimator is fit on the
    PIT train window and its fitted model prices the test window via
    :meth:`~numeraire.core.protocols.SupportsPricing.expected_returns`; realized ``(t, t+h]``
    returns are pulled from the full ``view`` (never the model), and the per-fold cross-sections are
    pooled into one ``(date x asset)`` panel pair tagged ``protocol="walk_forward"``. Works on
    a :class:`~numeraire.core.data.TimeSeriesView` (SDF-style N-asset block) or a
    :class:`~numeraire.core.data.CrossSectionView` (characteristic panel). ``n_jobs`` fans the folds
    over a thread pool (``-1`` = all cores); order-preserving, so identical output.
    """
    chash = config_hash(config)
    rid = run_id if run_id is not None else f"{method}-{chash}"

    def _run(fold: tuple[Any, Any]) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        train, test = fold
        model = estimator.fit(train)
        if capabilities.TO_PRICING not in model.capabilities() or not isinstance(
            model, SupportsPricing
        ):
            raise TypeError(f"{method}: fitted model does not support 'to_pricing'")
        pred = model.expected_returns(test)
        if pred.empty:
            return None
        pred, realized = _finalize_pricing(pred, _pricing_realized(view, pred))
        if pred.empty:
            return None
        return pred, realized

    results = _map_folds(_run, list(splitter.split(view)), n_jobs)
    p_parts: list[pd.DataFrame] = [r[0] for r in results if r is not None]
    r_parts: list[pd.DataFrame] = [r[1] for r in results if r is not None]
    if p_parts:
        predicted = pd.concat(p_parts).sort_index()
        realized_df = pd.concat(r_parts).sort_index()
    else:
        predicted = pd.DataFrame(columns=view.assets)
        realized_df = pd.DataFrame(columns=view.assets)
    return PricingOutput(
        predicted=predicted,
        realized=realized_df,
        method=method,
        config_hash=chash,
        data_vintage=data_vintage,
        run_id=rid,
        protocol="walk_forward",
    )


def backtest_pricing_in_sample(
    estimator: Estimator,
    view: Any,
    *,
    method: str,
    config: dict[str, Any] | None = None,
    data_vintage: str = "unknown",
    run_id: str | None = None,
) -> PricingOutput:
    """In-sample pricing: one full-sample fit, expected returns over the whole view (``in_sample``).

    The paper cross-sectional-pricing tradition — a single fit on all of ``view`` (no walk-forward
    discipline) whose expected returns are scored against the same sample's realized returns, tagged
    ``protocol="in_sample"`` so the explanatory nature of the number is explicit in each result row.
    Use :func:`backtest_pricing` for the out-of-sample counterpart.
    """
    chash = config_hash(config)
    rid = run_id if run_id is not None else f"{method}-{chash}"
    model = estimator.fit(view)
    if capabilities.TO_PRICING not in model.capabilities() or not isinstance(
        model, SupportsPricing
    ):
        raise TypeError(f"{method}: fitted model does not support 'to_pricing'")
    pred = model.expected_returns(view)
    if pred.empty:
        predicted, realized_df = pred, pred.copy()
    else:
        predicted, realized_df = _finalize_pricing(pred, _pricing_realized(view, pred))
    return PricingOutput(
        predicted=predicted,
        realized=realized_df,
        method=method,
        config_hash=chash,
        data_vintage=data_vintage,
        run_id=rid,
        protocol="in_sample",
    )


_DISPATCH_CAPS = (capabilities.TO_WEIGHTS, capabilities.TO_FORECAST, capabilities.TO_PRICING)


def backtest(
    estimator: Estimator,
    view: Any,
    splitter: Any = None,
    *,
    method: str,
    in_sample: bool = False,
    **kwargs: Any,
) -> WeightsOutput | ForecastOutput | PricingOutput | PanelWeightsOutput:
    """Backtest ``estimator`` over ``view``, dispatching to the right typed driver by capability.

    The discoverable entry point over the typed drivers. It routes on two things:

    - **which capability** the fitted model advertises — ``capabilities()`` intersected with
      ``{to_weights, to_forecast, to_pricing}``;
    - **the view type** — :class:`~numeraire.core.data.TimeSeriesView` vs
      :class:`~numeraire.core.data.CrossSectionView`.

    Dispatch rule (capability + view -> driver -> Output):

    - ``to_weights`` + ``TimeSeriesView`` -> :func:`backtest_weights` -> ``WeightsOutput``
    - ``to_weights`` + ``CrossSectionView`` -> :func:`backtest_panel` -> ``PanelWeightsOutput``
    - ``to_forecast`` + ``TimeSeriesView`` -> :func:`backtest_forecast` -> ``ForecastOutput``
    - ``to_pricing`` (either view) -> :func:`backtest_pricing` -> ``PricingOutput``
    - ``to_pricing`` + ``in_sample=True`` -> :func:`backtest_pricing_in_sample` -> ``PricingOutput``

    ``in_sample=True`` selects the single-full-sample-fit pricing path (explanatory in-sample R^2)
    and requires a ``to_pricing`` model. Every other path is walk-forward.

    To read the capabilities the model must first be fitted, so ``backtest`` does **one inspection
    fit on the full ``view``** and then delegates to the driver (which re-fits per fold / on the
    full sample as its discipline requires). The extra fit is intentional and cheap relative to a
    full walk-forward; power users who want to skip it — or who need the precise return type — can
    call the typed driver (``backtest_weights`` / ``backtest_forecast`` / ``backtest_panel`` /
    ``backtest_pricing`` / ``backtest_pricing_in_sample``) directly.

    A model advertising more than one of the three dispatchable capabilities is **ambiguous** and
    raises ``TypeError`` — call the specific typed driver in that case. Extra keyword arguments
    (``min_train``, ``window``, ``refit_every``, ``config``, ``data_vintage``, ``run_id``,
    ``n_jobs``, ...) are forwarded to the selected driver.
    """
    model = estimator.fit(view)  # one inspection fit to read the model's capabilities
    caps = set(model.capabilities()) & set(_DISPATCH_CAPS)

    if in_sample:
        if capabilities.TO_PRICING not in caps:
            raise TypeError(
                f"{method}: in_sample=True selects the in-sample pricing path, but the fitted "
                "model does not support 'to_pricing'"
            )
        return backtest_pricing_in_sample(estimator, view, method=method, **kwargs)

    if not caps:
        raise TypeError(
            f"{method}: fitted model advertises none of the dispatchable capabilities "
            f"{list(_DISPATCH_CAPS)}; nothing to backtest"
        )
    if len(caps) > 1:
        raise TypeError(
            f"{method}: fitted model advertises multiple dispatchable capabilities "
            f"{sorted(caps)}; backtest() cannot pick one — call the explicit typed driver "
            "(backtest_weights / backtest_forecast / backtest_panel / backtest_pricing)"
        )
    (cap,) = caps

    if cap == capabilities.TO_FORECAST:
        # The forecast route is windowed by ``min_train`` / ``window``, not a splitter. A splitter
        # here is a caller mistake — surface it rather than silently ignore the argument.
        if splitter is not None:
            raise TypeError(
                f"{method}: forecast backtests use `min_train`/`window`, not a splitter; "
                "drop the splitter argument"
            )
        return backtest_forecast(estimator, view, method=method, **kwargs)

    # The weights / panel / pricing routes are walk-forward and delegate to ``splitter.split``;
    # a missing splitter would otherwise die with a cryptic ``AttributeError`` on ``None``.
    if splitter is None:
        raise TypeError(
            f"{method}: this walk-forward backtest requires a `splitter` "
            "(e.g. WalkForwardSplitter); got splitter=None"
        )
    if cap == capabilities.TO_WEIGHTS:
        if isinstance(view, CrossSectionView):
            return backtest_panel(estimator, view, splitter, method=method, **kwargs)
        return backtest_weights(estimator, view, splitter, method=method, **kwargs)
    return backtest_pricing(estimator, view, splitter, method=method, **kwargs)


# --- deprecated aliases (one release): old walk_forward* names forward to the backtest* drivers ---
walk_forward = _deprecated_alias(backtest_weights, old="walk_forward", new="backtest_weights")
walk_forward_forecast = _deprecated_alias(
    backtest_forecast, old="walk_forward_forecast", new="backtest_forecast"
)
walk_forward_panel = _deprecated_alias(
    backtest_panel, old="walk_forward_panel", new="backtest_panel"
)
walk_forward_pricing = _deprecated_alias(
    backtest_pricing, old="walk_forward_pricing", new="backtest_pricing"
)
pricing_in_sample = _deprecated_alias(
    backtest_pricing_in_sample, old="pricing_in_sample", new="backtest_pricing_in_sample"
)
