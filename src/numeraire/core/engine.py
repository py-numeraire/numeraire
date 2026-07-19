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

import copy
import functools
import hashlib
import json
import os
import warnings
from collections.abc import Callable, Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal, ParamSpec, TypeVar, cast

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

MissingReturnPolicy = Literal["error", "zero", "renormalize_legs"]
_MISSING_RETURN_POLICIES: tuple[MissingReturnPolicy, ...] = (
    "error",
    "zero",
    "renormalize_legs",
)


@dataclass(frozen=True)
class _ScoringStats:
    """Audit counts produced while applying a missing-return scoring policy."""

    missing_held: int
    missing_dates: int
    renormalized_dates: int


def _validate_missing_return_policy(policy: str) -> MissingReturnPolicy:
    """Validate and narrow a public missing-return policy string."""
    if policy not in _MISSING_RETURN_POLICIES:
        raise ValueError(
            f"missing_returns must be one of {_MISSING_RETURN_POLICIES}; got {policy!r}"
        )
    return policy


def _weights_config(config: dict[str, Any] | None, policy: MissingReturnPolicy) -> dict[str, Any]:
    """Add the engine scoring convention to the configuration provenance."""
    merged = dict(config or {})
    if "missing_returns" in merged and merged["missing_returns"] != policy:
        raise ValueError(
            "config['missing_returns'] conflicts with the explicit missing_returns argument; "
            "pass the scoring convention only through missing_returns"
        )
    merged["missing_returns"] = policy
    return merged


def _scoring_array(
    weights: Float,
    realized: Float,
    groups: np.ndarray,
    group_labels: Sequence[object],
    asset_labels: np.ndarray,
    policy: MissingReturnPolicy,
) -> tuple[Float, _ScoringStats]:
    """Apply one scoring policy to flattened weights/returns grouped by decision date.

    Target weights are never mutated. For ``renormalize_legs``, observed positive and negative
    weights are separately rescaled back to their original per-date leg exposures. This preserves
    both gross and net target exposure while making the effective, ex-post scoring weights
    inspectable.
    """
    if weights.shape != realized.shape or weights.ndim != 1:
        raise ValueError("weights and realized returns must be aligned one-dimensional arrays")
    if len(groups) != len(weights) or len(asset_labels) != len(weights):
        raise ValueError("scoring labels must align exactly with weights")
    bad_weight = ~np.isfinite(weights)
    if bool(bad_weight.any()):
        i = int(np.flatnonzero(bad_weight)[0])
        date = group_labels[int(groups[i])]
        raise ValueError(
            f"non-finite target weight for asset {asset_labels[i]!r} at {date}; "
            "weights must be finite"
        )

    held = weights != 0.0
    observed = np.isfinite(realized)
    missing = held & ~observed
    missing_groups = np.unique(groups[missing])
    stats = _ScoringStats(
        missing_held=int(missing.sum()),
        missing_dates=len(missing_groups),
        renormalized_dates=len(missing_groups) if policy == "renormalize_legs" else 0,
    )
    if not bool(missing.any()):
        return weights.copy(), stats

    if policy == "error":
        i = int(np.flatnonzero(missing)[0])
        date = group_labels[int(groups[i])]
        same_date = missing & (groups == groups[i])
        names = [str(a) for a in asset_labels[same_date]]
        raise ValueError(
            f"non-finite return for held asset(s) {names} at {date}; "
            "handle delisting returns upstream or explicitly select "
            "missing_returns='zero'/'renormalize_legs'"
        )

    effective = weights.copy()
    if policy == "zero":
        return effective, stats

    effective[missing] = 0.0
    n_groups = len(group_labels)
    positive = weights > 0.0
    negative = weights < 0.0
    for leg_name, leg in (("positive", positive), ("negative", negative)):
        magnitudes = np.where(leg, np.abs(weights), 0.0)
        original = np.bincount(groups, weights=magnitudes, minlength=n_groups)
        observed_magnitudes = np.where(leg & observed, np.abs(weights), 0.0)
        available = np.bincount(groups, weights=observed_magnitudes, minlength=n_groups)
        unidentified = (original > 0.0) & (available == 0.0)
        if bool(unidentified.any()):
            group = int(np.flatnonzero(unidentified)[0])
            names = [str(a) for a in asset_labels[(groups == group) & leg & ~observed]]
            raise ValueError(
                f"all returns in the {leg_name} leg are non-finite at {group_labels[group]} "
                f"for held asset(s) {names}; portfolio return is unidentified"
            )
        scale = np.ones(n_groups, dtype=np.float64)
        scalable = available > 0.0
        scale[scalable] = original[scalable] / available[scalable]
        use = leg & observed
        effective[use] *= scale[groups[use]]
    return effective, stats


def _wide_scoring(
    weights: pd.DataFrame,
    realized: pd.DataFrame,
    policy: MissingReturnPolicy,
) -> tuple[pd.DataFrame, _ScoringStats]:
    """Return effective scoring weights and audit counts for a wide output."""
    if not weights.index.equals(realized.index) or not weights.columns.equals(realized.columns):
        raise ValueError("weights and realized must have identical indexes and columns")
    if not weights.index.is_unique or not weights.columns.is_unique:
        raise ValueError("weights axes must be unique")
    n_dates, n_assets = weights.shape
    flat_weights = weights.to_numpy(dtype=np.float64).reshape(-1)
    flat_realized = realized.to_numpy(dtype=np.float64).reshape(-1)
    groups = np.repeat(np.arange(n_dates, dtype=np.int64), n_assets)
    assets = np.tile(weights.columns.to_numpy(dtype=object), n_dates)
    effective, stats = _scoring_array(
        flat_weights,
        flat_realized,
        groups,
        list(weights.index),
        assets,
        policy,
    )
    frame = pd.DataFrame(
        effective.reshape(weights.shape), index=weights.index, columns=weights.columns
    )
    return frame, stats


def _add_scoring_stats(left: _ScoringStats, right: _ScoringStats) -> _ScoringStats:
    """Add per-date scoring counts without retaining an effective-weight artifact."""
    return _ScoringStats(
        missing_held=left.missing_held + right.missing_held,
        missing_dates=left.missing_dates + right.missing_dates,
        renormalized_dates=left.renormalized_dates + right.renormalized_dates,
    )


def _wide_scoring_stats(
    weights: pd.DataFrame,
    realized: pd.DataFrame,
    policy: MissingReturnPolicy,
) -> _ScoringStats:
    """Validate a wide output and collect audit counts with one date in memory at a time.

    Drivers only need validation and metadata, not the full effective-weight frame exposed by
    :meth:`WeightsOutput.scoring_weights`. Applying the shared policy one row at a time preserves
    identical fail-closed semantics (including an entirely missing long/short leg) without
    allocating flattened group/asset arrays or a second full-size weights frame.
    """
    if not weights.index.equals(realized.index) or not weights.columns.equals(realized.columns):
        raise ValueError("weights and realized must have identical indexes and columns")
    if not weights.index.is_unique or not weights.columns.is_unique:
        raise ValueError("weights axes must be unique")

    stats = _ScoringStats(0, 0, 0)
    assets = weights.columns.to_numpy(dtype=object)
    for row, date in enumerate(weights.index):
        row_weights = weights.iloc[row].to_numpy(dtype=np.float64, copy=False)
        row_realized = realized.iloc[row].to_numpy(dtype=np.float64, copy=False)
        _effective, current = _scoring_array(
            row_weights,
            row_realized,
            np.zeros(len(row_weights), dtype=np.int64),
            [date],
            assets,
            policy,
        )
        stats = _add_scoring_stats(stats, current)
    return stats


def _panel_scoring(
    weights: pd.Series,
    realized: pd.Series,
    policy: MissingReturnPolicy,
) -> tuple[pd.Series, _ScoringStats]:
    """Return effective scoring weights and audit counts for a long panel output."""
    if not weights.index.equals(realized.index):
        raise ValueError("panel weights and realized must have identical indexes")
    index = weights.index
    if not isinstance(index, pd.MultiIndex) or list(index.names) != ["date", "asset"]:
        raise TypeError("panel weights need a (date, asset) MultiIndex")
    if not index.is_unique:
        raise ValueError("panel weights need unique (date, asset) keys")
    dates = index.get_level_values("date")
    groups, labels = pd.factorize(dates, sort=False)
    effective, stats = _scoring_array(
        weights.to_numpy(dtype=np.float64),
        realized.to_numpy(dtype=np.float64),
        np.asarray(groups, dtype=np.int64),
        list(labels),
        index.get_level_values("asset").to_numpy(dtype=object),
        policy,
    )
    return pd.Series(effective, index=index, name=weights.name), stats


def _panel_scoring_stats(
    weights: pd.Series,
    realized: pd.Series,
    policy: MissingReturnPolicy,
) -> _ScoringStats:
    """Validate a panel output and collect audit counts one cross-section at a time.

    The normal driver output is sorted by ``(date, asset)``, so each group is a contiguous slice.
    A stable-order fallback keeps the helper correct for an otherwise valid non-contiguous input.
    Only one cross-section's temporary effective weights and masks are live at once.
    """
    if not weights.index.equals(realized.index):
        raise ValueError("panel weights and realized must have identical indexes")
    index = weights.index
    if not isinstance(index, pd.MultiIndex) or list(index.names) != ["date", "asset"]:
        raise TypeError("panel weights need a (date, asset) MultiIndex")
    if not index.is_unique:
        raise ValueError("panel weights need unique (date, asset) keys")
    if len(index) == 0:
        return _ScoringStats(0, 0, 0)

    date_codes = np.asarray(index.codes[0])
    asset_codes = np.asarray(index.codes[1])
    if bool((date_codes < 0).any()) or bool((asset_codes < 0).any()):
        raise ValueError("panel weight keys cannot contain missing date or asset labels")

    # Locate group boundaries with a one-byte-per-row mask. Sorted driver outputs take the fast
    # path; the fallback's integer order is allocated only for a manually interleaved date index.
    boundaries = np.empty(len(index), dtype=bool)
    boundaries[0] = True
    boundaries[1:] = date_codes[1:] != date_codes[:-1]
    starts = np.flatnonzero(boundaries)
    contiguous = len(np.unique(date_codes[starts])) == len(starts)
    order: np.ndarray | None = None
    grouped_codes = date_codes
    if not contiguous:
        order = np.argsort(date_codes, kind="stable")
        grouped_codes = date_codes[order]
        boundaries[0] = True
        boundaries[1:] = grouped_codes[1:] != grouped_codes[:-1]
        starts = np.flatnonzero(boundaries)
    ends = np.r_[starts[1:], len(index)]

    values = weights.to_numpy(dtype=np.float64, copy=False)
    outcomes = realized.to_numpy(dtype=np.float64, copy=False)
    date_labels = index.levels[0]
    asset_labels = index.levels[1]
    stats = _ScoringStats(0, 0, 0)
    for start, end in zip(starts, ends, strict=True):
        positions: slice | np.ndarray
        if order is None:
            positions = slice(int(start), int(end))
            code_position = int(start)
        else:
            positions = order[int(start) : int(end)]
            code_position = int(order[int(start)])
        current_assets = asset_labels.take(asset_codes[positions]).to_numpy(dtype=object)
        current_weights = values[positions]
        _effective, current = _scoring_array(
            current_weights,
            outcomes[positions],
            np.zeros(len(current_weights), dtype=np.int64),
            [date_labels[date_codes[code_position]]],
            current_assets,
            policy,
        )
        stats = _add_scoring_stats(stats, current)
    return stats


def _scoring_meta(policy: MissingReturnPolicy, stats: _ScoringStats) -> dict[str, Any]:
    """Serialize scoring provenance into an output's metadata."""
    return {
        "missing_returns": policy,
        "missing_held": stats.missing_held,
        "missing_dates": stats.missing_dates,
        "renormalized_dates": stats.renormalized_dates,
    }


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


def _fit_isolated(estimator: Estimator, view: Any, method: str) -> Any:
    """Deep-copy ``estimator``, fit the private copy, and return the fitted model.

    Every driver fits an isolated copy rather than the caller's instance — uniformly, serial *and*
    parallel. A fold's result therefore does not depend on which other folds were fitted first or
    on the thread-pool schedule, even for a stateful estimator (a warm start seeded from the last
    fit, a cached statistic): with ``n_jobs>1`` two workers would otherwise race on the same object,
    and serially the fits would chain. Isolation also mechanically enforces cross-fold fit purity,
    complementing :func:`numeraire.testing.check_fit_independence`.

    Two contracts this imposes on estimators:

    - **Deepcopy-able.** A pre-fit estimator is a parameter object, so the copy is cheap next to
      the fit it precedes. An estimator that holds an un-copyable resource (a live DB handle, an
      open socket) belongs behind a factory that builds that resource at ``fit`` time, not stored
      on the instance. A failing deepcopy raises a contextual ``TypeError`` naming the method.
    - **No fit-relevant mutable state shared across copies.** ``copy.deepcopy`` cannot sever class
      attributes, module globals, or containers a custom ``__deepcopy__`` deliberately aliases; an
      estimator that routes fit state through such shared channels defeats the isolation (and can
      still observe or mutate the caller's instance). The engine never fits the caller's instance
      directly, but only estimators honoring this contract get order- and schedule-independent
      folds — :func:`numeraire.testing.check_fold_isolation` probes the property.
    """
    try:
        isolated = copy.deepcopy(estimator)
    except Exception as exc:
        raise TypeError(
            f"{method}: estimator {type(estimator).__name__} is not deepcopy-able; the engine "
            "deep-copies the estimator before every fit to isolate folds — an estimator holding "
            "an un-copyable resource (a DB handle, an open socket) belongs behind a factory that "
            "builds the resource at fit time"
        ) from exc
    return isolated.fit(view)


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
    """OOS output for a ``to_weights`` method: target weights and aligned realized returns.

    ``weights`` and ``realized`` are both ``(date x asset)`` indexed by the prediction dates,
    where ``realized.loc[t]`` is the return over ``(t, t+h]`` (so ``strategy_returns`` is the
    realized, no-look-ahead P&L of holding ``weights.loc[t]`` over that period). ``weights`` always
    remains the model's target decision. If held returns are unavailable, ``missing_returns``
    controls scoring and :meth:`scoring_weights` exposes any ex-post effective weights separately.
    """

    weights: pd.DataFrame
    realized: pd.DataFrame
    method: str
    config_hash: str
    data_vintage: str
    run_id: str
    capability: str = capabilities.TO_WEIGHTS
    meta: dict[str, Any] = field(default_factory=dict)
    missing_returns: MissingReturnPolicy = "error"

    def __post_init__(self) -> None:
        """Reject malformed target-weight artifacts before any evaluator can consume them."""
        _validate_missing_return_policy(self.missing_returns)
        if not self.weights.index.equals(self.realized.index) or not self.weights.columns.equals(
            self.realized.columns
        ):
            raise ValueError("weights and realized must have identical indexes and columns")
        if not self.weights.index.is_unique or not self.weights.columns.is_unique:
            raise ValueError("weights axes must be unique")
        if not bool(np.isfinite(self.weights.to_numpy(dtype=np.float64)).all()):
            raise ValueError("target weights must all be finite")

    @property
    def universe(self) -> str:
        """Compact universe label (``n=<#assets>`` for panels, the name for a single asset)."""
        cols = [str(c) for c in self.weights.columns]
        return cols[0] if len(cols) == 1 else f"n={len(cols)}"

    def scoring_weights(self) -> pd.DataFrame:
        """Effective ex-post weights used only to score returns under ``missing_returns``.

        Exposure, turnover, and plots should continue to consume :attr:`weights`, which is the
        untouched target decision. This method makes any missing-return adjustment auditable.
        """
        policy = _validate_missing_return_policy(self.missing_returns)
        effective, _stats = _wide_scoring(self.weights, self.realized, policy)
        return effective

    def strategy_returns(self) -> pd.Series:
        """Realized portfolio return per date under the explicit missing-return policy."""
        effective = self.scoring_weights().to_numpy(dtype=np.float64)
        realized = self.realized.to_numpy(dtype=np.float64)
        safe_realized = np.where(np.isfinite(realized), realized, 0.0)
        return pd.Series(
            np.sum(effective * safe_realized, axis=1),
            index=self.weights.index,
            name="strategy_return",
        )


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
    results are order-preserved, so the output is identical to ``n_jobs=1``. Each refit block fits
    an isolated ``copy.deepcopy`` of ``estimator`` — never the caller's instance — so blocks stay
    order- and schedule-independent. The estimator must be deepcopy-able and must not route
    fit-relevant mutable state around the copy (class attributes, module globals, containers a
    custom ``__deepcopy__`` aliases); an un-copyable resource belongs behind a factory built at
    ``fit`` time.
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
        model = _fit_isolated(estimator, _train_at(block[0]), method)
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
            #
            # The forecast *origins* (dates) are engine-assigned from the view calendar, so — unlike
            # the weights / panel / pricing drivers, where the model returns a date-indexed output
            # and could inject out-of-fold or duplicate dates — the containment the model can
            # violate here is on the asset axis. Validate it before the reindex: duplicate labels
            # would make ``reindex`` raise cryptically, and a label absent from the view was
            # silently dropped
            # (a phantom-asset forecast scored as if the model had abstained). Both are caller bugs.
            fc = model.forecast(train)
            normalized = [str(i) for i in fc.index]
            forecast_labels = pd.Index(normalized)
            if not forecast_labels.is_unique:
                raise ValueError(f"{method}: forecast asset labels must be unique")
            extra = sorted(set(normalized) - set(assets))
            if extra:
                raise ValueError(f"{method}: forecast carries assets absent from the view: {extra}")
            f = fc.set_axis(normalized).reindex(assets)
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


def _scoreable_origins(view: TimeSeriesView | CrossSectionView, origins: pd.Index) -> np.ndarray:
    """Mask origins whose full horizon lies inside the original view calendar.

    This is the only safe reason for the engine to remove a model decision before scoring. A
    non-finite target at any earlier origin is data missingness, not an unrealized tail.
    """
    if not isinstance(origins, pd.DatetimeIndex):
        raise TypeError("weight dates must use a DatetimeIndex drawn from the view calendar")
    positions = view.calendar.get_indexer(origins)
    absent = positions < 0
    if bool(absent.any()):
        dates = [str(t) for t in origins[absent][:5]]
        raise ValueError(f"weight dates are absent from the view calendar: {dates}")
    return np.asarray(positions + view.horizon < len(view.calendar), dtype=bool)


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
    missing_returns: MissingReturnPolicy = "error",
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
        Order-preserving, so the result is identical to the serial ``n_jobs=1`` default. Each fold
        fits an isolated ``copy.deepcopy`` of ``estimator`` — never the caller's instance — so
        folds stay order- and schedule-independent. The estimator must be deepcopy-able and must
        not route fit-relevant mutable state around the copy (class attributes, module globals,
        containers a custom ``__deepcopy__`` aliases); an un-copyable resource belongs behind a
        factory built at ``fit`` time.
    missing_returns:
        Policy for a non-finite realized return on a non-zero target weight: fail closed
        (``"error"``), explicitly score it as zero (``"zero"``), or rescale the observed positive
        and negative legs separately to preserve target exposure (``"renormalize_legs"``).
    """
    policy = _validate_missing_return_policy(missing_returns)
    chash = config_hash(_weights_config(config, policy))
    rid = run_id if run_id is not None else f"{method}-{chash}"
    assets = view.assets

    def _run(
        fold: tuple[TimeSeriesView, TimeSeriesView],
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        train, test = fold
        model = _fit_isolated(estimator, train, method)
        if capabilities.TO_WEIGHTS not in model.capabilities() or not isinstance(
            model, SupportsWeights
        ):
            raise TypeError(f"{method}: fitted model does not support 'to_weights'")
        w = model.to_weights(test)
        # Reindex to the canonical ``view.assets`` order so the model's returned columns pair with
        # ``realized`` (built in ``view.assets`` order) BY LABEL, not by position. A model that
        # returns weight columns permuted or subset relative to ``view.assets`` would otherwise be
        # silently mis-scored downstream (``strategy_returns`` multiplies positionally by column).
        # Assets the model omits are zero-weighted. (The wide path expects a DataFrame; the panel
        # path, which handles a long Series, is ``backtest_panel``.) Column labels are str-normed
        # first so a model whose columns only str-match ``view.assets`` aligns.
        if not isinstance(w, pd.DataFrame):
            raise TypeError(f"{method}: time-series weights need a date x asset DataFrame")
        wide = w
        if not isinstance(wide.index, pd.DatetimeIndex):
            raise TypeError(f"{method}: weight dates need a DatetimeIndex")
        if not wide.index.is_unique:
            raise ValueError(f"{method}: weight dates must be unique")
        if not wide.columns.is_unique:
            raise ValueError(f"{method}: weight asset labels must be unique")
        outside_test = ~wide.index.isin(test.calendar)
        if bool(outside_test.any()):
            dates = [str(t) for t in wide.index[outside_test][:5]]
            raise ValueError(f"{method}: weight dates are outside the current test fold: {dates}")
        normalized_columns = [str(c) for c in wide.columns]
        if not pd.Index(normalized_columns).is_unique:
            raise ValueError(
                f"{method}: weight asset labels must remain unique after str alignment"
            )
        extra = sorted(set(normalized_columns) - set(assets))
        if extra:
            raise ValueError(f"{method}: weights carry assets absent from the view: {extra}")
        wide = wide.set_axis(normalized_columns, axis=1)
        w = wide.reindex(columns=assets, fill_value=0.0)
        if not bool(np.isfinite(w.to_numpy(dtype=np.float64)).all()):
            raise ValueError(f"{method}: target weights must all be finite")
        if len(w.index) == 0:
            return None
        realized = np.vstack([view.target_asof(t) for t in w.index])
        # Drop only the mechanically unrealized horizon tail. An earlier all-NaN row is ordinary
        # data missingness and must flow into the explicit policy rather than shorten the sample.
        keep = _scoreable_origins(view, w.index)
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

    stats = _wide_scoring_stats(weights, realized_df, policy)
    return WeightsOutput(
        weights=weights,
        realized=realized_df,
        method=method,
        config_hash=chash,
        data_vintage=data_vintage,
        run_id=rid,
        missing_returns=policy,
        meta=_scoring_meta(policy, stats),
    )


@dataclass(frozen=True)
class PanelWeightsOutput:
    """OOS output for a cross-sectional ``to_weights`` method over a ragged panel.

    ``weights`` and ``realized`` are long ``pd.Series`` on a ``(date, asset)`` MultiIndex; the wide,
    fixed-universe :class:`WeightsOutput` can't represent an entering/exiting universe, so the panel
    path carries the long form. ``realized`` is each name's ``(t, t+h]`` return, aligned by key.
    ``weights`` always remains the model's target decision; :meth:`scoring_weights` separately
    exposes any ex-post adjustment selected through ``missing_returns``.
    """

    weights: pd.Series
    realized: pd.Series
    method: str
    config_hash: str
    data_vintage: str
    run_id: str
    capability: str = capabilities.TO_WEIGHTS
    meta: dict[str, Any] = field(default_factory=dict)
    missing_returns: MissingReturnPolicy = "error"

    def __post_init__(self) -> None:
        """Reject malformed target-weight artifacts before any evaluator can consume them."""
        _validate_missing_return_policy(self.missing_returns)
        if not self.weights.index.equals(self.realized.index):
            raise ValueError("panel weights and realized must have identical indexes")
        index = self.weights.index
        if not isinstance(index, pd.MultiIndex) or list(index.names) != ["date", "asset"]:
            raise TypeError("panel weights need a (date, asset) MultiIndex")
        if not index.is_unique:
            raise ValueError("panel weights need unique (date, asset) keys")
        if not bool(np.isfinite(self.weights.to_numpy(dtype=np.float64)).all()):
            raise ValueError("panel target weights must all be finite")

    @property
    def universe(self) -> str:
        """Compact universe label (``n=<#assets>`` over the OOS panel; the name if single)."""
        names = self.weights.index.get_level_values("asset").unique()
        return str(names[0]) if len(names) == 1 else f"n={len(names)}"

    def scoring_weights(self) -> pd.Series:
        """Effective ex-post weights used only to score returns under ``missing_returns``."""
        policy = _validate_missing_return_policy(self.missing_returns)
        effective, _stats = _panel_scoring(self.weights, self.realized, policy)
        return effective

    def strategy_returns(self) -> pd.Series:
        """Cross-sectional portfolio return per date under the missing-return policy."""
        effective = self.scoring_weights()
        safe_realized = self.realized.where(np.isfinite(self.realized), 0.0)
        prod = effective * safe_realized
        return prod.groupby(level="date", sort=False).sum().rename("strategy_return")


def _panel_realized(view: CrossSectionView, keys: pd.MultiIndex, horizon: int) -> pd.Series:
    """Forward return by valid formation key (``nan`` on a later gap/non-finite input)."""
    dates = pd.DatetimeIndex(keys.get_level_values("date"))
    assets = keys.get_level_values("asset").to_numpy()
    out = np.full(len(keys), np.nan, dtype=np.float64)
    groups, unique_dates = pd.factorize(dates, sort=False)
    order = np.argsort(groups, kind="stable")
    counts = np.bincount(groups, minlength=len(unique_dates))
    ends = np.cumsum(counts)
    start = 0
    for group, t in enumerate(unique_dates):
        pos = order[start : int(ends[group])]
        start = int(ends[group])
        ids, y = view.target_asof(t, horizon=horizon)
        current = pd.Index([str(a) for a in ids])
        if not current.is_unique:
            raise ValueError(f"formation-universe asset labels collide after str alignment at {t}")
        requested = pd.Index([str(a) for a in assets[pos]])
        absent = ~requested.isin(current)
        if bool(absent.any()):
            names = [str(a) for a in requested[absent]]
            raise ValueError(
                f"panel weights carry asset(s) absent from the formation universe at {t}: {names}"
            )
        cross = pd.Series(y, index=current)
        out[pos] = cross.reindex(requested).to_numpy(dtype=np.float64)
    return pd.Series(out, index=keys, name="realized")


def _as_weight_series(w: object) -> pd.Series[Any]:
    """Normalize a panel model's weights to a long ``(date, asset)`` Series."""
    if isinstance(w, pd.DataFrame):
        if w.shape[1] != 1:
            raise TypeError(
                "panel to_weights must return a long (date, asset) Series or 1-col frame"
            )
        return w.iloc[:, 0]
    if not isinstance(w, pd.Series):
        raise TypeError("panel to_weights must return a pd.Series or one-column pd.DataFrame")
    return cast("pd.Series[Any]", w)


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
    missing_returns: MissingReturnPolicy = "error",
) -> PanelWeightsOutput:
    """Walk-forward OOS backtest of a cross-sectional ``to_weights`` estimator over a ragged panel.

    Mirrors :func:`backtest_weights` but for :class:`~numeraire.core.data.CrossSectionView`: the
    fitted model returns long ``(date, asset)`` target weights and realized forward returns are
    aligned by key. Only the mechanically unrealized horizon tail is removed; an earlier missing
    held return follows ``missing_returns`` (default ``"error"``). ``"renormalize_legs"`` rescales
    the observed positive and negative legs separately, preserving target gross/net exposure.
    ``n_jobs`` fans folds over a thread pool (``-1`` = all cores); output order is deterministic.
    Each fold fits an isolated ``copy.deepcopy`` of ``estimator`` — never the caller's instance —
    so folds stay order- and schedule-independent; the estimator must be deepcopy-able and must not
    share fit-relevant mutable state across copies (see :func:`backtest_weights`).
    """
    policy = _validate_missing_return_policy(missing_returns)
    chash = config_hash(_weights_config(config, policy))
    rid = run_id if run_id is not None else f"{method}-{chash}"

    def _run(fold: tuple[CrossSectionView, CrossSectionView]) -> tuple[pd.Series, pd.Series] | None:
        train, test = fold
        model = _fit_isolated(estimator, train, method)
        if capabilities.TO_WEIGHTS not in model.capabilities() or not isinstance(
            model, SupportsWeights
        ):
            raise TypeError(f"{method}: fitted model does not support 'to_weights'")
        w = _as_weight_series(model.to_weights(test))
        keys = w.index
        if not isinstance(keys, pd.MultiIndex):
            raise TypeError(f"{method}: panel weights need a (date, asset) MultiIndex")
        if list(keys.names) != ["date", "asset"]:
            raise TypeError(f"{method}: panel weights need index names ['date', 'asset']")
        if not keys.is_unique:
            raise ValueError(f"{method}: panel weights need unique (date, asset) keys")
        if not bool(np.isfinite(w.to_numpy(dtype=np.float64)).all()):
            raise ValueError(f"{method}: panel target weights must all be finite")
        dates = keys.get_level_values("date")
        if not isinstance(dates, pd.DatetimeIndex):
            raise TypeError(f"{method}: panel weight dates need a DatetimeIndex")
        outside_test = ~dates.isin(test.calendar)
        if bool(outside_test.any()):
            bad_dates = [str(t) for t in dates[outside_test].unique()[:5]]
            raise ValueError(
                f"{method}: panel weight dates are outside the current test fold: {bad_dates}"
            )
        normalized = pd.MultiIndex.from_arrays(
            [dates, [str(a) for a in keys.get_level_values("asset")]],
            names=["date", "asset"],
        )
        if not normalized.is_unique:
            raise ValueError(
                f"{method}: panel weight keys must remain unique after str asset alignment"
            )
        if len(w.index) == 0:
            return None
        # Validate every emitted formation key before dropping the structural tail. Otherwise a
        # model could hide a ghost/stale asset exclusively in that tail and still pass the driver.
        realized = _panel_realized(view, keys, view.horizon)
        keep = _scoreable_origins(view, dates)
        if not bool(keep.any()):
            return None
        w = w[keep]
        return w, realized[keep]

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

    stats = _panel_scoring_stats(weights, realized_s, policy)
    return PanelWeightsOutput(
        weights=weights,
        realized=realized_s,
        method=method,
        config_hash=chash,
        data_vintage=data_vintage,
        run_id=rid,
        missing_returns=policy,
        meta=_scoring_meta(policy, stats),
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


def _validate_pricing_labels(
    pred: pd.DataFrame, calendar: pd.Index, assets: Sequence[str], method: str, *, where: str
) -> pd.DataFrame:
    """Contain an expected-return panel to the fold and return it with str-normalized columns.

    Mirrors the containment guard the weights / panel drivers apply to their model output: the model
    chooses the ``(date x asset)`` index of :meth:`SupportsPricing.expected_returns`, so a stateful
    or adversarial pricer could otherwise emit dates before its train window, dates repeated across
    the panel, or a phantom asset absent from the view — all of which then flow into pooling and
    scoring as genuine OOS observations. Validation runs **before** :func:`_finalize_pricing` drops
    the structural horizon tail, so a bad label hidden exclusively in that tail cannot slip through,
    and before any emptiness short-circuit, so a zero-row panel cannot smuggle a phantom column.

    The returned frame carries the **str-normalized** column labels, so per-fold panels concatenate
    on one label set: validating ``str(c)`` but pooling the original labels would let one fold's
    integer column and another fold's string column of the same name each pass individually yet
    concatenate into two distinct assets. An empty panel (the documented "prices nothing"
    convention, a plain empty index) skips the date checks — a zero-row index carries no dates to
    contain — but never the column checks.
    """
    normalized_columns = [str(c) for c in pred.columns]
    if not pd.Index(normalized_columns).is_unique:
        raise ValueError(f"{method}: expected-return asset labels must be unique")
    extra = sorted(set(normalized_columns) - set(assets))
    if extra:
        raise ValueError(f"{method}: expected returns carry assets absent from the view: {extra}")
    if len(pred.index) > 0:
        if not isinstance(pred.index, pd.DatetimeIndex):
            raise TypeError(f"{method}: expected-return dates need a DatetimeIndex")
        if not pred.index.is_unique:
            raise ValueError(f"{method}: expected-return dates must be unique")
        outside = ~pred.index.isin(calendar)
        if bool(outside.any()):
            dates = [str(t) for t in pred.index[outside][:5]]
            raise ValueError(f"{method}: expected-return dates are outside {where}: {dates}")
    return pred.set_axis(normalized_columns, axis=1)


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
    over a thread pool (``-1`` = all cores); order-preserving, so identical output. Each fold fits
    an isolated ``copy.deepcopy`` of ``estimator`` — never the caller's instance — so folds stay
    order- and schedule-independent; the estimator must be deepcopy-able and must not share
    fit-relevant mutable state across copies (see :func:`backtest_weights`).
    """
    chash = config_hash(config)
    rid = run_id if run_id is not None else f"{method}-{chash}"

    def _run(fold: tuple[Any, Any]) -> tuple[pd.DataFrame, pd.DataFrame] | None:
        train, test = fold
        model = _fit_isolated(estimator, train, method)
        if capabilities.TO_PRICING not in model.capabilities() or not isinstance(
            model, SupportsPricing
        ):
            raise TypeError(f"{method}: fitted model does not support 'to_pricing'")
        # Validate the model-chosen (date x asset) labels before any emptiness short-circuit and
        # before dropping the structural tail, so an out-of-fold / duplicate date or a phantom
        # asset cannot be pooled as a real observation (or hide in a zero-row panel). The returned
        # frame carries str-normalized columns, so per-fold panels concatenate on one label set.
        pred = _validate_pricing_labels(
            model.expected_returns(test),
            test.calendar,
            view.assets,
            method,
            where="the current test fold",
        )
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
    model = _fit_isolated(estimator, view, method)
    if capabilities.TO_PRICING not in model.capabilities() or not isinstance(
        model, SupportsPricing
    ):
        raise TypeError(f"{method}: fitted model does not support 'to_pricing'")
    # Validated before the emptiness branch, so a zero-row panel cannot smuggle a phantom column.
    pred = _validate_pricing_labels(
        model.expected_returns(view), view.calendar, view.assets, method, where="the view calendar"
    )
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

_FORECAST_DEFAULT_MIN_TRAIN = 20  # keep in sync with backtest_forecast's ``min_train`` default


class _ReplaySplitter:
    """Replays folds already materialized from a user splitter (whose ``split`` ran exactly once).

    ``backtest`` needs the first fold's train view for the capability probe *and* the driver needs
    every fold. Calling the user splitter's ``split(view)`` twice would silently drop fold 0 when
    ``split`` returns a one-shot iterator, so the folds are materialized once and replayed through
    this stand-in. The drivers materialize the folds anyway (``list(splitter.split(view))``), so no
    laziness is lost.
    """

    def __init__(self, folds: list[tuple[Any, Any]]) -> None:
        self._folds = folds

    def split(self, view: Any) -> Iterator[tuple[Any, Any]]:
        return iter(self._folds)


def _probe_plan(
    view: Any, splitter: Any, in_sample: bool, kwargs: dict[str, Any]
) -> tuple[Any, Any]:
    """The ``(probe view, splitter)`` pair ``backtest`` uses: what to probe-fit on, what to forward.

    The probe view reproduces **the first fit the selected driver would itself perform**, never the
    full ``view`` ahead of a walk-forward run: a stateful estimator (a warm start, a cached
    statistic) must not observe post-train data while its capabilities are being read, or the
    capability probe becomes a contamination channel into the walk-forward loop. The three cases
    mirror the drivers exactly:

    - ``in_sample=True`` — the in-sample pricing driver fits the whole ``view``, so the probe does
      too (there is no earlier train window to isolate).
    - a ``splitter`` is given — the walk-forward drivers fit each fold's train window; the probe
      fits the **first** fold's train view. The user splitter's ``split(view)`` runs **exactly
      once**: the folds are materialized here and the driver receives a :class:`_ReplaySplitter`
      over them, so a one-shot ``split`` iterator loses no folds. If the splitter yields no folds
      the driver never fits — nothing exists to contaminate — and the probe falls back to ``view``.
    - no ``splitter`` — the forecast driver's first fit is its warm-up prefix; the probe rebuilds
      exactly that window (the first ``window``/``min_train`` calendar steps, rolled to ``window``
      when the view supports a rolling tail). A view shorter than the warm-up yields zero forecast
      origins in the driver, so — as in the empty-folds case — no result exists to contaminate and
      the probe falls back to ``view``.
    """
    if in_sample:
        return view, splitter
    if splitter is not None:
        folds = list(splitter.split(view))  # the user splitter's split() runs exactly once
        probe = view if not folds else folds[0][0]
        return probe, _ReplaySplitter(folds)
    window = kwargs.get("window")
    warmup = window if window is not None else kwargs.get("min_train", _FORECAST_DEFAULT_MIN_TRAIN)
    if warmup >= len(view.calendar):
        return view, None  # driver produces zero origins: no result exists to contaminate
    probe = view.window(view.calendar[warmup - 1])
    if window is not None and hasattr(probe, "tail"):
        probe = probe.tail(window)
    return probe, None


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

    To read the capabilities the model must first be fitted, so ``backtest`` does **one probe fit**
    before delegating. That probe fits on exactly the data the selected driver's *first* fit would
    see — the first fold's train window (walk-forward), the warm-up prefix (forecast), or the whole
    ``view`` (in-sample) — **never the full sample ahead of a walk-forward run**. This matters for a
    stateful estimator: fitting the probe on the full ``view`` would let it observe post-train data
    before the per-fold fits, a silent look-ahead channel; mirroring the first fold keeps the probe
    within the same information set the driver's first fit uses. A user splitter's ``split(view)``
    is called exactly once (its folds are replayed to the driver), so a one-shot ``split`` iterator
    loses no folds. The probe — like every per-fold fit the selected driver performs — runs on an
    isolated ``copy.deepcopy`` of the estimator, so the engine never fits the caller's instance
    directly; the estimator must be deepcopy-able and must not share fit-relevant mutable state
    across copies (see :func:`backtest_weights`). The extra fit is intentional and
    cheap relative to a full walk-forward; power users who want to skip it — or who need the precise
    return type — can call the typed driver (``backtest_weights`` / ``backtest_forecast`` /
    ``backtest_panel`` / ``backtest_pricing`` / ``backtest_pricing_in_sample``) directly.

    A model advertising more than one of the three dispatchable capabilities is **ambiguous** and
    raises ``TypeError`` — call the specific typed driver in that case. Extra keyword arguments
    (``min_train``, ``window``, ``refit_every``, ``missing_returns``, ``config``, ``data_vintage``,
    ``run_id``, ``n_jobs``, ...) are forwarded to the selected driver.
    """
    # Probe fit on the same data the selected driver's first fit would see (never the full sample
    # ahead of a walk-forward run), so a stateful estimator can't observe post-train data here.
    # A user splitter is materialized once and replayed to the driver (one-shot split() safe).
    # The probe fits an isolated ``copy.deepcopy`` so reading capabilities never fits the caller's
    # instance directly — otherwise a stateful probe fit would be a contamination channel into the
    # walk-forward loop (subject to the same no-shared-mutable-state contract as the fold fits).
    probe, splitter = _probe_plan(view, splitter, in_sample, kwargs)
    model = _fit_isolated(estimator, probe, method)
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
