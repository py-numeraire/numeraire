"""Portfolio sorts — the cross-sectional decile-sort constructor (anomaly / characteristic sorts).

At each date, assets are ranked on a signal into ``n_bins`` portfolios and each portfolio's return
is a (value- or equal-) weighted average; the long-short is the extreme-bin spread. The one subtlety
that matters for reproducing published anomaly returns is the **breakpoint universe**: NYSE-style
breakpoints are computed on a *subset* (e.g. NYSE stocks) but *applied* to the full cross-section,
so the many small NASDAQ names don't drag the cutoffs down. Pass ``breakpoint_universe`` to enable
this; leave it ``None`` for name-count (all-stock) breakpoints.

Formation and holding-period aggregation are deliberately separate. ``assign_portfolio_bins``
uses only the formation signal and formation-time masks. ``aggregate_assigned_portfolios`` then
joins those frozen assignments to realized returns. This boundary makes it impossible for a
missing future return to change a historical breakpoint or portfolio membership.

``signal`` and ``returns`` must describe the same date and asset labels (their input order may
differ): ``returns.loc[t]`` is the return earned over the holding period by the position formed
from ``signal.loc[t]`` (the engine / caller owns the PIT lag).
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import ParamSpec, TypeVar, cast

import numpy as np
import pandas as pd
from numpy.typing import NDArray

_P = ParamSpec("_P")
_RT = TypeVar("_RT")


def _deprecated_alias(replacement: Callable[_P, _RT], *, old: str, new: str) -> Callable[_P, _RT]:
    """Thin forwarder to ``replacement`` that emits a ``DeprecationWarning`` naming ``new``.

    Keeps a renamed public function working for one release (non-breaking). Signature and return
    type are preserved for type-checkers via ``ParamSpec``.
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


@dataclass(frozen=True)
class SortResult:
    """Per-period sorted-portfolio returns plus the long-short spread."""

    portfolios: pd.DataFrame  # (date x n_bins), columns 0..n_bins-1 (bin 0 = lowest signal)
    long_short: pd.Series  # (date,) direction * (top bin - bottom bin)
    counts: pd.DataFrame  # (date x n_bins) formation members, even when realized return is missing


@dataclass(frozen=True)
class SortAssignments:
    """Formation-time bin assignments and the breakpoints that produced them.

    ``bins`` has the same axes as the formation signal. Its finite values are integer bin labels
    in ``0 .. n_bins - 1``; ineligible names have ``NaN``. ``breakpoints`` has one row per date and
    columns ``1 .. n_bins - 1`` for the interior signal cutoffs.
    """

    bins: pd.DataFrame
    breakpoints: pd.DataFrame
    n_bins: int


def _validate_unique_axes(frame: pd.DataFrame, *, name: str) -> None:
    if not frame.index.is_unique:
        raise ValueError(f"{name} index labels must be unique")
    if not frame.columns.is_unique:
        raise ValueError(f"{name} column labels must be unique")


def _align_to(
    frame: pd.DataFrame,
    reference: pd.DataFrame,
    *,
    name: str,
    reference_name: str,
) -> pd.DataFrame:
    """Validate equal label sets and return ``frame`` in ``reference`` axis order."""
    _validate_unique_axes(frame, name=name)
    if (
        len(frame.index) != len(reference.index)
        or len(frame.columns) != len(reference.columns)
        or not bool(reference.index.isin(frame.index).all())
        or not bool(frame.index.isin(reference.index).all())
        or not bool(reference.columns.isin(frame.columns).all())
        or not bool(frame.columns.isin(reference.columns).all())
    ):
        raise ValueError(f"{name} and {reference_name} must have the same date and asset labels")
    return frame.reindex(index=reference.index, columns=reference.columns)


def _float_values(frame: pd.DataFrame, *, name: str) -> np.ndarray:
    try:
        values = frame.to_numpy(dtype=np.float64, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must contain numeric values") from exc
    if bool(np.isinf(values).any()):
        raise ValueError(f"{name} must not contain infinite values")
    return np.asarray(values, dtype=np.float64)


def _mask_values(frame: pd.DataFrame, *, name: str) -> np.ndarray:
    """Convert a nullable boolean/0-1 mask; missing means false and infinity is rejected."""
    values = _float_values(frame, name=name)
    finite = values[np.isfinite(values)]
    if bool((~np.isin(finite, np.asarray([0.0, 1.0]))).any()):
        raise ValueError(f"{name} must contain only boolean/0-1 values or missing values")
    return np.asarray(np.where(np.isnan(values), False, values == 1.0), dtype=bool)


def assign_portfolio_bins(
    signal: pd.DataFrame,
    *,
    n_bins: int = 10,
    breakpoint_universe: pd.DataFrame | None = None,
    eligibility: pd.DataFrame | None = None,
) -> SortAssignments:
    """Freeze formation-time portfolio memberships without consulting realized returns.

    ``signal`` is ``(date x asset)``. ``eligibility`` optionally restricts which assets may be
    assigned; missing mask entries mean ineligible. ``breakpoint_universe`` optionally restricts
    which eligible signals define the cutoffs, while the cutoffs are still applied to all eligible
    assets. Both masks are aligned to ``signal`` by pandas labels, so their input order is
    irrelevant.

    Every date must have at least ``n_bins`` finite, eligible breakpoint observations and at least
    ``n_bins`` distinct signal values, and its empirical quantiles must populate every requested
    bin. An empty, thin, or tie-degenerate breakpoint universe raises rather than silently
    switching to all-name breakpoints or emitting collapsed portfolios.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2; got {n_bins}")
    _validate_unique_axes(signal, name="signal")
    sig = _float_values(signal, name="signal")

    eligible = np.ones(sig.shape, dtype=bool)
    if eligibility is not None:
        aligned_eligibility = _align_to(
            eligibility,
            signal,
            name="eligibility",
            reference_name="signal",
        )
        eligible = _mask_values(aligned_eligibility, name="eligibility")
    eligible &= np.isfinite(sig)

    breakpoint_mask = eligible.copy()
    if breakpoint_universe is not None:
        aligned_universe = _align_to(
            breakpoint_universe,
            signal,
            name="breakpoint_universe",
            reference_name="signal",
        )
        breakpoint_mask &= _mask_values(aligned_universe, name="breakpoint_universe")

    assignments = np.full(sig.shape, np.nan, dtype=np.float64)
    cut_values = np.full((len(signal.index), n_bins - 1), np.nan, dtype=np.float64)
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]

    for i, date in enumerate(signal.index):
        n_breakpoint = int(breakpoint_mask[i].sum())
        if n_breakpoint < n_bins:
            raise ValueError(
                f"cannot form {n_bins} bins at date {date!r}: breakpoint universe has "
                f"{n_breakpoint} finite eligible observations"
            )
        breakpoint_signals = sig[i, breakpoint_mask[i]]
        n_distinct = int(np.unique(breakpoint_signals).size)
        if n_distinct < n_bins:
            raise ValueError(
                f"cannot form {n_bins} bins at date {date!r}: breakpoint universe has "
                f"{n_distinct} distinct signal values"
            )
        cuts = np.asarray(np.quantile(breakpoint_signals, quantiles), dtype=np.float64)
        breakpoint_bins = cast(
            NDArray[np.int64],
            np.digitize(breakpoint_signals, cuts, right=False),
        )
        populated = np.bincount(breakpoint_bins, minlength=n_bins)
        if bool((populated == 0).any()):
            raise ValueError(
                f"cannot form {n_bins} bins at date {date!r}: ties in the breakpoint universe "
                "leave at least one requested bin empty"
            )
        cut_values[i] = cuts
        assignments[i, eligible[i]] = np.searchsorted(
            cuts,
            sig[i, eligible[i]],
            side="right",
        )

    return SortAssignments(
        bins=pd.DataFrame(assignments, index=signal.index, columns=signal.columns),
        breakpoints=pd.DataFrame(
            cut_values,
            index=signal.index,
            columns=list(range(1, n_bins)),
        ),
        n_bins=n_bins,
    )


def aggregate_assigned_portfolios(
    assignments: SortAssignments,
    returns: pd.DataFrame,
    *,
    weights: pd.DataFrame | None = None,
    direction: int = 1,
) -> SortResult:
    """Aggregate frozen memberships into returns and the extreme-bin spread.

    ``returns`` and ``weights`` are aligned to ``assignments.bins`` by labels. ``counts`` always
    records formation membership, including an assigned asset whose realized return is missing.
    Equal-weighted returns use the finite realized returns in a bin. Value-weighted returns use
    only names having both a finite return and a strictly positive finite weight; missing,
    zero, or negative weights are excluded, and a bin with no usable weight remains ``NaN`` rather
    than silently falling back to equal weighting.
    """
    if assignments.n_bins < 2:
        raise ValueError(f"assignments.n_bins must be >= 2; got {assignments.n_bins}")
    if direction not in (1, -1):
        raise ValueError(f"direction must be +1 or -1; got {direction}")

    bins_frame = assignments.bins
    _validate_unique_axes(bins_frame, name="assignments.bins")
    bins = _float_values(bins_frame, name="assignments.bins")
    finite_bins = bins[np.isfinite(bins)]
    if bool(
        (
            (finite_bins != np.floor(finite_bins))
            | (finite_bins < 0)
            | (finite_bins >= assignments.n_bins)
        ).any()
    ):
        raise ValueError(
            "assignments.bins must contain integer labels in 0 .. assignments.n_bins - 1 or NaN"
        )

    aligned_returns = _align_to(
        returns,
        bins_frame,
        name="returns",
        reference_name="assignments.bins",
    )
    ret = _float_values(aligned_returns, name="returns")

    weight_values: np.ndarray | None = None
    if weights is not None:
        aligned_weights = _align_to(
            weights,
            bins_frame,
            name="weights",
            reference_name="assignments.bins",
        )
        weight_values = _float_values(aligned_weights, name="weights")

    n_dates = len(bins_frame.index)
    port = np.full((n_dates, assignments.n_bins), np.nan, dtype=np.float64)
    counts = np.zeros((n_dates, assignments.n_bins), dtype=np.int64)
    long_short = np.full(n_dates, np.nan, dtype=np.float64)

    for i in range(n_dates):
        for b in range(assignments.n_bins):
            member = bins[i] == float(b)
            counts[i, b] = int(member.sum())
            observed = member & np.isfinite(ret[i])
            if weight_values is None:
                if bool(observed.any()):
                    port[i, b] = float(ret[i, observed].mean())
                continue

            usable_weight = observed & np.isfinite(weight_values[i]) & (weight_values[i] > 0.0)
            if not bool(usable_weight.any()):
                continue
            denominator = float(weight_values[i, usable_weight].sum())
            if not np.isfinite(denominator):
                raise ValueError("weights sum must be finite within every assigned portfolio")
            port[i, b] = float(
                np.dot(
                    weight_values[i, usable_weight],
                    ret[i, usable_weight],
                )
                / denominator
            )

        top = port[i, assignments.n_bins - 1]
        bottom = port[i, 0]
        if np.isfinite(top) and np.isfinite(bottom):
            long_short[i] = direction * (top - bottom)

    columns = list(range(assignments.n_bins))
    return SortResult(
        portfolios=pd.DataFrame(port, index=bins_frame.index, columns=columns),
        long_short=pd.Series(long_short, index=bins_frame.index, name="long_short"),
        counts=pd.DataFrame(counts, index=bins_frame.index, columns=columns),
    )


def sort_portfolios(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    n_bins: int = 10,
    breakpoint_universe: pd.DataFrame | None = None,
    eligibility: pd.DataFrame | None = None,
    weights: pd.DataFrame | None = None,
    direction: int = 1,
) -> SortResult:
    """Cross-sectional ``n_bins`` sort of ``signal`` with per-period weighted portfolio returns.

    ``signal`` / ``returns`` are ``(date x asset)`` and aligned (see module docstring). Breakpoints
    are the ``n_bins``-quantiles of the signal over ``breakpoint_universe`` (a ``(date x asset)``
    boolean mask, e.g. NYSE membership) when given, else over all finite eligible signals; every
    eligible asset is then binned against those cutoffs. ``eligibility`` is an optional formation-
    time mask. ``weights`` (``(date x asset)``, e.g. market cap) gives value-weighting — omit for
    equal-weighting. ``direction`` (+1/-1) orients the long-short (``+1`` = long the top bin). Bins
    with no usable realized return get a NaN return.
    """
    assignments = assign_portfolio_bins(
        signal,
        n_bins=n_bins,
        breakpoint_universe=breakpoint_universe,
        eligibility=eligibility,
    )
    return aggregate_assigned_portfolios(
        assignments,
        returns,
        weights=weights,
        direction=direction,
    )


# --- deprecated alias (one release) ---
make_sorts = _deprecated_alias(sort_portfolios, old="make_sorts", new="sort_portfolios")
