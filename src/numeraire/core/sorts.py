"""Portfolio sorts — the cross-sectional decile-sort constructor (anomaly / characteristic sorts).

At each date, assets are ranked on a signal into ``n_bins`` portfolios and each portfolio's return
is a (value- or equal-) weighted average; the long-short is the extreme-bin spread. The one subtlety
that matters for reproducing published anomaly returns is the **breakpoint universe**: NYSE-style
breakpoints are computed on a *subset* (e.g. NYSE stocks) but *applied* to the full cross-section,
so the many small NASDAQ names don't drag the cutoffs down. Pass ``breakpoint_universe`` to enable
this; leave it ``None`` for name-count (all-stock) breakpoints.

``signal`` and ``returns`` are already **aligned**: ``returns.loc[t]`` is the return earned over the
holding period by the position formed from ``signal.loc[t]`` (the engine / caller owns the PIT lag).
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import ParamSpec, TypeVar

import numpy as np
import pandas as pd

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
    counts: pd.DataFrame  # (date x n_bins) number of assets in each bin


def sort_portfolios(
    signal: pd.DataFrame,
    returns: pd.DataFrame,
    *,
    n_bins: int = 10,
    breakpoint_universe: pd.DataFrame | None = None,
    weights: pd.DataFrame | None = None,
    direction: int = 1,
) -> SortResult:
    """Cross-sectional ``n_bins`` sort of ``signal`` with per-period weighted portfolio returns.

    ``signal`` / ``returns`` are ``(date x asset)`` and aligned (see module docstring). Breakpoints
    are the ``n_bins``-quantiles of the signal over ``breakpoint_universe`` (a ``(date x asset)``
    boolean mask, e.g. NYSE membership) when given, else over all valid names; every valid asset is
    then binned against those cutoffs. ``weights`` (``(date x asset)``, e.g. market cap) gives
    value-weighting — omit for equal-weighting. ``direction`` (+1/-1) orients the long-short
    (``+1`` = long the top bin). Bins with no assets get a NaN return.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2; got {n_bins}")
    if direction not in (1, -1):
        raise ValueError(f"direction must be +1 or -1; got {direction}")
    if not signal.index.equals(returns.index) or list(signal.columns) != list(returns.columns):
        raise ValueError("signal and returns must be aligned on the same (date x asset) axes")

    dates = signal.index
    sig = signal.to_numpy(dtype=np.float64)
    ret = returns.to_numpy(dtype=np.float64)
    wts = None if weights is None else weights.to_numpy(dtype=np.float64)
    uni = None if breakpoint_universe is None else breakpoint_universe.to_numpy(dtype=bool)

    port = np.full((len(dates), n_bins), np.nan)
    cnts = np.zeros((len(dates), n_bins), dtype=np.int64)
    ls = np.full(len(dates), np.nan)
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]  # interior quantile cut points

    for i in range(len(dates)):
        s = np.asarray(sig[i], dtype=np.float64)
        r = np.asarray(ret[i], dtype=np.float64)
        valid = ~(np.isnan(s) | np.isnan(r))
        if valid.sum() < n_bins:
            continue
        bp_mask = valid
        if uni is not None:
            bp_mask = valid & uni[i]
            if bp_mask.sum() == 0:
                bp_mask = valid  # fall back to all-name breakpoints when the universe is empty
        cuts = np.quantile(s[bp_mask], qs)
        bins = np.asarray(np.searchsorted(cuts, s[valid], side="right"), dtype=np.int64)
        w = None if wts is None else wts[i][valid]
        rv = r[valid]
        for b in range(n_bins):
            in_b = bins == b
            cnts[i, b] = int(in_b.sum())
            if not in_b.any():
                continue
            if w is not None and np.nansum(w[in_b]) > 0:
                wb = np.where(np.isnan(w[in_b]), 0.0, w[in_b])
                port[i, b] = float(np.dot(wb, rv[in_b]) / wb.sum())
            else:
                port[i, b] = float(rv[in_b].mean())
        top, bot = port[i, n_bins - 1], port[i, 0]
        if np.isfinite(top) and np.isfinite(bot):
            ls[i] = direction * (top - bot)

    cols = list(range(n_bins))
    return SortResult(
        portfolios=pd.DataFrame(port, index=dates, columns=cols),
        long_short=pd.Series(ls, index=dates, name="long_short"),
        counts=pd.DataFrame(cnts, index=dates, columns=cols),
    )


# --- deprecated alias (one release) ---
make_sorts = _deprecated_alias(sort_portfolios, old="make_sorts", new="sort_portfolios")
