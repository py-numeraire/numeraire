"""Universal weight baselines: the 1/N, minimum-variance and mean-variance rules.

These three are the reference points every ``to_weights`` method is measured against — the naive
benchmark (1/N) plus the two textbook sample-plug-in optimizers. They are bundled in ``numeraire``
itself (not a method package) because they are method-agnostic and every comparison needs them.

Weight functions (the single source of truth; extensions build on these rather than re-deriving):

- :func:`equal_weights` — ``w = 1/N`` (needs no estimation; the naive benchmark).
- :func:`minimum_variance_weights` — global minimum-variance ``w = S^-1 1 / (1' S^-1 1)`` (the
  first-order condition of ``min w'Sw`` s.t. ``1'w = 1``; always well defined for invertible ``S``).
- :func:`mean_variance_weights` — the plug-in tangency direction ``w ∝ S^-1 mu``, with the
  normalization made **explicit** (``"budget"`` divides by ``1' S^-1 mu`` so weights sum to one;
  ``"none"`` leaves the raw proportional direction). The ``"budget"`` divisor passes through zero
  when the tangency portfolio is nearly cash-neutral — the origin of sample mean-variance's famous
  weight/turnover explosion, which is a property of the rule, not a bug here.

The estimators (:class:`EqualWeight`, :class:`MinVariance`, :class:`MeanVariance`) are point-in-time
citizens: each ``to_weights(view)`` rebalances at every date on ``view.calendar``, estimating the
sample moments from that date's own trailing window (``window`` caps it to a rolling estimate; the
default expands from the start). They window internally, so look-ahead is structurally impossible.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from numeraire.core import capabilities
from numeraire.core.data import TimeSeriesView
from numeraire.core.protocols import DataView

Float = NDArray[np.float64]
Normalization = Literal["budget", "none"]


# --------------------------------------------------------------------------- weight functions


def equal_weights(n: int) -> Float:
    """The naive 1/N portfolio: ``w_i = 1/n`` (needs no moment estimation)."""
    if n < 1:
        raise ValueError(f"need at least one asset; got n={n}")
    return np.ones(n, dtype=np.float64) / n


def minimum_variance_weights(cov: Float) -> Float:
    """Global minimum-variance weights ``S^-1 1 / (1' S^-1 1)`` — the FOC of ``min w'Sw, 1'w=1``."""
    n = len(cov)
    x = np.linalg.solve(cov, np.ones(n, dtype=np.float64))
    return x / x.sum()


def mean_variance_weights(
    mu: Float, cov: Float, *, normalization: Normalization = "budget"
) -> Float:
    """Plug-in mean-variance (tangency) weights ``∝ S^-1 mu``, normalization made explicit.

    ``normalization``:

    - ``"budget"`` (default): divide by ``1' S^-1 mu`` so the weights sum to one — the
      DeMiguel-Garlappi-Uppal convention. The divisor passes through zero for a near-cash-neutral
      tangency portfolio, which is why sample mean-variance weights and turnover explode.
    - ``"none"``: the raw proportional direction ``S^-1 mu`` (no budget rescaling).
    """
    x = np.linalg.solve(cov, mu)
    if normalization == "budget":
        return x / x.sum()
    if normalization == "none":
        return x
    raise ValueError(f"normalization must be 'budget' or 'none'; got {normalization!r}")


# --------------------------------------------------------------------------- engine citizens


def _as_tsv(view: DataView) -> TimeSeriesView:
    if not isinstance(view, TimeSeriesView):
        raise TypeError("baseline weight rules run on a TimeSeriesView (asset-returns block)")
    return view


def _rolling_weights(
    view: TimeSeriesView, weight_fn: Callable[[Float], Float], *, window: int | None, min_obs: int
) -> pd.DataFrame:
    """Rebalance at every calendar date from that date's trailing window (PIT: windows internally).

    A date is skipped (warm-up) until at least ``min_obs`` past observations are available, and
    ``weight_fn`` gets the last ``window`` rows (all history if ``window`` is ``None``); it maps a
    ``(T x N)`` block of past returns to an ``N``-vector of weights.
    """
    assets = view.assets
    rows: list[Float] = []
    idx: list[pd.Timestamp] = []
    for t in view.calendar:
        hist = view.window(t).returns_frame().to_numpy(dtype=np.float64)
        if len(hist) < min_obs:
            continue  # warm-up
        block = hist if window is None else hist[-window:]
        if len(block) < min_obs:
            continue
        rows.append(weight_fn(block))
        idx.append(t)
    if not rows:
        return pd.DataFrame(columns=assets)
    return pd.DataFrame(np.vstack(rows), index=pd.DatetimeIndex(idx), columns=assets)


class _WeightModel:
    """Fitted (parameter-free, closed-form) baseline weight model over any view's calendar."""

    def __init__(
        self, weight_fn: Callable[[Float], Float], *, window: int | None, min_obs: int
    ) -> None:
        self._weight_fn = weight_fn
        self._window = window
        self._min_obs = min_obs

    def capabilities(self) -> set[str]:
        return {capabilities.TO_WEIGHTS}

    def to_weights(self, view: DataView) -> pd.DataFrame:
        return _rolling_weights(
            _as_tsv(view), self._weight_fn, window=self._window, min_obs=self._min_obs
        )


class EqualWeight:
    """The 1/N benchmark estimator: rebalances to equal weights over ``view.assets`` each period."""

    def fit(self, view: DataView) -> _WeightModel:
        _as_tsv(view)
        return _WeightModel(lambda b: equal_weights(b.shape[1]), window=None, min_obs=1)


def _resolve_min_obs(explicit: int | None, n_assets: int) -> int:
    """Warm-up length: an explicit floor, else one more row than assets (invertible covariance)."""
    return explicit if explicit is not None else n_assets + 1


class MinVariance:
    """Global minimum-variance estimator: sample covariance from the (optionally windowed) view.

    Parameters
    ----------
    window:
        Trailing window (in calendar steps) for the sample covariance; ``None`` (default) expands
        from the start. A rolling cap mirrors the skfolio adapter's estimation window.
    min_obs:
        Minimum observations before the first rebalance; ``None`` (default) requires strictly more
        rows than assets, so the sample covariance is non-singular.
    """

    def __init__(self, *, window: int | None = None, min_obs: int | None = None) -> None:
        if window is not None and window < 2:
            raise ValueError("window must be >= 2 (a covariance needs at least two rows)")
        if min_obs is not None and min_obs < 2:
            raise ValueError("min_obs must be >= 2")
        self.window = window
        self.min_obs = min_obs

    def fit(self, view: DataView) -> _WeightModel:
        tsv = _as_tsv(view)

        def _fn(block: Float) -> Float:
            return minimum_variance_weights(np.cov(block, rowvar=False))

        return _WeightModel(
            _fn, window=self.window, min_obs=_resolve_min_obs(self.min_obs, len(tsv.assets))
        )


class MeanVariance:
    """Plug-in mean-variance estimator: sample ``mu``/``S`` → ``S^-1 mu``, normalization explicit.

    Parameters
    ----------
    normalization:
        ``"budget"`` (default) divides by ``1' S^-1 mu`` (weights sum to one; DGU convention);
        ``"none"`` returns the raw proportional direction ``S^-1 mu``.
    window, min_obs:
        As for :class:`MinVariance` (rolling estimation window / warm-up; default warm-up is one
        row more than the asset count so the sample covariance is invertible).
    """

    def __init__(
        self,
        *,
        normalization: Normalization = "budget",
        window: int | None = None,
        min_obs: int | None = None,
    ) -> None:
        if normalization not in ("budget", "none"):
            raise ValueError(f"normalization must be 'budget' or 'none'; got {normalization!r}")
        if window is not None and window < 2:
            raise ValueError("window must be >= 2 (moments need at least two rows)")
        if min_obs is not None and min_obs < 2:
            raise ValueError("min_obs must be >= 2")
        self.normalization: Normalization = normalization
        self.window = window
        self.min_obs = min_obs

    def fit(self, view: DataView) -> _WeightModel:
        tsv = _as_tsv(view)
        norm = self.normalization

        def _fn(block: Float) -> Float:
            mu = block.mean(axis=0)
            cov = np.cov(block, rowvar=False)
            return mean_variance_weights(mu, cov, normalization=norm)

        return _WeightModel(
            _fn, window=self.window, min_obs=_resolve_min_obs(self.min_obs, len(tsv.assets))
        )
