"""Walk-forward splitters. Yields ``(train, test)`` views, PIT-aware.

The splitter only uses the :class:`~numeraire.core.protocols.DataView` calendar plus
``between``, so it is shape-agnostic. The horizon purge (a train fold's targets never reach
into its test fold) is enforced by the view's :meth:`~numeraire.core.data.TimeSeriesView.aligned`,
which drops any feature whose ``(t, t+h]`` target is not realized by the train cutoff;
``embargo`` adds an optional extra gap on top for serial-correlation safety.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta
from typing import Protocol, TypeVar, cast

import pandas as pd

from numeraire.core.data import TimeSeriesView


class _WindowedView(Protocol):
    """Structural view surface ``validation_split`` needs (both concrete views satisfy it)."""

    @property
    def calendar(self) -> pd.DatetimeIndex: ...
    def between(self, start: object, end: object) -> _WindowedView: ...


_V = TypeVar("_V", bound=_WindowedView)


def validation_split(view: _V, valid_size: int) -> tuple[_V, _V]:
    """Split a (train) view into PIT ``(fit, valid)``: valid = the last ``valid_size`` dates.

    The tuning pattern of the ML-cross-section protocols (fit candidate hyperparameters on
    ``fit``, score them on ``valid``, both strictly inside the train fold): ``fit`` keeps the
    fold's calendar up to the cutoff with **data truncated at the cutoff** (so its supervised
    pairs never see valid-period returns — the usual horizon purge applies at the seam), while
    ``valid`` keeps the trailing dates with full history available for lagged features.

    Estimators call this *inside* ``fit(train)``; the engine and splitter stay two-way. Assumes
    the view's calendar is a contiguous run of its dates (true for engine train folds).
    """
    cal = view.calendar
    if valid_size < 1:
        raise ValueError(f"valid_size must be >= 1; got {valid_size}")
    if valid_size >= len(cal) - 1:
        raise ValueError(
            f"valid_size={valid_size} leaves <2 fit dates on a {len(cal)}-date calendar"
        )
    cutoff = cal[-valid_size - 1]
    lo = cal[0] - pd.Timedelta(1, "ns")  # keep the fold's own start (rolling windows stay rolling)
    fit = view.between(lo, cutoff)
    valid = view.between(cutoff, cal[-1])
    return cast("_V", fit), cast("_V", valid)


@dataclass(frozen=True)
class WalkForwardSplitter:
    """Expanding- or rolling-window walk-forward splitter.

    Parameters
    ----------
    min_train:
        Minimum number of calendar observations in the first train fold.
    test_size:
        Number of calendar observations per test fold (also the step between folds).
    expanding:
        ``True`` → train grows from the start each fold; ``False`` → rolling window of
        ``min_train`` observations.
    embargo:
        Extra calendar observations to drop between the train cutoff and the first test date,
        on top of the automatic horizon purge. Default ``0``.
    """

    min_train: int = 60
    test_size: int = 12
    expanding: bool = True
    embargo: int = 0

    def __post_init__(self) -> None:
        if self.min_train < 1:
            raise ValueError("min_train must be >= 1")
        if self.test_size < 1:
            raise ValueError("test_size must be >= 1")
        if self.embargo < 0:
            raise ValueError("embargo must be >= 0")

    def split(self, view: TimeSeriesView) -> Iterator[tuple[TimeSeriesView, TimeSeriesView]]:
        """Yield ``(train, test)`` view pairs; each test calendar is strictly future of train."""
        cal = view.calendar
        n = len(cal)
        i = self.min_train
        while i + self.embargo + self.test_size <= n:
            cutoff = cal[i - 1]  # last training observation
            train_lo = 0 if self.expanding else i - self.min_train
            train_lb = _lower_bound(cal, train_lo)
            train = view.between(train_lb, cutoff)

            test_start = cal[i - 1 + self.embargo]
            test_end = cal[i - 1 + self.embargo + self.test_size]
            test = view.between(test_start, test_end)
            yield train, test
            i += self.test_size


def _lower_bound(cal: pd.DatetimeIndex, lo: int) -> pd.Timestamp:
    """Timestamp strictly below ``cal[lo]`` so ``between(lb, ...)`` includes ``cal[lo]``."""
    first: pd.Timestamp = cal[0]
    if lo == 0:
        return first - timedelta(days=1)
    return cal[lo - 1]
